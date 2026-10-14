"""把受控原始导出转换为可追溯的章节、原子事实及主题候选关系。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Protocol, Sequence

from training.v5.section_alignment_data.section_parser import parse_sections


NOTE_FIELDS = ("hadm_id", "note_id", "section_id", "parent_section_id", "heading",
               "section_role", "start_char", "end_char", "text", "parse_status",
               "content_status", "source_sha256", "timestamp_relation")
FACT_FIELDS = ("fact_id", "subject_id", "hadm_id", "source_table", "source_row_key",
               "concept_code", "concept_name", "value", "unit", "event_time",
               "available_time", "status", "context")
ALIGN_FIELDS = ("fact_id", "section_id", "rule_id", "relation_candidate")
LAB_NAMES = {
    50820: "ph", 50818: "paco2", 50821: "pao2", 50882: "bicarbonate",
    51301: "wbc", 51222: "hemoglobin", 50912: "creatinine", 50862: "albumin",
}
CONTEXT_FIELDS = (
    "age_at_admission", "gender", "discharge_location",
    "prior_coded_admission_count", "prior_strict_aecopd_count", "coded_admission_order",
)


def patient_split(subject_id: int) -> str:
    """与 training/v5/data.py 的 V5 患者级 MD5 划分合同保持一致。"""

    digest = hashlib.md5(f"aecopd-model-v1:{subject_id}".encode("ascii")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


class CsvRowWriter(Protocol):
    def writerow(self, row: Sequence[str]) -> object: ...


def _read(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _read_plain(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _time(value: str) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _add_candidates(
    fact_id: str, source: str, row: dict[str, str], discharge: str,
    roles: dict[str, list[str]], writer: CsvRowWriter,
) -> int:
    rules: list[tuple[str, str, str]] = []
    if source == "labs":
        rules.append(("pertinent_results", "lab_to_pertinent_results", "topic_only"))
    elif source == "diagnoses":
        rules.append(("discharge_diagnosis", "diagnosis_to_discharge_diagnosis", "topic_only"))
    elif source == "emar":
        rules.append(("hospital_course", "emar_to_hospital_course", "topic_only"))
    elif source == "prescriptions":
        rules.append(("hospital_course", "prescription_to_hospital_course", "topic_only"))
        start = _time(row["starttime"])
        end = _time(discharge)
        stop = _time(row["stoptime"])
        if start and end and timedelta(0) <= end - start <= timedelta(days=1) and (stop is None or stop >= end):
            rules.append(("discharge_medications", "near_discharge_order_unverified",
                          "topic_only_unverified_order_context"))
    elif source == "context" and row["field"] == "discharge_location":
        rules.append(("discharge_disposition", "location_to_discharge_disposition", "topic_only"))
    count = 0
    for role, rule, relation in rules:
        for section_id in roles.get(role, []):
            writer.writerow((fact_id, section_id, rule, relation))
            count += 1
    return count


def _fact(source: str, row: dict[str, str], row_number: int, discharge: str) -> tuple[str, ...]:
    fact_id = f"{source}:{row_number}"
    subject, hadm = row["subject_id"], row["hadm_id"]
    if source == "labs":
        item = int(row["itemid"])
        return (fact_id, subject, hadm, "mimiciv_hosp.labevents", row["labevent_id"],
                str(item), LAB_NAMES[item], row["valuenum"], row["valueuom"],
                row["charttime"], row["storetime"], row["flag"], "measured_lab")
    if source == "diagnoses":
        return (fact_id, subject, hadm, "mimiciv_hosp.diagnoses_icd",
                f"{hadm}:{row['seq_num']}:{row['icd_version']}:{row['icd_code']}",
                f"ICD{row['icd_version']}:{row['icd_code']}",
                row["icd_code"], "", "", discharge, discharge,
                "discharge_code_availability_assumed", "discharge_diagnosis_not_prior_history")
    if source == "emar":
        return (fact_id, subject, hadm, "mimiciv_hosp.emar",
                f"{row['emar_id']}:{row['emar_seq']}", row["medication"],
                row["medication"], row["event_txt"], "", row["charttime"],
                row["storetime"], "administered", "inpatient_administration")
    if source == "prescriptions":
        key = "|".join((row["pharmacy_id"], row["poe_id"], row["drug_type"], row["drug"]))
        return (fact_id, subject, hadm, "mimiciv_hosp.prescriptions", key,
                row["drug"], row["drug"], row["drug_type"], row["route"],
                row["starttime"], "", "availability_not_verified",
                "prescription_not_confirmed_discharge_medication")
    raise ValueError(source)


def build_dataset(
    raw_dir: Path, note_source: Path, output_dir: Path, base_source: Path | None = None,
) -> dict[str, object]:
    """只读输入；禁止覆盖已有交付物，输出汇总不包含任何病历正文。"""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    cohort: dict[str, dict[str, str]] = {}
    subjects: set[str] = set()
    for row in _read(raw_dir / "cohort.csv.gz"):
        hadm = row["hadm_id"]
        if hadm in cohort:
            raise ValueError(f"duplicate cohort hadm_id: {hadm}")
        cohort[hadm] = row
        subjects.add(row["subject_id"])
    if not cohort:
        raise ValueError("empty cohort")
    output_dir.mkdir(parents=True)
    summary: dict[str, object] = {
        "cohort_admissions": len(cohort), "cohort_patients": len(subjects),
        "note_admissions": 0, "sections": 0, "facts": 0, "candidates": 0,
        "unmatched_facts": 0, "source_facts": {}, "section_roles": {},
        "candidate_means_evidence": False,
        "prescriptions_are_confirmed_discharge_medications": False,
        "base_context_source_sha256": "",
    }
    with gzip.open(output_dir / "cohort_manifest.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subject_id", "hadm_id", "admittime", "dischtime", "patient_split",
                         "index_is_strict_aecopd"))
        for row in cohort.values():
            writer.writerow((row["subject_id"], row["hadm_id"], row["admittime"],
                             row["dischtime"], patient_split(int(row["subject_id"])),
                             row["index_is_strict_aecopd"]))

    roles_by_hadm: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    note_hadm: set[str] = set()
    role_counts: Counter[str] = Counter()
    with gzip.open(output_dir / "note_sections.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(NOTE_FIELDS)
        for row in _read(note_source):
            hadm = row["hadm_id"]
            if hadm not in cohort or row["subject_id"] != cohort[hadm]["subject_id"]:
                raise ValueError(f"note outside cohort or subject mismatch: {hadm}")
            if hadm in note_hadm:
                raise ValueError(f"multiple notes for admission: {hadm}")
            note_hadm.add(hadm)
            original = row["document_text"]
            document_hash = hashlib.sha256(original.encode("utf-8")).hexdigest()
            for section in parse_sections(original):
                if original[section.start_char:section.end_char] != section.text:
                    raise AssertionError("section offset mismatch")
                section_id = f"{row['note_id']}:{section.ordinal}"
                writer.writerow((hadm, row["note_id"], section_id, "", section.heading,
                                 section.role, section.start_char, section.end_char,
                                 section.text, section.parse_status, section.content_status,
                                 document_hash, row["timestamp_relation"]))
                roles_by_hadm[hadm][section.role].append(section_id)
                role_counts[section.role] += 1
                summary["sections"] += 1
    summary["note_admissions"] = len(note_hadm)
    summary["section_roles"] = dict(role_counts)

    counts: Counter[str] = Counter()
    with gzip.open(output_dir / "structured_facts.csv.gz", "wt", encoding="utf-8", newline="") as fact_handle, gzip.open(
        output_dir / "candidate_alignment.csv.gz", "wt", encoding="utf-8", newline=""
    ) as align_handle:
        facts = csv.writer(fact_handle)
        facts.writerow(FACT_FIELDS)
        align = csv.writer(align_handle)
        align.writerow(ALIGN_FIELDS)
        for source in ("labs", "diagnoses", "emar", "prescriptions"):
            for number, row in enumerate(_read(raw_dir / f"{source}.csv.gz"), start=1):
                hadm = row["hadm_id"]
                if hadm not in cohort or row["subject_id"] != cohort[hadm]["subject_id"]:
                    raise ValueError(f"fact outside cohort or subject mismatch: {source}:{number}")
                fact = _fact(source, row, number, cohort[hadm]["dischtime"])
                facts.writerow(fact)
                count = _add_candidates(fact[0], source, row, cohort[hadm]["dischtime"],
                                        roles_by_hadm[hadm], align)
                summary["facts"] += 1
                summary["candidates"] += count
                summary["unmatched_facts"] += int(count == 0)
                counts[source] += 1
        if base_source is not None:
            digest = hashlib.sha256()
            with base_source.open("rb") as source_handle:
                for block in iter(lambda: source_handle.read(1024 * 1024), b""):
                    digest.update(block)
            summary["base_context_source_sha256"] = digest.hexdigest()
            context_hadm: set[str] = set()
            for row in _read_plain(base_source):
                hadm = row["hadm_id"]
                if hadm not in cohort or row["subject_id"] != cohort[hadm]["subject_id"]:
                    raise ValueError("base context outside cohort")
                if hadm in context_hadm:
                    raise ValueError("duplicate base context admission")
                context_hadm.add(hadm)
                for field in CONTEXT_FIELDS:
                    value = row[field]
                    if value == "":
                        continue
                    number = counts["context"] + 1
                    is_discharge = field == "discharge_location"
                    event_time = row["dischtime"] if is_discharge else row["admittime"]
                    facts.writerow((f"context:{number}", row["subject_id"], hadm,
                                    "training.v5_export.BASE_SQL", f"{hadm}:{field}",
                                    field, field, value, "", event_time, event_time,
                                    "derived_as_of_discharge" if is_discharge else "derived_from_prior_or_static",
                                    "v5_context_without_outcome_label"))
                    count = _add_candidates(f"context:{number}", "context", {"field": field},
                                            row["dischtime"], roles_by_hadm[hadm], align)
                    summary["facts"] += 1
                    summary["candidates"] += count
                    summary["unmatched_facts"] += int(count == 0)
                    counts["context"] += 1
            if len(context_hadm) != len(cohort):
                raise ValueError("base context does not cover entire cohort")
    summary["source_facts"] = dict(counts)
    (output_dir / "qa_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--note-source", type=Path, required=True)
    parser.add_argument("--base-source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_dataset(args.raw_dir, args.note_source, args.output_dir, args.base_source)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
