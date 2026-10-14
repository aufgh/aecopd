"""以出院小结章节为锚，生成稀疏结构化概念 token 和文本章节位置。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterator

from training.v5.clinical_concepts import LABS, MED_NAMES, diagnosis_groups, medication_classes


SECTION_ROLES = (
    "patient_context", "allergies", "chief_complaint", "major_procedure",
    "history_of_present_illness", "past_medical_history", "social_history", "family_history",
    "physical_exam", "admission_physical_exam", "review_of_systems", "admission_medications",
    "admission_labs", "pertinent_results", "imaging", "microbiology", "hospital_course",
    "discharge_labs", "discharge_physical_exam", "discharge_medications", "discharge_diagnosis",
    "discharge_condition", "discharge_disposition", "discharge_instructions",
    "followup_instructions", "transitional_issues",
)
CHRONIC = ("heart_failure", "cardiac_other", "ckd", "diabetes", "malignancy", "chronic_respiratory")
ACUTE = ("acute_resp_failure", "pneumonia_or_aspiration", "sepsis", "aki")
TOKEN_FIELDS = ("subject_id", "hadm_id", "section_role", "token_type", "concept",
                "value0", "value1", "value2", "value3", "value4",
                "mask0", "mask1", "mask2", "mask3", "mask4", "unit", "provenance")
SLOT_FIELDS = ("subject_id", "hadm_id", "section_role", "text_section_ids",
               "text_nonblank_count", "structured_token_count", "has_discharge_note")


def _rows(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _days(after: str, before: str) -> float:
    return (datetime.fromisoformat(after) - datetime.fromisoformat(before)).total_seconds() / 86400.0


def _token(
    subject: str, hadm: str, role: str, kind: str, concept: str,
    values: tuple[float, ...], masks: tuple[int, ...], unit: str, provenance: str,
) -> tuple[str, ...]:
    if len(values) > 5 or len(values) != len(masks):
        raise ValueError("invalid token channel count")
    padded_values = (*values, *(0.0 for _ in range(5 - len(values))))
    padded_masks = (*masks, *(0 for _ in range(5 - len(masks))))
    return (subject, hadm, role, kind, concept,
            *(str(value) for value in padded_values),
            *(str(mask) for mask in padded_masks), unit, provenance)


def prior_chronic_history(
    cohort: dict[str, dict[str, str]], diagnosis: dict[str, set[str]],
) -> dict[str, set[str]]:
    """只汇总当前入院前已结束的、队列内更早住院的慢性诊断编码。"""

    patient_admissions: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in cohort.values():
        patient_admissions[row["subject_id"]].append(row)
    result: dict[str, set[str]] = {}
    for admissions in patient_admissions.values():
        admissions.sort(key=lambda row: (row["admittime"], row["hadm_id"]))
        previous: list[dict[str, str]] = []
        for current in admissions:
            result[current["hadm_id"]] = set().union(*(
                diagnosis[older["hadm_id"]] & set(CHRONIC)
                for older in previous if older["dischtime"] <= current["admittime"]
            ))
            previous.append(current)
    return result


def build_section_inputs(
    raw_dir: Path, aligned_dir: Path, base_csv: Path, output_dir: Path,
) -> dict[str, object]:
    """同一合格出院一组章节槽位；只读源数据，不写模型标签。"""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    validation = json.loads((aligned_dir / "qa_validation.json").read_text(encoding="utf-8"))
    if validation["status"] != "structural_checks_passed":
        raise ValueError("unaligned or unvalidated section dataset")
    for name, expected in validation["output_sha256"].items():
        if _hash(aligned_dir / name) != expected:
            raise ValueError(f"aligned source changed: {name}")
    cohort = {row["hadm_id"]: row for row in _rows(aligned_dir / "cohort_manifest.csv.gz")}
    base = {row["hadm_id"]: row for row in _rows(base_csv)}
    if len(cohort) != len(base) or cohort.keys() != base.keys():
        raise ValueError("base and section cohort differ")
    for hadm, row in base.items():
        if row["subject_id"] != cohort[hadm]["subject_id"] or row["dischtime"] != cohort[hadm]["dischtime"]:
            raise ValueError("base and section key/time mismatch")

    text_sections: dict[tuple[str, str], list[str]] = defaultdict(list)
    nonblank: Counter[tuple[str, str]] = Counter()
    note_hadms: set[str] = set()
    for row in _rows(aligned_dir / "note_sections.csv.gz"):
        hadm, role = row["hadm_id"], row["section_role"]
        note_hadms.add(hadm)
        if role not in SECTION_ROLES:
            continue
        text_sections[(hadm, role)].append(row["section_id"])
        if row["content_status"] == "informative":
            nonblank[(hadm, role)] += 1

    diagnosis: dict[str, set[str]] = defaultdict(set)
    for row in _rows(raw_dir / "diagnoses.csv.gz"):
        if row["hadm_id"] not in cohort:
            raise ValueError("diagnosis outside cohort")
        diagnosis[row["hadm_id"]].update(diagnosis_groups(row["icd_code"], int(row["icd_version"])))

    # 历史只包括本次入院前已结束的合格住院；当前出院编码另作一维代理。
    prior_chronic = prior_chronic_history(cohort, diagnosis)

    lab_tokens: dict[str, list[tuple[str, tuple[float, ...], tuple[int, ...], str]]] = defaultdict(list)
    current_key: tuple[str, str] | None = None
    series: list[float] = []
    unit = ""

    def flush_lab() -> None:
        if current_key is None:
            return
        hadm, itemid = current_key
        first, last = series[0], series[-1]
        values = (first, last, min(series), max(series), last - first if len(series) > 1 else 0.0)
        masks = (1, 1, 1, 1, int(len(series) > 1))
        lab_tokens[hadm].append((LABS[int(itemid)][0], values, masks, unit))

    for row in _rows(raw_dir / "labs.csv.gz"):
        key = row["hadm_id"], row["itemid"]
        if key != current_key:
            flush_lab()
            current_key, series, unit = key, [], row["valueuom"]
        elif row["valueuom"] != unit:
            raise ValueError("mixed lab units within admission/concept")
        series.append(float(row["valuenum"]))
    flush_lab()

    medications: dict[tuple[str, str], tuple[str, str]] = {}
    hadms_with_emar: set[str] = set()
    for row in _rows(raw_dir / "emar.csv.gz"):
        hadm = row["hadm_id"]
        hadms_with_emar.add(hadm)
        for drug_class in medication_classes(row["medication"]):
            key = hadm, drug_class
            time = row["charttime"]
            if key in medications:
                first, last = medications[key]
                medications[key] = min(first, time), max(last, time)
            else:
                medications[key] = time, time

    output_dir.mkdir(parents=True)
    counts: Counter[str] = Counter()
    tokens_by_slot: Counter[tuple[str, str]] = Counter()
    with gzip.open(output_dir / "structured_tokens.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TOKEN_FIELDS)

        def write(token: tuple[str, ...]) -> None:
            writer.writerow(token)
            counts[token[3]] += 1
            tokens_by_slot[(token[1], token[2])] += 1

        for hadm, current in cohort.items():
            subject, admission = current["subject_id"], current["admittime"]
            context = base[hadm]
            for concept, value in (
                ("age_at_admission", float(context["age_at_admission"])),
                ("male", float(context["gender"] == "M")),
                ("prior_coded_admissions", float(context["prior_coded_admission_count"])),
                ("prior_strict_aecopd_admissions", float(context["prior_strict_aecopd_count"])),
            ):
                write(_token(subject, hadm, "patient_context", "context", concept,
                             (value,), (1,), "", "v5_base_whitelist"))

            for concept in CHRONIC:
                if concept not in diagnosis[hadm] and concept not in prior_chronic[hadm]:
                    continue
                write(_token(subject, hadm, "past_medical_history", "chronic_comorbidity", concept,
                             (float(concept in diagnosis[hadm]), float(concept in prior_chronic[hadm])),
                             (1, 1), "", "current_discharge_code_proxy|prior_eligible_admission_code"))

            for concept, values, masks, value_unit in lab_tokens.get(hadm, []):
                write(_token(subject, hadm, "pertinent_results", "laboratory", concept,
                             values, masks, value_unit, "in_stay_labevents_recorded_by_discharge"))

            for concept in MED_NAMES:
                times = medications.get((hadm, concept))
                if times is None:
                    continue
                first, last = times
                write(_token(subject, hadm, "hospital_course", "administered_medication", concept,
                             (1.0, _days(first, admission), _days(last, admission), _days(last, first)),
                             (1, 1, 1, 1), "days", "in_stay_emar_not_discharge_prescription"))

            write(_token(subject, hadm, "discharge_diagnosis", "aecopd_index_code",
                         "strict_aecopd", (float(context["index_is_strict_aecopd"] in ("t", "true", "True", "1")),),
                         (1,), "", "current_discharge_diagnosis_code"))
            for concept in ACUTE:
                if concept not in diagnosis[hadm]:
                    continue
                write(_token(subject, hadm, "discharge_diagnosis", "acute_diagnosis", concept,
                             (float(concept in diagnosis[hadm]),), (1,), "",
                             "current_discharge_diagnosis_code"))

            destination = context["discharge_location"].upper()
            if destination:
                category = ("home" if "HOME" in destination else "facility" if any(
                    term in destination for term in ("REHAB", "SKILLED", "NURSING", "LONG TERM")
                ) else "other")
                write(_token(subject, hadm, "discharge_disposition", "disposition", category,
                             (1.0,), (1,), "", "v5_base_discharge_location"))

    with gzip.open(output_dir / "section_slots.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SLOT_FIELDS)
        for hadm, current in cohort.items():
            for role in SECTION_ROLES:
                ids = text_sections.get((hadm, role), [])
                writer.writerow((current["subject_id"], hadm, role,
                                 json.dumps(ids, separators=(",", ":")),
                                 nonblank[(hadm, role)], tokens_by_slot[(hadm, role)],
                                 int(hadm in note_hadms)))

    schema: dict[str, object] = {
        "cohort_admissions": len(cohort), "roles": SECTION_ROLES,
        "token_fields": TOKEN_FIELDS, "slot_fields": SLOT_FIELDS,
        "token_channels": {
            "context": ["value"],
            "chronic_comorbidity": ["coded_this_discharge", "coded_in_prior_eligible_admission"],
            "laboratory": ["first", "last", "minimum", "maximum", "last_minus_first"],
            "administered_medication": ["exposed", "first_day_from_admission", "last_day_from_admission", "exposure_span_days"],
            "aecopd_index_code": ["current_strict_aecopd_code"],
            "acute_diagnosis": ["coded_this_discharge"],
            "disposition": ["category_present"],
        },
        "structure_is_section_anchored_not_old_six_groups": True,
        "no_outcome_labels_in_outputs": True,
        "prescriptions_and_pharmacy_excluded": True,
        "diagnosis_tokens_are_positive_only": True,
        "missing_diagnosis_token_means_not_coded_not_disease_absent": True,
        "unmatched_text_only_roles_preserved": True,
        "source_sha256": {
            "aligned_qa_validation": _hash(aligned_dir / "qa_validation.json"),
            "v5_base": _hash(base_csv),
            "raw_labs": _hash(raw_dir / "labs.csv.gz"),
            "raw_diagnoses": _hash(raw_dir / "diagnoses.csv.gz"),
            "raw_emar": _hash(raw_dir / "emar.csv.gz"),
        },
    }
    (output_dir / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    report: dict[str, object] = {
        "admissions": len(cohort), "patients": len({row["subject_id"] for row in cohort.values()}),
        "roles_per_admission": len(SECTION_ROLES), "section_slots": len(cohort) * len(SECTION_ROLES),
        "structured_tokens": sum(counts.values()), "token_types": dict(counts),
        "admissions_with_note": len(note_hadms), "admissions_with_emar": len(hadms_with_emar),
        "admissions_with_any_lab": len(lab_tokens),
        "outcome_labels_exported": False,
    }
    (output_dir / "qa_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_section_inputs(args.raw_dir, args.aligned_dir, args.base_csv, args.output_dir),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
