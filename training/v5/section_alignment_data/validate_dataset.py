"""独立复核 V5 章节数据的键、原文偏移、时间边界和来源哈希。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterator

from training.v5.section_alignment_data.build_dataset import patient_split


def _read(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(
    raw_dir: Path, note_source: Path, dataset_dir: Path, base_source: Path | None = None,
) -> dict[str, object]:
    export_summary = json.loads((raw_dir / "export_summary.json").read_text(encoding="utf-8"))
    build_summary = json.loads((dataset_dir / "qa_summary.json").read_text(encoding="utf-8"))
    if _sha256(note_source) != export_summary["source_note_gzip_sha256"]:
        raise ValueError("note source hash changed")
    if base_source is not None and _sha256(base_source) != build_summary["base_context_source_sha256"]:
        raise ValueError("base context source hash changed")
    for name, expected in export_summary["sources"].items():
        if _sha256(raw_dir / f"{name}.csv.gz") != expected["sha256"]:
            raise ValueError(f"raw source hash changed: {name}")

    cohort: dict[str, tuple[str, datetime, datetime, str]] = {}
    split_patients: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    for row in _read(dataset_dir / "cohort_manifest.csv.gz"):
        hadm, subject = row["hadm_id"], row["subject_id"]
        if hadm in cohort:
            raise ValueError("duplicate cohort admission")
        if row["patient_split"] != patient_split(int(subject)):
            raise ValueError("patient split drift")
        if subject in split_patients and split_patients[subject] != row["patient_split"]:
            raise ValueError("patient crosses splits")
        split_patients[subject] = row["patient_split"]
        admission = datetime.fromisoformat(row["admittime"])
        discharge = datetime.fromisoformat(row["dischtime"])
        if discharge <= admission:
            raise ValueError("invalid admission time span")
        cohort[hadm] = (subject, admission, discharge, row["patient_split"])
        split_counts[row["patient_split"]] += 1
    if len(cohort) != build_summary["cohort_admissions"]:
        raise ValueError("manifest count drift")

    section_ids: set[str] = set()
    sections_by_hadm: dict[str, list[dict[str, str]]] = {}
    role_status: Counter[str] = Counter()
    for section in _read(dataset_dir / "note_sections.csv.gz"):
        hadm, sid = section["hadm_id"], section["section_id"]
        if hadm not in cohort or sid in section_ids:
            raise ValueError("unknown admission or duplicate section ID")
        section_ids.add(sid)
        sections_by_hadm.setdefault(hadm, []).append(section)
        role_status[f"{section['section_role']}|{section['content_status']}"] += 1
    if len(section_ids) != build_summary["sections"]:
        raise ValueError("section count drift")
    note_count = 0
    for note in _read(note_source):
        hadm = note["hadm_id"]
        original = note["document_text"]
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
        for section in sections_by_hadm.pop(hadm, []):
            start, end = int(section["start_char"]), int(section["end_char"])
            if not 0 <= start <= end <= len(original):
                raise ValueError(f"out-of-bounds section offset: role={section['section_role']}, "
                                 f"start={start}, end={end}, note_length={len(original)}")
            if original[start:end] != section["text"] or section["source_sha256"] != digest:
                raise ValueError("section text or source hash mismatch")
            if section["note_id"] != note["note_id"]:
                raise ValueError("section note ID mismatch")
        note_count += 1
    if sections_by_hadm or note_count != build_summary["note_admissions"]:
        raise ValueError("unmatched section or note")

    fact_counts: Counter[str] = Counter()
    temporal_violations: Counter[str] = Counter()
    fact_count = 0
    current_source = ""
    source_row_keys: set[str] = set()
    for fact in _read(dataset_dir / "structured_facts.csv.gz"):
        fact_id, hadm = fact["fact_id"], fact["hadm_id"]
        source, number_text = fact_id.split(":", 1)
        if source != current_source:
            current_source = source
            source_row_keys.clear()
        if fact["source_row_key"] in source_row_keys:
            raise ValueError(f"duplicate source row key in {source}")
        source_row_keys.add(fact["source_row_key"])
        number = int(number_text)
        fact_counts[source] += 1
        if number != fact_counts[source]:
            raise ValueError("fact ID sequence gap or duplicate")
        if hadm not in cohort or fact["subject_id"] != cohort[hadm][0]:
            raise ValueError("fact cohort mismatch")
        if fact["concept_code"] in {"event_type", "time_days"}:
            raise ValueError("outcome label leaked into structured facts")
        if source in {"labs", "emar", "context"}:
            event = datetime.fromisoformat(fact["event_time"])
            available = datetime.fromisoformat(fact["available_time"])
            _, admission, discharge, _ = cohort[hadm]
            if not admission <= event <= discharge:
                temporal_violations[f"{source}_event_outside_stay"] += 1
            if available > discharge:
                temporal_violations[f"{source}_recorded_after_discharge"] += 1
        fact_count += 1
    if fact_count != build_summary["facts"] or dict(fact_counts) != build_summary["source_facts"]:
        raise ValueError("fact count drift")
    if temporal_violations:
        raise ValueError(f"temporal violations: {dict(temporal_violations)}")

    candidate_count = 0
    rule_counts: Counter[str] = Counter()
    for candidate in _read(dataset_dir / "candidate_alignment.csv.gz"):
        if candidate["section_id"] not in section_ids:
            raise ValueError("candidate references unknown section")
        source, number_text = candidate["fact_id"].split(":", 1)
        if source not in fact_counts or not 1 <= int(number_text) <= fact_counts[source]:
            raise ValueError("candidate references unknown fact")
        if candidate["relation_candidate"] not in {"topic_only", "topic_only_unverified_order_context"}:
            raise ValueError("candidate incorrectly states evidence")
        candidate_count += 1
        rule_counts[candidate["rule_id"]] += 1
    if candidate_count != build_summary["candidates"]:
        raise ValueError("candidate count drift")
    result: dict[str, object] = {
        "status": "structural_checks_passed", "cohort_admissions": len(cohort),
        "cohort_patients": len(split_patients), "note_admissions": note_count,
        "missing_note_admissions": len(cohort) - note_count,
        "sections": len(section_ids), "facts": fact_count, "candidates": candidate_count,
        "unmatched_facts": build_summary["unmatched_facts"],
        "patient_split_admissions": dict(split_counts),
        "candidate_rules": dict(rule_counts), "section_content_status": dict(role_status),
        "temporal_violations": dict(temporal_violations),
        "clinical_evidence_review_complete": False,
        "output_sha256": {
            name: _sha256(dataset_dir / name) for name in (
                "cohort_manifest.csv.gz", "note_sections.csv.gz",
                "structured_facts.csv.gz", "candidate_alignment.csv.gz",
            )
        },
    }
    (dataset_dir / "qa_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--note-source", type=Path, required=True)
    parser.add_argument("--base-source", type=Path)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.raw_dir, args.note_source, args.dataset_dir, args.base_source)
    print(json.dumps({k: v for k, v in result.items() if k not in {"section_content_status"}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
