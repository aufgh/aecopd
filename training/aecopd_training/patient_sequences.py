"""Patient-level causal ordering and equal-weight metadata for index admissions."""

from __future__ import annotations

import csv
import gzip
import json
import os
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


REQUIRED_FIELDS = ("subject_id", "hadm_id", "data_split", "admittime", "dischtime")
OUTPUT_FIELDS = (
    "subject_id",
    "hadm_id",
    "data_split",
    "sequence_index",
    "patient_admission_count",
    "patient_loss_weight",
    "admittime",
    "dischtime",
    "days_since_previous_discharge",
)


def _parse_time(value: str, field: str) -> datetime:
    if not value:
        raise ValueError(f"Missing {field} in patient sequence source")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc


def build_patient_sequence_index(
    cohort_path: Path,
    output_path: Path,
    summary_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Order every patient's index admissions without truncating frequent users."""

    existing = [path for path in (output_path, summary_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {existing[0]}")
    if not cohort_path.is_file():
        raise FileNotFoundError(cohort_path)

    by_patient: dict[int, list[dict[str, str]]] = defaultdict(list)
    with gzip.open(cohort_path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in REQUIRED_FIELDS if field not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"Cohort table is missing fields: {', '.join(missing)}")
        for row in reader:
            by_patient[int(row["subject_id"])].append(row)

    ordered_patients: list[tuple[int, list[dict[str, str]]]] = []
    split_patients: Counter[str] = Counter()
    split_admissions: Counter[str] = Counter()
    admission_counts: list[int] = []
    for subject_id in sorted(by_patient):
        rows = by_patient[subject_id]
        splits = {row["data_split"] for row in rows}
        if len(splits) != 1:
            raise ValueError(f"Patient {subject_id} crosses data splits: {sorted(splits)}")
        rows.sort(key=lambda row: (_parse_time(row["dischtime"], "dischtime"), int(row["hadm_id"])))
        split = next(iter(splits))
        split_patients[split] += 1
        split_admissions[split] += len(rows)
        admission_counts.append(len(rows))
        ordered_patients.append((subject_id, rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    partial_output = output_path.with_name(f".{output_path.name}.{run_id}.partial")
    partial_summary = summary_path.with_name(f".{summary_path.name}.{run_id}.partial")
    try:
        with gzip.open(
            partial_output,
            "wt",
            encoding="utf-8",
            newline="",
            compresslevel=6,
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            for subject_id, rows in ordered_patients:
                count = len(rows)
                previous_discharge: datetime | None = None
                for sequence_index, row in enumerate(rows):
                    admission = _parse_time(row["admittime"], "admittime")
                    discharge = _parse_time(row["dischtime"], "dischtime")
                    if discharge < admission:
                        raise ValueError(f"dischtime precedes admittime for hadm_id={row['hadm_id']}")
                    gap = ""
                    if previous_discharge is not None:
                        gap_days = (admission - previous_discharge).total_seconds() / 86400.0
                        if gap_days < 0:
                            raise ValueError(
                                f"Overlapping index admissions for subject_id={subject_id}"
                            )
                        gap = repr(gap_days)
                    writer.writerow(
                        {
                            "subject_id": subject_id,
                            "hadm_id": row["hadm_id"],
                            "data_split": row["data_split"],
                            "sequence_index": sequence_index,
                            "patient_admission_count": count,
                            "patient_loss_weight": repr(1.0 / count),
                            "admittime": row["admittime"],
                            "dischtime": row["dischtime"],
                            "days_since_previous_discharge": gap,
                        }
                    )
                    previous_discharge = discharge

        summary: dict[str, object] = {
            "schema_version": "patient_sequence_index_v1",
            "patients": len(ordered_patients),
            "index_admissions": sum(admission_counts),
            "minimum_admissions_per_patient": min(admission_counts, default=0),
            "maximum_admissions_per_patient": max(admission_counts, default=0),
            "patients_with_at_least_5_admissions": sum(count >= 5 for count in admission_counts),
            "patients_with_at_least_10_admissions": sum(count >= 10 for count in admission_counts),
            "patients_with_at_least_20_admissions": sum(count >= 20 for count in admission_counts),
            "split_patients": {key: split_patients[key] for key in sorted(split_patients)},
            "split_admissions": {key: split_admissions[key] for key in sorted(split_admissions)},
            "ordering": "dischtime ascending, then hadm_id ascending",
            "truncation": "none",
            "loss_weight": "1 / patient index-admission count; weights sum to one within patient",
        }
        partial_summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(partial_output, output_path)
        os.replace(partial_summary, summary_path)
        return summary
    finally:
        partial_output.unlink(missing_ok=True)
        partial_summary.unlink(missing_ok=True)
