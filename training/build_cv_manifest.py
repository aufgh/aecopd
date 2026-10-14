from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def stable_hash(subject_id: int, seed: int) -> int:
    digest = hashlib.sha256(f"aecopd-cv:{seed}:{subject_id}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def stratified_patient_folds(
    rows: list[dict[str, str]], fold_count: int, seed: int
) -> dict[int, int]:
    by_subject: dict[int, Counter[str]] = {}
    for row in rows:
        subject = int(row["subject_id"])
        counts = by_subject.setdefault(subject, Counter())
        counts["patients"] = 1
        counts["admissions"] += 1
        counts[{"0": "censored", "1": "aecopd", "2": "death"}[row["event_type"]]] += 1
    fields = ("patients", "admissions", "censored", "aecopd", "death")
    targets = {
        field: sum(counts[field] for counts in by_subject.values()) / fold_count
        for field in fields
    }
    totals = [Counter() for _ in range(fold_count)]
    assignments: dict[int, int] = {}
    ordered = sorted(
        by_subject,
        key=lambda subject: (
            -by_subject[subject]["admissions"],
            -by_subject[subject]["death"],
            -by_subject[subject]["aecopd"],
            stable_hash(subject, seed),
        ),
    )
    for subject in ordered:
        counts = by_subject[subject]
        scores = []
        for fold in range(fold_count):
            score = sum(
                ((totals[fold][field] + counts[field]) / max(targets[field], 1.0)) ** 2
                for field in fields
            )
            scores.append((score, totals[fold]["patients"], fold))
        selected = min(scores)[2]
        assignments[subject] = selected
        totals[selected].update(counts)
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description="Build patient-isolated development CV folds")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("fold count must be at least two")

    rows: list[dict[str, str]] = []
    with gzip.open(args.cohort, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["data_split"] in {"train", "validation"}:
                rows.append(row)
    subjects = sorted({int(row["subject_id"]) for row in rows})
    assignments = stratified_patient_folds(rows, args.folds, args.seed)
    fold_statistics = []
    for fold in range(args.folds):
        selected = [row for row in rows if assignments[int(row["subject_id"])] == fold]
        events = Counter(row["event_type"] for row in selected)
        fold_statistics.append(
            {
                "fold": fold,
                "patients": len({int(row["subject_id"]) for row in selected}),
                "admissions": len(selected),
                "censored": events["0"],
                "aecopd": events["1"],
                "death": events["2"],
            }
        )
    payload = {
        "schema_version": "aecopd-development-patient-cv-1",
        "fold_count": args.folds,
        "seed": args.seed,
        "scope": "original train + validation; original test excluded",
        "subject_folds": {str(subject): fold for subject, fold in assignments.items()},
        "fold_statistics": fold_statistics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "subject_folds"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
