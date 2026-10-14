"""核对新导出的原始事件能否重建旧 V5 结构化导出；不读取结局。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Iterator


def _rows(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _cohort_parity(raw: Path, old: Path) -> dict[str, int]:
    previous = {row["hadm_id"]: row for row in _rows(old / "base.csv")}
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for row in _rows(raw / "cohort.csv.gz"):
        hadm = row["hadm_id"]
        if hadm in seen:
            counts["duplicate_new_hadm"] += 1
        seen.add(hadm)
        older = previous.get(hadm)
        if older is None:
            counts["new_hadm_not_old"] += 1
            continue
        for field in ("subject_id", "admittime", "dischtime", "index_is_strict_aecopd"):
            if row[field] != older[field]:
                counts[f"mismatch_{field}"] += 1
    counts["old_hadm_not_new"] = len(previous.keys() - seen)
    counts["new_rows"] = len(seen)
    counts["old_rows"] = len(previous)
    return dict(counts)


def _labs_parity(raw: Path, old: Path) -> dict[str, int]:
    expected: dict[tuple[str, str], tuple[int, Decimal, Decimal, Decimal, Decimal]] = {}
    for row in _rows(old / "labs.csv"):
        key = row["hadm_id"], row["itemid"]
        if key in expected:
            raise ValueError("duplicate old lab summary key")
        expected[key] = (
            int(row["measurement_count"]), *(Decimal(row[name]) for name in
                                                 ("first_value", "last_value", "min_value", "max_value")),
        )
    counts: Counter[str] = Counter()
    current_key: tuple[str, str] | None = None
    first = last = minimum = maximum = Decimal(0)
    measurement_count = 0

    def compare() -> None:
        if current_key is None:
            return
        counts["new_lab_groups"] += 1
        old_value = expected.pop(current_key, None)
        if old_value is None:
            counts["new_group_not_old"] += 1
        elif old_value != (measurement_count, first, last, minimum, maximum):
            counts["value_or_count_mismatch"] += 1

    for row in _rows(raw / "labs.csv.gz"):
        key = row["hadm_id"], row["itemid"]
        value = Decimal(row["valuenum"])
        if key != current_key:
            compare()
            current_key = key
            first = last = minimum = maximum = value
            measurement_count = 1
        else:
            last = value
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            measurement_count += 1
        counts["new_lab_events"] += 1
    compare()
    counts["old_group_not_new"] = len(expected)
    return dict(counts)


def _diagnoses_parity(raw: Path, old: Path) -> dict[str, int]:
    def key(row: dict[str, str]) -> tuple[str, str, str, str]:
        return row["hadm_id"], row["icd_version"], row["icd_code"].strip(), row["seq_num"]

    previous = Counter(key(row) for row in _rows(old / "diagnoses.csv"))
    current = Counter(key(row) for row in _rows(raw / "diagnoses.csv.gz"))
    return {
        "old_rows": sum(previous.values()), "new_rows": sum(current.values()),
        "old_only_rows": sum((previous - current).values()),
        "new_only_rows": sum((current - previous).values()),
    }


def _medications_parity(raw: Path, old: Path) -> dict[str, int]:
    previous: dict[tuple[str, str], tuple[str, str]] = {}
    for row in _rows(old / "medications.csv"):
        key = row["hadm_id"], row["medication"]
        if key in previous:
            raise ValueError("duplicate old medication summary key")
        previous[key] = row["first_charttime"], row["last_charttime"]
    current: dict[tuple[str, str], tuple[str, str]] = {}
    raw_rows = 0
    for row in _rows(raw / "emar.csv.gz"):
        key = row["hadm_id"], row["medication"].strip().lower()
        charttime = row["charttime"]
        if key in current:
            first, last = current[key]
            current[key] = min(first, charttime), max(last, charttime)
        else:
            current[key] = charttime, charttime
        raw_rows += 1
    common = previous.keys() & current.keys()
    return {
        "new_emar_events": raw_rows, "old_medication_groups": len(previous),
        "new_medication_groups": len(current),
        "old_group_not_new": len(previous.keys() - current.keys()),
        "new_group_not_old": len(current.keys() - previous.keys()),
        "first_or_last_time_mismatch": sum(previous[key] != current[key] for key in common),
    }


def audit(raw_dir: Path, old_dir: Path) -> dict[str, object]:
    components = {
        "cohort": _cohort_parity(raw_dir, old_dir),
        "labs": _labs_parity(raw_dir, old_dir),
        "diagnoses": _diagnoses_parity(raw_dir, old_dir),
        "medications": _medications_parity(raw_dir, old_dir),
    }
    mismatch = {
        "cohort": sum(value for name, value in components["cohort"].items()
                      if name.startswith(("mismatch_", "duplicate_")) or name.endswith(("not_old", "not_new"))),
        "labs": sum(components["labs"].get(name, 0) for name in
                    ("new_group_not_old", "old_group_not_new", "value_or_count_mismatch")),
        "diagnoses": components["diagnoses"]["old_only_rows"] + components["diagnoses"]["new_only_rows"],
        "medications": sum(components["medications"][name] for name in
                           ("old_group_not_new", "new_group_not_old", "first_or_last_time_mismatch")),
    }
    return {"status": "exact_parity" if not any(mismatch.values()) else "mismatch_found",
            "components": components, "mismatch_counts": mismatch,
            "old_export_reusable_as_baseline": not any(mismatch.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = audit(args.raw_dir, args.old_dir)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
