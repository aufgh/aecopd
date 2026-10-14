"""核对当前已入模检验的参考区间与可用于恢复轨迹的重复观测。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def numeric(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def audit(processed: Path, raw: Path) -> dict[str, object]:
    accepted = {}
    for row in rows(processed / "events.csv.gz"):
        if row["kind"] == "lab":
            accepted[row["source_id"]] = (row["hadm_id"], row["concept"],
                                           float(row["elapsed_days"]), float(row["value_raw"]))
    result = defaultdict(Counter)
    by_stay = defaultdict(list)
    found = 0
    for row in rows(raw / "raw/labevents.csv.gz"):
        key = row["labevent_id"]
        if key not in accepted:
            continue
        hadm, concept, days, value = accepted[key]
        if row["hadm_id"] != hadm or numeric(row["valuenum"]) != value:
            raise ValueError(f"accepted/raw laboratory row disagrees: {key}")
        found += 1
        counts = result[concept]
        counts["observations"] += 1
        lower, upper = numeric(row["ref_range_lower"]), numeric(row["ref_range_upper"])
        if lower is not None and upper is not None and lower < upper:
            counts["valid_two_sided_reference"] += 1
            state = "low" if value < lower else "high" if value > upper else "normal"
            counts[f"reference_{state}"] += 1
        if row["flag"].strip():
            counts["nonempty_flag"] += 1
            counts["flag_" + row["flag"].strip().lower()] += 1
        by_stay[(hadm, concept)].append((days, value))
    if found != len(accepted):
        raise ValueError(f"only {found} of {len(accepted)} accepted laboratory IDs found")
    for (_, concept), observations in by_stay.items():
        counts = result[concept]
        counts["admissions_with_any"] += 1
        if len(observations) >= 2:
            counts["admissions_with_at_least_two"] += 1
            days = [day for day, _ in observations]
            if max(days) - min(days) >= 1:
                counts["admissions_with_24h_span"] += 1
    return {"accepted_lab_events": found,
            "concepts": {concept: dict(counts) for concept, counts in sorted(result.items())}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(audit(args.processed, args.raw), ensure_ascii=False, indent=2),
                           encoding="utf-8")
