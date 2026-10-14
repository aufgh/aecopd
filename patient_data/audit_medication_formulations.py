"""聚合审查现行 eMAR 药名在详情表中的制剂和途径，不输出患者行。"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from patient_data.export_mimiciv import NULL, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    names = {}
    for row in rows(args.raw_root / "raw/emar.csv.gz"):
        if row["event_txt"] in {"Administered", "Started", "Applied", "Delayed Administered"}:
            names[row["emar_id"], row["emar_seq"]] = row["medication"].strip()
    combinations: Counter[tuple[str, str, str]] = Counter()
    per_source_seen: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in rows(args.raw_root / "raw/emar_detail.csv.gz"):
        key = row["emar_id"], row["emar_seq"]
        if key not in names:
            continue
        route = row["route"].strip().upper() if row["route"] not in (NULL, "") else "UNKNOWN"
        product = row["product_description"].strip() if row["product_description"] not in (NULL, "") else "UNKNOWN"
        pair = route, product
        if pair not in per_source_seen[key]:
            combinations[names[key], route, product] += 1
            per_source_seen[key].add(pair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("medication_name", "route", "product_description", "emar_sources"))
        for (name, route, product), count in sorted(combinations.items(), key=lambda kv: -kv[1]):
            writer.writerow((name, route, product, count))
    print(f"wrote {len(combinations)} aggregate formulations")


if __name__ == "__main__":
    main()
