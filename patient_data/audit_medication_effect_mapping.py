"""审计药理映射的原始药名、途径可见性和类别；不修改源记录。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def _rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def audit(raw: Path, processed: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "preprocessed_and_checked" or manifest.get("medication_mapping") not in (
            "gold_review_v2", "clinical_reviewed_2026_09"):
        raise ValueError("expected revised, validated medication mapping")
    digest = hashlib.sha256((processed / "events.csv.gz").read_bytes()).hexdigest()
    if digest != manifest["output_sha256"]["events.csv.gz"]:
        raise ValueError("processed medication events changed after validation")
    accepted: dict[str, set[str]] = defaultdict(set)
    for row in _rows(processed / "events.csv.gz"):
        if row["kind"] == "medication":
            accepted[row["source_id"]].add(row["concept"])
    names: dict[str, tuple[str, str]] = {}
    for row in _rows(raw / "raw/emar.csv.gz"):
        key = row["emar_id"] + ":" + row["emar_seq"]
        if key in accepted:
            if key in names:
                raise ValueError("duplicate accepted eMAR key")
            names[key] = row["medication"], row["hadm_id"]
    if set(names) != set(accepted):
        raise ValueError("accepted eMAR sources missing from original data")
    routes: dict[str, set[str]] = defaultdict(set)
    for row in _rows(raw / "raw/emar_detail.csv.gz"):
        key = row["emar_id"] + ":" + row["emar_seq"]
        if key in names and row["route"] not in ("", "\\N"):
            routes[key].add(row["route"].strip().upper())
    counts: Counter[tuple[str, str]] = Counter()
    admissions: dict[tuple[str, str], set[str]] = defaultdict(set)
    missing_routes: Counter[tuple[str, str]] = Counter()
    for key, (name, hadm) in names.items():
        group = (name, "|".join(sorted(accepted[key])))
        counts[group] += 1
        admissions[group].add(hadm)
        missing_routes[group] += not routes[key]
    output.mkdir(parents=True)
    with (output / "accepted_medication_names.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("原始药名", "映射类别", "实际给药来源数", "覆盖住院数", "无途径记录来源数"))
        for group, count in counts.most_common():
            writer.writerow((*group, count, len(admissions[group]), missing_routes[group]))
    summary = {"mapping": manifest["medication_mapping"],
               "processed_events_sha256": manifest["output_sha256"]["events.csv.gz"],
               "accepted_unique_administrations": len(names),
               "accepted_names": len(counts), "administrations_without_route": sum(missing_routes.values()),
               "concept_events": dict(Counter(concept for classes in accepted.values() for concept in classes))}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.raw, args.processed, args.output), ensure_ascii=False))
