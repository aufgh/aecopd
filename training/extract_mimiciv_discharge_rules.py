"""对完整 MIMIC-IV 严格索引小结运行保守规则并输出受控聚合审计。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from training.clinical_text_rules import extract_discharge, has_possible_post_discharge_fact
from training.prepare_clinical_text_pilot import DISCHARGE_FIELDS


def extract(source: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    timestamp_counts: Counter[str] = Counter()
    future_flags = 0
    seen: set[int] = set()
    feature_path = output_dir / "features.csv"
    with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as source_handle, feature_path.open(
        "w", encoding="utf-8", newline=""
    ) as feature_handle:
        reader = csv.DictReader(source_handle)
        required = {"subject_id", "hadm_id", "timestamp_relation", "document_text"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("source text contract is incomplete")
        writer = csv.DictWriter(feature_handle, fieldnames=(
            "subject_id", "hadm_id", "timestamp_relation", "possible_future_fact", *DISCHARGE_FIELDS,
        ))
        writer.writeheader()
        for row in reader:
            hadm = int(row["hadm_id"])
            if hadm in seen:
                raise ValueError("duplicate discharge admission")
            seen.add(hadm)
            findings = extract_discharge(row["document_text"])
            possible_future = has_possible_post_discharge_fact(row["document_text"])
            future_flags += int(possible_future)
            timestamp_counts[row["timestamp_relation"]] += 1
            item = {
                "subject_id": row["subject_id"], "hadm_id": row["hadm_id"],
                "timestamp_relation": row["timestamp_relation"],
                "possible_future_fact": int(possible_future),
            }
            for name in DISCHARGE_FIELDS:
                value = findings[name].value
                item[name] = value
                counts[name][value] += 1
            writer.writerow(item)
    summary: dict[str, object] = {
        "admissions_with_note": len(seen),
        "timestamp_relation": dict(timestamp_counts),
        "possible_post_discharge_fact_flags": future_flags,
        "field_value_counts": {name: dict(counts[name]) for name in DISCHARGE_FIELDS},
        "rule_accuracy_estimated": False,
        "outcome_labels_used": False,
        "training_ready": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.source, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
