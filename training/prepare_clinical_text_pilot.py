"""从本机审阅包生成盲法人工标注模板与规则预测，不导出原文。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from training.clinical_text_rules import (
    extract_discharge,
    extract_radiology,
    has_possible_post_discharge_fact,
)


DISCHARGE_FIELDS = (
    "discharge_oxygen_support",
    "new_or_escalated_home_oxygen",
    "discharge_mobility",
    "residual_wheeze",
    "discharge_steroid_taper",
)
RADIOLOGY_FIELDS = ("persistent_consolidation", "pulmonary_edema")


def select_reviews(manifest: list[dict[str, str]]) -> list[dict[str, str]]:
    """各队列角色和长度层等额抽样；不接触患者结局。"""

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        groups[(row["document_type"], row["cohort_group"], row["length_band"])].append(row)
    selected: list[dict[str, str]] = []
    for key, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: row["review_id"])
        count = 4 if key[:2] == ("discharge", "strict_index") else 2
        if len(ordered) < count:
            raise ValueError(f"not enough source reviews in stratum {key}")
        indices = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
        selected.extend(ordered[index] for index in indices)
    if len(selected) != 24 or len({row["review_id"] for row in selected}) != 24:
        raise ValueError("unexpected pilot review count")
    return sorted(selected, key=lambda row: row["review_id"])


def prepare(review_dir: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    with (review_dir / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        selected = select_reviews(list(csv.DictReader(handle)))
    output_dir.mkdir(parents=True)
    gold_rows: list[dict[str, str]] = []
    rule_rows: list[dict[str, str]] = []
    leak_flags = 0
    for item in selected:
        source_path = (review_dir / item["filename"]).resolve()
        if source_path.parent != review_dir.resolve():
            raise ValueError("review filename escapes source directory")
        original = source_path.read_text(encoding="utf-8")
        document_type = item["document_type"]
        fields = DISCHARGE_FIELDS if document_type == "discharge" else RADIOLOGY_FIELDS
        findings = extract_discharge(original) if document_type == "discharge" else extract_radiology(original)
        possible_future = has_possible_post_discharge_fact(original)
        leak_flags += int(possible_future)
        for field in fields:
            finding = findings[field]
            gold_rows.append({
                "review_id": item["review_id"], "document_type": document_type,
                "length_band": item["length_band"], "field": field,
                "gold_value": "", "gold_time": "", "gold_assertion": "",
                "reviewer": "", "review_complete": "",
            })
            rule_rows.append({
                "review_id": item["review_id"], "field": field,
                "rule_value": finding.value, "evidence_section": finding.section,
                "evidence_line": "" if finding.line is None else str(finding.line),
                "possible_post_discharge_fact": str(int(possible_future)),
            })
    with (output_dir / "gold_template.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(gold_rows[0]))
        writer.writeheader()
        writer.writerows(gold_rows)
    with (output_dir / "rule_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rule_rows[0]))
        writer.writeheader()
        writer.writerows(rule_rows)
    summary: dict[str, object] = {
        "documents": len(selected), "field_judgments": len(gold_rows),
        "document_groups": dict(Counter(f"{row['document_type']}:{row['cohort_group']}" for row in selected)),
        "possible_post_discharge_fact_documents": leak_flags,
        "rule_nonunknown_by_field": dict(Counter(row["field"] for row in rule_rows if row["rule_value"] != "unknown")),
        "gold_complete": False,
        "no_outcome_labels_used": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.review_dir, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
