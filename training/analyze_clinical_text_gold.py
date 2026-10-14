"""人工真值核对后的规则错误分析；未完成标注时拒绝给出准确率。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ALLOWED_VALUES = {
    "discharge_oxygen_support": {"0", "1", "unknown"},
    "new_or_escalated_home_oxygen": {"0", "1", "unknown"},
    "discharge_mobility": {"independent", "assisted", "bedbound", "unknown"},
    "residual_wheeze": {"0", "1", "unknown"},
    "discharge_steroid_taper": {"0", "1", "unknown"},
    "persistent_consolidation": {"0", "1", "unknown"},
    "pulmonary_edema": {"0", "1", "unknown"},
}
ALLOWED_TIMES = {"discharge", "past", "admission", "future", "unknown", "not_mentioned"}
ALLOWED_ASSERTIONS = {"affirmed", "negated", "conditional", "not_mentioned", "unknown"}


def analyze(gold_path: Path, rules_path: Path) -> dict[str, object]:
    with gold_path.open(encoding="utf-8-sig", newline="") as handle:
        gold_rows = list(csv.DictReader(handle))
    with rules_path.open(encoding="utf-8-sig", newline="") as handle:
        rule_rows = list(csv.DictReader(handle))
    rules = {(row["review_id"], row["field"]): row for row in rule_rows}
    if len(rules) != len(rule_rows) or len(gold_rows) != len(rule_rows):
        raise ValueError("duplicate or unmatched gold/rule rows")
    incomplete: list[str] = []
    by_field: dict[str, Counter[str]] = defaultdict(Counter)
    errors: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in gold_rows:
        key = row["review_id"], row["field"]
        if key in seen or key not in rules:
            raise ValueError(f"duplicate or unmatched gold key: {key}")
        seen.add(key)
        if row["review_complete"] != "1":
            incomplete.append(f"{key[0]}:{key[1]}")
            continue
        field = row["field"]
        if (
            field not in ALLOWED_VALUES
            or row["gold_value"] not in ALLOWED_VALUES[field]
            or row["gold_time"] not in ALLOWED_TIMES
            or row["gold_assertion"] not in ALLOWED_ASSERTIONS
            or not row["reviewer"].strip()
        ):
            raise ValueError(f"invalid completed gold annotation: {key}")
        predicted = rules[key]["rule_value"]
        if predicted not in ALLOWED_VALUES[field]:
            raise ValueError(f"invalid rule value: {key}")
        counts = by_field[field]
        counts["total"] += 1
        counts["gold_known"] += int(row["gold_value"] != "unknown")
        counts["rule_known"] += int(predicted != "unknown")
        counts["correct"] += int(predicted == row["gold_value"])
        if predicted != row["gold_value"]:
            category = (
                "missed_information" if predicted == "unknown"
                else "unsupported_assertion" if row["gold_value"] == "unknown"
                else "wrong_value"
            )
            counts[category] += 1
            errors.append({
                "review_id": key[0], "field": field, "error_type": category,
                "rule_value": predicted, "gold_value": row["gold_value"],
                "gold_time": row["gold_time"], "gold_assertion": row["gold_assertion"],
            })
    if incomplete:
        raise ValueError(
            f"manual gold incomplete: {len(incomplete)}/{len(gold_rows)} rows; "
            "accuracy and model training are not authorized"
        )
    return {
        "gold_rows": len(gold_rows),
        "by_field": {field: dict(counts) for field, counts in sorted(by_field.items())},
        "errors": errors,
        "error_rows_include_raw_text": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = analyze(args.gold, args.rules)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gold_rows": report["gold_rows"], "errors": len(report["errors"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
