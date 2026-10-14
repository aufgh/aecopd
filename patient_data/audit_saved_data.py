"""复核原始检验字段与标本可追溯性，仅输出聚合计数。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from export_mimiciv import NULL, rows


def audit(root: Path) -> dict[str, object]:
    specimen_types: dict[tuple[str, str], str] = {}
    blood_gases: list[tuple[str, str, str]] = []
    items: Counter[str] = Counter()
    fields: set[str] = set()
    for row in rows(root / "raw/labevents.csv.gz"):
        fields.update(row)
        items[row["itemid"]] += 1
        key = (row["subject_id"], row["specimen_id"])
        if row["itemid"] == "52033" and row["value"] != NULL:
            value = row["value"].strip().upper()
            previous = specimen_types.get(key)
            # 一份标本类型冲突时不能按最后一条覆盖。
            specimen_types[key] = value if previous in (None, value) else "CONFLICT"
        elif row["itemid"] in ("50818", "50820", "50821"):
            blood_gases.append((row["itemid"], *key))
    required = {"labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid",
                "charttime", "storetime", "value", "valuenum", "valueuom",
                "ref_range_lower", "ref_range_upper", "flag", "comments"}
    if not required.issubset(fields):
        raise ValueError("检验原始字段不完整")
    by_type: dict[str, Counter[str]] = {item: Counter() for item in ("50818", "50820", "50821")}
    for item, subject, specimen in blood_gases:
        by_type[item][specimen_types.get((subject, specimen), "UNKNOWN")] += 1
    if not specimen_types:
        raise ValueError("未保存标本类型，不能将血气视为动脉血气")
    result = {"status": "checked", "distinct_lab_items": len(items), "lab_records": sum(items.values()),
              "required_lab_fields_present": True, "blood_gas_specimen_type_counts": by_type,
              "note": "本报告仅审查原始底账，不清洗或重建模型输入；UNKNOWN/CONFLICT 必须由派生层明确处理。"}
    (root / "metadata/raw_lab_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), ensure_ascii=False))
