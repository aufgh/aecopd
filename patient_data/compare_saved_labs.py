"""将新检验事件与旧已用检验逐源记录核对，只输出聚合结果。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from export_mimiciv import rows, sha256


def compare(old_path: Path, output: Path) -> dict[str, object]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "preprocessed_and_checked":
        raise ValueError("预处理尚未校验完成")
    event_path = output / "events.csv.gz"
    if sha256(event_path) != manifest["output_sha256"][event_path.name]:
        raise ValueError("预处理事件文件变化")
    old = {r["labevent_id"]: r for r in rows(old_path)}
    remaining = set(old)
    added, kept = 0, 0
    for row in rows(event_path):
        if row["kind"] != "lab":
            continue
        previous = old.get(row["source_id"])
        if previous is None:
            added += 1
            continue
        if (row["hadm_id"] != previous["hadm_id"] or row["subject_id"] != previous["subject_id"]
                or float(row["value_raw"]) != float(previous["valuenum"])
                or row["value_unit"] != previous["valueuom"]
                or datetime.fromisoformat(row["charttime"]) != datetime.fromisoformat(previous["charttime"])):
            raise ValueError("新旧同一原始记录的患者／住院／数值／单位／时间不一致")
        remaining.discard(row["source_id"])
        kept += 1
    excluded = {r["source_id"]: r["reason"] for r in rows(output / "excluded_records.csv.gz") if r["source_table"] == "labevents"}
    if remaining - set(excluded):
        raise ValueError("旧检验有未说明原因的遗漏")
    result = {"status": "source_identity_checked", "old_lab_records": len(old), "old_records_retained": kept,
              "newly_added_records": added, "old_records_excluded": len(remaining),
              "exclusion_reasons": Counter(excluded[source] for source in remaining),
              "raw_values_units_times_and_members_unchanged": True,
              "old_source_sha256": sha256(old_path), "new_events_sha256": sha256(event_path)}
    (output / "source_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-labs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.old_labs, args.output)))
