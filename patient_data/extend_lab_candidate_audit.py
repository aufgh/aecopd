"""在不覆盖原审计的前提下，补充原始检验临床候选和其重复线索。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from patient_data.audit_feature_candidates import REVIEW_LABS
from patient_data.export_mimiciv import rows


def run(audit_dir: Path, raw_root: Path) -> Path:
    source = audit_dir / "all_raw_lab_items.csv"
    output = audit_dir / "all_raw_lab_items_reviewed.csv"
    if output.exists():
        raise FileExistsError(output)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        catalog = list(reader)
    available = {row["hadm_id"]: (row["admittime"], row["dischtime"])
                 for row in rows(raw_root / "cohort/admissions.csv.gz")
                 if row["discharge_landmark_eligible"] == "t"}
    seen_time: set[tuple[str, str, str]] = set()
    seen_source: set[str] = set()
    collisions: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows(raw_root / "raw/labevents.csv.gz"):
        itemid = row["itemid"]
        boundary = available.get(row["hadm_id"])
        if itemid not in REVIEW_LABS or boundary is None:
            continue
        if not boundary[0] <= row["charttime"] <= row["storetime"] <= boundary[1]:
            continue
        time_key = (row["hadm_id"], itemid, row["charttime"])
        collisions[itemid][0] += time_key in seen_time
        collisions[itemid][1] += row["labevent_id"] in seen_source
        seen_time.add(time_key)
        seen_source.add(row["labevent_id"])
    for row in catalog:
        itemid = row["itemid"]
        if itemid in REVIEW_LABS:
            row["review_status"] = "clinical_candidate_not_in_input"
            row["review_reason"] = REVIEW_LABS[itemid]
            row["same_timestamp_collisions_not_proven_duplicates"] = str(collisions[itemid][0])
            row["repeated_source_ids"] = str(collisions[itemid][1])
        elif row["review_status"] == "unreviewed":
            # 未审计不应写成零重复。
            row["same_timestamp_collisions_not_proven_duplicates"] = ""
            row["repeated_source_ids"] = ""
        else:
            # 当前入模概念的重复信息应读取 current_event_concepts.csv。
            row["same_timestamp_collisions_not_proven_duplicates"] = ""
            row["repeated_source_ids"] = ""
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(catalog)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    args = parser.parse_args()
    print(run(args.audit_dir, args.raw_root))
