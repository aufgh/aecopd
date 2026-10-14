"""核对旧基础表内嵌标签与现行标签的差别，不输出患者级记录。"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


def audit(base_csv: Path, current_labels: Path, run_dir: Path) -> dict[str, object]:
    with base_csv.open(encoding="utf-8-sig", newline="") as handle:
        base = list(csv.DictReader(handle))
    with gzip.open(current_labels, "rt", encoding="utf-8", newline="") as handle:
        labels = {r["hadm_id"]: r for r in csv.DictReader(handle)}
    if set(labels) != {r["hadm_id"] for r in base}:
        raise ValueError("旧基础表与现行标签住院成员不同")
    model_audit = json.loads((run_dir / "model_data_audit.json").read_text(encoding="utf-8"))
    result = {"cohort_admissions": len(base),
              "different_embedded_event_type": sum(int(r["event_type"]) != int(labels[r["hadm_id"]]["event_type"]) for r in base),
              "different_embedded_event_time": sum(abs(float(r["time_days"]) - float(labels[r["hadm_id"]]["time_days"])) > 1e-6 for r in base),
              "embedded_outcomes_used_for_training": False,
              "training_labels_match_current_raw_ledger": model_audit["current_labels_identical_to_raw_ledger"],
              "policy": "explicit current labels override obsolete embedded labels; raw-ledger parity must pass before training"}
    (run_dir / "embedded_label_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--current-labels", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.base_csv, args.current_labels, args.run_dir)))
