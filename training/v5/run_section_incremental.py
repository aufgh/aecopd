"""固定扩展字段词表，逐一训练新增小结章节与四字段基线。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from training.v5.extend_field_manifest import EXTRA_LABELS
from training.v5.field_text_baseline import FIELD_LABELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned_dir", "structured_dir", "base_csv", "processed_event_dir",
                 "raw_ledger_dir", "field_dir", "vectors", "labels_csv", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--only-arms", nargs="+", help="Subset of arm names for replication")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = tuple(FIELD_LABELS)
    arms = {"baseline_four_fields": baseline}
    arms.update({"plus_" + role: (*baseline, role) for role in EXTRA_LABELS})
    arms["plus_all_five"] = (*baseline, *EXTRA_LABELS)
    if args.only_arms is not None:
        if len(set(args.only_arms)) != len(args.only_arms) or set(args.only_arms) - set(arms):
            raise ValueError("unknown or duplicate replication arm")
        arms = {name: fields for name, fields in arms.items() if name in args.only_arms}
    state_path = args.output_dir / "progress.json"
    expected = {name: list(fields) for name, fields in arms.items()}
    if state_path.exists():
        progress = json.loads(state_path.read_text(encoding="utf-8"))
        if progress["seed"] != args.seed or progress["arms"] != expected:
            raise ValueError("existing experiment contract differs")
    else:
        progress = {"status": "running", "seed": args.seed, "arms": expected, "completed": []}
        state_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    shared = [
        "--aligned-dir", str(args.aligned_dir), "--structured-dir", str(args.structured_dir),
        "--base-csv", str(args.base_csv), "--processed-event-dir", str(args.processed_event_dir),
        "--raw-ledger-dir", str(args.raw_ledger_dir), "--field-dir", str(args.field_dir),
        "--vectors", str(args.vectors), "--labels-csv", str(args.labels_csv),
        "--event-encoder", "set", "--seed", str(args.seed),
    ]
    for name, fields in arms.items():
        run_dir = args.output_dir / name
        if name in progress["completed"]:
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            if report["status"] != "training_completed" or report["active_text_fields"] != list(fields):
                raise ValueError("completed arm no longer matches contract: " + name)
            continue
        if run_dir.exists():
            raise FileExistsError("incomplete arm requires inspection: " + str(run_dir))
        command = [sys.executable, "-m", "training.v5.train_event_sequence", *shared,
                   "--output-dir", str(run_dir), "--active-text-fields", ",".join(fields)]
        print(json.dumps({"starting_arm": name, "seed": args.seed}), flush=True)
        subprocess.run(command, check=True)
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        if report["status"] != "training_completed" or report["active_text_fields"] != list(fields):
            raise ValueError("arm did not complete as requested: " + name)
        progress["completed"].append(name)
        state_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    progress["status"] = "training_completed"
    state_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    print(json.dumps({"status": progress["status"], "completed": progress["completed"]}), flush=True)


if __name__ == "__main__":
    main()
