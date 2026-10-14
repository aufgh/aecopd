"""汇总药理组配对实验；不按单项指标自动筛选变量。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.v5.medication_effect_data import EFFECT_GROUPS


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "completed":
        raise ValueError(f"incomplete comparison: {path}")
    return value


def summarize(root: Path, output: Path, *, seeds: tuple[int, ...]) -> dict:
    if output.exists():
        raise FileExistsError(output)
    records = {}
    for seed in seeds:
        comparisons = {}
        for control in ("masked", "exposure_only"):
            path = root / f"comparison_{control}_seed{seed}.json"
            if path.is_file():
                comparisons[control] = _read(path)
        if "masked" not in comparisons:
            raise ValueError(f"missing whole-group comparison for seed {seed}")
        records[str(seed)] = {}
        for control, report in comparisons.items():
            if int(report["seed"]) != seed or report["control"] != control:
                raise ValueError("comparison identity mismatch")
            records[str(seed)][control] = {
                split: {cause: report["splits"][split][cause]["delta_active_minus_control"]
                        for cause in ("readmission", "death")}
                for split in ("validation", "historical_test")}
    group_ablations = {}
    for group in EFFECT_GROUPS:
        path = root / f"comparison_without_{group}_seed{seeds[0]}.json"
        if path.is_file():
            comparison = _read(path)
            group_ablations[group] = {
                "seed": seeds[0],
                "validation_readmission": comparison["splits"]["validation"]["readmission"]["delta_active_minus_control"],
                "historical_test_readmission": comparison["splits"]["historical_test"]["readmission"]["delta_active_minus_control"],
                "validation_paired_bootstrap": comparison["splits"]["validation"]["readmission"]["paired_bootstrap"],
            }
    result = {"status": "descriptive_not_final_selection", "seeds": records,
              "single_seed_group_ablations": group_ablations,
              "interpretation_boundaries": [
                  "Group ablations have one training seed; they cannot establish individual group value.",
                  "No group is automatically retained or removed from a single AUC or Brier difference.",
                  "Patient-level bootstrap intervals do not include training-seed uncertainty.",
                  "Historical test was examined before and is not a fresh blind test.",
                  "Drug exposure is confounded by clinical indication and eMAR documentation availability.",
              ]}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    args = parser.parse_args()
    summary = summarize(args.run_root, args.output, seeds=tuple(args.seeds))
    print(json.dumps({"status": summary["status"], "seeds": list(summary["seeds"]),
                      "groups": len(summary["single_seed_group_ablations"])}))
