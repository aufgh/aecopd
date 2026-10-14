"""汇总同队列新增章节消融的配对患者区间与验证/历史测试指标。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.v5.compare_strict_ablations import run as paired_compare
from training.v5.extend_field_manifest import EXTRA_LABELS
from training.v5.field_text_baseline import FIELD_LABELS


def summarize(root: Path) -> dict[str, object]:
    baseline_dir = root / "baseline_four_fields"
    baseline = json.loads((baseline_dir / "report.json").read_text(encoding="utf-8"))
    if baseline["active_text_fields"] != list(FIELD_LABELS):
        raise ValueError("four-field baseline contract mismatch")
    arms = ["plus_" + role for role in EXTRA_LABELS] + ["plus_all_five"]
    result = {"status": "same_cohort_section_incremental_comparison", "baseline": "baseline_four_fields",
              "arms": {}, "limitations": [
                  "One training seed; bootstrap intervals do not measure initialization variability.",
                  "Five single-section tests plus a combined test are exploratory and not multiplicity-corrected.",
                  "Historical test has been repeatedly inspected and is not blind.",
                  "Raw section embeddings may duplicate structured data or dilute other snippets.",
              ]}
    for name in arms:
        arm_dir = root / name
        report = json.loads((arm_dir / "report.json").read_text(encoding="utf-8"))
        expected = [*FIELD_LABELS, *EXTRA_LABELS] if name == "plus_all_five" else [*FIELD_LABELS, name.removeprefix("plus_")]
        if report["active_text_fields"] != expected or report["field_roles"] != baseline["field_roles"]:
            raise ValueError("arm field selection or model vocabulary differs: " + name)
        comparison_path = arm_dir / "paired_vs_baseline.json"
        if comparison_path.exists():
            paired = json.loads(comparison_path.read_text(encoding="utf-8"))
        else:
            paired = paired_compare(arm_dir, {"baseline": baseline_dir}, comparison_path)
        split_results = {}
        for split in ("validation", "historical_test"):
            full = report[split]["temporal_full"]
            base = baseline[split]["temporal_full"]
            paired_split = paired["comparisons"]["baseline"]["splits"][split]
            split_results[split] = {
                "mean_auc": full["mean_auc_7_365"],
                "mean_auc_delta": full["mean_auc_7_365"] - base["mean_auc_7_365"],
                "integrated_brier": full["integrated_brier_7_365"],
                "integrated_brier_delta": full["integrated_brier_7_365"] - base["integrated_brier_7_365"],
                "brier_paired_interval": paired_split["ibs_v5_minus_comparator_patient_cluster_95pct_interval"],
                "auc_30_90_365_paired": paired_split["auc_v5_minus_comparator"],
            }
        result["arms"][name] = {"active_text_fields": expected, "best_epoch": report["best_epoch"],
                                **split_results}
    target = root / "summary.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run_dir)
    print(json.dumps({name: {split: {key: values[key] for key in ("mean_auc_delta", "integrated_brier_delta")}
                              for split, values in arm.items() if split in ("validation", "historical_test")}
                      for name, arm in result["arms"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
