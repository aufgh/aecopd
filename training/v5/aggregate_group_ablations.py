"""汇总配对消融的跨种子方向，不自动裁决变量留删。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def run(comparison: Path, *, expected_seeds: int = 3) -> Path:
    payload = json.loads((comparison / "paired_group_metrics.json").read_text(encoding="utf-8"))
    seeds = payload["seeds"]
    if len(seeds) != expected_seeds:
        raise ValueError(f"expected {expected_seeds} seeds, got {len(seeds)}")
    arm_sets = [set(arms) for arms in seeds.values()]
    if any(arms != arm_sets[0] for arms in arm_sets[1:]):
        raise ValueError("some ablation arms are missing across seeds")
    output = comparison / "paired_group_overview.csv"
    if output.exists():
        raise FileExistsError(output)
    records = []
    for arm in sorted(arm_sets[0]):
        results = [seeds[seed][arm]["validation"] for seed in sorted(seeds)]
        readmission = [result["readmission"] for result in results]
        death = [result["death"] for result in results]
        def values(cause: list[dict[str, object]], metric: str) -> list[float]:
            return [float(item["delta_removed_minus_full"][metric]) for item in cause]

        read_brier = values(readmission, "integrated_brier_7_365")
        read_auc = values(readmission, "mean_auc_7_365")
        read_c90 = values(readmission, "c_index_90")
        read_c365 = values(readmission, "c_index_365")
        read_curve_gap = values(readmission, "mean_absolute_population_curve_gap_7_365")
        read_cal90 = values(readmission, "absolute_mean_calibration_gap_90")
        death_brier = values(death, "integrated_brier_7_365")
        death_auc = values(death, "mean_auc_7_365")
        death_c90 = values(death, "c_index_90")
        death_c365 = values(death, "c_index_365")
        death_curve_gap = values(death, "mean_absolute_population_curve_gap_7_365")
        death_cal90 = values(death, "absolute_mean_calibration_gap_90")
        intervals = [item["brier_interval"]["patient_cluster_200_draw_95_interval"]
                     for item in readmission]
        records.append({
            "arm": arm, "seed_count": len(results),
            "readmission_delta_integrated_brier_mean": mean(read_brier),
            "readmission_delta_integrated_brier_min": min(read_brier),
            "readmission_delta_integrated_brier_max": max(read_brier),
            "seeds_removal_worsens_readmission_brier": sum(value > 0 for value in read_brier),
            "seeds_brier_interval_entirely_above_zero": sum(float(low) > 0 for low, _ in intervals),
            "seeds_brier_interval_entirely_below_zero": sum(float(high) < 0 for _, high in intervals),
            "readmission_delta_mean_auc_mean": mean(read_auc),
            "seeds_removal_worsens_readmission_auc": sum(value < 0 for value in read_auc),
            "readmission_delta_c_index_90_mean": mean(read_c90),
            "seeds_removal_worsens_readmission_c_index_90": sum(value < 0 for value in read_c90),
            "readmission_delta_c_index_365_mean": mean(read_c365),
            "seeds_removal_worsens_readmission_c_index_365": sum(value < 0 for value in read_c365),
            "readmission_delta_population_curve_gap_mean": mean(read_curve_gap),
            "seeds_removal_worsens_population_curve_gap": sum(value > 0 for value in read_curve_gap),
            "readmission_delta_calibration_gap_90_mean": mean(read_cal90),
            "seeds_removal_worsens_readmission_calibration_90": sum(value > 0 for value in read_cal90),
            "death_delta_integrated_brier_mean": mean(death_brier),
            "seeds_removal_worsens_death_brier": sum(value > 0 for value in death_brier),
            "death_delta_mean_auc_mean": mean(death_auc),
            "seeds_removal_worsens_death_auc": sum(value < 0 for value in death_auc),
            "death_delta_c_index_90_mean": mean(death_c90),
            "seeds_removal_worsens_death_c_index_90": sum(value < 0 for value in death_c90),
            "death_delta_c_index_365_mean": mean(death_c365),
            "seeds_removal_worsens_death_c_index_365": sum(value < 0 for value in death_c365),
            "death_delta_population_curve_gap_mean": mean(death_curve_gap),
            "seeds_removal_worsens_death_curve_gap": sum(value > 0 for value in death_curve_gap),
            "death_delta_calibration_gap_90_mean": mean(death_cal90),
            "seeds_removal_worsens_death_calibration_90": sum(value > 0 for value in death_cal90),
        })
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=3)
    args = parser.parse_args()
    print(run(args.comparison, expected_seeds=args.expected_seeds))
