"""Compare independently retrained, one-group-removed clinical input arms with V5 full."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from training.v5.compare_screened_group_ablations import (
    CONTRACT_KEYS,
    _paired_auc,
    _paired_brier,
    _patient_draws,
    _predictions,
    _summary_for_cause,
)
from training.v5.train import GRID

GROUPS = (
    "med_short_beta_agonist", "med_short_muscarinic_antagonist",
    "med_long_beta_agonist", "med_long_muscarinic_antagonist",
    "med_inhaled_corticosteroid", "med_systemic_corticosteroid",
    "med_common_systemic_antibacterial", "med_diuretic",
    "lab_ventilation_acid_base", "lab_inflammation_infection",
    "lab_nutrition_hematology", "lab_renal", "lab_electrolytes",
    "lab_metabolic_perfusion", "lab_cardiac_burden",
    "med_short_bronchodilators", "med_maintenance_inhalers",
    "lean_clinical", "selected_group_removal",
)
SPLITS = ("validation", "historical_test")
CAUSES = ((1, "readmission"), (2, "death"))
EXTRA_CONTRACT = (
    "clinical_recovery_manifest_sha256", "abnormality_scaling",
    "medication_effect_groups", "medication_effect_feature_count",
    "abnormality_feature_count", "abnormality_scale_mode", "raw_events_enter_model",
)


def _report(path: Path) -> dict:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "training_completed" or report["event_encoder"] != "abnormal_medication":
        raise ValueError(f"Incomplete or wrong-model run: {path}")
    return report


def _check_contract(full: dict, removed: dict, group: str) -> None:
    for key in (*CONTRACT_KEYS, *EXTRA_CONTRACT):
        if full[key] != removed[key]:
            raise ValueError(f"Non-paired run for {group}: {key}")
    if full["clinical_recovery_mode"] != "active" or full["medication_effect_mode"] != "active":
        raise ValueError("Full reference does not enable both modalities")
    expected_lab = ("without_renal_metabolic_perfusion" if group == "selected_group_removal" else
                    "core_risk" if group == "lean_clinical" else
                    f"without_{group[4:]}" if group.startswith("lab_") else "active")
    expected_med = ("without_diuretic" if group == "selected_group_removal" else
                    "core_acute" if group == "lean_clinical" else
                    f"without_{group[4:]}" if group.startswith("med_") else "active")
    if (removed["clinical_recovery_mode"], removed["medication_effect_mode"]) != (expected_lab, expected_med):
        raise ValueError(f"Wrong ablation mode for {group}")


def compare(root: Path, *, seeds: tuple[int, ...], groups: tuple[str, ...],
            draws: int = 200) -> list[dict]:
    rows = []
    for seed in seeds:
        full_dir = root / f"full_seed{seed}"
        full_report = _report(full_dir)
        full_predictions = {split: _predictions(full_dir, split) for split in SPLITS}
        for group in groups:
            removed_dir = (root / f"selected_group_removal_seed{seed}" if group == "selected_group_removal" else
                           root / f"lean_clinical_seed{seed}" if group == "lean_clinical" else
                           root / "groups" / f"without_{group}_seed{seed}")
            removed_report = _report(removed_dir)
            _check_contract(full_report, removed_report, group)
            for split in SPLITS:
                full = full_predictions[split]
                removed = _predictions(removed_dir, split)
                for key in ("subject_ids", "hadm_ids", "times", "events"):
                    if not np.array_equal(full[key], removed[key]):
                        raise ValueError(f"Unpaired {key} for {group}, seed={seed}, split={split}")
                times, events, subjects = full["times"], full["events"], full["subject_ids"]
                bootstrap = _patient_draws(subjects, draws, seed + 100) if split == "validation" else None
                for cause, cause_name in CAUSES:
                    reference = _summary_for_cause(times, events, full["competing_cif"], cause)
                    candidate = _summary_for_cause(times, events, removed["competing_cif"], cause)
                    row = {"seed": seed, "group_removed": group, "split": split, "cause": cause_name,
                           "full_brier": reference["integrated_brier_7_365"],
                           "removed_brier": candidate["integrated_brier_7_365"],
                           "delta_brier": candidate["integrated_brier_7_365"] - reference["integrated_brier_7_365"],
                           "full_auc": reference["mean_auc_7_365"],
                           "removed_auc": candidate["mean_auc_7_365"],
                           "delta_auc": candidate["mean_auc_7_365"] - reference["mean_auc_7_365"],
                           "delta_c_index_90": candidate["c_index_90"] - reference["c_index_90"],
                           "delta_c_index_365": candidate["c_index_365"] - reference["c_index_365"],
                           "delta_auc_90": candidate["points"]["90"]["auc"] - reference["points"]["90"]["auc"],
                           "delta_brier_90": candidate["points"]["90"]["brier"] - reference["points"]["90"]["brier"],
                           "delta_curve_gap": candidate["mean_absolute_population_curve_gap_7_365"]
                           - reference["mean_absolute_population_curve_gap_7_365"],
                           "delta_calibration_gap_90": candidate["points"]["90"]["absolute_mean_calibration_gap"]
                           - reference["points"]["90"]["absolute_mean_calibration_gap"]}
                    if cause == 1:
                        for key in ("30d_0.10", "90d_0.20"):
                            row[f"delta_dca_net_benefit_{key}"] = (
                                candidate["exploratory_dca_net_benefit"][key]
                                - reference["exploratory_dca_net_benefit"][key])
                    if bootstrap is not None:
                        labels = ((events[:, None] == cause) & (times[:, None] <= GRID[None, :])).astype(float)
                        row["brier_95_interval"] = _paired_brier(
                            subjects, labels, removed["competing_cif"][:, :, cause],
                            full["competing_cif"][:, :, cause], bootstrap
                        )["patient_cluster_200_draw_95_interval"]
                        row["auc_90_95_interval"] = _paired_auc(
                            labels[:, 90].astype(bool), removed["competing_cif"][:, 90, cause],
                            full["competing_cif"][:, 90, cause], bootstrap
                        )["patient_cluster_200_draw_95_interval"]
                    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows = compare(args.run_root, seeds=tuple(args.seeds), groups=tuple(args.groups))
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "paired_group_metrics.json").write_text(
        json.dumps({"direction": "removed minus full; lower Brier and higher AUC are better",
                    "bootstrap": "200 patient-cluster draws per trained seed; not training uncertainty",
                    "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "paired_group_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [key for key in rows[0] if key not in ("brier_95_interval", "auc_90_95_interval")]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for group in args.groups:
        for split in SPLITS:
            for _, cause_name in CAUSES:
                matched = [row for row in rows if row["group_removed"] == group
                           and row["split"] == split and row["cause"] == cause_name]
                summary.append({
                    "group_removed": group, "split": split, "cause": cause_name,
                    "seeds": len(matched),
                    "mean_delta_brier": float(np.mean([row["delta_brier"] for row in matched])),
                    "mean_delta_auc": float(np.mean([row["delta_auc"] for row in matched])),
                    "mean_delta_c_index_365": float(np.mean([row["delta_c_index_365"] for row in matched])),
                    "mean_delta_calibration_gap_90": float(np.mean([row["delta_calibration_gap_90"] for row in matched])),
                    "seeds_brier_worse_after_removal": sum(row["delta_brier"] > 0 for row in matched),
                    "seeds_auc_worse_after_removal": sum(row["delta_auc"] < 0 for row in matched),
                })
    with (args.output_dir / "group_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps({"rows": len(rows), "seeds": args.seeds, "groups": args.groups}, ensure_ascii=False))


if __name__ == "__main__":
    main()
