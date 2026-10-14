"""Compare clinical preselection and single-axis removals under the same strict endpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from training.v5.compare_revised_group_ablations import EXTRA_CONTRACT
from training.v5.compare_screened_group_ablations import (
    CONTRACT_KEYS, _paired_auc, _paired_brier, _patient_draws,
    _predictions, _summary_for_cause,
)
from training.v5.lab_recovery_data import SELECTED_AXIS_SCREEN
from training.v5.strict_endpoint_labels import STRICT_PRIMARY_EVENT
from training.v5.train import GRID

SPLITS = ("validation", "historical_test")
CAUSES = ((1, "readmission"), (2, "death"))


def _report(path: Path, *, lab_mode: str, medication_mode: str) -> dict:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    if (report["status"] != "training_completed"
            or report["event_encoder"] != "abnormal_medication"
            or report["primary_event"] != STRICT_PRIMARY_EVENT
            or report["clinical_recovery_mode"] != lab_mode
            or report["medication_effect_mode"] != medication_mode
            or report["raw_events_enter_model"]):
        raise ValueError(f"Incorrect strict endpoint/input run: {path}")
    return report


def _arm(root: Path, arm: str, seed: int) -> tuple[Path, str, str]:
    if arm == "full":
        return root / f"full_seed{seed}", "active", "active"
    if arm == "selected":
        return root / f"selected_seed{seed}", "without_renal_metabolic_perfusion", "without_diuretic"
    if arm in SELECTED_AXIS_SCREEN:
        return (root / f"without_axis_{arm}_seed{seed}",
                f"without_renal_metabolic_perfusion_plus_{arm}", "without_diuretic")
    raise ValueError(f"Unknown arm: {arm}")


def compare(root: Path, *, seeds: tuple[int, ...], axes: tuple[str, ...],
            draws: int = 200) -> list[dict]:
    if not seeds or any(axis not in SELECTED_AXIS_SCREEN for axis in axes):
        raise ValueError("Invalid seeds or screened axes")
    rows = []
    comparisons = (("full", "selected"), *(("selected", axis) for axis in axes))
    for seed in seeds:
        cache: dict[str, tuple[dict, dict[str, dict[str, np.ndarray]]]] = {}
        for reference_name, candidate_name in comparisons:
            for arm_name in (reference_name, candidate_name):
                if arm_name not in cache:
                    directory, lab_mode, med_mode = _arm(root, arm_name, seed)
                    cache[arm_name] = (_report(directory, lab_mode=lab_mode, medication_mode=med_mode),
                                       {split: _predictions(directory, split) for split in SPLITS})
            reference_report, reference_predictions = cache[reference_name]
            candidate_report, candidate_predictions = cache[candidate_name]
            for key in (*CONTRACT_KEYS, *EXTRA_CONTRACT):
                if reference_report[key] != candidate_report[key]:
                    raise ValueError(f"Unpaired {reference_name}/{candidate_name} seed={seed}: {key}")
            for split in SPLITS:
                reference = reference_predictions[split]
                candidate = candidate_predictions[split]
                for key in ("subject_ids", "hadm_ids", "times", "events"):
                    if not np.array_equal(reference[key], candidate[key]):
                        raise ValueError(f"Unpaired {key}: {reference_name}/{candidate_name}, {seed}, {split}")
                times, events, subjects = reference["times"], reference["events"], reference["subject_ids"]
                bootstrap = _patient_draws(subjects, draws, seed + 100) if split == "validation" else None
                for cause, cause_name in CAUSES:
                    ref = _summary_for_cause(times, events, reference["competing_cif"], cause)
                    new = _summary_for_cause(times, events, candidate["competing_cif"], cause)
                    row = {
                        "seed": seed, "reference": reference_name, "candidate": candidate_name,
                        "split": split, "cause": cause_name,
                        "reference_brier": ref["integrated_brier_7_365"],
                        "candidate_brier": new["integrated_brier_7_365"],
                        "delta_brier": new["integrated_brier_7_365"] - ref["integrated_brier_7_365"],
                        "reference_auc": ref["mean_auc_7_365"],
                        "candidate_auc": new["mean_auc_7_365"],
                        "delta_auc": new["mean_auc_7_365"] - ref["mean_auc_7_365"],
                        "delta_c_index_365": new["c_index_365"] - ref["c_index_365"],
                        "delta_calibration_gap_90": new["points"]["90"]["absolute_mean_calibration_gap"]
                        - ref["points"]["90"]["absolute_mean_calibration_gap"],
                    }
                    if cause == 1:
                        for key in ("30d_0.10", "90d_0.20"):
                            row[f"delta_dca_net_benefit_{key}"] = (
                                new["exploratory_dca_net_benefit"][key]
                                - ref["exploratory_dca_net_benefit"][key]
                            )
                    if bootstrap is not None:
                        labels = ((events[:, None] == cause) & (times[:, None] <= GRID[None, :])).astype(float)
                        row["brier_95_interval"] = _paired_brier(
                            subjects, labels, candidate["competing_cif"][:, :, cause],
                            reference["competing_cif"][:, :, cause], bootstrap,
                        )["patient_cluster_200_draw_95_interval"]
                        row["auc_90_95_interval"] = _paired_auc(
                            labels[:, 90].astype(bool), candidate["competing_cif"][:, 90, cause],
                            reference["competing_cif"][:, 90, cause], bootstrap,
                        )["patient_cluster_200_draw_95_interval"]
                    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--axes", nargs="+", choices=SELECTED_AXIS_SCREEN, default=list(SELECTED_AXIS_SCREEN))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows = compare(args.run_root, seeds=tuple(args.seeds), axes=tuple(args.axes))
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "paired_results.json").write_text(json.dumps({
        "direction": "candidate minus reference; lower Brier and higher AUC are better",
        "endpoint": STRICT_PRIMARY_EVENT,
        "bootstrap": "200 validation patient-cluster draws per seed; not retraining uncertainty",
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "paired_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0]
                                                  if key not in ("brier_95_interval", "auc_90_95_interval")],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for reference, candidate in (("full", "selected"), *(("selected", axis) for axis in args.axes)):
        for split in SPLITS:
            for _, cause_name in CAUSES:
                matched = [row for row in rows if row["reference"] == reference and row["candidate"] == candidate
                           and row["split"] == split and row["cause"] == cause_name]
                summary.append({
                    "reference": reference, "candidate": candidate, "split": split, "cause": cause_name,
                    "seeds": len(matched),
                    "mean_delta_brier": float(np.mean([r["delta_brier"] for r in matched])),
                    "mean_delta_auc": float(np.mean([r["delta_auc"] for r in matched])),
                    "mean_delta_c_index_365": float(np.mean([r["delta_c_index_365"] for r in matched])),
                    "brier_improved_seeds": sum(r["delta_brier"] < 0 for r in matched),
                    "auc_improved_seeds": sum(r["delta_auc"] > 0 for r in matched),
                })
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps({"rows": len(rows), "seeds": args.seeds, "axes": args.axes}))


if __name__ == "__main__":
    main()
