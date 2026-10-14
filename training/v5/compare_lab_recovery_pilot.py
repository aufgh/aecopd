"""同初始化、同队列比较真实异常恢复摘要与全遮蔽摘要。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.v5.compare_screened_group_ablations import (
    CONTRACT_KEYS, _paired_auc, _paired_brier, _patient_draws, _predictions,
    _summary_for_cause,
)
from training.v5.train import GRID


def run(root: Path, output: Path, *, seed: int, draws: int = 200,
        control_mode: str = "masked") -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    if control_mode not in ("masked", "measurement_only", "core9"):
        raise ValueError("unknown clinical control mode")
    masked_dir = root / f"{control_mode}_seed{seed}"
    active_dir = root / f"active_seed{seed}"
    masked_report = json.loads((masked_dir / "report.json").read_text(encoding="utf-8"))
    active_report = json.loads((active_dir / "report.json").read_text(encoding="utf-8"))
    if any(report["status"] != "training_completed" or report["event_encoder"] not in ("set", "abnormal")
           for report in (masked_report, active_report)):
        raise ValueError("incomplete or incorrect clinical summary run")
    if (masked_report["clinical_recovery_mode"] != control_mode
            or active_report["clinical_recovery_mode"] != "active"
            or masked_report["clinical_recovery_manifest_sha256"]
            != active_report["clinical_recovery_manifest_sha256"]):
        raise ValueError("clinical recovery arm contract mismatch")
    for key in CONTRACT_KEYS:
        if masked_report[key] != active_report[key]:
            raise ValueError(f"training contract differs: {key}")
    result = {"status": "single_seed_pilot", "seed": seed,
              "control_mode": control_mode,
              "delta_definition": f"active clinical summary minus identical-architecture {control_mode} summary",
              "same_cohort_model_and_source_contract": True,
              "validation_patient_cluster_bootstrap_draws": draws,
              "splits": {}}
    for split in ("validation", "historical_test"):
        masked, active = _predictions(masked_dir, split), _predictions(active_dir, split)
        for key in ("subject_ids", "hadm_ids", "times", "events"):
            if not np.array_equal(masked[key], active[key]):
                raise ValueError(f"prediction samples differ: {split} {key}")
        times, events, subjects = masked["times"], masked["events"], masked["subject_ids"]
        result["splits"][split] = {}
        draws_index = _patient_draws(subjects, draws, seed + 200) if split == "validation" else []
        for cause, name in ((1, "readmission"), (2, "death")):
            no_summary = masked["competing_cif"][:, :, cause]
            with_summary = active["competing_cif"][:, :, cause]
            old = _summary_for_cause(times, events, masked["competing_cif"], cause)
            new = _summary_for_cause(times, events, active["competing_cif"], cause)
            score = {control_mode: old, "active": new,
                     f"delta_active_minus_{control_mode}": {
                         key: new[key] - old[key] for key in
                         ("integrated_brier_7_365", "mean_auc_7_365", "c_index_90", "c_index_365",
                          "mean_absolute_population_curve_gap_7_365")}}
            if split == "validation":
                labels = ((events[:, None] == cause) & (times[:, None] <= GRID[None, :])).astype(float)
                brier = _paired_brier(subjects, labels, with_summary, no_summary, draws_index)
                auc = _paired_auc(labels[:, 90].astype(bool), with_summary[:, 90],
                                  no_summary[:, 90], draws_index)
                score["paired_brier_interval"] = {
                    f"difference_active_minus_{control_mode}": brier["difference_removed_minus_full"],
                    "patient_cluster_200_draw_95_interval": brier["patient_cluster_200_draw_95_interval"]}
                score["paired_auc_90_interval"] = {
                    f"difference_active_minus_{control_mode}": auc["difference_removed_minus_full"],
                    "patient_cluster_200_draw_95_interval": auc["patient_cluster_200_draw_95_interval"]}
            result["splits"][split][name] = score
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--control-mode", choices=("masked", "measurement_only", "core9"), default="masked")
    args = parser.parse_args()
    print(json.dumps({"status": run(args.run_root, args.output, seed=args.seed,
                                     draws=args.bootstrap, control_mode=args.control_mode)["status"]}))
