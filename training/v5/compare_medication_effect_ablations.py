"""同种子、同队列配对比较药理作用组；不将给药暴露解释为疗效。"""

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


def run(root: Path, output: Path, *, seed: int, control: str = "masked", draws: int = 200) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if control not in ("masked", "exposure_only") and not control.startswith("without_"):
        raise ValueError("invalid medication comparison control")
    before = root / f"{control}_seed{seed}"
    after = root / f"active_seed{seed}"
    reports = [json.loads((path / "report.json").read_text(encoding="utf-8")) for path in (before, after)]
    if any(r["status"] != "training_completed" or r["event_encoder"] != "abnormal_medication"
           or r["clinical_recovery_mode"] != "active" for r in reports):
        raise ValueError("incomplete medication effect run")
    if reports[0]["medication_effect_mode"] != control or reports[1]["medication_effect_mode"] != "active":
        raise ValueError("incorrect medication effect modes")
    for key in (*CONTRACT_KEYS, "clinical_recovery_manifest_sha256", "medication_effect_groups",
                "medication_effect_feature_count"):
        if reports[0][key] != reports[1][key]:
            raise ValueError(f"paired contract differs: {key}")
    result = {"status": "completed", "seed": seed, "control": control,
              "comparison": "active minus control, independently trained from same seed",
              "limitations": ["Patient-cluster intervals condition on this seed and split.",
                              "Historical test is not blind.",
                              "Medication is an administration proxy, not a treatment effect or indication."],
              "splits": {}}
    for split in ("validation", "historical_test"):
        old, new = _predictions(before, split), _predictions(after, split)
        for key in ("subject_ids", "hadm_ids", "times", "events"):
            if not np.array_equal(old[key], new[key]):
                raise ValueError(f"different paired predictions: {key}")
        subjects, times, events = old["subject_ids"], old["times"], old["events"]
        patient_draws = _patient_draws(subjects, draws, seed + 100) if split == "validation" else []
        result["splits"][split] = {}
        for cause, name in ((1, "readmission"), (2, "death")):
            baseline = _summary_for_cause(times, events, old["competing_cif"], cause)
            candidate = _summary_for_cause(times, events, new["competing_cif"], cause)
            comparison = {"control": baseline, "active": candidate,
                          "delta_active_minus_control": {key: candidate[key] - baseline[key]
                              for key in ("mean_auc_7_365", "integrated_brier_7_365", "c_index_90",
                                          "c_index_365", "mean_absolute_population_curve_gap_7_365")}}
            if split == "validation":
                labels = ((events[:, None] == cause) & (times[:, None] <= GRID[None, :])).astype(float)
                brier = _paired_brier(subjects, labels, new["competing_cif"][:, :, cause],
                                      old["competing_cif"][:, :, cause], patient_draws)
                auc = _paired_auc(labels[:, 90].astype(bool), new["competing_cif"][:, 90, cause],
                                  old["competing_cif"][:, 90, cause], patient_draws)
                comparison["paired_bootstrap"] = {
                    "brier_active_minus_control": {
                        "point_difference": brier["difference_removed_minus_full"],
                        "patient_cluster_200_draw_95_interval": brier["patient_cluster_200_draw_95_interval"]},
                    "auc90_active_minus_control": {
                        "point_difference": auc["difference_removed_minus_full"],
                        "patient_cluster_200_draw_95_interval": auc["patient_cluster_200_draw_95_interval"]}}
            result["splits"][split][name] = comparison
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--control", default="masked")
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps({"status": run(args.run_root, args.output, seed=args.seed,
                                    control=args.control, draws=args.bootstrap)["status"]}))
