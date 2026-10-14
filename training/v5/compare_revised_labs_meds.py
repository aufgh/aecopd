"""比较同一新版输入上的四种检验/药物组合，不与旧版指标直接相减。"""

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


ARMS = {
    "baseline": ("masked", "masked"),
    "lab_only": ("active", "masked"),
    "med_only": ("masked", "active"),
    "full": ("active", "active"),
}
PAIRS = (("baseline", "lab_only"), ("baseline", "med_only"),
         ("lab_only", "full"), ("med_only", "full"), ("baseline", "full"))
METRICS = ("integrated_brier_7_365", "mean_auc_7_365", "c_index_90", "c_index_365",
           "mean_absolute_population_curve_gap_7_365")


def run(root: Path, output: Path, *, seeds: tuple[int, ...], draws: int = 200) -> dict:
    if output.exists():
        raise FileExistsError(output)
    result: dict[str, object] = {
        "status": "completed", "comparison": "second minus first, independently retrained",
        "patient_bootstrap_draws": draws, "seeds": {},
        "limitations": ["Historical test is not blind", "eMAR route is absent for many administrations",
                        "Drug exposure is not treatment efficacy"],
    }
    for seed in seeds:
        folders = {arm: root / f"{arm}_seed{seed}" for arm in ARMS}
        reports = {arm: json.loads((folder / "report.json").read_text(encoding="utf-8"))
                   for arm, folder in folders.items()}
        reference = reports["baseline"]
        for arm, report in reports.items():
            if (report["status"] != "training_completed" or report["event_encoder"] != "abnormal_medication"
                    or report["clinical_recovery_mode"] != ARMS[arm][0]
                    or report["medication_effect_mode"] != ARMS[arm][1]
                    or report["abnormality_scale_mode"] != "training_positive_p95"):
                raise ValueError(f"incorrect run contract: {arm} seed {seed}")
            for key in (*CONTRACT_KEYS, "clinical_recovery_manifest_sha256",
                        "abnormality_scaling", "abnormality_scale_mode", "medication_effect_groups"):
                if report[key] != reference[key]:
                    raise ValueError(f"paired run differs in {key}: {arm} seed {seed}")
        result["seeds"][str(seed)] = {}
        for split in ("validation", "historical_test"):
            arrays = {arm: _predictions(folder, split) for arm, folder in folders.items()}
            first = arrays["baseline"]
            for arm, item in arrays.items():
                for key in ("subject_ids", "hadm_ids", "times", "events"):
                    if not np.array_equal(item[key], first[key]):
                        raise ValueError(f"unpaired {split} {arm}: {key}")
            scores = {(arm, cause): _summary_for_cause(first["times"], first["events"],
                                                       item["competing_cif"], cause)
                      for arm, item in arrays.items() for cause in (1, 2)}
            draws_for_patients = (_patient_draws(first["subject_ids"], draws, seed + 100)
                                  if split == "validation" else [])
            result["seeds"][str(seed)][split] = {}
            for old, new in PAIRS:
                pair = {}
                for cause, name in ((1, "readmission"), (2, "death")):
                    before, after = scores[(old, cause)], scores[(new, cause)]
                    comparison = {"first": before, "second": after,
                                  "delta_second_minus_first": {key: after[key] - before[key]
                                                               for key in METRICS}}
                    if split == "validation":
                        labels = ((first["events"][:, None] == cause)
                                  & (first["times"][:, None] <= GRID[None, :])).astype(float)
                        brier = _paired_brier(first["subject_ids"], labels,
                                              arrays[new]["competing_cif"][:, :, cause],
                                              arrays[old]["competing_cif"][:, :, cause], draws_for_patients)
                        auc = _paired_auc(labels[:, 90].astype(bool),
                                          arrays[new]["competing_cif"][:, 90, cause],
                                          arrays[old]["competing_cif"][:, 90, cause], draws_for_patients)
                        comparison["paired_bootstrap"] = {"integrated_brier": brier, "auc90": auc}
                    pair[name] = comparison
                result["seeds"][str(seed)][split][f"{new}_minus_{old}"] = pair
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps({"status": run(args.run_root, args.output, seeds=tuple(args.seeds),
                                    draws=args.bootstrap)["status"]}))
