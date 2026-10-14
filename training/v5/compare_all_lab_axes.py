"""完整22轴逐轴配对比较；固定严格结局、二元给药及两个训练种子。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from patient_data.expanded_abnormality_features import AXES
from training.v5.compare_revised_group_ablations import EXTRA_CONTRACT
from training.v5.compare_screened_group_ablations import (
    CONTRACT_KEYS, _paired_auc, _paired_brier, _patient_draws, _predictions, _summary_for_cause,
)
from training.v5.compare_strict_axis_screen import _report
from training.v5.train import GRID

LABELS = (
    "low arterial pH", "high arterial pCO2", "low bicarbonate", "high bicarbonate",
    "high white blood cells", "low albumin", "low hemoglobin", "high creatinine",
    "hypercapnic acidosis", "low white blood cells", "low sodium", "high sodium",
    "low potassium", "high potassium", "high urea nitrogen", "low glucose", "high glucose",
    "low platelets", "low magnesium", "high lactate", "high CRP", "high NT-proBNP",
)


def compare(root: Path, output: Path, *, seeds: tuple[int, ...] = (2026,),
            draws: int = 100) -> list[dict]:
    if output.exists():
        raise FileExistsError(output)
    if len(set(seeds)) != len(seeds) or not seeds or not 20 <= draws <= 500:
        raise ValueError("invalid seeds or bootstrap count")
    rows = []
    details = []
    for seed in seeds:
        full_dir = root / f"full_seed{seed}"
        full_report = _report(full_dir, lab_mode="active", medication_mode="exposure_only")
        full = {split: _predictions(full_dir, split) for split in ("validation", "historical_test")}
        reference_scores = {(split, cause): _summary_for_cause(
            data["times"], data["events"], data["competing_cif"], cause
        ) for split, data in full.items() for cause in (1, 2)}
        bootstrap = _patient_draws(full["validation"]["subject_ids"], draws, seed + 100)
        for axis in AXES:
            directory = root / f"without_axis_{axis}_seed{seed}"
            report = _report(directory, lab_mode=f"without_axis_{axis}", medication_mode="exposure_only")
            for key in (*CONTRACT_KEYS, *EXTRA_CONTRACT, "shared_text_init_width", "deterministic_inference"):
                if report[key] != full_report[key]:
                    raise ValueError(f"unpaired contract: seed={seed}, axis={axis}, key={key}")
            for split, reference in full.items():
                candidate = _predictions(directory, split)
                for key in ("subject_ids", "hadm_ids", "times", "events"):
                    if not np.array_equal(reference[key], candidate[key]):
                        raise ValueError(f"unpaired {key}: {directory}/{split}")
                times, events = reference["times"], reference["events"]
                for cause, name in ((1, "readmission"), (2, "death")):
                    ref = reference_scores[(split, cause)]
                    new = _summary_for_cause(times, events, candidate["competing_cif"], cause)
                    row = dict(seed=seed, axis=axis, split=split, cause=name,
                               reference_brier=ref["integrated_brier_7_365"],
                               candidate_brier=new["integrated_brier_7_365"],
                               delta_brier=new["integrated_brier_7_365"] - ref["integrated_brier_7_365"],
                               reference_auc=ref["mean_auc_7_365"], candidate_auc=new["mean_auc_7_365"],
                               delta_auc=new["mean_auc_7_365"] - ref["mean_auc_7_365"],
                               delta_c_index_90=new["c_index_90"] - ref["c_index_90"],
                               delta_c_index_365=new["c_index_365"] - ref["c_index_365"],
                               delta_population_curve_gap=new["mean_absolute_population_curve_gap_7_365"]
                               - ref["mean_absolute_population_curve_gap_7_365"],
                               delta_calibration_gap_90=new["points"]["90"]["absolute_mean_calibration_gap"]
                               - ref["points"]["90"]["absolute_mean_calibration_gap"])
                    detail = dict(row, reference=ref, candidate=new)
                    if split == "validation":
                        labels = ((events[:, None] == cause) & (times[:, None] <= GRID[None, :])).astype(float)
                        detail["brier_95_interval"] = _paired_brier(
                            reference["subject_ids"], labels, candidate["competing_cif"][:, :, cause],
                            reference["competing_cif"][:, :, cause], bootstrap,
                        )["patient_cluster_200_draw_95_interval"]
                        detail["auc_90_95_interval"] = _paired_auc(
                            labels[:, 90].astype(bool), candidate["competing_cif"][:, 90, cause],
                            reference["competing_cif"][:, 90, cause], bootstrap,
                        )["patient_cluster_200_draw_95_interval"]
                    rows.append(row)
                    details.append(detail)
            print(f"COMPARE DONE seed={seed} axis={axis}", flush=True)
    output.mkdir(parents=True)
    with (output / "paired_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "paired_results.json").write_text(json.dumps({
        "status": "completed", "seeds": seeds, "axes": AXES,
        "bootstrap_patient_cluster_draws": draws,
        "direction": "removed minus full; negative Brier and positive AUC indicate improvement",
        "limitations": ["Exploratory comparisons; no multiplicity adjustment.",
                        "One seed is a quick screen; it cannot establish retraining stability.",
                        "Bootstrap covers patient sampling, not retraining uncertainty.",
                        "Historical test has been repeatedly viewed and is not blind.",
                        "Metrics use the existing 365-day observed-event World-A contract; not complete out-of-hospital follow-up."],
        "rows": details,
    }, indent=2), encoding="utf-8")
    summary = []
    for axis in AXES:
        for split in ("validation", "historical_test"):
            for cause in ("readmission", "death"):
                matched = [row for row in rows if row["axis"] == axis and row["split"] == split
                           and row["cause"] == cause]
                summary.append(dict(axis=axis, split=split, cause=cause,
                                    **{key: float(np.mean([row[key] for row in matched]))
                                       for key in matched[0] if key.startswith("delta_")},
                                    brier_improved_seeds=sum(row["delta_brier"] < 0 for row in matched),
                                    auc_improved_seeds=sum(row["delta_auc"] > 0 for row in matched)))
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    plot(rows, output / "validation_readmission_all22_deltas.png", seeds=seeds)
    return rows


def plot(rows: list[dict], output: Path, *, seeds: tuple[int, ...]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if len(LABELS) != len(AXES):
        raise ValueError("axis labels do not match the fixed vocabulary")
    fig, panels = plt.subplots(1, 2, figsize=(16, 11), sharey=True)
    for panel, key, title in zip(panels, ("delta_brier", "delta_auc"), (
            "Integrated Brier: removed - full (lower better)",
            "Mean AUC(t): removed - full (higher better)")):
        all_values = []
        for index, axis in enumerate(AXES):
            matched = [row for row in rows if row["axis"] == axis and row["split"] == "validation"
                       and row["cause"] == "readmission"]
            if sorted(row["seed"] for row in matched) != sorted(seeds):
                raise ValueError(f"incomplete plot row {axis}")
            values = np.array([row[key] for row in sorted(matched, key=lambda row: row["seed"])])
            all_values.extend(values)
            if len(seeds) > 1:
                panel.scatter(values, index + np.linspace(-.08, .08, len(seeds)), color="#c05621", s=32, alpha=.65)
            panel.scatter(values.mean(), index, color="#c05621", marker="D", s=55)
        limit = max(max(abs(value) for value in all_values) * 1.13, 1e-5)
        panel.set_xlim(-limit, limit)
        panel.axvline(0, color="#444444", linewidth=1.1)
        panel.set_title(title, fontsize=11)
        panel.set_xlabel(f"Paired difference across {len(seeds)} seeds")
        panel.grid(axis="x", alpha=.18)
        panel.set_axisbelow(True)
    panels[0].set_yticks(np.arange(len(AXES)), LABELS, fontsize=10)
    panels[0].invert_yaxis()
    fig.suptitle("Strict AECOPD readmission validation; each lab axis retrained from scratch", fontsize=13, y=.985)
    fig.text(.5, .945, "All 22 axes; binary medication exposures only; no multiplicity adjustment", ha="center", fontsize=10)
    legend = ([Line2D([0], [0], marker="o", linestyle="None", color="#c05621", alpha=.65,
                      label="Individual seed")] if len(seeds) > 1 else []) + [
        Line2D([0], [0], marker="D", linestyle="None", color="#c05621",
               label="Seed mean" if len(seeds) > 1 else "Single-seed estimate"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False)
    fig.subplots_adjust(left=.22, right=.985, top=.90, bottom=.09, wspace=.12)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--bootstrap", type=int, default=100)
    args = parser.parse_args()
    compare(args.run_root, args.output_dir, seeds=tuple(args.seeds), draws=args.bootstrap)


if __name__ == "__main__":
    main()
