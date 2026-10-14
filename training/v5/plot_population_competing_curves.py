"""画三种子完整模型的 0–365 天群体累计发生风险检查图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(root: Path, *, seeds: tuple[int, ...] = (2026, 2027, 2028)) -> tuple[Path, Path]:
    target = root / "population_competing_cif_curves.png"
    calibration_target = root / "calibration_90d_three_seeds.png"
    if target.exists() or calibration_target.exists():
        raise FileExistsError("population curve or calibration figure already exists")
    days = np.arange(366)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    calibration_fig, calibration_axes = plt.subplots(2, 2, figsize=(10, 9))
    for row, split in enumerate(("validation", "historical_test")):
        curves = []
        reference = None
        for seed in seeds:
            with np.load(root / f"full_seed{seed}" / f"{split}_predictions.npz", allow_pickle=False) as archive:
                times, events = archive["times"], archive["events"]
                subjects, admissions = archive["subject_ids"], archive["hadm_ids"]
                cif = archive["competing_cif"]
            if reference is None:
                reference = (times, events, subjects, admissions)
            elif any(not np.array_equal(a, b) for a, b in zip(reference, (times, events, subjects, admissions))):
                raise ValueError("full seed predictions are not paired on the same cohort")
            curves.append(cif)
        assert reference is not None
        times, events, _, _ = reference
        for column, (cause, label) in enumerate(((1, "Readmission"), (2, "Death"))):
            ax = axes[row, column]
            predicted = np.stack([cif[:, :, cause].mean(axis=0) for cif in curves])
            observed = ((events[:, None] == cause) & (times[:, None] <= days[None, :])).mean(axis=0)
            ax.plot(days, observed, color="black", linestyle="--", linewidth=1.8, label="Observed in database")
            ax.plot(days, predicted.mean(axis=0), color="#1f77b4", linewidth=1.8,
                    label="Mean prediction (3 seeds)")
            ax.fill_between(days, predicted.min(axis=0), predicted.max(axis=0),
                            color="#1f77b4", alpha=0.18, label="Seed range")
            ax.set_title(f"{split.replace('_', ' ').title()} — {label} (n={len(times)})")
            ax.set_ylabel("Cumulative incidence")
            ax.set_xlabel("Days since discharge")
            ax.set_xlim(0, 365)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, loc="upper left")
            risk_90 = np.mean(np.stack([cif[:, 90, cause] for cif in curves]), axis=0)
            observed_90 = ((events == cause) & (times <= 90)).astype(float)
            bins = np.array_split(np.argsort(risk_90), 10)
            mean_risk = np.array([risk_90[index].mean() for index in bins])
            event_rate = np.array([observed_90[index].mean() for index in bins])
            calibration_ax = calibration_axes[row, column]
            limit = max(.08, float(max(mean_risk.max(), event_rate.max()) * 1.08))
            calibration_ax.plot([0, limit], [0, limit], color="black", linestyle="--", label="Ideal")
            calibration_ax.plot(mean_risk, event_rate, "o-", color="#1f77b4", label="Prediction deciles")
            calibration_ax.set_title(f"{split.replace('_', ' ').title()} — {label} at day 90")
            calibration_ax.set_xlabel("Mean predicted CIF")
            calibration_ax.set_ylabel("Observed database fraction")
            calibration_ax.set_xlim(0, limit)
            calibration_ax.set_ylim(0, limit)
            calibration_ax.grid(alpha=.2)
            calibration_ax.legend(fontsize=8)
    fig.suptitle("World A: observed database events vs competing-risk CIF; historical test is not blind")
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)
    calibration_fig.suptitle("World A day-90 calibration by prediction decile; historical test is not blind")
    calibration_fig.tight_layout()
    calibration_fig.savefig(calibration_target, dpi=150)
    plt.close(calibration_fig)
    return target, calibration_target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    print(*run(args.run_root), sep="\n")
