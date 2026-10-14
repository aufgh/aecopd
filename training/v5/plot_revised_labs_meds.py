"""绘制新版四臂实验的跨种子时间依赖 AUC 与 Brier；不平滑原始曲线。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.v5.compare_screened_group_ablations import _predictions
from training.v5.train import GRID, metrics


ARMS = ("baseline", "lab_only", "med_only", "full")
LABELS = {"baseline": "Static + text + history", "lab_only": "+ laboratory",
          "med_only": "+ medication", "full": "+ laboratory + medication"}


def plot(root: Path, *, seeds: tuple[int, ...]) -> None:
    for split in ("validation", "historical_test"):
        curves: dict[str, list[dict[str, object]]] = {arm: [] for arm in ARMS}
        identity: tuple[np.ndarray, np.ndarray] | None = None
        for arm in ARMS:
            for seed in seeds:
                record = _predictions(root / f"{arm}_seed{seed}", split)
                if identity is None:
                    identity = record["times"], record["events"]
                elif not (np.array_equal(identity[0], record["times"])
                          and np.array_equal(identity[1], record["events"])):
                    raise ValueError("different evaluated admissions or labels")
                curves[arm].append(metrics(record["times"], record["events"], record["risk"]))
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        for arm in ARMS:
            for axis, name in zip(axes, ("auc_curve", "brier_curve"), strict=True):
                values = np.stack([np.asarray(curve[name]) for curve in curves[arm]])
                line, = axis.plot(GRID[7:], np.nanmean(values[:, 7:], axis=0), label=LABELS[arm])
                axis.fill_between(GRID[7:], np.nanmin(values[:, 7:], axis=0),
                                  np.nanmax(values[:, 7:], axis=0), color=line.get_color(), alpha=.12)
        axes[0].set(xlabel="Days after discharge", ylabel="Time-dependent cumulative AUC(t)",
                    ylim=(.45, .85))
        axes[1].set(xlabel="Days after discharge", ylabel="Brier(t) (lower is better)")
        for axis in axes:
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        figure.suptitle(f"{split}: 3-seed mean and range; observed in-database events (World A)")
        figure.tight_layout()
        figure.savefig(root / f"{split}_auc_brier_three_seeds.png", dpi=160)
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    args = parser.parse_args()
    plot(args.run_root, seeds=tuple(args.seeds))
