"""Plot paired validation deltas across three seeds; points are runs, diamonds are means."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["split"] == "validation" and row["cause"] == "readmission"]
    groups = list(dict.fromkeys(row["group_removed"] for row in rows))
    if not groups or any(sum(row["group_removed"] == group for row in rows) != 3 for group in groups):
        raise ValueError("each group requires exactly three paired seed results")
    labels = [group.removeprefix("med_").removeprefix("lab_").replace("_", " ") for group in groups]
    fig, axes = plt.subplots(1, 2, figsize=(15, max(8, len(groups) * .46)), sharey=True)
    for axis, field, title in zip(axes, ("delta_brier", "delta_auc"),
                                  ("Integrated Brier: removed - full (lower better)",
                                   "Mean AUC(t): removed - full (higher better)")):
        axis.axvline(0, color="#444444", linewidth=1)
        for index, group in enumerate(groups):
            values = np.array([float(row[field]) for row in rows if row["group_removed"] == group])
            color = "#2b6cb0" if group.startswith("med_") else "#c05621"
            axis.scatter(values, np.full(3, index) + np.array([-.12, 0, .12]),
                         color=color, s=16, alpha=.65, zorder=2)
            axis.scatter(values.mean(), index, color=color, marker="D", s=35, zorder=3)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Paired difference across three seeds")
        axis.grid(axis="x", alpha=.2)
    axes[0].set_yticks(np.arange(len(groups)), labels)
    axes[0].invert_yaxis()
    fig.suptitle("Readmission validation; groups retrained from scratch; no multiplicity adjustment", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.output)


if __name__ == "__main__":
    main()
