"""Plot paired strict-endpoint validation ablations from the saved per-seed results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


COMPARISONS = (
    ("full", "selected", "renal + metabolic/perfusion + diuretic", "#2b6cb0"),
    ("selected", "arterial_hypercapnic_acidosis", "hypercapnic acidosis", "#c05621"),
    ("selected", "platelet_low", "low platelets", "#c05621"),
    ("selected", "magnesium_low", "low magnesium", "#c05621"),
    ("selected", "ntprobnp_high", "high NT-proBNP", "#c05621"),
)
SEEDS = (2026, 2027, 2028)


def load_deltas(source: Path) -> dict[tuple[str, str], list[dict[str, float]]]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["split"] == "validation" and row["cause"] == "readmission"]

    expected = {(reference, candidate) for reference, candidate, _, _ in COMPARISONS}
    actual = {(row["reference"], row["candidate"]) for row in rows}
    if actual != expected:
        raise ValueError(f"comparison mismatch: missing={expected - actual}, extra={actual - expected}")

    results: dict[tuple[str, str], list[dict[str, float]]] = {}
    for reference, candidate, _, _ in COMPARISONS:
        matches = [row for row in rows
                   if row["reference"] == reference and row["candidate"] == candidate]
        seed_values = sorted(int(row["seed"]) for row in matches)
        if seed_values != list(SEEDS):
            raise ValueError(f"expected one paired result per seed for {candidate}: {seed_values}")
        results[(reference, candidate)] = [
            {"seed": int(row["seed"]),
             "delta_brier": float(row["delta_brier"]),
             "delta_auc": float(row["delta_auc"])}
            for row in sorted(matches, key=lambda row: int(row["seed"]))
        ]
    return results


def plot(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    results = load_deltas(source)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.9), sharey=True)

    panels = (
        ("delta_brier", "Integrated Brier: candidate - reference (lower better)"),
        ("delta_auc", "Mean AUC(t): candidate - reference (higher better)"),
    )
    for axis, (field, title) in zip(axes, panels):
        all_values = [result[field]
                      for reference, candidate, _, _ in COMPARISONS
                      for result in results[(reference, candidate)]]
        limit = max(abs(value) for value in all_values) * 1.14
        axis.set_xlim(-limit, limit)
        axis.axvline(0, color="#444444", linewidth=1.1, zorder=1)
        axis.axhline(.5, color="#999999", linewidth=.8, linestyle="--", alpha=.6)
        for index, (reference, candidate, _, color) in enumerate(COMPARISONS):
            values = np.array([row[field] for row in results[(reference, candidate)]])
            axis.scatter(values, np.full(len(SEEDS), index) + [-.10, 0, .10],
                         color=color, s=32, alpha=.65, zorder=2)
            axis.scatter(values.mean(), index, color=color, marker="D", s=65, zorder=3)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Paired difference across three seeds", fontsize=10)
        axis.grid(axis="x", alpha=.18)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelsize=9)

    labels = [label for _, _, label, _ in COMPARISONS]
    axes[0].set_yticks(np.arange(len(labels)), labels, fontsize=10)
    axes[0].invert_yaxis()
    fig.suptitle("Strict AECOPD readmission validation; retrained from scratch", fontsize=13, y=.99)
    fig.text(.5, .91,
             "Blue: three-group removal vs full  |  Orange: single-axis removal vs selected  |  No multiplicity adjustment",
             ha="center", fontsize=10)
    legend = (
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#777777",
               markeredgecolor="none", alpha=.65, markersize=6, label="Individual seed"),
        Line2D([0], [0], marker="D", linestyle="None", markerfacecolor="#777777",
               markeredgecolor="none", markersize=7, label="Three-seed mean"),
    )
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .01),
               ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.26, right=.985, top=.83, bottom=.16, wspace=.1)
    output.parent.mkdir(parents=True, exist_ok=True)
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
