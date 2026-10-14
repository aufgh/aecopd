"""只用聚合报告绘制严格 AECOPD 增量证据图，不读取患者级预测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ("full", "history_only_cox", "no_current_static", "no_medication_events",
            "no_lab_values", "no_lab_time", "one_lab_per_concept", "no_text")
LABELS = ("Full V5", "History Cox", "No other static", "No medications",
          "No lab values", "No lab time", "One lab/assay", "No text")


def run(result_dir: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    reports = {seed: json.loads((result_dir / f"strata_seed{seed}.json").read_text(encoding="utf-8"))
               for seed in (2026, 2027, 2028)}
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = ("#263b70", "#a45416", "#6e70a3")
    x = np.arange(len(VARIANTS))
    for axis, split in zip(axes, ("validation", "historical_test")):
        for k, seed in enumerate((2026, 2027, 2028)):
            models = reports[seed]["strata"][split]["all"]["models"]
            full = models["full"]["integrated_brier_7_365"]
            differences = [full - models[name]["integrated_brier_7_365"]
                           if name in models else np.nan for name in VARIANTS]
            axis.plot(x, differences, marker="o", color=colors[k], label=str(seed))
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title("Validation" if split == "validation" else "Reused historical test")
        axis.set_xticks(x, LABELS, rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("IBS(full V5) - IBS(comparator); lower favors full")
    axes[1].legend(title="Seed")
    figure.suptitle("Strict AECOPD readmission: same cohort, endpoint, and patient split")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.result_dir, args.output)


if __name__ == "__main__":
    main()
