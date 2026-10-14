"""将早期检验异常三种子实验画成配对差值图，不与新版药物实验混比。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


COMPARISONS = (
    ("原 9 轴异常 - 无检验", "core9", "masked"),
    ("仅测量过程 - 无检验", "measurement_only", "masked"),
    ("完整 22 轴 - 无检验", "active", "masked"),
    ("完整 22 轴 - 原 9 轴", "active", "core9"),
    ("完整 22 轴 - 仅测量过程", "active", "measurement_only"),
)


def plot(source: Path, output: Path, *, font_path: Path | None = None) -> None:
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads(source.read_text(encoding="utf-8"))
    if (summary["status"] != "exploratory_multi_seed" or summary["cohort_patients"] != 5351
            or summary["cohort_admissions"] != 19401):
        raise ValueError("unexpected laboratory experiment cohort or status")
    by_seed = {}
    for row in summary["validation_rows"]:
        key = (int(row["seed"]), row["mode"])
        if key in by_seed:
            raise ValueError(f"duplicate arm: {key}")
        by_seed[key] = row
    seeds = tuple(summary["seeds"])
    if len(seeds) != 3 or set(by_seed) != {(seed, mode) for seed in seeds
                                            for mode in ("masked", "core9", "measurement_only", "active")}:
        raise ValueError("expected four completed arms for each of three seeds")
    if font_path is not None:
        if not font_path.is_file():
            raise FileNotFoundError(font_path)
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    colors = ("#2b6cb0", "#718096", "#c05621", "#c05621", "#c05621")
    metrics = (("readmission_ibs", "积分 Brier 差值：负值更好", "Brier 差值（候选 - 参照）"),
               ("readmission_auc", "平均时间依赖 AUC 差值：正值更好", "AUC(t) 差值（候选 - 参照）"))
    for axis, (field, title, xlabel) in zip(axes, metrics):
        axis.axvline(0, color="#444444", linewidth=1)
        for index, (_, candidate, reference) in enumerate(COMPARISONS):
            values = np.array([by_seed[(seed, candidate)][field] - by_seed[(seed, reference)][field]
                               for seed in seeds])
            axis.scatter(values, np.full(3, index) + np.array([-.12, 0., .12]),
                         color=colors[index], s=44, alpha=.7, zorder=2)
            axis.scatter(values.mean(), index, color=colors[index], marker="D", s=75, zorder=3)
        axis.set_title(title, fontsize=13)
        axis.set_xlabel(xlabel, fontsize=11)
        axis.grid(axis="x", alpha=.2)
        axis.tick_params(axis="x", labelsize=10)
    axes[0].set_yticks(np.arange(len(COMPARISONS)), [label for label, _, _ in COMPARISONS])
    axes[0].tick_params(axis="y", labelsize=11)
    axes[0].invert_yaxis()
    fig.suptitle("早期检验异常实验：验证折再入院，三种子从头训练\n圆点＝各随机种子；菱形＝三种子均值；竖线＝无变化",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args()
    plot(args.source, args.output, font_path=args.font_path)


if __name__ == "__main__":
    main()
