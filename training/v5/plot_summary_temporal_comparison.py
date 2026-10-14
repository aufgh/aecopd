"""读取已完成实验的汇总 CSV，绘制同标签三模型曲线；不重新训练。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-run", type=Path, required=True)
    parser.add_argument("--temporal-run", type=Path, required=True)
    parser.add_argument("--set-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    directories = [args.summary_run, args.temporal_run]
    if args.set_run is not None:
        directories.append(args.set_run)
    reports = [json.loads((directory / "report.json").read_text(encoding="utf-8"))
               for directory in directories]
    for key in ("label_sha256", "field_manifest_sha256", "cohort_admissions"):
        if any(report[key] != reports[0][key] for report in reports[1:]):
            raise ValueError(f"comparison contract differs: {key}")
    curves: dict[tuple[str, str], list[dict[str, str]]] = {}
    for directory in directories:
        with (directory / "aggregate_curves.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["day"]) >= 7:
                    name = "event_set" if directory == args.set_run else row["model"]
                    curves.setdefault((row["split"], name), []).append(row)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"structured": "structured summaries",
              "structured_field_bert": "summaries + BERT",
              "temporal_full": "GRU events + BERT",
              "event_set": "STraTS-style event set + BERT"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col")
    for column, split in enumerate(("validation", "historical_test")):
        for (partition, model), rows in curves.items():
            if partition != split:
                continue
            days = [int(row["day"]) for row in rows]
            auc = [float(row["auc"]) if row["auc"] else float("nan") for row in rows]
            axes[0, column].plot(days, auc, label=labels[model])
            axes[1, column].plot(days, [float(row["brier"]) for row in rows], label=labels[model])
        axes[0, column].set(title=split, ylabel="Cumulative AUC(t)", ylim=(0.5, 0.8))
        axes[1, column].set(xlabel="Days after discharge", ylabel="Brier(t)")
        for axis in axes[:, column]:
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=9)
    figure.suptitle("Strict AECOPD at any position OR preset COPD/emphysema in top 3")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
