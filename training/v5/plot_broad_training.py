"""绘制同一宽口径终点的三组训练／验证学习曲线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-run", type=Path, required=True)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--process-run", type=Path)
    parser.add_argument("--title", default="Broad COPD-coded readmission; same cohort and BERT input")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histories = {
        "admission summaries": args.summary_run / "structured_field_bert_history.json",
        "raw event sequence": args.full_run / "history.json",
    }
    if args.process_run is not None:
        histories["process without lab values"] = args.process_run / "history.json"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for name, path in histories.items():
        records = json.loads(path.read_text(encoding="utf-8"))
        epochs = [entry["epoch"] for entry in records]
        axes[0].plot(epochs, [entry["train_nll"] for entry in records], marker=".", label=name)
        axes[1].plot(epochs, [entry["validation_integrated_brier"] for entry in records],
                     marker=".", label=name)
    axes[0].set(xlabel="Epoch", ylabel="Training competing-risk NLL",
                title="Training objective")
    axes[1].set(xlabel="Epoch", ylabel="Validation integrated Brier (7–365 d)",
                title="Checkpoint-selection metric")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(args.title)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
