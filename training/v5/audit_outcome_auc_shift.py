"""固定已训练模型和预测值，只替换结局，量化 AUC 的标签口径效应。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_data import load_data, normalize_on_training_patients, split_sequences
from training.v5.independent_fusion import IndependentFusionModel
from training.v5.train import metrics
from training.v5.train_independent import predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned_dir", "structured_dir", "base_csv", "field_dir", "vectors",
                 "broad_labels", "qualified_labels", "broad_model_run", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.plot is not None and args.plot.exists()):
        raise FileExistsError("audit output already exists")
    data = {}
    for name, label in (("strict", None), ("broad", args.broad_labels),
                        ("strict_or_top3", args.qualified_labels)):
        loaded = load_data(args.aligned_dir, args.structured_dir, args.base_csv,
                           args.field_dir, args.vectors, labels_csv=label)
        data[name] = normalize_on_training_patients(loaded)[0]
    anchor = data["broad"]
    for name, other in data.items():
        if (not np.array_equal(anchor.hadm_ids, other.hadm_ids)
                or not np.array_equal(anchor.subject_ids, other.subject_ids)
                or not np.array_equal(anchor.splits, other.splits)
                or anchor.manifest_sha256 != other.manifest_sha256):
            raise ValueError(f"{name} changed cohort, split or text manifest")
    if not torch.cuda.is_available():
        raise RuntimeError("fixed-model outcome audit requires CUDA")
    device = torch.device("cuda")
    model = IndependentFusionModel(
        concept_count=len(anchor.concept_names), type_count=len(anchor.type_names),
        field_count=len(FIELD_LABELS), text_width=anchor.text_width,
    ).to(device)
    model.load_state_dict(torch.load(args.broad_model_run / "structured_field_bert.pt",
                                     map_location=device, weights_only=True))
    report: dict[str, object] = {
        "fixed_model": "previously trained broad-outcome summary+BERT checkpoint",
        "cohort_admissions": len(anchor.hadm_ids),
        "field_manifest_sha256": anchor.manifest_sha256,
        "label_sha256": {name: item.label_sha256 for name, item in data.items()},
    }
    figures: dict[str, dict[str, object]] = {}
    for split in ("validation", "test"):
        sequences = split_sequences(anchor, split)
        indices = np.concatenate(sequences)
        # 三种结局共用同一份风险预测；只改变评价时的病例／对照定义。
        risk, _ = predict(model, anchor, sequences, include_text=True,
                          batch_size=24, device=device)
        details = {name: metrics(item.times[indices], item.events[indices], risk)
                   for name, item in data.items()}
        figures[split] = details
        report["historical_test" if split == "test" else split] = {
            name: {key: detail[key] for key in
                   ("admissions", "readmissions", "deaths", "integrated_brier_7_365",
                    "mean_auc_7_365")}
            for name, detail in details.items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.plot is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col")
        for column, split in enumerate(("validation", "test")):
            for name, detail in figures[split].items():
                days = np.arange(7, 366)
                axes[0, column].plot(days, detail["auc_curve"][7:], label=name)
                axes[1, column].plot(days, detail["case_curve"][7:], label=name)
            axes[0, column].set(title=split, ylabel="Cumulative AUC(t)", ylim=(0.5, 0.85))
            axes[1, column].set(xlabel="Days after discharge", ylabel="Cumulative cases")
            for axis in axes[:, column]:
                axis.grid(alpha=0.25)
        axes[0, 0].legend()
        figure.suptitle("Same model and predictions; only outcome labels differ")
        figure.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot, dpi=170)
        plt.close(figure)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
