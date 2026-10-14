"""用已保存的 V5 权重补充患者聚类 AUC 区间、训练和校准图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.baselines.rank_metrics import auc_rank
from training.baselines.continuous_competing_cox import fit_competing_cox, predict_cumulative_incidence
from training.v5.data import load_data
from training.v5.model import V5Model
from training.v5.train import GRID, preprocess, predict, split_sequences


POINTS = (7, 14, 30, 60, 90, 180, 365)


def auc_intervals(subject_ids: np.ndarray, times: np.ndarray, events: np.ndarray,
                  risk: np.ndarray, *, bootstrap: int = 200) -> dict[str, dict[str, float | int | None]]:
    patients, inverse = np.unique(subject_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(len(patients))]
    rng = np.random.default_rng(2026)
    result: dict[str, dict[str, float | int | None]] = {}
    for day in POINTS:
        labels = ((events == 1) & (times <= day)).astype(np.int8)
        count = int(labels.sum())
        if min(count, len(labels) - count) < 20:
            result[str(day)] = {"cases": count, "auc": None, "lower": None, "upper": None}
            continue
        point = auc_rank(risk[:, day], labels)
        values: list[float] = []
        for _ in range(bootstrap):
            draw = np.concatenate([groups[index] for index in rng.integers(len(groups), size=len(groups))])
            if min(int(labels[draw].sum()), len(draw) - int(labels[draw].sum())) >= 20:
                values.append(auc_rank(risk[draw, day], labels[draw]))
        if len(values) < bootstrap // 2:
            raise ValueError("too few valid patient bootstrap samples")
        lower, upper = np.quantile(values, (0.025, 0.975))
        result[str(day)] = {"cases": count, "auc": float(point),
                            "lower": float(lower), "upper": float(upper)}
    return result


def calibration_points(times: np.ndarray, events: np.ndarray, risk: np.ndarray,
                       day: int, bins: int = 10) -> list[dict[str, float | int]]:
    order = np.argsort(risk[:, day], kind="stable")
    result: list[dict[str, float | int]] = []
    for indices in np.array_split(order, bins):
        observed = (events[indices] == 1) & (times[indices] <= day)
        result.append({"count": int(len(indices)), "predicted": float(risk[indices, day].mean()),
                       "observed_world_a": float(observed.mean())})
    return result


def run(structured_dir: Path, manifest: Path, vectors: Path,
        training_dir: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required to replay V5 checkpoints")
    training_report = json.loads((training_dir / "report.json").read_text(encoding="utf-8"))
    text_name = str(training_report.get("text_experiment", "bert"))
    text_model_name = f"structured_{text_name}"
    data, _ = preprocess(load_data(structured_dir, manifest, vectors,
                                   vector_width=int(training_report.get("text_input_width", 768))))
    training = split_sequences(data, "train")
    anchor_indices = np.concatenate(training)
    anchor_feature = data.group_values[0][:, 3:4].astype(np.float64)
    anchor = fit_competing_cox(anchor_feature[anchor_indices], data.times[anchor_indices],
                               data.events[anchor_indices], l2=10.0)
    report: dict[str, object] = {"bootstrap_replicates": 200,
                                 "confidence": "patient-cluster 95% pointwise intervals",
                                 "splits": {}}
    plot_data: dict[str, dict[str, np.ndarray]] = {}
    for split in ("validation", "historical_test"):
        sequences = split_sequences(data, "test" if split == "historical_test" else split)
        indices = np.concatenate(sequences)
        risks: dict[str, np.ndarray] = {
            "prior_aecopd_cox": predict_cumulative_incidence(
                anchor, anchor_feature[indices], GRID)[:, :, 1]
        }
        for name in ("structured", text_model_name):
            model = V5Model(text_width=data.text_width if name == text_model_name else 768,
                            balance_text=bool(training_report.get("balanced_text_content", False))
                            if name == text_model_name else False).to("cuda")
            state = torch.load(training_dir / f"{name}.pt", map_location="cuda",
                               weights_only=True)
            model.load_state_dict(state)
            risks[name] = predict(model, data, sequences, include_text=name == text_model_name,
                                  device=torch.device("cuda"), batch_size=24)[0]
        plot_data[split] = risks
        split_report: dict[str, object] = {}
        for name, risk in risks.items():
            split_report[name] = {
                "auc_points": auc_intervals(data.subject_ids[indices], data.times[indices],
                                             data.events[indices], risk),
                "calibration": {str(day): calibration_points(
                    data.times[indices], data.events[indices], risk, day
                ) for day in (30, 90, 180, 365)},
            }
        report["splits"][split] = split_report
    output_dir.mkdir(parents=True)
    (output_dir / "aggregate_uncertainty_calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plots(training_dir, output_dir, plot_data, report, text_model_name)
    return report


def _plots(training_dir: Path, output_dir: Path,
           risks: dict[str, dict[str, np.ndarray]], report: dict[str, object],
           text_model_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histories = {name: json.loads((training_dir / f"{name}_history.json").read_text(encoding="utf-8"))
                 for name in ("structured", text_model_name)}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, history in histories.items():
        axes[0].plot([row["epoch"] for row in history],
                     [row["train_nll"] for row in history], label=f"{name} train")
        axes[0].plot([row["epoch"] for row in history],
                     [row["validation_nll"] for row in history], linestyle="--",
                     label=f"{name} validation")
        axes[1].plot([row["epoch"] for row in history],
                     [row["validation_integrated_brier"] for row in history], label=name)
    axes[0].set(xlabel="Epoch", ylabel="Negative log likelihood", title="Training and validation")
    axes[1].set(xlabel="Epoch", ylabel="Integrated Brier, days 7–365",
                title="Validation checkpoint selection")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=170)
    plt.close(figure)

    colors = {"prior_aecopd_cox": "#b36b00", "structured": "#1261a0",
              text_model_name: "#258b5a"}
    for split, models in risks.items():
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for name, risk in models.items():
            points = report["splits"][split][name]["auc_points"]
            x = np.array([day for day in POINTS if points[str(day)]["auc"] is not None])
            y = np.array([points[str(day)]["auc"] for day in x])
            lower = np.array([points[str(day)]["lower"] for day in x])
            upper = np.array([points[str(day)]["upper"] for day in x])
            axes[0].errorbar(x, y, yerr=np.stack((y - lower, upper - y)),
                             marker="o", capsize=2, linewidth=1.4, markersize=3,
                             label=name, color=colors[name])
            day = 365
            calibration = report["splits"][split][name]["calibration"][str(day)]
            axes[1].plot([item["predicted"] for item in calibration],
                         [item["observed_world_a"] for item in calibration],
                         marker="o", label=name, color=colors[name])
        axes[0].set(xlabel="Days after discharge", ylabel="Cumulative competing-risk AUC(t)",
                    title=f"{split}: patient-cluster 95% CI")
        axes[1].plot([0, 0.7], [0, 0.7], color="gray", linestyle="--")
        axes[1].set(xlabel="Predicted 365-day readmission risk",
                    ylabel="Observed fraction (World A)", title=f"{split}: calibration deciles")
        for axis in axes:
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output_dir / f"{split}_auc_ci_calibration.png", dpi=170)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--text-manifest", type=Path, required=True)
    parser.add_argument("--text-vectors", "--bert-vectors", dest="text_vectors", type=Path,
                        required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.structured_dir, args.text_manifest, args.text_vectors,
                 args.training_dir, args.output_dir)
    print(json.dumps({"outputs": str(args.output_dir), "splits": list(result["splits"])},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
