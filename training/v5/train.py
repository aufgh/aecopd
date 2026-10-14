"""V5 首轮完整样本训练：单特征 Cox、结构化融合、结构化＋规则文本。"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from training.baselines.rank_metrics import auc_rank
from training.baselines.continuous_competing_cox import fit_competing_cox, predict_cumulative_incidence
from training.v5.data import V5Data, collate, load_data
from training.v5.model import V5Model, competing_nll, predict_cif


GRID = np.arange(366, dtype=np.float64)
REPORT_DAYS = (7, 14, 30, 60, 90, 180, 365)
CONTINUOUS = ((0, 2, 3), (), (), tuple(range(15)), tuple(range(25)), (0,))


@dataclass(frozen=True)
class FittedModel:
    model: V5Model
    history: list[dict[str, float | int]]
    best_epoch: int
    validation_predictions: np.ndarray


def preprocess(data: V5Data) -> tuple[V5Data, list[dict[str, list[float]]]]:
    """中位数和 IQR 只在训练患者的已观测值上拟合，缺失保持掩码。"""

    train = data.splits == "train"
    groups: list[np.ndarray] = []
    report: list[dict[str, list[float]]] = []
    for group_index, (values, masks) in enumerate(zip(data.group_values, data.group_masks, strict=True)):
        transformed = values.copy()
        medians = np.zeros(values.shape[1], dtype=np.float64)
        scales = np.ones(values.shape[1], dtype=np.float64)
        for column in CONTINUOUS[group_index]:
            observed = values[train & (masks[:, column] > 0), column].astype(np.float64)
            if observed.size:
                median = float(np.median(observed))
                q1, q3 = np.percentile(observed, (25, 75))
                scale = max(float(q3 - q1), 1.0)
                medians[column], scales[column] = median, scale
                present = masks[:, column] > 0
                transformed[present, column] = np.clip(
                    (values[present, column] - median) / scale, -8.0, 8.0
                )
        transformed[masks == 0] = 0.0
        if not np.isfinite(transformed).all():
            raise ValueError("nonfinite preprocessed feature")
        groups.append(transformed.astype(np.float32))
        report.append({"medians": medians.tolist(), "scales": scales.tolist()})
    return replace(data, group_values=tuple(groups)), report


def split_sequences(data: V5Data, split: str) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for sequence in data.sequences:
        labels = set(data.splits[sequence].tolist())
        if len(labels) != 1:
            raise ValueError("patient crosses data splits")
        if split in labels:
            result.append(sequence)
    if not result:
        raise ValueError(f"empty {split} partition")
    return result


def _batches(sequences: list[np.ndarray], batch_size: int) -> list[list[np.ndarray]]:
    """按序列长度局部排序降低 padding；不改变患者内时序。"""

    ordered = sorted(sequences, key=len)
    return [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]


@torch.no_grad()
def predict(model: V5Model, data: V5Data, sequences: list[np.ndarray], *,
            include_text: bool, device: torch.device, batch_size: int) -> tuple[np.ndarray, float]:
    model.eval()
    log_amplitude = np.zeros((len(data.hadm_ids), 2), dtype=np.float32)
    filled = np.zeros(len(data.hadm_ids), dtype=bool)
    losses: list[tuple[float, int]] = []
    for members in _batches(sequences, batch_size):
        batch = collate(data, members, include_text=include_text).to(device)
        output = model(batch)
        count = int(batch.valid.sum().item())
        losses.append((float(competing_nll(output, batch).item()), count))
        probabilities = output.log_amplitude.detach().cpu().numpy()
        for position, sequence in enumerate(members):
            log_amplitude[sequence] = probabilities[position, :len(sequence)]
            filled[sequence] = True
    selected = np.concatenate(sequences)
    if not filled[selected].all() or int(filled.sum()) != len(selected):
        raise ValueError("prediction index mismatch")
    curves = predict_cif(log_amplitude[selected],
                         torch.nn.functional.softplus(model.raw_shape).detach().cpu().numpy() + 0.1,
                         GRID)
    nll = sum(loss * count for loss, count in losses) / sum(count for _, count in losses)
    return curves[:, :, 1], float(nll)


def metrics(times: np.ndarray, events: np.ndarray, risk: np.ndarray) -> dict[str, object]:
    if risk.shape != (len(times), 366):
        raise ValueError("risk array must have one daily curve per admission")
    observed = ((events[:, None] == 1) & (times[:, None] <= GRID[None, :])).astype(np.float64)
    brier = np.mean((risk - observed) ** 2, axis=0)
    auc = np.full(366, np.nan, dtype=np.float64)
    cases = observed.sum(axis=0).astype(int)
    for day in range(7, 366):
        if min(int(cases[day]), len(times) - int(cases[day])) >= 20:
            auc[day] = auc_rank(risk[:, day], observed[:, day].astype(np.int8))
    points = {str(day): {"auc": float(auc[day]) if np.isfinite(auc[day]) else None,
                         "brier": float(brier[day]), "cases": int(cases[day]),
                         "mean_predicted_risk": float(risk[:, day].mean()),
                         "observed_cumulative_fraction_world_a": float(observed[:, day].mean())}
              for day in REPORT_DAYS}
    return {
        "admissions": int(len(times)), "readmissions": int((events == 1).sum()),
        "deaths": int((events == 2).sum()),
        "integrated_brier_7_365": float(np.trapezoid(brier[7:], GRID[7:]) / (365 - 7)),
        "mean_auc_7_365": float(np.nanmean(auc[7:])),
        "points": points,
        "brier_curve": brier, "auc_curve": auc, "case_curve": cases,
    }


def _public_metrics(detail: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in detail.items()
            if key not in {"brier_curve", "auc_curve", "case_curve"}}


def train_model(data: V5Data, train_sequences: list[np.ndarray],
                validation_sequences: list[np.ndarray], *, include_text: bool,
                device: torch.device, max_epochs: int, batch_size: int,
                seed: int, balance_text: bool = False) -> FittedModel:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 结构化对照与文本编码维度无关，始终使用相同参数形状。
    model = V5Model(text_width=data.text_width if include_text else 768,
                    balance_text=balance_text if include_text else False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
    rng = np.random.default_rng(seed)
    best_epoch, best_score = 0, math.inf
    best_state = copy.deepcopy(model.state_dict())
    best_predictions: np.ndarray | None = None
    history: list[dict[str, float | int]] = []
    stale = 0
    val_indices = np.concatenate(validation_sequences)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(train_sequences))
        train_loss_sum, train_count = 0.0, 0
        for start in range(0, len(order), batch_size):
            members = [train_sequences[int(index)] for index in order[start:start + batch_size]]
            batch = collate(data, members, include_text=include_text).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = competing_nll(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite V5 training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(batch.valid.sum().item())
            train_loss_sum += float(loss.detach().item()) * count
            train_count += count
        validation_risk, val_nll = predict(
            model, data, validation_sequences, include_text=include_text,
            device=device, batch_size=batch_size,
        )
        evaluated = metrics(data.times[val_indices], data.events[val_indices], validation_risk)
        score = float(evaluated["integrated_brier_7_365"])
        entry: dict[str, float | int] = {
            "epoch": epoch, "train_nll": train_loss_sum / train_count,
            "validation_nll": val_nll, "validation_integrated_brier": score,
            "validation_mean_auc": float(evaluated["mean_auc_7_365"]),
        }
        history.append(entry)
        print(json.dumps({"text": include_text, **entry}), flush=True)
        if score < best_score - 1e-5:
            best_epoch, best_score = epoch, score
            best_state = copy.deepcopy(model.state_dict())
            best_predictions = validation_risk.copy()
            stale = 0
        else:
            stale += 1
        if stale >= 7:
            break
    model.load_state_dict(best_state)
    assert best_predictions is not None
    return FittedModel(model, history, best_epoch, best_predictions)


def clustered_brier_interval(subject_ids: np.ndarray, times: np.ndarray,
                             events: np.ndarray, full: np.ndarray,
                             comparator: np.ndarray, *, seed: int = 2026) -> list[float]:
    observed = ((events[:, None] == 1) & (times[:, None] <= GRID[None, :])).astype(float)
    delta = np.trapezoid((full[:, 7:] - observed[:, 7:]) ** 2
                         - (comparator[:, 7:] - observed[:, 7:]) ** 2,
                         GRID[7:], axis=1) / (365 - 7)
    patients, inverse = np.unique(subject_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(len(patients))]
    rng = np.random.default_rng(seed)
    samples = np.empty(300, dtype=np.float64)
    for iteration in range(len(samples)):
        draw = np.concatenate([groups[index] for index in rng.integers(len(groups), size=len(groups))])
        samples[iteration] = delta[draw].mean()
    return np.quantile(samples, (0.025, 0.975)).tolist()


def _save_curves(output_dir: Path, curves: dict[str, dict[str, dict[str, object]]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output_dir / "aggregate_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "model", "day", "auc", "brier", "cumulative_cases"))
        for split, models in curves.items():
            for name, values in models.items():
                for day in range(366):
                    score = values["auc_curve"][day]
                    writer.writerow((split, name, day, "" if not np.isfinite(score) else score,
                                     values["brier_curve"][day], values["case_curve"][day]))
    for split, models in curves.items():
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for name, values in models.items():
            axes[0].plot(GRID[7:], values["auc_curve"][7:], label=name)
            axes[1].plot(GRID[7:], values["brier_curve"][7:], label=name)
        axes[0].set(xlabel="Days after discharge", ylabel="Competing-risk cumulative AUC(t)",
                    ylim=(0.4, 0.9), title=f"{split}: discrimination")
        axes[1].set(xlabel="Days after discharge", ylabel="Brier(t)",
                    title=f"{split}: probability error")
        for axis in axes:
            axis.legend()
            axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output_dir / f"{split}_auc_brier.png", dpi=160)
        plt.close(figure)


def run(structured_dir: Path, manifest: Path, vectors: Path, output_dir: Path,
        *, text_name: str = "bert", vector_width: int | None = None,
        max_epochs: int = 35, batch_size: int = 24,
        balance_text: bool = False) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("V5 full run requires CUDA")
    expected_width = {"bert": 768, "qwen2560": 2560, "qwen768": 768, "qwen128": 128}
    if text_name not in expected_width:
        raise ValueError("unsupported text encoder experiment name")
    data, preprocessing = preprocess(load_data(structured_dir, manifest, vectors,
                                              vector_width=vector_width))
    expected_encoder = "Bio_ClinicalBERT" if text_name == "bert" else "Qwen3-Embedding-4B"
    if data.text_width != expected_width[text_name] or data.coverage["text_encoder"] != expected_encoder:
        raise ValueError("experiment name and text vector dimension disagree")
    train_sequences = split_sequences(data, "train")
    validation_sequences = split_sequences(data, "validation")
    test_sequences = split_sequences(data, "test")
    index = {split: np.concatenate(sequences) for split, sequences in
             (("train", train_sequences), ("validation", validation_sequences),
              ("historical_test", test_sequences))}
    device = torch.device("cuda")
    anchor_train = index["train"]
    anchor_feature = data.group_values[0][:, 3:4].astype(np.float64)
    anchor = fit_competing_cox(anchor_feature[anchor_train], data.times[anchor_train],
                               data.events[anchor_train], l2=10.0)
    if not anchor.readmission.converged or not anchor.death.converged:
        raise RuntimeError("single-feature competing Cox baseline failed")
    fitted = {
        "structured": train_model(data, train_sequences, validation_sequences,
                                  include_text=False, device=device, max_epochs=max_epochs,
                                  batch_size=batch_size, seed=2026),
        f"structured_{text_name}": train_model(data, train_sequences, validation_sequences,
                                               include_text=True, device=device, max_epochs=max_epochs,
                                               batch_size=batch_size, seed=2026,
                                               balance_text=balance_text),
    }
    curves: dict[str, dict[str, dict[str, object]]] = {}
    for split, sequences in (("validation", validation_sequences),
                             ("historical_test", test_sequences)):
        selected = index[split]
        risks: dict[str, np.ndarray] = {
            "prior_aecopd_cox": predict_cumulative_incidence(
                anchor, anchor_feature[selected], GRID
            )[:, :, 1]
        }
        for name, result in fitted.items():
            risks[name] = (result.validation_predictions if split == "validation"
                           else predict(result.model, data, sequences,
                                        include_text=name != "structured",
                                        device=device, batch_size=batch_size)[0])
        curves[split] = {name: metrics(data.times[selected], data.events[selected], risk)
                         for name, risk in risks.items()}
    validation = curves["validation"]
    interval = clustered_brier_interval(
        data.subject_ids[index["validation"]], data.times[index["validation"]],
        data.events[index["validation"]],
        fitted[f"structured_{text_name}"].validation_predictions,
        fitted["structured"].validation_predictions,
    )
    report: dict[str, object] = {
        "model": "V5 repeated-discharge structured/text causal Transformer competing Weibull",
        "text_experiment": text_name,
        "text_input_width": data.text_width,
        "balanced_text_content": balance_text,
        "status": "exploratory internal run; text rules not gold-standard validated",
        "source": "MIMIC-IV full discharge prediction sample set",
        "coverage": data.coverage,
        "split_counts": {split: {"admissions": int(len(rows)),
                                 "patients": int(len(sequences))}
                         for split, rows, sequences in
                         (("train", index["train"], train_sequences),
                          ("validation", index["validation"], validation_sequences),
                          ("historical_test", index["historical_test"], test_sequences))},
        "split_rule": "patient md5(aecopd-model-v1:subject_id), 70/15/15; historical test reused",
        "head": "two cause-specific Weibull amplitudes and two global shapes",
        "selection": "each network checkpoint selected by validation 7-365d integrated Brier",
        "validation": {name: _public_metrics(value) for name, value in curves["validation"].items()},
        "historical_test": {name: _public_metrics(value) for name, value in curves["historical_test"].items()},
        "text_minus_structured_validation_brier_95ci_patient_cluster": interval,
        "best_epochs": {name: result.best_epoch for name, result in fitted.items()},
        "limitations": [
            "No individual 365-day follow-up completeness or outside-hospital readmission capture is proven.",
            "Discharge summaries stored after discharge are treated as discharge-time retrospective documents.",
            "Rule text fields have not been validated against an annotated gold standard.",
            "The historical test split has been used by prior project experiments and is not a fresh blind test.",
            "eMAR drug-name mapping is conservative and does not prove an indication or treatment response.",
        ],
    }
    output_dir.mkdir(parents=True)
    for name, result in fitted.items():
        torch.save(result.model.state_dict(), output_dir / f"{name}.pt")
        (output_dir / f"{name}_history.json").write_text(
            json.dumps(result.history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output_dir / "preprocessing.json").write_text(
        json.dumps(preprocessing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_curves(output_dir, curves)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--text-manifest", type=Path, required=True)
    parser.add_argument("--text-vectors", "--bert-vectors", dest="text_vectors", type=Path,
                        required=True)
    parser.add_argument("--text-name", choices=("bert", "qwen2560", "qwen768", "qwen128"),
                        default="bert")
    parser.add_argument("--vector-width", type=int)
    parser.add_argument("--balance-text", action="store_true",
                        help="Layer-normalize projected snippet content before adding domain embedding")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    if args.max_epochs < 1 or args.batch_size < 1:
        raise ValueError("max epochs and batch size must be positive")
    report = run(args.structured_dir, args.text_manifest, args.text_vectors,
                 args.output_dir, text_name=args.text_name, vector_width=args.vector_width,
                 max_epochs=args.max_epochs, batch_size=args.batch_size,
                 balance_text=args.balance_text)
    print(json.dumps({"coverage": report["coverage"],
                      "validation": report["validation"],
                      "text_minus_structured_validation_brier_95ci_patient_cluster":
                          report["text_minus_structured_validation_brier_95ci_patient_cluster"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
