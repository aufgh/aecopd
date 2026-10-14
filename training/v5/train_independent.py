"""独立字段文本分支的 V5 增量实验；沿用既有竞争风险评价合同。"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_data import (
    IndependentData, collate, load_data, normalize_on_training_patients, split_sequences,
)
from training.v5.independent_fusion import IndependentFusionModel
from training.v5.model import competing_nll, predict_cif
from training.v5.train import GRID, _public_metrics, _save_curves, clustered_brier_interval, metrics


def _batches(sequences: list[np.ndarray], batch_size: int) -> list[list[np.ndarray]]:
    ordered = sorted(sequences, key=len)
    return [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]


@torch.no_grad()
def predict(model: IndependentFusionModel, data: IndependentData,
            sequences: list[np.ndarray], *, include_text: bool, batch_size: int,
            device: torch.device) -> tuple[np.ndarray, float]:
    model.eval()
    indices = np.concatenate(sequences)
    logits = np.zeros((len(data.hadm_ids), 2), dtype=np.float32)
    filled = np.zeros(len(data.hadm_ids), dtype=bool)
    loss_sum, count_sum = 0.0, 0
    for members in _batches(sequences, batch_size):
        batch = collate(data, members, include_text=include_text).to(device)
        output = model(batch)
        count = int(batch.valid.sum().item())
        loss_sum += float(competing_nll(output, batch).item()) * count
        count_sum += count
        values = output.log_amplitude.cpu().numpy()
        for position, sequence in enumerate(members):
            logits[sequence] = values[position, :len(sequence)]
            filled[sequence] = True
    if int(filled.sum()) != len(indices) or not filled[indices].all():
        raise ValueError("prediction row alignment failed")
    shape = torch.nn.functional.softplus(model.raw_shape).detach().cpu().numpy() + 0.1
    return predict_cif(logits[indices], shape, GRID)[:, :, 1], loss_sum / count_sum


def fit(data: IndependentData, train: list[np.ndarray], validation: list[np.ndarray],
        *, include_text: bool, max_epochs: int, batch_size: int, seed: int,
        device: torch.device) -> tuple[IndependentFusionModel, list[dict[str, float | int]], int, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = IndependentFusionModel(
        concept_count=len(data.concept_names), type_count=len(data.type_names),
        field_count=len(FIELD_LABELS), text_width=data.text_width,
    ).to(device)
    # 文本投影宽度不同会消耗不同数量的初始化随机数；重置训练随机流，
    # 确保共享层在无文本对照中的 dropout 序列和更新顺序完全一致。
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
    rng = np.random.default_rng(seed)
    val_indices = np.concatenate(validation)
    best_state = copy.deepcopy(model.state_dict())
    best_risk: np.ndarray | None = None
    best_epoch, best_score, stale = 0, math.inf, 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        shuffled = rng.permutation(len(train))
        loss_sum, count_sum = 0.0, 0
        for start in range(0, len(shuffled), batch_size):
            members = [train[int(index)] for index in shuffled[start:start + batch_size]]
            batch = collate(data, members, include_text=include_text).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = competing_nll(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(batch.valid.sum().item())
            loss_sum += float(loss.detach().item()) * count
            count_sum += count
        val_risk, val_nll = predict(model, data, validation, include_text=include_text,
                                    batch_size=batch_size, device=device)
        summary = metrics(data.times[val_indices], data.events[val_indices], val_risk)
        score = float(summary["integrated_brier_7_365"])
        record: dict[str, float | int] = {
            "epoch": epoch, "train_nll": loss_sum / count_sum,
            "validation_nll": val_nll, "validation_integrated_brier": score,
            "validation_mean_auc": float(summary["mean_auc_7_365"]),
        }
        history.append(record)
        print(json.dumps({"text": include_text, **record}), flush=True)
        if score < best_score - 1e-5:
            best_state, best_risk = copy.deepcopy(model.state_dict()), val_risk.copy()
            best_epoch, best_score, stale = epoch, score, 0
        else:
            stale += 1
        if stale >= 7:
            break
    model.load_state_dict(best_state)
    assert best_risk is not None
    return model, history, best_epoch, best_risk


def run(aligned_dir: Path, structured_dir: Path, base_csv: Path, field_dir: Path,
        vectors: Path, output_dir: Path, *, text_name: str = "bert",
        max_epochs: int, batch_size: int,
        reuse_structured_run: Path | None = None,
        labels_csv: Path | None = None) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("full independent-fusion experiment requires CUDA")
    data, preprocessing = normalize_on_training_patients(
        load_data(aligned_dir, structured_dir, base_csv, field_dir, vectors,
                  labels_csv=labels_csv)
    )
    if (text_name, data.text_width) not in (("bert", 768), ("qwen", 2560)):
        raise ValueError("text experiment name and vector width disagree")
    train = split_sequences(data, "train")
    validation = split_sequences(data, "validation")
    test = split_sequences(data, "test")
    partitions = {"validation": validation, "historical_test": test}
    device = torch.device("cuda")
    text_model_name = f"structured_field_{text_name}"
    reference_report: dict[str, object] | None = None
    if reuse_structured_run is None:
        structured = fit(data, train, validation, include_text=False,
                         max_epochs=max_epochs, batch_size=batch_size, seed=2026,
                         device=device)
    else:
        reference_report = json.loads((reuse_structured_run / "report.json").read_text(encoding="utf-8"))
        if (reference_report["field_manifest_sha256"] != data.manifest_sha256
                or reference_report.get("label_sha256") != data.label_sha256
                or reference_report["cohort_admissions"] != len(data.hadm_ids)
                or reference_report["structured_concepts"] != len(data.concept_names)
                or reference_report["structured_types"] != len(data.type_names)):
            raise ValueError("reused structured run has a different input contract")
        structured_model = IndependentFusionModel(
            concept_count=len(data.concept_names), type_count=len(data.type_names),
            field_count=len(FIELD_LABELS), text_width=int(reference_report["text_vector_width"]),
        ).to(device)
        structured_model.load_state_dict(torch.load(
            reuse_structured_run / "structured.pt", map_location=device, weights_only=True
        ))
        structured_history = json.loads(
            (reuse_structured_run / "structured_history.json").read_text(encoding="utf-8")
        )
        structured_risk, _ = predict(structured_model, data, validation,
                                      include_text=False, batch_size=batch_size, device=device)
        structured = (structured_model, structured_history,
                      int(reference_report["best_epochs"]["structured"]), structured_risk)
    fitted = {
        "structured": structured,
        text_model_name: fit(data, train, validation, include_text=True,
                             max_epochs=max_epochs, batch_size=batch_size,
                             seed=2026, device=device),
    }
    curves: dict[str, dict[str, dict[str, object]]] = {}
    for split, sequences in partitions.items():
        indices = np.concatenate(sequences)
        curves[split] = {}
        for name, (model, _, _, val_risk) in fitted.items():
            risk = val_risk if split == "validation" else predict(
                model, data, sequences, include_text=name != "structured",
                batch_size=batch_size, device=device,
            )[0]
            curves[split][name] = metrics(data.times[indices], data.events[indices], risk)
    if reference_report is not None:
        for split in partitions:
            score = float(curves[split]["structured"]["integrated_brier_7_365"])
            expected = float(reference_report[split]["structured"]["integrated_brier_7_365"])
            if abs(score - expected) > 1e-6:
                raise ValueError(f"reused structured checkpoint changed {split} predictions")
    val_indices = np.concatenate(validation)
    interval = clustered_brier_interval(
        data.subject_ids[val_indices], data.times[val_indices], data.events[val_indices],
        fitted[text_model_name][3], fitted["structured"][3],
    )
    report: dict[str, object] = {
        "model": "V5 independent sparse structured and field-text admission branches; causal patient Transformer; two-cause Weibull",
        "status": "exploratory internal comparison; clinical content not gold-standard validated",
        "cohort_admissions": int(len(data.hadm_ids)),
        "cohort_patients": int(len(data.sequences)),
        "structured_concepts": len(data.concept_names),
        "structured_types": len(data.type_names),
        "text_encoder": text_name,
        "field_manifest_sha256": data.manifest_sha256,
        "label_sha256": data.label_sha256,
        "primary_event": data.label_name or "strict AECOPD-coded readmission",
        "text_vector_width": data.text_width,
        "split_counts": {name: {"admissions": int(sum(map(len, sequences))),
                                "patients": len(sequences)}
                         for name, sequences in (("train", train), ("validation", validation),
                                                  ("historical_test", test))},
        "selection": "checkpoint by validation 7-365d integrated Brier; same patient split and shared seed",
        "structured_reference_run": str(reuse_structured_run) if reuse_structured_run else None,
        "validation": {name: _public_metrics(value) for name, value in curves["validation"].items()},
        "historical_test": {name: _public_metrics(value) for name, value in curves["historical_test"].items()},
        "text_minus_structured_validation_brier_95ci_patient_cluster": interval,
        "best_epochs": {name: result[2] for name, result in fitted.items()},
        "limitations": [
            "MIMIC-IV is a design proxy; no private-data sample or field contract was provided.",
            "The text field selection has not been checked against an annotated clinical gold standard.",
            "Outside-hospital readmission and complete 365-day follow-up are unproven.",
            "The historical test split has been used previously and is not blind.",
        ],
    }
    if reference_report is not None:
        for split in partitions:
            report[split]["structured"] = reference_report[split]["structured"]
    output_dir.mkdir(parents=True)
    for name, (model, history, _, _) in fitted.items():
        torch.save(model.state_dict(), output_dir / f"{name}.pt")
        (output_dir / f"{name}_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
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
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--field-dir", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--text-name", choices=("bert", "qwen"), default="bert")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--reuse-structured-run", type=Path,
                        help="Reuse one fitted structured comparator across text encoders")
    parser.add_argument("--labels-csv", type=Path,
                        help="Validated broad COPD-coded outcome overlay; legacy strict labels otherwise")
    args = parser.parse_args()
    if min(args.max_epochs, args.batch_size) < 1:
        raise ValueError("epochs and batch size must be positive")
    report = run(args.aligned_dir, args.structured_dir, args.base_csv, args.field_dir,
                 args.vectors, args.output_dir, text_name=args.text_name,
                 max_epochs=args.max_epochs,
                 batch_size=args.batch_size,
                 reuse_structured_run=args.reuse_structured_run,
                 labels_csv=args.labels_csv)
    print(json.dumps({"validation": report["validation"],
                      "historical_test": report["historical_test"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
