"""只读重放已完成的清洗事件 V5，补充竞争风险图与探索性随访 DCA。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from training.v5.event_data import EventData, collate_events
from training.v5.event_set_model import EventSetFusionModel
from training.v5.independent_data import split_sequences
from training.v5.model import predict_cif
from training.v5.processed_event_data import load_processed_event_data
from training.v5.train import GRID, metrics

POINTS = (30, 90, 180, 365)
DCA_DAYS = (30, 90)
DCA_THRESHOLDS = np.arange(1, 51, dtype=np.float64) / 100


def binary_roc(labels: np.ndarray, risk: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """固定时点累积病例 vs 未发生该事件者；先死亡者为对照。"""
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if min(positives, negatives) < 1:
        raise ValueError("ROC needs both cases and controls")
    order = np.argsort(-risk, kind="stable")
    sorted_risk = risk[order]
    sorted_labels = labels[order].astype(np.int64)
    end = np.r_[np.flatnonzero(np.diff(sorted_risk) != 0), len(labels) - 1]
    tp = np.cumsum(sorted_labels)[end]
    fp = end + 1 - tp
    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def decision_curve(labels: np.ndarray, risk: np.ndarray,
                   thresholds: np.ndarray = DCA_THRESHOLDS) -> dict[str, np.ndarray]:
    """世界观 A 下的时点净获益；只有明确行动及阈值后才可作临床解释。"""
    if np.any((thresholds <= 0) | (thresholds >= 1)):
        raise ValueError("DCA thresholds must lie strictly between zero and one")
    selected = risk[:, None] >= thresholds[None, :]
    true_positive = (selected & labels[:, None]).mean(axis=0)
    false_positive = (selected & ~labels[:, None]).mean(axis=0)
    odds = thresholds / (1 - thresholds)
    prevalence = float(labels.mean())
    return {"threshold": thresholds,
            "model": true_positive - false_positive * odds,
            "treat_all": prevalence - (1 - prevalence) * odds,
            "treat_none": np.zeros_like(thresholds)}


def calibration_bins(labels: np.ndarray, risk: np.ndarray, bins: int = 10) -> list[dict[str, float | int]]:
    order = np.argsort(risk, kind="stable")
    return [{"n": int(len(part)), "predicted": float(risk[part].mean()),
             "observed_world_a": float(labels[part].mean())}
            for part in np.array_split(order, bins)]


def patient_bootstrap_auc(subject_ids: np.ndarray, labels: np.ndarray,
                          risk: np.ndarray, *, draws: int = 200,
                          seed: int = 2026) -> tuple[float, float]:
    patients, inverse = np.unique(subject_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(patients))]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sampled = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if min(int(labels[sampled].sum()), len(sampled) - int(labels[sampled].sum())) >= 20:
            values.append(binary_roc(labels[sampled], risk[sampled])[2])
    if len(values) < draws // 2:
        raise ValueError("too few valid patient bootstrap samples")
    return tuple(float(x) for x in np.quantile(values, (0.025, 0.975)))


def evaluate_split(subject_ids: np.ndarray, times: np.ndarray, events: np.ndarray,
                   curves: np.ndarray, *, draws: int = 200) -> tuple[dict[str, object], dict[str, object]]:
    if curves.shape != (len(times), 366, 3) or not np.isfinite(curves).all():
        raise ValueError("invalid competing-risk curves")
    if np.max(np.abs(curves.sum(axis=2) - 1)) > 1e-5 or np.min(curves) < -1e-8:
        raise ValueError("competing-risk probabilities invalid")
    if np.any((events == 0) & (times != 365)):
        raise ValueError("early censoring needs a different IPCW/Aalen-Johansen evaluator")
    if not np.all(np.isin(events, (0, 1, 2))):
        raise ValueError("unknown event type")
    summary: dict[str, object] = {"n_admissions": int(len(times)),
                                   "n_patients": int(len(np.unique(subject_ids))),
                                   "bootstrap_patient_draws": draws,
                                   "followup_contract": "World A: all event_type=0 censored at day 365",
                                   "readmission": {}, "death": {}}
    detailed: dict[str, object] = {"roc": {}, "dca": {}, "calibration": {}}
    for cause, column in (("readmission", 1), ("death", 2)):
        for day in POINTS:
            labels = ((events == column) & (times <= day))
            predicted = curves[:, day, column]
            fpr, tpr, auc = binary_roc(labels, predicted)
            low, high = patient_bootstrap_auc(subject_ids, labels, predicted, draws=draws,
                                              seed=2026 + day + column)
            event_rate = float(labels.mean())
            brier = float(np.mean((predicted - labels) ** 2))
            null_brier = event_rate * (1 - event_rate)
            summary[cause][str(day)] = {
                "events": int(labels.sum()), "observed_fraction_world_a": event_rate,
                "mean_predicted_risk": float(predicted.mean()),
                "observed_to_expected": float(event_rate / predicted.mean()) if predicted.mean() else None,
                "auc": auc, "auc_patient_cluster_95ci": [low, high],
                "brier": brier, "scaled_brier_vs_constant_event_rate":
                    float(1 - brier / null_brier) if null_brier else None,
            }
            detailed["roc"][(cause, day)] = (fpr, tpr)
            detailed["calibration"][(cause, day)] = calibration_bins(labels, predicted)
            if cause == "readmission" and day in DCA_DAYS:
                detailed["dca"][day] = decision_curve(labels, predicted)
    return summary, detailed


@torch.no_grad()
def replay(model: EventSetFusionModel, data: EventData,
           sequences: list[np.ndarray], device: torch.device) -> np.ndarray:
    model.eval()
    selected = np.concatenate(sequences)
    logits = np.zeros((len(data.base.hadm_ids), 2), dtype=np.float32)
    filled = np.zeros(len(data.base.hadm_ids), dtype=bool)
    for start in range(0, len(sequences), 12):
        members = sequences[start:start + 12]
        batch = collate_events(data, members, include_text=True, process_only=False).to(device)
        values = model(batch).log_amplitude.cpu().numpy()
        for pos, sequence in enumerate(members):
            logits[sequence] = values[pos, :len(sequence)]
            filled[sequence] = True
    if int(filled.sum()) != len(selected) or not filled[selected].all():
        raise ValueError("checkpoint prediction alignment failed")
    shape = torch.nn.functional.softplus(model.raw_shape).detach().cpu().numpy() + 0.1
    return predict_cif(logits[selected], shape, GRID)


def save_figures(details: dict[str, dict[str, object]], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for split, detail in details.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for day in POINTS:
            fpr, tpr = detail["roc"][("readmission", day)]
            axes[0].plot(fpr, tpr, label=f"day {day}")
            points = detail["calibration"][("readmission", day)]
            axes[1].plot([p["predicted"] for p in points],
                         [p["observed_world_a"] for p in points], marker="o", label=f"day {day}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set(xlabel="False positive rate", ylabel="Sensitivity", title="Cumulative/dynamic ROC")
        axes[1].set(xlabel="Predicted CIF", ylabel="Observed fraction (World A)", title="Readmission calibration")
        for axis in axes:
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{split}_roc_calibration.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for axis, day in zip(axes, DCA_DAYS, strict=True):
            dca = detail["dca"][day]
            for name in ("model", "treat_all", "treat_none"):
                axis.plot(dca["threshold"], dca[name], label=name)
            axis.set(xlabel="Risk threshold", ylabel="Net benefit per discharge",
                     title=f"Day {day}: hypothetical intensified follow-up")
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{split}_exploratory_dca.png", dpi=160)
        plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint replay requires CUDA")
    report = json.loads((args.run_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "training_completed" or report.get("event_encoder") != "set" or not report.get("preprocessed_events"):
        raise ValueError("not a completed processed-event V5 checkpoint")
    data, _ = load_processed_event_data(
        args.aligned_dir, args.structured_dir, args.base_csv,
        args.processed_event_dir, args.raw_ledger_dir, labels_csv=args.labels_csv,
        field_dir=args.field_dir, vectors_path=args.vectors)
    if (report["cohort_admissions"] != len(data.base.hadm_ids)
            or report["label_sha256"] != data.base.label_sha256
            or report["field_manifest_sha256"] != data.base.manifest_sha256
            or report["event_concepts"] != list(data.concept_names)
            or report.get("disabled_static_concepts", []) != list(data.base.disabled_concepts)):
        raise ValueError("evaluation input differs from completed training contract")
    state = torch.load(args.run_dir / "model.pt", map_location="cpu", weights_only=True)
    model = EventSetFusionModel(
        concept_count=len(data.base.concept_names), type_count=len(data.base.type_names),
        field_count=len(data.base.field_roles), text_width=data.base.text_width,
        event_concept_count=len(data.concept_names)).to("cuda").eval()
    model.load_state_dict(state, strict=True)
    summary: dict[str, object] = {
        "status": "completed", "source_run": str(args.run_dir),
        "primary_event": report["primary_event"], "label_sha256": report["label_sha256"],
        "patient_bootstrap_draws": args.bootstrap,
        "limitations": [
            "DCA assumes a hypothetical intensified-follow-up action; no clinical threshold or intervention effect has been validated.",
            "World A treats unobserved events as absent through day 365; outside-hospital readmissions may be missed.",
            "Historical test patients have been viewed in previous experiments and are not a new blind test.",
        ], "splits": {}}
    details = {}
    for split in ("validation", "test"):
        sequences = split_sequences(data.base, split)
        indices = np.concatenate(sequences)
        curves = replay(model, data, sequences, torch.device("cuda"))
        original = metrics(data.base.times[indices], data.base.events[indices], curves[:, :, 1])
        name = "validation" if split == "validation" else "historical_test"
        reference = report[name]["temporal_full"]
        for metric in ("integrated_brier_7_365", "mean_auc_7_365"):
            if not np.isclose(original[metric], reference[metric], atol=1e-7):
                raise ValueError(f"checkpoint replay changed {name} {metric}")
        scores, detail = evaluate_split(data.base.subject_ids[indices], data.base.times[indices],
                                        data.base.events[indices], curves,
                                        draws=args.bootstrap)
        scores["replayed_integrated_brier_7_365"] = original["integrated_brier_7_365"]
        scores["replayed_mean_auc_7_365"] = original["mean_auc_7_365"]
        summary["splits"][name] = scores
        details[name] = detail
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "dca_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "day", "threshold", "model", "treat_all", "treat_none"))
        for split, detail in details.items():
            for day, dca in detail["dca"].items():
                for i, threshold in enumerate(dca["threshold"]):
                    writer.writerow((split, day, threshold, dca["model"][i],
                                     dca["treat_all"][i], dca["treat_none"][i]))
    save_figures(details, args.output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run_dir", "aligned_dir", "structured_dir", "base_csv", "processed_event_dir",
                 "raw_ledger_dir", "field_dir", "vectors", "labels_csv", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    if not 20 <= args.bootstrap <= 500:
        raise ValueError("bootstrap draws must be between 20 and 500")
    result = run(args)
    print(json.dumps({"status": result["status"], "output": str(args.output_dir),
                      "splits": list(result["splits"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
