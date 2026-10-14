"""同一严格结局与患者划分下，审计冻结 V5 和低容量连续时间参照。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from training.baselines.continuous_competing_cox import fit_competing_cox, predict_cumulative_incidence
from training.v5.event_set_model import EventSetFusionModel
from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_data import split_sequences
from training.v5.processed_event_data import load_processed_event_data
from training.v5.train import GRID, _public_metrics, _save_curves, clustered_brier_interval, metrics
from training.v5.train_event_sequence import predict


def _rows(base_csv: Path, hadm_ids: np.ndarray) -> list[dict[str, str]]:
    with base_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        by_hadm = {int(row["hadm_id"]): row for row in csv.DictReader(handle)}
    if set(by_hadm) != set(hadm_ids.tolist()):
        raise ValueError("simple baseline does not match model admissions")
    return [by_hadm[int(hadm)] for hadm in hadm_ids]


def _features(rows: list[dict[str, str]], train_mask: np.ndarray) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """预固定五项出院可得特征；只使用训练集拟合年龄中心。"""
    age = np.array([float(row["age_at_admission"]) for row in rows])
    if not np.isfinite(age).all():
        raise ValueError("nonfinite age in simple baseline")
    center = float(np.median(age[train_mask]))
    gender = [row["gender"].strip().upper() for row in rows]
    if set(gender) - {"M", "F"}:
        raise ValueError("unexpected gender in simple baseline")
    prior = np.array([int(row["prior_coded_admission_count"]) for row in rows])
    prior_strict = np.array([int(row["prior_strict_aecopd_count"]) for row in rows])
    strict_values = [row["index_is_strict_aecopd"].strip().lower() for row in rows]
    if set(strict_values) - {"t", "f"}:
        raise ValueError("unexpected current strict code flag")
    strict_now = np.array([value == "t" for value in strict_values], dtype=float)
    if (prior < 0).any() or (prior_strict < 0).any():
        raise ValueError("invalid admission history or current strict code")
    x = np.column_stack((
        (age - center) / 10.0,
        np.array([value == "M" for value in gender], dtype=float),
        np.log1p(prior),
        np.log1p(prior_strict),
        strict_now,
    ))
    return x, ["age_per_10_years", "male", "log1p_prior_coded", "log1p_prior_strict", "current_strict_code"], {"age_median_train": center}


def _check_saved(path: Path, data, indices: np.ndarray, risk: np.ndarray) -> None:
    saved = np.load(path)
    for key, actual in (("subject_ids", data.base.subject_ids[indices]),
                        ("hadm_ids", data.base.hadm_ids[indices]),
                        ("events", data.base.events[indices])):
        if not np.array_equal(saved[key], actual):
            raise ValueError(f"saved model predictions differ on {key}")
    if not np.allclose(saved["times"], data.base.times[indices], atol=1e-7, rtol=0):
        raise ValueError("saved model predictions differ on event times")
    if not np.allclose(saved["risk"], risk, atol=2e-6, rtol=0):
        raise ValueError("reloaded checkpoint does not reproduce saved risks")


def _calibration(times: np.ndarray, events: np.ndarray, risk: np.ndarray) -> dict[str, object]:
    answer = {}
    for day in (30, 90, 365):
        predicted = risk[:, day]
        observed = (events == 1) & (times <= day)
        edges = np.quantile(predicted, np.linspace(0, 1, 6))
        groups = []
        for bin_index in range(5):
            selected = (predicted >= edges[bin_index]) & (
                predicted < edges[bin_index + 1] if bin_index < 4 else predicted <= edges[bin_index + 1]
            )
            groups.append({"n": int(selected.sum()),
                           "predicted": float(predicted[selected].mean()) if selected.any() else None,
                           "observed_world_a": float(observed[selected].mean()) if selected.any() else None})
        answer[str(day)] = groups
    return answer


def _subgroups(data, rows: list[dict[str, str]], indices: np.ndarray, risk: np.ndarray) -> dict[str, object]:
    prior = np.array([int(row["prior_coded_admission_count"]) for row in rows])[indices]
    strict = np.array([row["index_is_strict_aecopd"].strip().lower() == "t" for row in rows])[indices]
    groups = {
        "first_coded_admission": prior == 0,
        "one_to_two_prior": (prior >= 1) & (prior <= 2),
        "three_plus_prior": prior >= 3,
        "current_strict_code": strict == 1,
        "current_non_strict_code": strict == 0,
    }
    out = {}
    for name, mask in groups.items():
        if not mask.any():
            continue
        detail = metrics(data.base.times[indices][mask], data.base.events[indices][mask], risk[mask])
        out[name] = {"patients": int(np.unique(data.base.subject_ids[indices][mask]).size),
                     **_public_metrics(detail)}
    return out


def _masked_data(data, name: str):
    if name == "text_absent":
        return replace(data, base=replace(data.base, text_by_admission=tuple(() for _ in data.base.hadm_ids)))
    if name == "events_absent":
        return replace(data, events_by_admission=tuple(np.empty(0, dtype=np.int64) for _ in data.base.hadm_ids))
    if name == "prior_counts_absent":
        missing = ("context:prior_coded_admissions", "context:prior_strict_aecopd_admissions")
        if not set(missing).issubset(data.base.concept_names):
            raise ValueError("prior count concepts missing")
        return replace(data, base=replace(data.base, disabled_concepts=missing))
    if name == "medications_absent":
        return replace(data, events_by_admission=tuple(refs[data.event_is_lab[refs]] for refs in data.events_by_admission))
    raise ValueError(name)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    existing = json.loads((args.candidate_dir / "report.json").read_text(encoding="utf-8"))
    if existing["status"] != "training_completed" or existing["event_encoder"] != "set":
        raise ValueError("candidate is not a complete set-encoder run")
    data, _ = load_processed_event_data(
        args.aligned_dir, args.structured_dir, args.base_csv, args.processed_event_dir,
        args.raw_ledger_dir, labels_csv=args.labels_csv, field_dir=args.field_dir,
        vectors_path=args.vectors,
    )
    if (existing["label_sha256"] != data.base.label_sha256 or
            existing["field_manifest_sha256"] != data.base.manifest_sha256 or
            existing["raw_source_sha256"] != data.source_hashes or
            existing["text_vector_width"] != data.base.text_width):
        raise ValueError("candidate data contract differs from current inputs")
    model = EventSetFusionModel(
        concept_count=len(data.base.concept_names), type_count=len(data.base.type_names),
        field_count=len(FIELD_LABELS), text_width=data.base.text_width,
        event_concept_count=len(data.concept_names),
    )
    model.load_state_dict(torch.load(args.candidate_dir / "model.pt", map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rows = _rows(args.base_csv, data.base.hadm_ids)
    train_mask = data.base.splits == "train"
    x, feature_names, preprocessing = _features(rows, train_mask)
    cox = fit_competing_cox(x[train_mask], data.base.times[train_mask], data.base.events[train_mask], l2=10.0)
    if not cox.readmission.converged or not cox.death.converged:
        raise ValueError("simple Cox baseline failed to converge")
    sequences = {split: split_sequences(data.base, split) for split in ("validation", "test")}
    all_curves = {}
    results = {}
    for split, members in sequences.items():
        label = "historical_test" if split == "test" else split
        indices = np.concatenate(members)
        times, events = data.base.times[indices], data.base.events[indices]
        full = predict(model, data, members, process_only=False, batch_size=12, device=device)[0]
        _check_saved(args.candidate_dir / f"{label}_predictions.npz", data, indices, full)
        simple = predict_cumulative_incidence(cox, x[indices], GRID)[:, :, 1]
        curves = {"v5_full": metrics(times, events, full),
                  "five_feature_cox": metrics(times, events, simple)}
        variants = {"lab_values_zeroed": predict(model, data, members, process_only=True, batch_size=12, device=device)[0]}
        for name in ("text_absent", "events_absent", "medications_absent", "prior_counts_absent"):
            variants[name] = predict(model, _masked_data(data, name), members,
                                     process_only=False, batch_size=12, device=device)[0]
        for name, risk in variants.items():
            curves[name] = metrics(times, events, risk)
        all_curves[label] = curves
        subject_ids = data.base.subject_ids[indices]
        results[label] = {
            "models": {name: _public_metrics(detail) for name, detail in curves.items()},
            "subgroups_v5": _subgroups(data, rows, indices, full),
            "subgroups_five_feature_cox": _subgroups(data, rows, indices, simple),
            "calibration_v5": _calibration(times, events, full),
            "calibration_five_feature_cox": _calibration(times, events, simple),
            "paired_ibs_difference_v5_minus_comparator": {
                name: {"point": curves["v5_full"]["integrated_brier_7_365"] - detail["integrated_brier_7_365"],
                       "patient_cluster_95pct_interval": clustered_brier_interval(subject_ids, times, events, full,
                                                                                     simple if name == "five_feature_cox" else variants[name])}
                for name, detail in curves.items() if name != "v5_full"
            },
        }
        print(json.dumps({"split": label, "v5_mean_auc": curves["v5_full"]["mean_auc_7_365"],
                          "cox_mean_auc": curves["five_feature_cox"]["mean_auc_7_365"]}), flush=True)
    args.output_dir.mkdir(parents=True)
    _save_curves(args.output_dir, all_curves)
    report = {
        "status": "internal_validation_completed",
        "candidate_dir": str(args.candidate_dir),
        "label_sha256": data.base.label_sha256,
        "field_manifest_sha256": data.base.manifest_sha256,
        "source_sha256": data.source_hashes,
        "device": str(device),
        "simple_baseline": {"model": "two cause-specific Cox models with cumulative incidence",
                            "features": feature_names, "l2": 10.0, "preprocessing": preprocessing,
                            "coefficients_readmission": cox.readmission.coefficients.tolist(),
                            "coefficients_death": cox.death.coefficients.tolist(),
                            "fit_on": "training patients only"},
        "splits": results,
        "limitations": [
            "World A: no-event discharge observed to day 365; out-of-system readmissions and death unmeasured.",
            "AUC(t) uses observed strict readmission by day t versus all others, including prior competing death.",
            "Input masking is an out-of-distribution frozen-checkpoint sensitivity test, not a retraining ablation.",
            "Historical test patients were used in earlier model exploration, so this is not a fresh blind test.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned_dir", "structured_dir", "base_csv", "processed_event_dir", "raw_ledger_dir",
                 "field_dir", "vectors", "labels_csv", "candidate_dir", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
