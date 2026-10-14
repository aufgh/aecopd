"""结局改变的二维审计：固定预测换标签，与相同标签下重新训练分开比较。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.baselines.rank_metrics import auc_rank
from training.v5.strict_endpoint_labels import STRICT_PRIMARY_EVENT
from training.v5.train import _public_metrics, _save_curves, clustered_brier_interval, metrics
from training.v5.train_event_sequence import PRIOR_COUNT_CONCEPTS


def compare(union_dir: Path, strict_dir: Path) -> dict[str, object]:
    union_report = json.loads((union_dir / "report.json").read_text(encoding="utf-8"))
    strict_report = json.loads((strict_dir / "report.json").read_text(encoding="utf-8"))
    if strict_report["primary_event"] != STRICT_PRIMARY_EVENT:
        raise ValueError("wrong strict endpoint")
    fields = ("cohort_admissions", "cohort_patients", "field_manifest_sha256", "raw_source_sha256",
              "parameter_count", "event_concepts", "text_vector_width", "seed", "learning_rate",
              "weight_decay", "patient_batch_size", "max_epochs", "patience", "event_encoder")
    for key in fields:
        if union_report[key] != strict_report[key]:
            raise ValueError("non-outcome contract differs: " + key)
    if any(tuple(r["disabled_static_concepts"]) != PRIOR_COUNT_CONCEPTS for r in (union_report, strict_report)):
        raise ValueError("count ablation differs")
    results: dict[str, object] = {}
    curves: dict[str, dict[str, dict[str, object]]] = {}
    for split in ("validation", "historical_test"):
        with np.load(union_dir / f"{split}_predictions.npz") as archive:
            union = {name: archive[name] for name in archive.files}
        with np.load(strict_dir / f"{split}_predictions.npz") as archive:
            strict = {name: archive[name] for name in archive.files}
        for key in ("subject_ids", "hadm_ids"):
            if not np.array_equal(union[key], strict[key]):
                raise ValueError("prediction member/order differs: " + key)
        table = {}
        for endpoint, labels in (("union", union), ("strict", strict)):
            detailed = {"union_trained_model": metrics(labels["times"], labels["events"], union["risk"]),
                        "strict_trained_model": metrics(labels["times"], labels["events"], strict["risk"])}
            curves[split + "_" + endpoint] = detailed
            table[endpoint + "_evaluation"] = {key: _public_metrics(value) for key, value in detailed.items()}
        # 主训练结果必须逐指标复现，禁止张冠李戴地混用结局。
        for report, endpoint, model in ((union_report, "union", "union_trained_model"),
                                         (strict_report, "strict", "strict_trained_model")):
            for key in ("mean_auc_7_365", "integrated_brier_7_365"):
                if not np.isclose(table[endpoint + "_evaluation"][model][key], report[split]["temporal_full"][key], atol=1e-8, rtol=0):
                    raise ValueError("diagonal evaluation does not reproduce report")
        subjects, times, events = strict["subject_ids"], strict["times"], strict["events"]
        patients, inverse = np.unique(subjects, return_inverse=True)
        groups = [np.flatnonzero(inverse == i) for i in range(len(patients))]
        rng = np.random.default_rng(2026)
        samples = [np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
                   for _ in range(300)]
        intervals = {}
        for day in (30, 90, 365):
            observed = ((events == 1) & (times <= day)).astype(np.int8)
            differences = [auc_rank(strict["risk"][draw, day], observed[draw])
                           - auc_rank(union["risk"][draw, day], observed[draw]) for draw in samples]
            intervals[str(day)] = {"strict_retraining_auc_difference":
                table["strict_evaluation"]["strict_trained_model"]["points"][str(day)]["auc"]
                - table["strict_evaluation"]["union_trained_model"]["points"][str(day)]["auc"],
                "patient_cluster_paired_95_interval": np.quantile(differences, (.025, .975)).tolist()}
        strict_scores = table["strict_evaluation"]
        results[split] = {**table,
            "strict_endpoint_retraining_mean_auc_difference": strict_scores["strict_trained_model"]["mean_auc_7_365"] - strict_scores["union_trained_model"]["mean_auc_7_365"],
            "strict_endpoint_retraining_brier_difference": strict_scores["strict_trained_model"]["integrated_brier_7_365"] - strict_scores["union_trained_model"]["integrated_brier_7_365"],
            "strict_endpoint_retraining_brier_paired_95_interval": clustered_brier_interval(subjects, times, events, strict["risk"], union["risk"]),
            "strict_endpoint_point_auc_intervals": intervals,
            "event_counts": {endpoint: {str(i): int((labels["events"] == i).sum()) for i in range(3)}
                             for endpoint, labels in (("union", union), ("strict", strict))}}
    result = {"status": "endpoint_comparison_completed", "only_training_contract_change": "outcome event definition and derived labels",
              "patient_cluster_bootstrap_replicates": 300, "bootstrap_seed": 2026,
              "comparison_matrix": "rows=evaluation endpoint; columns=model training endpoint",
              "limitations": ["Different endpoints change cases/controls: diagonal AUC difference is not architecture improvement.",
                              "Brier values across different event prevalences are not direct performance improvements.",
                              "Single training seed; intervals do not include initialization uncertainty.",
                              "Historical test was previously examined; MIMIC outside-hospital follow-up remains incomplete."],
              **results}
    (strict_dir / "endpoint_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    destination = strict_dir / "comparison"
    destination.mkdir(exist_ok=True)
    _save_curves(destination, curves)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--union-dir", type=Path, required=True)
    parser.add_argument("--strict-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.union_dir, args.strict_dir)
    print(json.dumps({split: {name: {model: {metric: score[metric] for metric in ("mean_auc_7_365", "integrated_brier_7_365")}
                                   for model, score in result[split][name].items()}
                             for name in ("union_evaluation", "strict_evaluation")}
                      for split in ("validation", "historical_test")}))
