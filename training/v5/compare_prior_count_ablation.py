"""在共同患者上比较显式既往次数消融；旧基线只重载推理，不重新训练。"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from training.baselines.rank_metrics import auc_rank
from training.v5.event_set_model import EventSetFusionModel
from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_data import split_sequences
from training.v5.processed_event_data import load_processed_event_data
from training.v5.strict_endpoint_labels import STRICT_PRIMARY_EVENT
from training.v5.train import _public_metrics, _save_curves, clustered_brier_interval, metrics
from training.v5.train_event_sequence import PRIOR_COUNT_CONCEPTS, predict


def run(baseline_dir: Path, ablation_dir: Path,
        comparison_dir: Path | None = None) -> dict[str, object]:
    baseline = json.loads((baseline_dir / "report.json").read_text(encoding="utf-8"))
    ablation = json.loads((ablation_dir / "report.json").read_text(encoding="utf-8"))
    for key in ("cohort_admissions", "cohort_patients", "field_manifest_sha256", "label_sha256",
                "raw_source_sha256", "parameter_count", "event_concepts", "text_vector_width",
                "seed", "learning_rate", "weight_decay", "patient_batch_size", "max_epochs", "patience"):
        if baseline[key] != ablation[key]:
            raise ValueError(f"comparison contract differs: {key}")
    if baseline.get("disabled_static_concepts") or tuple(ablation["disabled_static_concepts"]) != PRIOR_COUNT_CONCEPTS:
        raise ValueError("incorrect count ablation")
    # 本审计固定本轮数据版本；只读源文件并复核完整输入合同。
    private = Path("outputs/01a08b96-2988-7001-af1d-b40d901f3400/private")
    aligned = private / "v5_section_aligned"
    fields = aligned / "field_text_baseline_2026_09_27"
    if baseline["primary_event"] == STRICT_PRIMARY_EVENT:
        labels_path = Path("patient_data/derived/strict_aecopd_endpoint_2026-09-28/labels.csv.gz")
    else:
        labels_path = private / "v5_qualified_copd_labels_2026_09_27/labels.csv.gz"
    data, _ = load_processed_event_data(
        aligned / "dataset_v4", aligned / "section_inputs_sparse_2026_09_26",
        private / "v5_mimiciv_2026-09-25/structured/base.csv",
        Path("patient_data/processed/mimiciv_copd_events_2026-09-27"),
        Path("patient_data/mimiciv_copd/snapshot_2026-09-27"),
        labels_csv=labels_path,
        field_dir=fields, vectors_path=fields / "bert/bioclinicalbert.float16.npy")
    if data.base.label_sha256 != baseline["label_sha256"] or data.base.manifest_sha256 != baseline["field_manifest_sha256"]:
        raise ValueError("reloaded data differs from completed run")
    device = torch.device("cuda")
    model = EventSetFusionModel(concept_count=len(data.base.concept_names),
                                type_count=len(data.base.type_names), field_count=len(FIELD_LABELS),
                                text_width=data.base.text_width,
                                event_concept_count=len(data.concept_names)).to(device)
    model.load_state_dict(torch.load(baseline_dir / "model.pt", map_location=device, weights_only=True), strict=True)
    ablated_model = EventSetFusionModel(concept_count=len(data.base.concept_names),
                                       type_count=len(data.base.type_names), field_count=len(FIELD_LABELS),
                                       text_width=data.base.text_width,
                                       event_concept_count=len(data.concept_names)).to(device)
    ablated_model.load_state_dict(torch.load(ablation_dir / "model.pt", map_location=device, weights_only=True), strict=True)
    ablated_data = replace(data, base=replace(data.base, disabled_concepts=PRIOR_COUNT_CONCEPTS))
    results: dict[str, object] = {}
    reload_deltas: dict[str, float] = {}
    curves: dict[str, dict[str, dict[str, object]]] = {}
    for name, split in (("validation", "validation"), ("historical_test", "test")):
        sequences = split_sequences(data.base, split)
        indices = np.concatenate(sequences)
        original, _ = predict(model, data, sequences, process_only=False,
                              batch_size=int(baseline["patient_batch_size"]), device=device)
        with np.load(ablation_dir / f"{name}_predictions.npz") as saved:
            reduced = saved["risk"].copy()
            for key in ("subject_ids", "hadm_ids", "times", "events"):
                if not np.array_equal(saved[key], getattr(data.base, key)[indices]):
                    raise ValueError(f"paired prediction alignment differs: {key}")
        reloaded, _ = predict(ablated_model, ablated_data, sequences, process_only=False,
                              batch_size=int(ablation["patient_batch_size"]), device=device)
        reload_deltas[name] = float(np.max(np.abs(reloaded - reduced)))
        # GPU 重载可有亚微小浮点差；记录实测最大差，超过百万分之一则失败。
        if not np.allclose(reloaded, reduced, atol=1e-6, rtol=0):
            raise ValueError(f"ablation checkpoint with count mask does not reproduce saved predictions: {name}, max risk delta={float(np.max(np.abs(reloaded - reduced)))}")
        times, events, subjects = data.base.times[indices], data.base.events[indices], data.base.subject_ids[indices]
        full_metrics, removed_metrics = metrics(times, events, original), metrics(times, events, reduced)
        expected = baseline[name]["temporal_full"]
        for key in ("mean_auc_7_365", "integrated_brier_7_365"):
            if not np.isclose(full_metrics[key], expected[key], atol=1e-7, rtol=0):
                raise ValueError(f"checkpoint prediction does not reproduce baseline: {key}")
        unique, inverse = np.unique(subjects, return_inverse=True)
        groups = [np.flatnonzero(inverse == index) for index in range(len(unique))]
        rng = np.random.default_rng(2026)
        draws = [np.concatenate([groups[index] for index in rng.integers(len(groups), size=len(groups))])
                 for _ in range(300)]
        auc_intervals = {}
        for day in (30, 90, 365):
            labels = ((events == 1) & (times <= day)).astype(np.int8)
            differences = [auc_rank(reduced[draw, day], labels[draw])
                           - auc_rank(original[draw, day], labels[draw]) for draw in draws]
            auc_intervals[str(day)] = {
                "difference": removed_metrics["points"][str(day)]["auc"] - full_metrics["points"][str(day)]["auc"],
                "patient_cluster_paired_95_interval": np.quantile(differences, (.025, .975)).tolist()}
        results[name] = {
            "baseline": _public_metrics(full_metrics), "without_explicit_counts": _public_metrics(removed_metrics),
            "mean_auc_difference": removed_metrics["mean_auc_7_365"] - full_metrics["mean_auc_7_365"],
            "integrated_brier_difference": removed_metrics["integrated_brier_7_365"] - full_metrics["integrated_brier_7_365"],
            "integrated_brier_paired_95_interval": clustered_brier_interval(subjects, times, events, reduced, original),
            "point_auc_differences": auc_intervals}
        curves[name] = {"with_explicit_counts": full_metrics, "without_explicit_counts": removed_metrics}
    report = {"status": "paired_comparison_completed", "difference_direction": "without counts minus baseline",
              "bootstrap_patient_clusters": 300, "seed": 2026,
              "baseline_checkpoint_reproduced": True, "ablation_checkpoint_reproduced": True,
              "ablation_reload_max_risk_difference": reload_deltas,
              "checkpoint_prediction_absolute_tolerance": 1e-6,
              "contract_fields_checked": True,
              "limitations": ["Single training seed; bootstrap describes sampled patients, not initialization uncertainty.",
                              "History length, gaps, diagnoses and text still carry frequency proxies.",
                              "Historical test has been examined before and is not a new blind evaluation."],
              **results}
    destination = ablation_dir if comparison_dir is None else comparison_dir
    if not destination.is_dir():
        raise ValueError("comparison destination must be an existing run directory")
    (destination / "paired_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    chart_dir = destination / "comparison"
    chart_dir.mkdir(exist_ok=True)
    _save_curves(chart_dir, curves)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for directory, label in ((baseline_dir, "with explicit counts"), (ablation_dir, "without explicit counts")):
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        epochs = [row["epoch"] for row in history]
        for axis, key, title in zip(axes,
                ("train_nll", "validation_integrated_brier", "validation_mean_auc"),
                ("Training negative log likelihood", "Validation integrated Brier", "Validation mean AUC(t)")):
            axis.plot(epochs, [row[key] for row in history], label=label)
            axis.set(xlabel="Epoch", ylabel=title)
            axis.grid(alpha=.2)
            axis.legend()
    figure.tight_layout()
    figure.savefig(chart_dir / "training_comparison.png", dpi=160)
    plt.close(figure)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path)
    arguments = parser.parse_args()
    result = run(arguments.baseline_dir, arguments.ablation_dir, arguments.comparison_dir)
    print(json.dumps({name: {key: value for key, value in result[name].items()
                            if key not in {"baseline", "without_explicit_counts"}}
                      for name in ("validation", "historical_test")}))
