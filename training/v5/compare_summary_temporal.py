"""在相同患者、字段文本及结局下配对比较住院摘要与原始事件时序。"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from training.v5.event_data import load_event_data
from training.v5.event_model import TemporalFusionModel
from training.v5.event_set_model import EventSetFusionModel
from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_data import load_data, normalize_on_training_patients, split_sequences
from training.v5.independent_fusion import IndependentFusionModel
from training.v5.train import clustered_brier_interval, metrics
from training.v5.train_event_sequence import predict as predict_temporal
from training.v5.train_independent import predict as predict_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned_dir", "structured_dir", "base_csv", "raw_dir", "field_dir",
                 "vectors", "labels_csv", "summary_run", "temporal_run", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--gru-run", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("paired comparison requires CUDA")
    summary_data, _ = normalize_on_training_patients(load_data(
        args.aligned_dir, args.structured_dir, args.base_csv,
        args.field_dir, args.vectors, labels_csv=args.labels_csv,
    ))
    event_data, _ = load_event_data(
        args.aligned_dir, args.structured_dir, args.base_csv, args.raw_dir,
        labels_csv=args.labels_csv, field_dir=args.field_dir, vectors_path=args.vectors,
    )
    summary_report = json.loads((args.summary_run / "report.json").read_text(encoding="utf-8"))
    temporal_report = json.loads((args.temporal_run / "report.json").read_text(encoding="utf-8"))
    if (not np.array_equal(summary_data.hadm_ids, event_data.base.hadm_ids)
            or summary_data.label_sha256 != event_data.base.label_sha256
            or summary_report["label_sha256"] != summary_data.label_sha256
            or temporal_report["label_sha256"] != summary_data.label_sha256
            or summary_report["field_manifest_sha256"] != temporal_report["field_manifest_sha256"]
            or temporal_report["raw_source_sha256"] != event_data.source_hashes):
        raise ValueError("paired comparison cohort, label or source mismatch")
    device = torch.device("cuda")
    summary_model = IndependentFusionModel(
        concept_count=len(summary_data.concept_names), type_count=len(summary_data.type_names),
        field_count=len(FIELD_LABELS), text_width=summary_data.text_width,
    ).to(device)
    summary_model.load_state_dict(torch.load(
        args.summary_run / "structured_field_bert.pt", map_location=device, weights_only=True,
    ))
    model_class = EventSetFusionModel if temporal_report.get("event_encoder") == "set" else TemporalFusionModel
    event_model = model_class(
        concept_count=len(event_data.base.concept_names),
        type_count=len(event_data.base.type_names),
        field_count=len(FIELD_LABELS), text_width=event_data.base.text_width,
    ).to(device)
    event_model.load_state_dict(torch.load(
        args.temporal_run / "model.pt", map_location=device, weights_only=True,
    ))
    gru_model = None
    if args.gru_run is not None:
        reference = json.loads((args.gru_run / "report.json").read_text(encoding="utf-8"))
        for key in ("label_sha256", "field_manifest_sha256", "raw_source_sha256"):
            if reference[key] != temporal_report[key]:
                raise ValueError(f"GRU comparison differs: {key}")
        gru_model = TemporalFusionModel(
            concept_count=len(event_data.base.concept_names),
            type_count=len(event_data.base.type_names), field_count=len(FIELD_LABELS),
            text_width=event_data.base.text_width,
        ).to(device)
        gru_model.load_state_dict(torch.load(args.gru_run / "model.pt", map_location=device,
                                             weights_only=True))
    # 仅反转同时间戳的完整观测行，不破坏项目、数值和时间的绑定。
    permutation = np.arange(len(event_data.event_concepts))
    tied_rows = 0
    for refs in event_data.events_by_admission:
        if len(refs) < 2:
            continue
        for time in np.unique(event_data.event_values[refs, 1]):
            tied = refs[event_data.event_values[refs, 1] == time]
            if len(tied) > 1:
                permutation[tied] = tied[::-1]
                tied_rows += len(tied)
    reordered_data = replace(event_data,
                             event_values=event_data.event_values[permutation],
                             event_concepts=event_data.event_concepts[permutation],
                             event_is_lab=event_data.event_is_lab[permutation])
    report: dict[str, object] = {
        "primary_event": summary_data.label_name,
        "label_sha256": summary_data.label_sha256,
        "field_manifest_sha256": summary_data.manifest_sha256,
        "raw_source_sha256": event_data.source_hashes,
        "event_encoder": temporal_report.get("event_encoder", "gru"),
        "same_timestamp_reorder_total_rows": tied_rows,
        "comparison": "same outcome, cohort, patient split and frozen BERT text; validation-selected checkpoints",
    }
    for split in ("validation", "test"):
        sequences = split_sequences(summary_data, split)
        indices = np.concatenate(sequences)
        times, events = summary_data.times[indices], summary_data.events[indices]
        risks = {
            "summary": predict_summary(summary_model, summary_data, sequences,
                                       include_text=True, batch_size=24, device=device)[0],
            "temporal": predict_temporal(event_model, event_data, sequences,
                                         process_only=False, batch_size=12, device=device)[0],
            "lab_order_shuffled": predict_temporal(
                event_model, event_data, sequences, process_only=False,
                batch_size=12, device=device, shuffle_lab_order_seed=2026,
            )[0],
            "same_timestamp_reordered": predict_temporal(
                event_model, reordered_data, sequences, process_only=False,
                batch_size=12, device=device,
            )[0],
            "lab_values_zeroed": predict_temporal(
                event_model, event_data, sequences, process_only=True,
                batch_size=12, device=device,
            )[0],
        }
        if gru_model is not None:
            risks["gru"] = predict_temporal(gru_model, event_data, sequences,
                                             process_only=False, batch_size=12, device=device)[0]
        scores = {name: metrics(times, events, risk) for name, risk in risks.items()}
        paired = {}
        comparisons = [
            ("temporal_minus_summary", "temporal", "summary"),
            ("lab_order_shuffled_minus_temporal", "lab_order_shuffled", "temporal"),
            ("lab_values_zeroed_minus_temporal", "lab_values_zeroed", "temporal"),
        ]
        if gru_model is not None:
            comparisons.append(("temporal_minus_gru", "temporal", "gru"))
        for name, first, second in comparisons:
            paired[name] = {
                "estimate": float(scores[first]["integrated_brier_7_365"]
                                  - scores[second]["integrated_brier_7_365"]),
                "patient_cluster_95ci": clustered_brier_interval(
                    summary_data.subject_ids[indices], times, events,
                    risks[first], risks[second],
                ),
            }
        report["historical_test" if split == "test" else split] = {
            "lab_values_zeroed_max_abs_risk_change": float(np.max(np.abs(
                risks["lab_values_zeroed"] - risks["temporal"]))),
            "same_timestamp_max_abs_risk_change": float(np.max(np.abs(
                risks["same_timestamp_reordered"] - risks["temporal"]))),
            "same_timestamp_invariance_pass": bool(np.allclose(
                risks["same_timestamp_reordered"], risks["temporal"], atol=1e-5, rtol=1e-5)),
            "admissions": len(indices), "patients": len(sequences),
            "models": {name: {"readmissions": detail["readmissions"],
                              "deaths": detail["deaths"],
                              "integrated_brier_7_365": detail["integrated_brier_7_365"],
                              "mean_auc_7_365": detail["mean_auc_7_365"]}
                       for name, detail in scores.items()},
            "paired_brier_difference_95ci": paired,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
