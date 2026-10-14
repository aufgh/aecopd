"""同一宽口径标签、同一出院顺序下配对比较摘要、原始事件与时序消融。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.v5.event_data import load_event_data
from training.v5.event_model import TemporalFusionModel
from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_data import load_data, normalize_on_training_patients, split_sequences
from training.v5.independent_fusion import IndependentFusionModel
from training.v5.train import clustered_brier_interval, metrics
from training.v5.train_event_sequence import predict as predict_temporal
from training.v5.train_independent import predict as predict_summary


def _read_report(directory: Path) -> dict[str, object]:
    return json.loads((directory / "report.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned_dir", "structured_dir", "base_csv", "raw_dir", "field_dir",
                 "vectors", "labels_csv", "summary_run", "full_run", "process_run",
                 "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("paired model evaluation requires CUDA")
    data, _ = load_event_data(
        args.aligned_dir, args.structured_dir, args.base_csv, args.raw_dir,
        labels_csv=args.labels_csv, field_dir=args.field_dir, vectors_path=args.vectors,
    )
    summary_data, _ = normalize_on_training_patients(load_data(
        args.aligned_dir, args.structured_dir, args.base_csv, args.field_dir,
        args.vectors, labels_csv=args.labels_csv,
    ))
    if not np.array_equal(data.base.hadm_ids, summary_data.hadm_ids):
        raise ValueError("event and summary admission rows differ")
    reports = {"summary": _read_report(args.summary_run),
               "full": _read_report(args.full_run),
               "process": _read_report(args.process_run)}
    if len({report["label_sha256"] for report in reports.values()}) != 1:
        raise ValueError("different outcome labels")
    if any(report["label_sha256"] != data.base.label_sha256
           or report["field_manifest_sha256"] != data.base.manifest_sha256
           or report["cohort_admissions"] != len(data.base.hadm_ids)
           for report in reports.values()):
        raise ValueError("cohort or field manifest changed")
    if (reports["full"]["raw_source_sha256"] != reports["process"]["raw_source_sha256"]
            or reports["full"]["raw_source_sha256"] != data.source_hashes):
        raise ValueError("raw event sources differ")
    device = torch.device("cuda")
    summary = IndependentFusionModel(
        concept_count=len(data.base.concept_names), type_count=len(data.base.type_names),
        field_count=len(FIELD_LABELS), text_width=data.base.text_width,
    ).to(device)
    summary.load_state_dict(torch.load(args.summary_run / "structured_field_bert.pt",
                                       map_location=device, weights_only=True))
    temporal: dict[str, TemporalFusionModel] = {}
    for name, directory in (("full", args.full_run), ("process", args.process_run)):
        model = TemporalFusionModel(
            concept_count=len(data.base.concept_names), type_count=len(data.base.type_names),
            field_count=len(FIELD_LABELS), text_width=data.base.text_width,
        ).to(device)
        model.load_state_dict(torch.load(directory / "model.pt", map_location=device,
                                         weights_only=True))
        temporal[name] = model
    result: dict[str, object] = {
        "label_sha256": data.base.label_sha256,
        "field_manifest_sha256": data.base.manifest_sha256,
        "raw_source_sha256": data.source_hashes,
        "comparison": "same broad-coded endpoint, patients, BERT vectors; validation-selected models",
        "limitations": "patient-cluster intervals quantify sampling uncertainty only; historical test is not blind",
    }
    for split in ("validation", "test"):
        sequences = split_sequences(data.base, split)
        indices = np.concatenate(sequences)
        times, events = data.base.times[indices], data.base.events[indices]
        risks = {
            "summary": predict_summary(summary, summary_data, sequences, include_text=True,
                                       batch_size=24, device=device)[0],
            "full": predict_temporal(temporal["full"], data, sequences,
                                     process_only=False, batch_size=12, device=device)[0],
            "process": predict_temporal(temporal["process"], data, sequences,
                                        process_only=True, batch_size=12, device=device)[0],
            "order_shuffled": predict_temporal(
                temporal["full"], data, sequences, process_only=False,
                batch_size=12, device=device, shuffle_lab_order_seed=2026,
            )[0],
        }
        scores = {name: metrics(times, events, risk) for name, risk in risks.items()}
        split_result: dict[str, object] = {
            "admissions": len(indices), "patients": len(sequences),
            "models": {name: {"integrated_brier_7_365": detail["integrated_brier_7_365"],
                              "mean_auc_7_365": detail["mean_auc_7_365"]}
                       for name, detail in scores.items()},
            "paired_brier_difference_95ci": {},
        }
        for name, comparator in (("full_minus_summary", "summary"),
                                 ("full_minus_process", "process"),
                                 ("order_shuffled_minus_full", "full")):
            current = risks["order_shuffled"] if name.startswith("order_") else risks["full"]
            interval = clustered_brier_interval(data.base.subject_ids[indices], times, events,
                                                 current, risks[comparator])
            delta = (scores["order_shuffled"]["integrated_brier_7_365"]
                     if name.startswith("order_") else scores["full"]["integrated_brier_7_365"])
            delta -= scores[comparator]["integrated_brier_7_365"]
            split_result["paired_brier_difference_95ci"][name] = {
                "estimate": float(delta), "patient_cluster_95ci": interval,
            }
        result["historical_test" if split == "test" else split] = split_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
