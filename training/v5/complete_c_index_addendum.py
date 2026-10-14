"""对已完成的 V5 检查点只读重放，补充竞争风险 C-index 和整条曲线检查。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.v5.competing_concordance import cif_concordance
from training.v5.evaluate_completed_event_run import replay
from training.v5.event_set_model import EventSetFusionModel
from training.v5.independent_data import split_sequences
from training.v5.processed_event_data import load_processed_event_data
from training.v5.train import metrics


def run(args: argparse.Namespace) -> dict[str, object]:
    target = args.output_dir / "time_ranking_addendum.json"
    if not args.output_dir.is_dir() or target.exists():
        raise ValueError("evaluation output missing or C-index addendum already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint replay requires CUDA")
    report = json.loads((args.run_dir / "report.json").read_text(encoding="utf-8"))
    data, _ = load_processed_event_data(
        args.aligned_dir, args.structured_dir, args.base_csv,
        args.processed_event_dir, args.raw_ledger_dir, labels_csv=args.labels_csv,
        field_dir=args.field_dir, vectors_path=args.vectors)
    if (report["cohort_admissions"] != len(data.base.hadm_ids)
            or report["label_sha256"] != data.base.label_sha256
            or report["field_manifest_sha256"] != data.base.manifest_sha256
            or report["event_concepts"] != list(data.concept_names)):
        raise ValueError("checkpoint cohort/input contract mismatch")
    model = EventSetFusionModel(
        concept_count=len(data.base.concept_names), type_count=len(data.base.type_names),
        field_count=len(data.base.field_roles), text_width=data.base.text_width,
        event_concept_count=len(data.concept_names)).to("cuda").eval()
    state = torch.load(args.run_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    result: dict[str, object] = {
        "status": "completed", "source_run": str(args.run_dir),
        "definition": "Wolbers-style cause-specific truncated concordance using CIF at the stated horizon; competing deaths remain comparators, not censored",
        "observation_contract": "World A: event_type=0 observed to day 365; no early censoring",
        "limitations": ["Repeated discharges contribute admission-level pairs; no patient-cluster C-index interval was calculated.",
                        "Historical test is not blind; outside-hospital readmission may be missed."],
        "splits": {},
    }
    for split, key in (("validation", "validation"), ("test", "historical_test")):
        sequences = split_sequences(data.base, split)
        indices = np.concatenate(sequences)
        curves = replay(model, data, sequences, torch.device("cuda"))
        reference = report[key]["temporal_full"]
        replayed = metrics(data.base.times[indices], data.base.events[indices], curves[:, :, 1])
        for metric in ("integrated_brier_7_365", "mean_auc_7_365"):
            if not np.isclose(replayed[metric], reference[metric], atol=1e-7):
                raise ValueError("checkpoint replay metrics do not match original report")
        times, events = data.base.times[indices], data.base.events[indices]
        split_result = {"n_admissions": len(indices), "readmission": {}, "death": {}}
        for cause, name in ((1, "readmission"), (2, "death")):
            cifs = curves[:, :, cause]
            if np.any(np.diff(cifs, axis=1) < -1e-6):
                raise ValueError("predicted CIF curve is not monotone")
            for day in (90, 365):
                split_result[name][str(day)] = cif_concordance(times, events, cifs[:, day],
                                                               cause=cause, horizon=day)
            days = np.arange(7, 366)
            observed = np.array([np.mean((events == cause) & (times <= day)) for day in days])
            predicted = cifs[:, 7:366].mean(axis=0)
            split_result[name]["mean_absolute_population_curve_gap_7_365"] = float(np.mean(np.abs(predicted - observed)))
            split_result[name]["maximum_absolute_population_curve_gap_7_365"] = float(np.max(np.abs(predicted - observed)))
        result["splits"][key] = split_result
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run_dir", "aligned_dir", "structured_dir", "base_csv", "processed_event_dir",
                 "raw_ledger_dir", "field_dir", "vectors", "labels_csv", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps({"status": run(args)["status"], "output": str(args.output_dir / "time_ranking_addendum.json")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
