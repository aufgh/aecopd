"""同一旧结局、同一患者划分上配对比较独立分支 BERT 与 Qwen。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_data import load_data, normalize_on_training_patients, split_sequences
from training.v5.independent_fusion import IndependentFusionModel
from training.v5.train import clustered_brier_interval, metrics
from training.v5.train_independent import predict


def _read(run_dir: Path, name: str) -> dict[str, object]:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("text_encoder") != name:
        raise ValueError("encoder report does not match experiment")
    return report


def _risk(aligned_dir: Path, structured_dir: Path, base_csv: Path,
          field_dir: Path, vectors: Path, run_dir: Path, name: str,
          split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data, _ = normalize_on_training_patients(
        load_data(aligned_dir, structured_dir, base_csv, field_dir, vectors)
    )
    sequences = split_sequences(data, split)
    selected = np.concatenate(sequences)
    model = IndependentFusionModel(
        concept_count=len(data.concept_names), type_count=len(data.type_names),
        field_count=len(FIELD_LABELS), text_width=data.text_width,
    ).to("cuda")
    model.load_state_dict(torch.load(run_dir / f"structured_field_{name}.pt",
                                     map_location="cuda", weights_only=True))
    risk, _ = predict(model, data, sequences, include_text=True,
                      batch_size=24, device=torch.device("cuda"))
    return data.hadm_ids[selected], data.subject_ids[selected], data.times[selected], data.events[selected], risk


def compare(aligned_dir: Path, structured_dir: Path, base_csv: Path, field_dir: Path,
            bert_vectors: Path, bert_run: Path, qwen_vectors: Path, qwen_run: Path,
            output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("paired encoder comparison requires CUDA")
    reports = {"bert": _read(bert_run, "bert"), "qwen": _read(qwen_run, "qwen")}
    for key in ("field_manifest_sha256", "cohort_admissions", "cohort_patients", "split_counts"):
        if reports["bert"][key] != reports["qwen"][key]:
            raise ValueError(f"comparison cohort contract differs: {key}")
    for split in ("validation", "historical_test"):
        left = reports["bert"][split]["structured"]
        right = reports["qwen"][split]["structured"]
        if left != right:
            raise ValueError(f"structured comparator differs in {split}; encoder comparison invalid")
    bert_history = json.loads((bert_run / "structured_history.json").read_text(encoding="utf-8"))
    qwen_history = json.loads((qwen_run / "structured_history.json").read_text(encoding="utf-8"))
    if bert_history != qwen_history or reports["bert"]["best_epochs"]["structured"] != reports["qwen"]["best_epochs"]["structured"]:
        raise ValueError("structured training histories differ; random streams are not aligned")
    summaries: dict[str, object] = {}
    for data_split, name in (("validation", "validation"), ("test", "historical_test")):
        bert = _risk(aligned_dir, structured_dir, base_csv, field_dir,
                     bert_vectors, bert_run, "bert", data_split)
        qwen = _risk(aligned_dir, structured_dir, base_csv, field_dir,
                     qwen_vectors, qwen_run, "qwen", data_split)
        if not all(np.array_equal(a, b) for a, b in zip(bert[:4], qwen[:4], strict=True)):
            raise ValueError("comparison rows or labels differ")
        b = metrics(bert[2], bert[3], bert[4])
        q = metrics(qwen[2], qwen[3], qwen[4])
        summaries[name] = {
            "bert": {"integrated_brier_7_365": b["integrated_brier_7_365"],
                     "mean_auc_7_365": b["mean_auc_7_365"]},
            "qwen": {"integrated_brier_7_365": q["integrated_brier_7_365"],
                     "mean_auc_7_365": q["mean_auc_7_365"]},
            "qwen_minus_bert_integrated_brier":
                float(q["integrated_brier_7_365"] - b["integrated_brier_7_365"]),
            "patient_cluster_95ci": clustered_brier_interval(
                bert[1], bert[2], bert[3], qwen[4], bert[4]
            ),
        }
    result = {
        "status": "single-seed internal comparison; old strict outcome; historical test not blind",
        "manifest_sha256": reports["bert"]["field_manifest_sha256"],
        "identical_structured_baselines": True,
        "identical_structured_histories": True,
        "splits": summaries,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--field-dir", type=Path, required=True)
    parser.add_argument("--bert-vectors", type=Path, required=True)
    parser.add_argument("--bert-run", type=Path, required=True)
    parser.add_argument("--qwen-vectors", type=Path, required=True)
    parser.add_argument("--qwen-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.aligned_dir, args.structured_dir, args.base_csv,
                             args.field_dir, args.bert_vectors, args.bert_run,
                             args.qwen_vectors, args.qwen_run, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
