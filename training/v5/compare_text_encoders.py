"""在同一 V5 患者划分上配对比较冻结 BERT 和 Qwen 文本编码。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.v5.data import load_data
from training.v5.model import V5Model
from training.v5.train import clustered_brier_interval, metrics, predict, preprocess, split_sequences


def _risk(structured: Path, manifest: Path, vectors: Path, run_dir: Path,
          model_name: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
    training_report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    data, _ = preprocess(load_data(structured, manifest, vectors,
                                   vector_width=int(training_report.get("text_input_width", 768))))
    sequences = split_sequences(data, split)
    selected = np.concatenate(sequences)
    model = V5Model(text_width=data.text_width,
                    balance_text=bool(training_report.get("balanced_text_content", False))).to("cuda")
    model.load_state_dict(torch.load(run_dir / f"{model_name}.pt", map_location="cuda",
                                     weights_only=True))
    risk, _ = predict(model, data, sequences, include_text=True,
                      device=torch.device("cuda"), batch_size=24)
    return data.hadm_ids[selected], data.subject_ids[selected], data.times[selected], data.events[selected], risk


def run(structured: Path, manifest: Path, experiments: dict[str, tuple[Path, Path, str]],
        output: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint comparison requires CUDA")
    if output.exists():
        raise FileExistsError(output)
    report: dict[str, object] = {"primary_metric": "World A competing-risk IBS days 7-365",
                                 "patient_paired_bootstrap": 300, "splits": {}}
    for split in ("validation", "test"):
        predictions = {name: _risk(structured, manifest, vectors, directory, model_name, split)
                       for name, (vectors, directory, model_name) in experiments.items()}
        reference = predictions["bert"]
        summary: dict[str, object] = {"models": {}, "paired_differences_vs_bert": {}}
        for name, values in predictions.items():
            if not all(np.array_equal(a, b) for a, b in zip(reference[:4], values[:4], strict=True)):
                raise ValueError("encoder comparison rows or labels do not align")
            measured = metrics(values[2], values[3], values[4])
            summary["models"][name] = {
                "integrated_brier_7_365": measured["integrated_brier_7_365"],
                "mean_auc_7_365": measured["mean_auc_7_365"],
                "auc_365": measured["points"]["365"]["auc"],
                "mean_risk_365": measured["points"]["365"]["mean_predicted_risk"],
            }
            if name != "bert":
                difference = (float(measured["integrated_brier_7_365"])
                              - float(summary["models"]["bert"]["integrated_brier_7_365"]))
                summary["paired_differences_vs_bert"][name] = {
                    "brier_difference": difference,
                    "patient_cluster_95ci": clustered_brier_interval(
                        values[1], values[2], values[3], values[4], reference[4]
                    ),
                }
        narrow = predictions["qwen768"]
        full = predictions["qwen2560"]
        summary["qwen2560_minus_qwen768"] = {
            "brier_difference": (
                float(summary["models"]["qwen2560"]["integrated_brier_7_365"])
                - float(summary["models"]["qwen768"]["integrated_brier_7_365"])
            ),
            "patient_cluster_95ci": clustered_brier_interval(
                full[1], full[2], full[3], full[4], narrow[4]
            ),
        }
        if "qwen128" in predictions:
            smallest = predictions["qwen128"]
            summary["qwen768_minus_qwen128"] = {
                "brier_difference": (
                    float(summary["models"]["qwen768"]["integrated_brier_7_365"])
                    - float(summary["models"]["qwen128"]["integrated_brier_7_365"])
                ),
                "patient_cluster_95ci": clustered_brier_interval(
                    narrow[1], narrow[2], narrow[3], narrow[4], smallest[4]
                ),
            }
        for baseline, balanced in (("bert", "bert_balanced"),
                                   ("qwen2560", "qwen2560_balanced")):
            if balanced not in predictions:
                continue
            original, normalized = predictions[baseline], predictions[balanced]
            summary[f"{balanced}_minus_{baseline}"] = {
                "brier_difference": (
                    float(summary["models"][balanced]["integrated_brier_7_365"])
                    - float(summary["models"][baseline]["integrated_brier_7_365"])
                ),
                "patient_cluster_95ci": clustered_brier_interval(
                    normalized[1], normalized[2], normalized[3], normalized[4], original[4]
                ),
            }
        report["splits"]["historical_test" if split == "test" else split] = summary
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bert-vectors", type=Path, required=True)
    parser.add_argument("--bert-run", type=Path, required=True)
    parser.add_argument("--qwen-vectors", type=Path, required=True)
    parser.add_argument("--qwen768-run", type=Path, required=True)
    parser.add_argument("--qwen2560-run", type=Path, required=True)
    parser.add_argument("--qwen128-run", type=Path)
    parser.add_argument("--bert-balanced-run", type=Path)
    parser.add_argument("--qwen2560-balanced-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiments = {
        "bert": (args.bert_vectors, args.bert_run, "structured_bert"),
        "qwen768": (args.qwen_vectors, args.qwen768_run, "structured_qwen768"),
        "qwen2560": (args.qwen_vectors, args.qwen2560_run, "structured_qwen2560"),
    }
    if args.qwen128_run is not None:
        experiments["qwen128"] = (args.qwen_vectors, args.qwen128_run, "structured_qwen128")
    if args.bert_balanced_run is not None:
        experiments["bert_balanced"] = (args.bert_vectors, args.bert_balanced_run,
                                         "structured_bert")
    if args.qwen2560_balanced_run is not None:
        experiments["qwen2560_balanced"] = (args.qwen_vectors,
                                             args.qwen2560_balanced_run,
                                             "structured_qwen2560")
    result = run(args.structured_dir, args.manifest, experiments, args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
