"""从已冻结的最佳检查点纯评估 M1/B3，落地测试集逐住院预测并做患者聚类 bootstrap。

⚠ 纯评估：加载 checkpoints/best.pt，仅在测试集前向一次，不训练、不再用测试集做任何
模型选择（最佳轮早已在验证集选定）。目的只是给"已定型"模型的测试指标加患者聚类
95%CI，使 M1/B3/B2/B1 四者可并列比较。

复用训练脚本的全部构建块（同 collator/cache/outcomes/batcher），唯一新增：按展平住院
顺序收集 hadm_id，映射到 subject_id 供聚类 bootstrap；再复用 patient_clustered_bootstrap。

受控数据：不打印任何真实 subject_id/hadm_id，只落盘到受控 private 目录。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, "/workspace/training")
sys.path.insert(0, "/workspace/model_loader")

from aecopd_loader import AecopdDataBundle
from timeline_collator import SharedTimelineCollator, TimelineConfig

from aecopd_training.aligned_model import AlignedModelConfig
from aecopd_training.batching import load_timeline_lengths
from aecopd_training.conditional_model import ConditionalIntervalAecopdModel
from aecopd_training.conditional_outcomes import ConditionalOutcomeStore
from aecopd_training.longitudinal_batching import (
    LongitudinalBatchingConfig,
    PatientSequenceBatcher,
)
from aecopd_training.longitudinal_collator import (
    LongitudinalBatchCollator,
    PatientSequenceInput,
)
from aecopd_training.multimodal_collator import (
    FrozenTextEmbeddingStore,
    MultimodalBatchCollator,
)
from aecopd_training.patient_sequence_cache import PatientSequenceCache
from aecopd_training.patient_bootstrap import BootstrapConfig, patient_clustered_bootstrap


def _hadm_to_subject(cohort_path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    with gzip.open(cohort_path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            mapping[int(row["hadm_id"])] = int(row["subject_id"])
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--length-index", type=Path, required=True)
    parser.add_argument("--patient-cache", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, required=True)
    parser.add_argument("--text-index", type=Path, required=True)
    parser.add_argument("--text-state", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = AlignedModelConfig(**checkpoint["model_config"])
    training_config = checkpoint["training_config"]
    model = ConditionalIntervalAecopdModel(
        model_config,
        longitudinal_layers=int(training_config["longitudinal_layers"]),
        max_admissions=int(training_config["max_sequence_admissions"]),
        disable_longitudinal=bool(training_config.get("disable_longitudinal", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    artifacts = AecopdDataBundle.load_preprocessing(args.artifacts)
    outcomes = ConditionalOutcomeStore.from_cohort(args.cohort)
    text_store = FrozenTextEmbeddingStore(
        args.text_embeddings, args.text_index, state_path=args.text_state
    )
    collator = LongitudinalBatchCollator(
        MultimodalBatchCollator(
            SharedTimelineCollator(TimelineConfig(), artifacts=artifacts), text_store
        )
    )
    lengths = load_timeline_lengths(args.length_index, split=None)
    batcher = PatientSequenceBatcher(
        lengths,
        LongitudinalBatchingConfig(
            max_padded_tokens=int(training_config["max_padded_tokens"]),
            max_patients=int(training_config["max_patients"]),
            max_admissions=int(training_config["max_admissions"]),
            buffer_size=int(training_config["buffer_size"]),
        ),
        shuffle=False,
        seed=int(training_config["seed"]),
    )
    cache = PatientSequenceCache(args.patient_cache)
    use_bf16 = torch.cuda.is_bf16_supported()

    interval_probs: list[np.ndarray] = []
    interval_labels: list[np.ndarray] = []
    interval_masks: list[np.ndarray] = []
    tail_probs: list[np.ndarray] = []
    tail_labels: list[np.ndarray] = []
    tail_masks: list[np.ndarray] = []
    hadm_ids: list[int] = []

    sequences = cache.iter_sequences(args.split, shuffle=False)
    batches = batcher.iter_batches(sequences)
    with torch.inference_mode():
        for patient_sequences in batches:
            timeline_batch, tensors = collator(patient_sequences, device=device)
            tensors.update(outcomes.batch(timeline_batch.hadm_ids, device=device))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                output = model(tensors)
            interval_probs.append(output.interval_probabilities.float().cpu().numpy())
            interval_labels.append(tensors["conditional_interval_labels"].cpu().numpy())
            interval_masks.append(tensors["conditional_interval_mask"].cpu().numpy())
            tail_probs.append(output.tail_probabilities.float().cpu().numpy())
            tail_labels.append(tensors["conditional_tail_labels"].cpu().numpy())
            tail_masks.append(tensors["conditional_tail_mask"].cpu().numpy())
            hadm_ids.extend(int(h) for h in timeline_batch.hadm_ids)

    interval_probabilities = np.concatenate(interval_probs)
    il = np.concatenate(interval_labels)
    im = np.concatenate(interval_masks)
    tp = np.concatenate(tail_probs)
    tl = np.concatenate(tail_labels)
    tm = np.concatenate(tail_masks)
    hadm_array = np.array(hadm_ids, dtype=np.int64)

    mapping = _hadm_to_subject(args.cohort)
    subject_ids = np.array([mapping[int(h)] for h in hadm_array], dtype=np.int64)

    boot = patient_clustered_bootstrap(
        interval_probabilities,
        il,
        im,
        tp,
        tl,
        tm,
        subject_ids,
        prefix=args.split,
        config=BootstrapConfig(n_resamples=args.n_resamples, seed=args.seed),
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        f"{args.model_name}_{args.split}": {
            "metrics": boot.to_json(),
            "n_patients": int(np.unique(subject_ids).shape[0]),
            "n_admissions": int(hadm_array.shape[0]),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "architecture": checkpoint.get("architecture", "unknown"),
            "disable_longitudinal": bool(training_config.get("disable_longitudinal", False)),
            "note": "pure eval on frozen best checkpoint; test set not used for model selection",
        }
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    m = boot.to_json()["metrics"] if "metrics" in boot.to_json() else boot.to_json()
    def pt(k):
        v = m.get(k)
        return round(v["point"], 4) if isinstance(v, dict) else v
    print(f"[{args.model_name}] split={args.split} "
          f"n_patients={payload[f'{args.model_name}_{args.split}']['n_patients']} "
          f"n_admissions={hadm_array.shape[0]} epoch={checkpoint.get('epoch')}")
    for iv in ("days_0_30", "days_31_90", "days_91_365"):
        print(f"  AECOPD AUC {iv}: {pt(f'{args.split}_aecopd_auc_{iv}')}  "
              f"death AUC: {pt(f'{args.split}_death_auc_{iv}')}")
    print(f"  tail acc: {pt(f'{args.split}_tail_accuracy')}")
    print(f"[{args.model_name}] saved -> {args.output_json}")


if __name__ == "__main__":
    main()
