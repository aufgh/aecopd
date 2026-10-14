"""训练患者纵向的直接条件区间 AECOPD 竞争风险模型。

本入口从零初始化当前输出头，按验证集 AECOPD 条件区间平均 Brier 选择
最佳检查点，并仅在训练结束后加载最佳检查点评价测试集。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from aecopd_loader import AecopdDataBundle
from timeline_collator import SharedTimelineCollator, TimelineConfig

from aecopd_training.aligned_model import AlignedModelConfig
from aecopd_training.batching import load_timeline_lengths
from aecopd_training.conditional_losses import (
    conditional_admission_equal_loss,
    conditional_patient_equal_loss,
)
from aecopd_training.conditional_metrics import conditional_metrics
from aecopd_training.conditional_model import ConditionalIntervalAecopdModel
from aecopd_training.conditional_outcomes import ConditionalOutcomeStore, INTERVAL_NAMES
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
    StructuredBatchCollator,
)
from aecopd_training.patient_sequence_cache import PatientSequenceCache
from baselines.legacy_training_utils import atomic_json_dump, atomic_torch_save, set_seed, write_history


@dataclass(frozen=True)
class ConditionalTrainingConfig:
    """正式条件区间训练的超参数与基本合法性约束。"""

    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    update_patients: int = 43
    patience: int = 12
    min_delta: float = 1e-4
    seed: int = 2026
    max_padded_tokens: int = 32768
    max_patients: int = 8
    max_admissions: int = 32
    buffer_size: int = 128
    hidden_size: int = 256
    attention_heads: int = 4
    temporal_layers: int = 2
    within_window_layers: int = 1
    text_layers: int = 1
    fusion_layers: int = 1
    patient_layers: int = 2
    longitudinal_layers: int = 1
    max_sequence_admissions: int = 128
    dropout: float = 0.1
    selection_metric: str = "aecopd_interval_mean_brier"
    # ADR 0008：主要分析住院等权；患者等权只保留为敏感性分析。
    loss_reduction: str = "admission_equal"
    # B3 消融（opt-in，默认 False = 主模型 M1 行为不变）：去掉患者纵向因果序列。
    disable_longitudinal: bool = False
    # 文档语义层消融：不加载文本向量且不构造文本编码参数。
    disable_text: bool = False

    def __post_init__(self) -> None:
        positive = (
            self.epochs,
            self.update_patients,
            self.patience,
            self.max_padded_tokens,
            self.max_patients,
            self.max_admissions,
            self.buffer_size,
            self.max_sequence_admissions,
        )
        if any(value < 1 for value in positive):
            raise ValueError("epoch, batching, and sequence limits must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.selection_metric not in {
            "validation_loss",
            "aecopd_interval_mean_brier",
            "aecopd_death_interval_mean_brier",
            "all_head_mean_brier",
        }:
            raise ValueError("unsupported selection_metric")
        if self.loss_reduction not in {"admission_equal", "patient_equal"}:
            raise ValueError("loss_reduction must be admission_equal or patient_equal")


def grouped_patient_microbatches(
    batches: Iterable[list[PatientSequenceInput]],
    target_patients: int,
) -> Iterator[list[list[PatientSequenceInput]]]:
    """累积完整患者微批次，直到达到一次优化步骤的目标患者数。"""
    group: list[list[PatientSequenceInput]] = []
    count = 0
    for batch in batches:
        group.append(batch)
        count += len(batch)
        if count >= target_patients:
            yield group
            group = []
            count = 0
    if group:
        yield group


def _loss(
    output: Any,
    tensors: dict[str, torch.Tensor],
    loss_reduction: str,
) -> torch.Tensor:
    """按显式统计合同计算住院等权主损失或患者等权敏感性损失。"""
    arguments = (
        output.interval_logits,
        output.tail_logits,
        tensors["conditional_interval_labels"],
        tensors["conditional_interval_mask"],
        tensors["conditional_tail_labels"],
        tensors["conditional_tail_mask"],
    )
    if loss_reduction == "admission_equal":
        return conditional_admission_equal_loss(*arguments)
    if loss_reduction == "patient_equal":
        return conditional_patient_equal_loss(
            *arguments,
            tensors["longitudinal_patient_indices"],
        )
    raise ValueError(f"unsupported loss_reduction: {loss_reduction}")


def _mean_finite(values: Iterable[float]) -> float:
    """对有限值求均值；全为 NaN（空风险集）时返回 +inf，使其永不被当作最佳。"""
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.inf
    return sum(finite) / len(finite)


def composite_selection_scores(validation: dict[str, float]) -> dict[str, float]:
    """从验证指标派生候选检查点选择分数（H1：死亡与尾部不再无保护）。

    - aecopd_interval_mean_brier：仅 3 个 AECOPD 条件区间 Brier 的均值（历史默认，保持可比）。
    - aecopd_death_interval_mean_brier：6 个区间 Brier（AECOPD×3 + 死亡×3）均值，
      对系统性高估死亡风险施加选择压力。
    - all_head_mean_brier：9 个头 Brier（6 区间 + 3 尾部类别）均值，兼顾开放尾部。
    所有均值只对有限值求（空风险集产生 NaN 时跳过），不改变训练损失，只影响选最佳 epoch。
    """
    aecopd_interval = [
        validation.get(f"validation_aecopd_brier_{interval_name}", math.nan)
        for interval_name in INTERVAL_NAMES
    ]
    death_interval = [
        validation.get(f"validation_death_brier_{interval_name}", math.nan)
        for interval_name in INTERVAL_NAMES
    ]
    tail_brier = [
        validation.get(f"validation_tail_{outcome_name}_brier", math.nan)
        for outcome_name in ("none_observed", "aecopd", "death")
    ]
    return {
        "aecopd_interval_mean_brier": _mean_finite(aecopd_interval),
        "aecopd_death_interval_mean_brier": _mean_finite(aecopd_interval + death_interval),
        "all_head_mean_brier": _mean_finite(aecopd_interval + death_interval + tail_brier),
    }


def evaluate_patient_sequences(
    model: nn.Module,
    cache: PatientSequenceCache,
    outcomes: ConditionalOutcomeStore,
    split: str,
    batcher: PatientSequenceBatcher,
    collator: LongitudinalBatchCollator,
    device: torch.device,
    *,
    use_bf16: bool,
    loss_reduction: str = "admission_equal",
    max_batches: int | None = None,
) -> tuple[dict[str, float], dict[str, list[dict[str, float | int]]]]:
    """确定性评价一个患者划分，并返回指标与校准分箱。"""
    model.eval()
    weighted_loss_sum = 0.0
    loss_weight_count = 0
    patient_count = 0
    admission_count = 0
    interval_probabilities: list[np.ndarray] = []
    interval_labels: list[np.ndarray] = []
    interval_masks: list[np.ndarray] = []
    tail_probabilities: list[np.ndarray] = []
    tail_labels: list[np.ndarray] = []
    tail_masks: list[np.ndarray] = []
    sequences = cache.iter_sequences(split, shuffle=False)
    batches: Iterable[list[PatientSequenceInput]] = batcher.iter_batches(sequences)
    if max_batches is not None:
        batches = itertools.islice(batches, max_batches)

    with torch.inference_mode():
        for patient_sequences in batches:
            timeline_batch, tensors = collator(patient_sequences, device=device)
            tensors.update(outcomes.batch(timeline_batch.hadm_ids, device=device))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                output = model(tensors)
                loss = _loss(output, tensors, loss_reduction)
            current_patients = len(patient_sequences)
            current_admissions = len(timeline_batch.hadm_ids)
            current_weight = (
                current_admissions if loss_reduction == "admission_equal" else current_patients
            )
            weighted_loss_sum += float(loss.item()) * current_weight
            loss_weight_count += current_weight
            patient_count += current_patients
            admission_count += current_admissions
            interval_probabilities.append(output.interval_probabilities.float().cpu().numpy())
            interval_labels.append(tensors["conditional_interval_labels"].cpu().numpy())
            interval_masks.append(tensors["conditional_interval_mask"].cpu().numpy())
            tail_probabilities.append(output.tail_probabilities.float().cpu().numpy())
            tail_labels.append(tensors["conditional_tail_labels"].cpu().numpy())
            tail_masks.append(tensors["conditional_tail_mask"].cpu().numpy())
    if patient_count == 0:
        raise RuntimeError(f"{split} produced no patients")

    metrics, calibrations = conditional_metrics(
        np.concatenate(interval_probabilities),
        np.concatenate(interval_labels),
        np.concatenate(interval_masks),
        np.concatenate(tail_probabilities),
        np.concatenate(tail_labels),
        np.concatenate(tail_masks),
        prefix=split,
    )
    metrics[f"{split}_loss"] = weighted_loss_sum / loss_weight_count
    metrics[f"{split}_patients"] = float(patient_count)
    metrics[f"{split}_admissions"] = float(admission_count)
    return metrics, calibrations


def parse_args() -> argparse.Namespace:
    """解析数据、缓存、输出路径及可覆盖训练超参数。"""
    parser = argparse.ArgumentParser(
        description="Train direct conditional-interval AECOPD competing-risk model"
    )
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--length-index", type=Path, required=True)
    parser.add_argument("--patient-cache", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path)
    parser.add_argument("--text-index", type=Path)
    parser.add_argument("--text-state", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--update-patients", type=int, default=43)
    parser.add_argument("--max-padded-tokens", type=int, default=32768)
    parser.add_argument("--max-patients", type=int, default=8)
    parser.add_argument("--max-admissions", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--within-window-layers", type=int, default=1)
    parser.add_argument("--text-layers", type=int, default=1)
    parser.add_argument("--fusion-layers", type=int, default=1)
    parser.add_argument("--patient-layers", type=int, default=2)
    parser.add_argument("--longitudinal-layers", type=int, default=1)
    parser.add_argument("--max-sequence-admissions", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--loss-reduction",
        choices=("admission_equal", "patient_equal"),
        default="admission_equal",
        help="训练估计目标。主要分析按 ADR 0008 使用住院等权；患者等权仅作敏感性分析。",
    )
    parser.add_argument(
        "--selection-metric",
        choices=(
            "validation_loss",
            "aecopd_interval_mean_brier",
            "aecopd_death_interval_mean_brier",
            "all_head_mean_brier",
        ),
        default="aecopd_interval_mean_brier",
        help="检查点选择指标。默认沿用历史单指标（仅 AECOPD 区间 Brier，保持与既有运行可比）；"
             "aecopd_death_interval_mean_brier 与 all_head_mean_brier 为 H1 升级，"
             "对死亡/尾部施加选择压力。",
    )
    parser.add_argument(
        "--disable-longitudinal",
        action="store_true",
        help="B3 消融：去掉患者纵向因果 Transformer，每次索引住院独立编码。"
             "除此之外所有超参须与主模型 M1 一致，以隔离唯一变量。默认关闭（=M1）。",
    )
    parser.add_argument(
        "--disable-text",
        action="store_true",
        help="文本消融：不读取 Qwen 向量、文档角色或章节元数据，"
             "仅使用同一结构化时间轴、纵向编码器和结局头。",
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--max-test-batches", type=int)
    return parser.parse_args()


def main() -> None:
    """执行训练、验证选择、提前停止和一次最佳检查点测试评价。"""
    args = parse_args()
    config = ConditionalTrainingConfig(
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        update_patients=args.update_patients,
        max_padded_tokens=args.max_padded_tokens,
        max_patients=args.max_patients,
        max_admissions=args.max_admissions,
        buffer_size=args.buffer_size,
        hidden_size=args.hidden_size,
        attention_heads=args.attention_heads,
        temporal_layers=args.temporal_layers,
        within_window_layers=args.within_window_layers,
        text_layers=args.text_layers,
        fusion_layers=args.fusion_layers,
        patient_layers=args.patient_layers,
        longitudinal_layers=args.longitudinal_layers,
        max_sequence_admissions=args.max_sequence_admissions,
        dropout=args.dropout,
        seed=args.seed,
        selection_metric=args.selection_metric,
        loss_reduction=args.loss_reduction,
        disable_longitudinal=args.disable_longitudinal,
        disable_text=args.disable_text,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for longitudinal training")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    set_seed(config.seed)
    for name in ("checkpoints", "metrics", "tensorboard"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    artifacts = AecopdDataBundle.load_preprocessing(args.artifacts)
    outcomes = ConditionalOutcomeStore.from_cohort(args.cohort)
    timeline_collator = SharedTimelineCollator(TimelineConfig(), artifacts=artifacts)
    text_embedding_dim = 0
    if config.disable_text:
        base_collator = StructuredBatchCollator(timeline_collator)
    else:
        text_paths = (args.text_embeddings, args.text_index, args.text_state)
        if any(path is None for path in text_paths):
            raise ValueError(
                "--text-embeddings, --text-index and --text-state are required "
                "unless --disable-text is set"
            )
        assert args.text_embeddings is not None
        assert args.text_index is not None
        assert args.text_state is not None
        text_store = FrozenTextEmbeddingStore(
            args.text_embeddings,
            args.text_index,
            state_path=args.text_state,
        )
        text_embedding_dim = text_store.embedding_dim
        base_collator = MultimodalBatchCollator(timeline_collator, text_store)
    collator = LongitudinalBatchCollator(base_collator)
    lengths = load_timeline_lengths(args.length_index, split=None)
    batching_config = LongitudinalBatchingConfig(
        max_padded_tokens=config.max_padded_tokens,
        max_patients=config.max_patients,
        max_admissions=config.max_admissions,
        buffer_size=config.buffer_size,
    )
    train_batcher = PatientSequenceBatcher(lengths, batching_config, shuffle=True, seed=config.seed)
    validation_batcher = PatientSequenceBatcher(lengths, batching_config, shuffle=False, seed=config.seed)
    test_batcher = PatientSequenceBatcher(lengths, batching_config, shuffle=False, seed=config.seed)
    model_config = AlignedModelConfig.from_artifacts(
        args.artifacts,
        hidden_size=config.hidden_size,
        attention_heads=config.attention_heads,
        within_window_layers=config.within_window_layers,
        text_layers=config.text_layers,
        fusion_layers=config.fusion_layers,
        temporal_layers=config.temporal_layers,
        patient_layers=config.patient_layers,
        dropout=config.dropout,
        use_text=not config.disable_text,
        text_embedding_dim=text_embedding_dim,
        cause_specific_heads=False,
    )
    model = ConditionalIntervalAecopdModel(
        model_config,
        longitudinal_layers=config.longitudinal_layers,
        max_admissions=config.max_sequence_admissions,
        disable_longitudinal=config.disable_longitudinal,
    ).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    use_bf16 = torch.cuda.is_bf16_supported()

    history: list[dict[str, float]] = []
    best_score = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()
    cache = PatientSequenceCache(args.patient_cache)
    run_state: dict[str, Any] = {
        "started_at": datetime.now().astimezone().isoformat(),
        "status": "running",
        "architecture": (
            "structured_only_patient_longitudinal_conditional_intervals_v1"
            if config.disable_text
            else "b3_no_longitudinal_ablation_v1"
            if config.disable_longitudinal
            else "patient_longitudinal_conditional_intervals_v1"
        ),
        "config": asdict(config),
        "model_config": asdict(model_config),
        "cache_summary": cache.summary,
        "outcome_admissions": len(outcomes),
        "device": torch.cuda.get_device_name(0),
        "precision": "bf16" if use_bf16 else "fp32",
        "legacy_daily_head_used": False,
        "checkpoint_initialization": "from_scratch",
        "text_branch_enabled": not config.disable_text,
        "test_set_used": False,
        "scientific_scope": (
            "direct conditional competing risks for 0-30, 31-90, 91-365 days; "
            "database-defined open tail after day 365"
        ),
    }
    atomic_json_dump(run_state, args.output_dir / "run_state.json")
    writer = SummaryWriter(log_dir=args.output_dir / "tensorboard")

    try:
        for epoch in range(1, config.epochs + 1):
            epoch_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            model.train()
            train_loss_sum = 0.0
            train_loss_weight_count = 0
            train_patients = 0
            train_admissions = 0
            optimizer_steps = 0
            gradient_norm_sum = 0.0
            sequences = cache.iter_sequences("train", shuffle=True, seed=config.seed, epoch=epoch)
            batches: Iterable[list[PatientSequenceInput]] = train_batcher.iter_batches(
                sequences, epoch=epoch
            )
            if args.max_train_batches is not None:
                batches = itertools.islice(batches, args.max_train_batches)
            for group in grouped_patient_microbatches(batches, config.update_patients):
                optimizer.zero_grad(set_to_none=True)
                group_patients = sum(len(batch) for batch in group)
                group_admissions = sum(
                    len(sequence.admissions)
                    for batch in group
                    for sequence in batch
                )
                for patient_sequences in group:
                    timeline_batch, tensors = collator(patient_sequences, device=device)
                    tensors.update(outcomes.batch(timeline_batch.hadm_ids, device=device))
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
                    ):
                        output = model(tensors)
                        loss = _loss(output, tensors, config.loss_reduction)
                        current_admissions = len(timeline_batch.hadm_ids)
                        if config.loss_reduction == "admission_equal":
                            current_weight = current_admissions
                            group_weight = group_admissions
                        else:
                            current_weight = len(patient_sequences)
                            group_weight = group_patients
                        # 微批大小因 token 预算而变化。按当前估计目标的单位数
                        # 重标定后求和，等价于整个梯度累积组的一次加权平均。
                        weighted_loss = loss * (current_weight / group_weight)
                    weighted_loss.backward()
                    train_loss_sum += float(loss.item()) * current_weight
                    train_loss_weight_count += current_weight
                    train_patients += len(patient_sequences)
                    train_admissions += len(timeline_batch.hadm_ids)
                gradient_norm = nn.utils.clip_grad_norm_(
                    trainable_parameters, config.gradient_clip_norm
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"non-finite gradient norm at epoch {epoch}")
                gradient_norm_sum += float(gradient_norm.item())
                optimizer.step()
                optimizer_steps += 1
            if train_patients == 0:
                raise RuntimeError("training produced no patients")

            validation, calibrations = evaluate_patient_sequences(
                model,
                cache,
                outcomes,
                "validation",
                validation_batcher,
                collator,
                device,
                use_bf16=use_bf16,
                loss_reduction=config.loss_reduction,
                max_batches=args.max_validation_batches,
            )
            row: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": train_loss_sum / train_loss_weight_count,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "gradient_norm": gradient_norm_sum / max(1, optimizer_steps),
                "epoch_seconds": time.perf_counter() - epoch_started,
                "train_patients": float(train_patients),
                "train_admissions": float(train_admissions),
                "optimizer_steps": float(optimizer_steps),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
                "oversize_patient_batches": float(train_batcher.last_oversize_singletons),
            }
            row.update(validation)
            # 三个候选检查点选择分数每轮都算并记录（便于结果解释与事后复核）：
            #   aecopd_interval_mean_brier —— 历史默认，仅覆盖 3 个 AECOPD 区间；
            #   aecopd_death_interval_mean_brier / all_head_mean_brier —— H1 升级，
            #   把死亡与开放尾部纳入选择压力，避免"偏科"检查点当选。
            # 无论选哪个，死亡与尾部始终参与训练损失；此处只决定最佳 epoch 的判据。
            row.update(composite_selection_scores(validation))
            selection_score = row[config.selection_metric]
            row["selection_score"] = selection_score
            history.append(row)
            scheduler.step()

            improved = selection_score < best_score - config.min_delta
            if improved:
                best_score = selection_score
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "training_config": asdict(config),
                "model_config": asdict(model_config),
                "architecture": run_state["architecture"],
                "legacy_daily_head_used": False,
                "selection_metric": config.selection_metric,
            }
            atomic_torch_save(checkpoint, args.output_dir / "checkpoints" / "last.pt")
            if improved:
                atomic_torch_save(checkpoint, args.output_dir / "checkpoints" / "best.pt")
            write_history(history, args.output_dir / "metrics" / "results.csv")
            atomic_json_dump(
                {
                    "latest": row,
                    "best_epoch": best_epoch,
                    "selection_metric": config.selection_metric,
                    "best_selection_score": best_score,
                    "calibration": calibrations,
                },
                args.output_dir / "metrics" / "latest.json",
            )
            for name, value in row.items():
                if name != "epoch" and math.isfinite(value):
                    writer.add_scalar(name.replace("_", "/", 1), value, epoch)
            writer.flush()
            progress = {
                "epoch": epoch,
                "epochs": config.epochs,
                "train_loss": row["train_loss"],
                "validation_loss": row["validation_loss"],
                "aecopd_interval_mean_brier": row["aecopd_interval_mean_brier"],
                "aecopd_death_interval_mean_brier": row["aecopd_death_interval_mean_brier"],
                "all_head_mean_brier": row["all_head_mean_brier"],
                "active_selection_metric": config.selection_metric,
                "epoch_seconds": row["epoch_seconds"],
                "train_patients": train_patients,
                "train_admissions": train_admissions,
            }
            print(json.dumps(progress, ensure_ascii=False), flush=True)
            run_state.update(
                {
                    "completed_epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_selection_score": best_score,
                    "latest": progress,
                }
            )
            atomic_json_dump(run_state, args.output_dir / "run_state.json")
            if epochs_without_improvement >= config.patience:
                run_state["stop_reason"] = "early_stopping"
                break
        else:
            run_state["stop_reason"] = "max_epochs"

        # 测试集只在训练/提前停止完成后评价，并明确重新加载最佳检查点，
        # 避免把最后一轮（本次为 epoch 24）误当成最佳轮（epoch 12）。
        best_checkpoint = torch.load(
            args.output_dir / "checkpoints" / "best.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(best_checkpoint["model"])
        test_metrics, test_calibrations = evaluate_patient_sequences(
            model,
            cache,
            outcomes,
            "test",
            test_batcher,
            collator,
            device,
            use_bf16=use_bf16,
            loss_reduction=config.loss_reduction,
            max_batches=args.max_test_batches,
        )
        atomic_json_dump(
            {
                "best_epoch": best_epoch,
                "metrics": test_metrics,
                "calibration": test_calibrations,
                "scope": run_state["scientific_scope"],
            },
            args.output_dir / "metrics" / "test.json",
        )
        run_state.update(
            {"status": "complete", "test_set_used": True, "test_metrics": test_metrics}
        )
    except BaseException as exc:
        run_state["status"] = "failed"
        run_state["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        cache.close()
        writer.close()
        run_state["finished_at"] = datetime.now().astimezone().isoformat()
        run_state["elapsed_seconds"] = time.perf_counter() - started
        atomic_json_dump(run_state, args.output_dir / "run_state.json")


if __name__ == "__main__":
    main()
