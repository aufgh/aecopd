"""条件风险集和数据库开放尾部的区分度、误差与校准评价。"""

from __future__ import annotations

import math

import numpy as np

from .conditional_outcomes import INTERVAL_NAMES, OUTCOME_NAMES, TAIL_OUTCOME_NAMES
from .metrics import calibration_bins, horizon_metrics


def _validated_probabilities(probabilities: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != shape:
        raise ValueError(f"probabilities must have shape {shape}")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must be finite and between 0 and 1")
    if not np.allclose(values.sum(axis=-1), 1.0, atol=1e-5):
        raise ValueError("class probabilities must sum to one")
    return values


def conditional_metrics(
    interval_probabilities: np.ndarray,
    interval_labels: np.ndarray,
    interval_mask: np.ndarray,
    tail_probabilities: np.ndarray,
    tail_labels: np.ndarray,
    tail_mask: np.ndarray,
    *,
    prefix: str,
) -> tuple[dict[str, float], dict[str, list[dict[str, float | int]]]]:
    """在各自适用风险集中评价区间任务和数据库定义开放尾部。"""

    labels = np.asarray(interval_labels, dtype=np.int64)
    masks = np.asarray(interval_mask, dtype=bool)
    if labels.ndim != 2 or labels.shape[1] != 3 or masks.shape != labels.shape:
        raise ValueError("interval labels and masks must have shape [samples, 3]")
    sample_count = labels.shape[0]
    interval_probs = _validated_probabilities(
        interval_probabilities, (sample_count, 3, 3)
    )
    tail_y = np.asarray(tail_labels, dtype=np.int64).reshape(-1)
    tail_m = np.asarray(tail_mask, dtype=bool).reshape(-1)
    if tail_y.shape != (sample_count,) or tail_m.shape != (sample_count,):
        raise ValueError("tail labels and mask must have shape [samples]")
    tail_probs = _validated_probabilities(tail_probabilities, (sample_count, 3))
    if not np.isin(labels, (0, 1, 2)).all() or not np.isin(tail_y, (0, 1, 2)).all():
        raise ValueError("labels must contain only 0, 1, or 2")

    metrics: dict[str, float] = {}
    calibrations: dict[str, list[dict[str, float | int]]] = {}
    for interval_index, interval_name in enumerate(INTERVAL_NAMES):
        active = masks[:, interval_index]
        # 每个区间只使用仍在风险集的住院；不能把不同区间的 AUC 当成
        # 同一批样本上的普通多时间点二分类结果。
        metrics[f"{prefix}_count_{interval_name}"] = float(active.sum())
        for class_index, outcome_name in enumerate(OUTCOME_NAMES[1:], start=1):
            calibration_key = f"{prefix}_{outcome_name}_{interval_name}"
            if not active.any():
                for metric_name in ("auc", "brier", "observed_rate", "mean_predicted_risk"):
                    metrics[
                        f"{prefix}_{outcome_name}_{metric_name}_{interval_name}"
                    ] = math.nan
                continue
            binary_labels = (labels[active, interval_index] == class_index).astype(np.int64)
            risks = interval_probs[active, interval_index, class_index]
            values = horizon_metrics(risks, binary_labels)
            for metric_name, value in values.items():
                metrics[
                    f"{prefix}_{outcome_name}_{metric_name}_{interval_name}"
                ] = value
            calibrations[calibration_key] = calibration_bins(risks, binary_labels)

    active_tail = tail_m
    metrics[f"{prefix}_tail_count"] = float(active_tail.sum())
    if active_tail.any():
        active_labels = tail_y[active_tail]
        active_probabilities = tail_probs[active_tail]
        metrics[f"{prefix}_tail_accuracy"] = float(
            np.mean(np.argmax(active_probabilities, axis=1) == active_labels)
        )
        chosen = active_probabilities[np.arange(active_labels.size), active_labels]
        metrics[f"{prefix}_tail_log_loss"] = float(
            -np.mean(np.log(np.clip(chosen, 1e-12, 1.0)))
        )
        for class_index, outcome_name in enumerate(TAIL_OUTCOME_NAMES):
            calibration_key = f"{prefix}_tail_{outcome_name}"
            binary_labels = (active_labels == class_index).astype(np.int64)
            risks = active_probabilities[:, class_index]
            values = horizon_metrics(risks, binary_labels)
            for metric_name, value in values.items():
                metrics[f"{prefix}_tail_{outcome_name}_{metric_name}"] = value
            calibrations[calibration_key] = calibration_bins(risks, binary_labels)
    else:
        metrics[f"{prefix}_tail_accuracy"] = math.nan
        metrics[f"{prefix}_tail_log_loss"] = math.nan
    return metrics, calibrations
