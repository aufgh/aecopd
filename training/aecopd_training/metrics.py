from __future__ import annotations

import math
from typing import TypedDict

import numpy as np


class CalibrationBin(TypedDict):
    count: int
    mean_predicted: float
    observed_rate: float
    minimum_predicted: float
    maximum_predicted: float


def _validated_binary_inputs(
    outcomes: np.ndarray,
    risks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(outcomes, dtype=np.int64).reshape(-1)
    scores = np.asarray(risks, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or labels.size == 0:
        raise ValueError("outcomes and risks must be non-empty one-dimensional arrays of equal length")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("outcomes must contain only 0 and 1")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("risks must be finite probabilities between 0 and 1")
    return labels, scores


def binary_auc(outcomes: np.ndarray, risks: np.ndarray) -> float:
    """Compute ROC AUC from pairwise ranks, assigning half credit to ties."""
    labels, scores = _validated_binary_inputs(outcomes, risks)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return math.nan
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0)) / comparisons.size)


def horizon_metrics(risks: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    labels, scores = _validated_binary_inputs(outcomes, risks)
    return {
        "auc": binary_auc(labels, scores),
        "brier": float(np.mean(np.square(scores - labels))),
        "observed_rate": float(np.mean(labels)),
        "mean_predicted_risk": float(np.mean(scores)),
    }


def calibration_bins(
    risks: np.ndarray,
    outcomes: np.ndarray,
    *,
    bin_count: int = 10,
) -> list[CalibrationBin]:
    labels, scores = _validated_binary_inputs(outcomes, risks)
    if bin_count < 1:
        raise ValueError("bin_count must be positive")
    ordered = np.argsort(scores, kind="stable")
    groups = np.array_split(ordered, min(bin_count, scores.size))
    result: list[CalibrationBin] = []
    for group in groups:
        group_scores = scores[group]
        group_labels = labels[group]
        result.append(
            {
                "count": int(group.size),
                "mean_predicted": float(np.mean(group_scores)),
                "observed_rate": float(np.mean(group_labels)),
                "minimum_predicted": float(np.min(group_scores)),
                "maximum_predicted": float(np.max(group_scores)),
            }
        )
    return result
