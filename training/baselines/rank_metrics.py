"""V5 与统计基线共用的秩指标；从旧队列实验入口独立出来，算法不变。"""

from __future__ import annotations

import numpy as np


def auc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    """带并列分数校正的 ROC-AUC；死亡按累积病例定义留作对照。"""

    positive = int(np.sum(labels))
    negative = labels.size - positive
    if positive < 10 or negative < 10:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    _, starts, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    positives = np.add.reduceat(sorted_labels, starts)
    negatives = counts - positives
    negatives_before = np.cumsum(negatives) - negatives
    return float(np.sum(positives * (negatives_before + 0.5 * negatives)) / (positive * negative))
