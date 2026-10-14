"""条件区间模型的住院等权主损失与患者等权敏感性损失。"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _conditional_admission_losses(
    interval_logits: Tensor,
    tail_logits: Tensor,
    interval_labels: Tensor,
    interval_mask: Tensor,
    tail_labels: Tensor,
    tail_mask: Tensor,
) -> Tensor:
    """返回每次住院在全部适用条件任务上的平均交叉熵。

    三个区间属于递减条件风险集：事件一旦发生，后续区间 mask 为假；
    尾部任务只适用于第 365 天仍无事件的住院。
    """

    if interval_logits.ndim != 3 or interval_logits.shape[1:] != (3, 3):
        raise ValueError("interval_logits must have shape [batch, 3, 3]")
    batch_size = interval_logits.shape[0]
    if tail_logits.shape != (batch_size, 3):
        raise ValueError("tail_logits must have shape [batch, 3]")
    if interval_labels.shape != (batch_size, 3):
        raise ValueError("interval_labels must have shape [batch, 3]")
    if interval_mask.shape != (batch_size, 3) or interval_mask.dtype != torch.bool:
        raise ValueError("interval_mask must be boolean with shape [batch, 3]")
    if tail_labels.shape != (batch_size,):
        raise ValueError("tail_labels must have shape [batch]")
    if tail_mask.shape != (batch_size,) or tail_mask.dtype != torch.bool:
        raise ValueError("tail_mask must be boolean with shape [batch]")
    if torch.any((interval_labels < 0) | (interval_labels > 2)):
        raise ValueError("interval_labels must contain only 0, 1, or 2")
    if torch.any((tail_labels < 0) | (tail_labels > 2)):
        raise ValueError("tail_labels must contain only 0, 1, or 2")

    interval_losses = F.cross_entropy(
        interval_logits.reshape(-1, 3),
        interval_labels.reshape(-1),
        reduction="none",
    ).reshape(batch_size, 3)
    tail_losses = F.cross_entropy(tail_logits, tail_labels, reduction="none")
    # 每次住院可能适用 1～4 个任务。先除以适用任务数，避免随访到尾部的
    # 住院只因任务更多而自动获得更大的住院级损失权重。
    interval_weights = interval_mask.to(interval_losses.dtype)
    tail_weights = tail_mask.to(tail_losses.dtype)
    task_counts = interval_weights.sum(dim=1) + tail_weights
    if torch.any(task_counts == 0):
        raise ValueError("every admission must have at least one applicable task")
    return (
        (interval_losses * interval_weights).sum(dim=1)
        + tail_losses * tail_weights
    ) / task_counts


def conditional_admission_equal_loss(
    interval_logits: Tensor,
    tail_logits: Tensor,
    interval_labels: Tensor,
    interval_mask: Tensor,
    tail_labels: Tensor,
    tail_mask: Tensor,
) -> Tensor:
    """先平均住院内适用任务，再让每次索引住院获得相同总权重。"""

    admission_losses = _conditional_admission_losses(
        interval_logits,
        tail_logits,
        interval_labels,
        interval_mask,
        tail_labels,
        tail_mask,
    )
    return admission_losses.mean()


def conditional_patient_equal_loss(
    interval_logits: Tensor,
    tail_logits: Tensor,
    interval_labels: Tensor,
    interval_mask: Tensor,
    tail_labels: Tensor,
    tail_mask: Tensor,
    patient_ids: Tensor,
) -> Tensor:
    """患者等权敏感性分析：患者内平均住院，再平均患者。"""

    admission_losses = _conditional_admission_losses(
        interval_logits,
        tail_logits,
        interval_labels,
        interval_mask,
        tail_labels,
        tail_mask,
    )
    if patient_ids.shape != admission_losses.shape:
        raise ValueError("patient_ids must have shape [batch]")

    _, inverse, counts = torch.unique(
        patient_ids,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    patient_sums = admission_losses.new_zeros((counts.shape[0],))
    patient_sums.scatter_add_(0, inverse, admission_losses)
    # 敏感性分析按患者内平均后再患者间平均：65 次住院患者不会获得 65 倍总权重。
    return (patient_sums / counts.to(admission_losses.dtype)).mean()
