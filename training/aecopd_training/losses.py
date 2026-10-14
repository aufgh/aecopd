from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


def discrete_competing_risk_nll(
    logits: Tensor,
    event_types: Tensor,
    time_days: Tensor,
    *,
    reduction: Literal["mean", "none"] = "mean",
) -> Tensor:
    """Negative log likelihood for daily mutually exclusive competing risks.

    Class 0 means remaining event-free in an interval, class 1 means AECOPD
    readmission, and class 2 means all-cause death. Event times are assigned to
    ``ceil(time_days) - 1``; censoring contributes event-free likelihood through
    every fully or partially observed interval.
    """
    if logits.ndim != 3 or logits.shape[-1] != 3:
        raise ValueError("logits must have shape [batch, days, 3]")
    batch_size, horizon_days, _ = logits.shape
    if event_types.shape != (batch_size,) or time_days.shape != (batch_size,):
        raise ValueError("event_types and time_days must have shape [batch]")
    if torch.any((event_types < 0) | (event_types > 2)):
        raise ValueError("event_types must contain only 0, 1, or 2")
    if torch.any(~torch.isfinite(time_days)) or torch.any(time_days < 0):
        raise ValueError("time_days must be finite and non-negative")

    log_probabilities = logits.log_softmax(dim=-1)
    observed_intervals = torch.ceil(time_days).to(torch.long).clamp(0, horizon_days)
    event_interval = (observed_intervals - 1).clamp(0, horizon_days - 1)
    day_index = torch.arange(horizon_days, device=logits.device).unsqueeze(0)

    event_observed = event_types > 0
    survival_intervals = torch.where(
        event_observed,
        event_interval,
        observed_intervals,
    )
    survival_mask = day_index < survival_intervals.unsqueeze(1)
    log_likelihood = (
        log_probabilities[..., 0] * survival_mask.to(logits.dtype)
    ).sum(dim=1)

    if event_observed.any():
        batch_index = torch.arange(batch_size, device=logits.device)
        event_log_probability = log_probabilities[
            batch_index,
            event_interval,
            event_types,
        ]
        log_likelihood = log_likelihood + torch.where(
            event_observed,
            event_log_probability,
            torch.zeros_like(event_log_probability),
        )

    losses = -log_likelihood
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction must be 'mean' or 'none'")


def patient_equal_competing_risk_nll(
    logits: Tensor,
    event_types: Tensor,
    time_days: Tensor,
    patient_ids: Tensor,
) -> Tensor:
    """Average admission losses within patient, then average patients equally."""

    losses = discrete_competing_risk_nll(
        logits,
        event_types,
        time_days,
        reduction="none",
    )
    if patient_ids.shape != losses.shape:
        raise ValueError("patient_ids must have shape [batch]")
    _, inverse, counts = torch.unique(
        patient_ids,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    patient_sums = losses.new_zeros((counts.shape[0],))
    patient_sums.scatter_add_(0, inverse, losses)
    return (patient_sums / counts.to(losses.dtype)).mean()
