"""Adapter from patient admission sequences to the flattened admission collator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch


class AdmissionCollator(Protocol):
    """住院级 collator 的最小接口，便于纵向适配器解耦具体实现。"""

    def __call__(
        self,
        samples: Sequence[Any],
        *,
        device: str | torch.device | None = None,
    ) -> tuple[Any, dict[str, torch.Tensor]]: ...


@dataclass(frozen=True)
class PatientSequenceInput:
    """One complete, causally ordered patient sequence for a batch."""

    admissions: tuple[Any, ...]
    gap_days: tuple[float, ...]


class LongitudinalBatchCollator:
    """Flatten admissions once, then attach their padded patient-sequence map."""

    def __init__(self, admission_collator: AdmissionCollator) -> None:
        self.admission_collator = admission_collator

    def __call__(
        self,
        sequences: Sequence[PatientSequenceInput],
        *,
        device: str | torch.device | None = None,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        if not sequences:
            raise ValueError("at least one patient sequence is required")
        flattened: list[Any] = []
        patient_indices: list[int] = []
        position_indices: list[int] = []
        gaps: list[float] = []
        lengths: list[int] = []
        for patient_index, sequence in enumerate(sequences):
            if not sequence.admissions:
                raise ValueError("patient sequences cannot be empty")
            if len(sequence.admissions) != len(sequence.gap_days):
                raise ValueError("gap_days length must match admissions length")
            subject_ids = {int(admission.subject_id) for admission in sequence.admissions}
            if len(subject_ids) != 1:
                raise ValueError("a patient sequence contains multiple subject_id values")
            if float(sequence.gap_days[0]) != 0.0:
                raise ValueError("the first admission gap must be zero")
            if any(not math.isfinite(float(gap)) or float(gap) < 0 for gap in sequence.gap_days):
                raise ValueError("admission gaps must be finite and non-negative")
            lengths.append(len(sequence.admissions))
            for position, (admission, gap) in enumerate(
                zip(sequence.admissions, sequence.gap_days, strict=True)
            ):
                flattened.append(admission)
                patient_indices.append(patient_index)
                position_indices.append(position)
                gaps.append(float(gap))

        base_batch, tensors = self.admission_collator(flattened, device=device)
        # 展平只是计算优化；以下四个张量保存可逆映射，模型据此恢复患者
        # 序列。任何映射不一致都会在 longitudinal_model 中被拒绝。
        width = max(lengths)
        attention = torch.zeros(
            (len(sequences), width),
            dtype=torch.bool,
            device=device,
        )
        for patient_index, length in enumerate(lengths):
            attention[patient_index, :length] = True
        tensors.update(
            {
                "longitudinal_patient_indices": torch.as_tensor(
                    patient_indices, dtype=torch.long, device=device
                ),
                "longitudinal_position_indices": torch.as_tensor(
                    position_indices, dtype=torch.long, device=device
                ),
                "longitudinal_gap_days": torch.as_tensor(
                    gaps, dtype=torch.float32, device=device
                ),
                "longitudinal_attention_mask": attention,
            }
        )
        return base_batch, tensors
