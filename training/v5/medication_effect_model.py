"""在检验异常分支旁加入药理作用组暴露，不恢复原始给药事件时序。"""

from __future__ import annotations

import torch
from torch import nn

from training.v5.abnormality_model import AbnormalityFusionModel
from training.v5.event_data import EventBatch
from training.v5.medication_effect_data import EFFECT_GROUPS, VALUES_PER_GROUP


class MedicationEffectFusionModel(AbnormalityFusionModel):
    def __init__(self, *, concept_count: int, type_count: int, field_count: int,
                 text_width: int, abnormality_width: int, width: int = 128) -> None:
        super().__init__(concept_count=concept_count, type_count=type_count,
                         field_count=field_count, text_width=text_width,
                         abnormality_width=abnormality_width, width=width)
        self.medication_axis_embedding = nn.Embedding(len(EFFECT_GROUPS), 24)
        self.medication_values = nn.Sequential(nn.Linear(VALUES_PER_GROUP, 24), nn.GELU(),
                                               nn.LayerNorm(24))
        self.medication_score = nn.Linear(24, 1)
        self.medication_projection = nn.Sequential(nn.Linear(24, width), nn.GELU(),
                                                   nn.LayerNorm(width))
        self.medication_gate = nn.Linear(2 * width, 1)
        nn.init.constant_(self.medication_gate.bias, -1.0)

    def _events(self, batch: EventBatch) -> tuple[torch.Tensor, torch.Tensor]:
        laboratory, lab_present = super()._events(batch)
        values = batch.medication_by_admission
        expected = (*batch.event_counts.shape, len(EFFECT_GROUPS) * VALUES_PER_GROUP)
        if values is None or values.shape != expected or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("invalid medication-effect input")
        features = values.reshape(-1, len(EFFECT_GROUPS), VALUES_PER_GROUP)
        observed = features[:, :, 0] > 0
        med_present = observed.any(dim=1)
        medication = laboratory.new_zeros(laboratory.shape)
        if bool(med_present.any()):
            axis_ids = torch.arange(len(EFFECT_GROUPS), device=features.device)
            embedded = self.medication_values(features) + self.medication_axis_embedding(axis_ids)[None, :, :]
            scores = self.medication_score(embedded).squeeze(-1).masked_fill(~observed, -1e4)
            weights = torch.softmax(scores, dim=1) * observed
            pooled = (embedded * weights.unsqueeze(-1)).sum(dim=1)
            projected = self.medication_projection(pooled)
            medication[med_present] = projected[med_present]
        gate = torch.sigmoid(self.medication_gate(torch.cat((laboratory, medication), dim=-1)))
        gate = gate * med_present.unsqueeze(-1)
        return laboratory + gate * medication, lab_present | med_present
