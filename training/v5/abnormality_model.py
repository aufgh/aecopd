"""住院内仅编码检验异常与复查恢复，不构造原始检验／用药事件编码器。"""

from __future__ import annotations

import torch
from torch import nn

from training.v5.event_data import EventBatch
from training.v5.event_model import TemporalFusionModel


class AbnormalityFusionModel(TemporalFusionModel):
    """保留静态、文本、跨住院历史与竞争风险头；只替换住院内事件分支。"""

    def __init__(self, *, concept_count: int, type_count: int, field_count: int,
                 text_width: int, abnormality_width: int, width: int = 128) -> None:
        if abnormality_width < 6 or abnormality_width % 6:
            raise ValueError("abnormality input must have six values per axis")
        super().__init__(concept_count=concept_count, type_count=type_count,
                         field_count=field_count, text_width=text_width, width=width)
        # 父类仅用于复用住院融合和跨住院历史；这些原始事件参数不能进入新模型。
        del self.event_embedding
        del self.event_value_projection
        del self.event_gru
        del self.event_score
        del self.event_projection
        self.abnormality_width = abnormality_width
        axis_count = abnormality_width // 6
        self.axis_embedding = nn.Embedding(axis_count, 24)
        self.axis_values = nn.Sequential(nn.Linear(6, 24), nn.GELU(), nn.LayerNorm(24))
        self.axis_score = nn.Linear(24, 1)
        self.abnormality_projection = nn.Sequential(nn.Linear(24, width), nn.GELU(),
                                                    nn.LayerNorm(width))

    def _events(self, batch: EventBatch) -> tuple[torch.Tensor, torch.Tensor]:
        values = batch.recovery_by_admission
        if values is None or values.shape != (*batch.event_counts.shape, self.abnormality_width):
            raise ValueError("abnormality feature shape mismatch")
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("invalid abnormality input")
        features = values.reshape(-1, self.abnormality_width // 6, 6)
        observed = features[:, :, 0] > 0
        present = observed.any(dim=1)
        output = values.new_zeros((features.shape[0], self.gap_projection.out_features))
        if not bool(present.any()):
            return output, present
        axis_ids = torch.arange(features.shape[1], device=features.device)
        embedded = self.axis_values(features) + self.axis_embedding(axis_ids)[None, :, :]
        scores = self.axis_score(embedded).squeeze(-1).masked_fill(~observed, -1e4)
        weights = torch.softmax(scores, dim=1) * observed
        pooled = (embedded * weights.unsqueeze(-1)).sum(dim=1)
        projected = self.abnormality_projection(pooled)
        output[present] = projected[present]
        return output, present
