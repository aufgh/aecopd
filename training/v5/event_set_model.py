"""STraTS 式事件集合编码；不是含自监督预训练的原论文复现。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from training.v5.event_data import EVENT_NAMES, LAB_NAMES, EventBatch
from training.v5.event_model import TemporalFusionModel


class EventSetFusionModel(TemporalFusionModel):
    """仅替换住院内编码器；融合、患者历史及竞争风险头沿用原模型。"""

    def __init__(self, *, concept_count: int, type_count: int, field_count: int,
                 text_width: int, width: int = 128, event_width: int = 64,
                 event_concept_count: int | None = None,
                 clinical_recovery_width: int = 0) -> None:
        # 先构造原模型，保证保留模块（包括 event_gate）的初始权重一致。
        super().__init__(concept_count=concept_count, type_count=type_count,
                         field_count=field_count, text_width=text_width, width=width)
        del self.event_gru
        del self.event_value_projection
        vocabulary_size = len(EVENT_NAMES) if event_concept_count is None else event_concept_count
        if vocabulary_size < 1:
            raise ValueError("empty event vocabulary")
        self.requires_explicit_type = event_concept_count is not None
        self.event_embedding = nn.Embedding(vocabulary_size + 1, event_width, padding_idx=0)
        self.numeric_embedding = nn.Sequential(nn.Linear(1, 8), nn.Tanh(),
                                               nn.Linear(8, event_width))
        self.time_embedding = nn.Sequential(nn.Linear(1, 8), nn.Tanh(),
                                            nn.Linear(8, event_width))
        self.event_type_embedding = nn.Embedding(2, event_width)
        self.event_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(event_width, 4, 2 * event_width,
                                       dropout=0.1, batch_first=True),
            num_layers=2, enable_nested_tensor=False,
        )
        self.event_score = nn.Linear(event_width, 1)
        self.event_projection = nn.Sequential(nn.Linear(event_width, width),
                                              nn.GELU(), nn.LayerNorm(width))
        self.clinical_recovery_width = clinical_recovery_width
        if clinical_recovery_width:
            if clinical_recovery_width % 6:
                raise ValueError("clinical recovery features must contain six values per axis")
            self.recovery_projection = nn.Sequential(nn.Linear(clinical_recovery_width, 32),
                                                     nn.GELU(), nn.Linear(32, width),
                                                     nn.LayerNorm(width))
            self.recovery_gate = nn.Linear(2 * width, 1)
            nn.init.constant_(self.recovery_gate.bias, -1.0)

    def _combine_recovery(self, batch: EventBatch, output: torch.Tensor,
                          present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.clinical_recovery_width:
            if batch.recovery_by_admission is not None:
                raise ValueError("recovery input supplied to model without recovery branch")
            return output, present
        features = batch.recovery_by_admission
        if features is None or features.shape != (*batch.event_counts.shape, self.clinical_recovery_width):
            raise ValueError("clinical recovery feature shape mismatch")
        features = features.reshape(-1, self.clinical_recovery_width)
        if not bool(torch.isfinite(features).all()):
            raise ValueError("nonfinite clinical recovery feature")
        observed = (features[:, ::6] > 0).any(dim=1)
        if not bool(observed.any()):
            return output, present
        projected = self.recovery_projection(features)
        gate = torch.sigmoid(self.recovery_gate(torch.cat((output, projected), dim=-1)))
        return output + gate * projected * observed.unsqueeze(-1), present | observed

    def _events(self, batch: EventBatch) -> tuple[torch.Tensor, torch.Tensor]:
        counts = batch.event_counts.flatten()
        present = counts > 0
        output = batch.event_values.new_zeros((counts.numel(), self.gap_projection.out_features))
        if int(counts.sum()) != batch.event_values.shape[0]:
            raise ValueError("event offsets do not match flattened values")
        if not bool(present.any()):
            return self._combine_recovery(batch, output, present)
        if self.requires_explicit_type and (batch.event_is_lab is None or batch.event_numeric_mask is None):
            raise ValueError("new vocabulary requires explicit event type and numeric mask")
        is_lab = batch.event_concepts <= len(LAB_NAMES) if batch.event_is_lab is None else batch.event_is_lab
        numeric_mask = is_lab if batch.event_numeric_mask is None else batch.event_numeric_mask
        if is_lab.shape != batch.event_concepts.shape or numeric_mask.shape != is_lab.shape:
            raise ValueError("event type/mask shape mismatch")
        if bool((numeric_mask & ~is_lab).any()):
            raise ValueError("medication cannot have a numeric observation")
        # 仅使用项目、检验值及真实相对时间；不读取排序相关 gap 和距出院时间。
        encoded = (self.event_embedding(batch.event_concepts)
                   + self.time_embedding(batch.event_values[:, 1:2])
                   + self.event_type_embedding(is_lab.long())
                   + self.numeric_embedding(batch.event_values[:, :1]) * numeric_mask.unsqueeze(-1))
        sequences = torch.split(encoded, counts.tolist())
        ordered = torch.argsort(counts, descending=True)
        ordered = ordered[counts[ordered] > 0]
        cursor = 0
        while cursor < len(ordered):
            longest = int(counts[ordered[cursor]])
            # 限制二次方注意力开销，不截断原始事件。
            allowance = max(1, min(64, 8192 // longest, 2_000_000 // (longest * longest)))
            positions = ordered[cursor:cursor + allowance]
            lengths = counts[positions]
            padded = pad_sequence([sequences[int(i)] for i in positions], batch_first=True)
            valid = torch.arange(padded.shape[1], device=padded.device)[None, :] < lengths[:, None]
            states = self.event_encoder(padded, src_key_padding_mask=~valid)
            score = self.event_score(states).squeeze(-1).masked_fill(~valid, -torch.inf)
            pooled = (states * torch.softmax(score, dim=1).unsqueeze(-1)).sum(dim=1)
            output = output.index_copy(0, positions, self.event_projection(pooled))
            cursor += len(positions)
        return self._combine_recovery(batch, output, present)
