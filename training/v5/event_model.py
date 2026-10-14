"""V5 原始事件流分支：每次住院先编码不规则事件，再按时间编码患者住院史。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from training.v5.event_data import EVENT_NAMES, EventBatch
from training.v5.independent_fusion import IndependentFusionModel
from training.v5.model import V5Output


class TemporalFusionModel(IndependentFusionModel):
    """静态、原始事件、字段文本各自编码；住院级融合后作因果历史建模。"""

    def __init__(self, *, concept_count: int, type_count: int, field_count: int,
                 text_width: int, width: int = 128, event_width: int = 48) -> None:
        super().__init__(concept_count=concept_count, type_count=type_count,
                         field_count=field_count, text_width=text_width, width=width)
        self.event_embedding = nn.Embedding(len(EVENT_NAMES) + 1, event_width, padding_idx=0)
        self.event_value_projection = nn.Sequential(
            nn.Linear(4, event_width), nn.GELU(), nn.LayerNorm(event_width)
        )
        self.event_gru = nn.GRU(event_width, event_width, batch_first=True)
        self.event_score = nn.Linear(event_width, 1)
        self.event_projection = nn.Sequential(
            nn.Linear(2 * event_width, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.event_gate = nn.Linear(2 * width, 1)
        nn.init.constant_(self.event_gate.bias, -1.0)

    def _events(self, batch: EventBatch) -> tuple[torch.Tensor, torch.Tensor]:
        counts = batch.event_counts.flatten()
        flat_stays = counts.numel()
        present = counts > 0
        output = batch.event_values.new_zeros((flat_stays, self.gap_projection.out_features))
        if not bool(present.any()):
            return output, present
        if int(counts.sum()) != batch.event_values.shape[0]:
            raise ValueError("event offsets do not match flattened values")
        encoded = (self.event_value_projection(batch.event_values)
                   + self.event_embedding(batch.event_concepts))
        sequences = torch.split(encoded, counts.tolist())
        ordered = torch.argsort(counts, descending=True)
        ordered = ordered[counts[ordered] > 0]
        # 长度相近的住院合批，限制填充后 token 数；不截断任何原始事件。
        cursor = 0
        while cursor < len(ordered):
            first_length = int(counts[ordered[cursor]])
            allowance = max(1, min(64, 8192 // first_length))
            positions = ordered[cursor:cursor + allowance]
            lengths = counts[positions]
            padded = pad_sequence([sequences[int(index)] for index in positions], batch_first=True)
            packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True,
                                          enforce_sorted=True)
            transformed, final = self.event_gru(packed)
            states, _ = pad_packed_sequence(transformed, batch_first=True)
            time_index = torch.arange(states.shape[1], device=states.device)
            mask = time_index.unsqueeze(0) < lengths.unsqueeze(1)
            score = self.event_score(states).squeeze(-1).masked_fill(~mask, -1e4)
            weight = torch.softmax(score, dim=-1)
            pooled = (states * weight.unsqueeze(-1)).sum(dim=1)
            summary = self.event_projection(torch.cat((final[-1], pooled), dim=-1))
            output = output.index_copy(0, positions, summary)
            cursor += len(positions)
        return output, present

    def forward(self, batch: EventBatch) -> V5Output:
        source = batch.base
        batch_size, length = source.valid.shape
        flat_stays = batch_size * length
        structured = self._structured(source)
        events, event_present = self._events(batch)
        event_gate = torch.sigmoid(self.event_gate(torch.cat((structured, events), dim=-1)))
        event_gate = event_gate * event_present.unsqueeze(-1)
        history = structured + event_gate * events
        text, text_present = self._text(source, flat_stays)
        text_gate = torch.sigmoid(self.fusion_gate(torch.cat((history, text), dim=-1)))
        text_gate = text_gate * text_present.unsqueeze(-1)
        admission = self.fusion_norm(history + text_gate * text)
        admission = admission.view(batch_size, length, -1) + self.gap_projection(source.gap)
        causal = torch.triu(torch.ones((length, length), dtype=torch.bool,
                                       device=source.valid.device), diagonal=1)
        patient = self.patient_encoder(admission, mask=causal,
                                       src_key_padding_mask=~source.valid)
        log_amplitude = self.risk_head(patient).clamp(-10.0, 6.0)
        shape = F.softplus(self.raw_shape) + 0.1
        return V5Output(log_amplitude, shape, text_gate.view(batch_size, length, 1))
