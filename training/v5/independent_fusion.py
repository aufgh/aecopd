"""结构化事实与字段文本分别编码，只在住院级融合的竞争风险模型。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from training.v5.model import V5Output


@dataclass(frozen=True)
class IndependentBatch:
    """一批完整患者历史；文本索引为 [展平住院位置, 字段编号]。"""

    token_values: torch.Tensor        # [B, L, S, 5]
    token_masks: torch.Tensor         # [B, L, S, 5]
    token_concepts: torch.Tensor      # [B, L, S]
    token_types: torch.Tensor         # [B, L, S]
    token_present: torch.Tensor       # [B, L, S]
    snippet_vectors: torch.Tensor    # [N, text_width]
    snippet_index: torch.Tensor      # [N, 2]
    gap: torch.Tensor                # [B, L, 2]
    valid: torch.Tensor              # [B, L]
    times: torch.Tensor              # [B, L]
    events: torch.Tensor             # [B, L]

    def to(self, device: torch.device | str) -> IndependentBatch:
        return IndependentBatch(*(value.to(device) for value in (
            self.token_values, self.token_masks, self.token_concepts,
            self.token_types, self.token_present, self.snippet_vectors,
            self.snippet_index, self.gap, self.valid, self.times, self.events,
        )))


class IndependentFusionModel(nn.Module):
    """两个住院级独立分支；没有章节—结构化 token 的逐项注意力。"""

    def __init__(self, *, concept_count: int, type_count: int, field_count: int,
                 text_width: int, width: int = 128) -> None:
        super().__init__()
        if min(concept_count, type_count, field_count, text_width) < 1 or width % 4:
            raise ValueError("invalid model vocabulary or width")
        self.concept_embedding = nn.Embedding(concept_count + 1, width, padding_idx=0)
        self.type_embedding = nn.Embedding(type_count + 1, width, padding_idx=0)
        self.value_projection = nn.Sequential(nn.Linear(10, width), nn.GELU(), nn.LayerNorm(width))
        self.structured_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.1, batch_first=True),
            num_layers=1, enable_nested_tensor=False,
        )
        self.structured_score = nn.Linear(width, 1)
        self.field_embedding = nn.Embedding(field_count, width)
        self.text_score = nn.Linear(width, 1)
        self.text_content_norm = nn.LayerNorm(width)
        self.text_norm = nn.LayerNorm(width)
        self.fusion_gate = nn.Linear(2 * width, 1)
        nn.init.constant_(self.fusion_gate.bias, -1.0)
        self.fusion_norm = nn.LayerNorm(width)
        self.gap_projection = nn.Linear(2, width)
        self.patient_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.1, batch_first=True),
            num_layers=1, enable_nested_tensor=False,
        )
        self.risk_head = nn.Sequential(nn.Linear(width, 32), nn.ReLU(),
                                       nn.Dropout(0.2), nn.Linear(32, 2))
        self.raw_shape = nn.Parameter(torch.zeros(2))
        with torch.no_grad():
            self.risk_head[-1].bias.copy_(torch.tensor([-1.5, -3.0]))
        # 维度变化只影响最后初始化的文本投影；其余参数的初始值可逐项比较。
        self.text_projection = nn.Linear(text_width, width)

    def _structured(self, batch: IndependentBatch) -> torch.Tensor:
        batch_size, length, slots = batch.token_present.shape
        flat_stays = batch_size * length
        present = batch.token_present.reshape(flat_stays, slots)
        safe = present.clone()
        safe[~present.any(dim=1), 0] = True
        value = torch.cat((batch.token_values, batch.token_masks), dim=-1)
        value = value.reshape(flat_stays, slots, 10)
        tokens = (self.value_projection(value)
                  + self.concept_embedding(batch.token_concepts.reshape(flat_stays, slots))
                  + self.type_embedding(batch.token_types.reshape(flat_stays, slots)))
        tokens = self.structured_encoder(tokens, src_key_padding_mask=~safe)
        score = self.structured_score(tokens).squeeze(-1).masked_fill(~safe, -1e4)
        weight = torch.softmax(score, dim=1)
        pooled = (tokens * weight.unsqueeze(-1)).sum(dim=1)
        return pooled * present.any(dim=1).unsqueeze(-1)

    def _text(self, batch: IndependentBatch, flat_stays: int) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.snippet_vectors.shape[0] == 0:
            zero = batch.token_values.new_zeros((flat_stays, self.text_projection.out_features))
            return zero, torch.zeros(flat_stays, dtype=torch.bool, device=zero.device)
        segment = batch.snippet_index[:, 0]
        field = batch.snippet_index[:, 1]
        if bool((segment < 0).any()) or bool((segment >= flat_stays).any()):
            raise ValueError("snippet references admission outside batch")
        if bool((field < 0).any()) or bool((field >= self.field_embedding.num_embeddings).any()):
            raise ValueError("unknown text field")
        # 冻结的编码向量通常为单位范数；先平衡内容尺度，再加入字段身份。
        content = self.text_content_norm(self.text_projection(batch.snippet_vectors))
        projected = self.text_norm(content + self.field_embedding(field))
        score = self.text_score(torch.tanh(projected)).squeeze(-1)
        maximum = torch.full((flat_stays,), -torch.inf, device=score.device)
        maximum.scatter_reduce_(0, segment, score, reduce="amax", include_self=True)
        weight = torch.exp(score - maximum[segment])
        totals = torch.zeros((flat_stays,), device=score.device)
        totals.scatter_add_(0, segment, weight)
        weight = weight / totals[segment].clamp_min(1e-8)
        pooled = torch.zeros((flat_stays, projected.shape[-1]), device=score.device)
        pooled.index_add_(0, segment, weight.unsqueeze(-1) * projected)
        present = totals > 0
        return pooled, present

    def forward(self, batch: IndependentBatch) -> V5Output:
        batch_size, length = batch.valid.shape
        flat_stays = batch_size * length
        structured = self._structured(batch)
        text, text_present = self._text(batch, flat_stays)
        gate = torch.sigmoid(self.fusion_gate(torch.cat((structured, text), dim=-1)))
        gate = gate * text_present.unsqueeze(-1)
        admission = self.fusion_norm(structured + gate * text)
        admission = admission.view(batch_size, length, -1) + self.gap_projection(batch.gap)
        causal = torch.triu(torch.ones((length, length), dtype=torch.bool,
                                       device=batch.valid.device), diagonal=1)
        patient = self.patient_encoder(admission, mask=causal,
                                       src_key_padding_mask=~batch.valid)
        log_amplitude = self.risk_head(patient).clamp(-10.0, 6.0)
        shape = F.softplus(self.raw_shape) + 0.1
        return V5Output(log_amplitude, shape, gate.view(batch_size, length, 1))
