"""V5 六组结构化、八领域文本与患者因果序列的竞争风险模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training.v5.data import GROUP_WIDTHS, V5Batch


@dataclass(frozen=True)
class V5Output:
    log_amplitude: torch.Tensor
    shape: torch.Tensor
    gate: torch.Tensor


class V5Model(nn.Module):
    """每次合格出院读取其因果序列位置，输出两病因 Weibull 参数。"""

    def __init__(self, width: int = 128, text_width: int = 768,
                 balance_text: bool = False) -> None:
        super().__init__()
        if width % 4:
            raise ValueError("width must be divisible by four heads")
        self.balance_text = balance_text
        self.group_projections = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * size, width), nn.GELU(), nn.LayerNorm(width))
            for size in GROUP_WIDTHS
        )
        self.group_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, 256, dropout=0.1, batch_first=True),
            num_layers=1, enable_nested_tensor=False,
        )
        self.group_score = nn.Linear(width, 1)
        self.domain_embedding = nn.Embedding(8, width)
        self.snippet_score = nn.Linear(width, 1)
        self.domain_score = nn.Linear(width, 1)
        self.cross_attention = nn.MultiheadAttention(width, 4, dropout=0.1, batch_first=True)
        self.gate_layer = nn.Linear(3 * width, 2)
        self.fusion_norm = nn.LayerNorm(width)
        self.gap_projection = nn.Linear(2, width)
        self.patient_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, 256, dropout=0.1, batch_first=True),
            num_layers=1, enable_nested_tensor=False,
        )
        self.risk_head = nn.Sequential(nn.Linear(width, 32), nn.ReLU(),
                                       nn.Dropout(0.2), nn.Linear(32, 2))
        self.raw_shape = nn.Parameter(torch.zeros(2))
        with torch.no_grad():
            self.risk_head[-1].bias.copy_(torch.tensor([-1.5, -3.0]))
        # 输入维度不同只改变此层；其余层在同一随机种子下保持相同初始化。
        self.text_projection = nn.Linear(text_width, width)

    def _text_domains(self, batch: V5Batch, flat_stays: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按住院和临床领域对全部片段做可训练注意力池化，不截断片段。"""

        device = batch.valid.device
        domain_count = flat_stays * 8
        if batch.snippet_vectors.shape[0] == 0:
            return (torch.zeros((flat_stays, 8, self.text_projection.out_features), device=device),
                    torch.zeros((flat_stays, 8), dtype=torch.bool, device=device))
        field = batch.snippet_index[:, 1]
        segment = batch.snippet_index[:, 0] * 8 + field
        content = self.text_projection(batch.snippet_vectors)
        if self.balance_text:
            # 对照实验：使单位范数编码的内容与领域嵌入处于同一量级。
            content = F.layer_norm(content, (content.shape[-1],))
        projected = content + self.domain_embedding(field)
        score = self.snippet_score(torch.tanh(projected)).squeeze(-1)
        maximum = torch.full((domain_count,), -torch.inf, device=device)
        maximum.scatter_reduce_(0, segment, score, reduce="amax", include_self=True)
        weights = torch.exp(score - maximum[segment])
        totals = torch.zeros((domain_count,), device=device)
        totals.scatter_add_(0, segment, weights)
        weights = weights / totals[segment].clamp_min(1e-8)
        pooled = torch.zeros((domain_count, projected.shape[-1]), device=device)
        pooled.index_add_(0, segment, weights.unsqueeze(-1) * projected)
        counts = torch.zeros((domain_count,), device=device)
        counts.scatter_add_(0, segment, torch.ones_like(weights))
        return pooled.view(flat_stays, 8, -1), (counts > 0).view(flat_stays, 8)

    def forward(self, batch: V5Batch) -> V5Output:
        batch_size, length = batch.valid.shape
        flat_stays = batch_size * length
        tokens = torch.stack([
            layer(torch.cat((values, mask), dim=-1))
            for layer, values, mask in zip(self.group_projections, batch.group_values,
                                           batch.group_masks, strict=True)
        ], dim=2).reshape(flat_stays, len(GROUP_WIDTHS), -1)
        group_present = torch.stack([mask.any(dim=-1) for mask in batch.group_masks], dim=-1)
        group_present = group_present.reshape(flat_stays, len(GROUP_WIDTHS))
        group_present[:, 0] = True  # 填充住院也保留一个安全 token，最终由 valid 掩码清除。
        structured_tokens = self.group_encoder(tokens, src_key_padding_mask=~group_present)
        group_score = self.group_score(structured_tokens).squeeze(-1)
        group_score = group_score.masked_fill(~group_present, -1e4)
        group_weights = torch.softmax(group_score, dim=1)
        structured = (structured_tokens * group_weights.unsqueeze(-1)).sum(dim=1)

        domains, domain_present = self._text_domains(batch, flat_stays)
        any_text = domain_present.any(dim=1)
        safe_present = domain_present.clone()
        safe_present[~any_text, 0] = True
        domain_score = self.domain_score(torch.tanh(domains)).squeeze(-1)
        domain_score = domain_score.masked_fill(~safe_present, -1e4)
        domain_weights = torch.softmax(domain_score, dim=1)
        text = (domains * domain_weights.unsqueeze(-1)).sum(dim=1)
        text = text * any_text.unsqueeze(-1)
        cross_tokens, _ = self.cross_attention(structured_tokens, domains, domains,
                                                key_padding_mask=~safe_present,
                                                need_weights=False)
        cross = (cross_tokens * group_weights.unsqueeze(-1)).sum(dim=1)
        cross = cross * any_text.unsqueeze(-1)
        gate = torch.sigmoid(self.gate_layer(torch.cat((structured, text, cross), dim=-1)))
        gate = gate * any_text.unsqueeze(-1)
        admission = self.fusion_norm(structured + gate[:, :1] * text + gate[:, 1:] * cross)
        admission = admission.view(batch_size, length, -1)
        admission = admission + self.gap_projection(batch.gap)
        causal = torch.triu(torch.ones((length, length), dtype=torch.bool,
                                       device=batch.valid.device), diagonal=1)
        patient = self.patient_encoder(admission, mask=causal,
                                       src_key_padding_mask=~batch.valid)
        log_amplitude = self.risk_head(patient).clamp(-10.0, 6.0)
        shape = F.softplus(self.raw_shape) + 0.1
        return V5Output(log_amplitude, shape, gate.view(batch_size, length, 2))


def competing_nll(output: V5Output, batch: V5Batch) -> torch.Tensor:
    """两病因全似然；只在有效出院位置计算住院等权平均。"""

    if not torch.all((batch.events[batch.valid] >= 0) & (batch.events[batch.valid] <= 2)):
        raise ValueError("invalid event type")
    scaled = batch.times.clamp_min(1e-3).unsqueeze(-1) / 365.0
    cumulative = torch.exp(output.log_amplitude) * scaled.pow(output.shape)
    log_hazard = (output.log_amplitude + torch.log(output.shape)
                  - np.log(365.0) + (output.shape - 1.0) * torch.log(scaled))
    selected = log_hazard.gather(-1, (batch.events - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1)
    loss = cumulative.sum(dim=-1) - torch.where(batch.events > 0, selected, 0.0)
    return loss[batch.valid].mean()


def predict_cif(log_amplitude: np.ndarray, shape: np.ndarray,
                days: np.ndarray) -> np.ndarray:
    """按累计风险日增量积分，输出 [样本,时间,存活／再住院／死亡]。"""

    query = np.asarray(days, dtype=np.float64)
    if query.ndim != 1 or query[0] != 0 or query[-1] > 365 or np.any(np.diff(query) <= 0):
        raise ValueError("days must start at zero and increase within 365")
    amplitude = np.exp(np.asarray(log_amplitude, dtype=np.float64))
    powers = np.asarray(shape, dtype=np.float64)
    if amplitude.ndim != 2 or amplitude.shape[1] != 2 or powers.shape != (2,):
        raise ValueError("two cause-specific parameters required")
    whole_days = np.arange(366, dtype=np.float64)
    cumulative = amplitude[:, None, :] * (whole_days[None, :, None] / 365.0) ** powers
    increments = np.diff(cumulative, axis=1)
    total = increments.sum(axis=2)
    event_mass = -np.expm1(-total)
    share = np.divide(increments, total[:, :, None], out=np.zeros_like(increments),
                      where=total[:, :, None] > 0)
    survival = np.ones(amplitude.shape[0], dtype=np.float64)
    cif = np.zeros((amplitude.shape[0], 2), dtype=np.float64)
    curve = np.empty((amplitude.shape[0], 366, 3), dtype=np.float64)
    curve[:, 0] = np.column_stack((survival, cif))
    for day in range(365):
        cif += survival[:, None] * event_mass[:, day, None] * share[:, day]
        survival *= np.exp(-total[:, day])
        curve[:, day + 1] = np.column_stack((survival, cif))
    if np.array_equal(query, whole_days):
        return curve
    return np.stack([
        np.stack([np.interp(query, whole_days, row[:, cause]) for row in curve])
        for cause in range(3)
    ], axis=2)
