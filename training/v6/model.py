"""历史锚定 + 本次临床修正；无患者时序 Transformer，无新增文本编码。"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.risk import RiskOutput


@dataclass(frozen=True)
class ClinicalBatch:
    current_values: torch.Tensor    # [N,14]
    current_masks: torch.Tensor     # [N,14] bool
    lab_axis_inputs: torch.Tensor   # [N,22,9] 6值+3标记
    med_exposure: torch.Tensor      # [N,7] 二元记录暴露，不是实际未治疗的证据
    med_source_mask: torch.Tensor   # [N,7] bool，来源可用性而非用药时序
    snippet_vectors: torch.Tensor  # [K,D]
    snippet_index: torch.Tensor    # [K,2] admission_slot, field_id


class HistoryBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.amplitude = nn.Linear(6, 2)
        self.raw_shape = nn.Parameter(torch.zeros(2))
        nn.init.constant_(self.amplitude.bias, -2.)

    def forward(self, history: torch.Tensor) -> RiskOutput:
        if history.ndim != 2 or history.shape[1] != 6 or not torch.isfinite(history).all():
            raise ValueError("history requires six finite, preprocessed fields")
        return RiskOutput(self.amplitude(history).clamp(-10., 6.),
                          (F.softplus(self.raw_shape) + .1).expand(len(history), 2))


class ConstantCorrection(nn.Module):
    """不读取任何本次状态的4参数再校准，匹配临床头的幅度/形状自由度。"""
    def __init__(self, baseline: HistoryBaseline):
        super().__init__()
        self.baseline = baseline.requires_grad_(False).eval()
        self.correction = nn.Parameter(torch.zeros(2, 2))

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        return self

    def forward(self, history: torch.Tensor) -> RiskOutput:
        with torch.no_grad():
            base = self.baseline(history)
        delta = self.correction.unsqueeze(0).expand(len(history), -1, -1)
        shape = (F.softplus(self.baseline.raw_shape.detach() + self.correction[:, 1]) + .1).expand(len(history), 2)
        return RiskOutput((base.log_amplitude + delta[:, :, 0]).clamp(-10., 6.), shape, delta)


def masked_pool(tokens: torch.Tensor, present: torch.Tensor, score: nn.Module) -> torch.Tensor:
    safe = present.clone()
    safe[~present.any(1), 0] = True
    logits = score(tokens).squeeze(-1).masked_fill(~safe, -torch.inf)
    pooled = (tokens * torch.softmax(logits, 1).unsqueeze(-1)).sum(1)
    return torch.where(present.any(1, keepdim=True), pooled, torch.zeros_like(pooled))


class ClinicalEncoder(nn.Module):
    def __init__(self, text_width: int, *, width: int = 128):
        super().__init__()
        if text_width < 1 or width < 1:
            raise ValueError("invalid encoder width")
        self.width, self.text_width = width, text_width
        self.current = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, width))
        self.current_ids = nn.Embedding(14, width)
        self.current_score = nn.Linear(width, 1)
        self.lab = nn.Sequential(nn.Linear(9, 32), nn.GELU(), nn.Linear(32, width))
        self.lab_ids = nn.Embedding(22, width)
        self.lab_score = nn.Linear(width, 1)
        self.med = nn.Sequential(nn.Linear(14, 32), nn.GELU(), nn.Linear(32, width), nn.LayerNorm(width))
        self.field_ids = nn.Embedding(4, width)
        self.text_score = nn.Linear(width, 1)
        self.branch_norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(3))
        self.fusion = nn.Sequential(nn.Linear(4 * width + 3, width), nn.GELU(), nn.Dropout(.1),
                                    nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))
        # 维度依赖模块最后初始化；构造后训练入口还必须重设 RNG。
        self.text_projection = nn.Sequential(nn.LayerNorm(text_width), nn.Linear(text_width, width),
                                              nn.GELU(), nn.LayerNorm(width))

    def _text(self, batch: ClinicalBatch, n: int, *, mask_pmh: bool) -> tuple[torch.Tensor, torch.Tensor]:
        vectors, index = batch.snippet_vectors, batch.snippet_index
        if (vectors.ndim != 2 or vectors.shape[1] != self.text_width or index.shape != (len(vectors), 2)
                or index.dtype != torch.long or (index < 0).any()
                or (index[:, 0] >= n).any() or (index[:, 1] >= 4).any()):
            raise ValueError("invalid text row/index contract")
        if mask_pmh:
            keep = index[:, 1] != 1
            vectors, index = vectors[keep], index[keep]
        if not torch.isfinite(vectors).all():
            raise ValueError("nonfinite active text encoding")
        if not len(vectors):
            return batch.current_values.new_zeros((n, self.width)), torch.zeros(n, device=vectors.device, dtype=torch.bool)
        content = self.text_projection(vectors) + self.field_ids(index[:, 1])
        group = 4 * index[:, 0] + index[:, 1]
        score = self.text_score(content).squeeze(-1)
        maximum = score.new_full((4 * n,), -torch.inf)
        maximum.scatter_reduce_(0, group, score, reduce="amax", include_self=True)
        weights = torch.exp(score - maximum[group])
        total = score.new_zeros(4 * n).scatter_add_(0, group, weights)
        weights = weights / total[group]
        pooled = content.new_zeros((4 * n, self.width)).index_add_(0, group, weights[:, None] * content)
        present = (total > 0).reshape(n, 4)
        pooled = pooled.reshape(n, 4, self.width).sum(1) / present.sum(1).clamp_min(1)[:, None]
        return pooled, present.any(1)

    def forward(self, batch: ClinicalBatch, *, mask_pmh: bool = False) -> tuple[torch.Tensor, dict]:
        n = len(batch.current_values)
        if (n == 0 or batch.current_values.shape != (n, 14) or batch.current_masks.shape != (n, 14)
                or batch.current_masks.dtype != torch.bool or batch.lab_axis_inputs.shape != (n, 22, 9)
                or batch.med_exposure.shape != (n, 7) or batch.med_source_mask.shape != (n, 7)
                or batch.med_source_mask.dtype != torch.bool):
            raise ValueError("invalid current/laboratory/medication contract")
        current = torch.where(batch.current_masks, batch.current_values, 0.)
        observed = batch.lab_axis_inputs[:, :, 6] > 0
        labs = torch.where(observed[:, :, None], batch.lab_axis_inputs, 0.)
        med = torch.where(batch.med_source_mask, batch.med_exposure, 0.)
        if not all(torch.isfinite(x).all() for x in (current, labs, med)):
            raise ValueError("nonfinite active clinical input")
        if not ((med == 0) | (med == 1)).all():
            raise ValueError("medication input must be binary exposure only")
        ids = torch.arange(14, device=current.device)
        tokens = self.current(torch.stack((current, batch.current_masks.to(current.dtype)), -1)) + self.current_ids(ids)
        s = masked_pool(tokens, batch.current_masks, self.current_score)
        l = masked_pool(self.lab(labs) + self.lab_ids(torch.arange(22, device=labs.device)), observed, self.lab_score)
        med_present = batch.med_source_mask.any(1)
        m = self.med(torch.cat((med, batch.med_source_mask.to(med.dtype)), 1)) * med_present[:, None]
        s, l, m = (norm(x) for norm, x in zip(self.branch_norms, (s, l, m), strict=True))
        # LayerNorm 的可学习偏置不可令空分支凭空出现状态。
        s = s * batch.current_masks.any(1)[:, None]
        l = l * observed.any(1)[:, None]
        m = m * med_present[:, None]
        text, text_present = self._text(batch, n, mask_pmh=mask_pmh)
        presence = torch.stack((observed.any(1), med_present, text_present), 1).to(s.dtype)
        h = self.fusion(torch.cat((s, l, m, text, presence), 1))
        return h, {"structured": s, "lab": l, "medication": m, "text": text, "presence": presence}


class AnchoredModel(nn.Module):
    def __init__(self, baseline: HistoryBaseline, text_width: int, *, width: int = 128,
                 patient_shape: bool = True):
        super().__init__()
        self.baseline = baseline
        self.baseline.requires_grad_(False).eval()
        self.delta_head = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 4))
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        self.encoder = ClinicalEncoder(text_width, width=width)
        self.patient_shape = patient_shape

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        return self

    def forward(self, history: torch.Tensor, batch: ClinicalBatch, *, mask_pmh: bool = False) -> RiskOutput:
        with torch.no_grad():
            base = self.baseline(history)
        h, _ = self.encoder(batch, mask_pmh=mask_pmh)
        if len(h) != len(history):
            raise ValueError("history/current admission alignment mismatch")
        delta = self.delta_head(h).reshape(-1, 2, 2)
        shape_delta = delta[:, :, 1] if self.patient_shape else torch.zeros_like(delta[:, :, 1])
        shape = F.softplus(self.baseline.raw_shape.detach() + shape_delta) + .1
        return RiskOutput((base.log_amplitude + delta[:, :, 0]).clamp(-10., 6.), shape, delta)
