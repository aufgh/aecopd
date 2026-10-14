"""独立探索配方：扩展基础参照、保留状态、集合交互、数值编码。旧V6不改。"""
from dataclasses import dataclass, replace
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.model import HistoryBaseline, AnchoredModel, masked_pool
from training.v6.compact_model import CompactClinicalEncoder
from training.v6.compact_context_data import CompactContextBatch
from training.v6.risk import RiskOutput


class ExpandedBaseline(HistoryBaseline):
    """34维=原历史6+当前编码/去向14+有效性14；不是纯历史模型。"""
    def __init__(self, history):
        nn.Module.__init__(self)
        self.amplitude = nn.Linear(34, 2)
        self.raw_shape = nn.Parameter(history.raw_shape.detach().clone())
        with torch.no_grad():
            self.amplitude.weight.zero_()
            self.amplitude.weight[:, :6].copy_(history.amplitude.weight)
            self.amplitude.bias.copy_(history.amplitude.bias)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != 34 or not torch.isfinite(x).all():
            raise ValueError('expanded baseline requires 34 finite inputs')
        return RiskOutput(self.amplitude(x).clamp(-10, 6), (F.softplus(self.raw_shape)+.1).expand(len(x), 2))


def expanded_inputs(history, batch):
    b = batch.clinical
    return torch.cat((history, torch.where(b.current_masks, b.current_values, 0),
                      b.current_masks.to(history.dtype)), 1)


@dataclass(frozen=True)
class ExplorationBatch:
    compact: CompactContextBatch
    composite: torch.Tensor  # [N,5,9] 三原成分和NLR/SII；无效槽位及标记同时屏蔽
    candidate_vectors: torch.Tensor
    candidate_index: torch.Tensor  # 四个明确路由组，不等同旧B四原章节


class Piecewise(nn.Module):
    """每轴每数值六个训练分位区间；缺失遮罩仍在调用方保留。"""
    def __init__(self, knots):
        super().__init__()
        if knots.shape != (22, 6, 7) or not torch.isfinite(knots).all():
            raise ValueError('invalid train-only knots')
        self.register_buffer('knots', knots.clone())

    def forward(self, labs):
        left, right = self.knots[..., :-1], self.knots[..., 1:]
        values = ((labs[:, :, :6, None] - left) / (right-left).clamp_min(1e-6)).clamp(0, 1)
        return torch.cat((values.flatten(2), labs[:, :, 6:]), -1)


def field_states(encoder, b, n, *, mask_pmh=False):
    vectors, index = b.snippet_vectors, b.snippet_index
    if (vectors.ndim != 2 or vectors.shape[1] != encoder.text_width or index.shape != (len(vectors), 2)
            or index.dtype != torch.long or (index < 0).any() or (index[:, 0] >= n).any()
            or (index[:, 1] >= 4).any()):
        raise ValueError('invalid field index')
    if mask_pmh:
        keep = index[:, 1] != 1
        vectors, index = vectors[keep], index[keep]
    if not torch.isfinite(vectors).all(): raise ValueError('nonfinite text')
    if not len(vectors):
        return b.current_values.new_zeros(n, 4, encoder.width), torch.zeros(n, 4, device=vectors.device, dtype=torch.bool)
    content = encoder.text_projection(vectors) + encoder.field_ids(index[:, 1])
    group = 4*index[:, 0]+index[:, 1]
    score = encoder.text_score(content).squeeze(-1)
    maximum = score.new_full((4*n,), -torch.inf).scatter_reduce_(0, group, score, reduce='amax', include_self=True)
    weights = torch.exp(score-maximum[group])
    total = score.new_zeros(4*n).scatter_add_(0, group, weights)
    weights = weights/total[group]
    states = content.new_zeros(4*n, encoder.width).index_add_(0, group, weights[:, None]*content)
    return states.reshape(n, 4, encoder.width), (total > 0).reshape(n, 4)


class ExplorationEncoder(CompactClinicalEncoder):
    def __init__(self, text_width, *, width=128, hidden=32, mode='pooled', knots=None):
        super().__init__(text_width, width=width)
        self.mode = mode
        if mode not in ('pooled', 'retained', 'set', 'piecewise'): raise ValueError('invalid architecture')
        self.piecewise = Piecewise(knots) if mode == 'piecewise' else None
        self.lab = nn.Sequential(nn.Linear(39 if mode == 'piecewise' else 9, hidden), nn.GELU(), nn.Linear(hidden, width))
        self.med = nn.Sequential(nn.Linear(28, hidden), nn.GELU(), nn.Linear(hidden, width))
        self.microbiology = nn.Sequential(nn.Linear(10, hidden), nn.GELU(), nn.Linear(hidden, width))
        self.support = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, width))
        self.combination = nn.Sequential(nn.Linear(45, hidden), nn.GELU(), nn.Linear(hidden, width))
        if mode == 'set':
            layer = nn.TransformerEncoderLayer(width, 4, 2*width, .1, batch_first=True, norm_first=True)
            self.interaction = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.queries = nn.Parameter(torch.randn(1, 4, width)*.02)
            self.multi_pool = nn.MultiheadAttention(width, 4, .1, batch_first=True)
        lab_width = 198 if mode == 'retained' else 4*width if mode == 'set' else width
        # 四字段状态保留；候选四路由组另列，不与B的原章节混合池化。
        text_width_total = 4*width if mode in ('retained', 'set') else width
        self.fusion = nn.Sequential(nn.Linear(lab_width+3*width+text_width_total+4*width+width+6, width),
                                   nn.GELU(), nn.Dropout(.1), nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))

    def forward(self, batch, *, mask_pmh=False):
        if not isinstance(batch, ExplorationBatch): raise TypeError('explicit exploration batch required')
        c, b = batch.compact, batch.compact.clinical
        n = len(b.current_values)
        if batch.composite.shape != (n, 5, 9) or not torch.isfinite(batch.composite).all():
            raise ValueError('invalid combination features')
        observed = b.lab_axis_inputs[:, :, 6] > 0
        labs = torch.where(observed[:, :, None], b.lab_axis_inputs, 0)
        if labs.shape != (n, 22, 9) or not torch.isfinite(labs).all(): raise ValueError('invalid labs')
        if self.mode == 'retained':
            l = labs.flatten(1)
        else:
            tokens = self.lab(self.piecewise(labs) if self.piecewise else labs) + self.lab_ids(torch.arange(22, device=labs.device))
            if self.mode == 'set':
                safe = observed.clone(); safe[~safe.any(1), 0] = True
                tokens = self.interaction(tokens, src_key_padding_mask=~safe)
                queries = self.queries.expand(n, -1, -1)
                l = self.multi_pool(queries, tokens, tokens, key_padding_mask=~safe, need_weights=False)[0].flatten(1)
                l = l*observed.any(1)[:, None]
            else:
                l = masked_pool(tokens, observed, self.lab_score)
        m_present = c.med_features[:, :, 2:].bool().any(dim=(1, 2))
        m = self.med(c.med_features.flatten(1))*m_present[:, None]
        micro = self.microbiology(torch.cat((c.micro_values, c.micro_masks.float()), 1))*c.micro_masks.any(1)[:, None]
        support = self.support(torch.cat((c.support_values, c.support_masks.float()), 1))*c.support_masks.any(1)[:, None]
        if self.mode in ('pooled', 'piecewise'):
            l = self.branch_norms[1](l)*observed.any(1)[:, None]
        m = self.branch_norms[2](m)*m_present[:, None]
        micro = self.branch_norms[3](micro)*c.micro_masks.any(1)[:, None]
        support = self.branch_norms[4](support)*c.support_masks.any(1)[:, None]
        text, present = field_states(self, b, n, mask_pmh=mask_pmh)
        txt = text.flatten(1) if self.mode in ('retained', 'set') else text.sum(1)/present.sum(1).clamp_min(1)[:, None]
        candidate_b = replace(b, snippet_vectors=batch.candidate_vectors, snippet_index=batch.candidate_index)
        candidate, cp = field_states(self, candidate_b, n)  # 路由组1不是PMH，不能复用mask_pmh含义
        comb = self.combination(batch.composite.flatten(1))*(batch.composite[:, :, 6] > 0).any(1)[:, None]
        availability = torch.stack((observed.any(1), m_present, c.micro_masks.any(1), c.support_masks.any(1), present.any(1), cp.any(1)), 1).float()
        h = self.fusion(torch.cat((l, m, micro, support, txt, candidate.flatten(1), comb, availability), 1))
        return h, dict(lab=l, medication=m, microbiology=micro, support=support, text=txt, combination=comb)


class ExplorationModel(AnchoredModel):
    def __init__(self, baseline, text_width, *, width=128, hidden=32, mode='pooled', knots=None):
        nn.Module.__init__(self)
        self.baseline = baseline.requires_grad_(False).eval()
        self.delta_head = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 4))
        nn.init.zeros_(self.delta_head[-1].weight); nn.init.zeros_(self.delta_head[-1].bias)
        self.encoder = ExplorationEncoder(text_width, width=width, hidden=hidden, mode=mode, knots=knots)
        self.patient_shape = True
