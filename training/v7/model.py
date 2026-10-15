"""三层读取、药物/影像查询及显式历史衰减；不加载可训练 BERT。"""
import copy
from dataclasses import dataclass, fields, replace

import torch
from torch import nn
from torch.nn import functional as F

from training.v6.risk import RiskOutput

D_MODEL, HEADS, FFN, QUERIES = 256, 8, 1024, 4


@dataclass(frozen=True)
class V7Batch:
    context: torch.Tensor
    labs: torch.Tensor
    eos: torch.Tensor
    medications: torch.Tensor
    med_present: torch.Tensor
    coarse_meds: torch.Tensor
    micro: torch.Tensor
    micro_present: torch.Tensor
    support: torch.Tensor
    support_present: torch.Tensor
    text_vectors: torch.Tensor
    text_index: torch.Tensor
    text_slot: torch.Tensor
    radiology: torch.Tensor
    rad_present: torch.Tensor
    current: torch.Tensor
    past: torch.Tensor
    past_present: torch.Tensor
    gaps: torch.Tensor

    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)})


def safe_attention(attention, query, keys, present, *, bias=None):
    """全空来源临时开放零占位；读取结果随后清零，避免全遮罩 NaN。"""
    any_key = present.any(1)
    safe = present.clone()
    safe[:, 0] |= ~any_key
    clean = torch.where(present[:, :, None], keys, 0.)
    if bias is None:
        result, _ = attention(query, clean, clean, key_padding_mask=~safe, need_weights=False)
    else:
        # 与时间偏置同用浮点 mask，避免 bool/float mask 类型不匹配。
        padding = torch.zeros_like(present, dtype=query.dtype).masked_fill(~safe, float('-inf'))
        result, _ = attention(query, clean, clean, key_padding_mask=padding,
                              attn_mask=bias, need_weights=False)
    return result * any_key[:, None, None]


class MultiQueryPool(nn.Module):
    def __init__(self, *, decay=False):
        super().__init__()
        self.query = nn.Parameter(torch.randn(QUERIES, D_MODEL) * .02)
        self.attention = nn.MultiheadAttention(D_MODEL, HEADS, .1, batch_first=True)
        self.project = nn.Sequential(nn.Linear(QUERIES * D_MODEL, D_MODEL), nn.GELU(),
                                     nn.Dropout(.1), nn.LayerNorm(D_MODEL))
        self.decay = decay
        if decay:
            self.lambda_raw = nn.Parameter(torch.zeros(HEADS))

    def time_bias(self, gaps):
        # softplus(lambda)>0；给定内容相同，间隔越久未归一化权重越小。
        bias = -F.softplus(self.lambda_raw)[None, :, None, None] * torch.log1p(gaps.clamp_min(0))[:, None, None, :]
        return bias.expand(-1, -1, QUERIES, -1).reshape(-1, QUERIES, gaps.shape[1])

    def forward(self, tokens, present, gaps=None):
        if self.decay and gaps is None:
            raise ValueError('decay pool requires gaps')
        q = self.query[None].expand(len(tokens), -1, -1)
        bias = self.time_bias(gaps) if self.decay else None
        queries = safe_attention(self.attention, q, tokens, present, bias=bias)
        pooled = self.project(queries.flatten(1)) * present.any(1)[:, None]
        return pooled, queries


class CrossSource(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(D_MODEL, HEADS, .1, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(nn.Linear(D_MODEL, FFN), nn.GELU(), nn.Dropout(.1),
                                 nn.Linear(FFN, D_MODEL), nn.Dropout(.1))

    def forward(self, q, q_present, keys, key_present):
        read = safe_attention(self.attention, q, keys, key_present)
        x = self.norm1(q + read)
        return self.norm2(x + self.ffn(x)) * q_present[:, :, None]


class V7Model(nn.Module):
    def __init__(self, baseline, drug_parent, seed=42):
        super().__init__()
        self.baseline = copy.deepcopy(baseline).requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.context_projection = nn.Sequential(nn.LayerNorm(34), nn.Linear(34, D_MODEL),
                                                     nn.GELU(), nn.Dropout(.1))
            self.lab_projection = nn.Sequential(nn.Linear(9, D_MODEL), nn.GELU(), nn.Dropout(.1))
            self.lab_ids = nn.Embedding(36, D_MODEL)
            self.eos_projection = nn.Sequential(nn.Linear(16, D_MODEL), nn.GELU(), nn.Dropout(.1))
            self.drug_ids = nn.Embedding(38, D_MODEL)
            self.drug_parents = nn.Embedding(10, D_MODEL)
            parent = torch.as_tensor(drug_parent, dtype=torch.float32)
            if parent.shape != (38, 10) or (parent < 0).any() or not torch.isfinite(parent).all():
                raise ValueError('38 by 10 nonnegative drug parents required')
            self.register_buffer('drug_parent', parent / parent.sum(1, keepdim=True).clamp_min(1))
            self.drug_numeric = nn.Linear(4, D_MODEL)
            self.drug_units = nn.Linear(7, D_MODEL, bias=False)
            self.drug_norm = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Dropout(.1))
            self.coarse_projection = nn.Sequential(nn.Linear(4, D_MODEL), nn.GELU(), nn.Dropout(.1))
            self.coarse_ids = nn.Embedding(7, D_MODEL)
            self.micro_projection = nn.Sequential(nn.Linear(2, D_MODEL), nn.GELU(), nn.Dropout(.1))
            self.micro_ids = nn.Embedding(5, D_MODEL)
            self.support_projection = nn.Sequential(nn.Linear(2, D_MODEL), nn.GELU(), nn.Dropout(.1))
            self.support_ids = nn.Embedding(3, D_MODEL)
            self.text_projection = nn.Sequential(nn.Linear(768, D_MODEL), nn.GELU(), nn.Dropout(.1), nn.LayerNorm(D_MODEL))
            self.text_fields = nn.Embedding(4, D_MODEL)
            self.rad_projection = nn.Sequential(nn.Linear(1536, D_MODEL), nn.GELU(), nn.Dropout(.1), nn.LayerNorm(D_MODEL))
            self.rad_ids = nn.Embedding(2, D_MODEL)
            self.field_pool = nn.ModuleList([MultiQueryPool() for _ in range(4)])
            self.source_pool = nn.ModuleDict({n: MultiQueryPool() for n in ('lab', 'drug', 'coarse', 'micro', 'support', 'text', 'rad')})
            self.drug_cross, self.rad_cross = CrossSource(), CrossSource()
            self.fusion = nn.Sequential(nn.Linear(8 * D_MODEL + 8, D_MODEL), nn.GELU(),
                                        nn.Dropout(.3), nn.Linear(D_MODEL, D_MODEL), nn.LayerNorm(D_MODEL))
            self.gap_projection = nn.Sequential(nn.Linear(1, D_MODEL), nn.Tanh())
            layer = nn.TransformerEncoderLayer(D_MODEL, HEADS, FFN, .1, batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.history_pool = MultiQueryPool(decay=True)
            self.history_fusion = nn.Sequential(nn.Linear(2 * D_MODEL, D_MODEL), nn.GELU(), nn.Dropout(.3), nn.LayerNorm(D_MODEL))
            self.delta_head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Dropout(.3), nn.Linear(D_MODEL, 4))
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)
            # 对比读取跨源交互前的表示，避免正对已混入另一来源而形成捷径。
            self.contrast_projection = nn.ModuleDict({n: nn.Linear(D_MODEL, D_MODEL) for n in ('text', 'drug', 'lab', 'rad')})

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        return self

    def text(self, b, n):
        width = max(1, int(b.text_slot.max().item()) + 1) if len(b.text_slot) else 1
        dense = b.context.new_zeros(n, width, D_MODEL)
        field = torch.full((n, width), -1, dtype=torch.long, device=b.context.device)
        if len(b.text_vectors):
            v, f = b.text_index[:, 0], b.text_index[:, 1]
            dense[v, b.text_slot] = self.text_projection(b.text_vectors) + self.text_fields(f)
            field[v, b.text_slot] = f
        tokens, masks, states, state_masks = [], [], [], []
        for i, pool in enumerate(self.field_pool):
            mask = field == i
            state, q = pool(dense, mask)
            states.append(state)
            state_masks.append(mask.any(1))
            tokens.append(q)
            masks.append(mask.any(1)[:, None].expand(-1, QUERIES))
        tokens, present = torch.cat(tokens, 1), torch.cat(masks, 1)
        # 字段 M4 的4×256压缩投影参与住院读出；未压缩的查询仍用于跨源读取。
        field_states = torch.stack(states, 1)
        field_present = torch.stack(state_masks, 1)
        return self.source_pool['text'](field_states, field_present)[0], tokens, present

    def visits(self, b):
        n = len(b.context)
        lab_mask = b.labs[:, :, 6] > 0
        lab = self.lab_projection(torch.where(lab_mask[:, :, None], b.labs, 0.)) + self.lab_ids.weight
        eos_mask = lab_mask[:, 32:].any(1)
        lab = torch.cat((lab, self.eos_projection(torch.where(eos_mask[:, None], b.eos, 0.))[:, None]), 1)
        lab_mask = torch.cat((lab_mask, eos_mask[:, None]), 1)
        lab_rep = self.source_pool['lab'](lab, lab_mask)[0]
        numeric = b.medications[:, :, :4].clone()
        numeric[:, :, :2] = torch.where(numeric[:, :, 2:4].bool(), numeric[:, :, :2], 0.)
        drug = self.drug_ids.weight + self.drug_parent @ self.drug_parents.weight
        drug = self.drug_norm(drug + self.drug_numeric(numeric) + self.drug_units(b.medications[:, :, 4:]))
        text_rep, text_tokens, text_mask = self.text(b, n)
        rad = self.rad_projection(torch.where(b.rad_present[:, :, None], b.radiology, 0.)) + self.rad_ids.weight
        drug_rep = self.source_pool['drug'](drug, b.med_present)[0]
        rad_rep = self.source_pool['rad'](rad, b.rad_present)[0]
        source_reps = dict(text=text_rep, drug=drug_rep, lab=lab_rep, rad=rad_rep)
        source_masks = dict(text=text_mask.any(1), drug=b.med_present.any(1), lab=lab_mask.any(1), rad=b.rad_present.any(1))
        drug = self.drug_cross(drug, b.med_present, torch.cat((text_tokens, rad), 1), torch.cat((text_mask, b.rad_present), 1))
        rad = self.rad_cross(rad, b.rad_present, text_tokens, text_mask)
        coarse_mask = b.coarse_meds[:, :, 2:].bool().any(2)
        coarse = self.coarse_projection(torch.where(coarse_mask[:, :, None], b.coarse_meds, 0.)) + self.coarse_ids.weight
        micro = self.micro_projection(torch.stack((torch.where(b.micro_present, b.micro, 0.), b.micro_present.float()), 2)) + self.micro_ids.weight
        support = self.support_projection(torch.stack((torch.where(b.support_present, b.support, 0.), b.support_present.float()), 2)) + self.support_ids.weight
        branches = [self.context_projection(b.context), lab_rep,
                    self.source_pool['drug'](drug, b.med_present)[0],
                    self.source_pool['coarse'](coarse, coarse_mask)[0],
                    self.source_pool['micro'](micro, b.micro_present)[0],
                    self.source_pool['support'](support, b.support_present)[0], text_rep,
                    self.source_pool['rad'](rad, b.rad_present)[0]]
        present = torch.stack((torch.ones(n, dtype=torch.bool, device=b.context.device), source_masks['lab'],
                               source_masks['drug'], coarse_mask.any(1), b.micro_present.any(1),
                               b.support_present.any(1), source_masks['text'], source_masks['rad']), 1)
        return self.fusion(torch.cat((*branches, present.float()), 1)), source_reps, source_masks

    def forward(self, b, *, auxiliary=False):
        hidden, source_reps, source_masks = self.visits(b)
        current = hidden[b.current]
        past = hidden[b.past] + self.gap_projection(torch.log1p(b.gaps.clamp_min(0))[:, :, None] / 6.)
        past = torch.where(b.past_present[:, :, None], past, 0.)
        seq = torch.cat((current[:, None], past), 1)
        valid = torch.cat((torch.ones(len(current), 1, dtype=torch.bool, device=current.device), b.past_present), 1)
        encoded = self.temporal(seq, src_key_padding_mask=~valid)
        history, _ = self.history_pool(encoded[:, 1:], b.past_present, b.gaps)
        read = self.history_fusion(torch.cat((encoded[:, 0], history), 1))
        delta = self.delta_head(read).reshape(-1, 2, 2)
        with torch.no_grad():
            base = self.baseline(b.context[b.current])
        out = RiskOutput((base.log_amplitude + delta[:, :, 0]).clamp(-10., 6.),
                         F.softplus(self.baseline.raw_shape.detach() + delta[:, :, 1]) + .1, delta)
        if not auxiliary:
            return out
        reps = {k: F.normalize(self.contrast_projection[k](v[b.current]), dim=1) for k, v in source_reps.items()}
        masks = {k: v[b.current] for k, v in source_masks.items()}
        return out, reps, masks

    def optimizer_groups(self):
        slow, fast, names = [], [], {'cached_projection': [], 'new_modules': []}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(('text_projection.', 'rad_projection.')):
                slow.append(p); names['cached_projection'].append(name)
            else:
                fast.append(p); names['new_modules'].append(name)
        groups = [dict(params=slow, lr=1e-4, group_name='cached_projection'),
                  dict(params=fast, lr=3e-4, group_name='new_modules')]
        flat = [id(p) for g in groups for p in g['params']]
        if len(flat) != len(set(flat)) or set(flat) != {id(p) for p in self.parameters() if p.requires_grad}:
            raise ValueError('optimizer groups duplicate or incomplete')
        return groups, names
