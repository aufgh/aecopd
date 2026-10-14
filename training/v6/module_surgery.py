"""保留V6骨架，仅替换局部模块；官方包源码只读。"""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import rtdl_revisiting_models as rtdl
import rtdl_num_embeddings as numeric
import tabm

from training.v6.compact_model import CompactAnchoredModel
from training.v6.model import AnchoredModel
from training.v6.detail_extension_experiment import DetailEncoder
from training.v6.original_interaction import OriginalInteractionEncoder
from training.v6.risk import RiskOutput, competing_nll
from training.v6.token_model import fast_cif

ARMS = ('control', 'fusion_resnet', 'lab_ple', 'lab_periodic',
        'interaction_torch', 'interaction_ft', 'fusion_tabm')


def numeric_mask(x: torch.Tensor) -> torch.Tensor:
    """沿用原轴有效性，不把单次观测的恢复/峰后间隔当有效数值。"""
    observed = x[..., 6] > 0
    mask = observed[..., None].expand(*observed.shape, 6).clone()
    mask[..., 3] &= x[..., 7] > 0
    mask[..., 5] &= x[..., 7] > 0
    return mask


def fit_lab_bins(values: np.ndarray, training: np.ndarray) -> dict:
    if values.ndim != 3 or values.shape[1:] != (22, 9) or training.dtype != bool:
        raise ValueError('22-axis training-only numeric interface required')
    x = values[training]
    mask = np.broadcast_to((x[:, :, 6] > 0)[:, :, None], x[:, :, :6].shape).copy()
    mask[:, :, 3] &= x[:, :, 7] > 0
    mask[:, :, 5] &= x[:, :, 7] > 0
    bins, counts, fallback = [], [], []
    for axis in range(22):
        for channel in range(6):
            observed = x[:, axis, channel][mask[:, axis, channel]]
            if not np.isfinite(observed).all():
                raise ValueError('nonfinite observed train laboratory values')
            q = np.unique(np.quantile(observed, np.linspace(0, 1, 17))) if len(observed) else np.array([])
            constant = len(q) < 2
            if constant:
                q = np.array([-1., 1.])
            bins.append(q.astype(np.float32).tolist())
            counts.append(len(observed)); fallback.append(constant)
    return dict(bins=bins, observed_training_counts=counts, constant_or_empty_fallback=fallback,
                training_only=True, columns='axis-major six original numeric channels')


class NumericLabEncoder(nn.Module):
    """N×22×9 -> N×22×128；三标记、概念ID和池化仍由原V6处理。"""
    def __init__(self, kind: str, bins: list, embedding_width: int = 8):
        super().__init__()
        if kind == 'ple':
            self.embedding = numeric.PiecewiseLinearEmbeddings(
                [torch.as_tensor(b, dtype=torch.float32) for b in bins],
                embedding_width, activation=True, version='B')
        elif kind == 'periodic':
            self.embedding = numeric.PeriodicEmbeddings(132, embedding_width,
                n_frequencies=8, frequency_init_scale=.01, activation=True, lite=True)
        else:
            raise ValueError('unknown numerical embedding')
        self.projection = nn.Sequential(nn.Linear(6 * embedding_width + 3, 32),
                                         nn.GELU(), nn.Linear(32, 128))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (22, 9):
            raise ValueError('numeric module requires original N,22,9')
        mask = numeric_mask(x)
        clean = torch.where(mask, x[..., :6], 0.)
        e = self.embedding(clean.flatten(1)).reshape(len(x), 22, 6, -1)
        e = e * mask[..., None]
        return self.projection(torch.cat((e.flatten(2), x[..., 6:]), 2))


class MaskedFTBlock(nn.Module):
    """官方FT参数/FFN，本地只适配padding和token-to-token读出。"""
    def __init__(self):
        super().__init__()
        self.backbone = rtdl.FTTransformerBackbone(d_out=None, n_blocks=1,
            d_block=128, attention_n_heads=4, attention_dropout=.1,
            ffn_d_hidden_multiplier=4/3, ffn_dropout=.1, residual_dropout=0.)

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 128:
            raise ValueError('FT adapter requires N,T,128')
        present = (torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
                   if src_key_padding_mask is None else ~src_key_padding_mask)
        if present.shape != x.shape[:2] or present.dtype != torch.bool:
            raise ValueError('invalid token padding mask')
        safe = present.clone(); safe[~safe.any(1), 0] = True
        # 未观测内容不应通过投影或残差影响有效token。
        x = torch.where(present[..., None], x, 0.)
        for block in self.backbone.blocks:
            source = block['attention_normalization'](x) if 'attention_normalization' in block else x
            att = block['attention']
            def heads(v):
                return v.reshape(len(v), v.shape[1], 4, 32).transpose(1, 2)
            q, k, v = (heads(layer(source)) for layer in (att.W_q, att.W_k, att.W_v))
            dropout = att.dropout.p if att.dropout is not None and self.training else 0.
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=safe[:, None, None, :], dropout_p=dropout)
            y = y.transpose(1, 2).reshape_as(x)
            if att.W_out is not None:
                y = att.W_out(y)
            x = x + block['attention_residual_dropout'](y)
            x = x + block['ffn_residual_dropout'](block['ffn'](block['ffn_normalization'](x)))
            x = block['output'](x)
        return torch.where(present[..., None], x, 0.)


class TabMFusion(nn.Module):
    """保留774维分支接口；临时mean让旧编码器加回扩展，成员风险不取mean参数。"""
    def __init__(self):
        super().__init__()
        self.block = tabm.TabM.make(n_num_features=774, cat_cardinalities=[],
            d_out=128, n_blocks=2, d_block=128, k=16, dropout=.1)
        self.last_members: torch.Tensor | None = None

    def forward(self, x):
        self.last_members = self.block(x, None)
        return self.last_members.mean(1)


class SurgeryModel(nn.Module):
    def __init__(self, baseline, seed: int, arm: str, bins: list, text_width: int = 768):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        torch.manual_seed(seed)
        original = CompactAnchoredModel(copy.deepcopy(baseline), text_width)
        self.baseline, self.delta_head = original.baseline, original.delta_head
        detail = DetailEncoder(original.encoder)
        core = detail.base
        if arm == 'fusion_resnet':
            core.fusion = rtdl.ResNet(d_in=774, d_out=128, n_blocks=2, d_block=128,
                d_hidden=None, d_hidden_multiplier=2., dropout1=.1, dropout2=.1)
        elif arm in ('lab_ple', 'lab_periodic'):
            core.lab = NumericLabEncoder(arm.removeprefix('lab_'), bins)
        elif arm == 'fusion_tabm':
            core.fusion = TabMFusion()
        if arm.startswith('interaction_'):
            detail = OriginalInteractionEncoder(detail, interaction=True)
            if arm == 'interaction_ft':
                detail.attention = MaskedFTBlock()
        self.encoder = detail
        self.arm = arm
        self.patient_shape = True
        torch.manual_seed(seed)

    @property
    def detail(self):
        return self.encoder.original if isinstance(self.encoder, OriginalInteractionEncoder) else self.encoder

    def train(self, mode=True):
        super().train(mode); self.baseline.eval()
        return self

    def forward(self, history, batch):
        incoming = batch if isinstance(self.encoder, OriginalInteractionEncoder) else batch.base
        if self.arm != 'fusion_tabm':
            # 单成员直接使用原计算图，避免等价索引/reshape的反向数值路径漂移。
            return AnchoredModel.forward(self, history, incoming)
        with torch.no_grad():
            base = self.baseline(history)
        h, _ = self.encoder(incoming)
        if self.arm == 'fusion_tabm':
            members = self.detail.base.fusion.last_members
            # 原扩展加法逐成员相同；只是借旧forward复用，不提前丢掉集成。
            h = members + (h - members.mean(1))[:, None]
        delta = self.delta_head(h).reshape(*h.shape[:-1], 2, 2)
        if h.ndim == 3:
            amplitude = base.log_amplitude[:, None]
        else:
            amplitude = base.log_amplitude
        return RiskOutput((amplitude + delta[..., 0]).clamp(-10., 6.),
            F.softplus(self.baseline.raw_shape.detach() + delta[..., 1]) + .1, delta)


def member_nll(out: RiskOutput, times, events):
    if out.log_amplitude.ndim == 2:
        return competing_nll(out, times, events)
    members = out.log_amplitude.shape[1]
    flat = RiskOutput(out.log_amplitude.flatten(0, 1), out.shape.flatten(0, 1))
    return competing_nll(flat, times.repeat_interleave(members), events.repeat_interleave(members))


def probability_curve(a: np.ndarray, k: np.ndarray, *, device='cuda'):
    if a.ndim == 2:
        return fast_cif(a, k, device=device)
    n, members, _ = a.shape
    curve = fast_cif(a.reshape(-1, 2), k.reshape(-1, 2), device=device)
    return curve.reshape(n, members, 366, 3).mean(1)


def shared_initial_errors(model: SurgeryModel, control: SurgeryModel) -> dict:
    a, b = model.detail.state_dict(), control.detail.state_dict()
    ignore = ('base.fusion.',) if model.arm in ('fusion_resnet', 'fusion_tabm') else (
        ('base.lab.',) if model.arm in ('lab_ple', 'lab_periodic') else ())
    keys = [k for k in b if not k.startswith(ignore)]
    if any(k not in a or not torch.equal(a[k], b[k]) for k in keys):
        raise ValueError('unchanged V6 module initialization differs')
    if any(not torch.equal(v, control.delta_head.state_dict()[k]) for k, v in model.delta_head.state_dict().items()):
        raise ValueError('original shared correction head changed')
    if any(torch.count_nonzero(p) for p in model.delta_head[-1].parameters()):
        raise ValueError('correction head is not zero initialized')
    return dict(unchanged_detail_tensors=len(keys), maximum_error=0., correction_head_error=0.)
