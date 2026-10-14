"""官方表格骨干；共享联合连续时间风险头，包源码不修改。"""
import math
import types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import tabm
import rtdl_revisiting_models as rtdl
import rtdl_num_embeddings as numeric
from training.v6.risk import RiskOutput, competing_nll
from training.v6.token_model import fast_cif

ARMS = ('mlp', 'resnet_ple', 'resnet_periodic', 'tabm_ple', 'ft_transformer')


def sdpa_forward(self, x_q, x_kv):
    """与官方Attention同一投影/缩放/dropout；只替换运算内核。"""
    q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
    if self.key_compression is not None:
        k = self.key_compression(k.transpose(1, 2)).transpose(1, 2)
        compression = self.key_compression if self.value_compression is None else self.value_compression
        v = compression(v.transpose(1, 2)).transpose(1, 2)
    def heads(x):
        return x.reshape(len(x), x.shape[1], self._n_heads, -1).transpose(1, 2)
    p = self.dropout.p if self.dropout is not None and self.training else 0.
    x = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), dropout_p=p)
    x = x.transpose(1, 2).reshape(len(x_q), x_q.shape[1], -1)
    return self.W_out(x) if self.W_out is not None else x


def initial_bias(events):
    fractions = np.array([(events == e).mean() for e in (1, 2)])
    total = fractions.sum()
    if not 0 < total < 1 or np.any(fractions <= 0):
        raise ValueError('both events and event-free training observations required')
    hazard = -np.log1p(-total) * fractions / total
    return np.r_[np.log(hazard), np.repeat(np.log(np.expm1(.9)), 2)].astype(np.float32)


class PublicModel(nn.Module):
    def __init__(self, arm, n_num, n_binary, bins, bias, *, sdpa=True):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.k = 16 if arm == 'tabm_ple' else 1
        self.embedding = None
        width = 96 if arm == 'ft_transformer' else 256
        if arm in ('resnet_ple', 'tabm_ple'):
            embedding = numeric.PiecewiseLinearEmbeddings(bins, 8, activation=True, version='B')
        elif arm == 'resnet_periodic':
            embedding = numeric.PeriodicEmbeddings(n_num, 8, n_frequencies=8,
                frequency_init_scale=.01, activation=True, lite=True)
        else:
            embedding = None
        if arm == 'mlp':
            self.backbone = rtdl.MLP(d_in=n_num+n_binary, d_out=None, n_blocks=3, d_block=width, dropout=.1)
        elif arm.startswith('resnet'):
            self.embedding = embedding
            self.backbone = rtdl.ResNet(d_in=n_num*8+n_binary, d_out=None, n_blocks=2,
                d_block=width, d_hidden=None, d_hidden_multiplier=2., dropout1=.1, dropout2=.1)
        elif arm == 'tabm_ple':
            self.backbone = tabm.TabM.make(n_num_features=n_num, cat_cardinalities=[2]*n_binary,
                d_out=None, num_embeddings=embedding, n_blocks=2, d_block=width, k=self.k, dropout=.1)
        else:
            self.backbone = rtdl.FTTransformer(n_cont_features=n_num, cat_cardinalities=[2]*n_binary,
                d_out=None, n_blocks=2, d_block=width, attention_n_heads=4, attention_dropout=.1,
                ffn_d_hidden_multiplier=4/3, ffn_dropout=.1, residual_dropout=0.)
            if sdpa:
                for block in self.backbone.backbone.blocks:
                    attention = block['attention']
                    attention.forward = types.MethodType(sdpa_forward, attention)
        self.head = tabm.LinearEnsemble(width, 4, k=self.k) if self.k > 1 else nn.Linear(width, 4)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.as_tensor(bias).expand_as(self.head.bias))

    def forward(self, x_num, x_binary):
        if self.arm in ('ft_transformer', 'tabm_ple'):
            h = self.backbone(x_num, x_binary.long() if x_binary.shape[1] else None)
        else:
            x = self.embedding(x_num).flatten(1) if self.embedding is not None else x_num
            h = self.backbone(torch.cat((x, x_binary), 1))
        # 混合精度只用于骨干；头保持FP32，避免不同骨干的bias相加路径造成初始风险差。
        with torch.autocast(h.device.type, enabled=False):
            raw = self.head(h.float())
        if raw.ndim == 2:
            raw = raw[:, None]
        return raw[..., :2].clamp(-10., 6.), F.softplus(raw[..., 2:]) + .1


def ensemble_nll(a, k, times, events):
    """事件标签复制到成员，平均成员损失而非平均参数的损失。"""
    members = a.shape[1]
    return competing_nll(RiskOutput(a.flatten(0, 1), k.flatten(0, 1)),
        times.repeat_interleave(members), events.repeat_interleave(members))


def ensemble_curve(a, k, *, device='cuda'):
    if a.shape != k.shape or a.ndim != 3 or a.shape[-1] != 2:
        raise ValueError('expected [patients, ensemble, 2]')
    curves = fast_cif(a.reshape(-1, 2), k.reshape(-1, 2), device=device)
    return curves.reshape(len(a), a.shape[1], 366, 3).mean(1)
