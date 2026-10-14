"""V6住院编码器替换：晚池化、深层/共享层/多潜在槽位交互。"""
from __future__ import annotations

import copy
import torch
from torch import nn
from training.v6.compact_model import CompactAnchoredModel
from training.v6.model import AnchoredModel, masked_pool
from training.v6.original_interaction import ExpandedBatch

ARMS = ('control', 'late_gated', 'late_single', 'A_deep', 'B_shared', 'C_latent')
CANDIDATES = ARMS[1:]
WIDTH = 256


def block(width):
    return nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=.1,
                                     activation='gelu', batch_first=True, norm_first=True)


class ClinicalTokens(nn.Module):
    """与原V6相同信息；不在交互前聚合检验、药效或文本块。"""
    def __init__(self, text_width=768, width=WIDTH):
        super().__init__()
        self.width = width
        self.lab = nn.Sequential(nn.Linear(9, 32), nn.GELU(), nn.Linear(32, width))
        self.extra_regular = nn.Sequential(nn.Linear(10, 32), nn.GELU(), nn.Linear(32, width))
        self.extra_eos = nn.Sequential(nn.Linear(20, 32), nn.GELU(), nn.Linear(32, width))
        self.med = nn.Linear(4, width)
        self.micro = nn.Linear(2, width)
        self.support = nn.Linear(2, width)
        self.ids = nn.Embedding(44, width)  # 22+7+7+5+3，概念身份不是事件行号。
        self.types = nn.Embedding(6, width)
        self.fields = nn.Embedding(4, width)
        self.text = nn.Sequential(nn.LayerNorm(text_width), nn.Linear(text_width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, batch: ExpandedBatch, *, mask_pmh=False):
        d = batch.base
        c = d.base
        b = c.clinical
        n = len(b.current_values)
        # 诊断/去向留在H34；禁止未来扩展静默重新流入临床修正。
        if b.current_masks.any() or d.medications.ne(0).any():
            raise ValueError('requires frozen-H34 original V6 recipe; fine drugs inactive')
        obs = b.lab_axis_inputs[..., 6] > 0
        lab = self.lab(torch.where(obs[..., None], b.lab_axis_inputs, 0.))
        regular = d.labs[:, :50].reshape(n, 5, 10)
        eos = d.labs[:, 50:].reshape(n, 2, 20)
        extra_mask = torch.cat((regular[..., 5:].bool().any(2), eos[..., 6:12].bool().any(2)), 1)
        extra = torch.cat((self.extra_regular(torch.where(extra_mask[:, :5, None], regular, 0.)),
                           self.extra_eos(torch.where(extra_mask[:, 5:, None], eos, 0.))), 1)
        med_mask = c.med_features[..., 2:].bool().any(2)
        med = self.med(torch.where(med_mask[..., None], c.med_features, 0.))
        micro = self.micro(torch.stack((torch.where(c.micro_masks, c.micro_values, 0.), c.micro_masks.float()), -1))
        support = self.support(torch.stack((torch.where(c.support_masks, c.support_values, 0.), c.support_masks.float()), -1))
        pieces = (lab, extra, med, micro, support)
        masks = (obs, extra_mask, med_mask, c.micro_masks, c.support_masks)
        fixed = torch.cat(pieces, 1)
        fixed_types = torch.cat([torch.full((x.shape[1],), i, device=lab.device, dtype=torch.long)
                                 for i, x in enumerate(pieces)])
        fixed = fixed + self.ids(torch.arange(44, device=lab.device)) + self.types(fixed_types)
        present = torch.cat(masks, 1)
        index, vectors = b.snippet_index, b.snippet_vectors
        if mask_pmh:
            keep = index[:, 1] != 1
            index, vectors = index[keep], vectors[keep]
        length = max(1, int(torch.bincount(index[:, 0], minlength=n).max()) if len(index) else 1)
        dense = lab.new_zeros(n, length, self.width)
        text_mask = torch.zeros(n, length, device=lab.device, dtype=torch.bool)
        if len(index):
            if batch.text_slots is not None and not mask_pmh:
                slots = batch.text_slots
            else:
                slots = torch.zeros(len(index), device=lab.device, dtype=torch.long)
                for row in range(n):
                    where = torch.where(index[:, 0] == row)[0]
                    slots[where] = torch.arange(len(where), device=lab.device)
            dense[index[:, 0], slots] = self.text(vectors) + self.fields(index[:, 1]) + self.types.weight[5]
            text_mask[index[:, 0], slots] = True
        states = self.norm(torch.cat((fixed, dense), 1))
        present = torch.cat((present, text_mask), 1)
        return torch.where(present[..., None], states, 0.), present


class LatentRead(nn.Module):
    """潜在槽位向输入读取；不是完整Perceiver IO论文复现。"""
    def __init__(self, width):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.input_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=.1, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(.1), nn.Linear(2 * width, width))

    def forward(self, slots, inputs, safe):
        q, kv = self.query_norm(slots), self.input_norm(inputs)
        slots = slots + self.attention(q, kv, kv, key_padding_mask=~safe, need_weights=False)[0]
        return slots + self.ffn(self.norm(slots))


class DeepClinicalEncoder(nn.Module):
    def __init__(self, arm, *, text_width=768, width=WIDTH):
        super().__init__()
        if arm not in CANDIDATES:
            raise ValueError(arm)
        self.arm = arm
        # 通用模块均在各臂特有模块之前初始化，确保共同权重一致。
        self.tokens = ClinicalTokens(text_width, width)
        self.score = nn.Linear(width, 1)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.GELU(), nn.LayerNorm(128))
        first = block(width)
        if arm == 'A_deep':
            self.blocks = nn.ModuleList([first] + [block(width) for _ in range(5)])
        elif arm == 'C_latent':
            self.blocks = nn.ModuleList([first] + [block(width) for _ in range(3)])
            self.latents = nn.Parameter(torch.randn(16, width) * .02)
            self.cross = LatentRead(width)
        else:
            self.blocks = nn.ModuleList([first])
        if arm == 'late_gated':
            self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, batch, *, mask_pmh=False):
        states, present = self.tokens(batch, mask_pmh=mask_pmh)
        safe = present.clone()
        safe[~safe.any(1), 0] = True  # 空集合仅供数值计算；不冒充真实观测。
        input_norm = states.detach().square().sum(2).mean().sqrt()
        if self.arm == 'C_latent':
            x = self.latents[None].expand(len(states), -1, -1)
            x = self.cross(x, states, safe)
            for i, layer in enumerate(self.blocks):
                x = layer(x)
                if i == 1:
                    x = self.cross(x, states, safe)
            active = present.any(1)[:, None].expand(-1, x.shape[1])
        else:
            x, active = states, present
            rounds = 6 if self.arm == 'B_shared' else len(self.blocks)
            for i in range(rounds):
                layer = self.blocks[0] if self.arm == 'B_shared' else self.blocks[i]
                proposal = layer(x, src_key_padding_mask=~safe)
                x = x + torch.tanh(self.gate) * (proposal-x) if self.arm == 'late_gated' else proposal
                x = torch.where(present[..., None], x, 0.)
        pooled = masked_pool(x, active, self.score)
        hidden = self.output(pooled) * present.any(1)[:, None]
        return hidden, dict(token_count=present.sum(1), input_rms=input_norm,
            output_rms=x.detach().square().sum(2).mean().sqrt(),
            gate=torch.tanh(self.gate).detach() if hasattr(self, 'gate') else None)


class DeepModel(AnchoredModel):
    def __init__(self, baseline, seed, arm, *, text_width=768, width=WIDTH):
        nn.Module.__init__(self)
        torch.manual_seed(seed)
        original = CompactAnchoredModel(copy.deepcopy(baseline), text_width)
        self.baseline, self.delta_head = original.baseline, original.delta_head
        self.encoder = DeepClinicalEncoder(arm, text_width=text_width, width=width)
        self.patient_shape, self.arm = True, arm
        torch.manual_seed(seed)


def gradient_groups(model):
    result = {}
    for name, module in (('text', model.encoder.tokens.text), ('lab', model.encoder.tokens.lab),
                         ('medication', model.encoder.tokens.med), ('interaction', model.encoder.blocks),
                         ('readout', model.encoder.output), ('risk_head', model.delta_head)):
        result[name] = sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None)**.5
    return result
