"""显式batch上下文的融合前方案注入；原有效临床层联合学习。"""
import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from training.v6.model import masked_pool
from training.v6.risk import RiskOutput

ARMS = ('availability', 'content')
ACTIVE_PREFIXES = (
    'encoder.base.lab.', 'encoder.base.lab_ids.', 'encoder.base.lab_score.',
    'encoder.base.med.', 'encoder.base.microbiology.', 'encoder.base.support.',
    *(f'encoder.base.branch_norms.{i}.' for i in range(1, 5)),
    'encoder.base.field_ids.', 'encoder.base.text_score.',
    'encoder.base.text_projection.', 'encoder.base.fusion.',
    'encoder.extra_labs.', 'delta_head.',
)


@dataclass(frozen=True)
class RegimenBatch:
    original: object
    features: torch.Tensor
    available: torch.Tensor


class JointFusion(nn.Module):
    def __init__(self, teacher, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('unknown joint regimen arm')
        self.arm = arm
        self.teacher = copy.deepcopy(teacher)
        for name, p in self.teacher.named_parameters():
            p.requires_grad_(name.startswith(ACTIVE_PREFIXES))
        # 不让新模块初始化耗用训练dropout的全局随机流。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.regimen = nn.Linear(35, 128, bias=False)
        nn.init.zeros_(self.regimen.weight)

    @property
    def baseline(self):
        return self.teacher.baseline

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        return self

    def injection(self, features, available):
        n = len(features)
        if (features.shape != (n, 34) or available.shape != (n,)
                or available.dtype != torch.bool or not torch.isfinite(features).all()):
            raise ValueError('explicit regimen34 and boolean chapter mask required')
        x = features if self.arm == 'content' else torch.zeros_like(features)
        x = torch.cat((x, available[:, None].to(x.dtype)), 1)
        return self.regimen(x) * available[:, None]

    def clinical(self, batch, injected):
        """沿用原compact的分支顺序和算子；只改第一仿射后的一次加法。"""
        detail = self.teacher.encoder
        core = detail.base
        compact = batch.base
        b = compact.clinical
        n = len(b.current_values)
        current = torch.where(b.current_masks, b.current_values, 0.)
        observed = b.lab_axis_inputs[:, :, 6] > 0
        labs = torch.where(observed[:, :, None], b.lab_axis_inputs, 0.)
        s = masked_pool(core.current(torch.stack((current, b.current_masks.to(current.dtype)), -1))
                        + core.current_ids(torch.arange(14, device=current.device)),
                        b.current_masks, core.current_score)
        l = masked_pool(core.lab(labs) + core.lab_ids(torch.arange(22, device=labs.device)),
                        observed, core.lab_score)
        med = compact.med_features
        med_present = med[:, :, 2:].bool().any(dim=(1, 2))
        m = core.med(med.flatten(1))
        extra, extra_present = [], []
        for values, masks, module in (
                (compact.micro_values, compact.micro_masks, core.microbiology),
                (compact.support_values, compact.support_masks, core.support)):
            extra.append(module(torch.cat((values, masks.to(values.dtype)), 1)))
            extra_present.append(masks.any(1))
        present = [b.current_masks.any(1), observed.any(1), med_present, *extra_present]
        branches = [norm(x) * p[:, None] for norm, x, p in
                    zip(core.branch_norms, (s, l, m, *extra), present, strict=True)]
        text, text_present = core._text(b, n, mask_pmh=False)
        presence = torch.stack((*present, text_present), 1).to(s.dtype)
        x = torch.cat((*branches, text, presence), 1)
        h = core.fusion[0](x) + injected
        for layer in list(core.fusion)[1:]:
            h = layer(h)
        additions = []
        for values, width, module in ((batch.labs, 90, detail.extra_labs),
                                      (batch.medications, 192, detail.extra_meds)):
            if values.shape != (n, width) or not torch.isfinite(values).all():
                raise ValueError('invalid inherited detail batch')
            additions.append(module(values) * values.ne(0).any(1)[:, None])
        return h + additions[0] + additions[1]

    def forward(self, history, batch):
        if not isinstance(batch, RegimenBatch):
            raise TypeError('RegimenBatch required; no implicit context')
        with torch.no_grad():
            base = self.baseline(history)
        hidden = self.clinical(batch.original.base, self.injection(batch.features, batch.available))
        delta = self.teacher.delta_head(hidden).reshape(-1, 2, 2)
        return RiskOutput((base.log_amplitude + delta[:, :, 0]).clamp(-10., 6.),
                          F.softplus(self.baseline.raw_shape.detach() + delta[:, :, 1]) + .1, delta)
