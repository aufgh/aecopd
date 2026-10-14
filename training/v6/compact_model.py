"""V6 紧凑多来源编码器；显式消费新接口，旧模型及检查点不改。"""
from dataclasses import replace
import torch
from torch import nn
from training.v6.compact_context_data import CompactContextBatch
from training.v6.model import AnchoredModel, ClinicalEncoder, masked_pool

ARMS = ('full', 'without_medications', 'without_microbiology', 'without_support', 'without_labs')


def mask_group(batch: CompactContextBatch, arm: str) -> CompactContextBatch:
    """屏蔽整个组及其来源/有效性；保留全部网络槽位和参数。"""
    if arm not in ARMS:
        raise ValueError('unknown compact ablation')
    if arm == 'without_labs':
        return replace(batch, clinical=replace(batch.clinical, lab_axis_inputs=torch.zeros_like(batch.clinical.lab_axis_inputs)))
    if arm == 'without_medications':
        return replace(batch, med_features=torch.zeros_like(batch.med_features), clinical=replace(batch.clinical,
                       med_exposure=torch.zeros_like(batch.clinical.med_exposure),
                       med_source_mask=torch.zeros_like(batch.clinical.med_source_mask)))
    if arm == 'without_microbiology':
        return replace(batch, micro_values=torch.zeros_like(batch.micro_values), micro_masks=torch.zeros_like(batch.micro_masks))
    if arm == 'without_support':
        return replace(batch, support_values=torch.zeros_like(batch.support_values), support_masks=torch.zeros_like(batch.support_masks))
    return batch


class CompactClinicalEncoder(nn.Module):
    # 复用已验证的字段内池化；文本投影依赖维度，必须最后初始化。
    _text = ClinicalEncoder._text

    def __init__(self, text_width: int, *, width: int = 128):
        super().__init__()
        if text_width < 1 or width < 1:
            raise ValueError('invalid encoder width')
        self.width, self.text_width = width, text_width
        self.current = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, width))
        self.current_ids = nn.Embedding(14, width)
        self.current_score = nn.Linear(width, 1)
        self.lab = nn.Sequential(nn.Linear(9, 32), nn.GELU(), nn.Linear(32, width))
        self.lab_ids = nn.Embedding(22, width)
        self.lab_score = nn.Linear(width, 1)
        self.med = nn.Sequential(nn.Linear(28, 32), nn.GELU(), nn.Linear(32, width))
        self.microbiology = nn.Sequential(nn.Linear(10, 32), nn.GELU(), nn.Linear(32, width))
        self.support = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, width))
        self.branch_norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(5))
        self.field_ids = nn.Embedding(4, width)
        self.text_score = nn.Linear(width, 1)
        self.fusion = nn.Sequential(nn.Linear(6 * width + 6, width), nn.GELU(), nn.Dropout(.1),
                                    nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))
        self.text_projection = nn.Sequential(nn.LayerNorm(text_width), nn.Linear(text_width, width),
                                              nn.GELU(), nn.LayerNorm(width))

    def forward(self, batch: CompactContextBatch, *, mask_pmh=False):
        if not isinstance(batch, CompactContextBatch):
            raise TypeError('explicit CompactContextBatch required; legacy input is not accepted')
        b = batch.clinical
        n = len(b.current_values)
        if (not n or b.current_values.shape != (n, 14) or b.current_masks.shape != (n, 14)
                or b.current_masks.dtype != torch.bool or b.lab_axis_inputs.shape != (n, 22, 9)
                or batch.med_features.shape != (n, 7, 4)):
            raise ValueError('invalid compact clinical contract')
        med = batch.med_features
        if not torch.isfinite(med).all() or not ((med == 0) | (med == 1)).all():
            raise ValueError('medication channels must be finite binary')
        if ((med[:, :, 0] > med[:, :, 2]) | (med[:, :, 1] > med[:, :, 3])
                | ((med[:, :, 0] * med[:, :, 1]) > 0)).any():
            raise ValueError('invalid medication source / order-only contract')
        current = torch.where(b.current_masks, b.current_values, 0.)
        observed = b.lab_axis_inputs[:, :, 6] > 0
        labs = torch.where(observed[:, :, None], b.lab_axis_inputs, 0.)
        if not torch.isfinite(current).all() or not torch.isfinite(labs).all():
            raise ValueError('nonfinite active clinical input')
        s = masked_pool(self.current(torch.stack((current, b.current_masks.to(current.dtype)), -1))
                        + self.current_ids(torch.arange(14, device=current.device)), b.current_masks, self.current_score)
        l = masked_pool(self.lab(labs) + self.lab_ids(torch.arange(22, device=labs.device)), observed, self.lab_score)
        med_present = med[:, :, 2:].bool().any(dim=(1, 2))
        m = self.med(med.flatten(1))
        extra, extra_present = [], []
        for values, masks, count, module in ((batch.micro_values, batch.micro_masks, 5, self.microbiology),
                                            (batch.support_values, batch.support_masks, 3, self.support)):
            if (values.shape != (n, count) or masks.shape != values.shape or masks.dtype != torch.bool
                    or not torch.isfinite(values).all() or not ((values == 0) | (values == 1)).all()
                    or ((values > 0) & ~masks).any()):
                raise ValueError('invalid compact observed binary input')
            extra.append(module(torch.cat((values, masks.to(values.dtype)), 1)))
            extra_present.append(masks.any(1))
        present = [b.current_masks.any(1), observed.any(1), med_present, *extra_present]
        branches = [norm(x) * p[:, None] for norm, x, p in
                    zip(self.branch_norms, (s, l, m, *extra), present, strict=True)]
        text, text_present = self._text(b, n, mask_pmh=mask_pmh)
        presence = torch.stack((*present, text_present), 1).to(s.dtype)
        hidden = self.fusion(torch.cat((*branches, text, presence), 1))
        return hidden, dict(zip(('structured', 'lab', 'medication', 'microbiology', 'support'), branches, strict=True),
                            text=text, presence=presence)


class CompactAnchoredModel(AnchoredModel):
    def __init__(self, baseline, text_width, *, width=128, patient_shape=True):
        # 不先建立再抛弃旧encoder，避免无关参数和隐蔽RNG消耗。
        nn.Module.__init__(self)
        self.baseline = baseline.requires_grad_(False).eval()
        self.delta_head = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 4))
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        self.encoder = CompactClinicalEncoder(text_width, width=width)
        self.patient_shape = patient_shape
