"""保留父计算路径，仅校正纵向当前住院token的背景接入。"""
import torch
from torch.nn import functional as F
from training.v6.unified_longitudinal import LongitudinalModel
from training.v6.risk import RiskOutput


def visit_tokens(current, past, current_context, past_context, gap, *, symmetric):
    """背景都为同一读取器输出；不额外计算/相加过去风险。"""
    historical = past + past_context + gap
    now = current + current_context if symmetric else current
    return torch.cat((now[:, None], historical), 1)


class SymmetricContextModel(LongitudinalModel):
    def __init__(self, fresh, condition):
        if condition not in ('asymmetric', 'symmetric'):
            raise ValueError('unknown context condition')
        super().__init__(fresh, 'history_unified')
        self.condition = condition

    def forward(self, batch):
        if self.condition == 'asymmetric':
            return super().forward(batch)
        compact, legacy, labs, eos, context, current_index, past_index, present, gaps = batch
        hidden = self.visits(compact, legacy, labs, eos)
        current = hidden[current_index]
        delta = self.original.delta_head(current).reshape(-1, 2, 2)
        seq = visit_tokens(current, hidden[past_index], self.past_context(context[current_index]),
            self.past_context(context[past_index]),
            self.gap(torch.log1p(gaps.clamp_min(0))[:, :, None] / 6.), symmetric=True)
        seq = torch.cat((seq[:, :1], torch.where(present[:, :, None], seq[:, 1:], 0.)), 1)
        valid = torch.cat((torch.ones(len(current), 1, dtype=torch.bool, device=current.device), present), 1)
        read = self.temporal(seq, src_key_padding_mask=~valid)[:, 0]
        delta = delta + self.history_head(read).reshape(-1, 2, 2) * present.any(1)[:, None, None]
        with torch.no_grad():
            base = self.original.baseline(context[current_index])
        return RiskOutput((base.log_amplitude + delta[:, :, 0]).clamp(-10., 6.),
            F.softplus(self.original.baseline.raw_shape.detach() + delta[:, :, 1]) + .1, delta)
