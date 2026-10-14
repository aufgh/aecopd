"""独立正则化适配器：不改原V6/ABC源文件，连续时间概率保持。"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from training.v6.deep_interaction import DeepModel
from training.v6.module_surgery import SurgeryModel
from training.v6.risk import RiskOutput


@dataclass(frozen=True)
class Trial:
    name: str
    architecture: str = 'control'
    dropout: float = .1
    latent_dropout: float = 0.
    text_dropout: float = 0.
    weight_decay: float = 1e-4
    lr: float = 1e-4
    delta_penalty: float = 1e-4
    consistency: float = 0.
    brier: float = 0.
    depth_drop: float = 0.


TRIALS = (
    Trial('v6_control'),
    Trial('v6_dropout30', dropout=.3, latent_dropout=.2),
    Trial('v6_dropout50', dropout=.5, latent_dropout=.3),
    Trial('v6_text30', text_dropout=.3),
    Trial('v6_wd01', weight_decay=.01),
    Trial('v6_wd10', weight_decay=.1),
    Trial('v6_shrink01', delta_penalty=.01),
    Trial('v6_shrink10', delta_penalty=.1),
    Trial('v6_low_lr', lr=3e-5),
    Trial('v6_drop_wd', dropout=.3, latent_dropout=.2, text_dropout=.2, weight_decay=.01),
    Trial('v6_consistency', dropout=.3, latent_dropout=.2, consistency=1.),
    Trial('v6_brier2', brier=2.),
    Trial('v6_brier10', brier=10.),
    Trial('v6_drop_brier', dropout=.3, latent_dropout=.2, brier=5.),
    Trial('single_control', architecture='late_single'),
    Trial('single_drop_wd', architecture='late_single', dropout=.3, latent_dropout=.2, weight_decay=.01),
    Trial('A_control', architecture='A_deep'),
    Trial('A_dropout30', architecture='A_deep', dropout=.3, latent_dropout=.2),
    Trial('A_wd10', architecture='A_deep', weight_decay=.1),
    Trial('A_stochastic', architecture='A_deep', dropout=.2, depth_drop=.3),
    Trial('A_consistency', architecture='A_deep', dropout=.2, consistency=1., weight_decay=.01),
    Trial('A_brier', architecture='A_deep', dropout=.2, brier=5.),
    Trial('B_control', architecture='B_shared'),
    Trial('B_drop_wd', architecture='B_shared', dropout=.3, latent_dropout=.2, weight_decay=.01),
    Trial('B_stochastic', architecture='B_shared', dropout=.2, depth_drop=.3),
    Trial('B_consistency', architecture='B_shared', dropout=.2, consistency=1.),
)


class StochasticResidual(nn.Module):
    """每个样本随机绕过整个块；eval用完整块，不引入行号时序。"""
    def __init__(self, module, rate):
        super().__init__(); self.module, self.rate = module, rate
        if not 0 <= rate < 1:
            raise ValueError('invalid depth dropout')

    def forward(self, x, *args, **kwargs):
        update = self.module(x, *args, **kwargs)
        if self.training and self.rate:
            keep = (torch.rand(len(x), 1, 1, device=x.device) >= self.rate).to(x.dtype)
            return x + (update-x)*keep/(1-self.rate)
        return update


class OutputDropout(nn.Module):
    def __init__(self, encoder, probability):
        super().__init__(); self.encoder=encoder; self.dropout=nn.Dropout(probability)

    def forward(self, *args, **kwargs):
        h, diagnostics = self.encoder(*args, **kwargs)
        return self.dropout(h), diagnostics


def make_model(baseline, seed, trial, bins, *, text_width=768, width=256):
    if trial.architecture=='control':
        model=SurgeryModel(baseline,seed,'control',bins,text_width=text_width)
        core=model.encoder.base
        core.fusion[2].p=trial.dropout
        if trial.text_dropout:
            core.text_projection=nn.Sequential(core.text_projection,nn.Dropout(trial.text_dropout))
    else:
        model=DeepModel(baseline,seed,trial.architecture,text_width=text_width,width=width)
        # nn.MultiheadAttention使用数值dropout属性而不是独立nn.Dropout。
        for module in model.encoder.modules():
            if isinstance(module,nn.Dropout):module.p=trial.dropout
            if isinstance(module,nn.MultiheadAttention):module.dropout=trial.dropout
        if trial.depth_drop:
            model.encoder.blocks=nn.ModuleList(StochasticResidual(m,trial.depth_drop) for m in model.encoder.blocks)
    if trial.latent_dropout:
        model.encoder=OutputDropout(model.encoder,trial.latent_dropout)
    model.arm=trial.name
    torch.manual_seed(seed)
    return model


def tensor_cif(output: RiskOutput, days=(14,30,60,90,180,365), step=.5):
    """可微CIF；正时间网格避免零点奇异，双原因共享事件质量。"""
    if step<=0 or 365/step!=int(365/step) or any(d<=0 or d>365 for d in days):
        raise ValueError('invalid quadrature grid')
    device=output.shape.device
    grid=torch.arange(step,365+step/2,step,device=device,dtype=torch.float64)
    small=torch.exp(torch.linspace(-23.025850929940457,torch.log(grid[0]).item(),80,device=device,dtype=torch.float64))
    grid=torch.unique(torch.cat((small,grid,torch.tensor(days,device=device,dtype=torch.float64))),sorted=True)
    a,k=output.log_amplitude.double(),output.shape.double()
    cumulative=torch.exp(a[:,None]+k[:,None]*torch.log(grid[None,:,None]/365))
    previous=torch.cat((torch.zeros_like(cumulative[:,:1]),cumulative[:,:-1]),1)
    increment=(cumulative-previous).clamp_min(0)
    total=increment.sum(2,keepdim=True)
    survival_before=torch.exp(-previous.sum(2,keepdim=True))
    mass=survival_before*(-torch.expm1(-total))*increment/total.clamp_min(1e-300)
    cif=mass.cumsum(1)
    selected=torch.searchsorted(grid,torch.tensor(days,device=device,dtype=grid.dtype))
    survival=torch.exp(-cumulative[:,selected].sum(2,keepdim=True))
    return torch.cat((survival,cif[:,selected]),2)


def probability_consistency(left,right):
    p,q=tensor_cif(left),tensor_cif(right)
    p=p.clamp_min(1e-12);q=q.clamp_min(1e-12)
    p=p/p.sum(2,keepdim=True);q=q/q.sum(2,keepdim=True)
    return (.5*((p-q)*(p.log()-q.log())).sum(2)).mean()


def auxiliary_brier(output,times,events):
    days=torch.tensor((14,30,60,90,180,365),device=times.device)
    p=tensor_cif(output)[:,:,1:]
    y=((events[:,None,None]==torch.tensor((1,2),device=events.device)) &
       (times[:,None,None]<=days[None,:,None])).to(p.dtype)
    return (p-y).square().mean()
