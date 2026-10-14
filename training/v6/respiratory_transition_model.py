"""冻结原V6风险及32维状态，新96维呼吸事实作有限修正。"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.risk import RiskOutput

ARMS=('global','availability','timing','content')
AVAIL=[j*21+i for j in range(4) for i in (18,19)]+list(range(90,96))
TIMING=AVAIL+[j*21+i for j in range(4) for i in (5,6,8,14,15,17)]


@dataclass
class RespBatch:
    frozen: object
    features: torch.Tensor
    observed: torch.Tensor


def view(x,arm):
    if x.ndim!=2 or x.shape[1]!=96 or not torch.isfinite(x).all():raise ValueError('expected finite96 respiratory input')
    if arm=='content':return x
    if arm not in ('availability','timing'):raise ValueError('unknown feature view')
    keep=AVAIL if arm=='availability' else TIMING
    out=torch.zeros_like(x);out[:,keep]=x[:,keep]
    return out


class RespiratoryCorrection(nn.Module):
    def __init__(self,arm,seed=42):
        super().__init__()
        if arm not in ARMS:raise ValueError('invalid arm')
        self.arm=arm
        if arm=='global':self.raw=nn.Parameter(torch.zeros(4))
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed);self.head=nn.Sequential(nn.Linear(128,16),nn.GELU(),nn.Linear(16,4))
            nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,b):
        f=b.frozen
        if b.features.shape!=(len(f.hidden),96) or b.observed.shape!=(len(f.hidden),):raise ValueError('respiratory batch/index mismatch')
        if self.arm=='global':new=self.raw.expand(len(f.hidden),4)
        else:
            x=torch.cat((f.hidden,view(b.features,self.arm)),1)
            new=self.head(x)*b.observed[:,None]
        new=new.reshape(-1,2,2)
        delta=torch.stack((new[:,:,0],.25*torch.tanh(new[:,:,1])),2)
        original=f.raw.reshape(-1,2,2)
        return RiskOutput((f.prior+original[:,:,0]+delta[:,:,0]).clamp(-10,6),
            F.softplus(f.shape_raw+original[:,:,1]+delta[:,:,1])+.1,delta)
