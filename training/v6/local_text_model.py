"""同安全原句、同容量读取；保留原完整风险，仅学习文本增量。"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.risk import RiskOutput

ARMS=('availability','containing','local')


@dataclass
class LocalTextBatch:
    frozen:object
    vectors:torch.Tensor
    metadata:torch.Tensor
    owners:torch.Tensor
    aggregate:torch.Tensor
    observed:torch.Tensor
    local_vectors:torch.Tensor|None=None


class LocalTextCorrection(nn.Module):
    def __init__(self,arm,seed=42):
        super().__init__()
        if arm not in ARMS:raise ValueError('unknown text arm')
        self.arm=arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.projection=nn.Sequential(nn.Linear(768,16),nn.GELU(),nn.LayerNorm(16))
            self.metadata=nn.Linear(10,16);self.score=nn.Linear(16,1)
            self.head=nn.Sequential(nn.Linear(59,16),nn.GELU(),nn.Linear(16,4))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,b):
        f=b.frozen;n=len(f.hidden)
        if b.vectors.shape!=(len(b.owners),768) or b.metadata.shape!=(len(b.owners),10) or b.aggregate.shape!=(n,11):raise ValueError('text interface shape mismatch')
        if b.owners.dtype!=torch.long or (b.owners<0).any() or (b.owners>=n).any():raise ValueError('text owner index invalid')
        pooled=f.hidden.new_zeros(n,16)
        if len(b.owners):
            if self.arm=='local' and b.local_vectors is None:raise ValueError('explicit local vector view required')
            x=torch.zeros_like(b.vectors) if self.arm=='availability' else b.local_vectors if self.arm=='local' else b.vectors
            tokens=self.projection(x)+self.metadata(b.metadata);score=self.score(tokens).squeeze(-1)
            maximum=score.new_full((n,),-torch.inf);maximum.scatter_reduce_(0,b.owners,score,reduce='amax',include_self=True)
            w=torch.exp(score-maximum[b.owners]);total=score.new_zeros(n).scatter_add_(0,b.owners,w)
            w=w/total[b.owners];pooled.index_add_(0,b.owners,w[:,None]*tokens)
        raw=self.head(torch.cat((f.hidden,pooled,b.aggregate),1))*b.observed[:,None]
        raw=raw.reshape(n,2,2);delta=torch.stack((raw[:,:,0],.25*torch.tanh(raw[:,:,1])),2)
        original=f.raw.reshape(n,2,2)
        return RiskOutput((f.prior+original[:,:,0]+delta[:,:,0]).clamp(-10,6),F.softplus(f.shape_raw+original[:,:,1]+delta[:,:,1])+.1,delta)
