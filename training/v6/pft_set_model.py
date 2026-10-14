"""背景PFT不同值集合的分层读出；原完整V6风险不变起步。"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.risk import RiskOutput

ARMS=('availability','content')


@dataclass
class PFTBatch:
    frozen:object
    values:torch.Tensor
    concepts:torch.Tensor
    metadata:torch.Tensor
    owners:torch.Tensor
    aggregate:torch.Tensor
    observed:torch.Tensor


def segment_pool(tokens,owners,size,scorer):
    out=tokens.new_zeros(size,tokens.shape[-1])
    if not len(tokens):return out
    scores=scorer(tokens).squeeze(-1)
    maximum=scores.new_full((size,),-torch.inf);maximum.scatter_reduce_(0,owners,scores,reduce='amax',include_self=True)
    weights=torch.exp(scores-maximum[owners]);total=scores.new_zeros(size).scatter_add_(0,owners,weights)
    out.index_add_(0,owners,(weights/total[owners])[:,None]*tokens)
    return out


class PFTCorrection(nn.Module):
    def __init__(self,arm,seed=42):
        super().__init__()
        if arm not in ARMS:raise ValueError('unknown PFT arm')
        self.arm=arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.concept_embedding=nn.Embedding(6,8)
            self.value_projection=nn.Sequential(nn.Linear(1,8),nn.GELU(),nn.LayerNorm(8))
            self.metadata_projection=nn.Linear(3,8)
            self.value_score=nn.Linear(8,1);self.concept_score=nn.Linear(8,1)
            self.head=nn.Sequential(nn.Linear(49,16),nn.GELU(),nn.Linear(16,4))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,b):
        f=b.frozen;n=len(f.hidden);m=len(b.values)
        if b.values.shape!=(m,) or b.concepts.shape!=(m,) or b.metadata.shape!=(m,3) or b.owners.shape!=(m,) or b.aggregate.shape!=(n,9):
            raise ValueError('PFT interface shape mismatch')
        if b.owners.dtype!=torch.long or b.concepts.dtype!=torch.long or (b.owners<0).any() or (b.owners>=n).any() or (b.concepts<0).any() or (b.concepts>=6).any():
            raise ValueError('PFT concept/owner invalid')
        values=torch.zeros_like(b.values) if self.arm=='availability' else b.values
        tokens=self.concept_embedding(b.concepts)+self.value_projection(values[:,None])+self.metadata_projection(b.metadata)
        # 同概念的多个不同值先学习表示读出，再跨概念读出，不让记录多的概念占更多池化槽位。
        group=b.owners*6+b.concepts;within=segment_pool(tokens,group,n*6,self.value_score)
        present=torch.zeros(n*6,dtype=torch.bool,device=b.values.device);present[group]=True
        slots=torch.nonzero(present,as_tuple=False).squeeze(1)
        pooled=segment_pool(within[slots],slots//6,n,self.concept_score)
        raw=self.head(torch.cat((f.hidden,pooled,b.aggregate),1))*b.observed[:,None]
        raw=raw.reshape(n,2,2);delta=torch.stack((raw[:,:,0],.25*torch.tanh(raw[:,:,1])),2);original=f.raw.reshape(n,2,2)
        return RiskOutput((f.prior+original[:,:,0]+delta[:,:,0]).clamp(-10,6),F.softplus(f.shape_raw+original[:,:,1]+delta[:,:,1])+.1,delta)
