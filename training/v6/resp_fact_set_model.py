"""冻结原完整风险的低容量typed事实集合修正；来源对照真屏蔽。"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.pft_set_model import segment_pool
from training.v6.risk import RiskOutput

ARMS=('availability','content')

@dataclass
class FactBatch:
    frozen:object
    values:torch.Tensor
    concepts:torch.Tensor
    roles:torch.Tensor
    polarities:torch.Tensor
    units:torch.Tensor
    conditions:torch.Tensor
    numeric_present:torch.Tensor
    numeric_valid:torch.Tensor
    owners:torch.Tensor
    aggregate:torch.Tensor
    observed:torch.Tensor

class FactCorrection(nn.Module):
    def __init__(self,arm,seed=42):
        super().__init__()
        if arm not in ARMS:raise ValueError('unknown fact arm')
        self.arm=arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embeddings=nn.ModuleDict({k:nn.Embedding(n,8) for k,n in
                [('concepts',8),('roles',3),('polarities',2),('units',3),('conditions',4)]})
            self.value_projection=nn.Sequential(nn.Linear(3,8),nn.GELU(),nn.LayerNorm(8))
            self.value_score=nn.Linear(8,1);self.concept_score=nn.Linear(8,1)
            self.head=nn.Sequential(nn.Linear(43,16),nn.GELU(),nn.Linear(16,4))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,b):
        f=b.frozen;n=len(f.hidden);m=len(b.values)
        if f.hidden.shape!=(n,32) or b.aggregate.shape!=(n,3) or b.observed.shape!=(n,):raise ValueError('fact admission shape differs')
        if b.owners.shape!=(m,) or b.owners.dtype!=torch.long or (b.owners<0).any() or (b.owners>=n).any():raise ValueError('fact owners invalid')
        for name,limit in [('concepts',8),('roles',3),('polarities',2),('units',3),('conditions',4)]:
            v=getattr(b,name)
            if v.shape!=(m,) or v.dtype!=torch.long or (v<0).any() or (v>=limit).any():raise ValueError('invalid typed fact '+name)
        if b.numeric_present.shape!=(m,) or b.numeric_valid.shape!=(m,) or b.numeric_present.dtype!=torch.bool or b.numeric_valid.dtype!=torch.bool:
            raise ValueError('fact numeric masks differ')
        if (b.numeric_valid&~b.numeric_present).any() or not torch.isfinite(b.values).all():raise ValueError('invalid numeric validity/value')
        if not torch.equal(b.observed,b.aggregate.bool().any(1)):raise ValueError('source role gate differs')
        if self.arm=='availability':
            # 不读取typed属性/成员数量的表征：只保留共同三角色来源信息。
            pooled=f.hidden.new_zeros(n,8)
        else:
            tokens=sum(layer(getattr(b,key)) for key,layer in self.embeddings.items())
            numeric=torch.stack((b.values*b.numeric_valid,b.numeric_present.float(),b.numeric_valid.float()),1)
            tokens=tokens+self.value_projection(numeric)
            group=b.owners*8+b.concepts;within=segment_pool(tokens,group,n*8,self.value_score)
            present=torch.zeros(n*8,dtype=torch.bool,device=b.values.device);present[group]=True
            slots=torch.nonzero(present,as_tuple=False).squeeze(1)
            pooled=segment_pool(within[slots],slots//8,n,self.concept_score)
        raw=self.head(torch.cat((f.hidden,pooled,b.aggregate),1))*b.observed[:,None]
        raw=raw.reshape(n,2,2);delta=torch.stack((raw[:,:,0],.25*torch.tanh(raw[:,:,1])),2);original=f.raw.reshape(n,2,2)
        return RiskOutput((f.prior+original[:,:,0]+delta[:,:,0]).clamp(-10,6),F.softplus(f.shape_raw+original[:,:,1]+delta[:,:,1])+.1,delta)
