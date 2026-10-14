"""原V6教师冻结，仅读取过去住院临床内容；零初始化有界修正。"""
import torch
from torch import nn
from training.v6.risk import RiskOutput

ARMS=('current_only','times_only','past_last','past_mean','past_attention')


class HistoryReader(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError('unknown history reader')
        self.arm=arm
        self.projection=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,64),nn.GELU())
        self.time=nn.Linear(1,64)
        self.attention=nn.MultiheadAttention(64,4,dropout=.2,batch_first=True)
        self.head=nn.Sequential(nn.Linear(131,32),nn.GELU(),nn.Dropout(.2),nn.Linear(32,4))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,current,past,gaps,present,amplitude,shape):
        if (current.ndim!=2 or current.shape[1]!=128 or past.shape!=(len(current),present.shape[1],128) or
                gaps.shape!=present.shape or present.dtype!=torch.bool or amplitude.shape!=(len(current),2) or
                shape.shape!=amplitude.shape or not present.shape[1]):
            raise ValueError('invalid historical clinical interface')
        has=present.any(1);count=present.sum(1).clamp_min(1)
        time_value=torch.log1p(gaps.clamp_min(0))/6.
        clean=torch.where(present[:,:,None],past,0.)
        if self.arm=='times_only':clean=torch.zeros_like(clean)
        tokens=self.projection(clean)+self.time(time_value[:,:,None])
        tokens=torch.where(present[:,:,None],tokens,0.)
        query=self.projection(current)
        if self.arm=='current_only':pooled=torch.zeros_like(query)
        elif self.arm=='past_last':
            minimum=torch.where(present,gaps,torch.inf).min(1).values
            latest=present&(gaps==minimum[:,None])
            pooled=(tokens*latest[:,:,None]).sum(1)/latest.sum(1).clamp_min(1)[:,None]
        elif self.arm=='past_mean':pooled=tokens.sum(1)/count[:,None]
        else:
            safe=present.clone();safe[~has,0]=True
            pooled=self.attention(query[:,None],tokens,tokens,key_padding_mask=~safe,need_weights=False)[0][:,0]
            pooled=torch.where(has[:,None],pooled,0.)
        minimum=torch.where(present,time_value,torch.inf).min(1).values
        timing=torch.stack((torch.log1p(present.sum(1).float())/5.,torch.where(has,minimum,0.),
            (time_value*present).sum(1)/count),1)
        if self.arm=='current_only':timing=torch.zeros_like(timing)
        raw=self.head(torch.cat((query,pooled,timing),1)).reshape(-1,2,2)
        delta=torch.stack((.25*torch.tanh(raw[:,:,0]),.1*torch.tanh(raw[:,:,1])),2)
        delta=delta*has[:,None,None]
        return RiskOutput(amplitude+delta[:,:,0],shape*torch.exp(delta[:,:,1]),delta)
