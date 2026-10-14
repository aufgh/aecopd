"""与原V6每日CIF同网格的可微全年评分；不修改正式预测实现。"""
import numpy as np
import torch
from training.v6.risk import RiskOutput


def daily_cif(out: RiskOutput):
    a,k=out.log_amplitude.double(),out.shape.double()
    if (a.ndim!=2 or a.shape[1]!=2 or k.shape!=a.shape or not len(a)
            or not torch.isfinite(a).all() or not torch.isfinite(k).all() or (k<=0).any()):
        raise ValueError('finite N by 2 amplitudes and positive shapes required')
    grid=np.unique(np.r_[np.arange(0,365.0001,.25),np.geomspace(1e-10,.25,80)])
    g=torch.as_tensor(grid[1:],dtype=torch.float64,device=a.device)
    # 不在t=0计算参数导数；正时间部分与原fast_cif逐操作一致。
    positive=a[:,None,:].exp()*(g[None,:,None]/365).pow(k[:,None,:])
    cumulative=torch.cat((torch.zeros_like(positive[:,:1]),positive),1)
    increments=torch.diff(cumulative,dim=1).clamp_min(0)
    total=increments.sum(2)
    survival=(-cumulative.sum(2)).exp()
    mass=survival[:,:-1,None]*(-torch.expm1(-total))[:,:,None]*increments/total[:,:,None].clamp_min(1e-300)
    cif=torch.cat((torch.zeros_like(mass[:,:1]),mass.cumsum(1)),1)
    indices=torch.as_tensor(np.searchsorted(grid,np.arange(366)),device=a.device)
    return torch.cat((survival[:,:,None],cif),2)[:,indices]


def curve_scores(curve,times,events):
    """每记录7–365日三状态/原因评分；仅适用WorldA行政观察合同。"""
    n=len(times)
    if (curve.shape!=(n,366,3) or not n or times.shape!=(n,) or events.shape!=(n,)
            or events.dtype!=torch.long or not torch.isfinite(times).all()
            or ((times<0)|(times>365)).any() or ((events<0)|(events>2)).any()
            or ((events==0)&(times!=365)).any() or not torch.isfinite(curve).all()):
        raise ValueError('WorldA labels/daily three-state curve required; no early censoring')
    days=torch.arange(7,366,device=times.device)
    occurred=(events[:,None]>0)&(times[:,None]<=days[None,:])
    state=torch.where(occurred,events[:,None],0)
    target=torch.nn.functional.one_hot(state,num_classes=3).to(curve.dtype)
    squared=(curve[:,7:]-target).square()
    integrated=torch.trapezoid(squared,dx=1.,dim=1)/358.
    return dict(readmission=integrated[:,1],death=integrated[:,2],
                joint=integrated.sum(1))
