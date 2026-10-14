"""数学原型：独立原因时钟，保留教师365天累计概率；尚非队列实验。"""
import math
import numpy as np
import torch
from training.v6.risk import RiskOutput


def anchor_dense_to_daily(raw,grid,daily):
    """按每天端点重标定区间分位，不交错拼接两种积分误差。"""
    raw,grid,daily=(np.asarray(x,dtype=np.float64) for x in (raw,grid,daily))
    n=len(raw)
    if raw.shape!=(n,len(grid),3) or daily.shape!=(n,366,3) or grid[0]!=0 or grid[-1]!=365:
        raise ValueError('invalid dense/daily anchoring shape')
    slots=np.searchsorted(grid,np.arange(366))
    if not np.array_equal(grid[slots],np.arange(366)):raise ValueError('integer anchors missing')
    if (not np.isfinite(raw).all() or not np.isfinite(daily).all() or
        np.any(np.diff(raw[:,:,1:],axis=1)<-1e-10) or np.any(np.diff(daily[:,:,1:],axis=1)<-1e-10)):
        raise ValueError('nonmonotonic source CIF')
    lo=np.minimum(np.floor(grid).astype(int),364);hi=lo+1
    base=raw[:,slots,1:]
    denominator=base[:,hi]-base[:,lo]
    fallback=np.broadcast_to((grid-lo)[None,:,None],(n,len(grid),2)).copy()
    fraction=np.divide(raw[:,:,1:]-base[:,lo],denominator,out=fallback,where=denominator>1e-18)
    fraction=np.clip(fraction,0,1)
    result=np.empty_like(raw)
    result[:,:,1:]=daily[:,lo,1:]+fraction*(daily[:,hi,1:]-daily[:,lo,1:])
    result[:,:,0]=1-result[:,:,1:].sum(2)
    result=np.clip(result,0,1);result[:,slots]=daily
    if np.any(np.diff(result[:,:,1:],axis=1)<-1e-10):raise ValueError('anchoring did not preserve monotonicity')
    np.testing.assert_allclose(result.sum(2),1,atol=1e-9,rtol=0)
    return result


def clock_nll(prior:RiskOutput,log_clock:torch.Tensor,times:torch.Tensor,events:torch.Tensor):
    n=len(times)
    if (prior.log_amplitude.shape!=(n,2) or prior.shape.shape!=(n,2) or log_clock.shape!=(n,2) or
        events.shape!=(n,) or events.dtype!=torch.long or not n or
        not torch.isfinite(times).all() or (times<0).any() or (times>365).any() or
        ((events<0)|(events>2)).any() or ((events==0)&(times!=365)).any() or
        not torch.isfinite(log_clock).all() or not torch.isfinite(prior.log_amplitude).all() or
        not torch.isfinite(prior.shape).all() or (prior.shape<=0).any()):
        raise ValueError('clock prototype requires WorldA365 parameters/labels; early censoring unsupported')
    g=torch.exp(log_clock)
    s=times.clamp_min(1e-3)/365.;logs=torch.log(s)
    # 每个原因e的时钟q_e(t)=365*(t/365)^g_e：旧两危险率均在q_e求值。
    cumulative=(torch.exp(prior.log_amplitude[:,None,:])*s[:,None,None].pow(g[:,:,None]*prior.shape[:,None,:])).sum(2)
    log_density=prior.log_amplitude+torch.log(prior.shape)+log_clock-math.log(365.)+(g*prior.shape-1)*logs[:,None]-cumulative
    chosen=log_density.gather(1,(events-1).clamp_min(0)[:,None]).squeeze(1)
    # 所有无事件记录在365行政边界；该点S完全保留教师，不改结局。
    return torch.where(events>0,-chosen,torch.exp(prior.log_amplitude).sum(1)).mean()


def clock_curve(teacher_curve,teacher_grid,log_clock,days):
    """冻结教师密集CIF插值；正式使用仍需验证网格收敛与原模型精确重放。"""
    teacher_curve,grid,log_clock,days=(np.asarray(x,dtype=np.float64) for x in (teacher_curve,teacher_grid,log_clock,days))
    n=len(teacher_curve)
    if (teacher_curve.shape!=(n,len(grid),3) or log_clock.shape!=(n,2) or not n or
        grid.ndim!=1 or not len(grid) or grid[0]!=0 or grid[-1]!=365 or np.any(np.diff(grid)<=0) or
        days.ndim!=1 or not len(days) or days[0]!=0 or np.any(np.diff(days)<=0) or days[-1]>365 or
        not all(np.isfinite(x).all() for x in (teacher_curve,grid,log_clock,days)) or
        np.any((teacher_curve<0)|(teacher_curve>1)) or np.any(np.diff(teacher_curve[:,:,1:],axis=1)<-1e-10)):
        raise ValueError('invalid frozen CIF/grid/clock')
    np.testing.assert_allclose(teacher_curve.sum(2),1,atol=1e-9,rtol=0)
    result=np.empty((n,len(days),3))
    for i in range(n):
        for e in (0,1):
            query=365.*(days/365.)**np.exp(log_clock[i,e])
            # 零初始化精确取原查询，不用浮点幂引入极小坐标偏移。
            if log_clock[i,e]==0:query=days
            query=np.clip(query,0,365)
            result[i,:,e+1]=np.interp(query,grid,teacher_curve[i,:,e+1])
    result[:,:,0]=1-result[:,:,1:].sum(2)
    result[:,0]=teacher_curve[:,0]
    if days[-1]==365:result[:,-1]=teacher_curve[:,-1]
    if result.min()<-1e-9:raise ValueError('clock violated competing probability budget')
    result=np.clip(result,0,1)
    return result
