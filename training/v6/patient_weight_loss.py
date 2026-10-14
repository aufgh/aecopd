"""仅训练损失的患者权重；不改变模型输入、标签或评价人群。"""
import math
import numpy as np
import torch
from training.v6.risk import RiskOutput,competing_nll


def training_weights(subjects,power):
    subjects=np.asarray(subjects)
    if subjects.ndim!=1 or not len(subjects) or power not in (0.,.5,1.):
        raise ValueError('nonempty training subjects and fixed weight power required')
    _,inverse,count=np.unique(subjects,return_inverse=True,return_counts=True)
    weights=count[inverse].astype(np.float64)**(-power)
    # 全训练折均值1；不按每个minibatch重新归一化，避免改写目标。
    return (weights/weights.mean()).astype(np.float32)


def per_record_nll(out:RiskOutput,times,events):
    # 复用原校验；新向量式与原均值/梯度有显式回归对照。
    competing_nll(out,times,events)
    scaled=times.clamp_min(1e-3)[:,None]/365.
    a,k=out.log_amplitude,out.shape
    cumulative=torch.exp(a)*scaled.pow(k)
    log_hazard=a+torch.log(k)-math.log(365.)+(k-1)*torch.log(scaled)
    chosen=log_hazard.gather(1,(events-1).clamp_min(0)[:,None]).squeeze(1)
    return cumulative.sum(1)-torch.where(events>0,chosen,0.)


def weighted_nll(out,times,events,weights):
    if weights.shape!=times.shape or not torch.isfinite(weights).all() or (weights<=0).any():
        raise ValueError('positive globally normalized per-record weights required')
    return (per_record_nll(out,times,events)*weights).mean()
