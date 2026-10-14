"""训练折拟合的章节平衡冻结向量后处理，不读取结局拟合变换。"""
import numpy as np
import torch


def fit_geometry(vectors, indices, weights, fields=4):
    if vectors.ndim!=2 or indices.shape!=(len(vectors),) or weights.shape!=(len(vectors),):
        raise ValueError('invalid geometry rows')
    means,matrices,diagnostics=[],[],[]
    for field in range(fields):
        use=indices==field
        if int(use.sum())<2:raise ValueError('insufficient training field vectors')
        x=vectors[use].double();w=weights[use].double()
        if not torch.isfinite(x).all() or not torch.isfinite(w).all() or (w<=0).any():raise ValueError('nonfinite geometry')
        w=w/w.sum();mean=(x*w[:,None]).sum(0);center=x-mean
        covariance=center.T@(center*w[:,None]);covariance=(covariance+covariance.T)*.5
        eigen,rotation=torch.linalg.eigh(covariance)
        eigen=eigen.clamp_min(0);average=eigen.mean()
        if average<=0:raise ValueError('constant field representation')
        # 固定10%球形收缩，避免小特征值噪声无限放大；不是原论文逐式复现。
        shrunk=.9*eigen+.1*average
        transform=(rotation*shrunk.rsqrt()[None,:])@rotation.T
        means.append(mean.float());matrices.append(transform.float())
        diagnostics.append(dict(field=field,training_rows=len(x),mean_norm=float(mean.norm()),
            covariance_participation_rank=float(eigen.sum().square()/eigen.square().sum()),
            top1_variance_fraction=float(eigen[-1]/eigen.sum()),top10_variance_fraction=float(eigen[-10:].sum()/eigen.sum()),
            shrinkage=.1,maximum_gain=float(shrunk.rsqrt().max())))
    return torch.stack(means),torch.stack(matrices),diagnostics


def transform_rows(vectors, field, means, matrices=None):
    result=torch.empty_like(vectors)
    for f in range(len(means)):
        keep=field==f
        if not keep.any():continue
        x=vectors[keep]-means[f]
        if matrices is not None:x=x@matrices[f]
        result[keep]=torch.nn.functional.normalize(x,p=2,dim=1,eps=1e-8)
    if not torch.isfinite(result).all():raise ValueError('nonfinite transformed vectors')
    return result
