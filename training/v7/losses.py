"""连续时间竞争 NLL + 明确定义的软标签 Focal + 不同患者负对。"""
import torch
from torch.nn import functional as F

from training.v6.curve_score import daily_cif
from training.v6.risk import competing_nll

HORIZONS = (7, 14, 30, 60, 90, 180, 365)


def soft_focal(probability, target, *, gamma=2., alpha=.75, smoothing=.05):
    if probability.shape != target.shape or not torch.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('finite equal-shaped binary probabilities required')
    if ((target != 0) & (target != 1)).any() or not 0 <= smoothing < 1 or not 0 < alpha < 1 or gamma < 0:
        raise ValueError('invalid focal targets/settings')
    p = probability.clamp(1e-7, 1 - 1e-7)
    y = (1 - smoothing) * target.to(p.dtype) + smoothing / 2
    return (-alpha * y * (1 - p).pow(gamma) * p.log()
            -(1 - alpha) * (1 - y) * p.pow(gamma) * torch.log1p(-p)).mean()


def contrastive(reps, masks, subject_ids, *, margin=.5):
    anchor = reps['text']
    if subject_ids.shape != (len(anchor),) or margin <= 0:
        raise ValueError('patient ids/margin mismatch')
    positive, negative = [], []
    positive_count = negative_count = 0
    for source in ('drug', 'lab', 'rad'):
        left, right = masks['text'], masks[source]
        matched = left & right
        if matched.any():
            positive.append((anchor[matched] - reps[source][matched]).square().sum(1))
            positive_count += int(matched.sum())
        allowed = left[:, None] & right[None, :] & (subject_ids[:, None] != subject_ids[None, :])
        if allowed.any():
            # 只计算合法 pair 距离；零距离也有有限梯度。
            ii, jj = allowed.nonzero(as_tuple=True)
            distance = torch.linalg.vector_norm(anchor[ii] - reps[source][jj], dim=1)
            negative.append(F.relu(margin - distance).square())
            negative_count += len(ii)
    zero = sum(v.sum() * 0 for v in reps.values())
    pos = torch.cat(positive).mean() if positive else zero
    neg = torch.cat(negative).mean() if negative else zero
    return pos + neg, dict(positive_pairs=positive_count, negative_pairs=negative_count)


def objective(out, times, events, reps, masks, subject_ids, *, recommended):
    nll = competing_nll(out, times, events)
    penalty = 1e-4 * out.delta.square().mean()
    focal = contrast = nll * 0
    pairs = dict(positive_pairs=0, negative_pairs=0)
    if recommended:
        curve = daily_cif(out)
        days = torch.as_tensor(HORIZONS, device=times.device)
        causes = torch.tensor([1, 2], device=times.device)
        target = (events[:, None, None] == causes[None, None, :]) & (times[:, None, None] <= days[None, :, None])
        focal = soft_focal(curve[:, days, 1:], target)
        contrast, pairs = contrastive(reps, masks, subject_ids)
    total = nll + penalty + focal + .1 * contrast
    return total, dict(nll=nll, delta_penalty=penalty, focal=focal, contrast=contrast, **pairs)
