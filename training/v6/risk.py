"""患者相关双原因 Weibull 联合似然与 CIF 数值积分。"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch


@dataclass(frozen=True)
class RiskOutput:
    log_amplitude: torch.Tensor  # [N,2]，不是传统 log(scale)
    shape: torch.Tensor          # [N,2]
    delta: torch.Tensor | None = None  # [N,2,2]，每原因幅度/raw shape修正


def competing_nll(output: RiskOutput, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    n = len(times)
    if (times.shape != (n,) or events.shape != (n,) or events.dtype != torch.long
            or output.log_amplitude.shape != (n, 2) or output.shape.shape != (n, 2) or n == 0):
        raise ValueError("invalid labels/parameters shape or event dtype")
    if (not torch.isfinite(times).all() or (times < 0).any() or (times > 365).any()
            or (events < 0).any() or (events > 2).any()
            or not torch.isfinite(output.log_amplitude).all()
            or not torch.isfinite(output.shape).all() or (output.shape <= 0).any()):
        raise ValueError("invalid time/event/risk parameter")
    # 沿用 V5 对记录精度为同日事件的最小时间约定，不据此重写标签。
    scaled = times.clamp_min(1e-3).unsqueeze(-1) / 365.
    a, k = output.log_amplitude, output.shape
    cumulative = torch.exp(a) * scaled.pow(k)
    log_hazard = a + torch.log(k) - math.log(365.) + (k - 1) * torch.log(scaled)
    selected = log_hazard.gather(1, (events - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (cumulative.sum(1) - torch.where(events > 0, selected, 0.)).mean()


def predict_cif(log_amplitude: np.ndarray, shape: np.ndarray, days: np.ndarray,
                *, step_days: float = .25) -> np.ndarray:
    """[N,T,3] 为 S/F1/F2；步内按累计风险增量分配事件质量，须验证步长收敛。"""
    a, k, query = (np.asarray(x, dtype=np.float64) for x in (log_amplitude, shape, days))
    if (a.ndim != 2 or a.shape[1] != 2 or not len(a) or k.shape not in ((2,), a.shape)
            or query.ndim != 1 or not len(query) or query[0] != 0
            or np.any(np.diff(query) <= 0) or query[-1] > 365
            or not math.isfinite(step_days) or step_days <= 0
            or not all(np.isfinite(x).all() for x in (a, k, query)) or (k <= 0).any()):
        raise ValueError("invalid CIF parameters/grid")
    amplitude = np.exp(a)
    if not np.isfinite(amplitude).all():
        raise ValueError("nonfinite amplitude")
    k = np.broadcast_to(k, a.shape)
    survival, cif = np.ones(len(a)), np.zeros_like(a)
    result = np.empty((len(a), len(query), 3))
    result[:, 0] = np.column_stack((survival, cif))
    previous_h = np.zeros_like(a)
    # 每个要求时间点必须是精确网格边界，不对最终 CIF 随意插值。
    for q in range(1, len(query)):
        grid = np.linspace(query[q - 1], query[q], max(1, math.ceil((query[q] - query[q - 1]) / step_days)) + 1)
        if query[q - 1] == 0:
            # k<1 的 hazard 在零点可奇异；首个时间步细化，而非直接计算 h(0)。
            grid = np.unique(np.r_[grid, np.geomspace(1e-10, min(step_days, query[q]), 80)])
        for t in grid[1:]:
            cumulative = amplitude * (t / 365.) ** k
            increment = np.maximum(0, cumulative - previous_h)
            total = increment.sum(1)
            share = np.divide(increment, total[:, None], out=np.zeros_like(increment), where=total[:, None] > 0)
            cif += survival[:, None] * (-np.expm1(-total))[:, None] * share
            survival *= np.exp(-total)
            previous_h = cumulative
        result[:, q] = np.column_stack((survival, cif))
    return result
