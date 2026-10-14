"""低维连续时间竞争风险模型：AECOPD再住院与先于再住院的死亡。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CauseSpecificCox:
    coefficients: np.ndarray
    event_times: np.ndarray
    baseline_hazard_increments: np.ndarray
    converged: bool


@dataclass(frozen=True)
class CompetingCoxModel:
    readmission: CauseSpecificCox
    death: CauseSpecificCox
    horizon_days: float
    l2: float


def _validated_inputs(
    features: np.ndarray | Sequence[Sequence[float]],
    times: np.ndarray | Sequence[float],
    events: np.ndarray | Sequence[int],
    horizon_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(events, dtype=np.int64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must be a nonempty [admissions, features] matrix")
    if t.shape != (x.shape[0],) or e.shape != (x.shape[0],):
        raise ValueError("times and events must align with admissions")
    if not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError("features and times must be finite")
    if not np.isin(e, [0, 1, 2]).all():
        raise ValueError("event types must be 0, 1 or 2")
    if not np.isfinite(horizon_days) or horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if np.any(t < 0) or np.any(t > horizon_days):
        raise ValueError("event times must be within the study horizon")
    if np.any((e == 0) & (t != horizon_days)):
        raise ValueError("World A requires no-event admissions to end at the study horizon")
    return x, np.maximum(t, 1e-6), e


def _partial_likelihood(
    x: np.ndarray,
    t: np.ndarray,
    is_event: np.ndarray,
    beta: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Breslow 并列事件偏似然、梯度与正定观察信息矩阵。"""

    linear = x @ beta
    risk = np.exp(np.clip(linear, -30.0, 30.0))
    risk_total = np.cumsum(risk[::-1])[::-1]
    weighted = risk[:, None] * x
    risk_first = np.cumsum(weighted[::-1], axis=0)[::-1]
    weighted_outer = weighted[:, :, None] * x[:, None, :]
    risk_second = np.cumsum(weighted_outer[::-1], axis=0)[::-1]

    event_times, counts = np.unique(t[is_event], return_counts=True)
    objective = -0.5 * l2 * float(beta @ beta)
    gradient = -l2 * beta.copy()
    information = l2 * np.eye(x.shape[1])
    for event_time, count in zip(event_times, counts):
        first = int(np.searchsorted(t, event_time, side="left"))
        at_time = is_event & (t == event_time)
        x_events = x[at_time]
        mean = risk_first[first] / risk_total[first]
        objective += float(np.sum(linear[at_time]) - count * np.log(risk_total[first]))
        gradient += np.sum(x_events, axis=0) - count * mean
        information += count * (
            risk_second[first] / risk_total[first] - np.outer(mean, mean)
        )
    return objective, gradient, information


def _partial_objective(
    x: np.ndarray,
    t: np.ndarray,
    is_event: np.ndarray,
    beta: np.ndarray,
    l2: float,
) -> float:
    """回溯搜索仅计算目标值，避免反复构造三维 Hessian 张量。"""

    linear = x @ beta
    risk = np.exp(np.clip(linear, -30.0, 30.0))
    risk_total = np.cumsum(risk[::-1])[::-1]
    event_times, counts = np.unique(t[is_event], return_counts=True)
    starts = np.searchsorted(t, event_times, side="left")
    return float(
        np.sum(linear[is_event])
        - counts @ np.log(risk_total[starts])
        - 0.5 * l2 * (beta @ beta)
    )


def _fit_one_cause(
    x: np.ndarray,
    t: np.ndarray,
    is_event: np.ndarray,
    *,
    l2: float,
    max_iter: int,
    tolerance: float,
) -> CauseSpecificCox:
    order = np.argsort(t, kind="stable")
    x, t, is_event = x[order], t[order], is_event[order]
    beta = np.zeros(x.shape[1], dtype=np.float64)
    converged = False
    for _ in range(max_iter):
        objective, gradient, information = _partial_likelihood(x, t, is_event, beta, l2)
        step = np.linalg.solve(information, gradient)
        scale = 1.0
        accepted = False
        for _ in range(30):
            candidate = beta + scale * step
            candidate_objective = _partial_objective(x, t, is_event, candidate, l2)
            if candidate_objective >= objective - 1e-9:
                beta = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            raise RuntimeError("penalized Cox line search failed")
        if np.max(np.abs(scale * step)) < tolerance:
            converged = True
            break

    risk = np.exp(np.clip(x @ beta, -30.0, 30.0))
    risk_total = np.cumsum(risk[::-1])[::-1]
    event_times, counts = np.unique(t[is_event], return_counts=True)
    increments = np.array(
        [
            count / risk_total[int(np.searchsorted(t, event_time, side="left"))]
            for event_time, count in zip(event_times, counts)
        ],
        dtype=np.float64,
    )
    return CauseSpecificCox(beta, event_times, increments, converged)


def fit_competing_cox(
    features: np.ndarray | Sequence[Sequence[float]],
    times: np.ndarray | Sequence[float],
    events: np.ndarray | Sequence[int],
    *,
    horizon_days: float = 365.0,
    l2: float = 10.0,
    max_iter: int = 40,
    tolerance: float = 1e-6,
) -> CompetingCoxModel:
    """拟合两个病因特异风险，不把竞争死亡当作普通阴性。"""

    x, t, e = _validated_inputs(features, times, events, horizon_days)
    if l2 <= 0 or max_iter <= 0 or tolerance <= 0:
        raise ValueError("l2, max_iter and tolerance must be positive")
    if not np.any(e == 1) or not np.any(e == 2):
        raise ValueError("both readmission and death events are required in training")
    readmission = _fit_one_cause(
        x, t, e == 1, l2=l2, max_iter=max_iter, tolerance=tolerance
    )
    death = _fit_one_cause(x, t, e == 2, l2=l2, max_iter=max_iter, tolerance=tolerance)
    return CompetingCoxModel(readmission, death, horizon_days, l2)


def predict_cumulative_incidence(
    model: CompetingCoxModel,
    features: np.ndarray | Sequence[Sequence[float]],
    days: np.ndarray | Sequence[float],
) -> np.ndarray:
    """返回 [住院, 时间, 无事件/AECOPD/死亡] 的一致累积概率曲线。"""

    x = np.asarray(features, dtype=np.float64)
    query = np.asarray(days, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != model.readmission.coefficients.size:
        raise ValueError("feature width does not match the fitted model")
    if query.ndim != 1 or query.size == 0 or not np.isfinite(query).all():
        raise ValueError("days must be a finite nonempty one-dimensional array")
    if np.any(np.diff(query) < 0) or query[0] < 0 or query[-1] > model.horizon_days:
        raise ValueError("days must be sorted and inside the study horizon")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")

    grid = np.union1d(model.readmission.event_times, model.death.event_times)
    hazard_a = np.zeros(grid.size, dtype=np.float64)
    hazard_d = np.zeros(grid.size, dtype=np.float64)
    hazard_a[np.searchsorted(grid, model.readmission.event_times)] = (
        model.readmission.baseline_hazard_increments
    )
    hazard_d[np.searchsorted(grid, model.death.event_times)] = (
        model.death.baseline_hazard_increments
    )
    relative_a = np.exp(np.clip(x @ model.readmission.coefficients, -30.0, 30.0))
    relative_d = np.exp(np.clip(x @ model.death.coefficients, -30.0, 30.0))
    survival = np.ones(x.shape[0], dtype=np.float64)
    cif_a = np.zeros(x.shape[0], dtype=np.float64)
    cif_d = np.zeros(x.shape[0], dtype=np.float64)
    result = np.empty((x.shape[0], query.size, 3), dtype=np.float64)
    next_query = 0
    for index, event_time in enumerate(grid):
        while next_query < query.size and query[next_query] < event_time:
            result[:, next_query, :] = np.column_stack((survival, cif_a, cif_d))
            next_query += 1
        if next_query == query.size:
            break
        increment_a = relative_a * hazard_a[index]
        increment_d = relative_d * hazard_d[index]
        increment_total = increment_a + increment_d
        event_probability = -np.expm1(-increment_total)
        share_a = np.divide(
            increment_a,
            increment_total,
            out=np.zeros_like(increment_total),
            where=increment_total > 0,
        )
        cif_a += survival * event_probability * share_a
        cif_d += survival * event_probability * (1.0 - share_a)
        survival *= np.exp(-increment_total)
    while next_query < query.size:
        result[:, next_query, :] = np.column_stack((survival, cif_a, cif_d))
        next_query += 1
    return result


def integrated_competing_brier(
    times: np.ndarray | Sequence[float],
    events: np.ndarray | Sequence[int],
    readmission_cif: np.ndarray,
    days: np.ndarray | Sequence[float],
) -> float:
    """世界观A下整段时间的再住院Brier积分；不假装有逐人删失时间。"""

    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(events, dtype=np.int64)
    grid = np.asarray(days, dtype=np.float64)
    risk = np.asarray(readmission_cif, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("days must be a strictly increasing grid")
    if risk.shape != (t.size, grid.size) or e.shape != t.shape:
        raise ValueError("risk curves, times and events must align")
    if np.any((e == 0) & (t < grid[-1])):
        raise ValueError("early censoring needs an IPCW estimator, not World A Brier")
    if not np.isfinite(risk).all() or np.any((risk < 0) | (risk > 1)):
        raise ValueError("readmission CIF must be a finite probability")
    observed = ((e[:, None] == 1) & (t[:, None] <= grid[None, :])).astype(float)
    brier_by_day = np.mean((risk - observed) ** 2, axis=0)
    return float(np.trapezoid(brier_by_day, grid) / (grid[-1] - grid[0]))
