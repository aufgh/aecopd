"""World A本库记录终点评价：无提前删失权重，不冒充完整院外随访。"""
from __future__ import annotations
import numpy as np
from training.baselines.rank_metrics import auc_rank
from training.v5.competing_concordance import cif_concordance

GRID = np.arange(366, dtype=np.float64)
REPORT_DAYS = (7, 14, 30, 60, 90, 180, 365)


def validate(times, events, risk):
    if (times.ndim != 1 or events.shape != times.shape or risk.shape != (len(times), 366)
            or not len(times) or not np.isfinite(times).all() or np.any((times < 0) | (times > 365))
            or not np.isin(events, (0, 1, 2)).all() or np.any((events == 0) & (times != 365))
            or not np.isfinite(risk).all() or np.any((risk < 0) | (risk > 1))):
        raise ValueError("invalid World A outcomes/risk; early censoring needs another evaluation contract")


def outcome_matrix(times, events, cause=1):
    return ((events[:, None] == cause) & (times[:, None] <= GRID)).astype(np.float64)


def metrics(times, events, risk, *, cause=1, ordering=False):
    times, events, risk = (np.asarray(x) for x in (times, events, risk))
    validate(times, events, risk)
    y = outcome_matrix(times, events, cause)
    brier = np.mean((risk - y) ** 2, axis=0)
    cases = y.sum(0).astype(int)
    auc = np.full(366, np.nan)
    for day in range(7, 366):
        if min(cases[day], len(times) - cases[day]) >= 20:
            auc[day] = auc_rank(risk[:, day], y[:, day].astype(np.int8))
    valid_auc = np.isfinite(auc[7:])
    result = {"admissions": len(times), "target_events": int((events == cause).sum()),
              "integrated_brier_7_365": float(np.trapezoid(brier[7:], GRID[7:]) / 358),
              "mean_auc_7_365": float(np.mean(auc[7:][valid_auc])) if valid_auc.any() else None,
              "valid_auc_days": int(valid_auc.sum()), "cause": cause,
              "points": {str(d): {"auc": float(auc[d]) if np.isfinite(auc[d]) else None,
                                    "brier": float(brier[d]), "cases": int(cases[d]),
                                    "controls_including_competing_events": int(len(times) - cases[d]),
                                    "mean_predicted": float(risk[:, d].mean()),
                                    "observed_world_a_fraction": float(y[:, d].mean())} for d in REPORT_DAYS},
              "brier_curve": brier, "auc_curve": auc, "case_curve": cases}
    if ordering:
        result["project_cif_concordance"] = {}
        for horizon in (90, 365):
            try:
                result["project_cif_concordance"][str(horizon)] = cif_concordance(times, events, risk[:, horizon], cause=cause, horizon=horizon)
            except ValueError as error:
                result["project_cif_concordance"][str(horizon)] = {"c_index": None, "reason": str(error)}
    return result


def public_metrics(detail):
    return {k: v for k, v in detail.items() if not k.endswith('_curve')}


def paired_interval(subjects, times, events, full, comparator, *, replicates=200, seed=42):
    """同一次患者抽样用于两模型；保留重复住院权重；CI是条件于已拟合模型的探索区间。"""
    validate(times, events, full); validate(times, events, comparator)
    if len(subjects) != len(times) or replicates < 20:
        raise ValueError("invalid cluster/bootstrap contract")
    y = outcome_matrix(times, events)
    delta = np.trapezoid(((full - y)**2 - (comparator - y)**2)[:, 7:], GRID[7:], axis=1) / 358
    people, inverse = np.unique(subjects, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(people))]
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, 2))
    for r in range(replicates):
        ix = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        samples[r, 0] = delta[ix].mean()
        samples[r, 1] = auc_rank(full[ix, 90], y[ix, 90].astype(np.int8)) - auc_rank(comparator[ix, 90], y[ix, 90].astype(np.int8))
    def interval(x):
        values = x[np.isfinite(x)]
        return np.quantile(values, [.025, .975]).tolist() if len(values) else [None, None]
    return {"definition": "candidate minus comparator", "replicates": replicates, "patients": len(people),
            "integrated_brier_difference": float(delta.mean()), "integrated_brier_95ci": interval(samples[:, 0]),
            "auc90_difference": float(auc_rank(full[:, 90], y[:, 90].astype(np.int8)) - auc_rank(comparator[:, 90], y[:, 90].astype(np.int8))),
            "auc90_95ci": interval(samples[:, 1]), "valid_auc90_replicates": int(np.isfinite(samples[:, 1]).sum()),
            "limits": "single training seed; no multiplicity adjustment; patient clustering does not remove observation bias"}


def calibration_bins(times, events, risk, *, day=90, cause=1, bins=10):
    y = ((events == cause) & (times <= day)).astype(float)
    # 同分分数不强行拆到不同组；基于实际预测分位点。
    edges = np.unique(np.quantile(risk[:, day], np.linspace(0, 1, bins + 1)))
    group = np.searchsorted(edges[1:-1], risk[:, day], side='right')
    return [{"n": int((group == g).sum()), "predicted": float(risk[group == g, day].mean()),
             "observed": float(y[group == g].mean())} for g in range(max(1, len(edges) - 1)) if (group == g).any()]
