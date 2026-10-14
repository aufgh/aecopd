"""B2：经典结构化竞争风险模型（必需统计基线，纯 numpy 实现）。

设计对齐 docs/BASELINE_COMPARISON_PLAN.md B2。方法学参照：
- Austin PC, et al. Introduction to the Analysis of Survival Data in the
  Presence of Competing Risks. Circulation 2016 (PMID 26858290)。
- 校准评估参照 Austin PC, et al. Graphical calibration curves and the
  integrated calibration index (ICI) for competing risk models.
  Diagn Progn Res 2022 (PMID 35039069)。

核心思想：只用**预先定义的低维临床变量**（人口学、既往住院负担、住院次数、
住院间隔、住院时长、ICU、基线生命体征），拟合 cause-specific Cox（AECOPD 与
死亡各一个病因特异模型）；Aalen–Johansen 给出无模型的队列累计发生率描述。
不使用高维 Qwen 向量——目的是回答"复杂多模态输入是否优于常规临床变量"。

统计口径注意（写入报告，避免混用）：
- cause-specific Cox 建的是某一类事件的瞬时风险(hazard)；
- Fine–Gray 建的是子分布风险(subdistribution)，两者回答的问题不同；
- 本基线选 cause-specific，因为它与"给定仍在风险集时下一步发生什么"的
  条件区间语义更接近；如需累计发生率对照再叠加 Aalen–Johansen。

删失口径（2026-09-20 研究者确认）：**世界观 A**——出院后数据库再无记录=痊愈/阴性。
365 天内未观察到事件即在第 365 天行政删失（**非逐病人删失**），与主模型标签一致。
因此 Cox 只建模 0～365 天窗口；>365 天开放尾部无 hazard 信息，改由同一组协变量的
多项 logistic（仅在"365 天仍无事件"子集上训练）预测三分类尾部，退化时回退队列基率。

实现：Newton-Raphson 拟合 Cox 偏似然（Breslow 处理并列事件）→ 合并事件时间网格上
product-integral 求 CIF → 相邻边界差分得条件区间三分类概率 [N,3,3]、尾部 [N,3]。
标准化与所有拟合只用训练折；评估复用主项目 conditional_metrics（口径不漂移）。
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .shared_contract import (
    INTERVAL_UPPER_DAYS,
    BaselineEvalResult,
    SplitContract,
    evaluate_with_project_metrics,
)

# 预定义低维临床特征。lab 摘要暂不纳入（需从 numeric_summary 透视，另行扩展），
# 先用队列表自带的人口学/既往负担/住院/ICU/生命体征，保证基线可复现、口径固定。
CONTINUOUS_FEATURES: tuple[str, ...] = (
    "age_at_admission",
    "prior_aecopd_count",
    "prior_all_admission_count",
    "days_since_previous_aecopd",
    "hospital_los_hours",
    "icu_stay_count",
    "prior_bmi",
    "prior_sbp",
    "prior_dbp",
)
# 结构性缺失需带缺失指示列（首次住院无间隔；生命体征约三成缺）。
MISSING_INDICATOR_FEATURES: tuple[str, ...] = (
    "days_since_previous_aecopd",
    "prior_bmi",
    "prior_sbp",
    "prior_dbp",
)
BINARY_FEATURES: tuple[tuple[str, str], ...] = (
    ("gender", "M"),        # M=1, F=0
    ("entered_icu", "t"),   # t=1, f=0
    ("via_ed", "t"),        # t=1, f=0
)

INTERVAL_BOUNDARIES: tuple[float, float, float] = INTERVAL_UPPER_DAYS  # (30, 90, 365)


@dataclass(frozen=True)
class CoxBaselineConfig:
    """B2 配置。特征列表预先固定，避免高维/数据驱动特征选择污染对照。"""

    cohort_csv_gz_path: Path            # model_v3 cohort_labels_static（含标签与临床静态特征）
    split: SplitContract
    max_newton_iters: int = 50
    newton_tol: float = 1e-6
    ridge: float = 1e-6                 # Hessian 对角小岭，纯粹数值稳定
    tail_l2: float = 1.0                # 尾部多项 logistic 的 L2 强度
    tail_lr: float = 0.5
    tail_epochs: int = 500
    random_state: int = 2026
    # 世界观 A：固定 365 天行政删失，不使用逐病人随访终点。
    use_subject_specific_censoring: bool = False


# ----------------------------------------------------------------------------
# 数据组装
# ----------------------------------------------------------------------------
def _to_float(value: str | None) -> float:
    if value is None or value.strip() == "":
        return float("nan")
    return float(value)


@dataclass
class SurvivalFrame:
    """一个划分的生存分析数据框与对齐用键。"""

    hadm_ids: np.ndarray      # [N] int
    subject_ids: np.ndarray   # [N] int
    features: np.ndarray      # [N, P] 已标准化协变量
    duration: np.ndarray      # [N] 0~365，min(事件时间, 365)
    event_type: np.ndarray    # [N] 0 删失/无事件, 1 AECOPD, 2 死亡（窗口内）
    feature_names: tuple[str, ...]


@dataclass
class _StandardizerStats:
    means: dict[str, float]
    stds: dict[str, float]
    medians: dict[str, float]


def _read_cohort_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _fit_standardizer(train_rows: list[dict[str, str]]) -> _StandardizerStats:
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in CONTINUOUS_FEATURES:
        values = np.array([_to_float(r.get(name)) for r in train_rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        median = float(np.median(finite)) if finite.size else 0.0
        medians[name] = median
        imputed = np.where(np.isfinite(values), values, median)
        mean = float(np.mean(imputed))
        std = float(np.std(imputed))
        means[name] = mean
        stds[name] = std if std > 1e-8 else 1.0
    return _StandardizerStats(means=means, stds=stds, medians=medians)


def _build_feature_matrix(
    rows: list[dict[str, str]], stats: _StandardizerStats
) -> tuple[np.ndarray, tuple[str, ...]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    for name in CONTINUOUS_FEATURES:
        raw = np.array([_to_float(r.get(name)) for r in rows], dtype=np.float64)
        imputed = np.where(np.isfinite(raw), raw, stats.medians[name])
        columns.append((imputed - stats.means[name]) / stats.stds[name])
        names.append(name)
    for name in MISSING_INDICATOR_FEATURES:
        raw = np.array([_to_float(r.get(name)) for r in rows], dtype=np.float64)
        columns.append((~np.isfinite(raw)).astype(np.float64))
        names.append(f"{name}__missing")
    for name, positive in BINARY_FEATURES:
        columns.append(
            np.array([1.0 if (r.get(name) or "") == positive else 0.0 for r in rows], dtype=np.float64)
        )
        names.append(f"{name}__is_{positive}")
    return np.column_stack(columns), tuple(names)


def assemble_survival_frame(
    config: CoxBaselineConfig, split: str, stats: _StandardizerStats
) -> SurvivalFrame:
    """构造某划分的 (duration, event_type, covariates) 生存分析数据框（世界观 A）。

    duration = min(窗口内事件时间, 365)；event_type ∈ {0 无事件/删失, 1 AECOPD, 2 死亡}。
    event_type 与 time_days 直接取自队列表（与 ConditionalOutcomeStore 同源），
    保证与主模型标签逐 hadm 一致。连续变量用训练折统计标准化。
    """
    all_rows = _read_cohort_rows(config.cohort_csv_gz_path)
    rows = [r for r in all_rows if r["data_split"] == split]
    if not rows:
        raise ValueError(f"no rows for split={split}")
    features, names = _build_feature_matrix(rows, stats)
    hadm_ids = np.array([int(r["hadm_id"]) for r in rows], dtype=np.int64)
    subject_ids = np.array([int(r["subject_id"]) for r in rows], dtype=np.int64)
    event_type = np.array([int(r["event_type"]) for r in rows], dtype=np.int64)
    time_days = np.array([_to_float(r.get("time_days")) for r in rows], dtype=np.float64)
    # 世界观 A：event_type==0 在 365 天行政删失；事件行 duration=time_days。
    duration = np.where(event_type == 0, 365.0, time_days)
    duration = np.clip(duration, 1e-6, 365.0)
    return SurvivalFrame(
        hadm_ids=hadm_ids,
        subject_ids=subject_ids,
        features=features,
        duration=duration,
        event_type=event_type,
        feature_names=names,
    )


# ----------------------------------------------------------------------------
# cause-specific Cox（Newton-Raphson + Breslow）
# ----------------------------------------------------------------------------
@dataclass
class CauseSpecificCox:
    """单个病因的 Cox 系数与 Breslow 基线累计风险增量。"""

    beta: np.ndarray                 # [P]
    baseline_times: np.ndarray       # [K] 升序事件时间
    baseline_increments: np.ndarray  # [K] 每个事件时间的 dH0
    log_likelihood: float
    converged: bool


def _cox_partial_gradient_hessian(
    features: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    beta: np.ndarray,
    ridge: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Breslow 偏对数似然、梯度、（观测信息）Hessian。时间已按升序排列。"""
    n, p = features.shape
    # 线性预测子裁剪，避免 exp 溢出（不改变最优解，只保数值稳定）。
    theta = np.exp(np.clip(features @ beta, -30.0, 30.0))  # [n]
    # 风险集后缀和：R(t)={j: dur_j >= t}。升序排列 → 后缀和。
    s0 = np.cumsum(theta[::-1])[::-1]                     # [n]
    weighted = theta[:, None] * features                  # [n,p]
    s1 = np.cumsum(weighted[::-1], axis=0)[::-1]          # [n,p]
    outer = weighted[:, :, None] * features[:, None, :]   # [n,p,p]
    s2 = np.cumsum(outer[::-1], axis=0)[::-1]             # [n,p,p]

    log_lik = 0.0
    grad = np.zeros(p)
    hess = np.zeros((p, p))
    event_positions = np.nonzero(event == 1)[0]
    for i in event_positions:
        start = int(np.searchsorted(duration, duration[i], side="left"))
        s0_i = s0[start]
        xbar = s1[start] / s0_i                           # [p]
        log_lik += features[i] @ beta - np.log(s0_i)
        grad += features[i] - xbar
        hess += s2[start] / s0_i - np.outer(xbar, xbar)
    hess += ridge * np.eye(p)
    return log_lik, grad, hess


def fit_one_cause_cox(
    features: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    config: CoxBaselineConfig,
) -> CauseSpecificCox:
    """拟合单个病因的 cause-specific Cox 并计算 Breslow 基线增量。"""
    order = np.argsort(duration, kind="stable")
    x_sorted = features[order]
    d_sorted = duration[order]
    e_sorted = event[order].astype(np.int64)
    n, p = x_sorted.shape
    beta = np.zeros(p)
    converged = False
    log_lik = float("-inf")
    for _ in range(config.max_newton_iters):
        log_lik, grad, hess = _cox_partial_gradient_hessian(
            x_sorted, d_sorted, e_sorted, beta, config.ridge
        )
        step = np.linalg.solve(hess, grad)
        # 回溯线搜索：保证每步偏对数似然不下降，防止 Newton 步过冲发散。
        scale = 1.0
        applied = step
        for _ in range(40):
            candidate = beta + scale * step
            candidate_ll, _, _ = _cox_partial_gradient_hessian(
                x_sorted, d_sorted, e_sorted, candidate, config.ridge
            )
            if np.isfinite(candidate_ll) and candidate_ll >= log_lik - 1e-9:
                applied = scale * step
                break
            scale *= 0.5
        beta = beta + applied
        if np.max(np.abs(applied)) < config.newton_tol:
            converged = True
            break
    # Breslow 基线：每个唯一事件时间 dH0 = d_t / sum_{R(t)} theta。
    theta = np.exp(np.clip(x_sorted @ beta, -30.0, 30.0))
    s0 = np.cumsum(theta[::-1])[::-1]
    event_times = d_sorted[e_sorted == 1]
    unique_times, counts = np.unique(event_times, return_counts=True)
    increments = np.empty(unique_times.shape[0], dtype=np.float64)
    for k, t in enumerate(unique_times):
        start = int(np.searchsorted(d_sorted, t, side="left"))
        increments[k] = counts[k] / s0[start]
    return CauseSpecificCox(
        beta=beta,
        baseline_times=unique_times,
        baseline_increments=increments,
        log_likelihood=float(log_lik),
        converged=converged,
    )


def fit_cause_specific_cox(
    frame: SurvivalFrame, config: CoxBaselineConfig
) -> tuple[CauseSpecificCox, CauseSpecificCox]:
    """拟合两个病因特异 Cox：AECOPD（死亡视为删失）与死亡（AECOPD 视为删失）。"""
    aecopd_event = (frame.event_type == 1).astype(np.int64)
    death_event = (frame.event_type == 2).astype(np.int64)
    aecopd_model = fit_one_cause_cox(frame.features, frame.duration, aecopd_event, config)
    death_model = fit_one_cause_cox(frame.features, frame.duration, death_event, config)
    return aecopd_model, death_model


# ----------------------------------------------------------------------------
# CIF → 条件区间概率
# ----------------------------------------------------------------------------
def _cif_at_boundaries(
    features: np.ndarray,
    aecopd_model: CauseSpecificCox,
    death_model: CauseSpecificCox,
    boundaries: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在合并事件时间网格上 product-integral 求各边界的 (S, CIF_aecopd, CIF_death)。

    返回三个 [N, 3] 数组（列对应 30/90/365 边界处的取值）。
    """
    n = features.shape[0]
    grid = np.union1d(aecopd_model.baseline_times, death_model.baseline_times)
    grid = grid[grid <= boundaries[-1]]
    # 每个网格时间对应两因的 dH0（缺则 0）。
    dh_aecopd = np.zeros(grid.shape[0])
    dh_death = np.zeros(grid.shape[0])
    a_idx = np.searchsorted(grid, aecopd_model.baseline_times)
    valid_a = a_idx < grid.shape[0]
    dh_aecopd[a_idx[valid_a]] = aecopd_model.baseline_increments[valid_a]
    d_idx = np.searchsorted(grid, death_model.baseline_times)
    valid_d = d_idx < grid.shape[0]
    dh_death[d_idx[valid_d]] = death_model.baseline_increments[valid_d]

    theta_a = np.exp(np.clip(features @ aecopd_model.beta, -30.0, 30.0))   # [n]
    theta_d = np.exp(np.clip(features @ death_model.beta, -30.0, 30.0))    # [n]
    survival = np.ones(n)
    cif_a = np.zeros(n)
    cif_d = np.zeros(n)

    s_out = np.ones((n, 3))
    ca_out = np.zeros((n, 3))
    cd_out = np.zeros((n, 3))
    boundary_index = 0
    for step in range(grid.shape[0]):
        # 越过下一个边界前，先把当前累计值写入该边界快照（阶梯函数右连续）。
        while boundary_index < 3 and grid[step] > boundaries[boundary_index]:
            s_out[:, boundary_index] = survival
            ca_out[:, boundary_index] = cif_a
            cd_out[:, boundary_index] = cif_d
            boundary_index += 1
        inc_a = survival * theta_a * dh_aecopd[step]
        inc_d = survival * theta_d * dh_death[step]
        cif_a = cif_a + inc_a
        cif_d = cif_d + inc_d
        d_lambda = theta_a * dh_aecopd[step] + theta_d * dh_death[step]
        survival = np.clip(survival * (1.0 - d_lambda), 0.0, 1.0)
    while boundary_index < 3:
        s_out[:, boundary_index] = survival
        ca_out[:, boundary_index] = cif_a
        cd_out[:, boundary_index] = cif_d
        boundary_index += 1
    return s_out, ca_out, cd_out


def predict_interval_probabilities(
    features: np.ndarray,
    aecopd_model: CauseSpecificCox,
    death_model: CauseSpecificCox,
    boundaries: tuple[float, float, float] = INTERVAL_BOUNDARIES,
) -> np.ndarray:
    """CIF 差分 → 条件区间三分类概率 [N,3,3]，按各区间起点仍在风险集归一化。

    interval_probs[:, i, :] 列序 = (无事件, AECOPD, 死亡)，与 conditional_metrics 一致。
    """
    n = features.shape[0]
    s_b, ca_b, cd_b = _cif_at_boundaries(features, aecopd_model, death_model, boundaries)
    # 各区间的起点/终点取值（起点0处 S=1, CIF=0）。
    s_lo = np.column_stack([np.ones(n), s_b[:, 0], s_b[:, 1]])
    s_hi = s_b
    ca_lo = np.column_stack([np.zeros(n), ca_b[:, 0], ca_b[:, 1]])
    cd_lo = np.column_stack([np.zeros(n), cd_b[:, 0], cd_b[:, 1]])
    probs = np.zeros((n, 3, 3))
    for i in range(3):
        denom = np.clip(s_lo[:, i], 1e-12, None)
        p_no = s_hi[:, i] / denom
        p_ae = (ca_b[:, i] - ca_lo[:, i]) / denom
        p_de = (cd_b[:, i] - cd_lo[:, i]) / denom
        stacked = np.clip(np.column_stack([p_no, p_ae, p_de]), 0.0, None)
        row_sum = np.clip(stacked.sum(axis=1, keepdims=True), 1e-12, None)
        probs[:, i, :] = stacked / row_sum
    return probs


# ----------------------------------------------------------------------------
# 尾部：同协变量多项 logistic（仅在 365 天仍无事件子集上训练）
# ----------------------------------------------------------------------------
@dataclass
class TailMultinomial:
    weights: np.ndarray  # [P+1, 3] 含截距
    base_rate: np.ndarray  # [3] 训练基率（退化回退用）


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_tail_multinomial(
    features: np.ndarray, tail_labels: np.ndarray, config: CoxBaselineConfig
) -> TailMultinomial:
    """在"365 天仍无事件"子集上拟合三分类 softmax（GD + L2），退化时回退基率。"""
    n, p = features.shape
    design = np.column_stack([np.ones(n), features])       # [n, p+1]
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), tail_labels] = 1.0
    base_rate = onehot.mean(axis=0) if n else np.full(3, 1.0 / 3.0)
    weights = np.zeros((p + 1, 3))
    if n >= 30:  # 样本太少不拟合，直接用基率
        for _ in range(config.tail_epochs):
            probs = _softmax(design @ weights)
            grad = design.T @ (probs - onehot) / n
            grad[1:] += config.tail_l2 * weights[1:] / n   # 不罚截距
            weights -= config.tail_lr * grad
    return TailMultinomial(weights=weights, base_rate=base_rate)


def predict_tail_probabilities(features: np.ndarray, model: TailMultinomial) -> np.ndarray:
    """对全体住院给出尾部三分类概率 [N,3]（被 mask 掉的行也需合法概率）。"""
    n = features.shape[0]
    if not np.any(model.weights[1:]):  # 未拟合 → 广播基率
        return np.tile(model.base_rate, (n, 1))
    design = np.column_stack([np.ones(n), features])
    return _softmax(design @ model.weights)


# ----------------------------------------------------------------------------
# Aalen–Johansen（无模型队列累计发生率描述）
# ----------------------------------------------------------------------------
def fit_aalen_johansen(
    frame: SurvivalFrame, boundaries: tuple[float, float, float] = INTERVAL_BOUNDARIES
) -> dict[str, list[float]]:
    """无协变量队列累计发生率（beta=0 的 Breslow）。仅用于队列描述与对照。"""
    n = frame.features.shape[0]
    aecopd_event = (frame.event_type == 1).astype(np.int64)
    death_event = (frame.event_type == 2).astype(np.int64)
    order = np.argsort(frame.duration, kind="stable")
    d_sorted = frame.duration[order]
    theta = np.ones(n)
    s0 = np.cumsum(theta[::-1])[::-1]

    def _increments(event_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        times = d_sorted[event_sorted == 1]
        uniq, counts = np.unique(times, return_counts=True)
        inc = np.array([counts[k] / s0[int(np.searchsorted(d_sorted, t, side="left"))]
                        for k, t in enumerate(uniq)])
        return uniq, inc

    a_model = CauseSpecificCox(np.zeros(1), *_increments(aecopd_event[order]), 0.0, True)
    d_model = CauseSpecificCox(np.zeros(1), *_increments(death_event[order]), 0.0, True)
    # 用单行零协变量复用 CIF 计算。
    s_b, ca_b, cd_b = _cif_at_boundaries(np.zeros((1, 1)), a_model, d_model, boundaries)
    labels = ["days_0_30_cum", "days_0_90_cum", "days_0_365_cum"]
    return {
        "boundary_days": list(boundaries),
        "labels": labels,
        "survival": s_b[0].tolist(),
        "cif_aecopd": ca_b[0].tolist(),
        "cif_death": cd_b[0].tolist(),
    }


# ----------------------------------------------------------------------------
# 端到端
# ----------------------------------------------------------------------------
@dataclass
class B2Predictions:
    interval_probabilities: np.ndarray  # [N,3,3]
    tail_probabilities: np.ndarray      # [N,3]
    hadm_ids: np.ndarray
    subject_ids: np.ndarray


def predict_split(
    config: CoxBaselineConfig,
    stats: _StandardizerStats,
    aecopd_model: CauseSpecificCox,
    death_model: CauseSpecificCox,
    tail_model: TailMultinomial,
    split: str,
) -> B2Predictions:
    """对一个划分产出与主模型同形状的概率张量（不评估、不碰测试集选择）。"""
    frame = assemble_survival_frame(config, split, stats)
    interval = predict_interval_probabilities(frame.features, aecopd_model, death_model)
    tail = predict_tail_probabilities(frame.features, tail_model)
    return B2Predictions(
        interval_probabilities=interval,
        tail_probabilities=tail,
        hadm_ids=frame.hadm_ids,
        subject_ids=frame.subject_ids,
    )


def _aligned_labels(
    store, hadm_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按 hadm_ids 顺序从 ConditionalOutcomeStore 取标签/掩码，保证与概率张量对齐。"""
    interval_labels = np.array([store[int(h)].interval_labels for h in hadm_ids], dtype=np.int64)
    interval_mask = np.array([store[int(h)].interval_mask for h in hadm_ids], dtype=bool)
    tail_labels = np.array([store[int(h)].tail_label for h in hadm_ids], dtype=np.int64)
    tail_mask = np.array([store[int(h)].tail_mask for h in hadm_ids], dtype=bool)
    return interval_labels, interval_mask, tail_labels, tail_mask


def run_b2(config: CoxBaselineConfig, eval_split: str = "validation") -> BaselineEvalResult:
    """B2 端到端：训练折拟合 → 指定折预测评估。默认在 validation 上评估（测试集冻结）。

    注意：eval_split='test' 只应在最终定稿口径下调用一次。开发与对比一律用 validation。
    """
    config.split.validate_no_patient_leakage()
    train_rows = [r for r in _read_cohort_rows(config.cohort_csv_gz_path) if r["data_split"] == "train"]
    stats = _fit_standardizer(train_rows)
    train_frame = assemble_survival_frame(config, "train", stats)
    aecopd_model, death_model = fit_cause_specific_cox(train_frame, config)

    tail_subset = train_frame.event_type == 0
    from aecopd_training.conditional_outcomes import ConditionalOutcomeStore

    store = ConditionalOutcomeStore.from_cohort(config.cohort_csv_gz_path)
    train_tail_labels = np.array(
        [store[int(h)].tail_label for h in train_frame.hadm_ids[tail_subset]], dtype=np.int64
    )
    tail_model = fit_tail_multinomial(
        train_frame.features[tail_subset], train_tail_labels, config
    )

    predictions = predict_split(config, stats, aecopd_model, death_model, tail_model, eval_split)
    interval_labels, interval_mask, tail_labels, tail_mask = _aligned_labels(
        store, predictions.hadm_ids
    )
    metrics, calibrations = evaluate_with_project_metrics(
        predictions.interval_probabilities,
        interval_labels,
        interval_mask,
        predictions.tail_probabilities,
        tail_labels,
        tail_mask,
        prefix=eval_split,
    )
    result = BaselineEvalResult(model_name="B2_cause_specific_cox")
    for interval_name in ("days_0_30", "days_31_90", "days_91_365"):
        result.interval_aecopd_auc[interval_name] = metrics.get(
            f"{eval_split}_aecopd_auc_{interval_name}", float("nan")
        )
        result.interval_aecopd_brier[interval_name] = metrics.get(
            f"{eval_split}_aecopd_brier_{interval_name}", float("nan")
        )
        result.interval_death_auc[interval_name] = metrics.get(
            f"{eval_split}_death_auc_{interval_name}", float("nan")
        )
        result.interval_death_brier[interval_name] = metrics.get(
            f"{eval_split}_death_brier_{interval_name}", float("nan")
        )
    result.tail_accuracy = metrics.get(f"{eval_split}_tail_accuracy", float("nan"))
    result.tail_none_observed_auc = metrics.get(f"{eval_split}_tail_none_observed_auc", float("nan"))
    result.tail_aecopd_auc = metrics.get(f"{eval_split}_tail_aecopd_auc", float("nan"))
    result.tail_death_auc = metrics.get(f"{eval_split}_tail_death_auc", float("nan"))
    result.notes = (
        f"cause-specific Cox (numpy Newton-Raphson+Breslow), World A censoring at 365d; "
        f"aecopd_cox_converged={aecopd_model.converged}, death_cox_converged={death_model.converged}; "
        f"tail=multinomial logistic on event-free-at-365 subset; eval_split={eval_split}"
    )
    return result
