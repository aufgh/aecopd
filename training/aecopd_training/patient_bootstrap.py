"""患者聚类 bootstrap 置信区间（三基线与主模型共用）。

为什么按患者重采样而不是按住院：同一病人的多次住院不是相互独立的观测，
若按住院重采样会低估不确定性。这里以 subject_id 为不可分割的聚类单位，
有放回地抽取病人，被抽中的病人带上其全部住院，再在重采样集上用与主模型
完全相同的 conditional_metrics 重算指标，取分位数作为置信区间。

本模块只做数值计算，不训练、不读数据库、不碰测试集选择。点估计在全样本上算，
区间来自对病人的重采样分布；空风险集导致某次重采样的某指标为 NaN 时，
该指标在这一次重采样被跳过（不计入其分位数），并在结果里记录有效重采样次数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .conditional_metrics import conditional_metrics


@dataclass(frozen=True)
class BootstrapConfig:
    """患者聚类 bootstrap 的重采样与区间设置。"""

    n_resamples: int = 2000
    seed: int = 2026
    confidence: float = 0.95  # 双侧置信水平；0.95 → 取 2.5% 与 97.5% 分位

    def __post_init__(self) -> None:
        if self.n_resamples < 1:
            raise ValueError("n_resamples must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be strictly between 0 and 1")


@dataclass
class MetricInterval:
    """单个指标的点估计与置信区间。"""

    point: float
    lower: float
    upper: float
    std: float
    valid_resamples: int


@dataclass
class BootstrapResult:
    """一次患者聚类 bootstrap 的完整产物。"""

    prefix: str
    n_patients: int
    n_admissions: int
    n_resamples: int
    confidence: float
    metrics: dict[str, MetricInterval] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        """转成可 json 序列化的嵌套字典（NaN → None，便于写文件与报告）。"""

        def _num(value: float) -> float | None:
            return None if not math.isfinite(value) else float(value)

        return {
            "prefix": self.prefix,
            "n_patients": self.n_patients,
            "n_admissions": self.n_admissions,
            "n_resamples": self.n_resamples,
            "confidence": self.confidence,
            "metrics": {
                name: {
                    "point": _num(interval.point),
                    "lower": _num(interval.lower),
                    "upper": _num(interval.upper),
                    "std": _num(interval.std),
                    "valid_resamples": interval.valid_resamples,
                }
                for name, interval in self.metrics.items()
            },
        }


def _subject_row_groups(subject_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """返回唯一病人数组与「每个病人 → 其住院行下标」的分组列表（顺序一致）。"""
    unique_subjects, inverse = np.unique(subject_ids, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    boundaries = np.searchsorted(sorted_inverse, np.arange(unique_subjects.size))
    groups: list[np.ndarray] = []
    for index in range(unique_subjects.size):
        start = boundaries[index]
        stop = boundaries[index + 1] if index + 1 < unique_subjects.size else order.size
        groups.append(order[start:stop])
    return unique_subjects, groups


def patient_clustered_bootstrap(
    interval_probabilities: np.ndarray,  # [N, 3, 3]
    interval_labels: np.ndarray,         # [N, 3]
    interval_mask: np.ndarray,           # [N, 3] bool
    tail_probabilities: np.ndarray,      # [N, 3]
    tail_labels: np.ndarray,             # [N]
    tail_mask: np.ndarray,               # [N] bool
    subject_ids: np.ndarray,             # [N] 每次住院所属病人的 id
    *,
    prefix: str = "test",
    config: BootstrapConfig | None = None,
) -> BootstrapResult:
    """按病人重采样计算 conditional_metrics 的置信区间。

    点估计在全样本上调用一次 conditional_metrics；随后有放回抽取病人，
    每次重采样把被抽中病人的全部住院行拼接后重算指标，最后对每个指标取
    经验分位数。只有出现在两侧的数值型指标才产出区间；NaN 指标被逐次跳过。
    """
    settings = config or BootstrapConfig()
    subjects = np.asarray(subject_ids).reshape(-1)
    sample_count = subjects.shape[0]
    if sample_count == 0:
        raise ValueError("empty sample")
    if interval_labels.shape[0] != sample_count:
        raise ValueError("subject_ids length must match number of admissions")

    interval_probabilities = np.asarray(interval_probabilities, dtype=np.float64)
    interval_labels = np.asarray(interval_labels, dtype=np.int64)
    interval_mask = np.asarray(interval_mask, dtype=bool)
    tail_probabilities = np.asarray(tail_probabilities, dtype=np.float64)
    tail_labels = np.asarray(tail_labels, dtype=np.int64)
    tail_mask = np.asarray(tail_mask, dtype=bool)

    # 点估计（全样本，一次）。
    point_metrics, _ = conditional_metrics(
        interval_probabilities,
        interval_labels,
        interval_mask,
        tail_probabilities,
        tail_labels,
        tail_mask,
        prefix=prefix,
    )

    unique_subjects, subject_groups = _subject_row_groups(subjects)
    patient_count = unique_subjects.size
    rng = np.random.default_rng(settings.seed)

    collected: dict[str, list[float]] = {name: [] for name in point_metrics}
    for _ in range(settings.n_resamples):
        chosen = rng.integers(0, patient_count, size=patient_count)
        row_index = np.concatenate([subject_groups[patient] for patient in chosen])
        resample_metrics, _ = conditional_metrics(
            interval_probabilities[row_index],
            interval_labels[row_index],
            interval_mask[row_index],
            tail_probabilities[row_index],
            tail_labels[row_index],
            tail_mask[row_index],
            prefix=prefix,
        )
        for name, value in resample_metrics.items():
            if math.isfinite(value):
                collected[name].append(value)

    alpha = (1.0 - settings.confidence) / 2.0
    lower_q = 100.0 * alpha
    upper_q = 100.0 * (1.0 - alpha)
    metrics: dict[str, MetricInterval] = {}
    for name, point in point_metrics.items():
        samples = np.asarray(collected[name], dtype=np.float64)
        if samples.size == 0:
            metrics[name] = MetricInterval(
                point=point, lower=math.nan, upper=math.nan, std=math.nan, valid_resamples=0
            )
            continue
        metrics[name] = MetricInterval(
            point=float(point),
            lower=float(np.percentile(samples, lower_q)),
            upper=float(np.percentile(samples, upper_q)),
            std=float(np.std(samples, ddof=1)) if samples.size > 1 else math.nan,
            valid_resamples=int(samples.size),
        )

    return BootstrapResult(
        prefix=prefix,
        n_patients=int(patient_count),
        n_admissions=int(sample_count),
        n_resamples=settings.n_resamples,
        confidence=settings.confidence,
        metrics=metrics,
    )
