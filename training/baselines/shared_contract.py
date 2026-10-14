"""基线共享实验合同：划分、标签、评估口径的单一事实来源。

所有基线（B1/B2/B3）必须复用这里的定义，确保与主模型可比。本模块只定义
数据结构与契约常量，不做任何训练或数据库写入。
"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ------- 结局编码：与 conditional_outcomes.py 保持完全一致 -------
OUTCOME_NO_EVENT = 0
OUTCOME_AECOPD = 1
OUTCOME_DEATH = 2

INTERVAL_NAMES: tuple[str, ...] = ("days_0_30", "days_31_90", "days_91_365")
INTERVAL_UPPER_DAYS: tuple[float, ...] = (30.0, 90.0, 365.0)
TAIL_OUTCOME_NAMES: tuple[str, ...] = ("none_observed", "aecopd", "death")

# 主模型正式测试指标（single-split），供基线对照的参考锚点，不作为选择目标。
# 来源：training_runs/aecopd_patient_longitudinal_h256_conditional_intervals_tail_seed2026/metrics/test.json
MAIN_MODEL_TEST_AECOPD_AUC: dict[str, float] = {
    "days_0_30": 0.7207,
    "days_31_90": 0.7279,
    "days_91_365": 0.7466,
}
MAIN_MODEL_TEST_DEATH_AUC: dict[str, float] = {
    "days_0_30": 0.7793,
    "days_31_90": 0.6541,
    "days_91_365": 0.6899,
}
MAIN_MODEL_TAIL_ACCURACY: float = 0.6427
MAIN_MODEL_TAIL_MAJORITY_BASELINE: float = 0.6494  # none_observed 多数类基线


@dataclass(frozen=True)
class SplitContract:
    """患者级划分契约。基线必须直接复用主模型的划分文件，不得重新随机划分。"""

    # 患者序列索引，含 subject_id/hadm_id/data_split，权威来源见 data_v2 patient_sequence_index
    patient_sequence_index_path: Path
    # 允许的划分名
    splits: tuple[str, ...] = ("train", "validation", "test")

    def validate_no_patient_leakage(self) -> dict[str, int]:
        """校验同一 subject_id 不跨划分；返回各划分病人数摘要。

        直接读患者序列索引（含 subject_id/hadm_id/data_split），断言每个 subject_id
        只出现在一个 data_split。任何跨划分立即抛错——这是所有基线可比的前提。
        """
        splits_by_subject: dict[int, set[str]] = defaultdict(set)
        with gzip.open(self.patient_sequence_index_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"subject_id", "data_split"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"patient_sequence_index missing columns: {sorted(missing)}")
            for row in reader:
                subject = int(row["subject_id"])
                split = row["data_split"]
                if split not in self.splits:
                    raise ValueError(f"unexpected data_split value: {split!r}")
                splits_by_subject[subject].add(split)
        leaked = {subject for subject, splits in splits_by_subject.items() if len(splits) > 1}
        if leaked:
            raise ValueError(
                f"{len(leaked)} subject(s) leak across splits; patient-level split is violated"
            )
        summary: dict[str, int] = {split: 0 for split in self.splits}
        for splits in splits_by_subject.values():
            summary[next(iter(splits))] += 1
        return summary


@dataclass(frozen=True)
class LabelContract:
    """标签契约：直接调用主项目 ConditionalOutcomeStore，避免重复实现区间/掩码逻辑。"""

    cohort_csv_gz_path: Path  # 与 conditional_outcomes.ConditionalOutcomeStore.from_cohort 一致

    def load_targets(self):
        """返回按 hadm_id 可寻址的条件区间+尾部标签（复用主项目 ConditionalOutcomeStore）。

        不另写标签逻辑：世界观 A（出院后数据库再无记录=阴性）已是 ConditionalOutcomeStore
        的现有行为，直接从队列表构造，保证与主模型逐 hadm 一致。
        """
        from aecopd_training.conditional_outcomes import ConditionalOutcomeStore

        return ConditionalOutcomeStore.from_cohort(self.cohort_csv_gz_path)


@dataclass
class BaselineEvalResult:
    """统一的基线评估产物，字段命名对齐 conditional_metrics.py 以便直接并列对照。"""

    model_name: str
    interval_aecopd_auc: dict[str, float] = field(default_factory=dict)
    interval_aecopd_brier: dict[str, float] = field(default_factory=dict)
    interval_death_auc: dict[str, float] = field(default_factory=dict)
    interval_death_brier: dict[str, float] = field(default_factory=dict)
    tail_accuracy: float = float("nan")
    tail_none_observed_auc: float = float("nan")
    tail_aecopd_auc: float = float("nan")
    tail_death_auc: float = float("nan")
    notes: str = ""


def evaluate_with_project_metrics(
    interval_probabilities,  # np.ndarray [N, 3, 3]
    interval_labels,         # np.ndarray [N, 3]
    interval_mask,           # np.ndarray [N, 3] bool
    tail_probabilities,      # np.ndarray [N, 3]
    tail_labels,             # np.ndarray [N]
    tail_mask,               # np.ndarray [N] bool
    *,
    prefix: str = "test",
):
    """复用主项目 conditional_metrics，保证 AUC/Brier/校准口径与主模型完全一致。

    基线唯一的职责是产出与主模型同形状的概率张量（区间 [N,3,3]、尾部 [N,3]），
    评估交给同一函数，避免口径漂移。返回 (metrics, calibrations)。
    """
    from aecopd_training.conditional_metrics import conditional_metrics

    return conditional_metrics(
        interval_probabilities,
        interval_labels,
        interval_mask,
        tail_probabilities,
        tail_labels,
        tail_mask,
        prefix=prefix,
    )


def bootstrap_with_project_metrics(
    interval_probabilities,
    interval_labels,
    interval_mask,
    tail_probabilities,
    tail_labels,
    tail_mask,
    subject_ids,
    *,
    prefix: str = "test",
    n_resamples: int = 2000,
    seed: int = 2026,
    confidence: float = 0.95,
):
    """对基线预测做患者聚类 bootstrap 置信区间（与主模型共用同一工具）。

    薄封装 aecopd_training.patient_bootstrap.patient_clustered_bootstrap，
    subject_ids 为每次住院所属病人 id，确保重采样以病人为聚类单位。返回 BootstrapResult。
    """
    from aecopd_training.patient_bootstrap import (
        BootstrapConfig,
        patient_clustered_bootstrap,
    )

    return patient_clustered_bootstrap(
        interval_probabilities,
        interval_labels,
        interval_mask,
        tail_probabilities,
        tail_labels,
        tail_mask,
        subject_ids,
        prefix=prefix,
        config=BootstrapConfig(n_resamples=n_resamples, seed=seed, confidence=confidence),
    )
