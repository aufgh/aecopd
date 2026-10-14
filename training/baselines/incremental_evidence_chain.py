"""住院等权的低容量增量证据链。

同一患者划分、条件风险集和评价函数下，按以下顺序逐层增加信息：

1. B0：既往 AECOPD 次数单特征；
2. B1：低容量患者状态核心模型；
3. B2：以 B0 风险为固定 offset 的患者状态残差模型；
4. B3：增加患者状态数值轨迹摘要；
5. B4：增加诊疗行为静态变量和受控概念暴露；
6. B5：增加冻结文本向量的浅层 PCA 表示。

所有模型都是确定性的正则化二项 logistic。AECOPD 是主要事件，死亡在
“未发生 AECOPD”的剩余概率质量内建模，从而生成合法的三分类概率，同时
保证 AECOPD 排序不受死亡头二次改写。复杂 Transformer 不参与本实验。
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from aecopd_training.conditional_outcomes import ConditionalOutcomeStore, INTERVAL_NAMES
from aecopd_training.conditional_metrics import conditional_metrics
from baselines.shared_contract import SplitContract


PATIENT_STATE_CONTINUOUS: tuple[str, ...] = (
    "age_at_admission",
    "prior_all_admission_count",
    "days_since_previous_aecopd",
    "prior_weight_kg",
    "prior_height_cm",
    "prior_bmi",
    "prior_sbp",
    "prior_dbp",
    "days_since_prior_weight",
    "days_since_prior_bmi",
    "days_since_prior_bp",
)
CARE_PROCESS_CONTINUOUS: tuple[str, ...] = (
    "ed_hours",
    "hospital_los_hours",
    "icu_stay_count",
    "icu_los_days",
)
CARE_PROCESS_CATEGORICAL: tuple[str, ...] = (
    "admission_type",
    "admission_location",
    "discharge_location",
)
CARE_PROCESS_BINARY: tuple[str, ...] = ("via_ed", "entered_icu")
TRAJECTORY_STATISTICS: tuple[str, ...] = (
    "mean_value_scaled",
    "last_value_scaled",
    "first_to_last_change_scaled",
    "slope_per_hour_scaled",
)
MEASUREMENT_PROCESS_STATISTICS: tuple[str, ...] = (
    "measurement_count_log1p",
    "present",
)
TEXT_DOCUMENT_ROLES: tuple[str, ...] = (
    "current_discharge",
    "current_radiology",
    "prior_discharge",
    "prior_radiology",
)


@dataclass(frozen=True)
class IncrementalEvidenceConfig:
    """增量证据链的固定数据与正则化合同。"""

    data_dir: Path
    artifacts_path: Path
    split: SplitContract
    text_dir: Path | None = None
    eval_split: str = "validation"
    residual_l2: float = 0.01
    text_pca_dim: int = 32
    bootstrap_resamples: int = 1000
    confidence: float = 0.95
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.eval_split != "validation":
            raise ValueError(
                "incremental architecture selection is restricted to validation; "
                "the repeatedly inspected test split must not be used"
            )
        if self.residual_l2 < 0.0:
            raise ValueError("residual_l2 must be non-negative")
        if self.text_pca_dim < len(TEXT_DOCUMENT_ROLES):
            raise ValueError(
                f"text_pca_dim must be at least {len(TEXT_DOCUMENT_ROLES)} "
                "so every document role receives at least one component"
            )
        if self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be strictly between 0 and 1")


@dataclass(frozen=True)
class FeatureBlocks:
    """按同一住院顺序对齐的输入层；字典键是实验可切换的真实 seam。"""

    hadm_ids: np.ndarray
    subject_ids: np.ndarray
    splits: np.ndarray
    blocks: Mapping[str, np.ndarray]
    feature_names: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        count = self.hadm_ids.shape[0]
        if self.subject_ids.shape != (count,) or self.splits.shape != (count,):
            raise ValueError("feature keys must have one row per admission")
        for name, values in self.blocks.items():
            if values.ndim != 2 or values.shape[0] != count:
                raise ValueError(f"feature block {name!r} is not admission aligned")
            if len(self.feature_names[name]) != values.shape[1]:
                raise ValueError(f"feature names do not match block {name!r}")
            if not np.isfinite(values).all():
                raise ValueError(f"feature block {name!r} contains non-finite values")


@dataclass(frozen=True)
class BinaryLogitModel:
    """带可选固定 offset 的 L2 logistic 模型。"""

    intercept: float
    coefficients: np.ndarray
    converged: bool

    def decision_function(
        self,
        features: np.ndarray,
        *,
        offset: np.ndarray | None = None,
    ) -> np.ndarray:
        values = self.intercept + np.asarray(features, dtype=np.float64) @ self.coefficients
        if offset is not None:
            values = values + np.asarray(offset, dtype=np.float64)
        return values


@dataclass(frozen=True)
class StagePrediction:
    """一个证据层级的验证集概率和拟合审计。"""

    name: str
    label: str
    interval_probabilities: np.ndarray
    tail_probabilities: np.ndarray
    feature_count: int
    converged_heads: int
    total_heads: int


@dataclass(frozen=True)
class RetainedModel:
    """当前通过门控的模型形态，用于保证后续候选真正沿证据链递增。"""

    index: int
    blocks: tuple[str, ...]
    residual_anchor: bool


def advance_retained_model(
    current: RetainedModel,
    candidate: RetainedModel,
    *,
    supported: bool,
) -> RetainedModel:
    """只有候选获得稳定增益时才替换当前保留模型。"""

    return candidate if supported else current


def stage_gate_exclusion_reason(
    prediction: StagePrediction,
    aucs: Mapping[str, float],
) -> str | None:
    """返回候选不能进入统计门控的原因；None 表示可比较。"""

    if prediction.converged_heads != prediction.total_heads:
        return "not_all_heads_converged_or_estimable"
    if not all(np.isfinite(value) for value in aucs.values()):
        return "non_finite_primary_auc"
    return None


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def compose_competing_probabilities(
    aecopd_logits: np.ndarray,
    death_given_no_aecopd_logits: np.ndarray,
) -> np.ndarray:
    """把两个顺序二项 logit 合成 [无事件, AECOPD, 死亡] 概率。"""

    aecopd = _sigmoid(aecopd_logits)
    residual = 1.0 - aecopd
    death = residual * _sigmoid(death_given_no_aecopd_logits)
    no_event = residual - death
    probabilities = np.column_stack((no_event, aecopd, death))
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
        raise AssertionError("competing probabilities do not sum to one")
    return probabilities


def fit_binary_logit(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float,
    offset: np.ndarray | None = None,
    nonnegative_first_coefficient: bool = False,
) -> BinaryLogitModel:
    """拟合平均负对数似然；截距不惩罚，其他系数使用固定 L2。"""

    from scipy.optimize import minimize

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or y.size == 0:
        raise ValueError("features and labels must be non-empty and row aligned")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("binary labels must contain only zero and one")
    fixed = np.zeros(y.shape[0], dtype=np.float64) if offset is None else np.asarray(offset, dtype=np.float64)
    if fixed.shape != y.shape or not np.isfinite(fixed).all():
        raise ValueError("offset must be finite and aligned with labels")

    # 单类折没有有限的无惩罚截距最优解；使用 Jeffreys 平滑的常量概率
    # 明确回退，避免把 optimizer 未收敛结果静默带入证据链。
    if np.all(y == y[0]):
        prevalence = float((y.sum() + 0.5) / (y.size + 1.0))
        intercept = math.log(prevalence / (1.0 - prevalence)) - float(fixed.mean())
        return BinaryLogitModel(
            intercept=intercept,
            coefficients=np.zeros(x.shape[1], dtype=np.float64),
            converged=True,
        )

    prevalence = float(np.clip(y.mean(), 1e-5, 1.0 - 1e-5))
    initial_intercept = math.log(prevalence / (1.0 - prevalence)) - float(fixed.mean())
    initial = np.zeros(x.shape[1] + 1, dtype=np.float64)
    initial[0] = initial_intercept

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficients = parameters[1:]
        logits = fixed + intercept + x @ coefficients
        probability = _sigmoid(logits)
        loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
        loss += 0.5 * l2 * float(coefficients @ coefficients)
        error = probability - y
        gradient = np.empty_like(parameters)
        gradient[0] = float(error.mean())
        gradient[1:] = x.T @ error / y.size + l2 * coefficients
        return loss, gradient

    bounds: list[tuple[float | None, float | None]] = [(None, None)] * initial.size
    if nonnegative_first_coefficient and x.shape[1] > 0:
        bounds[1] = (1e-8, None)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
    )
    return BinaryLogitModel(
        intercept=float(result.x[0]),
        coefficients=np.asarray(result.x[1:], dtype=np.float64),
        converged=bool(result.success),
    )


def _read_cohort_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _one_hot_ids(
    ids: np.ndarray,
    *,
    size: int,
    prefix: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """保留 UNK/MISSING 与训练词表类别，丢弃只用于 padding 的 id=0。"""

    columns = [np.asarray(ids == category, dtype=np.float64) for category in range(1, size)]
    if not columns:
        return np.zeros((ids.shape[0], 0), dtype=np.float64), ()
    return np.column_stack(columns), tuple(f"{prefix}__id_{category}" for category in range(1, size))


def build_feature_blocks(config: IncrementalEvidenceConfig) -> FeatureBlocks:
    """从最新受控映射层构造住院对齐的患者状态、轨迹和诊疗行为块。"""

    from aecopd_loader import AecopdDataBundle

    artifacts = json.loads(config.artifacts_path.read_text(encoding="utf-8"))
    # 现有框架无关 loader 即使 include_text=False 也会建立文本行游标；无文本
    # 实验只借用同级 model_v2 索引完成 hadm 对齐，不读取任何 2,560 维向量。
    metadata_dir = config.text_dir or (config.data_dir.parent / "model_v2")
    bundle = AecopdDataBundle(
        config.data_dir,
        text_metadata_only=True,
        text_dir=metadata_dir,
    )
    samples = list(bundle.iter_all_samples(artifacts=artifacts, include_text=False))
    cohort_rows = _read_cohort_rows(config.data_dir / "cohort_labels_static.csv.gz")
    raw_by_hadm = {int(row["hadm_id"]): row for row in cohort_rows}
    if len(samples) != len(raw_by_hadm):
        raise ValueError("loader samples and cohort admissions are not one-to-one")

    hadm_ids = np.array([sample.hadm_id for sample in samples], dtype=np.int64)
    subject_ids = np.array([sample.subject_id for sample in samples], dtype=np.int64)
    splits = np.array([sample.data_split for sample in samples], dtype=object)
    count = hadm_ids.size

    anchor = np.array(
        [math.log1p(float(raw_by_hadm[int(hadm)]["prior_aecopd_count"])) for hadm in hadm_ids],
        dtype=np.float64,
    ).reshape(-1, 1)

    core_columns: list[np.ndarray] = []
    core_names: list[str] = []
    for name in PATIENT_STATE_CONTINUOUS:
        core_columns.append(np.array([sample.static_continuous[name] for sample in samples]))
        core_columns.append(
            np.array([float(sample.static_missing_mask[name]) for sample in samples])
        )
        core_names.extend((name, f"{name}__missing"))
    gender_ids = np.array([sample.static_categorical["gender"] for sample in samples])
    gender, gender_names = _one_hot_ids(
        gender_ids,
        size=len(artifacts["static_categorical"]["gender"]["vocabulary"]),
        prefix="gender",
    )
    core_columns.extend(gender.T)
    core_names.extend(gender_names)
    core = np.column_stack(core_columns) if core_columns else np.zeros((count, 0))

    summary_vocab = tuple(artifacts["numeric_summary_vocabulary"])
    summary_width = max(0, len(summary_vocab) - 3)
    trajectory = np.zeros((count, summary_width * len(TRAJECTORY_STATISTICS)), dtype=np.float64)
    measurement_process = np.zeros(
        (count, summary_width * len(MEASUREMENT_PROCESS_STATISTICS)),
        dtype=np.float64,
    )
    trajectory_names: list[str] = []
    for key in summary_vocab[3:]:
        trajectory_names.extend(f"{key}|{statistic}" for statistic in TRAJECTORY_STATISTICS)
    for row_index, sample in enumerate(samples):
        for summary in sample.numeric_summaries:
            concept_index = summary.concept_id - 3
            if concept_index < 0 or concept_index >= summary_width:
                continue
            offset = concept_index * len(TRAJECTORY_STATISTICS)
            trajectory[row_index, offset : offset + len(TRAJECTORY_STATISTICS)] = (
                summary.scaled_values["mean_value_scaled"],
                summary.scaled_values["last_value_scaled"],
                summary.scaled_values["first_to_last_change_scaled"],
                summary.scaled_values["slope_per_hour_scaled"],
            )
            process_offset = concept_index * len(MEASUREMENT_PROCESS_STATISTICS)
            measurement_process[
                row_index,
                process_offset : process_offset + len(MEASUREMENT_PROCESS_STATISTICS),
            ] = (summary.measurement_count_log1p, 1.0)

    care_columns: list[np.ndarray] = []
    care_names: list[str] = []
    for name in CARE_PROCESS_CONTINUOUS:
        care_columns.append(np.array([sample.static_continuous[name] for sample in samples]))
        care_columns.append(
            np.array([float(sample.static_missing_mask[name]) for sample in samples])
        )
        care_names.extend((name, f"{name}__missing"))
    for name in CARE_PROCESS_CATEGORICAL:
        category_ids = np.array([sample.static_categorical[name] for sample in samples])
        encoded, names = _one_hot_ids(
            category_ids,
            size=len(artifacts["static_categorical"][name]["vocabulary"]),
            prefix=name,
        )
        care_columns.extend(encoded.T)
        care_names.extend(names)
    for name in CARE_PROCESS_BINARY:
        care_columns.append(np.array([sample.static_binary[name] for sample in samples], dtype=np.float64))
        care_names.append(name)

    event_vocab = tuple(artifacts["event_vocabulary"])
    care_event_ids = tuple(
        index for index, key in enumerate(event_vocab) if key.startswith("care_process|")
    )
    care_event_index = {concept_id: position for position, concept_id in enumerate(care_event_ids)}
    care_events = np.zeros((count, len(care_event_ids) * 4), dtype=np.float64)
    for concept_id in care_event_ids:
        key = event_vocab[concept_id]
        care_names.extend(
            (
                f"{key}|present",
                f"{key}|first_los_fraction",
                f"{key}|last_los_fraction",
                f"{key}|span_los_fraction",
            )
        )
    for row_index, sample in enumerate(samples):
        grouped: dict[int, list[Any]] = {}
        for event in sample.events:
            if event.feature_layer == "care_process" and event.concept_id in care_event_index:
                grouped.setdefault(event.concept_id, []).append(event)
        scale = max(float(sample.hospital_los_hours), 24.0)
        for concept_id, events in grouped.items():
            base = care_event_index[concept_id] * 4
            care_events[row_index, base] = 1.0
            timed = [
                event
                for event in events
                if not event.evidence_type.startswith("retrospective_")
            ]
            if timed:
                hours = np.array([event.hours_from_admission for event in timed], dtype=np.float64)
                first = float(np.clip(hours.min() / scale, -1.0, 2.0))
                last = float(np.clip(hours.max() / scale, -1.0, 2.0))
                care_events[row_index, base + 1 : base + 4] = (
                    first,
                    last,
                    float(np.clip(last - first, 0.0, 3.0)),
                )
    care_columns.extend(care_events.T)
    # 检验频率与是否测量属于诊疗/记录过程，不属于患者生理轨迹。
    for key in summary_vocab[3:]:
        care_names.extend(
            f"measurement_process|{key}|{statistic}"
            for statistic in MEASUREMENT_PROCESS_STATISTICS
        )
    care_columns.extend(measurement_process.T)
    care = np.column_stack(care_columns) if care_columns else np.zeros((count, 0))

    return FeatureBlocks(
        hadm_ids=hadm_ids,
        subject_ids=subject_ids,
        splits=splits,
        blocks={
            "anchor": anchor,
            "core": core,
            "trajectory": trajectory,
            "care_process": care,
        },
        feature_names={
            "anchor": ("log1p_prior_aecopd_count",),
            "core": tuple(core_names),
            "trajectory": tuple(trajectory_names),
            "care_process": tuple(care_names),
        },
    )


def validate_feature_split_contract(
    feature_blocks: FeatureBlocks,
    split_contract: SplitContract,
) -> None:
    """逐住院核对实际特征行与权威患者索引，禁止陈旧索引假通过。"""

    expected: dict[int, tuple[int, str]] = {}
    with gzip.open(
        split_contract.patient_sequence_index_path,
        "rt",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            hadm = int(row["hadm_id"])
            value = (int(row["subject_id"]), row["data_split"])
            if hadm in expected and expected[hadm] != value:
                raise ValueError(f"conflicting split-contract rows for hadm_id={hadm}")
            expected[hadm] = value
    actual = {
        int(hadm): (int(subject), str(split))
        for hadm, subject, split in zip(
            feature_blocks.hadm_ids,
            feature_blocks.subject_ids,
            feature_blocks.splits,
            strict=True,
        )
    }
    if len(actual) != feature_blocks.hadm_ids.size:
        raise ValueError("duplicate hadm_id in feature blocks")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(
            "feature rows and split contract contain different admissions: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    mismatched = [hadm for hadm, value in actual.items() if expected[hadm] != value]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} admission(s) disagree with the patient split contract"
        )


def add_text_block(
    feature_blocks: FeatureBlocks,
    *,
    text_dir: Path,
    pca_dim: int,
    seed: int,
    embedding_path: Path | None = None,
) -> FeatureBlocks:
    """按四类文档角色分别 mean-pool，并只在训练集拟合角色内 PCA。"""

    from sklearn.decomposition import PCA

    wanted = {int(hadm) for hadm in feature_blocks.hadm_ids}
    rows_by_hadm: dict[int, dict[str, list[int]]] = {
        hadm: {role: [] for role in TEXT_DOCUMENT_ROLES} for hadm in wanted
    }
    index_path = text_dir / "qwen_section_embedding_index.csv.gz"
    with gzip.open(index_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            hadm = int(row["hadm_id"])
            role = row["document_role"]
            if hadm in rows_by_hadm and role in rows_by_hadm[hadm]:
                rows_by_hadm[hadm][role].append(int(row["embedding_row"]))
    embeddings = np.load(
        embedding_path or (text_dir / "qwen_section_embeddings.float16.npy"), mmap_mode="r"
    )
    train = feature_blocks.splits == "train"
    if int(train.sum()) < 2:
        raise ValueError("at least two training admissions are required for text PCA")
    base_dimension, remainder = divmod(pca_dim, len(TEXT_DOCUMENT_ROLES))
    text_parts: list[np.ndarray] = []
    names: list[str] = []
    for role_index, role in enumerate(TEXT_DOCUMENT_ROLES):
        pooled = np.zeros((feature_blocks.hadm_ids.size, embeddings.shape[1]), dtype=np.float32)
        presence = np.zeros((feature_blocks.hadm_ids.size, 1), dtype=np.float64)
        for index, hadm in enumerate(feature_blocks.hadm_ids):
            rows = rows_by_hadm[int(hadm)][role]
            if not rows:
                continue
            pooled[index] = np.asarray(embeddings[rows], dtype=np.float32).mean(axis=0)
            presence[index, 0] = 1.0
        requested = base_dimension + int(role_index < remainder)
        dimension = min(requested, int(train.sum()) - 1, pooled.shape[1])
        reducer = PCA(
            n_components=dimension,
            svd_solver="randomized",
            random_state=seed + role_index,
        )
        reducer.fit(pooled[train])
        reduced = reducer.transform(pooled).astype(np.float64, copy=False)
        text_parts.extend((presence, reduced))
        names.append(f"{role}|present")
        names.extend(f"{role}|pca_{index + 1}" for index in range(dimension))
    text = np.column_stack(text_parts)
    return FeatureBlocks(
        hadm_ids=feature_blocks.hadm_ids,
        subject_ids=feature_blocks.subject_ids,
        splits=feature_blocks.splits,
        blocks={**feature_blocks.blocks, "text": text},
        feature_names={**feature_blocks.feature_names, "text": tuple(names)},
    )


def _aligned_targets(
    store: ConditionalOutcomeStore,
    hadm_ids: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = [store[int(hadm)] for hadm in hadm_ids]
    return (
        np.array([value.interval_labels for value in values], dtype=np.int64),
        np.array([value.interval_mask for value in values], dtype=bool),
        np.array([value.tail_label for value in values], dtype=np.int64),
        np.array([value.tail_mask for value in values], dtype=bool),
    )


def _standardize_from_train(
    values: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_values = values[train_mask]
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    kept = std > 1e-10
    if not kept.any():
        return np.zeros((values.shape[0], 0), dtype=np.float64), kept
    return (values[:, kept] - mean[kept]) / std[kept], kept


def _fit_competing_task(
    train_features: np.ndarray,
    eval_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    l2: float,
    anchor_train: np.ndarray | None = None,
    anchor_eval: np.ndarray | None = None,
    direct_anchor: bool = False,
) -> tuple[np.ndarray, int, int]:
    """拟合主要 AECOPD 与条件死亡两个二项头，返回合法三分类概率。"""

    if train_labels.size == 0:
        # 整个训练风险集为空时不存在可估计目标。返回显式无信息概率，
        # 同时把两个头标为未收敛，使该阶段不能进入保留门控。
        return np.full((eval_features.shape[0], 3), 1.0 / 3.0), 0, 2

    if anchor_train is None:
        a_offset_train = None
        a_offset_eval = None
        d_offset_train = None
        d_offset_eval = None
    else:
        anchor_a = fit_binary_logit(
            anchor_train,
            train_labels == 1,
            l2=1e-8,
            nonnegative_first_coefficient=True,
        )
        non_aecopd = train_labels != 1
        a_offset_train = anchor_a.decision_function(anchor_train)
        a_offset_eval = anchor_a.decision_function(anchor_eval)
        if non_aecopd.any():
            anchor_d = fit_binary_logit(
                anchor_train[non_aecopd],
                train_labels[non_aecopd] == 2,
                l2=1e-8,
            )
            d_offset_train = anchor_d.decision_function(anchor_train[non_aecopd])
            d_offset_eval = anchor_d.decision_function(anchor_eval)
        else:
            d_offset_train = None
            d_offset_eval = None

    aecopd = fit_binary_logit(
        train_features,
        train_labels == 1,
        l2=l2,
        offset=a_offset_train,
        nonnegative_first_coefficient=direct_anchor,
    )
    non_aecopd = train_labels != 1
    aecopd_logits = aecopd.decision_function(eval_features, offset=a_offset_eval)
    if non_aecopd.any():
        death = fit_binary_logit(
            train_features[non_aecopd],
            train_labels[non_aecopd] == 2,
            l2=l2,
            offset=d_offset_train,
        )
        death_logits = death.decision_function(eval_features, offset=d_offset_eval)
        death_converged = int(death.converged)
    else:
        # 该训练风险集全部为 AECOPD 时，条件死亡没有可学习样本；明确回退
        # 到接近零的条件概率，而不是对空矩阵调用优化器。该头不可估计，
        # 因此不能伪装成已收敛并让阶段进入保留门控。
        death_logits = np.full(eval_features.shape[0], -20.0, dtype=np.float64)
        death_converged = 0
    return (
        compose_competing_probabilities(aecopd_logits, death_logits),
        int(aecopd.converged) + death_converged,
        2,
    )


def fit_stage(
    feature_blocks: FeatureBlocks,
    interval_labels: np.ndarray,
    interval_mask: np.ndarray,
    tail_labels: np.ndarray,
    tail_mask: np.ndarray,
    *,
    name: str,
    label: str,
    blocks: tuple[str, ...],
    residual_anchor: bool,
    l2: float,
) -> StagePrediction:
    """在训练划分拟合一个预声明层级，并只对验证划分产生预测。"""

    train_mask = feature_blocks.splits == "train"
    eval_mask = feature_blocks.splits == "validation"
    raw = np.column_stack([feature_blocks.blocks[block] for block in blocks])
    values, kept = _standardize_from_train(raw, train_mask)
    feature_count = int(kept.sum())
    anchor = feature_blocks.blocks["anchor"]
    eval_count = int(eval_mask.sum())
    interval_probabilities = np.zeros((eval_count, 3, 3), dtype=np.float64)
    converged_heads = 0
    total_heads = 0
    for interval_index in range(3):
        active_train = train_mask & interval_mask[:, interval_index]
        probabilities, converged, heads = _fit_competing_task(
            values[active_train],
            values[eval_mask],
            interval_labels[active_train, interval_index],
            l2=l2,
            anchor_train=anchor[active_train] if residual_anchor else None,
            anchor_eval=anchor[eval_mask] if residual_anchor else None,
            direct_anchor=(name == "B0_prior_count"),
        )
        interval_probabilities[:, interval_index] = probabilities
        converged_heads += converged
        total_heads += heads

    active_tail_train = train_mask & tail_mask
    tail_probabilities, converged, heads = _fit_competing_task(
        values[active_tail_train],
        values[eval_mask],
        tail_labels[active_tail_train],
        l2=l2,
        anchor_train=anchor[active_tail_train] if residual_anchor else None,
        anchor_eval=anchor[eval_mask] if residual_anchor else None,
        direct_anchor=(name == "B0_prior_count"),
    )
    converged_heads += converged
    total_heads += heads
    return StagePrediction(
        name=name,
        label=label,
        interval_probabilities=interval_probabilities,
        tail_probabilities=tail_probabilities,
        feature_count=feature_count,
        converged_heads=converged_heads,
        total_heads=total_heads,
    )


def _fast_binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    negatives = y.size - positives
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(s, kind="stable")
    sorted_scores = s[order]
    ranks = np.empty(y.size, dtype=np.float64)
    start = 0
    while start < y.size:
        stop = start + 1
        while stop < y.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _patient_equal_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    subject_ids: np.ndarray,
) -> float:
    """让风险集中的每位患者总权重相同，作为患者等权敏感性 AUC。"""

    from sklearn.metrics import roc_auc_score

    y = np.asarray(labels, dtype=np.int64)
    if np.unique(y).size < 2:
        return math.nan
    subjects = np.asarray(subject_ids)
    _, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return float(roc_auc_score(y, scores, sample_weight=weights))


def patient_equal_interval_aucs(
    probabilities: np.ndarray,
    interval_labels: np.ndarray,
    interval_mask: np.ndarray,
    subject_ids: np.ndarray,
) -> dict[str, float]:
    """计算三条件风险集的患者等权 AECOPD AUC。"""

    output: dict[str, float] = {}
    for interval_index, interval_name in enumerate(INTERVAL_NAMES):
        active = interval_mask[:, interval_index]
        output[interval_name] = _patient_equal_auc(
            interval_labels[active, interval_index] == 1,
            probabilities[active, interval_index, 1],
            subject_ids[active],
        )
    return output


def paired_patient_bootstrap_auc_delta(
    current: np.ndarray,
    previous: np.ndarray,
    interval_labels: np.ndarray,
    interval_mask: np.ndarray,
    subject_ids: np.ndarray,
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """按患者聚类、配对重采样三个 AECOPD 区间 AUC 差和均值差。"""

    subjects, inverse = np.unique(subject_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(subjects.size)]

    def deltas(index: np.ndarray) -> np.ndarray:
        result = np.full(3, np.nan, dtype=np.float64)
        for interval_index in range(3):
            active = interval_mask[index, interval_index]
            labels = interval_labels[index, interval_index][active] == 1
            current_auc = _fast_binary_auc(labels, current[index, interval_index, 1][active])
            previous_auc = _fast_binary_auc(labels, previous[index, interval_index, 1][active])
            result[interval_index] = current_auc - previous_auc
        return result

    full_index = np.arange(subject_ids.size)
    point = deltas(full_index)
    rng = np.random.default_rng(seed)
    samples = np.full((n_resamples, 4), np.nan, dtype=np.float64)
    for resample in range(n_resamples):
        chosen = rng.integers(0, subjects.size, size=subjects.size)
        rows = np.concatenate([groups[index] for index in chosen])
        interval_delta = deltas(rows)
        samples[resample, :3] = interval_delta
        # “三窗口平均差”必须始终是同一个估计目标；任何窗口在本次患者
        # 重采样中退化为单类，就丢弃该次均值，而不是改成两窗或单窗均值。
        if np.isfinite(interval_delta).all():
            samples[resample, 3] = float(np.mean(interval_delta))
    point_all = np.append(point, float(np.mean(point)))
    alpha = (1.0 - confidence) / 2.0
    names = (*INTERVAL_NAMES, "mean")
    output: dict[str, Any] = {}
    for index, metric_name in enumerate(names):
        finite = samples[np.isfinite(samples[:, index]), index]
        output[metric_name] = {
            "point": float(point_all[index]),
            "lower": float(np.quantile(finite, alpha)) if finite.size else None,
            "upper": float(np.quantile(finite, 1.0 - alpha)) if finite.size else None,
            "valid_resamples": int(finite.size),
        }
    return output


def run_incremental_evidence_chain(config: IncrementalEvidenceConfig) -> dict[str, Any]:
    """运行 B0→B5 验证实验并返回不含患者标识的聚合结果。"""

    split_summary = config.split.validate_no_patient_leakage()
    feature_blocks = build_feature_blocks(config)
    validate_feature_split_contract(feature_blocks, config.split)
    if config.text_dir is not None:
        feature_blocks = add_text_block(
            feature_blocks,
            text_dir=config.text_dir,
            pca_dim=config.text_pca_dim,
            seed=config.seed,
        )

    store = ConditionalOutcomeStore.from_cohort(config.data_dir / "cohort_labels_static.csv.gz")
    interval_labels, interval_mask, tail_labels, tail_mask = _aligned_targets(
        store, feature_blocks.hadm_ids
    )
    eval_mask = feature_blocks.splits == config.eval_split
    eval_interval_labels = interval_labels[eval_mask]
    eval_interval_mask = interval_mask[eval_mask]
    eval_tail_labels = tail_labels[eval_mask]
    eval_tail_mask = tail_mask[eval_mask]
    eval_subjects = feature_blocks.subject_ids[eval_mask]

    predictions: list[StagePrediction] = []
    stages: list[dict[str, Any]] = []

    def evaluate_candidate(
        name: str,
        label: str,
        blocks: tuple[str, ...],
        *,
        residual_anchor: bool,
        l2: float,
        reference_index: int | None,
    ) -> bool:
        """拟合候选并与当前保留层配对；返回是否有稳定平均 AUC 增量。"""

        stage_index = len(predictions)
        prediction = fit_stage(
            feature_blocks,
            interval_labels,
            interval_mask,
            tail_labels,
            tail_mask,
            name=name,
            label=label,
            blocks=blocks,
            residual_anchor=residual_anchor,
            l2=l2,
        )
        metrics, _ = conditional_metrics(
            prediction.interval_probabilities,
            eval_interval_labels,
            eval_interval_mask,
            prediction.tail_probabilities,
            eval_tail_labels,
            eval_tail_mask,
            prefix=config.eval_split,
        )
        aucs = {
            interval_name: metrics[f"{config.eval_split}_aecopd_auc_{interval_name}"]
            for interval_name in INTERVAL_NAMES
        }
        patient_equal_aucs = patient_equal_interval_aucs(
            prediction.interval_probabilities,
            eval_interval_labels,
            eval_interval_mask,
            eval_subjects,
        )
        stage_result: dict[str, Any] = {
            "name": name,
            "label": label,
            "input_blocks": list(blocks),
            "residual_anchor": residual_anchor,
            "feature_count_after_constant_filter": prediction.feature_count,
            "converged_heads": prediction.converged_heads,
            "total_heads": prediction.total_heads,
            "aecopd_auc": aucs,
            "aecopd_mean_auc": float(np.mean(list(aucs.values()))),
            "aecopd_auc_patient_equal": patient_equal_aucs,
            "aecopd_mean_auc_patient_equal": float(
                np.mean(list(patient_equal_aucs.values()))
            ),
            "metrics": metrics,
        }
        exclusion_reason = stage_gate_exclusion_reason(prediction, aucs)
        stage_result["gate_eligible"] = exclusion_reason is None
        stage_result["gate_exclusion_reason"] = exclusion_reason
        supported = False
        if reference_index is not None:
            delta = paired_patient_bootstrap_auc_delta(
                prediction.interval_probabilities,
                predictions[reference_index].interval_probabilities,
                eval_interval_labels,
                eval_interval_mask,
                eval_subjects,
                n_resamples=config.bootstrap_resamples,
                confidence=config.confidence,
                seed=config.seed + stage_index,
            )
            supported = bool(
                exclusion_reason is None
                and delta["mean"]["lower"] is not None
                and delta["mean"]["lower"] > 0.0
            )
            stage_result["reference_stage"] = predictions[reference_index].name
            stage_result["paired_delta_vs_reference"] = delta
            stage_result["supported_increment_vs_reference"] = supported
            # 兼容第一版聚合字段；语义已改为相对当前保留层，而非机械上一级。
            stage_result["paired_delta_vs_previous"] = delta
            stage_result["supported_increment_vs_previous"] = supported
        if stage_index > 0:
            stage_result["paired_delta_vs_anchor"] = paired_patient_bootstrap_auc_delta(
                prediction.interval_probabilities,
                predictions[0].interval_probabilities,
                eval_interval_labels,
                eval_interval_mask,
                eval_subjects,
                n_resamples=config.bootstrap_resamples,
                confidence=config.confidence,
                seed=config.seed + 100 + stage_index,
            )
        predictions.append(prediction)
        stages.append(stage_result)
        if reference_index is None and exclusion_reason is not None:
            raise RuntimeError(
                f"baseline stage {name!r} is not eligible for comparison: "
                f"{exclusion_reason}"
            )
        return supported

    evaluate_candidate(
        "B0_prior_count",
        "既往 AECOPD 次数",
        ("anchor",),
        residual_anchor=False,
        l2=1e-8,
        reference_index=None,
    )
    retained_indices = {0}
    # B0 自身是直接单变量模型；在它仍为 incumbent 时，新增信息以 B0
    # logit 为固定 offset，避免重新估计时悄悄改写锚点。
    retained = RetainedModel(index=0, blocks=(), residual_anchor=True)

    b1_supported = evaluate_candidate(
        "B1_patient_state_core",
        "低容量患者状态核心",
        ("anchor", "core"),
        residual_anchor=False,
        l2=config.residual_l2,
        reference_index=0,
    )
    retained = advance_retained_model(
        retained,
        RetainedModel(index=1, blocks=("anchor", "core"), residual_anchor=False),
        supported=b1_supported,
    )
    if b1_supported:
        retained_indices.add(1)

    core_supported = evaluate_candidate(
        "B2_anchored_core",
        "锚定患者状态残差",
        ("core",),
        residual_anchor=True,
        l2=config.residual_l2,
        reference_index=retained.index,
    )
    retained = advance_retained_model(
        retained,
        RetainedModel(index=2, blocks=("core",), residual_anchor=True),
        supported=core_supported,
    )
    if core_supported:
        retained_indices.add(2)

    layer_contracts = [
        ("B3", "trajectory", "患者状态数值轨迹"),
        ("B4", "care_process", "诊疗行为"),
    ]
    if "text" in feature_blocks.blocks:
        layer_contracts.append(("B5", "text", "Qwen 文本"))
    for prefix, block, _block_label in layer_contracts:
        candidate_blocks = (*retained.blocks, block)
        suffix = "_".join(candidate_blocks)
        supported = evaluate_candidate(
            f"{prefix}_{'anchored' if retained.residual_anchor else 'direct'}_{suffix}",
            ("锚定风险 + " if retained.residual_anchor else "") + " + ".join(
                {
                    "anchor": "既往 AECOPD 次数",
                    "core": "患者状态核心",
                    "trajectory": "患者状态数值轨迹",
                    "care_process": "诊疗行为",
                    "text": "Qwen 文本",
                }[name]
                for name in candidate_blocks
            ),
            candidate_blocks,
            residual_anchor=retained.residual_anchor,
            l2=config.residual_l2,
            reference_index=retained.index,
        )
        candidate_index = len(predictions) - 1
        retained = advance_retained_model(
            retained,
            RetainedModel(
                index=candidate_index,
                blocks=candidate_blocks,
                residual_anchor=retained.residual_anchor,
            ),
            supported=supported,
        )
        if supported:
            retained_indices.add(candidate_index)

    for index, stage in enumerate(stages):
        stage["retained_in_chain"] = index in retained_indices

    artifacts = json.loads(config.artifacts_path.read_text(encoding="utf-8"))
    return {
        "schema_version": "aecopd-incremental-evidence-chain-1",
        "selection_split": config.eval_split,
        "test_set_used": False,
        "estimand": "admission_equal",
        "sensitivity_estimand": "patient_equal",
        "split_patients": split_summary,
        "train_admissions": int(np.sum(feature_blocks.splits == "train")),
        "validation_admissions": int(eval_mask.sum()),
        "validation_patients": int(np.unique(eval_subjects).size),
        "clinical_feature_contract": artifacts.get("clinical_feature_contract"),
        "residual_l2": config.residual_l2,
        "text_pca_dim": config.text_pca_dim if "text" in feature_blocks.blocks else None,
        "bootstrap_resamples": config.bootstrap_resamples,
        "confidence": config.confidence,
        "seed": config.seed,
        "feature_block_widths": {
            name: int(values.shape[1]) for name, values in feature_blocks.blocks.items()
        },
        "stages": stages,
    }
