"""B1：HAIM 风格章节特征融合 + XGBoost（必需浅层基线，落地实现）。

设计对齐 docs/BASELINE_COMPARISON_PLAN.md B1 与 HAIM 框架
(Soenksen et al., Integrated multimodal AI framework, NPJ Digit Med 2022)。

核心思想：**继续使用相同的冻结 Qwen 章节向量**，但不训练 Transformer。
文本向量经"只在训练集拟合"的确定性池化/降维形成固定文本特征；结构化长表
转成概念×统计摘要的固定特征；二者与低维临床静态特征拼成一行一次索引住院的
固定向量，用 XGBoost 估计每个互斥区间的三分类条件概率。这样才能公平检验
"复杂纵向 Transformer 是否真的比浅层融合多带来价值"。

关键口径：
- 池化与顺序无关（mean+max），PCA/概念词表/缩放只在 split=='train' 上 fit；
- 区间条件风险集直接复用 ConditionalOutcomeStore 的 interval_mask，不自行切分；
- 世界观 A 标签（出院后数据库再无记录=阴性），与主模型逐 hadm 一致；
- 输出与主模型同形状：interval [N,3,3]、tail [N,3]，评估复用 conditional_metrics。
"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .shared_contract import (
    BaselineEvalResult,
    SplitContract,
    evaluate_with_project_metrics,
)
from . import b2_cause_specific_cox as b2  # 复用低维临床静态特征构造


@dataclass(frozen=True)
class HaimXgbConfig:
    """B1 配置。降维/池化/缩放的所有统计量只允许在训练集拟合。"""

    section_embedding_dir: Path        # model_v2 冻结章节向量目录
    events_path: Path                  # model_v3 结构化事件长表
    cohort_csv_gz_path: Path           # model_v3 cohort_labels_static（键、划分、标签、静态特征）
    split: SplitContract
    text_reduce_dim: int = 256         # 训练集拟合 PCA/SVD；池化后 mean+max=5120 → 256
    concept_min_train_count: int = 50  # 结构化概念入词表的最小训练出现次数
    xgb_num_boost_round: int = 300
    xgb_max_depth: int = 5
    xgb_eta: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample: float = 0.8
    random_state: int = 2026


# ----------------------------------------------------------------------------
# 文本特征：章节向量池化 + 训练集 PCA
# ----------------------------------------------------------------------------
def _load_section_index(section_dir: Path) -> dict[int, list[int]]:
    """hadm_id → 该住院全部章节向量在 .npy 中的行号列表。"""
    index_path = section_dir / "qwen_section_embedding_index.csv.gz"
    rows_by_hadm: dict[int, list[int]] = defaultdict(list)
    with gzip.open(index_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_hadm[int(row["hadm_id"])].append(int(row["embedding_row"]))
    return rows_by_hadm


def _pool_hadm_vectors(embeddings: np.ndarray, rows: list[int]) -> np.ndarray:
    """确定性池化：mean 与 max 拼接（顺序无关），无章节时返回零向量。"""
    dim = embeddings.shape[1]
    if not rows:
        return np.zeros(2 * dim, dtype=np.float64)
    block = np.asarray(embeddings[sorted(rows)], dtype=np.float64)
    return np.concatenate([block.mean(axis=0), block.max(axis=0)])


@dataclass
class _PCAStats:
    mean: np.ndarray
    components: np.ndarray  # [reduce_dim, pooled_dim]


def _fit_pca(pooled_train: np.ndarray, reduce_dim: int) -> _PCAStats:
    """训练集池化向量上拟合 PCA（SVD）。reduce_dim 受样本/维度上限约束。"""
    mean = pooled_train.mean(axis=0)
    centered = pooled_train - mean
    # 经济 SVD；取前 k 个右奇异向量为主成分。
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(reduce_dim, vt.shape[0])
    return _PCAStats(mean=mean, components=vt[:k])


def _apply_pca(pooled: np.ndarray, stats: _PCAStats) -> np.ndarray:
    return (pooled - stats.mean) @ stats.components.T


def build_text_features(
    section_dir: Path,
    hadm_order: np.ndarray,
    rows_by_hadm: dict[int, list[int]],
    embeddings: np.ndarray,
) -> np.ndarray:
    """按给定 hadm 顺序聚合章节向量为 [N, 2*dim] 池化矩阵（未降维）。"""
    pooled = np.zeros((hadm_order.shape[0], 2 * embeddings.shape[1]), dtype=np.float64)
    for i, hadm in enumerate(hadm_order):
        pooled[i] = _pool_hadm_vectors(embeddings, rows_by_hadm.get(int(hadm), []))
    return pooled


# ----------------------------------------------------------------------------
# 结构化特征：概念级 count + mean(value_scaled)，训练集拟合词表
# ----------------------------------------------------------------------------
@dataclass
class _StructuredVocab:
    concepts: tuple[str, ...]
    index: dict[str, int]


def _concept_key(row: dict[str, str]) -> str:
    return f"{row.get('event_domain','')}|{row.get('concept_name','')}"


def _aggregate_events(events_path: Path) -> dict[int, dict[str, tuple[int, float]]]:
    """流式聚合：hadm → concept → (count, sum_value_scaled)。只读 main_model 范围事件。"""
    acc: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    with gzip.open(events_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model_scope") not in ("main_model", "main_model_encoder"):
                continue
            hadm = int(row["hadm_id"])
            key = _concept_key(row)
            cell = acc[hadm][key]
            cell[0] += 1.0
            vs = row.get("value_scaled")
            if vs not in (None, ""):
                try:
                    cell[1] += float(vs)
                except ValueError:
                    pass
    return {h: {k: (int(v[0]), v[1]) for k, v in concepts.items()} for h, concepts in acc.items()}


def _fit_structured_vocab(
    aggregates: dict[int, dict[str, tuple[int, float]]],
    train_hadms: set[int],
    min_count: int,
) -> _StructuredVocab:
    counts: dict[str, int] = defaultdict(int)
    for hadm in train_hadms:
        for key, (cnt, _s) in aggregates.get(hadm, {}).items():
            counts[key] += cnt
    concepts = tuple(sorted(k for k, c in counts.items() if c >= min_count))
    return _StructuredVocab(concepts=concepts, index={k: i for i, k in enumerate(concepts)})


def build_structured_features(
    hadm_order: np.ndarray,
    aggregates: dict[int, dict[str, tuple[int, float]]],
    vocab: _StructuredVocab,
) -> np.ndarray:
    """每 hadm → [count_c ... , mean_scaled_c ...]，未见概念置 0。"""
    p = len(vocab.concepts)
    out = np.zeros((hadm_order.shape[0], 2 * p), dtype=np.float64)
    for i, hadm in enumerate(hadm_order):
        for key, (cnt, ssum) in aggregates.get(int(hadm), {}).items():
            j = vocab.index.get(key)
            if j is None:
                continue
            out[i, j] = cnt
            out[i, p + j] = ssum / cnt if cnt else 0.0
    return out


# ----------------------------------------------------------------------------
# XGBoost 头（multi:softprob, num_class=3 固定）
# ----------------------------------------------------------------------------
def _train_head(
    features: np.ndarray, labels: np.ndarray, config: HaimXgbConfig
):
    import xgboost as xgb

    dtrain = xgb.DMatrix(features, label=labels.astype(np.float64))
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": config.xgb_max_depth,
        "eta": config.xgb_eta,
        "subsample": config.xgb_subsample,
        "colsample_bytree": config.xgb_colsample,
        "seed": config.random_state,
        "verbosity": 0,
        "tree_method": "hist",
    }
    return xgb.train(params, dtrain, num_boost_round=config.xgb_num_boost_round)


def _predict_head(booster, features: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    proba = booster.predict(xgb.DMatrix(features))
    return np.asarray(proba, dtype=np.float64).reshape(features.shape[0], 3)


# ----------------------------------------------------------------------------
# 端到端
# ----------------------------------------------------------------------------
def _read_cohort_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _split_hadm_order(rows: list[dict[str, str]], split: str) -> np.ndarray:
    return np.array([int(r["hadm_id"]) for r in rows if r["data_split"] == split], dtype=np.int64)


def run_b1(
    config: HaimXgbConfig, eval_split: str = "validation"
) -> tuple[BaselineEvalResult, dict]:
    """B1 端到端：训练折拟合特征与 XGBoost → 指定折预测评估（默认 validation）。

    返回 (BaselineEvalResult, predictions dict)；predictions 供患者聚类 bootstrap 复用。
    """
    from aecopd_training.conditional_outcomes import ConditionalOutcomeStore

    config.split.validate_no_patient_leakage()
    all_rows = _read_cohort_rows(config.cohort_csv_gz_path)
    train_order = _split_hadm_order(all_rows, "train")
    eval_order = _split_hadm_order(all_rows, eval_split)
    train_hadms = set(int(h) for h in train_order)

    # --- 文本特征（章节向量池化 + 训练集 PCA） ---
    rows_by_hadm = _load_section_index(config.section_embedding_dir)
    embeddings = np.load(
        config.section_embedding_dir / "qwen_section_embeddings.float16.npy", mmap_mode="r"
    )
    pooled_train = build_text_features(config.section_embedding_dir, train_order, rows_by_hadm, embeddings)
    pca = _fit_pca(pooled_train, config.text_reduce_dim)
    text_train = _apply_pca(pooled_train, pca)
    pooled_eval = build_text_features(config.section_embedding_dir, eval_order, rows_by_hadm, embeddings)
    text_eval = _apply_pca(pooled_eval, pca)

    # --- 结构化特征（概念聚合，训练集词表） ---
    aggregates = _aggregate_events(config.events_path)
    vocab = _fit_structured_vocab(aggregates, train_hadms, config.concept_min_train_count)
    struct_train = build_structured_features(train_order, aggregates, vocab)
    struct_eval = build_structured_features(eval_order, aggregates, vocab)

    # --- 低维临床静态特征（复用 B2 的构造，训练集标准化） ---
    train_rows = [r for r in all_rows if r["data_split"] == "train"]
    stats = b2._fit_standardizer(train_rows)
    static_train, _ = b2._build_feature_matrix([r for r in all_rows if r["data_split"] == "train"], stats)
    static_eval, _ = b2._build_feature_matrix([r for r in all_rows if r["data_split"] == eval_split], stats)

    x_train = np.hstack([text_train, struct_train, static_train])
    x_eval = np.hstack([text_eval, struct_eval, static_eval])

    # --- 标签（复用 ConditionalOutcomeStore，按各折 hadm 顺序对齐） ---
    store = ConditionalOutcomeStore.from_cohort(config.cohort_csv_gz_path)

    def labels_for(order: np.ndarray):
        il = np.array([store[int(h)].interval_labels for h in order], dtype=np.int64)
        im = np.array([store[int(h)].interval_mask for h in order], dtype=bool)
        tl = np.array([store[int(h)].tail_label for h in order], dtype=np.int64)
        tm = np.array([store[int(h)].tail_mask for h in order], dtype=bool)
        return il, im, tl, tm

    il_tr, im_tr, tl_tr, tm_tr = labels_for(train_order)
    il_ev, im_ev, tl_ev, tm_ev = labels_for(eval_order)

    n_eval = eval_order.shape[0]
    interval_probs = np.zeros((n_eval, 3, 3))
    for i in range(3):
        active = im_tr[:, i]
        booster = _train_head(x_train[active], il_tr[active, i], config)
        interval_probs[:, i, :] = _predict_head(booster, x_eval)

    tail_booster = _train_head(x_train[tm_tr], tl_tr[tm_tr], config)
    tail_probs = _predict_head(tail_booster, x_eval)

    metrics, _ = evaluate_with_project_metrics(
        interval_probs, il_ev, im_ev, tail_probs, tl_ev, tm_ev, prefix=eval_split
    )
    result = BaselineEvalResult(model_name="B1_haim_xgboost")
    for interval_name in ("days_0_30", "days_31_90", "days_91_365"):
        result.interval_aecopd_auc[interval_name] = metrics.get(f"{eval_split}_aecopd_auc_{interval_name}", float("nan"))
        result.interval_aecopd_brier[interval_name] = metrics.get(f"{eval_split}_aecopd_brier_{interval_name}", float("nan"))
        result.interval_death_auc[interval_name] = metrics.get(f"{eval_split}_death_auc_{interval_name}", float("nan"))
        result.interval_death_brier[interval_name] = metrics.get(f"{eval_split}_death_brier_{interval_name}", float("nan"))
    result.tail_accuracy = metrics.get(f"{eval_split}_tail_accuracy", float("nan"))
    result.tail_none_observed_auc = metrics.get(f"{eval_split}_tail_none_observed_auc", float("nan"))
    result.tail_aecopd_auc = metrics.get(f"{eval_split}_tail_aecopd_auc", float("nan"))
    result.tail_death_auc = metrics.get(f"{eval_split}_tail_death_auc", float("nan"))
    result.notes = (
        f"HAIM-style: section-vector mean+max pooling -> train PCA({config.text_reduce_dim}); "
        f"concept count+mean(value_scaled) (|vocab|={len(vocab.concepts)}) + low-dim clinical static; "
        f"4x XGBoost multi:softprob; World A labels; eval_split={eval_split}; "
        f"n_train={train_order.shape[0]}, n_eval={n_eval}, n_features={x_train.shape[1]}"
    )
    # 供 bootstrap 复用的预测与键。
    result_predictions = {
        "interval_probabilities": interval_probs,
        "tail_probabilities": tail_probs,
        "hadm_ids": eval_order,
        "subject_ids": np.array(
            [int(r["subject_id"]) for r in all_rows if r["data_split"] == eval_split], dtype=np.int64
        ),
        "interval_labels": il_ev,
        "interval_mask": im_ev,
        "tail_labels": tl_ev,
        "tail_mask": tm_ev,
    }
    return result, result_predictions
