"""把已审计的住院级异常恢复特征按 hadm_id 对齐到固定 V5 队列。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from patient_data.lab_recovery_features import FEATURE_NAMES
from patient_data.expanded_abnormality_features import FEATURE_NAMES as ABNORMALITY_FEATURE_NAMES
from patient_data.expanded_abnormality_features import AXES as ABNORMALITY_AXES
from training.v5.event_data import EventData


ABNORMALITY_GROUPS = {
    "ventilation_acid_base": ("arterial_low_ph", "arterial_high_pco2", "arterial_hypercapnic_acidosis",
                              "bicarbonate_low", "bicarbonate_high"),
    "inflammation_infection": ("wbc_high", "wbc_low", "crp_high"),
    "nutrition_hematology": ("albumin_low", "hemoglobin_low", "platelet_low"),
    "renal": ("creatinine_high", "urea_nitrogen_high"),
    "electrolytes": ("sodium_low", "sodium_high", "potassium_low", "potassium_high", "magnesium_low"),
    "metabolic_perfusion": ("glucose_low", "glucose_high", "lactate_high"),
    "cardiac_burden": ("ntprobnp_high",),
}
SELECTED_REMOVED_AXES = (*ABNORMALITY_GROUPS["renal"], *ABNORMALITY_GROUPS["metabolic_perfusion"])
# 临床先筛：这些轴不是 AECOPD 核心指标，或与其他轴明显重合；是否再删由严格结局单轴实验决定。
SELECTED_AXIS_SCREEN = ("arterial_hypercapnic_acidosis", "platelet_low", "magnesium_low", "ntprobnp_high")
ABNORMALITY_ABLATIONS = {
    **ABNORMALITY_GROUPS,
    # 完整词表逐轴留一；不同于旧 selected 基线上的四项追加筛选。
    **{f"axis_{axis}": (axis,) for axis in ABNORMALITY_AXES},
    "renal_metabolic_perfusion": SELECTED_REMOVED_AXES,
    **{f"renal_metabolic_perfusion_plus_{axis}": (*SELECTED_REMOVED_AXES, axis)
       for axis in SELECTED_AXIS_SCREEN},
}
CORE_RISK_GROUPS = ("ventilation_acid_base", "inflammation_infection", "renal", "electrolytes")
if set().union(*map(set, ABNORMALITY_GROUPS.values())) != set(ABNORMALITY_AXES) or sum(
        len(group) for group in ABNORMALITY_GROUPS.values()) != len(ABNORMALITY_AXES):
    raise ValueError("abnormality groups must partition the fixed axis vocabulary")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attach_recovery(data: EventData, directory: Path, *, mode: str) -> EventData:
    if mode not in ("active", "masked"):
        raise ValueError("clinical recovery mode must be active or masked")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "lab_recovery.npz"
    if (manifest["status"] != "derived_not_yet_validated_by_prediction"
            or _sha(path) != manifest["output_sha256"]
            or manifest["input_sha256"]["processed_arrays"] != data.source_hashes["event_arrays"]):
        raise ValueError("recovery source contract mismatch")
    with np.load(path, allow_pickle=False) as archive:
        hadms = archive["hadm_ids"]
        features = archive["features"]
        names = archive["feature_names"]
    if (tuple(names.tolist()) != FEATURE_NAMES or features.shape != (len(hadms), len(FEATURE_NAMES))
            or len(set(hadms.tolist())) != len(hadms)
            or set(hadms.tolist()) != set(data.base.hadm_ids.tolist())
            or not np.isfinite(features).all() or (features < 0).any()):
        raise ValueError("recovery feature identity, shape or value mismatch")
    index = {int(hadm): i for i, hadm in enumerate(hadms)}
    aligned = features[[index[int(hadm)] for hadm in data.base.hadm_ids]].copy()
    if mode == "masked":
        aligned.fill(0.)
    return replace(data, recovery_by_admission=aligned)


def _scale_abnormalities_on_train(features: np.ndarray, splits: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """只用训练住院的阳性异常与观测跨度拟合正值尺度；0 和缺测 mask 不变。"""
    if len(splits) != len(features) or not np.any(splits == "train"):
        raise ValueError("training patients required for abnormality scaling")
    result = features.copy().reshape(len(features), -1, 6)
    parameters: dict[str, object] = {"method": "training_positive_p95", "axes": {}}
    for index, axis in enumerate(ABNORMALITY_FEATURE_NAMES[::6]):
        train = result[splits == "train", index]
        peak = train[train[:, 2] > 0, 2]
        span = train[train[:, 5] > 0, 5]
        severity_scale = float(np.percentile(peak, 95)) if peak.size else 1.0
        span_scale = float(np.percentile(span, 95)) if span.size else 1.0
        if not np.isfinite(severity_scale) or severity_scale <= 0 or not np.isfinite(span_scale) or span_scale <= 0:
            raise ValueError("invalid training abnormality scale")
        result[:, index, 1:4] = np.minimum(result[:, index, 1:4] / severity_scale, 5.0)
        result[:, index, 5] = np.minimum(result[:, index, 5] / span_scale, 5.0)
        parameters["axes"][axis.removesuffix("__observed")] = {
            "severity_log_p95_positive": severity_scale, "span_log_p95_positive": span_scale,
            "training_positive_peak": int(peak.size), "training_positive_span": int(span.size),
        }
    return result.reshape(features.shape), parameters


def attach_abnormalities(data: EventData, directory: Path, *, mode: str,
                         scale_mode: str = "none") -> EventData:
    """接入扩展异常特征；遮蔽臂保持完全相同的模型结构和参数量。"""
    if mode not in ("active", "masked", "measurement_only", "core9", "core_risk",
                    *(f"without_{group}" for group in ABNORMALITY_ABLATIONS)):
        raise ValueError("unknown abnormality mode")
    if scale_mode not in ("none", "training_positive_p95"):
        raise ValueError("unknown abnormality scale mode")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "abnormality.npz"
    if (manifest["status"] != "derived_not_yet_validated_by_prediction"
            or _sha(path) != manifest["output_sha256"]
            or manifest["input_sha256"]["processed_arrays"] != data.source_hashes["event_arrays"]):
        raise ValueError("abnormality source contract mismatch")
    with np.load(path, allow_pickle=False) as archive:
        hadms = archive["hadm_ids"]
        features = archive["features"]
        names = archive["feature_names"]
    if (tuple(names.tolist()) != ABNORMALITY_FEATURE_NAMES
            or features.shape != (len(hadms), len(ABNORMALITY_FEATURE_NAMES))
            or len(set(hadms.tolist())) != len(hadms)
            or set(hadms.tolist()) != set(data.base.hadm_ids.tolist())
            or not np.isfinite(features).all() or (features < 0).any()):
        raise ValueError("abnormality identity, shape or value mismatch")
    index = {int(hadm): i for i, hadm in enumerate(hadms)}
    aligned = features[[index[int(hadm)] for hadm in data.base.hadm_ids]].copy()
    scaling = None
    if scale_mode == "training_positive_p95":
        aligned, scaling = _scale_abnormalities_on_train(aligned, data.base.splits)
    if mode == "masked":
        aligned.fill(0.)
    elif mode == "core9":
        aligned[:, len(FEATURE_NAMES):] = 0.
    elif mode == "measurement_only":
        # 保留是否测过与观测跨度，仅遮蔽真正的异常程度／恢复内容。
        aligned.reshape(len(aligned), -1, 6)[:, :, 1:5] = 0.
    elif mode == "core_risk":
        shaped = aligned.reshape(len(aligned), -1, 6)
        kept = {axis for group in CORE_RISK_GROUPS for axis in ABNORMALITY_GROUPS[group]}
        for index, axis in enumerate(ABNORMALITY_AXES):
            if axis not in kept:
                shaped[:, index, :] = 0.
    elif mode.startswith("without_"):
        group = mode.removeprefix("without_")
        shaped = aligned.reshape(len(aligned), -1, 6)
        for axis in ABNORMALITY_ABLATIONS[group]:
            shaped[:, ABNORMALITY_AXES.index(axis), :] = 0.
    source_hashes = {**data.source_hashes, "abnormality_input": _sha(path)}
    if scaling is not None:
        source_hashes["abnormality_scaling"] = hashlib.sha256(
            json.dumps(scaling, sort_keys=True).encode("utf-8")).hexdigest()
    return replace(data, recovery_by_admission=aligned, abnormality_scaling=scaling,
                   source_hashes=source_hashes)
