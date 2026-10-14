"""将一次住院的结构化长表聚合为共享双锚点时间轴。

聚合会保留概念、域、单位、给药途径、证据类型和编码器标志，并用 25 个
统计通道描述同组事件。它不是把整次住院压成 25 个变量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from aecopd_loader import (
    STATIC_BINARY_FIELDS,
    STATIC_CATEGORICAL_FIELDS,
    STATIC_CONTINUOUS_FIELDS,
    SUMMARY_SCALED_FIELDS,
    AdmissionSample,
    EventInput,
    TextChunkInput,
)


TIMELINE_FEATURE_NAMES = (
    "count_log1p",
    "first_scaled",
    "last_scaled",
    "min_scaled",
    "max_scaled",
    "mean_scaled",
    "sd_scaled",
    "delta_scaled",
    "slope_scaled_per_hour",
    "hours_before_discharge_first_log1p",
    "hours_before_discharge_last_log1p",
    "observation_span_hours_log1p",
    "duration_hours_max_log1p",
    "normalization_available_fraction",
    "status_administered_fraction",
    "status_not_given_fraction",
    "status_flush_fraction",
    "status_not_flushed_fraction",
    "status_started_fraction",
    "status_stopped_fraction",
    "status_paused_fraction",
    "status_changed_fraction",
    "status_abnormal_fraction",
    "status_susceptible_fraction",
    "status_resistant_fraction",
)

WINDOW_NAMES = (
    "<PAD>",
    "入院第1天（0～24小时）",
    "入院第2天（24～48小时）",
    "入院第3天（48～72小时）",
    "入院第4天（72～96小时）",
    "入院第5天（96～120小时）",
    "入院第6天（120～144小时）",
    "入院第7天（144～168小时）",
    "长住院中间阶段",
    "出院前第7天（144～168小时）",
    "出院前第6天（120～144小时）",
    "出院前第5天（96～120小时）",
    "出院前第4天（72～96小时）",
    "出院前第3天（48～72小时）",
    "出院前第2天（24～48小时）",
    "出院前第1天（0～24小时）",
    "回顾性出院编码（无可靠发生时间）",
    "其他无可靠时间事件",
)

RETROSPECTIVE_EVIDENCE_TYPES = {
    "retrospective_discharge_code",
    "retrospective_procedure_code",
}

GLOBAL_SUMMARY_FEATURE_NAMES = (
    "measurement_count_log1p",
    *SUMMARY_SCALED_FIELDS,
    "normalization_available",
    "unseen_value_count_log1p",
)

HORIZON_LABEL_NAMES = (
    "aecopd_by_90d",
    "aecopd_by_180d",
    "aecopd_by_365d",
    "death_by_90d",
    "death_by_180d",
    "death_by_365d",
)

STATIC_CONTINUOUS_LAYERS = {
    "ed_hours": "care_process",
    "hospital_los_hours": "care_process",
    "icu_stay_count": "care_process",
    "icu_los_days": "care_process",
}
STATIC_CATEGORICAL_LAYERS = {
    "admission_type": "care_process",
    "admission_location": "care_process",
    "discharge_location": "care_process",
}
STATIC_BINARY_LAYERS = {
    "via_ed": "care_process",
    "entered_icu": "care_process",
}


@dataclass(frozen=True)
class TimelineConfig:
    """Dual-anchor daily timeline configuration for one admission."""

    max_tokens: int | None = 1280
    included_layers: tuple[str, ...] = ("patient_state", "care_process")


@dataclass(frozen=True)
class TimelineBatch:
    """一个补齐后的住院批次及其所有模型输入和审计元数据。"""

    timeline_feature_names: tuple[str, ...]
    timeline_window_names: tuple[str, ...]
    timeline_lengths: list[int]
    timeline_concept_ids: list[list[int]]
    timeline_domain_ids: list[list[int]]
    timeline_window_ids: list[list[int]]
    timeline_unit_ids: list[list[int]]
    timeline_route_ids: list[list[int]]
    timeline_evidence_ids: list[list[int]]
    timeline_encoder_flags: list[list[int]]
    timeline_features: list[list[list[float]]]
    timeline_feature_masks: list[list[list[bool]]]
    timeline_attention_mask: list[list[bool]]
    static_continuous_names: tuple[str, ...]
    static_continuous: list[list[float]]
    static_continuous_masks: list[list[bool]]
    static_categorical_names: tuple[str, ...]
    static_categorical: list[list[int]]
    static_binary_names: tuple[str, ...]
    static_binary: list[list[int]]
    global_summary_feature_names: tuple[str, ...]
    global_summary_lengths: list[int]
    global_summary_concept_ids: list[list[int]]
    global_summary_features: list[list[list[float]]]
    global_summary_feature_masks: list[list[list[bool]]]
    global_summary_attention_mask: list[list[bool]]
    horizon_label_names: tuple[str, ...]
    horizon_labels: list[list[int]]
    text_chunks: list[list[TextChunkInput]]
    subject_ids: list[int]
    hadm_ids: list[int]
    event_types: list[int]
    time_days: list[float]

    def to_torch(self, device: str | None = None) -> dict[str, object]:
        """Convert model-ready numeric fields to tensors using a lazy adapter."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required only for TimelineBatch.to_torch()") from exc

        batch_size = len(self.hadm_ids)
        timeline_width = max(self.timeline_lengths, default=0)
        summary_width = max(self.global_summary_lengths, default=0)

        def tensor(values: object, dtype: object) -> object:
            return torch.as_tensor(values, dtype=dtype, device=device)

        timeline_features = (
            torch.empty((batch_size, 0, len(self.timeline_feature_names)), dtype=torch.float32, device=device)
            if timeline_width == 0 else tensor(self.timeline_features, torch.float32)
        )
        timeline_feature_masks = (
            torch.empty((batch_size, 0, len(self.timeline_feature_names)), dtype=torch.bool, device=device)
            if timeline_width == 0 else tensor(self.timeline_feature_masks, torch.bool)
        )
        summary_features = (
            torch.empty((batch_size, 0, len(self.global_summary_feature_names)), dtype=torch.float32, device=device)
            if summary_width == 0 else tensor(self.global_summary_features, torch.float32)
        )
        summary_feature_masks = (
            torch.empty((batch_size, 0, len(self.global_summary_feature_names)), dtype=torch.bool, device=device)
            if summary_width == 0 else tensor(self.global_summary_feature_masks, torch.bool)
        )

        return {
            "timeline_concept_ids": tensor(self.timeline_concept_ids, torch.long),
            "timeline_domain_ids": tensor(self.timeline_domain_ids, torch.long),
            "timeline_window_ids": tensor(self.timeline_window_ids, torch.long),
            "timeline_unit_ids": tensor(self.timeline_unit_ids, torch.long),
            "timeline_route_ids": tensor(self.timeline_route_ids, torch.long),
            "timeline_evidence_ids": tensor(self.timeline_evidence_ids, torch.long),
            "timeline_encoder_flags": tensor(self.timeline_encoder_flags, torch.bool),
            "timeline_features": timeline_features,
            "timeline_feature_masks": timeline_feature_masks,
            "timeline_attention_mask": tensor(self.timeline_attention_mask, torch.bool),
            "static_continuous": tensor(self.static_continuous, torch.float32),
            "static_continuous_masks": tensor(self.static_continuous_masks, torch.bool),
            "static_categorical": tensor(self.static_categorical, torch.long),
            "static_binary": tensor(self.static_binary, torch.float32),
            "global_summary_concept_ids": tensor(self.global_summary_concept_ids, torch.long),
            "global_summary_features": summary_features,
            "global_summary_feature_masks": summary_feature_masks,
            "global_summary_attention_mask": tensor(self.global_summary_attention_mask, torch.bool),
            "event_types": tensor(self.event_types, torch.long),
            "time_days": tensor(self.time_days, torch.float32),
            "horizon_labels": tensor(self.horizon_labels, torch.float32),
            "subject_ids": tensor(self.subject_ids, torch.long),
            "hadm_ids": tensor(self.hadm_ids, torch.long),
        }


@dataclass
class _TokenAccumulator:
    events: list[EventInput]


@dataclass(frozen=True)
class _TimelineToken:
    concept_id: int
    domain_id: int
    window_id: int
    unit_id: int
    route_id: int
    evidence_id: int
    encoder_flag: int
    features: list[float]
    feature_masks: list[bool]


@dataclass(frozen=True)
class _GlobalSummaryRows:
    lengths: list[int]
    concept_ids: list[list[int]]
    features: list[list[list[float]]]
    feature_masks: list[list[list[bool]]]
    attention_mask: list[list[bool]]


def _canonical(value: str) -> str:
    value = value.strip().lower()
    return value if value else "<MISSING>"


def dual_anchor_window_id(hours_from_admission: float, hours_before_discharge: float) -> int:
    """Map a timestamp onto the shared admission/discharge 24-hour timeline."""
    admission_time_known = math.isfinite(hours_from_admission)
    discharge_time_known = math.isfinite(hours_before_discharge)

    # Give overlapping short stays to the admission anchor first. Consequently,
    # admissions lasting <=14 days retain every 24-hour slice without duplicate
    # events; only the middle of longer stays is temporally pooled.
    if admission_time_known:
        hours_from_admission = max(0.0, hours_from_admission)
        if hours_from_admission < 168.0:
            return min(7, int(hours_from_admission // 24.0) + 1)
    if discharge_time_known:
        hours_before_discharge = max(0.0, hours_before_discharge)
        if hours_before_discharge <= 168.0:
            return 15 - min(6, int(hours_before_discharge // 24.0))
    if admission_time_known or discharge_time_known:
        return 8
    return 17


def _window_id(event: EventInput) -> int:
    if event.evidence_type in RETROSPECTIVE_EVIDENCE_TYPES:
        return 16
    return dual_anchor_window_id(event.hours_from_admission, event.hours_before_discharge)


def _index(vocabulary: Sequence[str] | None) -> dict[str, int]:
    if vocabulary is None:
        return {}
    return {value: i for i, value in enumerate(vocabulary)}


def _lookup(index: Mapping[str, int], value: str) -> int:
    return index.get(value, 1)


def _sample_sd(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0.0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def _same_treatment_record(left: EventInput, right: EventInput) -> bool:
    """使用不入模的原始药物、途径和时间判定跨表重复记录。

    该规则故意偏保守：只删除高度疑似重复的低优先级来源，不会因为
    同一日窗出现一条 eMAR 就删掉其他时点的真实治疗。
    """
    if not left.dedup_identity or left.dedup_identity == "<MISSING>":
        return False
    if left.dedup_identity != right.dedup_identity:
        return False
    if (
        left.dedup_route
        and right.dedup_route
        and left.dedup_route != "<MISSING>"
        and right.dedup_route != "<MISSING>"
        and left.dedup_route != right.dedup_route
    ):
        return False
    admission_gap = abs(left.hours_from_admission - right.hours_from_admission)
    discharge_gap = abs(left.hours_before_discharge - right.hours_before_discharge)
    return admission_gap <= 1.0 or discharge_gap <= 1.0


def _deduplicate_treatment_events(events: Sequence[EventInput]) -> list[EventInput]:
    """保留所有非重复时点，对疑似重复记录优先保留更可靠来源。"""
    retained: list[EventInput] = []
    for event in sorted(
        events,
        key=lambda item: (-item.source_priority, item.hours_from_admission, item.event_id),
    ):
        duplicate_of_better_source = any(
            other.source_priority > event.source_priority
            and _same_treatment_record(event, other)
            for other in retained
        )
        if not duplicate_of_better_source:
            retained.append(event)
    return sorted(retained, key=lambda item: (item.hours_from_admission, item.event_id))


def _status_categories(status: str) -> set[str]:
    normalized = status.strip().lower()
    if not normalized:
        return set()
    categories: set[str] = set()
    if "not given" in normalized or "not administered" in normalized:
        categories.add("not_given")
    elif normalized == "administered" or normalized.startswith("administered ") or normalized in {"applied", "given", "infused"}:
        categories.add("administered")
    if "not flushed" in normalized:
        categories.add("not_flushed")
    elif "flushed" in normalized:
        categories.add("flush")
    if "started" in normalized:
        categories.add("started")
    if any(value in normalized for value in ("stopped", "removed", "discontinued")):
        categories.add("stopped")
    if "paused" in normalized:
        categories.add("paused")
    if "change" in normalized and any(value in normalized for value in ("dose", "rate")):
        categories.add("changed")
    if normalized == "abnormal":
        categories.add("abnormal")
    if normalized in {"s", "susceptible"}:
        categories.add("susceptible")
    if normalized in {"r", "resistant"}:
        categories.add("resistant")
    return categories


class SharedTimelineCollator:
    """Collapse raw events to padded concept-by-time-window batches.

    The public interface is deliberately framework-independent. Numeric arrays can
    be converted to PyTorch/JAX tensors by the training adapter without changing
    the grouping, masking, or leakage rules implemented here.
    """

    def __init__(
        self,
        config: TimelineConfig,
        *,
        artifacts: Mapping[str, object] | None = None,
    ) -> None:
        self.config = config
        artifacts = artifacts or {}
        self._domain_index = _index(artifacts.get("event_domain_vocabulary"))  # type: ignore[arg-type]
        self._unit_index = _index(artifacts.get("event_unit_vocabulary"))  # type: ignore[arg-type]
        self._route_index = _index(artifacts.get("event_route_vocabulary"))  # type: ignore[arg-type]
        self._evidence_index = _index(artifacts.get("event_evidence_vocabulary"))  # type: ignore[arg-type]

    def __call__(self, samples: Sequence[AdmissionSample]) -> TimelineBatch:
        token_rows = [self._aggregate_sample(sample) for sample in samples]
        if self.config.max_tokens is not None:
            for sample, row in zip(samples, token_rows):
                if len(row) > self.config.max_tokens:
                    raise ValueError(
                        f"timeline token count {len(row)} exceeds max_tokens={self.config.max_tokens} "
                        f"for hadm_id={sample.hadm_id}; revise aggregation instead of silently truncating"
                    )
        max_length = max((len(row) for row in token_rows), default=0)

        concept_ids: list[list[int]] = []
        domain_ids: list[list[int]] = []
        window_ids: list[list[int]] = []
        unit_ids: list[list[int]] = []
        route_ids: list[list[int]] = []
        evidence_ids: list[list[int]] = []
        encoder_flags: list[list[int]] = []
        features: list[list[list[float]]] = []
        feature_masks: list[list[list[bool]]] = []
        attention_masks: list[list[bool]] = []
        lengths: list[int] = []

        for row in token_rows:
            lengths.append(len(row))
            padding = max_length - len(row)
            concept_ids.append([token.concept_id for token in row] + [0] * padding)
            domain_ids.append([token.domain_id for token in row] + [0] * padding)
            window_ids.append([token.window_id for token in row] + [0] * padding)
            unit_ids.append([token.unit_id for token in row] + [0] * padding)
            route_ids.append([token.route_id for token in row] + [0] * padding)
            evidence_ids.append([token.evidence_id for token in row] + [0] * padding)
            encoder_flags.append([token.encoder_flag for token in row] + [0] * padding)
            features.append([token.features for token in row] + [[0.0] * len(TIMELINE_FEATURE_NAMES) for _ in range(padding)])
            feature_masks.append([token.feature_masks for token in row] + [[False] * len(TIMELINE_FEATURE_NAMES) for _ in range(padding)])
            attention_masks.append([True] * len(row) + [False] * padding)

        global_summaries = self._collate_global_summaries(samples)
        included_layers = frozenset(self.config.included_layers)

        def static_included(name: str, layers: Mapping[str, str]) -> bool:
            return layers.get(name, "patient_state") in included_layers

        return TimelineBatch(
            timeline_feature_names=TIMELINE_FEATURE_NAMES,
            timeline_window_names=WINDOW_NAMES,
            timeline_lengths=lengths,
            timeline_concept_ids=concept_ids,
            timeline_domain_ids=domain_ids,
            timeline_window_ids=window_ids,
            timeline_unit_ids=unit_ids,
            timeline_route_ids=route_ids,
            timeline_evidence_ids=evidence_ids,
            timeline_encoder_flags=encoder_flags,
            timeline_features=features,
            timeline_feature_masks=feature_masks,
            timeline_attention_mask=attention_masks,
            static_continuous_names=STATIC_CONTINUOUS_FIELDS,
            static_continuous=[
                [
                    sample.static_continuous.get(name, 0.0)
                    if static_included(name, STATIC_CONTINUOUS_LAYERS)
                    else 0.0
                    for name in STATIC_CONTINUOUS_FIELDS
                ]
                for sample in samples
            ],
            static_continuous_masks=[
                [
                    not sample.static_missing_mask.get(name, True)
                    and static_included(name, STATIC_CONTINUOUS_LAYERS)
                    for name in STATIC_CONTINUOUS_FIELDS
                ]
                for sample in samples
            ],
            static_categorical_names=STATIC_CATEGORICAL_FIELDS,
            static_categorical=[
                [
                    sample.static_categorical.get(name, 0)
                    if static_included(name, STATIC_CATEGORICAL_LAYERS)
                    else 2
                    for name in STATIC_CATEGORICAL_FIELDS
                ]
                for sample in samples
            ],
            static_binary_names=STATIC_BINARY_FIELDS,
            static_binary=[
                [
                    sample.static_binary.get(name, 0)
                    if static_included(name, STATIC_BINARY_LAYERS)
                    else 0
                    for name in STATIC_BINARY_FIELDS
                ]
                for sample in samples
            ],
            global_summary_feature_names=GLOBAL_SUMMARY_FEATURE_NAMES,
            global_summary_lengths=global_summaries.lengths,
            global_summary_concept_ids=global_summaries.concept_ids,
            global_summary_features=global_summaries.features,
            global_summary_feature_masks=global_summaries.feature_masks,
            global_summary_attention_mask=global_summaries.attention_mask,
            horizon_label_names=HORIZON_LABEL_NAMES,
            horizon_labels=[
                [sample.horizon_labels.get(name, 0) for name in HORIZON_LABEL_NAMES]
                for sample in samples
            ],
            text_chunks=[sample.text_chunks for sample in samples],
            subject_ids=[sample.subject_id for sample in samples],
            hadm_ids=[sample.hadm_id for sample in samples],
            event_types=[sample.event_type for sample in samples],
            time_days=[sample.time_days for sample in samples],
        )

    def timeline_length(self, sample: AdmissionSample) -> int:
        """Return the exact post-aggregation token count without padding a batch."""
        return len(self._aggregate_sample(sample))

    @staticmethod
    def _collate_global_summaries(samples: Sequence[AdmissionSample]) -> _GlobalSummaryRows:
        max_length = max((len(sample.numeric_summaries) for sample in samples), default=0)
        concept_ids: list[list[int]] = []
        features: list[list[list[float]]] = []
        masks: list[list[list[bool]]] = []
        attention: list[list[bool]] = []
        lengths: list[int] = []

        for sample in samples:
            lengths.append(len(sample.numeric_summaries))
            row_ids: list[int] = []
            row_features: list[list[float]] = []
            row_masks: list[list[bool]] = []
            for summary in sample.numeric_summaries:
                values = {
                    "measurement_count_log1p": summary.measurement_count_log1p,
                    **{name: summary.scaled_values.get(name, 0.0) for name in SUMMARY_SCALED_FIELDS},
                    "normalization_available": float(summary.normalization_available),
                    "unseen_value_count_log1p": math.log1p(summary.unseen_value_count),
                }
                present = {
                    "measurement_count_log1p": True,
                    **{name: not summary.missing_mask.get(name, True) for name in SUMMARY_SCALED_FIELDS},
                    "normalization_available": True,
                    "unseen_value_count_log1p": True,
                }
                row_ids.append(summary.concept_id)
                row_features.append([values[name] for name in GLOBAL_SUMMARY_FEATURE_NAMES])
                row_masks.append([present[name] for name in GLOBAL_SUMMARY_FEATURE_NAMES])

            padding = max_length - len(row_ids)
            concept_ids.append(row_ids + [0] * padding)
            features.append(row_features + [[0.0] * len(GLOBAL_SUMMARY_FEATURE_NAMES) for _ in range(padding)])
            masks.append(row_masks + [[False] * len(GLOBAL_SUMMARY_FEATURE_NAMES) for _ in range(padding)])
            attention.append([True] * len(row_ids) + [False] * padding)

        return _GlobalSummaryRows(
            lengths=lengths,
            concept_ids=concept_ids,
            features=features,
            feature_masks=masks,
            attention_mask=attention,
        )

    def _aggregate_sample(self, sample: AdmissionSample) -> list[_TimelineToken]:
        """按完整聚合键生成 token，并汇总数值趋势、时距和状态比例。"""
        groups: dict[tuple[int, int, int, int, int, int, int], _TokenAccumulator] = {}
        for event in sample.events:
            if event.feature_layer not in self.config.included_layers:
                continue
            window_id = _window_id(event)
            domain_id = _lookup(self._domain_index, _canonical(event.event_domain))
            if event.value_type == "binary_exposure":
                # 治疗途径和记录来源只用于上游临床判定。进入时间轴后，同一
                # 日窗、同一治疗类别只有一个二元暴露 token。
                unit_id = 2
                route_id = 2
                evidence_id = 2
                encoder_flag = 0
            else:
                unit_id = _lookup(self._unit_index, _canonical(event.unit)) if event.numeric_mask else 2
                route_id = 2
                evidence_id = _lookup(self._evidence_index, _canonical(event.evidence_type))
                encoder_flag = int(event.model_scope == "main_model_encoder")
            key = (window_id, event.concept_id, domain_id, unit_id, route_id, evidence_id, encoder_flag)
            groups.setdefault(key, _TokenAccumulator(events=[])).events.append(event)

        rows = []
        for key, accumulator in groups.items():
            window_id, concept_id, domain_id, unit_id, route_id, evidence_id, encoder_flag = key
            ordered = sorted(accumulator.events, key=lambda event: (event.hours_from_admission, event.event_id))
            value_types = {event.value_type for event in ordered}
            if len(value_types) != 1:
                raise ValueError(
                    "mixed value types in one concept-window token: "
                    f"hadm_id={sample.hadm_id}, concept_id={concept_id}, "
                    f"concept_names={sorted({event.concept_name for event in ordered})}, "
                    f"value_types={sorted(value_types)}"
                )
            binary_exposure = value_types == {"binary_exposure"}
            if binary_exposure:
                ordered = _deduplicate_treatment_events(ordered)
            numeric = [event for event in ordered if event.numeric_mask]
            values = [event.value_scaled for event in numeric]
            times = [event.hours_from_admission for event in numeric]
            sd = _sample_sd(values)
            slope = _slope(times, values)
            reliable_time = window_id not in {16, 17}
            first_hours = max(0.0, ordered[0].hours_before_discharge)
            last_hours = max(0.0, ordered[-1].hours_before_discharge)
            span = max(0.0, ordered[-1].hours_from_admission - ordered[0].hours_from_admission)
            durations = [event.duration_hours for event in ordered if event.duration_mask]
            statuses = [event.status for event in ordered if event.status.strip()]
            status_categories = [_status_categories(status) for status in statuses]

            def status_fraction(category: str) -> float:
                return sum(category in categories for categories in status_categories) / len(statuses) if statuses else 0.0

            exposure_duration_candidates = [*durations]
            if binary_exposure and len(ordered) > 1 and reliable_time:
                exposure_duration_candidates.append(span)
            exposure_duration = (
                max(exposure_duration_candidates)
                if binary_exposure and exposure_duration_candidates
                else None
            )
            feature_values = {
                "count_log1p": math.log1p(1 if binary_exposure else len(ordered)),
                "first_scaled": values[0] if values else 0.0,
                "last_scaled": values[-1] if values else 0.0,
                "min_scaled": min(values) if values else 0.0,
                "max_scaled": max(values) if values else 0.0,
                "mean_scaled": sum(values) / len(values) if values else 0.0,
                "sd_scaled": sd if sd is not None else 0.0,
                "delta_scaled": values[-1] - values[0] if values else 0.0,
                "slope_scaled_per_hour": slope if slope is not None else 0.0,
                "hours_before_discharge_first_log1p": math.log1p(first_hours) if reliable_time else 0.0,
                "hours_before_discharge_last_log1p": math.log1p(last_hours) if reliable_time else 0.0,
                "observation_span_hours_log1p": math.log1p(span) if reliable_time else 0.0,
                "duration_hours_max_log1p": (
                    math.log1p(exposure_duration)
                    if exposure_duration is not None
                    else math.log1p(max(durations)) if durations else 0.0
                ),
                "normalization_available_fraction": (
                    sum(event.normalization_available for event in numeric) / len(numeric) if numeric else 0.0
                ),
                "status_administered_fraction": (
                    1.0 if binary_exposure else status_fraction("administered")
                ),
                "status_not_given_fraction": status_fraction("not_given"),
                "status_flush_fraction": status_fraction("flush"),
                "status_not_flushed_fraction": status_fraction("not_flushed"),
                "status_started_fraction": status_fraction("started"),
                "status_stopped_fraction": status_fraction("stopped"),
                "status_paused_fraction": status_fraction("paused"),
                "status_changed_fraction": status_fraction("changed"),
                "status_abnormal_fraction": status_fraction("abnormal"),
                "status_susceptible_fraction": status_fraction("susceptible"),
                "status_resistant_fraction": status_fraction("resistant"),
            }
            feature_present = {
                "count_log1p": True,
                "first_scaled": bool(values),
                "last_scaled": bool(values),
                "min_scaled": bool(values),
                "max_scaled": bool(values),
                "mean_scaled": bool(values),
                "sd_scaled": sd is not None,
                "delta_scaled": bool(values),
                "slope_scaled_per_hour": slope is not None,
                "hours_before_discharge_first_log1p": reliable_time,
                "hours_before_discharge_last_log1p": reliable_time,
                "observation_span_hours_log1p": reliable_time,
                "duration_hours_max_log1p": (
                    exposure_duration is not None if binary_exposure else bool(durations)
                ),
                "normalization_available_fraction": bool(numeric),
                "status_administered_fraction": binary_exposure or bool(statuses),
                "status_not_given_fraction": bool(statuses),
                "status_flush_fraction": bool(statuses),
                "status_not_flushed_fraction": bool(statuses),
                "status_started_fraction": bool(statuses),
                "status_stopped_fraction": bool(statuses),
                "status_paused_fraction": bool(statuses),
                "status_changed_fraction": bool(statuses),
                "status_abnormal_fraction": bool(statuses),
                "status_susceptible_fraction": bool(statuses),
                "status_resistant_fraction": bool(statuses),
            }
            rows.append(_TimelineToken(
                concept_id=concept_id,
                domain_id=domain_id,
                window_id=window_id,
                unit_id=unit_id,
                route_id=route_id,
                evidence_id=evidence_id,
                encoder_flag=encoder_flag,
                features=[feature_values[name] for name in TIMELINE_FEATURE_NAMES],
                feature_masks=[feature_present[name] for name in TIMELINE_FEATURE_NAMES],
            ))

        rows.sort(key=lambda row: (
            row.window_id, row.domain_id, row.concept_id, row.unit_id,
            row.route_id, row.evidence_id, row.encoder_flag,
        ))
        if not rows:
            # 白名单收紧后，个别住院可能没有任何保留的结构化事件。
            # 使用全掩码的 <MISSING> 占位符避免 Transformer 收到全 padding；
            # 该 token 不表示任何临床事实。
            rows.append(_TimelineToken(
                concept_id=2,
                domain_id=2,
                window_id=17,
                unit_id=2,
                route_id=2,
                evidence_id=2,
                encoder_flag=0,
                features=[0.0] * len(TIMELINE_FEATURE_NAMES),
                feature_masks=[False] * len(TIMELINE_FEATURE_NAMES),
            ))
        return rows
