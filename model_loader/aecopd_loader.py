"""AECOPD model_v1 的框架无关流式数据加载器。

加载器按 hadm_id 合并静态字段、结构化事件、数值摘要和文本索引，输出
一次住院一个 ``AdmissionSample``。它不训练模型，也不计算评价指标。
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TypeVar

from clinical_feature_contract import ClinicalFeatureContract, MappedClinicalFeature

T = TypeVar("T")

STATIC_CONTINUOUS_FIELDS = (
    "age_at_admission", "ed_hours", "hospital_los_hours", "prior_aecopd_count",
    "prior_all_admission_count", "days_since_previous_aecopd", "icu_stay_count", "icu_los_days",
    "prior_weight_kg", "prior_height_cm", "prior_bmi", "prior_sbp", "prior_dbp",
    "days_since_prior_weight", "days_since_prior_bmi", "days_since_prior_bp",
)
STATIC_CATEGORICAL_FIELDS = (
    "gender", "admission_type", "admission_location", "discharge_location",
)
STATIC_BINARY_FIELDS = ("via_ed", "entered_icu")
SUMMARY_SCALED_FIELDS = (
    "min_value_scaled", "max_value_scaled", "mean_value_scaled", "sd_value_scaled",
    "first_value_scaled", "last_value_scaled", "first_to_last_change_scaled", "slope_per_hour_scaled",
)
RESERVED_VOCAB = ("<PAD>", "<UNK>", "<MISSING>")
DEFAULT_FEATURE_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "clinical_feature_contract.csv"
)
RETROSPECTIVE_EVIDENCE_TYPES = {
    "retrospective_discharge_code",
    "retrospective_procedure_code",
}


@dataclass(frozen=True)
class EventInput:
    """一条保留来源、时间、数值、状态和证据语义的结构化事件。"""

    event_id: str
    concept_id: int
    event_domain: str
    concept_name: str
    hours_from_admission: float
    hours_before_discharge: float
    value_scaled: float
    numeric_mask: bool
    normalization_available: bool
    normalization_method: str
    raw_code: str
    raw_name: str
    value_text: str
    unit: str
    route: str
    status: str
    evidence_type: str
    model_scope: str
    duration_hours: float = 0.0
    duration_mask: bool = False
    feature_layer: str = "patient_state"
    value_type: str = "clinical_state"
    source_priority: int = 0
    dedup_identity: str = ""
    dedup_route: str = ""


@dataclass(frozen=True)
class NumericSummaryInput:
    """一个概念在整次住院内的训练折归一化数值轨迹摘要。"""

    concept_id: int
    event_domain: str
    concept_name: str
    canonical_unit: str
    measurement_count_log1p: float
    scaled_values: dict[str, float]
    missing_mask: dict[str, bool]
    normalization_available: bool
    unseen_value_count: int


@dataclass(frozen=True)
class TextChunkInput:
    """一段文本或章节的索引与元数据；风险训练可不读取 ``chunk_text``。"""

    document_key: str
    document_role: str
    document_rank: int
    days_before_landmark: float
    chunk_index: int
    token_count: int
    chunk_text: str
    source_document_key: str = ""
    modality: str = "<UNK>"
    section_role: str = "<UNK>"
    section_order: int = 0
    section_instance: int = 1


@dataclass(frozen=True)
class AdmissionSample:
    """一次索引住院的完整模型无关样本。"""

    subject_id: int
    hadm_id: int
    data_split: str
    event_type: int
    event_name: str
    time_days: float
    horizon_labels: dict[str, int]
    static_continuous: dict[str, float]
    static_missing_mask: dict[str, bool]
    static_categorical: dict[str, int]
    static_binary: dict[str, int]
    events: list[EventInput]
    numeric_summaries: list[NumericSummaryInput]
    text_chunks: list[TextChunkInput]
    hospital_los_hours: float = math.nan


def _open_csv(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _float_or_none(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _finite_or_zero(value: str | None) -> float:
    parsed = _float_or_none(value)
    return 0.0 if parsed is None else parsed


def _int_or_zero(value: str | None) -> int:
    normalized = (value or "").strip().lower()
    if normalized in {"t", "true", "y", "yes"}:
        return 1
    if normalized in {"f", "false", "n", "no", ""}:
        return 0
    parsed = _float_or_none(value)
    return 0 if parsed is None else int(parsed)


def _bool_value(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "t", "true", "y", "yes"}


def _duration_hours(start: str | None, end: str | None) -> tuple[float, bool]:
    if not start or not end:
        return 0.0, False
    try:
        duration = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 3600.0
    except ValueError:
        return 0.0, False
    if not math.isfinite(duration) or duration < 0.0:
        return 0.0, False
    return duration, True


def _canonical(value: str | None) -> str:
    stripped = (value or "").strip()
    return stripped if stripped else "<MISSING>"


def _event_identity(row: Mapping[str, str]) -> str:
    """Return the train-fitted identity used by the structured event encoder.

    Events explicitly routed to a code/text encoder must retain the concrete raw
    item. Otherwise hundreds of distinct drugs, laboratory items, and ICD codes
    collapse to one generic placeholder concept.
    """
    base = f"{row['event_domain']}|{row['concept_name']}"
    if row.get("model_scope") != "main_model_encoder":
        return base
    raw_identity = _canonical(row.get("raw_code") or row.get("raw_name") or row.get("value_text"))
    return f"{base}|{raw_identity}"


def _mapped_event_identity(mapped: MappedClinicalFeature) -> str:
    return f"{mapped.feature_layer}|{mapped.canonical_concept}"


def _canonical_evidence(row: Mapping[str, str]) -> str:
    evidence_type = (row.get("evidence_type") or "").strip()
    if evidence_type in RETROSPECTIVE_EVIDENCE_TYPES:
        return evidence_type
    if _float_or_none(row.get("hours_from_admission")) is not None:
        return "observed_time"
    if _float_or_none(row.get("hours_before_discharge")) is not None:
        return "observed_time"
    return "unknown_time"


def _is_treatment_exposure(row: Mapping[str, str]) -> bool:
    """判断治疗记录是否足以支持实际暴露。

    处方只表示计划，不冒充实际给药；明确未执行、未开始、暂缓或
    暂停的记录也不能单独证明本次暴露。“Stopped”仍保留，因为它可以是已给药
    后的停药记录。
    """
    evidence_type = (row.get("evidence_type") or "").strip().lower()
    if evidence_type == "prescription_only":
        return False
    normalized = (row.get("status") or "").strip().lower()
    excluded_markers = (
        "not given",
        "not administered",
        "not started",
        "not confirmed",
        "hold dose",
        "held dose",
        "dose held",
        "paused",
    )
    if normalized in {"no", "delayed"}:
        return False
    return not any(marker in normalized for marker in excluded_markers)


def _has_observed_treatment_time(row: Mapping[str, str]) -> bool:
    """治疗暴露必须能定位到住院时间窗。

    无入院相对时间且无出院相对时间的记录不能默认放入第1天。
    """
    return (
        _float_or_none(row.get("hours_from_admission")) is not None
        or _float_or_none(row.get("hours_before_discharge")) is not None
    )


def _treatment_dedup_identity(row: Mapping[str, str]) -> str:
    """返回仅用于上游跨来源去重的原始药物标识。"""
    return _canonical(row.get("raw_code") or row.get("raw_name"))


def _summary_identity(mapped: MappedClinicalFeature, canonical_unit: str) -> str:
    return f"{mapped.feature_layer}|{mapped.canonical_concept}|{canonical_unit}"


def _source_priority(row: Mapping[str, str]) -> int:
    """返回仅供上游去重使用的来源优先级，不进入模型特征。"""
    source_table = (row.get("source_table") or "").strip().lower()
    evidence_type = (row.get("evidence_type") or "").strip().lower()
    if source_table == "mimiciv_hosp.emar" or evidence_type == "actual_administration":
        return 3
    if source_table == "mimiciv_icu.inputevents" or evidence_type == "actual_icu_input":
        return 2
    if source_table == "mimiciv_hosp.prescriptions" or evidence_type == "prescription_only":
        return 1
    return 0


def _vocabulary(counter: Counter[str]) -> list[str]:
    learned = [
        value
        for value, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        if value not in RESERVED_VOCAB
    ]
    return [*RESERVED_VOCAB, *learned]


def _vocab_index(vocabulary: Sequence[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate(vocabulary)}


def _quantiles(values: Sequence[float], probabilities: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantiles require at least one value")
    result = []
    for probability in probabilities:
        position = probability * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        result.append(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)
    return result


def buffered_shuffle(items: Iterable[T], buffer_size: int, seed: int) -> Iterator[T]:
    """Deterministically shuffle a stream with bounded memory."""
    if buffer_size < 1:
        raise ValueError("buffer_size must be at least 1")
    rng = random.Random(seed)
    buffer: list[T] = []
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = item
    rng.shuffle(buffer)
    yield from buffer


def _grouped_rows(path: Path) -> Iterator[tuple[int, list[dict[str, str]]]]:
    current_key: int | None = None
    rows: list[dict[str, str]] = []
    for row in _open_csv(path):
        key = int(row["hadm_id"])
        if current_key is None:
            current_key = key
        if key != current_key:
            if key < current_key:
                raise ValueError(f"{path.name} is not sorted by hadm_id")
            yield current_key, rows
            current_key = key
            rows = []
        rows.append(row)
    if current_key is not None:
        yield current_key, rows


class _GroupCursor:
    def __init__(self, path: Path) -> None:
        self._iterator = iter(_grouped_rows(path))
        self._current = next(self._iterator, None)

    def take(self, hadm_id: int) -> list[dict[str, str]]:
        while self._current is not None and self._current[0] < hadm_id:
            self._current = next(self._iterator, None)
        if self._current is not None and self._current[0] == hadm_id:
            rows = self._current[1]
            self._current = next(self._iterator, None)
            return rows
        return []


class AecopdDataBundle:
    """Stream admission samples with either v1 document or v2 section text."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        text_metadata_only: bool = False,
        text_dir: str | Path | None = None,
        feature_contract_path: str | Path | None = DEFAULT_FEATURE_CONTRACT_PATH,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.feature_contract = (
            None
            if feature_contract_path is None
            else ClinicalFeatureContract.from_csv(feature_contract_path)
        )
        text_root = self.data_dir if text_dir is None else Path(text_dir)
        if text_dir is None:
            text_name = (
                "qwen_embedding_index.csv.gz"
                if text_metadata_only
                else "qwen_text_chunks.csv.gz"
            )
        else:
            text_name = (
                "qwen_section_embedding_index.csv.gz"
                if text_metadata_only
                else "qwen_section_text_chunks.csv.gz"
            )
        self.paths = {
            "cohort": self.data_dir / "cohort_labels_static.csv.gz",
            "events": self.data_dir / "events.csv.gz",
            "summary": self.data_dir / "numeric_summary.csv.gz",
            "stats": self.data_dir / "numeric_normalization_stats.csv.gz",
            "text": text_root / text_name,
        }
        missing = [str(path) for path in self.paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing model_v1 files: {missing}")

    def fit_preprocessing(self, output_path: str | Path | None = None) -> dict[str, Any]:
        """Fit static scalers and vocabularies using train rows only."""
        continuous: dict[str, list[float]] = {field: [] for field in STATIC_CONTINUOUS_FIELDS}
        categorical: dict[str, Counter[str]] = {field: Counter() for field in STATIC_CATEGORICAL_FIELDS}
        train_admissions = 0

        for row in _open_csv(self.paths["cohort"]):
            if row["data_split"] != "train":
                continue
            train_admissions += 1
            for field in STATIC_CONTINUOUS_FIELDS:
                value = _float_or_none(row.get(field))
                if value is not None:
                    continuous[field].append(value)
            for field in STATIC_CATEGORICAL_FIELDS:
                categorical[field][_canonical(row.get(field))] += 1

        static_stats: dict[str, dict[str, float | int | str]] = {}
        for field, values in continuous.items():
            if values:
                p005, p25, median, p75, p995 = _quantiles(values, [0.005, 0.25, 0.5, 0.75, 0.995])
                if p75 > p25:
                    scale = p75 - p25
                    method = "train_iqr"
                elif p995 > p005:
                    scale = p995 - p005
                    method = "train_p005_p995_range"
                else:
                    scale = 1.0
                    method = "train_constant"
                static_stats[field] = {
                    "train_n": len(values), "p005": float(p005), "p25": float(p25),
                    "median": float(median), "p75": float(p75), "p995": float(p995),
                    "scale": float(scale), "method": method,
                }
            else:
                static_stats[field] = {
                    "train_n": 0, "p005": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0,
                    "p995": 0.0, "scale": 1.0, "method": "no_train_values",
                }

        event_counter: Counter[str] = Counter()
        event_domain_counter: Counter[str] = Counter()
        event_unit_counter: Counter[str] = Counter()
        event_route_counter: Counter[str] = Counter()
        event_evidence_counter: Counter[str] = Counter()
        source_train_event_rows = 0
        retained_train_source_rows = 0
        mapped_train_event_rows = 0
        mapped_train_layer_counts: Counter[str] = Counter()
        for row in _open_csv(self.paths["events"]):
            if row["data_split"] != "train":
                continue
            source_train_event_rows += 1
            if self.feature_contract is None:
                event_counter[_event_identity(row)] += 1
                event_domain_counter[_canonical(row.get("event_domain"))] += 1
                event_unit_counter[_canonical((row.get("unit") or "").lower())] += 1
                event_route_counter[_canonical((row.get("route") or "").lower())] += 1
                event_evidence_counter[_canonical(row.get("evidence_type"))] += 1
                retained_train_source_rows += 1
                mapped_train_event_rows += 1
                continue
            retained_current_row = False
            for mapped in self.feature_contract.map_event(row):
                if mapped.value_type == "binary_exposure":
                    if not _is_treatment_exposure(row) or not _has_observed_treatment_time(row):
                        continue
                retained_current_row = True
                event_counter[_mapped_event_identity(mapped)] += 1
                event_domain_counter[mapped.feature_layer] += 1
                unit = "" if mapped.value_type == "binary_exposure" else (row.get("unit") or "")
                event_unit_counter[_canonical(unit.lower())] += 1
                event_route_counter["<MISSING>"] += 1
                event_evidence_counter[_canonical_evidence(row)] += 1
                mapped_train_event_rows += 1
                mapped_train_layer_counts[mapped.feature_layer] += 1
            retained_train_source_rows += int(retained_current_row)

        summary_counter: Counter[str] = Counter()
        for row in _open_csv(self.paths["stats"]):
            if self.feature_contract is None:
                key = f"{row['event_domain']}|{row['concept_name']}|{row['canonical_unit']}"
                summary_counter[key] += int(row["train_n"])
                continue
            pseudo_event = {**row, "model_scope": "main_model"}
            for mapped in self.feature_contract.map_event(pseudo_event):
                if mapped.feature_layer != "patient_state" or mapped.value_type != "numeric_trajectory":
                    continue
                summary_counter[_summary_identity(mapped, row["canonical_unit"])] += int(row["train_n"])

        artifacts: dict[str, Any] = {
            "schema_version": "aecopd-curated-clinical-loader-1",
            "fit_split": "train",
            "train_admissions": train_admissions,
            "static_continuous": static_stats,
            "static_categorical": {
                field: {"vocabulary": _vocabulary(counter)} for field, counter in categorical.items()
            },
            "static_binary_fields": list(STATIC_BINARY_FIELDS),
            "event_vocabulary": _vocabulary(event_counter),
            "event_domain_vocabulary": _vocabulary(event_domain_counter),
            "event_unit_vocabulary": _vocabulary(event_unit_counter),
            "event_route_vocabulary": _vocabulary(event_route_counter),
            "event_evidence_vocabulary": _vocabulary(event_evidence_counter),
            "numeric_summary_vocabulary": _vocabulary(summary_counter),
            "reserved_ids": {"pad": 0, "unknown": 1, "missing": 2},
            "normalization_contract": {
                "event_values": "train IQR; fallback train p0.5-p99.5 range; constant groups use scale 1; unseen train groups use value 0 plus availability=false",
                "static_values": "train p0.5-p99.5 clipping followed by train robust scaling; missing values imputed with train median plus missing mask",
                "numeric_summary": "precomputed scaled columns; missing optional statistics use 0 plus missing mask",
            },
            "clinical_feature_contract": (
                None if self.feature_contract is None else self.feature_contract.metadata()
            ),
            "clinical_feature_audit": {
                "source_train_event_rows": source_train_event_rows,
                "retained_train_source_rows": retained_train_source_rows,
                "excluded_train_source_rows": source_train_event_rows - retained_train_source_rows,
                "mapped_train_event_rows": mapped_train_event_rows,
                "mapped_train_layer_counts": dict(sorted(mapped_train_layer_counts.items())),
            },
        }
        if output_path is not None:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")
        return artifacts

    @staticmethod
    def load_preprocessing(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def iter_samples(
        self,
        split: str | None,
        *,
        artifacts: Mapping[str, Any],
        include_text: bool = True,
    ) -> Iterator[AdmissionSample]:
        if split is not None and split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, test, or None")

        event_cursor = _GroupCursor(self.paths["events"])
        summary_cursor = _GroupCursor(self.paths["summary"])
        text_cursor = _GroupCursor(self.paths["text"])
        event_index = _vocab_index(artifacts["event_vocabulary"])
        summary_index = _vocab_index(artifacts["numeric_summary_vocabulary"])
        category_indices = {
            field: _vocab_index(artifacts["static_categorical"][field]["vocabulary"])
            for field in STATIC_CATEGORICAL_FIELDS
        }
        artifact_contract = artifacts.get("clinical_feature_contract")
        if self.feature_contract is not None:
            if not isinstance(artifact_contract, Mapping):
                raise ValueError(
                    "Preprocessing artifacts do not contain the active clinical feature contract; "
                    "refit preprocessing before loading samples"
                )
            if artifact_contract.get("sha256") != self.feature_contract.sha256:
                raise ValueError(
                    "Clinical feature contract changed after preprocessing was fit; refit preprocessing"
                )

        for row in _open_csv(self.paths["cohort"]):
            hadm_id = int(row["hadm_id"])
            event_rows = event_cursor.take(hadm_id)
            summary_rows = summary_cursor.take(hadm_id)
            text_rows = text_cursor.take(hadm_id)
            if split is not None and row["data_split"] != split:
                continue
            row_split = row["data_split"]
            if any(item["data_split"] != row_split for item in event_rows):
                raise ValueError(f"event split mismatch for hadm_id={hadm_id}")
            if any(item["data_split"] != row_split for item in summary_rows):
                raise ValueError(f"numeric summary split mismatch for hadm_id={hadm_id}")
            if any(item["data_split"] != row_split for item in text_rows):
                raise ValueError(f"text split mismatch for hadm_id={hadm_id}")

            static_continuous: dict[str, float] = {}
            static_missing: dict[str, bool] = {}
            for field in STATIC_CONTINUOUS_FIELDS:
                value = _float_or_none(row.get(field))
                static_missing[field] = value is None
                stats = artifacts["static_continuous"][field]
                if value is None:
                    value = float(stats["median"])
                clipped = min(float(stats["p995"]), max(float(stats["p005"]), value))
                static_continuous[field] = (clipped - float(stats["median"])) / float(stats["scale"])

            static_categorical = {}
            for field in STATIC_CATEGORICAL_FIELDS:
                value = _canonical(row.get(field))
                static_categorical[field] = category_indices[field].get(value, 1)

            events = []
            for event in event_rows:
                if self.feature_contract is not None:
                    mapped_features = self.feature_contract.map_event(event)
                else:
                    mapped_features = ()
                value_num = _float_or_none(event.get("value_num"))
                duration_hours, duration_mask = _duration_hours(event.get("event_time"), event.get("end_time"))
                if self.feature_contract is not None:
                    for mapped in mapped_features:
                        if mapped.value_type == "binary_exposure":
                            if not _is_treatment_exposure(event) or not _has_observed_treatment_time(event):
                                continue
                        binary_exposure = mapped.value_type == "binary_exposure"
                        event_key = _mapped_event_identity(mapped)
                        events.append(EventInput(
                            event_id=f"{event['event_id']}::{mapped.rule_id}",
                            concept_id=event_index.get(event_key, 1),
                            event_domain=mapped.feature_layer,
                            concept_name=mapped.canonical_concept,
                            hours_from_admission=_finite_or_zero(event.get("hours_from_admission")),
                            hours_before_discharge=_finite_or_zero(event.get("hours_before_discharge")),
                            value_scaled=(
                                0.0 if binary_exposure else _finite_or_zero(event.get("value_scaled"))
                            ),
                            numeric_mask=(value_num is not None and not binary_exposure),
                            normalization_available=(
                                _bool_value(event.get("normalization_available"))
                                if value_num is not None and not binary_exposure
                                else False
                            ),
                            normalization_method=(
                                event.get("normalization_method", "") if not binary_exposure else ""
                            ),
                            raw_code=event.get("raw_code", ""),
                            raw_name=event.get("raw_name", ""),
                            value_text=event.get("value_text", ""),
                            unit="" if binary_exposure else event.get("unit", ""),
                            route="",
                            # 保留原始状态供上游审计；二元暴露在
                            # collator 中统一表示为已暴露，原始字符串不入模。
                            status=event.get("status", ""),
                            evidence_type=_canonical_evidence(event),
                            model_scope="curated_clinical_feature",
                            duration_hours=duration_hours,
                            duration_mask=duration_mask,
                            feature_layer=mapped.feature_layer,
                            value_type=mapped.value_type,
                            source_priority=_source_priority(event),
                            dedup_identity=(
                                _treatment_dedup_identity(event) if binary_exposure else ""
                            ),
                            dedup_route=(
                                _canonical(event.get("route")) if binary_exposure else ""
                            ),
                        ))
                    continue
                specific_identity = "event_domain_vocabulary" in artifacts
                event_key = _event_identity(event) if specific_identity else f"{event['event_domain']}|{event['concept_name']}"
                events.append(EventInput(
                    event_id=event["event_id"],
                    concept_id=event_index.get(event_key, 1),
                    event_domain=event["event_domain"], concept_name=event["concept_name"],
                    hours_from_admission=_finite_or_zero(event.get("hours_from_admission")),
                    hours_before_discharge=_finite_or_zero(event.get("hours_before_discharge")),
                    value_scaled=_finite_or_zero(event.get("value_scaled")), numeric_mask=value_num is not None,
                    normalization_available=_bool_value(event.get("normalization_available")) if value_num is not None else False,
                    normalization_method=event.get("normalization_method", ""), raw_code=event.get("raw_code", ""),
                    raw_name=event.get("raw_name", ""), value_text=event.get("value_text", ""),
                    unit=event.get("unit", ""), route=event.get("route", ""), status=event.get("status", ""),
                    evidence_type=event.get("evidence_type", ""), model_scope=event.get("model_scope", ""),
                    duration_hours=duration_hours, duration_mask=duration_mask,
                ))

            summaries = []
            for summary in summary_rows:
                if self.feature_contract is not None:
                    mapped_summaries = tuple(
                        mapped
                        for mapped in self.feature_contract.map_event(
                            {**summary, "model_scope": "main_model"}
                        )
                        if mapped.feature_layer == "patient_state"
                        and mapped.value_type == "numeric_trajectory"
                    )
                    if not mapped_summaries:
                        continue
                else:
                    mapped_summaries = ()
                values: dict[str, float] = {}
                missing: dict[str, bool] = {}
                for field in SUMMARY_SCALED_FIELDS:
                    parsed = _float_or_none(summary.get(field))
                    missing[field] = parsed is None
                    values[field] = 0.0 if parsed is None else parsed
                if self.feature_contract is None:
                    mapped_rows = (
                        (
                            f"{summary['event_domain']}|{summary['concept_name']}|{summary['canonical_unit']}",
                            summary["event_domain"],
                            summary["concept_name"],
                        ),
                    )
                else:
                    mapped_rows = tuple(
                        (
                            _summary_identity(mapped, summary["canonical_unit"]),
                            mapped.feature_layer,
                            mapped.canonical_concept,
                        )
                        for mapped in mapped_summaries
                    )
                for key, event_domain, concept_name in mapped_rows:
                    summaries.append(NumericSummaryInput(
                        concept_id=summary_index.get(key, 1), event_domain=event_domain,
                        concept_name=concept_name, canonical_unit=summary["canonical_unit"],
                        measurement_count_log1p=math.log1p(_int_or_zero(summary.get("measurement_count"))),
                        scaled_values=values, missing_mask=missing,
                        normalization_available=_bool_value(summary.get("normalization_available")),
                        unseen_value_count=_int_or_zero(summary.get("unseen_value_count")),
                    ))

            texts = []
            if include_text:
                for text in text_rows:
                    texts.append(TextChunkInput(
                        document_key=text["document_key"], document_role=text["document_role"],
                        document_rank=int(text["document_rank"]),
                        days_before_landmark=_finite_or_zero(text.get("days_before_landmark")),
                        chunk_index=int(text["chunk_index"]), token_count=int(text["token_count"]),
                        chunk_text=text.get("chunk_text", ""),
                        source_document_key=text.get("source_document_key") or text["document_key"],
                        modality=text.get("modality") or "<UNK>",
                        section_role=text.get("section_role") or "<UNK>",
                        section_order=int(text.get("section_order") or text["chunk_index"]),
                        section_instance=int(text.get("section_instance") or 1),
                    ))

            horizon_fields = (
                "aecopd_by_90d", "aecopd_by_180d", "aecopd_by_365d",
                "death_by_90d", "death_by_180d", "death_by_365d",
            )
            yield AdmissionSample(
                subject_id=int(row["subject_id"]), hadm_id=hadm_id, data_split=row_split,
                event_type=int(row["event_type"]), event_name=row["event_name"],
                time_days=float(row["time_days"]),
                horizon_labels={field: _int_or_zero(row.get(field)) for field in horizon_fields},
                static_continuous=static_continuous, static_missing_mask=static_missing,
                static_categorical=static_categorical,
                static_binary={field: int(_bool_value(row.get(field))) for field in STATIC_BINARY_FIELDS},
                events=events, numeric_summaries=summaries, text_chunks=texts,
                hospital_los_hours=_finite_or_zero(row.get("hospital_los_hours")),
            )

    def iter_all_samples(
        self,
        *,
        artifacts: Mapping[str, Any],
        include_text: bool = True,
    ) -> Iterator[AdmissionSample]:
        yield from self.iter_samples(None, artifacts=artifacts, include_text=include_text)

    def iter_batches(
        self,
        split: str,
        *,
        artifacts: Mapping[str, Any],
        batch_size: int,
        shuffle_buffer_size: int = 0,
        seed: int = 0,
        include_text: bool = True,
    ) -> Iterator[list[AdmissionSample]]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        samples: Iterable[AdmissionSample] = self.iter_samples(split, artifacts=artifacts, include_text=include_text)
        if shuffle_buffer_size:
            samples = buffered_shuffle(samples, shuffle_buffer_size, seed)
        batch: list[AdmissionSample] = []
        for sample in samples:
            batch.append(sample)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def sample_to_dict(sample: AdmissionSample) -> dict[str, Any]:
    """Convert a sample to plain Python data for inspection or JSON debugging."""
    return asdict(sample)
