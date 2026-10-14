"""把已校验的稀疏结构化事实与独立字段文本接入同一出院序列。"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_fusion import IndependentBatch
from training.v5.numeric_preprocessing import robust_parameters


def _rows(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class IndependentData:
    subject_ids: np.ndarray
    hadm_ids: np.ndarray
    splits: np.ndarray
    times: np.ndarray
    events: np.ndarray
    gap: np.ndarray
    sequences: tuple[np.ndarray, ...]
    token_values: np.ndarray
    token_masks: np.ndarray
    token_concepts: np.ndarray
    token_types: np.ndarray
    tokens_by_admission: tuple[np.ndarray, ...]
    concept_names: tuple[str, ...]
    type_names: tuple[str, ...]
    text_by_admission: tuple[tuple[tuple[int, int], ...], ...]
    text_vectors: np.ndarray | None
    text_width: int
    manifest_sha256: str | None
    label_sha256: str | None
    label_name: str | None
    disabled_concepts: tuple[str, ...] = ()
    field_roles: tuple[str, ...] = tuple(FIELD_LABELS)


def load_data(
    aligned_dir: Path, structured_dir: Path, base_csv: Path,
    field_dir: Path | None = None, vectors_path: Path | None = None,
    labels_csv: Path | None = None,
) -> IndependentData:
    """结构化 token 不按章节与文本配对；可用已校验标签覆盖旧 base 结局。"""

    if (field_dir is None) != (vectors_path is None):
        raise ValueError("field manifest and vectors must be supplied together")
    validated = json.loads((structured_dir / "validation.json").read_text(encoding="utf-8"))
    token_path = structured_dir / "structured_tokens.csv.gz"
    if validated["status"] != "section_inputs_validated" or _sha(token_path) != validated["output_sha256"][token_path.name]:
        raise ValueError("structured token data unvalidated or changed")
    cohort = {row["hadm_id"]: row for row in _rows(aligned_dir / "cohort_manifest.csv.gz")}
    base = list(_rows(base_csv))
    if len(base) != 19401 or len(cohort) != len(base):
        raise ValueError("V5 full discharge cohort mismatch")
    hadms = np.array([int(row["hadm_id"]) for row in base], dtype=np.int64)
    subjects = np.array([int(row["subject_id"]) for row in base], dtype=np.int64)
    if len(set(hadms.tolist())) != len(base):
        raise ValueError("duplicate discharge admission")
    row_by_hadm = {str(hadm): index for index, hadm in enumerate(hadms)}
    label_sha: str | None = None
    label_name: str | None = None
    labels: dict[str, dict[str, str]] | None = None
    if labels_csv is not None:
        label_qa = json.loads((labels_csv.parent / "summary.json").read_text(encoding="utf-8"))
        label_sha = _sha(labels_csv)
        if (label_qa["status"] not in {"broad_labels_exported_and_checked",
                                       "qualified_labels_exported_and_checked",
                                       "strict_labels_reconstructed_and_checked"}
                or label_qa["label_sha256"] != label_sha
                or label_qa["cohort_admissions"] != len(base)
                or not label_qa.get("primary_event")):
            raise ValueError("outcome labels are unvalidated or changed")
        label_name = str(label_qa["primary_event"])
        labels = {row["hadm_id"]: row for row in _rows(labels_csv)}
        if len(labels) != len(base) or set(labels) != set(row_by_hadm):
            raise ValueError("outcome admission keys differ")
    splits = np.empty(len(base), dtype="U10")
    times = np.empty(len(base), dtype=np.float32)
    events = np.empty(len(base), dtype=np.int64)
    gap = np.zeros((len(base), 2), dtype=np.float32)
    patient_rows: dict[int, list[int]] = defaultdict(list)
    previous_discharge: dict[int, datetime] = {}
    for index, row in enumerate(base):
        hadm, subject = row["hadm_id"], int(row["subject_id"])
        manifest = cohort.get(hadm)
        if manifest is None or manifest["subject_id"] != row["subject_id"]:
            raise ValueError("cohort identifier mismatch")
        if manifest["admittime"] != row["admittime"] or manifest["dischtime"] != row["dischtime"]:
            raise ValueError("cohort prediction time mismatch")
        splits[index] = manifest["patient_split"]
        outcome = labels[hadm] if labels is not None else row
        if (outcome["subject_id"] != row["subject_id"]
                or outcome["admittime"] != row["admittime"]
                or outcome["dischtime"] != row["dischtime"]):
            raise ValueError("broad outcome identity or landmark mismatch")
        times[index], events[index] = float(outcome["time_days"]), int(outcome["event_type"])
        admission = datetime.fromisoformat(row["admittime"])
        discharge = datetime.fromisoformat(row["dischtime"])
        if not discharge > admission:
            raise ValueError("ineligible discharge in base")
        prior = previous_discharge.get(subject)
        if prior is not None:
            gap_days = max(0.0, (admission - prior).total_seconds() / 86400.0)
            gap[index] = (math.log1p(gap_days), float(admission < prior))
        previous_discharge[subject] = discharge
        patient_rows[subject].append(index)
    if not np.isfinite(times).all() or np.any((times < 0) | (times > 365)):
        raise ValueError("invalid event times")
    if not np.isin(events, (0, 1, 2)).all() or np.any((events == 0) & (times != 365)):
        raise ValueError("invalid event/censoring labels")
    sequences = tuple(np.asarray(rows, dtype=np.int64) for rows in patient_rows.values())
    if any(len(set(splits[sequence].tolist())) != 1 for sequence in sequences):
        raise ValueError("patient crosses train/validation/test splits")
    for sequence in sequences:
        admissions = [base[index]["admittime"] for index in sequence]
        if admissions != sorted(admissions):
            raise ValueError("patient admissions not chronological")

    token_rows = list(_rows(token_path))
    if len(token_rows) != validated["tokens"]:
        raise ValueError("structured token count drift")
    concept_names = tuple(sorted({row["token_type"] + ":" + row["concept"] for row in token_rows}))
    type_names = tuple(sorted({row["token_type"] for row in token_rows}))
    concept_id = {name: index + 1 for index, name in enumerate(concept_names)}
    type_id = {name: index + 1 for index, name in enumerate(type_names)}
    token_values = np.zeros((len(token_rows), 5), dtype=np.float32)
    token_masks = np.zeros_like(token_values)
    token_concepts = np.zeros(len(token_rows), dtype=np.int64)
    token_types = np.zeros(len(token_rows), dtype=np.int64)
    token_refs: list[list[int]] = [[] for _ in base]
    for position, row in enumerate(token_rows):
        hadm = row["hadm_id"]
        index = row_by_hadm.get(hadm)
        if index is None or row["subject_id"] != base[index]["subject_id"]:
            raise ValueError("structured token outside cohort")
        token_values[position] = [float(row[f"value{channel}"]) for channel in range(5)]
        token_masks[position] = [float(row[f"mask{channel}"]) for channel in range(5)]
        token_concepts[position] = concept_id[row["token_type"] + ":" + row["concept"]]
        token_types[position] = type_id[row["token_type"]]
        token_refs[index].append(position)
    if not np.isfinite(token_values).all() or not np.isin(token_masks, (0, 1)).all():
        raise ValueError("invalid structured token numerical values")
    if any(not refs for refs in token_refs):
        raise ValueError("admission lacks patient-context tokens")

    text_refs: list[list[tuple[int, int]]] = [[] for _ in base]
    vectors: np.ndarray | None = None
    text_width = 768
    manifest_sha: str | None = None
    field_roles = tuple(FIELD_LABELS)
    if field_dir is not None and vectors_path is not None:
        field_qa = json.loads((field_dir / "validation.json").read_text(encoding="utf-8"))
        field_summary = json.loads((field_dir / "summary.json").read_text(encoding="utf-8"))
        field_roles = tuple(field_summary["fields"])
        if (not field_roles or len(set(field_roles)) != len(field_roles)
                or field_roles[:len(FIELD_LABELS)] != tuple(FIELD_LABELS)):
            raise ValueError("field vocabulary must preserve the four baseline roles")
        manifest_path = field_dir / "shared_text.csv.gz"
        manifest_sha = _sha(manifest_path)
        state = json.loads((vectors_path.parent / "state.json").read_text(encoding="utf-8"))
        if (field_qa["status"] != "field_text_manifest_validated"
                or field_qa["manifest_sha256"] != manifest_sha
                or field_summary["manifest_sha256"] != manifest_sha):
            raise ValueError("unvalidated field-text manifest")
        if state.get("status") != "complete" or state.get("manifest_sha256") != manifest_sha:
            raise ValueError("incomplete or mismatched frozen text encoding")
        vectors = np.load(vectors_path, mmap_mode="r")
        if vectors.ndim != 2 or vectors.shape[1] not in (768, 2560):
            raise ValueError("unexpected text vector dimension")
        if vectors.shape[1] == 768 and state.get("model") != "Bio_ClinicalBERT":
            raise ValueError("BERT encoder metadata mismatch")
        if vectors.shape[1] == 2560 and state.get("encoder") != "Qwen3-Embedding-4B":
            raise ValueError("Qwen encoder metadata mismatch")
        text_width = int(vectors.shape[1])
        field_id = {role: index for index, role in enumerate(field_roles)}
        text_count = 0
        for row_number, row in enumerate(_rows(manifest_path)):
            index = row_by_hadm.get(row["hadm_id"])
            if index is None or row["subject_id"] != base[index]["subject_id"]:
                raise ValueError("text field outside cohort")
            if int(row["embedding_row"]) != row_number or row["field"] not in field_id:
                raise ValueError("field text row/order mismatch")
            text_refs[index].append((field_id[row["field"]], row_number))
            text_count += 1
        if text_count != vectors.shape[0]:
            raise ValueError("text vector row count mismatch")
    return IndependentData(
        subjects, hadms, splits, times, events, gap, sequences,
        token_values, token_masks, token_concepts, token_types,
        tuple(np.asarray(refs, dtype=np.int64) for refs in token_refs),
        concept_names, type_names, tuple(tuple(refs) for refs in text_refs),
        vectors, text_width, manifest_sha, label_sha, label_name,
        field_roles=field_roles,
    )


def normalize_on_training_patients(data: IndependentData) -> tuple[IndependentData, dict[str, object]]:
    """按概念／数值通道用训练患者拟合中位数和 IQR，其他折不参与。"""

    train_hadms = {int(hadm) for hadm in data.hadm_ids[data.splits == "train"]}
    token_hadm = np.empty(len(data.token_values), dtype=np.int64)
    for admission, refs in enumerate(data.tokens_by_admission):
        token_hadm[refs] = data.hadm_ids[admission]
    train = np.isin(token_hadm, list(train_hadms))
    transformed = data.token_values.copy()
    parameters: dict[str, object] = {}
    for concept_id, concept_name in enumerate(data.concept_names, start=1):
        selected = data.token_concepts == concept_id
        medians: list[float] = []
        scales: list[float] = []
        for channel in range(5):
            observed = data.token_values[selected & train & (data.token_masks[:, channel] > 0), channel]
            if observed.size:
                fitted = robust_parameters(observed)
                median, scale = float(fitted["median"]), float(fitted["scale"])
                # 二元编码不是连续量；不因稀有阳性而用标准差放大它。
                if np.all(np.isin(observed, (0.0, 1.0))):
                    scale = 1.0
            else:
                median, scale = 0.0, 1.0
            medians.append(median)
            scales.append(scale)
            keep = selected & (data.token_masks[:, channel] > 0)
            transformed[keep, channel] = np.clip((data.token_values[keep, channel] - median) / scale,
                                                  -8.0, 8.0)
        parameters[concept_name] = {"median": medians, "scale": scales}
    transformed[data.token_masks == 0] = 0.0
    if not np.isfinite(transformed).all():
        raise ValueError("nonfinite normalized structured token")
    return replace(data, token_values=transformed), parameters


def split_sequences(data: IndependentData, split: str) -> list[np.ndarray]:
    selected = [sequence for sequence in data.sequences if data.splits[sequence[0]] == split]
    if not selected:
        raise ValueError(f"empty {split} partition")
    return selected


def collate(data: IndependentData, sequences: list[np.ndarray], *, include_text: bool) -> IndependentBatch:
    if not sequences:
        raise ValueError("empty patient batch")
    batch_size = len(sequences)
    length = max(len(sequence) for sequence in sequences)
    slots = max(len(data.tokens_by_admission[int(row)]) for sequence in sequences for row in sequence)
    values = np.zeros((batch_size, length, slots, 5), dtype=np.float32)
    masks = np.zeros_like(values)
    concepts = np.zeros((batch_size, length, slots), dtype=np.int64)
    kinds = np.zeros_like(concepts)
    token_present = np.zeros((batch_size, length, slots), dtype=bool)
    valid = np.zeros((batch_size, length), dtype=bool)
    gap = np.zeros((batch_size, length, 2), dtype=np.float32)
    times = np.zeros((batch_size, length), dtype=np.float32)
    events = np.zeros((batch_size, length), dtype=np.int64)
    snippets: list[tuple[int, int, int]] = []
    disabled_ids = [index + 1 for index, name in enumerate(data.concept_names)
                    if name in data.disabled_concepts]
    if set(data.disabled_concepts) - set(data.concept_names):
        raise ValueError("disabled static concept absent from vocabulary")
    for batch_index, sequence in enumerate(sequences):
        count = len(sequence)
        valid[batch_index, :count] = True
        gap[batch_index, :count] = data.gap[sequence]
        times[batch_index, :count] = data.times[sequence]
        events[batch_index, :count] = data.events[sequence]
        for position, row in enumerate(sequence):
            refs = data.tokens_by_admission[int(row)]
            count_tokens = len(refs)
            values[batch_index, position, :count_tokens] = data.token_values[refs]
            masks[batch_index, position, :count_tokens] = data.token_masks[refs]
            concepts[batch_index, position, :count_tokens] = data.token_concepts[refs]
            kinds[batch_index, position, :count_tokens] = data.token_types[refs]
            enabled = ~np.isin(data.token_concepts[refs], disabled_ids)
            token_present[batch_index, position, :count_tokens] = enabled
            # 保留槽位和词表尺寸，完全屏蔽字段，保持共有层初始化和 dropout 形状。
            inactive = np.flatnonzero(~enabled)
            values[batch_index, position, inactive] = 0
            masks[batch_index, position, inactive] = 0
            concepts[batch_index, position, inactive] = 0
            kinds[batch_index, position, inactive] = 0
            if include_text:
                snippets.extend((batch_index * length + position, field, vector_row)
                                for field, vector_row in data.text_by_admission[int(row)])
    if snippets:
        if data.text_vectors is None:
            raise ValueError("text requested without frozen vectors")
        positions = np.asarray(snippets, dtype=np.int64)
        vectors = np.asarray(data.text_vectors[positions[:, 2]], dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("nonfinite frozen text vector")
        index = positions[:, :2]
    else:
        vectors = np.zeros((0, data.text_width), dtype=np.float32)
        index = np.zeros((0, 2), dtype=np.int64)
    return IndependentBatch(
        torch.from_numpy(values), torch.from_numpy(masks), torch.from_numpy(concepts),
        torch.from_numpy(kinds), torch.from_numpy(token_present),
        torch.from_numpy(vectors), torch.from_numpy(index),
        torch.from_numpy(gap), torch.from_numpy(valid),
        torch.from_numpy(times), torch.from_numpy(events),
    )
