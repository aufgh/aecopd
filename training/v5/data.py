"""将出院预测样本集转为六组结构化输入与完整患者序列。"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from training.v5.clinical_concepts import LABS, MED_NAMES, diagnosis_groups, medication_classes


LAB_ORDER = tuple(LABS)
GROUP_NAMES = ("baseline", "chronic_comorbidity", "current_diagnosis",
               "blood_gas", "other_labs", "treatment_and_discharge")
FIELD_NAMES = ("residual_dyspnea", "residual_cough_sputum", "oxygen_or_niv_at_discharge",
               "mobility_at_discharge", "inhaler_use_or_adherence", "social_support",
               "steroid_or_antibiotic_discharge_plan", "unresolved_infection_or_imaging")
GROUP_WIDTHS = (4, 6, 5, 15, 25, 16)

def patient_split(subject_id: int) -> str:
    digest = hashlib.md5(f"aecopd-model-v1:{subject_id}".encode("ascii")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


@dataclass(frozen=True)
class V5Data:
    subject_ids: np.ndarray
    hadm_ids: np.ndarray
    splits: np.ndarray
    times: np.ndarray
    events: np.ndarray
    group_values: tuple[np.ndarray, ...]
    group_masks: tuple[np.ndarray, ...]
    gap_features: np.ndarray
    sequences: tuple[np.ndarray, ...]
    snippets: tuple[tuple[tuple[int, int], ...], ...]
    text_vectors: np.ndarray | None
    text_width: int
    coverage: dict[str, object]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_data(structured_dir: Path, manifest: Path | None = None,
              vectors_path: Path | None = None, *, vector_width: int | None = None) -> V5Data:
    """严格按住院键连接，保持每位患者住院时间序列并隔离患者划分。"""

    base = _rows(structured_dir / "base.csv")
    if len(base) != 19401:
        raise ValueError("V5 requires the full discharge prediction sample set")
    hadm_ids = np.array([int(row["hadm_id"]) for row in base], dtype=np.int64)
    subject_ids = np.array([int(row["subject_id"]) for row in base], dtype=np.int64)
    if len(set(hadm_ids.tolist())) != len(base):
        raise ValueError("duplicate admission")
    row_by_hadm = {int(hadm): index for index, hadm in enumerate(hadm_ids)}
    splits = np.array([patient_split(int(subject)) for subject in subject_ids])
    times = np.array([float(row["time_days"]) for row in base], dtype=np.float32)
    events = np.array([int(row["event_type"]) for row in base], dtype=np.int64)
    if not np.isfinite(times).all() or np.any((times < 0) | (times > 365)):
        raise ValueError("invalid event time")
    if not np.isin(events, (0, 1, 2)).all() or np.any((events == 0) & (times != 365)):
        raise ValueError("invalid event type or administrative censoring")
    values = tuple(np.zeros((len(base), width), dtype=np.float32) for width in GROUP_WIDTHS)
    masks = tuple(np.zeros_like(group) for group in values)
    gap = np.zeros((len(base), 2), dtype=np.float32)
    patient_rows: dict[int, list[int]] = defaultdict(list)
    previous_discharge: dict[int, datetime] = {}
    for index, row in enumerate(base):
        subject = int(row["subject_id"])
        admission, discharge = _time(row["admittime"]), _time(row["dischtime"])
        if discharge <= admission:
            raise ValueError("ineligible discharge in V5 base")
        if patient_rows[subject] and admission < _time(base[patient_rows[subject][-1]]["admittime"]):
            raise ValueError("patient admissions are not chronologically ordered")
        patient_rows[subject].append(index)
        previous = previous_discharge.get(subject)
        if previous is not None:
            days = max(0.0, (admission - previous).total_seconds() / 86400)
            gap[index] = (math.log1p(days), float(admission < previous))
        previous_discharge[subject] = discharge
        values[0][index] = (
            float(row["age_at_admission"]), float(row["gender"] == "M"),
            math.log1p(int(row["prior_coded_admission_count"])),
            math.log1p(int(row["prior_strict_aecopd_count"])),
        )
        masks[0][index] = 1.0
        values[2][index, 0] = float(row["index_is_strict_aecopd"] in {"t", "true", "True", "1"})
        masks[2][index] = 1.0
        values[5][index, 0] = (discharge - admission).total_seconds() / 86400
        destination = row["discharge_location"].upper()
        values[5][index, 1:4] = (
            float("HOME" in destination),
            float(any(term in destination for term in ("REHAB", "SKILLED", "NURSING", "LONG TERM"))),
            float(bool(destination) and "HOME" not in destination
                  and not any(term in destination for term in ("REHAB", "SKILLED", "NURSING", "LONG TERM"))),
        )
        masks[5][index, :4] = 1.0
        masks[1][index] = 1.0
    lab_count = 0
    seen_labs: set[tuple[int, int]] = set()
    for row in _rows(structured_dir / "labs.csv"):
        hadm, itemid = int(row["hadm_id"]), int(row["itemid"])
        if hadm not in row_by_hadm or itemid not in LABS or (hadm, itemid) in seen_labs:
            raise ValueError("lab key mismatch or duplicate")
        seen_labs.add((hadm, itemid))
        index = row_by_hadm[hadm]
        group = LABS[itemid][1]
        position = (LAB_ORDER[:3] if group == 3 else LAB_ORDER[3:]).index(itemid) * 5
        first, last, minimum, maximum = (float(row[key]) for key in
                                          ("first_value", "last_value", "min_value", "max_value"))
        if not all(math.isfinite(number) and number >= 0 for number in (first, last, minimum, maximum)):
            raise ValueError("invalid lab value")
        values[group][index, position:position + 4] = first, last, minimum, maximum
        masks[group][index, position:position + 4] = 1.0
        if int(row["measurement_count"]) >= 2:
            values[group][index, position + 4] = last - first
            masks[group][index, position + 4] = 1.0
        lab_count += 1
    diagnosis_count = 0
    chronic = ("heart_failure", "cardiac_other", "ckd", "diabetes", "malignancy", "chronic_respiratory")
    acute = ("acute_resp_failure", "pneumonia_or_aspiration", "sepsis", "aki")
    for row in _rows(structured_dir / "diagnoses.csv"):
        hadm = int(row["hadm_id"])
        if hadm not in row_by_hadm:
            raise ValueError("diagnosis outside cohort")
        index = row_by_hadm[hadm]
        for concept in diagnosis_groups(row["icd_code"], int(row["icd_version"])):
            if concept in chronic:
                values[1][index, chronic.index(concept)] = 1.0
            else:
                values[2][index, 1 + acute.index(concept)] = 1.0
        diagnosis_count += 1
    medication_count = 0
    hadms_with_emar: set[int] = set()
    for row in _rows(structured_dir / "medications.csv"):
        hadm = int(row["hadm_id"])
        if hadm not in row_by_hadm:
            raise ValueError("eMAR medication outside cohort")
        index = row_by_hadm[hadm]
        hadms_with_emar.add(hadm)
        for concept in medication_classes(row["medication"]):
            values[5][index, 4 + MED_NAMES.index(concept)] = 1.0
        medication_count += 1
    for hadm in hadms_with_emar:
        masks[5][row_by_hadm[hadm], 4:] = 1.0
    snippets: list[list[tuple[int, int]]] = [[] for _ in base]
    text_vectors: np.ndarray | None = None
    text_width = 768
    text_encoder = "none"
    if manifest is not None or vectors_path is not None:
        if manifest is None or vectors_path is None:
            raise ValueError("manifest and text vectors must both be supplied")
        state_path = vectors_path.parent / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") != "complete" or state.get("manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest():
            raise ValueError("incomplete or mismatched text encoding")
        text_vectors = np.load(vectors_path, mmap_mode="r")
        if text_vectors.ndim != 2 or text_vectors.shape[1] not in (768, 2560):
            raise ValueError("expected Bio_ClinicalBERT 768- or Qwen3 2560-dimensional vectors")
        if text_vectors.shape[1] == 768 and state.get("model") != "Bio_ClinicalBERT":
            raise ValueError("BERT vector metadata mismatch")
        if text_vectors.shape[1] == 2560 and state.get("encoder") != "Qwen3-Embedding-4B":
            raise ValueError("Qwen vector metadata mismatch")
        text_encoder = "Bio_ClinicalBERT" if text_vectors.shape[1] == 768 else "Qwen3-Embedding-4B"
        text_width = vector_width or int(text_vectors.shape[1])
        if text_width != text_vectors.shape[1] and not (
            text_vectors.shape[1] == 2560 and text_width in (128, 768)
        ):
            raise ValueError("unsupported Qwen Matryoshka prefix dimension")
        text_rows = 0
        with gzip.open(manifest, "rt", encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle)):
                hadm = int(row["hadm_id"])
                if hadm not in row_by_hadm or int(row["embedding_row"]) != row_number:
                    raise ValueError("text manifest key/order mismatch")
                snippets[row_by_hadm[hadm]].append((FIELD_NAMES.index(row["field"]), row_number))
                text_rows += 1
        if text_rows != text_vectors.shape[0]:
            raise ValueError("text vector count mismatch")
    sequences = tuple(np.array(indices, dtype=np.int64) for indices in patient_rows.values())
    coverage: dict[str, object] = {
        "patients": len(sequences), "admissions": len(base), "lab_rows": lab_count,
        "diagnosis_rows": diagnosis_count, "emar_medication_rows": medication_count,
        "admissions_with_emar": len(hadms_with_emar),
        "admissions_with_rule_text": sum(bool(items) for items in snippets),
        "rule_snippets": sum(map(len, snippets)),
        "text_encoder": text_encoder,
    }
    return V5Data(subject_ids, hadm_ids, splits, times, events, values, masks,
                  gap, sequences, tuple(tuple(items) for items in snippets),
                  text_vectors, text_width, coverage)


@dataclass(frozen=True)
class V5Batch:
    group_values: tuple[torch.Tensor, ...]
    group_masks: tuple[torch.Tensor, ...]
    gap: torch.Tensor
    valid: torch.Tensor
    times: torch.Tensor
    events: torch.Tensor
    snippet_vectors: torch.Tensor
    snippet_index: torch.Tensor

    def to(self, device: torch.device | str) -> V5Batch:
        return V5Batch(tuple(value.to(device) for value in self.group_values),
                       tuple(mask.to(device) for mask in self.group_masks),
                       self.gap.to(device), self.valid.to(device), self.times.to(device),
                       self.events.to(device), self.snippet_vectors.to(device),
                       self.snippet_index.to(device))


def collate(data: V5Data, sequences: list[np.ndarray], *, include_text: bool) -> V5Batch:
    if not sequences:
        raise ValueError("empty patient batch")
    batch_size, length = len(sequences), max(len(sequence) for sequence in sequences)
    valid = np.zeros((batch_size, length), dtype=bool)
    groups = [np.zeros((batch_size, length, width), dtype=np.float32) for width in GROUP_WIDTHS]
    masks = [np.zeros_like(group) for group in groups]
    gap = np.zeros((batch_size, length, 2), dtype=np.float32)
    times = np.zeros((batch_size, length), dtype=np.float32)
    events = np.zeros((batch_size, length), dtype=np.int64)
    snippet_refs: list[tuple[int, int, int]] = []
    for batch_index, sequence in enumerate(sequences):
        count = len(sequence)
        valid[batch_index, :count] = True
        for group_index in range(len(GROUP_WIDTHS)):
            groups[group_index][batch_index, :count] = data.group_values[group_index][sequence]
            masks[group_index][batch_index, :count] = data.group_masks[group_index][sequence]
        gap[batch_index, :count] = data.gap_features[sequence]
        times[batch_index, :count] = data.times[sequence]
        events[batch_index, :count] = data.events[sequence]
        if include_text:
            for position, row_index in enumerate(sequence):
                snippet_refs.extend((batch_index * length + position, field, vector_row)
                                    for field, vector_row in data.snippets[int(row_index)])
    if snippet_refs:
        if data.text_vectors is None:
            raise ValueError("text requested without embeddings")
        positions = np.asarray(snippet_refs, dtype=np.int64)
        vectors = np.asarray(data.text_vectors[positions[:, 2], :data.text_width], dtype=np.float32)
        if data.text_width != data.text_vectors.shape[1]:
            # Qwen 的 Matryoshka 前缀裁剪后重新做 L2 归一化。
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
        if not np.isfinite(vectors).all():
            raise ValueError("nonfinite text vectors")
        index = positions[:, :2]
    else:
        vectors = np.zeros((0, data.text_width), dtype=np.float32)
        index = np.zeros((0, 2), dtype=np.int64)
    return V5Batch(tuple(torch.from_numpy(group) for group in groups),
                   tuple(torch.from_numpy(mask) for mask in masks),
                   torch.from_numpy(gap), torch.from_numpy(valid),
                   torch.from_numpy(times), torch.from_numpy(events),
                   torch.from_numpy(vectors), torch.from_numpy(index))
