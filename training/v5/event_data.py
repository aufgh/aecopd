"""把本次住院的原始化验和实际给药排成不规则事件流。"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from training.v5.clinical_concepts import LABS, MED_NAMES, medication_classes
from training.v5.independent_data import IndependentData, collate, load_data, normalize_on_training_patients
from training.v5.independent_fusion import IndependentBatch
from training.v5.numeric_preprocessing import robust_parameters


LAB_NAMES = tuple(sorted({name for name, _ in LABS.values()}))
EVENT_NAMES = LAB_NAMES + MED_NAMES


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EventData:
    base: IndependentData
    event_values: np.ndarray
    event_concepts: np.ndarray
    event_is_lab: np.ndarray
    events_by_admission: tuple[np.ndarray, ...]
    source_hashes: dict[str, str]
    concept_names: tuple[str, ...] = EVENT_NAMES
    event_numeric_mask: np.ndarray | None = None
    recovery_by_admission: np.ndarray | None = None
    medication_by_admission: np.ndarray | None = None
    abnormality_scaling: dict[str, object] | None = None


@dataclass(frozen=True)
class EventBatch:
    base: IndependentBatch
    event_values: torch.Tensor      # [N,4]: normalized lab value, elapsed, gap, before discharge
    event_concepts: torch.Tensor    # [N]
    event_counts: torch.Tensor      # [B,L]，对应每次住院在展平数组中的连续片段
    event_is_lab: torch.Tensor | None = None
    event_numeric_mask: torch.Tensor | None = None
    recovery_by_admission: torch.Tensor | None = None
    medication_by_admission: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> EventBatch:
        return EventBatch(self.base.to(device), self.event_values.to(device),
                          self.event_concepts.to(device), self.event_counts.to(device),
                          None if self.event_is_lab is None else self.event_is_lab.to(device),
                          None if self.event_numeric_mask is None else self.event_numeric_mask.to(device),
                          None if self.recovery_by_admission is None else self.recovery_by_admission.to(device),
                          None if self.medication_by_admission is None else self.medication_by_admission.to(device))


def load_event_data(aligned_dir: Path, structured_dir: Path, base_csv: Path,
                    raw_dir: Path, *, labels_csv: Path, field_dir: Path | None = None,
                    vectors_path: Path | None = None) -> tuple[EventData, dict[str, object]]:
    """仅保留静态结构化 token；化验与药物摘要由原始有时标事件替代。"""

    base, static_normalization = normalize_on_training_patients(
        load_data(aligned_dir, structured_dir, base_csv, field_dir, vectors_path,
                  labels_csv=labels_csv)
    )
    keep_types = {index + 1 for index, name in enumerate(base.type_names)
                  if name not in {"laboratory", "administered_medication"}}
    static_refs = tuple(refs[np.isin(base.token_types[refs], list(keep_types))]
                        for refs in base.tokens_by_admission)
    if any(len(refs) == 0 for refs in static_refs):
        raise ValueError("admission lost all static context tokens")
    base = replace(base, tokens_by_admission=static_refs)
    raw_qa = json.loads((raw_dir / "export_summary.json").read_text(encoding="utf-8"))
    source_hashes = {}
    for name in ("labs", "emar"):
        path = raw_dir / f"{name}.csv.gz"
        source_hashes[name] = _sha(path)
        if source_hashes[name] != raw_qa["sources"][name]["sha256"]:
            raise ValueError(f"raw {name} source changed")
    hadm_to_index = {str(hadm): index for index, hadm in enumerate(base.hadm_ids)}
    landmarks: dict[str, tuple[datetime, datetime]] = {}
    with base_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            landmarks[row["hadm_id"]] = (datetime.fromisoformat(row["admittime"]),
                                         datetime.fromisoformat(row["dischtime"]))
    concept_id = {name: index + 1 for index, name in enumerate(EVENT_NAMES)}
    source_events: list[list[tuple[datetime, int, float, bool, str]]] = [
        [] for _ in range(len(base.hadm_ids))
    ]
    with gzip.open(raw_dir / "labs.csv.gz", "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            hadm = row["hadm_id"]
            index = hadm_to_index.get(hadm)
            if index is None:
                raise ValueError("laboratory event outside cohort")
            moment = datetime.fromisoformat(row["charttime"])
            admission, discharge = landmarks[hadm]
            if not admission <= moment <= discharge:
                raise ValueError("laboratory event outside admission")
            concept = concept_id[LABS[int(row["itemid"])][0]]
            source_events[index].append((moment, concept, float(row["valuenum"]), True,
                                         row["labevent_id"]))
    with gzip.open(raw_dir / "emar.csv.gz", "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            hadm = row["hadm_id"]
            index = hadm_to_index.get(hadm)
            if index is None:
                raise ValueError("administration event outside cohort")
            moment = datetime.fromisoformat(row["charttime"])
            admission, discharge = landmarks[hadm]
            if not admission <= moment <= discharge:
                raise ValueError("administration event outside admission")
            for name in medication_classes(row["medication"]):
                source_events[index].append((moment, concept_id[name], 0.0, False,
                                             row["emar_id"] + ":" + row["emar_seq"]))

    raw_values: list[list[float]] = []
    concepts: list[int] = []
    lab_flags: list[bool] = []
    admission_refs: list[np.ndarray] = []
    for index, records in enumerate(source_events):
        records.sort(key=lambda item: (item[0], item[1], item[4]))
        start = len(raw_values)
        hadm = str(base.hadm_ids[index])
        admission, discharge = landmarks[hadm]
        previous = admission
        seen: set[tuple[datetime, int, str]] = set()
        for moment, concept, value, is_lab, source_id in records:
            key = (moment, concept, source_id)
            if key in seen:
                continue
            seen.add(key)
            elapsed = max(0.0, (moment - admission).total_seconds() / 86400.0)
            gap = max(0.0, (moment - previous).total_seconds() / 86400.0)
            before_discharge = max(0.0, (discharge - moment).total_seconds() / 86400.0)
            raw_values.append([value if is_lab else 0.0, math.log1p(elapsed),
                               math.log1p(gap), math.log1p(before_discharge)])
            concepts.append(concept)
            lab_flags.append(is_lab)
            previous = moment
        admission_refs.append(np.arange(start, len(raw_values), dtype=np.int64))
    values = np.asarray(raw_values, dtype=np.float32).reshape(-1, 4)
    concept_array = np.asarray(concepts, dtype=np.int64)
    lab_array = np.asarray(lab_flags, dtype=bool)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite event values")
    train_hadms = set(base.hadm_ids[base.splits == "train"].tolist())
    train_events = np.zeros(len(values), dtype=bool)
    for index, refs in enumerate(admission_refs):
        if int(base.hadm_ids[index]) in train_hadms:
            train_events[refs] = True
    event_normalization: dict[str, dict[str, float]] = {}
    for name in LAB_NAMES:
        selected = lab_array & (concept_array == concept_id[name])
        observed = values[selected & train_events, 0]
        if observed.size == 0:
            raise ValueError(f"no training measurements for {name}")
        fitted = robust_parameters(observed)
        median, scale = float(fitted["median"]), float(fitted["scale"])
        values[selected, 0] = np.clip((values[selected, 0] - median) / scale, -8.0, 8.0)
        event_normalization[name] = {"median": median, "scale": scale}
    return (EventData(base, values, concept_array, lab_array,
                      tuple(admission_refs), source_hashes),
            {"static": static_normalization, "event_lab": event_normalization})


def collate_events(data: EventData, sequences: list[np.ndarray], *, include_text: bool,
                   process_only: bool = False,
                   shuffle_lab_order_seed: int | None = None) -> EventBatch:
    if process_only and shuffle_lab_order_seed is not None:
        raise ValueError("process-only and lab-value order shuffle are separate ablations")
    base_batch = collate(data.base, sequences, include_text=include_text)
    batch_size = len(sequences)
    length = max(len(sequence) for sequence in sequences)
    counts = np.zeros((batch_size, length), dtype=np.int64)
    refs_by_stay: list[np.ndarray] = []
    for patient, sequence in enumerate(sequences):
        for position, row in enumerate(sequence):
            refs = data.events_by_admission[int(row)]
            counts[patient, position] = len(refs)
            refs_by_stay.append(refs)
        refs_by_stay.extend([np.empty(0, dtype=np.int64)] * (length - len(sequence)))
    refs = np.concatenate(refs_by_stay)
    values = data.event_values[refs].copy()
    concepts = data.event_concepts[refs]
    if shuffle_lab_order_seed is not None:
        offset = 0
        for patient, sequence in enumerate(sequences):
            for position in range(length):
                n = int(counts[patient, position])
                if position < len(sequence):
                    rng = np.random.default_rng(
                        shuffle_lab_order_seed + int(data.base.hadm_ids[int(sequence[position])])
                    )
                    for concept in np.unique(concepts[offset:offset + n]):
                        local = np.flatnonzero(concepts[offset:offset + n] == concept) + offset
                        if len(local) > 1 and data.event_is_lab[refs[local[0]]]:
                            values[local, 0] = rng.permutation(values[local, 0])
                offset += n
    if process_only:
        # 保留事件类型、频率与时间，仅去掉检验数值；用于诊疗过程对照。
        values[:, 0] = 0.0
    recovery = None
    if data.recovery_by_admission is not None:
        if data.recovery_by_admission.shape[0] != len(data.base.hadm_ids):
            raise ValueError("recovery/admission count mismatch")
        recovery = np.zeros((batch_size, length, data.recovery_by_admission.shape[1]), dtype=np.float32)
        for patient, sequence in enumerate(sequences):
            recovery[patient, :len(sequence)] = data.recovery_by_admission[sequence]
    medication = None
    if data.medication_by_admission is not None:
        if data.medication_by_admission.shape[0] != len(data.base.hadm_ids):
            raise ValueError("medication/admission count mismatch")
        medication = np.zeros((batch_size, length, data.medication_by_admission.shape[1]), dtype=np.float32)
        for patient, sequence in enumerate(sequences):
            medication[patient, :len(sequence)] = data.medication_by_admission[sequence]
    return EventBatch(base_batch, torch.from_numpy(values),
                      torch.from_numpy(concepts), torch.from_numpy(counts),
                      torch.from_numpy(data.event_is_lab[refs]),
                      torch.from_numpy(data.event_is_lab[refs] if data.event_numeric_mask is None
                                       else data.event_numeric_mask[refs]),
                      None if recovery is None else torch.from_numpy(recovery),
                      None if medication is None else torch.from_numpy(medication))
