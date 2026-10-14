"""受控患者纵向缓存的构建与只读访问。

基础加载器为流式 hadm_id 顺序，不能高效按患者随机读取完整序列。本模块
将样本压缩写入 SQLite；缓存仍含患者级训练输入，必须与原数据同级保护。
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import pickle
import random
import sqlite3
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

from .longitudinal_collator import PatientSequenceInput


class SampleBundle(Protocol):
    """构建缓存所需的流式数据包最小接口。"""

    def iter_samples(
        self,
        split: str | None,
        *,
        artifacts: Mapping[str, Any],
        include_text: bool,
    ) -> Iterator[Any]: ...


def _read_sequence_index(path: Path) -> tuple[dict[int, dict[str, Any]], dict[int, list[int]]]:
    """读取并验证住院到患者、位置、划分和间隔的唯一映射。"""
    admissions: dict[int, dict[str, Any]] = {}
    patients: dict[int, list[int]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            hadm_id = int(row["hadm_id"])
            subject_id = int(row["subject_id"])
            if hadm_id in admissions:
                raise ValueError(f"duplicate hadm_id in patient sequence index: {hadm_id}")
            sequence_index = int(row["sequence_index"])
            raw_gap = (row.get("days_since_previous_discharge") or "").strip()
            gap_days = 0.0 if sequence_index == 0 else float(raw_gap)
            if gap_days < 0:
                raise ValueError(f"negative longitudinal gap for hadm_id={hadm_id}")
            admissions[hadm_id] = {
                "subject_id": subject_id,
                "data_split": row["data_split"],
                "sequence_index": sequence_index,
                "gap_days": gap_days,
            }
            patients[subject_id].append(hadm_id)

    if not admissions:
        raise ValueError("patient sequence index is empty")
    for subject_id, hadm_ids in patients.items():
        ordered = sorted(hadm_ids, key=lambda value: admissions[value]["sequence_index"])
        positions = [admissions[value]["sequence_index"] for value in ordered]
        if positions != list(range(len(ordered))):
            raise ValueError(f"non-contiguous sequence positions for subject_id={subject_id}")
        splits = {admissions[value]["data_split"] for value in ordered}
        if len(splits) != 1:
            raise ValueError(f"patient crosses data splits: subject_id={subject_id}")
        patients[subject_id] = ordered
    return admissions, dict(patients)


def build_patient_sequence_cache(
    bundle: SampleBundle,
    *,
    artifacts: Mapping[str, Any],
    sequence_index_path: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """把流式住院样本物化为可按患者读取的本地压缩缓存。"""
    target = Path(output_path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"patient sequence cache already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    index_path = Path(sequence_index_path)
    admission_index, patient_index = _read_sequence_index(index_path)
    observed: set[int] = set()
    split_admissions: Counter[str] = Counter()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE patients (
                subject_id INTEGER PRIMARY KEY,
                data_split TEXT NOT NULL,
                admission_count INTEGER NOT NULL
            );
            CREATE TABLE admissions (
                hadm_id INTEGER PRIMARY KEY,
                subject_id INTEGER NOT NULL,
                data_split TEXT NOT NULL,
                sequence_index INTEGER NOT NULL,
                gap_days REAL NOT NULL,
                payload BLOB NOT NULL,
                UNIQUE(subject_id, sequence_index)
            );
            CREATE INDEX admissions_by_patient
                ON admissions(subject_id, sequence_index);
            """
        )
        for sample in bundle.iter_samples(None, artifacts=artifacts, include_text=True):
            hadm_id = int(sample.hadm_id)
            try:
                indexed = admission_index[hadm_id]
            except KeyError as exc:
                raise ValueError(f"sample is absent from patient sequence index: hadm_id={hadm_id}") from exc
            if hadm_id in observed:
                raise ValueError(f"duplicate sample while building cache: hadm_id={hadm_id}")
            if int(sample.subject_id) != indexed["subject_id"] or sample.data_split != indexed["data_split"]:
                raise ValueError(f"sample/index mismatch for hadm_id={hadm_id}")
            payload = zlib.compress(pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL), level=3)
            connection.execute(
                "INSERT INTO admissions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    hadm_id,
                    indexed["subject_id"],
                    indexed["data_split"],
                    indexed["sequence_index"],
                    indexed["gap_days"],
                    payload,
                ),
            )
            observed.add(hadm_id)
            split_admissions[indexed["data_split"]] += 1

        missing = set(admission_index) - observed
        if missing:
            preview = sorted(missing)[:5]
            raise ValueError(f"cache source is missing {len(missing)} indexed admissions; first={preview}")

        split_patients: Counter[str] = Counter()
        for subject_id, hadm_ids in patient_index.items():
            split = admission_index[hadm_ids[0]]["data_split"]
            connection.execute(
                "INSERT INTO patients VALUES (?, ?, ?)",
                (subject_id, split, len(hadm_ids)),
            )
            split_patients[split] += 1
        summary: dict[str, Any] = {
            "schema_version": "patient_sequence_cache_v1",
            "patients": len(patient_index),
            "admissions": len(admission_index),
            "split_patients": dict(sorted(split_patients.items())),
            "split_admissions": dict(sorted(split_admissions.items())),
            "sequence_index": str(index_path.resolve()),
            "compression": "pickle_protocol_highest+zlib_level_3",
        }
        connection.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            ("summary", json.dumps(summary, ensure_ascii=False, sort_keys=True)),
        )
        connection.commit()
    except BaseException:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        connection.close()
    os.replace(temporary, target)
    return summary


class PatientSequenceCache:
    """从本地缓存读取完整且有序的患者住院序列。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA query_only=ON")

    def __enter__(self) -> "PatientSequenceCache":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    @property
    def summary(self) -> dict[str, Any]:
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key='summary'"
        ).fetchone()
        if row is None:
            raise ValueError("patient sequence cache has no summary metadata")
        return json.loads(row[0])

    def iter_sequences(
        self,
        split: str,
        *,
        shuffle: bool,
        seed: int = 0,
        epoch: int = 0,
    ) -> Iterator[PatientSequenceInput]:
        """逐患者返回完整序列；只在患者之间洗牌，不截断患者内部。"""
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        subject_ids = [
            int(row[0])
            for row in self._connection.execute(
                "SELECT subject_id FROM patients WHERE data_split=? ORDER BY subject_id",
                (split,),
            )
        ]
        if shuffle:
            random.Random(seed + epoch).shuffle(subject_ids)
        for subject_id in subject_ids:
            rows = self._connection.execute(
                "SELECT gap_days, payload FROM admissions WHERE subject_id=? ORDER BY sequence_index",
                (subject_id,),
            ).fetchall()
            admissions = tuple(pickle.loads(zlib.decompress(row[1])) for row in rows)
            gaps = tuple(float(row[0]) for row in rows)
            yield PatientSequenceInput(admissions=admissions, gap_days=gaps)
