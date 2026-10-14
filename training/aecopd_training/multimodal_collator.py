"""把冻结章节向量与住院级结构化 batch 对齐。

风险训练阶段通过内存映射读取预计算 Qwen 向量，不读取原始正文，也不
加载或更新 Qwen 4B 参数。
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import torch

from timeline_collator import dual_anchor_window_id


TEXT_ROLES = (
    "<PAD>",
    "current_discharge",
    "current_radiology",
    "prior_discharge",
    "prior_radiology",
    "<UNK>",
)
TEXT_FEATURE_NAMES = (
    "token_count_log1p",
    "days_before_landmark_log1p",
    "document_rank_log1p",
)
SECTION_ROLES = (
    "<PAD>",
    "chief_complaint",
    "history_of_present_illness",
    "past_medical_history",
    "pertinent_results",
    "hospital_course",
    "discharge_diagnosis",
    "discharge_medications",
    "discharge_condition",
    "follow_up",
    "examination",
    "indication",
    "comparison",
    "technique",
    "findings",
    "impression",
    "other",
    "<UNK>",
)
TEXT_MODALITIES = (
    "<PAD>",
    "discharge_summary",
    "ct",
    "xray",
    "mri",
    "ultrasound",
    "nuclear_medicine",
    "other",
    "<UNK>",
)


class TextChunkLike(Protocol):
    """文本向量索引所需的最小章节元数据接口。"""

    document_key: str
    document_role: str
    document_rank: int
    days_before_landmark: float
    chunk_index: int
    token_count: int
    source_document_key: str
    modality: str
    section_role: str
    section_order: int
    section_instance: int


@dataclass(frozen=True)
class FrozenTextBatch:
    """补齐后的章节向量、文档映射、角色和位置张量。"""

    embeddings: torch.Tensor
    role_ids: torch.Tensor
    features: torch.Tensor
    attention_mask: torch.Tensor
    window_ids: torch.Tensor
    chunk_position_ids: torch.Tensor
    document_ids: torch.Tensor
    section_role_ids: torch.Tensor
    modality_ids: torch.Tensor
    section_position_ids: torch.Tensor
    section_instance_ids: torch.Tensor
    document_role_ids: torch.Tensor
    document_window_ids: torch.Tensor
    document_attention_mask: torch.Tensor
    lengths: list[int]
    document_lengths: list[int]

    def as_model_inputs(self) -> dict[str, torch.Tensor]:
        """返回与 ``TimeAlignedAecopdRiskModel`` 约定一致的输入键。"""
        return {
            "text_embeddings": self.embeddings,
            "text_role_ids": self.role_ids,
            "text_features": self.features,
            "text_attention_mask": self.attention_mask,
            "text_window_ids": self.window_ids,
            "text_chunk_position_ids": self.chunk_position_ids,
            "text_document_ids": self.document_ids,
            "text_section_role_ids": self.section_role_ids,
            "text_modality_ids": self.modality_ids,
            "text_section_position_ids": self.section_position_ids,
            "text_section_instance_ids": self.section_instance_ids,
            "text_document_role_ids": self.document_role_ids,
            "text_document_window_ids": self.document_window_ids,
            "text_document_attention_mask": self.document_attention_mask,
        }


class FrozenTextEmbeddingStore:
    """内存映射冻结 Qwen 向量，并在不读取正文的情况下组织文档。"""

    def __init__(
        self,
        embeddings_path: str | Path,
        index_path: str | Path,
        *,
        state_path: str | Path | None = None,
        require_complete: bool = True,
    ) -> None:
        self.embeddings_path = Path(embeddings_path)
        self.index_path = Path(index_path)
        self._vectors = np.load(self.embeddings_path, mmap_mode="r")
        if self._vectors.ndim != 2:
            raise ValueError("Text embedding store must have shape [rows, dimensions].")
        self.embedding_dim = int(self._vectors.shape[1])
        self._row_by_key: dict[tuple[str, int], int] = {}
        with gzip.open(self.index_path, "rt", encoding="utf-8-sig", newline="") as handle:
            for expected_row, row in enumerate(csv.DictReader(handle)):
                embedding_row = int(row["embedding_row"])
                if embedding_row != expected_row:
                    raise ValueError(f"Non-contiguous embedding_row at index row {expected_row}.")
                key = (row["document_key"], int(row["chunk_index"]))
                if key in self._row_by_key:
                    raise ValueError(f"Duplicate text embedding key: {key}")
                self._row_by_key[key] = embedding_row
        if len(self._row_by_key) != self._vectors.shape[0]:
            raise ValueError("Embedding index row count does not match NumPy store row count.")
        self.completed_rows = int(self._vectors.shape[0])
        if state_path is not None:
            state = json.loads(Path(state_path).read_text(encoding="utf-8"))
            if int(state["total_rows"]) != self._vectors.shape[0]:
                raise ValueError("Embedding state total_rows does not match NumPy store row count.")
            self.completed_rows = int(state["completed_rows"])
            if require_complete and state.get("status") != "complete":
                raise RuntimeError(
                    f"Frozen text embeddings are incomplete: {self.completed_rows}/{self._vectors.shape[0]} rows."
                )
        self._role_ids = {role: index for index, role in enumerate(TEXT_ROLES)}
        self._section_role_ids = {role: index for index, role in enumerate(SECTION_ROLES)}
        self._modality_ids = {modality: index for index, modality in enumerate(TEXT_MODALITIES)}

    def collate(
        self,
        chunks_by_admission: Sequence[Sequence[TextChunkLike]],
        *,
        hospital_los_hours: Sequence[float] | None = None,
        device: str | torch.device | None = None,
    ) -> FrozenTextBatch:
        batch_size = len(chunks_by_admission)
        lengths = [len(chunks) for chunks in chunks_by_admission]
        document_keys_by_admission = [
            list(
                dict.fromkeys(
                    (getattr(chunk, "source_document_key", "") or chunk.document_key)
                    for chunk in chunks
                )
            )
            for chunks in chunks_by_admission
        ]
        document_lengths = [len(keys) for keys in document_keys_by_admission]
        width = max(max(lengths, default=0), 1)
        document_width = max(max(document_lengths, default=0), 1)
        embeddings = np.zeros((batch_size, width, self.embedding_dim), dtype=np.float32)
        role_ids = np.zeros((batch_size, width), dtype=np.int64)
        features = np.zeros((batch_size, width, len(TEXT_FEATURE_NAMES)), dtype=np.float32)
        attention = np.zeros((batch_size, width), dtype=np.bool_)
        window_ids = np.zeros((batch_size, width), dtype=np.int64)
        chunk_position_ids = np.zeros((batch_size, width), dtype=np.int64)
        document_ids = np.zeros((batch_size, width), dtype=np.int64)
        section_role_ids = np.zeros((batch_size, width), dtype=np.int64)
        modality_ids = np.zeros((batch_size, width), dtype=np.int64)
        section_position_ids = np.zeros((batch_size, width), dtype=np.int64)
        section_instance_ids = np.zeros((batch_size, width), dtype=np.int64)
        document_role_ids = np.zeros((batch_size, document_width), dtype=np.int64)
        document_window_ids = np.zeros((batch_size, document_width), dtype=np.int64)
        document_attention = np.zeros((batch_size, document_width), dtype=np.bool_)
        if hospital_los_hours is not None and len(hospital_los_hours) != batch_size:
            raise ValueError("hospital_los_hours length must match the batch size")

        for batch_row, chunks in enumerate(chunks_by_admission):
            document_column_by_key = {
                key: column for column, key in enumerate(document_keys_by_admission[batch_row])
            }
            for column, chunk in enumerate(chunks):
                key = (chunk.document_key, int(chunk.chunk_index))
                try:
                    embedding_row = self._row_by_key[key]
                except KeyError as exc:
                    raise KeyError(f"Missing frozen text embedding for key: {key}") from exc
                if embedding_row >= self.completed_rows:
                    raise RuntimeError(
                        f"Frozen text embedding row {embedding_row} has not been generated; "
                        f"completed_rows={self.completed_rows}."
                    )
                vector = np.asarray(self._vectors[embedding_row], dtype=np.float32)
                if not np.isfinite(vector).all():
                    raise ValueError(f"Non-finite frozen text embedding at row {embedding_row}.")
                embeddings[batch_row, column] = vector
                role_ids[batch_row, column] = self._role_ids.get(chunk.document_role, self._role_ids["<UNK>"])
                features[batch_row, column] = (
                    math.log1p(max(0, int(chunk.token_count))),
                    math.log1p(max(0.0, float(chunk.days_before_landmark))),
                    math.log1p(max(0, int(chunk.document_rank))),
                )
                attention[batch_row, column] = True
                chunk_position_ids[batch_row, column] = max(0, int(chunk.chunk_index)) + 1
                source_document_key = (
                    getattr(chunk, "source_document_key", "") or chunk.document_key
                )
                document_column = document_column_by_key[source_document_key]
                document_ids[batch_row, column] = document_column + 1
                section_role = getattr(chunk, "section_role", "<UNK>") or "<UNK>"
                modality = getattr(chunk, "modality", "<UNK>") or "<UNK>"
                section_role_ids[batch_row, column] = self._section_role_ids.get(
                    section_role, self._section_role_ids["<UNK>"]
                )
                modality_ids[batch_row, column] = self._modality_ids.get(
                    modality, self._modality_ids["<UNK>"]
                )
                section_position_ids[batch_row, column] = (
                    max(0, int(getattr(chunk, "section_order", chunk.chunk_index))) + 1
                )
                section_instance_ids[batch_row, column] = max(
                    1, int(getattr(chunk, "section_instance", 1))
                )
                expected_role_id = self._role_ids.get(
                    chunk.document_role, self._role_ids["<UNK>"]
                )
                if document_attention[batch_row, document_column]:
                    if document_role_ids[batch_row, document_column] != expected_role_id:
                        raise ValueError(
                            f"Inconsistent document role for source_document_key={source_document_key}"
                        )
                else:
                    document_attention[batch_row, document_column] = True
                    document_role_ids[batch_row, document_column] = expected_role_id
                if chunk.document_role == "current_radiology" and hospital_los_hours is not None:
                    hours_before = max(0.0, float(chunk.days_before_landmark) * 24.0)
                    hours_from = max(0.0, float(hospital_los_hours[batch_row]) - hours_before)
                    window_ids[batch_row, column] = dual_anchor_window_id(hours_from, hours_before)
                document_window = window_ids[batch_row, column]
                existing_document_window = document_window_ids[batch_row, document_column]
                if existing_document_window not in {0, document_window}:
                    raise ValueError(
                        f"Inconsistent document window for source_document_key={source_document_key}"
                    )
                document_window_ids[batch_row, document_column] = document_window

        return FrozenTextBatch(
            embeddings=torch.as_tensor(embeddings, dtype=torch.float32, device=device),
            role_ids=torch.as_tensor(role_ids, dtype=torch.long, device=device),
            features=torch.as_tensor(features, dtype=torch.float32, device=device),
            attention_mask=torch.as_tensor(attention, dtype=torch.bool, device=device),
            window_ids=torch.as_tensor(window_ids, dtype=torch.long, device=device),
            chunk_position_ids=torch.as_tensor(chunk_position_ids, dtype=torch.long, device=device),
            document_ids=torch.as_tensor(document_ids, dtype=torch.long, device=device),
            section_role_ids=torch.as_tensor(section_role_ids, dtype=torch.long, device=device),
            modality_ids=torch.as_tensor(modality_ids, dtype=torch.long, device=device),
            section_position_ids=torch.as_tensor(
                section_position_ids, dtype=torch.long, device=device
            ),
            section_instance_ids=torch.as_tensor(
                section_instance_ids, dtype=torch.long, device=device
            ),
            document_role_ids=torch.as_tensor(
                document_role_ids, dtype=torch.long, device=device
            ),
            document_window_ids=torch.as_tensor(
                document_window_ids, dtype=torch.long, device=device
            ),
            document_attention_mask=torch.as_tensor(
                document_attention, dtype=torch.bool, device=device
            ),
            lengths=lengths,
            document_lengths=document_lengths,
        )


class MultimodalBatchCollator:
    """Combine the shared structured timeline with frozen text vectors."""

    def __init__(self, timeline_collator: Any, text_store: FrozenTextEmbeddingStore) -> None:
        self.timeline_collator = timeline_collator
        self.text_store = text_store

    def __call__(
        self,
        samples: Sequence[Any],
        *,
        device: str | torch.device | None = None,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        timeline_batch = self.timeline_collator(samples)
        tensors = timeline_batch.to_torch(device)
        text_batch = self.text_store.collate(
            timeline_batch.text_chunks,
            hospital_los_hours=[float(sample.hospital_los_hours) for sample in samples],
            device=device,
        )
        tensors.update(text_batch.as_model_inputs())
        return timeline_batch, tensors


class StructuredBatchCollator:
    """只输出结构化时间轴张量的住院级适配器。

    它与 ``MultimodalBatchCollator`` 满足同一个调用界面，因此纵向批处理、
    训练和评价代码不需要知道文本是否开启。
    """

    def __init__(self, timeline_collator: Any) -> None:
        self.timeline_collator = timeline_collator

    def __call__(
        self,
        samples: Sequence[Any],
        *,
        device: str | torch.device | None = None,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        timeline_batch = self.timeline_collator(samples)
        return timeline_batch, timeline_batch.to_torch(device)
