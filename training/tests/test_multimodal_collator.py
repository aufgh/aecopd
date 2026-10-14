from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from aecopd_training.multimodal_collator import FrozenTextEmbeddingStore, StructuredBatchCollator


@dataclass(frozen=True)
class Chunk:
    document_key: str
    document_role: str
    document_rank: int
    days_before_landmark: float
    chunk_index: int
    token_count: int
    source_document_key: str = ""
    modality: str = "<UNK>"
    section_role: str = "<UNK>"
    section_order: int = 0
    section_instance: int = 1


class FrozenTextEmbeddingStoreTests(unittest.TestCase):
    def test_structured_collator_returns_no_text_model_inputs(self) -> None:
        class TimelineBatch:
            def to_torch(self, device: str | None = None) -> dict[str, torch.Tensor]:
                return {"timeline_concept_ids": torch.tensor([[2]], device=device)}

        class TimelineCollator:
            def __call__(self, samples: object) -> TimelineBatch:
                return TimelineBatch()

        batch, tensors = StructuredBatchCollator(TimelineCollator())([object()], device="cpu")

        self.assertIsInstance(batch, TimelineBatch)
        self.assertEqual(set(tensors), {"timeline_concept_ids"})

    def test_collate_preserves_document_order_and_pads_missing_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "vectors.npy"
            index = root / "index.csv.gz"
            np.save(embeddings, np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float16))
            with gzip.open(index, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["embedding_row", "document_key", "chunk_index"])
                writer.writeheader()
                writer.writerow({"embedding_row": 0, "document_key": "a", "chunk_index": 0})
                writer.writerow({"embedding_row": 1, "document_key": "b", "chunk_index": 0})
            store = FrozenTextEmbeddingStore(embeddings, index)

            result = store.collate([
                [Chunk("b", "prior_radiology", 2, 5.0, 0, 100), Chunk("a", "current_discharge", 1, 0.0, 0, 20)],
                [],
            ])

            self.assertEqual(result.embeddings.shape, (2, 2, 3))
            self.assertEqual(result.lengths, [2, 0])
            self.assertEqual(result.embeddings[0, 0].tolist(), [4.0, 5.0, 6.0])
            self.assertEqual(result.embeddings[0, 1].tolist(), [1.0, 2.0, 3.0])
            self.assertEqual(result.attention_mask.tolist(), [[True, True], [False, False]])
            self.assertEqual(result.role_ids[0].tolist(), [4, 1])
            self.assertEqual(result.chunk_position_ids[0].tolist(), [1, 1])

    def test_current_radiology_is_aligned_to_shared_24_hour_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "vectors.npy"
            index = root / "index.csv.gz"
            np.save(embeddings, np.asarray([[1, 2]], dtype=np.float16))
            with gzip.open(index, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["embedding_row", "document_key", "chunk_index"])
                writer.writeheader()
                writer.writerow({"embedding_row": 0, "document_key": "image", "chunk_index": 0})
            store = FrozenTextEmbeddingStore(embeddings, index)
            result = store.collate(
                [[Chunk("image", "current_radiology", 1, 4.0, 0, 20)]],
                hospital_los_hours=[240.0],
            )
            self.assertEqual(result.window_ids.tolist(), [[7]])

    def test_section_metadata_builds_ordered_document_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "vectors.npy"
            index = root / "index.csv.gz"
            np.save(
                embeddings,
                np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float16),
            )
            with gzip.open(index, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["embedding_row", "document_key", "chunk_index"],
                )
                writer.writeheader()
                writer.writerow({"embedding_row": 0, "document_key": "a0", "chunk_index": 0})
                writer.writerow({"embedding_row": 1, "document_key": "a1", "chunk_index": 0})
                writer.writerow({"embedding_row": 2, "document_key": "b0", "chunk_index": 0})
            store = FrozenTextEmbeddingStore(embeddings, index)

            result = store.collate(
                [[
                    Chunk(
                        "a0", "current_discharge", 1, 0.0, 0, 10,
                        "doc-a", "discharge_summary", "chief_complaint", 0, 1,
                    ),
                    Chunk(
                        "a1", "current_discharge", 1, 0.0, 0, 20,
                        "doc-a", "discharge_summary", "hospital_course", 1, 1,
                    ),
                    Chunk(
                        "b0", "current_radiology", 1, 4.0, 0, 30,
                        "doc-b", "ct", "impression", 0, 1,
                    ),
                ]],
                hospital_los_hours=[240.0],
            )

            self.assertEqual(result.document_ids.tolist(), [[1, 1, 2]])
            self.assertEqual(result.section_position_ids.tolist(), [[1, 2, 1]])
            self.assertEqual(result.document_attention_mask.tolist(), [[True, True]])
            self.assertEqual(result.document_role_ids.tolist(), [[1, 2]])
            self.assertEqual(result.document_window_ids.tolist(), [[0, 7]])
            self.assertNotEqual(result.section_role_ids[0, 0], result.section_role_ids[0, 1])
            self.assertNotEqual(result.modality_ids[0, 0], result.modality_ids[0, 2])

    def test_missing_key_is_an_explicit_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "vectors.npy"
            index = root / "index.csv.gz"
            np.save(embeddings, np.asarray([[1, 2]], dtype=np.float16))
            with gzip.open(index, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["embedding_row", "document_key", "chunk_index"])
                writer.writeheader()
                writer.writerow({"embedding_row": 0, "document_key": "a", "chunk_index": 0})
            store = FrozenTextEmbeddingStore(embeddings, index)
            with self.assertRaises(KeyError):
                store.collate([[Chunk("missing", "current_discharge", 1, 0.0, 0, 20)]])

    def test_incomplete_state_is_rejected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "vectors.npy"
            index = root / "index.csv.gz"
            state = root / "state.json"
            np.save(embeddings, np.asarray([[1, 2], [3, 4]], dtype=np.float16))
            with gzip.open(index, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["embedding_row", "document_key", "chunk_index"])
                writer.writeheader()
                writer.writerow({"embedding_row": 0, "document_key": "a", "chunk_index": 0})
                writer.writerow({"embedding_row": 1, "document_key": "b", "chunk_index": 0})
            state.write_text(json.dumps({"total_rows": 2, "completed_rows": 1, "status": "in_progress"}))
            with self.assertRaises(RuntimeError):
                FrozenTextEmbeddingStore(embeddings, index, state_path=state)

            store = FrozenTextEmbeddingStore(embeddings, index, state_path=state, require_complete=False)
            with self.assertRaises(RuntimeError):
                store.collate([[Chunk("b", "prior_radiology", 1, 2.0, 0, 10)]])


if __name__ == "__main__":
    unittest.main()
