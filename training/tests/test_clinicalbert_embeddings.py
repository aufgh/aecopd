"""临床 BERT 分块与池化不能漏掉长病历尾部或混入 padding。"""

import csv
import gzip
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from aecopd_training.clinicalbert_embeddings import (
    TokenWindow,
    masked_content_mean,
    pack_token_windows,
    split_token_windows,
)
from generate_clinicalbert_embeddings import _write_state_with_retry, validate_index_alignment


class ClinicalBertEmbeddingTests(unittest.TestCase):
    def test_long_section_is_fully_covered_without_double_weight(self) -> None:
        windows = split_token_windows(list(range(1000)), 7, max_content_tokens=510, overlap_tokens=64)
        self.assertEqual([len(window.token_ids) for window in windows], [510, 510, 108])
        self.assertEqual(sum(window.aggregation_weight for window in windows), 1000)
        self.assertEqual(windows[-1].token_ids[-1], 999)
        self.assertTrue(all(window.source_row == 7 for window in windows))

    def test_batch_budget_uses_padded_length(self) -> None:
        windows = [TokenWindow(index, (1,) * 510, 510) for index in range(9)]
        batches = pack_token_windows(windows, token_budget=4096, max_batch_size=32)
        self.assertEqual([len(batch) for batch in batches], [8, 1])

    def test_masked_mean_excludes_special_and_padding(self) -> None:
        hidden = torch.tensor([[[9.0], [2.0], [4.0], [8.0], [100.0]]])
        mask = torch.tensor([[1, 1, 1, 1, 0]])
        pooled = masked_content_mean(hidden, mask)
        self.assertAlmostEqual(float(pooled[0, 0]), 3.0)

    def test_embedding_rows_must_match_existing_text_index(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            chunks_path = Path(folder) / "chunks.csv.gz"
            index_path = Path(folder) / "index.csv.gz"
            with gzip.open(chunks_path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("document_key", "chunk_index", "hadm_id", "document_role")
                )
                writer.writeheader()
                writer.writerow({"document_key": "a", "chunk_index": 0, "hadm_id": 1, "document_role": "current_discharge"})
            with gzip.open(index_path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("embedding_row", "document_key", "chunk_index", "hadm_id", "document_role")
                )
                writer.writeheader()
                writer.writerow({"embedding_row": 0, "document_key": "a", "chunk_index": 0, "hadm_id": 1, "document_role": "current_discharge"})
            self.assertEqual(validate_index_alignment(chunks_path, index_path), 1)
            with gzip.open(index_path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("embedding_row", "document_key", "chunk_index", "hadm_id", "document_role")
                )
                writer.writeheader()
                writer.writerow({"embedding_row": 0, "document_key": "wrong", "chunk_index": 0, "hadm_id": 1, "document_role": "current_discharge"})
            with self.assertRaisesRegex(ValueError, "misaligned"):
                validate_index_alignment(chunks_path, index_path)

    def test_transient_windows_state_lock_is_retried(self) -> None:
        with patch(
            "generate_clinicalbert_embeddings.atomic_write_json",
            side_effect=[PermissionError("locked"), PermissionError("locked"), None],
        ) as write:
            with patch("generate_clinicalbert_embeddings.time.sleep"):
                _write_state_with_retry(Path("state.json"), {"status": "in_progress"})
        self.assertEqual(write.call_count, 3)


if __name__ == "__main__":
    unittest.main()
