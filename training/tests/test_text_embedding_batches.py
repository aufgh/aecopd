from __future__ import annotations

import unittest
import csv
import gzip
import json
import tempfile
from pathlib import Path

from generate_text_embeddings import TextRow, count_rows, load_state, pack_batches, plan_reuse


class TextEmbeddingBatchTests(unittest.TestCase):
    def test_every_row_appears_once_and_budget_is_respected(self) -> None:
        rows = [TextRow(index, tokens, "x") for index, tokens in enumerate([10, 100, 200, 500, 1000, 9000])]
        batches = pack_batches(rows, token_budget=1024, max_batch_size=8)
        flattened = [row.row_number for batch in batches for row in batch]
        self.assertEqual(sorted(flattened), list(range(len(rows))))
        for batch in batches:
            padded = max(row.recorded_tokens + 1 for row in batch) * len(batch)
            if len(batch) > 1:
                self.assertLessEqual(padded, 1024)
        self.assertEqual([row.recorded_tokens for row in batches[-1]], [9000])

    def test_invalid_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pack_batches([TextRow(0, 10, "x")], token_budget=0, max_batch_size=1)

    def test_csv_record_count_does_not_treat_newlines_inside_text_as_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["token_count", "chunk_text"])
                writer.writeheader()
                writer.writerow({"token_count": 3, "chunk_text": "line one\nline two"})
                writer.writerow({"token_count": 2, "chunk_text": "plain"})
            self.assertEqual(count_rows(path), 2)

    def test_exact_duplicate_text_is_reused_from_completed_and_current_rows(self) -> None:
        first = TextRow(10, 4, "same")
        second = TextRow(11, 4, "new")
        third = TextRow(12, 4, "new")
        digest_same = __import__("hashlib").sha256(b"same").digest()

        plan = plan_reuse([first, second, third], {digest_same: 2})

        self.assertEqual([(row.row_number, source) for row, source in plan.existing_copies], [(10, 2)])
        self.assertEqual(len(plan.new_groups), 1)
        self.assertEqual([row.row_number for row in next(iter(plan.new_groups.values()))], [11, 12])

    def test_resume_rejects_changed_manifest_with_same_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps({"total_rows": 2, "completed_rows": 1,
                                        "manifest_sha256": "original"}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                load_state(path, 2, "changed")


if __name__ == "__main__":
    unittest.main()
