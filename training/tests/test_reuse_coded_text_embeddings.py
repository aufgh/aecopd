"""精确文本复用及缺失行拼装的行号校验。"""

from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.reuse_coded_text_embeddings import assemble, prepare


FIELDS = ("document_key", "chunk_index", "hadm_id", "document_role", "chunk_text")


def write_chunks(path: Path, texts: list[str]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, text in enumerate(texts):
            writer.writerow({
                "document_key": f"d{index}", "chunk_index": 0,
                "hadm_id": index + 1, "document_role": "current_radiology",
                "chunk_text": text,
            })


class ReuseCodedTextEmbeddingsTests(unittest.TestCase):
    def test_reuses_exact_text_and_assembles_in_new_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_chunks, new_chunks = root / "old.csv.gz", root / "new.csv.gz"
            write_chunks(old_chunks, ["same", "old-only"])
            write_chunks(new_chunks, ["new-only", "same", "same"])
            old_vectors = root / "old.npy"
            new_vectors = root / "new.npy"
            np.save(old_vectors, np.array([[1, 2], [3, 4]], dtype=np.float16))
            report = prepare(new_chunks, old_chunks, old_vectors, root / "reuse")
            self.assertEqual(report["exact_reuse_rows"], 2)
            self.assertEqual(report["rows_to_encode"], 1)
            np.save(new_vectors, np.array([[5, 6]], dtype=np.float16))
            assembled = root / "assembled.npy"
            assemble(root / "reuse/mapping.int32.npy", old_vectors, new_vectors, assembled)
            np.testing.assert_array_equal(
                np.load(assembled), np.array([[5, 6], [1, 2], [1, 2]], dtype=np.float16)
            )


if __name__ == "__main__":
    unittest.main()
