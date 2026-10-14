"""V5 冻结文本向量的对齐和断点状态校验。"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.v5.validate_vectors import validate


class V5VectorValidationTests(unittest.TestCase):
    def test_accepts_same_manifest_and_reused_duplicate_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "shared_text.csv.gz"
            with gzip.open(manifest, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("embedding_row", "chunk_text", "text_sha256"))
                writer.writeheader()
                for index in range(2):
                    writer.writerow({"embedding_row": index, "chunk_text": "same text",
                                     "text_sha256": hashlib.sha256(b"same text").hexdigest()})
            vectors = root / "vectors.npy"
            value = np.zeros((2, 2560), dtype=np.float16)
            value[:, 0] = 1.0
            np.save(vectors, value)
            state = root / "state.json"
            state.write_text(json.dumps({"status": "complete", "completed_rows": 2,
                                         "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}),
                             encoding="utf-8")
            self.assertEqual(validate(manifest, vectors, state)["duplicate_rows"], 1)
            state.write_text(json.dumps({"status": "complete", "completed_rows": 2,
                                         "manifest_sha256": "wrong"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different manifest"):
                validate(manifest, vectors, state)


if __name__ == "__main__":
    unittest.main()
