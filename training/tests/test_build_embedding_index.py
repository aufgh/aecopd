from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

from build_embedding_index import build_embedding_index


class EmbeddingIndexTests(unittest.TestCase):
    def test_section_metadata_is_retained_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.csv"
            output = root / "index.csv.gz"
            fields = [
                "document_key",
                "chunk_index",
                "hadm_id",
                "data_split",
                "document_role",
                "document_rank",
                "days_before_landmark",
                "token_count",
                "source_document_key",
                "modality",
                "section_role",
                "section_order",
                "section_instance",
            ]
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "document_key": "d1::section::000",
                        "chunk_index": 0,
                        "hadm_id": 101,
                        "data_split": "train",
                        "document_role": "current_discharge",
                        "document_rank": 1,
                        "days_before_landmark": 0,
                        "token_count": 25,
                        "source_document_key": "d1",
                        "modality": "discharge_summary",
                        "section_role": "hospital_course",
                        "section_order": 0,
                        "section_instance": 1,
                    }
                )

            result = build_embedding_index(metadata, output)

            self.assertEqual(result["rows"], 1)
            with gzip.open(output, "rt", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["embedding_row"], "0")
            self.assertEqual(row["source_document_key"], "d1")
            self.assertEqual(row["section_role"], "hospital_course")
            self.assertEqual(row["section_instance"], "1")


if __name__ == "__main__":
    unittest.main()
