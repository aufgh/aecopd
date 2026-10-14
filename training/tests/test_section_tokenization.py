from __future__ import annotations

import csv
import gzip
import json
import re
import tempfile
import unittest
from pathlib import Path

from aecopd_training.section_tokenization import tokenize_section_documents


class FakeFastTokenizer:
    is_fast = True

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        truncation: bool,
    ) -> dict[str, list[object]]:
        del add_special_tokens, return_offsets_mapping, truncation
        matches = list(re.finditer(r"\S+", text))
        return {
            "input_ids": list(range(len(matches))),
            "offset_mapping": [(match.start(), match.end()) for match in matches],
        }


class SectionTokenizationTests(unittest.TestCase):
    def test_tokenizes_long_section_and_propagates_section_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sections.csv.gz"
            chunks = root / "chunks.csv.gz"
            document_metadata = root / "documents.csv"
            chunk_metadata = root / "chunks.csv"
            summary = root / "summary.json"
            fields = [
                "document_key",
                "source_document_key",
                "subject_id",
                "hadm_id",
                "source_hadm_id",
                "data_split",
                "document_role",
                "document_rank",
                "days_before_landmark",
                "note_type",
                "charttime",
                "storetime",
                "modality",
                "section_role",
                "section_order",
                "section_instance",
                "document_text",
            ]
            with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "document_key": "d1::section::000",
                        "source_document_key": "d1",
                        "subject_id": "11",
                        "hadm_id": "101",
                        "source_hadm_id": "101",
                        "data_split": "train",
                        "document_role": "current_discharge",
                        "document_rank": "1",
                        "days_before_landmark": "0",
                        "note_type": "Discharge summary",
                        "charttime": "2024-01-01 12:00:00",
                        "storetime": "2024-01-02 12:00:00",
                        "modality": "discharge_summary",
                        "section_role": "hospital_course",
                        "section_order": "0",
                        "section_instance": "1",
                        "document_text": "one two three four five",
                    }
                )

            result = tokenize_section_documents(
                source_path=source,
                chunks_path=chunks,
                document_metadata_path=document_metadata,
                chunk_metadata_path=chunk_metadata,
                summary_path=summary,
                tokenizer=FakeFastTokenizer(),
                model_name="fake",
                whole_document_limit=4,
                long_document_chunk_tokens=3,
                overlap_tokens=1,
            )

            self.assertEqual(result["section_documents"], 1)
            self.assertEqual(result["chunks"], 2)
            with gzip.open(chunks, "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["chunk_text"] for row in rows], ["one two three", "three four five"])
            self.assertEqual(rows[0]["source_document_key"], "d1")
            self.assertEqual(rows[0]["section_role"], "hospital_course")
            self.assertEqual(rows[1]["chunk_index"], "1")
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8")), result)

    def test_rejects_invalid_chunk_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                tokenize_section_documents(
                    source_path=root / "missing.csv.gz",
                    chunks_path=root / "chunks.csv.gz",
                    document_metadata_path=root / "documents.csv",
                    chunk_metadata_path=root / "chunks.csv",
                    summary_path=root / "summary.json",
                    tokenizer=FakeFastTokenizer(),
                    model_name="fake",
                    whole_document_limit=4,
                    long_document_chunk_tokens=3,
                    overlap_tokens=3,
                )


if __name__ == "__main__":
    unittest.main()
