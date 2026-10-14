from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from aecopd_training.section_dataset import build_section_documents


class SectionDatasetTests(unittest.TestCase):
    def test_builds_compatible_section_documents_and_aggregate_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "documents.csv.gz"
            output = root / "v2" / "qwen_section_documents.csv.gz"
            summary = root / "v2" / "section_build_summary.json"
            fields = [
                "document_key",
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
                "document_text",
            ]
            with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "document_key": "d1",
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
                        "document_text": "Chief Complaint: Dyspnea\nHospital Course: Better.",
                    }
                )
                writer.writerow(
                    {
                        "document_key": "d2",
                        "subject_id": "11",
                        "hadm_id": "101",
                        "source_hadm_id": "99",
                        "data_split": "train",
                        "document_role": "prior_radiology",
                        "document_rank": "2",
                        "days_before_landmark": "20",
                        "note_type": "Radiology",
                        "charttime": "2023-12-01 10:00:00",
                        "storetime": "",
                        "document_text": "EXAM: CT CHEST\nIMPRESSION: Stable.",
                    }
                )

            result = build_section_documents(source, output, summary)

            self.assertEqual(result["source_documents"], 2)
            self.assertEqual(result["sections"], 4)
            self.assertNotIn("subject_id", result)
            with gzip.open(output, "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["document_key"], "d1::section::000")
            self.assertEqual(rows[0]["source_document_key"], "d1")
            self.assertEqual(rows[0]["section_role"], "chief_complaint")
            self.assertEqual(rows[2]["modality"], "ct")
            self.assertIn("Document role: prior_radiology", rows[2]["document_text"])
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8")), result)

    def test_refuses_to_overwrite_existing_outputs_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "documents.csv.gz"
            output = root / "sections.csv.gz"
            summary = root / "summary.json"
            output.write_bytes(b"already here")
            with self.assertRaises(FileExistsError):
                build_section_documents(source, output, summary)


if __name__ == "__main__":
    unittest.main()
