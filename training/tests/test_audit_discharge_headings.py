"""小标题统计只识别独立成行的章节标签。"""

from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

from training.audit_discharge_headings import HEADING, audit, normalize_title


class DischargeHeadingAuditTests(unittest.TestCase):
    def test_heading_definition_excludes_inline_and_numbered_problems(self) -> None:
        self.assertEqual(HEADING.fullmatch("Discharge Medications:").group(1),
                         "Discharge Medications")
        self.assertIsNone(HEADING.fullmatch("Discharge Medications: aspirin"))
        self.assertIsNone(HEADING.fullmatch("1. COPD:"))
        self.assertEqual(normalize_title("  DISCHARGE   Condition  "),
                         "discharge condition")

    def test_document_frequency_not_repeated_heading_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "notes.csv.gz"
            with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["document_text"])
                writer.writeheader()
                writer.writerow({"document_text": "Allergies:\nNone\nAllergies:\nNone\n"})
                writer.writerow({"document_text": "ALLERGIES:\nUnknown\n"})
            result = audit(source)
        self.assertEqual(result["documents"], 2)
        self.assertEqual(result["distinct_normalized_titles"], 1)
        self.assertEqual(result["titles_in_at_least_five_documents"], 0)


if __name__ == "__main__":
    unittest.main()
