"""全量出院小结导出与规则聚合的口径测试。"""

from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

from training.export_mimiciv_discharge_all import copy_sql
from training.extract_mimiciv_discharge_rules import extract


class MimicIVDischargeExportTests(unittest.TestCase):
    def test_sql_keeps_notes_saved_after_discharge_without_outcomes(self) -> None:
        sql = copy_sql()
        self.assertIn("mimiciv_note.discharge", sql)
        self.assertIn("n.storetime>c.dischtime", sql)
        self.assertNotIn("n.storetime<=c.dischtime", sql)
        self.assertNotIn("event_type", sql)

    def test_first_coded_index_export_is_separate_from_strict_v1(self) -> None:
        sql = copy_sql("first-coded-v2")
        self.assertIn("copd_coded_first_index_labels_v2", sql)
        self.assertNotIn("copd_coded_strict_index_labels_v1", sql)
        with self.assertRaises(ValueError):
            copy_sql("untrusted-table-name")

    def test_rule_export_never_copies_note_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "notes.csv.gz"
            with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "subject_id", "hadm_id", "timestamp_relation", "document_text",
                ))
                writer.writeheader()
                writer.writerow({
                    "subject_id": "1", "hadm_id": "2",
                    "timestamp_relation": "after_discharge",
                    "document_text": "DISCHARGE CONDITION:\nActivity Status: Ambulatory - Independent.\n",
                })
            summary = extract(source, root / "features")
            self.assertEqual(summary["admissions_with_note"], 1)
            output = (root / "features" / "features.csv").read_text(encoding="utf-8")
            self.assertIn("independent", output)
            self.assertNotIn("Activity Status", output)


if __name__ == "__main__":
    unittest.main()
