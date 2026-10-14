"""规则全量导出合同的小样本集成测试。"""

from __future__ import annotations

import csv
import gzip
import sqlite3
import tempfile
import unittest
from pathlib import Path

from training.extract_mimiciv_discharge_rules_v2 import extract


class RuleExportTests(unittest.TestCase):
    def test_all_notes_and_multiple_mentions_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "notes.csv.gz"
            with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "subject_id", "hadm_id", "timestamp_relation", "document_text",
                ))
                writer.writeheader()
                writer.writerow({
                    "subject_id": 1, "hadm_id": 10, "timestamp_relation": "by_discharge",
                    "document_text": ("HISTORY OF PRESENT ILLNESS:\nCough on admission.\n"
                                      "DISCHARGE CONDITION:\nCough resolved.\n"),
                })
                writer.writerow({
                    "subject_id": 2, "hadm_id": 20, "timestamp_relation": "after_discharge",
                    "document_text": "DISCHARGE CONDITION:\nAwake and comfortable.\n",
                })
            output = root / "rules"
            summary = extract(source, output)
            self.assertEqual(summary["documents"], 2)
            self.assertTrue(summary["unmatched_means_unknown_not_negative"])
            connection = sqlite3.connect(output / "rule_evidence.sqlite3")
            try:
                rows = connection.execute("""
                    SELECT temporality,eligible FROM mentions
                    WHERE hadm_id=10 AND field='residual_cough_sputum'
                    ORDER BY mention_id
                """).fetchall()
                self.assertEqual(rows, [("admission", 0), ("discharge_state", 1)])
                self.assertEqual(connection.execute(
                    "SELECT candidate_mentions FROM documents WHERE hadm_id=20"
                ).fetchone()[0], 0)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
