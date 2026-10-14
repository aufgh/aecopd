from __future__ import annotations

import csv
import gzip
import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from training.prepare_rule_text_encoding import prepare


class RuleTextManifestTests(unittest.TestCase):
    def test_source_match_time_gate_and_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv.gz"
            note = "DISCHARGE CONDITION: Breathing comfortably.\nHISTORY OF PRESENT ILLNESS: cough."
            with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("subject_id", "hadm_id", "timestamp_relation", "document_text"))
                writer.writerow((1, 2, "by_discharge", note))
            database = root / "rules.sqlite3"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE documents (hadm_id INT,subject_id INT,source_sha256 TEXT)")
            connection.execute("""CREATE TABLE mentions (
                mention_id INT,hadm_id INT,field TEXT,start_line INT,end_line INT,
                evidence_text TEXT,temporality TEXT,assertion TEXT,eligible INT)""")
            connection.execute("INSERT INTO documents VALUES (2,1,?)", (hashlib.sha256(note.encode()).hexdigest(),))
            connection.executemany("INSERT INTO mentions VALUES (?,?,?,?,?,?,?,?,?)", (
                (1, 2, "residual_dyspnea", 1, 1, "Breathing comfortably.", "discharge_state", "explicit_negative_or_resolved", 1),
                (2, 2, "residual_dyspnea", 1, 1, "Breathing comfortably.", "discharge_state", "explicit_negative_or_resolved", 1),
                (3, 2, "residual_cough_sputum", 2, 2, "cough.", "admission", "unclassified", 1),
            ))
            connection.commit()
            connection.close()
            report = prepare(source, database, root / "out")
            self.assertEqual(report["shared_text_rows"], 1)
            self.assertEqual(report["excluded"], {"duplicate_same_field": 1, "non_discharge_time": 1})
            with gzip.open(root / "out" / "shared_text.csv.gz", "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["chunk_text"], "Breathing comfortably.")
            self.assertEqual(rows[0]["embedding_row"], "0")

    def test_changed_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv.gz"
            with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("subject_id", "hadm_id", "timestamp_relation", "document_text"))
                writer.writerow((1, 2, "by_discharge", "changed"))
            database = root / "rules.sqlite3"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE documents (hadm_id INT,subject_id INT,source_sha256 TEXT)")
            connection.execute("""CREATE TABLE mentions (
                mention_id INT,hadm_id INT,field TEXT,start_line INT,end_line INT,
                evidence_text TEXT,temporality TEXT,assertion TEXT,eligible INT)""")
            connection.execute("INSERT INTO documents VALUES (2,1,?)", ("0" * 64,))
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare(source, database, root / "out")


if __name__ == "__main__":
    unittest.main()
