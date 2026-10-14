"""原文审阅包的本地生成与无标签抽样合同测试。"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from training.prepare_mimiciv_text_review import sample_sql, write_review_pack


class MimicIVTextReviewTests(unittest.TestCase):
    def test_sql_samples_without_future_outcome(self) -> None:
        sql = sample_sql()
        self.assertIn("mimiciv_note.discharge", sql)
        self.assertIn("mimiciv_note.radiology", sql)
        self.assertNotIn("event_type", sql)

    def test_pack_preserves_full_original_text_without_index_leak(self) -> None:
        text = "Admission: dyspnea\nDischarge: improved & stable\n"
        rows = [{
            "note_id": "note-1", "document_type": "discharge",
            "cohort_group": "strict_index", "length_band": "1",
            "text_chars": str(len(text)), "timestamp_relation": "after_discharge",
            "document_text": text,
        }]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review"
            result = write_review_pack(rows, output)
            self.assertEqual(result["documents"], 1)
            self.assertEqual((output / "R001.txt").read_bytes(), text.encode("utf-8"))
            self.assertNotIn(text, (output / "INDEX.html").read_text(encoding="utf-8"))
            with (output / "manifest.csv").open(encoding="utf-8", newline="") as handle:
                item = next(csv.DictReader(handle))
            self.assertEqual(item["filename"], "R001.txt")
            self.assertNotIn("document_text", item)


if __name__ == "__main__":
    unittest.main()
