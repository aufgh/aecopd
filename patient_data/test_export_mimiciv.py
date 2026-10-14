"""原始层导出合同测试，不访问或显示患者正文。"""
import gzip
import tempfile
import unittest
from pathlib import Path

from export_mimiciv import export, patient_split, rows, sources


class ExportContractTests(unittest.TestCase):
    def test_raw_labs_have_no_model_filters(self) -> None:
        sql = next(s.select for s in sources() if s.name == "raw/labevents")
        self.assertIn("SELECT *", sql)
        for token in ("itemid", "valuenum", "storetime", "charttime"):
            self.assertNotIn(token, sql)
        names = {s.name for s in sources()}
        self.assertIn("raw/emar_detail", names)
        self.assertIn("outcomes/readmission_death_labels", names)
        self.assertFalse(any("chartevents" in n for n in names))

    def test_multiline_text_and_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                handle.write('id,text,value\n1,"line one\nline two",\\N\n')
            records = list(rows(path))
            self.assertEqual(records[0]["text"], "line one\nline two")
            self.assertEqual(records[0]["value"], "\\N")

    def test_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                export(Path(directory), Path("missing-psql"), "unused")

    def test_split_is_deterministic(self) -> None:
        for subject in ("1", "2", "3", "100"):
            self.assertEqual(patient_split(subject), patient_split(subject))
            self.assertIn(patient_split(subject), ("train", "validation", "test"))


if __name__ == "__main__":
    unittest.main()
