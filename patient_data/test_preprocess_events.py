"""使用合成记录验证预处理合同，不读取患者正文。"""
from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime
from pathlib import Path

import numpy as np

from patient_data.export_mimiciv import NULL, patient_split, rows, sha256
from patient_data.preprocess_events import Admission, Event, accepted_lab, fit_lab_parameters, preprocess, specimen_type
from training.v5.numeric_preprocessing import robust_parameters


def write_records(root: Path, name: str, records: list[dict[str, str]], fields: tuple[str, ...] = ()) -> None:
    path = root / (name + ".csv.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]) if records else fields)
        writer.writeheader()
        writer.writerows(records)


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.admission = Admission("1", "10", datetime(2020, 1, 1), datetime(2020, 1, 4), "train", True)
        self.lab = {"subject_id": "1", "hadm_id": "10", "itemid": "50818", "labevent_id": "100",
                    "specimen_id": "200", "charttime": "2020-01-02 00:00:00", "storetime": "2020-01-02 01:00:00",
                    "valuenum": "45", "valueuom": "mm Hg", "value": "45"}

    def test_small_iqr_and_zero_fallback(self) -> None:
        self.assertAlmostEqual(robust_parameters(np.array([7.3, 7.35, 7.4, 7.45]))["scale"], .075)
        constant = robust_parameters(np.array([2.0, 2.0]))
        self.assertEqual(constant["method"], "constant")
        self.assertEqual(constant["scale"], 1.0)
        self.assertEqual(robust_parameters(np.array([1., 1., 1., 1., 2.]))["method"], "std_fallback")

    def test_training_only_fit(self) -> None:
        test_admission = Admission("2", "20", self.admission.start, self.admission.end, "test", False)
        events = [Event("10", "lab", "ph:arterial", "arterial", str(i), "2", self.admission.start,
                        self.admission.start, value, "units") for i, value in enumerate([7.3, 7.35, 7.4, 7.45])]
        parameters = fit_lab_parameters(events, {"10": self.admission, "20": test_admission})
        events.append(Event("20", "lab", "ph:arterial", "arterial", "test", "3", self.admission.start,
                            self.admission.start, 90., "units"))
        self.assertEqual(parameters, fit_lab_parameters(events, {"10": self.admission, "20": test_admission}))

    def test_specimen_types_do_not_merge(self) -> None:
        artery, _ = accepted_lab(self.lab, self.admission, "arterial")
        vein, _ = accepted_lab({**self.lab, "labevent_id": "101", "specimen_id": "201"}, self.admission, "venous")
        self.assertNotEqual(artery.concept, vein.concept)
        self.assertEqual(artery.moment, vein.moment)

    def test_unknown_and_conflicting_specimens(self) -> None:
        future = datetime(2020, 1, 5)
        self.assertEqual(specimen_type([("arterial", future, future)], self.admission.end), "unknown")
        moment = self.admission.start
        self.assertEqual(specimen_type([("arterial", moment, moment), ("venous", moment, moment)], self.admission.end), "conflict")

    def test_timestamp_unit_and_value_checks(self) -> None:
        for updates, expected in (({"storetime": "2020-01-05"}, "unavailable_at_discharge"),
                                  ({"charttime": "2019-12-31"}, "outside_admission"),
                                  ({"storetime": NULL}, "missing_or_invalid_timestamp"),
                                  ({"valueuom": "kPa"}, "unit_mismatch_or_missing"),
                                  ({"valuenum": "0"}, "nonfinite_or_nonpositive_value"),
                                  ({"valuenum": "nan"}, "nonfinite_or_nonpositive_value")):
            self.assertEqual(accepted_lab({**self.lab, **updates}, self.admission, "unknown")[1], expected)

    def test_end_to_end_files_and_history(self) -> None:
        train_subject = next(str(i) for i in range(1, 100) if patient_split(str(i)) == "train")
        test_subject = next(str(i) for i in range(1, 100) if patient_split(str(i)) == "test")
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / "raw", Path(directory) / "processed"
            cohort = [{"subject_id": subject, "hadm_id": hadm, "admittime": start, "dischtime": end,
                       "discharge_landmark_eligible": "t", "is_strict_aecopd": "t"}
                      for subject, hadm, start, end in ((train_subject, "10", "2020-01-01", "2020-01-04"),
                                                      (train_subject, "11", "2020-02-01", "2020-02-04"),
                                                      (test_subject, "20", "2020-01-01", "2020-01-04"))]
            write_records(root, "cohort/admissions", cohort)
            write_records(root, "cohort/patient_splits", [{"subject_id": s, "patient_split": patient_split(s)} for s in (train_subject, test_subject)])
            write_records(root, "raw/patients", [{"subject_id": s, "anchor_age": "60", "anchor_year": "2020", "gender": "F"} for s in (train_subject, test_subject)])
            labs = [{**self.lab, "subject_id": train_subject, "itemid": "50820", "valueuom": "units", "valuenum": str(v),
                     "labevent_id": str(i)} for i, v in enumerate([7.30, 7.35, 7.40, 7.45])]
            labs.append({**labs[0], "subject_id": test_subject, "hadm_id": "20", "labevent_id": "5", "valuenum": "90"})
            write_records(root, "raw/labevents", labs)
            write_records(root, "raw/diagnoses_icd", [{"subject_id": train_subject, "hadm_id": "11", "icd_code": "E119", "icd_version": "10"}])
            write_records(root, "raw/emar", [], ("subject_id", "hadm_id", "emar_id", "emar_seq", "medication", "event_txt", "charttime", "storetime"))
            write_records(root, "raw/emar_detail", [], ("emar_id", "emar_seq", "route"))
            write_records(root, "outcomes/readmission_death_labels", [{"subject_id": r["subject_id"], "hadm_id": r["hadm_id"]} for r in cohort])
            names = ("cohort/admissions", "cohort/patient_splits", "raw/patients", "raw/labevents", "raw/diagnoses_icd", "raw/emar", "raw/emar_detail", "outcomes/readmission_death_labels")
            manifest = {"status": "validated", "sources": {name: {"sha256": sha256(root / (name + ".csv.gz"))} for name in names}}
            (root / "metadata").mkdir()
            (root / "metadata/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            accesses: list[str] = []
            original_getitem = np.lib.npyio.NpzFile.__getitem__
            def counted_getitem(archive, key):
                accesses.append(key)
                return original_getitem(archive, key)
            with patch.object(np.lib.npyio.NpzFile, "__getitem__", counted_getitem):
                result = preprocess(root, output)
            # 逐行从压缩包取数组会反复解压整列，复杂度近似平方级。
            self.assertLessEqual(max(accesses.count(key) for key in set(accesses)), 1)
            self.assertEqual(result["admissions"], 3)
            self.assertEqual(result["admissions_without_events"], 1)
            parameters = json.loads((output / "normalization.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(parameters["lab"]["ph:unknown"]["scale"], .075)
            features = {(r["hadm_id"], r["feature"]): float(r["value_raw"]) for r in rows(output / "static_features.csv.gz")}
            self.assertEqual(features[("10", "prior_code:diabetes")], 0)
            self.assertEqual(features[("11", "current_code:diabetes")], 1)
            with np.load(output / "event_arrays.npz", allow_pickle=False) as arrays:
                self.assertNotIn("event_type", arrays.files)
                self.assertNotIn("time_days", arrays.files)
            with self.assertRaises(FileExistsError):
                preprocess(root, output)


if __name__ == "__main__":
    unittest.main()
