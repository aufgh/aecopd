from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from aecopd_training.patient_sequences import build_patient_sequence_index


class PatientSequenceIndexTests(unittest.TestCase):
    def test_orders_admissions_causally_and_assigns_patient_equal_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = root / "cohort.csv.gz"
            output = root / "patient_sequences.csv.gz"
            summary = root / "patient_sequences_summary.json"
            fields = ["subject_id", "hadm_id", "data_split", "admittime", "dischtime"]
            with gzip.open(cohort, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "subject_id": 10,
                            "hadm_id": 102,
                            "data_split": "train",
                            "admittime": "2020-03-01 00:00:00",
                            "dischtime": "2020-03-05 00:00:00",
                        },
                        {
                            "subject_id": 20,
                            "hadm_id": 201,
                            "data_split": "validation",
                            "admittime": "2020-02-01 00:00:00",
                            "dischtime": "2020-02-02 00:00:00",
                        },
                        {
                            "subject_id": 10,
                            "hadm_id": 101,
                            "data_split": "train",
                            "admittime": "2020-01-01 00:00:00",
                            "dischtime": "2020-01-05 00:00:00",
                        },
                    ]
                )

            result = build_patient_sequence_index(cohort, output, summary)

            with gzip.open(output, "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            patient_10 = [row for row in rows if row["subject_id"] == "10"]
            self.assertEqual([row["hadm_id"] for row in patient_10], ["101", "102"])
            self.assertEqual([row["sequence_index"] for row in patient_10], ["0", "1"])
            self.assertEqual([float(row["patient_loss_weight"]) for row in patient_10], [0.5, 0.5])
            self.assertEqual(float(patient_10[1]["days_since_previous_discharge"]), 56.0)
            self.assertEqual(result["patients"], 2)
            self.assertEqual(result["index_admissions"], 3)
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8")), result)

    def test_patient_cannot_cross_data_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = root / "cohort.csv.gz"
            fields = ["subject_id", "hadm_id", "data_split", "admittime", "dischtime"]
            with gzip.open(cohort, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {"subject_id": 10, "hadm_id": 101, "data_split": "train", "admittime": "2020-01-01", "dischtime": "2020-01-02"},
                        {"subject_id": 10, "hadm_id": 102, "data_split": "test", "admittime": "2020-02-01", "dischtime": "2020-02-02"},
                    ]
                )
            with self.assertRaises(ValueError):
                build_patient_sequence_index(
                    cohort,
                    root / "patient_sequences.csv.gz",
                    root / "summary.json",
                )


if __name__ == "__main__":
    unittest.main()
