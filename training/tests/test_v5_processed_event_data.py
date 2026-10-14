"""新事件输入的住院对齐、数值直通和显式类型测试。"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from training.v5.processed_event_data import attach_arrays


class ProcessedEventDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = SimpleNamespace(hadm_ids=np.array([20, 10]), subject_ids=np.array([2, 1]))
        self.arrays = {"hadm_ids": np.array([10, 20]), "subject_ids": np.array([1, 2]),
                       "offsets": np.array([0, 2, 3]), "concept_ids": np.array([2, 1, 2]),
                       "values": np.array([1.25, 0., -0.5], dtype=np.float32),
                       "numeric_mask": np.array([True, False, True]),
                       "elapsed_log_days": np.array([.1, .2, .3], dtype=np.float32)}

    def test_reorder_by_admission_and_do_not_renormalize(self) -> None:
        data = attach_arrays(self.base, self.arrays, ("saba", "ph:arterial"), {})
        self.assertEqual(data.events_by_admission[0].tolist(), [2])
        self.assertEqual(data.events_by_admission[1].tolist(), [0, 1])
        np.testing.assert_array_equal(data.event_values[:, 0], self.arrays["values"])
        np.testing.assert_array_equal(data.event_values[:, 2:], np.zeros((3, 2)))
        self.assertEqual(data.event_is_lab.tolist(), [True, False, True])

    def test_patient_mismatch_is_rejected(self) -> None:
        self.arrays["subject_ids"][0] = 99
        with self.assertRaisesRegex(ValueError, "identity"):
            attach_arrays(self.base, self.arrays, ("saba", "ph:arterial"), {})

    def test_cohort_mismatch_is_rejected(self) -> None:
        self.arrays["hadm_ids"][0] = 99
        with self.assertRaisesRegex(ValueError, "cohort"):
            attach_arrays(self.base, self.arrays, ("saba", "ph:arterial"), {})

    def test_medication_numeric_mask_is_rejected(self) -> None:
        self.arrays["numeric_mask"][1] = True
        with self.assertRaisesRegex(ValueError, "medication"):
            attach_arrays(self.base, self.arrays, ("saba", "ph:arterial"), {})

    def test_revised_tetracycline_is_a_medication_not_a_laboratory_value(self) -> None:
        data = attach_arrays(self.base, self.arrays, ("tetracycline", "ph:arterial"), {})
        self.assertEqual(data.event_is_lab.tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
