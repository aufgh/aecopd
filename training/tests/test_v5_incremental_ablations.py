"""增量证据链中的事件消融应只改变声明的输入。"""

import unittest

import numpy as np

from training.v5.event_data import EventData
from training.v5.train_event_sequence import ablate_event_inputs


class IncrementalAblationTests(unittest.TestCase):
    def setUp(self):
        self.data = EventData(
            base=None,
            event_values=np.array([[1., 0.2, 0., 0.], [3., 0.5, 0., 0.],
                                   [0., 0.4, 0., 0.], [7., 0.8, 0., 0.],
                                   [0., 0.9, 0., 0.]], dtype=np.float32),
            event_concepts=np.array([1, 1, 2, 3, 2]),
            event_is_lab=np.array([True, True, False, True, False]),
            events_by_admission=(np.array([0, 1, 2]), np.array([3, 4])),
            source_hashes={},
            concept_names=("lab_one", "drug", "lab_two"),
            event_numeric_mask=np.array([True, True, False, False, False]),
        )

    def test_remove_medication_preserves_labs_and_original(self):
        changed = ablate_event_inputs(self.data, disable_medications=True)
        np.testing.assert_array_equal(changed.events_by_admission[0], [0, 1])
        np.testing.assert_array_equal(changed.events_by_admission[1], [3])
        np.testing.assert_array_equal(self.data.events_by_admission[0], [0, 1, 2])

    def test_no_time_keeps_values_and_frequency(self):
        changed = ablate_event_inputs(self.data, lab_process="no_time")
        np.testing.assert_array_equal(changed.event_values[:, 0], self.data.event_values[:, 0])
        np.testing.assert_allclose(changed.event_values[:, 1], [0., 0., 0.4, 0., 0.9], atol=1e-6)
        np.testing.assert_array_equal(changed.events_by_admission[0], [0, 1, 2])

    def test_one_per_concept_averages_observed_values_and_keeps_drugs(self):
        changed = ablate_event_inputs(self.data, lab_process="one_per_concept")
        np.testing.assert_array_equal(changed.events_by_admission[0], [0, 2])
        np.testing.assert_array_equal(changed.events_by_admission[1], [3, 4])
        self.assertEqual(float(changed.event_values[0, 0]), 2.0)
        self.assertEqual(float(changed.event_values[0, 1]), 0.0)
        self.assertEqual(float(changed.event_values[3, 0]), 0.0)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            ablate_event_inputs(self.data, lab_process="unknown")

    def test_selected_concept_group_removed_without_renumbering(self):
        changed = ablate_event_inputs(self.data, excluded_concepts=("lab_one", "drug"))
        np.testing.assert_array_equal(changed.events_by_admission[0], [])
        np.testing.assert_array_equal(changed.events_by_admission[1], [3])
        self.assertEqual(changed.concept_names, self.data.concept_names)
        np.testing.assert_array_equal(self.data.events_by_admission[0], [0, 1, 2])

    def test_unknown_or_repeated_concept_rejected(self):
        for names in (("not_a_concept",), ("drug", "drug")):
            with self.subTest(names=names), self.assertRaises(ValueError):
                ablate_event_inputs(self.data, excluded_concepts=names)


if __name__ == "__main__":
    unittest.main()
