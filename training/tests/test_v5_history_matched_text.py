"""同层文本置换不跨患者、不跨历史层、保持字段片段数。"""

import unittest

import numpy as np

from training.v5.event_data import EventData
from training.v5.permute_history_matched_text import matched_field_swap, matched_lab_value_swap


class MatchedFieldSwapTests(unittest.TestCase):
    def test_matched_donor_and_unmatched_field(self):
        fields = (((0, 10), (0, 11), (1, 20)),
                  ((0, 30), (0, 31), (1, 40)),
                  ((0, 50),),
                  ((0, 60), (0, 61), (1, 70)))
        changed, count = matched_field_swap(
            fields, np.array([1, 2, 3, 4]), np.array([0, 0, 3, 0]),
            np.array([True, True, True, False]), np.arange(4), seed=7)
        self.assertEqual(changed[0], fields[1])
        self.assertEqual(changed[1], fields[0])
        self.assertEqual(changed[2], fields[2])
        self.assertEqual(changed[3], fields[3])
        self.assertEqual(count["fields_replaced"], 4)
        self.assertEqual(count["fields_unmatched"], 3)

    def test_same_patient_is_not_donor(self):
        fields = (((0, 10),), ((0, 20),))
        changed, count = matched_field_swap(fields, np.array([1, 1]), np.array([0, 0]),
                                            np.array([False, False]), np.arange(2), seed=7)
        self.assertEqual(changed, fields)
        self.assertEqual(count["fields_replaced"], 0)

    def test_matched_laboratory_swaps_values_but_not_time_or_events(self):
        events = EventData(
            base=None,
            event_values=np.array([[1., .1, 0., 0.], [2., .2, 0., 0.],
                                   [7., .3, 0., 0.], [8., .4, 0., 0.],
                                   [9., .5, 0., 0.]], dtype=np.float32),
            event_concepts=np.array([1, 1, 1, 1, 1]),
            event_is_lab=np.ones(5, dtype=bool),
            events_by_admission=(np.array([0, 1]), np.array([2, 3]), np.array([4])),
            source_hashes={}, event_numeric_mask=np.ones(5, dtype=bool),
        )
        changed, count = matched_lab_value_swap(
            events, np.array([1, 2, 3]), np.array([0, 0, 3]),
            np.array([True, True, True]), np.arange(3), seed=2)
        np.testing.assert_array_equal(changed[:4, 0], [7., 8., 1., 2.])
        np.testing.assert_array_equal(changed[:, 1], events.event_values[:, 1])
        np.testing.assert_array_equal(events.event_values[:, 0], [1., 2., 7., 8., 9.])
        self.assertEqual(count["trajectories_replaced"], 2)
        self.assertEqual(count["trajectories_unmatched"], 1)


if __name__ == "__main__":
    unittest.main()
