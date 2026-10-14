"""纯历史基线只读取既往信息，不混入本次严格编码或未来住院。"""

import unittest

import numpy as np

from training.v5.analyze_incremental_evidence import history_features


class HistoryFeaturesTests(unittest.TestCase):
    def test_unsorted_rows_and_current_code_do_not_change_past_gap(self):
        earlier = {"subject_id": "1", "hadm_id": "11", "admittime": "2100-01-01 00:00:00",
                   "dischtime": "2100-01-03 00:00:00", "prior_coded_admission_count": "0",
                   "prior_strict_aecopd_count": "0", "age_at_admission": "70", "gender": "F",
                   "index_is_strict_aecopd": "f"}
        later = {"subject_id": "1", "hadm_id": "12", "admittime": "2100-01-13 00:00:00",
                 "dischtime": "2100-01-16 00:00:00", "prior_coded_admission_count": "1",
                 "prior_strict_aecopd_count": "0", "age_at_admission": "70", "gender": "F",
                 "index_is_strict_aecopd": "t"}
        x, names = history_features([later, earlier])
        self.assertEqual(x.shape, (2, 6))
        self.assertNotIn("current_strict_code", names)
        self.assertAlmostEqual(x[0, 4], np.log1p(10.0))
        self.assertEqual(x[1, 4], 0.0)
        alternate = dict(later, index_is_strict_aecopd="f")
        other, _ = history_features([alternate, earlier])
        np.testing.assert_array_equal(x, other)


if __name__ == "__main__":
    unittest.main()
