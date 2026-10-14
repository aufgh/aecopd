"""严格结局验证脚本的低维参照和审计分层边界。"""

from __future__ import annotations

import unittest

import numpy as np

from training.v5.validate_strict_candidate import _features
from training.v5.compare_strict_ablations import _paired_auc_interval


class StrictCandidateValidationTests(unittest.TestCase):
    def test_five_features_use_training_center_and_boolean_code(self) -> None:
        rows = [
            {"age_at_admission": "60", "gender": "F", "prior_coded_admission_count": "0",
             "prior_strict_aecopd_count": "0", "index_is_strict_aecopd": "f"},
            {"age_at_admission": "80", "gender": "M", "prior_coded_admission_count": "3",
             "prior_strict_aecopd_count": "1", "index_is_strict_aecopd": "t"},
            {"age_at_admission": "100", "gender": "F", "prior_coded_admission_count": "1",
             "prior_strict_aecopd_count": "0", "index_is_strict_aecopd": "f"},
        ]
        x, names, preprocessing = _features(rows, np.array([True, True, False]))
        self.assertEqual(x.shape, (3, 5))
        self.assertEqual(len(names), 5)
        self.assertEqual(preprocessing["age_median_train"], 70.0)
        np.testing.assert_allclose(x[:, 0], [-1., 1., 3.])
        np.testing.assert_array_equal(x[:, 4], [0., 1., 0.])
        np.testing.assert_allclose(x[1, 2:4], np.log1p([3, 1]))

    def test_rejects_unexpected_code_flag(self) -> None:
        row = {"age_at_admission": "60", "gender": "F", "prior_coded_admission_count": "0",
               "prior_strict_aecopd_count": "0", "index_is_strict_aecopd": "unknown"}
        with self.assertRaisesRegex(ValueError, "unexpected"):
            _features([row], np.array([True]))

    def test_paired_auc_bootstrap_preserves_patient_clustering(self) -> None:
        n = 100
        subjects = np.repeat(np.arange(n // 2), 2)
        events = np.array([1] * 50 + [0] * 50)
        times = np.where(events == 1, 20., 365.)
        full = np.zeros((n, 366))
        other = np.zeros_like(full)
        full[:, 30] = np.where(events == 1, .8, .2)
        other[:, 30] = np.where(events == 1, .2, .8)
        comparison = _paired_auc_interval(subjects, times, events, full, other, 30, draws=20)
        self.assertAlmostEqual(comparison["point"], 1.)
        self.assertEqual(comparison["valid_draws"], 20)


if __name__ == "__main__":
    unittest.main()
