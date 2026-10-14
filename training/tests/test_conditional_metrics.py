from __future__ import annotations

import unittest

import numpy as np

from aecopd_training.conditional_metrics import conditional_metrics


class ConditionalMetricsTests(unittest.TestCase):
    def test_metrics_use_each_conditional_risk_set_and_open_tail(self) -> None:
        interval_probabilities = np.array(
            [
                [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1]],
                [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]],
            ]
        )
        interval_labels = np.array([[1, 0, 0], [0, 2, 0]])
        interval_mask = np.array([[True, False, False], [True, True, False]])
        tail_probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
        tail_labels = np.array([0, 2])
        tail_mask = np.array([True, False])

        metrics, calibration = conditional_metrics(
            interval_probabilities,
            interval_labels,
            interval_mask,
            tail_probabilities,
            tail_labels,
            tail_mask,
            prefix="validation",
        )

        self.assertEqual(metrics["validation_count_days_0_30"], 2.0)
        self.assertEqual(metrics["validation_count_days_31_90"], 1.0)
        self.assertEqual(metrics["validation_count_days_91_365"], 0.0)
        self.assertEqual(metrics["validation_tail_count"], 1.0)
        self.assertAlmostEqual(metrics["validation_aecopd_brier_days_0_30"], 0.025)
        self.assertIn("validation_aecopd_days_0_30", calibration)
        self.assertIn("validation_tail_none_observed", calibration)


if __name__ == "__main__":
    unittest.main()
