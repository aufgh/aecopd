from __future__ import annotations

import unittest

import numpy as np

from aecopd_training.metrics import binary_auc, calibration_bins, horizon_metrics


class CompetingRiskMetricTests(unittest.TestCase):
    def test_binary_auc_is_one_for_perfect_ranking_and_handles_ties(self) -> None:
        self.assertEqual(binary_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(binary_auc(np.array([0, 1]), np.array([0.5, 0.5])), 0.5)

    def test_horizon_metrics_report_auc_brier_and_observed_rate(self) -> None:
        metrics = horizon_metrics(
            risks=np.array([0.1, 0.2, 0.8, 0.9]),
            outcomes=np.array([0, 0, 1, 1]),
        )
        self.assertAlmostEqual(metrics["auc"], 1.0)
        self.assertAlmostEqual(metrics["brier"], 0.025)
        self.assertAlmostEqual(metrics["observed_rate"], 0.5)
        self.assertAlmostEqual(metrics["mean_predicted_risk"], 0.5)

    def test_calibration_bins_keep_every_sample_exactly_once(self) -> None:
        bins = calibration_bins(
            risks=np.array([0.05, 0.10, 0.20, 0.80, 0.90]),
            outcomes=np.array([0, 0, 1, 1, 1]),
            bin_count=3,
        )
        self.assertEqual(sum(item["count"] for item in bins), 5)
        self.assertTrue(all(0.0 <= item["mean_predicted"] <= 1.0 for item in bins))
        self.assertTrue(all(0.0 <= item["observed_rate"] <= 1.0 for item in bins))


if __name__ == "__main__":
    unittest.main()
