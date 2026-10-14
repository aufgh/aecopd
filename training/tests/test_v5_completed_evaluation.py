"""补充评价只计算固定竞争风险曲线，不改变训练或患者成员。"""

from __future__ import annotations

import unittest

import numpy as np

from training.v5.evaluate_completed_event_run import (
    binary_roc, decision_curve, evaluate_split, patient_bootstrap_auc,
)


class CompletedEvaluationTests(unittest.TestCase):
    def test_roc_handles_ties_and_death_as_control(self) -> None:
        labels = np.array([True, True, False, False])
        fpr, tpr, auc = binary_roc(labels, np.array([0.9, 0.8, 0.8, 0.1]))
        self.assertAlmostEqual(auc, 0.875)
        self.assertEqual((fpr[0], tpr[0]), (0.0, 0.0))
        self.assertEqual((fpr[-1], tpr[-1]), (1.0, 1.0))

    def test_dca_matches_hand_calculation(self) -> None:
        result = decision_curve(np.array([True, False, False, False]),
                                np.array([0.8, 0.6, 0.2, 0.1]),
                                np.array([0.5]))
        self.assertAlmostEqual(result["model"][0], 0.0)
        self.assertAlmostEqual(result["treat_all"][0], -0.5)
        self.assertEqual(result["treat_none"][0], 0.0)

    def test_competing_death_not_a_readmission_case(self) -> None:
        n = 60
        times = np.full(n, 365.0)
        events = np.zeros(n, dtype=int)
        events[:30] = 1
        times[:30] = 20
        events[30:] = 2
        times[30:] = 10
        curves = np.zeros((n, 366, 3), dtype=float)
        curves[:, :, 0] = 0.6
        curves[:, :, 1] = 0.2
        curves[:, :, 2] = 0.2
        scores, detail = evaluate_split(np.arange(n).astype(str), times, events, curves, draws=20)
        self.assertEqual(scores["readmission"]["30"]["events"], 30)
        self.assertEqual(scores["death"]["30"]["events"], 30)
        self.assertEqual(len(detail["dca"][30]["threshold"]), 50)

    def test_early_censor_requires_new_contract(self) -> None:
        n = 60
        curves = np.empty((n, 366, 3), dtype=float)
        curves[:] = (0.6, 0.2, 0.2)
        times = np.full(n, 365.0)
        times[0] = 5
        with self.assertRaisesRegex(ValueError, "early censoring"):
            evaluate_split(np.arange(n).astype(str), times, np.zeros(n, dtype=int), curves)

    def test_patient_bootstrap_keeps_repeated_stays_together(self) -> None:
        subjects = np.repeat(np.arange(40), 2)
        labels = np.arange(80) % 2 == 0
        risk = np.where(labels, 0.8, 0.2)
        lower, upper = patient_bootstrap_auc(subjects, labels, risk, draws=20)
        self.assertEqual((lower, upper), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
