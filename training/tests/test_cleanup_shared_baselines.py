"""目录整理的算法回归：只用合成数据，不训练患者模型。"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np

from training.baselines.continuous_competing_cox import (
    fit_competing_cox,
    integrated_competing_brier,
    predict_cumulative_incidence,
)
from training.baselines.rank_metrics import auc_rank


ROOT = Path(__file__).resolve().parents[2]


class SharedBaselineCleanupTests(unittest.TestCase):
    def test_rank_auc_perfect_reversed_and_tied(self) -> None:
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.arange(20, dtype=float)
        self.assertEqual(auc_rank(scores, labels), 1.0)
        self.assertEqual(auc_rank(-scores, labels), 0.0)
        self.assertEqual(auc_rank(np.zeros(20), labels), 0.5)

    def test_rank_auc_preserves_small_sample_threshold(self) -> None:
        self.assertTrue(np.isnan(auc_rank(np.arange(19), np.array([0] * 10 + [1] * 9))))

    def test_cox_curves_are_coherent(self) -> None:
        x = np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
        times = np.array([10.0, 20.0, 30.0, 50.0, 365.0, 365.0])
        events = np.array([1, 2, 1, 2, 0, 0])
        model = fit_competing_cox(x, times, events, l2=2.0)
        curves = predict_cumulative_incidence(model, x, [0.0, 15.0, 35.0, 365.0])
        np.testing.assert_allclose(curves[:, 0], [[1.0, 0.0, 0.0]] * len(x))
        np.testing.assert_allclose(curves.sum(axis=2), 1.0, atol=1e-10)
        self.assertTrue(np.all(np.diff(curves[:, :, 1:], axis=1) >= -1e-12))

    def test_brier_retains_competing_event_semantics(self) -> None:
        times = np.array([10.0, 20.0, 365.0])
        events = np.array([1, 2, 0])
        grid = np.array([0.0, 15.0, 25.0])
        cif = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertEqual(integrated_competing_brier(times, events, cif, grid), 0.0)
        with self.assertRaises(ValueError):
            integrated_competing_brier([10.0, 20.0, 5.0], events, cif, grid)

    def test_current_v5_no_longer_imports_legacy_risk_experiments(self) -> None:
        for path in (ROOT / "training/v5").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(node.module, "training.run_coded_cohort_v1", str(path))
                    self.assertFalse((node.module or "").startswith("training.v4_survival"), str(path))


if __name__ == "__main__":
    unittest.main()
