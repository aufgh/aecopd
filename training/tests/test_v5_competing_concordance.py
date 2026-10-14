"""原因别 C-index 的竞争事件、同日及提前删失边界。"""

import unittest

import numpy as np

from training.v5.competing_concordance import cif_concordance


class CompetingConcordanceTest(unittest.TestCase):
    def test_competing_death_is_comparator_even_when_earlier(self):
        result = cif_concordance(np.array([20, 5, 365]), np.array([1, 2, 0]),
                                 np.array([0.8, 0.2, 0.1]), horizon=90)
        self.assertEqual(result["comparable_pairs"], 2)
        self.assertEqual(result["c_index"], 1.0)

    def test_later_target_event_and_risk_tie(self):
        result = cif_concordance(np.array([10, 40, 10]), np.array([1, 1, 1]),
                                 np.array([0.5, 0.5, 0.9]), horizon=90)
        self.assertEqual(result["comparable_pairs"], 2)
        self.assertEqual(result["c_index"], 0.75)

    def test_censored_early_rejected(self):
        with self.assertRaisesRegex(ValueError, "IPCW"):
            cif_concordance(np.array([10, 20]), np.array([0, 1]),
                            np.array([0.1, 0.9]), horizon=90)


if __name__ == "__main__":
    unittest.main()
