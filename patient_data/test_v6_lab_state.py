"""V6 摘要边界；不使用患者原文。"""
import unittest
import numpy as np
from patient_data.v6_lab_state import summarize, scale_on_train, strict_interval_exposure, AXES


class LabStateTests(unittest.TestCase):
    def test_early_recovery_is_retained(self):
        x, _ = summarize([(1, .9), (3, 0)], 8)
        self.assertAlmostEqual(x[3], .9)
        self.assertAlmostEqual(x[4], np.log1p(5))
        self.assertEqual(x[7], 1)

    def test_single_plateau_normal_and_latest_peak(self):
        single, _ = summarize([(1, 2)], 8)
        plateau, _ = summarize([(1, 2), (3, 2)], 8)
        normal, _ = summarize([(1, 0), (3, 0)], 8)
        worsening, _ = summarize([(1, 0), (3, 2)], 8)
        self.assertEqual(tuple(single[7:]), (0, 1))
        self.assertEqual(tuple(plateau[7:]), (1, 1))
        self.assertEqual(tuple(normal[7:]), (1, 0))
        self.assertEqual(tuple(worsening[7:]), (0, 1))

    def test_missing_not_normal(self):
        empty, _ = summarize([], 8)
        normal, _ = summarize([(1, 0)], 8)
        self.assertEqual(empty[6], 0)
        self.assertEqual(normal[6], 1)

    def test_permutation_peak_and_no_cap(self):
        samples = [(1, 6), (3, 9), (5, 8), (5, 8)]
        a, m = summarize(samples, 8)
        b, _ = summarize(list(reversed(samples)), 8)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a[1], 9)
        self.assertEqual(a[3], 1)
        self.assertEqual(m["first_peak_day"], 3)

    def test_future_and_nonfinite_fail(self):
        for obs in [[(9, 1)], [(1, float('nan'))], [(1, -1)]]:
            with self.assertRaises(ValueError):
                summarize(obs, 8)

    def test_scaling_train_only_and_masks(self):
        raw = np.zeros((3, len(AXES), 9))
        for i, value in enumerate([.1, .2, 1000]):
            raw[i, 0], _ = summarize([(1, value)], 8)
        x, mask, p = scale_on_train(raw, np.array(['train', 'train', 'test']))
        raw[2, 0, :4] = 1000000
        _, _, changed = scale_on_train(raw, np.array(['train', 'train', 'test']))
        self.assertEqual(p, changed)
        self.assertTrue(mask[0, 0, 0])
        self.assertFalse(mask[0, 0, 3])
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[:, :, 6:], raw[:, :, 6:])

    def test_drug_boundaries_do_not_imply_order(self):
        self.assertFalse(strict_interval_exposure(1, 3, [1, 3]))
        self.assertTrue(strict_interval_exposure(1, 3, [1, 2, 3]))
        self.assertFalse(strict_interval_exposure(1, 3, []))


if __name__ == '__main__':
    unittest.main()
