"""固定阈值读出的边界，不涉及患者原文或模型训练。"""
import unittest
import numpy as np
from training.v6.fixed_threshold_readout import threshold_stats, point, validate, aligned


class ThresholdReadoutTest(unittest.TestCase):
    def test_equal_threshold_and_counts(self):
        r = threshold_stats(np.array([1, 0, 1, 0], bool), np.array([.2, .2, .1, .01]), .2, np.array([0, 1, 0, 0], bool))
        self.assertEqual((r['tp'], r['fp'], r['fn'], r['tn']), (1, 1, 1, 1))
        self.assertEqual(r['ppv'], .5)
        self.assertEqual(r['alerts_per_100'], 50.)
        self.assertEqual(r['competing_event_false_alerts'], 1)
        self.assertAlmostEqual(r['net_benefit'], .1875)

    def test_empty_alerts(self):
        r = threshold_stats(np.array([1, 0], bool), np.zeros(2), .1, np.zeros(2, bool))
        self.assertIsNone(r['ppv']); self.assertEqual(r['sensitivity'], 0)

    def test_no_cases(self):
        r = threshold_stats(np.zeros(2, bool), np.ones(2), .1, np.zeros(2, bool))
        self.assertIsNone(r['sensitivity']); self.assertIsNone(r['balanced_accuracy'])
        self.assertIsNone(r['npv'])

    def test_fixed_bins_keep_equal_scores(self):
        r = point(np.array([1, 0], bool), np.array([.1, .1]), np.zeros(2, bool))
        self.assertEqual(len(r['fixed_bins']), 1)
        self.assertEqual(r['fixed_bins'][0]['lower'], .1)
        self.assertAlmostEqual(r['brier'], .41)

    def sample(self):
        c = np.zeros((2, 366, 3)); c[:, :, 0] = 1
        c[:, -1] = [.5, .3, .2]
        return dict(hadm_ids=np.array([1, 2]), subject_ids=np.array([3, 3]),
                    times=np.array([365., 365.]), events=np.array([0, 1]), curve=c)

    def test_last_day_valid(self):
        validate(self.sample())

    def test_invalid_early_censoring(self):
        a = self.sample(); a['times'][0] = 90
        with self.assertRaises(ValueError): validate(a)

    def test_invalid_probability(self):
        a = self.sample(); a['curve'][0, -1, 0] = .6
        with self.assertRaises(ValueError): validate(a)

    def test_nonmonotone_curve(self):
        a = self.sample(); a['curve'][0, 2] = [.9, .05, .05]
        with self.assertRaises(ValueError): validate(a)

    def test_alignment_failure(self):
        a, b = self.sample(), self.sample(); b['hadm_ids'] = b['hadm_ids'][::-1]
        with self.assertRaises(AssertionError): aligned(a, b)

    def test_conflicting_events_and_nan(self):
        for scores, other in ((np.array([.2]), np.array([True])), (np.array([np.nan]), np.array([False]))):
            with self.assertRaises(ValueError): threshold_stats(np.array([True]), scores, .1, other)


if __name__ == '__main__': unittest.main()
