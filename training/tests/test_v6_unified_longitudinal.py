"""统一异常与全年F1的边界；真实历史/GPU另用新入口预检。"""
import unittest
import numpy as np
from patient_data.unified_lab_state import abnormal
from patient_data.v6_lab_state import summarize
from training.v6.unified_longitudinal import best_f1, f1_at


class UnifiedLongitudinalTests(unittest.TestCase):
    def test_low_high_direction(self):
        self.assertEqual(abnormal(8,10,20,'low'),.2)
        self.assertEqual(abnormal(8,10,20,'high'),0)
        self.assertEqual(abnormal(22,10,20,'high'),.2)

    def test_missing_range_invalid(self):
        for lo,hi in ((10,10),(np.nan,20)):
            with self.assertRaises(ValueError):abnormal(15,lo,hi,'high')

    def test_normal_is_observed(self):
        x,_=summarize([(1.,0.)],3.)
        self.assertEqual(x[6],1);self.assertEqual(x[7],0);self.assertEqual(x[8],0)

    def test_recovery_after_peak(self):
        x,_=summarize([(1.,2.),(2.,.5)],3.)
        self.assertEqual(x[3],1.5);self.assertEqual(x[7],1)
        y,_=summarize([(1.,.5),(2.,2.)],3.)
        self.assertEqual(y[3],0);self.assertEqual(y[8],1)

    def test_equal_scores_grouped(self):
        y=np.array([1,0,1,0],bool);s=np.array([.3,.3,.2,.1])
        best=best_f1(y,s)
        self.assertEqual(best['threshold'],.2);self.assertEqual(best['tp'],2);self.assertEqual(best['fp'],1)
        self.assertEqual(best['f1'],.8)

    def test_no_events(self):
        self.assertEqual(best_f1(np.zeros(3,bool),np.array([.1,.3,.5]))['f1'],0)

    def test_brute_force_f1(self):
        rng=np.random.default_rng(42)
        for _ in range(20):
            y=rng.random(30)<.3;s=np.round(rng.random(30),1)
            result=best_f1(y,s)
            maximum=max(f1_at(y,s,x)['f1'] for x in np.unique(s))
            self.assertAlmostEqual(result['f1'],maximum)

    def test_nonfinite_scores(self):
        with self.assertRaises(ValueError):best_f1(np.array([True]),np.array([np.nan]))


if __name__=='__main__':unittest.main()
