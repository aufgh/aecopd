import unittest
import numpy as np
from training.v6.multihorizon_readout import horizon_readout, decision, segmented_auc
from training.v6.fixed_threshold_readout import aligned, validate


def fixture():
    c=np.zeros((4,366,3));c[:,:,0]=1
    for i,level in enumerate((.4,.3,.2,.1)):
        c[i,:,1]=np.arange(366)/365*level;c[i,:,0]-=c[i,:,1]
    return dict(hadm_ids=np.arange(4),subject_ids=np.arange(4),
                times=np.array([30.,180.,90.,365.]),events=np.array([1,1,2,0]),curve=c)


class MultihorizonTests(unittest.TestCase):
    def test_true_event_boundary_and_competing(self):
        a=fixture();validate(a)
        self.assertEqual(horizon_readout(a,14,1,.2)['events'],0)
        self.assertEqual(horizon_readout(a,30,1,.2)['events'],1)
        self.assertEqual(horizon_readout(a,365,1,.2)['events'],2)
        self.assertEqual(horizon_readout(a,90,2,.2)['events'],1)

    def test_optimal_no_worse_than_fixed(self):
        r=horizon_readout(fixture(),365,1,.25)
        for fixed in r['thresholds']:
            self.assertGreaterEqual(r['tuning_best_f1']['f1'],fixed['f1'])

    def test_annual_threshold_not_retuned(self):
        r=horizon_readout(fixture(),30,1,.25)
        self.assertEqual(r['annual_threshold_readout']['threshold'],.25)
        self.assertEqual(r['annual_threshold_readout']['alerts'],0)
        self.assertGreater(r['tuning_best_f1']['alerts'],0)

    def test_equal_threshold_is_alert(self):
        d=decision(np.array([True,False]),np.array([.2,.1]),.2)
        self.assertEqual(d['tp'],1);self.assertEqual(d['accuracy'],1)

    def test_auc_segments_do_not_fill_missing(self):
        x=np.full(366,np.nan);x[7]=.8;x[30]=.6;x[31]=.7
        r=segmented_auc(x)
        self.assertAlmostEqual(r['7_30']['mean_auc'],.7)
        self.assertEqual(r['7_30']['valid_days'],2)
        self.assertIsNone(r['91_365']['mean_auc'])

    def test_alignment_mismatch_rejected(self):
        a=fixture();b=fixture();b['times'][0]=31
        with self.assertRaises(AssertionError):aligned(a,b)

    def test_future_or_missing_horizon_rejected(self):
        with self.assertRaises(ValueError):horizon_readout(fixture(),400,1,.2)
        with self.assertRaises(ValueError):horizon_readout(fixture(),30,3,.2)

    def test_invalid_curve_rejected(self):
        a=fixture();a['curve'][0,30,1]=.9
        with self.assertRaises(ValueError):validate(a)


if __name__ == '__main__':unittest.main()
