"""原值、采样去重、轨迹有效性与内折尺度隔离。"""
import unittest
import numpy as np
from training.v7.research_features import trajectory,encode


class ResearchFeatureTests(unittest.TestCase):
    def test_same_time_dedup_and_recovery_direction(self):
        x,m=trajectory([(0.,80.),(0.,100.),(1.,60.),(2.,50.)],3.,1)
        self.assertEqual(x[0],90.);self.assertEqual(x[6],3.)
        self.assertEqual(x[7],20.);self.assertTrue(m[8])
        low,valid=trajectory([(0.,7.2),(1.,7.3)],2.,-1)
        self.assertAlmostEqual(low[7],.1);self.assertTrue(valid[7]);self.assertFalse(valid[8])

    def test_missing_and_no_valid_recovery(self):
        x,m=trajectory([],1.,1);self.assertFalse(m.any());self.assertEqual(x.sum(),0)
        x,m=trajectory([(1.,5.)],2.,1);self.assertFalse(m[7]);self.assertFalse(m[8]);self.assertTrue(m[9])
        with self.assertRaises(ValueError):trajectory([(3.,1.)],2.,1)

    def test_encoding_does_not_fit_heldout_values(self):
        raw=np.ones((5,9,10));raw[:,:,0]=np.arange(5)[:,None]+1;raw[:,:,1]=raw[:,:,0]+.1
        a=dict(raw=raw,valid=np.ones_like(raw,bool),clinical=np.zeros((5,6)))
        for view in ('linear','ple','velocity','density','curvature'):
            x,parameters=encode(a,np.array([0,1,2]),view)
            changed=dict(a,raw=raw.copy());changed['raw'][3:]*=1e5
            y,other=encode(changed,np.array([0,1,2]),view)
            self.assertEqual(parameters,other);np.testing.assert_array_equal(x[:3],y[:3])

    def test_clinical_comparison_only_changes_interaction(self):
        a=dict(raw=np.zeros((2,9,10)),valid=np.zeros((2,9,10),bool),clinical=np.ones((2,6)))
        main,_=encode(a,np.array([0]),'clinical_main');full,_=encode(a,np.array([0]),'clinical_interaction')
        self.assertEqual(np.count_nonzero(main!=full),2);np.testing.assert_array_equal(main[:,:5],full[:,:5])


if __name__=='__main__':unittest.main()
