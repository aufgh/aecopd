import unittest
import numpy as np
from training.v6.core_lab_continuous import feature_view,scaled_values,lock

class ContinuousTests(unittest.TestCase):
    def test_actual_physical_blinding(self):
        x=np.arange(64,dtype=np.float32).reshape(2,32);a=feature_view(x,'summary')
        self.assertFalse(a[:,:10].any());np.testing.assert_array_equal(a[:,10:],x[:,10:])
        x[:,:10]=999;np.testing.assert_array_equal(a,feature_view(x,'summary'))
    def test_content_unchanged(self):
        x=np.ones((3,32));np.testing.assert_array_equal(x,feature_view(x,'continuous'))
    def test_training_only_true_iqr(self):
        x=np.zeros((5,5,2));x[:4]=np.array([7.36,7.38,7.40,7.42])[:,None,None];x[4]=100
        m=np.ones((5,5),bool);train=np.array([1,1,1,1,0],bool);a,s=scaled_values(x,m,train)
        self.assertLess(s['arterial_low_ph/0']['scale'],.1);self.assertEqual(a[4,0,0],8)
        x[4]=200;_,t=scaled_values(x,m,train);self.assertEqual(s,t)
    def test_missing_stays_masked(self):
        x=np.ones((4,5,2));m=np.ones((4,5),bool);m[3]=False
        a,_=scaled_values(x,m,np.ones(4,bool));self.assertFalse(a[3].any())
    def test_unknown_arm(self):
        with self.assertRaises(ValueError):feature_view(np.zeros((1,32)),'new')
    def test_lock_requires_both(self):
        def f(v):return dict(metrics=dict(readmission=dict(integrated_brier_7_365=v)))
        r=dict(fits=dict(summary=f(.1),continuous=f(.1)),reference=f(.1)['metrics']);self.assertFalse(lock(r)['nominated'])
        r['fits']['continuous']=f(.099);self.assertEqual(lock(r)['nominated'],['continuous'])

if __name__=='__main__':unittest.main()
