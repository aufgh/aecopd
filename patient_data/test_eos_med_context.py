import unittest
from datetime import datetime,timedelta
import numpy as np
from patient_data.eos_med_context import canonical_eos, relation, summarize_eos


class EosTests(unittest.TestCase):
    def setUp(self): self.t=datetime(2100,1,1)
    def test_units_and_zero_valid(self):
        self.assertEqual(canonical_eos('.3','K/uL','absolute'),300)
        self.assertEqual(canonical_eos('300','#/uL','absolute'),300)
        self.assertEqual(canonical_eos('0','%','percent'),0)
        self.assertIsNone(canonical_eos('2','mg/dL','absolute'))
        self.assertIsNone(canonical_eos('101','%','percent'))
    def test_before_after_tie_unknown_and_earlier_ambiguity(self):
        self.assertEqual(relation(self.t,self.t+timedelta(hours=1)),0)
        self.assertEqual(relation(self.t+timedelta(hours=2),self.t),1)
        self.assertEqual(relation(self.t,self.t),2)
        self.assertEqual(relation(self.t,None),3)
        self.assertEqual(relation(self.t,self.t+timedelta(hours=1),[self.t-timedelta(hours=1)]),3)
    def test_missing_differs_from_observed_zero(self):
        end=self.t+timedelta(days=8)
        v,m,c=summarize_eos([],end,None)
        self.assertFalse(m.any()); self.assertEqual(c.sum(),0)
        v,m,c=summarize_eos([(self.t,0)],end,None)
        self.assertTrue(m[0]); self.assertFalse(m[4]); self.assertEqual(c[3],1)
    def test_pre_value_and_latest_after_preserved(self):
        end=self.t+timedelta(days=8); steroid=self.t+timedelta(hours=4)
        v,m,c=summarize_eos([(self.t,300),(self.t+timedelta(days=1),0)],end,steroid)
        np.testing.assert_array_equal(v[:5],[300,0,300,0,300]); self.assertTrue(m[4])
        self.assertEqual(c[0],1); self.assertEqual(c[5],1)
    def test_same_timestamp_no_order_and_permutation(self):
        samples=[(self.t,0),(self.t,200),(self.t+timedelta(days=1),50)]
        a=summarize_eos(samples,self.t+timedelta(days=3),self.t)
        b=summarize_eos(list(reversed(samples)),self.t+timedelta(days=3),self.t)
        for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
        self.assertEqual(a[0][0],100); self.assertEqual(a[2][2],1)


if __name__=='__main__': unittest.main()
