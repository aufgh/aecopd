import unittest
from datetime import datetime,timedelta
import numpy as np
import json
from unittest.mock import patch
from patient_data.respiratory_transition import summarize,support_pairs,collapse,normalize,icu_contains


class Tests(unittest.TestCase):
    def test_derived_metadata_json_types(self):
        # 真实源数据反馈环：统计指标必须能用allow_nan=False持久化。
        from patient_data.respiratory_transition import derive
        _,report=derive()
        self.assertIsInstance(report['coverage']['spo2']['training_latest_age_over2days'],int)
        json.dumps(report,allow_nan=False)

    def test_missing_icu_end_is_unknown_not_a_timestamp(self):
        t=datetime(2020,1,1)
        self.assertIsNone(icu_contains({'intime':'2020-01-01','outtime':r'\N'},t))
        self.assertIsNone(icu_contains({'intime':'2020-01-01','outtime':r'\\N'},t))
        self.assertTrue(icu_contains({'intime':'2020-01-01','outtime':'2020-01-02'},t))

    def test_single_is_not_zero_recovery(self):
        t=datetime(2020,1,1);raw,mask,flags=summarize([(t,30.,'1')],t,t+timedelta(days=5))
        self.assertFalse(mask[3]);self.assertFalse(mask[4]);self.assertEqual(flags.tolist(),[1,0,1])
        self.assertAlmostEqual(np.expm1(raw[6]),5.)

    def test_peak_at_latest_and_normal_constant_are_not_recovery(self):
        t=datetime(2020,1,1)
        _,mask,flags=summarize([(t,20.,'1'),(t+timedelta(days=1),30.,'1')],t,t+timedelta(days=5))
        self.assertFalse(mask[4]);self.assertEqual(flags[2],1)
        _,mask,flags=summarize([(t,20.,'1'),(t+timedelta(days=1),20.,'1')],t,t+timedelta(days=5))
        self.assertFalse(mask[4]);self.assertEqual(flags[2],1)

    def test_peak_before_latest_and_same_time_order_invariant(self):
        t=datetime(2020,1,1);obs=[(t,30.,'1'),(t+timedelta(days=1),20.,'1'),(t,32.,'1')]
        a=summarize(obs,t,t+timedelta(days=5));b=summarize(obs[::-1],t,t+timedelta(days=5))
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertEqual(a[0][4],-11.);self.assertTrue(a[1][4]);self.assertAlmostEqual(np.expm1(a[0][7]),1.)

    def test_support_pair_requires_same_stay_and_one_hour(self):
        t=datetime(2020,1,1);spo=[(t,88.,'1'),(t+timedelta(days=1),95.,'1')]
        flow=[(t,2.,'1'),(t+timedelta(days=1),5.,'1')]
        raw,mask=support_pairs(spo,flow,[]);self.assertEqual(raw[2],3.);self.assertTrue(mask[2])
        _,mask=support_pairs(spo,[(t,2.,'2')],[]);self.assertFalse(mask.any())
        _,mask=support_pairs(spo,[(t+timedelta(hours=2),2.,'1')],[]);self.assertFalse(mask.any())

    def test_support_change_is_not_defined_for_single_spo2(self):
        t=datetime(2020,1,1);_,mask=support_pairs([(t,95.,'1')],[(t,2.,'1')],[])
        self.assertTrue(mask[0]);self.assertTrue(mask[1]);self.assertFalse(mask[2])

    def test_normalization_uses_only_train_observations(self):
        raw=np.array([[1.,2.],[2.,4.],[100.,200.]])
        mask=np.ones_like(raw,bool);train=np.array([True,True,False])
        a,p=normalize(raw,mask,train);raw[-1]*=100;b,q=normalize(raw,mask,train)
        self.assertEqual(p,q);np.testing.assert_array_equal(a[:2],b[:2])


if __name__=='__main__':unittest.main()
