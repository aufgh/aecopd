from datetime import datetime
import unittest
import numpy as np
from patient_data.core_lab_value_audit import accepted,readout

class CoreValueTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(charttime='2020-01-02',storetime='2020-01-02',valuenum='14',
            ref_range_lower='12',ref_range_upper='16',valueuom='g/dL',labevent_id='1')
        self.bounds=(datetime(2020,1,1),datetime(2020,1,3))
    def test_accept(self):self.assertIsNone(accepted(self.row,self.bounds,'g/dL')[1])
    def test_units(self):self.assertEqual(accepted(self.row,self.bounds,'mg/dL')[1],'unit_mismatch')
    def test_arterial_proof(self):
        self.assertEqual(accepted(self.row,self.bounds,'g/dL',set())[1],'not_old_verified_arterial')
        self.assertIsNone(accepted(self.row,self.bounds,'g/dL',{'1'})[1])
    def test_future_store(self):
        self.row['storetime']='2020-01-04';self.assertEqual(accepted(self.row,self.bounds,'g/dL')[1],'not_hospital_visible')
    def test_bad_values(self):
        for v in ('nan','0','-1',r'\N'):
            self.row['valuenum']=v;self.assertIsNotNone(accepted(self.row,self.bounds,'g/dL')[1])
    def test_no_observation_not_normal(self):
        raw,p,z,info=readout([],3);self.assertFalse(info['observed']);self.assertFalse(raw.any())
    def test_retains_values_with_zero_severity(self):
        x,p,_,info=readout([(.2,13,12,16,0),(2,15,12,16,0)],3)
        self.assertTrue(info['entirely_zero_direction'] and info['original_value_varies']);np.testing.assert_array_equal(p,[13,15])
        y,_,_,_=readout([(.2,14,12,16,0),(2,14,12,16,0)],3);np.testing.assert_array_equal(x,y)
    def test_same_time_not_repeated_draw(self):
        _,p,_,info=readout([(1,13,12,16,0),(1,15,12,16,0)],3)
        self.assertEqual(info['timestamps'],1);np.testing.assert_array_equal(p,[14,14])
    def test_reordering_invariant(self):
        samples=[(.2,14,12,16,0),(2,15,12,16,0)];a=readout(samples,3);b=readout(samples[::-1],3)
        for x,y in zip(a[:3],b[:3]):np.testing.assert_array_equal(x,y)
        self.assertEqual(a[3],b[3])

if __name__=='__main__':unittest.main()
