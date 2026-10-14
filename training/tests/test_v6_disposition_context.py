import unittest
import numpy as np
from patient_data.disposition_audit import eligibility,time_role
from training.v6.disposition_context import classify,encode,feature_view,lock,CATEGORIES

class DispositionTests(unittest.TestCase):
    def test_home_distinction(self):self.assertNotEqual(classify('HOME'),classify('HOME HEALTH CARE'))
    def test_special_destinations(self):
        self.assertEqual(classify('HOSPICE'),5);self.assertEqual(classify('AGAINST ADVICE'),6)
    def test_unknown_and_died(self):
        for value in ('',r'\N','DIED','new_unverified',None):self.assertIsNone(classify(value))
        np.testing.assert_array_equal(encode(['DIED'])['features'],np.zeros((1,32)))
        self.assertFalse(encode(['DIED'])['observed'][0])
    def test_content_hiding(self):
        x=encode(list(CATEGORIES[:-1])+['PSYCH FACILITY'])['features']
        np.testing.assert_array_equal(feature_view(x,'availability'),np.zeros_like(x))
        np.testing.assert_array_equal(feature_view(x,'content'),x)
    def test_one_hot_and_padding(self):
        x=encode(['HOME','REHAB','ACUTE HOSPITAL'])['features']
        np.testing.assert_array_equal(x.sum(1),np.ones(3));self.assertFalse(x[:,8:].any())
    def test_qualification_not_destination(self):
        row=dict(hospital_expire_flag='0',admittime='2020-01-01',dischtime='2020-01-02',discharge_location='DIED')
        self.assertTrue(eligibility(row));row['hospital_expire_flag']='1';self.assertFalse(eligibility(row))
        row['hospital_expire_flag']='0';row['dischtime']=row['admittime'];self.assertFalse(eligibility(row))
    def test_death_time_role_qc_only(self):
        row=dict(admittime='2020-01-02',dischtime='2020-01-03')
        for v,e in ((r'\N','unrecorded'),('2020-01-01','before_admission'),('2020-01-03','within_hospital'),('2020-01-04','after_discharge')):
            self.assertEqual(time_role(v,row),e)
    def test_lock_needs_actual_content_gain(self):
        def r(v):return dict(metrics=dict(readmission=dict(integrated_brier_7_365=v)))
        report=dict(fits=dict(availability=r(.1),content=r(.1)),reference=r(.1)['metrics'])
        self.assertEqual(lock(report)['nominated'],[])
        report['fits']['content']=r(.099);self.assertEqual(lock(report)['nominated'],['content'])

if __name__=='__main__':unittest.main()
