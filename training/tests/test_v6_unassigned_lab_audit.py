from datetime import datetime
import unittest
from patient_data.unassigned_lab_audit import attribution,key

class AnchorTests(unittest.TestCase):
    def setUp(self):
        self.r=dict(subject_id='p',specimen_id='s',charttime='2020-01-02',storetime='2020-01-02')
        self.a={'h':dict(visible_times={'2020-01-02'})}
        self.bounds={'h':(datetime(2020,1,1),datetime(2020,1,3))}
    def result(self,a=None,e=None,dup=False):
        return attribution(self.r,self.a if a is None else a,self.bounds,{'h'} if e is None else e,dup)
    def test_good(self):self.assertEqual(self.result(),('h','same_specimen_hospital_evidence'))
    def test_no_anchor(self):self.assertEqual(self.result({})[1],'no_assigned_specimen_anchor')
    def test_multi_anchor(self):self.assertEqual(self.result({'h':self.a['h'],'other':self.a['h']})[1],'multiple_hospital_anchors')
    def test_missing_specimen(self):
        self.r['specimen_id']=r'\N';self.assertIsNone(key(self.r));self.assertEqual(self.result()[1],'missing_patient_or_specimen')
    def test_patient_key(self):
        k=key(self.r);self.r['subject_id']='other';self.assertNotEqual(k,key(self.r))
    def test_after_discharge(self):
        self.r['storetime']='2020-01-04';self.assertEqual(self.result()[1],'candidate_not_hospital_visible')
    def test_different_or_unavailable_anchor_time(self):
        self.assertEqual(self.result({'h':dict(visible_times={'2020-01-01'})})[1],'no_visible_same_time_anchor')
        self.assertEqual(self.result({'h':dict(visible_times=set())})[1],'no_visible_same_time_anchor')
    def test_not_training(self):self.assertEqual(self.result(e=set())[1],'anchor_not_eligible_training_admission')
    def test_duplicate(self):self.assertEqual(self.result(dup=True)[1],'duplicate_labevent_id')
    def test_unknown_times(self):
        self.r['charttime']=r'\N';self.assertEqual(self.result()[1],'candidate_not_hospital_visible')

if __name__=='__main__':unittest.main()
