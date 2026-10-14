import unittest
from datetime import datetime,timedelta
from patient_data.resp_micro_context import (timestamp_interval,available_micro,micro_scope,invalid_micro,
                                            target_organism,union_hours,canonical_resp,viral_positive)


class ContextTests(unittest.TestCase):
    def setUp(self): self.t=datetime(2100,1,1)
    def test_date_only_is_interval_not_midnight(self):
        interval=timestamp_interval(r'\N','2100-01-01 00:00:00')
        self.assertGreater(interval[1],interval[0])
        self.assertIsNone(timestamp_interval(r'\N',r'\N'))
    def test_future_result_and_boundary_unknown(self):
        row={'charttime':'2100-01-01 01:00:00','chartdate':'2100-01-01','storetime':'2100-01-02 03:00:00','storedate':'2100-01-02'}
        self.assertFalse(available_micro(row,self.t,self.t+timedelta(days=1))[0])
        row['storetime']=r'\N'
        self.assertFalse(available_micro(row,self.t,self.t+timedelta(days=1,hours=10))[0])
        self.assertTrue(available_micro(row,self.t,self.t+timedelta(days=2))[0])
    def test_not_pseudomonas_and_hflu_aliases(self):
        self.assertIsNone(target_organism('NON-FERMENTER, NOT PSEUDOMONAS AERUGINOSA'))
        self.assertEqual(target_organism('PSEUDOMONAS AERUGINOSA'),1)
        self.assertEqual(target_organism('HAEMOPHILUS INFLUENZAE, BETA-LACTAMASE NEGATIVE'),2)
    def test_specimen_and_invalid_distinctions(self):
        self.assertEqual(micro_scope({'spec_type_desc':'URINE','test_name':'URINE CULTURE'}),'other')
        self.assertEqual(micro_scope({'spec_type_desc':'SPUTUM','test_name':'RESPIRATORY CULTURE'}),'lower')
        self.assertTrue(invalid_micro({'org_name':r'\N','comments':'Culture results are invalid due to contamination'}))
        self.assertFalse(invalid_micro({'org_name':r'\N','comments':'NO GROWTH'}))
    def test_union_no_duplicate_duration(self):
        a=self.t; b=a+timedelta(hours=4); c=a+timedelta(hours=6)
        self.assertEqual(union_hours([(a,b),(a,b),(b,c)]),6)
        self.assertEqual(union_hours([]),0)
        with self.assertRaises(ValueError): union_hours([(b,a)])
    def test_canonical_units_and_fraction_fio2(self):
        self.assertEqual(canonical_resp('223835','.4',r'\N'),40)
        self.assertEqual(canonical_resp('223835','40','%'),40)
        self.assertIsNone(canonical_resp('223835','4','%'))
        self.assertIsNone(canonical_resp('220277','150','%'))
        self.assertEqual(canonical_resp('223834','0','L/min'),0)
        self.assertIsNone(canonical_resp('220210','20','L/min'))
    def test_viral_negative_is_not_positive(self):
        self.assertFalse(viral_positive(r'\N','NOT DETECTED: INFLUENZA A'))
        self.assertFalse(viral_positive('NEGATIVE FOR INFLUENZA A',''))
        self.assertTrue(viral_positive(r'\N','POSITIVE FOR INFLUENZA A'))


if __name__=='__main__': unittest.main()
