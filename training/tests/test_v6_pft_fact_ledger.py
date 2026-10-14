import unittest
from patient_data.pft_fact_ledger import canonical,add_fact


class Tests(unittest.TestCase):
    def test_ml_to_l_explicit_conversion(self):
        self.assertEqual(canonical('FEV1','volume_ml',1200),('fev1_volume_l',1.2,'L'))

    def test_ratio_percent_not_percent_predicted(self):
        self.assertEqual(canonical('FEV1/FVC','ratio_percent',65),('fev1_fvc_ratio',.65,'ratio'))
        self.assertIsNone(canonical('FEV1/FVC','ratio_percent_predicted',65))

    def test_predicted_percent_retains_percentage_scale(self):
        self.assertEqual(canonical('FEV1','percent_predicted',40),('fev1_percent_predicted',40,'%predicted'))

    def test_unknown_unit_and_unknown_percent_not_invented(self):
        self.assertIsNone(canonical('FEV1','unit_unknown',1.2))
        self.assertIsNone(canonical('FEV1','percent_unspecified',40))

    def test_same_value_dedup_preserves_distinct_evidence(self):
        bag={};add_fact(bag,1,2,'FEV1','volume_l',1.2,{'source':'a'})
        add_fact(bag,1,2,'FEV1','volume_ml',1200,{'source':'b'})
        add_fact(bag,1,2,'FEV1','volume_l',1.2,{'source':'a'})
        self.assertEqual(len(bag),1);self.assertEqual(len(next(iter(bag.values()))['evidence']),2)

    def test_distinct_values_and_admissions_not_averaged(self):
        bag={}
        for hadm,value in ((1,1.2),(1,1.5),(3,1.2)):add_fact(bag,hadm,2,'FEV1','volume_l',value,{})
        self.assertEqual(len(bag),3)

    def test_patient_conflict_rejected(self):
        bag={};add_fact(bag,1,2,'FEV1','volume_l',1.2,{})
        with self.assertRaises(ValueError):add_fact(bag,1,3,'FEV1','volume_l',1.2,{})

    def test_current_measurement_unknown_not_false_normal(self):
        bag={};add_fact(bag,1,2,'FEV1','volume_l',1.2,{})
        self.assertEqual(next(iter(bag.values()))['current_discharge_measurement'],'unknown')


if __name__=='__main__':unittest.main()
