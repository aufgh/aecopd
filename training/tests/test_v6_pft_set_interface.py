import unittest
import numpy as np
from patient_data.pft_set_interface import temporal_reason,normalized
from patient_data.pft_fact_ledger import CONCEPTS


class Tests(unittest.TestCase):
    def test_explicit_post_discharge_isolated(self):
        for t in ('PFT after discharge showed FEV1 40% predicted','PFT following the discharge','PFT post-discharge','PFT after leaving hospital'):
            self.assertEqual(temporal_reason(t),'explicit_post_discharge_words')

    def test_future_testing_plans_isolated(self):
        for t in ('will repeat PFT tomorrow','PFT is scheduled next month','recommended spirometry after review','plan for FEV1 testing'):
            self.assertEqual(temporal_reason(t),'future_PFT_plan_words')

    def test_existing_background_not_automatically_future(self):
        self.assertIsNone(temporal_reason('History of COPD, FEV1 40% predicted on prior PFT.'))

    def test_only_training_scaling(self):
        c=np.tile(np.arange(6),3);p=np.repeat(np.arange(3),6);folds=np.array(['train','train','validation'])
        raw=np.r_[np.ones(6),np.ones(6)*3,np.ones(6)*1000]
        values,scales=normalized(raw,c,p,folds)
        self.assertEqual(scales[CONCEPTS[0]]['median'],2)
        self.assertEqual(scales[CONCEPTS[0]]['training_observations'],2)
        self.assertTrue((values[-6:]==8).all())

    def test_small_iqr_not_floored(self):
        c=np.tile(np.arange(6),2);p=np.repeat(np.arange(2),6);folds=np.array(['train','train'])
        _,s=normalized(np.r_[np.ones(6)*.3,np.ones(6)*.4],c,p,folds)
        self.assertLess(s[CONCEPTS[0]]['scale'],.1)

    def test_no_nontraining_scale_fallback(self):
        with self.assertRaises(ValueError):normalized(np.ones(6),np.arange(6),np.zeros(6,dtype=int),np.array(['validation']))


if __name__=='__main__':unittest.main()
