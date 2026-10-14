"""验证患者分割与固定候选清单，防止未来数据/患者混折。"""
import unittest
import numpy as np
from training.v6.regularization_model import TRIALS
from training.v6.regularization_search import validation_parts


class Tests(unittest.TestCase):
    def test_split_is_patientwise_and_reproducible(self):
        people=np.repeat(np.arange(100),3)
        splits=np.array(['validation']*210+['train']*60+['test']*30)
        a,b=validation_parts(people,splits)
        aa,bb=validation_parts(people,splits)
        np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)
        self.assertFalse(set(people[a]) & set(people[b]))
        np.testing.assert_array_equal(np.sort(np.r_[a,b]),np.flatnonzero(splits=='validation'))

    def test_nonvalidation_changes_do_not_choose_validation_patients(self):
        people=np.repeat(np.arange(100),3)
        splits=np.array(['validation']*210+['train']*60+['test']*30)
        a,b=validation_parts(people,splits)
        people[210:]+=10000
        aa,bb=validation_parts(people,splits)
        np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)

    def test_trial_ids_unique_and_control_unchanged(self):
        self.assertEqual(len(TRIALS),26)
        self.assertEqual(len({x.name for x in TRIALS}),len(TRIALS))
        control=TRIALS[0]
        self.assertEqual(control.architecture,'control')
        self.assertEqual(control.dropout,.1)
        self.assertEqual(control.consistency,0.)
        self.assertEqual(control.brier,0.)


if __name__=='__main__':unittest.main()
