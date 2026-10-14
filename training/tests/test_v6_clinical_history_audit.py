import unittest
import numpy as np
from training.v6.clinical_history_audit import previous_positions


class Tests(unittest.TestCase):
    def test_completed_history_only_with_no_same_time_or_future(self):
        records=[dict(hadm_id=str(i),subject_id='1',admittime=a,dischtime=b) for i,a,b in (
            (11,'2020-01-01','2020-01-03'),(12,'2020-01-02','2020-01-06'),
            (13,'2020-01-03','2020-01-05'),(14,'2020-01-03','2020-01-04'),
            (15,'2020-01-07','2020-01-09'))]
        index,offset,gap=previous_positions(records,np.array([11,12,13,14,15]),np.ones(5,int),np.array(['train']*5))
        np.testing.assert_array_equal(index[offset[1]:offset[2]],[])
        np.testing.assert_array_equal(index[offset[2]:offset[3]],[0])
        np.testing.assert_array_equal(index[offset[3]:offset[4]],[0])
        np.testing.assert_array_equal(index[offset[4]:offset[5]],[0,3,2,1])
        self.assertTrue((gap>=0).all())

    def test_rejects_cross_patient_fold_and_duplicate_ids(self):
        rows=[dict(hadm_id='11',subject_id='1',admittime='2020-01-01',dischtime='2020-01-02'),
            dict(hadm_id='12',subject_id='1',admittime='2020-01-03',dischtime='2020-01-04')]
        with self.assertRaises(ValueError):previous_positions(rows,np.array([11,12]),np.ones(2,int),np.array(['train','val']))
        with self.assertRaises(ValueError):previous_positions(rows,np.array([11,11]),np.ones(2,int),np.array(['train']*2))


if __name__=='__main__':unittest.main()
