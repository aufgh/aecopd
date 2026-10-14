import unittest
import numpy as np
from training.v6.regularization_diagnostics import history_counts,calibration


class Tests(unittest.TestCase):
    def test_raw_log_counts_not_standardized_values(self):
        x=np.zeros((5,6));x[:,2]=np.log1p([0,1,2,3,10])
        np.testing.assert_array_equal(history_counts(x),[0,1,2,3,10])
        x[:,2]=.4
        with self.assertRaises(ValueError):history_counts(x)

    def test_calibration_keeps_competing_death_as_noncase(self):
        times=np.array([30.,30.,365.]);events=np.array([1,2,0])
        curve=np.zeros((3,366,3));curve[:,:,0]=.8;curve[:,:,1:]=.1
        out=calibration(times,events,curve,1)
        self.assertAlmostEqual(out['90']['observed'],1/3)
        self.assertAlmostEqual(out['90']['mean_predicted'],.1)


if __name__=='__main__':unittest.main()
