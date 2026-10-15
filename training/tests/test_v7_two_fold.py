"""两折患者隔离、互斥风险集、日曲线守恒和共同温度。"""
import unittest
import numpy as np
from training.v7.two_fold_ensemble import patient_folds,period_rows,interval_cif,temper,common_weights


class TwoFoldTests(unittest.TestCase):
    def test_patient_fold_and_no_future_samples(self):
        subjects=np.array([1,1,2,2,3,4,5]);fold=patient_folds(subjects,np.arange(6))
        self.assertEqual(fold[0],fold[1]);self.assertEqual(fold[2],fold[3]);self.assertEqual(fold[6],-1)
        x=np.zeros((3,2));xx,yy=period_rows(x,np.array([5.,10.,365.]),np.array([2,1,0]),np.arange(3))
        self.assertEqual(len(xx),1+2+7);self.assertEqual((yy==2).sum(),1);self.assertEqual((yy==1).sum(),1)

    def test_interval_probability_and_temperature(self):
        p=np.broadcast_to(np.array([.8,.12,.08]),(3,7,3)).copy();curve=interval_cif(p)
        np.testing.assert_allclose(curve.sum(2),1,atol=1e-12)
        self.assertTrue((np.diff(curve[:,:,1:],axis=1)>=-1e-12).all())
        np.testing.assert_allclose(temper(curve,1),curve,atol=1e-12)
        for t in (.8,1.2):
            calibrated=temper(curve,t);np.testing.assert_allclose(calibrated.sum(2),1,atol=1e-12)
            self.assertTrue((np.diff(calibrated[:,:,1:],axis=1)>=-1e-12).all())

    def test_one_shared_convex_weight_vector(self):
        c=interval_cif(np.broadcast_to(np.array([.8,.1,.1]),(4,7,3)))
        curves=np.stack((c,c,c));w,value=common_weights(curves,np.array([10.,30.,100.,365.]),np.array([1,2,1,0]))
        self.assertTrue((w>=0).all());self.assertAlmostEqual(w.sum(),1);self.assertTrue(np.isfinite(value))


if __name__=='__main__':unittest.main()
