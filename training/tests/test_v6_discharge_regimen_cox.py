import unittest
import numpy as np
from training.v6.discharge_regimen_cox import views,lock,probability_readout


class RegimenCoxTests(unittest.TestCase):
    def setUp(self):
        self.h=np.ones((4,34),np.float32);self.x=np.zeros((4,34),np.float32);self.m=np.ones(4,bool)

    def test_width_and_common_columns(self):
        v=views(self.h,self.x,self.m);self.assertEqual(v['availability'].shape,(4,35));self.assertEqual(v['content'].shape,(4,69))
        np.testing.assert_array_equal(v['availability'],v['content'][:,:35])

    def test_content_blinded(self):
        a=views(self.h,self.x,self.m);self.x[:]=1;b=views(self.h,self.x,self.m)
        np.testing.assert_array_equal(a['availability'],b['availability']);self.assertFalse(np.array_equal(a['content'],b['content']))

    def test_missing_source_cannot_have_content(self):
        self.m[0]=False;self.x[0,0]=1
        with self.assertRaises(ValueError):views(self.h,self.x,self.m)

    def test_nonfinite(self):
        self.x[0,0]=np.nan
        with self.assertRaises(ValueError):views(self.h,self.x,self.m)

    def test_binary_required(self):
        self.x[0,0]=.5
        with self.assertRaises(ValueError):views(self.h,self.x,self.m)

    def test_source_must_be_mask(self):
        with self.assertRaises(ValueError):views(self.h,self.x,self.m.astype(int))

    def test_inputs_unchanged(self):
        old=self.x.copy();v=views(self.h,self.x,self.m);v['content'][:]=0;np.testing.assert_array_equal(old,self.x)

    def test_lock_not_auc_only(self):
        def item(ibs,auc,death):return dict(metrics=dict(readmission=dict(integrated_brier_7_365=ibs,points={'90':dict(auc=auc)}),death=dict(integrated_brier_7_365=death)))
        base=item(.1,.78,.06);r=dict(availability=base,content=item(.1001,.80,.06))
        self.assertFalse(lock(r,base)['material_signal']);self.assertIsNone(lock(r,base)['conditional_signal'])
        r['content']=item(.098,.79,.059);self.assertTrue(lock(r,base)['material_signal'])
        r['content']=item(.098,.79,.061);self.assertFalse(lock(r,base)['material_signal'])

    def curve(self):
        curve=np.zeros((4,366,3));curve[:,:,0]=1;curve[:,1:,0]=.7;curve[:,1:,1]=.2;curve[:,1:,2]=.1
        return curve

    def test_probability_readout_not_accuracy(self):
        r=probability_readout(np.array([20.,40.,365.,365.]),np.array([1,2,0,0]),self.curve())
        self.assertAlmostEqual(r['readmission']['90']['mean_predicted'],.2);self.assertEqual(r['readmission']['90']['observed_fraction'],.25)
        self.assertAlmostEqual(r['readmission']['90']['binned_weighted_absolute_gap'],.05)
        self.assertEqual(r['death']['30']['observed_fraction'],0)

    def test_probability_conservation_required(self):
        curve=self.curve();curve[:,:,0]=1
        with self.assertRaises(ValueError):probability_readout(np.array([20.,40.,365.,365.]),np.array([1,2,0,0]),curve)

    def test_tied_scores_not_forced_into_bins(self):
        r=probability_readout(np.array([20.,40.,365.,365.]),np.array([1,2,0,0]),self.curve())
        self.assertEqual(len(r['readmission']['90']['bins']),1);self.assertEqual(r['readmission']['90']['bins'][0]['n'],4)


if __name__=='__main__':unittest.main()
