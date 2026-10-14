import unittest
import numpy as np
from training.v6.core_lab_cox_diagnostic import views,lock


class CoreLabCoxTests(unittest.TestCase):
    def setUp(self):
        self.h=np.ones((4,34),np.float32);self.x=np.zeros((4,32),np.float32)
        self.m=np.ones((4,5),bool);self.x[:,20:25]=self.m

    def test_common_columns(self):
        v=views(self.h,self.x,self.m)
        self.assertEqual(v['summary'].shape,(4,54));self.assertEqual(v['continuous'].shape,(4,64))
        np.testing.assert_array_equal(v['summary'][:,:34],v['continuous'][:,:34])
        np.testing.assert_array_equal(v['summary'][:,34:],v['continuous'][:,44:])

    def test_summary_blinds_physical_values(self):
        old=views(self.h,self.x,self.m);self.x[:,:10]=3
        new=views(self.h,self.x,self.m)
        np.testing.assert_array_equal(old['summary'],new['summary'])
        self.assertFalse(np.array_equal(old['continuous'],new['continuous']))

    def test_padding_rejected(self):
        self.x[:,30]=1
        with self.assertRaises(ValueError):views(self.h,self.x,self.m)

    def test_mask_must_match(self):
        self.m[:,0]=False
        with self.assertRaises(ValueError):views(self.h,self.x,self.m)

    def test_missing_physical_rejected(self):
        self.m[:,0]=False;self.x[:,20]=0;self.x[:,0]=4
        with self.assertRaises(ValueError):views(self.h,self.x,self.m)

    def test_nonfinite_rejected(self):
        self.x[0,0]=np.nan
        with self.assertRaises(ValueError):views(self.h,self.x,self.m)

    def test_decision_needs_combined_signal(self):
        def item(ibs,auc,death):
            return dict(metrics=dict(readmission=dict(integrated_brier_7_365=ibs,points={'90':dict(auc=auc)}),
                                     death=dict(integrated_brier_7_365=death)))
        r=dict(summary=item(.1,.78,.06),continuous=item(.0999,.79,.06))
        self.assertFalse(lock(r)['material_signal'])
        r['continuous']=item(.098,.786,.059);self.assertTrue(lock(r)['material_signal'])
        r['continuous']=item(.098,.786,.061);self.assertFalse(lock(r)['material_signal'])


if __name__=='__main__':unittest.main()
