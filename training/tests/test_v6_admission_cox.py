import tempfile
from pathlib import Path
import unittest
import numpy as np
from training.v6.admission_cox_diagnostic import (
    input_view, save_model, load_model, validate_curve, decision, SETTINGS,
)
from training.baselines.continuous_competing_cox import fit_competing_cox, predict_cumulative_incidence


class AdmissionCoxTests(unittest.TestCase):
    def setUp(self):
        self.h = np.ones((5,34), np.float32)
        self.m = np.ones((5,4), bool)
        self.x = np.zeros((5,32), np.float32); self.x[:,2:6] = self.m

    def test_width_and_common_columns(self):
        for arm, width in (('h34',34), ('availability',38), ('content',59)):
            value = input_view(self.h,self.x,self.m,arm)
            self.assertEqual(value.shape,(5,width)); np.testing.assert_array_equal(value[:,:34],self.h)

    def test_availability_hides_content(self):
        a = input_view(self.h,self.x,self.m,'availability')
        self.x[:,:2] = 7; self.x[:,6:25] = 1
        np.testing.assert_array_equal(a,input_view(self.h,self.x,self.m,'availability'))
        self.assertEqual(input_view(self.h,self.x,self.m,'content')[0,34],7)

    def test_padding_rejected(self):
        self.x[:,31] = 1
        with self.assertRaises(ValueError): input_view(self.h,self.x,self.m,'content')

    def test_mask_alignment(self):
        self.x[:,2] = 0
        with self.assertRaises(ValueError): input_view(self.h,self.x,self.m,'content')

    def test_nonfinite_and_unknown_arm(self):
        with self.assertRaises(ValueError): input_view(self.h,self.x,self.m,'fake')
        self.h[0,0] = np.nan
        with self.assertRaises(ValueError): input_view(self.h,self.x,self.m,'h34')

    def test_solver_conservation_and_replay(self):
        rng = np.random.default_rng(42); x = rng.normal(size=(60,3))
        e = np.tile([0,1,2],20); t = np.where(e==0,365.,np.arange(60)+1.)
        model = fit_competing_cox(x,t,e,**SETTINGS)
        self.assertTrue(model.readmission.converged and model.death.converged)
        curve = predict_cumulative_incidence(model,x,np.arange(366)); validate_curve(curve)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'model.npz'; save_model(path,model)
            np.testing.assert_array_equal(curve,predict_cumulative_incidence(load_model(path),x,np.arange(366)))

    def test_no_early_censoring(self):
        with self.assertRaises(ValueError): fit_competing_cox(np.ones((3,2)),[10.,20.,30.],[0,1,2])

    def test_invalid_probability(self):
        with self.assertRaises(ValueError): validate_curve(np.ones((2,366,3)))

    def test_lock_needs_both_comparators(self):
        def item(ibs,auc=.8,death=.05):
            return dict(metrics=dict(readmission=dict(integrated_brier_7_365=ibs,points={'90':{'auc':auc}}),
                                     death=dict(integrated_brier_7_365=death)))
        results = dict(h34=item(.1),availability=item(.098),content=item(.099,.81))
        self.assertIsNone(decision(results)['conditional_signal'])
        results['content'] = item(.097,.806)
        self.assertTrue(decision(results)['material_signal'])
        results['content'] = item(.097,.806,.06)
        self.assertFalse(decision(results)['material_signal'])


if __name__ == '__main__': unittest.main()
