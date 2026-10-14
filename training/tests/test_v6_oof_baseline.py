import unittest
import numpy as np
import torch
from training.v6.oof_baseline_diagnostic import fold_assignment, fit_scales, matrix, FreshH34, fit_model, weight_digest


class OOFTests(unittest.TestCase):
    def raw(self):
        return np.array([[50+i,i%2,np.log1p(i%4),np.log1p(i%3),np.log1p(i+1),0.] for i in range(20)],np.float32)

    def test_patient_folds_and_permutation(self):
        subjects=np.repeat(np.arange(12),3); folds=fold_assignment(subjects)
        self.assertEqual(len(set(folds)),5)
        for s in np.unique(subjects):self.assertEqual(len(set(folds[subjects==s])),1)
        p=np.random.default_rng(1).permutation(len(subjects))
        np.testing.assert_array_equal(fold_assignment(subjects[p]),folds[p])

    def test_scale_does_not_use_holdout(self):
        raw=self.raw();ix=np.arange(15);scales=fit_scales(raw,ix)
        changed=raw.copy();changed[15:,0:5]=1e6
        self.assertEqual(scales,fit_scales(changed,ix))

    def test_gap_missing_and_binary(self):
        raw=self.raw();raw[0,5]=1;raw[0,4]=999
        scales=fit_scales(raw,np.arange(20))
        self.assertEqual(scales['log_previous_discharge_gap_days']['training_observations'],19)
        current=np.ones((20,14));mask=np.ones((20,14),bool);mask[0,1]=False
        x=matrix(raw,current,mask,scales)
        self.assertEqual(x.shape,(20,34));self.assertEqual(x[0,4],0);self.assertEqual(x[0,5],1)
        self.assertEqual(x[0,7],0);self.assertEqual(x[0,21],0)
        np.testing.assert_array_equal(x[:,1],raw[:,1])

    def test_fresh_initialization_does_not_depend_on_global_rng(self):
        a=FreshH34();torch.manual_seed(8);_ = torch.rand(100);b=FreshH34()
        self.assertEqual(weight_digest(a),weight_digest(b))
        self.assertEqual(sum(p.numel() for p in a.parameters()),72)
        self.assertTrue(torch.equal(a.raw_shape,torch.zeros(2)))

    def test_scale_reject_duplicates(self):
        with self.assertRaises(ValueError):fit_scales(self.raw(),np.array([1,1]))

    def test_risk_and_single_step(self):
        x=torch.tensor(np.random.default_rng(3).normal(size=(12,34)),dtype=torch.float32)
        t=torch.arange(1,13,dtype=torch.float32);e=torch.tensor([0,1,2]*4)
        m,info=fit_model(x,t,e,updates=1)
        self.assertEqual(info['updates'],1);self.assertNotEqual(weight_digest(m),weight_digest(FreshH34()))
        r=m(x);self.assertEqual(r.shape.shape,(12,2));self.assertTrue(torch.isfinite(r.log_amplitude).all())


if __name__=='__main__':unittest.main()
