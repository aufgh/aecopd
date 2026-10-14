import unittest
import numpy as np
import torch
from training.v6.early_stopping import EarlyStopping
from training.v6.text_geometry import fit_geometry,transform_rows


class Tests(unittest.TestCase):
    def test_early_stop_does_not_require_cap(self):
        stop=EarlyStopping(patience=3,min_epochs=4)
        states=[stop.observe(i,v) for i,v in enumerate([1.,.9,.91,.92,.93])]
        self.assertEqual(states[-1],(False,True));self.assertEqual(stop.best_epoch,1)

    def test_tiny_best_is_saved_without_resetting_patience(self):
        stop=EarlyStopping(patience=2,min_epochs=0,min_delta=.01)
        self.assertEqual(stop.observe(0,1.),(True,False))
        self.assertEqual(stop.observe(1,.999),(True,False))
        self.assertEqual(stop.observe(2,.998),(True,True));self.assertEqual(stop.best_epoch,2)

    def test_nonfinite_and_repeated_epoch(self):
        with self.assertRaises(ValueError):EarlyStopping().observe(0,float('nan'))
        s=EarlyStopping();s.observe(0,1.)
        with self.assertRaises(ValueError):s.observe(0,.9)

    def test_holdout_cannot_change_train_transform(self):
        torch.manual_seed(2);x=torch.randn(60,8)+2;fields=torch.arange(60)%4;w=torch.ones(60)
        a,b,_=fit_geometry(x[:40],fields[:40],w[:40])
        x[40:]=100
        ar,br,_=fit_geometry(x[:40],fields[:40],w[:40])
        self.assertTrue(torch.equal(a,ar));self.assertTrue(torch.equal(b,br))

    def test_transform_finite_permutation_and_zero(self):
        torch.manual_seed(3);x=torch.randn(40,8)+2;f=torch.arange(40)%4
        m,k,diag=fit_geometry(x,f,torch.ones(40))
        y=transform_rows(x,f,m,k);p=torch.randperm(40)
        torch.testing.assert_close(transform_rows(x[p],f[p],m,k),y[p])
        torch.testing.assert_close(y.norm(dim=1),torch.ones(40))
        z=transform_rows(m,torch.arange(4),m)
        self.assertTrue(torch.equal(z,torch.zeros_like(z)))
        self.assertTrue(all(0<d['covariance_participation_rank']<=8 for d in diag))


if __name__=='__main__':unittest.main()
