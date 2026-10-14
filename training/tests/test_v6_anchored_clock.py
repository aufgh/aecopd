import unittest
import numpy as np
import torch
from training.v6.anchored_clock import clock_nll,clock_curve,anchor_dense_to_daily
from training.v6.risk import RiskOutput,competing_nll,predict_cif


class Tests(unittest.TestCase):
    def test_identity_density_matches_original_joint_nll(self):
        a=torch.tensor([[-1.,-.7],[-.3,-1.2],[-2.,-2.]])
        k=torch.tensor([[.8,1.2],[1.3,.7],[.9,.8]])
        prior=RiskOutput(a,k);t=torch.tensor([20.,40.,365.]);e=torch.tensor([1,2,0])
        clock=torch.zeros(3,2,requires_grad=True)
        loss=clock_nll(prior,clock,t,e)
        torch.testing.assert_close(loss,competing_nll(prior,t,e),atol=1e-6,rtol=1e-6)
        loss.backward();self.assertTrue(torch.isfinite(clock.grad).all())
        self.assertEqual(float(clock.grad[2].abs().sum()),0.)

    def test_endpoint_preserved_and_competing_mass_coherent(self):
        days=np.arange(366);grid=np.arange(0,365.01,.5)
        a=np.array([[-1.,-.7],[-.3,-1.2]]);k=np.array([[.8,1.2],[1.3,.7]])
        curve=predict_cif(a,k,grid,step_days=.5)
        moved=clock_curve(curve,grid,np.array([[-.25,.25],[.2,-.2]]),days)
        np.testing.assert_array_equal(moved[:,-1],curve[:,-1])
        np.testing.assert_allclose(moved.sum(2),1,atol=1e-12,rtol=0)
        self.assertGreaterEqual(moved.min(),0.)
        self.assertTrue((np.diff(moved[:,:,1:],axis=1)>=-1e-12).all())
        unchanged=clock_curve(curve,grid,np.zeros((2,2)),days)
        np.testing.assert_array_equal(unchanged[:,:,1:],curve[:,::2,1:])

    def test_does_not_silently_accept_early_censoring(self):
        prior=RiskOutput(torch.zeros(1,2),torch.ones(1,2))
        with self.assertRaises(ValueError):clock_nll(prior,torch.zeros(1,2),torch.tensor([100.]),torch.tensor([0]))

    def test_daily_anchor_does_not_create_saturated_cif_sawtooth(self):
        grid=np.arange(0,365.01,.25)
        raw=np.zeros((1,len(grid),3));raw[:,:,1]=.4;raw[:,:,2]=.6;raw[:,:,0]=0.
        raw[:,0]=[1.,0.,0.]
        daily=raw[:,::4].copy();daily[:,1:,1]+=.0000008;daily[:,1:,2]-=.0000008
        broken=raw.copy();broken[:,::4]=daily
        self.assertLess(float(np.diff(broken[:,:,1:],axis=1).min()),-1e-10)
        fixed=anchor_dense_to_daily(raw,grid,daily)
        np.testing.assert_array_equal(fixed[:,::4],daily)
        self.assertTrue((np.diff(fixed[:,:,1:],axis=1)>=-1e-10).all())
        clock_curve(fixed,grid,np.zeros((1,2)),np.arange(366))


if __name__=='__main__':unittest.main()
