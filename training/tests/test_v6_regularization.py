"""概率正则适配与随机深度验证；不凭工程通过宣称性能提升。"""
import unittest
import torch
import numpy as np
from training.v6.risk import RiskOutput,predict_cif,competing_nll
from training.v6.model import HistoryBaseline
from training.tests.test_v6_deep_interaction import example
from training.v6.regularization_model import (Trial,StochasticResidual,make_model,tensor_cif,
    probability_consistency,auxiliary_brier,TRIALS)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_probabilities_agree_with_reference_and_differentiate(self):
        a=torch.tensor([[-2.,-3.],[-1.,-.5]],requires_grad=True)
        k=torch.tensor([[.65,1.5],[1.2,.8]],requires_grad=True)
        out=RiskOutput(a,k)
        p=tensor_cif(out)
        expected=predict_cif(a.detach().numpy(),k.detach().numpy(),np.array([0,14,30,60,90,180,365]),step_days=.5)[:,1:]
        np.testing.assert_allclose(p.detach().numpy(),expected,atol=3e-6,rtol=3e-5)
        torch.testing.assert_close(p.sum(2),torch.ones_like(p[:,:,0]),atol=1e-10,rtol=0)
        self.assertTrue((torch.diff(p[:,:,1:],dim=1)>=0).all())
        auxiliary_brier(out,torch.tensor([90.,365.]),torch.tensor([1,0])).backward()
        self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(k.grad).all())
        self.assertGreater(float(a.grad.abs().sum()),0.)

    def test_consistency_zero_and_positive(self):
        a=torch.randn(3,2,requires_grad=True);k=torch.ones(3,2,requires_grad=True)
        x=RiskOutput(a,k)
        self.assertEqual(float(probability_consistency(x,x).detach()),0.)
        penalty=probability_consistency(x,RiskOutput(a+.2,k+.1))
        self.assertGreater(float(penalty.detach()),0.)
        penalty.backward();self.assertTrue(torch.isfinite(a.grad).all())

    def test_stochastic_depth_eval_and_expectation(self):
        class Double(torch.nn.Module):
            def forward(self,x):return x*2
        m=StochasticResidual(Double(),.5)
        x=torch.ones(10000,1,1)
        torch.manual_seed(7); y=m(x)
        self.assertTrue(((y==1)|(y==3)).all())
        self.assertAlmostEqual(float(y.mean()),2.,delta=.04)
        torch.testing.assert_close(m.eval()(x),x*2,atol=0,rtol=0)

    def test_all_trials_zero_risk_start_and_frozen_baseline(self):
        base=HistoryBaseline();b=example();h=torch.randn(3,6)
        for trial in TRIALS:
            m=make_model(base,42,trial,[[-1.,1.]]*132,text_width=8,width=32).eval()
            a,expected=m(h,b),m.baseline(h)
            torch.testing.assert_close(a.log_amplitude,expected.log_amplitude,atol=0,rtol=0)
            torch.testing.assert_close(a.shape,expected.shape,atol=0,rtol=0)
            self.assertTrue(all(not p.requires_grad for p in m.baseline.parameters()))

    def test_gpu_auxiliary_losses_get_gradient(self):
        if not torch.cuda.is_available():self.skipTest('GPU unavailable')
        a=torch.randn(8,2,device='cuda',requires_grad=True)
        k=torch.ones(8,2,device='cuda',requires_grad=True)
        out=RiskOutput(a,k)
        loss=auxiliary_brier(out,torch.linspace(1,365,8,device='cuda'),torch.arange(8,device='cuda')%3)
        loss+=probability_consistency(out,RiskOutput(a+.1,k+.1))
        loss.backward();self.assertTrue(torch.isfinite(a.grad).all())


if __name__=='__main__':unittest.main()
