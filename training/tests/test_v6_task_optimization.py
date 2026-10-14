import unittest
import torch
from training.tests.test_v6_deep_interaction import example
from training.v6.model import HistoryBaseline
from training.v6.task_optimization import TRIALS,build,projected_sum,backward
from training.v6.task_gradient_diagnostic import cause_losses
from training.v6.risk import competing_nll


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_initial_risk_and_encoder_weights_unchanged(self):
        base=HistoryBaseline();b=example();h=torch.randn(3,6)
        control=build(base,42,TRIALS[0],[[-1.,1.]]*132,text_width=8,device='cpu')
        for trial in TRIALS:
            m=build(base,42,trial,[[-1.,1.]]*132,text_width=8,device='cpu').eval()
            out,expected=m(h,b),m.baseline(h)
            torch.testing.assert_close(out.log_amplitude,expected.log_amplitude,atol=0,rtol=0)
            torch.testing.assert_close(out.shape,expected.shape,atol=0,rtol=0)
            encoders=m.encoders if trial.independent_encoders else [m.encoder]
            for enc in encoders:
                for k,v in control.encoder.state_dict().items():torch.testing.assert_close(v,enc.state_dict()[k],atol=0,rtol=0)

    def test_projection_nonconflict_and_conflict(self):
        left=[torch.tensor([1.,0.])];right=[torch.tensor([0.,1.])]
        torch.testing.assert_close(projected_sum(left,right)[0],torch.tensor([1.,1.]))
        left=[torch.tensor([1.,0.])];right=[torch.tensor([-1.,1.])]
        result=projected_sum(left,right)[0]
        self.assertGreaterEqual(float((result*left[0]).sum()),0.)
        self.assertGreaterEqual(float((result*right[0]).sum()),0.)
        self.assertTrue(torch.isfinite(projected_sum([torch.zeros(2)],[torch.zeros(2)],balance=True)[0]).all())

    def test_independent_branch_has_no_other_cause_gradient(self):
        m=build(HistoryBaseline(),42,TRIALS[4],[[-1.,1.]]*132,text_width=8,device='cpu')
        for head in m.head.heads:torch.nn.init.normal_(head[-1].weight,std=.1)
        out=m(torch.randn(3,6),example());loss=cause_losses(out,torch.tensor([20.,50.,365.]),torch.tensor([1,2,0]))
        loss[0].backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.encoders[0].parameters() if p.grad is not None),0.)
        self.assertEqual(sum(float(p.grad.abs().sum()) for p in m.encoders[1].parameters() if p.grad is not None),0.)

    def test_all_trials_backpropagate_without_foundation_gradient(self):
        for trial in TRIALS:
            m=build(HistoryBaseline(),42,trial,[[-1.,1.]]*132,text_width=8,device='cpu')
            if trial.independent_encoders:
                for head in m.head.heads:torch.nn.init.normal_(head[-1].weight,std=.02)
            elif hasattr(m.delta_head,'heads'):
                for head in m.delta_head.heads:torch.nn.init.normal_(head[-1].weight,std=.02)
            else:torch.nn.init.normal_(m.delta_head[-1].weight,std=.02)
            t=torch.tensor([20.,50.,365.]);e=torch.tensor([1,2,0]);out=m(torch.randn(3,6),example())
            nll=competing_nll(out,t,e);penalty=out.delta.square().mean()*1e-4
            backward(m,out,nll,penalty,t,e,trial)
            self.assertTrue(all(p.grad is None for p in m.baseline.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None),0.)


if __name__=='__main__':unittest.main()
