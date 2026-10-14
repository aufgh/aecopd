import unittest
import torch
from training.v6.model import HistoryBaseline
from training.v6.module_surgery import SurgeryModel
from training.tests.test_v6_deep_interaction import example
from training.v6.warm_residual import TRIALS,build
from training.v6.risk import competing_nll


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def teacher(self):
        base=HistoryBaseline();teacher=SurgeryModel(base,42,'control',[[-1,1]]*132,text_width=8)
        torch.nn.init.normal_(teacher.delta_head[-1].weight,std=.02)
        return base,teacher.eval()

    def test_every_arm_starts_at_trained_teacher_not_base(self):
        base,teacher=self.teacher();h=torch.randn(3,6);b=example();expected=teacher(h,b)
        for trial in TRIALS:
            m=build(base,42,trial,[[-1,1]]*132,text_width=8,device='cpu',teacher_state=teacher.state_dict()).eval()
            out=m(h,b)
            torch.testing.assert_close(out.log_amplitude,expected.log_amplitude,atol=0,rtol=0)
            torch.testing.assert_close(out.shape,expected.shape,atol=0,rtol=0)

    def test_adapter_teacher_frozen_and_amplitude_only_keeps_shape(self):
        base,teacher=self.teacher();h=torch.randn(3,6);b=example()
        for trial in TRIALS[:3]:
            m=build(base,42,trial,[[-1,1]]*132,text_width=8,device='cpu',teacher_state=teacher.state_dict())
            torch.nn.init.normal_(m.head[-1].weight,std=.02)
            out=m(h,b);competing_nll(out,torch.tensor([20.,50.,365.]),torch.tensor([1,2,0])).backward()
            self.assertTrue(all(p.grad is None for p in m.teacher.parameters()))
            self.assertGreater(float(m.head[-1].weight.grad.abs().sum()),0.)
            if trial.mode!='bounded_shape':torch.testing.assert_close(out.shape,teacher(h,b).shape,atol=0,rtol=0)
            else:self.assertLessEqual(float(out.delta[:,:,1].detach().abs().max()),.25)

    def test_finetune_keeps_history_frozen_but_updates_clinical(self):
        base,teacher=self.teacher();m=build(base,42,TRIALS[-1],[[-1,1]]*132,text_width=8,device='cpu',teacher_state=teacher.state_dict())
        out=m(torch.randn(3,6),example());competing_nll(out,torch.tensor([20.,50.,365.]),torch.tensor([1,2,0])).backward()
        self.assertTrue(all(p.grad is None for p in m.baseline.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.encoder.parameters() if p.grad is not None),0.)


if __name__=='__main__':unittest.main()
