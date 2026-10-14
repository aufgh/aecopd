"""组屏蔽内容不影响输出，冻结/解冻同参数初值但梯度约束不同。"""
import unittest
from dataclasses import replace
import torch
from training.tests.test_v6_core import batch
from training.v6.model import HistoryBaseline, AnchoredModel
from training.v6.group_ablation import mask_group, UnfrozenClinicalModel
from training.v6.risk import competing_nll


class GroupTests(unittest.TestCase):
    def test_masked_source_values_cannot_change_prediction(self):
        torch.set_num_threads(1)
        model=AnchoredModel(HistoryBaseline(),8,width=16).eval()
        torch.nn.init.normal_(model.delta_head[-1].weight,std=.01)
        b=batch(); h=torch.randn(3,6)
        for arm,changed in (
            ('without_current',replace(b,current_values=b.current_values+1000)),
            ('without_labs',replace(b,lab_axis_inputs=b.lab_axis_inputs+1000)),
            ('without_medications',replace(b,med_exposure=torch.zeros_like(b.med_exposure)))):
            one=model(h,mask_group(b,arm)); two=model(h,mask_group(changed,arm))
            torch.testing.assert_close(one.log_amplitude,two.log_amplitude,rtol=0,atol=0)
            torch.testing.assert_close(one.shape,two.shape,rtol=0,atol=0)

    def test_same_initial_state_but_history_gradients_differ(self):
        # 架构宽度采用正式默认值；所有参数来自共同状态。
        frozen=AnchoredModel(HistoryBaseline(),8)
        unfrozen=UnfrozenClinicalModel(HistoryBaseline(),8)
        unfrozen.load_state_dict(frozen.state_dict())
        h=torch.randn(3,6); b=batch()
        frozen.eval(); unfrozen.eval()
        f=frozen(h,b); u=unfrozen(h,b)
        torch.testing.assert_close(f.log_amplitude,u.log_amplitude,rtol=0,atol=0)
        torch.testing.assert_close(f.shape,u.shape,rtol=0,atol=0)
        t=torch.tensor([20.,30.,365.]); e=torch.tensor([1,2,0])
        competing_nll(f,t,e).backward(); competing_nll(u,t,e).backward()
        self.assertTrue(all(p.grad is None for p in frozen.baseline.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in unfrozen.baseline.parameters()),0)
        self.assertEqual(sum(p.numel() for p in frozen.parameters()),sum(p.numel() for p in unfrozen.parameters()))


if __name__=='__main__': unittest.main()
