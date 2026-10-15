"""检查真正停用模块、保留历史gap及原公式等价，不镜像训练循环。"""
import unittest
from dataclasses import replace
import torch
from training.tests.test_v7_model import ToyBaseline,toy_batch
from training.v7.model import V7Model
from training.v7.sequential_variants import Variant,SequentialV7,loss_terms
from training.v7.losses import objective


class SequentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_unchanged_risk_path_equals_original(self):
        b=toy_batch();base=ToyBaseline();parents=torch.ones(38,10)
        old=V7Model(base,parents).eval();new=SequentialV7(base,parents,Variant('control')).eval()
        for name,value in old.state_dict().items():torch.testing.assert_close(value,new.state_dict()[name],rtol=0,atol=0)
        torch.testing.assert_close(old.visits(b)[0],new.visits(b)[0],rtol=0,atol=0)
        with torch.no_grad():old.delta_head[-1].weight.fill_(.001);new.delta_head[-1].weight.fill_(.001)
        torch.testing.assert_close(old(b).log_amplitude,new(b).log_amplitude,rtol=0,atol=0)

    def test_no_cross_is_really_bypassed(self):
        b=toy_batch();m=SequentialV7(ToyBaseline(),torch.ones(38,10),Variant('off',cross=False)).eval()
        expected=m.visits(b)[0]
        with torch.no_grad():
            for module in (m.drug_cross,m.rad_cross):
                for p in module.parameters():p.fill_(100.)
        torch.testing.assert_close(expected,m.visits(b)[0],rtol=0,atol=0)
        self.assertFalse(any(p.requires_grad for p in m.drug_cross.parameters()))

    def test_no_decay_keeps_gap_and_history(self):
        b=toy_batch();m=SequentialV7(ToyBaseline(),torch.ones(38,10),Variant('off',decay=False)).eval()
        self.assertFalse(m.history_pool.decay);self.assertFalse(m.history_pool.lambda_raw.requires_grad)
        with torch.no_grad():m.delta_head[-1].weight.normal_(0,.01)
        expected=m(b).log_amplitude
        self.assertGreater(float((m(replace(b,gaps=b.gaps+100.)).log_amplitude-expected).abs().max().detach()),0)

    def test_added_lab_keys_respects_missing_values(self):
        b=toy_batch();m=SequentialV7(ToyBaseline(),torch.ones(38,10),Variant('lab',lab_keys=True)).eval()
        labs=b.labs.clone();labs[:,:,6]=0
        missing=replace(b,labs=labs);expected=m.visits(missing)[0]
        labs[:,:,:6]=1e7;labs[:,:,7:]=1e7
        torch.testing.assert_close(expected,m.visits(replace(missing,labs=labs))[0],rtol=0,atol=0)
        self.assertTrue(torch.isfinite(expected).all())

    def test_control_loss_equals_locked_nll(self):
        b=toy_batch();m=SequentialV7(ToyBaseline(),torch.ones(38,10),Variant('control'))
        out,reps,masks=m(b,auxiliary=True);t=torch.tensor([10.,100.]);e=torch.tensor([1,2])
        new,_=loss_terms(out,t,e,Variant('control'))
        old,_=objective(out,t,e,reps,masks,torch.tensor([1,2]),recommended=False)
        torch.testing.assert_close(new,old,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
