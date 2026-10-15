"""完整强风险的零修正保留；不误称纯H34。"""
import unittest
import torch
from training.tests.test_v7_model import ToyBaseline,toy_batch
from training.v7.strong_anchor_experiment import StrongV7,StrongBatch


class StrongAnchorTests(unittest.TestCase):
    def test_initial_equals_strong_and_can_start(self):
        torch.set_num_threads(2);b=toy_batch()
        anchors=dict(anchor_a=torch.tensor([[-2.,-3.],[-1.,-2.]]).numpy(),anchor_k=torch.tensor([[.2,2.],[1.3,.5]]).numpy(),valid=torch.ones(2,dtype=torch.bool).numpy())
        m=StrongV7(ToyBaseline(),torch.ones(38,10),anchors).eval();out=m(StrongBatch(b,torch.tensor([0,1])))
        torch.testing.assert_close(out.log_amplitude,torch.as_tensor(anchors['anchor_a']),rtol=0,atol=0)
        torch.testing.assert_close(out.shape,torch.as_tensor(anchors['anchor_k']),rtol=0,atol=0)
        out.log_amplitude.sum().backward()
        self.assertGreater(float(m.delta_head[-1].weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in m.baseline.parameters()))


if __name__=='__main__':unittest.main()
