"""固定映射对单人推理一致，零门梯度能够启动。"""
import unittest
import numpy as np
import torch
from training.v7.residual_controls import fixed_projection,Model,DenseBatch


class ResidualControlTests(unittest.TestCase):
    def test_fixed_projection_heldout_batch_independent(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(8,3));h=rng.normal(size=(8,6))
        residual,b=fixed_projection(x,h,np.arange(5));single=h[6]-np.r_[1.,x[6]]@b
        np.testing.assert_allclose(single,residual[6],atol=1e-14)
        altered=h.copy();altered[5:]+=100
        _,other=fixed_projection(x,altered,np.arange(5));np.testing.assert_array_equal(b,other)

    def test_zero_gate_nonzero_head_has_initial_gradient(self):
        torch.set_num_threads(2);m=Model('zero_gate');b=DenseBatch(torch.randn(3,512),torch.zeros(3,2),torch.ones(3,2))
        out=m(b);torch.testing.assert_close(out.log_amplitude,b.a,rtol=0,atol=0)
        out.log_amplitude.sum().backward();self.assertGreater(float(m.alpha.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
