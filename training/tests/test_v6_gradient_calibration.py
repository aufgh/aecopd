import unittest
import numpy as np
import torch
from training.v6.gradient_calibration_decomposition import components,common_center_gradient


class Tests(unittest.TestCase):
    def test_exact_mean_covariance_reconstruction(self):
        rng=np.random.default_rng(3);w=rng.normal(size=(4,32));b=rng.normal(size=4);mean=rng.normal(size=32)
        p=components(w,b,mean)
        np.testing.assert_allclose(p['mean']+p['covariance'],p['raw'],atol=1e-15,rtol=0)
        np.testing.assert_array_equal(p['covariance'][-4:],np.zeros(4))

    def test_shared_train_center_matches_actual_autograd(self):
        torch.manual_seed(12);x=torch.randn(6,32,dtype=torch.float64);raw=torch.randn(6,4,dtype=torch.float64)
        train_mean=torch.randn(32,dtype=torch.float64)  # 与当前折均值不同，核验共同训练中心。
        native=torch.nn.Linear(32,4,dtype=torch.float64)
        torch.nn.init.zeros_(native.weight);torch.nn.init.zeros_(native.bias)
        loss=(raw+native(x)).square().mean();loss.backward()
        expected=common_center_gradient(native.weight.grad.numpy(),native.bias.grad.numpy(),train_mean.numpy())
        centered=torch.nn.Linear(32,4,bias=False,dtype=torch.float64);torch.nn.init.zeros_(centered.weight)
        (raw+centered(x-train_mean)).square().mean().backward()
        np.testing.assert_allclose(centered.weight.grad.numpy(),expected,atol=1e-15,rtol=0)


if __name__=='__main__':unittest.main()
