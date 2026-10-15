"""全局训练映射、风险路径和梯度边界；不依赖患者数据。"""
import unittest
import numpy as np
import torch
from training.tests.test_v7_model import ToyBaseline, toy_batch
from training.v7.strong_anchor_experiment import StrongV7, StrongBatch
from training.v7.projected_strong import ProjectedStrongV7, ridge_operator, fit_coefficients


class ProjectedStrongTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        # 与真实实验setup一致，避免no_grad捕获切入另一套注意力内核。
        torch.backends.mha.set_fastpath_enabled(False)

    def make(self):
        b = toy_batch()
        anchors = dict(anchor_a=np.array([[-2., -3.], [-1., -2.]], np.float32),
            anchor_k=np.array([[.2, 2.], [1.3, .5]], np.float32), valid=np.ones(2, bool))
        return StrongBatch(b, torch.tensor([0, 1])), anchors

    def test_training_operator_does_not_read_heldout_and_handles_collinearity(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(60, 34)); x[:, 1:] = x[:, :1]
        design, op = ridge_operator(x[:40]); h = rng.normal(size=(40, 256))
        expected = fit_coefficients(op, h)
        x[40:] = 1e10
        _, other = ridge_operator(x[:40])
        np.testing.assert_array_equal(expected, fit_coefficients(other, h))
        self.assertTrue(np.isfinite(expected).all())
        np.testing.assert_allclose((h-design @ expected).mean(0), 0, atol=2e-7)
        with self.assertRaises(ValueError):
            ridge_operator(np.empty((0, 34)))
        with self.assertRaises(ValueError):
            fit_coefficients(op, h[:-1])

    def test_disabled_projection_exactly_matches_parent_and_zero_preserves_full_anchor(self):
        batch, anchors = self.make(); base = ToyBaseline(); parent = torch.ones(38, 10)
        old = StrongV7(base, parent, anchors).eval()
        new = ProjectedStrongV7(base, parent, anchors, enabled=False).eval()
        with torch.no_grad():
            old.delta_head[-1].weight.fill_(.01); old.delta_head[-1].bias.fill_(.02)
            new.delta_head[-1].load_state_dict(old.delta_head[-1].state_dict())
        a, b = old(batch), new(batch)
        torch.testing.assert_close(a.log_amplitude, b.log_amplitude, atol=0, rtol=0)
        torch.testing.assert_close(a.shape, b.shape, atol=0, rtol=0)
        new.projection_enabled = True
        with torch.no_grad():
            new.delta_head[-1].weight.zero_(); new.delta_head[-1].bias.zero_()
            new.projection_coefficients.normal_()
        out = new(batch)
        torch.testing.assert_close(out.log_amplitude, torch.as_tensor(anchors['anchor_a']), atol=0, rtol=0)
        torch.testing.assert_close(out.shape, torch.as_tensor(anchors['anchor_k']), atol=0, rtol=0)

    def test_projection_is_used_but_capture_is_raw_and_gradients_reach_encoder(self):
        batch, anchors = self.make()
        m = ProjectedStrongV7(ToyBaseline(), torch.ones(38, 10), anchors).eval()
        hidden = m.representations(batch).clone()
        with torch.no_grad():
            m.projection_coefficients[0].fill_(.1)
            m.delta_head[-1].weight.normal_(std=.01)
        seen = []
        handle = m.delta_head.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
        # 用与映射提取相同的no_grad路径核对变换；梯度另作独立断言。
        with torch.no_grad():
            m(batch)
        handle.remove()
        torch.testing.assert_close(seen[0], hidden-.1, atol=1e-7, rtol=0)
        torch.testing.assert_close(m.representations(batch), hidden, atol=0, rtol=0)
        m(batch).log_amplitude.sum().backward()
        self.assertGreater(float(m.history_fusion[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(m.projection_coefficients.grad)
        self.assertTrue(all(p.grad is None for p in m.baseline.parameters()))


if __name__ == '__main__':
    unittest.main()
