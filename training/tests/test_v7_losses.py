"""软标签 Focal 的定义、近边界梯度和患者配对排除。"""
import unittest
import torch
from torch.nn import functional as F
from training.v6.curve_score import daily_cif
from training.v6.risk import RiskOutput
from training.v7.losses import soft_focal, contrastive


class V7LossTests(unittest.TestCase):
    def test_formula(self):
        p = torch.tensor([.2, .8], dtype=torch.float64)
        y = torch.tensor([0., 1.], dtype=torch.float64)
        ys = torch.tensor([.025, .975], dtype=torch.float64)
        expected = (-.75 * ys * (1-p)**2 * p.log() - .25 * (1-ys) * p**2 * torch.log1p(-p)).mean()
        torch.testing.assert_close(soft_focal(p, y), expected, rtol=0, atol=0)

    def test_boundary_finite(self):
        p = torch.tensor([0., 1., 1e-6, 1-1e-6], requires_grad=True)
        soft_focal(p, torch.tensor([1., 0., 0., 1.])).backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        with self.assertRaises(ValueError): soft_focal(torch.tensor([float('nan')]), torch.ones(1))

    def test_same_patient_excluded_and_missing(self):
        reps = {k: F.normalize(torch.randn(3, 256), dim=1).requires_grad_() for k in ('text', 'lab', 'drug', 'rad')}
        masks = {k: torch.ones(3, dtype=torch.bool) for k in reps}
        masks['rad'][:] = False
        loss, stats = contrastive(reps, masks, torch.tensor([1, 1, 2]))
        self.assertEqual(stats, dict(positive_pairs=6, negative_pairs=8))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(reps['rad'].grad)

    def test_empty_differentiable(self):
        reps = {k: torch.zeros(2, 256, requires_grad=True) for k in ('text', 'lab', 'drug', 'rad')}
        masks = {k: torch.zeros(2, dtype=torch.bool) for k in reps}
        loss, stats = contrastive(reps, masks, torch.tensor([1, 1]))
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(stats['negative_pairs'], 0)
        loss.backward()
        self.assertTrue(all(v.grad is not None and not v.grad.any() for v in reps.values()))

    def test_cif_monotone_conserved_differentiable(self):
        a = torch.tensor([[-1., -2.], [0., -1.]], requires_grad=True)
        k = torch.tensor([[.7, 1.2], [1.3, .6]], requires_grad=True)
        curve = daily_cif(RiskOutput(a, k))
        torch.testing.assert_close(curve.sum(2), torch.ones_like(curve[:, :, 0]), rtol=0, atol=1e-12)
        self.assertTrue((torch.diff(curve[:, :, 1:], dim=1) >= -1e-14).all())
        soft_focal(curve[:, 365, 1], torch.tensor([1., 0.])).backward()
        self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(k.grad).all())


if __name__ == '__main__':
    unittest.main()
