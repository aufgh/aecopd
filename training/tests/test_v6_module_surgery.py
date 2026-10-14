"""局部替换合同：共同初值、冻结、掩码、FT官方参数和成员概率。"""
import copy
from dataclasses import replace
import unittest
import numpy as np
import torch
from training.tests.test_v6_compact_model import compact
from training.v6.model import HistoryBaseline
from training.v6.compact_model import CompactAnchoredModel
from training.v6.detail_extension_experiment import DetailEncoder
from training.v6.detail_extension_experiment import DetailBatch
from training.v6.original_interaction import ExpandedBatch
from training.v6.module_surgery import (ARMS, SurgeryModel, MaskedFTBlock, NumericLabEncoder,
    fit_lab_bins, member_nll, probability_curve, shared_initial_errors)


def sample():
    b = compact()
    return ExpandedBatch(DetailBatch(b, torch.randn(3, 90), torch.zeros(3, 192)),
        torch.zeros(3, 10, 10), torch.zeros(3, 38, 23), torch.zeros(3, 38, dtype=torch.bool))


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, arm):
        return SurgeryModel(self.base, 42, arm, [[-1., 1.]] * 132, text_width=8)

    def setUp(self):
        torch.manual_seed(11); self.base = HistoryBaseline(); self.b = sample()

    def test_unchanged_weights_and_zero_risk_start(self):
        control = self.model('control'); history = torch.randn(3, 6)
        for arm in ARMS:
            m = self.model(arm).eval(); shared_initial_errors(m, control)
            out, expected = m(history, self.b), m.baseline(history)
            a, k = out.log_amplitude, out.shape
            if a.ndim == 3:
                a, k = a[:, 0], k[:, 0]
                torch.testing.assert_close(out.log_amplitude, expected.log_amplitude[:, None].expand(-1, 16, -1), rtol=0, atol=0)
            torch.testing.assert_close(a, expected.log_amplitude, rtol=0, atol=0)
            torch.testing.assert_close(k, expected.shape, rtol=0, atol=0)
            self.assertEqual(m.detail.base.text_projection[1].in_features, 8)

    def test_control_original_forward_and_backprop_are_exact(self):
        torch.manual_seed(42)
        old = CompactAnchoredModel(copy.deepcopy(self.base), 8)
        old.encoder = DetailEncoder(old.encoder)
        m = self.model('control')
        for name, value in old.state_dict().items():
            torch.testing.assert_close(m.state_dict()[name], value, rtol=0, atol=0)
        with torch.no_grad():
            weight = torch.randn_like(old.delta_head[-1].weight) * .03
            old.delta_head[-1].weight.copy_(weight); m.delta_head[-1].weight.copy_(weight)
        h = torch.randn(3, 6); t = torch.tensor([30., 90., 365.]); e = torch.tensor([1, 2, 0])
        torch.manual_seed(99); a = old(h, self.b.base)
        torch.manual_seed(99); b = m(h, self.b)
        torch.testing.assert_close(a.log_amplitude, b.log_amplitude, rtol=0, atol=0)
        torch.testing.assert_close(a.shape, b.shape, rtol=0, atol=0)
        member_nll(a, t, e).backward(); member_nll(b, t, e).backward()
        grads = dict(m.named_parameters())
        for name, value in old.named_parameters():
            if value.grad is not None:
                torch.testing.assert_close(value.grad, grads[name].grad, rtol=0, atol=0)

    def test_all_candidates_get_gradient_without_baseline_gradient(self):
        h = torch.randn(3, 6); t = torch.tensor([30., 90., 365.]); e = torch.tensor([1, 2, 0])
        for arm in ARMS:
            m = self.model(arm)
            torch.nn.init.normal_(m.delta_head[-1].weight, std=.03)
            if arm.startswith('interaction'):
                m.encoder.interaction_gate.data.fill_(.2)
            member_nll(m(h, self.b), t, e).backward()
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.baseline.parameters()))
            target = (m.detail.base.lab if arm.startswith('lab_') else
                      m.encoder.attention if arm.startswith('interaction') else m.detail.base.fusion)
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in target.parameters() if p.grad is not None), 0)

    def test_numeric_missing_recovery_is_ignored(self):
        for kind in ('ple', 'periodic'):
            m = NumericLabEncoder(kind, [[-1., 1.]] * 132).eval()
            x = torch.randn(3, 22, 9); x[:, :, 6:] = 1; x[:, :, 7] = 0
            a = m(x); changed = x.clone(); changed[:, :, (3, 5)] = 999
            torch.testing.assert_close(a, m(changed), rtol=0, atol=0)
            x[:, :, 6] = 0; changed = x.clone(); changed[:, :, :6] = 999
            torch.testing.assert_close(m(x), m(changed), rtol=0, atol=0)

    def test_bins_use_only_observed_training(self):
        x = np.ones((10, 22, 9), np.float32); train = np.arange(10) < 5
        x[:5, :, :6] = np.arange(5)[:, None, None]; x[5:, :, :6] = 999
        a = fit_lab_bins(x, train); x[5:, :, :6] = -999
        self.assertEqual(a, fit_lab_bins(x, train)); self.assertEqual(a['bins'][0][-1], 4.)
        x[:5, :, 7] = 0
        b = fit_lab_bins(x, train); self.assertEqual(b['observed_training_counts'][3], 0)

    def test_ft_unmasked_cls_matches_official_forward_and_gradients(self):
        m = MaskedFTBlock().eval(); official = copy.deepcopy(m.backbone).eval()
        x = torch.randn(3, 7, 128, requires_grad=True); y = x.detach().clone().requires_grad_()
        a, b = m(x)[:, 0], official(y)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
        a.square().sum().backward(); b.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, atol=3e-6, rtol=3e-6)

    def test_ft_padding_permutation_and_empty(self):
        m = MaskedFTBlock().eval(); x = torch.randn(3, 7, 128)
        present = torch.tensor([[1, 1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 0], [0] * 7], dtype=torch.bool)
        a = m(x, ~present); changed = x.clone(); changed[~present] = 1000
        torch.testing.assert_close(a, m(changed, ~present), rtol=0, atol=0)
        p = torch.tensor([4, 2, 0, 6, 3, 5, 1]); b = m(x[:, p], ~present[:, p])
        torch.testing.assert_close(a[:, p], b, atol=2e-6, rtol=2e-6)
        self.assertTrue(torch.isfinite(a).all()); self.assertEqual(float(a[2].detach().abs().sum()), 0.)

    def test_interaction_gate_zero_preserves_original_latent(self):
        control = self.model('control').eval(); expected, _ = control.encoder(self.b.base)
        for arm in ('interaction_torch', 'interaction_ft'):
            m = self.model(arm).eval(); actual, _ = m.encoder(self.b)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_tabm_member_likelihood_and_probability(self):
        m = self.model('fusion_tabm').eval()
        torch.nn.init.normal_(m.delta_head[-1].weight, std=.02)
        out = m(torch.randn(3, 6), self.b); t = torch.tensor([30., 90., 365.]); e = torch.tensor([1, 2, 0])
        loss = member_nll(out, t, e); self.assertTrue(torch.isfinite(loss))
        a, k = out.log_amplitude.detach().numpy(), out.shape.detach().numpy()
        curve = probability_curve(a, k, device='cpu')
        manual = np.mean([probability_curve(a[:, i], k[:, i], device='cpu') for i in range(16)], axis=0)
        np.testing.assert_allclose(curve, manual, atol=1e-12)
        np.testing.assert_allclose(curve.sum(2), 1, atol=1e-10)
        self.assertTrue((np.diff(curve[:, :, 1:], axis=1) >= -1e-12).all())


if __name__ == '__main__':
    unittest.main()
