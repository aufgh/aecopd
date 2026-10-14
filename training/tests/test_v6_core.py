"""历史隔离、零初始化、字段池化和连续概率；合成输入，不训练真实队列。"""
import unittest
from dataclasses import replace
import numpy as np
import torch
from training.v6.model import AnchoredModel, HistoryBaseline, ClinicalBatch
from training.v6.risk import competing_nll, predict_cif


def batch(text_width=8, device='cpu'):
    return ClinicalBatch(torch.randn(3, 14, device=device),
                         torch.ones(3, 14, dtype=torch.bool, device=device),
                         torch.cat((torch.randn(3, 22, 6, device=device), torch.ones(3, 22, 3, device=device)), 2),
                         torch.ones(3, 7, device=device),
                         torch.ones(3, 7, dtype=torch.bool, device=device),
                         torch.randn(6, text_width, device=device),
                         torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 3], [1, 3]], device=device))


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initial_baseline_and_two_step_gradient(self):
        torch.manual_seed(17)
        base = HistoryBaseline()
        model = AnchoredModel(base, 8, width=16)
        h, b = torch.randn(3, 6), batch()
        initial, expected = model(h, b), base(h)
        torch.testing.assert_close(initial.log_amplitude, expected.log_amplitude, rtol=0, atol=0)
        torch.testing.assert_close(initial.shape, expected.shape, rtol=0, atol=0)
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        t, e = torch.tensor([30., 90., 365.]), torch.tensor([1, 2, 0])
        loss = competing_nll(initial, t, e)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        self.assertGreater(model.delta_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.encoder.lab[0].weight.grad.abs().sum().item(), 0)
        opt.step(); opt.zero_grad()
        competing_nll(model(h, b), t, e).backward()
        self.assertGreater(model.encoder.lab[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in base.parameters()))

    def test_history_cannot_change_delta(self):
        model = AnchoredModel(HistoryBaseline(), 8, width=16).eval()
        torch.nn.init.normal_(model.delta_head[-1].weight, std=.01)
        b = batch()
        a = model(torch.zeros(3, 6), b)
        changed = model(torch.ones(3, 6), b)
        torch.testing.assert_close(a.delta, changed.delta, rtol=0, atol=0)
        self.assertFalse(torch.equal(a.log_amplitude, changed.log_amplitude))

    def test_text_permutation_pmh_and_empty(self):
        model = AnchoredModel(HistoryBaseline(), 8, width=16).eval()
        b = batch()
        expected, _ = model.encoder(b)
        order = torch.tensor([5, 4, 2, 1, 3, 0])
        shuffled = replace(b, snippet_vectors=b.snippet_vectors[order], snippet_index=b.snippet_index[order])
        actual, _ = model.encoder(shuffled)
        torch.testing.assert_close(expected, actual)
        changed = b.snippet_vectors.clone()
        changed[b.snippet_index[:, 1] == 1] = 10000
        masked, _ = model.encoder(b, mask_pmh=True)
        masked_changed, _ = model.encoder(replace(b, snippet_vectors=changed), mask_pmh=True)
        torch.testing.assert_close(masked, masked_changed, rtol=0, atol=0)
        empty = replace(b, current_masks=torch.zeros_like(b.current_masks),
                        lab_axis_inputs=torch.zeros_like(b.lab_axis_inputs),
                        med_source_mask=torch.zeros_like(b.med_source_mask),
                        snippet_vectors=b.snippet_vectors[:0], snippet_index=b.snippet_index[:0])
        result, branches = model.encoder(empty)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(all(branches[k].abs().sum() == 0 for k in ('structured', 'lab', 'medication', 'text')))

    def test_nontext_initialization_invariant(self):
        states = []
        for dimension in (8, 20):
            torch.manual_seed(33)
            model = AnchoredModel(HistoryBaseline(), dimension, width=16)
            states.append({k: v for k, v in model.state_dict().items() if 'text_projection' not in k})
        self.assertEqual(states[0].keys(), states[1].keys())
        for k in states[0]:
            torch.testing.assert_close(states[0][k], states[1][k], rtol=0, atol=0)

    def test_probability_conservation_and_common_shape(self):
        a = np.array([[-2., -3.], [-1., -4.]])
        days = np.array([0., .001, 7., 30., 90., 365.])
        k = np.array([[.5, .5], [1.2, 1.2]])
        curve = predict_cif(a, k, days)
        np.testing.assert_allclose(curve.sum(2), 1, atol=2e-12)
        self.assertTrue((curve >= 0).all() and (curve <= 1).all())
        self.assertTrue((np.diff(curve[:, :, 0], axis=1) <= 1e-12).all())
        self.assertTrue((np.diff(curve[:, :, 1:], axis=1) >= -1e-12).all())
        H = np.exp(a).sum(1)[:, None] * (days[None, :] / 365.) ** k[:, :1]
        np.testing.assert_allclose(curve[:, :, 0], np.exp(-H), atol=2e-12)
        share = np.exp(a) / np.exp(a).sum(1)[:, None]
        np.testing.assert_allclose(curve[:, :, 1:], (1 - np.exp(-H))[:, :, None] * share[:, None, :], atol=2e-12)

    def test_patient_shape_convergence_and_validation(self):
        a = np.array([[-1., -2.], [-2., -3.]])
        k = np.array([[.25, 1.4], [.7, 2.]])
        days = np.array([0., 7., 90., 365.])
        coarse = predict_cif(a, k, days, step_days=.25)
        fine = predict_cif(a, k, days, step_days=.125)
        np.testing.assert_allclose(coarse, fine, atol=1e-4, rtol=0)
        for invalid in (np.array([]), np.array([1., 10.]), np.array([0., 366.])):
            with self.assertRaises(ValueError):
                predict_cif(a, k, invalid)
        model = AnchoredModel(HistoryBaseline(), 8, width=16)
        with self.assertRaises(ValueError):
            competing_nll(model(torch.zeros(3, 6), batch()), torch.tensor([1., 2., 3.]), torch.tensor([1, 3, 0]))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_forward_backward(self):
        model = AnchoredModel(HistoryBaseline(), 8, width=16).cuda()
        output = model(torch.zeros(3, 6, device='cuda'), batch(device='cuda'))
        loss = competing_nll(output, torch.tensor([0., 10., 365.], device='cuda'),
                             torch.tensor([1, 2, 0], device='cuda'))
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == '__main__':
    unittest.main()
