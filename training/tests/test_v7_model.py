"""V7 掩码、排列、初始化和冻结边界；真实来源/GPU由入口另验。"""
import unittest
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from training.v6.risk import RiskOutput
from training.v7.model import V7Model, V7Batch, MultiQueryPool


class ToyBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.amplitude = nn.Linear(34, 2)
        self.raw_shape = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        return RiskOutput(self.amplitude(x).clamp(-10, 6), (F.softplus(self.raw_shape) + .1).expand(len(x), 2))


def toy_batch():
    torch.manual_seed(5)
    labs = torch.randn(4, 36, 9); labs[:, :, 6] = 1
    return V7Batch(torch.randn(4, 34), labs, torch.randn(4, 16), torch.randn(4, 38, 11),
                   torch.ones(4, 38, dtype=torch.bool), torch.ones(4, 7, 4), torch.randn(4, 5),
                   torch.ones(4, 5, dtype=torch.bool), torch.randn(4, 3), torch.ones(4, 3, dtype=torch.bool),
                   torch.randn(6, 768), torch.tensor([[0, 0], [0, 0], [0, 1], [1, 2], [2, 1], [3, 3]]),
                   torch.tensor([0, 1, 2, 0, 0, 0]), torch.randn(4, 2, 1536),
                   torch.ones(4, 2, dtype=torch.bool), torch.tensor([0, 1]), torch.tensor([[2, 3], [3, 0]]),
                   torch.tensor([[True, True], [True, False]]), torch.tensor([[10., 50.], [20., 0.]]))


class V7ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_pool_padding_and_all_missing(self):
        pool = MultiQueryPool().eval()
        tokens = torch.randn(2, 3, 256)
        present = torch.tensor([[True, False, False], [False, False, False]])
        expected = pool(tokens, present)[0]
        altered = torch.where(present[:, :, None], tokens, torch.full_like(tokens, 1e7))
        torch.testing.assert_close(expected, pool(altered, present)[0], rtol=0, atol=0)
        self.assertEqual(expected[1].abs().sum(), 0)

    def test_time_decay_order_and_no_history(self):
        pool = MultiQueryPool(decay=True).eval()
        bias = pool.time_bias(torch.tensor([[1., 5., 100.]]))
        self.assertTrue((bias[:, :, 0] > bias[:, :, 1]).all())
        self.assertTrue((bias[:, :, 1] > bias[:, :, 2]).all())
        empty, _ = pool(torch.randn(1, 3, 256), torch.zeros(1, 3, dtype=torch.bool), torch.ones(1, 3))
        self.assertEqual(empty.abs().sum(), 0)

    def test_initialization_groups_and_frozen_baseline(self):
        base = ToyBaseline()
        first, second = [V7Model(base, torch.ones(38, 10)).eval() for _ in range(2)]
        for p, q in zip(first.parameters(), second.parameters(), strict=True):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        b = toy_batch()
        out = first(b)
        expected = base(b.context[b.current])
        torch.testing.assert_close(out.log_amplitude, expected.log_amplitude, rtol=0, atol=0)
        torch.testing.assert_close(out.shape, expected.shape, rtol=0, atol=0)
        groups, _ = first.optimizer_groups()
        self.assertEqual([g['lr'] for g in groups], [1e-4, 3e-4])
        out.log_amplitude.sum().backward()
        self.assertTrue(all(p.grad is None for p in first.baseline.parameters()))

    def test_text_permutation_and_empty_modalities(self):
        model = V7Model(ToyBaseline(), torch.ones(38, 10)).eval()
        nn.init.normal_(model.delta_head[-1].weight, std=.02)
        b = toy_batch()
        expected = model(b).log_amplitude
        order = torch.tensor([1, 0, 2, 3, 4, 5])
        altered = replace(b, text_vectors=b.text_vectors[order], text_index=b.text_index[order])
        torch.testing.assert_close(expected, model(altered).log_amplitude, rtol=1e-5, atol=1e-6)
        empty = replace(b, text_vectors=b.text_vectors[:0], text_index=b.text_index[:0], text_slot=b.text_slot[:0],
                        med_present=torch.zeros_like(b.med_present), rad_present=torch.zeros_like(b.rad_present),
                        past_present=torch.zeros_like(b.past_present), labs=torch.zeros_like(b.labs))
        out = model(empty)
        self.assertTrue(torch.isfinite(out.log_amplitude).all())
        contaminated = replace(empty, radiology=torch.full_like(empty.radiology, 1e5),
                               medications=torch.full_like(empty.medications, 1e5))
        torch.testing.assert_close(out.log_amplitude, model(contaminated).log_amplitude, rtol=0, atol=0)

    def test_history_padding_and_real_history_sensitivity(self):
        model = V7Model(ToyBaseline(), torch.ones(38, 10)).eval()
        nn.init.normal_(model.delta_head[-1].weight, std=.02)
        b = toy_batch()
        expected = model(b).log_amplitude
        altered = replace(b, past=b.past.clone(), gaps=b.gaps.clone())
        altered.past[~b.past_present] = 3
        altered.gaps[~b.past_present] = 1e6
        torch.testing.assert_close(expected, model(altered).log_amplitude, rtol=1e-5, atol=1e-6)
        absent = replace(b, past_present=torch.zeros_like(b.past_present))
        self.assertGreater(float((expected - model(absent).log_amplitude).abs().max().detach()), 1e-5)

    def test_field_compression_on_risk_gradient_path(self):
        model = V7Model(ToyBaseline(), torch.ones(38, 10)).eval()
        nn.init.normal_(model.delta_head[-1].weight, std=.02)
        model(toy_batch()).log_amplitude.sum().backward()
        for i, pool in enumerate(model.field_pool):
            for name, p in pool.project.named_parameters():
                self.assertIsNotNone(p.grad, f'field {i} {name} disconnected')
                self.assertGreater(float(p.grad.abs().sum()), 0, f'field {i} {name} has zero gradient')


if __name__ == '__main__':
    unittest.main()
