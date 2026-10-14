"""V5 核心张量合同与时间泄漏防护测试。"""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import torch

from training.v5.data import (GROUP_WIDTHS, V5Batch, V5Data, collate,
                              diagnosis_groups, medication_classes)
from training.v5.model import V5Model, competing_nll, predict_cif


def batch_with_two_visits(*, text: bool) -> V5Batch:
    values = tuple(torch.zeros((1, 2, width), dtype=torch.float32) for width in GROUP_WIDTHS)
    masks = tuple(torch.ones_like(value) for value in values)
    values[0][0, 0, 0] = 60.0
    values[0][0, 1, 0] = 72.0
    vectors = torch.ones((1, 768)) if text else torch.zeros((0, 768))
    index = torch.tensor([[0, 2]], dtype=torch.int64) if text else torch.zeros((0, 2), dtype=torch.int64)
    return V5Batch(values, masks, torch.zeros((1, 2, 2)),
                   torch.ones((1, 2), dtype=torch.bool),
                   torch.tensor([[15.0, 365.0]]), torch.tensor([[1, 0]]),
                   vectors, index)


class V5ModelTests(unittest.TestCase):
    def test_medication_and_diagnosis_mapping_is_conservative(self) -> None:
        self.assertEqual(medication_classes("Ipratropium-Albuterol neb"), {"saba", "sama"})
        self.assertIn("inhaled_steroid", medication_classes("Fluticasone-Salmeterol diskus"))
        self.assertNotIn("inhaled_steroid", medication_classes("Fluticasone propionate nasal"))
        self.assertEqual(diagnosis_groups("I50.9", 10), {"heart_failure"})
        self.assertEqual(diagnosis_groups("51881", 9), {"acute_resp_failure"})

    def test_missing_text_gate_is_zero_and_loss_has_gradient(self) -> None:
        torch.manual_seed(7)
        model = V5Model(width=32)
        batch = batch_with_two_visits(text=False)
        result = model(batch)
        self.assertTrue(torch.equal(result.gate, torch.zeros_like(result.gate)))
        loss = competing_nll(result, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.risk_head[0].weight.grad)

    def test_future_visit_does_not_change_first_prediction(self) -> None:
        torch.manual_seed(8)
        model = V5Model(width=32).eval()
        batch = batch_with_two_visits(text=True)
        with torch.no_grad():
            before = model(batch).log_amplitude[0, 0].clone()
            batch.group_values[0][0, 1, 0] = 20.0
            after = model(batch).log_amplitude[0, 0]
        self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_competing_probabilities_sum_to_one(self) -> None:
        curve = predict_cif(np.array([[-1.5, -3.0], [-1.0, -2.0]]),
                            np.array([0.8, 1.2]), np.arange(366))
        self.assertEqual(curve.shape, (2, 366, 3))
        np.testing.assert_allclose(curve.sum(axis=2), 1.0, atol=1e-9)
        self.assertTrue(np.all(np.diff(curve[:, :, 1:], axis=1) >= -1e-12))

    def test_qwen_prefix_is_renormalized_and_full_width_runs(self) -> None:
        vectors = np.zeros((1, 2560), dtype=np.float16)
        vectors[0, :2] = (3, 4)
        base = V5Data(np.array([1]), np.array([10]), np.array(["train"]),
                      np.array([30], dtype=np.float32), np.array([1]),
                      tuple(np.zeros((1, width), dtype=np.float32) for width in GROUP_WIDTHS),
                      tuple(np.ones((1, width), dtype=np.float32) for width in GROUP_WIDTHS),
                      np.zeros((1, 2), dtype=np.float32), (np.array([0]),),
                      (((0, 0),),), vectors, 128, {})
        prefix_batch = collate(base, [base.sequences[0]], include_text=True)
        self.assertEqual(tuple(prefix_batch.snippet_vectors.shape), (1, 128))
        self.assertAlmostEqual(float(prefix_batch.snippet_vectors.norm()), 1.0, places=6)
        full_batch = collate(replace(base, text_width=2560),
                             [base.sequences[0]], include_text=True)
        self.assertEqual(tuple(V5Model(width=32, text_width=2560)(full_batch).log_amplitude.shape),
                         (1, 1, 2))

    def test_text_width_does_not_change_shared_initialization(self) -> None:
        torch.manual_seed(2026)
        narrow = V5Model(width=32, text_width=768).state_dict()
        torch.manual_seed(2026)
        full = V5Model(width=32, text_width=2560).state_dict()
        for name, tensor in narrow.items():
            if not name.startswith("text_projection."):
                self.assertTrue(torch.equal(tensor, full[name]), name)

    def test_balanced_text_branch_keeps_content_gradient(self) -> None:
        model = V5Model(width=32, balance_text=True)
        batch = batch_with_two_visits(text=True)
        loss = competing_nll(model(batch), batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.text_projection.weight.grad)
        self.assertGreater(float(model.text_projection.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
