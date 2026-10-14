"""独立分支的缺失、初始化、梯度和患者历史因果性。"""

from __future__ import annotations

import unittest

import torch

from training.v5.independent_fusion import IndependentBatch, IndependentFusionModel
from training.v5.model import competing_nll


def _batch(*, text: bool) -> IndependentBatch:
    values = torch.zeros((1, 2, 3, 5))
    masks = torch.zeros_like(values)
    masks[:, :, 0, 0] = 1
    values[0, 0, 0, 0] = 60
    values[0, 1, 0, 0] = 65
    concept = torch.tensor([[[1, 2, 0], [1, 2, 0]]])
    kinds = torch.tensor([[[1, 2, 0], [1, 2, 0]]])
    present = torch.tensor([[[True, True, False], [True, True, False]]])
    vectors = torch.randn((2, 8)) if text else torch.zeros((0, 8))
    index = torch.tensor([[0, 0], [1, 1]]) if text else torch.zeros((0, 2), dtype=torch.long)
    return IndependentBatch(values, masks, concept, kinds, present, vectors, index,
                            torch.zeros((1, 2, 2)), torch.ones((1, 2), dtype=torch.bool),
                            torch.tensor([[20.0, 365.0]]), torch.tensor([[1, 0]]))


class IndependentFusionTests(unittest.TestCase):
    def test_missing_text_has_zero_gate_and_finite_loss(self) -> None:
        model = IndependentFusionModel(concept_count=3, type_count=2, field_count=2,
                                       text_width=8, width=32)
        batch = _batch(text=False)
        output = model(batch)
        self.assertTrue(torch.equal(output.gate, torch.zeros_like(output.gate)))
        loss = competing_nll(output, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.risk_head[0].weight.grad)

    def test_text_branch_has_gradient_without_structured_token(self) -> None:
        model = IndependentFusionModel(concept_count=3, type_count=2, field_count=2,
                                       text_width=8, width=32)
        batch = _batch(text=True)
        batch.token_present[:] = False
        loss = competing_nll(model(batch), batch)
        loss.backward()
        self.assertGreater(float(model.text_projection.weight.grad.abs().sum()), 0.0)

    def test_future_visit_cannot_change_previous_risk(self) -> None:
        model = IndependentFusionModel(concept_count=3, type_count=2, field_count=2,
                                       text_width=8, width=32).eval()
        batch = _batch(text=True)
        with torch.no_grad():
            before = model(batch).log_amplitude[0, 0].clone()
            batch.token_values[0, 1, 0, 0] = 999
            batch.snippet_vectors[1] *= -10
            after = model(batch).log_amplitude[0, 0]
        self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_text_width_does_not_change_shared_initialization(self) -> None:
        torch.manual_seed(2026)
        small = IndependentFusionModel(concept_count=3, type_count=2, field_count=2,
                                       text_width=8, width=32).state_dict()
        torch.manual_seed(2026)
        large = IndependentFusionModel(concept_count=3, type_count=2, field_count=2,
                                       text_width=16, width=32).state_dict()
        for name, tensor in small.items():
            if not name.startswith("text_projection."):
                self.assertTrue(torch.equal(tensor, large[name]), name)

    def test_reset_after_construction_aligns_dropout_for_structured_comparator(self) -> None:
        batch = _batch(text=False)
        torch.manual_seed(2026)
        small = IndependentFusionModel(concept_count=3, type_count=2, field_count=2,
                                       text_width=8, width=32).train()
        torch.manual_seed(2026)
        small_output = small(batch).log_amplitude
        torch.manual_seed(2026)
        large = IndependentFusionModel(concept_count=3, type_count=2, field_count=2,
                                       text_width=16, width=32).train()
        torch.manual_seed(2026)
        large_output = large(batch).log_amplitude
        self.assertTrue(torch.equal(small_output, large_output))


if __name__ == "__main__":
    unittest.main()
