"""原始事件模型的空序列、梯度和跨住院因果性。"""

from __future__ import annotations

import unittest

import torch

from training.tests.test_v5_independent_fusion import _batch
from training.v5.event_data import EventBatch
from training.v5.event_model import TemporalFusionModel
from training.v5.model import competing_nll


class TemporalFusionTests(unittest.TestCase):
    def _model(self) -> TemporalFusionModel:
        return TemporalFusionModel(concept_count=3, type_count=2, field_count=2,
                                   text_width=8, width=32, event_width=16)

    def test_irregular_events_receive_gradients(self) -> None:
        batch = EventBatch(_batch(text=False),
                           torch.tensor([[1.0, 0.1, 0.1, 1.0],
                                         [2.0, 0.4, 0.3, 0.7],
                                         [0.0, 0.2, 0.2, 0.8]]),
                           torch.tensor([1, 1, 9]), torch.tensor([[2, 1]]))
        model = self._model()
        loss = competing_nll(model(batch), batch.base)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.event_gru.weight_ih_l0.grad.abs().sum()), 0.0)

    def test_missing_events_and_future_visit_are_safe(self) -> None:
        batch = EventBatch(_batch(text=False), torch.tensor([[0.0, 0.2, 0.2, 0.8]]),
                           torch.tensor([9]), torch.tensor([[0, 1]]))
        model = self._model().eval()
        with torch.no_grad():
            first = model(batch).log_amplitude[0, 0].clone()
            batch.event_values[0, 1] = 5.0
            second = model(batch).log_amplitude[0, 0]
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_count_mismatch_rejected(self) -> None:
        batch = EventBatch(_batch(text=False), torch.zeros((1, 4)),
                           torch.tensor([1]), torch.tensor([[2, 0]]))
        with self.assertRaisesRegex(ValueError, "offsets"):
            self._model()(batch)


if __name__ == "__main__":
    unittest.main()
