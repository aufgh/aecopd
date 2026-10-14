"""事件集合的排列不变性、真实时间、初始化和因果性回归测试。"""

from __future__ import annotations

import unittest

import torch

from training.tests.test_v5_independent_fusion import _batch
from training.v5.event_data import EventBatch
from training.v5.event_model import TemporalFusionModel
from training.v5.event_set_model import EventSetFusionModel
from training.v5.model import competing_nll


class EventSetTests(unittest.TestCase):
    def test_new_vocabulary_uses_explicit_event_type(self) -> None:
        model = EventSetFusionModel(concept_count=3, type_count=2, field_count=2,
                                    text_width=8, width=32, event_width=16,
                                    event_concept_count=32).eval()
        batch = EventBatch(_batch(text=False), torch.tensor([[1., .1, 0., 0.], [0., .2, 0., 0.]]),
                           torch.tensor([31, 1]), torch.tensor([[1, 1]]),
                           event_is_lab=torch.tensor([True, False]),
                           event_numeric_mask=torch.tensor([True, False]))
        with torch.no_grad():
            before = model(batch).log_amplitude.clone()
            batch.event_values[1, 0] = 999.
            after = model(batch).log_amplitude
        self.assertTrue(torch.equal(before, after))
        with self.assertRaisesRegex(ValueError, "explicit"):
            model(EventBatch(batch.base, batch.event_values, batch.event_concepts, batch.event_counts))

    def _model(self) -> EventSetFusionModel:
        return EventSetFusionModel(concept_count=3, type_count=2, field_count=2,
                                   text_width=8, width=32, event_width=16)

    def _events(self) -> EventBatch:
        return EventBatch(_batch(text=True),
                          torch.tensor([[1., .1, .1, 1.], [2., .1, 0., 1.],
                                        [3., .5, .4, .6], [0., .2, .2, .8]]),
                          torch.tensor([1, 2, 1, 9]), torch.tensor([[3, 1]]))

    def test_reordering_records_preserves_predictions(self) -> None:
        model, batch = self._model().eval(), self._events()
        for order in ([1, 0, 2, 3], [2, 0, 1, 3]):
            permutation = torch.tensor(order)
            other = EventBatch(batch.base, batch.event_values[permutation],
                               batch.event_concepts[permutation], batch.event_counts)
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(batch).log_amplitude,
                                               model(other).log_amplitude, atol=1e-5, rtol=1e-5))

    def test_gap_discharge_and_medication_numeric_value_are_ignored(self) -> None:
        model, batch = self._model().eval(), self._events()
        with torch.no_grad():
            before = model(batch).log_amplitude
            batch.event_values[:, 2:] = 999
            batch.event_values[-1, 0] = 999
            after = model(batch).log_amplitude
        self.assertTrue(torch.equal(before, after))

    def test_real_time_and_numeric_value_have_gradients(self) -> None:
        model, batch = self._model(), self._events()
        loss = competing_nll(model(batch), batch.base)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for module in (model.time_embedding, model.numeric_embedding):
            self.assertGreater(float(module[0].weight.grad.abs().sum()), 0.)

    def test_future_events_cannot_change_previous_risk(self) -> None:
        model, batch = self._model().eval(), self._events()
        with torch.no_grad():
            before = model(batch).log_amplitude[0, 0].clone()
            batch.event_values[-1, 1] = 999
            after = model(batch).log_amplitude[0, 0]
        self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_empty_event_stays_and_bad_counts(self) -> None:
        model = self._model()
        empty = EventBatch(_batch(text=False), torch.empty((0, 4)),
                           torch.empty(0, dtype=torch.long), torch.zeros((1, 2), dtype=torch.long))
        self.assertTrue(torch.isfinite(model(empty).log_amplitude).all())
        bad = self._events()
        bad.event_counts[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "offsets"):
            model(bad)

    def test_retained_shared_weights_match_gru_initialization(self) -> None:
        arguments = dict(concept_count=3, type_count=2, field_count=2, text_width=8, width=32)
        torch.manual_seed(2026)
        original = TemporalFusionModel(**arguments).state_dict()
        torch.manual_seed(2026)
        candidate = EventSetFusionModel(**arguments).state_dict()
        for name, value in original.items():
            if not name.startswith("event_") or name.startswith("event_gate."):
                self.assertTrue(torch.equal(value, candidate[name]), name)


if __name__ == "__main__":
    unittest.main()
