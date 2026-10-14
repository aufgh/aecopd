from __future__ import annotations

import unittest

import torch

from aecopd_training.conditional_losses import (
    conditional_admission_equal_loss,
    conditional_patient_equal_loss,
)


class ConditionalPatientEqualLossTests(unittest.TestCase):
    def _loss(
        self,
        interval_logits: torch.Tensor,
        tail_logits: torch.Tensor,
        *,
        interval_labels: torch.Tensor | None = None,
        interval_mask: torch.Tensor | None = None,
        tail_labels: torch.Tensor | None = None,
        tail_mask: torch.Tensor | None = None,
        patient_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = interval_logits.shape[0]
        return conditional_patient_equal_loss(
            interval_logits,
            tail_logits,
            interval_labels
            if interval_labels is not None
            else torch.zeros((batch_size, 3), dtype=torch.long),
            interval_mask
            if interval_mask is not None
            else torch.ones((batch_size, 3), dtype=torch.bool),
            tail_labels
            if tail_labels is not None
            else torch.zeros(batch_size, dtype=torch.long),
            tail_mask
            if tail_mask is not None
            else torch.ones(batch_size, dtype=torch.bool),
            patient_ids
            if patient_ids is not None
            else torch.arange(batch_size, dtype=torch.long),
        )

    def test_correct_interval_and_tail_logits_have_lower_loss(self) -> None:
        labels = torch.tensor([[1, 2, 0]], dtype=torch.long)
        correct_interval = torch.full((1, 3, 3), -4.0)
        correct_interval[0, torch.arange(3), labels[0]] = 4.0
        wrong_interval = -correct_interval
        correct_tail = torch.tensor([[-4.0, 4.0, -4.0]])
        wrong_tail = torch.tensor([[4.0, -4.0, -4.0]])

        correct = self._loss(
            correct_interval,
            correct_tail,
            interval_labels=labels,
            tail_labels=torch.tensor([1]),
        )
        wrong = self._loss(
            wrong_interval,
            wrong_tail,
            interval_labels=labels,
            tail_labels=torch.tensor([1]),
        )

        self.assertLess(correct.item(), wrong.item())

    def test_masked_interval_and_tail_logits_do_not_change_loss(self) -> None:
        interval_logits = torch.zeros((1, 3, 3))
        tail_logits = torch.zeros((1, 3))
        interval_mask = torch.tensor([[True, False, False]])
        tail_mask = torch.tensor([False])
        original = self._loss(
            interval_logits,
            tail_logits,
            interval_mask=interval_mask,
            tail_mask=tail_mask,
        )
        changed_interval = interval_logits.clone()
        changed_interval[:, 1:] = torch.tensor([1000.0, -1000.0, 0.0])
        changed_tail = torch.tensor([[1000.0, -1000.0, 0.0]])
        changed = self._loss(
            changed_interval,
            changed_tail,
            interval_mask=interval_mask,
            tail_mask=tail_mask,
        )

        self.assertTrue(torch.allclose(original, changed))

    def test_duplicate_admission_does_not_upweight_a_patient(self) -> None:
        high_interval = torch.tensor([[[0.0, -4.0, 4.0]] * 3])
        low_interval = torch.tensor([[[4.0, -4.0, -4.0]] * 3])
        high_tail = torch.tensor([[0.0, -4.0, 4.0]])
        low_tail = torch.tensor([[4.0, -4.0, -4.0]])
        original = self._loss(
            torch.cat((high_interval, low_interval)),
            torch.cat((high_tail, low_tail)),
            patient_ids=torch.tensor([10, 20]),
        )
        duplicated = self._loss(
            torch.cat((high_interval, high_interval, low_interval)),
            torch.cat((high_tail, high_tail, low_tail)),
            patient_ids=torch.tensor([10, 10, 20]),
        )

        self.assertTrue(torch.allclose(original, duplicated))

    def test_admission_equal_loss_weights_every_index_discharge_equally(self) -> None:
        high_interval = torch.tensor([[[0.0, -4.0, 4.0]] * 3])
        low_interval = torch.tensor([[[4.0, -4.0, -4.0]] * 3])
        high_tail = torch.tensor([[0.0, -4.0, 4.0]])
        low_tail = torch.tensor([[4.0, -4.0, -4.0]])

        def admission_loss(interval: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
            batch_size = interval.shape[0]
            return conditional_admission_equal_loss(
                interval,
                tail,
                torch.zeros((batch_size, 3), dtype=torch.long),
                torch.ones((batch_size, 3), dtype=torch.bool),
                torch.zeros(batch_size, dtype=torch.long),
                torch.ones(batch_size, dtype=torch.bool),
            )

        original = admission_loss(
            torch.cat((high_interval, low_interval)),
            torch.cat((high_tail, low_tail)),
        )
        duplicated = admission_loss(
            torch.cat((high_interval, high_interval, low_interval)),
            torch.cat((high_tail, high_tail, low_tail)),
        )

        self.assertGreater(duplicated.item(), original.item())


if __name__ == "__main__":
    unittest.main()
