from __future__ import annotations

import unittest

import torch

from aecopd_training.losses import (
    discrete_competing_risk_nll,
    patient_equal_competing_risk_nll,
)


class DiscreteCompetingRiskLossTest(unittest.TestCase):
    def test_correct_event_probability_has_lower_loss(self) -> None:
        event_types = torch.tensor([1], dtype=torch.long)
        time_days = torch.tensor([2.0])
        correct = torch.tensor([
            [[4.0, -4.0, -4.0], [0.0, 4.0, -4.0], [0.0, 0.0, 0.0]],
        ])
        wrong = torch.tensor([
            [[4.0, -4.0, -4.0], [0.0, -4.0, 4.0], [0.0, 0.0, 0.0]],
        ])

        correct_loss = discrete_competing_risk_nll(correct, event_types, time_days)
        wrong_loss = discrete_competing_risk_nll(wrong, event_types, time_days)

        self.assertLess(correct_loss.item(), wrong_loss.item())

    def test_patient_equal_loss_is_invariant_to_duplicate_admissions_within_patient(self) -> None:
        high_loss = torch.tensor([[[0.0, -4.0, 4.0]]])
        low_loss = torch.tensor([[[0.0, 4.0, -4.0]]])
        logits = torch.cat([high_loss, low_loss], dim=0)
        event_types = torch.tensor([1, 1], dtype=torch.long)
        time_days = torch.tensor([1.0, 1.0])
        patient_ids = torch.tensor([10, 20], dtype=torch.long)
        original = patient_equal_competing_risk_nll(
            logits, event_types, time_days, patient_ids
        )

        duplicated = patient_equal_competing_risk_nll(
            torch.cat([high_loss, high_loss, low_loss], dim=0),
            torch.tensor([1, 1, 1], dtype=torch.long),
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor([10, 10, 20], dtype=torch.long),
        )

        self.assertTrue(torch.allclose(original, duplicated))


if __name__ == "__main__":
    unittest.main()
