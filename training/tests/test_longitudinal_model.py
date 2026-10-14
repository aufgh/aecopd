from __future__ import annotations

import unittest

import torch

from aecopd_training.longitudinal_model import LongitudinalAecopdRiskModel
from test_aligned_model import aligned_batch, config


class LongitudinalAecopdRiskModelTests(unittest.TestCase):
    def test_future_admission_cannot_change_earlier_prediction(self) -> None:
        torch.manual_seed(11)
        model = LongitudinalAecopdRiskModel(
            config(),
            admission_layers=1,
            max_admissions=8,
        ).eval()
        original = aligned_batch()
        original.update(
            {
                "longitudinal_patient_indices": torch.tensor([0, 0]),
                "longitudinal_position_indices": torch.tensor([0, 1]),
                "longitudinal_gap_days": torch.tensor([0.0, 30.0]),
                "longitudinal_attention_mask": torch.tensor([[True, True]]),
            }
        )
        changed = {name: value.clone() for name, value in original.items()}
        changed["timeline_features"][1] += 100.0
        changed["text_embeddings"][1] -= 100.0
        changed["static_continuous"][1] += 100.0

        first = model(original)
        second = model(changed)

        self.assertTrue(torch.equal(first.logits[0], second.logits[0]))
        self.assertFalse(torch.equal(first.logits[1], second.logits[1]))


if __name__ == "__main__":
    unittest.main()
