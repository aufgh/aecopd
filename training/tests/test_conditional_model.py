from __future__ import annotations

import unittest

import torch

from aecopd_training.conditional_model import ConditionalIntervalAecopdModel
from aecopd_training.conditional_losses import conditional_patient_equal_loss
from test_aligned_model import aligned_batch, config


class ConditionalIntervalAecopdModelTests(unittest.TestCase):
    @staticmethod
    def _longitudinal_batch() -> dict[str, torch.Tensor]:
        batch = aligned_batch()
        batch.update(
            {
                "longitudinal_patient_indices": torch.tensor([0, 0]),
                "longitudinal_position_indices": torch.tensor([0, 1]),
                "longitudinal_gap_days": torch.tensor([0.0, 30.0]),
                "longitudinal_attention_mask": torch.tensor([[True, True]]),
            }
        )
        return batch

    def test_outputs_three_conditional_intervals_and_one_open_tail(self) -> None:
        model = ConditionalIntervalAecopdModel(
            config(), longitudinal_layers=1, max_admissions=8
        ).eval()
        batch = self._longitudinal_batch()

        output = model(batch)

        self.assertEqual(output.interval_logits.shape, (2, 3, 3))
        self.assertEqual(output.tail_logits.shape, (2, 3))
        self.assertTrue(torch.allclose(output.interval_probabilities.sum(-1), torch.ones(2, 3)))
        self.assertTrue(torch.allclose(output.tail_probabilities.sum(-1), torch.ones(2)))
        self.assertFalse(
            any("risk_head" in name for name, _ in model.named_parameters())
        )

    def test_loss_reaches_every_major_new_architecture_module(self) -> None:
        torch.manual_seed(2026)
        model = ConditionalIntervalAecopdModel(
            config(), longitudinal_layers=1, max_admissions=8
        ).train()
        output = model(self._longitudinal_batch())
        loss = conditional_patient_equal_loss(
            output.interval_logits,
            output.tail_logits,
            torch.tensor([[1, 0, 0], [0, 2, 0]]),
            torch.ones((2, 3), dtype=torch.bool),
            torch.tensor([0, 2]),
            torch.ones(2, dtype=torch.bool),
            torch.tensor([0, 0]),
        )
        loss.backward()

        for prefix in (
            "encoder.admission_model.",
            "encoder.admission_encoder.",
            "interval_head.",
            "tail_head.",
        ):
            gradients = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if name.startswith(prefix) and parameter.requires_grad
            ]
            self.assertTrue(gradients, prefix)
            self.assertTrue(
                any(
                    gradient is not None
                    and torch.isfinite(gradient).all()
                    and torch.count_nonzero(gradient) > 0
                    for gradient in gradients
                ),
                prefix,
            )


if __name__ == "__main__":
    unittest.main()
