from __future__ import annotations

import unittest

import torch

from aecopd_training.model import AecopdRiskModel, ModelConfig


def synthetic_batch() -> dict[str, torch.Tensor]:
    return {
        "timeline_concept_ids": torch.tensor([[3, 4, 0], [5, 0, 0]]),
        "timeline_domain_ids": torch.tensor([[3, 4, 0], [3, 0, 0]]),
        "timeline_window_ids": torch.tensor([[1, 2, 0], [16, 0, 0]]),
        "timeline_unit_ids": torch.tensor([[3, 2, 0], [2, 0, 0]]),
        "timeline_route_ids": torch.tensor([[2, 3, 0], [2, 0, 0]]),
        "timeline_evidence_ids": torch.tensor([[3, 4, 0], [5, 0, 0]]),
        "timeline_encoder_flags": torch.tensor([[False, True, False], [False, False, False]]),
        "timeline_features": torch.randn(2, 3, 25),
        "timeline_feature_masks": torch.ones(2, 3, 25, dtype=torch.bool),
        "timeline_attention_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "static_continuous": torch.randn(2, 16),
        "static_continuous_masks": torch.ones(2, 16, dtype=torch.bool),
        "static_categorical": torch.tensor([[3, 4, 5, 6], [4, 3, 4, 3]]),
        "static_binary": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        "global_summary_concept_ids": torch.tensor([[3, 4], [5, 0]]),
        "global_summary_features": torch.randn(2, 2, 11),
        "global_summary_feature_masks": torch.ones(2, 2, 11, dtype=torch.bool),
        "global_summary_attention_mask": torch.tensor([[True, True], [True, False]]),
    }


class AecopdRiskModelTest(unittest.TestCase):
    def test_forward_returns_valid_monotone_competing_risks(self) -> None:
        config = ModelConfig(
            concept_vocab_size=20,
            domain_vocab_size=10,
            window_vocab_size=18,
            unit_vocab_size=10,
            route_vocab_size=10,
            evidence_vocab_size=10,
            summary_vocab_size=20,
            static_category_sizes=(5, 12, 14, 16),
            hidden_size=32,
            attention_heads=4,
            temporal_layers=1,
            dropout=0.0,
            horizon_days=365,
        )
        model = AecopdRiskModel(config).eval()

        output = model(synthetic_batch())

        self.assertEqual(output.logits.shape, (2, 365, 3))
        self.assertEqual(output.cumulative_incidence.shape, (2, 365, 2))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(torch.all(output.cumulative_incidence[:, 1:] >= output.cumulative_incidence[:, :-1]))
        self.assertTrue(torch.all(output.cumulative_incidence.sum(dim=-1) <= 1.00001))
        self.assertEqual(output.risk_at(90).shape, (2, 2))
        self.assertTrue(torch.equal(output.risk_at(365), output.cumulative_incidence[:, 364]))
        self.assertTrue(torch.all(output.risk_at(90) < output.risk_at(365)))
        self.assertTrue(torch.all(output.survival[:, -1] > 0.05))

    def test_padding_values_do_not_change_predictions(self) -> None:
        config = ModelConfig(
            concept_vocab_size=20,
            domain_vocab_size=10,
            window_vocab_size=18,
            unit_vocab_size=10,
            route_vocab_size=10,
            evidence_vocab_size=10,
            summary_vocab_size=20,
            static_category_sizes=(5, 12, 14, 16),
            hidden_size=32,
            attention_heads=4,
            temporal_layers=1,
            dropout=0.0,
        )
        model = AecopdRiskModel(config).eval()
        original = synthetic_batch()
        changed = {name: value.clone() for name, value in original.items()}
        changed["timeline_features"][:, 2] = 999.0
        changed["timeline_concept_ids"][:, 2] = 19
        changed["global_summary_features"][1, 1] = -999.0
        changed["global_summary_concept_ids"][1, 1] = 19

        first = model(original).logits
        second = model(changed).logits

        self.assertTrue(torch.equal(first, second))

    def test_text_document_padding_does_not_change_predictions(self) -> None:
        config = ModelConfig(
            concept_vocab_size=20,
            domain_vocab_size=10,
            window_vocab_size=18,
            unit_vocab_size=10,
            route_vocab_size=10,
            evidence_vocab_size=10,
            summary_vocab_size=20,
            static_category_sizes=(5, 12, 14, 16),
            hidden_size=32,
            attention_heads=4,
            temporal_layers=1,
            dropout=0.0,
            text_embedding_dim=8,
        )
        model = AecopdRiskModel(config).eval()
        original = synthetic_batch()
        original.update({
            "text_embeddings": torch.randn(2, 3, 8),
            "text_role_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
            "text_features": torch.randn(2, 3, 3),
            "text_attention_mask": torch.tensor([[True, True, False], [True, False, False]]),
        })
        changed = {name: value.clone() for name, value in original.items()}
        changed["text_embeddings"][0, 2] = 999.0
        changed["text_embeddings"][1, 1:] = -999.0
        changed["text_role_ids"][:, 2] = 5
        changed["text_features"][:, 2] = 88.0

        first = model(original).logits
        second = model(changed).logits

        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
