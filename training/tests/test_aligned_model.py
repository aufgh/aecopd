from __future__ import annotations

import unittest

import torch

from aecopd_training.aligned_model import AlignedModelConfig, TimeAlignedAecopdRiskModel
from test_model import synthetic_batch


def aligned_batch() -> dict[str, torch.Tensor]:
    batch = synthetic_batch()
    batch.update(
        {
            "text_embeddings": torch.randn(2, 4, 8),
            "text_role_ids": torch.tensor([[1, 2, 3, 0], [2, 4, 0, 0]]),
            "text_features": torch.randn(2, 4, 3),
            "text_attention_mask": torch.tensor(
                [[True, True, True, False], [True, True, False, False]]
            ),
            "text_window_ids": torch.tensor([[0, 2, 0, 0], [1, 0, 0, 0]]),
            "text_chunk_position_ids": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "text_document_ids": torch.tensor([[1, 1, 2, 0], [1, 2, 0, 0]]),
            "text_section_role_ids": torch.tensor([[1, 5, 15, 0], [14, 16, 0, 0]]),
            "text_modality_ids": torch.tensor([[1, 1, 2, 0], [3, 1, 0, 0]]),
            "text_section_position_ids": torch.tensor([[1, 2, 1, 0], [1, 1, 0, 0]]),
            "text_section_instance_ids": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "text_document_role_ids": torch.tensor([[1, 3], [2, 4]]),
            "text_document_window_ids": torch.tensor([[0, 0], [1, 0]]),
            "text_document_attention_mask": torch.tensor([[True, True], [True, True]]),
        }
    )
    return batch


def config() -> AlignedModelConfig:
    return AlignedModelConfig(
        concept_vocab_size=20,
        domain_vocab_size=10,
        window_vocab_size=18,
        unit_vocab_size=10,
        route_vocab_size=10,
        evidence_vocab_size=10,
        summary_vocab_size=20,
        static_category_sizes=(8, 12, 14, 16),
        hidden_size=32,
        attention_heads=4,
        text_embedding_dim=8,
        temporal_layers=1,
        patient_layers=1,
        dropout=0.0,
    )


class TimeAlignedAecopdRiskModelTest(unittest.TestCase):
    def test_structured_only_model_runs_without_any_text_tensors(self) -> None:
        structured_only_config = AlignedModelConfig(
            **{
                **config().__dict__,
                "use_text": False,
                "text_embedding_dim": 0,
            }
        )

        output = TimeAlignedAecopdRiskModel(structured_only_config).eval()(synthetic_batch())

        self.assertEqual(output.logits.shape, (2, 365, 3))
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_forward_is_finite_and_has_competing_risk_contract(self) -> None:
        output = TimeAlignedAecopdRiskModel(config()).eval()(aligned_batch())
        self.assertEqual(output.logits.shape, (2, 365, 3))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(
            torch.all(output.cumulative_incidence[:, 1:] >= output.cumulative_incidence[:, :-1])
        )
        self.assertTrue(torch.all(output.cumulative_incidence.sum(dim=-1) <= 1.00001))

    def test_cause_specific_heads_both_receive_gradients(self) -> None:
        specific_config = AlignedModelConfig(**{**config().__dict__, "cause_specific_heads": True})
        model = TimeAlignedAecopdRiskModel(specific_config)
        output = model(aligned_batch())
        output.logits.sum().backward()
        self.assertIsNotNone(model.aecopd_head)
        self.assertIsNotNone(model.death_head)
        assert model.aecopd_head is not None
        assert model.death_head is not None
        self.assertIsNotNone(model.aecopd_head.weight.grad)
        self.assertIsNotNone(model.death_head.weight.grad)

    def test_padding_values_do_not_change_predictions(self) -> None:
        model = TimeAlignedAecopdRiskModel(config()).eval()
        original = aligned_batch()
        changed = {name: value.clone() for name, value in original.items()}
        changed["text_embeddings"][0, 3] = 999.0
        changed["text_embeddings"][1, 2:] = -999.0
        changed["text_role_ids"][:, 3] = 5
        changed["text_window_ids"][:, 3] = 15
        changed["text_document_ids"][:, 3] = 2
        changed["text_section_role_ids"][:, 3] = 17
        changed["text_modality_ids"][:, 3] = 8
        changed["text_section_position_ids"][:, 3] = 63
        changed["text_section_instance_ids"][:, 3] = 31
        changed["timeline_features"][:, 2] = 999.0
        changed["timeline_concept_ids"][:, 2] = 19
        self.assertTrue(torch.equal(model(original).logits, model(changed).logits))

    def test_document_boundaries_change_hierarchical_text_representation(self) -> None:
        torch.manual_seed(7)
        model = TimeAlignedAecopdRiskModel(config()).eval()
        grouped = aligned_batch()
        regrouped = {name: value.clone() for name, value in grouped.items()}
        regrouped["text_document_ids"][0] = torch.tensor([1, 2, 2, 0])

        grouped_output = model(grouped)
        regrouped_output = model(regrouped)

        self.assertFalse(torch.equal(grouped_output.representation, regrouped_output.representation))


if __name__ == "__main__":
    unittest.main()
