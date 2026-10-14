from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np

from baselines.incremental_evidence_chain import (
    FeatureBlocks,
    IncrementalEvidenceConfig,
    RetainedModel,
    StagePrediction,
    advance_retained_model,
    compose_competing_probabilities,
    fit_binary_logit,
    fit_stage,
    paired_patient_bootstrap_auc_delta,
    stage_gate_exclusion_reason,
    validate_feature_split_contract,
)
from baselines.shared_contract import SplitContract


class IncrementalEvidenceChainTests(unittest.TestCase):
    def test_competing_probabilities_are_valid_and_preserve_aecopd_ranking(self) -> None:
        aecopd_logits = np.array([-2.0, 0.0, 2.0])
        first = compose_competing_probabilities(aecopd_logits, np.array([-5.0, 0.0, 5.0]))
        second = compose_competing_probabilities(aecopd_logits, np.array([5.0, 0.0, -5.0]))

        np.testing.assert_allclose(first.sum(axis=1), 1.0)
        np.testing.assert_allclose(second.sum(axis=1), 1.0)
        np.testing.assert_allclose(first[:, 1], second[:, 1])
        self.assertTrue(np.all(np.diff(first[:, 1]) > 0.0))

    def test_anchor_logit_enforces_nonnegative_prior_count_direction(self) -> None:
        features = np.arange(8, dtype=np.float64).reshape(-1, 1)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        model = fit_binary_logit(
            features,
            labels,
            l2=1e-8,
            nonnegative_first_coefficient=True,
        )
        scores = model.decision_function(features)

        self.assertGreater(model.coefficients[0], 0.0)
        self.assertTrue(np.all(np.diff(scores) > 0.0))

    def test_paired_cluster_bootstrap_returns_zero_for_identical_predictions(self) -> None:
        probabilities = np.array(
            [
                [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.3, 0.1]],
                [[0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.4, 0.5, 0.1]],
                [[0.4, 0.5, 0.1], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1]],
                [[0.2, 0.7, 0.1], [0.1, 0.8, 0.1], [0.05, 0.9, 0.05]],
            ],
            dtype=np.float64,
        )
        labels = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]])
        mask = np.ones((4, 3), dtype=bool)
        subjects = np.array([10, 10, 20, 30])

        result = paired_patient_bootstrap_auc_delta(
            probabilities,
            probabilities,
            labels,
            mask,
            subjects,
            n_resamples=50,
            confidence=0.95,
            seed=2026,
        )

        self.assertEqual(result["mean"]["point"], 0.0)
        self.assertEqual(result["mean"]["lower"], 0.0)
        self.assertEqual(result["mean"]["upper"], 0.0)

    def test_architecture_selection_refuses_the_repeatedly_inspected_test_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "restricted to validation"):
            IncrementalEvidenceConfig(
                data_dir=Path("model_v3"),
                artifacts_path=Path("artifacts.json"),
                split=SplitContract(Path("index.csv.gz")),
                eval_split="test",
            )

    def test_text_pca_budget_cannot_drop_a_document_role(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 4"):
            IncrementalEvidenceConfig(
                data_dir=Path("model_v3"),
                artifacts_path=Path("artifacts.json"),
                split=SplitContract(Path("index.csv.gz")),
                text_pca_dim=3,
            )

    def test_constant_binary_label_uses_finite_smoothed_fallback(self) -> None:
        features = np.arange(6, dtype=np.float64).reshape(-1, 1)
        model = fit_binary_logit(features, np.ones(6), l2=0.01)

        self.assertTrue(model.converged)
        np.testing.assert_array_equal(model.coefficients, np.zeros(1))
        self.assertTrue(np.isfinite(model.decision_function(features)).all())

    def test_feature_rows_must_match_authoritative_split_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "index.csv.gz"
            with gzip.open(index_path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("subject_id", "hadm_id", "data_split"),
                )
                writer.writeheader()
                writer.writerow(
                    {"subject_id": 10, "hadm_id": 100, "data_split": "train"}
                )
            blocks = FeatureBlocks(
                hadm_ids=np.array([100]),
                subject_ids=np.array([10]),
                splits=np.array(["validation"]),
                blocks={"anchor": np.zeros((1, 1))},
                feature_names={"anchor": ("prior",)},
            )

            with self.assertRaisesRegex(ValueError, "disagree"):
                validate_feature_split_contract(blocks, SplitContract(index_path))

    def test_all_aecopd_risk_set_has_safe_empty_death_head(self) -> None:
        blocks = FeatureBlocks(
            hadm_ids=np.arange(5),
            subject_ids=np.arange(5),
            splits=np.array(["train", "train", "train", "validation", "validation"]),
            blocks={"anchor": np.arange(5, dtype=np.float64).reshape(-1, 1)},
            feature_names={"anchor": ("prior",)},
        )
        prediction = fit_stage(
            blocks,
            np.ones((5, 3), dtype=np.int64),
            np.ones((5, 3), dtype=bool),
            np.ones(5, dtype=np.int64),
            np.ones(5, dtype=bool),
            name="B0_prior_count",
            label="anchor",
            blocks=("anchor",),
            residual_anchor=False,
            l2=1e-8,
        )

        self.assertTrue(np.isfinite(prediction.interval_probabilities).all())
        np.testing.assert_allclose(prediction.interval_probabilities.sum(axis=2), 1.0)
        self.assertLess(prediction.converged_heads, prediction.total_heads)
        self.assertEqual(
            stage_gate_exclusion_reason(
                prediction,
                {"days_0_30": 0.7, "days_31_90": 0.7, "days_91_365": 0.7},
            ),
            "not_all_heads_converged_or_estimable",
        )

    def test_empty_training_risk_set_is_marked_not_estimable(self) -> None:
        blocks = FeatureBlocks(
            hadm_ids=np.arange(5),
            subject_ids=np.arange(5),
            splits=np.array(["train", "train", "train", "validation", "validation"]),
            blocks={"anchor": np.arange(5, dtype=np.float64).reshape(-1, 1)},
            feature_names={"anchor": ("prior",)},
        )
        interval_mask = np.ones((5, 3), dtype=bool)
        interval_mask[:3, 0] = False
        prediction = fit_stage(
            blocks,
            np.zeros((5, 3), dtype=np.int64),
            interval_mask,
            np.zeros(5, dtype=np.int64),
            np.ones(5, dtype=bool),
            name="B0_prior_count",
            label="anchor",
            blocks=("anchor",),
            residual_anchor=False,
            l2=1e-8,
        )

        np.testing.assert_allclose(prediction.interval_probabilities[:, 0], 1.0 / 3.0)
        self.assertLess(prediction.converged_heads, prediction.total_heads)

    def test_nonconverged_stage_is_ineligible_for_gate(self) -> None:
        prediction = StagePrediction(
            name="candidate",
            label="candidate",
            interval_probabilities=np.zeros((1, 3, 3)),
            tail_probabilities=np.zeros((1, 3)),
            feature_count=1,
            converged_heads=7,
            total_heads=8,
        )

        self.assertEqual(
            stage_gate_exclusion_reason(
                prediction,
                {"days_0_30": 0.7, "days_31_90": 0.7, "days_91_365": 0.7},
            ),
            "not_all_heads_converged_or_estimable",
        )

    def test_bootstrap_mean_requires_all_three_intervals_to_be_valid(self) -> None:
        probabilities = np.full((4, 3, 3), 0.0)
        probabilities[:, :, 0] = 0.6
        probabilities[:, :, 1] = np.array(
            [[0.1, 0.2, 0.3], [0.2, 0.4, 0.1], [0.8, 0.1, 0.5], [0.7, 0.8, 0.9]]
        )
        probabilities[:, :, 2] = 1.0 - probabilities[:, :, :2].sum(axis=2)
        labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        result = paired_patient_bootstrap_auc_delta(
            probabilities,
            probabilities,
            labels,
            np.ones((4, 3), dtype=bool),
            np.arange(4),
            n_resamples=100,
            confidence=0.95,
            seed=7,
        )

        interval_valid = [result[name]["valid_resamples"] for name in result if name != "mean"]
        self.assertLessEqual(result["mean"]["valid_resamples"], min(interval_valid))

    def test_supported_candidate_becomes_next_reference_model(self) -> None:
        anchor = RetainedModel(index=0, blocks=(), residual_anchor=True)
        patient_state = RetainedModel(
            index=1,
            blocks=("anchor", "core"),
            residual_anchor=False,
        )

        self.assertEqual(
            advance_retained_model(anchor, patient_state, supported=True),
            patient_state,
        )
        self.assertEqual(
            advance_retained_model(anchor, patient_state, supported=False),
            anchor,
        )


if __name__ == "__main__":
    unittest.main()
