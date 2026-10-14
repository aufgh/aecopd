from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aecopd_loader import AdmissionSample, EventInput, NumericSummaryInput
from timeline_collator import SharedTimelineCollator, TimelineConfig


def make_event(
    event_id: str,
    *,
    concept_id: int,
    hours_before_discharge: float,
    hours_from_admission: float | None = None,
    value_scaled: float = 0.0,
    numeric_mask: bool = False,
    event_domain: str = "检验",
    concept_name: str = "白细胞",
    unit: str = "K/uL",
    route: str = "",
    evidence_type: str = "measured",
    status: str = "",
    duration_hours: float = 0.0,
    duration_mask: bool = False,
    feature_layer: str = "patient_state",
    value_type: str = "clinical_state",
    source_priority: int = 0,
    dedup_identity: str = "",
    dedup_route: str = "",
) -> EventInput:
    return EventInput(
        event_id=event_id,
        concept_id=concept_id,
        event_domain=event_domain,
        concept_name=concept_name,
        hours_from_admission=(
            48.0 - hours_before_discharge
            if hours_from_admission is None
            else hours_from_admission
        ),
        hours_before_discharge=hours_before_discharge,
        value_scaled=value_scaled,
        numeric_mask=numeric_mask,
        normalization_available=numeric_mask,
        normalization_method="train_iqr" if numeric_mask else "",
        raw_code="",
        raw_name=concept_name,
        value_text="",
        unit=unit,
        route=route,
        status=status,
        evidence_type=evidence_type,
        model_scope="main_model",
        duration_hours=duration_hours,
        duration_mask=duration_mask,
        feature_layer=feature_layer,
        value_type=value_type,
        source_priority=source_priority,
        dedup_identity=dedup_identity,
        dedup_route=dedup_route,
    )


def make_sample(events: list[EventInput]) -> AdmissionSample:
    return AdmissionSample(
        subject_id=1,
        hadm_id=101,
        data_split="train",
        event_type=1,
        event_name="AECOPD_READMISSION",
        time_days=30.0,
        horizon_labels={"aecopd_by_90d": 1},
        static_continuous={},
        static_missing_mask={},
        static_categorical={},
        static_binary={},
        events=events,
        numeric_summaries=[],
        text_chunks=[],
    )


class SharedTimelineCollatorTest(unittest.TestCase):
    def test_empty_curated_timeline_receives_only_a_missing_placeholder(self) -> None:
        batch = SharedTimelineCollator(TimelineConfig())([make_sample([])])

        self.assertEqual(batch.timeline_lengths, [1])
        self.assertEqual(batch.timeline_concept_ids[0][0], 2)
        self.assertEqual(batch.timeline_window_ids[0][0], 17)
        self.assertTrue(batch.timeline_attention_mask[0][0])
        self.assertFalse(any(batch.timeline_feature_masks[0][0]))

    def test_patient_state_only_configuration_excludes_care_process_tokens(self) -> None:
        sample = replace(make_sample([
            make_event(
                "lab",
                concept_id=10,
                hours_before_discharge=10,
                feature_layer="patient_state",
            ),
            make_event(
                "treatment",
                concept_id=70,
                hours_before_discharge=10,
                event_domain="care_process",
                concept_name="抗菌药物",
                feature_layer="care_process",
                value_type="binary_exposure",
            ),
        ]),
            static_continuous={"age_at_admission": 0.5, "hospital_los_hours": 2.0},
            static_missing_mask={"age_at_admission": False, "hospital_los_hours": False},
            static_categorical={"gender": 3, "discharge_location": 4},
            static_binary={"via_ed": 1, "entered_icu": 1},
        )

        batch = SharedTimelineCollator(
            TimelineConfig(included_layers=("patient_state",))
        )([sample])

        self.assertEqual(batch.timeline_lengths, [1])
        self.assertEqual(batch.timeline_concept_ids[0][0], 10)
        continuous = dict(zip(batch.static_continuous_names, batch.static_continuous[0]))
        continuous_masks = dict(
            zip(batch.static_continuous_names, batch.static_continuous_masks[0])
        )
        categorical = dict(zip(batch.static_categorical_names, batch.static_categorical[0]))
        binary = dict(zip(batch.static_binary_names, batch.static_binary[0]))
        self.assertEqual(continuous["age_at_admission"], 0.5)
        self.assertEqual(continuous["hospital_los_hours"], 0.0)
        self.assertFalse(continuous_masks["hospital_los_hours"])
        self.assertEqual(categorical["gender"], 3)
        self.assertEqual(categorical["discharge_location"], 2)
        self.assertEqual(binary["via_ed"], 0)
        self.assertEqual(binary["entered_icu"], 0)

    def test_distinct_unknown_concepts_do_not_merge_or_mix_value_types(self) -> None:
        sample = make_sample([
            make_event(
                "unknown-numeric",
                concept_id=1,
                hours_before_discharge=10,
                concept_name="训练未见数值概念",
                value_scaled=1.0,
                numeric_mask=True,
                value_type="numeric_trajectory",
            ),
            make_event(
                "unknown-state",
                concept_id=1,
                hours_before_discharge=10,
                concept_name="训练未见状态概念",
                value_type="clinical_state",
            ),
        ])

        batch = SharedTimelineCollator(TimelineConfig())([sample])

        self.assertEqual(batch.timeline_lengths, [2])
        self.assertEqual(batch.timeline_concept_ids[0][:2], [1, 1])

    def test_treatment_is_one_binary_token_independent_of_route_source_and_record_count(self) -> None:
        sample = make_sample([
            make_event(
                "lower-priority-source",
                concept_id=70,
                hours_from_admission=2.5,
                hours_before_discharge=45.5,
                event_domain="care_process",
                concept_name="全身糖皮质激素",
                route="IV",
                evidence_type="icu_input",
                status="administered",
                feature_layer="care_process",
                value_type="binary_exposure",
                source_priority=1,
                dedup_identity="methylprednisolone",
                dedup_route="po",
            ),
            make_event(
                "steroid-po",
                concept_id=70,
                hours_from_admission=2,
                hours_before_discharge=46,
                event_domain="care_process",
                concept_name="全身糖皮质激素",
                route="PO",
                evidence_type="actual_administration",
                status="administered",
                feature_layer="care_process",
                value_type="binary_exposure",
                source_priority=3,
                dedup_identity="methylprednisolone",
                dedup_route="po",
            ),
            make_event(
                "steroid-iv",
                concept_id=70,
                hours_from_admission=20,
                hours_before_discharge=28,
                event_domain="care_process",
                concept_name="全身糖皮质激素",
                route="IV",
                evidence_type="icu_input",
                status="administered",
                feature_layer="care_process",
                value_type="binary_exposure",
                source_priority=3,
                dedup_identity="methylprednisolone",
                dedup_route="iv",
            ),
        ])

        batch = SharedTimelineCollator(TimelineConfig())([sample])

        self.assertEqual(batch.timeline_lengths, [1])
        self.assertEqual(batch.timeline_route_ids[0][0], 2)
        self.assertEqual(batch.timeline_evidence_ids[0][0], 2)
        features = dict(zip(batch.timeline_feature_names, batch.timeline_features[0][0]))
        masks = dict(zip(batch.timeline_feature_names, batch.timeline_feature_masks[0][0]))
        self.assertAlmostEqual(features["count_log1p"], __import__("math").log1p(1))
        self.assertAlmostEqual(
            features["duration_hours_max_log1p"], __import__("math").log1p(18.0)
        )
        self.assertTrue(masks["duration_hours_max_log1p"])
        self.assertEqual(features["status_administered_fraction"], 1.0)

    def test_distinct_lower_priority_treatment_time_is_not_discarded(self) -> None:
        sample = make_sample([
            make_event(
                "icu-distinct-time",
                concept_id=70,
                hours_from_admission=0,
                hours_before_discharge=48,
                event_domain="care_process",
                concept_name="全身糖皮质激素",
                feature_layer="care_process",
                value_type="binary_exposure",
                source_priority=2,
                dedup_identity="methylprednisolone",
                dedup_route="iv",
            ),
            make_event(
                "emar-later",
                concept_id=70,
                hours_from_admission=20,
                hours_before_discharge=28,
                event_domain="care_process",
                concept_name="全身糖皮质激素",
                feature_layer="care_process",
                value_type="binary_exposure",
                source_priority=3,
                dedup_identity="methylprednisolone",
                dedup_route="iv",
            ),
        ])

        batch = SharedTimelineCollator(TimelineConfig())([sample])

        features = dict(zip(batch.timeline_feature_names, batch.timeline_features[0][0]))
        self.assertAlmostEqual(
            features["duration_hours_max_log1p"], __import__("math").log1p(20.0)
        )

    def test_single_treatment_without_end_time_has_unknown_duration(self) -> None:
        sample = make_sample([
            make_event(
                "single-dose",
                concept_id=70,
                hours_from_admission=2,
                hours_before_discharge=46,
                event_domain="care_process",
                concept_name="全身糖皮质激素",
                feature_layer="care_process",
                value_type="binary_exposure",
                source_priority=3,
            )
        ])

        batch = SharedTimelineCollator(TimelineConfig())([sample])

        duration_index = batch.timeline_feature_names.index("duration_hours_max_log1p")
        self.assertFalse(batch.timeline_feature_masks[0][0][duration_index])

    def test_timeline_length_matches_collated_batch_length(self) -> None:
        sample = make_sample([
            make_event("e1", concept_id=10, hours_before_discharge=10),
            make_event("e2", concept_id=11, hours_before_discharge=10),
        ])
        collator = SharedTimelineCollator(TimelineConfig())

        observed = collator.timeline_length(sample)

        self.assertEqual(observed, collator([sample]).timeline_lengths[0])

    def test_same_concept_and_window_become_one_token_with_numeric_trajectory(self) -> None:
        sample = make_sample([
            make_event("e1", concept_id=10, hours_before_discharge=10, value_scaled=1.0, numeric_mask=True),
            make_event("e2", concept_id=10, hours_before_discharge=8, value_scaled=3.0, numeric_mask=True),
        ])

        batch = SharedTimelineCollator(TimelineConfig())([sample])

        self.assertEqual(batch.timeline_lengths, [1])
        self.assertEqual(batch.timeline_concept_ids[0][0], 10)
        self.assertEqual(batch.timeline_window_ids[0][0], 2)  # 入院第2天
        features = dict(zip(batch.timeline_feature_names, batch.timeline_features[0][0]))
        masks = dict(zip(batch.timeline_feature_names, batch.timeline_feature_masks[0][0]))
        self.assertAlmostEqual(features["first_scaled"], 1.0)
        self.assertAlmostEqual(features["last_scaled"], 3.0)
        self.assertAlmostEqual(features["delta_scaled"], 2.0)
        self.assertAlmostEqual(features["mean_scaled"], 2.0)
        self.assertTrue(masks["slope_scaled_per_hour"])

    def test_token_limit_never_silently_discards_clinical_events(self) -> None:
        sample = make_sample([
            make_event("e1", concept_id=10, hours_before_discharge=10),
            make_event("e2", concept_id=11, hours_before_discharge=10),
        ])

        collator = SharedTimelineCollator(TimelineConfig(max_tokens=1))

        with self.assertRaisesRegex(ValueError, "hadm_id=101"):
            collator([sample])

    def test_batch_contains_padded_static_and_global_numeric_inputs(self) -> None:
        summary = NumericSummaryInput(
            concept_id=21,
            event_domain="检验",
            concept_name="白细胞",
            canonical_unit="k/ul",
            measurement_count_log1p=1.5,
            scaled_values={"mean_value_scaled": 0.25},
            missing_mask={"mean_value_scaled": False},
            normalization_available=True,
            unseen_value_count=0,
        )
        first = replace(
            make_sample([make_event("e1", concept_id=10, hours_before_discharge=10)]),
            static_continuous={"age_at_admission": -0.5},
            static_missing_mask={"age_at_admission": False},
            static_categorical={"gender": 3},
            static_binary={"via_ed": 1},
            numeric_summaries=[summary],
        )
        second = replace(make_sample([]), subject_id=2, hadm_id=102, event_type=0, numeric_summaries=[])

        batch = SharedTimelineCollator(TimelineConfig())([first, second])

        self.assertEqual(batch.timeline_attention_mask, [[True], [True]])
        self.assertEqual(batch.timeline_concept_ids[1][0], 2)
        self.assertEqual(batch.static_continuous[0][0], -0.5)
        self.assertEqual(batch.static_continuous_masks[0][0], True)
        self.assertEqual(batch.global_summary_lengths, [1, 0])
        self.assertEqual(batch.global_summary_concept_ids, [[21], [0]])
        mean_index = batch.global_summary_feature_names.index("mean_value_scaled")
        self.assertEqual(batch.global_summary_features[0][0][mean_index], 0.25)
        self.assertTrue(batch.global_summary_feature_masks[0][0][mean_index])

    def test_treatment_duration_is_retained_without_expanding_every_raw_event(self) -> None:
        sample = make_sample([
            make_event(
                "t1",
                concept_id=30,
                hours_before_discharge=30,
                event_domain="ICU操作",
                concept_name="无创通气",
                evidence_type="actual_icu_procedure",
                duration_hours=18.0,
                duration_mask=True,
            )
        ])

        batch = SharedTimelineCollator(TimelineConfig())([sample])

        features = dict(zip(batch.timeline_feature_names, batch.timeline_features[0][0]))
        masks = dict(zip(batch.timeline_feature_names, batch.timeline_feature_masks[0][0]))
        self.assertAlmostEqual(features["duration_hours_max_log1p"], __import__("math").log1p(18.0))
        self.assertTrue(masks["duration_hours_max_log1p"])

    def test_retrospective_codes_are_not_given_false_clinical_timing(self) -> None:
        sample = make_sample([
            make_event(
                "dx1",
                concept_id=40,
                hours_before_discharge=0,
                event_domain="诊断",
                concept_name="慢性呼吸衰竭",
                evidence_type="retrospective_discharge_code",
            )
        ])

        batch = SharedTimelineCollator(TimelineConfig())([sample])

        self.assertEqual(batch.timeline_window_ids[0][0], 16)
        masks = dict(zip(batch.timeline_feature_names, batch.timeline_feature_masks[0][0]))
        self.assertFalse(masks["hours_before_discharge_first_log1p"])
        self.assertFalse(masks["hours_before_discharge_last_log1p"])

    def test_dual_anchor_windows_preserve_admission_start_middle_and_discharge_end(self) -> None:
        sample = make_sample([
            make_event(
                "early", concept_id=61, hours_from_admission=12, hours_before_discharge=468,
            ),
            make_event(
                "middle", concept_id=62, hours_from_admission=200, hours_before_discharge=280,
            ),
            make_event(
                "late", concept_id=63, hours_from_admission=468, hours_before_discharge=12,
            ),
        ])

        batch = SharedTimelineCollator(TimelineConfig())([sample])

        self.assertEqual(batch.timeline_window_ids[0][:3], [1, 8, 15])
        self.assertEqual(batch.timeline_window_names[1], "入院第1天（0～24小时）")
        self.assertEqual(batch.timeline_window_names[8], "长住院中间阶段")
        self.assertEqual(batch.timeline_window_names[15], "出院前第1天（0～24小时）")

    def test_torch_adapter_has_an_explicit_dependency_error(self) -> None:
        batch = SharedTimelineCollator(TimelineConfig())([make_sample([])])

        original_import = __import__

        def import_without_torch(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch":
                raise ImportError("simulated missing optional dependency")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_torch):
            with self.assertRaisesRegex(RuntimeError, "PyTorch"):
                batch.to_torch()

    def test_medication_states_distinguish_administered_from_not_given(self) -> None:
        sample = make_sample([
            make_event(
                "m1", concept_id=50, hours_before_discharge=8, event_domain="药物",
                concept_name="抗菌药物", status="Administered", evidence_type="actual_administration",
            ),
            make_event(
                "m2", concept_id=50, hours_before_discharge=7, event_domain="药物",
                concept_name="抗菌药物", status="Not Given", evidence_type="actual_administration",
            ),
        ])

        batch = SharedTimelineCollator(TimelineConfig())([sample])

        self.assertEqual(batch.timeline_lengths, [1])
        features = dict(zip(batch.timeline_feature_names, batch.timeline_features[0][0]))
        self.assertEqual(features["status_administered_fraction"], 0.5)
        self.assertEqual(features["status_not_given_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
