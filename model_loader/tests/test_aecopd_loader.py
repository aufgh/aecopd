from __future__ import annotations

import csv
import gzip
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aecopd_loader import AecopdDataBundle, buffered_shuffle


def write_gzip_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class AecopdLoaderContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        cohort_fields = [
            "subject_id", "hadm_id", "data_split", "training_eligible", "admittime", "dischtime",
            "event_type", "event_name", "time_days", "next_observed_aecopd_days", "observed_death_days",
            "aecopd_by_90d", "aecopd_by_180d", "aecopd_by_365d", "death_by_90d", "death_by_180d",
            "death_by_365d", "age_at_admission", "gender", "admission_type", "admission_location",
            "discharge_location", "via_ed", "ed_hours", "hospital_los_hours", "prior_aecopd_count",
            "prior_all_admission_count", "days_since_previous_aecopd", "entered_icu", "icu_stay_count",
            "icu_los_days", "prior_weight_kg", "prior_height_cm", "prior_bmi", "prior_sbp", "prior_dbp",
            "days_since_prior_weight", "days_since_prior_bmi", "days_since_prior_bp",
        ]
        base = {field: "" for field in cohort_fields}
        rows = []
        for subject_id, hadm_id, split, age, gender, event_type in [
            (1, 101, "train", 60, "M", 0),
            (2, 102, "train", 80, "F", 1),
            (3, 103, "validation", 1000, "X", 2),
        ]:
            row = dict(base)
            row.update({
                "subject_id": subject_id, "hadm_id": hadm_id, "data_split": split,
                "training_eligible": "t", "event_type": event_type,
                "event_name": ["CENSORED_365D", "AECOPD_READMISSION", "ALL_CAUSE_DEATH"][event_type],
                "time_days": 30, "age_at_admission": age, "gender": gender,
                "aecopd_by_90d": "t" if event_type == 1 else "f",
                "admission_type": "URGENT", "admission_location": "ED",
                "discharge_location": "HOME", "via_ed": "t", "hospital_los_hours": 48,
                "prior_aecopd_count": 0, "prior_all_admission_count": 1, "entered_icu": "f",
                "icu_stay_count": 0,
            })
            rows.append(row)
        write_gzip_csv(self.root / "cohort_labels_static.csv.gz", cohort_fields, rows)

        event_fields = [
            "event_id", "subject_id", "hadm_id", "data_split", "event_domain", "concept_name",
            "raw_code", "raw_name", "event_time", "end_time", "store_time", "hours_from_admission",
            "hours_before_discharge", "value_num", "value_text", "unit", "specimen", "route", "status",
            "evidence_type", "source_table", "model_scope", "availability_note", "value_clipped",
            "value_scaled", "normalization_available", "normalization_method",
        ]
        write_gzip_csv(self.root / "events.csv.gz", event_fields, [
            {**{f: "" for f in event_fields}, "event_id": "e1", "subject_id": 1, "hadm_id": 101,
             "data_split": "train", "event_domain": "检验", "concept_name": "白细胞", "value_num": 10,
             "unit": "k/uL", "hours_from_admission": 1, "hours_before_discharge": 47,
             "value_scaled": 0.5, "normalization_available": "t", "normalization_method": "train_iqr",
             "model_scope": "main_model"},
            {**{f: "" for f in event_fields}, "event_id": "e1a", "subject_id": 1, "hadm_id": 101,
             "data_split": "train", "event_domain": "药物", "concept_name": "泼尼松或泼尼松龙",
             "raw_code": "Prednisone", "raw_name": "Prednisone", "hours_from_admission": 2,
             "hours_before_discharge": 46, "route": "PO", "status": "Administered",
             "evidence_type": "actual_administration", "source_table": "mimiciv_hosp.emar",
             "model_scope": "main_model"},
            {**{f: "" for f in event_fields}, "event_id": "e1b", "subject_id": 1, "hadm_id": 101,
             "data_split": "train", "event_domain": "药物", "concept_name": "甲泼尼龙",
             "raw_code": "Methylprednisolone", "raw_name": "Methylprednisolone",
             "hours_from_admission": 3, "hours_before_discharge": 45, "route": "IV",
             "status": "Administered", "evidence_type": "actual_administration",
             "source_table": "mimiciv_hosp.emar", "model_scope": "main_model"},
            {**{f: "" for f in event_fields}, "event_id": "e1c", "subject_id": 1, "hadm_id": 101,
             "data_split": "train", "event_domain": "药物", "concept_name": "地塞米松",
             "raw_code": "Dexamethasone", "raw_name": "Dexamethasone",
             "hours_from_admission": 4, "hours_before_discharge": 44, "route": "IV",
             "status": "Not Given", "evidence_type": "actual_administration",
             "source_table": "mimiciv_hosp.emar", "model_scope": "main_model"},
            {**{f: "" for f in event_fields}, "event_id": "e1d", "subject_id": 1, "hadm_id": 101,
             "data_split": "train", "event_domain": "药物", "concept_name": "甲泼尼龙",
             "raw_code": "Methylprednisolone", "raw_name": "Prescription only steroid",
             "hours_from_admission": 5, "hours_before_discharge": 43, "route": "IV",
             "status": "", "evidence_type": "prescription_only",
             "source_table": "mimiciv_hosp.prescriptions", "model_scope": "main_model"},
            {**{f: "" for f in event_fields}, "event_id": "e1e", "subject_id": 1, "hadm_id": 101,
             "data_split": "train", "event_domain": "药物", "concept_name": "甲泼尼龙",
             "raw_code": "Methylprednisolone", "raw_name": "Held steroid",
             "hours_from_admission": 6, "hours_before_discharge": 42, "route": "IV",
             "status": "Hold Dose", "evidence_type": "actual_administration",
             "source_table": "mimiciv_hosp.emar", "model_scope": "main_model"},
            {**{f: "" for f in event_fields}, "event_id": "e1f", "subject_id": 1, "hadm_id": 101,
             "data_split": "train", "event_domain": "药物", "concept_name": "甲泼尼龙",
             "raw_code": "Methylprednisolone", "raw_name": "Paused steroid",
             "hours_from_admission": 7, "hours_before_discharge": 41, "route": "IV",
             "status": "Paused", "evidence_type": "actual_administration",
             "source_table": "mimiciv_hosp.emar", "model_scope": "main_model"},
            {**{f: "" for f in event_fields}, "event_id": "e1g", "subject_id": 1, "hadm_id": 101,
             "data_split": "train", "event_domain": "药物", "concept_name": "甲泼尼龙",
             "raw_code": "Methylprednisolone", "raw_name": "Untimed steroid",
             "route": "IV", "status": "Administered",
             "evidence_type": "actual_administration",
             "source_table": "mimiciv_hosp.emar", "model_scope": "main_model"},
            {**{f: "" for f in event_fields}, "event_id": "e1h", "subject_id": 1, "hadm_id": 101,
             "data_split": "train", "event_domain": "药物", "concept_name": "甲泼尼龙",
             "raw_code": "Methylprednisolone", "raw_name": "No steroid",
             "hours_from_admission": 8, "hours_before_discharge": 40, "route": "IV",
             "status": "No", "evidence_type": "actual_administration",
             "source_table": "mimiciv_hosp.emar", "model_scope": "main_model"},
            {**{f: "" for f in event_fields}, "event_id": "e1i", "subject_id": 1, "hadm_id": 101,
             "data_split": "train", "event_domain": "药物", "concept_name": "甲泼尼龙",
             "raw_code": "Methylprednisolone", "raw_name": "Unconfirmed steroid",
             "hours_from_admission": 9, "hours_before_discharge": 39, "route": "IV",
             "status": "Not Confirmed", "evidence_type": "actual_administration",
             "source_table": "mimiciv_hosp.emar", "model_scope": "main_model"},
            {**{f: "" for f in event_fields}, "event_id": "e1j", "subject_id": 1, "hadm_id": 101,
             "data_split": "train", "event_domain": "药物", "concept_name": "甲泼尼龙",
             "raw_code": "Methylprednisolone", "raw_name": "Delayed steroid",
             "hours_from_admission": 10, "hours_before_discharge": 38, "route": "IV",
             "status": "Delayed", "evidence_type": "actual_administration",
             "source_table": "mimiciv_hosp.emar", "model_scope": "main_model"},
            {**{f: "" for f in event_fields}, "event_id": "e2", "subject_id": 3, "hadm_id": 103,
             "data_split": "validation", "event_domain": "检验", "concept_name": "C反应蛋白",
             "value_num": 5, "hours_from_admission": 2, "hours_before_discharge": 46,
             "value_scaled": 0, "normalization_available": "f", "normalization_method": "unseen_in_train",
             "model_scope": "main_model"},
        ])

        summary_fields = [
            "subject_id", "hadm_id", "data_split", "event_domain", "concept_name", "canonical_unit",
            "measurement_count", "min_value", "max_value", "mean_value", "sd_value", "first_value",
            "last_value", "first_to_last_change", "slope_per_hour", "hours_before_discharge_last",
            "min_value_scaled", "max_value_scaled", "mean_value_scaled", "sd_value_scaled",
            "first_value_scaled", "last_value_scaled", "first_to_last_change_scaled",
            "slope_per_hour_scaled", "unseen_value_count", "normalization_available", "normalization_method",
        ]
        write_gzip_csv(self.root / "numeric_summary.csv.gz", summary_fields, [])

        stats_fields = ["event_domain", "concept_name", "canonical_unit", "train_n", "p005", "p25", "median", "p75", "p995"]
        write_gzip_csv(self.root / "numeric_normalization_stats.csv.gz", stats_fields, [
            {"event_domain": "检验", "concept_name": "白细胞", "canonical_unit": "k/ul", "train_n": 1,
             "p005": 10, "p25": 10, "median": 10, "p75": 10, "p995": 10},
        ])

        text_fields = [
            "document_key", "subject_id", "hadm_id", "source_hadm_id", "data_split", "document_role",
            "document_rank", "days_before_landmark", "chunk_index", "token_start", "token_end",
            "char_start", "char_end", "token_count", "chunk_text",
        ]
        write_gzip_csv(self.root / "qwen_text_chunks.csv.gz", text_fields, [
            {"document_key": "d1", "subject_id": 1, "hadm_id": 101, "source_hadm_id": 101,
             "data_split": "train", "document_role": "current_discharge", "document_rank": 1,
             "days_before_landmark": 0, "chunk_index": 0, "token_start": 0, "token_end": 3,
             "char_start": 0, "char_end": 4, "token_count": 3, "chunk_text": "test"},
        ])
        index_fields = [
            "embedding_row", "document_key", "chunk_index", "hadm_id", "data_split",
            "document_role", "document_rank", "days_before_landmark", "token_count",
        ]
        write_gzip_csv(self.root / "qwen_embedding_index.csv.gz", index_fields, [
            {"embedding_row": 0, "document_key": "d1", "chunk_index": 0, "hadm_id": 101,
             "data_split": "train", "document_role": "current_discharge", "document_rank": 1,
             "days_before_landmark": 0, "token_count": 3},
        ])

        self.section_text_root = self.root / "model_v2"
        self.section_text_root.mkdir()
        section_index_fields = [
            *index_fields,
            "source_document_key",
            "modality",
            "section_role",
            "section_order",
            "section_instance",
        ]
        write_gzip_csv(
            self.section_text_root / "qwen_section_embedding_index.csv.gz",
            section_index_fields,
            [
                {
                    "embedding_row": 0,
                    "document_key": "d1::section::000",
                    "chunk_index": 0,
                    "hadm_id": 101,
                    "data_split": "train",
                    "document_role": "current_discharge",
                    "document_rank": 1,
                    "days_before_landmark": 0,
                    "token_count": 5,
                    "source_document_key": "d1",
                    "modality": "discharge_summary",
                    "section_role": "chief_complaint",
                    "section_order": 0,
                    "section_instance": 1,
                },
                {
                    "embedding_row": 1,
                    "document_key": "d1::section::001",
                    "chunk_index": 0,
                    "hadm_id": 101,
                    "data_split": "train",
                    "document_role": "current_discharge",
                    "document_rank": 1,
                    "days_before_landmark": 0,
                    "token_count": 9,
                    "source_document_key": "d1",
                    "modality": "discharge_summary",
                    "section_role": "hospital_course",
                    "section_order": 1,
                    "section_instance": 1,
                },
            ],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_preprocessing_is_fit_from_train_only(self) -> None:
        bundle = AecopdDataBundle(self.root)
        artifacts = bundle.fit_preprocessing()

        age = artifacts["static_continuous"]["age_at_admission"]
        self.assertEqual(age["median"], 70.0)
        self.assertNotIn("X", artifacts["static_categorical"]["gender"]["vocabulary"])
        self.assertNotIn("patient_state|C反应蛋白", artifacts["event_vocabulary"])

    def test_treatments_use_canonical_identity_without_route_or_source_vocabularies(self) -> None:
        bundle = AecopdDataBundle(self.root)
        artifacts = bundle.fit_preprocessing()

        vocabulary = artifacts["event_vocabulary"]
        self.assertIn("care_process|全身糖皮质激素", vocabulary)
        self.assertNotIn("Prednisone", "\n".join(vocabulary))
        self.assertNotIn("Methylprednisolone", "\n".join(vocabulary))
        self.assertEqual(artifacts["event_route_vocabulary"], ["<PAD>", "<UNK>", "<MISSING>"])
        self.assertEqual(
            artifacts["event_evidence_vocabulary"],
            ["<PAD>", "<UNK>", "<MISSING>", "observed_time"],
        )

    def test_real_sample_contract_has_finite_values_and_unknown_mapping(self) -> None:
        bundle = AecopdDataBundle(self.root)
        artifacts = bundle.fit_preprocessing()
        validation = list(bundle.iter_samples("validation", artifacts=artifacts))

        self.assertEqual(len(validation), 1)
        sample = validation[0]
        self.assertEqual(sample.hadm_id, 103)
        self.assertEqual(sample.event_type, 2)
        self.assertEqual(sample.horizon_labels["aecopd_by_90d"], 0)
        self.assertEqual(sample.static_categorical["gender"], 1)  # <UNK>
        self.assertEqual(sample.events[0].concept_id, 1)  # <UNK>
        self.assertFalse(sample.events[0].normalization_available)
        self.assertTrue(all(math.isfinite(v) for v in sample.static_continuous.values()))

    def test_not_given_treatment_does_not_create_an_exposure_event(self) -> None:
        bundle = AecopdDataBundle(self.root)
        artifacts = bundle.fit_preprocessing()
        sample = next(bundle.iter_samples("train", artifacts=artifacts))

        self.assertNotIn("Dexamethasone", {event.raw_name for event in sample.events})
        self.assertNotIn("Prescription only steroid", {event.raw_name for event in sample.events})
        self.assertNotIn("Held steroid", {event.raw_name for event in sample.events})
        self.assertNotIn("Paused steroid", {event.raw_name for event in sample.events})
        self.assertNotIn("Untimed steroid", {event.raw_name for event in sample.events})
        self.assertNotIn("No steroid", {event.raw_name for event in sample.events})
        self.assertNotIn("Unconfirmed steroid", {event.raw_name for event in sample.events})
        self.assertNotIn("Delayed steroid", {event.raw_name for event in sample.events})
        steroids = [
            event for event in sample.events if event.concept_name == "全身糖皮质激素"
        ]
        self.assertEqual(len(steroids), 2)
        self.assertTrue(all(event.value_type == "binary_exposure" for event in steroids))
        self.assertTrue(all(event.route == "" for event in steroids))

    def test_buffer_shuffle_is_seeded_and_lossless(self) -> None:
        first = list(buffered_shuffle(range(20), buffer_size=4, seed=7))
        second = list(buffered_shuffle(range(20), buffer_size=4, seed=7))
        third = list(buffered_shuffle(range(20), buffer_size=4, seed=8))

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual(sorted(first), list(range(20)))

    def test_training_can_load_text_metadata_without_decompressing_raw_text(self) -> None:
        bundle = AecopdDataBundle(self.root, text_metadata_only=True)
        artifacts = bundle.fit_preprocessing()
        sample = next(bundle.iter_samples("train", artifacts=artifacts))

        self.assertEqual(sample.text_chunks[0].document_key, "d1")
        self.assertEqual(sample.text_chunks[0].token_count, 3)
        self.assertEqual(sample.text_chunks[0].chunk_text, "")

    def test_section_text_directory_preserves_document_and_section_metadata(self) -> None:
        bundle = AecopdDataBundle(
            self.root,
            text_metadata_only=True,
            text_dir=self.section_text_root,
        )
        artifacts = bundle.fit_preprocessing()
        sample = next(bundle.iter_samples("train", artifacts=artifacts))

        self.assertEqual(len(sample.text_chunks), 2)
        self.assertEqual(sample.text_chunks[0].source_document_key, "d1")
        self.assertEqual(sample.text_chunks[0].section_role, "chief_complaint")
        self.assertEqual(sample.text_chunks[1].section_role, "hospital_course")
        self.assertEqual(sample.text_chunks[1].section_order, 1)
        self.assertEqual(sample.text_chunks[1].modality, "discharge_summary")


if __name__ == "__main__":
    unittest.main()
