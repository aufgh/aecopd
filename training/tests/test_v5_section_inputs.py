"""章节级结构化输入的概念映射和数值槽位回归测试。"""

from __future__ import annotations

import unittest

from training.v5.clinical_concepts import LABS, MED_NAMES, diagnosis_groups, medication_classes
from training.v5.section_alignment_data.build_section_inputs import (
    SECTION_ROLES, _token, prior_chronic_history,
)
from training.v5.section_alignment_data.prepare_section_text import chunk_offsets


class ClinicalConceptTests(unittest.TestCase):
    def test_old_v5_concept_contract_is_preserved(self) -> None:
        self.assertEqual(len(LABS), 8)
        self.assertEqual(len(MED_NAMES), 12)
        self.assertEqual(medication_classes("Ipratropium-Albuterol neb"), {"saba", "sama"})
        self.assertIn("inhaled_steroid", medication_classes("Fluticasone-Salmeterol diskus"))
        self.assertNotIn("inhaled_steroid", medication_classes("Fluticasone propionate nasal"))
        self.assertEqual(diagnosis_groups("I50.9", 10), {"heart_failure"})
        self.assertEqual(diagnosis_groups("51881", 9), {"acute_resp_failure"})

    def test_text_only_chapters_remain_slots(self) -> None:
        self.assertIn("admission_medications", SECTION_ROLES)
        self.assertIn("discharge_medications", SECTION_ROLES)
        self.assertIn("discharge_condition", SECTION_ROLES)
        self.assertIn("social_history", SECTION_ROLES)

    def test_token_channels_are_padded_and_masked(self) -> None:
        token = _token("1", "10", "pertinent_results", "laboratory", "creatinine",
                       (1.0, 1.2), (1, 1), "mg/dL", "labevents")
        self.assertEqual(token[5:10], ("1.0", "1.2", "0.0", "0.0", "0.0"))
        self.assertEqual(token[10:15], ("1", "1", "0", "0", "0"))
        self.assertEqual(len(token), 17)

    def test_prior_history_excludes_current_future_and_overlapping_stays(self) -> None:
        cohort = {
            "10": {"subject_id": "1", "hadm_id": "10", "admittime": "2020-01-01", "dischtime": "2020-01-05"},
            "11": {"subject_id": "1", "hadm_id": "11", "admittime": "2020-01-04", "dischtime": "2020-01-06"},
            "12": {"subject_id": "1", "hadm_id": "12", "admittime": "2020-01-10", "dischtime": "2020-01-12"},
        }
        diagnosis = {"10": {"heart_failure"}, "11": {"ckd"}, "12": {"diabetes"}}
        history = prior_chronic_history(cohort, diagnosis)
        self.assertEqual(history["10"], set())
        self.assertEqual(history["11"], set())
        self.assertEqual(history["12"], {"heart_failure", "ckd"})

    def test_text_chunks_keep_original_character_boundaries(self) -> None:
        offsets = ((0, 3), (4, 7), (8, 11), (12, 15), (16, 19))
        self.assertEqual(chunk_offsets(offsets, 3, 1),
                         ((0, 11, 0), (8, 19, 1)))
        with self.assertRaises(ValueError):
            chunk_offsets(offsets, 3, 3)


if __name__ == "__main__":
    unittest.main()
