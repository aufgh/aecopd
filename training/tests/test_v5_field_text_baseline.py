"""独立字段文本清单不会暗中生成章节与结构化事实的配对。"""

from __future__ import annotations

import unittest

from training.v5.field_text_baseline import eligible, field_text


class FieldTextBaselineTests(unittest.TestCase):
    def test_field_name_is_explicit_context(self) -> None:
        self.assertEqual(field_text("past_medical_history", "COPD on home oxygen"),
                         "Clinical field: Past Medical History\nCOPD on home oxygen")
        with self.assertRaises(ValueError):
            field_text("unknown", "Some text")

    def test_screening_keeps_narrative_but_flags_admin_header(self) -> None:
        self.assertEqual(eligible("hospital_course", "Heart failure treated.", "by_discharge"),
                         (True, "screened_candidate_not_clinically_validated"))
        self.assertEqual(eligible("history_of_present_illness", "PCP: ___\nDyspnea", "by_discharge"),
                         (False, "hpi_administrative_header_review"))

    def test_condition_keeps_function_but_not_template_or_long_mixed_note(self) -> None:
        self.assertTrue(eligible("discharge_condition", "Activity Status: Ambulatory - Independent.",
                                 "by_discharge")[0])
        self.assertFalse(eligible("discharge_condition", "Stable.", "by_discharge")[0])
        self.assertFalse(eligible("discharge_condition", "Activity Status: Ambulatory. " * 30,
                                  "by_discharge")[0])

    def test_medication_lists_wait_for_separate_semantic_pilot(self) -> None:
        self.assertFalse(eligible("discharge_medications", "Prednisone 20 mg PO daily", "by_discharge")[0])


if __name__ == "__main__":
    unittest.main()
