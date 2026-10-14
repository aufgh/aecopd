"""章节文本风险标记只能提示人工核对，不可静默删除临床内容。"""

from __future__ import annotations

import unittest

from training.v5.section_alignment_data.screen_section_content import screen_flags


class SectionContentScreenTests(unittest.TestCase):
    def test_admin_header_is_local_to_hpi(self) -> None:
        text = "PCP: [redacted]\nPatient has dyspnea and cough."
        self.assertIn("administrative_header_first_500_chars",
                      screen_flags("history_of_present_illness", text, "by_discharge"))
        self.assertNotIn("administrative_header_first_500_chars",
                         screen_flags("hospital_course", text, "by_discharge"))

    def test_condition_boundary_and_function_are_separate_flags(self) -> None:
        self.assertIn("generic_condition_only", screen_flags("discharge_condition", "Stable.", "by_discharge"))
        self.assertIn("function_or_respiratory_cue", screen_flags(
            "discharge_condition", "Activity Status: Ambulatory - Independent.", "after_discharge"))
        self.assertIn("condition_over_500_chars_review_boundary", screen_flags(
            "discharge_condition", "Discharge examination: " + "breathing comfortably. " * 30,
            "by_discharge"))

    def test_medication_and_course_flags_do_not_claim_relevance(self) -> None:
        self.assertIn("drug_list_format", screen_flags(
            "discharge_medications", "Prednisone 20 mg PO DAILY", "by_discharge"))
        self.assertIn("problem_subheadings", screen_flags(
            "hospital_course", "# Heart failure\nDiuresed.\n# COPD\nTreated.", "by_discharge"))
        self.assertEqual(screen_flags("hospital_course", "Heart failure treated.", "by_discharge"), ())


if __name__ == "__main__":
    unittest.main()
