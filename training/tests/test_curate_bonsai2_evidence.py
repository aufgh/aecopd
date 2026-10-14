"""出院时点与字段语义质量门的无病历测试。"""

from __future__ import annotations

import unittest

from training.curate_bonsai2_evidence import curate_span


class CurateEvidenceTests(unittest.TestCase):
    def test_admission_cough_is_not_residual_discharge_cough(self) -> None:
        note = "HISTORY OF PRESENT ILLNESS:\nPatient reports productive cough.\n"
        result = curate_span("residual_cough_sputum", 2, note)
        self.assertIsNone(result.quote)
        self.assertEqual(result.reason, "not_discharge_state")

    def test_discharge_cough_is_retained(self) -> None:
        note = "DISCHARGE CONDITION:\nCough had resolved at discharge.\n"
        result = curate_span("residual_cough_sputum", 2, note)
        self.assertEqual(result.reason, "accepted")

    def test_inhaler_medication_list_is_not_adherence(self) -> None:
        note = "DISCHARGE MEDICATIONS:\nAlbuterol inhaler one puff every four hours.\n"
        result = curate_span("inhaler_use_or_adherence", 2, note)
        self.assertEqual(result.reason, "medication_list_not_adherence")

    def test_wrapped_inhaler_education_is_rejoined(self) -> None:
        note = ("TRANSITIONAL ISSUES:\nHome nurse will ensure the patient is taught\n"
                "how to use her inhaler correctly.\n")
        result = curate_span("inhaler_use_or_adherence", 2, note)
        self.assertEqual(result.reason, "accepted")
        self.assertIn("how to use", result.quote or "")

    def test_raw_radiology_result_is_not_unresolved_discharge_problem(self) -> None:
        note = "PERTINENT RESULTS:\nChest radiograph showed bibasilar opacities.\n"
        result = curate_span("unresolved_infection_or_imaging", 2, note)
        self.assertEqual(result.reason, "raw_imaging_not_discharge_status")

    def test_discharge_followup_nodule_is_retained(self) -> None:
        note = ("TRANSITIONAL ISSUES:\nPulmonary nodule remains on chest CT;\n"
                "repeat imaging is planned after discharge.\n")
        result = curate_span("unresolved_infection_or_imaging", 2, note)
        self.assertEqual(result.reason, "accepted")


if __name__ == "__main__":
    unittest.main()
