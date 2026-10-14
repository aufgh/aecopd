"""全文候选、换行重组和出院时点判定的无病历回归测试。"""

from __future__ import annotations

import unittest

from training.clinical_discharge_rules_v2 import extract_mentions, text_units


class DischargeRulesV2Tests(unittest.TestCase):
    def test_wrapped_shortness_of_breath_is_a_single_candidate(self) -> None:
        note = "DISCHARGE CONDITION:\nNo shortness of\nbreath at rest.\n"
        matches = [item for item in extract_mentions(note) if item.field == "residual_dyspnea"]
        self.assertEqual(len(matches), 1)
        self.assertEqual((matches[0].start_line, matches[0].end_line), (2, 3))
        self.assertTrue(matches[0].eligible)

    def test_admission_cough_is_retained_but_not_eligible(self) -> None:
        note = ("HISTORY OF PRESENT ILLNESS:\nProductive cough on arrival.\n"
                "DISCHARGE CONDITION:\nCough resolved before discharge.\n")
        matches = [item for item in extract_mentions(note) if item.field == "residual_cough_sputum"]
        self.assertEqual(len(matches), 2)
        self.assertEqual([item.eligible for item in matches], [False, True])
        self.assertEqual(matches[0].temporality, "admission")

    def test_followup_heading_does_not_inherit_discharge_condition(self) -> None:
        note = ("DISCHARGE CONDITION:\nActivity Status: Ambulatory - Independent\n"
                "FOLLOWUP INSTRUCTIONS:\nRepeat chest CT for lung nodule.\n")
        units = text_units(note)
        self.assertEqual(units[-1].section, "FOLLOWUP INSTRUCTIONS")
        imaging = [item for item in extract_mentions(note)
                   if item.field == "unresolved_infection_or_imaging"]
        self.assertEqual(imaging[-1].section, "FOLLOWUP INSTRUCTIONS")
        self.assertTrue(imaging[-1].eligible)

    def test_medication_list_is_candidate_but_not_adherence(self) -> None:
        note = "DISCHARGE MEDICATIONS:\nAlbuterol inhaler one puff every four hours.\n"
        matches = [item for item in extract_mentions(note) if item.field == "inhaler_use_or_adherence"]
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].eligible)
        self.assertEqual(matches[0].exclusion_reason, "drug_or_device_mention_only")

    def test_unmentioned_field_has_no_negative_candidate(self) -> None:
        note = "DISCHARGE CONDITION:\nAwake and comfortable.\n"
        self.assertEqual(extract_mentions(note), [])

    def test_discharge_sentence_does_not_relabel_presenting_dyspnea(self) -> None:
        note = ("BRIEF HOSPITAL COURSE:\nPatient presented with shortness of breath. "
                "She improved and was discharged on prednisone.\n")
        dyspnea = [item for item in extract_mentions(note) if item.field == "residual_dyspnea"]
        self.assertEqual(len(dyspnea), 1)
        self.assertFalse(dyspnea[0].eligible)

    def test_sputum_culture_is_not_residual_sputum_symptom(self) -> None:
        note = "BRIEF HOSPITAL COURSE:\nSputum culture was pending at discharge.\n"
        sputum = [item for item in extract_mentions(note) if item.field == "residual_cough_sputum"]
        self.assertEqual(len(sputum), 1)
        self.assertFalse(sputum[0].eligible)

    def test_oxygen_saturation_alone_is_not_oxygen_support(self) -> None:
        note = "DISCHARGE CONDITION:\nO2 saturation was 96%.\n"
        oxygen = [item for item in extract_mentions(note)
                  if item.field == "oxygen_or_niv_at_discharge"]
        self.assertEqual(len(oxygen), 1)
        self.assertFalse(oxygen[0].eligible)

    def test_corpus_discovered_doe_and_group_home_are_candidates(self) -> None:
        note = ("SOCIAL HISTORY:\nLives in group home with two roommates.\n"
                "DISCHARGE CONDITION:\nStable DOE on walking.\n")
        mentions = extract_mentions(note)
        self.assertTrue(any(item.field == "social_support" for item in mentions))
        self.assertTrue(any(item.field == "residual_dyspnea" and item.eligible
                            for item in mentions))

    def test_by_the_day_of_discharge_cough_is_retained(self) -> None:
        note = "BRIEF HOSPITAL COURSE:\nBy the day of discharge, cough had improved.\n"
        cough = [item for item in extract_mentions(note)
                 if item.field == "residual_cough_sputum"]
        self.assertEqual(len(cough), 1)
        self.assertTrue(cough[0].eligible)


if __name__ == "__main__":
    unittest.main()
