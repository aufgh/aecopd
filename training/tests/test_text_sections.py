from __future__ import annotations

import unittest

from aecopd_training.text_sections import (
    build_section_embedding_text,
    split_clinical_document,
)


class ClinicalTextSectionTests(unittest.TestCase):
    def test_discharge_note_is_split_in_source_order_and_keeps_preamble(self) -> None:
        parsed = split_clinical_document(
            document_role="current_discharge",
            note_type="Discharge summary",
            text=(
                "Administrative preamble\r\n"
                "Chief Complaint: Worsening dyspnea\r\n"
                "HPI:\r\nThree days of cough.\r\n"
                "Brief Hospital Course:\r\nImproved after treatment.\r\n"
                "Follow-up: Pulmonary clinic"
            ),
        )

        self.assertEqual(parsed.modality, "discharge_summary")
        self.assertEqual(
            [section.section_role for section in parsed.sections],
            [
                "other",
                "chief_complaint",
                "history_of_present_illness",
                "hospital_course",
                "follow_up",
            ],
        )
        self.assertEqual(parsed.sections[0].text, "Administrative preamble")
        self.assertEqual(parsed.sections[1].text, "Worsening dyspnea")
        self.assertEqual(parsed.sections[2].text, "Three days of cough.")

    def test_repeated_headers_remain_separate_ordered_instances(self) -> None:
        parsed = split_clinical_document(
            document_role="prior_discharge",
            note_type="Discharge summary",
            text="Hospital Course:\nFirst phase.\nHospital Course:\nSecond phase.",
        )

        self.assertEqual(len(parsed.sections), 2)
        self.assertEqual([section.section_instance for section in parsed.sections], [1, 2])
        self.assertEqual([section.text for section in parsed.sections], ["First phase.", "Second phase."])

    def test_document_without_known_header_is_preserved_as_other(self) -> None:
        parsed = split_clinical_document(
            document_role="current_discharge",
            note_type="Discharge summary",
            text="Narrative discharge summary without a standard heading.",
        )

        self.assertEqual(len(parsed.sections), 1)
        self.assertEqual(parsed.sections[0].section_role, "other")
        self.assertEqual(parsed.sections[0].text, "Narrative discharge summary without a standard heading.")

    def test_radiology_headers_and_modality_are_detected_without_parsing_dates(self) -> None:
        parsed = split_clinical_document(
            document_role="current_radiology",
            note_type="Radiology",
            text=(
                "EXAMINATION: CT CHEST\n"
                "INDICATION: Dyspnea since 2024-03-02\n"
                "FINDINGS: Emphysema.\n"
                "IMPRESSION: No acute process."
            ),
        )

        self.assertEqual(parsed.modality, "ct")
        self.assertEqual(
            [section.section_role for section in parsed.sections],
            ["examination", "indication", "findings", "impression"],
        )
        self.assertIn("2024-03-02", parsed.sections[1].text)

    def test_header_word_inside_prose_does_not_create_a_section(self) -> None:
        parsed = split_clinical_document(
            document_role="prior_radiology",
            note_type="Radiology",
            text="Narrative says the findings are stable without using a titled line.",
        )

        self.assertEqual([section.section_role for section in parsed.sections], ["other"])

    def test_unknown_titled_paragraph_is_isolated_as_other(self) -> None:
        parsed = split_clinical_document(
            document_role="current_discharge",
            note_type="Discharge summary",
            text=(
                "Past Medical History:\nCOPD.\n"
                "Social History: Former smoker.\n"
                "Pertinent Results:\nFEV1 reduced."
            ),
        )

        self.assertEqual(
            [section.section_role for section in parsed.sections],
            ["past_medical_history", "other", "pertinent_results"],
        )
        self.assertEqual(parsed.sections[1].text, "Social History: Former smoker.")

    def test_arbitrary_key_value_line_does_not_create_an_other_section(self) -> None:
        parsed = split_clinical_document(
            document_role="current_discharge",
            note_type="Discharge summary",
            text=(
                "Hospital Course:\n"
                "HR: 90 bpm\n"
                "Improved after bronchodilators.\n"
                "Discharge Diagnosis: COPD exacerbation"
            ),
        )

        self.assertEqual(
            [section.section_role for section in parsed.sections],
            ["hospital_course", "discharge_diagnosis"],
        )
        self.assertIn("HR: 90 bpm", parsed.sections[0].text)

    def test_embedding_prefix_makes_section_context_part_of_deduplication_key(self) -> None:
        discharge = split_clinical_document(
            document_role="current_discharge",
            note_type="Discharge summary",
            text="IMPRESSION: Stable.",
        ).sections[0]
        radiology_parsed = split_clinical_document(
            document_role="current_radiology",
            note_type="Radiology",
            text="IMPRESSION: Stable.",
        )
        radiology = radiology_parsed.sections[0]

        discharge_text = build_section_embedding_text(
            document_role="current_discharge",
            modality="discharge_summary",
            section=discharge,
        )
        radiology_text = build_section_embedding_text(
            document_role="current_radiology",
            modality=radiology_parsed.modality,
            section=radiology,
        )

        self.assertNotEqual(discharge_text, radiology_text)
        self.assertIn("Document role: current_radiology", radiology_text)
        self.assertIn("Section: impression", radiology_text)

    def test_timed_mimiciii_clinical_note_keeps_role_and_note_type(self) -> None:
        parsed = split_clinical_document(
            document_role="current_clinical_note",
            note_type="Respiratory",
            text="  Dyspnea improved after treatment.  ",
        )
        self.assertEqual(parsed.modality, "clinical_note")
        self.assertEqual(len(parsed.sections), 1)
        self.assertEqual(parsed.sections[0].text, "Dyspnea improved after treatment.")
        encoded = build_section_embedding_text(
            "current_clinical_note", parsed.modality, parsed.sections[0], "Respiratory"
        )
        self.assertIn("Note type: Respiratory", encoded)


if __name__ == "__main__":
    unittest.main()
