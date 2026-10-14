"""V5 章节切分与原文位置的回归测试。"""

from __future__ import annotations

import unittest
import csv
import gzip
import tempfile
from pathlib import Path

from training.v5.section_alignment_data.audit_old_export_parity import _labs_parity
from training.v5.section_alignment_data.build_dataset import build_dataset
from training.v5.section_alignment_data.section_parser import parse_sections


class SectionParserTests(unittest.TestCase):
    def test_empty_section_has_nonreversed_offsets(self) -> None:
        note = "Admission Physical Exam:\n \nDischarge Physical Exam:\nStable\n"
        sections = parse_sections(note)
        self.assertEqual(sections[0].content_status, "placeholder")
        self.assertLessEqual(sections[0].start_char, sections[0].end_char)
        self.assertEqual(note[sections[0].start_char:sections[0].end_char], "")

    def test_attending_does_not_contaminate_allergies(self) -> None:
        note = "Allergies:\nPenicillin\nAttending: ___\nChief Complaint:\nShortness of breath\n"
        sections = parse_sections(note)
        self.assertEqual(sections[0].role, "allergies")
        self.assertEqual(sections[0].text, "Penicillin")
        self.assertEqual(sections[1].role, "metadata")
        self.assertEqual(sections[2].role, "chief_complaint")
        for section in sections:
            self.assertEqual(note[section.start_char:section.end_char].strip(), section.text)


class SectionDatasetTests(unittest.TestCase):
    @staticmethod
    def _write(path: Path, fields: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)

    def test_sources_stay_distinct_and_candidates_are_not_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            self._write(raw / "cohort.csv.gz",
                        ("subject_id", "hadm_id", "admittime", "dischtime", "index_is_strict_aecopd"),
                        [("1", "10", "2020-01-01 00:00:00", "2020-01-04 00:00:00", "t")])
            self._write(root / "notes.csv.gz",
                        ("subject_id", "hadm_id", "note_id", "charttime", "storetime", "dischtime", "timestamp_relation", "document_text"),
                        [("1", "10", "N1", "", "2020-01-05 00:00:00", "2020-01-04 00:00:00", "after_discharge",
                          "Pertinent Results:\nCr 1.2\nBrief Hospital Course:\nImproved\nDischarge Medications:\nInhaler\n")])
            self._write(raw / "labs.csv.gz",
                        ("subject_id", "hadm_id", "labevent_id", "itemid", "charttime", "storetime", "valuenum", "valueuom", "flag"),
                        [("1", "10", "L1", "50912", "2020-01-02 00:00:00", "2020-01-02 01:00:00", "1.2", "mg/dL", "")])
            self._write(raw / "diagnoses.csv.gz",
                        ("subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"),
                        [("1", "10", "1", "J441", "10"),
                         ("1", "10", "1", "J449", "10")])
            self._write(raw / "emar.csv.gz",
                        ("subject_id", "hadm_id", "emar_id", "emar_seq", "pharmacy_id", "charttime", "storetime", "medication", "event_txt"),
                        [("1", "10", "E1", "1", "P1", "2020-01-02 00:00:00", "2020-01-02 01:00:00", "Drug X", "Administered")])
            self._write(raw / "prescriptions.csv.gz",
                        ("subject_id", "hadm_id", "pharmacy_id", "poe_id", "starttime", "stoptime", "drug_type", "drug", "gsn", "route"),
                        [("1", "10", "P1", "O1", "2020-01-03 12:00:00", "", "MAIN", "Drug X", "", "PO")])
            with (root / "base.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("subject_id", "hadm_id", "admittime", "dischtime", "age_at_admission",
                                 "gender", "discharge_location", "prior_coded_admission_count",
                                 "prior_strict_aecopd_count", "coded_admission_order", "event_type", "time_days"))
                writer.writerow(("1", "10", "2020-01-01 00:00:00", "2020-01-04 00:00:00",
                                 "60", "F", "HOME", "0", "0", "1", "death", "7"))
            result = build_dataset(raw, root / "notes.csv.gz", root / "built", root / "base.csv")
            self.assertEqual(result["cohort_admissions"], 1)
            self.assertEqual(result["facts"], 11)
            with gzip.open(root / "built" / "candidate_alignment.csv.gz", "rt", newline="") as handle:
                candidates = list(csv.DictReader(handle))
            self.assertIn("lab_to_pertinent_results", {row["rule_id"] for row in candidates})
            self.assertIn("emar_to_hospital_course", {row["rule_id"] for row in candidates})
            self.assertNotIn("prescription_as_confirmed_discharge_med", {row["rule_id"] for row in candidates})
            with gzip.open(root / "built" / "structured_facts.csv.gz", "rt", newline="") as handle:
                fact_rows = list(csv.DictReader(handle))
            self.assertNotIn("death", {row["value"] for row in fact_rows})
            self.assertNotIn("7", {row["value"] for row in fact_rows})
            diagnosis_keys = [row["source_row_key"] for row in fact_rows
                              if row["source_table"] == "mimiciv_hosp.diagnoses_icd"]
            self.assertEqual(len(set(diagnosis_keys)), 2)
            with gzip.open(root / "built" / "note_sections.csv.gz", "rt", newline="") as handle:
                sections = list(csv.DictReader(handle))
            self.assertEqual({row["timestamp_relation"] for row in sections}, {"after_discharge"})


class OldExportParityTests(unittest.TestCase):
    def test_lab_parity_detects_changed_extreme(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, old = root / "raw", root / "old"
            raw.mkdir()
            old.mkdir()
            SectionDatasetTests._write(
                raw / "labs.csv.gz",
                ("hadm_id", "itemid", "valuenum"),
                [("10", "50912", "1.2"), ("10", "50912", "1.5")],
            )
            with (old / "labs.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("hadm_id", "itemid", "measurement_count", "first_value",
                                 "last_value", "min_value", "max_value"))
                writer.writerow(("10", "50912", "2", "1.2", "1.5", "1.2", "1.5"))
            self.assertEqual(_labs_parity(raw, old).get("value_or_count_mismatch", 0), 0)
            with (old / "labs.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("hadm_id", "itemid", "measurement_count", "first_value",
                                 "last_value", "min_value", "max_value"))
                writer.writerow(("10", "50912", "2", "1.2", "1.5", "1.2", "1.7"))
            self.assertEqual(_labs_parity(raw, old)["value_or_count_mismatch"], 1)

    def test_diagnosis_after_disposition_is_its_own_section(self) -> None:
        note = (
            "Discharge Disposition:\nHome With Service\nFacility: ___\n"
            "Diagnosis:\nPrimary: COPD exacerbation\n"
            "Followup Instructions:\n___\n"
        )
        sections = parse_sections(note)
        self.assertEqual([section.role for section in sections], [
            "discharge_disposition", "discharge_diagnosis", "followup_instructions",
        ])
        self.assertNotIn("COPD", sections[0].text)
        self.assertEqual(sections[2].content_status, "placeholder")

    def test_ambiguous_history_cannot_be_claimed_as_social_history(self) -> None:
        note = "Social History:\n___\nHistory:\nFather died of stroke\nFamily History:\nUnknown\n"
        sections = parse_sections(note)
        self.assertEqual(sections[0].role, "social_history")
        self.assertEqual(sections[0].content_status, "placeholder")
        self.assertEqual(sections[1].role, "unclassified")
        self.assertEqual(sections[1].parse_status, "uncertain")
        self.assertEqual(sections[2].role, "family_history")

    def test_inline_heading_and_duplicate_section_keep_source_offsets(self) -> None:
        note = "Discharge Condition: Stable on room air\nDischarge Condition:\nAmbulatory with cane\n"
        sections = parse_sections(note)
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].text, "Stable on room air")
        self.assertEqual(sections[1].text, "Ambulatory with cane")
        self.assertEqual([section.ordinal for section in sections], [0, 1])
        for section in sections:
            self.assertEqual(note[section.start_char:section.end_char].strip(), section.text)


if __name__ == "__main__":
    unittest.main()
