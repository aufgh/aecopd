"""临床文本规则的时态、否定与未知值契约测试。"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from training.analyze_clinical_text_gold import analyze
from training.clinical_text_rules import (
    extract_discharge,
    extract_radiology,
    has_possible_post_discharge_fact,
)
from training.prepare_clinical_text_pilot import prepare


class ClinicalTextRuleTests(unittest.TestCase):
    def test_admission_wheeze_not_misread_as_discharge_wheeze(self) -> None:
        text = (
            "PHYSICAL EXAM:\nDiffuse wheezing.\n"
            "DISCHARGE CONDITION:\nLungs clear, no wheezing.\n"
        )
        self.assertEqual(extract_discharge(text)["residual_wheeze"].value, "0")

    def test_unmentioned_is_not_negative(self) -> None:
        result = extract_discharge("DISCHARGE CONDITION:\nStable.\n")
        self.assertEqual(result["discharge_oxygen_support"].value, "unknown")
        self.assertEqual(result["new_or_escalated_home_oxygen"].value, "unknown")
        self.assertEqual(result["discharge_mobility"].value, "unknown")

    def test_mobility_and_taper_are_section_anchored(self) -> None:
        text = (
            "DISCHARGE CONDITION:\nActivity Status: Ambulatory - Independent.\n"
            "DISCHARGE MEDICATIONS:\nPrednisone taper prescribed.\n"
        )
        result = extract_discharge(text)
        self.assertEqual(result["discharge_mobility"].value, "independent")
        self.assertEqual(result["discharge_steroid_taper"].value, "1")

    def test_post_discharge_is_flagged_not_deleted(self) -> None:
        text = (
            "DISCHARGE CONDITION:\nRoom air.\n"
            "BRIEF HOSPITAL COURSE:\nPost-discharge phone call was made.\n"
        )
        self.assertTrue(has_possible_post_discharge_fact(text))
        self.assertEqual(extract_discharge(text)["discharge_oxygen_support"].value, "0")
        self.assertEqual(extract_discharge(text)["new_or_escalated_home_oxygen"].value, "0")

    def test_oxygen_support_does_not_imply_new_oxygen(self) -> None:
        result = extract_discharge("DISCHARGE CONDITION:\nStable on 2 L nasal cannula.\n")
        self.assertEqual(result["discharge_oxygen_support"].value, "1")
        self.assertEqual(result["new_or_escalated_home_oxygen"].value, "unknown")

    def test_radiology_negation_and_unknown(self) -> None:
        self.assertEqual(
            extract_radiology("IMPRESSION:\nNo pulmonary edema.")["pulmonary_edema"].value,
            "0",
        )
        self.assertEqual(
            extract_radiology("FINDINGS:\nPersistent right lower lobe consolidation.")["persistent_consolidation"].value,
            "1",
        )
        self.assertEqual(
            extract_radiology("IMPRESSION:\nNo change.")["persistent_consolidation"].value,
            "unknown",
        )

    def test_gold_template_has_no_text_or_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "review"
            source.mkdir()
            manifest = []
            index = 0
            for document_type, cohort_group, count in (
                ("discharge", "other_coded", 5),
                ("discharge", "strict_index", 15),
                ("radiology", "strict_index", 4),
            ):
                for band in (1, 2, 3):
                    for _ in range(count):
                        index += 1
                        review_id = f"R{index:03d}"
                        filename = review_id + ".txt"
                        (source / filename).write_text(
                            "DISCHARGE CONDITION:\nStable.\n", encoding="utf-8"
                        )
                        manifest.append({
                            "review_id": review_id, "filename": filename,
                            "document_type": document_type,
                            "cohort_group": cohort_group, "length_band": str(band),
                        })
            with (source / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(manifest[0]))
                writer.writeheader()
                writer.writerows(manifest)
            summary = prepare(source, root / "pilot")
            self.assertEqual(summary["documents"], 24)
            gold = (root / "pilot" / "gold_template.csv").read_text(encoding="utf-8")
            self.assertNotIn("Stable.", gold)
            self.assertNotIn("event_type", gold)
            with self.assertRaisesRegex(ValueError, "manual gold incomplete"):
                analyze(root / "pilot" / "gold_template.csv", root / "pilot" / "rule_predictions.csv")

    def test_gold_analysis_classifies_rule_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.csv"
            rules_path = root / "rules.csv"
            with gold_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "review_id", "field", "gold_value", "gold_time",
                    "gold_assertion", "reviewer", "review_complete",
                ))
                writer.writeheader()
                writer.writerow({
                    "review_id": "R001", "field": "residual_wheeze", "gold_value": "1",
                    "gold_time": "discharge", "gold_assertion": "affirmed",
                    "reviewer": "reviewer-1", "review_complete": "1",
                })
            with rules_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("review_id", "field", "rule_value"))
                writer.writeheader()
                writer.writerow({"review_id": "R001", "field": "residual_wheeze", "rule_value": "unknown"})
            report = analyze(gold_path, rules_path)
            self.assertEqual(report["by_field"]["residual_wheeze"]["missed_information"], 1)


if __name__ == "__main__":
    unittest.main()
