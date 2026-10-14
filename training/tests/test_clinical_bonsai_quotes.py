"""原文摘录允许换行差异，但不允许模型改写。"""

from __future__ import annotations

import json
import unittest

from training.clinical_bonsai_quotes import complete_note, parse_quotes
from training.clinical_discharge_sections import field_contexts, split_major_sections
from training.clinical_discharge_rules_v2 import FIELDS


class BonsaiQuoteTests(unittest.TestCase):
    def test_verbatim_multiline_quote(self) -> None:
        note = "At discharge,\nshe required 2 L oxygen.\n"
        response = {field: [] for field in FIELDS}
        response["oxygen_or_niv_at_discharge"] = ["At discharge, she required 2 L oxygen."]
        accepted, rejected = parse_quotes(json.dumps(response), note)
        self.assertEqual(rejected, {})
        self.assertEqual([(q.field, q.text) for q in accepted], [
            ("oxygen_or_niv_at_discharge", "At discharge, she required 2 L oxygen.")
        ])

    def test_paraphrase_rejects_just_its_field(self) -> None:
        note = "Discharge: Breathing comfortably.\nAmbulatory with assistance."
        response = {field: [] for field in FIELDS}
        response["residual_dyspnea"] = ["No shortness of breath at discharge."]
        response["mobility_at_discharge"] = ["Ambulatory with assistance."]
        accepted, rejected = parse_quotes(json.dumps(response), note)
        self.assertEqual([q.field for q in accepted], ["mobility_at_discharge"])
        self.assertEqual(rejected, {"residual_dyspnea": "not_verbatim_source"})

    def test_invalid_quote_does_not_keep_other_quotes_in_same_field(self) -> None:
        note = "At discharge breathing comfortably."
        response = {field: [] for field in FIELDS}
        response["residual_dyspnea"] = ["breathing comfortably.", "No wheeze."]
        accepted, rejected = parse_quotes(json.dumps(response), note)
        self.assertEqual(accepted, [])
        self.assertEqual(rejected, {"residual_dyspnea": "not_verbatim_source"})

    def test_complete_note_refuses_silent_truncation(self) -> None:
        self.assertEqual(complete_note("First\nSecond"), "First\nSecond")
        with self.assertRaises(ValueError):
            complete_note("x" * 30, max_chars=20)

    def test_section_filter_keeps_nested_problem_in_parent(self) -> None:
        note = (
            "History of Present Illness:\nDyspnea on admission.\n"
            "Brief Hospital Course:\nCOPD:\nAt discharge, breathing comfortably.\n"
            "Discharge Medications:\nAlbuterol inhaler daily.\n"
        )
        sections = split_major_sections(note)
        self.assertEqual([section.title for section in sections], [
            "history of present illness", "brief hospital course", "discharge medications",
        ])
        context, allowed = field_contexts(note)
        self.assertNotIn("Dyspnea on admission.", context)
        self.assertIn("COPD:\nAt discharge, breathing comfortably.", context)
        response = {field: [] for field in FIELDS}
        response["residual_dyspnea"] = ["At discharge, breathing comfortably."]
        response["inhaler_use_or_adherence"] = ["Albuterol inhaler daily."]
        accepted, rejected = parse_quotes(json.dumps(response), note, allowed)
        self.assertEqual([quote.field for quote in accepted], ["residual_dyspnea"])
        self.assertEqual(rejected, {"inhaler_use_or_adherence": "outside_allowed_section"})


if __name__ == "__main__":
    unittest.main()
