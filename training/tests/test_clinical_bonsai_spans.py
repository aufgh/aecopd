"""Bonsai 2 片段抽取合同的轻量测试，不调用模型或读取病历。"""

from __future__ import annotations

import json
import unittest

from training.clinical_bonsai_spans import (
    FIELDS, parse_evidence, parse_evidence_partial, select_note_context,
)


class BonsaiSpanTests(unittest.TestCase):
    def test_discharge_lines_survive_long_unrelated_section(self) -> None:
        note = "HISTORY OF PRESENT ILLNESS:\n" + ("unrelated history\n" * 1500)
        note += "DISCHARGE CONDITION:\nNo dyspnea at rest.\nLives alone.\n"
        context = select_note_context(note, max_chars=2000)
        self.assertIn("No dyspnea at rest.", context)
        self.assertIn("Lives alone.", context)
        self.assertLessEqual(len(context), 2000)

    def test_only_existing_context_lines_are_accepted(self) -> None:
        note = "DISCHARGE CONDITION:\nNo dyspnea at rest.\n"
        response = {field: None for field in FIELDS}
        response["residual_dyspnea"] = 2
        span = parse_evidence(json.dumps(response), note, select_note_context(note))
        self.assertEqual((span[0].line, span[0].quote), (2, "No dyspnea at rest."))
        response["residual_dyspnea"] = 99
        with self.assertRaisesRegex(ValueError, "invalid source line"):
            parse_evidence(json.dumps(response), note, select_note_context(note))

    def test_partial_parser_keeps_other_fields_when_one_line_is_invalid(self) -> None:
        note = "DISCHARGE CONDITION:\nNo dyspnea at rest.\n\nActivity Status: Ambulatory.\n"
        response = {field: None for field in FIELDS}
        response["residual_dyspnea"] = 2
        response["residual_cough_sputum"] = 3
        response["mobility_at_discharge"] = 4
        spans, rejected = parse_evidence_partial(
            json.dumps(response), note, select_note_context(note)
        )
        self.assertEqual({span.field for span in spans},
                         {"residual_dyspnea", "mobility_at_discharge"})
        self.assertEqual(rejected, {"residual_cough_sputum": "blank_source_line"})


if __name__ == "__main__":
    unittest.main()
