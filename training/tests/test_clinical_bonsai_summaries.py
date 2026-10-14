"""Bonsai2 语义摘要输出既保留综合表述，又限制证据行号。"""

from __future__ import annotations

import json
import unittest

from training.clinical_bonsai_summaries import numbered_note, parse_summaries
from training.clinical_discharge_rules_v2 import FIELDS


class BonsaiSemanticSummariesTests(unittest.TestCase):
    def test_synthesis_is_accepted_without_copying_source_sentence(self) -> None:
        note = "DISCHARGE CONDITION:\nAt discharge, she was comfortable at rest.\nShe had dyspnea on walking.\n"
        response = {field: None for field in FIELDS}
        response["residual_dyspnea"] = {
            "summary": "At discharge, no dyspnea at rest but breathlessness with walking.",
            "evidence_lines": [2, 3],
        }
        accepted, rejected = parse_summaries(json.dumps(response), note)
        self.assertEqual(rejected, {})
        self.assertEqual(len(accepted), 1)
        self.assertNotIn(accepted[0].summary, note)
        self.assertEqual(accepted[0].evidence_lines, (2, 3))

    def test_invalid_line_rejects_only_its_field(self) -> None:
        note = "DISCHARGE CONDITION:\nBreathing comfortably.\nAmbulatory with assistance.\n"
        response = {field: None for field in FIELDS}
        response["residual_dyspnea"] = {
            "summary": "At discharge, breathing comfortably.", "evidence_lines": [99],
        }
        response["mobility_at_discharge"] = {
            "summary": "At discharge, ambulation required assistance.", "evidence_lines": [3],
        }
        accepted, rejected = parse_summaries(json.dumps(response), note)
        self.assertEqual([item.field for item in accepted], ["mobility_at_discharge"])
        self.assertEqual(rejected, {"residual_dyspnea": "invalid_evidence_line"})

    def test_complete_context_refuses_silent_truncation(self) -> None:
        self.assertEqual(numbered_note("First\nSecond"), "L1: First\nL2: Second")
        with self.assertRaises(ValueError):
            numbered_note("x" * 30, max_chars=20)


if __name__ == "__main__":
    unittest.main()
