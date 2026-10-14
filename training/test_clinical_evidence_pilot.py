"""结构与逐字溯源测试不等于临床抽取准确率。"""
import unittest

from training.bonsai_clinical_evidence_pilot import (
    decode_content, find_source_spans, local_api, sources_for_note, validate_evidence,
)
from training.clinical_text_catalog import FIELDS


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.original = 'At discharge: improved\r\n dyspnea, but still on 3 L oxygen.'
        self.sources = sources_for_note(self.original, radiology=True)
        self.data = {f.key: [] for f in FIELDS}

    def add(self, quote):
        self.data['dyspnea'] = [quote]

    def test_whitespace_and_offsets(self):
        self.add('improved dyspnea, but still on 3 L oxygen.')
        accepted, errors = validate_evidence(self.data, self.sources, self.original)
        self.assertEqual(errors, [])
        item = accepted['dyspnea'][0]
        match = item['source_matches'][0]
        self.assertEqual(item['quote'], self.original[match['start_char']:match['end_char']])
        self.assertEqual(item['clinical_review_status'], 'not_reviewed')

    def test_paraphrase_rejected(self):
        self.add('dyspnea resolved')
        accepted, errors = validate_evidence(self.data, self.sources, self.original)
        self.assertEqual(accepted['dyspnea'], [])
        self.assertEqual(errors[0]['type'], 'quote_absent')

    def test_unknown_field_invalidates_schema(self):
        self.data['invented'] = []
        self.assertEqual(validate_evidence(self.data, self.sources, self.original)[1][0]['type'], 'schema_keys')

    def test_table_is_created_by_code_not_model(self):
        accepted, errors = validate_evidence({}, self.sources, self.original)
        self.assertFalse(errors)
        self.assertEqual(set(accepted), {f.key for f in FIELDS})

    def test_model_metadata_not_accepted(self):
        self.data['dyspnea'] = [{'quote': 'improved', 'time_role': 'discharge'}]
        self.assertTrue(validate_evidence(self.data, self.sources, self.original)[1])

    def test_ambiguous_quote(self):
        self.assertEqual(len(find_source_spans('improved', 'improved, improved')), 2)

    def test_no_ellipsis_substitution(self):
        self.assertEqual(find_source_spans('improved ... oxygen.', self.original), [])

    def test_invalid_source_offsets_rejected(self):
        self.sources[0]['start'] = 2
        with self.assertRaises(ValueError):
            validate_evidence(self.data, self.sources, self.original)

    def test_repeated_original_preserves_all_locations(self):
        original = 'Dyspnea improved. Dyspnea improved.'
        self.add('Dyspnea improved.')
        accepted, errors = validate_evidence(self.data, sources_for_note(original, True), original)
        self.assertFalse(errors)
        self.assertEqual(len(accepted['dyspnea'][0]['source_matches']), 2)

    def test_remote_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            local_api('https://example.com', '/v1/chat/completions', {})

    def test_malformed_json_not_empty_success(self):
        with self.assertRaises(ValueError):
            decode_content('{"dyspnea":')

    def test_duplicate_keys_cannot_silently_erase_evidence(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON field'):
            decode_content('{"dyspnea":["Dyspnea"],"dyspnea":[]}')

    def test_valid_object_still_decodes(self):
        self.assertEqual(decode_content('{"dyspnea":["Dyspnea"]}'), {'dyspnea': ['Dyspnea']})

    def test_exact_quote_can_still_be_semantically_wrong(self):
        # 自动匹配不能证明字段类别或完整否定语境正确。
        self.add('improved')
        self.assertFalse(validate_evidence(self.data, self.sources, self.original)[1])


if __name__ == '__main__':
    unittest.main()
