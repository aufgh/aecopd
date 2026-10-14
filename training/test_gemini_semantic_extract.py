import json
from pathlib import Path
import tempfile
import unittest

from training.gemini_semantic_extract import atomic_json, ingest, parse_json, prompt_for, split_batches
from training.gemini_field_catalog import FIELDS, field_table


class GeminiExtractionTests(unittest.TestCase):
    def test_fenced_and_pty_json(self):
        self.assertEqual(parse_json('\x1b[0m\r\n```json\n{"x":1}\n```'), {'x': 1})
        with self.assertRaises(ValueError):
            parse_json('{"x":')

    def test_budgets_never_drop_units(self):
        candidates = [{'unit_id': str(i), 'text': 'x'*1000, 'section': 'course',
                       'problem_heading': ''} for i in range(35)]
        batches = split_batches(candidates, 'course_results')
        self.assertEqual([c for batch in batches for c in batch], candidates)
        self.assertGreater(len(batches), 1)

    def test_exact_offsets_reject_invented_and_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            text = 'No wheezing. She improved with oxygen.'
            request = {'id': '00000_course_results_00', 'note_id': 'synthetic', 'subject_id': 1,
                       'hadm_id': 2, 'patient_split': 'train', 'source_sha256': 'test',
                       'prompt_sha256': 'test', 'route': 'course_results',
                       'candidates': [{'unit_id': 'U0000', 'text': text, 'start': 0, 'end': len(text),
                                       'section': 'brief_hospital_course', 'problem_heading': ''}]}
            atomic_json(output/'requests'/f"{request['id']}.json", request)
            atomic_json(output/'sources/00000.json', {'text': text})
            item = {'unit_id': 'U0000', 'quote': 'No wheezing.', 'fields': ['wheezing_respiratory_effort']}
            second = {**item, 'fields': ['dyspnea_activity_course']}
            data = {'fragments': [item, second, {**item, 'quote': 'Invented fever.'}]}
            envelope = {'status': 'SUCCESS', 'response': json.dumps(data), 'duration_seconds': 1}
            result = ingest(output, request['id'], {'output': json.dumps(envelope)})
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['fragments'], 1)
            saved = json.loads((output/'results'/f"{request['id']}.json").read_text())
            fragment = saved['fragments'][0]
            self.assertEqual(text[fragment['start']:fragment['end']], fragment['text'])
            self.assertEqual(fragment['fields'], ['wheezing_respiratory_effort', 'dyspnea_activity_course'])
            self.assertEqual(set(saved['fields']), set(FIELDS))
            self.assertEqual(saved['fields']['oxygen_requirement_course'], [])
            with self.assertRaises(ValueError):
                ingest(output, request['id'], {'output': json.dumps(envelope)})

    def test_catalog_is_complete_and_empty_is_not_normal(self):
        self.assertEqual(len(FIELDS), 16)
        self.assertEqual(field_table([]), {key: [] for key in FIELDS})
        prompt = prompt_for([], 'baseline')
        self.assertIn('not normal', prompt)
        self.assertIn('Admission', prompt)
        self.assertIn('NOT a hard filter', prompt)
        self.assertIn('default alert/coherent/independent', prompt)
        for key in FIELDS:
            self.assertIn(key, prompt)


if __name__ == '__main__':
    unittest.main()
