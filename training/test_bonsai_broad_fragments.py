import json
import unittest
from unittest.mock import patch

from training.bonsai_broad_fragments import TOPICS, PATTERNS, units, payload, extract_note, refined_fragments


class BroadFragmentsTests(unittest.TestCase):
    def test_admission_symptom_allowed_and_no_discharge_claim(self):
        note = 'History of Present Illness:\nThree days of worsening shortness of breath.\n'
        items = units(note)
        self.assertTrue(any(PATTERNS['respiratory_course'].search(u['text']) for u in items))
        self.assertIn('Admission symptoms ARE allowed', payload('respiratory_course', items)['messages'][0]['content'])

    def test_offsets_unicode_and_long_source(self):
        note = 'Brief Hospital Course:\n# COPD\n'+('Oxygen was weaned.\u0085 '+ 'x'*350+'\n')*15
        items = units(note)
        self.assertTrue(items)
        for item in items:
            self.assertLessEqual(len(item['text']), 1000)
            self.assertEqual(note[item['start']:item['end']], item['text'])
        observed = ''.join(u['text'] for u in items)
        self.assertEqual(observed.count('x'), note.count('x'))

    def test_one_original_multi_topic_duplicate_collapsed(self):
        text = 'Brief Hospital Course:\nCOPD improved with albuterol and oxygen.\n'
        def fake_api(base, endpoint, body=None):
            quotes = [s['text'] for s in json.loads(body['messages'][1]['content'])]
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'quotes': quotes+quotes})}}]}
        row = {'text': text, 'hadm_id': 1, 'subject_id': 2, 'source_sha256': 'test'}
        with patch('training.bonsai_broad_fragments.batches', side_effect=lambda t,c,n: [c] if c else []), patch('training.bonsai_broad_fragments.local_api', side_effect=fake_api):
            result = extract_note(row, 8192)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['fragments']), 1)
        self.assertGreaterEqual(len(result['fragments'][0]['topics']), 3)

    def test_hallucination_rejected_but_good_quote_survives(self):
        text = 'Chief Complaint:\nShortness of breath.\n'
        def fake_api(base, endpoint, body=None):
            candidates = json.loads(body['messages'][1]['content'])
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'quotes': [candidates[0]['text'], 'Invented disease']})}}]}
        with patch('training.bonsai_broad_fragments.batches', side_effect=lambda t,c,n: [c] if c else []), patch('training.bonsai_broad_fragments.local_api', side_effect=fake_api):
            result = extract_note({'text': text}, 8192)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(len(result['fragments']), 1)
        self.assertEqual(result['hard_failures'][0]['type'], 'unverifiable_quote')

    def test_truncated_not_empty_normal(self):
        with patch('training.bonsai_broad_fragments.batches', side_effect=lambda t,c,n: [c] if c else []), patch('training.bonsai_broad_fragments.local_api', return_value={'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]}):
            result = extract_note({'text': 'Chief Complaint:\nShortness of breath.\n'}, 8192)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['topic_status']['respiratory_course'], 'PARTIAL_OR_FAILED_NOT_MISSING')

    def test_prompt_exact_original_and_soft_uncertainty(self):
        for topic in TOPICS:
            body = payload(topic, [{'text': 'No wheezing at discharge.', 'start': 0, 'end': 24}])
            self.assertEqual(body['response_format']['schema']['properties']['quotes']['items']['enum'], ['No wheezing at discharge.'])
            self.assertIn('Overlap with other topics is allowed', body['messages'][0]['content'])
            self.assertTrue(body['chat_template_kwargs']['enable_thinking'])

    def test_head_ct_baseline_not_respiratory_candidate(self):
        note = 'Brief Hospital Course:\nHead CT remained stable with baseline deltoid weakness.\n'
        self.assertFalse(any(PATTERNS['respiratory_course'].search(u['text']) for u in units(note)))

    def test_sentence_refinement_does_not_copy_whole_neuro_paragraph(self):
        note = 'Brief Hospital Course:\nShe has COPD and CHF. Head CT showed a contusion.\n'
        section = [u for u in units(note) if 'COPD' in u['text']][0]
        first = section['start']
        last = note.index('\n', first)
        old = {'start': first, 'end': last, 'text': note[first:last], 'topics': ['respiratory_course']}
        refined = refined_fragments([old], note)
        self.assertTrue(refined)
        self.assertNotIn('Head CT', '\n'.join(f['text'] for f in refined))

    def test_refined_context_never_overlaps_twice(self):
        note = 'Brief Hospital Course:\nThe patient was ambulating with assistance. The patient uses home oxygen.\n'
        a = note.index('The patient')
        b = len(note)-1
        source = {'start': a, 'end': b, 'text': note[a:b], 'topics': ['respiratory_support', 'comorbidity_function_care']}
        refined = refined_fragments([source], note)
        self.assertTrue(refined)
        for left, right in zip(refined, refined[1:]):
            self.assertLessEqual(left['end'], right['start'])

    def test_unrelated_medication_list_and_routine_normal_not_encoded(self):
        note = 'Discharge Medications:\nzoloft\nlasix\naspirin 81 qd\n'
        u = units(note)[0]
        self.assertEqual(refined_fragments([{**u, 'topics': ['comorbidity_function_care']}], note), [])
        note = 'Physical Exam:\nLungs: CTA bilaterally.\n'
        u = units(note)[0]
        self.assertEqual(refined_fragments([{**u, 'topics': ['respiratory_course']}], note), [])


if __name__ == '__main__':
    unittest.main()
