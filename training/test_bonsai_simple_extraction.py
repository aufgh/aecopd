import unittest

from training.bonsai_simple_extraction import FIELDS, TASKS, build_job
from training.bonsai_clinical_evidence_pilot import validate_evidence


class SimpleExtractionTests(unittest.TestCase):
    def test_six_concrete_fields_owned_once(self):
        keys = [key for _, group, _ in TASKS for key in group]
        self.assertEqual(len(keys), 6)
        self.assertEqual(len(set(keys)), 6)
        self.assertEqual(set(keys), {f.key for f in FIELDS})

    def test_routing_original_not_truncated(self):
        row = {'review_id': 'synthetic', 'text': 'History of Present Illness:\nCough worsened.\nDischarge Medications:\nAlbuterol IH BID.'}
        job = build_job(row, TASKS[1])
        self.assertTrue(any('Cough worsened.' in s['text'] for s in job['sources']))
        self.assertTrue(any('Albuterol IH BID.' in s['text'] for s in job['sources']))
        for s in job['sources']:
            self.assertEqual(s['text'], row['text'][s['start']:s['end']])

    def test_nonverbatim_is_rejected(self):
        row = {'review_id': 'synthetic', 'text': 'Discharge Medications:\nAlbuterol IH BID.'}
        job = build_job(row, TASKS[1])
        evidence, issues = validate_evidence({'inhaled_medications': ['Albuterol IH DAILY.']},
                                             job['sources'], row['text'], job['fields'])
        self.assertFalse(evidence['inhaled_medications'])
        self.assertEqual(issues[0]['type'], 'quote_absent')

    def test_state_keeps_course_under_imaging_heading(self):
        row = {'review_id': 'synthetic', 'text': 'Imaging:\nCXR: LLL opacity.\n# CAP: Weaned from 4L oxygen to room air.\nDischarge Medications:\nAzithromycin.'}
        job = build_job(row, TASKS[0])
        self.assertTrue(any('Weaned from 4L oxygen to room air.' in s['text'] for s in job['sources']))

    def test_candidate_offsets_and_context(self):
        from training.bonsai_candidate_probe import candidates
        text = 'History of Present Illness:\nShe was given oxygen in the field.\nShe had cough.\nDischarge Condition:\nBreathing on room air without supplemental oxygen.'
        parts = candidates(text, 'oxygen_support')
        self.assertEqual(len(parts), 2)
        for part in parts:
            self.assertEqual(part['text'], text[part['start']:part['end']])
        self.assertTrue(any('without supplemental oxygen' in p['text'] for p in parts))

    def test_candidate_filter_not_vitals_or_ecg(self):
        from training.bonsai_candidate_probe import candidates
        self.assertFalse(candidates('Physical Exam:\nO2 99% 2L.', 'oxygen_support'))
        self.assertFalse(candidates('Pertinent Results:\nThe ST segments showed LVH on ECG.', 'chest_abnormality'))

    def test_candidate_lab_and_image_do_not_merge(self):
        from training.bonsai_candidate_probe import candidates
        text = 'Pertinent Results:\r\r\nWBC-19.1 HGB-13.0\r\r\n\r\r\nIMPRESSION: New left opacity concerning for pneumonia.'
        parts = candidates(text, 'chest_abnormality')
        self.assertEqual(len(parts), 1)
        self.assertNotIn('WBC', parts[0]['text'])
        self.assertIn('concerning for pneumonia', parts[0]['text'])


if __name__ == '__main__':
    unittest.main()
