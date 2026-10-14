import unittest

from training.bonsai_catalog16 import FIELDS, TASKS, candidate_sources, make_payload, field_status


class Catalog16Tests(unittest.TestCase):
    def test_unique_ownership(self):
        keys = [k for _, group in TASKS for k in group]
        self.assertEqual(len(keys), 16)
        self.assertEqual(set(keys), {f.key for f in FIELDS})
        self.assertEqual(len(set(keys)), len(keys))

    def test_source_offsets_and_blank_paragraph(self):
        note = 'Pertinent Results:\nWBC-5\n\nCXR: Small pleural effusion.\n'
        candidates, _ = candidate_sources(note, 'chest_imaging_findings')
        self.assertTrue(candidates)
        for s in candidates:
            self.assertEqual(note[s['start']:s['end']], s['text'])
            self.assertNotIn('WBC', s['text'])

    def test_no_hard_section_gate(self):
        note = 'Imaging:\nAt discharge she was weaned to room air.\n'
        candidates, _ = candidate_sources(note, 'oxygen_trajectory_discharge')
        self.assertTrue(candidates)

    def test_indication_context_retained(self):
        note = 'Brief Hospital Course:\n# UTI\nCiprofloxacin was given for complicated UTI.\n\nDischarge Medications:\n1. Ciprofloxacin 500 mg PO BID.\n'
        candidates, _ = candidate_sources(note, 'respiratory_antibiotic_course')
        self.assertTrue(candidates)
        self.assertIn('UTI', '\n'.join(s['context'] for s in candidates))

    def test_no_candidate_not_normal(self):
        self.assertEqual(field_status([], [], []), 'NO_CANDIDATE')

    def test_schema_exact_quote_only(self):
        note = 'Brief Hospital Course:\nAt discharge she was on room air.\n'
        sources, _ = candidate_sources(note, 'oxygen_trajectory_discharge')
        payload = make_payload(('oxygen_trajectory_discharge',), {'oxygen_trajectory_discharge': sources})
        schema = payload['response_format']['schema']
        self.assertEqual(schema['required'], ['oxygen_trajectory_discharge'])
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual(schema['properties']['oxygen_trajectory_discharge']['items']['enum'], [s['text'] for s in sources])


if __name__ == '__main__':
    unittest.main()
