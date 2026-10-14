import unittest

from training.benchmark_bonsai_sections import TASKS, section_sources, task_sources
from training.clinical_text_catalog import FIELDS


class SectionTests(unittest.TestCase):
    def test_fixed_fields_have_one_owner(self):
        keys = [k for _, ks, _ in TASKS for k in ks]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys), {f.key for f in FIELDS})

    def test_note_without_heading_preserved(self):
        text = 'Dyspnea worse. No standard headings.'
        parts = section_sources(text)
        self.assertEqual(parts[0]['text'], text)
        self.assertEqual(parts[0]['role'], 'unclassified')

    def test_original_offsets_and_nonwhitespace_coverage(self):
        text = 'Leading clinical text\nHistory of Present Illness:\nDyspnea.\nUnknown:\nMore.\nDischarge Medications:\nAlbuterol.'
        parts = section_sources(text)
        covered = set()
        for s in parts:
            self.assertEqual(text[s['start']:s['end']], s['text'])
            covered.update(range(s['start'], s['end']))
        self.assertTrue(all(i in covered for i, c in enumerate(text) if not c.isspace()))

    def test_problem_headings_preserve_nonrespiratory_indication(self):
        text = 'Brief Hospital Course:\n# COPD: Improved.\n# Cholangitis: Antibiotics.\n'
        parts = section_sources(text)
        self.assertTrue(any('Cholangitis' in s['text'] for s in parts))
        self.assertTrue(any(s.get('problem_heading', '').startswith('# COPD') for s in parts))

    def test_medication_task_has_indication_context(self):
        _, _, roles = next(t for t in TASKS if t[0] == 'treatment')
        self.assertIn('hospital_course', roles)
        self.assertIn('discharge_medications', roles)

    def test_unknown_source_goes_to_fallback(self):
        sources = [{'role': 'unclassified'}, {'role': 'social_history'}, {'role': 'discharge_medications'}]
        self.assertEqual(task_sources(sources, ('social_history',)), sources[:2])


if __name__ == '__main__':
    unittest.main()
