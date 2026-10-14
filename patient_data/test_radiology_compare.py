import unittest
from patient_data.clinical_facts.radiology_compare import graph_chunks, graph_facts
from patient_data.clinical_facts.short_topics import candidates


class RadiologyComparisonTest(unittest.TestCase):
    def test_chunks_preserve_all_characters(self):
        text = ('A long imaging report.\n' * 500)
        chunks = graph_chunks(text)
        self.assertEqual(''.join(s for _, _, s in chunks), text)
        self.assertTrue(all(text[a:b] == s for a, b, s in chunks))

    def test_model_polarity_and_verbatim_offsets(self):
        source = 'No pleural effusion. Possible consolidation.'
        annotation = {'text': 'No pleural effusion . Possible consolidation .', 'entities': {
            '1': dict(tokens='pleural effusion', label='Observation::definitely absent', start_ix=1, end_ix=2, relations=[]),
            '2': dict(tokens='consolidation', label='Observation::uncertain', start_ix=5, end_ix=5, relations=[])}}
        cxr, ct, evidence = graph_facts(annotation, source)
        self.assertEqual(cxr['Pleural Effusion'], 2)
        self.assertEqual(cxr['Consolidation'], 3)
        self.assertTrue(all(source[e['start']:e['end']] == e['quote'] for e in evidence))

    def test_patent_artery_does_not_mean_enlarged(self):
        source = 'Patent pulmonary artery.'
        a = {'text': 'Patent pulmonary artery .', 'entities': {'1': dict(tokens='pulmonary artery', label='Observation::definitely present', start_ix=1, end_ix=2, relations=[])}}
        _, facts, _ = graph_facts(a, source)
        self.assertFalse(any(f['field'] == 'enlarged_pulmonary_artery' for f in facts))

    def test_modifier_polarity_not_copied_to_disease(self):
        source = 'Acute pneumonia.'
        a = {'text': 'Acute pneumonia .', 'entities': {
            '1': dict(tokens='Acute', label='Observation::definitely present', start_ix=0, end_ix=0, relations=[['modify', '2']]),
            '2': dict(tokens='pneumonia', label='Observation::uncertain', start_ix=1, end_ix=1, relations=[])}}
        cxr, _, evidence = graph_facts(a, source)
        self.assertEqual(cxr['Pneumonia'], 3)
        self.assertEqual(len(evidence), 1)

    def test_short_topic_maps_preserve_offsets(self):
        source = 'Kidney injury improved.\n\nAt discharge on home oxygen.'
        s = dict(text=source, role='hospital_course', start_char=100, end_char=100 + len(source))
        spans = candidates(s, 'respiratory')
        self.assertEqual(len(spans), 1)
        self.assertTrue(all(source[p['start_char'] - 100:p['end_char'] - 100] == p['text'] for p in spans))


if __name__ == '__main__': unittest.main()
