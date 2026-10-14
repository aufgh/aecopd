import unittest

from training.bonsai_catalog16_guarded import guarded_sources


class GuardedTests(unittest.TestCase):
    def test_admission_and_prn_not_residual(self):
        note = 'History of Present Illness:\nShe has SOB worse than baseline.\n\nDischarge Medications:\n1. Albuterol IH PRN sob.\n'
        sources, _ = guarded_sources(note, 'discharge_residual_dyspnea')
        self.assertFalse(sources)

    def test_discharge_exam_line_not_whole_exam(self):
        note = 'Physical Exam:\nDISCHARGE PHYSICAL EXAM:\nVITALS: Stable\nHEENT: Normal\nLUNGS: Breathing comfortably on RA\nNECK: Supple.\n'
        sources, _ = guarded_sources(note, 'oxygen_trajectory_discharge')
        self.assertTrue(sources)
        self.assertEqual(sources[0]['text'], 'LUNGS: Breathing comfortably on RA')
        self.assertEqual(note[sources[0]['start']:sources[0]['end']], sources[0]['text'])

    def test_catch_breath_retrieved_in_discharge_subsection(self):
        note = 'Physical Exam:\nAdmission\nGeneral: No acute distress.\n\nDischarge\nGeneral: pauses intermittently when speaking to catch breath, improved from prior\n'
        sources, _ = guarded_sources(note, 'discharge_residual_dyspnea')
        self.assertTrue(sources)
        self.assertIn('catch breath', sources[0]['text'])

    def test_generic_adherence_not_inhaler_assessment(self):
        note = 'History of Present Illness:\nShe reports medication adherence.\n\nDischarge Instructions:\nPlease remember to take inhalers as prescribed.\n'
        sources, _ = guarded_sources(note, 'inhaler_technique_adherence')
        self.assertFalse(sources)

    def test_respiratory_antibiotic_nonadherence_retained(self):
        note = 'History of Present Illness:\nShe was started on Azithromycin; however, she did not tolerate this due to GI upset and did not take it as prescribed.\n'
        sources, _ = guarded_sources(note, 'inhaler_technique_adherence')
        self.assertTrue(sources)

    def test_next_item_number_not_attached(self):
        note = 'Discharge Medications:\n1. Albuterol IH PRN sob\n2. Aspirin PO daily.\n'
        sources, _ = guarded_sources(note, 'inhaled_medication_changes')
        self.assertTrue(sources)
        self.assertNotIn('2.', sources[0]['text'])


if __name__ == '__main__':
    unittest.main()
