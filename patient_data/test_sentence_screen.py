import unittest
from patient_data.clinical_facts.sentence_screen import units, screen
from patient_data.clinical_facts.sentence_rules import classify


class SentenceScreenTests(unittest.TestCase):
    def test_exact_full_character_coverage(self):
        text = 'Dr. Smith prescribed prednisone 2.5 mg.\n1. Continue for 3 days.\n2. Stop thereafter.'
        spans = units(text)
        self.assertIn('Dr. Smith prescribed prednisone 2.5 mg.', [text[a:b] for a, b in spans])
        self.assertEqual(''.join(text[a:b] for a, b in spans).replace(' ', '').replace('\n', ''), text.replace(' ', '').replace('\n', ''))

    def test_negative_sentence_kept_without_forcing_polarity(self):
        r = classify('No pleural effusion or pneumothorax.', 'FINDINGS', 'cxr')
        self.assertTrue(r['selected']); self.assertIn('negation_words', r['context_cues'])
        self.assertNotIn('state', r)

    def test_devices_do_not_erase_disease(self):
        self.assertTrue(classify('ETT position normal but right basilar consolidation persists.', 'FINDINGS', 'cxr')['selected'])
        self.assertEqual(classify('ETT tip 2.5 cm above the carina.', 'FINDINGS', 'cxr')['reason'], 'device_only')

    def test_generic_warning_not_patient_status(self):
        self.assertFalse(classify('If you develop shortness of breath, call 911.', 'discharge_instructions', 'discharge')['selected'])
        self.assertTrue(classify('Patient still has dyspnea; if you develop worse breathing, call 911.', 'discharge_instructions', 'discharge')['selected'])

    def test_plan_not_completed(self):
        r = classify('Continue prednisone 40 mg for three more days.', 'discharge_medications', 'discharge')
        self.assertTrue(r['selected']); self.assertIn('plan_words', r['context_cues']); self.assertNotIn('completed', r)

    def test_dyspnea_is_not_desaturation(self):
        r = classify('Still dyspneic with exertion.', 'discharge_condition', 'discharge')
        self.assertEqual(r['topics'], ['respiratory_status'])

    def test_context_within_section_and_exact_offsets(self):
        original = 'Header\nShe has COPD. This improved.\nNext\nHome.'
        start = original.index('She'); end = original.index('\nNext')
        result = screen(original, [dict(role='hospital_course', start_char=start, end_char=end, text=original[start:end])], 'discharge')
        for r in result:
            self.assertEqual(original[r['start_char']:r['end_char']], r['quote'])
            self.assertLessEqual(r['context_end'], end)

    def test_invalid_reverse_offsets_rejected(self):
        with self.assertRaises(ValueError): screen('abc', [dict(role='x', start_char=2, end_char=1, text='')], 'discharge')

    def test_history_kept_separate_from_discharge(self):
        self.assertTrue(classify('History of COPD on home oxygen.', 'past_medical_history', 'discharge')['selected'])
        self.assertFalse(classify('Mother has COPD.', 'family_history', 'discharge')['selected'])

    def test_future_observed_outcome_excluded(self):
        self.assertFalse(classify('Subsequently readmitted for COPD.', 'hospital_course', 'discharge')['selected'])

    def test_explicit_normal_imaging_is_not_dropped(self):
        self.assertTrue(classify('No acute cardiopulmonary abnormality.', 'IMPRESSION', 'cxr')['selected'])

    def test_decimal_time_does_not_split(self):
        text = 'Oxygen at 2.5 L/min. FEV1 was 1.25 L.'
        self.assertEqual([text[a:b] for a, b in units(text)], ['Oxygen at 2.5 L/min.', 'FEV1 was 1.25 L.'])

    def test_numeric_bullets_remain_attached(self):
        text = '1. Severe emphysema.\n2. No pleural effusion.'
        self.assertEqual([text[a:b] for a, b in units(text)], ['1. Severe emphysema.', '2. No pleural effusion.'])

    def test_source_review_errors_regressions(self):
        self.assertTrue(classify('No areas of mucoid impaction.', 'FINDINGS', 'chest_ct')['selected'])
        self.assertTrue(classify('Re-expansion edema has cleared.', 'IMPRESSION', 'cxr')['selected'])
        self.assertFalse(classify('Right adrenal mass causes mass effect on liver.', 'IMPRESSION', 'chest_ct')['selected'])
        self.assertFalse(classify('Renal collection treated with 14d meropenem.', 'past_medical_history', 'discharge')['selected'])
        self.assertTrue(classify('Renal mass and right lower lobe consolidation.', 'FINDINGS', 'chest_ct')['selected'])

    def test_third_review_regressions(self):
        self.assertFalse(classify('Hydrocortisone Cream 2.5% TP TID PRN pruritis.', 'discharge_medications', 'discharge')['selected'])
        self.assertFalse(classify('Fluticasone nasal spray daily.', 'discharge_medications', 'discharge')['selected'])
        self.assertTrue(classify('Hydrocortisone cream for rash; COPD still symptomatic.', 'hospital_course', 'discharge')['selected'])
        self.assertTrue(classify('The ventilation at the left lung base is minimally improved.', 'FINDINGS', 'cxr')['selected'])
        self.assertTrue(classify('Home With Service', 'discharge_disposition', 'discharge')['selected'])

    def test_second_review_regressions(self):
        self.assertTrue(classify('Scarring at the right apex.', 'IMPRESSION', 'cxr')['selected'])
        self.assertTrue(classify('Increased opacification of the left hemithorax.', 'IMPRESSION', 'cxr')['selected'])
        self.assertFalse(classify('No definite colonic mass.', 'IMPRESSION', 'chest_ct')['selected'])
        self.assertFalse(classify('Vitals- BP 130/80 HR 80 O2 sat 96% on RA.', 'physical_exam', 'discharge')['selected'])
        self.assertTrue(classify('O2 sat 96% on RA after weaning oxygen.', 'discharge_condition', 'discharge')['selected'])


if __name__ == '__main__': unittest.main()
