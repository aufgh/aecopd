import unittest
from patient_data.discharge_regimen_audit import lexical,span,check_maps,covered


class RegimenInventoryTests(unittest.TestCase):
    def test_combination_mapping(self):
        self.assertEqual(lexical('Advair Diskus one puff BID')['classes'],['inhaled_steroid','laba'])
        self.assertEqual(lexical('Trelegy Ellipta inhale daily')['classes'],['inhaled_steroid','laba','lama'])

    def test_nasal_not_pulmonary(self):
        self.assertEqual(lexical('Fluticasone nasal spray daily')['classes'],[])

    def test_oral_and_bare_name(self):
        for s in ('Budesonide','Budesonide capsule PO daily','Glycopyrrolate tablet PO'):
            self.assertEqual(lexical(s)['classes'],[])

    def test_stop_and_condition_remain_candidates(self):
        v=lexical('Stop Advair Diskus if instructed')
        self.assertIn('stop_hold_words',v['flags']);self.assertIn('conditional_words',v['flags'])
        self.assertFalse(v['clinically_verified']);self.assertNotIn('prescribed',v)

    def test_dose_frequency_strings_not_values(self):
        v=lexical('Spiriva inhale 18 mcg daily')
        self.assertIn('dose_string_present',v['flags']);self.assertIn('frequency_string_present',v['flags'])
        self.assertNotIn('dose',v)

    def test_exact_offset_and_reverse_empty(self):
        span('abc',0,3,'abc')
        with self.assertRaises(ValueError):span('abc',2,1,'')

    def test_task_section_coverage(self):
        source=dict(text='abcdef',sections=[dict(role='discharge_medications',start_char=0,end_char=6)])
        task=dict(text='abc',source_maps=[dict(role='discharge_medications',original_start=0,original_end=3,input_start=0,input_end=3)])
        self.assertEqual(check_maps(source,task),[(0,3)])
        covered(source['sections'][0],[(0,3),(3,6)])
        for rr in ([(0,3),(4,6)],[(0,3),(2,6)],[(0,3)]):
            with self.assertRaises(ValueError):covered(source['sections'][0],rr)


if __name__=='__main__':unittest.main()
