"""多来源药物和紧凑临床概念的边界；不读取患者文本、不训练。"""
from datetime import datetime, timedelta
import unittest
import numpy as np
from patient_data.compact_multisource import order_available, order_groups, bacteria_group, compact_support, compact_micro_scope


class CompactTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2100, 1, 1)
        self.end = self.start + timedelta(days=5)
        self.row = {'starttime': str(self.start), 'stoptime': str(self.end),
                    'entertime': str(self.start), 'status': 'Discontinued via patient discharge'}

    def test_historical_discontinuation_is_not_cancellation(self):
        self.assertTrue(order_available(self.row, (self.start, self.end))[0])
        self.assertFalse(order_available({**self.row, 'status': 'H'}, (self.start, self.end))[0])
        self.assertFalse(order_available({**self.row, 'status': 'Cancelled'}, (self.start, self.end))[0])

    def test_missing_entry_time_not_replaced_by_start(self):
        self.assertFalse(order_available({**self.row, 'entertime': r'\N'}, (self.start, self.end))[0])
        self.assertTrue(order_available({**self.row, 'entertime': r'\N'}, (self.start, self.end), entered=self.start)[0])

    def test_future_and_pre_admission_intervals(self):
        self.assertFalse(order_available({**self.row, 'entertime': str(self.end+timedelta(hours=1))}, (self.start, self.end))[0])
        self.assertFalse(order_available({**self.row, 'starttime': str(self.end+timedelta(hours=1)), 'stoptime': r'\N'}, (self.start, self.end))[0])
        # 入院前开立且延续到本次住院的医嘱仍有交集。
        self.assertTrue(order_available({**self.row, 'starttime': str(self.start-timedelta(days=2))}, (self.start, self.end))[0])
        self.assertFalse(order_available({**self.row, 'stoptime': str(self.start-timedelta(hours=1))}, (self.start, self.end))[0])

    def test_invalid_stop_and_negative_interval(self):
        self.assertFalse(order_available({**self.row, 'stoptime': 'bad'}, (self.start, self.end))[0])
        self.assertFalse(order_available({**self.row, 'starttime': str(self.end), 'stoptime': str(self.start)}, (self.start, self.end))[0])

    def test_combo_classes_and_route_guard(self):
        self.assertEqual(order_groups('DuoNeb', 'NEB'), {0, 1})
        self.assertEqual(order_groups('Trelegy Ellipta', 'IH'), {2, 3, 4})
        self.assertEqual(order_groups('budesonide', 'PO'), set())
        self.assertEqual(order_groups('budesonide', 'IH'), {4})
        self.assertEqual(order_groups('hydrocortisone cream', 'TP'), set())
        self.assertEqual(order_groups('hydrocortisone', 'IV'), {5})
        self.assertEqual(order_groups('azithromycin', 'PO/NG'), {6})
        self.assertEqual(order_groups('gentamicin', 'IV'), set())
        self.assertEqual(order_groups('azithromycin', 'NEB'), set())
        self.assertEqual(order_groups('prednisone', r'\N'), set())

    def test_bacterial_compression_and_no_negation_false_positive(self):
        for org in ('HAEMOPHILUS INFLUENZAE, BETA-LACTAMASE NEGATIVE', 'STREPTOCOCCUS PNEUMONIAE', 'MORAXELLA CATARRHALIS'):
            self.assertEqual(bacteria_group(org), 0)
        self.assertEqual(bacteria_group('PSEUDOMONAS AERUGINOSA'), 1)
        self.assertEqual(bacteria_group('STAPH AUREUS COAG +'), 2)
        for org in ('NON-FERMENTER, NOT PSEUDOMONAS AERUGINOSA', 'GRAM NEGATIVE ROD(S)', 'YEAST', 'MIXED BACTERIAL FLORA', r'\N'):
            self.assertIsNone(bacteria_group(org))

    def test_targeted_culture_does_not_prove_routine_bacteria_assessed(self):
        row = {'spec_type_desc': 'SPUTUM', 'test_name': 'RESPIRATORY CULTURE'}
        self.assertEqual(compact_micro_scope(row), 'bacteria')
        for test in ('FUNGAL CULTURE', 'ACID FAST CULTURE', 'LEGIONELLA CULTURE', 'GRAM STAIN'):
            self.assertEqual(compact_micro_scope({**row, 'test_name': test}), 'excluded_targeted_or_nonculture_test')
        self.assertEqual(compact_micro_scope({**row, 'test_name': 'Respiratory Viral Antigen Screen'}), 'virus')

    def test_support_union_and_oxygen_threshold(self):
        s = np.zeros((4, 9), np.float32); sm = np.zeros_like(s, bool)
        s[0, 2] = 1; sm[0, 2] = True
        s[1, 3] = 1; sm[1, 3] = True
        v = np.zeros((4, 8, 5)); vm = np.zeros_like(v, bool)
        v[2, 5, 2] = 21; vm[2, 5, 2] = True
        v[3, 5, 2] = 30; vm[3, 5, 2] = True
        out, masks = compact_support(s, sm, v, vm, [])
        self.assertEqual(out[0, 0], 1); self.assertEqual(out[1, 1], 1)
        self.assertEqual(out[2, 2], 0); self.assertTrue(masks[2, 2])
        self.assertEqual(out[3, 2], 1)

    def test_cpap_trach_and_brand_are_not_niv_or_invasive(self):
        s = np.zeros((3, 9), np.float32); sm = np.zeros_like(s, bool)
        s[0, 4] = 1; sm[0, 4] = True; s[1, 6] = 1; sm[1, 6] = True
        v = np.zeros((3, 8, 5)); vm = np.zeros_like(v, bool)
        out, masks = compact_support(s, sm, v, vm, [(0, '223849', 'CPAP/PSV'), (1, '223848', 'Drager'), (2, '229314', 'NIV-ST')])
        self.assertFalse(out[:2, :2].any()); self.assertFalse(masks[:2, :2].any())
        self.assertEqual(out[2, 1], 1); self.assertTrue(masks[2, 1])

    def test_missing_support_not_normal(self):
        out, masks = compact_support(np.zeros((1, 9)), np.zeros((1, 9), bool), np.zeros((1, 8, 5)), np.zeros((1, 8, 5), bool), [])
        self.assertFalse(out.any()); self.assertFalse(masks.any())


if __name__ == '__main__': unittest.main()
