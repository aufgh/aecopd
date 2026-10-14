import unittest
from patient_data.pft_evidence import bindings, repeat_stats


class Tests(unittest.TestCase):
    def test_volume_and_annotated_predicted_separate(self):
        r=bindings('FEV1 1.2 L (40% predicted), FVC 2.4 L')[0]
        self.assertEqual([q['kind'] for q in r['quantities']],['volume_l','percent_predicted'])
        self.assertEqual([q['value'] for q in r['quantities']],[1.2,40])

    def test_unlabelled_annotation_not_assumed_predicted(self):
        q=bindings('FEV1 1.2 L (40%)')[0]['quantities'][1]
        self.assertEqual(q['kind'],'percent_unspecified')

    def test_ratio_not_fev1_predicted(self):
        r=bindings('FEV1/FVC 65% predicted')[0]
        self.assertEqual(r['quantities'][0]['kind'],'ratio_percent_predicted')
        self.assertFalse(r['scalar_binding_eligible'])

    def test_raw_ratio_percent_bound_explicit(self):
        self.assertTrue(bindings('FEV1/FVC 65%')[0]['scalar_binding_eligible'])

    def test_unlabelled_ratio_not_automatic_scaled(self):
        r=bindings('FEV1/FVC .65')[0]
        self.assertEqual(r['quantities'][0]['value'],.65)
        self.assertFalse(r['scalar_binding_eligible'])

    def test_table_columns_rejected(self):
        self.assertEqual(bindings('FEV1 3.1 1.4 45')[0]['status'],'unlabeled_numeric_columns')

    def test_dates_not_measurements(self):
        for text in ('FEV1 2112-03-05','FEV1 03/05/2112','FEV1 2112'):
            self.assertEqual(bindings(text)[0]['status'],'date_like_number')

    def test_bounds_ranges_not_scalar(self):
        self.assertEqual(bindings('FEV1 <30% predicted')[0]['status'],'bound_not_scalar')
        self.assertEqual(bindings('FEV1 30-40%')[0]['status'],'range_ambiguous')

    def test_no_unit_borrowing_or_annotation_across_metric(self):
        r=bindings('FEV1 0.9 FVC 2 L (50% predicted)')[0]
        self.assertEqual(len(r['quantities']),1)
        self.assertEqual(r['quantities'][0]['kind'],'unit_unknown')

    def test_percent_predicted_words_bound_adjacent(self):
        r=bindings('FEV1 40% of predicted')[0]
        self.assertTrue(r['scalar_binding_eligible'])

    def test_volume_ml_not_l(self):
        self.assertEqual(bindings('FEV1 1200 mL')[0]['quantities'][0]['kind'],'volume_ml')

    def test_liter_unit_not_ratio(self):
        self.assertEqual(bindings('FEV1/FVC 0.6 L')[0]['status'],'volume_unit_metric_mismatch')

    def test_absent_measurement_and_plan(self):
        self.assertEqual(bindings('FEV1 will be measured tomorrow')[0]['status'],'no_adjacent_numeric')

    def test_repeated_same_patient_different_admissions_only(self):
        r=[dict(subject_id=1,hadm_id=1,q='x'),dict(subject_id=1,hadm_id=1,q='x'),
           dict(subject_id=2,hadm_id=2,q='x')]
        self.assertEqual(repeat_stats(r,'q')['cross_admission_groups'],0)
        r.append(dict(subject_id=1,hadm_id=3,q='x'))
        self.assertEqual(repeat_stats(r,'q'),dict(cross_admission_groups=1,patients=1,admissions=2))


if __name__=='__main__':unittest.main()
