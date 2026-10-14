import unittest

from training.clinical_evidence_checks import assess, gate_evidence


class ClinicalCheckTests(unittest.TestCase):
    def test_bare_copd_not_severity_but_episode_preserved(self):
        self.assertEqual(assess('baseline_respiratory_severity', 'COPD')[0], 'reject')
        self.assertEqual(assess('baseline_respiratory_severity', 'Severe COPD on home O2.')[0], 'pass')
        self.assertEqual(assess('episode_diagnosis', 'COPD: No evidence of exacerbation.')[0], 'pass')

    def test_routes(self):
        self.assertEqual(assess('maintenance_rescue_inhalers', 'Fluticasone NASAL 1 SPRY NU DAILY')[0], 'reject')
        self.assertEqual(assess('maintenance_rescue_inhalers', 'Ipratropium 0.06% NU DAILY')[0], 'reject')
        self.assertEqual(assess('maintenance_rescue_inhalers', 'Fluticasone-Salmeterol 1 INH IH BID')[0], 'pass')
        self.assertEqual(assess('maintenance_rescue_inhalers', 'Flonase NASAL, albuterol IH PRN')[0], 'needs_review')

    def test_support_indication(self):
        self.assertEqual(assess('ventilation_support', 'NRB then nasal cannula')[0], 'reject')
        self.assertEqual(assess('ventilation_support', 'NRB then BiPAP')[0], 'pass')

    def test_no_blanket_long_span_deletion(self):
        self.assertEqual(assess('dyspnea', 'Worsening dyspnea. ' * 100)[0], 'needs_review')

    def test_withheld_is_not_missing(self):
        quote = 'Fluticasone NASAL 1 SPRY NU DAILY'
        result = gate_evidence({'maintenance_rescue_inhalers': [{'quote': quote}]})
        self.assertEqual(result['rejected'][0]['quote'], quote)
        self.assertEqual(result['field_states']['maintenance_rescue_inhalers'], 'rejected_candidates')
        self.assertFalse(result['clinical_truth_validated'])


if __name__ == '__main__':
    unittest.main()
