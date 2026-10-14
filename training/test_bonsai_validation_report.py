import unittest

from training.report_bonsai_validation import summarize


class AuditedReportTests(unittest.TestCase):
    def test_focused_quote_comparison_preserves_whitespace_equivalence_and_field(self):
        result = {'review_id': 'R', 'reasoning': 'none', 'finish_reason': 'stop',
                  'end_to_end_seconds': 1, 'evidence': {
                      'dyspnea': [{'quote': 'Dyspnea\r\n improved.'}],
                      'respiratory_recovery': [{'quote': 'Dyspnea\r\n improved.'}]}}
        entries = [dict(review_id='R', reasoning='none', field='dyspnea',
                        quote='Dyspnea improved.', source_found=True, clinical_grade='correct'),
                   dict(review_id='R', reasoning='none', field='respiratory_recovery',
                        quote='Dyspnea improved.', source_found=True, clinical_grade='incorrect')]
        protocol = {'cases': {'R': {'positive': [
            {'field': 'dyspnea', 'anchors': ['improved']},
            {'field': 'respiratory_recovery', 'anchors': ['improved']}], 'empty': []}}}
        report = summarize([result], entries, protocol, {})['none']
        self.assertEqual(report['key_facts_present_in_source_verified_quotes'], 2)
        self.assertEqual(report['key_facts_present_in_directly_usable_quotes'], 1)

    def test_invalid_annotation_excluded_without_changing_positives(self):
        result = {'review_id': 'R', 'reasoning': 'none', 'finish_reason': 'stop',
                  'end_to_end_seconds': 1, 'evidence': {}}
        protocol = {'cases': {'R': {'positive': [], 'empty': ['dyspnea', 'hypercapnia_acidosis']}}}
        report = summarize([result], [], protocol, {'R': ['hypercapnia_acidosis']})['none']
        self.assertEqual(report['valid_empty_checkpoints'], 1)
        self.assertEqual(report['valid_empty_passed'], 1)
        self.assertEqual(report['all_quotes_clinically_correct_nonempty_documents'], 0)


if __name__ == '__main__':
    unittest.main()
