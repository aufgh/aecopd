import unittest

from training.benchmark_bonsai_verbatim import checkpoint_report


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.case = {'positive': [{'field': 'dyspnea', 'anchors': ['dyspnea', 'improved']}],
                     'empty': ['inhaler_technique']}

    def test_failures_do_not_earn_negative_credit(self):
        report = checkpoint_report({'status': 'request_failed'}, self.case)
        self.assertFalse(report['negative'][0]['pass'])

    def test_schema_failure_does_not_earn_negative_credit(self):
        result = {'finish_reason': 'stop', 'evidence': {}, 'validation_issues': [{'type': 'schema_keys'}]}
        self.assertFalse(checkpoint_report(result, self.case)['negative'][0]['pass'])

    def test_supported_checkpoint(self):
        result = {'finish_reason': 'stop', 'evidence': {'dyspnea': [{'quote': 'Dyspnea improved.'}]}}
        report = checkpoint_report(result, self.case)
        self.assertTrue(report['positive'][0]['detected'])
        self.assertTrue(report['negative'][0]['pass'])

    def test_absent_quote_cannot_earn_positive_credit(self):
        result = {'finish_reason': 'stop', 'evidence': {}, 'validation_issues': [{'type': 'quote_absent'}]}
        self.assertFalse(checkpoint_report(result, self.case)['positive'][0]['detected'])


if __name__ == '__main__':
    unittest.main()
