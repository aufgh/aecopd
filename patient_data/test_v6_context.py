"""V6 历史与去向边界；不依赖训练框架。"""
import unittest
import numpy as np
from training.v6.data import history_for, disposition, CURRENT_NAMES, HISTORY_NAMES


def admission(start, end, strict='f'):
    return {'admittime': start, 'dischtime': end, 'is_strict_aecopd': strict}


class ContextTests(unittest.TestCase):
    def test_only_previous_completed_positive_duration_admissions(self):
        row = admission('2100-01-10 00:00:00', '2100-01-15 00:00:00')
        previous = [admission('2100-01-01 00:00:00', '2100-01-03 00:00:00', 't'),
                    admission('2100-01-05 00:00:00', '2100-01-08 00:00:00'),
                    admission('2100-01-09 00:00:00', '2100-01-12 00:00:00'),
                    admission('2100-01-20 00:00:00', '2100-01-25 00:00:00'),
                    admission('2100-01-07 00:00:00', '2100-01-06 00:00:00')]
        h = history_for(row, previous, {'gender': 'M', 'anchor_age': '60', 'anchor_year': '2100'})
        np.testing.assert_allclose(h, [60, 0, np.log1p(2), np.log1p(1), np.log1p(2), 0])

    def test_no_history_and_future_cannot_change_it(self):
        row = admission('2100-01-10 00:00:00', '2100-01-15 00:00:00')
        p = {'gender': 'F', 'anchor_age': '65', 'anchor_year': '2100'}
        a = history_for(row, [], p)
        b = history_for(row, [admission('2100-02-01 00:00:00', '2100-02-02 00:00:00', 't')], p)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a[5], 1)

    def test_unknown_disposition_not_other(self):
        for unknown in ('', '\\N'):
            value, mask = disposition(unknown)
            self.assertEqual(value.sum(), 0)
            self.assertEqual(mask.sum(), 0)
        self.assertEqual(disposition('HOME HEALTH CARE')[0][0], 1)
        self.assertEqual(disposition('SKILLED NURSING FACILITY')[0][1], 1)
        self.assertEqual(disposition('HOSPICE')[0][2], 1)

    def test_current_branch_catalog_excludes_history(self):
        self.assertEqual(len(CURRENT_NAMES), 14)
        self.assertEqual(len(HISTORY_NAMES), 6)
        self.assertFalse(any('prior' in n or 'gap' in n for n in CURRENT_NAMES))


if __name__ == '__main__':
    unittest.main()
