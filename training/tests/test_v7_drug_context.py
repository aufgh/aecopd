"""首末窗口和剂量/途径未知的边界，不把无记录当减药。"""
import unittest
from training.v7.drug_context_features import summary,route_kind


class DrugContextTests(unittest.TestCase):
    def test_both_windows_needed(self):
        x,m=summary([(0.1,10.,'iv'),(.5,10.,'iv')],4.)
        self.assertFalse(m[5]);self.assertFalse(m[6])
        x,m=summary([(.1,10.,'iv'),(3.5,20.,'enteral')],4.)
        self.assertTrue(m[5]);self.assertTrue(m[6]);self.assertAlmostEqual(x[6],10/3)
        self.assertEqual(x[8],1);self.assertTrue(m[8])

    def test_unknown_dose_not_zero(self):
        x,m=summary([(.1,None,'iv'),(3.5,20.,'enteral')],4.)
        self.assertTrue(m[5]);self.assertFalse(m[6])
        x,m=summary([(.1,10.,'iv'),(.5,20.,'enteral')],1.)
        self.assertFalse(m[5]);self.assertFalse(m[6])

    def test_route_alias_and_conflict(self):
        self.assertEqual(route_kind({'PO','ORAL'}),'enteral')
        self.assertEqual(route_kind({'IV','IV PUSH'}),'iv')
        self.assertIsNone(route_kind({'IV','PO'}));self.assertIsNone(route_kind({'UNKNOWN'}))


if __name__=='__main__':unittest.main()
