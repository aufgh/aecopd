"""只读统计的边界测试；不加载真实患者文件。"""
import unittest
from datetime import datetime, timedelta

from patient_data.audit_v6_input_candidates import CHEST_CT, measurement_stats, number


class InputAuditTests(unittest.TestCase):
    def test_recheck_before_last_48h_is_retained(self):
        start = datetime(2020, 1, 1)
        result = measurement_stats(
            {"a": {start, start + timedelta(days=2)}},
            {"a": (start, start + timedelta(days=8))}, {"a": "person"})
        self.assertEqual(result["two_distinct_timestamps"], 1)
        self.assertEqual(result["last_measurement_older_than_48h"], 1)
        self.assertEqual(result["first_in_24h_and_last_in_48h"], 0)

    def test_single_measurement_not_repeated(self):
        moment = datetime(2020, 1, 1)
        result = measurement_stats({"a": {moment}},
                                  {"a": (moment, moment + timedelta(days=1))}, {"a": "person"})
        self.assertEqual(result["observed_admissions"], 1)
        self.assertEqual(result["two_distinct_timestamps"], 0)

    def test_ct_title_is_candidate_not_general_chest_radiograph(self):
        self.assertIsNotNone(CHEST_CT.search("EXAMINATION: CT CHEST W/O CONTRAST"))
        self.assertIsNotNone(CHEST_CT.search("CTA PULMONARY"))
        self.assertIsNone(CHEST_CT.search("CHEST (PORTABLE AP)"))
        self.assertIsNone(CHEST_CT.search("CT HEAD WITHOUT CONTRAST"))

    def test_nonfinite_is_not_numeric_observation(self):
        for value in ("NaN", "inf", "\\N", ""):
            self.assertIsNone(number(value))
        self.assertEqual(number("0"), 0)


if __name__ == "__main__":
    unittest.main()
