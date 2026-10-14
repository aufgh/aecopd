"""新增异常方向与输出词表的基本约束。"""

import unittest

from patient_data.expanded_abnormality_features import ADDED_AXES, AXES, FEATURE_NAMES, directional_severity


class ExpandedAbnormalityFeatureTests(unittest.TestCase):
    def test_directional_reference_boundaries(self):
        self.assertEqual(directional_severity(140, 135, 145, "low"), 0)
        self.assertAlmostEqual(directional_severity(130, 135, 145, "low"), .5)
        self.assertAlmostEqual(directional_severity(150, 135, 145, "high"), .5)
        self.assertEqual(directional_severity(140, 135, 145, "high"), 0)
        self.assertEqual(directional_severity(210, 135, 145, "high"), 5)
        self.assertGreater(directional_severity(210, 135, 145, "high", severity_cap=None), 5)

    def test_fixed_six_values_per_axis_and_distinct_concepts(self):
        self.assertEqual(len(FEATURE_NAMES), len(AXES) * 6)
        self.assertEqual(len(set(FEATURE_NAMES)), len(FEATURE_NAMES))
        self.assertEqual(len(ADDED_AXES), 13)


if __name__ == "__main__":
    unittest.main()
