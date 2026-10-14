"""组消融总表必须同时保留两种结局的辨别与校准方向。"""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from training.v5.aggregate_group_ablations import run


class GroupAggregationTests(unittest.TestCase):
    def test_preserves_both_competing_outcome_metrics(self):
        metrics = {
            "integrated_brier_7_365": .01,
            "mean_auc_7_365": -.02,
            "c_index_90": -.03,
            "c_index_365": -.04,
            "mean_absolute_population_curve_gap_7_365": .05,
            "absolute_mean_calibration_gap_90": .06,
        }
        cause = {"delta_removed_minus_full": metrics,
                 "brier_interval": {"patient_cluster_200_draw_95_interval": [.001, .02]}}
        payload = {"seeds": {str(seed): {"no_labs": {"validation": {
            "readmission": cause, "death": cause}}} for seed in (2026, 2027, 2028)}}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "paired_group_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
            output = run(root)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["seeds_removal_worsens_readmission_brier"], "3")
        self.assertEqual(row["seeds_removal_worsens_death_auc"], "3")
        self.assertEqual(row["seeds_removal_worsens_death_c_index_365"], "3")
        self.assertEqual(row["seeds_removal_worsens_death_calibration_90"], "3")


if __name__ == "__main__":
    unittest.main()
