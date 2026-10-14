from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

from aecopd_training.conditional_outcomes import ConditionalOutcomeStore


class ConditionalOutcomeStoreTests(unittest.TestCase):
    def test_builds_mutually_exclusive_interval_and_open_tail_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.csv.gz"
            fields = (
                "hadm_id", "event_type", "time_days",
                "next_observed_aecopd_days", "observed_death_days",
            )
            rows = (
                (1, 1, 30, 30, ""),
                (2, 2, 90, "", 90),
                (3, 1, 365, 365, ""),
                (4, 0, 365, 500, 400),
                (5, 0, 365, 500, 500),
                (6, 0, 365, "", ""),
            )
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(zip(fields, row)))

            store = ConditionalOutcomeStore.from_cohort(path)

            self.assertEqual(store[1].interval_labels, (1, 0, 0))
            self.assertEqual(store[1].interval_mask, (True, False, False))
            self.assertFalse(store[1].tail_mask)
            self.assertEqual(store[2].interval_labels, (0, 2, 0))
            self.assertEqual(store[2].interval_mask, (True, True, False))
            self.assertEqual(store[3].interval_labels, (0, 0, 1))
            self.assertEqual(store[4].tail_label, 2)
            self.assertTrue(store[4].tail_mask)
            self.assertEqual(store[5].tail_label, 1)  # ties follow existing AECOPD-first rule
            self.assertEqual(store[6].interval_labels, (0, 0, 0))
            self.assertEqual(store[6].interval_mask, (True, True, True))
            self.assertEqual(store[6].tail_label, 0)


if __name__ == "__main__":
    unittest.main()
