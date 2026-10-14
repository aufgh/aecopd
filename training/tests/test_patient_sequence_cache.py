from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from aecopd_training.patient_sequence_cache import (
    PatientSequenceCache,
    build_patient_sequence_cache,
)


@dataclass(frozen=True)
class Sample:
    subject_id: int
    hadm_id: int
    data_split: str


class FakeBundle:
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def iter_samples(self, split, *, artifacts, include_text):
        del artifacts, include_text
        yield from (sample for sample in self.samples if split is None or sample.data_split == split)


class PatientSequenceCacheTests(unittest.TestCase):
    def test_builds_complete_causal_sequences_and_deterministic_patient_shuffle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "patient_sequence_index.csv.gz"
            with gzip.open(index, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "subject_id",
                        "hadm_id",
                        "data_split",
                        "sequence_index",
                        "days_since_previous_discharge",
                    ),
                )
                writer.writeheader()
                writer.writerows(
                    (
                        {"subject_id": 10, "hadm_id": 101, "data_split": "train", "sequence_index": 0, "days_since_previous_discharge": ""},
                        {"subject_id": 10, "hadm_id": 102, "data_split": "train", "sequence_index": 1, "days_since_previous_discharge": 15.5},
                        {"subject_id": 20, "hadm_id": 201, "data_split": "train", "sequence_index": 0, "days_since_previous_discharge": ""},
                        {"subject_id": 30, "hadm_id": 301, "data_split": "validation", "sequence_index": 0, "days_since_previous_discharge": ""},
                    )
                )
            bundle = FakeBundle(
                [Sample(20, 201, "train"), Sample(10, 102, "train"), Sample(30, 301, "validation"), Sample(10, 101, "train")]
            )
            cache_path = root / "patient_sequences.sqlite"
            summary = build_patient_sequence_cache(
                bundle,
                artifacts={},
                sequence_index_path=index,
                output_path=cache_path,
            )

            self.assertEqual(summary["patients"], 3)
            self.assertEqual(summary["admissions"], 4)
            with PatientSequenceCache(cache_path) as cache:
                ordered = list(cache.iter_sequences("train", shuffle=False))
                self.assertEqual([sequence.admissions[0].subject_id for sequence in ordered], [10, 20])
                self.assertEqual([sample.hadm_id for sample in ordered[0].admissions], [101, 102])
                self.assertEqual(ordered[0].gap_days, (0.0, 15.5))
                first = [sequence.admissions[0].subject_id for sequence in cache.iter_sequences("train", shuffle=True, seed=7, epoch=2)]
                second = [sequence.admissions[0].subject_id for sequence in cache.iter_sequences("train", shuffle=True, seed=7, epoch=2)]
                self.assertEqual(first, second)
                self.assertEqual(sorted(first), [10, 20])
                validation = list(cache.iter_sequences("validation", shuffle=False))
                self.assertEqual([sample.hadm_id for sample in validation[0].admissions], [301])


if __name__ == "__main__":
    unittest.main()
