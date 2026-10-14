from __future__ import annotations

import unittest
from dataclasses import dataclass

from aecopd_training.longitudinal_batching import (
    LongitudinalBatchingConfig,
    PatientSequenceBatcher,
)
from aecopd_training.longitudinal_collator import PatientSequenceInput


@dataclass(frozen=True)
class Sample:
    subject_id: int
    hadm_id: int


def sequence(subject_id: int, *hadm_ids: int) -> PatientSequenceInput:
    return PatientSequenceInput(
        admissions=tuple(Sample(subject_id, hadm_id) for hadm_id in hadm_ids),
        gap_days=tuple(0.0 if index == 0 else 1.0 for index in range(len(hadm_ids))),
    )


class PatientSequenceBatcherTests(unittest.TestCase):
    def test_preserves_complete_patients_and_respects_padding_budget(self) -> None:
        sequences = [sequence(1, 11, 12), sequence(2, 21), sequence(3, 31, 32)]
        lengths = {11: 3, 12: 4, 21: 8, 31: 2, 32: 2}
        batcher = PatientSequenceBatcher(
            lengths,
            LongitudinalBatchingConfig(
                max_padded_tokens=12,
                max_patients=2,
                max_admissions=4,
                buffer_size=3,
            ),
            shuffle=False,
        )

        batches = list(batcher.iter_batches(sequences))
        flattened = [item for batch in batches for item in batch]
        self.assertEqual(
            sorted(item.admissions[0].subject_id for item in flattened),
            [1, 2, 3],
        )
        self.assertTrue(all(len({sample.subject_id for sample in item.admissions}) == 1 for item in flattened))
        for batch in batches:
            admission_lengths = [lengths[sample.hadm_id] for item in batch for sample in item.admissions]
            cost = len(admission_lengths) * max(admission_lengths)
            self.assertLessEqual(cost, 12)

    def test_oversize_patient_is_yielded_once_as_singleton_without_truncation(self) -> None:
        long_sequence = sequence(9, 91, 92, 93)
        batcher = PatientSequenceBatcher(
            {91: 10, 92: 10, 93: 10},
            LongitudinalBatchingConfig(
                max_padded_tokens=20,
                max_patients=2,
                max_admissions=2,
                buffer_size=2,
            ),
            shuffle=False,
        )

        batches = list(batcher.iter_batches([long_sequence]))
        self.assertEqual(batches, [[long_sequence]])
        self.assertEqual(batcher.last_oversize_singletons, 1)


if __name__ == "__main__":
    unittest.main()
