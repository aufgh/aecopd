from __future__ import annotations

import unittest
from dataclasses import dataclass

import torch

from aecopd_training.longitudinal_collator import (
    LongitudinalBatchCollator,
    PatientSequenceInput,
)


@dataclass(frozen=True)
class Sample:
    subject_id: int
    hadm_id: int


class RecordingAdmissionCollator:
    def __init__(self) -> None:
        self.hadm_ids: list[int] = []

    def __call__(self, samples: list[Sample], *, device: str | None = None):
        del device
        self.hadm_ids = [sample.hadm_id for sample in samples]
        return "admission-batch", {"existing": torch.tensor([1])}


class LongitudinalBatchCollatorTests(unittest.TestCase):
    def test_flattens_patient_sequences_and_adds_causal_mapping(self) -> None:
        admission_collator = RecordingAdmissionCollator()
        collator = LongitudinalBatchCollator(admission_collator)
        sequences = [
            PatientSequenceInput(
                admissions=(Sample(10, 101), Sample(10, 102)),
                gap_days=(0.0, 30.0),
            ),
            PatientSequenceInput(
                admissions=(Sample(20, 201),),
                gap_days=(0.0,),
            ),
        ]

        base_batch, tensors = collator(sequences)

        self.assertEqual(base_batch, "admission-batch")
        self.assertEqual(admission_collator.hadm_ids, [101, 102, 201])
        self.assertEqual(tensors["longitudinal_patient_indices"].tolist(), [0, 0, 1])
        self.assertEqual(tensors["longitudinal_position_indices"].tolist(), [0, 1, 0])
        self.assertEqual(tensors["longitudinal_gap_days"].tolist(), [0.0, 30.0, 0.0])
        self.assertEqual(
            tensors["longitudinal_attention_mask"].tolist(),
            [[True, True], [True, False]],
        )

    def test_rejects_mixed_patient_sequence(self) -> None:
        collator = LongitudinalBatchCollator(RecordingAdmissionCollator())
        with self.assertRaises(ValueError):
            collator(
                [
                    PatientSequenceInput(
                        admissions=(Sample(10, 101), Sample(20, 201)),
                        gap_days=(0.0, 30.0),
                    )
                ]
            )


if __name__ == "__main__":
    unittest.main()
