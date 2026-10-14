"""跨患者同领域文本置换必须保持覆盖和患者隔离。"""

from __future__ import annotations

import unittest

import numpy as np

from training.v5.data import V5Data
from training.v5.text_content_permutation import permute_same_field_across_patients


class TextContentPermutationTests(unittest.TestCase):
    def test_same_fields_and_coverage_without_self_donation(self) -> None:
        snippets = (
            ((0, 0), (1, 4)),
            ((0, 1),),
            ((0, 2), (1, 5)),
            ((0, 3),),
        )
        data = V5Data(
            subject_ids=np.array([1, 1, 2, 3]),
            hadm_ids=np.array([10, 11, 20, 30]),
            splits=np.array(["validation"] * 4),
            times=np.full(4, 365.0), events=np.zeros(4, dtype=np.int64),
            group_values=(), group_masks=(), gap_features=np.zeros((4, 2)),
            sequences=(np.array([0, 1]), np.array([2]), np.array([3])),
            snippets=snippets, text_vectors=np.zeros((6, 768)), text_width=768,
            coverage={},
        )
        changed, audit = permute_same_field_across_patients(
            data, np.arange(4), tuple(f"text-{index}" for index in range(6)),
            seed=2026,
        )
        owner = {0: 1, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2}
        for row, (original, shuffled) in enumerate(zip(snippets, changed.snippets, strict=True)):
            self.assertEqual([field for field, _ in original],
                             [field for field, _ in shuffled])
            for _, vector_row in shuffled:
                self.assertNotEqual(owner[vector_row], int(data.subject_ids[row]))
        self.assertEqual(audit["fields"]["residual_dyspnea"]["snippets"], 4)
        self.assertEqual(audit["fields"]["residual_cough_sputum"]["snippets"], 2)
        self.assertEqual(data.snippets, snippets)

    def test_unselected_admission_is_unchanged(self) -> None:
        snippets = (((0, 0),), ((0, 1),), ((0, 2),))
        data = V5Data(
            subject_ids=np.array([1, 2, 3]), hadm_ids=np.array([10, 20, 30]),
            splits=np.array(["validation", "validation", "train"]),
            times=np.full(3, 365.0), events=np.zeros(3, dtype=np.int64),
            group_values=(), group_masks=(), gap_features=np.zeros((3, 2)),
            sequences=(np.array([0]), np.array([1]), np.array([2])),
            snippets=snippets, text_vectors=np.zeros((3, 768)), text_width=768,
            coverage={},
        )
        changed, _ = permute_same_field_across_patients(
            data, np.array([0, 1]), ("a", "b", "c"), seed=7,
        )
        self.assertEqual(changed.snippets[2], snippets[2])


if __name__ == "__main__":
    unittest.main()
