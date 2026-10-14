"""候选章节消融保留同一字段词表和行索引。"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from training.v5.permute_added_section import permute_role
from training.v5.train_event_sequence import select_active_text_fields


@dataclass(frozen=True)
class _MinimalData:
    field_roles: tuple[str, ...]
    text_by_admission: tuple[tuple[tuple[int, int], ...], ...]


class ActiveTextFieldTests(unittest.TestCase):
    def test_mask_preserves_vocabulary_and_vector_rows(self) -> None:
        source = _MinimalData(("hpi", "course", "discharge_medications"),
                              (((0, 7), (2, 12), (1, 9)), ((2, 3),)))
        baseline = select_active_text_fields(source, ("hpi", "course"))
        extended = select_active_text_fields(source, ("hpi", "course", "discharge_medications"))
        self.assertEqual(baseline.field_roles, extended.field_roles)
        self.assertEqual(baseline.text_by_admission, (((0, 7), (1, 9)), ()))
        self.assertEqual(extended.text_by_admission, source.text_by_admission)
        self.assertEqual(source.text_by_admission[1], ((2, 3),))

    def test_rejects_unknown_and_duplicate_roles(self) -> None:
        source = _MinimalData(("hpi",), (((0, 0),),))
        for names in ((), ("hpi", "hpi"), ("missing",)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                select_active_text_fields(source, names)

    def test_candidate_permutation_keeps_other_fields_and_missing_pattern(self) -> None:
        @dataclass(frozen=True)
        class Base:
            field_roles: tuple[str, ...]
            subject_ids: np.ndarray
            text_by_admission: tuple[tuple[tuple[int, int], ...], ...]

        @dataclass(frozen=True)
        class Data:
            base: Base

        source = Data(Base(
            ("history_of_present_illness", "discharge_medications"),
            np.array([10, 20, 30]),
            (((0, 1), (1, 7)), ((0, 2), (1, 8)), ((0, 3),)),
        ))
        shuffled, count = permute_role(source, [np.array([0, 1, 2])], "discharge_medications", 71)
        self.assertEqual(count, 2)
        self.assertEqual([len(x) for x in shuffled.base.text_by_admission], [2, 2, 1])
        self.assertEqual([x[0] for x in shuffled.base.text_by_admission], [(0, 1), (0, 2), (0, 3)])
        self.assertEqual(shuffled.base.text_by_admission[0][1], (1, 8))
        self.assertEqual(shuffled.base.text_by_admission[1][1], (1, 7))
        self.assertEqual(source.base.text_by_admission[0][1], (1, 7))


if __name__ == "__main__":
    unittest.main()
