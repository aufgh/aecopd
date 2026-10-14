"""计数字段消融必须完全屏蔽信息，且不改变输入槽位和参数尺寸。"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from training.v5.independent_data import collate
from training.v5.independent_fusion import IndependentFusionModel
from training.v5.train_event_sequence import PRIOR_COUNT_CONCEPTS


class CountAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = SimpleNamespace(
            tokens_by_admission=(np.array([0, 1, 2]),),
            concept_names=("context:age_at_admission", *PRIOR_COUNT_CONCEPTS),
            disabled_concepts=(), token_values=np.ones((3, 5), dtype=np.float32),
            token_masks=np.ones((3, 5), dtype=np.float32),
            token_concepts=np.array([1, 2, 3]), token_types=np.ones(3, dtype=np.int64),
            gap=np.zeros((1, 2), dtype=np.float32), times=np.array([365.], dtype=np.float32),
            events=np.array([0]), text_width=8, text_by_admission=((),), text_vectors=None)

    def test_mask_keeps_slots_and_clears_all_count_channels(self) -> None:
        full = collate(self.data, [np.array([0])], include_text=False)
        self.data.disabled_concepts = PRIOR_COUNT_CONCEPTS
        removed = collate(self.data, [np.array([0])], include_text=False)
        self.assertEqual(full.token_values.shape, removed.token_values.shape)
        self.assertEqual(removed.token_present.tolist(), [[[True, False, False]]])
        for array in (removed.token_values, removed.token_masks, removed.token_concepts,
                      removed.token_types):
            self.assertEqual(int(torch.count_nonzero(array[:, :, 1:])), 0)
        torch.testing.assert_close(full.token_values[:, :, 0], removed.token_values[:, :, 0])

    def test_disabled_values_cannot_change_prediction(self) -> None:
        self.data.disabled_concepts = PRIOR_COUNT_CONCEPTS
        model = IndependentFusionModel(concept_count=3, type_count=1, field_count=1,
                                       text_width=8, width=32).eval()
        with torch.no_grad():
            before = model(collate(self.data, [np.array([0])], include_text=False)).log_amplitude
            self.data.token_values[1:] = 10000
            after = model(collate(self.data, [np.array([0])], include_text=False)).log_amplitude
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_unknown_disabled_concept_fails(self) -> None:
        self.data.disabled_concepts = ("unknown",)
        with self.assertRaisesRegex(ValueError, "absent"):
            collate(self.data, [np.array([0])], include_text=False)


if __name__ == "__main__":
    unittest.main()
