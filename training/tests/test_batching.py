from __future__ import annotations

import unittest
from dataclasses import dataclass

from aecopd_training.batching import BatchingConfig, LengthBucketBatcher


@dataclass(frozen=True)
class Sample:
    hadm_id: int


class LengthBucketBatcherTest(unittest.TestCase):
    def test_each_sample_appears_once_and_every_batch_respects_token_budget(self) -> None:
        samples = [Sample(hadm_id=value) for value in range(1, 9)]
        lengths = {1: 11, 2: 40, 3: 120, 4: 250, 5: 300, 6: 500, 7: 700, 8: 1200}
        batcher = LengthBucketBatcher(
            lengths,
            BatchingConfig(
                max_padded_tokens=1200,
                max_batch_size=4,
                buffer_size=8,
                bucket_boundaries=(256, 512, 768, 1024, 1280),
            ),
            shuffle=True,
            seed=17,
        )

        batches = list(batcher.iter_batches(samples, epoch=0))

        observed = [sample.hadm_id for batch in batches for sample in batch]
        self.assertEqual(sorted(observed), list(range(1, 9)))
        self.assertEqual(len(observed), len(set(observed)))
        for batch in batches:
            padded_tokens = max(lengths[sample.hadm_id] for sample in batch) * len(batch)
            self.assertLessEqual(padded_tokens, 1200)

    def test_seed_and_epoch_make_training_order_reproducible(self) -> None:
        samples = [Sample(hadm_id=value) for value in range(1, 13)]
        lengths = {value: 100 for value in range(1, 13)}
        config = BatchingConfig(max_padded_tokens=400, max_batch_size=4, buffer_size=12)
        batcher = LengthBucketBatcher(lengths, config, shuffle=True, seed=9)

        first = [[sample.hadm_id for sample in batch] for batch in batcher.iter_batches(samples, epoch=2)]
        repeated = [[sample.hadm_id for sample in batch] for batch in batcher.iter_batches(samples, epoch=2)]
        next_epoch = [[sample.hadm_id for sample in batch] for batch in batcher.iter_batches(samples, epoch=3)]

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_epoch)
        self.assertCountEqual(
            [value for batch in next_epoch for value in batch],
            lengths,
        )

    def test_missing_length_is_reported_instead_of_guessing(self) -> None:
        batcher = LengthBucketBatcher(
            {1: 20},
            BatchingConfig(max_padded_tokens=100, max_batch_size=2, buffer_size=2),
            shuffle=False,
        )

        with self.assertRaisesRegex(KeyError, "hadm_id=2"):
            list(batcher.iter_batches([Sample(1), Sample(2)]))


if __name__ == "__main__":
    unittest.main()
