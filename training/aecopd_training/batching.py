from __future__ import annotations

import bisect
import csv
import gzip
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Iterable, Iterator, Mapping, Protocol, TypeVar


class HasAdmissionId(Protocol):
    hadm_id: int


SampleT = TypeVar("SampleT", bound=HasAdmissionId)


@dataclass(frozen=True)
class BatchingConfig:
    max_padded_tokens: int = 8192
    max_batch_size: int = 32
    buffer_size: int = 256
    bucket_boundaries: tuple[int, ...] = (256, 512, 768, 1024, 1280)

    def __post_init__(self) -> None:
        if self.max_padded_tokens < 1:
            raise ValueError("max_padded_tokens must be positive")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if self.buffer_size < self.max_batch_size:
            raise ValueError("buffer_size must be at least max_batch_size")
        if not self.bucket_boundaries or any(
            left >= right
            for left, right in zip(self.bucket_boundaries, self.bucket_boundaries[1:])
        ):
            raise ValueError("bucket_boundaries must be strictly increasing")


def load_timeline_lengths(path: str | Path, *, split: str | None = None) -> dict[int, int]:
    """Load the exact timeline token count keyed by admission ID."""
    lengths: dict[int, int] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row["data_split"] != split:
                continue
            hadm_id = int(row["hadm_id"])
            if hadm_id in lengths:
                raise ValueError(f"duplicate hadm_id in length index: {hadm_id}")
            length = int(row["timeline_token_count"])
            if length < 1:
                raise ValueError(f"invalid timeline length for hadm_id={hadm_id}: {length}")
            lengths[hadm_id] = length
    if not lengths:
        suffix = f" for split={split}" if split is not None else ""
        raise ValueError(f"timeline length index is empty{suffix}")
    return lengths


class LengthBucketBatcher(Generic[SampleT]):
    """Create padded-token-budgeted batches from a streaming sample iterable.

    Only ``buffer_size`` samples are retained at once. Training can shuffle within
    each buffer and across its completed batches; validation remains deterministic.
    """

    def __init__(
        self,
        lengths: Mapping[int, int],
        config: BatchingConfig | None = None,
        *,
        shuffle: bool,
        seed: int = 0,
    ) -> None:
        self.lengths = dict(lengths)
        self.config = config or BatchingConfig()
        self.shuffle = shuffle
        self.seed = seed

    def iter_batches(
        self,
        samples: Iterable[SampleT],
        *,
        epoch: int = 0,
    ) -> Iterator[list[SampleT]]:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        randomizer = random.Random(self.seed + epoch)
        buffer: list[SampleT] = []
        seen: set[int] = set()
        for sample in samples:
            if sample.hadm_id in seen:
                raise ValueError(f"duplicate sample in stream: hadm_id={sample.hadm_id}")
            self._length(sample)
            seen.add(sample.hadm_id)
            buffer.append(sample)
            if len(buffer) == self.config.buffer_size:
                yield from self._flush(buffer, randomizer)
                buffer = []
        if buffer:
            yield from self._flush(buffer, randomizer)

    def _length(self, sample: SampleT) -> int:
        try:
            length = self.lengths[sample.hadm_id]
        except KeyError as exc:
            raise KeyError(f"missing timeline length for hadm_id={sample.hadm_id}") from exc
        if length > self.config.bucket_boundaries[-1]:
            raise ValueError(
                f"timeline length {length} exceeds largest bucket boundary "
                f"{self.config.bucket_boundaries[-1]} for hadm_id={sample.hadm_id}"
            )
        if length > self.config.max_padded_tokens:
            raise ValueError(
                f"timeline length {length} exceeds per-batch token budget "
                f"{self.config.max_padded_tokens} for hadm_id={sample.hadm_id}"
            )
        return length

    def _flush(
        self,
        samples: list[SampleT],
        randomizer: random.Random,
    ) -> Iterator[list[SampleT]]:
        buckets: dict[int, list[SampleT]] = {}
        for sample in samples:
            bucket = bisect.bisect_left(
                self.config.bucket_boundaries,
                self._length(sample),
            )
            buckets.setdefault(bucket, []).append(sample)

        completed: list[list[SampleT]] = []
        for bucket in sorted(buckets):
            candidates = buckets[bucket]
            if self.shuffle:
                randomizer.shuffle(candidates)
            batch: list[SampleT] = []
            maximum = 0
            for sample in candidates:
                length = self._length(sample)
                candidate_maximum = max(maximum, length)
                candidate_size = len(batch) + 1
                exceeds_budget = (
                    candidate_maximum * candidate_size > self.config.max_padded_tokens
                )
                exceeds_size = candidate_size > self.config.max_batch_size
                if batch and (exceeds_budget or exceeds_size):
                    completed.append(batch)
                    batch = []
                    maximum = 0
                batch.append(sample)
                maximum = max(maximum, length)
            if batch:
                completed.append(batch)
        if self.shuffle:
            randomizer.shuffle(completed)
        yield from completed
