from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping

from .longitudinal_collator import PatientSequenceInput


@dataclass(frozen=True)
class LongitudinalBatchingConfig:
    max_padded_tokens: int = 8192
    max_patients: int = 8
    max_admissions: int = 32
    buffer_size: int = 128

    def __post_init__(self) -> None:
        if self.max_padded_tokens < 1:
            raise ValueError("max_padded_tokens must be positive")
        if self.max_patients < 1 or self.max_admissions < 1:
            raise ValueError("max_patients and max_admissions must be positive")
        if self.buffer_size < self.max_patients:
            raise ValueError("buffer_size must be at least max_patients")


class PatientSequenceBatcher:
    """Batch complete patient sequences under a flattened-admission token budget.

    A patient is never split or truncated. A single patient whose full history is
    over either limit is emitted as an explicit singleton and counted.
    """

    def __init__(
        self,
        lengths: Mapping[int, int],
        config: LongitudinalBatchingConfig | None = None,
        *,
        shuffle: bool,
        seed: int = 0,
    ) -> None:
        self.lengths = dict(lengths)
        self.config = config or LongitudinalBatchingConfig()
        self.shuffle = shuffle
        self.seed = seed
        self.last_oversize_singletons = 0

    def iter_batches(
        self,
        sequences: Iterable[PatientSequenceInput],
        *,
        epoch: int = 0,
    ) -> Iterator[list[PatientSequenceInput]]:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.last_oversize_singletons = 0
        randomizer = random.Random(self.seed + epoch)
        buffer: list[PatientSequenceInput] = []
        seen: set[int] = set()
        for sequence in sequences:
            if not sequence.admissions:
                raise ValueError("patient sequence cannot be empty")
            subject_id = int(sequence.admissions[0].subject_id)
            if subject_id in seen:
                raise ValueError(f"duplicate patient sequence: subject_id={subject_id}")
            if any(int(sample.subject_id) != subject_id for sample in sequence.admissions):
                raise ValueError(f"mixed patient sequence: subject_id={subject_id}")
            self._sequence_shape(sequence)
            seen.add(subject_id)
            buffer.append(sequence)
            if len(buffer) == self.config.buffer_size:
                yield from self._flush(buffer, randomizer)
                buffer = []
        if buffer:
            yield from self._flush(buffer, randomizer)

    def _sequence_shape(self, sequence: PatientSequenceInput) -> tuple[int, int]:
        lengths: list[int] = []
        for sample in sequence.admissions:
            try:
                length = int(self.lengths[int(sample.hadm_id)])
            except KeyError as exc:
                raise KeyError(f"missing timeline length for hadm_id={sample.hadm_id}") from exc
            if length < 1:
                raise ValueError(f"invalid timeline length for hadm_id={sample.hadm_id}: {length}")
            lengths.append(length)
        return len(lengths), max(lengths)

    def _flush(
        self,
        sequences: list[PatientSequenceInput],
        randomizer: random.Random,
    ) -> Iterator[list[PatientSequenceInput]]:
        candidates = list(sequences)
        if self.shuffle:
            randomizer.shuffle(candidates)
        candidates.sort(key=lambda item: self._sequence_shape(item)[1])

        completed: list[list[PatientSequenceInput]] = []
        batch: list[PatientSequenceInput] = []
        admissions = 0
        maximum_length = 0
        for sequence in candidates:
            sequence_admissions, sequence_maximum = self._sequence_shape(sequence)
            candidate_admissions = admissions + sequence_admissions
            candidate_maximum = max(maximum_length, sequence_maximum)
            exceeds = (
                len(batch) + 1 > self.config.max_patients
                or candidate_admissions > self.config.max_admissions
                or candidate_admissions * candidate_maximum > self.config.max_padded_tokens
            )
            if batch and exceeds:
                completed.append(batch)
                batch = []
                admissions = 0
                maximum_length = 0
                candidate_admissions = sequence_admissions
                candidate_maximum = sequence_maximum
            if not batch and (
                sequence_admissions > self.config.max_admissions
                or sequence_admissions * sequence_maximum > self.config.max_padded_tokens
            ):
                completed.append([sequence])
                self.last_oversize_singletons += 1
                continue
            batch.append(sequence)
            admissions = candidate_admissions
            maximum_length = candidate_maximum
        if batch:
            completed.append(batch)
        if self.shuffle:
            randomizer.shuffle(completed)
        yield from completed
