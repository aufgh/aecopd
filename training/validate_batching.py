from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aecopd_training import BatchingConfig, LengthBucketBatcher, load_timeline_lengths


@dataclass(frozen=True)
class IndexedAdmission:
    hadm_id: int


def _percentile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)] if ordered else 0


def _sequential_padded_tokens(lengths: list[int], batch_size: int) -> int:
    total = 0
    for start in range(0, len(lengths), batch_size):
        batch = lengths[start : start + batch_size]
        total += max(batch) * len(batch)
    return total


def validate(
    index_path: Path,
    *,
    split: str,
    config: BatchingConfig,
) -> dict[str, Any]:
    lengths = load_timeline_lengths(index_path, split=split)
    admissions = [IndexedAdmission(hadm_id) for hadm_id in lengths]
    batcher = LengthBucketBatcher(lengths, config, shuffle=True, seed=20260912)
    batches = list(batcher.iter_batches(admissions, epoch=0))
    flattened = [sample.hadm_id for batch in batches for sample in batch]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("batcher duplicated at least one admission")
    if set(flattened) != set(lengths):
        raise RuntimeError("batcher lost or introduced admissions")

    actual_tokens = sum(lengths.values())
    padded_tokens = sum(
        max(lengths[sample.hadm_id] for sample in batch) * len(batch)
        for batch in batches
    )
    sequential_padded = _sequential_padded_tokens(
        list(lengths.values()),
        config.max_batch_size,
    )
    batch_sizes = [len(batch) for batch in batches]
    maximum_batch_tokens = max(
        max(lengths[sample.hadm_id] for sample in batch) * len(batch)
        for batch in batches
    )
    return {
        "passed": maximum_batch_tokens <= config.max_padded_tokens,
        "split": split,
        "admissions": len(flattened),
        "unique_admissions": len(set(flattened)),
        "batches": len(batches),
        "configuration": {
            "max_padded_tokens": config.max_padded_tokens,
            "max_batch_size": config.max_batch_size,
            "buffer_size": config.buffer_size,
            "bucket_boundaries": list(config.bucket_boundaries),
        },
        "batch_size_distribution": {
            "min": min(batch_sizes),
            "median": _percentile(batch_sizes, 0.50),
            "p90": _percentile(batch_sizes, 0.90),
            "max": max(batch_sizes),
        },
        "tokens": {
            "actual": actual_tokens,
            "bucketed_padded": padded_tokens,
            "sequential_fixed_batch_padded": sequential_padded,
            "maximum_single_batch_padded": maximum_batch_tokens,
        },
        "efficiency": {
            "bucketed_useful_fraction": actual_tokens / padded_tokens,
            "sequential_fixed_batch_useful_fraction": actual_tokens / sequential_padded,
            "padded_token_reduction_vs_sequential": 1.0 - padded_tokens / sequential_padded,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate streaming length-bucket batching.")
    parser.add_argument("--length-index", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-padded-tokens", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=256)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate(
        args.length_index,
        split=args.split,
        config=BatchingConfig(
            max_padded_tokens=args.max_padded_tokens,
            max_batch_size=args.max_batch_size,
            buffer_size=args.buffer_size,
        ),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
