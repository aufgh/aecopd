from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "model_loader"))

from aecopd_loader import AecopdDataBundle  # noqa: E402
from timeline_collator import SharedTimelineCollator, TimelineConfig  # noqa: E402


def _percentile(values: list[int], probability: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def build_length_index(
    data_dir: Path,
    artifacts_path: Path,
    output_path: Path,
    expected_admissions: int = 5656,
    text_dir: Path | None = None,
) -> dict[str, Any]:
    if text_dir is not None:
        bundle = AecopdDataBundle(data_dir, text_metadata_only=True, text_dir=text_dir)
    else:
        bundle = AecopdDataBundle(data_dir)
    artifacts = bundle.load_preprocessing(artifacts_path)
    collator = SharedTimelineCollator(TimelineConfig(max_tokens=None), artifacts=artifacts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | str]] = []
    seen: set[int] = set()
    lengths: list[int] = []
    split_counts: Counter[str] = Counter()

    for sample in bundle.iter_all_samples(artifacts=artifacts, include_text=False):
        if sample.hadm_id in seen:
            raise RuntimeError(f"duplicate hadm_id in model data: {sample.hadm_id}")
        length = collator.timeline_length(sample)
        if length < 1 or length > 1280:
            raise RuntimeError(
                f"timeline length outside validated range for hadm_id={sample.hadm_id}: {length}"
            )
        seen.add(sample.hadm_id)
        lengths.append(length)
        split_counts[sample.data_split] += 1
        rows.append(
            {
                "hadm_id": sample.hadm_id,
                "data_split": sample.data_split,
                "timeline_token_count": length,
                "raw_event_count": len(sample.events),
                "numeric_summary_count": len(sample.numeric_summaries),
            }
        )

    if len(rows) != expected_admissions:
        raise RuntimeError(f"expected {expected_admissions} admissions, observed {len(rows)}")

    with tempfile.NamedTemporaryFile(
        prefix="timeline_lengths_",
        suffix=".csv.gz",
        dir=output_path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return {
        "passed": True,
        "output": str(output_path),
        "admissions": len(rows),
        "unique_hadm_ids": len(seen),
        "split_counts": dict(sorted(split_counts.items())),
        "timeline_length_distribution": {
            "min": min(lengths),
            "median": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "max": max(lengths),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build exact per-admission timeline lengths.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-admissions", type=int, default=5656)
    parser.add_argument("--text-dir", type=Path, default=None,
                        help="Reuse section text index from another dir (e.g. model_v2). "
                             "Length computation never reads text, but the bundle validates its path.")
    args = parser.parse_args()
    report = build_length_index(
        args.data_dir, args.artifacts, args.output, args.expected_admissions, args.text_dir
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
