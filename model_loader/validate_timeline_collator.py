from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from aecopd_loader import AecopdDataBundle, _is_treatment_exposure
from timeline_collator import SharedTimelineCollator, TimelineConfig


def percentile(values: Sequence[int], probability: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round(probability * (len(ordered) - 1))]


def validate(
    data_dir: Path,
    artifacts_path: Path,
    report_path: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    bundle = AecopdDataBundle(data_dir)
    artifacts = bundle.fit_preprocessing(artifacts_path)
    collator = SharedTimelineCollator(TimelineConfig(max_tokens=None), artifacts=artifacts)

    admission_count = 0
    raw_event_count = 0
    timeline_token_count = 0
    global_summary_count = 0
    timeline_lengths: list[int] = []
    violations: Counter[str] = Counter()
    window_token_counts: Counter[str] = Counter()
    unknown_concept_events: Counter[str] = Counter()
    duration_event_count = 0
    duration_token_count = 0
    mapped_layer_event_counts: Counter[str] = Counter()
    binary_exposure_event_count = 0
    invalid_binary_exposure_event_count = 0

    def consume(samples: list[Any]) -> None:
        nonlocal admission_count, raw_event_count, global_summary_count
        nonlocal duration_event_count, timeline_token_count, duration_token_count
        nonlocal binary_exposure_event_count, invalid_binary_exposure_event_count
        batch = collator(samples)
        admission_count += len(samples)
        raw_event_count += sum(len(sample.events) for sample in samples)
        global_summary_count += sum(len(sample.numeric_summaries) for sample in samples)
        duration_event_count += sum(event.duration_mask for sample in samples for event in sample.events)
        mapped_layer_event_counts.update(
            event.feature_layer for sample in samples for event in sample.events
        )
        binary_exposure_event_count += sum(
            event.value_type == "binary_exposure"
            for sample in samples
            for event in sample.events
        )
        invalid_binary_exposure_event_count += sum(
            event.value_type == "binary_exposure"
            and (
                not _is_treatment_exposure(
                    {"status": event.status, "evidence_type": event.evidence_type}
                )
                or event.route != ""
            )
            for sample in samples
            for event in sample.events
        )
        timeline_lengths.extend(batch.timeline_lengths)
        timeline_token_count += sum(batch.timeline_lengths)
        for sample in samples:
            unknown_concept_events[sample.data_split] += sum(event.concept_id == 1 for event in sample.events)
        _validate_batch(batch, violations, window_token_counts)
        duration_index = batch.timeline_feature_names.index("duration_hours_max_log1p")
        duration_token_count += sum(
            batch.timeline_feature_masks[i][j][duration_index]
            for i, length in enumerate(batch.timeline_lengths)
            for j in range(length)
        )

    pending: list[Any] = []
    for sample in bundle.iter_all_samples(artifacts=artifacts, include_text=False):
        pending.append(sample)
        if len(pending) == batch_size:
            consume(pending)
            pending = []
    if pending:
        consume(pending)

    maximum = max(timeline_lengths, default=0)
    compression = 1.0 - timeline_token_count / raw_event_count if raw_event_count else 0.0
    event_vocabulary = artifacts["event_vocabulary"]
    route_vocabulary = artifacts["event_route_vocabulary"]
    feature_contract = artifacts.get("clinical_feature_contract")
    checks = [
        _check("住院数", admission_count, 5656, admission_count == 5656),
        _check("映射后结构化事件非空", raw_event_count, ">0", raw_event_count > 0),
        _check("映射后全局数值摘要非空", global_summary_count, ">0", global_summary_count > 0),
        _check("训练折未知概念事件", unknown_concept_events["train"], 0, unknown_concept_events["train"] == 0),
        _check(
            "给药途径不进入新词表",
            route_vocabulary,
            ["<PAD>", "<UNK>", "<MISSING>"],
            route_vocabulary == ["<PAD>", "<UNK>", "<MISSING>"],
        ),
        _check("临床特征合同存在", bool(feature_contract), True, bool(feature_contract)),
        _check("治疗二元暴露事件非空", binary_exposure_event_count, ">0", binary_exposure_event_count > 0),
        _check(
            "未执行/暂停/未开始/执行不确定不得形成暴露",
            invalid_binary_exposure_event_count,
            0,
            invalid_binary_exposure_event_count == 0,
        ),
        _check("默认1280-token上限", maximum, "<=1280", maximum <= 1280),
        _check("collator违例", sum(violations.values()), 0, sum(violations.values()) == 0),
    ]
    report = {
        "schema_version": "aecopd-curated-clinical-timeline-validation-1",
        "passed": all(item["passed"] for item in checks),
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "time_windows": list(collator([]).timeline_window_names),
        "timeline_feature_names": list(collator([]).timeline_feature_names),
        "counts": {
            "admissions": admission_count,
            "mapped_events": raw_event_count,
            "timeline_tokens": timeline_token_count,
            "global_numeric_summaries": global_summary_count,
            "duration_events": duration_event_count,
            "tokens_with_duration": duration_token_count,
            "binary_exposure_events": binary_exposure_event_count,
            "invalid_binary_exposure_events": invalid_binary_exposure_event_count,
            "event_vocabulary_size": len(event_vocabulary),
            "summary_vocabulary_size": len(artifacts["numeric_summary_vocabulary"]),
        },
        "feature_contract": feature_contract,
        "feature_audit": artifacts.get("clinical_feature_audit"),
        "mapped_layer_event_counts": dict(sorted(mapped_layer_event_counts.items())),
        "compression": {
            "mapped_event_to_timeline_reduction": compression,
            "mapped_events_per_admission_mean": (
                raw_event_count / admission_count if admission_count else 0.0
            ),
            "timeline_tokens_per_admission_mean": (
                timeline_token_count / admission_count if admission_count else 0.0
            ),
        },
        "timeline_length_distribution": {
            "min": min(timeline_lengths, default=0),
            "median": percentile(timeline_lengths, 0.50),
            "p90": percentile(timeline_lengths, 0.90),
            "p95": percentile(timeline_lengths, 0.95),
            "p99": percentile(timeline_lengths, 0.99),
            "max": maximum,
            "over_256": sum(value > 256 for value in timeline_lengths),
            "over_512": sum(value > 512 for value in timeline_lengths),
            "over_640": sum(value > 640 for value in timeline_lengths),
            "over_768": sum(value > 768 for value in timeline_lengths),
            "over_1024": sum(value > 1024 for value in timeline_lengths),
            "over_1280": sum(value > 1280 for value in timeline_lengths),
        },
        "window_token_counts": dict(sorted(window_token_counts.items())),
        "unknown_concept_event_counts": dict(sorted(unknown_concept_events.items())),
        "violations": dict(sorted(violations.items())),
        "validation_scope": "full 5,656-admission model-independent shared-timeline collator dry run; no model training",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _validate_batch(batch: Any, violations: Counter[str], window_counts: Counter[str]) -> None:
    for i, length in enumerate(batch.timeline_lengths):
        if sum(batch.timeline_attention_mask[i]) != length:
            violations["timeline_attention_length_mismatch"] += 1
        observed_windows = batch.timeline_window_ids[i][:length]
        if observed_windows != sorted(observed_windows):
            violations["timeline_not_sorted"] += 1
        for j in range(length):
            if batch.timeline_concept_ids[i][j] == 0:
                violations["real_token_has_pad_concept"] += 1
            if any(not math.isfinite(value) for value in batch.timeline_features[i][j]):
                violations["timeline_nonfinite"] += 1
            window_id = batch.timeline_window_ids[i][j]
            window_counts[batch.timeline_window_names[window_id]] += 1
        for j in range(length, len(batch.timeline_attention_mask[i])):
            if batch.timeline_attention_mask[i][j]:
                violations["padding_marked_real"] += 1
            if batch.timeline_concept_ids[i][j] != 0:
                violations["padding_concept_nonzero"] += 1

    for i, length in enumerate(batch.global_summary_lengths):
        if sum(batch.global_summary_attention_mask[i]) != length:
            violations["summary_attention_length_mismatch"] += 1
        for j in range(length):
            if any(not math.isfinite(value) for value in batch.global_summary_features[i][j]):
                violations["summary_nonfinite"] += 1


def _check(name: str, observed: object, expected: object, passed: bool) -> dict[str, object]:
    return {"name": name, "observed": observed, "expected": expected, "passed": passed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the final shared-timeline collator without training.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    report = validate(args.data_dir, args.artifacts, args.report, batch_size=args.batch_size)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
