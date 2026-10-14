from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from aecopd_loader import AecopdDataBundle, buffered_shuffle


EXPECTED_ADMISSIONS = {"train": 4008, "validation": 797, "test": 851}
EXPECTED_PATIENTS = {"train": 2316, "validation": 486, "test": 524}
EXPECTED_TOTALS = {"events": 2_446_659, "numeric_summaries": 324_383, "text_chunks": 154_168}


def check(name: str, observed: Any, expected: Any, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "observed": observed, "expected": expected, "passed": bool(passed), "detail": detail}


def validate(data_dir: Path, artifacts_path: Path, report_path: Path) -> dict[str, Any]:
    bundle = AecopdDataBundle(data_dir)
    artifacts = bundle.fit_preprocessing(artifacts_path)

    admission_counts: Counter[str] = Counter()
    patients: dict[str, set[int]] = defaultdict(set)
    subject_splits: dict[int, set[str]] = defaultdict(set)
    hadm_ids: set[int] = set()
    hadm_ids_by_split: dict[str, list[int]] = defaultdict(list)
    label_counts: Counter[int] = Counter()
    normalization_methods: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    violations: Counter[str] = Counter()
    maxima = {"events_per_admission": 0, "summaries_per_admission": 0, "texts_per_admission": 0, "text_tokens": 0}

    for sample in bundle.iter_all_samples(artifacts=artifacts, include_text=True):
        admission_counts[sample.data_split] += 1
        patients[sample.data_split].add(sample.subject_id)
        subject_splits[sample.subject_id].add(sample.data_split)
        hadm_ids_by_split[sample.data_split].append(sample.hadm_id)
        label_counts[sample.event_type] += 1
        if sample.hadm_id in hadm_ids:
            violations["duplicate_hadm"] += 1
        hadm_ids.add(sample.hadm_id)
        if not 0.0 <= sample.time_days <= 365.0:
            violations["label_time_out_of_range"] += 1

        for value in sample.static_continuous.values():
            if not math.isfinite(value):
                violations["static_nonfinite"] += 1

        totals["events"] += len(sample.events)
        totals["numeric_summaries"] += len(sample.numeric_summaries)
        totals["text_chunks"] += len(sample.text_chunks)
        maxima["events_per_admission"] = max(maxima["events_per_admission"], len(sample.events))
        maxima["summaries_per_admission"] = max(maxima["summaries_per_admission"], len(sample.numeric_summaries))
        maxima["texts_per_admission"] = max(maxima["texts_per_admission"], len(sample.text_chunks))

        previous_time = -math.inf
        for event in sample.events:
            if event.hours_from_admission < previous_time:
                violations["event_time_not_sorted"] += 1
            previous_time = event.hours_from_admission
            if not math.isfinite(event.value_scaled):
                violations["event_scaled_nonfinite"] += 1
            if event.numeric_mask:
                normalization_methods[event.normalization_method] += 1
                if not event.normalization_method:
                    violations["numeric_without_method"] += 1

        for summary in sample.numeric_summaries:
            if not math.isfinite(summary.measurement_count_log1p):
                violations["summary_count_nonfinite"] += 1
            if any(not math.isfinite(value) for value in summary.scaled_values.values()):
                violations["summary_scaled_nonfinite"] += 1

        role_ranks: dict[str, list[int]] = defaultdict(list)
        for text in sample.text_chunks:
            maxima["text_tokens"] = max(maxima["text_tokens"], text.token_count)
            role_ranks[text.document_role].append(text.document_rank)
            if text.token_count <= 0 or text.token_count > 32_000:
                violations["text_token_rule"] += 1
            if text.chunk_text == "":
                violations["empty_text_chunk"] += 1
        for ranks in role_ranks.values():
            if len(ranks) != len(set(ranks)):
                violations["duplicate_text_rank"] += 1

    deterministic_checks = {}
    for split, ids in hadm_ids_by_split.items():
        first = list(buffered_shuffle(ids, 256, 20260912))[:64]
        second = list(buffered_shuffle(ids, 256, 20260912))[:64]
        deterministic_checks[split] = first == second

    checks: list[dict[str, Any]] = []
    checks.append(check("住院键总数", len(hadm_ids), 5656, len(hadm_ids) == 5656))
    for split in ("train", "validation", "test"):
        checks.append(check(f"{split}住院数", admission_counts[split], EXPECTED_ADMISSIONS[split], admission_counts[split] == EXPECTED_ADMISSIONS[split]))
        checks.append(check(f"{split}患者数", len(patients[split]), EXPECTED_PATIENTS[split], len(patients[split]) == EXPECTED_PATIENTS[split]))
        checks.append(check(f"{split}固定种子批次可重复", deterministic_checks.get(split), True, deterministic_checks.get(split) is True))
    cross_split = sum(1 for splits in subject_splits.values() if len(splits) > 1)
    checks.append(check("患者跨数据集数", cross_split, 0, cross_split == 0))
    checks.append(check("标签分布", dict(sorted(label_counts.items())), {0: 2881, 1: 1755, 2: 1020}, label_counts == Counter({0: 2881, 1: 1755, 2: 1020})))
    for key, expected in EXPECTED_TOTALS.items():
        checks.append(check(key, totals[key], expected, totals[key] == expected))
    checks.append(check("训练未见数值事件", normalization_methods["unseen_in_train"], 51, normalization_methods["unseen_in_train"] == 51, "缩放值为0且availability=false"))
    checks.append(check("所有装载器违例", sum(violations.values()), 0, sum(violations.values()) == 0, json.dumps(violations, ensure_ascii=False, sort_keys=True)))

    report = {
        "schema_version": "aecopd-model-v1-loader-validation-1",
        "passed": all(item["passed"] for item in checks),
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "normalization_methods": dict(sorted(normalization_methods.items())),
        "maxima": maxima,
        "violations": dict(sorted(violations.items())),
        "validation_scope": "model-independent data assembly only; no model, loss, prediction, discrimination, or calibration metrics",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the AECOPD model_v1 streaming data loader without training a model.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.data_dir, args.artifacts, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
