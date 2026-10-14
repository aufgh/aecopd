"""从已接受检验中提取低容量、方向明确的异常及复查恢复表示。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# 只把动脉 pH / pCO2 用于 COPD 高碳酸血症线索；不把静脉氧分压当 PaO2。
# PaO2 的给氧状态不可用，因此本轮不设单项氧合“恢复”阈值。
AXIS_CONCEPT = {
    "arterial_low_ph": "ph:arterial",
    "arterial_high_pco2": "pco2:arterial",
    "bicarbonate_low": "bicarbonate",
    "bicarbonate_high": "bicarbonate",
    "wbc_high": "wbc",
    "albumin_low": "albumin",
    "hemoglobin_low": "hemoglobin",
    "creatinine_high": "creatinine",
}
PAIRED_AXIS = "arterial_hypercapnic_acidosis"
AXES = (*AXIS_CONCEPT, PAIRED_AXIS)
SUMMARY_NAMES = ("observed", "first_severity", "peak_severity", "last_severity",
                 "post_peak_recovery", "observed_span_log_days")
FEATURE_NAMES = tuple(f"{axis}__{name}" for axis in AXES for name in SUMMARY_NAMES)


def abnormal_severity(axis: str, value: float, lower: float, upper: float,
                      *, severity_cap: float | None = 5.0) -> float:
    """按项目参考范围构建无量纲偏离；旧输入默认截断于 5。"""
    if not all(map(math.isfinite, (value, lower, upper))) or not lower < upper:
        raise ValueError("invalid laboratory result or reference interval")
    if axis == "arterial_low_ph":
        severity = (7.35 - value) / .10
    elif axis == "arterial_high_pco2":
        severity = (value - 45.0) / 15.0
    elif axis in AXIS_CONCEPT:
        width = upper - lower
        severity = ((lower - value) if axis.endswith("_low") else (value - upper)) / width
    else:
        raise ValueError(f"unknown abnormality axis: {axis}")
    if severity_cap is not None and severity_cap <= 0:
        raise ValueError("severity cap must be positive")
    return float(min(severity_cap, max(0.0, severity)) if severity_cap is not None
                 else max(0.0, severity))


def summarize_axis(samples: list[tuple[float, float]]) -> tuple[float, ...]:
    """只在峰值后有更晚复查时给恢复分；缺测和同刻记录不能冒充正常。"""
    if not samples:
        return (0., 0., 0., 0., 0., 0.)
    by_time: dict[float, list[float]] = defaultdict(list)
    for day, severity in samples:
        if not math.isfinite(day) or not math.isfinite(severity) or day < 0 or severity < 0:
            raise ValueError("invalid sample")
        by_time[day].append(severity)
    ordered = [(day, float(np.median(values))) for day, values in sorted(by_time.items())]
    peak_index = max(range(len(ordered)), key=lambda i: ordered[i][1])
    peak = ordered[peak_index][1]
    last = ordered[-1][1]
    recovery = max(0., (peak - last) / peak) if peak > 0 and peak_index < len(ordered) - 1 else 0.
    return (1., math.log1p(ordered[0][1]), math.log1p(peak), math.log1p(last),
            float(recovery), math.log1p(ordered[-1][0] - ordered[0][0]))


def _rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(processed: Path, raw: Path, output: Path,
          *, severity_cap: float | None = 5.0) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    processed_manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    raw_manifest = json.loads((raw / "metadata/manifest.json").read_text(encoding="utf-8"))
    if processed_manifest["status"] != "preprocessed_and_checked" or raw_manifest["status"] != "validated":
        raise ValueError("unvalidated laboratory sources")
    for name in ("events.csv.gz", "event_arrays.npz"):
        if _sha(processed / name) != processed_manifest["output_sha256"][name]:
            raise ValueError(f"processed laboratory source changed: {name}")
    if _sha(raw / "raw/labevents.csv.gz") != raw_manifest["sources"]["raw/labevents"]["sha256"]:
        raise ValueError("raw labevents source changed")
    with np.load(processed / "event_arrays.npz", allow_pickle=False) as archive:
        hadm_ids = archive["hadm_ids"]
    accepted = {}
    for row in _rows(processed / "events.csv.gz"):
        if row["kind"] == "lab":
            key = row["source_id"]
            if key in accepted:
                raise ValueError("repeated accepted laboratory source")
            accepted[key] = (row["hadm_id"], row["concept"], row["specimen_id"],
                             float(row["elapsed_days"]), float(row["value_raw"]))
    samples: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    specimens: dict[tuple[str, str], dict[str, tuple[float, float]]] = defaultdict(dict)
    matched = 0
    for row in _rows(raw / "raw/labevents.csv.gz"):
        record = accepted.get(row["labevent_id"])
        if record is None:
            continue
        hadm, concept, specimen_id, day, value = record
        if hadm != row["hadm_id"] or specimen_id != row["specimen_id"] or value != float(row["valuenum"]):
            raise ValueError("accepted/raw laboratory row mismatch")
        matched += 1
        lower, upper = float(row["ref_range_lower"]), float(row["ref_range_upper"])
        for axis, expected in AXIS_CONCEPT.items():
            if concept == expected:
                samples[(hadm, axis)].append((day, abnormal_severity(
                    axis, value, lower, upper, severity_cap=severity_cap)))
        if concept in ("ph:arterial", "pco2:arterial") and specimen_id not in ("", "\\N"):
            slot = specimens[(hadm, specimen_id)]
            if concept in slot:
                raise ValueError("multiple accepted arterial values in one specimen")
            slot[concept] = (day, abnormal_severity(
                "arterial_low_ph" if concept == "ph:arterial" else "arterial_high_pco2",
                value, lower, upper, severity_cap=severity_cap))
    if matched != len(accepted):
        raise ValueError("not all accepted laboratory sources were found")
    paired = 0
    for (hadm, _), slot in specimens.items():
        if len(slot) != 2:
            continue
        ph_day, ph_severity = slot["ph:arterial"]
        co2_day, co2_severity = slot["pco2:arterial"]
        if abs(ph_day - co2_day) > 1 / 24:
            continue
        samples[(hadm, PAIRED_AXIS)].append((max(ph_day, co2_day), min(ph_severity, co2_severity)))
        paired += 1
    output.mkdir(parents=True)
    features = np.zeros((len(hadm_ids), len(FEATURE_NAMES)), dtype=np.float32)
    coverage: Counter[str] = Counter()
    recovery: Counter[str] = Counter()
    for index, hadm in enumerate(hadm_ids):
        for axis_index, axis in enumerate(AXES):
            summary = summarize_axis(samples.get((str(hadm), axis), []))
            features[index, axis_index * len(SUMMARY_NAMES):(axis_index + 1) * len(SUMMARY_NAMES)] = summary
            coverage[axis] += bool(summary[0])
            recovery[axis] += summary[4] > 0
    np.savez_compressed(output / "lab_recovery.npz", hadm_ids=hadm_ids,
                        features=features, feature_names=np.asarray(FEATURE_NAMES))
    report = {"status": "derived_not_yet_validated_by_prediction", "admissions": len(hadm_ids),
              "severity_cap": severity_cap,
              "accepted_lab_events": matched, "paired_arterial_specimens": paired,
              "feature_count": len(FEATURE_NAMES), "coverage_admissions": dict(coverage),
              "post_peak_recovery_admissions": dict(recovery),
              "input_sha256": {"processed_events": _sha(processed / "events.csv.gz"),
                               "processed_arrays": _sha(processed / "event_arrays.npz"),
                               "raw_labevents": _sha(raw / "raw/labevents.csv.gz")},
              "output_sha256": _sha(output / "lab_recovery.npz"),
              "limitations": ["Post-peak recovery requires a later observed measurement; no observation is not normal.",
                              "Arterial pH and pCO2 are not interchangeable with venous samples.",
                              "PaO2 is excluded because supplemental oxygen/FiO2 is not available in this input.",
                              "Laboratory reference range abnormality is not a diagnosis or treatment effect."]}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uncapped", action="store_true", help="retain raw reference-range departure before log1p")
    args = parser.parse_args()
    print(json.dumps(build(args.processed, args.raw, args.output,
                           severity_cap=None if args.uncapped else 5.0), ensure_ascii=False))
