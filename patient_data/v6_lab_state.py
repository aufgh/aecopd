"""V6 全住院检验状态：只读源数据，独立派生，不读取结局内容。"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import time

import numpy as np

from patient_data.export_mimiciv import rows, sha256
from patient_data.expanded_abnormality_features import AXES, ADDED_AXES, directional_severity
from patient_data.lab_recovery_features import AXIS_CONCEPT, abnormal_severity, PAIRED_AXIS
from patient_data.preprocess_events import LABS
from training.v5.numeric_preprocessing import robust_parameters

NUMERIC = ("first_observed_severity", "observed_peak_severity", "latest_observed_severity",
           "recovery_delta", "latest_measurement_age", "peak_to_latest_elapsed")
FLAGS = ("axis_observed", "post_peak_recheck_valid", "latest_at_abnormal_peak")
# 与已经临床核验的类别一致；不接入 other_antibiotic，不加入利尿剂。
MED_GROUPS = {
    "short_beta_agonist": ("saba",), "short_muscarinic_antagonist": ("sama",),
    "long_beta_agonist": ("laba",), "long_muscarinic_antagonist": ("lama",),
    "inhaled_corticosteroid": ("inhaled_steroid",),
    "systemic_corticosteroid": ("systemic_steroid",),
    "common_systemic_antibacterial": ("macrolide", "beta_lactam", "fluoroquinolone", "tetracycline"),
}


def summarize(samples: list[tuple[float, float]], discharge_day: float) -> tuple[np.ndarray, dict]:
    """同刻取中位数；在无截顶异常程度上定位最早/最后峰值。"""
    if not math.isfinite(discharge_day) or discharge_day <= 0:
        raise ValueError("invalid discharge duration")
    grouped = defaultdict(list)
    for day, severity in samples:
        if not (math.isfinite(day) and math.isfinite(severity)
                and 0 <= day <= discharge_day and severity >= 0):
            raise ValueError("invalid observation")
        grouped[day].append(severity)
    if not grouped:
        return np.zeros(9), {"measurements": 0}
    ordered = [(t, float(np.median(v))) for t, v in sorted(grouped.items())]
    peak = max(v for _, v in ordered)
    # 定位峰值用精确最大值，末次峰标记另用固定浮点容差。
    peak_times = [t for t, v in ordered if v == peak]
    first_peak = peak_times[0]
    latest_time, latest = ordered[-1]
    valid = latest_time > first_peak
    result = np.array([ordered[0][1], peak, latest, peak - latest if valid else 0.,
                       math.log1p(discharge_day - latest_time), math.log1p(latest_time - first_peak),
                       1., float(valid), float(peak > 0 and math.isclose(latest, peak, rel_tol=1e-7, abs_tol=1e-8))])
    return result, {"measurements": len(ordered), "first_peak_day": first_peak,
                    "last_peak_day": peak_times[-1], "latest_day": latest_time}


def scale_on_train(raw: np.ndarray, splits: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """只有有测/有效恢复的训练住院参与尺度拟合；标记保持二元。"""
    if raw.ndim != 3 or raw.shape[2] != 9 or len(splits) != len(raw):
        raise ValueError("invalid state shape")
    scaled = raw.copy()
    valid = np.broadcast_to(raw[:, :, 6:7] > 0, raw[:, :, :6].shape).copy()
    valid[:, :, 3] &= raw[:, :, 7] > 0
    valid[:, :, 5] &= raw[:, :, 7] > 0
    transformed = raw[:, :, :6].copy()
    transformed[:, :, :4] = np.log1p(transformed[:, :, :4])
    parameters = {}
    for a, axis in enumerate(AXES):
        for f, name in enumerate(NUMERIC):
            selected = valid[:, a, f] & (splits == "train")
            key = f"{axis}__{name}"
            if not selected.any():
                parameters[key] = {"method": "no_training_observation", "training_observations": 0}
                valid[:, a, f] = False
                scaled[:, a, f] = 0
                continue
            p = robust_parameters(transformed[selected, a, f])
            values = (transformed[:, a, f] - p["median"]) / p["scale"]
            scaled[:, a, f] = np.where(valid[:, a, f], np.clip(values, -8, 8), 0)
            parameters[key] = p
    return scaled.astype(np.float32), valid, parameters


def strict_interval_exposure(start: float, end: float, times: list[float]) -> bool:
    """只计严格位于两次采样之间的执行记录；边界同刻不推断先后。"""
    if not start < end:
        raise ValueError("interval must increase")
    return bisect_right(times, start) < bisect_left(times, end)


def build(snapshot: Path, processed: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    source_manifest = snapshot / "metadata/manifest.json"
    sm = json.loads(source_manifest.read_text(encoding="utf-8"))
    pm_path = processed / "manifest.json"
    pm = json.loads(pm_path.read_text(encoding="utf-8"))
    if sm["status"] != "validated" or pm["status"] != "preprocessed_and_checked":
        raise ValueError("unvalidated source")
    if pm.get("medication_mapping") != "clinical_reviewed_2026_09":
        raise ValueError("expected clinically reviewed medication mapping")
    inputs = {}
    for name in ("cohort/admissions", "cohort/patient_splits", "raw/labevents"):
        digest = sha256(snapshot / (name + ".csv.gz"))
        if digest != sm["sources"][name]["sha256"] or digest != pm["input_sha256"][name]:
            raise ValueError(f"raw lineage mismatch: {name}")
        inputs[name] = digest
    events_path = processed / "events.csv.gz"
    if sha256(events_path) != pm["output_sha256"]["events.csv.gz"]:
        raise ValueError("processed events changed")
    inputs["processed_events"] = pm["output_sha256"]["events.csv.gz"]
    admissions = {}
    subjects = {}
    for row in rows(snapshot / "cohort/admissions.csv.gz"):
        if row["discharge_landmark_eligible"] not in ("t", "true", "1"):
            continue
        hadm = row["hadm_id"]
        if hadm in admissions:
            raise ValueError("duplicate admission")
        admissions[hadm] = (datetime.fromisoformat(row["admittime"]), datetime.fromisoformat(row["dischtime"]))
        subjects[hadm] = row["subject_id"]
    if len(admissions) != 19401 or len(set(subjects.values())) != 5351:
        raise ValueError("cohort drift")
    split_rows = list(rows(snapshot / "cohort/patient_splits.csv.gz"))
    splits = {r["subject_id"]: r["patient_split"] for r in split_rows}
    if len(splits) != len(split_rows) or not set(splits.values()) <= {"train", "validation", "test"}:
        raise ValueError("split drift")
    hadms = sorted(admissions, key=lambda h: (int(subjects[h]), admissions[h][0], int(h)))
    by_class = {c: g for g, cs in MED_GROUPS.items() for c in cs}
    medication = defaultdict(dict)
    accepted_gas = set()
    for row in rows(events_path):
        h = row["hadm_id"]
        if h not in admissions or row["patient_split"] != splits[subjects[h]]:
            raise ValueError("processed member/split mismatch")
        if row["kind"] == "lab" and row["concept"] in ("ph:arterial", "pco2:arterial"):
            accepted_gas.add(row["source_id"])
        if row["kind"] != "medication" or row["concept"] not in by_class:
            continue
        day = float(row["elapsed_days"])
        length = (admissions[h][1] - admissions[h][0]).total_seconds() / 86400
        if not math.isfinite(day) or not 0 <= day <= length:
            raise ValueError("medication outside admission")
        slot = medication[h, by_class[row["concept"]]]
        previous = slot.setdefault(row["source_id"], day)
        if previous != day:
            raise ValueError("conflicting administration time")
    samples = defaultdict(list)
    paired = defaultdict(dict)
    axis_audit = defaultdict(Counter)
    item_axes = defaultdict(list)
    for axis, concept in AXIS_CONCEPT.items():
        item, unit = next((i, u) for i, (c, u) in LABS.items() if c == concept.split(":")[0])
        item_axes[item].append((axis, "", unit))
    for axis, (item, direction, unit) in ADDED_AXES.items():
        item_axes[str(item)].append((axis, direction, unit))
    total = 0
    for row in rows(snapshot / "raw/labevents.csv.gz"):
        total += 1
        h, item = row["hadm_id"], row["itemid"]
        if h not in admissions or item not in item_axes:
            continue
        start, end = admissions[h]
        try:
            chart, stored = datetime.fromisoformat(row["charttime"]), datetime.fromisoformat(row["storetime"])
            values = tuple(float(row[k]) for k in ("valuenum", "ref_range_lower", "ref_range_upper"))
            value, lower, upper = values
            reason = ("time_unavailable" if not start <= chart <= stored <= end else
                      "invalid_numeric_or_reference" if not all(map(math.isfinite, values)) or value <= 0 or lower >= upper else None)
        except (ValueError, TypeError):
            reason = "invalid_time_numeric_or_reference"
        for axis, direction, unit in item_axes[item]:
            counts = axis_audit[axis]
            counts["raw_rows"] += 1
            reject = reason
            if row["valueuom"].strip() != unit:
                reject = "unit_mismatch_or_missing"
            if axis.startswith("arterial_") and row["labevent_id"] not in accepted_gas:
                reject = "not_in_validated_arterial_events"
            if reject:
                counts[reject] += 1
                continue
            day = (chart - start).total_seconds() / 86400
            severity = (directional_severity(value, lower, upper, direction, severity_cap=None) if direction
                        else abnormal_severity(axis, value, lower, upper, severity_cap=None))
            samples[h, axis].append((day, severity))
            counts["accepted_rows"] += 1
            if axis.startswith("arterial_") and row["specimen_id"] not in ("", "\\N"):
                slot = paired[h, row["specimen_id"]]
                if axis in slot:
                    raise ValueError("duplicate gas component within specimen")
                slot[axis] = (day, severity)
    if total != sm["sources"]["raw/labevents"]["rows"]:
        raise ValueError("raw laboratory count changed")
    for (h, _), slot in paired.items():
        if len(slot) != 2:
            continue
        (t1, s1), (t2, s2) = slot["arterial_low_ph"], slot["arterial_high_pco2"]
        if abs(t1 - t2) <= 1 / 24:
            samples[h, PAIRED_AXIS].append((max(t1, t2), min(s1, s2)))
    print(json.dumps({"stage": "raw_labs_scanned", "rows": total}), flush=True)
    raw = np.zeros((len(hadms), len(AXES), 9), dtype=np.float64)
    med = np.zeros((len(hadms), len(MED_GROUPS)), dtype=np.float32)
    peak_times = np.full((len(hadms), len(AXES), 3), np.nan)
    cycle_stats = defaultdict(Counter)
    for i, h in enumerate(hadms):
        duration = (admissions[h][1] - admissions[h][0]).total_seconds() / 86400
        drug_times = {g: sorted(set(medication[h, g].values())) for g in MED_GROUPS}
        for g, group in enumerate(MED_GROUPS):
            med[i, g] = bool(drug_times[group])
        for a, axis in enumerate(AXES):
            obs = samples.get((h, axis), [])
            raw[i, a], metadata = summarize(obs, duration)
            if metadata["measurements"]:
                peak_times[i, a] = [metadata["first_peak_day"], metadata["last_peak_day"], metadata["latest_day"]]
            for name, flag in zip(FLAGS, raw[i, a, 6:], strict=True):
                axis_audit[axis][name + "_admissions"] += int(flag)
            axis_audit[axis]["observed_recovery_admissions"] += int(raw[i, a, 3] > 0)
            times = sorted({t for t, _ in obs})
            if len(times) < 2:
                continue
            cycle_stats[axis]["admissions_with_pairs"] += 1
            cycle_stats[axis]["pairs"] += len(times) - 1
            for left, right in zip(times[:-1], times[1:], strict=True):
                for group in MED_GROUPS:
                    cycle_stats[axis][group + "_exposed_intervals"] += strict_interval_exposure(left, right, drug_times[group])
    split_array = np.asarray([splits[subjects[h]] for h in hadms])
    scaled, value_masks, parameters = scale_on_train(raw, split_array)
    # 在写文件前再次核对源文件，防止运行期间源被替换。
    for name in ("cohort/admissions", "cohort/patient_splits", "raw/labevents"):
        if sha256(snapshot / (name + ".csv.gz")) != inputs[name]:
            raise ValueError("source changed during build")
    if sha256(events_path) != inputs["processed_events"]:
        raise ValueError("events changed during build")
    output.mkdir(parents=True)
    file = output / "clinical_state.npz"
    np.savez_compressed(file, hadm_ids=np.asarray(hadms, dtype=np.int64),
                        subject_ids=np.asarray([subjects[h] for h in hadms], dtype=np.int64),
                        patient_splits=split_array, lab_axis_raw=raw, lab_axis_inputs=scaled,
                        lab_value_masks=value_masks, axis_names=np.asarray(AXES),
                        feature_names=np.asarray(NUMERIC + FLAGS), peak_audit_days=peak_times,
                        med_exposure=med, med_group_names=np.asarray(tuple(MED_GROUPS)))
    with np.load(file, allow_pickle=False) as check:
        loaded = {k: check[k] for k in check.files}
    if not np.array_equal(loaded["lab_axis_inputs"], scaled) or not np.isfinite(scaled).all():
        raise ValueError("derived roundtrip failed")
    (output / "normalization.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    report = {"status": "derived_and_roundtrip_checked_not_prediction_validated",
              "admissions": len(hadms), "patients": len(set(subjects.values())),
              "lab_shape": list(raw.shape), "med_shape": list(med.shape),
              "source_hashes": inputs, "source_manifest_sha256": sha256(source_manifest),
              "processed_manifest_sha256": sha256(pm_path), "builder_sha256": sha256(Path(__file__)),
              "output_sha256": {"clinical_state.npz": sha256(file),
                                "normalization.json": sha256(output / "normalization.json")},
              "axis_audit": {a: dict(axis_audit[a]) for a in AXES},
              "lab_intervention_interval_coverage": {a: dict(cycle_stats[a]) for a in AXES},
              "medication_coverage": {g: int(med[:, i].sum()) for i, g in enumerate(MED_GROUPS)},
              "limitations": ["Frozen eMAR mapping is inherited, not a new dose or indication adjudication",
                              "No eMAR evidence is not confirmed absence of treatment",
                              "Cycle coverage is not a treatment effect; boundary-time drug events excluded",
                              "Per-axis timestamp aggregation is not a dense state vector",
                              "No new predictors, labels, text encoding or model training in this builder",
                              "Lab measurements require positive numeric values and valid reference interval"],
              "elapsed_seconds": round(time.perf_counter() - started, 2)}
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.snapshot, args.processed, args.output)
    print(json.dumps({k: result[k] for k in ("status", "admissions", "patients", "elapsed_seconds")}))
