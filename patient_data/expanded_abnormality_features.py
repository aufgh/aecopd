"""从出院前真实检验构建方向明确的异常轨迹；不输入原始事件序列。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from patient_data.lab_recovery_features import AXES as ORIGINAL_AXES
from patient_data.lab_recovery_features import FEATURE_NAMES as ORIGINAL_NAMES
from patient_data.lab_recovery_features import SUMMARY_NAMES, summarize_axis


# 同一项目的双向异常共用原始观测；不把相关项目（如 Hb/Hct）都加进来。
ADDED_AXES: dict[str, tuple[int, str, str]] = {
    "wbc_low": (51301, "low", "K/uL"),
    "sodium_low": (50983, "low", "mEq/L"),
    "sodium_high": (50983, "high", "mEq/L"),
    "potassium_low": (50971, "low", "mEq/L"),
    "potassium_high": (50971, "high", "mEq/L"),
    "urea_nitrogen_high": (51006, "high", "mg/dL"),
    "glucose_low": (50931, "low", "mg/dL"),
    "glucose_high": (50931, "high", "mg/dL"),
    "platelet_low": (51265, "low", "K/uL"),
    "magnesium_low": (50960, "low", "mg/dL"),
    "lactate_high": (50813, "high", "mmol/L"),
    "crp_high": (50889, "high", "mg/L"),
    "ntprobnp_high": (50963, "high", "pg/mL"),
}
AXES = ORIGINAL_AXES + tuple(ADDED_AXES)
FEATURE_NAMES = ORIGINAL_NAMES + tuple(
    f"{axis}__{name}" for axis in ADDED_AXES for name in SUMMARY_NAMES
)


def _rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def directional_severity(value: float, lower: float, upper: float, direction: str,
                         *, severity_cap: float | None = 5.0) -> float:
    if direction not in ("low", "high") or not all(map(math.isfinite, (value, lower, upper))) or lower >= upper:
        raise ValueError("invalid result/reference range")
    offset = lower - value if direction == "low" else value - upper
    departure = max(0.0, offset / (upper - lower))
    return min(severity_cap, departure) if severity_cap is not None else departure


def build(original: Path, raw: Path, output: Path,
          *, severity_cap: float | None = 5.0) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    old_manifest = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    if old_manifest.get("severity_cap", 5.0) != severity_cap:
        raise ValueError("original and added abnormality axes must use the same severity cap")
    raw_manifest = json.loads((raw / "metadata/manifest.json").read_text(encoding="utf-8"))
    old_file = original / "lab_recovery.npz"
    raw_file = raw / "raw/labevents.csv.gz"
    if (old_manifest["status"] != "derived_not_yet_validated_by_prediction"
            or raw_manifest["status"] != "validated"
            or _sha(old_file) != old_manifest["output_sha256"]
            or _sha(raw_file) != raw_manifest["sources"]["raw/labevents"]["sha256"]):
        raise ValueError("unvalidated abnormality source")
    with np.load(old_file, allow_pickle=False) as archive:
        hadms = archive["hadm_ids"]
        old_features = archive["features"]
        if tuple(archive["feature_names"].tolist()) != ORIGINAL_NAMES:
            raise ValueError("original abnormality vocabulary changed")
    if old_features.shape != (len(hadms), len(ORIGINAL_NAMES)) or len(set(hadms.tolist())) != len(hadms):
        raise ValueError("original abnormality shape/identity changed")
    selected_hadms = {str(hadm) for hadm in hadms}
    landmarks = {}
    for row in _rows(raw / "cohort/admissions.csv.gz"):
        if row["hadm_id"] in selected_hadms:
            landmarks[row["hadm_id"]] = (datetime.fromisoformat(row["admittime"]),
                                           datetime.fromisoformat(row["dischtime"]))
    if set(landmarks) != selected_hadms:
        raise ValueError("abnormality admissions differ from raw cohort")
    by_item: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
    for axis, (itemid, direction, unit) in ADDED_AXES.items():
        by_item[itemid].append((axis, direction, unit))
    observations: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    audit: dict[str, Counter[str]] = defaultdict(Counter)
    for row in _rows(raw_file):
        itemid = int(row["itemid"])
        if itemid not in by_item or row["hadm_id"] not in landmarks:
            continue
        admission, discharge = landmarks[row["hadm_id"]]
        chart = datetime.fromisoformat(row["charttime"])
        stored = datetime.fromisoformat(row["storetime"]) if row["storetime"] not in ("", "\\N") else None
        value, lower, upper = (_number(row[key]) for key in ("valuenum", "ref_range_lower", "ref_range_upper"))
        for axis, direction, unit in by_item[itemid]:
            counts = audit[axis]
            counts["raw_rows"] += 1
            if not admission <= chart <= discharge or stored is None or not chart <= stored <= discharge:
                counts["not_available_by_discharge"] += 1
                continue
            if value is None or lower is None or upper is None or lower >= upper:
                counts["invalid_numeric_or_reference"] += 1
                continue
            if row["valueuom"] != unit:
                counts["unit_mismatch"] += 1
                continue
            day = (chart - admission).total_seconds() / 86400.0
            observations[(row["hadm_id"], axis)].append(
                (day, directional_severity(value, lower, upper, direction,
                                           severity_cap=severity_cap)))
            counts["accepted_rows"] += 1
    added = np.zeros((len(hadms), len(ADDED_AXES) * len(SUMMARY_NAMES)), dtype=np.float32)
    for index, hadm in enumerate(hadms):
        for axis_index, axis in enumerate(ADDED_AXES):
            summary = summarize_axis(observations.get((str(hadm), axis), []))
            lo = axis_index * len(SUMMARY_NAMES)
            added[index, lo:lo + len(SUMMARY_NAMES)] = summary
            audit[axis]["observed_admissions"] += bool(summary[0])
            audit[axis]["abnormal_admissions"] += summary[2] > 0
            audit[axis]["observed_recovery_admissions"] += summary[4] > 0
    features = np.concatenate((old_features, added), axis=1)
    if not np.isfinite(features).all() or (features < 0).any():
        raise ValueError("invalid derived abnormality values")
    output.mkdir(parents=True)
    np.savez_compressed(output / "abnormality.npz", hadm_ids=hadms, features=features,
                        feature_names=np.asarray(FEATURE_NAMES))
    report = {
        "status": "derived_not_yet_validated_by_prediction", "admissions": len(hadms),
        "severity_cap": severity_cap,
        "axes": AXES, "feature_count": len(FEATURE_NAMES),
        "added_axis_audit": {axis: dict(audit[axis]) for axis in ADDED_AXES},
        "input_sha256": {"original_abnormality": _sha(old_file), "raw_labevents": _sha(raw_file),
                         "processed_arrays": old_manifest["input_sha256"]["processed_arrays"]},
        "output_sha256": _sha(output / "abnormality.npz"),
        "limitations": ["Absent test is not normal", "Recovery requires later measured result",
                        "Reference-range departure is not COPD-specific or a treatment effect",
                        "Only results stored by discharge are eligible"],
    }
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uncapped", action="store_true", help="retain raw reference-range departure before log1p")
    args = parser.parse_args()
    result = build(args.original, args.raw, args.output,
                   severity_cap=None if args.uncapped else 5.0)
    print(json.dumps({"admissions": result["admissions"], "feature_count": result["feature_count"],
                      "added_axis_audit": result["added_axis_audit"]}, ensure_ascii=False))
