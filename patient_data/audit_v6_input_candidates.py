"""只读核对 V6 未用候选与复测时间；输出聚合统计，不导出患者文本。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import re
import time

from patient_data.export_mimiciv import rows, sha256
from patient_data.preprocess_events import specimen_name


CANDIDATES = {
    "eosinophils_percent": ("51200",),
    "eosinophils_absolute": ("51199", "52073"),
    "neutrophils_percent": ("51256",),
    "neutrophils_absolute": ("52075",),
    "blood_gas_oxygen_saturation": ("50817",),
    "blood_gas_po2": ("50821",),
    "base_excess": ("50802",),
    "anion_gap": ("50868",),
    "chloride": ("50902",),
    "rdw": ("51277",),
    "hematocrit": ("51221",),
    "troponin": ("51002", "51003", "52642"),
}
TRACKS = {
    "arterial_ph": ("50820",), "arterial_pco2": ("50818",),
    "bicarbonate": ("50882",), "wbc": ("51301",),
    "albumin": ("50862",), "hemoglobin": ("51222",),
    "creatinine": ("50912",), "sodium": ("50983",),
    "potassium": ("50971",), "urea_nitrogen": ("51006",),
    "glucose": ("50931",), "platelet": ("51265",),
    "magnesium": ("50960",), "lactate": ("50813",),
    "crp": ("50889",), "ntprobnp": ("50963",),
}
# 只识别检查名称/标题，不依据 body 中“建议未来CT”判定当前确有CT。
CHEST_CT = re.compile(
    r"(?:\bCT\s+(?:OF\s+(?:THE\s+)?)?(?:CHEST|THORAX)\b|"
    r"\bCHEST\s+CT\b|\bCTA\s+(?:CHEST|PULMONARY)\b|"
    r"\bCT\s+(?:PULMONARY\s+)?ANGIOGRAM\b)", re.I)


def number(value: str) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def measurement_stats(samples, admissions, subjects):
    observed = len(samples)
    repeated = late_absent = early_and_late = 0
    for hadm, timestamps in samples.items():
        start, end = admissions[hadm]
        ordered = sorted(timestamps)
        repeated += len(ordered) >= 2
        late_absent += (end - ordered[-1]).total_seconds() > 48 * 3600
        early_and_late += ((ordered[0] - start).total_seconds() <= 24 * 3600
                           and (end - ordered[-1]).total_seconds() <= 48 * 3600)
    return {"observed_admissions": observed,
            "observed_patients": len({subjects[h] for h in samples}),
            "two_distinct_timestamps": repeated,
            "last_measurement_older_than_48h": late_absent,
            "first_in_24h_and_last_in_48h": early_and_late}


def audit(snapshot: Path, labels: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    manifest = json.loads((snapshot / "metadata/manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "validated":
        raise ValueError("snapshot not validated")
    admissions, subjects = {}, {}
    for row in rows(snapshot / "cohort/admissions.csv.gz"):
        if row["discharge_landmark_eligible"].lower() not in ("1", "t", "true"):
            continue
        hadm = row["hadm_id"]
        admissions[hadm] = (datetime.fromisoformat(row["admittime"]),
                            datetime.fromisoformat(row["dischtime"]))
        subjects[hadm] = row["subject_id"]
    if len(admissions) != 19401:
        raise ValueError("eligible cohort drift")
    dictionary = {r["itemid"]: r for r in rows(snapshot / "raw/d_labitems.csv.gz")}
    groups = {**CANDIDATES, **TRACKS}
    procalcitonin = tuple(k for k, v in dictionary.items()
                         if "procalciton" in v["label"].lower())
    if procalcitonin:
        groups["procalcitonin"] = procalcitonin
    by_item = defaultdict(list)
    for name, items in groups.items():
        for item in items:
            by_item[item].append(name)
    measurements = {name: defaultdict(set) for name in groups}
    units = {name: Counter() for name in groups}
    rejects = {name: Counter() for name in groups}
    specimen_types = defaultdict(set)
    gas = []
    raw_count = 0
    for row in rows(snapshot / "raw/labevents.csv.gz"):
        raw_count += 1
        hadm, item = row["hadm_id"], row["itemid"]
        if hadm not in admissions or (item not in by_item and item != "52033"):
            continue
        start, end = admissions[hadm]
        try:
            chart = datetime.fromisoformat(row["charttime"])
            store = datetime.fromisoformat(row["storetime"])
            available = start <= chart <= store <= end
        except ValueError:
            available = False
        if item == "52033":
            if available:
                specimen_types[(hadm, row["specimen_id"])].add(specimen_name(row["value"]))
            continue
        for name in by_item[item]:
            if not available:
                rejects[name]["time_unavailable"] += 1
                continue
            if number(row["valuenum"]) is None:
                rejects[name]["no_finite_numeric_value"] += 1
                continue
            if row["valueuom"] in ("", "\\N"):
                rejects[name]["no_unit"] += 1
                continue
            if name in ("arterial_ph", "arterial_pco2"):
                gas.append((name, hadm, row["specimen_id"], chart, row["valueuom"]))
            else:
                measurements[name][hadm].add(chart)
                units[name][row["valueuom"]] += 1
    if raw_count != manifest["sources"]["raw/labevents"]["rows"]:
        raise ValueError("raw laboratory row count drift")
    for name, hadm, specimen, chart, unit in gas:
        types = specimen_types.get((hadm, specimen), set())
        if types == {"arterial"}:
            measurements[name][hadm].add(chart)
            units[name][unit] += 1
        else:
            rejects[name]["not_confirmed_arterial_or_conflict"] += 1
    print(json.dumps({"stage": "raw_labs_scanned", "rows": raw_count}), flush=True)
    if not measurements["arterial_ph"] or not measurements["arterial_pco2"]:
        raise ValueError("arterial mapping missing despite existing validated arterial inputs")
    lab_results = {}
    for name, items in groups.items():
        lab_results[name] = {
            "items": [dictionary[item] for item in items],
            "units_rows": dict(units[name]), "excluded_rows": dict(rejects[name]),
            **measurement_stats(measurements[name], admissions, subjects)}
    ct_hadms, ct_subjects = set(), set()
    ct_count = report_count = 0
    for row in rows(snapshot / "raw/radiology.csv.gz"):
        report_count += 1
        hadm = row["hadm_id"]
        if hadm not in admissions:
            continue
        start, end = admissions[hadm]
        try:
            if not start <= datetime.fromisoformat(row["charttime"]) <= datetime.fromisoformat(row["storetime"]) <= end:
                continue
        except ValueError:
            continue
        # 常见当前检查标题在全文开头；启发式候选，不宣称临床真值/召回率。
        if CHEST_CT.search(row["text"][:900]):
            ct_count += 1
            ct_hadms.add(hadm); ct_subjects.add(subjects[hadm])
    if report_count != manifest["sources"]["raw/radiology"]["rows"]:
        raise ValueError("raw radiology row count drift")
    splits = {r["subject_id"]: r["patient_split"]
              for r in rows(snapshot / "cohort/patient_splits.csv.gz")}
    boundaries = [0, 7, 14, 30, 60, 90, 180, 365]
    events = {split: {str(cause): [0] * 7 for cause in (1, 2)}
              for split in ("train", "validation", "test")}
    target_sets = {split: [set() for _ in range(7)] for split in events}
    label_rows = list(rows(labels))
    if {r["hadm_id"] for r in label_rows} != set(admissions) or len(label_rows) != len(admissions):
        raise ValueError("strict label membership drift")
    for row in label_rows:
        cause, day = int(row["event_type"]), float(row["time_days"])
        if cause == 0:
            continue
        # 与候选模型一致，段为 [a,b)，365 位于末段；同日 0 在首段。
        k = min(sum(day >= edge for edge in boundaries[1:-1]), 6)
        split = splits[row["subject_id"]]
        events[split][str(cause)][k] += 1
        if cause == 1:
            target = row["next_copd_hadm_id"]
            if target in ("", "\\N"):
                raise ValueError("positive readmission label has no target admission")
            target_sets[split][k].add(target)
    result = {
        "status": "aggregate_readonly_input_audit", "eligible_admissions": len(admissions),
        "eligible_patients": len(set(subjects.values())),
        "source_manifest_sha256": sha256(snapshot / "metadata/manifest.json"),
        "source_file_sha256_verified": {
            name: sha256(snapshot / (name + ".csv.gz")) == manifest["sources"][name]["sha256"]
            for name in ("raw/labevents", "raw/radiology", "raw/d_labitems")},
        "strict_labels_sha256": sha256(labels),
        "lab_scan_rows": raw_count, "lab_candidates_and_tracks": lab_results,
        "procalcitonin_dictionary_items": list(procalcitonin),
        "chest_ct_title_candidate": {"all_radiology_rows": report_count,
            "candidate_reports": ct_count, "candidate_admissions": len(ct_hadms),
            "candidate_patients": len(ct_subjects),
            "method": "first 900 chars title regex; chart/store available by discharge; not adjudicated CT truth"},
        "strict_event_counts_by_time_segment": events,
        "distinct_readmission_targets_per_segment": {
            split: [len(targets) for targets in bins] for split, bins in target_sets.items()},
        "distinct_readmission_targets_per_split": {
            split: len(set().union(*bins)) for split, bins in target_sets.items()},
        "segment_boundaries": boundaries,
        "limitations": ["Numeric/time/unit presence is not full model inclusion QA",
                        "Candidate lab codes not equivalent to approved predictors",
                        "Repeat timestamp counts do not prove abnormality or recovery",
                        "One future admission can label multiple landmarks and appear in different time segments",
                        "Whole dataset coverage is not supervised validation evidence",
                        "No live PostgreSQL access or microbiology export in this audit"],
        "elapsed_seconds": round(time.perf_counter() - started, 2)}
    if not all(result["source_file_sha256_verified"].values()):
        raise ValueError("source SHA mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output),
                      "seconds": result["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.snapshot, args.labels, args.output)
