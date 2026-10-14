"""只输出聚合信息，分析换结局后病例组成与固定风险分数为何改变。"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np


def records(path: Path) -> dict[str, dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return {row["hadm_id"]: row for row in csv.DictReader(handle)}


def analyze(union_labels: Path, strict_labels: Path, base_csv: Path, cohort_csv: Path,
            union_run: Path, output_path: Path) -> dict[str, object]:
    union = records(union_labels)
    strict = records(strict_labels)
    base = records(base_csv)
    cohort = records(cohort_csv)
    if set(union) != set(strict) or set(union) != set(base) or not set(union).issubset(cohort):
        raise ValueError("endpoint audit membership mismatch")
    for hadm in union:
        if any(union[hadm][key] != strict[hadm][key] for key in ("subject_id", "admittime", "dischtime")):
            raise ValueError("endpoint audit index identity mismatch")
    groups: dict[str, list[str]] = {name: [] for name in
                                    ("strict_readmission", "union_only_readmission", "union_death", "union_censor")}
    for hadm, broad in union.items():
        if strict[hadm]["event_type"] == "1":
            group = "strict_readmission"
        elif broad["event_type"] == "1":
            group = "union_only_readmission"
        elif broad["event_type"] == "2":
            group = "union_death"
        else:
            group = "union_censor"
        groups[group].append(hadm)
    with np.load(union_run / "historical_test_predictions.npz") as archive:
        hadms = archive["hadm_ids"].astype(str)
        risk = archive["risk"]
        predicted_events = archive["events"]
        predicted_times = archive["times"]
    if len(set(hadms.tolist())) != len(hadms):
        raise ValueError("duplicate historical prediction")
    if any(predicted_events[i] != int(union[h]["event_type"])
           or not np.isclose(predicted_times[i], float(union[h]["time_days"]), atol=1e-4)
           for i, h in enumerate(hadms)):
        raise ValueError("historical prediction uses different union labels")
    by_hadm = {hadm: position for position, hadm in enumerate(hadms)}
    aggregate: dict[str, dict[str, object]] = {}
    for name, members in groups.items():
        subjects = {union[h]["subject_id"] for h in members}
        current_strict = sum(cohort[h]["is_strict_aecopd"] == "t" for h in members)
        prior_coded = np.array([float(base[h]["prior_coded_admission_count"]) for h in members])
        prior_strict = np.array([float(base[h]["prior_strict_aecopd_count"]) for h in members])
        positions = [by_hadm[h] for h in members if h in by_hadm]
        scores = risk[positions] if positions else np.zeros((0, risk.shape[1]))
        aggregate[name] = {"index_discharges": len(members), "unique_patients": len(subjects),
            "current_index_has_strict_code_fraction": current_strict / len(members),
            "prior_coded_admissions_mean": float(prior_coded.mean()),
            "prior_coded_admissions_median": float(np.median(prior_coded)),
            "prior_strict_admissions_mean": float(prior_strict.mean()),
            "historical_test_index_discharges": len(positions),
            "fixed_union_model_risk_90_mean": float(scores[:, 90].mean()) if len(scores) else None,
            "fixed_union_model_risk_365_mean": float(scores[:, 365].mean()) if len(scores) else None}
    both = groups["strict_readmission"]
    same_first = sum(union[h]["next_copd_hadm_id"] == strict[h]["next_copd_hadm_id"] for h in both)
    later_strict = len(both) - same_first
    dropped = groups["union_only_readmission"]
    nonstrict_first = sum(union[h]["next_is_strict_aecopd"] == "f" for h in dropped)
    if nonstrict_first != len(dropped):
        raise ValueError("union-only event unexpectedly has strict first event")
    result = {"status": "aggregate_endpoint_shift_checked", "groups": aggregate,
              "strict_events_with_same_first_union_event": same_first,
              "strict_events_after_an_earlier_nonstrict_union_event": later_strict,
              "union_only_first_future_event_is_nonstrict": nonstrict_first,
              "disclaimer": "Risk means use fixed union-trained model on historical test, and are descriptive, not causal effects or clinical adjudication."}
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("union_labels", "strict_labels", "base_csv", "cohort_csv", "union_run", "output_path"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.union_labels, args.strict_labels, args.base_csv,
                             args.cohort_csv, args.union_run, args.output_path)))
