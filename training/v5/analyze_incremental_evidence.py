"""严格结局同患者增量证据：纯历史 Cox、层内比较与患者聚类区间。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from training.baselines.continuous_competing_cox import fit_competing_cox, predict_cumulative_incidence
from training.v5.section_alignment_data.build_dataset import patient_split
from training.v5.train import GRID, _public_metrics, metrics
from training.v5.compare_strict_ablations import _paired_auc_interval
from training.baselines.rank_metrics import auc_rank


def _rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _load_predictions(run_dir: Path, split: str):
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "training_completed":
        raise ValueError(f"unfinished training: {run_dir}")
    with np.load(run_dir / f"{split}_predictions.npz") as archive:
        prediction = {key: archive[key] for key in archive.files}
    return report, prediction


def history_features(base_rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    """只用当前入院前已结束的历史与人口信息；不含本次诊断/治疗/文本。"""
    by_subject: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(base_rows):
        by_subject[int(row["subject_id"])].append(i)
    gaps = np.zeros(len(base_rows), dtype=np.float64)
    for indices in by_subject.values():
        indices.sort(key=lambda i: (base_rows[i]["admittime"], int(base_rows[i]["hadm_id"])))
        previous = None
        for i in indices:
            admission = datetime.fromisoformat(base_rows[i]["admittime"])
            if previous is not None:
                gaps[i] = np.log1p(max(0.0, (admission - previous).total_seconds() / 86400.0))
            previous = datetime.fromisoformat(base_rows[i]["dischtime"])
    prior = np.array([int(row["prior_coded_admission_count"]) for row in base_rows])
    strict = np.array([int(row["prior_strict_aecopd_count"]) for row in base_rows])
    age = np.array([float(row["age_at_admission"]) for row in base_rows])
    sex = np.array([row["gender"].upper() for row in base_rows])
    if np.any(prior < 0) or np.any(strict < 0) or not np.isfinite(age).all() or set(sex) - {"M", "F"}:
        raise ValueError("invalid history features")
    features = np.column_stack(((age - 70.0) / 10.0, sex == "M", np.log1p(prior),
                                np.log1p(strict), gaps, prior == 0)).astype(np.float64)
    return features, ["age_per_10_years", "male", "log1p_prior_coded",
                      "log1p_prior_strict", "log1p_days_since_previous_discharge", "first_coded_admission"]


def _paired_brier(subjects, times, events, candidate, baseline, *, seed=2026, draws=300):
    observed = ((events[:, None] == 1) & (times[:, None] <= GRID[None, :])).astype(float)
    sq_candidate = (candidate[:, 7:] - observed[:, 7:]) ** 2
    sq_baseline = (baseline[:, 7:] - observed[:, 7:]) ** 2
    per_stay = np.trapezoid(sq_candidate - sq_baseline, GRID[7:], axis=1) / (365 - 7)
    patients, inverse = np.unique(subjects, return_inverse=True)
    group_sum = np.bincount(inverse, weights=per_stay)
    group_n = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    choice = rng.integers(len(patients), size=(draws, len(patients)))
    draws_value = group_sum[choice].sum(axis=1) / group_n[choice].sum(axis=1)
    return {"point": float(per_stay.mean()),
            "patient_cluster_95pct_interval": np.quantile(draws_value, [0.025, 0.975]).tolist()}


def _auc_with_sparse_guard(subjects, times, events, candidate, baseline, day):
    observed = ((events == 1) & (times <= day)).astype(np.int8)
    if min(int(observed.sum()), len(observed) - int(observed.sum())) < 20:
        return {"point": None, "patient_cluster_95pct_interval": None,
                "reason": "fewer than 20 cases or controls"}
    try:
        return _paired_auc_interval(subjects, times, events, candidate, baseline, day)
    except ValueError as error:
        if str(error) != "too few valid patient bootstrap draws":
            raise
        return {"point": float(auc_rank(candidate[:, day], observed) -
                               auc_rank(baseline[:, day], observed)),
                "patient_cluster_95pct_interval": None,
                "reason": "too few valid patient-bootstrap draws with at least 20 cases and controls"}


def run(base_csv: Path, labels_csv: Path, full_dir: Path,
        comparators: dict[str, Path], output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(output_path)
    rows = list(_rows(base_csv))
    labels = {int(row["hadm_id"]): row for row in _rows(labels_csv)}
    if len(rows) != 19401 or len(labels) != len(rows):
        raise ValueError("history baseline cohort mismatch")
    hadms = np.array([int(row["hadm_id"]) for row in rows])
    if len(np.unique(hadms)) != len(hadms) or set(hadms) != set(labels):
        raise ValueError("history baseline admission IDs mismatch")
    subjects = np.array([int(row["subject_id"]) for row in rows])
    splits = np.array([patient_split(int(subject)) for subject in subjects])
    times = np.array([float(labels[int(hadm)]["time_days"]) for hadm in hadms])
    events = np.array([int(labels[int(hadm)]["event_type"]) for hadm in hadms])
    for i, hadm in enumerate(hadms):
        if int(labels[int(hadm)]["subject_id"]) != subjects[i]:
            raise ValueError("history baseline label/subject mismatch")
    features, names = history_features(rows)
    fitted = fit_competing_cox(features[splits == "train"], times[splits == "train"],
                               events[splits == "train"], l2=10.0)
    if not fitted.readmission.converged or not fitted.death.converged:
        raise ValueError("history Cox failed to converge")
    report_full = json.loads((full_dir / "report.json").read_text(encoding="utf-8"))
    results = {"status": "internal_incremental_evidence", "endpoint": report_full["primary_event"],
               "label_sha256": report_full["label_sha256"], "features_history_only": names,
               "cox_l2": 10.0, "history_cox_fit": "training patients only",
               "strata": {}, "limitations": [
                   "Previously reused historical test is exploratory, not a new blind test.",
                   "World A treats no observed event as followed to day 365.",
                   "Cox fit treats repeat discharges as observations; uncertainty here clusters patients only at evaluation.",
                   "Ablation differences are associations, not causal feature effects."]}
    all_dirs = {"full": full_dir, **comparators}
    for split in ("validation", "historical_test"):
        predictions = {}
        for name, directory in all_dirs.items():
            report, pred = _load_predictions(directory, split)
            for key in ("label_sha256", "field_manifest_sha256", "raw_source_sha256",
                        "text_vector_width", "event_concept_count", "cohort_admissions",
                        "cohort_patients", "primary_event", "parameter_count"):
                if report[key] != report_full[key]:
                    raise ValueError(f"{name}: mismatched {key}")
            if predictions:
                anchor = predictions["full"]
                for key in ("subject_ids", "hadm_ids", "times", "events"):
                    if not np.array_equal(anchor[key], pred[key]):
                        raise ValueError(f"{name}/{split}: prediction order or labels differ")
            predictions[name] = pred
        anchor = predictions["full"]
        index_by_hadm = {int(hadm): i for i, hadm in enumerate(hadms)}
        indices = np.array([index_by_hadm[int(hadm)] for hadm in anchor["hadm_ids"]])
        expected_split = "test" if split == "historical_test" else split
        if not np.all(splits[indices] == expected_split) or not np.array_equal(subjects[indices], anchor["subject_ids"]):
            raise ValueError("prediction split or patient IDs differ")
        if not np.array_equal(events[indices], anchor["events"]) or not np.allclose(times[indices], anchor["times"], atol=1e-6):
            raise ValueError("prediction outcomes differ from independent labels")
        predictions["history_only_cox"] = {"risk": predict_cumulative_incidence(
            fitted, features[indices], GRID)[:, :, 1]}
        prior = np.array([int(rows[i]["prior_coded_admission_count"]) for i in indices])
        groups = {"first_coded_admission": prior == 0,
                  "one_to_two_prior": (prior >= 1) & (prior <= 2),
                  "three_plus_prior": prior >= 3,
                  "all": np.ones(len(prior), dtype=bool)}
        results["strata"][split] = {}
        for group_name, mask in groups.items():
            y_times, y_events, patient_ids = times[indices][mask], events[indices][mask], subjects[indices][mask]
            full_risk = predictions["full"]["risk"][mask]
            group = {"admissions": int(mask.sum()), "patients": int(np.unique(patient_ids).size),
                     "events": int((y_events == 1).sum()), "models": {}, "full_minus_comparator": {}}
            for name, prediction in predictions.items():
                risk = prediction["risk"][mask]
                group["models"][name] = _public_metrics(metrics(y_times, y_events, risk))
                if name != "full":
                    group["full_minus_comparator"][name] = {
                        "ibs": _paired_brier(patient_ids, y_times, y_events, full_risk, risk),
                        "auc": {str(day): _auc_with_sparse_guard(patient_ids, y_times, y_events,
                                                                 full_risk, risk, day)
                                for day in (30, 90, 365)}}
            results["strata"][split][group_name] = group
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--comparator", nargs=2, action="append", metavar=("NAME", "DIR"), default=[])
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    names = [name for name, _ in args.comparator]
    if len(set(names)) != len(names) or "full" in names or "history_only_cox" in names:
        raise ValueError("duplicate or reserved comparator name")
    result = run(args.base_csv, args.labels_csv, args.full_dir,
                 {name: Path(directory) for name, directory in args.comparator}, args.output_path)
    print(json.dumps({split: {name: {"admissions": group["admissions"],
                                       "history_ibs": group["models"]["history_only_cox"]["integrated_brier_7_365"],
                                       "full_ibs": group["models"]["full"]["integrated_brier_7_365"]}
                              for name, group in groups.items()}
                      for split, groups in result["strata"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
