# 08_TRAINING_V5 · 1

源文件正文或经过字段白名单提取的汇总数据，来源身份见每段元数据。

## training/v5/__init__.py · lines 1–1

<!-- REVIEW_FRAGMENT {"path":"training/v5/__init__.py","part":1,"start_line":1,"end_line":1,"source_sha256":"7aec948e3755d99acd7832668426c170f0f61a6f8cab78f6e660a963dd2d7b36","fragment_sha256":"7aec948e3755d99acd7832668426c170f0f61a6f8cab78f6e660a963dd2d7b36","added_terminal_newline":false} -->
````python
"""V5 逐次出院多模态连续时间模型。"""
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/abnormality_model.py · lines 1–54

<!-- REVIEW_FRAGMENT {"path":"training/v5/abnormality_model.py","part":1,"start_line":1,"end_line":54,"source_sha256":"b333e05ad92b0a5214e4e3f57c8600d714161183de655fc58906dfb31dc1bac9","fragment_sha256":"b333e05ad92b0a5214e4e3f57c8600d714161183de655fc58906dfb31dc1bac9","added_terminal_newline":false} -->
````python
"""住院内仅编码检验异常与复查恢复，不构造原始检验／用药事件编码器。"""

from __future__ import annotations

import torch
from torch import nn

from training.v5.event_data import EventBatch
from training.v5.event_model import TemporalFusionModel


class AbnormalityFusionModel(TemporalFusionModel):
    """保留静态、文本、跨住院历史与竞争风险头；只替换住院内事件分支。"""

    def __init__(self, *, concept_count: int, type_count: int, field_count: int,
                 text_width: int, abnormality_width: int, width: int = 128) -> None:
        if abnormality_width < 6 or abnormality_width % 6:
            raise ValueError("abnormality input must have six values per axis")
        super().__init__(concept_count=concept_count, type_count=type_count,
                         field_count=field_count, text_width=text_width, width=width)
        # 父类仅用于复用住院融合和跨住院历史；这些原始事件参数不能进入新模型。
        del self.event_embedding
        del self.event_value_projection
        del self.event_gru
        del self.event_score
        del self.event_projection
        self.abnormality_width = abnormality_width
        axis_count = abnormality_width // 6
        self.axis_embedding = nn.Embedding(axis_count, 24)
        self.axis_values = nn.Sequential(nn.Linear(6, 24), nn.GELU(), nn.LayerNorm(24))
        self.axis_score = nn.Linear(24, 1)
        self.abnormality_projection = nn.Sequential(nn.Linear(24, width), nn.GELU(),
                                                    nn.LayerNorm(width))

    def _events(self, batch: EventBatch) -> tuple[torch.Tensor, torch.Tensor]:
        values = batch.recovery_by_admission
        if values is None or values.shape != (*batch.event_counts.shape, self.abnormality_width):
            raise ValueError("abnormality feature shape mismatch")
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("invalid abnormality input")
        features = values.reshape(-1, self.abnormality_width // 6, 6)
        observed = features[:, :, 0] > 0
        present = observed.any(dim=1)
        output = values.new_zeros((features.shape[0], self.gap_projection.out_features))
        if not bool(present.any()):
            return output, present
        axis_ids = torch.arange(features.shape[1], device=features.device)
        embedded = self.axis_values(features) + self.axis_embedding(axis_ids)[None, :, :]
        scores = self.axis_score(embedded).squeeze(-1).masked_fill(~observed, -1e4)
        weights = torch.softmax(scores, dim=1) * observed
        pooled = (embedded * weights.unsqueeze(-1)).sum(dim=1)
        projected = self.abnormality_projection(pooled)
        output[present] = projected[present]
        return output, present
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/aggregate_group_ablations.py · lines 1–88

<!-- REVIEW_FRAGMENT {"path":"training/v5/aggregate_group_ablations.py","part":1,"start_line":1,"end_line":88,"source_sha256":"5d8382fe40bf04b8d2e4b32e95baba02a274cc3b6b57dd9d740cbaf72992eef1","fragment_sha256":"5d8382fe40bf04b8d2e4b32e95baba02a274cc3b6b57dd9d740cbaf72992eef1","added_terminal_newline":false} -->
````python
"""汇总配对消融的跨种子方向，不自动裁决变量留删。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def run(comparison: Path, *, expected_seeds: int = 3) -> Path:
    payload = json.loads((comparison / "paired_group_metrics.json").read_text(encoding="utf-8"))
    seeds = payload["seeds"]
    if len(seeds) != expected_seeds:
        raise ValueError(f"expected {expected_seeds} seeds, got {len(seeds)}")
    arm_sets = [set(arms) for arms in seeds.values()]
    if any(arms != arm_sets[0] for arms in arm_sets[1:]):
        raise ValueError("some ablation arms are missing across seeds")
    output = comparison / "paired_group_overview.csv"
    if output.exists():
        raise FileExistsError(output)
    records = []
    for arm in sorted(arm_sets[0]):
        results = [seeds[seed][arm]["validation"] for seed in sorted(seeds)]
        readmission = [result["readmission"] for result in results]
        death = [result["death"] for result in results]
        def values(cause: list[dict[str, object]], metric: str) -> list[float]:
            return [float(item["delta_removed_minus_full"][metric]) for item in cause]

        read_brier = values(readmission, "integrated_brier_7_365")
        read_auc = values(readmission, "mean_auc_7_365")
        read_c90 = values(readmission, "c_index_90")
        read_c365 = values(readmission, "c_index_365")
        read_curve_gap = values(readmission, "mean_absolute_population_curve_gap_7_365")
        read_cal90 = values(readmission, "absolute_mean_calibration_gap_90")
        death_brier = values(death, "integrated_brier_7_365")
        death_auc = values(death, "mean_auc_7_365")
        death_c90 = values(death, "c_index_90")
        death_c365 = values(death, "c_index_365")
        death_curve_gap = values(death, "mean_absolute_population_curve_gap_7_365")
        death_cal90 = values(death, "absolute_mean_calibration_gap_90")
        intervals = [item["brier_interval"]["patient_cluster_200_draw_95_interval"]
                     for item in readmission]
        records.append({
            "arm": arm, "seed_count": len(results),
            "readmission_delta_integrated_brier_mean": mean(read_brier),
            "readmission_delta_integrated_brier_min": min(read_brier),
            "readmission_delta_integrated_brier_max": max(read_brier),
            "seeds_removal_worsens_readmission_brier": sum(value > 0 for value in read_brier),
            "seeds_brier_interval_entirely_above_zero": sum(float(low) > 0 for low, _ in intervals),
            "seeds_brier_interval_entirely_below_zero": sum(float(high) < 0 for _, high in intervals),
            "readmission_delta_mean_auc_mean": mean(read_auc),
            "seeds_removal_worsens_readmission_auc": sum(value < 0 for value in read_auc),
            "readmission_delta_c_index_90_mean": mean(read_c90),
            "seeds_removal_worsens_readmission_c_index_90": sum(value < 0 for value in read_c90),
            "readmission_delta_c_index_365_mean": mean(read_c365),
            "seeds_removal_worsens_readmission_c_index_365": sum(value < 0 for value in read_c365),
            "readmission_delta_population_curve_gap_mean": mean(read_curve_gap),
            "seeds_removal_worsens_population_curve_gap": sum(value > 0 for value in read_curve_gap),
            "readmission_delta_calibration_gap_90_mean": mean(read_cal90),
            "seeds_removal_worsens_readmission_calibration_90": sum(value > 0 for value in read_cal90),
            "death_delta_integrated_brier_mean": mean(death_brier),
            "seeds_removal_worsens_death_brier": sum(value > 0 for value in death_brier),
            "death_delta_mean_auc_mean": mean(death_auc),
            "seeds_removal_worsens_death_auc": sum(value < 0 for value in death_auc),
            "death_delta_c_index_90_mean": mean(death_c90),
            "seeds_removal_worsens_death_c_index_90": sum(value < 0 for value in death_c90),
            "death_delta_c_index_365_mean": mean(death_c365),
            "seeds_removal_worsens_death_c_index_365": sum(value < 0 for value in death_c365),
            "death_delta_population_curve_gap_mean": mean(death_curve_gap),
            "seeds_removal_worsens_death_curve_gap": sum(value > 0 for value in death_curve_gap),
            "death_delta_calibration_gap_90_mean": mean(death_cal90),
            "seeds_removal_worsens_death_calibration_90": sum(value > 0 for value in death_cal90),
        })
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=3)
    args = parser.parse_args()
    print(run(args.comparison, expected_seeds=args.expected_seeds))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/analyze_incremental_evidence.py · lines 1–199

<!-- REVIEW_FRAGMENT {"path":"training/v5/analyze_incremental_evidence.py","part":1,"start_line":1,"end_line":199,"source_sha256":"0c98dd013f5934bf9a59977f925f3f6f98da4a655ac8c1f9f5c16ee8f1f4960d","fragment_sha256":"0c98dd013f5934bf9a59977f925f3f6f98da4a655ac8c1f9f5c16ee8f1f4960d","added_terminal_newline":false} -->
````python
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
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/audit_embedded_labels.py · lines 1–35

<!-- REVIEW_FRAGMENT {"path":"training/v5/audit_embedded_labels.py","part":1,"start_line":1,"end_line":35,"source_sha256":"b4340790933732b2e78e3c1140718b7edba44e71ebe90d653ad6e086071fba39","fragment_sha256":"b4340790933732b2e78e3c1140718b7edba44e71ebe90d653ad6e086071fba39","added_terminal_newline":false} -->
````python
"""核对旧基础表内嵌标签与现行标签的差别，不输出患者级记录。"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


def audit(base_csv: Path, current_labels: Path, run_dir: Path) -> dict[str, object]:
    with base_csv.open(encoding="utf-8-sig", newline="") as handle:
        base = list(csv.DictReader(handle))
    with gzip.open(current_labels, "rt", encoding="utf-8", newline="") as handle:
        labels = {r["hadm_id"]: r for r in csv.DictReader(handle)}
    if set(labels) != {r["hadm_id"] for r in base}:
        raise ValueError("旧基础表与现行标签住院成员不同")
    model_audit = json.loads((run_dir / "model_data_audit.json").read_text(encoding="utf-8"))
    result = {"cohort_admissions": len(base),
              "different_embedded_event_type": sum(int(r["event_type"]) != int(labels[r["hadm_id"]]["event_type"]) for r in base),
              "different_embedded_event_time": sum(abs(float(r["time_days"]) - float(labels[r["hadm_id"]]["time_days"])) > 1e-6 for r in base),
              "embedded_outcomes_used_for_training": False,
              "training_labels_match_current_raw_ledger": model_audit["current_labels_identical_to_raw_ledger"],
              "policy": "explicit current labels override obsolete embedded labels; raw-ledger parity must pass before training"}
    (run_dir / "embedded_label_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--current-labels", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.base_csv, args.current_labels, args.run_dir)))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/audit_medication_effect_features.py · lines 1–61

<!-- REVIEW_FRAGMENT {"path":"training/v5/audit_medication_effect_features.py","part":1,"start_line":1,"end_line":61,"source_sha256":"dffc47af044d54f8d618b90971e8d2588017828118b722ddee60436ee4adfb7c","fragment_sha256":"dffc47af044d54f8d618b90971e8d2588017828118b722ddee60436ee4adfb7c","added_terminal_newline":false} -->
````python
"""仅输出药理作用组住院级覆盖与暴露摘要，不输出患者级资料。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from training.v5.event_data import EventData
from training.v5.medication_effect_data import EFFECT_GROUPS, VALUES_PER_GROUP, attach_medication_effects


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(processed: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    with np.load(processed / "event_arrays.npz", allow_pickle=False) as archive:
        hadms = archive["hadm_ids"]
    dummy = EventData(SimpleNamespace(hadm_ids=hadms), np.empty((0, 4)),
                      np.empty(0), np.empty(0), (),
                      {"processed_manifest": _sha(processed / "manifest.json")})
    features = attach_medication_effects(dummy, processed, mode="active").medication_by_admission
    rows = features.reshape(len(hadms), len(EFFECT_GROUPS), VALUES_PER_GROUP)
    groups = {}
    for index, name in enumerate(EFFECT_GROUPS):
        exposed = rows[:, index, 0] > 0
        groups[name] = {
            "admissions_with_accepted_administration": int(exposed.sum()),
            "fraction_of_all_admissions": float(exposed.mean()),
            "median_unique_administrations_among_exposed": (
                float(np.median(np.expm1(rows[exposed, index, 1]))) if exposed.any() else None),
            "median_hospital_day_bins_among_exposed": (
                float(np.median(np.expm1(rows[exposed, index, 2]))) if exposed.any() else None),
        }
    result = {"status": "audited_aggregate_only", "admissions": len(hadms),
              "processed_manifest_sha256": _sha(processed / "manifest.json"),
              "any_target_medication_admissions": int((rows[:, :, 0] > 0).any(axis=1).sum()),
              "groups": groups,
              "limitation": "No accepted target class is not proof that no drug was administered."}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-event-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.processed_event_dir, args.output)
    print(json.dumps({"status": result["status"], "admissions": result["admissions"]}))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/audit_outcome_auc_shift.py · lines 1–99

<!-- REVIEW_FRAGMENT {"path":"training/v5/audit_outcome_auc_shift.py","part":1,"start_line":1,"end_line":99,"source_sha256":"de4c0f3015d6e20ac379b1237139f2360ad54f404146a75ca4bb18544e33f325","fragment_sha256":"de4c0f3015d6e20ac379b1237139f2360ad54f404146a75ca4bb18544e33f325","added_terminal_newline":false} -->
````python
"""固定已训练模型和预测值，只替换结局，量化 AUC 的标签口径效应。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_data import load_data, normalize_on_training_patients, split_sequences
from training.v5.independent_fusion import IndependentFusionModel
from training.v5.train import metrics
from training.v5.train_independent import predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned_dir", "structured_dir", "base_csv", "field_dir", "vectors",
                 "broad_labels", "qualified_labels", "broad_model_run", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.plot is not None and args.plot.exists()):
        raise FileExistsError("audit output already exists")
    data = {}
    for name, label in (("strict", None), ("broad", args.broad_labels),
                        ("strict_or_top3", args.qualified_labels)):
        loaded = load_data(args.aligned_dir, args.structured_dir, args.base_csv,
                           args.field_dir, args.vectors, labels_csv=label)
        data[name] = normalize_on_training_patients(loaded)[0]
    anchor = data["broad"]
    for name, other in data.items():
        if (not np.array_equal(anchor.hadm_ids, other.hadm_ids)
                or not np.array_equal(anchor.subject_ids, other.subject_ids)
                or not np.array_equal(anchor.splits, other.splits)
                or anchor.manifest_sha256 != other.manifest_sha256):
            raise ValueError(f"{name} changed cohort, split or text manifest")
    if not torch.cuda.is_available():
        raise RuntimeError("fixed-model outcome audit requires CUDA")
    device = torch.device("cuda")
    model = IndependentFusionModel(
        concept_count=len(anchor.concept_names), type_count=len(anchor.type_names),
        field_count=len(FIELD_LABELS), text_width=anchor.text_width,
    ).to(device)
    model.load_state_dict(torch.load(args.broad_model_run / "structured_field_bert.pt",
                                     map_location=device, weights_only=True))
    report: dict[str, object] = {
        "fixed_model": "previously trained broad-outcome summary+BERT checkpoint",
        "cohort_admissions": len(anchor.hadm_ids),
        "field_manifest_sha256": anchor.manifest_sha256,
        "label_sha256": {name: item.label_sha256 for name, item in data.items()},
    }
    figures: dict[str, dict[str, object]] = {}
    for split in ("validation", "test"):
        sequences = split_sequences(anchor, split)
        indices = np.concatenate(sequences)
        # 三种结局共用同一份风险预测；只改变评价时的病例／对照定义。
        risk, _ = predict(model, anchor, sequences, include_text=True,
                          batch_size=24, device=device)
        details = {name: metrics(item.times[indices], item.events[indices], risk)
                   for name, item in data.items()}
        figures[split] = details
        report["historical_test" if split == "test" else split] = {
            name: {key: detail[key] for key in
                   ("admissions", "readmissions", "deaths", "integrated_brier_7_365",
                    "mean_auc_7_365")}
            for name, detail in details.items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.plot is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col")
        for column, split in enumerate(("validation", "test")):
            for name, detail in figures[split].items():
                days = np.arange(7, 366)
                axes[0, column].plot(days, detail["auc_curve"][7:], label=name)
                axes[1, column].plot(days, detail["case_curve"][7:], label=name)
            axes[0, column].set(title=split, ylabel="Cumulative AUC(t)", ylim=(0.5, 0.85))
            axes[1, column].set(xlabel="Days after discharge", ylabel="Cumulative cases")
            for axis in axes[:, column]:
                axis.grid(alpha=0.25)
        axes[0, 0].legend()
        figure.suptitle("Same model and predictions; only outcome labels differ")
        figure.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot, dpi=170)
        plt.close(figure)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/audit_revised_group_coverage.py · lines 1–122

<!-- REVIEW_FRAGMENT {"path":"training/v5/audit_revised_group_coverage.py","part":1,"start_line":1,"end_line":122,"source_sha256":"31120493048689f799f195cda278d2a6c16451adb338e826034c9b613d7943d5","fragment_sha256":"31120493048689f799f195cda278d2a6c16451adb338e826034c9b613d7943d5","added_terminal_newline":false} -->
````python
"""Audit admission/patient coverage of the fixed reviewed medication and laboratory groups."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from patient_data.expanded_abnormality_features import AXES, FEATURE_NAMES
from training.v5.lab_recovery_data import ABNORMALITY_GROUPS
from training.v5.medication_effect_data import EFFECT_GROUPS


def _rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def audit(processed: Path, abnormality: Path) -> dict:
    subject_by_hadm = {}
    for row in _rows(processed / "static_features.csv.gz"):
        hadm, subject = row["hadm_id"], row["subject_id"]
        previous = subject_by_hadm.setdefault(hadm, subject)
        if previous != subject:
            raise ValueError("one admission has multiple subjects")
    class_to_groups = defaultdict(list)
    for group, members in EFFECT_GROUPS.items():
        for member in members:
            class_to_groups[member].append(group)
    med_hadms = defaultdict(set)
    med_patients = defaultdict(set)
    med_administrations = defaultdict(set)
    for row in _rows(processed / "events.csv.gz"):
        if row["kind"] != "medication" or row["concept"] not in class_to_groups:
            continue
        hadm = row["hadm_id"]
        subject = subject_by_hadm[hadm]
        for group in class_to_groups[row["concept"]]:
            med_hadms[group].add(hadm)
            med_patients[group].add(subject)
            med_administrations[group].add((hadm, row["source_id"]))
    medication_overlap = []
    groups = list(EFFECT_GROUPS)
    n_admissions = len(subject_by_hadm)
    for left_index, left in enumerate(groups):
        for right in groups[left_index + 1:]:
            a, b = med_hadms[left], med_hadms[right]
            both = len(a & b)
            denom = len(a | b)
            p_a, p_b, p_both = len(a) / n_admissions, len(b) / n_admissions, both / n_admissions
            phi_denom = (p_a * (1 - p_a) * p_b * (1 - p_b)) ** .5
            medication_overlap.append({"left": left, "right": right, "coexposed_admissions": both,
                                       "jaccard": both / denom if denom else 0.,
                                       "binary_phi": (p_both - p_a * p_b) / phi_denom if phi_denom else 0.})
    with np.load(abnormality / "abnormality.npz", allow_pickle=False) as archive:
        hadms = archive["hadm_ids"]
        features = archive["features"]
        names = archive["feature_names"]
    if tuple(names.tolist()) != FEATURE_NAMES or features.shape != (len(hadms), len(FEATURE_NAMES)):
        raise ValueError("abnormality feature schema mismatch")
    if set(map(str, hadms)) != set(subject_by_hadm):
        raise ValueError("abnormality and medication admission sets differ")
    shaped = features.reshape(len(hadms), len(AXES), 6)
    per_axis = {}
    for index, axis in enumerate(AXES):
        observed = shaped[:, index, 0] > 0
        abnormal = shaped[:, index, 2] > 0
        recovered = shaped[:, index, 4] > 0
        per_axis[axis] = {
            "observed_admissions": int(observed.sum()),
            "observed_patients": len({subject_by_hadm[str(hadm)] for hadm in hadms[observed]}),
            "abnormal_admissions": int(abnormal.sum()),
            "abnormal_patients": len({subject_by_hadm[str(hadm)] for hadm in hadms[abnormal]}),
            "post_peak_recovery_admissions": int(recovered.sum()),
        }
    lab = {}
    for group, members in ABNORMALITY_GROUPS.items():
        indices = [AXES.index(member) for member in members]
        observed = (shaped[:, indices, 0] > 0).any(axis=1)
        abnormal = (shaped[:, indices, 2] > 0).any(axis=1)
        recovered = (shaped[:, indices, 4] > 0).any(axis=1)
        lab[group] = {
            "axes": list(members),
            "observed_admissions": int(observed.sum()),
            "observed_patients": len({subject_by_hadm[str(hadm)] for hadm in hadms[observed]}),
            "abnormal_admissions": int(abnormal.sum()),
            "abnormal_patients": len({subject_by_hadm[str(hadm)] for hadm in hadms[abnormal]}),
            "post_peak_recovery_admissions": int(recovered.sum()),
        }
    return {"cohort_admissions": len(subject_by_hadm),
            "cohort_patients": len(set(subject_by_hadm.values())),
            "medication": {
                group: {"concepts": list(EFFECT_GROUPS[group]),
                        "exposed_admissions": len(med_hadms[group]),
                        "exposed_patients": len(med_patients[group]),
                        "unique_administrations": len(med_administrations[group])}
                for group in EFFECT_GROUPS},
            "medication_pairwise_overlap": medication_overlap,
            "laboratory": lab, "laboratory_by_axis": per_axis}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--abnormality-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.processed_dir, args.abnormality_dir)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"admissions": result["cohort_admissions"],
                      "patients": result["cohort_patients"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/audit_section_incremental.py · lines 1–70

<!-- REVIEW_FRAGMENT {"path":"training/v5/audit_section_incremental.py","part":1,"start_line":1,"end_line":70,"source_sha256":"23ce804d627a01502815e0534d46003a87d897c93e351777a49e4dc762671caf","fragment_sha256":"23ce804d627a01502815e0534d46003a87d897c93e351777a49e4dc762671caf","added_terminal_newline":false} -->
````python
"""聚合新增小结章节的正文覆盖与模板空栏；不导出患者级文本。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from training.v5.extend_field_manifest import EXTRA_LABELS


def audit(path: Path) -> dict[str, object]:
    roles = (*EXTRA_LABELS, "followup_instructions")
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    admissions: dict[str, set[str]] = defaultdict(set)
    lengths: dict[str, list[int]] = defaultdict(list)
    hashes: dict[str, Counter[str]] = defaultdict(Counter)
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            role = row["section_role"]
            if role not in roles:
                continue
            counts[role]["sections"] += 1
            counts[role][row["content_status"]] += 1
            if row["content_status"] == "informative":
                admissions[role].add(row["hadm_id"])
                content = row["text"].strip()
                lengths[role].append(len(content))
                hashes[role][hashlib.sha256(content.encode("utf-8")).hexdigest()] += 1
    return {
        "status": "aggregate_section_content_audit",
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "roles": {role: {
            "sections": counts[role]["sections"],
            "informative_sections": counts[role]["informative"],
            "placeholder_sections": counts[role]["placeholder"],
            "admissions_with_informative_section": len(admissions[role]),
            "median_informative_chars": float(np.median(lengths[role])) if lengths[role] else None,
            "p90_informative_chars": float(np.quantile(lengths[role], .9)) if lengths[role] else None,
            "unique_informative_texts": len(hashes[role]),
            "max_identical_text_repetitions": max(hashes[role].values(), default=0),
        } for role in roles},
        "limitations": [
            "Informative means only that text contains a letter or digit; it is not clinical adjudication.",
            "Distinct section text is not proof of new information beyond structured data.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.sections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["roles"], ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/broad_fragment_data.py · lines 1–118

<!-- REVIEW_FRAGMENT {"path":"training/v5/broad_fragment_data.py","part":1,"start_line":1,"end_line":118,"source_sha256":"6b84f36f0dd74a360153faaae70dfb5ea225b60be602ac6668d9d0fbb0a2a383","fragment_sha256":"6b84f36f0dd74a360153faaae70dfb5ea225b60be602ac6668d9d0fbb0a2a383","added_terminal_newline":false} -->
````python
"""新片段只替换文本引用；沿用核验后的结构化输入与完整患者历史。"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from training.v5.event_data import EventData

EXCERPT_ROLE = 'clinical_excerpt'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_selection(data: EventData, selection_file: Path) -> dict:
    selection = json.loads(selection_file.read_text(encoding='utf-8'))
    if sha(selection_file.parent/selection['members_file']) != selection['member_sha256']:
        raise ValueError('Frozen batch membership changed')
    base = data.base
    by_hadm = {int(h): i for i, h in enumerate(base.hadm_ids)}
    ids = [int(r['hadm_id']) for r in selection['members']]
    if len(ids) != len(set(ids)) or len(ids) != selection['selected_notes']:
        raise ValueError('Duplicate or inconsistent batch members')
    for member in selection['members']:
        i = by_hadm.get(int(member['hadm_id']))
        if (i is None or int(base.subject_ids[i]) != int(member['subject_id']) or
                str(base.splits[i]) != member['patient_split']):
            raise ValueError('Batch patient/admission/split differs from validated model data')
    return selection


def preserve_selected_histories(data: EventData, selection: dict) -> EventData:
    """仅筛选训练/评价患者序列，保留选中患者没有小结的住院及原索引。"""
    subjects = {int(r['subject_id']) for r in selection['members']}
    base = data.base
    sequences = tuple(seq for seq in base.sequences if int(base.subject_ids[seq[0]]) in subjects)
    seen = {int(base.subject_ids[seq[0]]) for seq in sequences}
    if seen != subjects or any(len(set(base.subject_ids[seq].tolist())) != 1 or
                               len(set(base.splits[seq].tolist())) != 1 for seq in sequences):
        raise ValueError('Selected patient history missing or crosses patients/splits')
    expected = set(np.flatnonzero(np.isin(base.subject_ids, list(subjects))).tolist())
    if set(np.concatenate(sequences).tolist()) != expected:
        raise ValueError('Selected full discharge histories were truncated')
    return replace(data, base=replace(base, sequences=sequences))


def common_roles(data: EventData) -> tuple[str, ...]:
    roles = data.base.field_roles
    return roles if EXCERPT_ROLE in roles else (*roles, EXCERPT_ROLE)


def old_or_no_text(data: EventData, selection: dict, *, enabled: bool) -> EventData:
    hadms = {int(r['hadm_id']) for r in selection['members']}
    base = data.base
    refs = tuple(refs if enabled and int(hadm) in hadms else ()
                 for hadm, refs in zip(base.hadm_ids, base.text_by_admission, strict=True))
    return replace(data, base=replace(base, text_by_admission=refs, field_roles=common_roles(data)))


def attach_broad_text(data: EventData, selection: dict, directory: Path, encoder: str) -> EventData:
    if encoder not in ('bert', 'qwen'):
        raise ValueError('Unsupported encoder')
    path = directory/'shared_text.csv.gz'
    manifest_sha = sha(path)
    summary = json.loads((directory/'manifest_summary.json').read_text(encoding='utf-8'))
    validation = json.loads((directory/'vector_validation.json').read_text(encoding='utf-8'))
    state = json.loads((directory/encoder/'state.json').read_text(encoding='utf-8'))
    width = 768 if encoder == 'bert' else 2560
    filename = 'bioclinicalbert.float16.npy' if encoder == 'bert' else 'qwen3_embedding.float16.npy'
    if (summary['status'] != 'manifest_ready' or not summary['provenance_validated'] or
            summary['manifest_sha256'] != manifest_sha or validation['manifest_sha256'] != manifest_sha or
            validation['encoders'][encoder]['status'] != 'validated' or
            state['status'] != 'complete' or state['manifest_sha256'] != manifest_sha or
            state['completed_rows'] != summary['rows']):
        raise ValueError('Incomplete or changed shared manifest/encoding')
    if ((encoder == 'bert' and state.get('model') != 'Bio_ClinicalBERT') or
            (encoder == 'qwen' and state.get('encoder') != 'Qwen3-Embedding-4B')):
        raise ValueError('Encoder identity mismatch')
    vectors = np.load(directory/encoder/filename, mmap_mode='r')
    if vectors.shape != (summary['rows'], width) or vectors.dtype != np.float16:
        raise ValueError('Wrong vector width/rows/dtype')
    members = {int(r['hadm_id']): r for r in selection['members']}
    base = data.base
    by_hadm = {int(h): i for i, h in enumerate(base.hadm_ids)}
    roles = common_roles(data)
    role = roles.index(EXCERPT_ROLE)
    refs: list[list[tuple[int, int]]] = [[] for _ in base.hadm_ids]
    csv.field_size_limit(2**31-1)
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(vectors):
        raise ValueError('Vector row count mismatch')
    for row_number, row in enumerate(rows):
        hadm = int(row['hadm_id'])
        member = members.get(hadm)
        if member is None or hadm not in by_hadm:
            raise ValueError('Text outside selected batch')
        if (int(row['embedding_row']) != row_number or int(row['subject_id']) != int(member['subject_id']) or
                row['note_id'] != member['note_id'] or row['patient_split'] != member['patient_split'] or
                row['source_sha256'] != member['source_sha256'] or
                hashlib.sha256(row['chunk_text'].encode()).hexdigest() != row['text_sha256']):
            raise ValueError('Text identity/order/content mismatch')
        # 主题为检索提示，不复制一个向量给多个主题，不将其当成已证实语义标签。
        refs[by_hadm[hadm]].append((role, row_number))
    for start in range(0, len(vectors), 2048):
        values = vectors[start:start+2048].astype(np.float32)
        norms = np.linalg.norm(values, axis=1)
        if not np.isfinite(values).all() or np.any((norms < .98) | (norms > 1.02)):
            raise ValueError('Invalid or unnormalized frozen vectors')
    return replace(data, base=replace(base, text_by_admission=tuple(tuple(r) for r in refs),
                    text_vectors=vectors, text_width=width, manifest_sha256=manifest_sha, field_roles=roles))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/clinical_concepts.py · lines 1–148

<!-- REVIEW_FRAGMENT {"path":"training/v5/clinical_concepts.py","part":1,"start_line":1,"end_line":148,"source_sha256":"c05e2ec100ac2c12ff23410d878fa57cc77cc5aa390f7eac95d0434322dab80d","fragment_sha256":"c05e2ec100ac2c12ff23410d878fa57cc77cc5aa390f7eac95d0434322dab80d","added_terminal_newline":false} -->
````python
"""V5 共用的轻量临床概念映射；不依赖训练框架。"""

from __future__ import annotations

import re


LABS: dict[int, tuple[str, int]] = {
    50820: ("ph", 3), 50818: ("paco2", 3), 50821: ("pao2", 3),
    50882: ("bicarbonate", 4), 51301: ("wbc", 4),
    51222: ("hemoglobin", 4), 50912: ("creatinine", 4), 50862: ("albumin", 4),
}
MED_NAMES = ("saba", "sama", "laba", "lama", "inhaled_steroid", "systemic_steroid",
             "macrolide", "beta_lactam", "fluoroquinolone", "other_antibiotic",
             "loop_diuretic", "other_diuretic")

_MED_PATTERNS = {
    "saba": r"\b(?:albuterol|salbutamol|levalbuterol)\b",
    "sama": r"\bipratropium\b",
    "laba": r"\b(?:salmeterol|formoterol|arformoterol|indacaterol|olodaterol|vilanterol)\b",
    "lama": r"\b(?:tiotropium|umeclidinium|glycopyrrolate|aclidinium|revefenacin)\b",
    "inhaled_steroid": r"\b(?:budesonide|fluticasone|beclomethasone|mometasone|ciclesonide)\b",
    "systemic_steroid": r"\b(?:prednisone|prednisolone|methylprednisolone|dexamethasone|hydrocortisone)\b",
    "macrolide": r"\b(?:azithromycin|clarithromycin|erythromycin)\b",
    "beta_lactam": r"\b(?:cef\w+|cephalexin|amoxicillin|ampicillin|piperacillin|tazobactam|meropenem|ertapenem|imipenem|aztreonam|penicillin)\b",
    "fluoroquinolone": r"\b(?:levofloxacin|ciprofloxacin|moxifloxacin|ofloxacin)\b",
    "other_antibiotic": r"\b(?:vancomycin|doxycycline|tetracycline|linezolid|clindamycin|metronidazole|trimethoprim|sulfamethoxazole|gentamicin|tobramycin)\b",
    "loop_diuretic": r"\b(?:furosemide|torsemide|bumetanide)\b",
    "other_diuretic": r"\b(?:spironolactone|hydrochlorothiazide|chlorothiazide|metolazone|amiloride|eplerenone)\b",
}
_COMPILED_MED = {name: re.compile(pattern, re.I) for name, pattern in _MED_PATTERNS.items()}
_BRANDS = {
    "symbicort": {"laba", "inhaled_steroid"},
    "advair": {"laba", "inhaled_steroid"},
    "breo ellipta": {"laba", "inhaled_steroid"},
    "trelegy ellipta": {"laba", "lama", "inhaled_steroid"},
    "spiriva": {"lama"}, "incruse ellipta": {"lama"},
    "anoro ellipta": {"laba", "lama"}, "stiolto respimat": {"laba", "lama"},
    "duoneb": {"saba", "sama"}, "combivent": {"saba", "sama"},
    "pulmicort": {"inhaled_steroid"}, "ventolin": {"saba"},
    "proair": {"saba"}, "xopenex": {"saba"}, "atrovent": {"sama"},
}
_LOCAL_FORM = re.compile(r"ophth|\botic\b|nasal|topical|\bcream\b|\boint\b|\bointment\b|\blotion\b|"
                         r"\beye\s*(?:drop|gel)|\bear\s*drop|\bvaginal\b|\brectal\b|"
                         r"\benema\b|\bsuppositor(?:y|ies)\b|\bnepafenac\b|"
                         r"\bclindamycin\s*1%\s*solution\b", re.I)
_INHALED_CLASSES = {"saba", "sama", "laba", "lama", "inhaled_steroid"}
_ORAL_ROUTES = {"PO", "NG", "GT", "PEG", "OGT"}
_TETRACYCLINE = re.compile(r"\b(?:doxycycline|tetracycline|minocycline)\b", re.I)
_OTHER_ANTIBIOTIC_NON_TETRACYCLINE = re.compile(
    r"\b(?:vancomycin|linezolid|clindamycin|metronidazole|trimethoprim|sulfamethoxazole|gentamicin|tobramycin)\b", re.I)


def medication_classes(name: str, *, product_descriptions: tuple[str, ...] = (),
                       routes: tuple[str, ...] = (), mapping_version: str = "legacy") -> set[str]:
    """同一复方可贡献多个类别；给药详情只用于剂型判定，不作为模型特征。"""

    lower = name.lower()
    if mapping_version not in ("legacy", "gold_review_v2", "clinical_reviewed_2026_09"):
        raise ValueError("unknown medication mapping version")
    result = {key for key, pattern in _COMPILED_MED.items() if pattern.search(lower)}
    if mapping_version != "legacy" and _TETRACYCLINE.search(lower):
        result.add("tetracycline")
        if not _OTHER_ANTIBIOTIC_NON_TETRACYCLINE.search(lower):
            result.discard("other_antibiotic")
    if mapping_version == "clinical_reviewed_2026_09":
        # 这些记录不是可解释为本次系统治疗的完整给药。
        if re.search(r"\block\b|graded challenge|study drug|w/ lidocaine", lower):
            return set()
        if "umeclidin" in lower:
            result.add("lama")
        if "vilanter" in lower:
            result.add("laba")
    for brand, classes in _BRANDS.items():
        if brand in lower:
            result.update(classes)
    descriptions = " ".join(product_descriptions).lower()
    if _LOCAL_FORM.search(lower) or _LOCAL_FORM.search(descriptions):
        return set()
    if mapping_version == "clinical_reviewed_2026_09":
        if (lower.strip() in {"mometasone", "fluticasone"}
                and not re.search(r"inhal|aerosol|diskus|ellipta|asmanex|flovent", descriptions)):
            result.discard("inhaled_steroid")
        if "prednisolone acetate" in lower and not re.search(r"\boral\b|\bpo\b", descriptions):
            result.discard("systemic_steroid")
    # 单独的 budesonide 在本底账同时有雾化液和口服肠溶胶囊；仅凭药名不可判为 ICS。
    if lower.strip() == "budesonide" and not re.search(r"neb|inhal|aerosol|respule", descriptions):
        result.discard("inhaled_steroid")
    # 本底账的裸名 glycopyrrolate 多为片剂或麻醉逆转注射，不能推断为吸入 LAMA。
    if "glycopyrrolate" in lower and not re.search(r"neb|inhal|lonhala|seebri", lower + " " + descriptions):
        result.discard("lama")
    if "albuterol" in lower and "extended release" in lower:
        result.discard("saba")
    if routes and set(routes).issubset(_ORAL_ROUTES):
        result.difference_update(_INHALED_CLASSES)
    return result


def diagnosis_groups(code: str, version: int) -> set[str]:
    """保守 ICD 家族映射；本次出院编码不自动等于既往史。"""

    code = code.strip().upper().replace(".", "")
    result: set[str] = set()
    if version == 10:
        if code.startswith(("I50", "I110", "I130", "I132")):
            result.add("heart_failure")
        if code.startswith(("I20", "I21", "I22", "I23", "I24", "I25", "I48")):
            result.add("cardiac_other")
        if code.startswith("N18"):
            result.add("ckd")
        if code.startswith(("E08", "E09", "E10", "E11", "E13")):
            result.add("diabetes")
        if re.match(r"^C\d\d", code):
            result.add("malignancy")
        if code.startswith(("J961", "J962", "G4733")):
            result.add("chronic_respiratory")
        if code.startswith(("J960", "J962")):
            result.add("acute_resp_failure")
        if re.match(r"^J(?:1[2-8]|69)", code):
            result.add("pneumonia_or_aspiration")
        if code.startswith(("A40", "A41", "R652")):
            result.add("sepsis")
        if code.startswith("N17"):
            result.add("aki")
    elif version == 9:
        if code.startswith("428"):
            result.add("heart_failure")
        if code.startswith(("410", "411", "412", "413", "414", "42731")):
            result.add("cardiac_other")
        if code.startswith("585"):
            result.add("ckd")
        if code.startswith("250"):
            result.add("diabetes")
        if len(code) >= 3 and code[:3].isdigit() and 140 <= int(code[:3]) <= 208:
            result.add("malignancy")
        if code.startswith(("51883", "51884", "32723")):
            result.add("chronic_respiratory")
        if code.startswith(("51881", "51884")):
            result.add("acute_resp_failure")
        if len(code) >= 3 and code[:3].isdigit() and (480 <= int(code[:3]) <= 486 or code[:3] == "507"):
            result.add("pneumonia_or_aspiration")
        if code.startswith(("038", "99591", "99592")):
            result.add("sepsis")
        if code.startswith("584"):
            result.add("aki")
    else:
        raise ValueError(f"unsupported ICD version: {version}")
    return result
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/compare_abnormality_controls.ps1 · lines 1–24

<!-- REVIEW_FRAGMENT {"path":"training/v5/compare_abnormality_controls.ps1","part":1,"start_line":1,"end_line":24,"source_sha256":"609664131ccc5fca6bf288a5b6bc94377848a7b18b811483b6b0caa155f3be09","fragment_sha256":"609664131ccc5fca6bf288a5b6bc94377848a7b18b811483b6b0caa155f3be09","added_terminal_newline":false} -->
````powershell
param([int[]]$Seeds = @(2026, 2027, 2028))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = 'training_runs/v5_abnormality_only_2026-09-29'
foreach ($seed in $Seeds) {
    foreach ($control in @('core9', 'measurement_only')) {
        $output = "$runRoot/comparison_active_minus_${control}_seed${seed}.json"
        if (Test-Path -LiteralPath $output) {
            $existing = Get-Content -Raw -LiteralPath $output | ConvertFrom-Json
            if ($existing.status -ne 'single_seed_pilot' -or $existing.control_mode -ne $control) {
                throw "Unrecognized comparison already exists: $output"
            }
            Write-Output "SKIP active minus $control seed $seed"
            continue
        }
        & docker run --rm -v "${root}:/workspace" -w /workspace -e PYTHONPATH=/workspace `
            aecopd-training:v1 python -m training.v5.compare_lab_recovery_pilot `
            --run-root $runRoot --output $output --seed $seed --bootstrap 200 `
            --control-mode $control
        if ($LASTEXITCODE -ne 0) { throw "Comparison failed: $control seed $seed" }
        Write-Output "DONE active minus $control seed $seed"
    }
}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/compare_all_lab_axes.py · lines 1–177

<!-- REVIEW_FRAGMENT {"path":"training/v5/compare_all_lab_axes.py","part":1,"start_line":1,"end_line":177,"source_sha256":"d4215e3611cb2b9518f6e509939c6bcab370acb01cad3be0e11a62592eea2b73","fragment_sha256":"d4215e3611cb2b9518f6e509939c6bcab370acb01cad3be0e11a62592eea2b73","added_terminal_newline":false} -->
````python
"""完整22轴逐轴配对比较；固定严格结局、二元给药及两个训练种子。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from patient_data.expanded_abnormality_features import AXES
from training.v5.compare_revised_group_ablations import EXTRA_CONTRACT
from training.v5.compare_screened_group_ablations import (
    CONTRACT_KEYS, _paired_auc, _paired_brier, _patient_draws, _predictions, _summary_for_cause,
)
from training.v5.compare_strict_axis_screen import _report
from training.v5.train import GRID

LABELS = (
    "low arterial pH", "high arterial pCO2", "low bicarbonate", "high bicarbonate",
    "high white blood cells", "low albumin", "low hemoglobin", "high creatinine",
    "hypercapnic acidosis", "low white blood cells", "low sodium", "high sodium",
    "low potassium", "high potassium", "high urea nitrogen", "low glucose", "high glucose",
    "low platelets", "low magnesium", "high lactate", "high CRP", "high NT-proBNP",
)


def compare(root: Path, output: Path, *, seeds: tuple[int, ...] = (2026,),
            draws: int = 100) -> list[dict]:
    if output.exists():
        raise FileExistsError(output)
    if len(set(seeds)) != len(seeds) or not seeds or not 20 <= draws <= 500:
        raise ValueError("invalid seeds or bootstrap count")
    rows = []
    details = []
    for seed in seeds:
        full_dir = root / f"full_seed{seed}"
        full_report = _report(full_dir, lab_mode="active", medication_mode="exposure_only")
        full = {split: _predictions(full_dir, split) for split in ("validation", "historical_test")}
        reference_scores = {(split, cause): _summary_for_cause(
            data["times"], data["events"], data["competing_cif"], cause
        ) for split, data in full.items() for cause in (1, 2)}
        bootstrap = _patient_draws(full["validation"]["subject_ids"], draws, seed + 100)
        for axis in AXES:
            directory = root / f"without_axis_{axis}_seed{seed}"
            report = _report(directory, lab_mode=f"without_axis_{axis}", medication_mode="exposure_only")
            for key in (*CONTRACT_KEYS, *EXTRA_CONTRACT, "shared_text_init_width", "deterministic_inference"):
                if report[key] != full_report[key]:
                    raise ValueError(f"unpaired contract: seed={seed}, axis={axis}, key={key}")
            for split, reference in full.items():
                candidate = _predictions(directory, split)
                for key in ("subject_ids", "hadm_ids", "times", "events"):
                    if not np.array_equal(reference[key], candidate[key]):
                        raise ValueError(f"unpaired {key}: {directory}/{split}")
                times, events = reference["times"], reference["events"]
                for cause, name in ((1, "readmission"), (2, "death")):
                    ref = reference_scores[(split, cause)]
                    new = _summary_for_cause(times, events, candidate["competing_cif"], cause)
                    row = dict(seed=seed, axis=axis, split=split, cause=name,
                               reference_brier=ref["integrated_brier_7_365"],
                               candidate_brier=new["integrated_brier_7_365"],
                               delta_brier=new["integrated_brier_7_365"] - ref["integrated_brier_7_365"],
                               reference_auc=ref["mean_auc_7_365"], candidate_auc=new["mean_auc_7_365"],
                               delta_auc=new["mean_auc_7_365"] - ref["mean_auc_7_365"],
                               delta_c_index_90=new["c_index_90"] - ref["c_index_90"],
                               delta_c_index_365=new["c_index_365"] - ref["c_index_365"],
                               delta_population_curve_gap=new["mean_absolute_population_curve_gap_7_365"]
                               - ref["mean_absolute_population_curve_gap_7_365"],
                               delta_calibration_gap_90=new["points"]["90"]["absolute_mean_calibration_gap"]
                               - ref["points"]["90"]["absolute_mean_calibration_gap"])
                    detail = dict(row, reference=ref, candidate=new)
                    if split == "validation":
                        labels = ((events[:, None] == cause) & (times[:, None] <= GRID[None, :])).astype(float)
                        detail["brier_95_interval"] = _paired_brier(
                            reference["subject_ids"], labels, candidate["competing_cif"][:, :, cause],
                            reference["competing_cif"][:, :, cause], bootstrap,
                        )["patient_cluster_200_draw_95_interval"]
                        detail["auc_90_95_interval"] = _paired_auc(
                            labels[:, 90].astype(bool), candidate["competing_cif"][:, 90, cause],
                            reference["competing_cif"][:, 90, cause], bootstrap,
                        )["patient_cluster_200_draw_95_interval"]
                    rows.append(row)
                    details.append(detail)
            print(f"COMPARE DONE seed={seed} axis={axis}", flush=True)
    output.mkdir(parents=True)
    with (output / "paired_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "paired_results.json").write_text(json.dumps({
        "status": "completed", "seeds": seeds, "axes": AXES,
        "bootstrap_patient_cluster_draws": draws,
        "direction": "removed minus full; negative Brier and positive AUC indicate improvement",
        "limitations": ["Exploratory comparisons; no multiplicity adjustment.",
                        "One seed is a quick screen; it cannot establish retraining stability.",
                        "Bootstrap covers patient sampling, not retraining uncertainty.",
                        "Historical test has been repeatedly viewed and is not blind.",
                        "Metrics use the existing 365-day observed-event World-A contract; not complete out-of-hospital follow-up."],
        "rows": details,
    }, indent=2), encoding="utf-8")
    summary = []
    for axis in AXES:
        for split in ("validation", "historical_test"):
            for cause in ("readmission", "death"):
                matched = [row for row in rows if row["axis"] == axis and row["split"] == split
                           and row["cause"] == cause]
                summary.append(dict(axis=axis, split=split, cause=cause,
                                    **{key: float(np.mean([row[key] for row in matched]))
                                       for key in matched[0] if key.startswith("delta_")},
                                    brier_improved_seeds=sum(row["delta_brier"] < 0 for row in matched),
                                    auc_improved_seeds=sum(row["delta_auc"] > 0 for row in matched)))
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    plot(rows, output / "validation_readmission_all22_deltas.png", seeds=seeds)
    return rows


def plot(rows: list[dict], output: Path, *, seeds: tuple[int, ...]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if len(LABELS) != len(AXES):
        raise ValueError("axis labels do not match the fixed vocabulary")
    fig, panels = plt.subplots(1, 2, figsize=(16, 11), sharey=True)
    for panel, key, title in zip(panels, ("delta_brier", "delta_auc"), (
            "Integrated Brier: removed - full (lower better)",
            "Mean AUC(t): removed - full (higher better)")):
        all_values = []
        for index, axis in enumerate(AXES):
            matched = [row for row in rows if row["axis"] == axis and row["split"] == "validation"
                       and row["cause"] == "readmission"]
            if sorted(row["seed"] for row in matched) != sorted(seeds):
                raise ValueError(f"incomplete plot row {axis}")
            values = np.array([row[key] for row in sorted(matched, key=lambda row: row["seed"])])
            all_values.extend(values)
            if len(seeds) > 1:
                panel.scatter(values, index + np.linspace(-.08, .08, len(seeds)), color="#c05621", s=32, alpha=.65)
            panel.scatter(values.mean(), index, color="#c05621", marker="D", s=55)
        limit = max(max(abs(value) for value in all_values) * 1.13, 1e-5)
        panel.set_xlim(-limit, limit)
        panel.axvline(0, color="#444444", linewidth=1.1)
        panel.set_title(title, fontsize=11)
        panel.set_xlabel(f"Paired difference across {len(seeds)} seeds")
        panel.grid(axis="x", alpha=.18)
        panel.set_axisbelow(True)
    panels[0].set_yticks(np.arange(len(AXES)), LABELS, fontsize=10)
    panels[0].invert_yaxis()
    fig.suptitle("Strict AECOPD readmission validation; each lab axis retrained from scratch", fontsize=13, y=.985)
    fig.text(.5, .945, "All 22 axes; binary medication exposures only; no multiplicity adjustment", ha="center", fontsize=10)
    legend = ([Line2D([0], [0], marker="o", linestyle="None", color="#c05621", alpha=.65,
                      label="Individual seed")] if len(seeds) > 1 else []) + [
        Line2D([0], [0], marker="D", linestyle="None", color="#c05621",
               label="Seed mean" if len(seeds) > 1 else "Single-seed estimate"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False)
    fig.subplots_adjust(left=.22, right=.985, top=.90, bottom=.09, wspace=.12)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--bootstrap", type=int, default=100)
    args = parser.parse_args()
    compare(args.run_root, args.output_dir, seeds=tuple(args.seeds), draws=args.bootstrap)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/compare_broad_fragments.py · lines 1–126

<!-- REVIEW_FRAGMENT {"path":"training/v5/compare_broad_fragments.py","part":1,"start_line":1,"end_line":126,"source_sha256":"ce22501910519cf00f1d66e4608161433f769286f0a72afcab3baa54e232151f","fragment_sha256":"ce22501910519cf00f1d66e4608161433f769286f0a72afcab3baa54e232151f","added_terminal_newline":false} -->
````python
"""同患者/严格结局配对指标与英文科研图；单种子，探索性区间。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from training.bonsai_broad_fragments import atomic_json
from training.baselines.rank_metrics import auc_rank
from training.v5.compare_screened_group_ablations import _patient_draws, _predictions, _summary_for_cause
from training.v5.train import GRID, metrics

ARMS = ('no_text', 'old_fields_bert', 'fragments_bert', 'fragments_qwen')
LABELS = ('No text', 'Old four fields: BERT', 'Clinical fragments: BERT', 'Clinical fragments: Qwen')
PAIRS = (('fragments_bert', 'no_text'), ('fragments_qwen', 'no_text'),
         ('fragments_bert', 'old_fields_bert'), ('fragments_qwen', 'fragments_bert'))


def interval(times, events, subjects, candidate, reference, cause, *, draws=100):
    labels = ((events[:, None] == cause) & (times[:, None] <= GRID[None, :])).astype(float)
    error = np.trapezoid((candidate[:, 7:, cause]-labels[:, 7:])**2-
                         (reference[:, 7:, cause]-labels[:, 7:])**2, GRID[7:], axis=1)/358
    sampled = _patient_draws(subjects, draws, 2126)
    ibs = np.array([error[ix].mean() for ix in sampled])
    auc = []
    if min(int(labels[:, 90].sum()), len(times)-int(labels[:, 90].sum())) >= 20:
        for ix in sampled:
            y = labels[ix, 90].astype(int)
            if min(int(y.sum()), len(y)-int(y.sum())) >= 20:
                auc.append(auc_rank(candidate[ix, 90, cause], y)-auc_rank(reference[ix, 90, cause], y))
    return {'bootstrap_patient_cluster_draws': draws, 'delta_integrated_brier_95ci': np.quantile(ibs, (.025, .975)).tolist(),
        'delta_auc90_95ci': np.quantile(auc, (.025, .975)).tolist() if len(auc) >= draws//2 else None,
        'valid_auc90_draws': len(auc), 'interval_scope': 'patient sampling for one trained seed; not training randomness'}


def compare(root: Path) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = root/'paired_analysis'
    output.mkdir(parents=True, exist_ok=False)
    reports = {arm: json.loads((root/arm/'report.json').read_text(encoding='utf-8')) for arm in ARMS}
    for arm, report in reports.items():
        if report['status'] != 'training_completed':
            raise ValueError('Incomplete comparison arm')
        for key in ('member_sha256', 'label_sha256', 'primary_event', 'seed', 'shared_initialization_sha256',
                    'split_counts', 'medication_effect_mode', 'clinical_recovery_axes', 'field_roles',
                    'max_epochs', 'learning_rate', 'weight_decay', 'patient_batch_size'):
            if report[key] != reports['no_text'][key]:
                raise ValueError('Pairing contract differs: '+key)
    records, detailed, absolute = [], {}, {}
    for split in ('validation', 'historical_test'):
        predictions = {arm: _predictions(root/arm, split) for arm in ARMS}
        full = predictions['no_text']
        for arm, prediction in predictions.items():
            for key in ('subject_ids', 'hadm_ids', 'times', 'events', 'note_member'):
                if not np.array_equal(full[key], prediction[key]):
                    raise ValueError('Prediction pairing differs: '+key)
        absolute[split] = {arm: {name: _summary_for_cause(p['times'], p['events'], p['competing_cif'], cause)
                          for cause, name in ((1, 'readmission'), (2, 'death'))} for arm, p in predictions.items()}
        detailed[split] = []
        for added, reference in PAIRS:
            for cause, name in ((1, 'readmission'), (2, 'death')):
                a, b = absolute[split][added][name], absolute[split][reference][name]
                row = {'split': split, 'added': added, 'reference': reference, 'cause': name,
                    'delta_ibs': a['integrated_brier_7_365']-b['integrated_brier_7_365'],
                    'delta_mean_auc': a['mean_auc_7_365']-b['mean_auc_7_365'],
                    'delta_c_index365': a['c_index_365']-b['c_index_365'],
                    'delta_population_curve_gap': a['mean_absolute_population_curve_gap_7_365']-b['mean_absolute_population_curve_gap_7_365'],
                    'delta_calibration_gap90': a['points']['90']['absolute_mean_calibration_gap']-b['points']['90']['absolute_mean_calibration_gap']}
                records.append(row)
                detailed[split].append({**row, **interval(full['times'], full['events'], full['subject_ids'],
                    predictions[added]['competing_cif'], predictions[reference]['competing_cif'], cause)})
        figure, axes = plt.subplots(2, 2, figsize=(13, 9))
        for arm, label in zip(ARMS, LABELS, strict=True):
            p = predictions[arm]
            score = metrics(p['times'], p['events'], p['risk'])
            axes[0, 0].plot(GRID[7:], score['auc_curve'][7:], label=label)
            axes[0, 1].plot(GRID[7:], score['brier_curve'][7:], label=label)
            history = json.loads((root/arm/'history.json').read_text(encoding='utf-8'))
            axes[1, 0].plot([r['epoch'] for r in history], [r['validation_integrated_brier'] for r in history], label=label)
            axes[1, 1].plot(GRID, p['risk'].mean(axis=0), label=label)
        observed = ((full['events'][:, None] == 1) & (full['times'][:, None] <= GRID[None, :])).mean(axis=0)
        axes[1, 1].plot(GRID, observed, 'k--', label='Observed cumulative fraction (World A)')
        axes[0, 0].set(xlabel='Days after discharge', ylabel='AUC(t)', ylim=(0, 1), title='Strict readmission discrimination')
        axes[0, 1].set(xlabel='Days after discharge', ylabel='Brier(t)', title='Probability error (lower is better)')
        axes[1, 0].set(xlabel='Epoch', ylabel='Validation integrated Brier', title='Training checkpoint selection')
        axes[1, 1].set(xlabel='Days after discharge', ylabel='Cumulative probability', title='Population risk curves')
        for axis in axes.flat:
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
        figure.suptitle(f'{split}: matched patient histories; one seed; exploratory', fontsize=13)
        figure.tight_layout()
        figure.savefig(output/f'{split}_overview.png', dpi=160)
        plt.close(figure)
        readmission = [r for r in detailed[split] if r['cause'] == 'readmission']
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        labels = [f"{r['added']} - {r['reference']}" for r in readmission]
        for i, row in enumerate(readmission):
            lo, hi = row['delta_integrated_brier_95ci']
            axes[0].plot([lo, hi], [i, i], color='tab:blue', lw=2)
            axes[0].plot(row['delta_ibs'], i, 'D', color='tab:blue')
            axes[1].plot(row['delta_mean_auc'], i, 'D', color='tab:orange')
        axes[0].set_yticks(range(len(labels)), labels)
        axes[0].invert_yaxis()
        axes[0].set(xlabel='Added - reference', title='Integrated Brier difference (negative is better)')
        axes[1].set(xlabel='Added - reference', title='Mean AUC(t) difference (positive is better)')
        for axis in axes:
            axis.axvline(0, color='grey')
            axis.grid(alpha=.2)
        figure.suptitle(f'{split}: one seed; Brier CI = 100 patient-cluster draws; AUC = point estimate')
        figure.tight_layout()
        figure.savefig(output/f'{split}_paired_differences.png', dpi=160)
        plt.close(figure)
    with (output/'summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = {'status': 'paired_analysis_completed', 'absolute': absolute, 'paired': detailed,
        'primary_decision': 'jointly review readmission/death IBS, supported AUC(t), C-index and calibration; no automatic winner',
        'bootstrap_draws': 100, 'no_multiplicity_adjustment': True, 'single_seed': True,
        'not_comparable_to_old_full_cohort_auc': True}
    atomic_json(output/'paired_results.json', result)
    return result
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/compare_endpoint_experiment.py · lines 1–98

<!-- REVIEW_FRAGMENT {"path":"training/v5/compare_endpoint_experiment.py","part":1,"start_line":1,"end_line":98,"source_sha256":"e84b6c806badacc712a5d2dd62f1fdd51016d69d3ec837efaa1e5a411cd0d8a4","fragment_sha256":"e84b6c806badacc712a5d2dd62f1fdd51016d69d3ec837efaa1e5a411cd0d8a4","added_terminal_newline":false} -->
````python
"""结局改变的二维审计：固定预测换标签，与相同标签下重新训练分开比较。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.baselines.rank_metrics import auc_rank
from training.v5.strict_endpoint_labels import STRICT_PRIMARY_EVENT
from training.v5.train import _public_metrics, _save_curves, clustered_brier_interval, metrics
from training.v5.train_event_sequence import PRIOR_COUNT_CONCEPTS


def compare(union_dir: Path, strict_dir: Path) -> dict[str, object]:
    union_report = json.loads((union_dir / "report.json").read_text(encoding="utf-8"))
    strict_report = json.loads((strict_dir / "report.json").read_text(encoding="utf-8"))
    if strict_report["primary_event"] != STRICT_PRIMARY_EVENT:
        raise ValueError("wrong strict endpoint")
    fields = ("cohort_admissions", "cohort_patients", "field_manifest_sha256", "raw_source_sha256",
              "parameter_count", "event_concepts", "text_vector_width", "seed", "learning_rate",
              "weight_decay", "patient_batch_size", "max_epochs", "patience", "event_encoder")
    for key in fields:
        if union_report[key] != strict_report[key]:
            raise ValueError("non-outcome contract differs: " + key)
    if any(tuple(r["disabled_static_concepts"]) != PRIOR_COUNT_CONCEPTS for r in (union_report, strict_report)):
        raise ValueError("count ablation differs")
    results: dict[str, object] = {}
    curves: dict[str, dict[str, dict[str, object]]] = {}
    for split in ("validation", "historical_test"):
        with np.load(union_dir / f"{split}_predictions.npz") as archive:
            union = {name: archive[name] for name in archive.files}
        with np.load(strict_dir / f"{split}_predictions.npz") as archive:
            strict = {name: archive[name] for name in archive.files}
        for key in ("subject_ids", "hadm_ids"):
            if not np.array_equal(union[key], strict[key]):
                raise ValueError("prediction member/order differs: " + key)
        table = {}
        for endpoint, labels in (("union", union), ("strict", strict)):
            detailed = {"union_trained_model": metrics(labels["times"], labels["events"], union["risk"]),
                        "strict_trained_model": metrics(labels["times"], labels["events"], strict["risk"])}
            curves[split + "_" + endpoint] = detailed
            table[endpoint + "_evaluation"] = {key: _public_metrics(value) for key, value in detailed.items()}
        # 主训练结果必须逐指标复现，禁止张冠李戴地混用结局。
        for report, endpoint, model in ((union_report, "union", "union_trained_model"),
                                         (strict_report, "strict", "strict_trained_model")):
            for key in ("mean_auc_7_365", "integrated_brier_7_365"):
                if not np.isclose(table[endpoint + "_evaluation"][model][key], report[split]["temporal_full"][key], atol=1e-8, rtol=0):
                    raise ValueError("diagonal evaluation does not reproduce report")
        subjects, times, events = strict["subject_ids"], strict["times"], strict["events"]
        patients, inverse = np.unique(subjects, return_inverse=True)
        groups = [np.flatnonzero(inverse == i) for i in range(len(patients))]
        rng = np.random.default_rng(2026)
        samples = [np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
                   for _ in range(300)]
        intervals = {}
        for day in (30, 90, 365):
            observed = ((events == 1) & (times <= day)).astype(np.int8)
            differences = [auc_rank(strict["risk"][draw, day], observed[draw])
                           - auc_rank(union["risk"][draw, day], observed[draw]) for draw in samples]
            intervals[str(day)] = {"strict_retraining_auc_difference":
                table["strict_evaluation"]["strict_trained_model"]["points"][str(day)]["auc"]
                - table["strict_evaluation"]["union_trained_model"]["points"][str(day)]["auc"],
                "patient_cluster_paired_95_interval": np.quantile(differences, (.025, .975)).tolist()}
        strict_scores = table["strict_evaluation"]
        results[split] = {**table,
            "strict_endpoint_retraining_mean_auc_difference": strict_scores["strict_trained_model"]["mean_auc_7_365"] - strict_scores["union_trained_model"]["mean_auc_7_365"],
            "strict_endpoint_retraining_brier_difference": strict_scores["strict_trained_model"]["integrated_brier_7_365"] - strict_scores["union_trained_model"]["integrated_brier_7_365"],
            "strict_endpoint_retraining_brier_paired_95_interval": clustered_brier_interval(subjects, times, events, strict["risk"], union["risk"]),
            "strict_endpoint_point_auc_intervals": intervals,
            "event_counts": {endpoint: {str(i): int((labels["events"] == i).sum()) for i in range(3)}
                             for endpoint, labels in (("union", union), ("strict", strict))}}
    result = {"status": "endpoint_comparison_completed", "only_training_contract_change": "outcome event definition and derived labels",
              "patient_cluster_bootstrap_replicates": 300, "bootstrap_seed": 2026,
              "comparison_matrix": "rows=evaluation endpoint; columns=model training endpoint",
              "limitations": ["Different endpoints change cases/controls: diagonal AUC difference is not architecture improvement.",
                              "Brier values across different event prevalences are not direct performance improvements.",
                              "Single training seed; intervals do not include initialization uncertainty.",
                              "Historical test was previously examined; MIMIC outside-hospital follow-up remains incomplete."],
              **results}
    (strict_dir / "endpoint_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    destination = strict_dir / "comparison"
    destination.mkdir(exist_ok=True)
    _save_curves(destination, curves)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--union-dir", type=Path, required=True)
    parser.add_argument("--strict-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.union_dir, args.strict_dir)
    print(json.dumps({split: {name: {model: {metric: score[metric] for metric in ("mean_auc_7_365", "integrated_brier_7_365")}
                                   for model, score in result[split][name].items()}
                             for name in ("union_evaluation", "strict_evaluation")}
                      for split in ("validation", "historical_test")}))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/compare_independent_text.py · lines 1–115

<!-- REVIEW_FRAGMENT {"path":"training/v5/compare_independent_text.py","part":1,"start_line":1,"end_line":115,"source_sha256":"5100d20cf2332be3b5d844f2358f617e4719d0ce87a966e53f9251c5919d81cb","fragment_sha256":"5100d20cf2332be3b5d844f2358f617e4719d0ce87a966e53f9251c5919d81cb","added_terminal_newline":false} -->
````python
"""同一旧结局、同一患者划分上配对比较独立分支 BERT 与 Qwen。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_data import load_data, normalize_on_training_patients, split_sequences
from training.v5.independent_fusion import IndependentFusionModel
from training.v5.train import clustered_brier_interval, metrics
from training.v5.train_independent import predict


def _read(run_dir: Path, name: str) -> dict[str, object]:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("text_encoder") != name:
        raise ValueError("encoder report does not match experiment")
    return report


def _risk(aligned_dir: Path, structured_dir: Path, base_csv: Path,
          field_dir: Path, vectors: Path, run_dir: Path, name: str,
          split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data, _ = normalize_on_training_patients(
        load_data(aligned_dir, structured_dir, base_csv, field_dir, vectors)
    )
    sequences = split_sequences(data, split)
    selected = np.concatenate(sequences)
    model = IndependentFusionModel(
        concept_count=len(data.concept_names), type_count=len(data.type_names),
        field_count=len(FIELD_LABELS), text_width=data.text_width,
    ).to("cuda")
    model.load_state_dict(torch.load(run_dir / f"structured_field_{name}.pt",
                                     map_location="cuda", weights_only=True))
    risk, _ = predict(model, data, sequences, include_text=True,
                      batch_size=24, device=torch.device("cuda"))
    return data.hadm_ids[selected], data.subject_ids[selected], data.times[selected], data.events[selected], risk


def compare(aligned_dir: Path, structured_dir: Path, base_csv: Path, field_dir: Path,
            bert_vectors: Path, bert_run: Path, qwen_vectors: Path, qwen_run: Path,
            output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("paired encoder comparison requires CUDA")
    reports = {"bert": _read(bert_run, "bert"), "qwen": _read(qwen_run, "qwen")}
    for key in ("field_manifest_sha256", "cohort_admissions", "cohort_patients", "split_counts"):
        if reports["bert"][key] != reports["qwen"][key]:
            raise ValueError(f"comparison cohort contract differs: {key}")
    for split in ("validation", "historical_test"):
        left = reports["bert"][split]["structured"]
        right = reports["qwen"][split]["structured"]
        if left != right:
            raise ValueError(f"structured comparator differs in {split}; encoder comparison invalid")
    bert_history = json.loads((bert_run / "structured_history.json").read_text(encoding="utf-8"))
    qwen_history = json.loads((qwen_run / "structured_history.json").read_text(encoding="utf-8"))
    if bert_history != qwen_history or reports["bert"]["best_epochs"]["structured"] != reports["qwen"]["best_epochs"]["structured"]:
        raise ValueError("structured training histories differ; random streams are not aligned")
    summaries: dict[str, object] = {}
    for data_split, name in (("validation", "validation"), ("test", "historical_test")):
        bert = _risk(aligned_dir, structured_dir, base_csv, field_dir,
                     bert_vectors, bert_run, "bert", data_split)
        qwen = _risk(aligned_dir, structured_dir, base_csv, field_dir,
                     qwen_vectors, qwen_run, "qwen", data_split)
        if not all(np.array_equal(a, b) for a, b in zip(bert[:4], qwen[:4], strict=True)):
            raise ValueError("comparison rows or labels differ")
        b = metrics(bert[2], bert[3], bert[4])
        q = metrics(qwen[2], qwen[3], qwen[4])
        summaries[name] = {
            "bert": {"integrated_brier_7_365": b["integrated_brier_7_365"],
                     "mean_auc_7_365": b["mean_auc_7_365"]},
            "qwen": {"integrated_brier_7_365": q["integrated_brier_7_365"],
                     "mean_auc_7_365": q["mean_auc_7_365"]},
            "qwen_minus_bert_integrated_brier":
                float(q["integrated_brier_7_365"] - b["integrated_brier_7_365"]),
            "patient_cluster_95ci": clustered_brier_interval(
                bert[1], bert[2], bert[3], qwen[4], bert[4]
            ),
        }
    result = {
        "status": "single-seed internal comparison; old strict outcome; historical test not blind",
        "manifest_sha256": reports["bert"]["field_manifest_sha256"],
        "identical_structured_baselines": True,
        "identical_structured_histories": True,
        "splits": summaries,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--field-dir", type=Path, required=True)
    parser.add_argument("--bert-vectors", type=Path, required=True)
    parser.add_argument("--bert-run", type=Path, required=True)
    parser.add_argument("--qwen-vectors", type=Path, required=True)
    parser.add_argument("--qwen-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.aligned_dir, args.structured_dir, args.base_csv,
                             args.field_dir, args.bert_vectors, args.bert_run,
                             args.qwen_vectors, args.qwen_run, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/compare_lab_recovery_pilot.py · lines 1–88

<!-- REVIEW_FRAGMENT {"path":"training/v5/compare_lab_recovery_pilot.py","part":1,"start_line":1,"end_line":88,"source_sha256":"11a7f1d0711bf80e4a21ad536dddd0e138fb6530553dc980b77f05e930509cac","fragment_sha256":"11a7f1d0711bf80e4a21ad536dddd0e138fb6530553dc980b77f05e930509cac","added_terminal_newline":false} -->
````python
"""同初始化、同队列比较真实异常恢复摘要与全遮蔽摘要。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.v5.compare_screened_group_ablations import (
    CONTRACT_KEYS, _paired_auc, _paired_brier, _patient_draws, _predictions,
    _summary_for_cause,
)
from training.v5.train import GRID


def run(root: Path, output: Path, *, seed: int, draws: int = 200,
        control_mode: str = "masked") -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    if control_mode not in ("masked", "measurement_only", "core9"):
        raise ValueError("unknown clinical control mode")
    masked_dir = root / f"{control_mode}_seed{seed}"
    active_dir = root / f"active_seed{seed}"
    masked_report = json.loads((masked_dir / "report.json").read_text(encoding="utf-8"))
    active_report = json.loads((active_dir / "report.json").read_text(encoding="utf-8"))
    if any(report["status"] != "training_completed" or report["event_encoder"] not in ("set", "abnormal")
           for report in (masked_report, active_report)):
        raise ValueError("incomplete or incorrect clinical summary run")
    if (masked_report["clinical_recovery_mode"] != control_mode
            or active_report["clinical_recovery_mode"] != "active"
            or masked_report["clinical_recovery_manifest_sha256"]
            != active_report["clinical_recovery_manifest_sha256"]):
        raise ValueError("clinical recovery arm contract mismatch")
    for key in CONTRACT_KEYS:
        if masked_report[key] != active_report[key]:
            raise ValueError(f"training contract differs: {key}")
    result = {"status": "single_seed_pilot", "seed": seed,
              "control_mode": control_mode,
              "delta_definition": f"active clinical summary minus identical-architecture {control_mode} summary",
              "same_cohort_model_and_source_contract": True,
              "validation_patient_cluster_bootstrap_draws": draws,
              "splits": {}}
    for split in ("validation", "historical_test"):
        masked, active = _predictions(masked_dir, split), _predictions(active_dir, split)
        for key in ("subject_ids", "hadm_ids", "times", "events"):
            if not np.array_equal(masked[key], active[key]):
                raise ValueError(f"prediction samples differ: {split} {key}")
        times, events, subjects = masked["times"], masked["events"], masked["subject_ids"]
        result["splits"][split] = {}
        draws_index = _patient_draws(subjects, draws, seed + 200) if split == "validation" else []
        for cause, name in ((1, "readmission"), (2, "death")):
            no_summary = masked["competing_cif"][:, :, cause]
            with_summary = active["competing_cif"][:, :, cause]
            old = _summary_for_cause(times, events, masked["competing_cif"], cause)
            new = _summary_for_cause(times, events, active["competing_cif"], cause)
            score = {control_mode: old, "active": new,
                     f"delta_active_minus_{control_mode}": {
                         key: new[key] - old[key] for key in
                         ("integrated_brier_7_365", "mean_auc_7_365", "c_index_90", "c_index_365",
                          "mean_absolute_population_curve_gap_7_365")}}
            if split == "validation":
                labels = ((events[:, None] == cause) & (times[:, None] <= GRID[None, :])).astype(float)
                brier = _paired_brier(subjects, labels, with_summary, no_summary, draws_index)
                auc = _paired_auc(labels[:, 90].astype(bool), with_summary[:, 90],
                                  no_summary[:, 90], draws_index)
                score["paired_brier_interval"] = {
                    f"difference_active_minus_{control_mode}": brier["difference_removed_minus_full"],
                    "patient_cluster_200_draw_95_interval": brier["patient_cluster_200_draw_95_interval"]}
                score["paired_auc_90_interval"] = {
                    f"difference_active_minus_{control_mode}": auc["difference_removed_minus_full"],
                    "patient_cluster_200_draw_95_interval": auc["patient_cluster_200_draw_95_interval"]}
            result["splits"][split][name] = score
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--control-mode", choices=("masked", "measurement_only", "core9"), default="masked")
    args = parser.parse_args()
    print(json.dumps({"status": run(args.run_root, args.output, seed=args.seed,
                                     draws=args.bootstrap, control_mode=args.control_mode)["status"]}))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/compare_medication_effect_ablations.py · lines 1–83

<!-- REVIEW_FRAGMENT {"path":"training/v5/compare_medication_effect_ablations.py","part":1,"start_line":1,"end_line":83,"source_sha256":"6254f1448efe3724aa04326f7b1f2706fcc4d0c579aa193a7b4fd54e9c1e84af","fragment_sha256":"6254f1448efe3724aa04326f7b1f2706fcc4d0c579aa193a7b4fd54e9c1e84af","added_terminal_newline":false} -->
````python
"""同种子、同队列配对比较药理作用组；不将给药暴露解释为疗效。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.v5.compare_screened_group_ablations import (
    CONTRACT_KEYS, _paired_auc, _paired_brier, _patient_draws, _predictions,
    _summary_for_cause,
)
from training.v5.train import GRID


def run(root: Path, output: Path, *, seed: int, control: str = "masked", draws: int = 200) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if control not in ("masked", "exposure_only") and not control.startswith("without_"):
        raise ValueError("invalid medication comparison control")
    before = root / f"{control}_seed{seed}"
    after = root / f"active_seed{seed}"
    reports = [json.loads((path / "report.json").read_text(encoding="utf-8")) for path in (before, after)]
    if any(r["status"] != "training_completed" or r["event_encoder"] != "abnormal_medication"
           or r["clinical_recovery_mode"] != "active" for r in reports):
        raise ValueError("incomplete medication effect run")
    if reports[0]["medication_effect_mode"] != control or reports[1]["medication_effect_mode"] != "active":
        raise ValueError("incorrect medication effect modes")
    for key in (*CONTRACT_KEYS, "clinical_recovery_manifest_sha256", "medication_effect_groups",
                "medication_effect_feature_count"):
        if reports[0][key] != reports[1][key]:
            raise ValueError(f"paired contract differs: {key}")
    result = {"status": "completed", "seed": seed, "control": control,
              "comparison": "active minus control, independently trained from same seed",
              "limitations": ["Patient-cluster intervals condition on this seed and split.",
                              "Historical test is not blind.",
                              "Medication is an administration proxy, not a treatment effect or indication."],
              "splits": {}}
    for split in ("validation", "historical_test"):
        old, new = _predictions(before, split), _predictions(after, split)
        for key in ("subject_ids", "hadm_ids", "times", "events"):
            if not np.array_equal(old[key], new[key]):
                raise ValueError(f"different paired predictions: {key}")
        subjects, times, events = old["subject_ids"], old["times"], old["events"]
        patient_draws = _patient_draws(subjects, draws, seed + 100) if split == "validation" else []
        result["splits"][split] = {}
        for cause, name in ((1, "readmission"), (2, "death")):
            baseline = _summary_for_cause(times, events, old["competing_cif"], cause)
            candidate = _summary_for_cause(times, events, new["competing_cif"], cause)
            comparison = {"control": baseline, "active": candidate,
                          "delta_active_minus_control": {key: candidate[key] - baseline[key]
                              for key in ("mean_auc_7_365", "integrated_brier_7_365", "c_index_90",
                                          "c_index_365", "mean_absolute_population_curve_gap_7_365")}}
            if split == "validation":
                labels = ((events[:, None] == cause) & (times[:, None] <= GRID[None, :])).astype(float)
                brier = _paired_brier(subjects, labels, new["competing_cif"][:, :, cause],
                                      old["competing_cif"][:, :, cause], patient_draws)
                auc = _paired_auc(labels[:, 90].astype(bool), new["competing_cif"][:, 90, cause],
                                  old["competing_cif"][:, 90, cause], patient_draws)
                comparison["paired_bootstrap"] = {
                    "brier_active_minus_control": {
                        "point_difference": brier["difference_removed_minus_full"],
                        "patient_cluster_200_draw_95_interval": brier["patient_cluster_200_draw_95_interval"]},
                    "auc90_active_minus_control": {
                        "point_difference": auc["difference_removed_minus_full"],
                        "patient_cluster_200_draw_95_interval": auc["patient_cluster_200_draw_95_interval"]}}
            result["splits"][split][name] = comparison
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--control", default="masked")
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps({"status": run(args.run_root, args.output, seed=args.seed,
                                    control=args.control, draws=args.bootstrap)["status"]}))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/compare_prior_count_ablation.py · lines 1–153

<!-- REVIEW_FRAGMENT {"path":"training/v5/compare_prior_count_ablation.py","part":1,"start_line":1,"end_line":153,"source_sha256":"923273d35a2fc15a837e70ce35ce70d2d8a43d310a3e7a3b1f102394fc8c0f5f","fragment_sha256":"923273d35a2fc15a837e70ce35ce70d2d8a43d310a3e7a3b1f102394fc8c0f5f","added_terminal_newline":false} -->
````python
"""在共同患者上比较显式既往次数消融；旧基线只重载推理，不重新训练。"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from training.baselines.rank_metrics import auc_rank
from training.v5.event_set_model import EventSetFusionModel
from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_data import split_sequences
from training.v5.processed_event_data import load_processed_event_data
from training.v5.strict_endpoint_labels import STRICT_PRIMARY_EVENT
from training.v5.train import _public_metrics, _save_curves, clustered_brier_interval, metrics
from training.v5.train_event_sequence import PRIOR_COUNT_CONCEPTS, predict


def run(baseline_dir: Path, ablation_dir: Path,
        comparison_dir: Path | None = None) -> dict[str, object]:
    baseline = json.loads((baseline_dir / "report.json").read_text(encoding="utf-8"))
    ablation = json.loads((ablation_dir / "report.json").read_text(encoding="utf-8"))
    for key in ("cohort_admissions", "cohort_patients", "field_manifest_sha256", "label_sha256",
                "raw_source_sha256", "parameter_count", "event_concepts", "text_vector_width",
                "seed", "learning_rate", "weight_decay", "patient_batch_size", "max_epochs", "patience"):
        if baseline[key] != ablation[key]:
            raise ValueError(f"comparison contract differs: {key}")
    if baseline.get("disabled_static_concepts") or tuple(ablation["disabled_static_concepts"]) != PRIOR_COUNT_CONCEPTS:
        raise ValueError("incorrect count ablation")
    # 本审计固定本轮数据版本；只读源文件并复核完整输入合同。
    private = Path("outputs/01a08b96-2988-7001-af1d-b40d901f3400/private")
    aligned = private / "v5_section_aligned"
    fields = aligned / "field_text_baseline_2026_09_27"
    if baseline["primary_event"] == STRICT_PRIMARY_EVENT:
        labels_path = Path("patient_data/derived/strict_aecopd_endpoint_2026-09-28/labels.csv.gz")
    else:
        labels_path = private / "v5_qualified_copd_labels_2026_09_27/labels.csv.gz"
    data, _ = load_processed_event_data(
        aligned / "dataset_v4", aligned / "section_inputs_sparse_2026_09_26",
        private / "v5_mimiciv_2026-09-25/structured/base.csv",
        Path("patient_data/processed/mimiciv_copd_events_2026-09-27"),
        Path("patient_data/mimiciv_copd/snapshot_2026-09-27"),
        labels_csv=labels_path,
        field_dir=fields, vectors_path=fields / "bert/bioclinicalbert.float16.npy")
    if data.base.label_sha256 != baseline["label_sha256"] or data.base.manifest_sha256 != baseline["field_manifest_sha256"]:
        raise ValueError("reloaded data differs from completed run")
    device = torch.device("cuda")
    model = EventSetFusionModel(concept_count=len(data.base.concept_names),
                                type_count=len(data.base.type_names), field_count=len(FIELD_LABELS),
                                text_width=data.base.text_width,
                                event_concept_count=len(data.concept_names)).to(device)
    model.load_state_dict(torch.load(baseline_dir / "model.pt", map_location=device, weights_only=True), strict=True)
    ablated_model = EventSetFusionModel(concept_count=len(data.base.concept_names),
                                       type_count=len(data.base.type_names), field_count=len(FIELD_LABELS),
                                       text_width=data.base.text_width,
                                       event_concept_count=len(data.concept_names)).to(device)
    ablated_model.load_state_dict(torch.load(ablation_dir / "model.pt", map_location=device, weights_only=True), strict=True)
    ablated_data = replace(data, base=replace(data.base, disabled_concepts=PRIOR_COUNT_CONCEPTS))
    results: dict[str, object] = {}
    reload_deltas: dict[str, float] = {}
    curves: dict[str, dict[str, dict[str, object]]] = {}
    for name, split in (("validation", "validation"), ("historical_test", "test")):
        sequences = split_sequences(data.base, split)
        indices = np.concatenate(sequences)
        original, _ = predict(model, data, sequences, process_only=False,
                              batch_size=int(baseline["patient_batch_size"]), device=device)
        with np.load(ablation_dir / f"{name}_predictions.npz") as saved:
            reduced = saved["risk"].copy()
            for key in ("subject_ids", "hadm_ids", "times", "events"):
                if not np.array_equal(saved[key], getattr(data.base, key)[indices]):
                    raise ValueError(f"paired prediction alignment differs: {key}")
        reloaded, _ = predict(ablated_model, ablated_data, sequences, process_only=False,
                              batch_size=int(ablation["patient_batch_size"]), device=device)
        reload_deltas[name] = float(np.max(np.abs(reloaded - reduced)))
        # GPU 重载可有亚微小浮点差；记录实测最大差，超过百万分之一则失败。
        if not np.allclose(reloaded, reduced, atol=1e-6, rtol=0):
            raise ValueError(f"ablation checkpoint with count mask does not reproduce saved predictions: {name}, max risk delta={float(np.max(np.abs(reloaded - reduced)))}")
        times, events, subjects = data.base.times[indices], data.base.events[indices], data.base.subject_ids[indices]
        full_metrics, removed_metrics = metrics(times, events, original), metrics(times, events, reduced)
        expected = baseline[name]["temporal_full"]
        for key in ("mean_auc_7_365", "integrated_brier_7_365"):
            if not np.isclose(full_metrics[key], expected[key], atol=1e-7, rtol=0):
                raise ValueError(f"checkpoint prediction does not reproduce baseline: {key}")
        unique, inverse = np.unique(subjects, return_inverse=True)
        groups = [np.flatnonzero(inverse == index) for index in range(len(unique))]
        rng = np.random.default_rng(2026)
        draws = [np.concatenate([groups[index] for index in rng.integers(len(groups), size=len(groups))])
                 for _ in range(300)]
        auc_intervals = {}
        for day in (30, 90, 365):
            labels = ((events == 1) & (times <= day)).astype(np.int8)
            differences = [auc_rank(reduced[draw, day], labels[draw])
                           - auc_rank(original[draw, day], labels[draw]) for draw in draws]
            auc_intervals[str(day)] = {
                "difference": removed_metrics["points"][str(day)]["auc"] - full_metrics["points"][str(day)]["auc"],
                "patient_cluster_paired_95_interval": np.quantile(differences, (.025, .975)).tolist()}
        results[name] = {
            "baseline": _public_metrics(full_metrics), "without_explicit_counts": _public_metrics(removed_metrics),
            "mean_auc_difference": removed_metrics["mean_auc_7_365"] - full_metrics["mean_auc_7_365"],
            "integrated_brier_difference": removed_metrics["integrated_brier_7_365"] - full_metrics["integrated_brier_7_365"],
            "integrated_brier_paired_95_interval": clustered_brier_interval(subjects, times, events, reduced, original),
            "point_auc_differences": auc_intervals}
        curves[name] = {"with_explicit_counts": full_metrics, "without_explicit_counts": removed_metrics}
    report = {"status": "paired_comparison_completed", "difference_direction": "without counts minus baseline",
              "bootstrap_patient_clusters": 300, "seed": 2026,
              "baseline_checkpoint_reproduced": True, "ablation_checkpoint_reproduced": True,
              "ablation_reload_max_risk_difference": reload_deltas,
              "checkpoint_prediction_absolute_tolerance": 1e-6,
              "contract_fields_checked": True,
              "limitations": ["Single training seed; bootstrap describes sampled patients, not initialization uncertainty.",
                              "History length, gaps, diagnoses and text still carry frequency proxies.",
                              "Historical test has been examined before and is not a new blind evaluation."],
              **results}
    destination = ablation_dir if comparison_dir is None else comparison_dir
    if not destination.is_dir():
        raise ValueError("comparison destination must be an existing run directory")
    (destination / "paired_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    chart_dir = destination / "comparison"
    chart_dir.mkdir(exist_ok=True)
    _save_curves(chart_dir, curves)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for directory, label in ((baseline_dir, "with explicit counts"), (ablation_dir, "without explicit counts")):
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        epochs = [row["epoch"] for row in history]
        for axis, key, title in zip(axes,
                ("train_nll", "validation_integrated_brier", "validation_mean_auc"),
                ("Training negative log likelihood", "Validation integrated Brier", "Validation mean AUC(t)")):
            axis.plot(epochs, [row[key] for row in history], label=label)
            axis.set(xlabel="Epoch", ylabel=title)
            axis.grid(alpha=.2)
            axis.legend()
    figure.tight_layout()
    figure.savefig(chart_dir / "training_comparison.png", dpi=160)
    plt.close(figure)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path)
    arguments = parser.parse_args()
    result = run(arguments.baseline_dir, arguments.ablation_dir, arguments.comparison_dir)
    print(json.dumps({name: {key: value for key, value in result[name].items()
                            if key not in {"baseline", "without_explicit_counts"}}
                      for name in ("validation", "historical_test")}))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/compare_revised_group_ablations.py · lines 1–167

<!-- REVIEW_FRAGMENT {"path":"training/v5/compare_revised_group_ablations.py","part":1,"start_line":1,"end_line":167,"source_sha256":"e9cef6f16ad9f1de5692c41c499bafdbb22a3ca551814bc11ec974cb53a54224","fragment_sha256":"e9cef6f16ad9f1de5692c41c499bafdbb22a3ca551814bc11ec974cb53a54224","added_terminal_newline":false} -->
````python
"""Compare independently retrained, one-group-removed clinical input arms with V5 full."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from training.v5.compare_screened_group_ablations import (
    CONTRACT_KEYS,
    _paired_auc,
    _paired_brier,
    _patient_draws,
    _predictions,
    _summary_for_cause,
)
from training.v5.train import GRID

GROUPS = (
    "med_short_beta_agonist", "med_short_muscarinic_antagonist",
    "med_long_beta_agonist", "med_long_muscarinic_antagonist",
    "med_inhaled_corticosteroid", "med_systemic_corticosteroid",
    "med_common_systemic_antibacterial", "med_diuretic",
    "lab_ventilation_acid_base", "lab_inflammation_infection",
    "lab_nutrition_hematology", "lab_renal", "lab_electrolytes",
    "lab_metabolic_perfusion", "lab_cardiac_burden",
    "med_short_bronchodilators", "med_maintenance_inhalers",
    "lean_clinical", "selected_group_removal",
)
SPLITS = ("validation", "historical_test")
CAUSES = ((1, "readmission"), (2, "death"))
EXTRA_CONTRACT = (
    "clinical_recovery_manifest_sha256", "abnormality_scaling",
    "medication_effect_groups", "medication_effect_feature_count",
    "abnormality_feature_count", "abnormality_scale_mode", "raw_events_enter_model",
)


def _report(path: Path) -> dict:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "training_completed" or report["event_encoder"] != "abnormal_medication":
        raise ValueError(f"Incomplete or wrong-model run: {path}")
    return report


def _check_contract(full: dict, removed: dict, group: str) -> None:
    for key in (*CONTRACT_KEYS, *EXTRA_CONTRACT):
        if full[key] != removed[key]:
            raise ValueError(f"Non-paired run for {group}: {key}")
    if full["clinical_recovery_mode"] != "active" or full["medication_effect_mode"] != "active":
        raise ValueError("Full reference does not enable both modalities")
    expected_lab = ("without_renal_metabolic_perfusion" if group == "selected_group_removal" else
                    "core_risk" if group == "lean_clinical" else
                    f"without_{group[4:]}" if group.startswith("lab_") else "active")
    expected_med = ("without_diuretic" if group == "selected_group_removal" else
                    "core_acute" if group == "lean_clinical" else
                    f"without_{group[4:]}" if group.startswith("med_") else "active")
    if (removed["clinical_recovery_mode"], removed["medication_effect_mode"]) != (expected_lab, expected_med):
        raise ValueError(f"Wrong ablation mode for {group}")


def compare(root: Path, *, seeds: tuple[int, ...], groups: tuple[str, ...],
            draws: int = 200) -> list[dict]:
    rows = []
    for seed in seeds:
        full_dir = root / f"full_seed{seed}"
        full_report = _report(full_dir)
        full_predictions = {split: _predictions(full_dir, split) for split in SPLITS}
        for group in groups:
            removed_dir = (root / f"selected_group_removal_seed{seed}" if group == "selected_group_removal" else
                           root / f"lean_clinical_seed{seed}" if group == "lean_clinical" else
                           root / "groups" / f"without_{group}_seed{seed}")
            removed_report = _report(removed_dir)
            _check_contract(full_report, removed_report, group)
            for split in SPLITS:
                full = full_predictions[split]
                removed = _predictions(removed_dir, split)
                for key in ("subject_ids", "hadm_ids", "times", "events"):
                    if not np.array_equal(full[key], removed[key]):
                        raise ValueError(f"Unpaired {key} for {group}, seed={seed}, split={split}")
                times, events, subjects = full["times"], full["events"], full["subject_ids"]
                bootstrap = _patient_draws(subjects, draws, seed + 100) if split == "validation" else None
                for cause, cause_name in CAUSES:
                    reference = _summary_for_cause(times, events, full["competing_cif"], cause)
                    candidate = _summary_for_cause(times, events, removed["competing_cif"], cause)
                    row = {"seed": seed, "group_removed": group, "split": split, "cause": cause_name,
                           "full_brier": reference["integrated_brier_7_365"],
                           "removed_brier": candidate["integrated_brier_7_365"],
                           "delta_brier": candidate["integrated_brier_7_365"] - reference["integrated_brier_7_365"],
                           "full_auc": reference["mean_auc_7_365"],
                           "removed_auc": candidate["mean_auc_7_365"],
                           "delta_auc": candidate["mean_auc_7_365"] - reference["mean_auc_7_365"],
                           "delta_c_index_90": candidate["c_index_90"] - reference["c_index_90"],
                           "delta_c_index_365": candidate["c_index_365"] - reference["c_index_365"],
                           "delta_auc_90": candidate["points"]["90"]["auc"] - reference["points"]["90"]["auc"],
                           "delta_brier_90": candidate["points"]["90"]["brier"] - reference["points"]["90"]["brier"],
                           "delta_curve_gap": candidate["mean_absolute_population_curve_gap_7_365"]
                           - reference["mean_absolute_population_curve_gap_7_365"],
                           "delta_calibration_gap_90": candidate["points"]["90"]["absolute_mean_calibration_gap"]
                           - reference["points"]["90"]["absolute_mean_calibration_gap"]}
                    if cause == 1:
                        for key in ("30d_0.10", "90d_0.20"):
                            row[f"delta_dca_net_benefit_{key}"] = (
                                candidate["exploratory_dca_net_benefit"][key]
                                - reference["exploratory_dca_net_benefit"][key])
                    if bootstrap is not None:
                        labels = ((events[:, None] == cause) & (times[:, None] <= GRID[None, :])).astype(float)
                        row["brier_95_interval"] = _paired_brier(
                            subjects, labels, removed["competing_cif"][:, :, cause],
                            full["competing_cif"][:, :, cause], bootstrap
                        )["patient_cluster_200_draw_95_interval"]
                        row["auc_90_95_interval"] = _paired_auc(
                            labels[:, 90].astype(bool), removed["competing_cif"][:, 90, cause],
                            full["competing_cif"][:, 90, cause], bootstrap
                        )["patient_cluster_200_draw_95_interval"]
                    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows = compare(args.run_root, seeds=tuple(args.seeds), groups=tuple(args.groups))
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "paired_group_metrics.json").write_text(
        json.dumps({"direction": "removed minus full; lower Brier and higher AUC are better",
                    "bootstrap": "200 patient-cluster draws per trained seed; not training uncertainty",
                    "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "paired_group_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [key for key in rows[0] if key not in ("brier_95_interval", "auc_90_95_interval")]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for group in args.groups:
        for split in SPLITS:
            for _, cause_name in CAUSES:
                matched = [row for row in rows if row["group_removed"] == group
                           and row["split"] == split and row["cause"] == cause_name]
                summary.append({
                    "group_removed": group, "split": split, "cause": cause_name,
                    "seeds": len(matched),
                    "mean_delta_brier": float(np.mean([row["delta_brier"] for row in matched])),
                    "mean_delta_auc": float(np.mean([row["delta_auc"] for row in matched])),
                    "mean_delta_c_index_365": float(np.mean([row["delta_c_index_365"] for row in matched])),
                    "mean_delta_calibration_gap_90": float(np.mean([row["delta_calibration_gap_90"] for row in matched])),
                    "seeds_brier_worse_after_removal": sum(row["delta_brier"] > 0 for row in matched),
                    "seeds_auc_worse_after_removal": sum(row["delta_auc"] < 0 for row in matched),
                })
    with (args.output_dir / "group_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps({"rows": len(rows), "seeds": args.seeds, "groups": args.groups}, ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/compare_revised_labs_meds.py · lines 1–99

<!-- REVIEW_FRAGMENT {"path":"training/v5/compare_revised_labs_meds.py","part":1,"start_line":1,"end_line":99,"source_sha256":"dfcc9931601ba70ac4b269019dc1373c2b18f9f2dd88de34fc0305a797d53922","fragment_sha256":"dfcc9931601ba70ac4b269019dc1373c2b18f9f2dd88de34fc0305a797d53922","added_terminal_newline":false} -->
````python
"""比较同一新版输入上的四种检验/药物组合，不与旧版指标直接相减。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.v5.compare_screened_group_ablations import (
    CONTRACT_KEYS, _paired_auc, _paired_brier, _patient_draws, _predictions,
    _summary_for_cause,
)
from training.v5.train import GRID


ARMS = {
    "baseline": ("masked", "masked"),
    "lab_only": ("active", "masked"),
    "med_only": ("masked", "active"),
    "full": ("active", "active"),
}
PAIRS = (("baseline", "lab_only"), ("baseline", "med_only"),
         ("lab_only", "full"), ("med_only", "full"), ("baseline", "full"))
METRICS = ("integrated_brier_7_365", "mean_auc_7_365", "c_index_90", "c_index_365",
           "mean_absolute_population_curve_gap_7_365")


def run(root: Path, output: Path, *, seeds: tuple[int, ...], draws: int = 200) -> dict:
    if output.exists():
        raise FileExistsError(output)
    result: dict[str, object] = {
        "status": "completed", "comparison": "second minus first, independently retrained",
        "patient_bootstrap_draws": draws, "seeds": {},
        "limitations": ["Historical test is not blind", "eMAR route is absent for many administrations",
                        "Drug exposure is not treatment efficacy"],
    }
    for seed in seeds:
        folders = {arm: root / f"{arm}_seed{seed}" for arm in ARMS}
        reports = {arm: json.loads((folder / "report.json").read_text(encoding="utf-8"))
                   for arm, folder in folders.items()}
        reference = reports["baseline"]
        for arm, report in reports.items():
            if (report["status"] != "training_completed" or report["event_encoder"] != "abnormal_medication"
                    or report["clinical_recovery_mode"] != ARMS[arm][0]
                    or report["medication_effect_mode"] != ARMS[arm][1]
                    or report["abnormality_scale_mode"] != "training_positive_p95"):
                raise ValueError(f"incorrect run contract: {arm} seed {seed}")
            for key in (*CONTRACT_KEYS, "clinical_recovery_manifest_sha256",
                        "abnormality_scaling", "abnormality_scale_mode", "medication_effect_groups"):
                if report[key] != reference[key]:
                    raise ValueError(f"paired run differs in {key}: {arm} seed {seed}")
        result["seeds"][str(seed)] = {}
        for split in ("validation", "historical_test"):
            arrays = {arm: _predictions(folder, split) for arm, folder in folders.items()}
            first = arrays["baseline"]
            for arm, item in arrays.items():
                for key in ("subject_ids", "hadm_ids", "times", "events"):
                    if not np.array_equal(item[key], first[key]):
                        raise ValueError(f"unpaired {split} {arm}: {key}")
            scores = {(arm, cause): _summary_for_cause(first["times"], first["events"],
                                                       item["competing_cif"], cause)
                      for arm, item in arrays.items() for cause in (1, 2)}
            draws_for_patients = (_patient_draws(first["subject_ids"], draws, seed + 100)
                                  if split == "validation" else [])
            result["seeds"][str(seed)][split] = {}
            for old, new in PAIRS:
                pair = {}
                for cause, name in ((1, "readmission"), (2, "death")):
                    before, after = scores[(old, cause)], scores[(new, cause)]
                    comparison = {"first": before, "second": after,
                                  "delta_second_minus_first": {key: after[key] - before[key]
                                                               for key in METRICS}}
                    if split == "validation":
                        labels = ((first["events"][:, None] == cause)
                                  & (first["times"][:, None] <= GRID[None, :])).astype(float)
                        brier = _paired_brier(first["subject_ids"], labels,
                                              arrays[new]["competing_cif"][:, :, cause],
                                              arrays[old]["competing_cif"][:, :, cause], draws_for_patients)
                        auc = _paired_auc(labels[:, 90].astype(bool),
                                          arrays[new]["competing_cif"][:, 90, cause],
                                          arrays[old]["competing_cif"][:, 90, cause], draws_for_patients)
                        comparison["paired_bootstrap"] = {"integrated_brier": brier, "auc90": auc}
                    pair[name] = comparison
                result["seeds"][str(seed)][split][f"{new}_minus_{old}"] = pair
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps({"status": run(args.run_root, args.output, seeds=tuple(args.seeds),
                                    draws=args.bootstrap)["status"]}))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/compare_screened_group_ablations.py · lines 1–218

<!-- REVIEW_FRAGMENT {"path":"training/v5/compare_screened_group_ablations.py","part":1,"start_line":1,"end_line":218,"source_sha256":"0ba4acf8c1dfd10f612f33d65a27fceb73330b886cfad8fc559fd71e82144d99","fragment_sha256":"0ba4acf8c1dfd10f612f33d65a27fceb73330b886cfad8fc559fd71e82144d99","added_terminal_newline":false} -->
````python
"""同标签同患者下比较从头重训的临床信息组；正值/负值方向均显式报告。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from training.baselines.rank_metrics import auc_rank
from training.v5.competing_concordance import cif_concordance
from training.v5.train import GRID, metrics

CONTRACT_KEYS = ("cohort_admissions", "cohort_patients", "field_manifest_sha256", "label_sha256",
                 "raw_source_sha256", "parameter_count", "event_concepts", "text_vector_width",
                 "seed", "learning_rate", "weight_decay", "patient_batch_size", "max_epochs", "patience",
                 "primary_event", "event_encoder")
POINTS = (30, 90, 180, 365)


def _read_run(directory: Path) -> dict[str, object]:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "training_completed" or report["event_encoder"] != "set":
        raise ValueError(f"incomplete or incorrect run: {directory}")
    return report


def _predictions(directory: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(directory / f"{split}_predictions.npz", allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    if result["competing_cif"].shape != (len(result["times"]), 366, 3):
        raise ValueError(f"missing full competing-risk curve in {directory}")
    if not np.allclose(result["risk"], result["competing_cif"][:, :, 1], atol=1e-7):
        raise ValueError("readmission curve mismatch")
    cifs = result["competing_cif"]
    if (not np.isfinite(cifs).all() or np.any(cifs < -1e-7) or np.any(cifs > 1 + 1e-7)
            or np.any(np.diff(cifs[:, :, 1:], axis=1) < -1e-6)
            or not np.allclose(cifs.sum(axis=2), 1, atol=1e-6)
            or np.any(cifs[:, :, 1] + cifs[:, :, 2] > 1 + 1e-6)):
        raise ValueError(f"invalid competing cumulative-risk curves in {directory}")
    return result


def _patient_draws(subjects: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    unique, inverse = np.unique(subjects, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            for _ in range(count)]


def _paired_brier(subjects: np.ndarray, labels: np.ndarray,
                  candidate: np.ndarray, full: np.ndarray,
                  draws: list[np.ndarray]) -> dict[str, object]:
    difference = np.trapezoid((candidate[:, 7:] - labels[:, 7:]) ** 2
                              - (full[:, 7:] - labels[:, 7:]) ** 2, GRID[7:], axis=1) / (365 - 7)
    values = np.array([difference[draw].mean() for draw in draws])
    return {"difference_removed_minus_full": float(difference.mean()),
            "patient_cluster_200_draw_95_interval": np.quantile(values, (.025, .975)).tolist()}


def _paired_auc(labels: np.ndarray, candidate: np.ndarray, full: np.ndarray,
                draws: list[np.ndarray]) -> dict[str, object]:
    values = []
    for draw in draws:
        sampled = labels[draw]
        if 0 < sampled.sum() < len(sampled):
            values.append(auc_rank(candidate[draw], sampled) - auc_rank(full[draw], sampled))
    if len(values) < len(draws) // 2:
        raise ValueError("too few valid paired AUC bootstrap draws")
    return {"difference_removed_minus_full": float(auc_rank(candidate, labels) - auc_rank(full, labels)),
            "patient_cluster_200_draw_95_interval": np.quantile(values, (.025, .975)).tolist()}


def _summary_for_cause(times: np.ndarray, events: np.ndarray, cif: np.ndarray,
                       cause: int) -> dict[str, object]:
    coded_events = np.where(events == cause, 1, np.where(events == 0, 0, 2))
    score = metrics(times, coded_events, cif[:, :, cause])
    observed_curve = ((events[:, None] == cause) & (times[:, None] <= GRID[None, :]))
    population_curve_gap = np.abs(cif[:, 7:, cause].mean(axis=0) - observed_curve[:, 7:].mean(axis=0))
    points = {}
    for day in POINTS:
        label = ((events == cause) & (times <= day))
        predicted = cif[:, day, cause]
        points[str(day)] = {"auc": score["points"][str(day)]["auc"],
                            "brier": score["points"][str(day)]["brier"],
                            "observed_fraction": float(label.mean()),
                            "mean_predicted": float(predicted.mean()),
                            "absolute_mean_calibration_gap": float(abs(predicted.mean() - label.mean()))}
    dca = {}
    if cause == 1:
        for day, threshold in ((30, .05), (30, .10), (90, .10), (90, .20)):
            predicted_positive = cif[:, day, cause] >= threshold
            actual_positive = (events == cause) & (times <= day)
            dca[f"{day}d_{threshold:.2f}"] = float(np.mean(predicted_positive & actual_positive)
                                                   - np.mean(predicted_positive & ~actual_positive)
                                                   * threshold / (1 - threshold))
    return {"mean_auc_7_365": score["mean_auc_7_365"],
            "integrated_brier_7_365": score["integrated_brier_7_365"],
            "mean_absolute_population_curve_gap_7_365": float(population_curve_gap.mean()),
            "c_index_90": cif_concordance(times, events, cif[:, 90, cause], cause=cause, horizon=90)["c_index"],
            "c_index_365": cif_concordance(times, events, cif[:, 365, cause], cause=cause, horizon=365)["c_index"],
            "points": points, "exploratory_dca_net_benefit": dca}


def run(root: Path, output: Path, *, draws: int = 200) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    arms = sorted(path for path in root.iterdir() if path.is_dir() and (path / "report.json").is_file())
    if not arms:
        raise ValueError("no completed ablation runs")
    grouped: dict[int, dict[str, Path]] = {}
    for path in arms:
        report = _read_run(path)
        seed = int(report["seed"])
        name = path.name.removesuffix(f"_seed{seed}")
        if name == path.name or name in grouped.setdefault(seed, {}):
            raise ValueError("invalid or repeated run name")
        grouped[seed][name] = path
    result: dict[str, object] = {
        "status": "completed", "comparison": "ablated-from-scratch minus full; same seed, patients, endpoint and input sources",
        "bootstrap_patient_cluster_draws": draws,
        "decision_rule": "jointly review readmission and death Brier, AUC, C-index and calibration; no single-metric automatic retention",
        "limitations": ["Bootstrap intervals only describe patient sampling within one trained seed.",
                        "The historical test has been viewed before and is not a new blind test.",
                        "Current drug categories are exposure proxies, not treatment efficacy or indications."],
        "seeds": {},
    }
    flat = []
    for seed, names in sorted(grouped.items()):
        if "full" not in names:
            raise ValueError(f"full reference missing for seed {seed}")
        full_dir = names["full"]
        full_report = _read_run(full_dir)
        references = {split: _predictions(full_dir, split)
                      for split in ("validation", "historical_test")}
        reference_scores = {
            (split, cause): _summary_for_cause(reference["times"], reference["events"],
                                              reference["competing_cif"], cause)
            for split, reference in references.items() for cause in (1, 2)
        }
        result["seeds"][str(seed)] = {}
        for arm, directory in sorted(names.items()):
            if arm == "full":
                continue
            report = _read_run(directory)
            for key in CONTRACT_KEYS:
                if report[key] != full_report[key]:
                    raise ValueError(f"comparison contract differs for {arm}: {key}")
            result["seeds"][str(seed)][arm] = {}
            for split in ("validation", "historical_test"):
                full = references[split]
                reduced = _predictions(directory, split)
                for key in ("subject_ids", "hadm_ids", "times", "events"):
                    if not np.array_equal(full[key], reduced[key]):
                        raise ValueError(f"paired sample mismatch in {arm}: {key}")
                times, events, subjects = full["times"], full["events"], full["subject_ids"]
                full_cif, reduced_cif = full["competing_cif"], reduced["competing_cif"]
                split_result = {}
                bootstrap = _patient_draws(subjects, draws, seed + 100) if split == "validation" else []
                for cause, name in ((1, "readmission"), (2, "death")):
                    full_score = reference_scores[(split, cause)]
                    reduced_score = _summary_for_cause(times, events, reduced_cif, cause)
                    paired = {"full": full_score, "removed": reduced_score,
                              "delta_removed_minus_full": {
                                  "integrated_brier_7_365": reduced_score["integrated_brier_7_365"] - full_score["integrated_brier_7_365"],
                                  "mean_auc_7_365": reduced_score["mean_auc_7_365"] - full_score["mean_auc_7_365"],
                                  "c_index_90": reduced_score["c_index_90"] - full_score["c_index_90"],
                                  "c_index_365": reduced_score["c_index_365"] - full_score["c_index_365"],
                                  "mean_absolute_population_curve_gap_7_365":
                                      reduced_score["mean_absolute_population_curve_gap_7_365"]
                                      - full_score["mean_absolute_population_curve_gap_7_365"],
                                  "absolute_mean_calibration_gap_90": reduced_score["points"]["90"]["absolute_mean_calibration_gap"] - full_score["points"]["90"]["absolute_mean_calibration_gap"],
                              }}
                    if cause == 1:
                        paired["delta_removed_minus_full"]["exploratory_dca_30d_0.10"] = (
                            reduced_score["exploratory_dca_net_benefit"]["30d_0.10"]
                            - full_score["exploratory_dca_net_benefit"]["30d_0.10"])
                        paired["delta_removed_minus_full"]["exploratory_dca_90d_0.20"] = (
                            reduced_score["exploratory_dca_net_benefit"]["90d_0.20"]
                            - full_score["exploratory_dca_net_benefit"]["90d_0.20"])
                    if split == "validation":
                        labels = ((events[:, None] == cause) & (times[:, None] <= GRID[None, :])).astype(float)
                        paired["brier_interval"] = _paired_brier(subjects, labels, reduced_cif[:, :, cause],
                                                                 full_cif[:, :, cause], bootstrap)
                        paired["auc_90_interval"] = _paired_auc(labels[:, 90].astype(bool),
                                                                  reduced_cif[:, 90, cause], full_cif[:, 90, cause], bootstrap)
                    split_result[name] = paired
                    flat.append({"seed": seed, "arm": arm, "split": split, "cause": name,
                                 **paired["delta_removed_minus_full"]})
                result["seeds"][str(seed)][arm][split] = split_result
    if not flat:
        raise ValueError("no completed ablation arm to compare with full")
    output.mkdir(parents=True)
    (output / "paired_group_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "paired_group_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    if not 20 <= args.bootstrap <= 500:
        raise ValueError("bootstrap draws outside supported range")
    result = run(args.run_root, args.output, draws=args.bootstrap)
    print(json.dumps({"status": result["status"], "seeds": list(result["seeds"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/compare_strict_ablations.py · lines 1–92

<!-- REVIEW_FRAGMENT {"path":"training/v5/compare_strict_ablations.py","part":1,"start_line":1,"end_line":92,"source_sha256":"53e422a876bd85ea8cd28313772f696ae6f85d79dee6bf313e3c79592ffad550","fragment_sha256":"53e422a876bd85ea8cd28313772f696ae6f85d79dee6bf313e3c79592ffad550","added_terminal_newline":false} -->
````python
"""对同患者严格结局实验做配对比较，不混用历史队列或评价口径。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.baselines.rank_metrics import auc_rank
from training.v5.train import clustered_brier_interval


def _load(run_dir: Path, split: str) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "training_completed":
        raise ValueError(f"incomplete run: {run_dir}")
    with np.load(run_dir / f"{split}_predictions.npz") as archive:
        predictions = {name: archive[name] for name in archive.files}
    return report, predictions


def _paired_auc_interval(subjects: np.ndarray, times: np.ndarray, events: np.ndarray,
                         full: np.ndarray, comparator: np.ndarray, day: int,
                         *, seed: int = 2026, draws: int = 300) -> dict[str, object]:
    observed = ((events == 1) & (times <= day)).astype(np.int8)
    if min(int(observed.sum()), int(len(observed) - observed.sum())) < 20:
        return {"point": None, "patient_cluster_95pct_interval": None}
    point = float(auc_rank(full[:, day], observed) - auc_rank(comparator[:, day], observed))
    patients, inverse = np.unique(subjects, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(len(patients))]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected = np.concatenate([groups[index] for index in rng.integers(len(groups), size=len(groups))])
        labels = observed[selected]
        if min(int(labels.sum()), int(len(labels) - labels.sum())) < 20:
            continue
        samples.append(float(auc_rank(full[selected, day], labels) -
                             auc_rank(comparator[selected, day], labels)))
    if len(samples) < 0.9 * draws:
        raise ValueError("too few valid patient bootstrap draws")
    return {"point": point, "patient_cluster_95pct_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
            "valid_draws": len(samples)}


def run(candidate_dir: Path, comparators: dict[str, Path], output_path: Path) -> dict[str, object]:
    result: dict[str, object] = {"status": "paired_internal_comparison", "candidate": str(candidate_dir), "comparisons": {}}
    candidate_report = json.loads((candidate_dir / "report.json").read_text(encoding="utf-8"))
    for name, run_dir in comparators.items():
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        for key in ("label_sha256", "field_manifest_sha256", "raw_source_sha256", "text_vector_width",
                    "event_concept_count", "cohort_admissions", "cohort_patients", "primary_event",
                    "seed", "learning_rate", "weight_decay", "patient_batch_size", "selection",
                    "parameter_count"):
            if candidate_report[key] != report[key]:
                raise ValueError(f"{name}: mismatched {key}")
        result["comparisons"][name] = {"run_dir": str(run_dir), "splits": {}}
        for split in ("validation", "historical_test"):
            _, full = _load(candidate_dir, split)
            _, other = _load(run_dir, split)
            for key in ("subject_ids", "hadm_ids", "times", "events"):
                if not np.array_equal(full[key], other[key]):
                    raise ValueError(f"{name}/{split}: patient labels or order differ on {key}")
            subjects, times, events = full["subject_ids"], full["times"], full["events"]
            result["comparisons"][name]["splits"][split] = {
                "ibs_v5_minus_comparator_patient_cluster_95pct_interval": clustered_brier_interval(
                    subjects, times, events, full["risk"], other["risk"]),
                "auc_v5_minus_comparator": {
                    str(day): _paired_auc_interval(subjects, times, events, full["risk"], other["risk"], day)
                    for day in (30, 90, 365)
                },
            }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--comparator", nargs=2, action="append", metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    names = [pair[0] for pair in args.comparator]
    if len(names) != len(set(names)) or args.output_path.exists():
        raise ValueError("comparison names must be unique and output must not exist")
    run(args.candidate_dir, {name: Path(path) for name, path in args.comparator}, args.output_path)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/compare_strict_axis_screen.py · lines 1–157

<!-- REVIEW_FRAGMENT {"path":"training/v5/compare_strict_axis_screen.py","part":1,"start_line":1,"end_line":157,"source_sha256":"9d8b73ba179dfda143cfd78c1ee09bd85f33566777a6b1d53e9ad2539d77c20d","fragment_sha256":"9d8b73ba179dfda143cfd78c1ee09bd85f33566777a6b1d53e9ad2539d77c20d","added_terminal_newline":false} -->
````python
"""Compare clinical preselection and single-axis removals under the same strict endpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from training.v5.compare_revised_group_ablations import EXTRA_CONTRACT
from training.v5.compare_screened_group_ablations import (
    CONTRACT_KEYS, _paired_auc, _paired_brier, _patient_draws,
    _predictions, _summary_for_cause,
)
from training.v5.lab_recovery_data import SELECTED_AXIS_SCREEN
from training.v5.strict_endpoint_labels import STRICT_PRIMARY_EVENT
from training.v5.train import GRID

SPLITS = ("validation", "historical_test")
CAUSES = ((1, "readmission"), (2, "death"))


def _report(path: Path, *, lab_mode: str, medication_mode: str) -> dict:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    if (report["status"] != "training_completed"
            or report["event_encoder"] != "abnormal_medication"
            or report["primary_event"] != STRICT_PRIMARY_EVENT
            or report["clinical_recovery_mode"] != lab_mode
            or report["medication_effect_mode"] != medication_mode
            or report["raw_events_enter_model"]):
        raise ValueError(f"Incorrect strict endpoint/input run: {path}")
    return report


def _arm(root: Path, arm: str, seed: int) -> tuple[Path, str, str]:
    if arm == "full":
        return root / f"full_seed{seed}", "active", "active"
    if arm == "selected":
        return root / f"selected_seed{seed}", "without_renal_metabolic_perfusion", "without_diuretic"
    if arm in SELECTED_AXIS_SCREEN:
        return (root / f"without_axis_{arm}_seed{seed}",
                f"without_renal_metabolic_perfusion_plus_{arm}", "without_diuretic")
    raise ValueError(f"Unknown arm: {arm}")


def compare(root: Path, *, seeds: tuple[int, ...], axes: tuple[str, ...],
            draws: int = 200) -> list[dict]:
    if not seeds or any(axis not in SELECTED_AXIS_SCREEN for axis in axes):
        raise ValueError("Invalid seeds or screened axes")
    rows = []
    comparisons = (("full", "selected"), *(("selected", axis) for axis in axes))
    for seed in seeds:
        cache: dict[str, tuple[dict, dict[str, dict[str, np.ndarray]]]] = {}
        for reference_name, candidate_name in comparisons:
            for arm_name in (reference_name, candidate_name):
                if arm_name not in cache:
                    directory, lab_mode, med_mode = _arm(root, arm_name, seed)
                    cache[arm_name] = (_report(directory, lab_mode=lab_mode, medication_mode=med_mode),
                                       {split: _predictions(directory, split) for split in SPLITS})
            reference_report, reference_predictions = cache[reference_name]
            candidate_report, candidate_predictions = cache[candidate_name]
            for key in (*CONTRACT_KEYS, *EXTRA_CONTRACT):
                if reference_report[key] != candidate_report[key]:
                    raise ValueError(f"Unpaired {reference_name}/{candidate_name} seed={seed}: {key}")
            for split in SPLITS:
                reference = reference_predictions[split]
                candidate = candidate_predictions[split]
                for key in ("subject_ids", "hadm_ids", "times", "events"):
                    if not np.array_equal(reference[key], candidate[key]):
                        raise ValueError(f"Unpaired {key}: {reference_name}/{candidate_name}, {seed}, {split}")
                times, events, subjects = reference["times"], reference["events"], reference["subject_ids"]
                bootstrap = _patient_draws(subjects, draws, seed + 100) if split == "validation" else None
                for cause, cause_name in CAUSES:
                    ref = _summary_for_cause(times, events, reference["competing_cif"], cause)
                    new = _summary_for_cause(times, events, candidate["competing_cif"], cause)
                    row = {
                        "seed": seed, "reference": reference_name, "candidate": candidate_name,
                        "split": split, "cause": cause_name,
                        "reference_brier": ref["integrated_brier_7_365"],
                        "candidate_brier": new["integrated_brier_7_365"],
                        "delta_brier": new["integrated_brier_7_365"] - ref["integrated_brier_7_365"],
                        "reference_auc": ref["mean_auc_7_365"],
                        "candidate_auc": new["mean_auc_7_365"],
                        "delta_auc": new["mean_auc_7_365"] - ref["mean_auc_7_365"],
                        "delta_c_index_365": new["c_index_365"] - ref["c_index_365"],
                        "delta_calibration_gap_90": new["points"]["90"]["absolute_mean_calibration_gap"]
                        - ref["points"]["90"]["absolute_mean_calibration_gap"],
                    }
                    if cause == 1:
                        for key in ("30d_0.10", "90d_0.20"):
                            row[f"delta_dca_net_benefit_{key}"] = (
                                new["exploratory_dca_net_benefit"][key]
                                - ref["exploratory_dca_net_benefit"][key]
                            )
                    if bootstrap is not None:
                        labels = ((events[:, None] == cause) & (times[:, None] <= GRID[None, :])).astype(float)
                        row["brier_95_interval"] = _paired_brier(
                            subjects, labels, candidate["competing_cif"][:, :, cause],
                            reference["competing_cif"][:, :, cause], bootstrap,
                        )["patient_cluster_200_draw_95_interval"]
                        row["auc_90_95_interval"] = _paired_auc(
                            labels[:, 90].astype(bool), candidate["competing_cif"][:, 90, cause],
                            reference["competing_cif"][:, 90, cause], bootstrap,
                        )["patient_cluster_200_draw_95_interval"]
                    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--axes", nargs="+", choices=SELECTED_AXIS_SCREEN, default=list(SELECTED_AXIS_SCREEN))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows = compare(args.run_root, seeds=tuple(args.seeds), axes=tuple(args.axes))
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "paired_results.json").write_text(json.dumps({
        "direction": "candidate minus reference; lower Brier and higher AUC are better",
        "endpoint": STRICT_PRIMARY_EVENT,
        "bootstrap": "200 validation patient-cluster draws per seed; not retraining uncertainty",
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "paired_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0]
                                                  if key not in ("brier_95_interval", "auc_90_95_interval")],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for reference, candidate in (("full", "selected"), *(("selected", axis) for axis in args.axes)):
        for split in SPLITS:
            for _, cause_name in CAUSES:
                matched = [row for row in rows if row["reference"] == reference and row["candidate"] == candidate
                           and row["split"] == split and row["cause"] == cause_name]
                summary.append({
                    "reference": reference, "candidate": candidate, "split": split, "cause": cause_name,
                    "seeds": len(matched),
                    "mean_delta_brier": float(np.mean([r["delta_brier"] for r in matched])),
                    "mean_delta_auc": float(np.mean([r["delta_auc"] for r in matched])),
                    "mean_delta_c_index_365": float(np.mean([r["delta_c_index_365"] for r in matched])),
                    "brier_improved_seeds": sum(r["delta_brier"] < 0 for r in matched),
                    "auc_improved_seeds": sum(r["delta_auc"] > 0 for r in matched),
                })
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps({"rows": len(rows), "seeds": args.seeds, "axes": args.axes}))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/compare_summary_temporal.py · lines 1–170

<!-- REVIEW_FRAGMENT {"path":"training/v5/compare_summary_temporal.py","part":1,"start_line":1,"end_line":170,"source_sha256":"930f3bbc9b66a290dfff7641d08e6c406ba2c61192b2058ca080a7e81977ec1e","fragment_sha256":"930f3bbc9b66a290dfff7641d08e6c406ba2c61192b2058ca080a7e81977ec1e","added_terminal_newline":false} -->
````python
"""在相同患者、字段文本及结局下配对比较住院摘要与原始事件时序。"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from training.v5.event_data import load_event_data
from training.v5.event_model import TemporalFusionModel
from training.v5.event_set_model import EventSetFusionModel
from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_data import load_data, normalize_on_training_patients, split_sequences
from training.v5.independent_fusion import IndependentFusionModel
from training.v5.train import clustered_brier_interval, metrics
from training.v5.train_event_sequence import predict as predict_temporal
from training.v5.train_independent import predict as predict_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned_dir", "structured_dir", "base_csv", "raw_dir", "field_dir",
                 "vectors", "labels_csv", "summary_run", "temporal_run", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--gru-run", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("paired comparison requires CUDA")
    summary_data, _ = normalize_on_training_patients(load_data(
        args.aligned_dir, args.structured_dir, args.base_csv,
        args.field_dir, args.vectors, labels_csv=args.labels_csv,
    ))
    event_data, _ = load_event_data(
        args.aligned_dir, args.structured_dir, args.base_csv, args.raw_dir,
        labels_csv=args.labels_csv, field_dir=args.field_dir, vectors_path=args.vectors,
    )
    summary_report = json.loads((args.summary_run / "report.json").read_text(encoding="utf-8"))
    temporal_report = json.loads((args.temporal_run / "report.json").read_text(encoding="utf-8"))
    if (not np.array_equal(summary_data.hadm_ids, event_data.base.hadm_ids)
            or summary_data.label_sha256 != event_data.base.label_sha256
            or summary_report["label_sha256"] != summary_data.label_sha256
            or temporal_report["label_sha256"] != summary_data.label_sha256
            or summary_report["field_manifest_sha256"] != temporal_report["field_manifest_sha256"]
            or temporal_report["raw_source_sha256"] != event_data.source_hashes):
        raise ValueError("paired comparison cohort, label or source mismatch")
    device = torch.device("cuda")
    summary_model = IndependentFusionModel(
        concept_count=len(summary_data.concept_names), type_count=len(summary_data.type_names),
        field_count=len(FIELD_LABELS), text_width=summary_data.text_width,
    ).to(device)
    summary_model.load_state_dict(torch.load(
        args.summary_run / "structured_field_bert.pt", map_location=device, weights_only=True,
    ))
    model_class = EventSetFusionModel if temporal_report.get("event_encoder") == "set" else TemporalFusionModel
    event_model = model_class(
        concept_count=len(event_data.base.concept_names),
        type_count=len(event_data.base.type_names),
        field_count=len(FIELD_LABELS), text_width=event_data.base.text_width,
    ).to(device)
    event_model.load_state_dict(torch.load(
        args.temporal_run / "model.pt", map_location=device, weights_only=True,
    ))
    gru_model = None
    if args.gru_run is not None:
        reference = json.loads((args.gru_run / "report.json").read_text(encoding="utf-8"))
        for key in ("label_sha256", "field_manifest_sha256", "raw_source_sha256"):
            if reference[key] != temporal_report[key]:
                raise ValueError(f"GRU comparison differs: {key}")
        gru_model = TemporalFusionModel(
            concept_count=len(event_data.base.concept_names),
            type_count=len(event_data.base.type_names), field_count=len(FIELD_LABELS),
            text_width=event_data.base.text_width,
        ).to(device)
        gru_model.load_state_dict(torch.load(args.gru_run / "model.pt", map_location=device,
                                             weights_only=True))
    # 仅反转同时间戳的完整观测行，不破坏项目、数值和时间的绑定。
    permutation = np.arange(len(event_data.event_concepts))
    tied_rows = 0
    for refs in event_data.events_by_admission:
        if len(refs) < 2:
            continue
        for time in np.unique(event_data.event_values[refs, 1]):
            tied = refs[event_data.event_values[refs, 1] == time]
            if len(tied) > 1:
                permutation[tied] = tied[::-1]
                tied_rows += len(tied)
    reordered_data = replace(event_data,
                             event_values=event_data.event_values[permutation],
                             event_concepts=event_data.event_concepts[permutation],
                             event_is_lab=event_data.event_is_lab[permutation])
    report: dict[str, object] = {
        "primary_event": summary_data.label_name,
        "label_sha256": summary_data.label_sha256,
        "field_manifest_sha256": summary_data.manifest_sha256,
        "raw_source_sha256": event_data.source_hashes,
        "event_encoder": temporal_report.get("event_encoder", "gru"),
        "same_timestamp_reorder_total_rows": tied_rows,
        "comparison": "same outcome, cohort, patient split and frozen BERT text; validation-selected checkpoints",
    }
    for split in ("validation", "test"):
        sequences = split_sequences(summary_data, split)
        indices = np.concatenate(sequences)
        times, events = summary_data.times[indices], summary_data.events[indices]
        risks = {
            "summary": predict_summary(summary_model, summary_data, sequences,
                                       include_text=True, batch_size=24, device=device)[0],
            "temporal": predict_temporal(event_model, event_data, sequences,
                                         process_only=False, batch_size=12, device=device)[0],
            "lab_order_shuffled": predict_temporal(
                event_model, event_data, sequences, process_only=False,
                batch_size=12, device=device, shuffle_lab_order_seed=2026,
            )[0],
            "same_timestamp_reordered": predict_temporal(
                event_model, reordered_data, sequences, process_only=False,
                batch_size=12, device=device,
            )[0],
            "lab_values_zeroed": predict_temporal(
                event_model, event_data, sequences, process_only=True,
                batch_size=12, device=device,
            )[0],
        }
        if gru_model is not None:
            risks["gru"] = predict_temporal(gru_model, event_data, sequences,
                                             process_only=False, batch_size=12, device=device)[0]
        scores = {name: metrics(times, events, risk) for name, risk in risks.items()}
        paired = {}
        comparisons = [
            ("temporal_minus_summary", "temporal", "summary"),
            ("lab_order_shuffled_minus_temporal", "lab_order_shuffled", "temporal"),
            ("lab_values_zeroed_minus_temporal", "lab_values_zeroed", "temporal"),
        ]
        if gru_model is not None:
            comparisons.append(("temporal_minus_gru", "temporal", "gru"))
        for name, first, second in comparisons:
            paired[name] = {
                "estimate": float(scores[first]["integrated_brier_7_365"]
                                  - scores[second]["integrated_brier_7_365"]),
                "patient_cluster_95ci": clustered_brier_interval(
                    summary_data.subject_ids[indices], times, events,
                    risks[first], risks[second],
                ),
            }
        report["historical_test" if split == "test" else split] = {
            "lab_values_zeroed_max_abs_risk_change": float(np.max(np.abs(
                risks["lab_values_zeroed"] - risks["temporal"]))),
            "same_timestamp_max_abs_risk_change": float(np.max(np.abs(
                risks["same_timestamp_reordered"] - risks["temporal"]))),
            "same_timestamp_invariance_pass": bool(np.allclose(
                risks["same_timestamp_reordered"], risks["temporal"], atol=1e-5, rtol=1e-5)),
            "admissions": len(indices), "patients": len(sequences),
            "models": {name: {"readmissions": detail["readmissions"],
                              "deaths": detail["deaths"],
                              "integrated_brier_7_365": detail["integrated_brier_7_365"],
                              "mean_auc_7_365": detail["mean_auc_7_365"]}
                       for name, detail in scores.items()},
            "paired_brier_difference_95ci": paired,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/compare_text_encoders.py · lines 1–141

<!-- REVIEW_FRAGMENT {"path":"training/v5/compare_text_encoders.py","part":1,"start_line":1,"end_line":141,"source_sha256":"dedabd647829b2725544dea2b88b9af8f58ba39dd185197ad937549e9088d4b1","fragment_sha256":"dedabd647829b2725544dea2b88b9af8f58ba39dd185197ad937549e9088d4b1","added_terminal_newline":false} -->
````python
"""在同一 V5 患者划分上配对比较冻结 BERT 和 Qwen 文本编码。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.v5.data import load_data
from training.v5.model import V5Model
from training.v5.train import clustered_brier_interval, metrics, predict, preprocess, split_sequences


def _risk(structured: Path, manifest: Path, vectors: Path, run_dir: Path,
          model_name: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
    training_report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    data, _ = preprocess(load_data(structured, manifest, vectors,
                                   vector_width=int(training_report.get("text_input_width", 768))))
    sequences = split_sequences(data, split)
    selected = np.concatenate(sequences)
    model = V5Model(text_width=data.text_width,
                    balance_text=bool(training_report.get("balanced_text_content", False))).to("cuda")
    model.load_state_dict(torch.load(run_dir / f"{model_name}.pt", map_location="cuda",
                                     weights_only=True))
    risk, _ = predict(model, data, sequences, include_text=True,
                      device=torch.device("cuda"), batch_size=24)
    return data.hadm_ids[selected], data.subject_ids[selected], data.times[selected], data.events[selected], risk


def run(structured: Path, manifest: Path, experiments: dict[str, tuple[Path, Path, str]],
        output: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint comparison requires CUDA")
    if output.exists():
        raise FileExistsError(output)
    report: dict[str, object] = {"primary_metric": "World A competing-risk IBS days 7-365",
                                 "patient_paired_bootstrap": 300, "splits": {}}
    for split in ("validation", "test"):
        predictions = {name: _risk(structured, manifest, vectors, directory, model_name, split)
                       for name, (vectors, directory, model_name) in experiments.items()}
        reference = predictions["bert"]
        summary: dict[str, object] = {"models": {}, "paired_differences_vs_bert": {}}
        for name, values in predictions.items():
            if not all(np.array_equal(a, b) for a, b in zip(reference[:4], values[:4], strict=True)):
                raise ValueError("encoder comparison rows or labels do not align")
            measured = metrics(values[2], values[3], values[4])
            summary["models"][name] = {
                "integrated_brier_7_365": measured["integrated_brier_7_365"],
                "mean_auc_7_365": measured["mean_auc_7_365"],
                "auc_365": measured["points"]["365"]["auc"],
                "mean_risk_365": measured["points"]["365"]["mean_predicted_risk"],
            }
            if name != "bert":
                difference = (float(measured["integrated_brier_7_365"])
                              - float(summary["models"]["bert"]["integrated_brier_7_365"]))
                summary["paired_differences_vs_bert"][name] = {
                    "brier_difference": difference,
                    "patient_cluster_95ci": clustered_brier_interval(
                        values[1], values[2], values[3], values[4], reference[4]
                    ),
                }
        narrow = predictions["qwen768"]
        full = predictions["qwen2560"]
        summary["qwen2560_minus_qwen768"] = {
            "brier_difference": (
                float(summary["models"]["qwen2560"]["integrated_brier_7_365"])
                - float(summary["models"]["qwen768"]["integrated_brier_7_365"])
            ),
            "patient_cluster_95ci": clustered_brier_interval(
                full[1], full[2], full[3], full[4], narrow[4]
            ),
        }
        if "qwen128" in predictions:
            smallest = predictions["qwen128"]
            summary["qwen768_minus_qwen128"] = {
                "brier_difference": (
                    float(summary["models"]["qwen768"]["integrated_brier_7_365"])
                    - float(summary["models"]["qwen128"]["integrated_brier_7_365"])
                ),
                "patient_cluster_95ci": clustered_brier_interval(
                    narrow[1], narrow[2], narrow[3], narrow[4], smallest[4]
                ),
            }
        for baseline, balanced in (("bert", "bert_balanced"),
                                   ("qwen2560", "qwen2560_balanced")):
            if balanced not in predictions:
                continue
            original, normalized = predictions[baseline], predictions[balanced]
            summary[f"{balanced}_minus_{baseline}"] = {
                "brier_difference": (
                    float(summary["models"][balanced]["integrated_brier_7_365"])
                    - float(summary["models"][baseline]["integrated_brier_7_365"])
                ),
                "patient_cluster_95ci": clustered_brier_interval(
                    normalized[1], normalized[2], normalized[3], normalized[4], original[4]
                ),
            }
        report["splits"]["historical_test" if split == "test" else split] = summary
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bert-vectors", type=Path, required=True)
    parser.add_argument("--bert-run", type=Path, required=True)
    parser.add_argument("--qwen-vectors", type=Path, required=True)
    parser.add_argument("--qwen768-run", type=Path, required=True)
    parser.add_argument("--qwen2560-run", type=Path, required=True)
    parser.add_argument("--qwen128-run", type=Path)
    parser.add_argument("--bert-balanced-run", type=Path)
    parser.add_argument("--qwen2560-balanced-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiments = {
        "bert": (args.bert_vectors, args.bert_run, "structured_bert"),
        "qwen768": (args.qwen_vectors, args.qwen768_run, "structured_qwen768"),
        "qwen2560": (args.qwen_vectors, args.qwen2560_run, "structured_qwen2560"),
    }
    if args.qwen128_run is not None:
        experiments["qwen128"] = (args.qwen_vectors, args.qwen128_run, "structured_qwen128")
    if args.bert_balanced_run is not None:
        experiments["bert_balanced"] = (args.bert_vectors, args.bert_balanced_run,
                                         "structured_bert")
    if args.qwen2560_balanced_run is not None:
        experiments["qwen2560_balanced"] = (args.qwen_vectors,
                                             args.qwen2560_balanced_run,
                                             "structured_qwen2560")
    result = run(args.structured_dir, args.manifest, experiments, args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/competing_concordance.py · lines 1–41

<!-- REVIEW_FRAGMENT {"path":"training/v5/competing_concordance.py","part":1,"start_line":1,"end_line":41,"source_sha256":"1c1764227f44e39692acc1fdabe15482fc6a817951ece6c9e4764f086bd33d5b","fragment_sha256":"1c1764227f44e39692acc1fdabe15482fc6a817951ece6c9e4764f086bd33d5b","added_terminal_newline":false} -->
````python
"""World A 完整观察合同下、基于原因别累计发生概率的截断 C-index。"""

from __future__ import annotations

import numpy as np


def cif_concordance(times: np.ndarray, events: np.ndarray, predicted_cif: np.ndarray,
                    *, cause: int = 1, horizon: int = 365) -> dict[str, float | int]:
    """Wolbers 式可比对：目标事件较早 vs 目标事件较晚或竞争事件。

    无提前失访是此简化实现的必要条件。风险分数固定为 F_cause(horizon)，
    同时发生目标事件的配对不计入；同分风险给半分。重复出院的点估计按住院配对。
    """
    times = np.asarray(times)
    events = np.asarray(events)
    risk = np.asarray(predicted_cif)
    if not (len(times) == len(events) == len(risk)) or not len(times):
        raise ValueError("outcome/risk length mismatch")
    if cause not in (1, 2) or horizon < 1 or not np.all(np.isin(events, (0, 1, 2))):
        raise ValueError("invalid cause, horizon or event type")
    if np.any((events == 0) & (times < horizon)):
        raise ValueError("early censoring requires IPCW concordance")
    if not np.isfinite(risk).all() or np.any((risk < 0) | (risk > 1)):
        raise ValueError("invalid CIF risk")
    case_indices = np.flatnonzero((events == cause) & (times <= horizon))
    total, concordant = 0, 0.0
    for index in case_indices:
        # 竞争事件可早于目标事件；其后已不可能发生目标事件。
        comparable = (events != cause) | (times > times[index])
        comparable[index] = False
        if not np.any(comparable):
            continue
        difference = risk[index] - risk[comparable]
        total += len(difference)
        concordant += float(np.count_nonzero(difference > 1e-10))
        concordant += 0.5 * float(np.count_nonzero(np.abs(difference) <= 1e-10))
    if not total:
        raise ValueError("no comparable event pairs")
    return {"c_index": concordant / total, "target_events": int(len(case_indices)),
            "comparable_pairs": total, "horizon_days": horizon, "cause": cause}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/complete_c_index_addendum.py · lines 1–89

<!-- REVIEW_FRAGMENT {"path":"training/v5/complete_c_index_addendum.py","part":1,"start_line":1,"end_line":89,"source_sha256":"536bef15b11f446294157ab24e684019dac05d347cad5fb641aa336035797027","fragment_sha256":"536bef15b11f446294157ab24e684019dac05d347cad5fb641aa336035797027","added_terminal_newline":false} -->
````python
"""对已完成的 V5 检查点只读重放，补充竞争风险 C-index 和整条曲线检查。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.v5.competing_concordance import cif_concordance
from training.v5.evaluate_completed_event_run import replay
from training.v5.event_set_model import EventSetFusionModel
from training.v5.independent_data import split_sequences
from training.v5.processed_event_data import load_processed_event_data
from training.v5.train import metrics


def run(args: argparse.Namespace) -> dict[str, object]:
    target = args.output_dir / "time_ranking_addendum.json"
    if not args.output_dir.is_dir() or target.exists():
        raise ValueError("evaluation output missing or C-index addendum already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint replay requires CUDA")
    report = json.loads((args.run_dir / "report.json").read_text(encoding="utf-8"))
    data, _ = load_processed_event_data(
        args.aligned_dir, args.structured_dir, args.base_csv,
        args.processed_event_dir, args.raw_ledger_dir, labels_csv=args.labels_csv,
        field_dir=args.field_dir, vectors_path=args.vectors)
    if (report["cohort_admissions"] != len(data.base.hadm_ids)
            or report["label_sha256"] != data.base.label_sha256
            or report["field_manifest_sha256"] != data.base.manifest_sha256
            or report["event_concepts"] != list(data.concept_names)):
        raise ValueError("checkpoint cohort/input contract mismatch")
    model = EventSetFusionModel(
        concept_count=len(data.base.concept_names), type_count=len(data.base.type_names),
        field_count=len(data.base.field_roles), text_width=data.base.text_width,
        event_concept_count=len(data.concept_names)).to("cuda").eval()
    state = torch.load(args.run_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    result: dict[str, object] = {
        "status": "completed", "source_run": str(args.run_dir),
        "definition": "Wolbers-style cause-specific truncated concordance using CIF at the stated horizon; competing deaths remain comparators, not censored",
        "observation_contract": "World A: event_type=0 observed to day 365; no early censoring",
        "limitations": ["Repeated discharges contribute admission-level pairs; no patient-cluster C-index interval was calculated.",
                        "Historical test is not blind; outside-hospital readmission may be missed."],
        "splits": {},
    }
    for split, key in (("validation", "validation"), ("test", "historical_test")):
        sequences = split_sequences(data.base, split)
        indices = np.concatenate(sequences)
        curves = replay(model, data, sequences, torch.device("cuda"))
        reference = report[key]["temporal_full"]
        replayed = metrics(data.base.times[indices], data.base.events[indices], curves[:, :, 1])
        for metric in ("integrated_brier_7_365", "mean_auc_7_365"):
            if not np.isclose(replayed[metric], reference[metric], atol=1e-7):
                raise ValueError("checkpoint replay metrics do not match original report")
        times, events = data.base.times[indices], data.base.events[indices]
        split_result = {"n_admissions": len(indices), "readmission": {}, "death": {}}
        for cause, name in ((1, "readmission"), (2, "death")):
            cifs = curves[:, :, cause]
            if np.any(np.diff(cifs, axis=1) < -1e-6):
                raise ValueError("predicted CIF curve is not monotone")
            for day in (90, 365):
                split_result[name][str(day)] = cif_concordance(times, events, cifs[:, day],
                                                               cause=cause, horizon=day)
            days = np.arange(7, 366)
            observed = np.array([np.mean((events == cause) & (times <= day)) for day in days])
            predicted = cifs[:, 7:366].mean(axis=0)
            split_result[name]["mean_absolute_population_curve_gap_7_365"] = float(np.mean(np.abs(predicted - observed)))
            split_result[name]["maximum_absolute_population_curve_gap_7_365"] = float(np.max(np.abs(predicted - observed)))
        result["splits"][key] = split_result
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run_dir", "aligned_dir", "structured_dir", "base_csv", "processed_event_dir",
                 "raw_ledger_dir", "field_dir", "vectors", "labels_csv", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps({"status": run(args)["status"], "output": str(args.output_dir / "time_ranking_addendum.json")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/data.py · lines 1–272

<!-- REVIEW_FRAGMENT {"path":"training/v5/data.py","part":1,"start_line":1,"end_line":272,"source_sha256":"cb5f32751f59cf7ad6cfb32769bfb9b2ffa376ad20a1ce5a87ec18a0c9ace4fa","fragment_sha256":"cb5f32751f59cf7ad6cfb32769bfb9b2ffa376ad20a1ce5a87ec18a0c9ace4fa","added_terminal_newline":false} -->
````python
"""将出院预测样本集转为六组结构化输入与完整患者序列。"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from training.v5.clinical_concepts import LABS, MED_NAMES, diagnosis_groups, medication_classes


LAB_ORDER = tuple(LABS)
GROUP_NAMES = ("baseline", "chronic_comorbidity", "current_diagnosis",
               "blood_gas", "other_labs", "treatment_and_discharge")
FIELD_NAMES = ("residual_dyspnea", "residual_cough_sputum", "oxygen_or_niv_at_discharge",
               "mobility_at_discharge", "inhaler_use_or_adherence", "social_support",
               "steroid_or_antibiotic_discharge_plan", "unresolved_infection_or_imaging")
GROUP_WIDTHS = (4, 6, 5, 15, 25, 16)

def patient_split(subject_id: int) -> str:
    digest = hashlib.md5(f"aecopd-model-v1:{subject_id}".encode("ascii")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


@dataclass(frozen=True)
class V5Data:
    subject_ids: np.ndarray
    hadm_ids: np.ndarray
    splits: np.ndarray
    times: np.ndarray
    events: np.ndarray
    group_values: tuple[np.ndarray, ...]
    group_masks: tuple[np.ndarray, ...]
    gap_features: np.ndarray
    sequences: tuple[np.ndarray, ...]
    snippets: tuple[tuple[tuple[int, int], ...], ...]
    text_vectors: np.ndarray | None
    text_width: int
    coverage: dict[str, object]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_data(structured_dir: Path, manifest: Path | None = None,
              vectors_path: Path | None = None, *, vector_width: int | None = None) -> V5Data:
    """严格按住院键连接，保持每位患者住院时间序列并隔离患者划分。"""

    base = _rows(structured_dir / "base.csv")
    if len(base) != 19401:
        raise ValueError("V5 requires the full discharge prediction sample set")
    hadm_ids = np.array([int(row["hadm_id"]) for row in base], dtype=np.int64)
    subject_ids = np.array([int(row["subject_id"]) for row in base], dtype=np.int64)
    if len(set(hadm_ids.tolist())) != len(base):
        raise ValueError("duplicate admission")
    row_by_hadm = {int(hadm): index for index, hadm in enumerate(hadm_ids)}
    splits = np.array([patient_split(int(subject)) for subject in subject_ids])
    times = np.array([float(row["time_days"]) for row in base], dtype=np.float32)
    events = np.array([int(row["event_type"]) for row in base], dtype=np.int64)
    if not np.isfinite(times).all() or np.any((times < 0) | (times > 365)):
        raise ValueError("invalid event time")
    if not np.isin(events, (0, 1, 2)).all() or np.any((events == 0) & (times != 365)):
        raise ValueError("invalid event type or administrative censoring")
    values = tuple(np.zeros((len(base), width), dtype=np.float32) for width in GROUP_WIDTHS)
    masks = tuple(np.zeros_like(group) for group in values)
    gap = np.zeros((len(base), 2), dtype=np.float32)
    patient_rows: dict[int, list[int]] = defaultdict(list)
    previous_discharge: dict[int, datetime] = {}
    for index, row in enumerate(base):
        subject = int(row["subject_id"])
        admission, discharge = _time(row["admittime"]), _time(row["dischtime"])
        if discharge <= admission:
            raise ValueError("ineligible discharge in V5 base")
        if patient_rows[subject] and admission < _time(base[patient_rows[subject][-1]]["admittime"]):
            raise ValueError("patient admissions are not chronologically ordered")
        patient_rows[subject].append(index)
        previous = previous_discharge.get(subject)
        if previous is not None:
            days = max(0.0, (admission - previous).total_seconds() / 86400)
            gap[index] = (math.log1p(days), float(admission < previous))
        previous_discharge[subject] = discharge
        values[0][index] = (
            float(row["age_at_admission"]), float(row["gender"] == "M"),
            math.log1p(int(row["prior_coded_admission_count"])),
            math.log1p(int(row["prior_strict_aecopd_count"])),
        )
        masks[0][index] = 1.0
        values[2][index, 0] = float(row["index_is_strict_aecopd"] in {"t", "true", "True", "1"})
        masks[2][index] = 1.0
        values[5][index, 0] = (discharge - admission).total_seconds() / 86400
        destination = row["discharge_location"].upper()
        values[5][index, 1:4] = (
            float("HOME" in destination),
            float(any(term in destination for term in ("REHAB", "SKILLED", "NURSING", "LONG TERM"))),
            float(bool(destination) and "HOME" not in destination
                  and not any(term in destination for term in ("REHAB", "SKILLED", "NURSING", "LONG TERM"))),
        )
        masks[5][index, :4] = 1.0
        masks[1][index] = 1.0
    lab_count = 0
    seen_labs: set[tuple[int, int]] = set()
    for row in _rows(structured_dir / "labs.csv"):
        hadm, itemid = int(row["hadm_id"]), int(row["itemid"])
        if hadm not in row_by_hadm or itemid not in LABS or (hadm, itemid) in seen_labs:
            raise ValueError("lab key mismatch or duplicate")
        seen_labs.add((hadm, itemid))
        index = row_by_hadm[hadm]
        group = LABS[itemid][1]
        position = (LAB_ORDER[:3] if group == 3 else LAB_ORDER[3:]).index(itemid) * 5
        first, last, minimum, maximum = (float(row[key]) for key in
                                          ("first_value", "last_value", "min_value", "max_value"))
        if not all(math.isfinite(number) and number >= 0 for number in (first, last, minimum, maximum)):
            raise ValueError("invalid lab value")
        values[group][index, position:position + 4] = first, last, minimum, maximum
        masks[group][index, position:position + 4] = 1.0
        if int(row["measurement_count"]) >= 2:
            values[group][index, position + 4] = last - first
            masks[group][index, position + 4] = 1.0
        lab_count += 1
    diagnosis_count = 0
    chronic = ("heart_failure", "cardiac_other", "ckd", "diabetes", "malignancy", "chronic_respiratory")
    acute = ("acute_resp_failure", "pneumonia_or_aspiration", "sepsis", "aki")
    for row in _rows(structured_dir / "diagnoses.csv"):
        hadm = int(row["hadm_id"])
        if hadm not in row_by_hadm:
            raise ValueError("diagnosis outside cohort")
        index = row_by_hadm[hadm]
        for concept in diagnosis_groups(row["icd_code"], int(row["icd_version"])):
            if concept in chronic:
                values[1][index, chronic.index(concept)] = 1.0
            else:
                values[2][index, 1 + acute.index(concept)] = 1.0
        diagnosis_count += 1
    medication_count = 0
    hadms_with_emar: set[int] = set()
    for row in _rows(structured_dir / "medications.csv"):
        hadm = int(row["hadm_id"])
        if hadm not in row_by_hadm:
            raise ValueError("eMAR medication outside cohort")
        index = row_by_hadm[hadm]
        hadms_with_emar.add(hadm)
        for concept in medication_classes(row["medication"]):
            values[5][index, 4 + MED_NAMES.index(concept)] = 1.0
        medication_count += 1
    for hadm in hadms_with_emar:
        masks[5][row_by_hadm[hadm], 4:] = 1.0
    snippets: list[list[tuple[int, int]]] = [[] for _ in base]
    text_vectors: np.ndarray | None = None
    text_width = 768
    text_encoder = "none"
    if manifest is not None or vectors_path is not None:
        if manifest is None or vectors_path is None:
            raise ValueError("manifest and text vectors must both be supplied")
        state_path = vectors_path.parent / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") != "complete" or state.get("manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest():
            raise ValueError("incomplete or mismatched text encoding")
        text_vectors = np.load(vectors_path, mmap_mode="r")
        if text_vectors.ndim != 2 or text_vectors.shape[1] not in (768, 2560):
            raise ValueError("expected Bio_ClinicalBERT 768- or Qwen3 2560-dimensional vectors")
        if text_vectors.shape[1] == 768 and state.get("model") != "Bio_ClinicalBERT":
            raise ValueError("BERT vector metadata mismatch")
        if text_vectors.shape[1] == 2560 and state.get("encoder") != "Qwen3-Embedding-4B":
            raise ValueError("Qwen vector metadata mismatch")
        text_encoder = "Bio_ClinicalBERT" if text_vectors.shape[1] == 768 else "Qwen3-Embedding-4B"
        text_width = vector_width or int(text_vectors.shape[1])
        if text_width != text_vectors.shape[1] and not (
            text_vectors.shape[1] == 2560 and text_width in (128, 768)
        ):
            raise ValueError("unsupported Qwen Matryoshka prefix dimension")
        text_rows = 0
        with gzip.open(manifest, "rt", encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle)):
                hadm = int(row["hadm_id"])
                if hadm not in row_by_hadm or int(row["embedding_row"]) != row_number:
                    raise ValueError("text manifest key/order mismatch")
                snippets[row_by_hadm[hadm]].append((FIELD_NAMES.index(row["field"]), row_number))
                text_rows += 1
        if text_rows != text_vectors.shape[0]:
            raise ValueError("text vector count mismatch")
    sequences = tuple(np.array(indices, dtype=np.int64) for indices in patient_rows.values())
    coverage: dict[str, object] = {
        "patients": len(sequences), "admissions": len(base), "lab_rows": lab_count,
        "diagnosis_rows": diagnosis_count, "emar_medication_rows": medication_count,
        "admissions_with_emar": len(hadms_with_emar),
        "admissions_with_rule_text": sum(bool(items) for items in snippets),
        "rule_snippets": sum(map(len, snippets)),
        "text_encoder": text_encoder,
    }
    return V5Data(subject_ids, hadm_ids, splits, times, events, values, masks,
                  gap, sequences, tuple(tuple(items) for items in snippets),
                  text_vectors, text_width, coverage)


@dataclass(frozen=True)
class V5Batch:
    group_values: tuple[torch.Tensor, ...]
    group_masks: tuple[torch.Tensor, ...]
    gap: torch.Tensor
    valid: torch.Tensor
    times: torch.Tensor
    events: torch.Tensor
    snippet_vectors: torch.Tensor
    snippet_index: torch.Tensor

    def to(self, device: torch.device | str) -> V5Batch:
        return V5Batch(tuple(value.to(device) for value in self.group_values),
                       tuple(mask.to(device) for mask in self.group_masks),
                       self.gap.to(device), self.valid.to(device), self.times.to(device),
                       self.events.to(device), self.snippet_vectors.to(device),
                       self.snippet_index.to(device))


def collate(data: V5Data, sequences: list[np.ndarray], *, include_text: bool) -> V5Batch:
    if not sequences:
        raise ValueError("empty patient batch")
    batch_size, length = len(sequences), max(len(sequence) for sequence in sequences)
    valid = np.zeros((batch_size, length), dtype=bool)
    groups = [np.zeros((batch_size, length, width), dtype=np.float32) for width in GROUP_WIDTHS]
    masks = [np.zeros_like(group) for group in groups]
    gap = np.zeros((batch_size, length, 2), dtype=np.float32)
    times = np.zeros((batch_size, length), dtype=np.float32)
    events = np.zeros((batch_size, length), dtype=np.int64)
    snippet_refs: list[tuple[int, int, int]] = []
    for batch_index, sequence in enumerate(sequences):
        count = len(sequence)
        valid[batch_index, :count] = True
        for group_index in range(len(GROUP_WIDTHS)):
            groups[group_index][batch_index, :count] = data.group_values[group_index][sequence]
            masks[group_index][batch_index, :count] = data.group_masks[group_index][sequence]
        gap[batch_index, :count] = data.gap_features[sequence]
        times[batch_index, :count] = data.times[sequence]
        events[batch_index, :count] = data.events[sequence]
        if include_text:
            for position, row_index in enumerate(sequence):
                snippet_refs.extend((batch_index * length + position, field, vector_row)
                                    for field, vector_row in data.snippets[int(row_index)])
    if snippet_refs:
        if data.text_vectors is None:
            raise ValueError("text requested without embeddings")
        positions = np.asarray(snippet_refs, dtype=np.int64)
        vectors = np.asarray(data.text_vectors[positions[:, 2], :data.text_width], dtype=np.float32)
        if data.text_width != data.text_vectors.shape[1]:
            # Qwen 的 Matryoshka 前缀裁剪后重新做 L2 归一化。
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
        if not np.isfinite(vectors).all():
            raise ValueError("nonfinite text vectors")
        index = positions[:, :2]
    else:
        vectors = np.zeros((0, data.text_width), dtype=np.float32)
        index = np.zeros((0, 2), dtype=np.int64)
    return V5Batch(tuple(torch.from_numpy(group) for group in groups),
                   tuple(torch.from_numpy(mask) for mask in masks),
                   torch.from_numpy(gap), torch.from_numpy(valid),
                   torch.from_numpy(times), torch.from_numpy(events),
                   torch.from_numpy(vectors), torch.from_numpy(index))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/evaluate_broad_temporal.py · lines 1–126

<!-- REVIEW_FRAGMENT {"path":"training/v5/evaluate_broad_temporal.py","part":1,"start_line":1,"end_line":126,"source_sha256":"12ef5bf4afd8db874425b910ee55d6589afab804ee862f1ae261f6bb6d8d35d1","fragment_sha256":"12ef5bf4afd8db874425b910ee55d6589afab804ee862f1ae261f6bb6d8d35d1","added_terminal_newline":false} -->
````python
"""同一宽口径标签、同一出院顺序下配对比较摘要、原始事件与时序消融。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.v5.event_data import load_event_data
from training.v5.event_model import TemporalFusionModel
from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_data import load_data, normalize_on_training_patients, split_sequences
from training.v5.independent_fusion import IndependentFusionModel
from training.v5.train import clustered_brier_interval, metrics
from training.v5.train_event_sequence import predict as predict_temporal
from training.v5.train_independent import predict as predict_summary


def _read_report(directory: Path) -> dict[str, object]:
    return json.loads((directory / "report.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned_dir", "structured_dir", "base_csv", "raw_dir", "field_dir",
                 "vectors", "labels_csv", "summary_run", "full_run", "process_run",
                 "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("paired model evaluation requires CUDA")
    data, _ = load_event_data(
        args.aligned_dir, args.structured_dir, args.base_csv, args.raw_dir,
        labels_csv=args.labels_csv, field_dir=args.field_dir, vectors_path=args.vectors,
    )
    summary_data, _ = normalize_on_training_patients(load_data(
        args.aligned_dir, args.structured_dir, args.base_csv, args.field_dir,
        args.vectors, labels_csv=args.labels_csv,
    ))
    if not np.array_equal(data.base.hadm_ids, summary_data.hadm_ids):
        raise ValueError("event and summary admission rows differ")
    reports = {"summary": _read_report(args.summary_run),
               "full": _read_report(args.full_run),
               "process": _read_report(args.process_run)}
    if len({report["label_sha256"] for report in reports.values()}) != 1:
        raise ValueError("different outcome labels")
    if any(report["label_sha256"] != data.base.label_sha256
           or report["field_manifest_sha256"] != data.base.manifest_sha256
           or report["cohort_admissions"] != len(data.base.hadm_ids)
           for report in reports.values()):
        raise ValueError("cohort or field manifest changed")
    if (reports["full"]["raw_source_sha256"] != reports["process"]["raw_source_sha256"]
            or reports["full"]["raw_source_sha256"] != data.source_hashes):
        raise ValueError("raw event sources differ")
    device = torch.device("cuda")
    summary = IndependentFusionModel(
        concept_count=len(data.base.concept_names), type_count=len(data.base.type_names),
        field_count=len(FIELD_LABELS), text_width=data.base.text_width,
    ).to(device)
    summary.load_state_dict(torch.load(args.summary_run / "structured_field_bert.pt",
                                       map_location=device, weights_only=True))
    temporal: dict[str, TemporalFusionModel] = {}
    for name, directory in (("full", args.full_run), ("process", args.process_run)):
        model = TemporalFusionModel(
            concept_count=len(data.base.concept_names), type_count=len(data.base.type_names),
            field_count=len(FIELD_LABELS), text_width=data.base.text_width,
        ).to(device)
        model.load_state_dict(torch.load(directory / "model.pt", map_location=device,
                                         weights_only=True))
        temporal[name] = model
    result: dict[str, object] = {
        "label_sha256": data.base.label_sha256,
        "field_manifest_sha256": data.base.manifest_sha256,
        "raw_source_sha256": data.source_hashes,
        "comparison": "same broad-coded endpoint, patients, BERT vectors; validation-selected models",
        "limitations": "patient-cluster intervals quantify sampling uncertainty only; historical test is not blind",
    }
    for split in ("validation", "test"):
        sequences = split_sequences(data.base, split)
        indices = np.concatenate(sequences)
        times, events = data.base.times[indices], data.base.events[indices]
        risks = {
            "summary": predict_summary(summary, summary_data, sequences, include_text=True,
                                       batch_size=24, device=device)[0],
            "full": predict_temporal(temporal["full"], data, sequences,
                                     process_only=False, batch_size=12, device=device)[0],
            "process": predict_temporal(temporal["process"], data, sequences,
                                        process_only=True, batch_size=12, device=device)[0],
            "order_shuffled": predict_temporal(
                temporal["full"], data, sequences, process_only=False,
                batch_size=12, device=device, shuffle_lab_order_seed=2026,
            )[0],
        }
        scores = {name: metrics(times, events, risk) for name, risk in risks.items()}
        split_result: dict[str, object] = {
            "admissions": len(indices), "patients": len(sequences),
            "models": {name: {"integrated_brier_7_365": detail["integrated_brier_7_365"],
                              "mean_auc_7_365": detail["mean_auc_7_365"]}
                       for name, detail in scores.items()},
            "paired_brier_difference_95ci": {},
        }
        for name, comparator in (("full_minus_summary", "summary"),
                                 ("full_minus_process", "process"),
                                 ("order_shuffled_minus_full", "full")):
            current = risks["order_shuffled"] if name.startswith("order_") else risks["full"]
            interval = clustered_brier_interval(data.base.subject_ids[indices], times, events,
                                                 current, risks[comparator])
            delta = (scores["order_shuffled"]["integrated_brier_7_365"]
                     if name.startswith("order_") else scores["full"]["integrated_brier_7_365"])
            delta -= scores[comparator]["integrated_brier_7_365"]
            split_result["paired_brier_difference_95ci"][name] = {
                "estimate": float(delta), "patient_cluster_95ci": interval,
            }
        result["historical_test" if split == "test" else split] = split_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/evaluate_completed_event_run.py · lines 1–260

<!-- REVIEW_FRAGMENT {"path":"training/v5/evaluate_completed_event_run.py","part":1,"start_line":1,"end_line":260,"source_sha256":"1356157075753cb5b863263fa3787a769c3c357e69b95121da01591eadecf240","fragment_sha256":"1356157075753cb5b863263fa3787a769c3c357e69b95121da01591eadecf240","added_terminal_newline":false} -->
````python
"""只读重放已完成的清洗事件 V5，补充竞争风险图与探索性随访 DCA。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from training.v5.event_data import EventData, collate_events
from training.v5.event_set_model import EventSetFusionModel
from training.v5.independent_data import split_sequences
from training.v5.model import predict_cif
from training.v5.processed_event_data import load_processed_event_data
from training.v5.train import GRID, metrics

POINTS = (30, 90, 180, 365)
DCA_DAYS = (30, 90)
DCA_THRESHOLDS = np.arange(1, 51, dtype=np.float64) / 100


def binary_roc(labels: np.ndarray, risk: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """固定时点累积病例 vs 未发生该事件者；先死亡者为对照。"""
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if min(positives, negatives) < 1:
        raise ValueError("ROC needs both cases and controls")
    order = np.argsort(-risk, kind="stable")
    sorted_risk = risk[order]
    sorted_labels = labels[order].astype(np.int64)
    end = np.r_[np.flatnonzero(np.diff(sorted_risk) != 0), len(labels) - 1]
    tp = np.cumsum(sorted_labels)[end]
    fp = end + 1 - tp
    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def decision_curve(labels: np.ndarray, risk: np.ndarray,
                   thresholds: np.ndarray = DCA_THRESHOLDS) -> dict[str, np.ndarray]:
    """世界观 A 下的时点净获益；只有明确行动及阈值后才可作临床解释。"""
    if np.any((thresholds <= 0) | (thresholds >= 1)):
        raise ValueError("DCA thresholds must lie strictly between zero and one")
    selected = risk[:, None] >= thresholds[None, :]
    true_positive = (selected & labels[:, None]).mean(axis=0)
    false_positive = (selected & ~labels[:, None]).mean(axis=0)
    odds = thresholds / (1 - thresholds)
    prevalence = float(labels.mean())
    return {"threshold": thresholds,
            "model": true_positive - false_positive * odds,
            "treat_all": prevalence - (1 - prevalence) * odds,
            "treat_none": np.zeros_like(thresholds)}


def calibration_bins(labels: np.ndarray, risk: np.ndarray, bins: int = 10) -> list[dict[str, float | int]]:
    order = np.argsort(risk, kind="stable")
    return [{"n": int(len(part)), "predicted": float(risk[part].mean()),
             "observed_world_a": float(labels[part].mean())}
            for part in np.array_split(order, bins)]


def patient_bootstrap_auc(subject_ids: np.ndarray, labels: np.ndarray,
                          risk: np.ndarray, *, draws: int = 200,
                          seed: int = 2026) -> tuple[float, float]:
    patients, inverse = np.unique(subject_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(patients))]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sampled = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if min(int(labels[sampled].sum()), len(sampled) - int(labels[sampled].sum())) >= 20:
            values.append(binary_roc(labels[sampled], risk[sampled])[2])
    if len(values) < draws // 2:
        raise ValueError("too few valid patient bootstrap samples")
    return tuple(float(x) for x in np.quantile(values, (0.025, 0.975)))


def evaluate_split(subject_ids: np.ndarray, times: np.ndarray, events: np.ndarray,
                   curves: np.ndarray, *, draws: int = 200) -> tuple[dict[str, object], dict[str, object]]:
    if curves.shape != (len(times), 366, 3) or not np.isfinite(curves).all():
        raise ValueError("invalid competing-risk curves")
    if np.max(np.abs(curves.sum(axis=2) - 1)) > 1e-5 or np.min(curves) < -1e-8:
        raise ValueError("competing-risk probabilities invalid")
    if np.any((events == 0) & (times != 365)):
        raise ValueError("early censoring needs a different IPCW/Aalen-Johansen evaluator")
    if not np.all(np.isin(events, (0, 1, 2))):
        raise ValueError("unknown event type")
    summary: dict[str, object] = {"n_admissions": int(len(times)),
                                   "n_patients": int(len(np.unique(subject_ids))),
                                   "bootstrap_patient_draws": draws,
                                   "followup_contract": "World A: all event_type=0 censored at day 365",
                                   "readmission": {}, "death": {}}
    detailed: dict[str, object] = {"roc": {}, "dca": {}, "calibration": {}}
    for cause, column in (("readmission", 1), ("death", 2)):
        for day in POINTS:
            labels = ((events == column) & (times <= day))
            predicted = curves[:, day, column]
            fpr, tpr, auc = binary_roc(labels, predicted)
            low, high = patient_bootstrap_auc(subject_ids, labels, predicted, draws=draws,
                                              seed=2026 + day + column)
            event_rate = float(labels.mean())
            brier = float(np.mean((predicted - labels) ** 2))
            null_brier = event_rate * (1 - event_rate)
            summary[cause][str(day)] = {
                "events": int(labels.sum()), "observed_fraction_world_a": event_rate,
                "mean_predicted_risk": float(predicted.mean()),
                "observed_to_expected": float(event_rate / predicted.mean()) if predicted.mean() else None,
                "auc": auc, "auc_patient_cluster_95ci": [low, high],
                "brier": brier, "scaled_brier_vs_constant_event_rate":
                    float(1 - brier / null_brier) if null_brier else None,
            }
            detailed["roc"][(cause, day)] = (fpr, tpr)
            detailed["calibration"][(cause, day)] = calibration_bins(labels, predicted)
            if cause == "readmission" and day in DCA_DAYS:
                detailed["dca"][day] = decision_curve(labels, predicted)
    return summary, detailed


@torch.no_grad()
def replay(model: EventSetFusionModel, data: EventData,
           sequences: list[np.ndarray], device: torch.device) -> np.ndarray:
    model.eval()
    selected = np.concatenate(sequences)
    logits = np.zeros((len(data.base.hadm_ids), 2), dtype=np.float32)
    filled = np.zeros(len(data.base.hadm_ids), dtype=bool)
    for start in range(0, len(sequences), 12):
        members = sequences[start:start + 12]
        batch = collate_events(data, members, include_text=True, process_only=False).to(device)
        values = model(batch).log_amplitude.cpu().numpy()
        for pos, sequence in enumerate(members):
            logits[sequence] = values[pos, :len(sequence)]
            filled[sequence] = True
    if int(filled.sum()) != len(selected) or not filled[selected].all():
        raise ValueError("checkpoint prediction alignment failed")
    shape = torch.nn.functional.softplus(model.raw_shape).detach().cpu().numpy() + 0.1
    return predict_cif(logits[selected], shape, GRID)


def save_figures(details: dict[str, dict[str, object]], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for split, detail in details.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for day in POINTS:
            fpr, tpr = detail["roc"][("readmission", day)]
            axes[0].plot(fpr, tpr, label=f"day {day}")
            points = detail["calibration"][("readmission", day)]
            axes[1].plot([p["predicted"] for p in points],
                         [p["observed_world_a"] for p in points], marker="o", label=f"day {day}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set(xlabel="False positive rate", ylabel="Sensitivity", title="Cumulative/dynamic ROC")
        axes[1].set(xlabel="Predicted CIF", ylabel="Observed fraction (World A)", title="Readmission calibration")
        for axis in axes:
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{split}_roc_calibration.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for axis, day in zip(axes, DCA_DAYS, strict=True):
            dca = detail["dca"][day]
            for name in ("model", "treat_all", "treat_none"):
                axis.plot(dca["threshold"], dca[name], label=name)
            axis.set(xlabel="Risk threshold", ylabel="Net benefit per discharge",
                     title=f"Day {day}: hypothetical intensified follow-up")
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{split}_exploratory_dca.png", dpi=160)
        plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint replay requires CUDA")
    report = json.loads((args.run_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "training_completed" or report.get("event_encoder") != "set" or not report.get("preprocessed_events"):
        raise ValueError("not a completed processed-event V5 checkpoint")
    data, _ = load_processed_event_data(
        args.aligned_dir, args.structured_dir, args.base_csv,
        args.processed_event_dir, args.raw_ledger_dir, labels_csv=args.labels_csv,
        field_dir=args.field_dir, vectors_path=args.vectors)
    if (report["cohort_admissions"] != len(data.base.hadm_ids)
            or report["label_sha256"] != data.base.label_sha256
            or report["field_manifest_sha256"] != data.base.manifest_sha256
            or report["event_concepts"] != list(data.concept_names)
            or report.get("disabled_static_concepts", []) != list(data.base.disabled_concepts)):
        raise ValueError("evaluation input differs from completed training contract")
    state = torch.load(args.run_dir / "model.pt", map_location="cpu", weights_only=True)
    model = EventSetFusionModel(
        concept_count=len(data.base.concept_names), type_count=len(data.base.type_names),
        field_count=len(data.base.field_roles), text_width=data.base.text_width,
        event_concept_count=len(data.concept_names)).to("cuda").eval()
    model.load_state_dict(state, strict=True)
    summary: dict[str, object] = {
        "status": "completed", "source_run": str(args.run_dir),
        "primary_event": report["primary_event"], "label_sha256": report["label_sha256"],
        "patient_bootstrap_draws": args.bootstrap,
        "limitations": [
            "DCA assumes a hypothetical intensified-follow-up action; no clinical threshold or intervention effect has been validated.",
            "World A treats unobserved events as absent through day 365; outside-hospital readmissions may be missed.",
            "Historical test patients have been viewed in previous experiments and are not a new blind test.",
        ], "splits": {}}
    details = {}
    for split in ("validation", "test"):
        sequences = split_sequences(data.base, split)
        indices = np.concatenate(sequences)
        curves = replay(model, data, sequences, torch.device("cuda"))
        original = metrics(data.base.times[indices], data.base.events[indices], curves[:, :, 1])
        name = "validation" if split == "validation" else "historical_test"
        reference = report[name]["temporal_full"]
        for metric in ("integrated_brier_7_365", "mean_auc_7_365"):
            if not np.isclose(original[metric], reference[metric], atol=1e-7):
                raise ValueError(f"checkpoint replay changed {name} {metric}")
        scores, detail = evaluate_split(data.base.subject_ids[indices], data.base.times[indices],
                                        data.base.events[indices], curves,
                                        draws=args.bootstrap)
        scores["replayed_integrated_brier_7_365"] = original["integrated_brier_7_365"]
        scores["replayed_mean_auc_7_365"] = original["mean_auc_7_365"]
        summary["splits"][name] = scores
        details[name] = detail
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "dca_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "day", "threshold", "model", "treat_all", "treat_none"))
        for split, detail in details.items():
            for day, dca in detail["dca"].items():
                for i, threshold in enumerate(dca["threshold"]):
                    writer.writerow((split, day, threshold, dca["model"][i],
                                     dca["treat_all"][i], dca["treat_none"][i]))
    save_figures(details, args.output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run_dir", "aligned_dir", "structured_dir", "base_csv", "processed_event_dir",
                 "raw_ledger_dir", "field_dir", "vectors", "labels_csv", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    if not 20 <= args.bootstrap <= 500:
        raise ValueError("bootstrap draws must be between 20 and 500")
    result = run(args)
    print(json.dumps({"status": result["status"], "output": str(args.output_dir),
                      "splits": list(result["splits"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/evaluate_medication_documentation.py · lines 1–95

<!-- REVIEW_FRAGMENT {"path":"training/v5/evaluate_medication_documentation.py","part":1,"start_line":1,"end_line":95,"source_sha256":"f3c7dca7ccd32258d628fbf9d82f9489ff3b07e1c671dd3b876748c0cf69a348","fragment_sha256":"f3c7dca7ccd32258d628fbf9d82f9489ff3b07e1c671dd3b876748c0cf69a348","added_terminal_newline":false} -->
````python
"""在有/无出院时可见 eMAR 文书的住院层内审查药理组对照。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from training.v5.compare_screened_group_ablations import _predictions, _summary_for_cause


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def evaluate(root: Path, raw: Path, output: Path, *, seeds: tuple[int, ...]) -> dict:
    if output.exists():
        raise FileExistsError(output)
    raw_manifest = json.loads((raw / "metadata/manifest.json").read_text(encoding="utf-8"))
    emar_file = raw / "raw/emar.csv.gz"
    if raw_manifest["status"] != "validated" or _sha(emar_file) != raw_manifest["sources"]["raw/emar"]["sha256"]:
        raise ValueError("unvalidated eMAR ledger")
    admission_times = {}
    for row in _rows(raw / "cohort/admissions.csv.gz"):
        admission_times[row["hadm_id"]] = (datetime.fromisoformat(row["admittime"]),
                                            datetime.fromisoformat(row["dischtime"]))
    documented = set()
    for row in _rows(emar_file):
        times = admission_times.get(row["hadm_id"])
        if times is None or row["charttime"] in ("", "\\N") or row["storetime"] in ("", "\\N"):
            continue
        chart, stored = datetime.fromisoformat(row["charttime"]), datetime.fromisoformat(row["storetime"])
        if times[0] <= chart <= stored <= times[1]:
            documented.add(int(row["hadm_id"]))
    result = {"status": "completed", "source_sha256": _sha(emar_file),
              "definition": "at least one eMAR record with admission <= charttime <= storetime <= discharge; not proof of complete medication capture",
              "seeds": {}}
    for seed in seeds:
        before = root / f"masked_seed{seed}"
        after = root / f"active_seed{seed}"
        reports = [json.loads((path / "report.json").read_text(encoding="utf-8")) for path in (before, after)]
        if any(report["status"] != "training_completed" or report["event_encoder"] != "abnormal_medication"
               for report in reports) or reports[0]["medication_effect_mode"] != "masked" or reports[1]["medication_effect_mode"] != "active":
            raise ValueError("incorrect paired reports")
        result["seeds"][str(seed)] = {}
        for split in ("validation", "historical_test"):
            old, new = _predictions(before, split), _predictions(after, split)
            for key in ("subject_ids", "hadm_ids", "times", "events"):
                if not np.array_equal(old[key], new[key]):
                    raise ValueError("different paired prediction rows")
            available = np.isin(old["hadm_ids"], list(documented))
            result["seeds"][str(seed)][split] = {}
            for subgroup, selector in (("any_emar_record", available), ("no_emar_record", ~available)):
                if selector.sum() < 100:
                    raise ValueError("eMAR subgroup too small")
                subgroup_result = {"admissions": int(selector.sum()), "patients": int(np.unique(old["subject_ids"][selector]).size)}
                for cause, name in ((1, "readmission"), (2, "death")):
                    baseline = _summary_for_cause(old["times"][selector], old["events"][selector],
                                                  old["competing_cif"][selector], cause)
                    candidate = _summary_for_cause(new["times"][selector], new["events"][selector],
                                                   new["competing_cif"][selector], cause)
                    subgroup_result[name] = {
                        "masked": {key: baseline[key] for key in ("mean_auc_7_365", "integrated_brier_7_365")},
                        "active": {key: candidate[key] for key in ("mean_auc_7_365", "integrated_brier_7_365")},
                        "delta_active_minus_masked": {key: candidate[key] - baseline[key]
                                                       for key in ("mean_auc_7_365", "integrated_brier_7_365")}}
                result["seeds"][str(seed)][split][subgroup] = subgroup_result
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--raw-ledger-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    args = parser.parse_args()
    report = evaluate(args.run_root, args.raw_ledger_dir, args.output, seeds=tuple(args.seeds))
    print(json.dumps({"status": report["status"], "seeds": tuple(report["seeds"])}, ensure_ascii=False))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/event_data.py · lines 1–236

<!-- REVIEW_FRAGMENT {"path":"training/v5/event_data.py","part":1,"start_line":1,"end_line":236,"source_sha256":"59a4da0121c4620243574c7112000297c0a01ed40ec1b2dc9d81008944f8bd59","fragment_sha256":"59a4da0121c4620243574c7112000297c0a01ed40ec1b2dc9d81008944f8bd59","added_terminal_newline":false} -->
````python
"""把本次住院的原始化验和实际给药排成不规则事件流。"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from training.v5.clinical_concepts import LABS, MED_NAMES, medication_classes
from training.v5.independent_data import IndependentData, collate, load_data, normalize_on_training_patients
from training.v5.independent_fusion import IndependentBatch
from training.v5.numeric_preprocessing import robust_parameters


LAB_NAMES = tuple(sorted({name for name, _ in LABS.values()}))
EVENT_NAMES = LAB_NAMES + MED_NAMES


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EventData:
    base: IndependentData
    event_values: np.ndarray
    event_concepts: np.ndarray
    event_is_lab: np.ndarray
    events_by_admission: tuple[np.ndarray, ...]
    source_hashes: dict[str, str]
    concept_names: tuple[str, ...] = EVENT_NAMES
    event_numeric_mask: np.ndarray | None = None
    recovery_by_admission: np.ndarray | None = None
    medication_by_admission: np.ndarray | None = None
    abnormality_scaling: dict[str, object] | None = None


@dataclass(frozen=True)
class EventBatch:
    base: IndependentBatch
    event_values: torch.Tensor      # [N,4]: normalized lab value, elapsed, gap, before discharge
    event_concepts: torch.Tensor    # [N]
    event_counts: torch.Tensor      # [B,L]，对应每次住院在展平数组中的连续片段
    event_is_lab: torch.Tensor | None = None
    event_numeric_mask: torch.Tensor | None = None
    recovery_by_admission: torch.Tensor | None = None
    medication_by_admission: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> EventBatch:
        return EventBatch(self.base.to(device), self.event_values.to(device),
                          self.event_concepts.to(device), self.event_counts.to(device),
                          None if self.event_is_lab is None else self.event_is_lab.to(device),
                          None if self.event_numeric_mask is None else self.event_numeric_mask.to(device),
                          None if self.recovery_by_admission is None else self.recovery_by_admission.to(device),
                          None if self.medication_by_admission is None else self.medication_by_admission.to(device))


def load_event_data(aligned_dir: Path, structured_dir: Path, base_csv: Path,
                    raw_dir: Path, *, labels_csv: Path, field_dir: Path | None = None,
                    vectors_path: Path | None = None) -> tuple[EventData, dict[str, object]]:
    """仅保留静态结构化 token；化验与药物摘要由原始有时标事件替代。"""

    base, static_normalization = normalize_on_training_patients(
        load_data(aligned_dir, structured_dir, base_csv, field_dir, vectors_path,
                  labels_csv=labels_csv)
    )
    keep_types = {index + 1 for index, name in enumerate(base.type_names)
                  if name not in {"laboratory", "administered_medication"}}
    static_refs = tuple(refs[np.isin(base.token_types[refs], list(keep_types))]
                        for refs in base.tokens_by_admission)
    if any(len(refs) == 0 for refs in static_refs):
        raise ValueError("admission lost all static context tokens")
    base = replace(base, tokens_by_admission=static_refs)
    raw_qa = json.loads((raw_dir / "export_summary.json").read_text(encoding="utf-8"))
    source_hashes = {}
    for name in ("labs", "emar"):
        path = raw_dir / f"{name}.csv.gz"
        source_hashes[name] = _sha(path)
        if source_hashes[name] != raw_qa["sources"][name]["sha256"]:
            raise ValueError(f"raw {name} source changed")
    hadm_to_index = {str(hadm): index for index, hadm in enumerate(base.hadm_ids)}
    landmarks: dict[str, tuple[datetime, datetime]] = {}
    with base_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            landmarks[row["hadm_id"]] = (datetime.fromisoformat(row["admittime"]),
                                         datetime.fromisoformat(row["dischtime"]))
    concept_id = {name: index + 1 for index, name in enumerate(EVENT_NAMES)}
    source_events: list[list[tuple[datetime, int, float, bool, str]]] = [
        [] for _ in range(len(base.hadm_ids))
    ]
    with gzip.open(raw_dir / "labs.csv.gz", "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            hadm = row["hadm_id"]
            index = hadm_to_index.get(hadm)
            if index is None:
                raise ValueError("laboratory event outside cohort")
            moment = datetime.fromisoformat(row["charttime"])
            admission, discharge = landmarks[hadm]
            if not admission <= moment <= discharge:
                raise ValueError("laboratory event outside admission")
            concept = concept_id[LABS[int(row["itemid"])][0]]
            source_events[index].append((moment, concept, float(row["valuenum"]), True,
                                         row["labevent_id"]))
    with gzip.open(raw_dir / "emar.csv.gz", "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            hadm = row["hadm_id"]
            index = hadm_to_index.get(hadm)
            if index is None:
                raise ValueError("administration event outside cohort")
            moment = datetime.fromisoformat(row["charttime"])
            admission, discharge = landmarks[hadm]
            if not admission <= moment <= discharge:
                raise ValueError("administration event outside admission")
            for name in medication_classes(row["medication"]):
                source_events[index].append((moment, concept_id[name], 0.0, False,
                                             row["emar_id"] + ":" + row["emar_seq"]))

    raw_values: list[list[float]] = []
    concepts: list[int] = []
    lab_flags: list[bool] = []
    admission_refs: list[np.ndarray] = []
    for index, records in enumerate(source_events):
        records.sort(key=lambda item: (item[0], item[1], item[4]))
        start = len(raw_values)
        hadm = str(base.hadm_ids[index])
        admission, discharge = landmarks[hadm]
        previous = admission
        seen: set[tuple[datetime, int, str]] = set()
        for moment, concept, value, is_lab, source_id in records:
            key = (moment, concept, source_id)
            if key in seen:
                continue
            seen.add(key)
            elapsed = max(0.0, (moment - admission).total_seconds() / 86400.0)
            gap = max(0.0, (moment - previous).total_seconds() / 86400.0)
            before_discharge = max(0.0, (discharge - moment).total_seconds() / 86400.0)
            raw_values.append([value if is_lab else 0.0, math.log1p(elapsed),
                               math.log1p(gap), math.log1p(before_discharge)])
            concepts.append(concept)
            lab_flags.append(is_lab)
            previous = moment
        admission_refs.append(np.arange(start, len(raw_values), dtype=np.int64))
    values = np.asarray(raw_values, dtype=np.float32).reshape(-1, 4)
    concept_array = np.asarray(concepts, dtype=np.int64)
    lab_array = np.asarray(lab_flags, dtype=bool)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite event values")
    train_hadms = set(base.hadm_ids[base.splits == "train"].tolist())
    train_events = np.zeros(len(values), dtype=bool)
    for index, refs in enumerate(admission_refs):
        if int(base.hadm_ids[index]) in train_hadms:
            train_events[refs] = True
    event_normalization: dict[str, dict[str, float]] = {}
    for name in LAB_NAMES:
        selected = lab_array & (concept_array == concept_id[name])
        observed = values[selected & train_events, 0]
        if observed.size == 0:
            raise ValueError(f"no training measurements for {name}")
        fitted = robust_parameters(observed)
        median, scale = float(fitted["median"]), float(fitted["scale"])
        values[selected, 0] = np.clip((values[selected, 0] - median) / scale, -8.0, 8.0)
        event_normalization[name] = {"median": median, "scale": scale}
    return (EventData(base, values, concept_array, lab_array,
                      tuple(admission_refs), source_hashes),
            {"static": static_normalization, "event_lab": event_normalization})


def collate_events(data: EventData, sequences: list[np.ndarray], *, include_text: bool,
                   process_only: bool = False,
                   shuffle_lab_order_seed: int | None = None) -> EventBatch:
    if process_only and shuffle_lab_order_seed is not None:
        raise ValueError("process-only and lab-value order shuffle are separate ablations")
    base_batch = collate(data.base, sequences, include_text=include_text)
    batch_size = len(sequences)
    length = max(len(sequence) for sequence in sequences)
    counts = np.zeros((batch_size, length), dtype=np.int64)
    refs_by_stay: list[np.ndarray] = []
    for patient, sequence in enumerate(sequences):
        for position, row in enumerate(sequence):
            refs = data.events_by_admission[int(row)]
            counts[patient, position] = len(refs)
            refs_by_stay.append(refs)
        refs_by_stay.extend([np.empty(0, dtype=np.int64)] * (length - len(sequence)))
    refs = np.concatenate(refs_by_stay)
    values = data.event_values[refs].copy()
    concepts = data.event_concepts[refs]
    if shuffle_lab_order_seed is not None:
        offset = 0
        for patient, sequence in enumerate(sequences):
            for position in range(length):
                n = int(counts[patient, position])
                if position < len(sequence):
                    rng = np.random.default_rng(
                        shuffle_lab_order_seed + int(data.base.hadm_ids[int(sequence[position])])
                    )
                    for concept in np.unique(concepts[offset:offset + n]):
                        local = np.flatnonzero(concepts[offset:offset + n] == concept) + offset
                        if len(local) > 1 and data.event_is_lab[refs[local[0]]]:
                            values[local, 0] = rng.permutation(values[local, 0])
                offset += n
    if process_only:
        # 保留事件类型、频率与时间，仅去掉检验数值；用于诊疗过程对照。
        values[:, 0] = 0.0
    recovery = None
    if data.recovery_by_admission is not None:
        if data.recovery_by_admission.shape[0] != len(data.base.hadm_ids):
            raise ValueError("recovery/admission count mismatch")
        recovery = np.zeros((batch_size, length, data.recovery_by_admission.shape[1]), dtype=np.float32)
        for patient, sequence in enumerate(sequences):
            recovery[patient, :len(sequence)] = data.recovery_by_admission[sequence]
    medication = None
    if data.medication_by_admission is not None:
        if data.medication_by_admission.shape[0] != len(data.base.hadm_ids):
            raise ValueError("medication/admission count mismatch")
        medication = np.zeros((batch_size, length, data.medication_by_admission.shape[1]), dtype=np.float32)
        for patient, sequence in enumerate(sequences):
            medication[patient, :len(sequence)] = data.medication_by_admission[sequence]
    return EventBatch(base_batch, torch.from_numpy(values),
                      torch.from_numpy(concepts), torch.from_numpy(counts),
                      torch.from_numpy(data.event_is_lab[refs]),
                      torch.from_numpy(data.event_is_lab[refs] if data.event_numeric_mask is None
                                       else data.event_numeric_mask[refs]),
                      None if recovery is None else torch.from_numpy(recovery),
                      None if medication is None else torch.from_numpy(medication))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/event_model.py · lines 1–90

<!-- REVIEW_FRAGMENT {"path":"training/v5/event_model.py","part":1,"start_line":1,"end_line":90,"source_sha256":"364d9d3919192aeb5f1412f4ef5af84c1559372ff9027bc0a4d6bcba67212f75","fragment_sha256":"364d9d3919192aeb5f1412f4ef5af84c1559372ff9027bc0a4d6bcba67212f75","added_terminal_newline":false} -->
````python
"""V5 原始事件流分支：每次住院先编码不规则事件，再按时间编码患者住院史。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from training.v5.event_data import EVENT_NAMES, EventBatch
from training.v5.independent_fusion import IndependentFusionModel
from training.v5.model import V5Output


class TemporalFusionModel(IndependentFusionModel):
    """静态、原始事件、字段文本各自编码；住院级融合后作因果历史建模。"""

    def __init__(self, *, concept_count: int, type_count: int, field_count: int,
                 text_width: int, width: int = 128, event_width: int = 48) -> None:
        super().__init__(concept_count=concept_count, type_count=type_count,
                         field_count=field_count, text_width=text_width, width=width)
        self.event_embedding = nn.Embedding(len(EVENT_NAMES) + 1, event_width, padding_idx=0)
        self.event_value_projection = nn.Sequential(
            nn.Linear(4, event_width), nn.GELU(), nn.LayerNorm(event_width)
        )
        self.event_gru = nn.GRU(event_width, event_width, batch_first=True)
        self.event_score = nn.Linear(event_width, 1)
        self.event_projection = nn.Sequential(
            nn.Linear(2 * event_width, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.event_gate = nn.Linear(2 * width, 1)
        nn.init.constant_(self.event_gate.bias, -1.0)

    def _events(self, batch: EventBatch) -> tuple[torch.Tensor, torch.Tensor]:
        counts = batch.event_counts.flatten()
        flat_stays = counts.numel()
        present = counts > 0
        output = batch.event_values.new_zeros((flat_stays, self.gap_projection.out_features))
        if not bool(present.any()):
            return output, present
        if int(counts.sum()) != batch.event_values.shape[0]:
            raise ValueError("event offsets do not match flattened values")
        encoded = (self.event_value_projection(batch.event_values)
                   + self.event_embedding(batch.event_concepts))
        sequences = torch.split(encoded, counts.tolist())
        ordered = torch.argsort(counts, descending=True)
        ordered = ordered[counts[ordered] > 0]
        # 长度相近的住院合批，限制填充后 token 数；不截断任何原始事件。
        cursor = 0
        while cursor < len(ordered):
            first_length = int(counts[ordered[cursor]])
            allowance = max(1, min(64, 8192 // first_length))
            positions = ordered[cursor:cursor + allowance]
            lengths = counts[positions]
            padded = pad_sequence([sequences[int(index)] for index in positions], batch_first=True)
            packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True,
                                          enforce_sorted=True)
            transformed, final = self.event_gru(packed)
            states, _ = pad_packed_sequence(transformed, batch_first=True)
            time_index = torch.arange(states.shape[1], device=states.device)
            mask = time_index.unsqueeze(0) < lengths.unsqueeze(1)
            score = self.event_score(states).squeeze(-1).masked_fill(~mask, -1e4)
            weight = torch.softmax(score, dim=-1)
            pooled = (states * weight.unsqueeze(-1)).sum(dim=1)
            summary = self.event_projection(torch.cat((final[-1], pooled), dim=-1))
            output = output.index_copy(0, positions, summary)
            cursor += len(positions)
        return output, present

    def forward(self, batch: EventBatch) -> V5Output:
        source = batch.base
        batch_size, length = source.valid.shape
        flat_stays = batch_size * length
        structured = self._structured(source)
        events, event_present = self._events(batch)
        event_gate = torch.sigmoid(self.event_gate(torch.cat((structured, events), dim=-1)))
        event_gate = event_gate * event_present.unsqueeze(-1)
        history = structured + event_gate * events
        text, text_present = self._text(source, flat_stays)
        text_gate = torch.sigmoid(self.fusion_gate(torch.cat((history, text), dim=-1)))
        text_gate = text_gate * text_present.unsqueeze(-1)
        admission = self.fusion_norm(history + text_gate * text)
        admission = admission.view(batch_size, length, -1) + self.gap_projection(source.gap)
        causal = torch.triu(torch.ones((length, length), dtype=torch.bool,
                                       device=source.valid.device), diagonal=1)
        patient = self.patient_encoder(admission, mask=causal,
                                       src_key_padding_mask=~source.valid)
        log_amplitude = self.risk_head(patient).clamp(-10.0, 6.0)
        shape = F.softplus(self.raw_shape) + 0.1
        return V5Output(log_amplitude, shape, text_gate.view(batch_size, length, 1))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/event_set_model.py · lines 1–110

<!-- REVIEW_FRAGMENT {"path":"training/v5/event_set_model.py","part":1,"start_line":1,"end_line":110,"source_sha256":"025577831f44139263aeab5826b1fa971e19800549eb2175756b9387f29bc279","fragment_sha256":"025577831f44139263aeab5826b1fa971e19800549eb2175756b9387f29bc279","added_terminal_newline":false} -->
````python
"""STraTS 式事件集合编码；不是含自监督预训练的原论文复现。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from training.v5.event_data import EVENT_NAMES, LAB_NAMES, EventBatch
from training.v5.event_model import TemporalFusionModel


class EventSetFusionModel(TemporalFusionModel):
    """仅替换住院内编码器；融合、患者历史及竞争风险头沿用原模型。"""

    def __init__(self, *, concept_count: int, type_count: int, field_count: int,
                 text_width: int, width: int = 128, event_width: int = 64,
                 event_concept_count: int | None = None,
                 clinical_recovery_width: int = 0) -> None:
        # 先构造原模型，保证保留模块（包括 event_gate）的初始权重一致。
        super().__init__(concept_count=concept_count, type_count=type_count,
                         field_count=field_count, text_width=text_width, width=width)
        del self.event_gru
        del self.event_value_projection
        vocabulary_size = len(EVENT_NAMES) if event_concept_count is None else event_concept_count
        if vocabulary_size < 1:
            raise ValueError("empty event vocabulary")
        self.requires_explicit_type = event_concept_count is not None
        self.event_embedding = nn.Embedding(vocabulary_size + 1, event_width, padding_idx=0)
        self.numeric_embedding = nn.Sequential(nn.Linear(1, 8), nn.Tanh(),
                                               nn.Linear(8, event_width))
        self.time_embedding = nn.Sequential(nn.Linear(1, 8), nn.Tanh(),
                                            nn.Linear(8, event_width))
        self.event_type_embedding = nn.Embedding(2, event_width)
        self.event_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(event_width, 4, 2 * event_width,
                                       dropout=0.1, batch_first=True),
            num_layers=2, enable_nested_tensor=False,
        )
        self.event_score = nn.Linear(event_width, 1)
        self.event_projection = nn.Sequential(nn.Linear(event_width, width),
                                              nn.GELU(), nn.LayerNorm(width))
        self.clinical_recovery_width = clinical_recovery_width
        if clinical_recovery_width:
            if clinical_recovery_width % 6:
                raise ValueError("clinical recovery features must contain six values per axis")
            self.recovery_projection = nn.Sequential(nn.Linear(clinical_recovery_width, 32),
                                                     nn.GELU(), nn.Linear(32, width),
                                                     nn.LayerNorm(width))
            self.recovery_gate = nn.Linear(2 * width, 1)
            nn.init.constant_(self.recovery_gate.bias, -1.0)

    def _combine_recovery(self, batch: EventBatch, output: torch.Tensor,
                          present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.clinical_recovery_width:
            if batch.recovery_by_admission is not None:
                raise ValueError("recovery input supplied to model without recovery branch")
            return output, present
        features = batch.recovery_by_admission
        if features is None or features.shape != (*batch.event_counts.shape, self.clinical_recovery_width):
            raise ValueError("clinical recovery feature shape mismatch")
        features = features.reshape(-1, self.clinical_recovery_width)
        if not bool(torch.isfinite(features).all()):
            raise ValueError("nonfinite clinical recovery feature")
        observed = (features[:, ::6] > 0).any(dim=1)
        if not bool(observed.any()):
            return output, present
        projected = self.recovery_projection(features)
        gate = torch.sigmoid(self.recovery_gate(torch.cat((output, projected), dim=-1)))
        return output + gate * projected * observed.unsqueeze(-1), present | observed

    def _events(self, batch: EventBatch) -> tuple[torch.Tensor, torch.Tensor]:
        counts = batch.event_counts.flatten()
        present = counts > 0
        output = batch.event_values.new_zeros((counts.numel(), self.gap_projection.out_features))
        if int(counts.sum()) != batch.event_values.shape[0]:
            raise ValueError("event offsets do not match flattened values")
        if not bool(present.any()):
            return self._combine_recovery(batch, output, present)
        if self.requires_explicit_type and (batch.event_is_lab is None or batch.event_numeric_mask is None):
            raise ValueError("new vocabulary requires explicit event type and numeric mask")
        is_lab = batch.event_concepts <= len(LAB_NAMES) if batch.event_is_lab is None else batch.event_is_lab
        numeric_mask = is_lab if batch.event_numeric_mask is None else batch.event_numeric_mask
        if is_lab.shape != batch.event_concepts.shape or numeric_mask.shape != is_lab.shape:
            raise ValueError("event type/mask shape mismatch")
        if bool((numeric_mask & ~is_lab).any()):
            raise ValueError("medication cannot have a numeric observation")
        # 仅使用项目、检验值及真实相对时间；不读取排序相关 gap 和距出院时间。
        encoded = (self.event_embedding(batch.event_concepts)
                   + self.time_embedding(batch.event_values[:, 1:2])
                   + self.event_type_embedding(is_lab.long())
                   + self.numeric_embedding(batch.event_values[:, :1]) * numeric_mask.unsqueeze(-1))
        sequences = torch.split(encoded, counts.tolist())
        ordered = torch.argsort(counts, descending=True)
        ordered = ordered[counts[ordered] > 0]
        cursor = 0
        while cursor < len(ordered):
            longest = int(counts[ordered[cursor]])
            # 限制二次方注意力开销，不截断原始事件。
            allowance = max(1, min(64, 8192 // longest, 2_000_000 // (longest * longest)))
            positions = ordered[cursor:cursor + allowance]
            lengths = counts[positions]
            padded = pad_sequence([sequences[int(i)] for i in positions], batch_first=True)
            valid = torch.arange(padded.shape[1], device=padded.device)[None, :] < lengths[:, None]
            states = self.event_encoder(padded, src_key_padding_mask=~valid)
            score = self.event_score(states).squeeze(-1).masked_fill(~valid, -torch.inf)
            pooled = (states * torch.softmax(score, dim=1).unsqueeze(-1)).sum(dim=1)
            output = output.index_copy(0, positions, self.event_projection(pooled))
            cursor += len(positions)
        return self._combine_recovery(batch, output, present)
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/explain_endpoint_shift.py · lines 1–93

<!-- REVIEW_FRAGMENT {"path":"training/v5/explain_endpoint_shift.py","part":1,"start_line":1,"end_line":93,"source_sha256":"bafaba8652f4b5d77bd0e797950d63a6dac3ccc2ed377d319c8b157ad1bd8b34","fragment_sha256":"bafaba8652f4b5d77bd0e797950d63a6dac3ccc2ed377d319c8b157ad1bd8b34","added_terminal_newline":false} -->
````python
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
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/export_broad_labels.py · lines 1–109

<!-- REVIEW_FRAGMENT {"path":"training/v5/export_broad_labels.py","part":1,"start_line":1,"end_line":109,"source_sha256":"2c42ddd61ac07a79a20781019bae9414917a5b93bede2be944d18e66fe6f07fc","fragment_sha256":"2c42ddd61ac07a79a20781019bae9414917a5b93bede2be944d18e66fe6f07fc","added_terminal_newline":false} -->
````python
"""只读导出新的 COPD 编码再住院结局；不改动旧严格结局或特征。"""

from __future__ import annotations

import argparse
import csv
import getpass
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path


SQL = """COPY (
  SELECT subject_id,hadm_id,admittime,dischtime,event_type,time_days,
         next_copd_hadm_id,next_copd_best_code_seq_num,next_is_strict_aecopd,
         next_has_49122,next_has_j440
  FROM project_aecopd.copd_broad_readmission_labels_v1
  ORDER BY subject_id,admittime,hadm_id
) TO STDOUT WITH (FORMAT CSV,HEADER TRUE)"""


def export(output_dir: Path, psql: Path, password: str, *, sql: str = SQL,
           status_name: str = "broad_labels_exported_and_checked",
           primary_event: str = "first non-transfer admission with any preset COPD/emphysema code in any diagnosis position") -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not psql.is_file():
        raise FileNotFoundError(psql)
    output_dir.mkdir(parents=True)
    target = output_dir / "labels.csv.gz"
    partial = output_dir / "labels.csv.gz.partial"
    environment = os.environ.copy()
    environment["PGPASSWORD"] = password
    with gzip.open(partial, "wb", compresslevel=6) as destination:
        process = subprocess.Popen(
            [str(psql), "-X", "-q", "-w", "-h", "localhost", "-U", "postgres",
             "-d", "mimiciv", "-v", "ON_ERROR_STOP=1", "-c", sql],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
        )
        assert process.stdout is not None and process.stderr is not None
        shutil.copyfileobj(process.stdout, destination)
        error = process.stderr.read()
        status = process.wait()
    if status:
        raise RuntimeError(error.decode("utf-8", errors="replace"))
    partial.replace(target)
    counts: Counter[int] = Counter()
    subtypes: Counter[str] = Counter()
    positions: Counter[str] = Counter()
    hadms: set[str] = set()
    subjects: set[str] = set()
    with gzip.open(target, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            hadm = row["hadm_id"]
            if hadm in hadms:
                raise ValueError("duplicate label admission")
            hadms.add(hadm)
            subjects.add(row["subject_id"])
            event = int(row["event_type"])
            time = float(row["time_days"])
            if event not in (0, 1, 2) or not 0 <= time <= 365:
                raise ValueError("invalid competing event label")
            if event == 0 and time != 365:
                raise ValueError("administrative censoring time changed")
            counts[event] += 1
            if event == 1:
                sequence = int(row["next_copd_best_code_seq_num"])
                positions["principal" if sequence == 1 else
                          "secondary_2_3" if sequence <= 3 else "position_4_plus"] += 1
                subtypes["strict" if row["next_is_strict_aecopd"] == "t" else
                         "acute_infection_or_bronchitis" if row["next_has_49122"] == "t"
                         or row["next_has_j440"] == "t" else
                         "other_copd_or_emphysema"] += 1
    if len(hadms) != 19401 or len(subjects) != 5351:
        raise ValueError("broad label cohort changed")
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    report: dict[str, object] = {
        "status": status_name,
        "cohort_admissions": len(hadms), "cohort_patients": len(subjects),
        "events": {str(key): counts[key] for key in (0, 1, 2)},
        "readmission_subtypes": dict(subtypes),
        "readmission_copd_code_position": dict(positions),
        "label_sha256": sha,
        "primary_event": primary_event,
        "individual_followup_complete": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--psql", type=Path,
                        default=Path("C:/Program Files/PostgreSQL/18/bin/psql.exe"))
    args = parser.parse_args()
    password = os.environ.get("PGPASSWORD") or getpass.getpass("PostgreSQL password: ")
    print(json.dumps(export(args.output_dir, args.psql, password), ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/export_qualified_labels.py · lines 1–40

<!-- REVIEW_FRAGMENT {"path":"training/v5/export_qualified_labels.py","part":1,"start_line":1,"end_line":40,"source_sha256":"676af0ce8127e461b390b304fc91f6b795491c930ede43c2b94bfc10d1f814c2","fragment_sha256":"676af0ce8127e461b390b304fc91f6b795491c930ede43c2b94bfc10d1f814c2","added_terminal_newline":false} -->
````python
"""导出严格 AECOPD 任意位或既定 COPD／肺气肿前三位的并集再住院标签。"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

from training.v5.export_broad_labels import export


SQL = """COPY (
  SELECT subject_id,hadm_id,admittime,dischtime,event_type,time_days,
         next_qualified_hadm_id AS next_copd_hadm_id,
         next_copd_best_code_seq_num,next_is_strict_aecopd,
         next_has_49122,next_has_j440
  FROM project_aecopd.copd_qualified_readmission_labels_v1
  ORDER BY subject_id,admittime,hadm_id
) TO STDOUT WITH (FORMAT CSV,HEADER TRUE)"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--psql", type=Path,
                        default=Path("C:/Program Files/PostgreSQL/18/bin/psql.exe"))
    args = parser.parse_args()
    password = os.environ.get("PGPASSWORD") or getpass.getpass("PostgreSQL password: ")
    report = export(
        args.output_dir, args.psql, password, sql=SQL,
        status_name="qualified_labels_exported_and_checked",
        primary_event="first non-transfer readmission with strict 491.21/J44.1 at any position or preset COPD/emphysema code in diagnosis positions 1-3",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/extend_field_manifest.py · lines 1–184

<!-- REVIEW_FRAGMENT {"path":"training/v5/extend_field_manifest.py","part":1,"start_line":1,"end_line":184,"source_sha256":"ba0f93565f9acc116b048e855b9e7e578088799118d1281da3ffb1da7d07cd56","fragment_sha256":"ba0f93565f9acc116b048e855b9e7e578088799118d1281da3ffb1da7d07cd56","added_terminal_newline":false} -->
````python
"""将额外小结章节追加到冻结四字段清单；旧行与向量保持不变。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from training.v5.field_text_baseline import FIELD_LABELS, FIELDS


EXTRA_LABELS = {
    "discharge_medications": "Discharge Medications",
    "discharge_instructions": "Discharge Instructions",
    "social_history": "Social History",
    "pertinent_results": "Pertinent Results",
    "discharge_physical_exam": "Discharge Physical Exam",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def prepare(base_dir: Path, candidate_dir: Path, output_dir: Path,
            bert_tokenizer: object, qwen_tokenizer: object) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    base_path = base_dir / "shared_text.csv.gz"
    source_path = candidate_dir / "shared_text.csv.gz"
    base_qa = json.loads((base_dir / "validation.json").read_text(encoding="utf-8"))
    candidate_qa = json.loads((candidate_dir / "validation.json").read_text(encoding="utf-8"))
    if (base_qa["status"] != "field_text_manifest_validated"
            or base_qa["manifest_sha256"] != _sha(base_path)
            or candidate_qa["status"] != "manifest_structurally_validated"
            or candidate_qa["manifest_sha256"] != _sha(source_path)):
        raise ValueError("source manifests are not validated or have changed")
    candidate_summary = json.loads((candidate_dir / "summary.json").read_text(encoding="utf-8"))
    if tuple(candidate_summary["roles"]) != tuple(EXTRA_LABELS):
        raise ValueError("unexpected candidate role vocabulary")
    output_dir.mkdir(parents=True)
    candidate_path = output_dir / "candidate_text.csv.gz"
    combined_path = output_dir / "shared_text.csv.gz"
    counts: Counter[str] = Counter()
    admissions: set[str] = set()
    with (gzip.open(candidate_path, "wt", encoding="utf-8", newline="") as candidate_handle,
          gzip.open(combined_path, "wt", encoding="utf-8", newline="") as combined_handle):
        candidate_writer = csv.writer(candidate_handle)
        combined_writer = csv.writer(combined_handle)
        candidate_writer.writerow(FIELDS)
        combined_writer.writerow(FIELDS)
        for index, row in enumerate(_rows(base_path)):
            if int(row["embedding_row"]) != index or row["field"] not in FIELD_LABELS:
                raise ValueError("baseline row/order changed")
            combined_writer.writerow([row[name] for name in FIELDS])
            counts["baseline"] += 1
            admissions.add(row["hadm_id"])
        for source_index, source in enumerate(_rows(source_path)):
            role = source["section_role"]
            if role not in EXTRA_LABELS or int(source["embedding_row"]) != source_index:
                raise ValueError("candidate row/order changed")
            raw = source["chunk_text"]
            if hashlib.sha256(raw.encode("utf-8")).hexdigest() != source["text_sha256"]:
                raise ValueError("candidate content hash mismatch")
            text = f"Clinical field: {EXTRA_LABELS[role]}\n{raw}"
            bert_count = len(bert_tokenizer(text, add_special_tokens=True)["input_ids"])
            qwen_count = len(qwen_tokenizer(text, add_special_tokens=False)["input_ids"])
            if not 1 <= bert_count <= 512 or not 1 <= qwen_count <= 8192:
                raise ValueError("candidate text exceeds encoder context")
            common = (
                source["subject_id"], source["hadm_id"], role, source["section_id"],
                source["chunk_index"], source_index, source["source_start_char"],
                source["source_end_char"], source["source_section_sha256"],
                source["timestamp_relation"], bert_count, qwen_count,
                source["text_sha256"], hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
            )
            candidate_writer.writerow((source_index, *common))
            combined_writer.writerow((counts["baseline"] + source_index, *common))
            counts[role] += 1
            admissions.add(source["hadm_id"])
    if sum(counts[role] for role in EXTRA_LABELS) != candidate_qa["chunks"]:
        raise ValueError("candidate row count changed")
    summary = {
        "status": "screened_candidate_not_encoded",
        "fields": [*FIELD_LABELS, *EXTRA_LABELS],
        "baseline_rows": counts["baseline"],
        "candidate_rows": candidate_qa["chunks"],
        "chunks": counts["baseline"] + candidate_qa["chunks"],
        "admissions_with_text": len(admissions),
        "chunks_by_extra_field": {role: counts[role] for role in EXTRA_LABELS},
        "baseline_manifest_sha256": _sha(base_path),
        "source_candidate_manifest_sha256": _sha(source_path),
        "candidate_manifest_sha256": _sha(candidate_path),
        "manifest_sha256": _sha(combined_path),
        "clinical_semantics_validated": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def combine_vectors(base_dir: Path, extension_dir: Path, encoded_dir: Path) -> dict[str, object]:
    summary = json.loads((extension_dir / "summary.json").read_text(encoding="utf-8"))
    candidate_path = extension_dir / "candidate_text.csv.gz"
    combined_path = extension_dir / "shared_text.csv.gz"
    candidate_state = json.loads((encoded_dir / "state.json").read_text(encoding="utf-8"))
    if (_sha(candidate_path) != summary["candidate_manifest_sha256"]
            or _sha(combined_path) != summary["manifest_sha256"]
            or candidate_state["status"] != "complete"
            or candidate_state["manifest_sha256"] != summary["candidate_manifest_sha256"]):
        raise ValueError("candidate encoding or manifest incomplete")
    combined_rows = _rows(combined_path)
    for original, extended in zip(_rows(base_dir / "shared_text.csv.gz"), combined_rows):
        if original != extended:
            raise ValueError("baseline text row changed in extension")
    old = np.load(base_dir / "bert/bioclinicalbert.float16.npy", mmap_mode="r")
    new = np.load(encoded_dir / "bioclinicalbert.float16.npy", mmap_mode="r")
    if old.shape != (summary["baseline_rows"], 768) or new.shape != (summary["candidate_rows"], 768):
        raise ValueError("embedding shape mismatch")
    target_dir = extension_dir / "bert"
    if target_dir.exists():
        raise FileExistsError(target_dir)
    target_dir.mkdir()
    target = np.lib.format.open_memmap(target_dir / "bioclinicalbert.float16.npy", mode="w+",
                                       dtype=np.float16, shape=(summary["chunks"], 768))
    target[:len(old)] = old
    target[len(old):] = new
    target.flush()
    if not np.array_equal(target[:len(old)], old) or not np.array_equal(target[len(old):], new):
        raise ValueError("combined vectors differ from sources")
    state = {"status": "complete", "model": "Bio_ClinicalBERT", "device": "cuda",
             "rows": summary["chunks"], "completed_rows": summary["chunks"],
             "dimension": 768, "manifest_sha256": summary["manifest_sha256"],
             "baseline_vector_sha256": _sha(base_dir / "bert/bioclinicalbert.float16.npy"),
             "candidate_vector_sha256": _sha(encoded_dir / "bioclinicalbert.float16.npy")}
    (target_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    validation = {"status": "field_text_manifest_validated",
                  "manifest_sha256": summary["manifest_sha256"],
                  "chunks": summary["chunks"], "admissions": summary["admissions_with_text"],
                  "baseline_rows_unchanged": True, "clinical_semantics_validated": False}
    (extension_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "combine-vectors"))
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--encoded-dir", type=Path)
    parser.add_argument("--bert-model-dir", type=Path)
    parser.add_argument("--qwen-model-dir", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        from transformers import AutoTokenizer
        if args.candidate_dir is None or args.bert_model_dir is None or args.qwen_model_dir is None:
            parser.error("prepare requires candidate and both tokenizer directories")
        bert = AutoTokenizer.from_pretrained(args.bert_model_dir, local_files_only=True)
        qwen = AutoTokenizer.from_pretrained(args.qwen_model_dir, local_files_only=True)
        result = prepare(args.base_dir, args.candidate_dir, args.extension_dir, bert, qwen)
    else:
        if args.encoded_dir is None:
            parser.error("combine-vectors requires encoded directory")
        result = combine_vectors(args.base_dir, args.extension_dir, args.encoded_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/field_text_baseline.py · lines 1–185

<!-- REVIEW_FRAGMENT {"path":"training/v5/field_text_baseline.py","part":1,"start_line":1,"end_line":185,"source_sha256":"1a94f14cdab0d851e626b539c0ce9bc798c0ebd2ae07e7c3d264e3e38181f86c","fragment_sha256":"1a94f14cdab0d851e626b539c0ce9bc798c0ebd2ae07e7c3d264e3e38181f86c","added_terminal_newline":false} -->
````python
"""生成独立字段文本分支的共同 BERT／Qwen 输入，不构造章节—结构化配对。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterator, Protocol

from training.v5.section_alignment_data.screen_section_content import screen_flags


FIELD_LABELS = {
    "history_of_present_illness": "History of Present Illness",
    "past_medical_history": "Past Medical History",
    "hospital_course": "Hospital Course",
    "discharge_condition": "Discharge Condition",
}
FIELDS = (
    "embedding_row", "subject_id", "hadm_id", "field", "section_id", "chunk_index",
    "source_manifest_row", "source_start_char", "source_end_char", "source_section_sha256",
    "timestamp_relation", "bert_token_count", "token_count", "source_chunk_sha256",
    "text_sha256", "chunk_text",
)


class Tokenizer(Protocol):
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]: ...


def _rows(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def field_text(role: str, raw_chunk: str) -> str:
    """字段名只是文本上下文，既不声称有结构化证据，也不加入结局信息。"""

    if role not in FIELD_LABELS or not raw_chunk.strip():
        raise ValueError("unsupported or empty field text")
    return f"Clinical field: {FIELD_LABELS[role]}\n{raw_chunk}"


def eligible(role: str, original: str, timestamp_relation: str) -> tuple[bool, str]:
    """保守筛查整段来源；不把标记解释为临床真值。"""

    if role not in FIELD_LABELS:
        return False, "outside_first_baseline"
    flags = set(screen_flags(role, original, timestamp_relation))
    if role == "history_of_present_illness" and "administrative_header_first_500_chars" in flags:
        return False, "hpi_administrative_header_review"
    if role == "discharge_condition":
        if "condition_over_500_chars_review_boundary" in flags:
            return False, "long_condition_boundary_review"
        if "generic_condition_only" in flags:
            return False, "generic_condition_only"
        if "function_or_respiratory_cue" not in flags:
            return False, "condition_without_selected_cue"
    return True, "screened_candidate_not_clinically_validated"


def prepare(
    aligned_dir: Path, candidate_dir: Path, output_dir: Path,
    bert_tokenizer: Tokenizer, qwen_tokenizer: Tokenizer,
) -> dict[str, object]:
    """同一字段清单给两个冻结编码器；拒绝覆盖、超长和未校验来源。"""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    aligned_qa = json.loads((aligned_dir / "qa_validation.json").read_text(encoding="utf-8"))
    candidate_qa = json.loads((candidate_dir / "validation.json").read_text(encoding="utf-8"))
    source_sections = aligned_dir / "note_sections.csv.gz"
    source_manifest = candidate_dir / "shared_text.csv.gz"
    if aligned_qa["status"] != "structural_checks_passed" or candidate_qa["status"] != "manifest_structurally_validated":
        raise ValueError("unvalidated source data")
    if _sha(source_sections) != aligned_qa["output_sha256"][source_sections.name]:
        raise ValueError("source sections changed")
    if _sha(source_manifest) != candidate_qa["manifest_sha256"]:
        raise ValueError("source chunk manifest changed")

    section_status: dict[str, tuple[bool, str]] = {}
    for row in _rows(source_sections):
        if row["section_role"] not in FIELD_LABELS:
            continue
        section_status[row["section_id"]] = eligible(
            row["section_role"], row["text"], row["timestamp_relation"]
        )
    output_dir.mkdir(parents=True)
    counts: Counter[str] = Counter()
    covered_hadms: set[str] = set()
    emitted_sections: set[str] = set()
    max_bert = max_qwen = 0
    with gzip.open(output_dir / "shared_text.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for source_row, row in enumerate(_rows(source_manifest)):
            role = row["section_role"]
            if role not in FIELD_LABELS:
                continue
            if int(row["embedding_row"]) != source_row:
                raise ValueError("non-contiguous source manifest rows")
            status = section_status.get(row["section_id"])
            if status is None:
                raise ValueError("source chunk references unknown field section")
            if not status[0]:
                counts["excluded_" + status[1]] += 1
                continue
            raw = row["chunk_text"]
            if hashlib.sha256(raw.encode("utf-8")).hexdigest() != row["text_sha256"]:
                raise ValueError("source chunk content hash mismatch")
            text = field_text(role, raw)
            bert_count = len(bert_tokenizer(text, add_special_tokens=True)["input_ids"])
            qwen_count = len(qwen_tokenizer(text, add_special_tokens=False)["input_ids"])
            if not 1 <= bert_count <= 512 or not 1 <= qwen_count <= 8192:
                raise ValueError("field text exceeds frozen encoder context")
            writer.writerow((
                counts["chunks"], row["subject_id"], row["hadm_id"], role,
                row["section_id"], row["chunk_index"], source_row,
                row["source_start_char"], row["source_end_char"],
                row["source_section_sha256"], row["timestamp_relation"],
                bert_count, qwen_count, row["text_sha256"],
                hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
            ))
            counts["chunks"] += 1
            counts["chunks_" + role] += 1
            covered_hadms.add(row["hadm_id"])
            emitted_sections.add(row["section_id"])
            max_bert = max(max_bert, bert_count)
            max_qwen = max(max_qwen, qwen_count)
    if not counts["chunks"]:
        raise ValueError("no field text selected")
    manifest = output_dir / "shared_text.csv.gz"
    result: dict[str, object] = {
        "status": "screened_candidate_not_encoded",
        "cohort_source_sections_sha256": _sha(source_sections),
        "source_candidate_manifest_sha256": _sha(source_manifest),
        "manifest_sha256": _sha(manifest),
        "fields": list(FIELD_LABELS),
        "admissions_with_text": len(covered_hadms),
        "source_sections": len(emitted_sections),
        "chunks": counts["chunks"],
        "chunks_by_field": {role: counts["chunks_" + role] for role in FIELD_LABELS},
        "excluded_chunks_by_reason": {
            key.removeprefix("excluded_"): value
            for key, value in sorted(counts.items()) if key.startswith("excluded_")
        },
        "max_bert_tokens": max_bert,
        "max_qwen_tokens": max_qwen,
        "clinical_semantics_validated": False,
        "outcome_labels_exported": False,
        "paired_structured_facts_assumed": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bert-model-dir", type=Path, default=Path("/model-cache/Bio_ClinicalBERT"))
    parser.add_argument("--qwen-model-dir", type=Path, default=Path("/model-cache/Qwen3-Embedding-4B"))
    args = parser.parse_args()
    bert = AutoTokenizer.from_pretrained(args.bert_model_dir, local_files_only=True)
    qwen = AutoTokenizer.from_pretrained(args.qwen_model_dir, local_files_only=True)
    print(json.dumps(prepare(args.aligned_dir, args.candidate_dir, args.output_dir, bert, qwen), ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/finish_all_lab_axes.ps1 · lines 1–21

<!-- REVIEW_FRAGMENT {"path":"training/v5/finish_all_lab_axes.ps1","part":1,"start_line":1,"end_line":21,"source_sha256":"79bf17b6f7fd9c375262c6631fc7e3ffcbe664e77569420b5582920f55c35fe8","fragment_sha256":"79bf17b6f7fd9c375262c6631fc7e3ffcbe664e77569420b5582920f55c35fe8","added_terminal_newline":false} -->
````powershell
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $root
$runRoot = Join-Path $root 'training_runs/v5_strict_all22_binary_meds_2026-09-30'
$exitCode = & docker wait aecopd-lab22-binary-20260930
& docker logs aecopd-lab22-binary-20260930 *> (Join-Path $runRoot 'training.log')
if ($LASTEXITCODE -ne 0 -or "$exitCode".Trim() -ne '0') {
    throw "Lab-axis batch failed; inspect $runRoot/training.log"
}
$analysis = Join-Path $runRoot 'paired_analysis/paired_results.json'
if (Test-Path -LiteralPath $analysis) {
    $saved = Get-Content -Raw -LiteralPath $analysis | ConvertFrom-Json
    if ($saved.status -ne 'completed' -or $saved.rows.Count -ne 176) { throw 'Incomplete paired analysis' }
    return
}
& docker run --rm -v "${root}:/workspace" -w /workspace -e PYTHONPATH=/workspace `
    -e MPLCONFIGDIR=/tmp/matplotlib aecopd-training:v1 python -u -m training.v5.compare_all_lab_axes `
    --run-root training_runs/v5_strict_all22_binary_meds_2026-09-30 `
    --output-dir training_runs/v5_strict_all22_binary_meds_2026-09-30/paired_analysis `
    *> (Join-Path $runRoot 'analysis.log')
if ($LASTEXITCODE -ne 0) { throw 'Paired analysis failed; inspect analysis.log' }
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/independent_data.py · lines 1–327

<!-- REVIEW_FRAGMENT {"path":"training/v5/independent_data.py","part":1,"start_line":1,"end_line":327,"source_sha256":"0d8a711690d40ff6e3d7e1918300cdf5dfec93fbf36ff439a5ea6e009bfd20c7","fragment_sha256":"0d8a711690d40ff6e3d7e1918300cdf5dfec93fbf36ff439a5ea6e009bfd20c7","added_terminal_newline":false} -->
````python
"""把已校验的稀疏结构化事实与独立字段文本接入同一出院序列。"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_fusion import IndependentBatch
from training.v5.numeric_preprocessing import robust_parameters


def _rows(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class IndependentData:
    subject_ids: np.ndarray
    hadm_ids: np.ndarray
    splits: np.ndarray
    times: np.ndarray
    events: np.ndarray
    gap: np.ndarray
    sequences: tuple[np.ndarray, ...]
    token_values: np.ndarray
    token_masks: np.ndarray
    token_concepts: np.ndarray
    token_types: np.ndarray
    tokens_by_admission: tuple[np.ndarray, ...]
    concept_names: tuple[str, ...]
    type_names: tuple[str, ...]
    text_by_admission: tuple[tuple[tuple[int, int], ...], ...]
    text_vectors: np.ndarray | None
    text_width: int
    manifest_sha256: str | None
    label_sha256: str | None
    label_name: str | None
    disabled_concepts: tuple[str, ...] = ()
    field_roles: tuple[str, ...] = tuple(FIELD_LABELS)


def load_data(
    aligned_dir: Path, structured_dir: Path, base_csv: Path,
    field_dir: Path | None = None, vectors_path: Path | None = None,
    labels_csv: Path | None = None,
) -> IndependentData:
    """结构化 token 不按章节与文本配对；可用已校验标签覆盖旧 base 结局。"""

    if (field_dir is None) != (vectors_path is None):
        raise ValueError("field manifest and vectors must be supplied together")
    validated = json.loads((structured_dir / "validation.json").read_text(encoding="utf-8"))
    token_path = structured_dir / "structured_tokens.csv.gz"
    if validated["status"] != "section_inputs_validated" or _sha(token_path) != validated["output_sha256"][token_path.name]:
        raise ValueError("structured token data unvalidated or changed")
    cohort = {row["hadm_id"]: row for row in _rows(aligned_dir / "cohort_manifest.csv.gz")}
    base = list(_rows(base_csv))
    if len(base) != 19401 or len(cohort) != len(base):
        raise ValueError("V5 full discharge cohort mismatch")
    hadms = np.array([int(row["hadm_id"]) for row in base], dtype=np.int64)
    subjects = np.array([int(row["subject_id"]) for row in base], dtype=np.int64)
    if len(set(hadms.tolist())) != len(base):
        raise ValueError("duplicate discharge admission")
    row_by_hadm = {str(hadm): index for index, hadm in enumerate(hadms)}
    label_sha: str | None = None
    label_name: str | None = None
    labels: dict[str, dict[str, str]] | None = None
    if labels_csv is not None:
        label_qa = json.loads((labels_csv.parent / "summary.json").read_text(encoding="utf-8"))
        label_sha = _sha(labels_csv)
        if (label_qa["status"] not in {"broad_labels_exported_and_checked",
                                       "qualified_labels_exported_and_checked",
                                       "strict_labels_reconstructed_and_checked"}
                or label_qa["label_sha256"] != label_sha
                or label_qa["cohort_admissions"] != len(base)
                or not label_qa.get("primary_event")):
            raise ValueError("outcome labels are unvalidated or changed")
        label_name = str(label_qa["primary_event"])
        labels = {row["hadm_id"]: row for row in _rows(labels_csv)}
        if len(labels) != len(base) or set(labels) != set(row_by_hadm):
            raise ValueError("outcome admission keys differ")
    splits = np.empty(len(base), dtype="U10")
    times = np.empty(len(base), dtype=np.float32)
    events = np.empty(len(base), dtype=np.int64)
    gap = np.zeros((len(base), 2), dtype=np.float32)
    patient_rows: dict[int, list[int]] = defaultdict(list)
    previous_discharge: dict[int, datetime] = {}
    for index, row in enumerate(base):
        hadm, subject = row["hadm_id"], int(row["subject_id"])
        manifest = cohort.get(hadm)
        if manifest is None or manifest["subject_id"] != row["subject_id"]:
            raise ValueError("cohort identifier mismatch")
        if manifest["admittime"] != row["admittime"] or manifest["dischtime"] != row["dischtime"]:
            raise ValueError("cohort prediction time mismatch")
        splits[index] = manifest["patient_split"]
        outcome = labels[hadm] if labels is not None else row
        if (outcome["subject_id"] != row["subject_id"]
                or outcome["admittime"] != row["admittime"]
                or outcome["dischtime"] != row["dischtime"]):
            raise ValueError("broad outcome identity or landmark mismatch")
        times[index], events[index] = float(outcome["time_days"]), int(outcome["event_type"])
        admission = datetime.fromisoformat(row["admittime"])
        discharge = datetime.fromisoformat(row["dischtime"])
        if not discharge > admission:
            raise ValueError("ineligible discharge in base")
        prior = previous_discharge.get(subject)
        if prior is not None:
            gap_days = max(0.0, (admission - prior).total_seconds() / 86400.0)
            gap[index] = (math.log1p(gap_days), float(admission < prior))
        previous_discharge[subject] = discharge
        patient_rows[subject].append(index)
    if not np.isfinite(times).all() or np.any((times < 0) | (times > 365)):
        raise ValueError("invalid event times")
    if not np.isin(events, (0, 1, 2)).all() or np.any((events == 0) & (times != 365)):
        raise ValueError("invalid event/censoring labels")
    sequences = tuple(np.asarray(rows, dtype=np.int64) for rows in patient_rows.values())
    if any(len(set(splits[sequence].tolist())) != 1 for sequence in sequences):
        raise ValueError("patient crosses train/validation/test splits")
    for sequence in sequences:
        admissions = [base[index]["admittime"] for index in sequence]
        if admissions != sorted(admissions):
            raise ValueError("patient admissions not chronological")

    token_rows = list(_rows(token_path))
    if len(token_rows) != validated["tokens"]:
        raise ValueError("structured token count drift")
    concept_names = tuple(sorted({row["token_type"] + ":" + row["concept"] for row in token_rows}))
    type_names = tuple(sorted({row["token_type"] for row in token_rows}))
    concept_id = {name: index + 1 for index, name in enumerate(concept_names)}
    type_id = {name: index + 1 for index, name in enumerate(type_names)}
    token_values = np.zeros((len(token_rows), 5), dtype=np.float32)
    token_masks = np.zeros_like(token_values)
    token_concepts = np.zeros(len(token_rows), dtype=np.int64)
    token_types = np.zeros(len(token_rows), dtype=np.int64)
    token_refs: list[list[int]] = [[] for _ in base]
    for position, row in enumerate(token_rows):
        hadm = row["hadm_id"]
        index = row_by_hadm.get(hadm)
        if index is None or row["subject_id"] != base[index]["subject_id"]:
            raise ValueError("structured token outside cohort")
        token_values[position] = [float(row[f"value{channel}"]) for channel in range(5)]
        token_masks[position] = [float(row[f"mask{channel}"]) for channel in range(5)]
        token_concepts[position] = concept_id[row["token_type"] + ":" + row["concept"]]
        token_types[position] = type_id[row["token_type"]]
        token_refs[index].append(position)
    if not np.isfinite(token_values).all() or not np.isin(token_masks, (0, 1)).all():
        raise ValueError("invalid structured token numerical values")
    if any(not refs for refs in token_refs):
        raise ValueError("admission lacks patient-context tokens")

    text_refs: list[list[tuple[int, int]]] = [[] for _ in base]
    vectors: np.ndarray | None = None
    text_width = 768
    manifest_sha: str | None = None
    field_roles = tuple(FIELD_LABELS)
    if field_dir is not None and vectors_path is not None:
        field_qa = json.loads((field_dir / "validation.json").read_text(encoding="utf-8"))
        field_summary = json.loads((field_dir / "summary.json").read_text(encoding="utf-8"))
        field_roles = tuple(field_summary["fields"])
        if (not field_roles or len(set(field_roles)) != len(field_roles)
                or field_roles[:len(FIELD_LABELS)] != tuple(FIELD_LABELS)):
            raise ValueError("field vocabulary must preserve the four baseline roles")
        manifest_path = field_dir / "shared_text.csv.gz"
        manifest_sha = _sha(manifest_path)
        state = json.loads((vectors_path.parent / "state.json").read_text(encoding="utf-8"))
        if (field_qa["status"] != "field_text_manifest_validated"
                or field_qa["manifest_sha256"] != manifest_sha
                or field_summary["manifest_sha256"] != manifest_sha):
            raise ValueError("unvalidated field-text manifest")
        if state.get("status") != "complete" or state.get("manifest_sha256") != manifest_sha:
            raise ValueError("incomplete or mismatched frozen text encoding")
        vectors = np.load(vectors_path, mmap_mode="r")
        if vectors.ndim != 2 or vectors.shape[1] not in (768, 2560):
            raise ValueError("unexpected text vector dimension")
        if vectors.shape[1] == 768 and state.get("model") != "Bio_ClinicalBERT":
            raise ValueError("BERT encoder metadata mismatch")
        if vectors.shape[1] == 2560 and state.get("encoder") != "Qwen3-Embedding-4B":
            raise ValueError("Qwen encoder metadata mismatch")
        text_width = int(vectors.shape[1])
        field_id = {role: index for index, role in enumerate(field_roles)}
        text_count = 0
        for row_number, row in enumerate(_rows(manifest_path)):
            index = row_by_hadm.get(row["hadm_id"])
            if index is None or row["subject_id"] != base[index]["subject_id"]:
                raise ValueError("text field outside cohort")
            if int(row["embedding_row"]) != row_number or row["field"] not in field_id:
                raise ValueError("field text row/order mismatch")
            text_refs[index].append((field_id[row["field"]], row_number))
            text_count += 1
        if text_count != vectors.shape[0]:
            raise ValueError("text vector row count mismatch")
    return IndependentData(
        subjects, hadms, splits, times, events, gap, sequences,
        token_values, token_masks, token_concepts, token_types,
        tuple(np.asarray(refs, dtype=np.int64) for refs in token_refs),
        concept_names, type_names, tuple(tuple(refs) for refs in text_refs),
        vectors, text_width, manifest_sha, label_sha, label_name,
        field_roles=field_roles,
    )


def normalize_on_training_patients(data: IndependentData) -> tuple[IndependentData, dict[str, object]]:
    """按概念／数值通道用训练患者拟合中位数和 IQR，其他折不参与。"""

    train_hadms = {int(hadm) for hadm in data.hadm_ids[data.splits == "train"]}
    token_hadm = np.empty(len(data.token_values), dtype=np.int64)
    for admission, refs in enumerate(data.tokens_by_admission):
        token_hadm[refs] = data.hadm_ids[admission]
    train = np.isin(token_hadm, list(train_hadms))
    transformed = data.token_values.copy()
    parameters: dict[str, object] = {}
    for concept_id, concept_name in enumerate(data.concept_names, start=1):
        selected = data.token_concepts == concept_id
        medians: list[float] = []
        scales: list[float] = []
        for channel in range(5):
            observed = data.token_values[selected & train & (data.token_masks[:, channel] > 0), channel]
            if observed.size:
                fitted = robust_parameters(observed)
                median, scale = float(fitted["median"]), float(fitted["scale"])
                # 二元编码不是连续量；不因稀有阳性而用标准差放大它。
                if np.all(np.isin(observed, (0.0, 1.0))):
                    scale = 1.0
            else:
                median, scale = 0.0, 1.0
            medians.append(median)
            scales.append(scale)
            keep = selected & (data.token_masks[:, channel] > 0)
            transformed[keep, channel] = np.clip((data.token_values[keep, channel] - median) / scale,
                                                  -8.0, 8.0)
        parameters[concept_name] = {"median": medians, "scale": scales}
    transformed[data.token_masks == 0] = 0.0
    if not np.isfinite(transformed).all():
        raise ValueError("nonfinite normalized structured token")
    return replace(data, token_values=transformed), parameters


def split_sequences(data: IndependentData, split: str) -> list[np.ndarray]:
    selected = [sequence for sequence in data.sequences if data.splits[sequence[0]] == split]
    if not selected:
        raise ValueError(f"empty {split} partition")
    return selected


def collate(data: IndependentData, sequences: list[np.ndarray], *, include_text: bool) -> IndependentBatch:
    if not sequences:
        raise ValueError("empty patient batch")
    batch_size = len(sequences)
    length = max(len(sequence) for sequence in sequences)
    slots = max(len(data.tokens_by_admission[int(row)]) for sequence in sequences for row in sequence)
    values = np.zeros((batch_size, length, slots, 5), dtype=np.float32)
    masks = np.zeros_like(values)
    concepts = np.zeros((batch_size, length, slots), dtype=np.int64)
    kinds = np.zeros_like(concepts)
    token_present = np.zeros((batch_size, length, slots), dtype=bool)
    valid = np.zeros((batch_size, length), dtype=bool)
    gap = np.zeros((batch_size, length, 2), dtype=np.float32)
    times = np.zeros((batch_size, length), dtype=np.float32)
    events = np.zeros((batch_size, length), dtype=np.int64)
    snippets: list[tuple[int, int, int]] = []
    disabled_ids = [index + 1 for index, name in enumerate(data.concept_names)
                    if name in data.disabled_concepts]
    if set(data.disabled_concepts) - set(data.concept_names):
        raise ValueError("disabled static concept absent from vocabulary")
    for batch_index, sequence in enumerate(sequences):
        count = len(sequence)
        valid[batch_index, :count] = True
        gap[batch_index, :count] = data.gap[sequence]
        times[batch_index, :count] = data.times[sequence]
        events[batch_index, :count] = data.events[sequence]
        for position, row in enumerate(sequence):
            refs = data.tokens_by_admission[int(row)]
            count_tokens = len(refs)
            values[batch_index, position, :count_tokens] = data.token_values[refs]
            masks[batch_index, position, :count_tokens] = data.token_masks[refs]
            concepts[batch_index, position, :count_tokens] = data.token_concepts[refs]
            kinds[batch_index, position, :count_tokens] = data.token_types[refs]
            enabled = ~np.isin(data.token_concepts[refs], disabled_ids)
            token_present[batch_index, position, :count_tokens] = enabled
            # 保留槽位和词表尺寸，完全屏蔽字段，保持共有层初始化和 dropout 形状。
            inactive = np.flatnonzero(~enabled)
            values[batch_index, position, inactive] = 0
            masks[batch_index, position, inactive] = 0
            concepts[batch_index, position, inactive] = 0
            kinds[batch_index, position, inactive] = 0
            if include_text:
                snippets.extend((batch_index * length + position, field, vector_row)
                                for field, vector_row in data.text_by_admission[int(row)])
    if snippets:
        if data.text_vectors is None:
            raise ValueError("text requested without frozen vectors")
        positions = np.asarray(snippets, dtype=np.int64)
        vectors = np.asarray(data.text_vectors[positions[:, 2]], dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("nonfinite frozen text vector")
        index = positions[:, :2]
    else:
        vectors = np.zeros((0, data.text_width), dtype=np.float32)
        index = np.zeros((0, 2), dtype=np.int64)
    return IndependentBatch(
        torch.from_numpy(values), torch.from_numpy(masks), torch.from_numpy(concepts),
        torch.from_numpy(kinds), torch.from_numpy(token_present),
        torch.from_numpy(vectors), torch.from_numpy(index),
        torch.from_numpy(gap), torch.from_numpy(valid),
        torch.from_numpy(times), torch.from_numpy(events),
    )
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/independent_fusion.py · lines 1–131

<!-- REVIEW_FRAGMENT {"path":"training/v5/independent_fusion.py","part":1,"start_line":1,"end_line":131,"source_sha256":"83128642af2b2e52c9f6d0eb057edeba02928d70039ba7873870a3b5c1d40eb7","fragment_sha256":"83128642af2b2e52c9f6d0eb057edeba02928d70039ba7873870a3b5c1d40eb7","added_terminal_newline":false} -->
````python
"""结构化事实与字段文本分别编码，只在住院级融合的竞争风险模型。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from training.v5.model import V5Output


@dataclass(frozen=True)
class IndependentBatch:
    """一批完整患者历史；文本索引为 [展平住院位置, 字段编号]。"""

    token_values: torch.Tensor        # [B, L, S, 5]
    token_masks: torch.Tensor         # [B, L, S, 5]
    token_concepts: torch.Tensor      # [B, L, S]
    token_types: torch.Tensor         # [B, L, S]
    token_present: torch.Tensor       # [B, L, S]
    snippet_vectors: torch.Tensor    # [N, text_width]
    snippet_index: torch.Tensor      # [N, 2]
    gap: torch.Tensor                # [B, L, 2]
    valid: torch.Tensor              # [B, L]
    times: torch.Tensor              # [B, L]
    events: torch.Tensor             # [B, L]

    def to(self, device: torch.device | str) -> IndependentBatch:
        return IndependentBatch(*(value.to(device) for value in (
            self.token_values, self.token_masks, self.token_concepts,
            self.token_types, self.token_present, self.snippet_vectors,
            self.snippet_index, self.gap, self.valid, self.times, self.events,
        )))


class IndependentFusionModel(nn.Module):
    """两个住院级独立分支；没有章节—结构化 token 的逐项注意力。"""

    def __init__(self, *, concept_count: int, type_count: int, field_count: int,
                 text_width: int, width: int = 128) -> None:
        super().__init__()
        if min(concept_count, type_count, field_count, text_width) < 1 or width % 4:
            raise ValueError("invalid model vocabulary or width")
        self.concept_embedding = nn.Embedding(concept_count + 1, width, padding_idx=0)
        self.type_embedding = nn.Embedding(type_count + 1, width, padding_idx=0)
        self.value_projection = nn.Sequential(nn.Linear(10, width), nn.GELU(), nn.LayerNorm(width))
        self.structured_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.1, batch_first=True),
            num_layers=1, enable_nested_tensor=False,
        )
        self.structured_score = nn.Linear(width, 1)
        self.field_embedding = nn.Embedding(field_count, width)
        self.text_score = nn.Linear(width, 1)
        self.text_content_norm = nn.LayerNorm(width)
        self.text_norm = nn.LayerNorm(width)
        self.fusion_gate = nn.Linear(2 * width, 1)
        nn.init.constant_(self.fusion_gate.bias, -1.0)
        self.fusion_norm = nn.LayerNorm(width)
        self.gap_projection = nn.Linear(2, width)
        self.patient_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.1, batch_first=True),
            num_layers=1, enable_nested_tensor=False,
        )
        self.risk_head = nn.Sequential(nn.Linear(width, 32), nn.ReLU(),
                                       nn.Dropout(0.2), nn.Linear(32, 2))
        self.raw_shape = nn.Parameter(torch.zeros(2))
        with torch.no_grad():
            self.risk_head[-1].bias.copy_(torch.tensor([-1.5, -3.0]))
        # 维度变化只影响最后初始化的文本投影；其余参数的初始值可逐项比较。
        self.text_projection = nn.Linear(text_width, width)

    def _structured(self, batch: IndependentBatch) -> torch.Tensor:
        batch_size, length, slots = batch.token_present.shape
        flat_stays = batch_size * length
        present = batch.token_present.reshape(flat_stays, slots)
        safe = present.clone()
        safe[~present.any(dim=1), 0] = True
        value = torch.cat((batch.token_values, batch.token_masks), dim=-1)
        value = value.reshape(flat_stays, slots, 10)
        tokens = (self.value_projection(value)
                  + self.concept_embedding(batch.token_concepts.reshape(flat_stays, slots))
                  + self.type_embedding(batch.token_types.reshape(flat_stays, slots)))
        tokens = self.structured_encoder(tokens, src_key_padding_mask=~safe)
        score = self.structured_score(tokens).squeeze(-1).masked_fill(~safe, -1e4)
        weight = torch.softmax(score, dim=1)
        pooled = (tokens * weight.unsqueeze(-1)).sum(dim=1)
        return pooled * present.any(dim=1).unsqueeze(-1)

    def _text(self, batch: IndependentBatch, flat_stays: int) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.snippet_vectors.shape[0] == 0:
            zero = batch.token_values.new_zeros((flat_stays, self.text_projection.out_features))
            return zero, torch.zeros(flat_stays, dtype=torch.bool, device=zero.device)
        segment = batch.snippet_index[:, 0]
        field = batch.snippet_index[:, 1]
        if bool((segment < 0).any()) or bool((segment >= flat_stays).any()):
            raise ValueError("snippet references admission outside batch")
        if bool((field < 0).any()) or bool((field >= self.field_embedding.num_embeddings).any()):
            raise ValueError("unknown text field")
        # 冻结的编码向量通常为单位范数；先平衡内容尺度，再加入字段身份。
        content = self.text_content_norm(self.text_projection(batch.snippet_vectors))
        projected = self.text_norm(content + self.field_embedding(field))
        score = self.text_score(torch.tanh(projected)).squeeze(-1)
        maximum = torch.full((flat_stays,), -torch.inf, device=score.device)
        maximum.scatter_reduce_(0, segment, score, reduce="amax", include_self=True)
        weight = torch.exp(score - maximum[segment])
        totals = torch.zeros((flat_stays,), device=score.device)
        totals.scatter_add_(0, segment, weight)
        weight = weight / totals[segment].clamp_min(1e-8)
        pooled = torch.zeros((flat_stays, projected.shape[-1]), device=score.device)
        pooled.index_add_(0, segment, weight.unsqueeze(-1) * projected)
        present = totals > 0
        return pooled, present

    def forward(self, batch: IndependentBatch) -> V5Output:
        batch_size, length = batch.valid.shape
        flat_stays = batch_size * length
        structured = self._structured(batch)
        text, text_present = self._text(batch, flat_stays)
        gate = torch.sigmoid(self.fusion_gate(torch.cat((structured, text), dim=-1)))
        gate = gate * text_present.unsqueeze(-1)
        admission = self.fusion_norm(structured + gate * text)
        admission = admission.view(batch_size, length, -1) + self.gap_projection(batch.gap)
        causal = torch.triu(torch.ones((length, length), dtype=torch.bool,
                                       device=batch.valid.device), diagonal=1)
        patient = self.patient_encoder(admission, mask=causal,
                                       src_key_padding_mask=~batch.valid)
        log_amplitude = self.risk_head(patient).clamp(-10.0, 6.0)
        shape = F.softplus(self.raw_shape) + 0.1
        return V5Output(log_amplitude, shape, gate.view(batch_size, length, 1))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/lab_recovery_data.py · lines 1–153

<!-- REVIEW_FRAGMENT {"path":"training/v5/lab_recovery_data.py","part":1,"start_line":1,"end_line":153,"source_sha256":"ca615c5305addbe3a918b2c701374b175dd517c9db38d9f71d710ee1ae183227","fragment_sha256":"ca615c5305addbe3a918b2c701374b175dd517c9db38d9f71d710ee1ae183227","added_terminal_newline":false} -->
````python
"""把已审计的住院级异常恢复特征按 hadm_id 对齐到固定 V5 队列。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from patient_data.lab_recovery_features import FEATURE_NAMES
from patient_data.expanded_abnormality_features import FEATURE_NAMES as ABNORMALITY_FEATURE_NAMES
from patient_data.expanded_abnormality_features import AXES as ABNORMALITY_AXES
from training.v5.event_data import EventData


ABNORMALITY_GROUPS = {
    "ventilation_acid_base": ("arterial_low_ph", "arterial_high_pco2", "arterial_hypercapnic_acidosis",
                              "bicarbonate_low", "bicarbonate_high"),
    "inflammation_infection": ("wbc_high", "wbc_low", "crp_high"),
    "nutrition_hematology": ("albumin_low", "hemoglobin_low", "platelet_low"),
    "renal": ("creatinine_high", "urea_nitrogen_high"),
    "electrolytes": ("sodium_low", "sodium_high", "potassium_low", "potassium_high", "magnesium_low"),
    "metabolic_perfusion": ("glucose_low", "glucose_high", "lactate_high"),
    "cardiac_burden": ("ntprobnp_high",),
}
SELECTED_REMOVED_AXES = (*ABNORMALITY_GROUPS["renal"], *ABNORMALITY_GROUPS["metabolic_perfusion"])
# 临床先筛：这些轴不是 AECOPD 核心指标，或与其他轴明显重合；是否再删由严格结局单轴实验决定。
SELECTED_AXIS_SCREEN = ("arterial_hypercapnic_acidosis", "platelet_low", "magnesium_low", "ntprobnp_high")
ABNORMALITY_ABLATIONS = {
    **ABNORMALITY_GROUPS,
    # 完整词表逐轴留一；不同于旧 selected 基线上的四项追加筛选。
    **{f"axis_{axis}": (axis,) for axis in ABNORMALITY_AXES},
    "renal_metabolic_perfusion": SELECTED_REMOVED_AXES,
    **{f"renal_metabolic_perfusion_plus_{axis}": (*SELECTED_REMOVED_AXES, axis)
       for axis in SELECTED_AXIS_SCREEN},
}
CORE_RISK_GROUPS = ("ventilation_acid_base", "inflammation_infection", "renal", "electrolytes")
if set().union(*map(set, ABNORMALITY_GROUPS.values())) != set(ABNORMALITY_AXES) or sum(
        len(group) for group in ABNORMALITY_GROUPS.values()) != len(ABNORMALITY_AXES):
    raise ValueError("abnormality groups must partition the fixed axis vocabulary")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attach_recovery(data: EventData, directory: Path, *, mode: str) -> EventData:
    if mode not in ("active", "masked"):
        raise ValueError("clinical recovery mode must be active or masked")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "lab_recovery.npz"
    if (manifest["status"] != "derived_not_yet_validated_by_prediction"
            or _sha(path) != manifest["output_sha256"]
            or manifest["input_sha256"]["processed_arrays"] != data.source_hashes["event_arrays"]):
        raise ValueError("recovery source contract mismatch")
    with np.load(path, allow_pickle=False) as archive:
        hadms = archive["hadm_ids"]
        features = archive["features"]
        names = archive["feature_names"]
    if (tuple(names.tolist()) != FEATURE_NAMES or features.shape != (len(hadms), len(FEATURE_NAMES))
            or len(set(hadms.tolist())) != len(hadms)
            or set(hadms.tolist()) != set(data.base.hadm_ids.tolist())
            or not np.isfinite(features).all() or (features < 0).any()):
        raise ValueError("recovery feature identity, shape or value mismatch")
    index = {int(hadm): i for i, hadm in enumerate(hadms)}
    aligned = features[[index[int(hadm)] for hadm in data.base.hadm_ids]].copy()
    if mode == "masked":
        aligned.fill(0.)
    return replace(data, recovery_by_admission=aligned)


def _scale_abnormalities_on_train(features: np.ndarray, splits: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """只用训练住院的阳性异常与观测跨度拟合正值尺度；0 和缺测 mask 不变。"""
    if len(splits) != len(features) or not np.any(splits == "train"):
        raise ValueError("training patients required for abnormality scaling")
    result = features.copy().reshape(len(features), -1, 6)
    parameters: dict[str, object] = {"method": "training_positive_p95", "axes": {}}
    for index, axis in enumerate(ABNORMALITY_FEATURE_NAMES[::6]):
        train = result[splits == "train", index]
        peak = train[train[:, 2] > 0, 2]
        span = train[train[:, 5] > 0, 5]
        severity_scale = float(np.percentile(peak, 95)) if peak.size else 1.0
        span_scale = float(np.percentile(span, 95)) if span.size else 1.0
        if not np.isfinite(severity_scale) or severity_scale <= 0 or not np.isfinite(span_scale) or span_scale <= 0:
            raise ValueError("invalid training abnormality scale")
        result[:, index, 1:4] = np.minimum(result[:, index, 1:4] / severity_scale, 5.0)
        result[:, index, 5] = np.minimum(result[:, index, 5] / span_scale, 5.0)
        parameters["axes"][axis.removesuffix("__observed")] = {
            "severity_log_p95_positive": severity_scale, "span_log_p95_positive": span_scale,
            "training_positive_peak": int(peak.size), "training_positive_span": int(span.size),
        }
    return result.reshape(features.shape), parameters


def attach_abnormalities(data: EventData, directory: Path, *, mode: str,
                         scale_mode: str = "none") -> EventData:
    """接入扩展异常特征；遮蔽臂保持完全相同的模型结构和参数量。"""
    if mode not in ("active", "masked", "measurement_only", "core9", "core_risk",
                    *(f"without_{group}" for group in ABNORMALITY_ABLATIONS)):
        raise ValueError("unknown abnormality mode")
    if scale_mode not in ("none", "training_positive_p95"):
        raise ValueError("unknown abnormality scale mode")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "abnormality.npz"
    if (manifest["status"] != "derived_not_yet_validated_by_prediction"
            or _sha(path) != manifest["output_sha256"]
            or manifest["input_sha256"]["processed_arrays"] != data.source_hashes["event_arrays"]):
        raise ValueError("abnormality source contract mismatch")
    with np.load(path, allow_pickle=False) as archive:
        hadms = archive["hadm_ids"]
        features = archive["features"]
        names = archive["feature_names"]
    if (tuple(names.tolist()) != ABNORMALITY_FEATURE_NAMES
            or features.shape != (len(hadms), len(ABNORMALITY_FEATURE_NAMES))
            or len(set(hadms.tolist())) != len(hadms)
            or set(hadms.tolist()) != set(data.base.hadm_ids.tolist())
            or not np.isfinite(features).all() or (features < 0).any()):
        raise ValueError("abnormality identity, shape or value mismatch")
    index = {int(hadm): i for i, hadm in enumerate(hadms)}
    aligned = features[[index[int(hadm)] for hadm in data.base.hadm_ids]].copy()
    scaling = None
    if scale_mode == "training_positive_p95":
        aligned, scaling = _scale_abnormalities_on_train(aligned, data.base.splits)
    if mode == "masked":
        aligned.fill(0.)
    elif mode == "core9":
        aligned[:, len(FEATURE_NAMES):] = 0.
    elif mode == "measurement_only":
        # 保留是否测过与观测跨度，仅遮蔽真正的异常程度／恢复内容。
        aligned.reshape(len(aligned), -1, 6)[:, :, 1:5] = 0.
    elif mode == "core_risk":
        shaped = aligned.reshape(len(aligned), -1, 6)
        kept = {axis for group in CORE_RISK_GROUPS for axis in ABNORMALITY_GROUPS[group]}
        for index, axis in enumerate(ABNORMALITY_AXES):
            if axis not in kept:
                shaped[:, index, :] = 0.
    elif mode.startswith("without_"):
        group = mode.removeprefix("without_")
        shaped = aligned.reshape(len(aligned), -1, 6)
        for axis in ABNORMALITY_ABLATIONS[group]:
            shaped[:, ABNORMALITY_AXES.index(axis), :] = 0.
    source_hashes = {**data.source_hashes, "abnormality_input": _sha(path)}
    if scaling is not None:
        source_hashes["abnormality_scaling"] = hashlib.sha256(
            json.dumps(scaling, sort_keys=True).encode("utf-8")).hexdigest()
    return replace(data, recovery_by_admission=aligned, abnormality_scaling=scaling,
                   source_hashes=source_hashes)
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/medication_effect_data.py · lines 1–110

<!-- REVIEW_FRAGMENT {"path":"training/v5/medication_effect_data.py","part":1,"start_line":1,"end_line":110,"source_sha256":"2ee73a351b5a5cf113dde226af60507e598fcc1b20eca22ddf0a7bbfedb1b449","fragment_sha256":"2ee73a351b5a5cf113dde226af60507e598fcc1b20eca22ddf0a7bbfedb1b449","added_terminal_newline":false} -->
````python
"""从已核验的院内实际给药构造药理作用组，不把药名或给药途径输入模型。"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from training.v5.event_data import EventData


# 抗菌药仅是全身抗菌暴露代理，不能推断肺部感染适应证或治疗有效。
# 利尿剂单列共病/容量管理组，不并入 COPD 特异治疗组。
EFFECT_GROUPS = {
    "short_beta_agonist": ("saba",),
    "short_muscarinic_antagonist": ("sama",),
    "long_beta_agonist": ("laba",),
    "long_muscarinic_antagonist": ("lama",),
    "inhaled_corticosteroid": ("inhaled_steroid",),
    "systemic_corticosteroid": ("systemic_steroid",),
    "common_systemic_antibacterial": ("macrolide", "beta_lactam", "fluoroquinolone", "tetracycline"),
    "diuretic": ("loop_diuretic", "other_diuretic"),
}
# 两个高共暴露治疗组合用于联合消融；不增加输入维度或新药理类别。
EFFECT_ABLATIONS = {
    **{group: (group,) for group in EFFECT_GROUPS},
    "short_bronchodilators": ("short_beta_agonist", "short_muscarinic_antagonist"),
    "maintenance_inhalers": ("long_beta_agonist", "long_muscarinic_antagonist",
                            "inhaled_corticosteroid"),
}
CORE_ACUTE_GROUPS = ("short_beta_agonist", "short_muscarinic_antagonist",
                     "systemic_corticosteroid", "common_systemic_antibacterial")
VALUES_PER_GROUP = 6
FEATURES = ("exposed", "administrations_log", "hospital_day_bins_log", "first_day_log",
            "last_day_log", "span_days_log")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attach_medication_effects(data: EventData, processed_dir: Path, *, mode: str) -> EventData:
    """在相同词表/参数量下遮蔽作用组；复方的同一来源只计一次。"""
    if mode not in ("active", "masked", "exposure_only", "core_acute",
                    *(f"without_{group}" for group in EFFECT_ABLATIONS)):
        raise ValueError("unknown medication-effect ablation")
    manifest_path = processed_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events_path = processed_dir / "events.csv.gz"
    if (manifest["status"] != "preprocessed_and_checked"
            or _sha(manifest_path) != data.source_hashes["processed_manifest"]
            or _sha(events_path) != manifest["output_sha256"]["events.csv.gz"]):
        raise ValueError("medication source contract mismatch")
    hadm_index = {str(hadm): i for i, hadm in enumerate(data.base.hadm_ids)}
    if len(hadm_index) != len(data.base.hadm_ids):
        raise ValueError("duplicate model admissions")
    by_class = defaultdict(list)
    for group, members in EFFECT_GROUPS.items():
        for member in members:
            by_class[member].append(group)
    # source_id 为 eMAR ID + seq；同一复方可对应多类，但同一作用组只算一次。
    administration_days: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    with gzip.open(events_path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] != "medication" or row["concept"] not in by_class:
                continue
            position = hadm_index.get(row["hadm_id"])
            if position is None:
                raise ValueError("accepted medication outside model cohort")
            day = float(row["elapsed_days"])
            if not math.isfinite(day) or day < 0 or not row["source_id"]:
                raise ValueError("invalid accepted administration")
            for group in by_class[row["concept"]]:
                entries = administration_days[(position, group)]
                previous = entries.setdefault(row["source_id"], day)
                if abs(previous - day) > 1e-5:
                    raise ValueError("one eMAR administration has conflicting times")
    features = np.zeros((len(hadm_index), len(EFFECT_GROUPS) * VALUES_PER_GROUP), dtype=np.float32)
    for (position, group), entries in administration_days.items():
        days = sorted(entries.values())
        first, last = days[0], days[-1]
        offset = list(EFFECT_GROUPS).index(group) * VALUES_PER_GROUP
        features[position, offset:offset + VALUES_PER_GROUP] = (
            1., math.log1p(len(days)), math.log1p(len({math.floor(day) for day in days})),
            math.log1p(first), math.log1p(last), math.log1p(last - first))
    if mode == "masked":
        features.fill(0.)
    elif mode == "exposure_only":
        features.reshape(len(hadm_index), len(EFFECT_GROUPS), VALUES_PER_GROUP)[:, :, 1:] = 0.
    elif mode == "core_acute":
        for group_index, group in enumerate(EFFECT_GROUPS):
            if group not in CORE_ACUTE_GROUPS:
                offset = group_index * VALUES_PER_GROUP
                features[:, offset:offset + VALUES_PER_GROUP] = 0.
    elif mode.startswith("without_"):
        for group in EFFECT_ABLATIONS[mode.removeprefix("without_")]:
            offset = list(EFFECT_GROUPS).index(group) * VALUES_PER_GROUP
            features[:, offset:offset + VALUES_PER_GROUP] = 0.
    return replace(data, medication_by_admission=features)
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/medication_effect_model.py · lines 1–48

<!-- REVIEW_FRAGMENT {"path":"training/v5/medication_effect_model.py","part":1,"start_line":1,"end_line":48,"source_sha256":"6e558f6e11feba33e105229f2f7a27159151dce6cffa2516225b3c4e1163ce77","fragment_sha256":"6e558f6e11feba33e105229f2f7a27159151dce6cffa2516225b3c4e1163ce77","added_terminal_newline":false} -->
````python
"""在检验异常分支旁加入药理作用组暴露，不恢复原始给药事件时序。"""

from __future__ import annotations

import torch
from torch import nn

from training.v5.abnormality_model import AbnormalityFusionModel
from training.v5.event_data import EventBatch
from training.v5.medication_effect_data import EFFECT_GROUPS, VALUES_PER_GROUP


class MedicationEffectFusionModel(AbnormalityFusionModel):
    def __init__(self, *, concept_count: int, type_count: int, field_count: int,
                 text_width: int, abnormality_width: int, width: int = 128) -> None:
        super().__init__(concept_count=concept_count, type_count=type_count,
                         field_count=field_count, text_width=text_width,
                         abnormality_width=abnormality_width, width=width)
        self.medication_axis_embedding = nn.Embedding(len(EFFECT_GROUPS), 24)
        self.medication_values = nn.Sequential(nn.Linear(VALUES_PER_GROUP, 24), nn.GELU(),
                                               nn.LayerNorm(24))
        self.medication_score = nn.Linear(24, 1)
        self.medication_projection = nn.Sequential(nn.Linear(24, width), nn.GELU(),
                                                   nn.LayerNorm(width))
        self.medication_gate = nn.Linear(2 * width, 1)
        nn.init.constant_(self.medication_gate.bias, -1.0)

    def _events(self, batch: EventBatch) -> tuple[torch.Tensor, torch.Tensor]:
        laboratory, lab_present = super()._events(batch)
        values = batch.medication_by_admission
        expected = (*batch.event_counts.shape, len(EFFECT_GROUPS) * VALUES_PER_GROUP)
        if values is None or values.shape != expected or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("invalid medication-effect input")
        features = values.reshape(-1, len(EFFECT_GROUPS), VALUES_PER_GROUP)
        observed = features[:, :, 0] > 0
        med_present = observed.any(dim=1)
        medication = laboratory.new_zeros(laboratory.shape)
        if bool(med_present.any()):
            axis_ids = torch.arange(len(EFFECT_GROUPS), device=features.device)
            embedded = self.medication_values(features) + self.medication_axis_embedding(axis_ids)[None, :, :]
            scores = self.medication_score(embedded).squeeze(-1).masked_fill(~observed, -1e4)
            weights = torch.softmax(scores, dim=1) * observed
            pooled = (embedded * weights.unsqueeze(-1)).sum(dim=1)
            projected = self.medication_projection(pooled)
            medication[med_present] = projected[med_present]
        gate = torch.sigmoid(self.medication_gate(torch.cat((laboratory, medication), dim=-1)))
        gate = gate * med_present.unsqueeze(-1)
        return laboratory + gate * medication, lab_present | med_present
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/model.py · lines 1–177

<!-- REVIEW_FRAGMENT {"path":"training/v5/model.py","part":1,"start_line":1,"end_line":177,"source_sha256":"0c467c8a883ad14811613180ccdbf50ae3fe3b96876ec448d70bb89be5115b97","fragment_sha256":"0c467c8a883ad14811613180ccdbf50ae3fe3b96876ec448d70bb89be5115b97","added_terminal_newline":false} -->
````python
"""V5 六组结构化、八领域文本与患者因果序列的竞争风险模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training.v5.data import GROUP_WIDTHS, V5Batch


@dataclass(frozen=True)
class V5Output:
    log_amplitude: torch.Tensor
    shape: torch.Tensor
    gate: torch.Tensor


class V5Model(nn.Module):
    """每次合格出院读取其因果序列位置，输出两病因 Weibull 参数。"""

    def __init__(self, width: int = 128, text_width: int = 768,
                 balance_text: bool = False) -> None:
        super().__init__()
        if width % 4:
            raise ValueError("width must be divisible by four heads")
        self.balance_text = balance_text
        self.group_projections = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * size, width), nn.GELU(), nn.LayerNorm(width))
            for size in GROUP_WIDTHS
        )
        self.group_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, 256, dropout=0.1, batch_first=True),
            num_layers=1, enable_nested_tensor=False,
        )
        self.group_score = nn.Linear(width, 1)
        self.domain_embedding = nn.Embedding(8, width)
        self.snippet_score = nn.Linear(width, 1)
        self.domain_score = nn.Linear(width, 1)
        self.cross_attention = nn.MultiheadAttention(width, 4, dropout=0.1, batch_first=True)
        self.gate_layer = nn.Linear(3 * width, 2)
        self.fusion_norm = nn.LayerNorm(width)
        self.gap_projection = nn.Linear(2, width)
        self.patient_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, 256, dropout=0.1, batch_first=True),
            num_layers=1, enable_nested_tensor=False,
        )
        self.risk_head = nn.Sequential(nn.Linear(width, 32), nn.ReLU(),
                                       nn.Dropout(0.2), nn.Linear(32, 2))
        self.raw_shape = nn.Parameter(torch.zeros(2))
        with torch.no_grad():
            self.risk_head[-1].bias.copy_(torch.tensor([-1.5, -3.0]))
        # 输入维度不同只改变此层；其余层在同一随机种子下保持相同初始化。
        self.text_projection = nn.Linear(text_width, width)

    def _text_domains(self, batch: V5Batch, flat_stays: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按住院和临床领域对全部片段做可训练注意力池化，不截断片段。"""

        device = batch.valid.device
        domain_count = flat_stays * 8
        if batch.snippet_vectors.shape[0] == 0:
            return (torch.zeros((flat_stays, 8, self.text_projection.out_features), device=device),
                    torch.zeros((flat_stays, 8), dtype=torch.bool, device=device))
        field = batch.snippet_index[:, 1]
        segment = batch.snippet_index[:, 0] * 8 + field
        content = self.text_projection(batch.snippet_vectors)
        if self.balance_text:
            # 对照实验：使单位范数编码的内容与领域嵌入处于同一量级。
            content = F.layer_norm(content, (content.shape[-1],))
        projected = content + self.domain_embedding(field)
        score = self.snippet_score(torch.tanh(projected)).squeeze(-1)
        maximum = torch.full((domain_count,), -torch.inf, device=device)
        maximum.scatter_reduce_(0, segment, score, reduce="amax", include_self=True)
        weights = torch.exp(score - maximum[segment])
        totals = torch.zeros((domain_count,), device=device)
        totals.scatter_add_(0, segment, weights)
        weights = weights / totals[segment].clamp_min(1e-8)
        pooled = torch.zeros((domain_count, projected.shape[-1]), device=device)
        pooled.index_add_(0, segment, weights.unsqueeze(-1) * projected)
        counts = torch.zeros((domain_count,), device=device)
        counts.scatter_add_(0, segment, torch.ones_like(weights))
        return pooled.view(flat_stays, 8, -1), (counts > 0).view(flat_stays, 8)

    def forward(self, batch: V5Batch) -> V5Output:
        batch_size, length = batch.valid.shape
        flat_stays = batch_size * length
        tokens = torch.stack([
            layer(torch.cat((values, mask), dim=-1))
            for layer, values, mask in zip(self.group_projections, batch.group_values,
                                           batch.group_masks, strict=True)
        ], dim=2).reshape(flat_stays, len(GROUP_WIDTHS), -1)
        group_present = torch.stack([mask.any(dim=-1) for mask in batch.group_masks], dim=-1)
        group_present = group_present.reshape(flat_stays, len(GROUP_WIDTHS))
        group_present[:, 0] = True  # 填充住院也保留一个安全 token，最终由 valid 掩码清除。
        structured_tokens = self.group_encoder(tokens, src_key_padding_mask=~group_present)
        group_score = self.group_score(structured_tokens).squeeze(-1)
        group_score = group_score.masked_fill(~group_present, -1e4)
        group_weights = torch.softmax(group_score, dim=1)
        structured = (structured_tokens * group_weights.unsqueeze(-1)).sum(dim=1)

        domains, domain_present = self._text_domains(batch, flat_stays)
        any_text = domain_present.any(dim=1)
        safe_present = domain_present.clone()
        safe_present[~any_text, 0] = True
        domain_score = self.domain_score(torch.tanh(domains)).squeeze(-1)
        domain_score = domain_score.masked_fill(~safe_present, -1e4)
        domain_weights = torch.softmax(domain_score, dim=1)
        text = (domains * domain_weights.unsqueeze(-1)).sum(dim=1)
        text = text * any_text.unsqueeze(-1)
        cross_tokens, _ = self.cross_attention(structured_tokens, domains, domains,
                                                key_padding_mask=~safe_present,
                                                need_weights=False)
        cross = (cross_tokens * group_weights.unsqueeze(-1)).sum(dim=1)
        cross = cross * any_text.unsqueeze(-1)
        gate = torch.sigmoid(self.gate_layer(torch.cat((structured, text, cross), dim=-1)))
        gate = gate * any_text.unsqueeze(-1)
        admission = self.fusion_norm(structured + gate[:, :1] * text + gate[:, 1:] * cross)
        admission = admission.view(batch_size, length, -1)
        admission = admission + self.gap_projection(batch.gap)
        causal = torch.triu(torch.ones((length, length), dtype=torch.bool,
                                       device=batch.valid.device), diagonal=1)
        patient = self.patient_encoder(admission, mask=causal,
                                       src_key_padding_mask=~batch.valid)
        log_amplitude = self.risk_head(patient).clamp(-10.0, 6.0)
        shape = F.softplus(self.raw_shape) + 0.1
        return V5Output(log_amplitude, shape, gate.view(batch_size, length, 2))


def competing_nll(output: V5Output, batch: V5Batch) -> torch.Tensor:
    """两病因全似然；只在有效出院位置计算住院等权平均。"""

    if not torch.all((batch.events[batch.valid] >= 0) & (batch.events[batch.valid] <= 2)):
        raise ValueError("invalid event type")
    scaled = batch.times.clamp_min(1e-3).unsqueeze(-1) / 365.0
    cumulative = torch.exp(output.log_amplitude) * scaled.pow(output.shape)
    log_hazard = (output.log_amplitude + torch.log(output.shape)
                  - np.log(365.0) + (output.shape - 1.0) * torch.log(scaled))
    selected = log_hazard.gather(-1, (batch.events - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1)
    loss = cumulative.sum(dim=-1) - torch.where(batch.events > 0, selected, 0.0)
    return loss[batch.valid].mean()


def predict_cif(log_amplitude: np.ndarray, shape: np.ndarray,
                days: np.ndarray) -> np.ndarray:
    """按累计风险日增量积分，输出 [样本,时间,存活／再住院／死亡]。"""

    query = np.asarray(days, dtype=np.float64)
    if query.ndim != 1 or query[0] != 0 or query[-1] > 365 or np.any(np.diff(query) <= 0):
        raise ValueError("days must start at zero and increase within 365")
    amplitude = np.exp(np.asarray(log_amplitude, dtype=np.float64))
    powers = np.asarray(shape, dtype=np.float64)
    if amplitude.ndim != 2 or amplitude.shape[1] != 2 or powers.shape != (2,):
        raise ValueError("two cause-specific parameters required")
    whole_days = np.arange(366, dtype=np.float64)
    cumulative = amplitude[:, None, :] * (whole_days[None, :, None] / 365.0) ** powers
    increments = np.diff(cumulative, axis=1)
    total = increments.sum(axis=2)
    event_mass = -np.expm1(-total)
    share = np.divide(increments, total[:, :, None], out=np.zeros_like(increments),
                      where=total[:, :, None] > 0)
    survival = np.ones(amplitude.shape[0], dtype=np.float64)
    cif = np.zeros((amplitude.shape[0], 2), dtype=np.float64)
    curve = np.empty((amplitude.shape[0], 366, 3), dtype=np.float64)
    curve[:, 0] = np.column_stack((survival, cif))
    for day in range(365):
        cif += survival[:, None] * event_mass[:, day, None] * share[:, day]
        survival *= np.exp(-total[:, day])
        curve[:, day + 1] = np.column_stack((survival, cif))
    if np.array_equal(query, whole_days):
        return curve
    return np.stack([
        np.stack([np.interp(query, whole_days, row[:, cause]) for row in curve])
        for cause in range(3)
    ], axis=2)
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/numeric_preprocessing.py · lines 1–21

<!-- REVIEW_FRAGMENT {"path":"training/v5/numeric_preprocessing.py","part":1,"start_line":1,"end_line":21,"source_sha256":"fc128e8c6b439dcd32b6e0761faf004b2cfc19a20ccc874e4ca9c56c52f09d38","fragment_sha256":"fc128e8c6b439dcd32b6e0761faf004b2cfc19a20ccc874e4ca9c56c52f09d38","added_terminal_newline":false} -->
````python
"""纯数值预处理：小量纲使用真实 IQR，零跨度显式回退。"""
from __future__ import annotations

import numpy as np


def robust_parameters(observed: np.ndarray) -> dict[str, float | int | str]:
    values = np.asarray(observed, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("尺度拟合需要非空、有限的训练观测")
    median = float(np.median(values))
    q1, q3 = np.percentile(values, (25, 75))
    iqr = float(q3 - q1)
    tolerance = max(abs(median), 1.0) * 1e-8
    if iqr > tolerance:
        scale, method = iqr, "iqr"
    else:
        deviation = float(np.std(values))
        scale, method = (deviation, "std_fallback") if deviation > tolerance else (1.0, "constant")
    return {"median": median, "scale": scale, "iqr": iqr,
            "method": method, "training_observations": int(values.size)}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/permute_added_section.py · lines 1–132

<!-- REVIEW_FRAGMENT {"path":"training/v5/permute_added_section.py","part":1,"start_line":1,"end_line":132,"source_sha256":"5a8dd010092c6235f550158914bfa9d0bb42545094affef793366965e3e32058","fragment_sha256":"5a8dd010092c6235f550158914bfa9d0bb42545094affef793366965e3e32058","added_terminal_newline":false} -->
````python
"""固定已训练模型，只跨患者替换一个新增章节的 BERT 内容。"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from training.v5.compare_strict_ablations import _paired_auc_interval
from training.v5.event_set_model import EventSetFusionModel
from training.v5.extend_field_manifest import EXTRA_LABELS
from training.v5.independent_data import split_sequences
from training.v5.processed_event_data import load_processed_event_data
from training.v5.train import _public_metrics, clustered_brier_interval, metrics
from training.v5.train_event_sequence import predict, select_active_text_fields


def permute_role(data, sequences: list[np.ndarray], role: str, seed: int):
    """保留章节是否存在和每次住院片段数，只改该章节的患者来源。"""

    field_id = data.base.field_roles.index(role)
    selected = np.concatenate(sequences)
    pool = [(int(data.base.subject_ids[admission]), vector_row)
            for admission in selected
            for field, vector_row in data.base.text_by_admission[int(admission)]
            if field == field_id]
    if len({subject for subject, _ in pool}) < 2:
        raise ValueError("too few donor patients")
    rng = np.random.default_rng(seed)
    refs = list(data.base.text_by_admission)
    changed = 0
    for admission in selected:
        subject = int(data.base.subject_ids[admission])
        replacement = []
        for field, vector_row in refs[int(admission)]:
            if field == field_id:
                while True:
                    donor_subject, vector_row = pool[int(rng.integers(len(pool)))]
                    if donor_subject != subject:
                        break
                changed += 1
            replacement.append((field, vector_row))
        refs[int(admission)] = tuple(replacement)
    return replace(data, base=replace(data.base, text_by_admission=tuple(refs))), changed


def run(aligned_dir: Path, structured_dir: Path, base_csv: Path, processed_event_dir: Path,
        raw_ledger_dir: Path, field_dir: Path, vectors: Path, labels_csv: Path,
        fitted_dir: Path, role: str, seed: int = 71) -> dict[str, object]:
    if role not in EXTRA_LABELS:
        raise ValueError("not an added section")
    output = fitted_dir / "content_permutation.json"
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((fitted_dir / "report.json").read_text(encoding="utf-8"))
    if role not in report["active_text_fields"] or report["event_encoder"] != "set":
        raise ValueError("checkpoint does not use requested field or event encoder")
    data, _ = load_processed_event_data(
        aligned_dir, structured_dir, base_csv, processed_event_dir, raw_ledger_dir,
        labels_csv=labels_csv, field_dir=field_dir, vectors_path=vectors,
    )
    if (report["field_manifest_sha256"] != data.base.manifest_sha256
            or report["label_sha256"] != data.base.label_sha256
            or report["raw_source_sha256"] != data.source_hashes
            or report["field_roles"] != list(data.base.field_roles)):
        raise ValueError("checkpoint and data contract differ")
    data = replace(data, base=select_active_text_fields(data.base, tuple(report["active_text_fields"])))
    model = EventSetFusionModel(
        concept_count=len(data.base.concept_names), type_count=len(data.base.type_names),
        field_count=len(data.base.field_roles), text_width=data.base.text_width,
        event_concept_count=len(data.concept_names),
    ).to("cuda")
    model.load_state_dict(torch.load(fitted_dir / "model.pt", map_location="cuda", weights_only=True))
    result: dict[str, object] = {"status": "fixed_checkpoint_candidate_content_permutation",
                                 "role": role, "seed": seed, "splits": {}}
    for split, key in (("validation", "validation"), ("test", "historical_test")):
        sequences = split_sequences(data.base, split)
        indices = np.concatenate(sequences)
        real, _ = predict(model, data, sequences, process_only=False,
                          batch_size=12, device=torch.device("cuda"))
        with np.load(fitted_dir / f"{key}_predictions.npz") as saved:
            if (not np.array_equal(saved["hadm_ids"], data.base.hadm_ids[indices])
                    or not np.allclose(saved["risk"], real, atol=1e-6, rtol=0)):
                raise ValueError("checkpoint fails to reproduce saved predictions")
        changed, count = permute_role(data, sequences, role, seed)
        other, _ = predict(model, changed, sequences, process_only=False,
                           batch_size=12, device=torch.device("cuda"))
        subjects, times, events = (data.base.subject_ids[indices],
                                   data.base.times[indices], data.base.events[indices])
        original = metrics(times, events, real)
        permuted = metrics(times, events, other)
        result["splits"][key] = {
            "snippets_replaced": count,
            "original": _public_metrics(original),
            "permuted": _public_metrics(permuted),
            "permuted_minus_original_brier_interval": clustered_brier_interval(
                subjects, times, events, other, real, seed=seed),
            "permuted_minus_original_auc": {
                str(day): _paired_auc_interval(subjects, times, events, other, real, day, seed=seed)
                for day in (30, 90, 365)
            },
        }
    result["limitation"] = ("One random content permutation; it tests use of this frozen embedding "
                             "within the fitted model, not causal clinical value.")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned_dir", "structured_dir", "base_csv", "processed_event_dir",
                 "raw_ledger_dir", "field_dir", "vectors", "labels_csv", "fitted_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--role", choices=tuple(EXTRA_LABELS), required=True)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint permutation")
    result = run(args.aligned_dir, args.structured_dir, args.base_csv,
                 args.processed_event_dir, args.raw_ledger_dir, args.field_dir,
                 args.vectors, args.labels_csv, args.fitted_dir, args.role, args.seed)
    print(json.dumps({split: {"snippets_replaced": data["snippets_replaced"],
                              "brier_interval": data["permuted_minus_original_brier_interval"]}
                      for split, data in result["splits"].items()}))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/permute_field_content.py · lines 1–108

<!-- REVIEW_FRAGMENT {"path":"training/v5/permute_field_content.py","part":1,"start_line":1,"end_line":108,"source_sha256":"230236046897beaf6ccb2e68785c924beefe1df95b33771cd4d65fa15d9afd70","fragment_sha256":"230236046897beaf6ccb2e68785c924beefe1df95b33771cd4d65fa15d9afd70","added_terminal_newline":false} -->
````python
"""固定训练模型，跨患者置换同名字段向量，检验内容而非文本可用性。"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_data import (
    IndependentData, load_data, normalize_on_training_patients, split_sequences,
)
from training.v5.independent_fusion import IndependentFusionModel
from training.v5.train import _public_metrics, clustered_brier_interval, metrics
from training.v5.train_independent import predict


def permute_content(data: IndependentData, sequences: list[np.ndarray],
                    seed: int) -> tuple[IndependentData, int]:
    """每个片段只更换同字段、不同患者的向量；字段数与缺失模式不变。"""

    selected = np.concatenate(sequences)
    donors: dict[int, list[tuple[int, int]]] = {index: [] for index in range(len(FIELD_LABELS))}
    for admission in selected:
        for field, vector_row in data.text_by_admission[int(admission)]:
            donors[field].append((int(data.subject_ids[admission]), vector_row))
    rng = np.random.default_rng(seed)
    new_refs = list(data.text_by_admission)
    replacements = 0
    for admission in selected:
        subject = int(data.subject_ids[admission])
        changed = []
        for field, _ in data.text_by_admission[int(admission)]:
            pool = donors[field]
            if len({member for member, _ in pool}) < 2:
                raise ValueError("field has too few donor patients")
            while True:
                donor_subject, donor_row = pool[int(rng.integers(len(pool)))]
                if donor_subject != subject:
                    break
            changed.append((field, donor_row))
            replacements += 1
        new_refs[int(admission)] = tuple(changed)
    return replace(data, text_by_admission=tuple(new_refs)), replacements


def run(aligned_dir: Path, structured_dir: Path, base_csv: Path, field_dir: Path,
        vectors: Path, fitted_dir: Path, *, seed: int = 71, batch_size: int = 24) -> dict[str, object]:
    target = fitted_dir / "field_content_permutation.json"
    if target.exists():
        raise FileExistsError(target)
    data, _ = normalize_on_training_patients(
        load_data(aligned_dir, structured_dir, base_csv, field_dir, vectors)
    )
    validation = split_sequences(data, "validation")
    indices = np.concatenate(validation)
    model = IndependentFusionModel(
        concept_count=len(data.concept_names), type_count=len(data.type_names),
        field_count=len(FIELD_LABELS), text_width=data.text_width,
    ).to("cuda")
    model.load_state_dict(torch.load(fitted_dir / "structured_field_bert.pt",
                                     map_location="cuda", weights_only=True))
    real, _ = predict(model, data, validation, include_text=True,
                      batch_size=batch_size, device=torch.device("cuda"))
    changed, count = permute_content(data, validation, seed)
    permuted, _ = predict(model, changed, validation, include_text=True,
                          batch_size=batch_size, device=torch.device("cuda"))
    baseline = metrics(data.times[indices], data.events[indices], real)
    control = metrics(data.times[indices], data.events[indices], permuted)
    interval = clustered_brier_interval(
        data.subject_ids[indices], data.times[indices], data.events[indices],
        permuted, real, seed=seed,
    )
    result = {
        "method": "same-field frozen-vector replacement from another validation patient; model weights fixed",
        "manifest_sha256": data.manifest_sha256,
        "seed": seed,
        "validation_chunks_replaced": count,
        "original": _public_metrics(baseline),
        "permuted": _public_metrics(control),
        "permuted_minus_original_integrated_brier_95ci_patient_cluster": interval,
        "interpretation_limit": "one permutation and a previously used validation cohort cannot prove causal text utility",
    }
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--field-dir", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--fitted-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("permutation diagnosis requires CUDA")
    print(json.dumps(run(args.aligned_dir, args.structured_dir, args.base_csv,
                         args.field_dir, args.vectors, args.fitted_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/permute_history_matched_text.py · lines 1–183

<!-- REVIEW_FRAGMENT {"path":"training/v5/permute_history_matched_text.py","part":1,"start_line":1,"end_line":183,"source_sha256":"9d118acd46d75f3eaed6a383297e32e190250630b18de6cc2e40c2cb1f7f89e3","fragment_sha256":"9d118acd46d75f3eaed6a383297e32e190250630b18de6cc2e40c2cb1f7f89e3","added_terminal_newline":false} -->
````python
"""冻结 V5：在相同既往次数层、当前严格编码与字段片段数内替换文本内容。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from training.v5.event_set_model import EventSetFusionModel
from training.v5.independent_data import split_sequences
from training.v5.processed_event_data import load_processed_event_data
from training.v5.train import _public_metrics, clustered_brier_interval, metrics
from training.v5.train_event_sequence import predict


def matched_field_swap(text_by_admission, subjects: np.ndarray, prior: np.ndarray,
                       strict: np.ndarray, selected: np.ndarray, *, seed: int):
    """交换完整字段的片段集合；没有跨患者同层供体时保留原字段并报告。"""
    if len(text_by_admission) != len(subjects) or len(prior) != len(subjects):
        raise ValueError("admission metadata differs")
    rng = np.random.default_rng(seed)
    strata = np.where(prior == 0, 0, np.where(prior <= 2, 1, 2))
    role_refs = []
    pools = defaultdict(list)
    for index in selected:
        grouped = defaultdict(list)
        for role, row in text_by_admission[int(index)]:
            grouped[role].append(row)
        role_refs.append((int(index), grouped))
        for role, rows in grouped.items():
            pools[(int(strata[index]), int(strict[index]), role, len(rows))].append(int(index))
    lookup = {index: grouped for index, grouped in role_refs}
    changed = list(text_by_admission)
    replaced = 0
    unmatched = 0
    for index, grouped in role_refs:
        new = []
        for role, rows in grouped.items():
            key = (int(strata[index]), int(strict[index]), role, len(rows))
            donors = [donor for donor in pools[key] if subjects[donor] != subjects[index]]
            if donors:
                donor = donors[int(rng.integers(len(donors)))]
                new.extend((role, row) for row in lookup[donor][role])
                replaced += 1
            else:
                new.extend((role, row) for row in rows)
                unmatched += 1
        changed[index] = tuple(new)
    return tuple(changed), {"fields_replaced": replaced, "fields_unmatched": unmatched,
                            "matching": "same split, prior bin (0/1-2/3+), current strict code, role and chunk count; different patient"}


def matched_lab_value_swap(data, subjects: np.ndarray, prior: np.ndarray,
                           strict: np.ndarray, selected: np.ndarray, *, seed: int):
    """同层同检验项目、相同观测次数的住院之间交换完整数值轨迹；保留本人的时点。"""
    if data.event_numeric_mask is None:
        raise ValueError("explicit numeric mask is required")
    rng = np.random.default_rng(seed)
    strata = np.where(prior == 0, 0, np.where(prior <= 2, 1, 2))
    pools = defaultdict(list)
    by_stay = {}
    for index in selected:
        observed = data.events_by_admission[int(index)]
        observed = observed[data.event_is_lab[observed] & data.event_numeric_mask[observed]]
        by_concept = {int(concept): observed[data.event_concepts[observed] == concept]
                      for concept in np.unique(data.event_concepts[observed])}
        by_stay[int(index)] = by_concept
        for concept, refs in by_concept.items():
            pools[(int(strata[index]), int(strict[index]), concept, len(refs))].append(int(index))
    changed = data.event_values.copy()
    replaced = 0
    unmatched = 0
    for index, grouped in by_stay.items():
        for concept, refs in grouped.items():
            key = (int(strata[index]), int(strict[index]), concept, len(refs))
            donors = [donor for donor in pools[key] if subjects[donor] != subjects[index]]
            if donors:
                donor = donors[int(rng.integers(len(donors)))]
                donor_refs = by_stay[donor][concept]
                changed[refs, 0] = data.event_values[donor_refs, 0]
                replaced += 1
            else:
                unmatched += 1
    return changed, {"trajectories_replaced": replaced, "trajectories_unmatched": unmatched,
                     "matching": "same split, prior bin, current strict code, laboratory concept and observed count; different patient; original times retained"}


def run(args):
    if args.output_path.exists():
        raise FileExistsError(args.output_path)
    data, _ = load_processed_event_data(
        args.aligned_dir, args.structured_dir, args.base_csv, args.processed_event_dir,
        args.raw_ledger_dir, labels_csv=args.labels_csv, field_dir=args.field_dir,
        vectors_path=args.vectors,
    )
    report = json.loads((args.full_dir / "report.json").read_text(encoding="utf-8"))
    if (report["status"] != "training_completed" or report["label_sha256"] != data.base.label_sha256
            or report["field_manifest_sha256"] != data.base.manifest_sha256
            or report["raw_source_sha256"] != data.source_hashes
            or report["text_vector_width"] != data.base.text_width):
        raise ValueError("frozen checkpoint and current data contract differ")
    model = EventSetFusionModel(
        concept_count=len(data.base.concept_names), type_count=len(data.base.type_names),
        field_count=len(data.base.field_roles), text_width=data.base.text_width,
        event_concept_count=len(data.concept_names),
    ).to("cuda")
    model.load_state_dict(torch.load(args.full_dir / "model.pt", map_location="cuda", weights_only=True))
    by_hadm = {}
    with args.base_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            by_hadm[int(row["hadm_id"])] = row
    prior = np.array([int(by_hadm[int(hadm)]["prior_coded_admission_count"]) for hadm in data.base.hadm_ids])
    strict = np.array([by_hadm[int(hadm)]["index_is_strict_aecopd"] == "t" for hadm in data.base.hadm_ids])
    output = {"status": "history_matched_frozen_content_permutation", "model": str(args.full_dir),
              "label_sha256": data.base.label_sha256, "splits": {}, "limitations": [
                  "Frozen-checkpoint content perturbation is not retraining or causal inference.",
                  "Historical test split was previously used for model exploration.",
                  "Matched donor text can still differ in unmeasured clinical severity."]}
    for split in ("validation", "test"):
        sequences = split_sequences(data.base, split)
        indices = np.concatenate(sequences)
        saved_name = "historical_test" if split == "test" else split
        with np.load(args.full_dir / f"{saved_name}_predictions.npz") as archive:
            saved = {key: archive[key] for key in archive.files}
        if not np.array_equal(saved["hadm_ids"], data.base.hadm_ids[indices]):
            raise ValueError("saved admission order differs")
        original, _ = predict(model, data, sequences, process_only=False,
                              batch_size=12, device=torch.device("cuda"))
        if not np.allclose(original, saved["risk"], atol=2e-6, rtol=0):
            raise ValueError("frozen checkpoint does not reproduce saved risks")
        base_metric = _public_metrics(metrics(data.base.times[indices], data.base.events[indices], original))
        variants = {}
        for modality in ("text", "laboratory_values"):
            rows = []
            for seed in args.seeds:
                if modality == "text":
                    refs, counts = matched_field_swap(data.base.text_by_admission,
                                                      data.base.subject_ids, prior, strict, indices, seed=seed)
                    changed = replace(data, base=replace(data.base, text_by_admission=refs))
                else:
                    values, counts = matched_lab_value_swap(data, data.base.subject_ids,
                                                            prior, strict, indices, seed=seed)
                    changed = replace(data, event_values=values)
                risk, _ = predict(model, changed, sequences, process_only=False,
                                  batch_size=12, device=torch.device("cuda"))
                detail = _public_metrics(metrics(data.base.times[indices], data.base.events[indices], risk))
                rows.append({"seed": seed, **counts, "metrics": detail,
                             "permuted_minus_original_ibs_patient_cluster_95pct_interval":
                             clustered_brier_interval(data.base.subject_ids[indices],
                                                      data.base.times[indices], data.base.events[indices],
                                                      risk, original)})
            variants[modality] = rows
        output["splits"][saved_name] = {"original": base_metric, "permutations": variants}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned_dir", "structured_dir", "base_csv", "processed_event_dir", "raw_ledger_dir",
                 "field_dir", "vectors", "labels_csv", "full_dir", "output_path"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[71, 72, 73])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("frozen permutation requires CUDA")
    result = run(args)
    print(json.dumps({name: {"original_ibs": value["original"]["integrated_brier_7_365"],
                              "permuted_ibs": {modality: [row["metrics"]["integrated_brier_7_365"]
                                                           for row in rows]
                                               for modality, rows in value["permutations"].items()}}
                      for name, value in result["splits"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/plot_abnormality_multiseed_deltas.py · lines 1–82

<!-- REVIEW_FRAGMENT {"path":"training/v5/plot_abnormality_multiseed_deltas.py","part":1,"start_line":1,"end_line":82,"source_sha256":"12b604705a67c15fa026270371a1f07348efa1cdd753b78457f4424a4a6a563e","fragment_sha256":"12b604705a67c15fa026270371a1f07348efa1cdd753b78457f4424a4a6a563e","added_terminal_newline":false} -->
````python
"""将早期检验异常三种子实验画成配对差值图，不与新版药物实验混比。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


COMPARISONS = (
    ("原 9 轴异常 - 无检验", "core9", "masked"),
    ("仅测量过程 - 无检验", "measurement_only", "masked"),
    ("完整 22 轴 - 无检验", "active", "masked"),
    ("完整 22 轴 - 原 9 轴", "active", "core9"),
    ("完整 22 轴 - 仅测量过程", "active", "measurement_only"),
)


def plot(source: Path, output: Path, *, font_path: Path | None = None) -> None:
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads(source.read_text(encoding="utf-8"))
    if (summary["status"] != "exploratory_multi_seed" or summary["cohort_patients"] != 5351
            or summary["cohort_admissions"] != 19401):
        raise ValueError("unexpected laboratory experiment cohort or status")
    by_seed = {}
    for row in summary["validation_rows"]:
        key = (int(row["seed"]), row["mode"])
        if key in by_seed:
            raise ValueError(f"duplicate arm: {key}")
        by_seed[key] = row
    seeds = tuple(summary["seeds"])
    if len(seeds) != 3 or set(by_seed) != {(seed, mode) for seed in seeds
                                            for mode in ("masked", "core9", "measurement_only", "active")}:
        raise ValueError("expected four completed arms for each of three seeds")
    if font_path is not None:
        if not font_path.is_file():
            raise FileNotFoundError(font_path)
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    colors = ("#2b6cb0", "#718096", "#c05621", "#c05621", "#c05621")
    metrics = (("readmission_ibs", "积分 Brier 差值：负值更好", "Brier 差值（候选 - 参照）"),
               ("readmission_auc", "平均时间依赖 AUC 差值：正值更好", "AUC(t) 差值（候选 - 参照）"))
    for axis, (field, title, xlabel) in zip(axes, metrics):
        axis.axvline(0, color="#444444", linewidth=1)
        for index, (_, candidate, reference) in enumerate(COMPARISONS):
            values = np.array([by_seed[(seed, candidate)][field] - by_seed[(seed, reference)][field]
                               for seed in seeds])
            axis.scatter(values, np.full(3, index) + np.array([-.12, 0., .12]),
                         color=colors[index], s=44, alpha=.7, zorder=2)
            axis.scatter(values.mean(), index, color=colors[index], marker="D", s=75, zorder=3)
        axis.set_title(title, fontsize=13)
        axis.set_xlabel(xlabel, fontsize=11)
        axis.grid(axis="x", alpha=.2)
        axis.tick_params(axis="x", labelsize=10)
    axes[0].set_yticks(np.arange(len(COMPARISONS)), [label for label, _, _ in COMPARISONS])
    axes[0].tick_params(axis="y", labelsize=11)
    axes[0].invert_yaxis()
    fig.suptitle("早期检验异常实验：验证折再入院，三种子从头训练\n圆点＝各随机种子；菱形＝三种子均值；竖线＝无变化",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args()
    plot(args.source, args.output, font_path=args.font_path)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/plot_broad_training.py · lines 1–52

<!-- REVIEW_FRAGMENT {"path":"training/v5/plot_broad_training.py","part":1,"start_line":1,"end_line":52,"source_sha256":"8b97111323d545a42c7f3966f8bf584256b77deb6d5359d92064a345182cfad9","fragment_sha256":"8b97111323d545a42c7f3966f8bf584256b77deb6d5359d92064a345182cfad9","added_terminal_newline":false} -->
````python
"""绘制同一宽口径终点的三组训练／验证学习曲线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-run", type=Path, required=True)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--process-run", type=Path)
    parser.add_argument("--title", default="Broad COPD-coded readmission; same cohort and BERT input")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histories = {
        "admission summaries": args.summary_run / "structured_field_bert_history.json",
        "raw event sequence": args.full_run / "history.json",
    }
    if args.process_run is not None:
        histories["process without lab values"] = args.process_run / "history.json"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for name, path in histories.items():
        records = json.loads(path.read_text(encoding="utf-8"))
        epochs = [entry["epoch"] for entry in records]
        axes[0].plot(epochs, [entry["train_nll"] for entry in records], marker=".", label=name)
        axes[1].plot(epochs, [entry["validation_integrated_brier"] for entry in records],
                     marker=".", label=name)
    axes[0].set(xlabel="Epoch", ylabel="Training competing-risk NLL",
                title="Training objective")
    axes[1].set(xlabel="Epoch", ylabel="Validation integrated Brier (7–365 d)",
                title="Checkpoint-selection metric")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(args.title)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/plot_incremental_evidence.py · lines 1–59

<!-- REVIEW_FRAGMENT {"path":"training/v5/plot_incremental_evidence.py","part":1,"start_line":1,"end_line":59,"source_sha256":"bca3cdb918df94250c19deb2932eb995336ed0bef0ea1a1cc6395f2d0e377059","fragment_sha256":"bca3cdb918df94250c19deb2932eb995336ed0bef0ea1a1cc6395f2d0e377059","added_terminal_newline":false} -->
````python
"""只用聚合报告绘制严格 AECOPD 增量证据图，不读取患者级预测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ("full", "history_only_cox", "no_current_static", "no_medication_events",
            "no_lab_values", "no_lab_time", "one_lab_per_concept", "no_text")
LABELS = ("Full V5", "History Cox", "No other static", "No medications",
          "No lab values", "No lab time", "One lab/assay", "No text")


def run(result_dir: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    reports = {seed: json.loads((result_dir / f"strata_seed{seed}.json").read_text(encoding="utf-8"))
               for seed in (2026, 2027, 2028)}
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = ("#263b70", "#a45416", "#6e70a3")
    x = np.arange(len(VARIANTS))
    for axis, split in zip(axes, ("validation", "historical_test")):
        for k, seed in enumerate((2026, 2027, 2028)):
            models = reports[seed]["strata"][split]["all"]["models"]
            full = models["full"]["integrated_brier_7_365"]
            differences = [full - models[name]["integrated_brier_7_365"]
                           if name in models else np.nan for name in VARIANTS]
            axis.plot(x, differences, marker="o", color=colors[k], label=str(seed))
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title("Validation" if split == "validation" else "Reused historical test")
        axis.set_xticks(x, LABELS, rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("IBS(full V5) - IBS(comparator); lower favors full")
    axes[1].legend(title="Seed")
    figure.suptitle("Strict AECOPD readmission: same cohort, endpoint, and patient split")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.result_dir, args.output)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/plot_population_competing_curves.py · lines 1–85

<!-- REVIEW_FRAGMENT {"path":"training/v5/plot_population_competing_curves.py","part":1,"start_line":1,"end_line":85,"source_sha256":"6b5787f63af6e07b8191ed65e164acd4b0e153df7d0ce84c27d1bfec39c6efdd","fragment_sha256":"6b5787f63af6e07b8191ed65e164acd4b0e153df7d0ce84c27d1bfec39c6efdd","added_terminal_newline":false} -->
````python
"""画三种子完整模型的 0–365 天群体累计发生风险检查图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(root: Path, *, seeds: tuple[int, ...] = (2026, 2027, 2028)) -> tuple[Path, Path]:
    target = root / "population_competing_cif_curves.png"
    calibration_target = root / "calibration_90d_three_seeds.png"
    if target.exists() or calibration_target.exists():
        raise FileExistsError("population curve or calibration figure already exists")
    days = np.arange(366)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    calibration_fig, calibration_axes = plt.subplots(2, 2, figsize=(10, 9))
    for row, split in enumerate(("validation", "historical_test")):
        curves = []
        reference = None
        for seed in seeds:
            with np.load(root / f"full_seed{seed}" / f"{split}_predictions.npz", allow_pickle=False) as archive:
                times, events = archive["times"], archive["events"]
                subjects, admissions = archive["subject_ids"], archive["hadm_ids"]
                cif = archive["competing_cif"]
            if reference is None:
                reference = (times, events, subjects, admissions)
            elif any(not np.array_equal(a, b) for a, b in zip(reference, (times, events, subjects, admissions))):
                raise ValueError("full seed predictions are not paired on the same cohort")
            curves.append(cif)
        assert reference is not None
        times, events, _, _ = reference
        for column, (cause, label) in enumerate(((1, "Readmission"), (2, "Death"))):
            ax = axes[row, column]
            predicted = np.stack([cif[:, :, cause].mean(axis=0) for cif in curves])
            observed = ((events[:, None] == cause) & (times[:, None] <= days[None, :])).mean(axis=0)
            ax.plot(days, observed, color="black", linestyle="--", linewidth=1.8, label="Observed in database")
            ax.plot(days, predicted.mean(axis=0), color="#1f77b4", linewidth=1.8,
                    label="Mean prediction (3 seeds)")
            ax.fill_between(days, predicted.min(axis=0), predicted.max(axis=0),
                            color="#1f77b4", alpha=0.18, label="Seed range")
            ax.set_title(f"{split.replace('_', ' ').title()} — {label} (n={len(times)})")
            ax.set_ylabel("Cumulative incidence")
            ax.set_xlabel("Days since discharge")
            ax.set_xlim(0, 365)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, loc="upper left")
            risk_90 = np.mean(np.stack([cif[:, 90, cause] for cif in curves]), axis=0)
            observed_90 = ((events == cause) & (times <= 90)).astype(float)
            bins = np.array_split(np.argsort(risk_90), 10)
            mean_risk = np.array([risk_90[index].mean() for index in bins])
            event_rate = np.array([observed_90[index].mean() for index in bins])
            calibration_ax = calibration_axes[row, column]
            limit = max(.08, float(max(mean_risk.max(), event_rate.max()) * 1.08))
            calibration_ax.plot([0, limit], [0, limit], color="black", linestyle="--", label="Ideal")
            calibration_ax.plot(mean_risk, event_rate, "o-", color="#1f77b4", label="Prediction deciles")
            calibration_ax.set_title(f"{split.replace('_', ' ').title()} — {label} at day 90")
            calibration_ax.set_xlabel("Mean predicted CIF")
            calibration_ax.set_ylabel("Observed database fraction")
            calibration_ax.set_xlim(0, limit)
            calibration_ax.set_ylim(0, limit)
            calibration_ax.grid(alpha=.2)
            calibration_ax.legend(fontsize=8)
    fig.suptitle("World A: observed database events vs competing-risk CIF; historical test is not blind")
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)
    calibration_fig.suptitle("World A day-90 calibration by prediction decile; historical test is not blind")
    calibration_fig.tight_layout()
    calibration_fig.savefig(calibration_target, dpi=150)
    plt.close(calibration_fig)
    return target, calibration_target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    print(*run(args.run_root), sep="\n")
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/plot_revised_group_ablations.py · lines 1–54

<!-- REVIEW_FRAGMENT {"path":"training/v5/plot_revised_group_ablations.py","part":1,"start_line":1,"end_line":54,"source_sha256":"bb33da2bf8f260524ab0c244a67779602c3f60ff0ba19da8fcc5149651889651","fragment_sha256":"bb33da2bf8f260524ab0c244a67779602c3f60ff0ba19da8fcc5149651889651","added_terminal_newline":false} -->
````python
"""Plot paired validation deltas across three seeds; points are runs, diamonds are means."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["split"] == "validation" and row["cause"] == "readmission"]
    groups = list(dict.fromkeys(row["group_removed"] for row in rows))
    if not groups or any(sum(row["group_removed"] == group for row in rows) != 3 for group in groups):
        raise ValueError("each group requires exactly three paired seed results")
    labels = [group.removeprefix("med_").removeprefix("lab_").replace("_", " ") for group in groups]
    fig, axes = plt.subplots(1, 2, figsize=(15, max(8, len(groups) * .46)), sharey=True)
    for axis, field, title in zip(axes, ("delta_brier", "delta_auc"),
                                  ("Integrated Brier: removed - full (lower better)",
                                   "Mean AUC(t): removed - full (higher better)")):
        axis.axvline(0, color="#444444", linewidth=1)
        for index, group in enumerate(groups):
            values = np.array([float(row[field]) for row in rows if row["group_removed"] == group])
            color = "#2b6cb0" if group.startswith("med_") else "#c05621"
            axis.scatter(values, np.full(3, index) + np.array([-.12, 0, .12]),
                         color=color, s=16, alpha=.65, zorder=2)
            axis.scatter(values.mean(), index, color=color, marker="D", s=35, zorder=3)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Paired difference across three seeds")
        axis.grid(axis="x", alpha=.2)
    axes[0].set_yticks(np.arange(len(groups)), labels)
    axes[0].invert_yaxis()
    fig.suptitle("Readmission validation; groups retrained from scratch; no multiplicity adjustment", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.output)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/plot_revised_labs_meds.py · lines 1–59

<!-- REVIEW_FRAGMENT {"path":"training/v5/plot_revised_labs_meds.py","part":1,"start_line":1,"end_line":59,"source_sha256":"ae8957d30f3ec2d47d04628b35838febf2d2081fe6bbc624719969b793f68491","fragment_sha256":"ae8957d30f3ec2d47d04628b35838febf2d2081fe6bbc624719969b793f68491","added_terminal_newline":false} -->
````python
"""绘制新版四臂实验的跨种子时间依赖 AUC 与 Brier；不平滑原始曲线。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.v5.compare_screened_group_ablations import _predictions
from training.v5.train import GRID, metrics


ARMS = ("baseline", "lab_only", "med_only", "full")
LABELS = {"baseline": "Static + text + history", "lab_only": "+ laboratory",
          "med_only": "+ medication", "full": "+ laboratory + medication"}


def plot(root: Path, *, seeds: tuple[int, ...]) -> None:
    for split in ("validation", "historical_test"):
        curves: dict[str, list[dict[str, object]]] = {arm: [] for arm in ARMS}
        identity: tuple[np.ndarray, np.ndarray] | None = None
        for arm in ARMS:
            for seed in seeds:
                record = _predictions(root / f"{arm}_seed{seed}", split)
                if identity is None:
                    identity = record["times"], record["events"]
                elif not (np.array_equal(identity[0], record["times"])
                          and np.array_equal(identity[1], record["events"])):
                    raise ValueError("different evaluated admissions or labels")
                curves[arm].append(metrics(record["times"], record["events"], record["risk"]))
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        for arm in ARMS:
            for axis, name in zip(axes, ("auc_curve", "brier_curve"), strict=True):
                values = np.stack([np.asarray(curve[name]) for curve in curves[arm]])
                line, = axis.plot(GRID[7:], np.nanmean(values[:, 7:], axis=0), label=LABELS[arm])
                axis.fill_between(GRID[7:], np.nanmin(values[:, 7:], axis=0),
                                  np.nanmax(values[:, 7:], axis=0), color=line.get_color(), alpha=.12)
        axes[0].set(xlabel="Days after discharge", ylabel="Time-dependent cumulative AUC(t)",
                    ylim=(.45, .85))
        axes[1].set(xlabel="Days after discharge", ylabel="Brier(t) (lower is better)")
        for axis in axes:
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        figure.suptitle(f"{split}: 3-seed mean and range; observed in-database events (World A)")
        figure.tight_layout()
        figure.savefig(root / f"{split}_auc_brier_three_seeds.png", dpi=160)
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    args = parser.parse_args()
    plot(args.run_root, seeds=tuple(args.seeds))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/plot_strict_axis_screen.py · lines 1–109

<!-- REVIEW_FRAGMENT {"path":"training/v5/plot_strict_axis_screen.py","part":1,"start_line":1,"end_line":109,"source_sha256":"2c8521a707168592b2ee8b66fc513d83ea6f1f7d3d2c53214e09722e2ab73bfe","fragment_sha256":"2c8521a707168592b2ee8b66fc513d83ea6f1f7d3d2c53214e09722e2ab73bfe","added_terminal_newline":false} -->
````python
"""Plot paired strict-endpoint validation ablations from the saved per-seed results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


COMPARISONS = (
    ("full", "selected", "renal + metabolic/perfusion + diuretic", "#2b6cb0"),
    ("selected", "arterial_hypercapnic_acidosis", "hypercapnic acidosis", "#c05621"),
    ("selected", "platelet_low", "low platelets", "#c05621"),
    ("selected", "magnesium_low", "low magnesium", "#c05621"),
    ("selected", "ntprobnp_high", "high NT-proBNP", "#c05621"),
)
SEEDS = (2026, 2027, 2028)


def load_deltas(source: Path) -> dict[tuple[str, str], list[dict[str, float]]]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["split"] == "validation" and row["cause"] == "readmission"]

    expected = {(reference, candidate) for reference, candidate, _, _ in COMPARISONS}
    actual = {(row["reference"], row["candidate"]) for row in rows}
    if actual != expected:
        raise ValueError(f"comparison mismatch: missing={expected - actual}, extra={actual - expected}")

    results: dict[tuple[str, str], list[dict[str, float]]] = {}
    for reference, candidate, _, _ in COMPARISONS:
        matches = [row for row in rows
                   if row["reference"] == reference and row["candidate"] == candidate]
        seed_values = sorted(int(row["seed"]) for row in matches)
        if seed_values != list(SEEDS):
            raise ValueError(f"expected one paired result per seed for {candidate}: {seed_values}")
        results[(reference, candidate)] = [
            {"seed": int(row["seed"]),
             "delta_brier": float(row["delta_brier"]),
             "delta_auc": float(row["delta_auc"])}
            for row in sorted(matches, key=lambda row: int(row["seed"]))
        ]
    return results


def plot(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    results = load_deltas(source)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.9), sharey=True)

    panels = (
        ("delta_brier", "Integrated Brier: candidate - reference (lower better)"),
        ("delta_auc", "Mean AUC(t): candidate - reference (higher better)"),
    )
    for axis, (field, title) in zip(axes, panels):
        all_values = [result[field]
                      for reference, candidate, _, _ in COMPARISONS
                      for result in results[(reference, candidate)]]
        limit = max(abs(value) for value in all_values) * 1.14
        axis.set_xlim(-limit, limit)
        axis.axvline(0, color="#444444", linewidth=1.1, zorder=1)
        axis.axhline(.5, color="#999999", linewidth=.8, linestyle="--", alpha=.6)
        for index, (reference, candidate, _, color) in enumerate(COMPARISONS):
            values = np.array([row[field] for row in results[(reference, candidate)]])
            axis.scatter(values, np.full(len(SEEDS), index) + [-.10, 0, .10],
                         color=color, s=32, alpha=.65, zorder=2)
            axis.scatter(values.mean(), index, color=color, marker="D", s=65, zorder=3)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Paired difference across three seeds", fontsize=10)
        axis.grid(axis="x", alpha=.18)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelsize=9)

    labels = [label for _, _, label, _ in COMPARISONS]
    axes[0].set_yticks(np.arange(len(labels)), labels, fontsize=10)
    axes[0].invert_yaxis()
    fig.suptitle("Strict AECOPD readmission validation; retrained from scratch", fontsize=13, y=.99)
    fig.text(.5, .91,
             "Blue: three-group removal vs full  |  Orange: single-axis removal vs selected  |  No multiplicity adjustment",
             ha="center", fontsize=10)
    legend = (
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#777777",
               markeredgecolor="none", alpha=.65, markersize=6, label="Individual seed"),
        Line2D([0], [0], marker="D", linestyle="None", markerfacecolor="#777777",
               markeredgecolor="none", markersize=7, label="Three-seed mean"),
    )
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .01),
               ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.26, right=.985, top=.83, bottom=.16, wspace=.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.output)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/plot_summary_temporal_comparison.py · lines 1–65

<!-- REVIEW_FRAGMENT {"path":"training/v5/plot_summary_temporal_comparison.py","part":1,"start_line":1,"end_line":65,"source_sha256":"1c595168f81404e25a413b72dd14cd36c06df888be62114ae536f36acaa49a56","fragment_sha256":"1c595168f81404e25a413b72dd14cd36c06df888be62114ae536f36acaa49a56","added_terminal_newline":false} -->
````python
"""读取已完成实验的汇总 CSV，绘制同标签三模型曲线；不重新训练。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-run", type=Path, required=True)
    parser.add_argument("--temporal-run", type=Path, required=True)
    parser.add_argument("--set-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    directories = [args.summary_run, args.temporal_run]
    if args.set_run is not None:
        directories.append(args.set_run)
    reports = [json.loads((directory / "report.json").read_text(encoding="utf-8"))
               for directory in directories]
    for key in ("label_sha256", "field_manifest_sha256", "cohort_admissions"):
        if any(report[key] != reports[0][key] for report in reports[1:]):
            raise ValueError(f"comparison contract differs: {key}")
    curves: dict[tuple[str, str], list[dict[str, str]]] = {}
    for directory in directories:
        with (directory / "aggregate_curves.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["day"]) >= 7:
                    name = "event_set" if directory == args.set_run else row["model"]
                    curves.setdefault((row["split"], name), []).append(row)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"structured": "structured summaries",
              "structured_field_bert": "summaries + BERT",
              "temporal_full": "GRU events + BERT",
              "event_set": "STraTS-style event set + BERT"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col")
    for column, split in enumerate(("validation", "historical_test")):
        for (partition, model), rows in curves.items():
            if partition != split:
                continue
            days = [int(row["day"]) for row in rows]
            auc = [float(row["auc"]) if row["auc"] else float("nan") for row in rows]
            axes[0, column].plot(days, auc, label=labels[model])
            axes[1, column].plot(days, [float(row["brier"]) for row in rows], label=labels[model])
        axes[0, column].set(title=split, ylabel="Cumulative AUC(t)", ylim=(0.5, 0.8))
        axes[1, column].set(xlabel="Days after discharge", ylabel="Brier(t)")
        for axis in axes[:, column]:
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=9)
    figure.suptitle("Strict AECOPD at any position OR preset COPD/emphysema in top 3")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/processed_event_data.py · lines 1–202

<!-- REVIEW_FRAGMENT {"path":"training/v5/processed_event_data.py","part":1,"start_line":1,"end_line":202,"source_sha256":"d0913a8075764a8bd6341b2d3d28a921e18aa975a1cb0e76ea1183a5f97d940a","fragment_sha256":"d0913a8075764a8bd6341b2d3d28a921e18aa975a1cb0e76ea1183a5f97d940a","added_terminal_newline":false} -->
````python
"""接入已校验事件底账，沿用旧静态／文本分支并强制核对现行队列。"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from training.v5.clinical_concepts import MED_NAMES, diagnosis_groups
from training.v5.event_data import EventData
from training.v5.independent_data import IndependentData, load_data, normalize_on_training_patients
from training.v5.section_alignment_data.build_dataset import patient_split
from training.v5.strict_endpoint_labels import STRICT_PRIMARY_EVENT, checked_strict_labels


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path):
    csv.field_size_limit(2**31 - 1)
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def attach_arrays(base: IndependentData, arrays: dict[str, np.ndarray], concepts: tuple[str, ...],
                  source_hashes: dict[str, str]) -> EventData:
    """按 hadm_id 重排关联，已归一化值原样接入，不重复拟合。"""
    hadms, subjects = arrays["hadm_ids"], arrays["subject_ids"]
    offsets, ids = arrays["offsets"], arrays["concept_ids"]
    n = len(arrays["values"])
    if len(set(hadms.tolist())) != len(hadms) or set(hadms.tolist()) != set(base.hadm_ids.tolist()):
        raise ValueError("processed event admissions differ from model cohort")
    if len(subjects) != len(hadms) or len(offsets) != len(hadms) + 1:
        raise ValueError("processed subject/offset shape mismatch")
    if offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) < 0):
        raise ValueError("processed event offsets invalid")
    if not concepts or len(set(concepts)) != len(concepts):
        raise ValueError("processed event dictionary invalid")
    if any(len(arrays[k]) != n for k in ("concept_ids", "numeric_mask", "elapsed_log_days")):
        raise ValueError("processed event array shape mismatch")
    if n and (ids.min() < 1 or ids.max() > len(concepts)):
        raise ValueError("processed concept ID outside dictionary")
    for field in ("values", "elapsed_log_days"):
        if not np.isfinite(arrays[field]).all():
            raise ValueError("processed event has nonfinite values")
    if np.any(arrays["elapsed_log_days"] < 0):
        raise ValueError("processed event precedes admission")
    med_lookup = np.array([False, *(name in MED_NAMES or name == "tetracycline" for name in concepts)], dtype=bool)
    is_lab = ~med_lookup[ids]
    numeric = arrays["numeric_mask"].astype(bool)
    if np.any(numeric & ~is_lab):
        raise ValueError("processed medication has a numeric mask")
    by_hadm = {int(hadm): i for i, hadm in enumerate(hadms)}
    refs = []
    for hadm, subject in zip(base.hadm_ids, base.subject_ids, strict=True):
        position = by_hadm[int(hadm)]
        if int(subjects[position]) != int(subject):
            raise ValueError("processed patient/admission identity mismatch")
        refs.append(np.arange(offsets[position], offsets[position + 1], dtype=np.int64))
    values = np.zeros((n, 4), dtype=np.float32)
    values[:, 0], values[:, 1] = arrays["values"], arrays["elapsed_log_days"]
    return EventData(base, values, ids.astype(np.int64), is_lab, tuple(refs), source_hashes,
                     concepts, numeric)


def load_processed_event_data(aligned_dir: Path, structured_dir: Path, base_csv: Path,
                              processed_dir: Path, raw_ledger_dir: Path, *, labels_csv: Path,
                              field_dir: Path, vectors_path: Path) -> tuple[EventData, dict[str, object]]:
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    raw_manifest = json.loads((raw_ledger_dir / "metadata/manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "preprocessed_and_checked" or raw_manifest["status"] != "validated":
        raise ValueError("unvalidated processed or raw ledger")
    for name, expected in manifest["output_sha256"].items():
        if _sha(processed_dir / name) != expected:
            raise ValueError("processed event artifact changed: " + name)
    for name, expected in manifest["input_sha256"].items():
        if raw_manifest["sources"][name]["sha256"] != expected or _sha(raw_ledger_dir / (name + ".csv.gz")) != expected:
            raise ValueError("processed event raw source changed: " + name)
    base = load_data(aligned_dir, structured_dir, base_csv, field_dir, vectors_path, labels_csv)
    raw_cohort = {r["hadm_id"]: r for r in _rows(raw_ledger_dir / "cohort/admissions.csv.gz")}
    raw_by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in raw_cohort.values():
        raw_by_subject[record["subject_id"]].append(record)
    eligible = {h: r for h, r in raw_cohort.items() if r["discharge_landmark_eligible"] == "t"}
    labels = {r["hadm_id"]: r for r in _rows(raw_ledger_dir / "outcomes/readmission_death_labels.csv.gz")}
    alternate_endpoint = base.label_name == STRICT_PRIMARY_EVENT
    if alternate_endpoint:
        strict_labels, strict_hashes = checked_strict_labels(raw_ledger_dir)
        strict_summary = json.loads((labels_csv.parent / "summary.json").read_text(encoding="utf-8"))
        if strict_summary.get("source_sha256") != strict_hashes:
            raise ValueError("strict label source contract differs")
        labels = strict_labels
    raw_splits = {r["subject_id"]: r["patient_split"] for r in _rows(raw_ledger_dir / "cohort/patient_splits.csv.gz")}
    context = {r["hadm_id"]: r for r in _rows(base_csv)}
    patients = {r["subject_id"]: r for r in _rows(raw_ledger_dir / "raw/patients.csv.gz")}
    if set(eligible) != set(map(str, base.hadm_ids)) or set(labels) != set(eligible):
        raise ValueError("legacy model cohort conflicts with current raw cohort")
    patient_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, hadm_id in enumerate(base.hadm_ids):
        hadm, subject = str(hadm_id), str(base.subject_ids[index])
        current, old, label = eligible[hadm], context[hadm], labels[hadm]
        for key in ("subject_id", "admittime", "dischtime"):
            if old[key] != current[key] or label[key] != current[key]:
                raise ValueError("legacy model identity/landmark conflicts with current cohort")
        if str(label["subject_id"]) != subject:
            raise ValueError("model subject conflicts with current cohort")
        if (int(label["event_type"]) != base.events[index]
                or not np.isclose(float(label["time_days"]), base.times[index], atol=1e-4)):
            raise ValueError("legacy labels conflict with current endpoint")
        if base.splits[index] != raw_splits[subject] or raw_splits[subject] != patient_split(int(subject)):
            raise ValueError("legacy patient split conflicts with current cohort")
        patient = patients[subject]
        age = min(90, max(0, int(patient["anchor_age"]) + datetime.fromisoformat(current["admittime"]).year - int(patient["anchor_year"])))
        if not np.isclose(float(old["age_at_admission"]), age) or old["gender"] != patient["gender"]:
            raise ValueError("legacy demographic context conflicts with raw patients")
        strict_history = sum(row["is_strict_aecopd"] == "t"
                             and row["admittime"] < current["admittime"] and row["dischtime"] <= current["admittime"]
                             for row in raw_by_subject[subject])
        if int(old["prior_strict_aecopd_count"]) != strict_history:
            raise ValueError("legacy strict history count conflicts with raw cohort")
        patient_rows[subject].append(current)
    unavailable_history = 0
    for records in patient_rows.values():
        records.sort(key=lambda r: (r["admittime"], r["hadm_id"]))
        for index, current in enumerate(records):
            unavailable_history += sum(datetime.fromisoformat(past["dischtime"]) > datetime.fromisoformat(current["dischtime"])
                                       for past in records[:index])
    if unavailable_history:
        raise ValueError("legacy admission-order attention includes history unavailable at prediction time")
    # 原静态 token 不重建，只核对确有来源的当前／既往共病与上下文字段。
    diagnosis: dict[str, set[str]] = defaultdict(set)
    for row in _rows(raw_ledger_dir / "raw/diagnoses_icd.csv.gz"):
        if row["hadm_id"] in eligible:
            diagnosis[row["hadm_id"]].update(diagnosis_groups(row["icd_code"], int(row["icd_version"])))
    histories: dict[str, set[str]] = {}
    for records in patient_rows.values():
        for current in records:
            histories[current["hadm_id"]] = set().union(*(diagnosis[past["hadm_id"]] for past in records
                if past["hadm_id"] != current["hadm_id"] and past["admittime"] < current["admittime"]
                and past["dischtime"] <= current["admittime"]))
    checked_static = 0
    for row in _rows(structured_dir / "structured_tokens.csv.gz"):
        hadm, kind, concept = row["hadm_id"], row["token_type"], row["concept"]
        current, old = eligible[hadm], context[hadm]
        expected: tuple[float, ...] | None = None
        if kind == "context":
            expected = ({"age_at_admission": float(old["age_at_admission"]), "male": float(old["gender"] == "M"),
                         "prior_coded_admissions": float(current["prior_coded_admission_count"]),
                         "prior_strict_aecopd_admissions": float(old["prior_strict_aecopd_count"])}[concept],)
        elif kind == "chronic_comorbidity":
            expected = float(concept in diagnosis[hadm]), float(concept in histories[hadm])
        elif kind == "acute_diagnosis":
            expected = (float(current["is_strict_aecopd"] == "t") if concept == "strict_aecopd" else float(concept in diagnosis[hadm]),)
        elif kind == "disposition":
            destination = current["discharge_location"].upper()
            category = "home" if "HOME" in destination else "facility" if any(t in destination for t in ("REHAB", "SKILLED", "NURSING", "LONG TERM")) else "other"
            if concept != category:
                raise ValueError("legacy discharge disposition conflicts with raw cohort")
            expected = (1.0,)
        if expected is not None:
            if any(not np.isclose(float(row[f"value{i}"]), value) for i, value in enumerate(expected)):
                raise ValueError("legacy static token conflicts with current raw source")
            checked_static += 1
    base, static_normalization = normalize_on_training_patients(base)
    retained = {i + 1 for i, name in enumerate(base.type_names) if name not in {"laboratory", "administered_medication"}}
    base = replace(base, tokens_by_admission=tuple(ref[np.isin(base.token_types[ref], list(retained))] for ref in base.tokens_by_admission))
    if any(len(ref) == 0 for ref in base.tokens_by_admission):
        raise ValueError("static context missing after removing obsolete summaries")
    with np.load(processed_dir / "event_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    hashes = {"processed_manifest": _sha(processed_dir / "manifest.json"),
              "event_arrays": _sha(processed_dir / "event_arrays.npz"),
              "text_vectors": _sha(vectors_path), "structured_tokens": _sha(structured_dir / "structured_tokens.csv.gz")}
    data = attach_arrays(base, arrays, tuple(manifest["concepts"]), hashes)
    split_admissions = Counter(base.splits.tolist())
    split_patients = Counter(str(base.splits[sequence[0]]) for sequence in base.sequences)
    text_admissions = sum(bool(ref) for ref in base.text_by_admission)
    audit = {"status": "model_data_contract_checked", "cohort_admissions": len(base.hadm_ids),
             "cohort_patients": len(base.sequences), "split_admissions": split_admissions,
             "split_patients": split_patients, "outcome_counts": Counter(map(int, base.events)),
             "current_labels_identical_to_raw_ledger": not alternate_endpoint,
             "strict_labels_independently_reconstructed_from_raw": alternate_endpoint,
             "landmarks_and_patient_splits_identical": True,
             "unavailable_future_history": unavailable_history, "static_tokens_checked": checked_static,
             "static_summary_lab_med_tokens_excluded": True, "text_admissions": text_admissions,
             "text_snippets": sum(len(ref) for ref in base.text_by_admission), "text_width": base.text_width,
             "events": len(data.event_values), "event_concepts": data.concept_names,
             "numeric_values_renormalized": False, "input_hashes": hashes}
    normalization = json.loads((processed_dir / "normalization.json").read_text(encoding="utf-8"))
    return data, {"static": static_normalization, "event_lab": normalization["lab"], "contract_audit": audit}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_abnormality_only.ps1 · lines 1–59

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_abnormality_only.ps1","part":1,"start_line":1,"end_line":59,"source_sha256":"7447fb0c430e5f0b3dba6f013c811c7a4e85639ad6f226650e2b85a8ed15d416","fragment_sha256":"7447fb0c430e5f0b3dba6f013c811c7a4e85639ad6f226650e2b85a8ed15d416","added_terminal_newline":false} -->
````powershell
param([int[]]$Seeds = @(2026), [string[]]$Modes = @('masked', 'active'))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = Join-Path $root 'training_runs/v5_abnormality_only_2026-09-29'
$features = 'patient_data/processed/abnormality_only_2026-09-29'
$processed = 'patient_data/processed/mimiciv_copd_events_formulation_verified_2026-09-29'
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$manifest = Get-Content -Raw -LiteralPath (Join-Path $root "$features/manifest.json") | ConvertFrom-Json
if ($manifest.status -ne 'derived_not_yet_validated_by_prediction' -or
    $manifest.feature_count -ne 132) { throw 'Abnormality features are not audited.' }
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$common = @(
    'run', '--rm', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    'aecopd-training:v1', 'python', '-m', 'training.v5.train_event_sequence',
    '--aligned-dir', "$private/v5_section_aligned/dataset_v3",
    '--structured-dir', "$private/v5_section_aligned/section_inputs_sparse_2026_09_26",
    '--base-csv', "$private/v5_mimiciv_2026-09-25/structured/base.csv",
    '--field-dir', "$private/v5_section_aligned/field_text_baseline_2026_09_27",
    '--vectors', "$private/v5_section_aligned/field_text_baseline_2026_09_27/bert/bioclinicalbert.float16.npy",
    '--labels-csv', "$private/v5_qualified_copd_labels_2026_09_27/labels.csv.gz",
    '--processed-event-dir', $processed,
    '--raw-ledger-dir', 'patient_data/mimiciv_copd/snapshot_2026-09-27',
    '--clinical-recovery-dir', $features,
    '--event-encoder', 'abnormal', '--max-epochs', '35', '--batch-size', '12'
)
foreach ($seed in $Seeds) {
    foreach ($mode in $Modes) {
        $name = "${mode}_seed${seed}"
        $output = Join-Path $runRoot $name
        if (Test-Path -LiteralPath (Join-Path $output 'report.json')) {
            $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
            if ($report.status -ne 'training_completed') { throw "Incomplete run: $output" }
            Write-Output "SKIP $name"
            continue
        }
        if (Test-Path -LiteralPath $output) { throw "Partial run requires inspection: $output" }
        Write-Output "START $name"
        & docker @common '--clinical-recovery-mode' $mode '--seed' "$seed" '--output-dir' "training_runs/v5_abnormality_only_2026-09-29/$name" *> (Join-Path $runRoot "$name.log")
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath (Join-Path $runRoot "$name.log") -Tail 50
            throw "Training failed: $name"
        }
        $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
        if ($report.status -ne 'training_completed' -or $report.raw_events_enter_model) {
            throw "Incorrect completed report: $name"
        }
        Write-Output "DONE $name epoch=$($report.best_epoch) valIBS=$($report.validation.temporal_full.integrated_brier_7_365)"
    }
    if (($Modes -contains 'masked') -and ($Modes -contains 'active')) {
        & docker run --rm -v "${root}:/workspace" -w /workspace -e PYTHONPATH=/workspace `
            aecopd-training:v1 python -m training.v5.compare_lab_recovery_pilot `
            --run-root training_runs/v5_abnormality_only_2026-09-29 `
            --output "training_runs/v5_abnormality_only_2026-09-29/comparison_seed${seed}.json" `
            --seed $seed --bootstrap 200
        if ($LASTEXITCODE -ne 0) { throw "Comparison failed: $seed" }
    }
}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_all_lab_axes.py · lines 1–100

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_all_lab_axes.py","part":1,"start_line":1,"end_line":100,"source_sha256":"d77a980f7c4f88be1c7e524d3a022a7bb59e300b7a2f34273f3121e7c9b8469e","fragment_sha256":"d77a980f7c4f88be1c7e524d3a022a7bb59e300b7a2f34273f3121e7c9b8469e","added_terminal_newline":false} -->
````python
"""同一已核验来源上运行完整22轴逐轴消融；给药只有二元暴露。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from patient_data.expanded_abnormality_features import AXES
from training.v5.medication_effect_data import attach_medication_effects
from training.v5.processed_event_data import load_processed_event_data
from training.v5.strict_endpoint_labels import STRICT_PRIMARY_EVENT
from training.v5.train_event_sequence import run

PRIVATE = Path("outputs/01a08b96-2988-7001-af1d-b40d901f3400/private")
RUN_ROOT = Path("training_runs/v5_strict_all22_binary_meds_one_seed_2026-09-30")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--axes", nargs="+", choices=AXES, default=list(AXES))
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--worker-name", default="batch")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.axes)) != len(args.axes):
        raise ValueError("duplicate seed or axis")
    paths = dict(
        aligned_dir=PRIVATE / "v5_section_aligned/dataset_v3",
        structured_dir=PRIVATE / "v5_section_aligned/section_inputs_sparse_2026_09_26",
        base_csv=PRIVATE / "v5_mimiciv_2026-09-25/structured/base.csv",
        field_dir=PRIVATE / "v5_section_aligned/field_text_baseline_2026_09_27",
        vectors=PRIVATE / "v5_section_aligned/field_text_baseline_2026_09_27/bert/bioclinicalbert.float16.npy",
        labels_csv=Path("patient_data/derived/strict_aecopd_endpoint_2026-09-28/labels.csv.gz"),
        processed_event_dir=Path("patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29"),
        raw_ledger_dir=Path("patient_data/mimiciv_copd/snapshot_2026-09-27"),
    )
    started = time.perf_counter()
    print("VERIFY common source once; no new text encoding", flush=True)
    source = load_processed_event_data(
        paths["aligned_dir"], paths["structured_dir"], paths["base_csv"],
        paths["processed_event_dir"], paths["raw_ledger_dir"],
        labels_csv=paths["labels_csv"], field_dir=paths["field_dir"], vectors_path=paths["vectors"],
    )
    exposures = attach_medication_effects(source[0], paths["processed_event_dir"],
                                         mode="exposure_only").medication_by_admission
    exposures.setflags(write=False)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    completed = []
    for seed in args.seeds:
        for axis in args.axes if args.skip_baseline else (None, *args.axes):
            name = f"full_seed{seed}" if axis is None else f"without_axis_{axis}_seed{seed}"
            mode = "active" if axis is None else f"without_axis_{axis}"
            output = RUN_ROOT / name
            if output.exists():
                report = json.loads((output / "report.json").read_text(encoding="utf-8"))
                status = json.loads((output / "status.json").read_text(encoding="utf-8"))
                if (status["status"] != "training_completed" or report["status"] != "training_completed"
                        or report["primary_event"] != STRICT_PRIMARY_EVENT or report["seed"] != seed
                        or report["clinical_recovery_mode"] != mode
                        or report["medication_effect_mode"] != "exposure_only"
                        or report["label_sha256"] != source[0].base.label_sha256
                        or report["field_manifest_sha256"] != source[0].base.manifest_sha256
                        or report["max_epochs"] != 35 or report["raw_events_enter_model"]):
                    raise ValueError(f"partial or inconsistent run: {output}")
                print(f"SKIP {name}", flush=True)
            else:
                print(f"START {name}", flush=True)
                report = run(
                    **paths, raw_dir=None, output_dir=output, process_only=False,
                    max_epochs=35, batch_size=12, event_encoder="abnormal_medication",
                    clinical_recovery_dir=Path("patient_data/processed/abnormality_clinical_reviewed_2026-09-29"),
                    clinical_recovery_mode=mode, abnormality_scale_mode="training_positive_p95",
                    medication_effect_mode="exposure_only", seed=seed,
                    prepared_source=source, prepared_exposure_only=exposures,
                )
                print(f"DONE {name} epochs={report['epochs_run']} best={report['best_epoch']} "
                      f"IBS={report['validation']['temporal_full']['integrated_brier_7_365']:.6f}", flush=True)
            completed.append(name)
            (RUN_ROOT / f"{args.worker_name}_status.json").write_text(json.dumps({
                "status": "running", "completed": completed,
                "planned": len(args.seeds) * (len(args.axes) + (not args.skip_baseline)),
                "elapsed_seconds": time.perf_counter() - started,
            }, indent=2), encoding="utf-8")
    (RUN_ROOT / f"{args.worker_name}_status.json").write_text(json.dumps({
        "status": "training_completed", "completed": completed, "planned": len(completed),
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2), encoding="utf-8")
    if tuple(args.axes) == tuple(AXES) and not args.skip_baseline:
        from training.v5.compare_all_lab_axes import compare
        compare(RUN_ROOT, RUN_ROOT / "paired_analysis", seeds=tuple(args.seeds), draws=100)
        (RUN_ROOT / f"{args.worker_name}_status.json").write_text(json.dumps({
            "status": "training_and_analysis_completed", "completed": completed, "planned": len(completed),
            "elapsed_seconds": time.perf_counter() - started,
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_broad_fragment_comparison.py · lines 1–206

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_broad_fragment_comparison.py","part":1,"start_line":1,"end_line":206,"source_sha256":"4771f29a479397088c235428fa6b647c6abc568a1d4a0a73982022386654c2cc","fragment_sha256":"4771f29a479397088c235428fa6b647c6abc568a1d4a0a73982022386654c2cc","added_terminal_newline":false} -->
````python
"""固定扩展文本批次的单种子四臂比较；不修改原模型或正式队列。"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from training.bonsai_broad_fragments import atomic_json
from training.v5.broad_fragment_data import (
    attach_broad_text, checked_selection, old_or_no_text, preserve_selected_histories, sha,
)
from training.v5.compare_screened_group_ablations import _summary_for_cause
from training.v5.independent_data import split_sequences
from training.v5.lab_recovery_data import attach_abnormalities
from training.v5.medication_effect_data import EFFECT_GROUPS, attach_medication_effects
from training.v5.medication_effect_model import MedicationEffectFusionModel
from training.v5.processed_event_data import load_processed_event_data
from training.v5.strict_endpoint_labels import STRICT_PRIMARY_EVENT
from training.v5.train import _save_curves, metrics
from training.v5.train_event_sequence import _build_model_with_matched_text_initialization, fit, predict

ARMS = ('no_text', 'old_fields_bert', 'fragments_bert', 'fragments_qwen')
PRIVATE = Path('outputs/01a08b96-2988-7001-af1d-b40d901f3400/private')
SNAPSHOT = Path('patient_data/mimiciv_copd/snapshot_2026-09-27')
PROCESSED = Path('patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29')
LABELS = Path('patient_data/derived/strict_aecopd_endpoint_2026-09-28/labels.csv.gz')


def build_model(data, device):
    return _build_model_with_matched_text_initialization(
        MedicationEffectFusionModel, concept_count=len(data.base.concept_names),
        type_count=len(data.base.type_names), field_count=len(data.base.field_roles),
        text_width=data.base.text_width, shared_text_init_width=768,
        event_arguments={'abnormality_width': data.recovery_by_admission.shape[1]}, device=device)


def initial_shared_hash(data, seed, device):
    torch.manual_seed(seed)
    model = build_model(data, device)
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not name.startswith('text_projection.'):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    count = sum(p.numel() for p in model.parameters())
    del model
    return digest.hexdigest(), count


def train_arm(data, selection, output: Path, arm: str, seed: int, epochs: int) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    sequences = {s: split_sequences(data.base, s) for s in ('train', 'validation', 'test')}
    indices = {s: np.concatenate(rs) for s, rs in sequences.items()}
    initial_hash, parameter_count = initial_shared_hash(data, seed, device)
    started = time.perf_counter()
    atomic_json(output/'status.json', {'status': 'training', 'arm': arm})
    model, history, best_epoch, selected_risk = fit(
        data, sequences['train'], sequences['validation'], process_only=False,
        max_epochs=epochs, batch_size=12, device=device, event_encoder='abnormal_medication',
        seed=seed, shared_text_init_width=768, history_path=output/'history.json')
    torch.save(model.state_dict(), output/'model.pt')
    reloaded = build_model(data, device)
    reloaded.load_state_dict(torch.load(output/'model.pt', map_location=device, weights_only=True))
    reloaded.eval()
    curves, scores, note_scores = {}, {}, {}
    note_ids = {int(r['hadm_id']) for r in selection['members']}
    for split, key in (('validation', 'validation'), ('historical_test', 'test')):
        index = indices[key]
        cif, _ = predict(reloaded, data, sequences[key], process_only=False, batch_size=12,
                         device=device, return_all_causes=True)
        if split == 'validation':
            difference = float(np.max(np.abs(cif[:, :, 1]-selected_risk)))
            atomic_json(output/'checkpoint_prediction_check.json', {
                'passed': difference <= 1e-6, 'maximum_absolute_difference': difference,
                'absolute_tolerance': 1e-6, 'reloaded_from_saved_weights': True})
            if difference > 1e-6:
                raise ValueError('Reloaded best checkpoint differs from selected validation predictions')
        times, events = data.base.times[index], data.base.events[index]
        curves[split] = {arm: metrics(times, events, cif[:, :, 1])}
        scores[split] = {name: _summary_for_cause(times, events, cif, cause)
                         for cause, name in ((1, 'readmission'), (2, 'death'))}
        mask = np.isin(data.base.hadm_ids[index], list(note_ids))
        note_scores[split] = {name: _summary_for_cause(times[mask], events[mask], cif[mask], cause)
                              for cause, name in ((1, 'readmission'), (2, 'death'))}
        np.savez_compressed(output/f'{split}_predictions.npz',
            subject_ids=data.base.subject_ids[index], hadm_ids=data.base.hadm_ids[index],
            times=times, events=events, risk=cif[:, :, 1], competing_cif=cif, note_member=mask)
    _save_curves(output, curves)
    report = {'status': 'training_completed', 'arm': arm, 'seed': seed,
        'model': 'V5 static + lab abnormality + binary medication + gated text; causal patient Transformer; two-cause Weibull',
        'cohort_patients': len(data.base.sequences), 'cohort_admissions': sum(len(s) for s in data.base.sequences),
        'split_counts': {s: {'patients': len(sequences[s]), 'admissions': len(indices[s]),
            'readmissions': int((data.base.events[indices[s]] == 1).sum()),
            'deaths': int((data.base.events[indices[s]] == 2).sum())} for s in sequences},
        'primary_event': data.base.label_name, 'label_sha256': data.base.label_sha256,
        'member_sha256': selection['member_sha256'], 'field_manifest_sha256': data.base.manifest_sha256,
        'field_roles': data.base.field_roles, 'text_width': data.base.text_width,
        'text_vector_sha256': None if arm == 'no_text' else hashlib.sha256(np.asarray(data.base.text_vectors).tobytes()).hexdigest(),
        'shared_initialization_sha256': initial_hash, 'shared_text_init_width': 768,
        'parameter_count': parameter_count, 'epochs_run': len(history), 'best_epoch': best_epoch,
        'max_epochs': epochs, 'patience': 7, 'learning_rate': 8e-4, 'weight_decay': .02,
        'patient_batch_size': 12, 'device': torch.cuda.get_device_name(device),
        'selection': 'validation 7-365d integrated Brier', 'raw_events_enter_model': False,
        'clinical_recovery_axes': 22, 'medication_effect_mode': 'exposure_only; diuretic masked',
        'medication_nonconstant_inputs': 7,
        'old_text_restricted_to_same_note_members': True,
        'text_admissions_selected_histories': sum(bool(data.base.text_by_admission[i]) for ix in indices.values() for i in ix),
        'all_eligible_histories_retained': True, 'selected_note_histories_complete': True,
        'structural_scaling': 'existing full training fold static normalization and training-positive P95; identical frozen values across arms; no validation/test fitting',
        'validation': scores['validation'], 'historical_test': scores['historical_test'],
        'note_member_only': note_scores, 'seconds': time.perf_counter()-started,
        'limitations': ['Single seed; exploratory small training subset of 108 patients.',
            'Historical test previously examined; evaluation subset expanded after support audit, not a new blind test.',
            'Source comparison changes clinical content coverage and text roles; not proof of an isolated LLM causal contribution.',
            'Qwen and BERT have different frozen widths and projection parameter counts.',
            'World A observed 365-day administrative contract; no complete outside-hospital follow-up is proven.',
            'Source spans verified, not a gold-standard clinical extraction accuracy assessment.']}
    atomic_json(output/'report.json', report)
    atomic_json(output/'status.json', {'status': 'training_completed'})
    del model, reloaded
    torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-epochs', type=int, default=35)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU training fallback')
    audit = json.loads((args.batch/'audit_2026-10-05/summary.json').read_text(encoding='utf-8'))
    if audit['status'] != 'source_provenance_checked_evaluation_ready':
        raise ValueError('Batch provenance/event support has not been checked')
    old = PRIVATE/'v5_section_aligned/field_text_baseline_2026_09_27'
    args.output.mkdir(parents=True)
    atomic_json(args.output/'status.json', {'status': 'checking_sources', 'planned_arms': ARMS})
    try:
        source, preprocessing = load_processed_event_data(
            PRIVATE/'v5_section_aligned/dataset_v3',
            PRIVATE/'v5_section_aligned/section_inputs_sparse_2026_09_26',
            PRIVATE/'v5_mimiciv_2026-09-25/structured/base.csv', PROCESSED, SNAPSHOT,
            labels_csv=LABELS, field_dir=old, vectors_path=old/'bert/bioclinicalbert.float16.npy')
        selection = checked_selection(source, args.batch/'selection.json')
        if (selection['member_sha256'] != audit['member_sha256'] or
                source.base.label_sha256 != audit['label_sha256'] or source.base.label_name != STRICT_PRIMARY_EVENT):
            raise ValueError('Audit membership/endpoint changed')
        source = attach_abnormalities(source, Path('patient_data/processed/abnormality_clinical_reviewed_2026-09-29'),
                                      mode='active', scale_mode='training_positive_p95')
        source = attach_medication_effects(source, PROCESSED, mode='exposure_only')
        medication = source.medication_by_admission.copy().reshape(len(source.base.hadm_ids), len(EFFECT_GROUPS), 6)
        medication[:, tuple(EFFECT_GROUPS).index('diuretic'), :] = 0
        if np.any(medication[:, :, 1:] != 0) or not np.isin(medication[:, :, 0], (0., 1.)).all():
            raise ValueError('Nonbinary/time medication input entered comparison')
        source = replace(source, medication_by_admission=medication.reshape(len(medication), -1),
                         events_by_admission=tuple(np.empty(0, dtype=np.int64) for _ in source.base.hadm_ids))
        source = preserve_selected_histories(source, selection)
        atomic_json(args.output/'preprocessing.json', preprocessing)
        structural_digest = hashlib.sha256()
        for array in (source.base.token_values, source.base.token_masks, source.base.token_concepts,
                      source.base.token_types, source.base.gap, source.recovery_by_admission,
                      source.medication_by_admission):
            structural_digest.update(np.asarray(array).tobytes())
        atomic_json(args.output/'contract.json', {'member_sha256': selection['member_sha256'],
            'label_sha256': source.base.label_sha256, 'primary_event': source.base.label_name,
            'structural_input_sha256': structural_digest.hexdigest(), 'source_hashes': source.source_hashes,
            'same_structural_object_for_all_arms': True, 'training_patients': 108,
            'field_vocabulary': (*source.base.field_roles, 'clinical_excerpt'),
            'raw_events_enter_model': False, 'medications_binary_only': True, 'diuretic_masked': True,
            'all_22_lab_axes_retained_no_unapproved_axis_deletion': True,
            'arms': ARMS, 'seed': args.seed, 'max_epochs': args.max_epochs})
        reports, expected_hash = {}, None
        for arm in ARMS:
            data = (old_or_no_text(source, selection, enabled=arm == 'old_fields_bert')
                    if arm in ('no_text', 'old_fields_bert') else
                    attach_broad_text(source, selection, args.batch/'encoding',
                                      'bert' if arm == 'fragments_bert' else 'qwen'))
            initial_hash, _ = initial_shared_hash(data, args.seed, torch.device('cuda'))
            if expected_hash is not None and initial_hash != expected_hash:
                raise ValueError('Shared non-text-projection weights differ between arms')
            expected_hash = initial_hash
            atomic_json(args.output/'status.json', {'status': 'training', 'current_arm': arm,
                                                   'completed_arms': list(reports)})
            reports[arm] = train_arm(data, selection, args.output/arm, arm, args.seed, args.max_epochs)
            print(json.dumps({'completed_arm': arm, 'validation': reports[arm]['validation']['readmission']}, ensure_ascii=False), flush=True)
        from training.v5.compare_broad_fragments import compare
        compare(args.output)
        atomic_json(args.output/'status.json', {'status': 'training_and_analysis_completed', 'completed_arms': list(reports)})
    except Exception as error:
        atomic_json(args.output/'status.json', {'status': 'failed', 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_incremental_analysis.ps1 · lines 1–52

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_incremental_analysis.ps1","part":1,"start_line":1,"end_line":52,"source_sha256":"5ac5a1bebccc0f054050e02c7e39843812c5471016dbac312402600422890e18","fragment_sha256":"5ac5a1bebccc0f054050e02c7e39843812c5471016dbac312402600422890e18","added_terminal_newline":false} -->
````powershell
param([int[]]$Seeds = @(2026, 2027, 2028))
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$resultRoot = 'training_runs/v5_incremental_history_evidence_2026-09-28'
$dataRoot = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$dockerBase = @('run', '--rm', '-v', "${root}:/workspace", '-w', '/workspace',
                '-e', 'PYTHONPATH=/workspace:/workspace/training:/workspace/model_loader',
                'aecopd-baselines:v1', 'python', '-m')
$fullDirs = @{
    2026 = 'training_runs/v5_with_prior_counts_strict_aecopd_bert_2026-09-28'
    2027 = 'training_runs/v5_strict_full_seed2027_2026-09-28'
    2028 = 'training_runs/v5_strict_full_seed2028_2026-09-28'
}
foreach ($seed in $Seeds) {
    $comparators = @()
    foreach ($name in @('no_current_static', 'no_medication_events', 'no_lab_time', 'one_lab_per_concept')) {
        $directory = "$resultRoot/${name}_seed$seed"
        if ($seed -eq 2026 -and $name -eq 'no_current_static') {
            $directory = 'training_runs/v5_history_evidence_no_current_static_seed2026_2026-09-28'
        }
        if (-not (Test-Path -LiteralPath (Join-Path $directory 'report.json'))) {
            throw "Missing completed training: $directory"
        }
        $comparators += @('--comparator', $name, $directory)
    }
    if ($seed -eq 2026) {
        foreach ($name in @('history_of_present_illness', 'past_medical_history', 'hospital_course', 'discharge_condition')) {
            $comparators += @('--comparator', "without_text_$name", "$resultRoot/without_text_${name}_seed2026")
        }
        $comparators += @('--comparator', 'no_text', 'training_runs/v5_strict_retrained_without_text_2026-09-28')
        $comparators += @('--comparator', 'no_lab_values', 'training_runs/v5_strict_retrained_without_lab_values_2026-09-28')
        $comparators += @('--comparator', 'no_events', 'training_runs/v5_strict_retrained_without_events_2026-09-28')
    } else {
        $comparators += @('--comparator', 'no_lab_values', "training_runs/v5_strict_no_lab_values_seed${seed}_2026-09-28")
    }
    $paired = "$resultRoot/paired_seed$seed.json"
    if (-not (Test-Path -LiteralPath $paired)) {
        Write-Output "START paired seed=$seed"
        & docker @dockerBase 'training.v5.compare_strict_ablations' '--candidate-dir' $fullDirs[$seed] @comparators '--output-path' $paired
        if ($LASTEXITCODE -ne 0) { throw "Paired comparison failed for seed $seed" }
    }
    $strata = "$resultRoot/strata_seed$seed.json"
    if (-not (Test-Path -LiteralPath $strata)) {
        Write-Output "START strata seed=$seed"
        & docker @dockerBase 'training.v5.analyze_incremental_evidence' `
            '--base-csv' "$dataRoot/v5_mimiciv_2026-09-25/structured/base.csv" `
            '--labels-csv' 'patient_data/derived/strict_aecopd_endpoint_2026-09-28/labels.csv.gz' `
            '--full-dir' $fullDirs[$seed] @comparators '--output-path' $strata
        if ($LASTEXITCODE -ne 0) { throw "Stratified comparison failed for seed $seed" }
    }
    Write-Output "DONE evidence seed=$seed"
}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_incremental_evidence.ps1 · lines 1–68

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_incremental_evidence.ps1","part":1,"start_line":1,"end_line":68,"source_sha256":"1bd56e04e33c39b195500126d32bac41ba26f383213235a0927248570759a52f","fragment_sha256":"1bd56e04e33c39b195500126d32bac41ba26f383213235a0927248570759a52f","added_terminal_newline":false} -->
````powershell
param(
    [int[]]$Seeds = @(2026, 2027, 2028)
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$outputRoot = Join-Path $root 'training_runs/v5_incremental_history_evidence_2026-09-28'
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$dataRoot = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$common = @(
    'run', '--rm', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    'aecopd-training:v1', 'python', '-m', 'training.v5.train_event_sequence',
    '--aligned-dir', "$dataRoot/v5_section_aligned/dataset_v3",
    '--structured-dir', "$dataRoot/v5_section_aligned/section_inputs_sparse_2026_09_26",
    '--base-csv', "$dataRoot/v5_mimiciv_2026-09-25/structured/base.csv",
    '--field-dir', "$dataRoot/v5_section_aligned/field_text_baseline_2026_09_27",
    '--vectors', "$dataRoot/v5_section_aligned/field_text_baseline_2026_09_27/bert/bioclinicalbert.float16.npy",
    '--labels-csv', 'patient_data/derived/strict_aecopd_endpoint_2026-09-28/labels.csv.gz',
    '--processed-event-dir', 'patient_data/processed/mimiciv_copd_events_2026-09-27',
    '--raw-ledger-dir', 'patient_data/mimiciv_copd/snapshot_2026-09-27',
    '--event-encoder', 'set'
)
$variants = @(
    @{ Name = 'no_current_static'; Args = @('--disable-current-static') },
    @{ Name = 'no_medication_events'; Args = @('--disable-medications') },
    @{ Name = 'no_lab_time'; Args = @('--lab-process', 'no_time') },
    @{ Name = 'one_lab_per_concept'; Args = @('--lab-process', 'one_per_concept') }
)
$textFields = @('history_of_present_illness', 'past_medical_history', 'hospital_course', 'discharge_condition')
foreach ($field in $textFields) {
    $active = ($textFields | Where-Object { $_ -ne $field }) -join ','
    $variants += @{ Name = "without_text_$field"; Args = @('--active-text-fields', $active); OnlySeed2026 = $true }
}
foreach ($seed in $Seeds) {
    foreach ($variant in $variants) {
        if ($variant.OnlySeed2026 -and $seed -ne 2026) { continue }
        $name = "$($variant.Name)_seed$seed"
        if ($name -eq 'no_current_static_seed2026') {
            $existing = Join-Path $root 'training_runs/v5_history_evidence_no_current_static_seed2026_2026-09-28/report.json'
            if (-not (Test-Path -LiteralPath $existing)) { throw "Initial run is not complete: $existing" }
            $initialReport = Get-Content -Raw -LiteralPath $existing | ConvertFrom-Json
            if ($initialReport.status -ne 'training_completed') { throw "Initial run is incomplete: $existing" }
            Write-Output "SKIP $name (initial run already complete)"
            continue
        }
        $output = Join-Path $outputRoot $name
        $reportPath = Join-Path $output 'report.json'
        if (Test-Path -LiteralPath $reportPath) {
            $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
            if ($report.status -ne 'training_completed') { throw "Incomplete report: $reportPath" }
            Write-Output "SKIP $name (already complete)"
            continue
        }
        if (Test-Path -LiteralPath $output) { throw "Partial run needs inspection: $output" }
        Write-Output "START $name"
        $log = Join-Path $outputRoot "$name.log"
        $variantArgs = $variant.Args
        & docker @common '--output-dir' "training_runs/v5_incremental_history_evidence_2026-09-28/$name" '--seed' "$seed" @variantArgs *> $log
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath $log -Tail 60
            throw "Training failed: $name (see $log)"
        }
        $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
        if ($report.status -ne 'training_completed') { throw "Missing complete report: $reportPath" }
        Write-Output "DONE $name best_epoch=$($report.best_epoch) validation_IBS=$($report.validation.temporal_full.integrated_brier_7_365)"
    }
}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_lab_recovery_pilot.ps1 · lines 1–49

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_lab_recovery_pilot.ps1","part":1,"start_line":1,"end_line":49,"source_sha256":"7c1f93e196d97a5f42ec2c28fb8a6772f0b41352746f276ee27eb31e216afaef","fragment_sha256":"7c1f93e196d97a5f42ec2c28fb8a6772f0b41352746f276ee27eb31e216afaef","added_terminal_newline":false} -->
````powershell
param([int[]]$Seeds = @(2026))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = Join-Path $root 'training_runs/v5_lab_recovery_pilot_2026-09-29'
$features = 'patient_data/processed/lab_recovery_2026-09-29'
$processed = 'patient_data/processed/mimiciv_copd_events_formulation_verified_2026-09-29'
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$featureManifest = Get-Content -Raw -LiteralPath (Join-Path $root "$features/manifest.json") | ConvertFrom-Json
if ($featureManifest.status -ne 'derived_not_yet_validated_by_prediction' -or
    $featureManifest.feature_count -ne 54) { throw 'Clinical recovery features are not audited.' }
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$common = @(
    'run', '--rm', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    'aecopd-training:v1', 'python', '-m', 'training.v5.train_event_sequence',
    '--aligned-dir', "$private/v5_section_aligned/dataset_v3",
    '--structured-dir', "$private/v5_section_aligned/section_inputs_sparse_2026_09_26",
    '--base-csv', "$private/v5_mimiciv_2026-09-25/structured/base.csv",
    '--field-dir', "$private/v5_section_aligned/field_text_baseline_2026_09_27",
    '--vectors', "$private/v5_section_aligned/field_text_baseline_2026_09_27/bert/bioclinicalbert.float16.npy",
    '--labels-csv', "$private/v5_qualified_copd_labels_2026_09_27/labels.csv.gz",
    '--processed-event-dir', $processed,
    '--raw-ledger-dir', 'patient_data/mimiciv_copd/snapshot_2026-09-27',
    '--clinical-recovery-dir', $features,
    '--event-encoder', 'set', '--max-epochs', '35', '--batch-size', '12'
)
foreach ($seed in $Seeds) {
    foreach ($mode in @('masked', 'active')) {
        $name = "${mode}_seed${seed}"
        $output = Join-Path $runRoot $name
        if (Test-Path -LiteralPath (Join-Path $output 'report.json')) {
            $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
            if ($report.status -ne 'training_completed') { throw "Incomplete run: $output" }
            Write-Output "SKIP $name"
            continue
        }
        if (Test-Path -LiteralPath $output) { throw "Partial run requires inspection: $output" }
        Write-Output "START $name"
        & docker @common '--clinical-recovery-mode' $mode '--seed' "$seed" '--output-dir' "training_runs/v5_lab_recovery_pilot_2026-09-29/$name" *> (Join-Path $runRoot "$name.log")
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath (Join-Path $runRoot "$name.log") -Tail 40
            throw "Training failed: $name"
        }
        $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
        if ($report.status -ne 'training_completed') { throw "Missing completed report: $name" }
        Write-Output "DONE $name epoch=$($report.best_epoch) valIBS=$($report.validation.temporal_full.integrated_brier_7_365)"
    }
}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_medication_effect_ablations.ps1 · lines 1–64

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_medication_effect_ablations.ps1","part":1,"start_line":1,"end_line":64,"source_sha256":"c7f97db0be19fcf8dfa1b06f1069cb5138928b4f08bc0649f23b1e02121dd0aa","fragment_sha256":"c7f97db0be19fcf8dfa1b06f1069cb5138928b4f08bc0649f23b1e02121dd0aa","added_terminal_newline":false} -->
````powershell
param([int[]]$Seeds = @(2026, 2027, 2028),
      [string[]]$Modes = @('masked', 'active'))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = Join-Path $root 'training_runs/v5_medication_effect_2026-09-29'
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$processed = 'patient_data/processed/mimiciv_copd_events_formulation_verified_2026-09-29'
$features = 'patient_data/processed/abnormality_only_2026-09-29'
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$common = @(
    'run', '--rm', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    'aecopd-training:v1', 'python', '-m', 'training.v5.train_event_sequence',
    '--aligned-dir', "$private/v5_section_aligned/dataset_v3",
    '--structured-dir', "$private/v5_section_aligned/section_inputs_sparse_2026_09_26",
    '--base-csv', "$private/v5_mimiciv_2026-09-25/structured/base.csv",
    '--field-dir', "$private/v5_section_aligned/field_text_baseline_2026_09_27",
    '--vectors', "$private/v5_section_aligned/field_text_baseline_2026_09_27/bert/bioclinicalbert.float16.npy",
    '--labels-csv', "$private/v5_qualified_copd_labels_2026_09_27/labels.csv.gz",
    '--processed-event-dir', $processed,
    '--raw-ledger-dir', 'patient_data/mimiciv_copd/snapshot_2026-09-27',
    '--clinical-recovery-dir', $features, '--clinical-recovery-mode', 'active',
    '--event-encoder', 'abnormal_medication', '--max-epochs', '35', '--batch-size', '12'
)
foreach ($seed in $Seeds) {
    foreach ($mode in $Modes) {
        $name = "${mode}_seed${seed}"
        $output = Join-Path $runRoot $name
        if (Test-Path -LiteralPath (Join-Path $output 'report.json')) {
            $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
            if ($report.status -ne 'training_completed' -or $report.medication_effect_mode -ne $mode) {
                throw "Inconsistent completed run: $output"
            }
            Write-Output "SKIP $name"
            continue
        }
        if (Test-Path -LiteralPath $output) { throw "Partial run requires inspection: $output" }
        Write-Output "START $name"
        & docker @common '--medication-effect-mode' $mode '--seed' "$seed" `
            '--output-dir' "training_runs/v5_medication_effect_2026-09-29/$name" `
            *> (Join-Path $runRoot "$name.log")
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath (Join-Path $runRoot "$name.log") -Tail 70
            throw "Training failed: $name"
        }
        $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
        if ($report.status -ne 'training_completed' -or $report.raw_events_enter_model) {
            throw "Incorrect run contract: $name"
        }
        Write-Output "DONE $name epoch=$($report.best_epoch) valIBS=$($report.validation.temporal_full.integrated_brier_7_365)"
    }
    if (($Modes -contains 'masked') -and ($Modes -contains 'active')) {
        $comparison = Join-Path $runRoot "comparison_masked_seed${seed}.json"
        if (-not (Test-Path -LiteralPath $comparison)) {
            & docker run --rm -v "${root}:/workspace" -w /workspace -e PYTHONPATH=/workspace `
                aecopd-training:v1 python -m training.v5.compare_medication_effect_ablations `
                --run-root training_runs/v5_medication_effect_2026-09-29 `
                --output "training_runs/v5_medication_effect_2026-09-29/comparison_masked_seed${seed}.json" `
                --seed $seed --control masked --bootstrap 200
            if ($LASTEXITCODE -ne 0) { throw "Comparison failed: $seed" }
        }
    }
}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_medication_exposure_only.ps1 · lines 1–21

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_medication_exposure_only.ps1","part":1,"start_line":1,"end_line":21,"source_sha256":"2c7335299bcf9fca50df3440ac66e7213b064e93eec887de4d39bc2f0a930e9b","fragment_sha256":"2c7335299bcf9fca50df3440ac66e7213b064e93eec887de4d39bc2f0a930e9b","added_terminal_newline":false} -->
````powershell
param([int[]]$Seeds = @(2026, 2027, 2028))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = Join-Path $root 'training_runs/v5_medication_effect_2026-09-29'
foreach ($seed in $Seeds) {
    if (-not (Test-Path -LiteralPath (Join-Path $runRoot "active_seed${seed}/report.json"))) {
        throw "Active reference run missing for seed $seed"
    }
    & (Join-Path $PSScriptRoot 'run_medication_effect_ablations.ps1') -Seeds $seed -Modes 'exposure_only'
    if ($LASTEXITCODE -ne 0) { throw "Exposure-only training failed: seed=$seed" }
    $comparison = Join-Path $runRoot "comparison_exposure_only_seed${seed}.json"
    if (-not (Test-Path -LiteralPath $comparison)) {
        & docker run --rm -v "${root}:/workspace" -w /workspace -e PYTHONPATH=/workspace `
            aecopd-training:v1 python -m training.v5.compare_medication_effect_ablations `
            --run-root training_runs/v5_medication_effect_2026-09-29 `
            --output "training_runs/v5_medication_effect_2026-09-29/comparison_exposure_only_seed${seed}.json" `
            --seed $seed --control exposure_only --bootstrap 200
        if ($LASTEXITCODE -ne 0) { throw "Exposure-only comparison failed: seed=$seed" }
    }
}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_medication_group_ablations.ps1 · lines 1–30

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_medication_group_ablations.ps1","part":1,"start_line":1,"end_line":30,"source_sha256":"4f41f52c997ddb2b2dec45e3464b9c661bd195df41ebcf76f87b4e5cf8ff8b5e","fragment_sha256":"4f41f52c997ddb2b2dec45e3464b9c661bd195df41ebcf76f87b4e5cf8ff8b5e","added_terminal_newline":false} -->
````powershell
param([int[]]$Seeds = @(2026))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = Join-Path $root 'training_runs/v5_medication_effect_2026-09-29'
$groups = @(
    'short_beta_agonist', 'short_muscarinic_antagonist',
    'long_beta_agonist', 'long_muscarinic_antagonist',
    'inhaled_corticosteroid', 'systemic_corticosteroid',
    'common_systemic_antibacterial', 'diuretic'
)
foreach ($seed in $Seeds) {
    if (-not (Test-Path -LiteralPath (Join-Path $runRoot "active_seed${seed}/report.json"))) {
        throw "Active reference run missing for seed $seed"
    }
    foreach ($group in $groups) {
        $mode = "without_$group"
        & (Join-Path $PSScriptRoot 'run_medication_effect_ablations.ps1') -Seeds $seed -Modes $mode
        if ($LASTEXITCODE -ne 0) { throw "Ablation failed: $mode seed=$seed" }
        $comparison = Join-Path $runRoot "comparison_${mode}_seed${seed}.json"
        if (-not (Test-Path -LiteralPath $comparison)) {
            & docker run --rm -v "${root}:/workspace" -w /workspace -e PYTHONPATH=/workspace `
                aecopd-training:v1 python -m training.v5.compare_medication_effect_ablations `
                --run-root training_runs/v5_medication_effect_2026-09-29 `
                --output "training_runs/v5_medication_effect_2026-09-29/comparison_${mode}_seed${seed}.json" `
                --seed $seed --control $mode --bootstrap 200
            if ($LASTEXITCODE -ne 0) { throw "Comparison failed: $mode seed=$seed" }
        }
    }
}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_parallel_all_lab_axes.ps1 · lines 1–33

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_parallel_all_lab_axes.ps1","part":1,"start_line":1,"end_line":33,"source_sha256":"2b6823bec5bb1aa6e6ce4ac6890f86cae605e723afb3fffaca9aa7b1e9dca4ba","fragment_sha256":"2b6823bec5bb1aa6e6ce4ac6890f86cae605e723afb3fffaca9aa7b1e9dca4ba","added_terminal_newline":false} -->
````powershell
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $root
$runRoot = Join-Path $root 'training_runs/v5_strict_all22_binary_meds_one_seed_2026-09-30'
if (-not (Test-Path -LiteralPath $runRoot)) { New-Item -ItemType Directory -Path $runRoot | Out-Null }
$firstAxes = @('arterial_low_ph', 'arterial_high_pco2', 'bicarbonate_low', 'bicarbonate_high',
    'wbc_high', 'albumin_low', 'hemoglobin_low', 'creatinine_high', 'arterial_hypercapnic_acidosis',
    'wbc_low', 'sodium_low')
$secondAxes = @('sodium_high', 'potassium_low', 'potassium_high', 'urea_nitrogen_high',
    'glucose_low', 'glucose_high', 'platelet_low', 'magnesium_low', 'lactate_high', 'crp_high', 'ntprobnp_high')
$common = @('run', '-d', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    '-e', 'CUBLAS_WORKSPACE_CONFIG=:4096:8', '-e', 'OMP_NUM_THREADS=2', '-e', 'MKL_NUM_THREADS=2')
& docker @common '--name' 'aecopd-lab22-single-a' 'aecopd-training:v1' `
    'python' '-u' '-m' 'training.v5.run_all_lab_axes' '--seeds' '2026' '--worker-name' 'worker_a' '--axes' @firstAxes
if ($LASTEXITCODE -ne 0) { throw 'First worker could not start' }
& docker @common '--name' 'aecopd-lab22-single-b' 'aecopd-training:v1' `
    'python' '-u' '-m' 'training.v5.run_all_lab_axes' '--seeds' '2026' '--worker-name' 'worker_b' `
    '--skip-baseline' '--axes' @secondAxes
if ($LASTEXITCODE -ne 0) { throw 'Second worker could not start; first worker may still be running' }
$failed = @()
foreach ($name in @('aecopd-lab22-single-a', 'aecopd-lab22-single-b')) {
    $exitCode = & docker wait $name
    & docker logs $name *> (Join-Path $runRoot "$name.log")
    if ($LASTEXITCODE -ne 0 -or "$exitCode".Trim() -ne '0') { $failed += $name }
}
if ($failed.Count -gt 0) { throw "Training failed: $failed; completed artifacts are preserved" }
& docker run --rm -v "${root}:/workspace" -w /workspace -e PYTHONPATH=/workspace `
    -e MPLCONFIGDIR=/tmp/matplotlib aecopd-training:v1 python -u -m training.v5.compare_all_lab_axes `
    --run-root training_runs/v5_strict_all22_binary_meds_one_seed_2026-09-30 `
    --output-dir training_runs/v5_strict_all22_binary_meds_one_seed_2026-09-30/paired_analysis `
    --seeds 2026 *> (Join-Path $runRoot 'analysis.log')
if ($LASTEXITCODE -ne 0) { throw 'Paired analysis failed; inspect analysis.log' }
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_revised_group_ablations.ps1 · lines 1–85

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_revised_group_ablations.ps1","part":1,"start_line":1,"end_line":85,"source_sha256":"1304686c61fc66063026a54a0f69ec07fedd442a222c7e4be5da0a747f43e9c3","fragment_sha256":"1304686c61fc66063026a54a0f69ec07fedd442a222c7e4be5da0a747f43e9c3","added_terminal_newline":false} -->
````powershell
param([int[]]$Seeds = @(2026, 2027, 2028),
      [string[]]$Groups = @(
          'med_short_beta_agonist', 'med_short_muscarinic_antagonist',
          'med_long_beta_agonist', 'med_long_muscarinic_antagonist',
          'med_inhaled_corticosteroid', 'med_systemic_corticosteroid',
          'med_common_systemic_antibacterial', 'med_diuretic',
          'lab_ventilation_acid_base', 'lab_inflammation_infection',
          'lab_nutrition_hematology', 'lab_renal', 'lab_electrolytes',
          'lab_metabolic_perfusion', 'lab_cardiac_burden'))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$parent = Join-Path $root 'training_runs/v5_revised_labs_meds_2026-09-29'
$runRoot = Join-Path $parent 'groups'
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$processed = 'patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29'
$features = 'patient_data/processed/abnormality_clinical_reviewed_2026-09-29'
$medGroups = @('short_beta_agonist', 'short_muscarinic_antagonist', 'long_beta_agonist',
               'long_muscarinic_antagonist', 'inhaled_corticosteroid',
               'systemic_corticosteroid', 'common_systemic_antibacterial', 'diuretic',
               'short_bronchodilators', 'maintenance_inhalers')
$labGroups = @('ventilation_acid_base', 'inflammation_infection', 'nutrition_hematology',
               'renal', 'electrolytes', 'metabolic_perfusion', 'cardiac_burden')
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$common = @(
    'run', '--rm', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    'aecopd-training:v1', 'python', '-m', 'training.v5.train_event_sequence',
    '--aligned-dir', "$private/v5_section_aligned/dataset_v3",
    '--structured-dir', "$private/v5_section_aligned/section_inputs_sparse_2026_09_26",
    '--base-csv', "$private/v5_mimiciv_2026-09-25/structured/base.csv",
    '--field-dir', "$private/v5_section_aligned/field_text_baseline_2026_09_27",
    '--vectors', "$private/v5_section_aligned/field_text_baseline_2026_09_27/bert/bioclinicalbert.float16.npy",
    '--labels-csv', "$private/v5_qualified_copd_labels_2026_09_27/labels.csv.gz",
    '--processed-event-dir', $processed,
    '--raw-ledger-dir', 'patient_data/mimiciv_copd/snapshot_2026-09-27',
    '--clinical-recovery-dir', $features,
    '--abnormality-scale-mode', 'training_positive_p95',
    '--event-encoder', 'abnormal_medication', '--max-epochs', '35', '--batch-size', '12'
)
foreach ($seed in $Seeds) {
    $full = Join-Path $parent "full_seed${seed}/report.json"
    if (-not (Test-Path -LiteralPath $full)) { throw "Full reference missing: $full" }
    foreach ($group in $Groups) {
        if ($group.StartsWith('med_')) {
            $groupName = $group.Substring(4)
            if ($groupName -notin $medGroups) { throw "Unknown medication group: $group" }
            $labMode = 'active'
            $medMode = "without_$groupName"
        } elseif ($group.StartsWith('lab_')) {
            $groupName = $group.Substring(4)
            if ($groupName -notin $labGroups) { throw "Unknown laboratory group: $group" }
            $labMode = "without_$groupName"
            $medMode = 'active'
        } else { throw "Unknown group: $group" }
        $name = "without_${group}_seed${seed}"
        $output = Join-Path $runRoot $name
        if (Test-Path -LiteralPath (Join-Path $output 'report.json')) {
            $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
            if ($report.status -ne 'training_completed' -or
                $report.clinical_recovery_mode -ne $labMode -or
                $report.medication_effect_mode -ne $medMode -or
                $report.abnormality_scale_mode -ne 'training_positive_p95') {
                throw "Inconsistent completed run: $output"
            }
            Write-Output "SKIP $name"
            continue
        }
        if (Test-Path -LiteralPath $output) { throw "Partial run needs inspection: $output" }
        Write-Output "START $name"
        & docker @common '--clinical-recovery-mode' $labMode `
            '--medication-effect-mode' $medMode '--seed' "$seed" `
            '--output-dir' "training_runs/v5_revised_labs_meds_2026-09-29/groups/$name" `
            *> (Join-Path $runRoot "$name.log")
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath (Join-Path $runRoot "$name.log") -Tail 70
            throw "Training failed: $name"
        }
        $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
        if ($report.status -ne 'training_completed' -or $report.raw_events_enter_model) {
            throw "Incorrect run contract: $name"
        }
        Write-Output "DONE $name epoch=$($report.best_epoch) valIBS=$($report.validation.temporal_full.integrated_brier_7_365)"
    }
}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_revised_labs_meds.ps1 · lines 1–69

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_revised_labs_meds.ps1","part":1,"start_line":1,"end_line":69,"source_sha256":"055f952a37462ef291912a41b59b907ae3871ee49a361ebac3accbfa7ca6fc5c","fragment_sha256":"055f952a37462ef291912a41b59b907ae3871ee49a361ebac3accbfa7ca6fc5c","added_terminal_newline":false} -->
````powershell
param([int[]]$Seeds = @(2026, 2027, 2028),
      [string[]]$Arms = @('baseline', 'lab_only', 'med_only', 'full'))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = Join-Path $root 'training_runs/v5_revised_labs_meds_2026-09-29'
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$processed = 'patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29'
$features = 'patient_data/processed/abnormality_clinical_reviewed_2026-09-29'
if (-not (Test-Path -LiteralPath (Join-Path $processed 'manifest.json')) -or
    -not (Test-Path -LiteralPath (Join-Path $features 'manifest.json'))) {
    throw 'Revised input artifacts are missing'
}
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$common = @(
    'run', '--rm', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    'aecopd-training:v1', 'python', '-m', 'training.v5.train_event_sequence',
    '--aligned-dir', "$private/v5_section_aligned/dataset_v3",
    '--structured-dir', "$private/v5_section_aligned/section_inputs_sparse_2026_09_26",
    '--base-csv', "$private/v5_mimiciv_2026-09-25/structured/base.csv",
    '--field-dir', "$private/v5_section_aligned/field_text_baseline_2026_09_27",
    '--vectors', "$private/v5_section_aligned/field_text_baseline_2026_09_27/bert/bioclinicalbert.float16.npy",
    '--labels-csv', "$private/v5_qualified_copd_labels_2026_09_27/labels.csv.gz",
    '--processed-event-dir', $processed,
    '--raw-ledger-dir', 'patient_data/mimiciv_copd/snapshot_2026-09-27',
    '--clinical-recovery-dir', $features,
    '--abnormality-scale-mode', 'training_positive_p95',
    '--event-encoder', 'abnormal_medication', '--max-epochs', '35', '--batch-size', '12'
)
$modes = @{
    baseline = @('masked', 'masked')
    lab_only = @('active', 'masked')
    med_only = @('masked', 'active')
    full = @('active', 'active')
}
foreach ($seed in $Seeds) {
    foreach ($arm in $Arms) {
        if (-not $modes.ContainsKey($arm)) { throw "Unknown arm: $arm" }
        $name = "${arm}_seed${seed}"
        $output = Join-Path $runRoot $name
        if (Test-Path -LiteralPath (Join-Path $output 'report.json')) {
            $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
            if ($report.status -ne 'training_completed' -or
                $report.abnormality_scale_mode -ne 'training_positive_p95' -or
                $report.clinical_recovery_mode -ne $modes[$arm][0] -or
                $report.medication_effect_mode -ne $modes[$arm][1]) {
                throw "Inconsistent completed run: $output"
            }
            Write-Output "SKIP $name"
            continue
        }
        if (Test-Path -LiteralPath $output) { throw "Partial run needs inspection: $output" }
        Write-Output "START $name"
        & docker @common '--clinical-recovery-mode' $modes[$arm][0] `
            '--medication-effect-mode' $modes[$arm][1] '--seed' "$seed" `
            '--output-dir' "training_runs/v5_revised_labs_meds_2026-09-29/$name" `
            *> (Join-Path $runRoot "$name.log")
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath (Join-Path $runRoot "$name.log") -Tail 70
            throw "Training failed: $name"
        }
        $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
        if ($report.status -ne 'training_completed' -or $report.raw_events_enter_model) {
            throw "Incorrect run contract: $name"
        }
        Write-Output "DONE $name epoch=$($report.best_epoch) valIBS=$($report.validation.temporal_full.integrated_brier_7_365)"
    }
}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_revised_lean_candidate.ps1 · lines 1–51

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_revised_lean_candidate.ps1","part":1,"start_line":1,"end_line":51,"source_sha256":"5c767d147b31f82e001eaa7964928f2dbed193959f28292a1a4fbd25b65ac6c1","fragment_sha256":"5c767d147b31f82e001eaa7964928f2dbed193959f28292a1a4fbd25b65ac6c1","added_terminal_newline":false} -->
````powershell
param([int[]]$Seeds = @(2026, 2027, 2028))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$parent = Join-Path $root 'training_runs/v5_revised_labs_meds_2026-09-29'
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$processed = 'patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29'
$features = 'patient_data/processed/abnormality_clinical_reviewed_2026-09-29'
$common = @(
    'run', '--rm', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    'aecopd-training:v1', 'python', '-m', 'training.v5.train_event_sequence',
    '--aligned-dir', "$private/v5_section_aligned/dataset_v3",
    '--structured-dir', "$private/v5_section_aligned/section_inputs_sparse_2026_09_26",
    '--base-csv', "$private/v5_mimiciv_2026-09-25/structured/base.csv",
    '--field-dir', "$private/v5_section_aligned/field_text_baseline_2026_09_27",
    '--vectors', "$private/v5_section_aligned/field_text_baseline_2026_09_27/bert/bioclinicalbert.float16.npy",
    '--labels-csv', "$private/v5_qualified_copd_labels_2026_09_27/labels.csv.gz",
    '--processed-event-dir', $processed,
    '--raw-ledger-dir', 'patient_data/mimiciv_copd/snapshot_2026-09-27',
    '--clinical-recovery-dir', $features,
    '--abnormality-scale-mode', 'training_positive_p95',
    '--event-encoder', 'abnormal_medication', '--max-epochs', '35', '--batch-size', '12',
    '--clinical-recovery-mode', 'core_risk', '--medication-effect-mode', 'core_acute'
)
foreach ($seed in $Seeds) {
    $full = Join-Path $parent "full_seed${seed}/report.json"
    if (-not (Test-Path -LiteralPath $full)) { throw "Full reference missing: $full" }
    $name = "lean_clinical_seed${seed}"
    $output = Join-Path $parent $name
    if (Test-Path -LiteralPath (Join-Path $output 'report.json')) {
        $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
        if ($report.status -ne 'training_completed' -or $report.clinical_recovery_mode -ne 'core_risk' -or
            $report.medication_effect_mode -ne 'core_acute') { throw "Inconsistent completed run: $output" }
        Write-Output "SKIP $name"
        continue
    }
    if (Test-Path -LiteralPath $output) { throw "Partial run needs inspection: $output" }
    Write-Output "START $name"
    & docker @common '--seed' "$seed" '--output-dir' "training_runs/v5_revised_labs_meds_2026-09-29/$name" `
        *> (Join-Path $parent "$name.log")
    if ($LASTEXITCODE -ne 0) {
        Get-Content -LiteralPath (Join-Path $parent "$name.log") -Tail 70
        throw "Training failed: $name"
    }
    $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
    if ($report.status -ne 'training_completed' -or $report.raw_events_enter_model) {
        throw "Incorrect run contract: $name"
    }
    Write-Output "DONE $name epoch=$($report.best_epoch) valIBS=$($report.validation.temporal_full.integrated_brier_7_365)"
}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_screened_group_ablations.ps1 · lines 1–81

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_screened_group_ablations.ps1","part":1,"start_line":1,"end_line":81,"source_sha256":"de796c691a900cb4831f0192cb3c5d65461d5fa5ddf4b512a6f87b1632e40366","fragment_sha256":"de796c691a900cb4831f0192cb3c5d65461d5fa5ddf4b512a6f87b1632e40366","added_terminal_newline":false} -->
````powershell
param(
    [int[]]$Seeds = @(2026, 2027, 2028),
    [string[]]$Arms = @('full', 'no_all_events', 'no_all_labs', 'no_all_medications', 'no_text', 'no_noncontext_static',
                       'no_prior_counts', 'no_blood_gas', 'no_routine_labs',
                       'no_sparse_blood_gas', 'no_bronchodilators', 'no_steroids',
                       'no_antibiotics', 'no_diuretics')
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = Join-Path $root 'training_runs/v5_feature_groups_formulation_verified_2026-09-29'
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$processed = 'patient_data/processed/mimiciv_copd_events_formulation_verified_2026-09-29'
$manifest = Get-Content -Raw -LiteralPath (Join-Path $root "$processed/manifest.json") | ConvertFrom-Json
if ($manifest.status -ne 'preprocessed_and_checked') { throw 'Corrected event data is not validated.' }
$known = @($manifest.concepts)
$groups = @{
    no_all_labs = @($known | Where-Object { $_ -match '^(ph|pco2|po2):' -or $_ -in @('albumin','bicarbonate','creatinine','hemoglobin','wbc') })
    no_blood_gas = @($known | Where-Object { $_ -match '^(ph|pco2|po2):' })
    no_sparse_blood_gas = @($known | Where-Object { $_ -match '^(ph|pco2|po2):(central_venous|mixed|unknown)$' })
    no_routine_labs = @('albumin','bicarbonate','creatinine','hemoglobin','wbc')
    no_bronchodilators = @('saba','sama','laba','lama')
    no_steroids = @('inhaled_steroid','systemic_steroid')
    no_antibiotics = @('macrolide','beta_lactam','fluoroquinolone','other_antibiotic')
    no_diuretics = @('loop_diuretic','other_diuretic')
}
foreach ($entry in $groups.GetEnumerator()) {
    if (-not $entry.Value -or @($entry.Value | Where-Object { $_ -notin $known }).Count) {
        throw "Invalid event concept group: $($entry.Key)"
    }
}
$validArms = @('full','no_all_events','no_all_labs','no_all_medications','no_text','no_noncontext_static',
               'no_prior_counts','no_blood_gas','no_routine_labs',
               'no_sparse_blood_gas','no_bronchodilators','no_steroids','no_antibiotics','no_diuretics')
if (@($Arms | Where-Object { $_ -notin $validArms }).Count) { throw 'Unknown arm requested.' }
$common = @(
    'run', '--rm', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    'aecopd-training:v1', 'python', '-m', 'training.v5.train_event_sequence',
    '--aligned-dir', "$private/v5_section_aligned/dataset_v3",
    '--structured-dir', "$private/v5_section_aligned/section_inputs_sparse_2026_09_26",
    '--base-csv', "$private/v5_mimiciv_2026-09-25/structured/base.csv",
    '--field-dir', "$private/v5_section_aligned/field_text_baseline_2026_09_27",
    '--vectors', "$private/v5_section_aligned/field_text_baseline_2026_09_27/bert/bioclinicalbert.float16.npy",
    '--labels-csv', "$private/v5_qualified_copd_labels_2026_09_27/labels.csv.gz",
    '--processed-event-dir', $processed,
    '--raw-ledger-dir', 'patient_data/mimiciv_copd/snapshot_2026-09-27',
    '--event-encoder', 'set', '--max-epochs', '35', '--batch-size', '12'
)
foreach ($seed in $Seeds) {
    foreach ($arm in $Arms) {
        $name = "${arm}_seed${seed}"
        $output = Join-Path $runRoot $name
        $reportPath = Join-Path $output 'report.json'
        if (Test-Path -LiteralPath $reportPath) {
            $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
            if ($report.status -ne 'training_completed') { throw "Incomplete run: $output" }
            Write-Output "SKIP $name (already complete)"
            continue
        }
        if (Test-Path -LiteralPath $output) { throw "Partial run needs inspection: $output" }
        $variantArgs = @()
        if ($arm -eq 'no_all_events') { $variantArgs = @('--disable-events') }
        elseif ($arm -eq 'no_all_medications') { $variantArgs = @('--disable-medications') }
        elseif ($arm -eq 'no_text') { $variantArgs = @('--disable-text') }
        elseif ($arm -eq 'no_noncontext_static') { $variantArgs = @('--disable-current-static') }
        elseif ($arm -eq 'no_prior_counts') { $variantArgs = @('--exclude-prior-admission-counts') }
        elseif ($groups.ContainsKey($arm)) { $variantArgs = @('--exclude-event-concepts', ($groups[$arm] -join ',')) }
        $log = Join-Path $runRoot "$name.log"
        Write-Output "START $name"
        & docker @common '--output-dir' "training_runs/v5_feature_groups_formulation_verified_2026-09-29/$name" '--seed' "$seed" @variantArgs *> $log
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath $log -Tail 50
            throw "Training failed: $name (see $log)"
        }
        $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
        if ($report.status -ne 'training_completed') { throw "Missing completed report: $reportPath" }
        Write-Output "DONE $name best_epoch=$($report.best_epoch) validation_IBS=$($report.validation.temporal_full.integrated_brier_7_365)"
    }
}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_section_incremental.py · lines 1–72

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_section_incremental.py","part":1,"start_line":1,"end_line":72,"source_sha256":"93764a3cd92ecbfcf3b4e6f25eba3ba7d7e4068bcf54c95059f55cc2a6914e23","fragment_sha256":"93764a3cd92ecbfcf3b4e6f25eba3ba7d7e4068bcf54c95059f55cc2a6914e23","added_terminal_newline":false} -->
````python
"""固定扩展字段词表，逐一训练新增小结章节与四字段基线。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from training.v5.extend_field_manifest import EXTRA_LABELS
from training.v5.field_text_baseline import FIELD_LABELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned_dir", "structured_dir", "base_csv", "processed_event_dir",
                 "raw_ledger_dir", "field_dir", "vectors", "labels_csv", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--only-arms", nargs="+", help="Subset of arm names for replication")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = tuple(FIELD_LABELS)
    arms = {"baseline_four_fields": baseline}
    arms.update({"plus_" + role: (*baseline, role) for role in EXTRA_LABELS})
    arms["plus_all_five"] = (*baseline, *EXTRA_LABELS)
    if args.only_arms is not None:
        if len(set(args.only_arms)) != len(args.only_arms) or set(args.only_arms) - set(arms):
            raise ValueError("unknown or duplicate replication arm")
        arms = {name: fields for name, fields in arms.items() if name in args.only_arms}
    state_path = args.output_dir / "progress.json"
    expected = {name: list(fields) for name, fields in arms.items()}
    if state_path.exists():
        progress = json.loads(state_path.read_text(encoding="utf-8"))
        if progress["seed"] != args.seed or progress["arms"] != expected:
            raise ValueError("existing experiment contract differs")
    else:
        progress = {"status": "running", "seed": args.seed, "arms": expected, "completed": []}
        state_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    shared = [
        "--aligned-dir", str(args.aligned_dir), "--structured-dir", str(args.structured_dir),
        "--base-csv", str(args.base_csv), "--processed-event-dir", str(args.processed_event_dir),
        "--raw-ledger-dir", str(args.raw_ledger_dir), "--field-dir", str(args.field_dir),
        "--vectors", str(args.vectors), "--labels-csv", str(args.labels_csv),
        "--event-encoder", "set", "--seed", str(args.seed),
    ]
    for name, fields in arms.items():
        run_dir = args.output_dir / name
        if name in progress["completed"]:
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            if report["status"] != "training_completed" or report["active_text_fields"] != list(fields):
                raise ValueError("completed arm no longer matches contract: " + name)
            continue
        if run_dir.exists():
            raise FileExistsError("incomplete arm requires inspection: " + str(run_dir))
        command = [sys.executable, "-m", "training.v5.train_event_sequence", *shared,
                   "--output-dir", str(run_dir), "--active-text-fields", ",".join(fields)]
        print(json.dumps({"starting_arm": name, "seed": args.seed}), flush=True)
        subprocess.run(command, check=True)
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        if report["status"] != "training_completed" or report["active_text_fields"] != list(fields):
            raise ValueError("arm did not complete as requested: " + name)
        progress["completed"].append(name)
        state_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    progress["status"] = "training_completed"
    state_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    print(json.dumps({"status": progress["status"], "completed": progress["completed"]}), flush=True)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_selected_group_removal.ps1 · lines 1–55

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_selected_group_removal.ps1","part":1,"start_line":1,"end_line":55,"source_sha256":"dc8a79cf3d8495165c01642cd262da2102dff0b441491d5a3171995809be36db","fragment_sha256":"dc8a79cf3d8495165c01642cd262da2102dff0b441491d5a3171995809be36db","added_terminal_newline":false} -->
````powershell
param([int[]]$Seeds = @(2026, 2027, 2028))

$ErrorActionPreference = 'Stop'
Write-Warning 'Historical strict-or-top3 composite endpoint only; current strict endpoint uses run_strict_axis_screen.ps1.'
$root = (Resolve-Path -LiteralPath '.').Path
$parent = Join-Path $root 'training_runs/v5_revised_labs_meds_2026-09-29'
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$processed = 'patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29'
$features = 'patient_data/processed/abnormality_clinical_reviewed_2026-09-29'
$labMode = 'without_renal_metabolic_perfusion'
$medMode = 'without_diuretic'
$common = @(
    'run', '--rm', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    'aecopd-training:v1', 'python', '-m', 'training.v5.train_event_sequence',
    '--aligned-dir', "$private/v5_section_aligned/dataset_v3",
    '--structured-dir', "$private/v5_section_aligned/section_inputs_sparse_2026_09_26",
    '--base-csv', "$private/v5_mimiciv_2026-09-25/structured/base.csv",
    '--field-dir', "$private/v5_section_aligned/field_text_baseline_2026_09_27",
    '--vectors', "$private/v5_section_aligned/field_text_baseline_2026_09_27/bert/bioclinicalbert.float16.npy",
    '--labels-csv', "$private/v5_qualified_copd_labels_2026_09_27/labels.csv.gz",
    '--processed-event-dir', $processed,
    '--raw-ledger-dir', 'patient_data/mimiciv_copd/snapshot_2026-09-27',
    '--clinical-recovery-dir', $features,
    '--abnormality-scale-mode', 'training_positive_p95',
    '--event-encoder', 'abnormal_medication', '--max-epochs', '35', '--batch-size', '12',
    '--clinical-recovery-mode', $labMode, '--medication-effect-mode', $medMode
)
foreach ($seed in $Seeds) {
    $full = Join-Path $parent "full_seed${seed}/report.json"
    if (-not (Test-Path -LiteralPath $full)) { throw "Full reference missing: $full" }
    $name = "selected_group_removal_seed${seed}"
    $output = Join-Path $parent $name
    if (Test-Path -LiteralPath (Join-Path $output 'report.json')) {
        $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
        if ($report.status -ne 'training_completed' -or $report.clinical_recovery_mode -ne $labMode -or
            $report.medication_effect_mode -ne $medMode) { throw "Inconsistent completed run: $output" }
        Write-Output "SKIP $name"
        continue
    }
    if (Test-Path -LiteralPath $output) { throw "Partial run needs inspection: $output" }
    Write-Output "START $name"
    & docker @common '--seed' "$seed" '--output-dir' "training_runs/v5_revised_labs_meds_2026-09-29/$name" `
        *> (Join-Path $parent "$name.log")
    if ($LASTEXITCODE -ne 0) {
        Get-Content -LiteralPath (Join-Path $parent "$name.log") -Tail 70
        throw "Training failed: $name"
    }
    $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
    if ($report.status -ne 'training_completed' -or $report.raw_events_enter_model -or
        $report.clinical_recovery_mode -ne $labMode -or $report.medication_effect_mode -ne $medMode) {
        throw "Incorrect run contract: $name"
    }
    Write-Output "DONE $name epoch=$($report.best_epoch) valIBS=$($report.validation.temporal_full.integrated_brier_7_365)"
}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_strict_axis_screen.ps1 · lines 1–73

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_strict_axis_screen.ps1","part":1,"start_line":1,"end_line":73,"source_sha256":"c0f18d556f41e54d991cbcf578dc6a85e3c8cb6172bb27ea2752f70b842b9926","fragment_sha256":"c0f18d556f41e54d991cbcf578dc6a85e3c8cb6172bb27ea2752f70b842b9926","added_terminal_newline":false} -->
````powershell
param(
    [int[]]$Seeds = @(2026, 2027, 2028),
    [string[]]$Arms = @('full', 'selected', 'arterial_hypercapnic_acidosis', 'platelet_low', 'magnesium_low', 'ntprobnp_high')
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = Join-Path $root 'training_runs/v5_strict_axis_screen_2026-09-30'
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$strictLabels = 'patient_data/derived/strict_aecopd_endpoint_2026-09-28/labels.csv.gz'
$strictSummary = Get-Content -Raw -LiteralPath 'patient_data/derived/strict_aecopd_endpoint_2026-09-28/summary.json' | ConvertFrom-Json
if ($strictSummary.status -ne 'strict_labels_reconstructed_and_checked' -or
    $strictSummary.cohort_admissions -ne 19401 -or -not (Test-Path -LiteralPath $strictLabels)) {
    throw 'Strict endpoint labels are missing or unverified'
}
$validArms = @('full', 'selected', 'arterial_hypercapnic_acidosis', 'platelet_low', 'magnesium_low', 'ntprobnp_high')
foreach ($arm in $Arms) {
    if ($arm -notin $validArms) { throw "Unknown screening arm: $arm" }
}
$common = @(
    'run', '--rm', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    'aecopd-training:v1', 'python', '-m', 'training.v5.train_event_sequence',
    '--aligned-dir', "$private/v5_section_aligned/dataset_v3",
    '--structured-dir', "$private/v5_section_aligned/section_inputs_sparse_2026_09_26",
    '--base-csv', "$private/v5_mimiciv_2026-09-25/structured/base.csv",
    '--field-dir', "$private/v5_section_aligned/field_text_baseline_2026_09_27",
    '--vectors', "$private/v5_section_aligned/field_text_baseline_2026_09_27/bert/bioclinicalbert.float16.npy",
    '--labels-csv', $strictLabels,
    '--processed-event-dir', 'patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29',
    '--raw-ledger-dir', 'patient_data/mimiciv_copd/snapshot_2026-09-27',
    '--clinical-recovery-dir', 'patient_data/processed/abnormality_clinical_reviewed_2026-09-29',
    '--abnormality-scale-mode', 'training_positive_p95',
    '--event-encoder', 'abnormal_medication', '--max-epochs', '35', '--batch-size', '12'
)
if (-not (Test-Path -LiteralPath $runRoot)) { New-Item -ItemType Directory -Path $runRoot | Out-Null }
foreach ($seed in $Seeds) {
    foreach ($arm in $Arms) {
        if ($arm -eq 'full') {
            $labMode, $medMode, $name = 'active', 'active', "full_seed${seed}"
        } elseif ($arm -eq 'selected') {
            $labMode, $medMode, $name = 'without_renal_metabolic_perfusion', 'without_diuretic', "selected_seed${seed}"
        } else {
            $labMode, $medMode, $name = "without_renal_metabolic_perfusion_plus_${arm}", 'without_diuretic', "without_axis_${arm}_seed${seed}"
        }
        $output = Join-Path $runRoot $name
        if (Test-Path -LiteralPath (Join-Path $output 'report.json')) {
            $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
            if ($report.status -ne 'training_completed' -or $report.clinical_recovery_mode -ne $labMode -or
                $report.medication_effect_mode -ne $medMode -or $report.label_sha256 -ne $strictSummary.label_sha256) {
                throw "Inconsistent completed run: $output"
            }
            Write-Output "SKIP $name"
            continue
        }
        if (Test-Path -LiteralPath $output) { throw "Partial run needs inspection: $output" }
        Write-Output "START $name"
        & docker @common '--clinical-recovery-mode' $labMode '--medication-effect-mode' $medMode `
            '--seed' "$seed" '--output-dir' "training_runs/v5_strict_axis_screen_2026-09-30/$name" `
            *> (Join-Path $runRoot "$name.log")
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath (Join-Path $runRoot "$name.log") -Tail 70
            throw "Training failed: $name"
        }
        $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
        if ($report.status -ne 'training_completed' -or $report.raw_events_enter_model -or
            $report.clinical_recovery_mode -ne $labMode -or $report.medication_effect_mode -ne $medMode -or
            $report.label_sha256 -ne $strictSummary.label_sha256) {
            throw "Incorrect run contract: $name"
        }
        Write-Output "DONE $name epoch=$($report.best_epoch) valIBS=$($report.validation.temporal_full.integrated_brier_7_365)"
    }
}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/run_strict_qwen_bert_comparison.ps1 · lines 1–68

<!-- REVIEW_FRAGMENT {"path":"training/v5/run_strict_qwen_bert_comparison.ps1","part":1,"start_line":1,"end_line":68,"source_sha256":"2ba27ad07317c133f773b0d226dbf3957a7d9a0ecc43741bdfdae8b62e9bd1bb","fragment_sha256":"2ba27ad07317c133f773b0d226dbf3957a7d9a0ecc43741bdfdae8b62e9bd1bb","added_terminal_newline":false} -->
````powershell
param([int[]]$Seeds = @(2026, 2027, 2028))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = Join-Path $root 'training_runs/v5_strict_qwen_bert_2026-09-30'
$bertRoot = Join-Path $root 'training_runs/v5_strict_axis_screen_2026-09-30'
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$fieldRoot = "$private/v5_section_aligned/field_text_baseline_2026_09_27"
$fieldLocal = Join-Path $root ($fieldRoot -replace '/', '\')
$manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $fieldLocal 'shared_text.csv.gz')).Hash.ToLowerInvariant()
$qwenState = Get-Content -Raw -LiteralPath (Join-Path $fieldLocal 'qwen/state.json') | ConvertFrom-Json
$bertState = Get-Content -Raw -LiteralPath (Join-Path $fieldLocal 'bert/state.json') | ConvertFrom-Json
if ($qwenState.status -ne 'complete' -or $bertState.status -ne 'complete' -or
    $qwenState.completed_rows -ne 76974 -or $bertState.completed_rows -ne 76974 -or
    $qwenState.manifest_sha256 -ne $manifestHash -or $bertState.manifest_sha256 -ne $manifestHash -or
    -not (Test-Path -LiteralPath (Join-Path $fieldLocal 'qwen/qwen3_embedding.float16.npy'))) {
    throw 'Qwen and BERT must use the same completed 76,974-row manifest'
}
$strictSummary = Get-Content -Raw -LiteralPath 'patient_data/derived/strict_aecopd_endpoint_2026-09-28/summary.json' | ConvertFrom-Json
$common = @(
    'run', '--rm', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    'aecopd-training:v1', 'python', '-m', 'training.v5.train_event_sequence',
    '--aligned-dir', "$private/v5_section_aligned/dataset_v3",
    '--structured-dir', "$private/v5_section_aligned/section_inputs_sparse_2026_09_26",
    '--base-csv', "$private/v5_mimiciv_2026-09-25/structured/base.csv",
    '--field-dir', $fieldRoot,
    '--vectors', "$fieldRoot/qwen/qwen3_embedding.float16.npy",
    '--labels-csv', 'patient_data/derived/strict_aecopd_endpoint_2026-09-28/labels.csv.gz',
    '--processed-event-dir', 'patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29',
    '--raw-ledger-dir', 'patient_data/mimiciv_copd/snapshot_2026-09-27',
    '--clinical-recovery-dir', 'patient_data/processed/abnormality_clinical_reviewed_2026-09-29',
    '--abnormality-scale-mode', 'training_positive_p95',
    '--clinical-recovery-mode', 'active', '--medication-effect-mode', 'active',
    '--event-encoder', 'abnormal_medication', '--shared-text-init-width', '768',
    '--max-epochs', '35', '--batch-size', '12'
)
if (-not (Test-Path -LiteralPath $runRoot)) { New-Item -ItemType Directory -Path $runRoot | Out-Null }
foreach ($seed in $Seeds) {
    $bertReport = Get-Content -Raw -LiteralPath (Join-Path $bertRoot "full_seed${seed}/report.json") | ConvertFrom-Json
    if ($bertReport.status -ne 'training_completed' -or $bertReport.label_sha256 -ne $strictSummary.label_sha256 -or
        $bertReport.field_manifest_sha256 -ne $manifestHash -or $bertReport.text_vector_width -ne 768 -or
        $bertReport.clinical_recovery_mode -ne 'active' -or $bertReport.medication_effect_mode -ne 'active') {
        throw "BERT reference differs from full strict-endpoint contract: $seed"
    }
    $output = Join-Path $runRoot "qwen_seed${seed}"
    if (Test-Path -LiteralPath (Join-Path $output 'report.json')) {
        $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
        if ($report.status -ne 'training_completed' -or $report.label_sha256 -ne $strictSummary.label_sha256 -or
            $report.field_manifest_sha256 -ne $manifestHash -or $report.text_vector_width -ne 2560 -or
            $report.shared_text_init_width -ne 768) { throw "Inconsistent Qwen run: $seed" }
        Write-Output "SKIP qwen_seed${seed}"
        continue
    }
    if (Test-Path -LiteralPath $output) { throw "Partial run needs inspection: $output" }
    Write-Output "START qwen_seed${seed}"
    & docker @common '--seed' "$seed" '--output-dir' "training_runs/v5_strict_qwen_bert_2026-09-30/qwen_seed${seed}" `
        *> (Join-Path $runRoot "qwen_seed${seed}.log")
    if ($LASTEXITCODE -ne 0) {
        Get-Content -LiteralPath (Join-Path $runRoot "qwen_seed${seed}.log") -Tail 70
        throw "Qwen training failed: $seed"
    }
    $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
    if ($report.status -ne 'training_completed' -or $report.label_sha256 -ne $strictSummary.label_sha256 -or
        $report.field_manifest_sha256 -ne $manifestHash -or $report.text_vector_width -ne 2560 -or
        $report.shared_text_init_width -ne 768) { throw "Qwen report contract failed: $seed" }
    Write-Output "DONE qwen_seed${seed} epoch=$($report.best_epoch)"
}
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/section_alignment_data/__init__.py · lines 1–1

<!-- REVIEW_FRAGMENT {"path":"training/v5/section_alignment_data/__init__.py","part":1,"start_line":1,"end_line":1,"source_sha256":"e9e6b2fedcf43fbde84c752e42320e4352bce7b4f618a3f213af36bdd1db8225","fragment_sha256":"e9e6b2fedcf43fbde84c752e42320e4352bce7b4f618a3f213af36bdd1db8225","added_terminal_newline":false} -->
````python
"""V5 章节锚定数据准备：只读来源与可追溯切分。"""
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/section_alignment_data/audit_old_export_parity.py · lines 1–163

<!-- REVIEW_FRAGMENT {"path":"training/v5/section_alignment_data/audit_old_export_parity.py","part":1,"start_line":1,"end_line":163,"source_sha256":"2b8793d3eb467c5dac5f94836a5588b068d7b9676897d5dc086bf67ed0aea2ca","fragment_sha256":"2b8793d3eb467c5dac5f94836a5588b068d7b9676897d5dc086bf67ed0aea2ca","added_terminal_newline":false} -->
````python
"""核对新导出的原始事件能否重建旧 V5 结构化导出；不读取结局。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Iterator


def _rows(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _cohort_parity(raw: Path, old: Path) -> dict[str, int]:
    previous = {row["hadm_id"]: row for row in _rows(old / "base.csv")}
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for row in _rows(raw / "cohort.csv.gz"):
        hadm = row["hadm_id"]
        if hadm in seen:
            counts["duplicate_new_hadm"] += 1
        seen.add(hadm)
        older = previous.get(hadm)
        if older is None:
            counts["new_hadm_not_old"] += 1
            continue
        for field in ("subject_id", "admittime", "dischtime", "index_is_strict_aecopd"):
            if row[field] != older[field]:
                counts[f"mismatch_{field}"] += 1
    counts["old_hadm_not_new"] = len(previous.keys() - seen)
    counts["new_rows"] = len(seen)
    counts["old_rows"] = len(previous)
    return dict(counts)


def _labs_parity(raw: Path, old: Path) -> dict[str, int]:
    expected: dict[tuple[str, str], tuple[int, Decimal, Decimal, Decimal, Decimal]] = {}
    for row in _rows(old / "labs.csv"):
        key = row["hadm_id"], row["itemid"]
        if key in expected:
            raise ValueError("duplicate old lab summary key")
        expected[key] = (
            int(row["measurement_count"]), *(Decimal(row[name]) for name in
                                                 ("first_value", "last_value", "min_value", "max_value")),
        )
    counts: Counter[str] = Counter()
    current_key: tuple[str, str] | None = None
    first = last = minimum = maximum = Decimal(0)
    measurement_count = 0

    def compare() -> None:
        if current_key is None:
            return
        counts["new_lab_groups"] += 1
        old_value = expected.pop(current_key, None)
        if old_value is None:
            counts["new_group_not_old"] += 1
        elif old_value != (measurement_count, first, last, minimum, maximum):
            counts["value_or_count_mismatch"] += 1

    for row in _rows(raw / "labs.csv.gz"):
        key = row["hadm_id"], row["itemid"]
        value = Decimal(row["valuenum"])
        if key != current_key:
            compare()
            current_key = key
            first = last = minimum = maximum = value
            measurement_count = 1
        else:
            last = value
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            measurement_count += 1
        counts["new_lab_events"] += 1
    compare()
    counts["old_group_not_new"] = len(expected)
    return dict(counts)


def _diagnoses_parity(raw: Path, old: Path) -> dict[str, int]:
    def key(row: dict[str, str]) -> tuple[str, str, str, str]:
        return row["hadm_id"], row["icd_version"], row["icd_code"].strip(), row["seq_num"]

    previous = Counter(key(row) for row in _rows(old / "diagnoses.csv"))
    current = Counter(key(row) for row in _rows(raw / "diagnoses.csv.gz"))
    return {
        "old_rows": sum(previous.values()), "new_rows": sum(current.values()),
        "old_only_rows": sum((previous - current).values()),
        "new_only_rows": sum((current - previous).values()),
    }


def _medications_parity(raw: Path, old: Path) -> dict[str, int]:
    previous: dict[tuple[str, str], tuple[str, str]] = {}
    for row in _rows(old / "medications.csv"):
        key = row["hadm_id"], row["medication"]
        if key in previous:
            raise ValueError("duplicate old medication summary key")
        previous[key] = row["first_charttime"], row["last_charttime"]
    current: dict[tuple[str, str], tuple[str, str]] = {}
    raw_rows = 0
    for row in _rows(raw / "emar.csv.gz"):
        key = row["hadm_id"], row["medication"].strip().lower()
        charttime = row["charttime"]
        if key in current:
            first, last = current[key]
            current[key] = min(first, charttime), max(last, charttime)
        else:
            current[key] = charttime, charttime
        raw_rows += 1
    common = previous.keys() & current.keys()
    return {
        "new_emar_events": raw_rows, "old_medication_groups": len(previous),
        "new_medication_groups": len(current),
        "old_group_not_new": len(previous.keys() - current.keys()),
        "new_group_not_old": len(current.keys() - previous.keys()),
        "first_or_last_time_mismatch": sum(previous[key] != current[key] for key in common),
    }


def audit(raw_dir: Path, old_dir: Path) -> dict[str, object]:
    components = {
        "cohort": _cohort_parity(raw_dir, old_dir),
        "labs": _labs_parity(raw_dir, old_dir),
        "diagnoses": _diagnoses_parity(raw_dir, old_dir),
        "medications": _medications_parity(raw_dir, old_dir),
    }
    mismatch = {
        "cohort": sum(value for name, value in components["cohort"].items()
                      if name.startswith(("mismatch_", "duplicate_")) or name.endswith(("not_old", "not_new"))),
        "labs": sum(components["labs"].get(name, 0) for name in
                    ("new_group_not_old", "old_group_not_new", "value_or_count_mismatch")),
        "diagnoses": components["diagnoses"]["old_only_rows"] + components["diagnoses"]["new_only_rows"],
        "medications": sum(components["medications"][name] for name in
                           ("old_group_not_new", "new_group_not_old", "first_or_last_time_mismatch")),
    }
    return {"status": "exact_parity" if not any(mismatch.values()) else "mismatch_found",
            "components": components, "mismatch_counts": mismatch,
            "old_export_reusable_as_baseline": not any(mismatch.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = audit(args.raw_dir, args.old_dir)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/section_alignment_data/audit_section_text_pilot.py · lines 1–63

<!-- REVIEW_FRAGMENT {"path":"training/v5/section_alignment_data/audit_section_text_pilot.py","part":1,"start_line":1,"end_line":63,"source_sha256":"a4ec7f5778f2b6a58fb739deeb99c3a44063b7fea55e6d3f3094790407f6d364","fragment_sha256":"a4ec7f5778f2b6a58fb739deeb99c3a44063b7fea55e6d3f3094790407f6d364","added_terminal_newline":false} -->
````python
"""统计候选章节文本的长度、重复和数字密度，不保存原文。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def audit(manifest_dir: Path) -> dict[str, object]:
    with gzip.open(manifest_dir / "shared_text.csv.gz", "rt", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_role: dict[str, list[dict[str, str]]] = defaultdict(list)
    chunks_by_section: Counter[str] = Counter()
    after_discharge: Counter[str] = Counter()
    normalized: dict[str, Counter[str]] = defaultdict(Counter)
    numeric_heavy: Counter[str] = Counter()
    for row in rows:
        role = row["section_role"]
        by_role[role].append(row)
        chunks_by_section[row["section_id"]] += 1
        after_discharge[role] += int(row["timestamp_relation"] == "after_discharge")
        canonical = re.sub(r"\d", "#", " ".join(row["chunk_text"].lower().split()))
        normalized[role][hashlib.sha256(canonical.encode("utf-8")).hexdigest()] += 1
        letters = sum(character.isalpha() for character in row["chunk_text"])
        digits = sum(character.isdigit() for character in row["chunk_text"])
        numeric_heavy[role] += int(digits > letters)
    result: dict[str, object] = {"status": "content_statistics_not_clinical_truth", "roles": {}}
    for role, items in by_role.items():
        sections = {item["section_id"] for item in items}
        section_chunks = [chunks_by_section[sid] for sid in sections]
        bert_lengths = [int(item["bert_token_count"]) for item in items]
        duplicates = sum(count for count in normalized[role].values() if count >= 5)
        result["roles"][role] = {
            "sections": len(sections), "chunks": len(items),
            "median_bert_tokens_per_chunk": median(bert_lengths),
            "max_chunks_per_section": max(section_chunks),
            "sections_over_4_chunks": sum(count > 4 for count in section_chunks),
            "chunks_in_repeated_normalized_text_5plus": duplicates,
            "numeric_heavy_chunks": numeric_heavy[role],
            "stored_after_discharge_chunks": after_discharge[role],
        }
    (manifest_dir / "content_statistics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/section_alignment_data/build_dataset.py · lines 1–256

<!-- REVIEW_FRAGMENT {"path":"training/v5/section_alignment_data/build_dataset.py","part":1,"start_line":1,"end_line":256,"source_sha256":"5b66c2b521f879ab042722518b2c388b31df631856761c78874a094521722ce3","fragment_sha256":"5b66c2b521f879ab042722518b2c388b31df631856761c78874a094521722ce3","added_terminal_newline":false} -->
````python
"""把受控原始导出转换为可追溯的章节、原子事实及主题候选关系。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Protocol, Sequence

from training.v5.section_alignment_data.section_parser import parse_sections


NOTE_FIELDS = ("hadm_id", "note_id", "section_id", "parent_section_id", "heading",
               "section_role", "start_char", "end_char", "text", "parse_status",
               "content_status", "source_sha256", "timestamp_relation")
FACT_FIELDS = ("fact_id", "subject_id", "hadm_id", "source_table", "source_row_key",
               "concept_code", "concept_name", "value", "unit", "event_time",
               "available_time", "status", "context")
ALIGN_FIELDS = ("fact_id", "section_id", "rule_id", "relation_candidate")
LAB_NAMES = {
    50820: "ph", 50818: "paco2", 50821: "pao2", 50882: "bicarbonate",
    51301: "wbc", 51222: "hemoglobin", 50912: "creatinine", 50862: "albumin",
}
CONTEXT_FIELDS = (
    "age_at_admission", "gender", "discharge_location",
    "prior_coded_admission_count", "prior_strict_aecopd_count", "coded_admission_order",
)


def patient_split(subject_id: int) -> str:
    """与 training/v5/data.py 的 V5 患者级 MD5 划分合同保持一致。"""

    digest = hashlib.md5(f"aecopd-model-v1:{subject_id}".encode("ascii")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


class CsvRowWriter(Protocol):
    def writerow(self, row: Sequence[str]) -> object: ...


def _read(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _read_plain(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _time(value: str) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _add_candidates(
    fact_id: str, source: str, row: dict[str, str], discharge: str,
    roles: dict[str, list[str]], writer: CsvRowWriter,
) -> int:
    rules: list[tuple[str, str, str]] = []
    if source == "labs":
        rules.append(("pertinent_results", "lab_to_pertinent_results", "topic_only"))
    elif source == "diagnoses":
        rules.append(("discharge_diagnosis", "diagnosis_to_discharge_diagnosis", "topic_only"))
    elif source == "emar":
        rules.append(("hospital_course", "emar_to_hospital_course", "topic_only"))
    elif source == "prescriptions":
        rules.append(("hospital_course", "prescription_to_hospital_course", "topic_only"))
        start = _time(row["starttime"])
        end = _time(discharge)
        stop = _time(row["stoptime"])
        if start and end and timedelta(0) <= end - start <= timedelta(days=1) and (stop is None or stop >= end):
            rules.append(("discharge_medications", "near_discharge_order_unverified",
                          "topic_only_unverified_order_context"))
    elif source == "context" and row["field"] == "discharge_location":
        rules.append(("discharge_disposition", "location_to_discharge_disposition", "topic_only"))
    count = 0
    for role, rule, relation in rules:
        for section_id in roles.get(role, []):
            writer.writerow((fact_id, section_id, rule, relation))
            count += 1
    return count


def _fact(source: str, row: dict[str, str], row_number: int, discharge: str) -> tuple[str, ...]:
    fact_id = f"{source}:{row_number}"
    subject, hadm = row["subject_id"], row["hadm_id"]
    if source == "labs":
        item = int(row["itemid"])
        return (fact_id, subject, hadm, "mimiciv_hosp.labevents", row["labevent_id"],
                str(item), LAB_NAMES[item], row["valuenum"], row["valueuom"],
                row["charttime"], row["storetime"], row["flag"], "measured_lab")
    if source == "diagnoses":
        return (fact_id, subject, hadm, "mimiciv_hosp.diagnoses_icd",
                f"{hadm}:{row['seq_num']}:{row['icd_version']}:{row['icd_code']}",
                f"ICD{row['icd_version']}:{row['icd_code']}",
                row["icd_code"], "", "", discharge, discharge,
                "discharge_code_availability_assumed", "discharge_diagnosis_not_prior_history")
    if source == "emar":
        return (fact_id, subject, hadm, "mimiciv_hosp.emar",
                f"{row['emar_id']}:{row['emar_seq']}", row["medication"],
                row["medication"], row["event_txt"], "", row["charttime"],
                row["storetime"], "administered", "inpatient_administration")
    if source == "prescriptions":
        key = "|".join((row["pharmacy_id"], row["poe_id"], row["drug_type"], row["drug"]))
        return (fact_id, subject, hadm, "mimiciv_hosp.prescriptions", key,
                row["drug"], row["drug"], row["drug_type"], row["route"],
                row["starttime"], "", "availability_not_verified",
                "prescription_not_confirmed_discharge_medication")
    raise ValueError(source)


def build_dataset(
    raw_dir: Path, note_source: Path, output_dir: Path, base_source: Path | None = None,
) -> dict[str, object]:
    """只读输入；禁止覆盖已有交付物，输出汇总不包含任何病历正文。"""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    cohort: dict[str, dict[str, str]] = {}
    subjects: set[str] = set()
    for row in _read(raw_dir / "cohort.csv.gz"):
        hadm = row["hadm_id"]
        if hadm in cohort:
            raise ValueError(f"duplicate cohort hadm_id: {hadm}")
        cohort[hadm] = row
        subjects.add(row["subject_id"])
    if not cohort:
        raise ValueError("empty cohort")
    output_dir.mkdir(parents=True)
    summary: dict[str, object] = {
        "cohort_admissions": len(cohort), "cohort_patients": len(subjects),
        "note_admissions": 0, "sections": 0, "facts": 0, "candidates": 0,
        "unmatched_facts": 0, "source_facts": {}, "section_roles": {},
        "candidate_means_evidence": False,
        "prescriptions_are_confirmed_discharge_medications": False,
        "base_context_source_sha256": "",
    }
    with gzip.open(output_dir / "cohort_manifest.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subject_id", "hadm_id", "admittime", "dischtime", "patient_split",
                         "index_is_strict_aecopd"))
        for row in cohort.values():
            writer.writerow((row["subject_id"], row["hadm_id"], row["admittime"],
                             row["dischtime"], patient_split(int(row["subject_id"])),
                             row["index_is_strict_aecopd"]))

    roles_by_hadm: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    note_hadm: set[str] = set()
    role_counts: Counter[str] = Counter()
    with gzip.open(output_dir / "note_sections.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(NOTE_FIELDS)
        for row in _read(note_source):
            hadm = row["hadm_id"]
            if hadm not in cohort or row["subject_id"] != cohort[hadm]["subject_id"]:
                raise ValueError(f"note outside cohort or subject mismatch: {hadm}")
            if hadm in note_hadm:
                raise ValueError(f"multiple notes for admission: {hadm}")
            note_hadm.add(hadm)
            original = row["document_text"]
            document_hash = hashlib.sha256(original.encode("utf-8")).hexdigest()
            for section in parse_sections(original):
                if original[section.start_char:section.end_char] != section.text:
                    raise AssertionError("section offset mismatch")
                section_id = f"{row['note_id']}:{section.ordinal}"
                writer.writerow((hadm, row["note_id"], section_id, "", section.heading,
                                 section.role, section.start_char, section.end_char,
                                 section.text, section.parse_status, section.content_status,
                                 document_hash, row["timestamp_relation"]))
                roles_by_hadm[hadm][section.role].append(section_id)
                role_counts[section.role] += 1
                summary["sections"] += 1
    summary["note_admissions"] = len(note_hadm)
    summary["section_roles"] = dict(role_counts)

    counts: Counter[str] = Counter()
    with gzip.open(output_dir / "structured_facts.csv.gz", "wt", encoding="utf-8", newline="") as fact_handle, gzip.open(
        output_dir / "candidate_alignment.csv.gz", "wt", encoding="utf-8", newline=""
    ) as align_handle:
        facts = csv.writer(fact_handle)
        facts.writerow(FACT_FIELDS)
        align = csv.writer(align_handle)
        align.writerow(ALIGN_FIELDS)
        for source in ("labs", "diagnoses", "emar", "prescriptions"):
            for number, row in enumerate(_read(raw_dir / f"{source}.csv.gz"), start=1):
                hadm = row["hadm_id"]
                if hadm not in cohort or row["subject_id"] != cohort[hadm]["subject_id"]:
                    raise ValueError(f"fact outside cohort or subject mismatch: {source}:{number}")
                fact = _fact(source, row, number, cohort[hadm]["dischtime"])
                facts.writerow(fact)
                count = _add_candidates(fact[0], source, row, cohort[hadm]["dischtime"],
                                        roles_by_hadm[hadm], align)
                summary["facts"] += 1
                summary["candidates"] += count
                summary["unmatched_facts"] += int(count == 0)
                counts[source] += 1
        if base_source is not None:
            digest = hashlib.sha256()
            with base_source.open("rb") as source_handle:
                for block in iter(lambda: source_handle.read(1024 * 1024), b""):
                    digest.update(block)
            summary["base_context_source_sha256"] = digest.hexdigest()
            context_hadm: set[str] = set()
            for row in _read_plain(base_source):
                hadm = row["hadm_id"]
                if hadm not in cohort or row["subject_id"] != cohort[hadm]["subject_id"]:
                    raise ValueError("base context outside cohort")
                if hadm in context_hadm:
                    raise ValueError("duplicate base context admission")
                context_hadm.add(hadm)
                for field in CONTEXT_FIELDS:
                    value = row[field]
                    if value == "":
                        continue
                    number = counts["context"] + 1
                    is_discharge = field == "discharge_location"
                    event_time = row["dischtime"] if is_discharge else row["admittime"]
                    facts.writerow((f"context:{number}", row["subject_id"], hadm,
                                    "training.v5_export.BASE_SQL", f"{hadm}:{field}",
                                    field, field, value, "", event_time, event_time,
                                    "derived_as_of_discharge" if is_discharge else "derived_from_prior_or_static",
                                    "v5_context_without_outcome_label"))
                    count = _add_candidates(f"context:{number}", "context", {"field": field},
                                            row["dischtime"], roles_by_hadm[hadm], align)
                    summary["facts"] += 1
                    summary["candidates"] += count
                    summary["unmatched_facts"] += int(count == 0)
                    counts["context"] += 1
            if len(context_hadm) != len(cohort):
                raise ValueError("base context does not cover entire cohort")
    summary["source_facts"] = dict(counts)
    (output_dir / "qa_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--note-source", type=Path, required=True)
    parser.add_argument("--base-source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_dataset(args.raw_dir, args.note_source, args.output_dir, args.base_source)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/section_alignment_data/build_section_inputs.py · lines 1–292

<!-- REVIEW_FRAGMENT {"path":"training/v5/section_alignment_data/build_section_inputs.py","part":1,"start_line":1,"end_line":292,"source_sha256":"e6ef56bae1d69345a1b59f9c9edcef3fd9513e098ea137938b0e0d0512e400f6","fragment_sha256":"e6ef56bae1d69345a1b59f9c9edcef3fd9513e098ea137938b0e0d0512e400f6","added_terminal_newline":false} -->
````python
"""以出院小结章节为锚，生成稀疏结构化概念 token 和文本章节位置。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterator

from training.v5.clinical_concepts import LABS, MED_NAMES, diagnosis_groups, medication_classes


SECTION_ROLES = (
    "patient_context", "allergies", "chief_complaint", "major_procedure",
    "history_of_present_illness", "past_medical_history", "social_history", "family_history",
    "physical_exam", "admission_physical_exam", "review_of_systems", "admission_medications",
    "admission_labs", "pertinent_results", "imaging", "microbiology", "hospital_course",
    "discharge_labs", "discharge_physical_exam", "discharge_medications", "discharge_diagnosis",
    "discharge_condition", "discharge_disposition", "discharge_instructions",
    "followup_instructions", "transitional_issues",
)
CHRONIC = ("heart_failure", "cardiac_other", "ckd", "diabetes", "malignancy", "chronic_respiratory")
ACUTE = ("acute_resp_failure", "pneumonia_or_aspiration", "sepsis", "aki")
TOKEN_FIELDS = ("subject_id", "hadm_id", "section_role", "token_type", "concept",
                "value0", "value1", "value2", "value3", "value4",
                "mask0", "mask1", "mask2", "mask3", "mask4", "unit", "provenance")
SLOT_FIELDS = ("subject_id", "hadm_id", "section_role", "text_section_ids",
               "text_nonblank_count", "structured_token_count", "has_discharge_note")


def _rows(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _days(after: str, before: str) -> float:
    return (datetime.fromisoformat(after) - datetime.fromisoformat(before)).total_seconds() / 86400.0


def _token(
    subject: str, hadm: str, role: str, kind: str, concept: str,
    values: tuple[float, ...], masks: tuple[int, ...], unit: str, provenance: str,
) -> tuple[str, ...]:
    if len(values) > 5 or len(values) != len(masks):
        raise ValueError("invalid token channel count")
    padded_values = (*values, *(0.0 for _ in range(5 - len(values))))
    padded_masks = (*masks, *(0 for _ in range(5 - len(masks))))
    return (subject, hadm, role, kind, concept,
            *(str(value) for value in padded_values),
            *(str(mask) for mask in padded_masks), unit, provenance)


def prior_chronic_history(
    cohort: dict[str, dict[str, str]], diagnosis: dict[str, set[str]],
) -> dict[str, set[str]]:
    """只汇总当前入院前已结束的、队列内更早住院的慢性诊断编码。"""

    patient_admissions: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in cohort.values():
        patient_admissions[row["subject_id"]].append(row)
    result: dict[str, set[str]] = {}
    for admissions in patient_admissions.values():
        admissions.sort(key=lambda row: (row["admittime"], row["hadm_id"]))
        previous: list[dict[str, str]] = []
        for current in admissions:
            result[current["hadm_id"]] = set().union(*(
                diagnosis[older["hadm_id"]] & set(CHRONIC)
                for older in previous if older["dischtime"] <= current["admittime"]
            ))
            previous.append(current)
    return result


def build_section_inputs(
    raw_dir: Path, aligned_dir: Path, base_csv: Path, output_dir: Path,
) -> dict[str, object]:
    """同一合格出院一组章节槽位；只读源数据，不写模型标签。"""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    validation = json.loads((aligned_dir / "qa_validation.json").read_text(encoding="utf-8"))
    if validation["status"] != "structural_checks_passed":
        raise ValueError("unaligned or unvalidated section dataset")
    for name, expected in validation["output_sha256"].items():
        if _hash(aligned_dir / name) != expected:
            raise ValueError(f"aligned source changed: {name}")
    cohort = {row["hadm_id"]: row for row in _rows(aligned_dir / "cohort_manifest.csv.gz")}
    base = {row["hadm_id"]: row for row in _rows(base_csv)}
    if len(cohort) != len(base) or cohort.keys() != base.keys():
        raise ValueError("base and section cohort differ")
    for hadm, row in base.items():
        if row["subject_id"] != cohort[hadm]["subject_id"] or row["dischtime"] != cohort[hadm]["dischtime"]:
            raise ValueError("base and section key/time mismatch")

    text_sections: dict[tuple[str, str], list[str]] = defaultdict(list)
    nonblank: Counter[tuple[str, str]] = Counter()
    note_hadms: set[str] = set()
    for row in _rows(aligned_dir / "note_sections.csv.gz"):
        hadm, role = row["hadm_id"], row["section_role"]
        note_hadms.add(hadm)
        if role not in SECTION_ROLES:
            continue
        text_sections[(hadm, role)].append(row["section_id"])
        if row["content_status"] == "informative":
            nonblank[(hadm, role)] += 1

    diagnosis: dict[str, set[str]] = defaultdict(set)
    for row in _rows(raw_dir / "diagnoses.csv.gz"):
        if row["hadm_id"] not in cohort:
            raise ValueError("diagnosis outside cohort")
        diagnosis[row["hadm_id"]].update(diagnosis_groups(row["icd_code"], int(row["icd_version"])))

    # 历史只包括本次入院前已结束的合格住院；当前出院编码另作一维代理。
    prior_chronic = prior_chronic_history(cohort, diagnosis)

    lab_tokens: dict[str, list[tuple[str, tuple[float, ...], tuple[int, ...], str]]] = defaultdict(list)
    current_key: tuple[str, str] | None = None
    series: list[float] = []
    unit = ""

    def flush_lab() -> None:
        if current_key is None:
            return
        hadm, itemid = current_key
        first, last = series[0], series[-1]
        values = (first, last, min(series), max(series), last - first if len(series) > 1 else 0.0)
        masks = (1, 1, 1, 1, int(len(series) > 1))
        lab_tokens[hadm].append((LABS[int(itemid)][0], values, masks, unit))

    for row in _rows(raw_dir / "labs.csv.gz"):
        key = row["hadm_id"], row["itemid"]
        if key != current_key:
            flush_lab()
            current_key, series, unit = key, [], row["valueuom"]
        elif row["valueuom"] != unit:
            raise ValueError("mixed lab units within admission/concept")
        series.append(float(row["valuenum"]))
    flush_lab()

    medications: dict[tuple[str, str], tuple[str, str]] = {}
    hadms_with_emar: set[str] = set()
    for row in _rows(raw_dir / "emar.csv.gz"):
        hadm = row["hadm_id"]
        hadms_with_emar.add(hadm)
        for drug_class in medication_classes(row["medication"]):
            key = hadm, drug_class
            time = row["charttime"]
            if key in medications:
                first, last = medications[key]
                medications[key] = min(first, time), max(last, time)
            else:
                medications[key] = time, time

    output_dir.mkdir(parents=True)
    counts: Counter[str] = Counter()
    tokens_by_slot: Counter[tuple[str, str]] = Counter()
    with gzip.open(output_dir / "structured_tokens.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TOKEN_FIELDS)

        def write(token: tuple[str, ...]) -> None:
            writer.writerow(token)
            counts[token[3]] += 1
            tokens_by_slot[(token[1], token[2])] += 1

        for hadm, current in cohort.items():
            subject, admission = current["subject_id"], current["admittime"]
            context = base[hadm]
            for concept, value in (
                ("age_at_admission", float(context["age_at_admission"])),
                ("male", float(context["gender"] == "M")),
                ("prior_coded_admissions", float(context["prior_coded_admission_count"])),
                ("prior_strict_aecopd_admissions", float(context["prior_strict_aecopd_count"])),
            ):
                write(_token(subject, hadm, "patient_context", "context", concept,
                             (value,), (1,), "", "v5_base_whitelist"))

            for concept in CHRONIC:
                if concept not in diagnosis[hadm] and concept not in prior_chronic[hadm]:
                    continue
                write(_token(subject, hadm, "past_medical_history", "chronic_comorbidity", concept,
                             (float(concept in diagnosis[hadm]), float(concept in prior_chronic[hadm])),
                             (1, 1), "", "current_discharge_code_proxy|prior_eligible_admission_code"))

            for concept, values, masks, value_unit in lab_tokens.get(hadm, []):
                write(_token(subject, hadm, "pertinent_results", "laboratory", concept,
                             values, masks, value_unit, "in_stay_labevents_recorded_by_discharge"))

            for concept in MED_NAMES:
                times = medications.get((hadm, concept))
                if times is None:
                    continue
                first, last = times
                write(_token(subject, hadm, "hospital_course", "administered_medication", concept,
                             (1.0, _days(first, admission), _days(last, admission), _days(last, first)),
                             (1, 1, 1, 1), "days", "in_stay_emar_not_discharge_prescription"))

            write(_token(subject, hadm, "discharge_diagnosis", "aecopd_index_code",
                         "strict_aecopd", (float(context["index_is_strict_aecopd"] in ("t", "true", "True", "1")),),
                         (1,), "", "current_discharge_diagnosis_code"))
            for concept in ACUTE:
                if concept not in diagnosis[hadm]:
                    continue
                write(_token(subject, hadm, "discharge_diagnosis", "acute_diagnosis", concept,
                             (float(concept in diagnosis[hadm]),), (1,), "",
                             "current_discharge_diagnosis_code"))

            destination = context["discharge_location"].upper()
            if destination:
                category = ("home" if "HOME" in destination else "facility" if any(
                    term in destination for term in ("REHAB", "SKILLED", "NURSING", "LONG TERM")
                ) else "other")
                write(_token(subject, hadm, "discharge_disposition", "disposition", category,
                             (1.0,), (1,), "", "v5_base_discharge_location"))

    with gzip.open(output_dir / "section_slots.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SLOT_FIELDS)
        for hadm, current in cohort.items():
            for role in SECTION_ROLES:
                ids = text_sections.get((hadm, role), [])
                writer.writerow((current["subject_id"], hadm, role,
                                 json.dumps(ids, separators=(",", ":")),
                                 nonblank[(hadm, role)], tokens_by_slot[(hadm, role)],
                                 int(hadm in note_hadms)))

    schema: dict[str, object] = {
        "cohort_admissions": len(cohort), "roles": SECTION_ROLES,
        "token_fields": TOKEN_FIELDS, "slot_fields": SLOT_FIELDS,
        "token_channels": {
            "context": ["value"],
            "chronic_comorbidity": ["coded_this_discharge", "coded_in_prior_eligible_admission"],
            "laboratory": ["first", "last", "minimum", "maximum", "last_minus_first"],
            "administered_medication": ["exposed", "first_day_from_admission", "last_day_from_admission", "exposure_span_days"],
            "aecopd_index_code": ["current_strict_aecopd_code"],
            "acute_diagnosis": ["coded_this_discharge"],
            "disposition": ["category_present"],
        },
        "structure_is_section_anchored_not_old_six_groups": True,
        "no_outcome_labels_in_outputs": True,
        "prescriptions_and_pharmacy_excluded": True,
        "diagnosis_tokens_are_positive_only": True,
        "missing_diagnosis_token_means_not_coded_not_disease_absent": True,
        "unmatched_text_only_roles_preserved": True,
        "source_sha256": {
            "aligned_qa_validation": _hash(aligned_dir / "qa_validation.json"),
            "v5_base": _hash(base_csv),
            "raw_labs": _hash(raw_dir / "labs.csv.gz"),
            "raw_diagnoses": _hash(raw_dir / "diagnoses.csv.gz"),
            "raw_emar": _hash(raw_dir / "emar.csv.gz"),
        },
    }
    (output_dir / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    report: dict[str, object] = {
        "admissions": len(cohort), "patients": len({row["subject_id"] for row in cohort.values()}),
        "roles_per_admission": len(SECTION_ROLES), "section_slots": len(cohort) * len(SECTION_ROLES),
        "structured_tokens": sum(counts.values()), "token_types": dict(counts),
        "admissions_with_note": len(note_hadms), "admissions_with_emar": len(hadms_with_emar),
        "admissions_with_any_lab": len(lab_tokens),
        "outcome_labels_exported": False,
    }
    (output_dir / "qa_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_section_inputs(args.raw_dir, args.aligned_dir, args.base_csv, args.output_dir),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/section_alignment_data/export_raw_sources.py · lines 1–140

<!-- REVIEW_FRAGMENT {"path":"training/v5/section_alignment_data/export_raw_sources.py","part":1,"start_line":1,"end_line":140,"source_sha256":"f9454e7632ff648ab92361cf284e4ba9fb3cc87f4c82368eec8c98b4c4bdefb8","fragment_sha256":"f9454e7632ff648ab92361cf284e4ba9fb3cc87f4c82368eec8c98b4c4bdefb8","added_terminal_newline":false} -->
````python
"""只读导出 V5 章节对齐所需的原始结构化事件。"""

from __future__ import annotations

import argparse
import csv
import getpass
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


COHORT = "project_aecopd.copd_coded_all_index_labels_v2"


def _copy_sql(select: str) -> str:
    return f"COPY ({select}) TO STDOUT WITH (FORMAT CSV, HEADER TRUE)"


QUERIES: dict[str, str] = {
    "cohort": _copy_sql(f"""
        SELECT l.subject_id,l.hadm_id,l.admittime,l.dischtime,l.index_is_strict_aecopd
        FROM {COHORT} l ORDER BY l.subject_id,l.admittime,l.hadm_id
    """),
    "labs": _copy_sql(f"""
        SELECT x.subject_id,x.hadm_id,x.labevent_id,x.itemid,x.charttime,x.storetime,
               x.valuenum,x.valueuom,x.flag
        FROM {COHORT} l JOIN mimiciv_hosp.labevents x USING (hadm_id)
        WHERE x.itemid IN (50818,50820,50821,50862,50882,50912,51222,51301)
          AND x.charttime BETWEEN l.admittime AND l.dischtime
          AND x.storetime <= l.dischtime AND x.valuenum IS NOT NULL AND x.valuenum >= 0
        ORDER BY x.hadm_id,x.itemid,x.charttime,x.labevent_id
    """),
    "diagnoses": _copy_sql(f"""
        SELECT d.subject_id,d.hadm_id,d.seq_num,btrim(d.icd_code) AS icd_code,d.icd_version
        FROM {COHORT} l JOIN mimiciv_hosp.diagnoses_icd d USING (hadm_id)
        ORDER BY d.hadm_id,d.seq_num,d.icd_code
    """),
    "emar": _copy_sql(f"""
        SELECT e.subject_id,e.hadm_id,e.emar_id,e.emar_seq,e.pharmacy_id,
               e.charttime,e.storetime,e.medication,e.event_txt
        FROM {COHORT} l JOIN mimiciv_hosp.emar e USING (hadm_id)
        WHERE e.charttime BETWEEN l.admittime AND l.dischtime
          AND e.storetime <= l.dischtime
          AND e.event_txt IN ('Administered','Started','Applied','Delayed Administered')
          AND e.medication IS NOT NULL AND btrim(e.medication) <> ''
        ORDER BY e.hadm_id,e.charttime,e.emar_id
    """),
    "prescriptions": _copy_sql(f"""
        SELECT p.subject_id,p.hadm_id,p.pharmacy_id,p.poe_id,p.starttime,p.stoptime,
               p.drug_type,p.drug,p.gsn,p.route
        FROM {COHORT} l JOIN mimiciv_hosp.prescriptions p USING (hadm_id)
        ORDER BY p.hadm_id,p.pharmacy_id,p.drug_type,p.drug
    """),
    "pharmacy": _copy_sql(f"""
        SELECT p.subject_id,p.hadm_id,p.pharmacy_id,p.poe_id,p.starttime,p.stoptime,
               p.entertime,p.verifiedtime,p.medication,p.status,p.proc_type
        FROM {COHORT} l JOIN mimiciv_hosp.pharmacy p USING (hadm_id)
        ORDER BY p.hadm_id,p.pharmacy_id
    """),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_count(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def export(output_dir: Path, psql: Path, password: str, note_source: Path) -> dict[str, object]:
    """SQL 全部为只读 COPY；每份压缩文件完成后再原子改名。"""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not psql.is_file() or not note_source.is_file():
        raise FileNotFoundError("psql or source notes missing")
    if any(not sql.lstrip().upper().startswith("COPY (") for sql in QUERIES.values()):
        raise ValueError("export queries must be read-only COPY")
    output_dir.mkdir(parents=True)
    environment = os.environ.copy()
    environment["PGPASSWORD"] = password
    summary: dict[str, object] = {
        "cohort": "MIMIC-IV V5 discharge prediction sample set",
        "source_note_gzip_sha256": _sha256(note_source),
        "sources": {},
        "prescriptions_are_not_confirmed_discharge_medications": True,
    }
    for name, sql in QUERIES.items():
        target = output_dir / f"{name}.csv.gz"
        partial = output_dir / f"{name}.csv.gz.partial"
        with gzip.open(partial, "wb", compresslevel=6) as destination:
            process = subprocess.Popen(
                [str(psql), "-X", "-q", "-w", "-h", "localhost", "-U", "postgres",
                 "-d", "mimiciv", "-v", "ON_ERROR_STOP=1", "-c", sql],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
            )
            assert process.stdout is not None and process.stderr is not None
            shutil.copyfileobj(process.stdout, destination)
            error = process.stderr.read()
            status = process.wait()
        if status:
            raise RuntimeError(f"{name} export failed: {error.decode('utf-8', errors='replace')}")
        partial.replace(target)
        summary["sources"][name] = {
            "rows": _row_count(target), "sha256": _sha256(target),
            "bytes_compressed": target.stat().st_size,
        }
        print(json.dumps({"exported": name, **summary["sources"][name]}), flush=True)
    if summary["sources"]["cohort"]["rows"] != 19401:
        raise ValueError("cohort row count changed")
    (output_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--note-source", type=Path, required=True)
    parser.add_argument("--psql", type=Path,
                        default=Path("C:/Program Files/PostgreSQL/18/bin/psql.exe"))
    args = parser.parse_args()
    password = os.environ.get("PGPASSWORD") or getpass.getpass("PostgreSQL password: ")
    export(args.output_dir, args.psql, password, args.note_source)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/section_alignment_data/prepare_section_text.py · lines 1–198

<!-- REVIEW_FRAGMENT {"path":"training/v5/section_alignment_data/prepare_section_text.py","part":1,"start_line":1,"end_line":198,"source_sha256":"87adcca9a6bf7c162fd81a40641b667f53138d7198e158505e47564e2abb064d","fragment_sha256":"87adcca9a6bf7c162fd81a40641b667f53138d7198e158505e47564e2abb064d","added_terminal_newline":false} -->
````python
"""将已校验的小结章节切成 BERT/Qwen 共用、可回查原文位置的编码清单。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterator, Protocol, Sequence


PILOT_ROLES = (
    "history_of_present_illness", "past_medical_history", "admission_medications",
    "hospital_course", "discharge_medications", "discharge_condition",
)
FIELDS = (
    "embedding_row", "subject_id", "hadm_id", "section_role", "section_id", "chunk_index",
    "source_start_char", "source_end_char", "source_section_sha256", "timestamp_relation",
    "bert_token_count", "token_count", "text_sha256", "chunk_text",
)


class OffsetTokenizer(Protocol):
    def __call__(self, text: str, *, add_special_tokens: bool,
                 return_offsets_mapping: bool) -> dict[str, Sequence[tuple[int, int]]]: ...


def _rows(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_offsets(
    offsets: Sequence[tuple[int, int]], max_tokens: int, overlap_tokens: int,
) -> tuple[tuple[int, int, int], ...]:
    """返回原片段字符起止与块索引；字符偏移右端不包含。"""

    if max_tokens < 1 or not 0 <= overlap_tokens < max_tokens:
        raise ValueError("invalid token window")
    if not offsets:
        return ()
    chunks: list[tuple[int, int, int]] = []
    start = 0
    while start < len(offsets):
        end = min(start + max_tokens, len(offsets))
        left, right = offsets[start][0], offsets[end - 1][1]
        if not 0 <= left < right:
            raise ValueError("invalid tokenizer character offsets")
        chunks.append((left, right, len(chunks)))
        if end == len(offsets):
            break
        start = end - overlap_tokens
    return tuple(chunks)


def prepare(
    aligned_dir: Path, section_inputs_dir: Path, output_dir: Path,
    bert_tokenizer: OffsetTokenizer, qwen_tokenizer: object,
    *, roles: tuple[str, ...] = PILOT_ROLES, max_tokens: int = 384,
    overlap_tokens: int = 32,
) -> dict[str, object]:
    """同一块原文分别记精确 BERT/Qwen 长度，不做医学语义改写。"""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not roles or len(set(roles)) != len(roles):
        raise ValueError("empty or duplicate section roles")
    aligned_validation = json.loads((aligned_dir / "qa_validation.json").read_text(encoding="utf-8"))
    input_validation = json.loads((section_inputs_dir / "validation.json").read_text(encoding="utf-8"))
    if aligned_validation["status"] != "structural_checks_passed" or input_validation["status"] != "section_inputs_validated":
        raise ValueError("unvalidated source dataset")
    if _sha256(aligned_dir / "note_sections.csv.gz") != aligned_validation["output_sha256"]["note_sections.csv.gz"]:
        raise ValueError("source section file changed")
    if _sha256(section_inputs_dir / "section_slots.csv.gz") != input_validation["output_sha256"]["section_slots.csv.gz"]:
        raise ValueError("section slots changed")
    cohort = {row["hadm_id"]: row["subject_id"]
              for row in _rows(aligned_dir / "cohort_manifest.csv.gz")}
    # 只允许章节级输入实际引用过的片段，避免旁路引入未选中的文书。
    referenced: set[str] = set()
    for row in _rows(section_inputs_dir / "section_slots.csv.gz"):
        if row["section_role"] in roles:
            referenced.update(json.loads(row["text_section_ids"]))

    output_dir.mkdir(parents=True)
    counts: Counter[str] = Counter()
    unique_hadms: set[str] = set()
    max_bert = max_qwen = max_chunks = 0
    source_sections = 0
    with gzip.open(output_dir / "shared_text.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for section in _rows(aligned_dir / "note_sections.csv.gz"):
            role, sid = section["section_role"], section["section_id"]
            if role not in roles:
                continue
            counts["section_total_" + role] += 1
            if sid not in referenced:
                raise ValueError("selected section not referenced by section slot")
            if section["parse_status"] == "uncertain":
                counts["excluded_uncertain_" + role] += 1
                continue
            if section["content_status"] != "informative":
                counts["excluded_placeholder_" + role] += 1
                continue
            text = section["text"]
            tokenized = bert_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = tokenized["offset_mapping"]
            source_sections += 1
            chunks = chunk_offsets(offsets, max_tokens, overlap_tokens)
            if not chunks:
                raise ValueError("informative section tokenized to zero chunks")
            max_chunks = max(max_chunks, len(chunks))
            for left, right, chunk_index in chunks:
                chunk = text[left:right]
                global_left = int(section["start_char"]) + left
                global_right = int(section["start_char"]) + right
                if global_right > int(section["end_char"]) or not chunk.strip():
                    raise ValueError("chunk exceeds source section or is blank")
                bert_count = len(bert_tokenizer(chunk, add_special_tokens=True,
                                                return_offsets_mapping=True)["offset_mapping"])
                qwen_count = len(qwen_tokenizer(chunk, add_special_tokens=False)["input_ids"])
                if bert_count > 512 or qwen_count > 8192:
                    raise ValueError("chunk exceeds encoder context limit")
                row_number = counts["chunks"]
                writer.writerow((row_number, cohort[section["hadm_id"]], section["hadm_id"],
                                 role, sid, chunk_index, global_left, global_right,
                                 section["source_sha256"], section["timestamp_relation"],
                                 bert_count, qwen_count, hashlib.sha256(chunk.encode("utf-8")).hexdigest(), chunk))
                counts["chunks"] += 1
                counts["chunks_" + role] += 1
                unique_hadms.add(section["hadm_id"])
                max_bert = max(max_bert, bert_count)
                max_qwen = max(max_qwen, qwen_count)
            counts["included_" + role] += 1
    if counts["chunks"] == 0:
        raise ValueError("no text chunks selected")
    manifest = output_dir / "shared_text.csv.gz"
    report: dict[str, object] = {
        "status": "candidate_manifest_not_encoded", "roles": roles,
        "source_sections_included": source_sections,
        "admissions_with_text": len(unique_hadms), "chunks": counts["chunks"],
        "max_chunks_per_section": max_chunks, "max_bert_tokens": max_bert,
        "max_qwen_tokens": max_qwen,
        "section_role_counts": {role: {
            "total": counts["section_total_" + role],
            "included": counts["included_" + role],
            "excluded_placeholder": counts["excluded_placeholder_" + role],
            "excluded_uncertain": counts["excluded_uncertain_" + role],
            "chunks": counts["chunks_" + role],
        } for role in roles},
        "bert_chunk_target_tokens": max_tokens,
        "bert_overlap_tokens": overlap_tokens,
        "source_note_sections_sha256": aligned_validation["output_sha256"]["note_sections.csv.gz"],
        "source_section_slots_sha256": input_validation["output_sha256"]["section_slots.csv.gz"],
        "manifest_sha256": _sha256(manifest),
        "outcome_labels_exported": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--section-inputs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bert-model-dir", type=Path, default=Path("/model-cache/Bio_ClinicalBERT"))
    parser.add_argument("--qwen-model-dir", type=Path, default=Path("/model-cache/Qwen3-Embedding-4B"))
    parser.add_argument("--max-bert-tokens", type=int, default=384)
    parser.add_argument("--overlap-tokens", type=int, default=32)
    parser.add_argument("--roles", nargs="+", default=PILOT_ROLES,
                        help="Selected parsed section roles; omit for the original six-role pilot")
    args = parser.parse_args()
    bert = AutoTokenizer.from_pretrained(args.bert_model_dir, local_files_only=True, use_fast=True)
    qwen = AutoTokenizer.from_pretrained(args.qwen_model_dir, local_files_only=True)
    if not bert.is_fast:
        raise ValueError("BERT tokenizer must provide exact character offsets")
    result = prepare(args.aligned_dir, args.section_inputs_dir, args.output_dir, bert, qwen,
                     roles=tuple(args.roles), max_tokens=args.max_bert_tokens,
                     overlap_tokens=args.overlap_tokens)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/section_alignment_data/screen_section_content.py · lines 1–118

<!-- REVIEW_FRAGMENT {"path":"training/v5/section_alignment_data/screen_section_content.py","part":1,"start_line":1,"end_line":118,"source_sha256":"9f0662c9e6bfab859df5d570ee4d2911339b1d735715016f8d826adef9bfbf09","fragment_sha256":"9f0662c9e6bfab859df5d570ee4d2911339b1d735715016f8d826adef9bfbf09","added_terminal_newline":false} -->
````python
"""对候选章节做可复现的内容风险筛查；规则仅供复核，不自动删改原文。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from training.v5.section_alignment_data.prepare_section_text import PILOT_ROLES


_ADMIN_HEADER = re.compile(
    r"(?im)^\s*(?:PCP|Primary Care (?:Physician|Provider)|Contact|Phone|Fax)\s*:"
)
_DRUG_LIST_CUE = re.compile(
    r"(?i)(?:\bSig\s*:|\bPO\s+(?:DAILY|BID|TID)\b|\b\d+(?:\.\d+)?\s*(?:mg|mcg)\b)"
)
_SHORT_CONDITION = re.compile(
    r"(?is)^\s*(?:stable|good|fair|improved|guarded|satisfactory|unchanged)\.?\s*$"
)
_FUNCTION_CUE = re.compile(
    r"(?i)\b(?:activity status|ambulatory|ambulates?|mental status|"
    r"level of consciousness|alert|independent|assistance|oxygen|room air)\b"
)
_COURSE_SUBHEADING = re.compile(r"(?m)^\s*#{1,3}\s*\S")


def screen_flags(role: str, text: str, timestamp_relation: str) -> tuple[str, ...]:
    """返回待复核风险标记；它们既不是临床标签，也不是编码过滤条件。"""

    flags: list[str] = []
    if timestamp_relation == "after_discharge":
        flags.append("stored_after_discharge")
    if len(text) > 2500:
        flags.append("long_over_2500_chars")
    if role == "history_of_present_illness" and _ADMIN_HEADER.search(text[:500]):
        flags.append("administrative_header_first_500_chars")
    if role in {"admission_medications", "discharge_medications"} and _DRUG_LIST_CUE.search(text):
        flags.append("drug_list_format")
    if role == "hospital_course" and _COURSE_SUBHEADING.search(text):
        flags.append("problem_subheadings")
    if role == "discharge_condition":
        if len(text) > 500:
            flags.append("condition_over_500_chars_review_boundary")
        if _SHORT_CONDITION.fullmatch(text):
            flags.append("generic_condition_only")
        if _FUNCTION_CUE.search(text):
            flags.append("function_or_respiratory_cue")
    return tuple(flags)


def screen(aligned_dir: Path, output_dir: Path) -> dict[str, object]:
    """核对来源指纹并保存无正文计数及分层复核索引。"""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    qa = json.loads((aligned_dir / "qa_validation.json").read_text(encoding="utf-8"))
    source = aligned_dir / "note_sections.csv.gz"
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if qa["status"] != "structural_checks_passed" or digest.hexdigest() != qa["output_sha256"][source.name]:
        raise ValueError("source section dataset not validated")

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    # 固定哈希顺序，短／中／长篇每类两例；只保存源键，正文仍留在私有原始清单。
    examples: dict[tuple[str, str], list[tuple[str, dict[str, str]]]] = defaultdict(list)
    with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            role = row["section_role"]
            if role not in PILOT_ROLES or row["parse_status"] == "uncertain" or row["content_status"] != "informative":
                continue
            text = row["text"]
            counter = counts[role]
            counter["sections"] += 1
            for flag in screen_flags(role, text, row["timestamp_relation"]):
                counter[flag] += 1
            bucket = "short" if len(text) < 500 else "long" if len(text) > 2500 else "medium"
            score = hashlib.sha256(("sample-v1-" + row["section_id"]).encode("utf-8")).hexdigest()
            examples[(role, bucket)].append((score, row))

    output_dir.mkdir(parents=True)
    with (output_dir / "sample_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("section_role", "length_stratum", "section_id", "hadm_id", "note_id",
                         "start_char", "end_char", "timestamp_relation", "text_length"))
        for role in PILOT_ROLES:
            for bucket in ("short", "medium", "long"):
                for _, row in sorted(examples[(role, bucket)], key=lambda item: item[0])[:2]:
                    writer.writerow((role, bucket, row["section_id"], row["hadm_id"], row["note_id"],
                                     row["start_char"], row["end_char"], row["timestamp_relation"], len(row["text"])))
    result: dict[str, object] = {
        "status": "screening_flags_not_clinical_truth",
        "source_note_sections_sha256": digest.hexdigest(),
        "counts": {role: dict(counts[role]) for role in PILOT_ROLES},
        "sampling": "two SHA-256-ranked sections per role and length stratum; first 800 characters manually inspected after redaction",
        "automatic_exclusions": 0,
    }
    (output_dir / "screening.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(screen(args.aligned_dir, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/section_alignment_data/section_parser.py · lines 1–120

<!-- REVIEW_FRAGMENT {"path":"training/v5/section_alignment_data/section_parser.py","part":1,"start_line":1,"end_line":120,"source_sha256":"b3038f2bef67824dfb399a278ae038d899162030076092c9b983a3a01c720251","fragment_sha256":"b3038f2bef67824dfb399a278ae038d899162030076092c9b983a3a01c720251","added_terminal_newline":false} -->
````python
"""保留原文偏移的出院小结主章节切分。"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedSection:
    """同一小结中的一段原文；偏移为 Python 字符索引，右端不包含。"""

    ordinal: int
    role: str
    heading: str
    start_char: int
    end_char: int
    text: str
    parse_status: str
    content_status: str


_ROLES: dict[str, str] = {
    "allergies": "allergies",
    "chief complaint": "chief_complaint",
    "major surgical or invasive procedure": "major_procedure",
    "history of present illness": "history_of_present_illness",
    "hpi": "history_of_present_illness",
    "past medical history": "past_medical_history",
    "social history": "social_history",
    "family history": "family_history",
    "physical exam": "physical_exam",
    "pertinent results": "pertinent_results",
    "brief hospital course": "hospital_course",
    "hospital course": "hospital_course",
    "medications on admission": "admission_medications",
    "discharge medications": "discharge_medications",
    "discharge disposition": "discharge_disposition",
    "discharge diagnosis": "discharge_diagnosis",
    "discharge diagnoses": "discharge_diagnosis",
    "discharge condition": "discharge_condition",
    "discharge instructions": "discharge_instructions",
    "followup instructions": "followup_instructions",
    "follow-up instructions": "followup_instructions",
    "transitional issues": "transitional_issues",
    "admission physical exam": "admission_physical_exam",
    "admission exam": "admission_physical_exam",
    "discharge physical exam": "discharge_physical_exam",
    "discharge exam": "discharge_physical_exam",
    "admission labs": "admission_labs",
    "discharge labs": "discharge_labs",
    "imaging": "imaging",
    "microbiology": "microbiology",
    "review of systems": "review_of_systems",
    "attending": "metadata",
    "service": "metadata",
}

_HEADER = re.compile(r"^\s*([A-Za-z][A-Za-z /&()'\-]{1,64})\s*:\s*(?P<inline>.*)$")
_NON_PLACEHOLDER = re.compile(r"[A-Za-z0-9]")


def _normalize_heading(value: str) -> str:
    return " ".join(value.lower().split())


def _header_role(heading: str, previous_role: str | None) -> tuple[str, str] | None:
    role = _ROLES.get(heading)
    if role is not None:
        return role, "recognized"
    if heading in {"diagnosis", "primary diagnosis", "secondary diagnosis", "primary diagnoses",
                   "secondary diagnoses", "primary", "secondary"}:
        if previous_role in {"discharge_disposition", "discharge_diagnosis"}:
            if previous_role == "discharge_diagnosis":
                return None  # 诊断节内部的分层标签，不另开章节。
            return "discharge_diagnosis", "contextual_alias"
        return None
    if heading == "history" and previous_role == "social_history":
        return "unclassified", "uncertain"
    return None


def parse_sections(original: str) -> tuple[ParsedSection, ...]:
    """切分主章节；未知标题保留在原文，歧义分界不伪装成社会史。"""

    sections: list[ParsedSection] = []
    current: tuple[str, str, str, int] | None = None

    def flush(boundary: int) -> None:
        if current is None:
            return
        role, heading, parse_status, start = current
        raw = original[start:boundary]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        start_char = start + left
        end_char = max(start_char, start + right)
        body = original[start_char:end_char]
        sections.append(ParsedSection(
            ordinal=len(sections), role=role, heading=heading,
            start_char=start_char, end_char=end_char, text=body,
            parse_status=parse_status,
            content_status="informative" if _NON_PLACEHOLDER.search(body) else "placeholder",
        ))

    offset = 0
    for line in original.splitlines(keepends=True):
        match = _HEADER.fullmatch(line.rstrip("\r\n"))
        if match is not None:
            heading = _normalize_heading(match.group(1))
            result = _header_role(heading, current[0] if current else None)
            if result is not None:
                flush(offset)
                role, parse_status = result
                inline = match.group("inline")
                start = offset + match.start("inline") if inline.strip() else offset + len(line)
                current = (role, heading, parse_status, start)
        offset += len(line)
    flush(len(original))
    return tuple(sections)
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/section_alignment_data/source_inventory.sql · lines 1–59

<!-- REVIEW_FRAGMENT {"path":"training/v5/section_alignment_data/source_inventory.sql","part":1,"start_line":1,"end_line":59,"source_sha256":"6e6bd9634f934722f6c3c0f9d2f01a9ce75d879c95e0bcdc3b30aed457e44a07","fragment_sha256":"6e6bd9634f934722f6c3c0f9d2f01a9ce75d879c95e0bcdc3b30aed457e44a07","added_terminal_newline":false} -->
````sql
-- V5 章节对齐的只读来源盘点；不得修改现有队列或标签。
BEGIN TRANSACTION READ ONLY;

SELECT 'cohort' AS source, count(*) AS rows,
       count(DISTINCT subject_id) AS patients,
       count(DISTINCT hadm_id) AS admissions
FROM project_aecopd.copd_coded_all_index_labels_v2;

SELECT 'discharge_notes' AS source, count(*) AS rows,
       count(DISTINCT n.subject_id) AS patients,
       count(DISTINCT n.hadm_id) AS admissions,
       count(*) FILTER (WHERE n.storetime > l.dischtime) AS stored_after_discharge
FROM project_aecopd.copd_coded_all_index_labels_v2 l
JOIN mimiciv_note.discharge n USING (subject_id, hadm_id)
WHERE n.text IS NOT NULL AND btrim(n.text) <> '';

SELECT 'selected_labevents' AS source, count(*) AS rows,
       count(DISTINCT x.hadm_id) AS admissions,
       count(DISTINCT x.itemid) AS concepts
FROM project_aecopd.copd_coded_all_index_labels_v2 l
JOIN mimiciv_hosp.labevents x USING (hadm_id)
WHERE x.itemid IN (50818,50820,50821,50862,50882,50912,51222,51301)
  AND x.charttime BETWEEN l.admittime AND l.dischtime
  AND x.storetime <= l.dischtime AND x.valuenum IS NOT NULL AND x.valuenum >= 0;

SELECT 'diagnoses_icd' AS source, count(*) AS rows,
       count(DISTINCT d.hadm_id) AS admissions,
       count(DISTINCT (d.icd_version, d.icd_code)) AS concepts
FROM project_aecopd.copd_coded_all_index_labels_v2 l
JOIN mimiciv_hosp.diagnoses_icd d USING (hadm_id);

SELECT 'emar_administered' AS source, count(*) AS rows,
       count(DISTINCT e.hadm_id) AS admissions,
       count(DISTINCT lower(btrim(e.medication))) AS medication_names
FROM project_aecopd.copd_coded_all_index_labels_v2 l
JOIN mimiciv_hosp.emar e USING (hadm_id)
WHERE e.charttime BETWEEN l.admittime AND l.dischtime
  AND e.storetime <= l.dischtime
  AND e.event_txt IN ('Administered','Started','Applied','Delayed Administered')
  AND e.medication IS NOT NULL AND btrim(e.medication) <> '';

SELECT 'prescriptions' AS source, count(*) AS rows,
       count(DISTINCT p.hadm_id) AS admissions,
       count(*) FILTER (WHERE p.starttime >= l.dischtime - interval '24 hours'
                        AND p.starttime <= l.dischtime) AS started_last_24h,
       count(*) FILTER (WHERE p.stoptime IS NULL OR p.stoptime >= l.dischtime)
         AS not_stopped_before_discharge
FROM project_aecopd.copd_coded_all_index_labels_v2 l
JOIN mimiciv_hosp.prescriptions p USING (hadm_id);

SELECT 'pharmacy' AS source, count(*) AS rows,
       count(DISTINCT p.hadm_id) AS admissions,
       count(*) FILTER (WHERE p.entertime <= l.dischtime) AS entered_by_discharge,
       count(*) FILTER (WHERE p.starttime >= l.dischtime - interval '24 hours'
                        AND p.starttime <= l.dischtime) AS started_last_24h
FROM project_aecopd.copd_coded_all_index_labels_v2 l
JOIN mimiciv_hosp.pharmacy p USING (hadm_id);

ROLLBACK;
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/section_alignment_data/validate_dataset.py · lines 1–181

<!-- REVIEW_FRAGMENT {"path":"training/v5/section_alignment_data/validate_dataset.py","part":1,"start_line":1,"end_line":181,"source_sha256":"ba8c03450bad1c9247bad65423a19b226ac0a86194a94a3b8e69164b0114e32b","fragment_sha256":"ba8c03450bad1c9247bad65423a19b226ac0a86194a94a3b8e69164b0114e32b","added_terminal_newline":false} -->
````python
"""独立复核 V5 章节数据的键、原文偏移、时间边界和来源哈希。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterator

from training.v5.section_alignment_data.build_dataset import patient_split


def _read(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(
    raw_dir: Path, note_source: Path, dataset_dir: Path, base_source: Path | None = None,
) -> dict[str, object]:
    export_summary = json.loads((raw_dir / "export_summary.json").read_text(encoding="utf-8"))
    build_summary = json.loads((dataset_dir / "qa_summary.json").read_text(encoding="utf-8"))
    if _sha256(note_source) != export_summary["source_note_gzip_sha256"]:
        raise ValueError("note source hash changed")
    if base_source is not None and _sha256(base_source) != build_summary["base_context_source_sha256"]:
        raise ValueError("base context source hash changed")
    for name, expected in export_summary["sources"].items():
        if _sha256(raw_dir / f"{name}.csv.gz") != expected["sha256"]:
            raise ValueError(f"raw source hash changed: {name}")

    cohort: dict[str, tuple[str, datetime, datetime, str]] = {}
    split_patients: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    for row in _read(dataset_dir / "cohort_manifest.csv.gz"):
        hadm, subject = row["hadm_id"], row["subject_id"]
        if hadm in cohort:
            raise ValueError("duplicate cohort admission")
        if row["patient_split"] != patient_split(int(subject)):
            raise ValueError("patient split drift")
        if subject in split_patients and split_patients[subject] != row["patient_split"]:
            raise ValueError("patient crosses splits")
        split_patients[subject] = row["patient_split"]
        admission = datetime.fromisoformat(row["admittime"])
        discharge = datetime.fromisoformat(row["dischtime"])
        if discharge <= admission:
            raise ValueError("invalid admission time span")
        cohort[hadm] = (subject, admission, discharge, row["patient_split"])
        split_counts[row["patient_split"]] += 1
    if len(cohort) != build_summary["cohort_admissions"]:
        raise ValueError("manifest count drift")

    section_ids: set[str] = set()
    sections_by_hadm: dict[str, list[dict[str, str]]] = {}
    role_status: Counter[str] = Counter()
    for section in _read(dataset_dir / "note_sections.csv.gz"):
        hadm, sid = section["hadm_id"], section["section_id"]
        if hadm not in cohort or sid in section_ids:
            raise ValueError("unknown admission or duplicate section ID")
        section_ids.add(sid)
        sections_by_hadm.setdefault(hadm, []).append(section)
        role_status[f"{section['section_role']}|{section['content_status']}"] += 1
    if len(section_ids) != build_summary["sections"]:
        raise ValueError("section count drift")
    note_count = 0
    for note in _read(note_source):
        hadm = note["hadm_id"]
        original = note["document_text"]
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
        for section in sections_by_hadm.pop(hadm, []):
            start, end = int(section["start_char"]), int(section["end_char"])
            if not 0 <= start <= end <= len(original):
                raise ValueError(f"out-of-bounds section offset: role={section['section_role']}, "
                                 f"start={start}, end={end}, note_length={len(original)}")
            if original[start:end] != section["text"] or section["source_sha256"] != digest:
                raise ValueError("section text or source hash mismatch")
            if section["note_id"] != note["note_id"]:
                raise ValueError("section note ID mismatch")
        note_count += 1
    if sections_by_hadm or note_count != build_summary["note_admissions"]:
        raise ValueError("unmatched section or note")

    fact_counts: Counter[str] = Counter()
    temporal_violations: Counter[str] = Counter()
    fact_count = 0
    current_source = ""
    source_row_keys: set[str] = set()
    for fact in _read(dataset_dir / "structured_facts.csv.gz"):
        fact_id, hadm = fact["fact_id"], fact["hadm_id"]
        source, number_text = fact_id.split(":", 1)
        if source != current_source:
            current_source = source
            source_row_keys.clear()
        if fact["source_row_key"] in source_row_keys:
            raise ValueError(f"duplicate source row key in {source}")
        source_row_keys.add(fact["source_row_key"])
        number = int(number_text)
        fact_counts[source] += 1
        if number != fact_counts[source]:
            raise ValueError("fact ID sequence gap or duplicate")
        if hadm not in cohort or fact["subject_id"] != cohort[hadm][0]:
            raise ValueError("fact cohort mismatch")
        if fact["concept_code"] in {"event_type", "time_days"}:
            raise ValueError("outcome label leaked into structured facts")
        if source in {"labs", "emar", "context"}:
            event = datetime.fromisoformat(fact["event_time"])
            available = datetime.fromisoformat(fact["available_time"])
            _, admission, discharge, _ = cohort[hadm]
            if not admission <= event <= discharge:
                temporal_violations[f"{source}_event_outside_stay"] += 1
            if available > discharge:
                temporal_violations[f"{source}_recorded_after_discharge"] += 1
        fact_count += 1
    if fact_count != build_summary["facts"] or dict(fact_counts) != build_summary["source_facts"]:
        raise ValueError("fact count drift")
    if temporal_violations:
        raise ValueError(f"temporal violations: {dict(temporal_violations)}")

    candidate_count = 0
    rule_counts: Counter[str] = Counter()
    for candidate in _read(dataset_dir / "candidate_alignment.csv.gz"):
        if candidate["section_id"] not in section_ids:
            raise ValueError("candidate references unknown section")
        source, number_text = candidate["fact_id"].split(":", 1)
        if source not in fact_counts or not 1 <= int(number_text) <= fact_counts[source]:
            raise ValueError("candidate references unknown fact")
        if candidate["relation_candidate"] not in {"topic_only", "topic_only_unverified_order_context"}:
            raise ValueError("candidate incorrectly states evidence")
        candidate_count += 1
        rule_counts[candidate["rule_id"]] += 1
    if candidate_count != build_summary["candidates"]:
        raise ValueError("candidate count drift")
    result: dict[str, object] = {
        "status": "structural_checks_passed", "cohort_admissions": len(cohort),
        "cohort_patients": len(split_patients), "note_admissions": note_count,
        "missing_note_admissions": len(cohort) - note_count,
        "sections": len(section_ids), "facts": fact_count, "candidates": candidate_count,
        "unmatched_facts": build_summary["unmatched_facts"],
        "patient_split_admissions": dict(split_counts),
        "candidate_rules": dict(rule_counts), "section_content_status": dict(role_status),
        "temporal_violations": dict(temporal_violations),
        "clinical_evidence_review_complete": False,
        "output_sha256": {
            name: _sha256(dataset_dir / name) for name in (
                "cohort_manifest.csv.gz", "note_sections.csv.gz",
                "structured_facts.csv.gz", "candidate_alignment.csv.gz",
            )
        },
    }
    (dataset_dir / "qa_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--note-source", type=Path, required=True)
    parser.add_argument("--base-source", type=Path)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.raw_dir, args.note_source, args.dataset_dir, args.base_source)
    print(json.dumps({k: v for k, v in result.items() if k not in {"section_content_status"}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/section_alignment_data/validate_section_inputs.py · lines 1–196

<!-- REVIEW_FRAGMENT {"path":"training/v5/section_alignment_data/validate_section_inputs.py","part":1,"start_line":1,"end_line":196,"source_sha256":"c07d9a71a201def802dcb6237b31c5967b2a3b2573cdbeed0f6130a37b286561","fragment_sha256":"c07d9a71a201def802dcb6237b31c5967b2a3b2573cdbeed0f6130a37b286561","added_terminal_newline":false} -->
````python
"""独立校验章节级输入的角色、键、掩码及旧检验／给药摘要一致性。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterator

from training.v5.clinical_concepts import LABS, medication_classes
from training.v5.section_alignment_data.build_section_inputs import SECTION_ROLES


def _rows(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _days(after: str, before: str) -> Decimal:
    seconds = (datetime.fromisoformat(after) - datetime.fromisoformat(before)).total_seconds()
    return Decimal(str(seconds)) / Decimal(86400)


def validate(
    aligned_dir: Path, inputs_dir: Path, old_dir: Path,
) -> dict[str, object]:
    schema = json.loads((inputs_dir / "schema.json").read_text(encoding="utf-8"))
    expected = json.loads((inputs_dir / "qa_summary.json").read_text(encoding="utf-8"))
    if tuple(schema["roles"]) != SECTION_ROLES or not schema["no_outcome_labels_in_outputs"]:
        raise ValueError("section input schema drift")
    if _sha256(aligned_dir / "qa_validation.json") != schema["source_sha256"]["aligned_qa_validation"]:
        raise ValueError("validated source changed")
    if _sha256(old_dir / "base.csv") != schema["source_sha256"]["v5_base"]:
        raise ValueError("base source changed")

    cohort = {row["hadm_id"]: row for row in _rows(aligned_dir / "cohort_manifest.csv.gz")}
    base = {row["hadm_id"]: row for row in _rows(old_dir / "base.csv")}
    note_sections = {row["section_id"]: (row["hadm_id"], row["section_role"], row["content_status"])
                     for row in _rows(aligned_dir / "note_sections.csv.gz")}
    note_hadms = {hadm for hadm, _, _ in note_sections.values()}
    slot_seen: set[tuple[str, str]] = set()
    text_counts: Counter[tuple[str, str]] = Counter()
    slot_count = 0
    for row in _rows(inputs_dir / "section_slots.csv.gz"):
        hadm, role = row["hadm_id"], row["section_role"]
        key = hadm, role
        if hadm not in cohort or row["subject_id"] != cohort[hadm]["subject_id"]:
            raise ValueError("slot outside cohort")
        if role not in SECTION_ROLES or key in slot_seen:
            raise ValueError("unknown or repeated section slot")
        slot_seen.add(key)
        ids = json.loads(row["text_section_ids"])
        if len(ids) != len(set(ids)):
            raise ValueError("repeated text section ID in slot")
        for sid in ids:
            if note_sections.get(sid, (None, None, None))[:2] != key:
                raise ValueError("slot references unrelated text section")
            text_counts[key] += int(note_sections[sid][2] == "informative")
        if int(row["text_nonblank_count"]) != text_counts[key]:
            raise ValueError("text count mismatch")
        if int(row["has_discharge_note"]) != int(hadm in note_hadms):
            raise ValueError("note presence mismatch")
        slot_count += 1
    if slot_count != len(cohort) * len(SECTION_ROLES):
        raise ValueError("slot count mismatch")

    old_labs = {(row["hadm_id"], LABS[int(row["itemid"])][0]): row
                for row in _rows(old_dir / "labs.csv")}
    old_medications: dict[tuple[str, str], tuple[str, str]] = {}
    for row in _rows(old_dir / "medications.csv"):
        for category in medication_classes(row["medication"]):
            key = row["hadm_id"], category
            if key in old_medications:
                first, last = old_medications[key]
                old_medications[key] = (min(first, row["first_charttime"]),
                                        max(last, row["last_charttime"]))
            else:
                old_medications[key] = row["first_charttime"], row["last_charttime"]

    token_seen: set[tuple[str, str, str, str]] = set()
    by_slot: Counter[tuple[str, str]] = Counter()
    types: Counter[str] = Counter()
    labs_seen = 0
    meds_seen = 0
    first_by_patient: dict[str, tuple[str, str]] = {}
    for hadm, row in cohort.items():
        subject = row["subject_id"]
        candidate = row["admittime"], hadm
        if subject not in first_by_patient or candidate < first_by_patient[subject]:
            first_by_patient[subject] = candidate
    for row in _rows(inputs_dir / "structured_tokens.csv.gz"):
        hadm, role, kind, concept = (row[name] for name in
                                     ("hadm_id", "section_role", "token_type", "concept"))
        key = hadm, role, kind, concept
        if key in token_seen or (hadm, role) not in slot_seen:
            raise ValueError("duplicate token or unknown slot")
        token_seen.add(key)
        if row["subject_id"] != cohort[hadm]["subject_id"]:
            raise ValueError("token subject mismatch")
        if concept in {"event_type", "time_days"}:
            raise ValueError("outcome leakage")
        values = [float(row[f"value{index}"]) for index in range(5)]
        masks = [int(row[f"mask{index}"]) for index in range(5)]
        if not all(math.isfinite(value) for value in values) or not set(masks) <= {0, 1}:
            raise ValueError("invalid value or mask")
        if any(not mask and value != 0 for value, mask in zip(values, masks)):
            raise ValueError("nonzero masked value")
        by_slot[(hadm, role)] += 1
        types[kind] += 1
        if kind == "laboratory":
            old = old_labs.pop((hadm, concept), None)
            if old is None or role != "pertinent_results":
                raise ValueError("unmatched laboratory token")
            old_values = [Decimal(old[name]) for name in
                          ("first_value", "last_value", "min_value", "max_value")]
            if any(abs(Decimal(str(new)) - original) > Decimal("0.000001")
                   for new, original in zip(values[:4], old_values)):
                raise ValueError("laboratory summary mismatch")
            if masks[4] != int(int(old["measurement_count"]) >= 2):
                raise ValueError("laboratory trend mask mismatch")
            if masks[4] and abs(Decimal(str(values[4])) - (old_values[1] - old_values[0])) > Decimal("0.000001"):
                raise ValueError("laboratory delta mismatch")
            labs_seen += 1
        elif kind == "administered_medication":
            times = old_medications.pop((hadm, concept), None)
            if times is None or role != "hospital_course":
                raise ValueError("unmatched medication token")
            admission = base[hadm]["admittime"]
            expected_days = (Decimal(1), _days(times[0], admission),
                             _days(times[1], admission), _days(times[1], times[0]))
            if any(abs(Decimal(str(new)) - target) > Decimal("0.000001")
                   for new, target in zip(values[:4], expected_days)):
                raise ValueError("medication span mismatch")
            meds_seen += 1
        elif kind == "chronic_comorbidity":
            if role != "past_medical_history":
                raise ValueError("chronic comorbidity misplaced")
            if values[0] == 0 and values[1] == 0:
                raise ValueError("uncoded chronic category became a token")
            subject = cohort[hadm]["subject_id"]
            if (cohort[hadm]["admittime"], hadm) == first_by_patient[subject] and values[1] != 0:
                raise ValueError("first eligible admission has fabricated prior history")
        elif kind == "acute_diagnosis":
            if role != "discharge_diagnosis" or values[0] != 1:
                raise ValueError("uncoded or misplaced acute diagnosis")
        elif kind == "disposition" and role != "discharge_disposition":
            raise ValueError("disposition misplaced")
    if old_labs or old_medications:
        raise ValueError("missing lab or medication tokens")
    for row in _rows(inputs_dir / "section_slots.csv.gz"):
        if int(row["structured_token_count"]) != by_slot[(row["hadm_id"], row["section_role"])]:
            raise ValueError("slot structured-token count mismatch")
    if len(token_seen) != expected["structured_tokens"] or dict(types) != expected["token_types"]:
        raise ValueError("token count drift")
    result: dict[str, object] = {
        "status": "section_inputs_validated", "admissions": len(cohort),
        "slots": slot_count, "tokens": len(token_seen),
        "laboratory_tokens_reconciled": labs_seen,
        "medication_tokens_reconciled": meds_seen,
        "text_sections_referenced": sum(len(json.loads(row["text_section_ids"]))
                                        for row in _rows(inputs_dir / "section_slots.csv.gz")),
        "output_sha256": {name: _sha256(inputs_dir / name) for name in
                          ("structured_tokens.csv.gz", "section_slots.csv.gz", "schema.json")},
        "clinical_content_validated": False,
    }
    (inputs_dir / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--old-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.aligned_dir, args.inputs_dir, args.old_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/section_alignment_data/validate_section_text.py · lines 1–96

<!-- REVIEW_FRAGMENT {"path":"training/v5/section_alignment_data/validate_section_text.py","part":1,"start_line":1,"end_line":96,"source_sha256":"b283dfb869d5d1060a007c7db99eb293def77b88e5c5a10c9bcb034666f22f3c","fragment_sha256":"b283dfb869d5d1060a007c7db99eb293def77b88e5c5a10c9bcb034666f22f3c","added_terminal_newline":false} -->
````python
"""回查章节编码清单的原文位置、来源哈希和同队列角色。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterator


def _rows(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(aligned_dir: Path, manifest_dir: Path) -> dict[str, object]:
    summary = json.loads((manifest_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = manifest_dir / "shared_text.csv.gz"
    if _sha256(manifest) != summary["manifest_sha256"]:
        raise ValueError("text manifest hash changed")
    cohort = {row["hadm_id"]: row["subject_id"]
              for row in _rows(aligned_dir / "cohort_manifest.csv.gz")}
    roles = set(summary["roles"])
    sections = {row["section_id"]: row for row in _rows(aligned_dir / "note_sections.csv.gz")
                if row["section_role"] in roles and row["content_status"] == "informative"
                and row["parse_status"] != "uncertain"}
    counts: Counter[str] = Counter()
    chunk_indices: dict[str, int] = {}
    source_hadms: set[str] = set()
    for number, row in enumerate(_rows(manifest)):
        if int(row["embedding_row"]) != number:
            raise ValueError("non-contiguous embedding row")
        sid = row["section_id"]
        original = sections.get(sid)
        if original is None:
            raise ValueError("manifest references ineligible section")
        if row["hadm_id"] != original["hadm_id"] or row["section_role"] != original["section_role"]:
            raise ValueError("section key or role mismatch")
        if row["subject_id"] != cohort[row["hadm_id"]]:
            raise ValueError("subject mismatch")
        if row["timestamp_relation"] != original["timestamp_relation"] or row["source_section_sha256"] != original["source_sha256"]:
            raise ValueError("source metadata mismatch")
        start, end = int(row["source_start_char"]), int(row["source_end_char"])
        source_start, source_end = int(original["start_char"]), int(original["end_char"])
        if not source_start <= start < end <= source_end:
            raise ValueError("chunk offset outside section")
        if original["text"][start - source_start:end - source_start] != row["chunk_text"]:
            raise ValueError("chunk text cannot be found at original offset")
        if hashlib.sha256(row["chunk_text"].encode("utf-8")).hexdigest() != row["text_sha256"]:
            raise ValueError("chunk text hash mismatch")
        if int(row["chunk_index"]) != chunk_indices.get(sid, 0):
            raise ValueError("non-contiguous chunk index within section")
        chunk_indices[sid] = int(row["chunk_index"]) + 1
        if not 1 <= int(row["bert_token_count"]) <= 512 or not 1 <= int(row["token_count"]) <= 8192:
            raise ValueError("encoder token count outside limit")
        counts[row["section_role"]] += 1
        source_hadms.add(row["hadm_id"])
    if sum(counts.values()) != summary["chunks"]:
        raise ValueError("manifest row count drift")
    if len(chunk_indices) != summary["source_sections_included"]:
        raise ValueError("section count drift")
    if len(source_hadms) != summary["admissions_with_text"]:
        raise ValueError("admission count drift")
    result: dict[str, object] = {
        "status": "manifest_structurally_validated", "chunks": sum(counts.values()),
        "source_sections": len(chunk_indices), "admissions": len(source_hadms),
        "chunks_by_role": dict(counts), "manifest_sha256": summary["manifest_sha256"],
        "clinical_semantics_review_complete": False,
        "encoding_started": False,
    }
    (manifest_dir / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.aligned_dir, args.manifest_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/strict_endpoint_labels.py · lines 1–135

<!-- REVIEW_FRAGMENT {"path":"training/v5/strict_endpoint_labels.py","part":1,"start_line":1,"end_line":135,"source_sha256":"d86b594f9a098fa5a51eb7102f2d4b8910a1325b77c03abdaf9c48d053f1000d","fragment_sha256":"d86b594f9a098fa5a51eb7102f2d4b8910a1325b77c03abdaf9c48d053f1000d","added_terminal_newline":false} -->
````python
"""由冻结原始底账生成严格结局，保持当前并集实验的转院、死亡和索引规则。"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

STRICT_PRIMARY_EVENT = "first non-transfer readmission with strict 491.21/J44.1 at any diagnosis position; current all-coded transfer rule"
SOURCES = ("cohort/admissions", "raw/patients", "raw/diagnoses_icd",
           "outcomes/readmission_death_labels")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def diagnosis_key(record: dict[str, str]) -> tuple[int, str]:
    """PostgreSQL 定长 ICD 字段含尾随空格，不把空格当成代码的一部分。"""
    return int(record["icd_version"]), record["icd_code"].strip().replace(".", "").upper()


def reconstruct(cohort: list[dict[str, str]], patients: dict[str, dict[str, str]], *,
                strict_only: bool) -> dict[str, dict[str, object]]:
    """首次符合结局的非续接住院；其他住院不结束观察，死亡为竞争事件。"""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in cohort:
        groups[record["subject_id"]].append(record)
    output: dict[str, dict[str, object]] = {}
    for subject, records in groups.items():
        records.sort(key=lambda r: (r["admittime"], int(r["hadm_id"])))
        candidates = []
        previous_end = None
        for record in records:
            start, end = datetime.fromisoformat(record["admittime"]), datetime.fromisoformat(record["dischtime"])
            continuation = (record["admission_location"] == "TRANSFER FROM HOSPITAL"
                            and previous_end is not None and previous_end <= start <= previous_end + timedelta(days=2))
            qualifying = record["is_strict_aecopd"] == "t" or (not strict_only and int(record["best_copd_seq_num"]) <= 3)
            if qualifying and not continuation:
                candidates.append((start, int(record["hadm_id"]), record))
            # 不因本次结局定义而改变转院参照住院集合。
            previous_end = end
        dod_text = patients[subject]["dod"]
        dod = None if dod_text in {"", "\\N"} else datetime.fromisoformat(dod_text).date()
        for record in records:
            if record["discharge_landmark_eligible"] != "t":
                continue
            start, end = datetime.fromisoformat(record["admittime"]), datetime.fromisoformat(record["dischtime"])
            next_event = next((item for item in candidates
                               if (item[0], item[1]) > (start, int(record["hadm_id"])) and item[0] >= end), None)
            next_days = None if next_event is None else (next_event[0] - end).total_seconds() / 86400
            death_days = None if dod is None or dod < end.date() else float((dod - end.date()).days)
            if next_days is not None and next_days <= 365 and (death_days is None or next_days <= death_days):
                event, time = 1, next_days
            elif death_days is not None and death_days <= 365 and (next_days is None or death_days < next_days):
                event, time = 2, death_days
            else:
                event, time = 0, 365.
            output[record["hadm_id"]] = {"subject_id": subject, "hadm_id": record["hadm_id"],
                "admittime": record["admittime"], "dischtime": record["dischtime"],
                "event_type": event, "time_days": time,
                "next_copd_hadm_id": "" if next_event is None else next_event[1]}
    return output


def checked_strict_labels(raw_dir: Path) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    manifest = json.loads((raw_dir / "metadata/manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "validated":
        raise ValueError("raw ledger not validated")
    hashes = {name: sha(raw_dir / (name + ".csv.gz")) for name in SOURCES}
    if any(hashes[name] != manifest["sources"][name]["sha256"] for name in SOURCES):
        raise ValueError("raw endpoint source changed")
    cohort = rows(raw_dir / "cohort/admissions.csv.gz")
    patients = {r["subject_id"]: r for r in rows(raw_dir / "raw/patients.csv.gz")}
    strict_ids = {r["hadm_id"] for r in rows(raw_dir / "raw/diagnoses_icd.csv.gz")
                  if diagnosis_key(r) in {(9, "49121"), (10, "J441")}}
    if any((r["is_strict_aecopd"] == "t") != (r["hadm_id"] in strict_ids) for r in cohort):
        raise ValueError("strict flags disagree with raw diagnosis codes")
    broad = reconstruct(cohort, patients, strict_only=False)
    reference = {r["hadm_id"]: r for r in rows(raw_dir / "outcomes/readmission_death_labels.csv.gz")}
    if set(broad) != set(reference) or len(broad) != 19401:
        raise ValueError("reconstructed endpoint cohort differs")
    for hadm, label in broad.items():
        old = reference[hadm]
        if int(old["event_type"]) != label["event_type"] or not np.isclose(float(old["time_days"]), label["time_days"], atol=1e-8, rtol=0):
            raise ValueError("union label reconstruction differs from frozen ledger")
    strict = reconstruct(cohort, patients, strict_only=True)
    if any(strict[h]["event_type"] == 1 and
           (str(strict[h]["next_copd_hadm_id"]) not in strict_ids or
            broad[h]["event_type"] != 1 or float(strict[h]["time_days"]) < float(broad[h]["time_days"])) for h in strict):
        raise ValueError("strict event not a valid subset/later occurrence")
    return strict, hashes


def export(raw_dir: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    strict, hashes = checked_strict_labels(raw_dir)
    output_dir.mkdir(parents=True)
    target = output_dir / "labels.csv.gz"
    with gzip.open(target, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(strict.values()))))
        writer.writeheader()
        writer.writerows(strict.values())
    original = {r["hadm_id"]: r for r in rows(raw_dir / "outcomes/readmission_death_labels.csv.gz")}
    transitions = Counter(f"{original[h]['event_type']}->{label['event_type']}" for h, label in strict.items())
    summary = {"status": "strict_labels_reconstructed_and_checked", "primary_event": STRICT_PRIMARY_EVENT,
               "cohort_admissions": len(strict), "cohort_patients": len({r["subject_id"] for r in strict.values()}),
               "events": dict(Counter(str(r["event_type"]) for r in strict.values())),
               "event_type_transitions_from_union": dict(transitions), "label_sha256": sha(target),
               "source_sha256": hashes, "union_reconstruction_identical": True,
               "strict_flags_checked_against_diagnoses": True, "individual_followup_complete": False,
               "transfer_rule": "all coded admissions; hospital transfer within two days of preceding coded discharge",
               "code_position": "any diagnosis position"}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.raw_dir, args.output_dir)))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/summarize_abnormality_experiment.py · lines 1–67

<!-- REVIEW_FRAGMENT {"path":"training/v5/summarize_abnormality_experiment.py","part":1,"start_line":1,"end_line":67,"source_sha256":"c853a4b814ae16fba0de12a8d14e72d65bb36f24285f6f1f7d0e0de1dbcf9101","fragment_sha256":"c853a4b814ae16fba0de12a8d14e72d65bb36f24285f6f1f7d0e0de1dbcf9101","added_terminal_newline":false} -->
````python
"""汇总异常分支多种子配对试验，不把重复查看的历史测试集用于模型选择。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(root: Path, output: Path, seeds: tuple[int, ...]) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("invalid seed list")
    rows = []
    reference: dict[str, object] | None = None
    for seed in seeds:
        reports = {
            mode: json.loads((root / f"{mode}_seed{seed}" / "report.json").read_text(encoding="utf-8"))
            for mode in ("masked", "core9", "measurement_only", "active")
        }
        control = reports["masked"]
        if reference is None:
            reference = control
        else:
            for key in ("cohort_patients", "cohort_admissions", "label_sha256",
                        "field_manifest_sha256", "raw_source_sha256", "parameter_count",
                        "text_vector_width", "event_encoder", "clinical_recovery_manifest_sha256"):
                if control[key] != reference[key]:
                    raise ValueError(f"{key} differs between seeds")
        for mode, report in reports.items():
            for key in ("cohort_patients", "cohort_admissions", "label_sha256",
                        "field_manifest_sha256", "raw_source_sha256", "parameter_count",
                        "text_vector_width", "event_encoder", "clinical_recovery_manifest_sha256"):
                if report[key] != control[key]:
                    raise ValueError(f"{key} differs for {mode}, seed {seed}")
            if (report["status"] != "training_completed" or report["seed"] != seed
                    or report["event_encoder"] != "abnormal" or report["raw_events_enter_model"]
                    or report["clinical_recovery_mode"] != mode):
                raise ValueError(f"invalid experiment arm: {mode}, seed {seed}")
            readmission = report["validation"]["temporal_full"]
            death = report["validation"]["death_temporal_full"]
            rows.append({"seed": seed, "mode": mode, "best_epoch": report["best_epoch"],
                         "readmission_ibs": readmission["integrated_brier_7_365"],
                         "readmission_auc": readmission["mean_auc_7_365"],
                         "death_ibs": death["integrated_brier_7_365"],
                         "death_auc": death["mean_auc_7_365"]})
    result = {"status": "exploratory_multi_seed", "seeds": seeds, "arms": len(rows),
              "cohort_patients": control["cohort_patients"],
              "cohort_admissions": control["cohort_admissions"],
              "parameter_count_each_arm": control["parameter_count"],
              "validation_only_for_design": True,
              "notes": ["All four arms share architecture, source, patient split, labels and seed within comparison.",
                        "A 200-draw patient-cluster interval is reported separately for each seed.",
                        "The historical test split is not blind and is not used to select axes."],
              "validation_rows": rows}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    args = parser.parse_args()
    print(json.dumps({"status": summarize(args.run_root, args.output, tuple(args.seeds))["status"]}))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/summarize_medication_effects.py · lines 1–71

<!-- REVIEW_FRAGMENT {"path":"training/v5/summarize_medication_effects.py","part":1,"start_line":1,"end_line":71,"source_sha256":"c3bf5b78764372dd61a7db9d5cb38359f24a344a5b761dd9114f4167ec11f48f","fragment_sha256":"c3bf5b78764372dd61a7db9d5cb38359f24a344a5b761dd9114f4167ec11f48f","added_terminal_newline":false} -->
````python
"""汇总药理组配对实验；不按单项指标自动筛选变量。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.v5.medication_effect_data import EFFECT_GROUPS


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "completed":
        raise ValueError(f"incomplete comparison: {path}")
    return value


def summarize(root: Path, output: Path, *, seeds: tuple[int, ...]) -> dict:
    if output.exists():
        raise FileExistsError(output)
    records = {}
    for seed in seeds:
        comparisons = {}
        for control in ("masked", "exposure_only"):
            path = root / f"comparison_{control}_seed{seed}.json"
            if path.is_file():
                comparisons[control] = _read(path)
        if "masked" not in comparisons:
            raise ValueError(f"missing whole-group comparison for seed {seed}")
        records[str(seed)] = {}
        for control, report in comparisons.items():
            if int(report["seed"]) != seed or report["control"] != control:
                raise ValueError("comparison identity mismatch")
            records[str(seed)][control] = {
                split: {cause: report["splits"][split][cause]["delta_active_minus_control"]
                        for cause in ("readmission", "death")}
                for split in ("validation", "historical_test")}
    group_ablations = {}
    for group in EFFECT_GROUPS:
        path = root / f"comparison_without_{group}_seed{seeds[0]}.json"
        if path.is_file():
            comparison = _read(path)
            group_ablations[group] = {
                "seed": seeds[0],
                "validation_readmission": comparison["splits"]["validation"]["readmission"]["delta_active_minus_control"],
                "historical_test_readmission": comparison["splits"]["historical_test"]["readmission"]["delta_active_minus_control"],
                "validation_paired_bootstrap": comparison["splits"]["validation"]["readmission"]["paired_bootstrap"],
            }
    result = {"status": "descriptive_not_final_selection", "seeds": records,
              "single_seed_group_ablations": group_ablations,
              "interpretation_boundaries": [
                  "Group ablations have one training seed; they cannot establish individual group value.",
                  "No group is automatically retained or removed from a single AUC or Brier difference.",
                  "Patient-level bootstrap intervals do not include training-seed uncertainty.",
                  "Historical test was examined before and is not a fresh blind test.",
                  "Drug exposure is confounded by clinical indication and eMAR documentation availability.",
              ]}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    args = parser.parse_args()
    summary = summarize(args.run_root, args.output, seeds=tuple(args.seeds))
    print(json.dumps({"status": summary["status"], "seeds": list(summary["seeds"]),
                      "groups": len(summary["single_seed_group_ablations"])}))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/summarize_section_incremental.py · lines 1–71

<!-- REVIEW_FRAGMENT {"path":"training/v5/summarize_section_incremental.py","part":1,"start_line":1,"end_line":71,"source_sha256":"00d132ed66cfd23ff1f5ff48dfd333da67e81efe2dde97608d185ad7c0c2efce","fragment_sha256":"00d132ed66cfd23ff1f5ff48dfd333da67e81efe2dde97608d185ad7c0c2efce","added_terminal_newline":false} -->
````python
"""汇总同队列新增章节消融的配对患者区间与验证/历史测试指标。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.v5.compare_strict_ablations import run as paired_compare
from training.v5.extend_field_manifest import EXTRA_LABELS
from training.v5.field_text_baseline import FIELD_LABELS


def summarize(root: Path) -> dict[str, object]:
    baseline_dir = root / "baseline_four_fields"
    baseline = json.loads((baseline_dir / "report.json").read_text(encoding="utf-8"))
    if baseline["active_text_fields"] != list(FIELD_LABELS):
        raise ValueError("four-field baseline contract mismatch")
    arms = ["plus_" + role for role in EXTRA_LABELS] + ["plus_all_five"]
    result = {"status": "same_cohort_section_incremental_comparison", "baseline": "baseline_four_fields",
              "arms": {}, "limitations": [
                  "One training seed; bootstrap intervals do not measure initialization variability.",
                  "Five single-section tests plus a combined test are exploratory and not multiplicity-corrected.",
                  "Historical test has been repeatedly inspected and is not blind.",
                  "Raw section embeddings may duplicate structured data or dilute other snippets.",
              ]}
    for name in arms:
        arm_dir = root / name
        report = json.loads((arm_dir / "report.json").read_text(encoding="utf-8"))
        expected = [*FIELD_LABELS, *EXTRA_LABELS] if name == "plus_all_five" else [*FIELD_LABELS, name.removeprefix("plus_")]
        if report["active_text_fields"] != expected or report["field_roles"] != baseline["field_roles"]:
            raise ValueError("arm field selection or model vocabulary differs: " + name)
        comparison_path = arm_dir / "paired_vs_baseline.json"
        if comparison_path.exists():
            paired = json.loads(comparison_path.read_text(encoding="utf-8"))
        else:
            paired = paired_compare(arm_dir, {"baseline": baseline_dir}, comparison_path)
        split_results = {}
        for split in ("validation", "historical_test"):
            full = report[split]["temporal_full"]
            base = baseline[split]["temporal_full"]
            paired_split = paired["comparisons"]["baseline"]["splits"][split]
            split_results[split] = {
                "mean_auc": full["mean_auc_7_365"],
                "mean_auc_delta": full["mean_auc_7_365"] - base["mean_auc_7_365"],
                "integrated_brier": full["integrated_brier_7_365"],
                "integrated_brier_delta": full["integrated_brier_7_365"] - base["integrated_brier_7_365"],
                "brier_paired_interval": paired_split["ibs_v5_minus_comparator_patient_cluster_95pct_interval"],
                "auc_30_90_365_paired": paired_split["auc_v5_minus_comparator"],
            }
        result["arms"][name] = {"active_text_fields": expected, "best_epoch": report["best_epoch"],
                                **split_results}
    target = root / "summary.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run_dir)
    print(json.dumps({name: {split: {key: values[key] for key in ("mean_auc_delta", "integrated_brier_delta")}
                              for split, values in arm.items() if split in ("validation", "historical_test")}
                      for name, arm in result["arms"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/text_content_permutation.py · lines 1–233

<!-- REVIEW_FRAGMENT {"path":"training/v5/text_content_permutation.py","part":1,"start_line":1,"end_line":233,"source_sha256":"b4acb4df54bcaddbb64fb4216a75f602c10ac4f623e929335076f86ee34792b9","fragment_sha256":"b4acb4df54bcaddbb64fb4216a75f602c10ac4f623e929335076f86ee34792b9","added_terminal_newline":false} -->
````python
"""冻结 V5 检查点，跨患者打乱同领域片段，检验文本内容的增量信息。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from training.v5.data import FIELD_NAMES, V5Data, load_data
from training.v5.model import V5Model
from training.v5.train import GRID, metrics, predict, preprocess, split_sequences


def _manifest_hashes(path: Path) -> tuple[str, ...]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(int(row["embedding_row"]) != index for index, row in enumerate(rows)):
        raise ValueError("text manifest row order changed")
    return tuple(row["text_sha256"] for row in rows)


def permute_same_field_across_patients(
    data: V5Data, rows: np.ndarray, hashes: tuple[str, ...], *, seed: int,
) -> tuple[V5Data, dict[str, object]]:
    """只替换向量行号；保留每次出院的领域、片段数和缺失模式。"""

    rng = np.random.default_rng(seed)
    changed = [list(items) for items in data.snippets]
    field_stats: dict[str, dict[str, int]] = {}
    for field, name in enumerate(FIELD_NAMES):
        slots = [(int(row), position, vector_row, int(data.subject_ids[row]))
                 for row in rows for position, (item_field, vector_row)
                 in enumerate(data.snippets[int(row)]) if item_field == field]
        if not slots:
            field_stats[name] = {"snippets": 0, "different_text": 0}
            continue
        subjects = sorted({slot[3] for slot in slots})
        if len(subjects) < 2:
            raise ValueError(f"{name} has fewer than two patients in split")
        order = {subject: index for index, subject in enumerate(rng.permutation(subjects))}
        slots.sort(key=lambda slot: (order[slot[3]], slot[0], slot[1]))
        largest = max(Counter(slot[3] for slot in slots).values())
        if len(slots) >= 2 * largest:
            # 同领域向量做双射轮转，既保持词条频次，又禁止本患者给自己供片段。
            offset = int(rng.integers(largest, len(slots) - largest + 1))
            donors = [slots[(index + offset) % len(slots)] for index in range(len(slots))]
        else:
            # 极端不平衡领域无法同时满足双射和跨患者，显式退化为跨患者有放回抽样。
            donors = []
            for slot in slots:
                other = [candidate for candidate in slots if candidate[3] != slot[3]]
                donors.append(other[int(rng.integers(len(other)))])
        different = 0
        for recipient, donor in zip(slots, donors, strict=True):
            row, position, vector_row, subject = recipient
            if subject == donor[3]:
                raise AssertionError("patient received their own text")
            changed[row][position] = (field, donor[2])
            different += int(hashes[vector_row] != hashes[donor[2]])
        field_stats[name] = {"snippets": len(slots), "different_text": different}
    if any(len(original) != len(shuffled) or
           any(left[0] != right[0] for left, right in zip(original, shuffled, strict=True))
           for original, shuffled in zip(data.snippets, changed, strict=True)):
        raise AssertionError("text permutation changed field or missingness")
    return replace(data, snippets=tuple(tuple(items) for items in changed)), {"fields": field_stats}


def _integrated_error_by_stay(times: np.ndarray, events: np.ndarray,
                              risk: np.ndarray) -> np.ndarray:
    observed = ((events[:, None] == 1) & (times[:, None] <= GRID[None, :])).astype(float)
    return np.trapezoid((risk[:, 7:] - observed[:, 7:]) ** 2,
                        GRID[7:], axis=1) / (365 - 7)


def _patient_interval(subject_ids: np.ndarray, difference: np.ndarray,
                      *, seed: int) -> list[float]:
    patients, inverse = np.unique(subject_ids, return_inverse=True)
    totals = np.bincount(inverse, weights=difference)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(patients), size=(1000, len(patients)))
    means = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return np.quantile(means, (0.025, 0.975)).tolist()


def _load_model(run_dir: Path, model_name: str, text_width: int,
                *, include_text: bool, device: torch.device) -> V5Model:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    model = V5Model(text_width=text_width if include_text else 768,
                    balance_text=bool(report.get("balanced_text_content", False))
                    if include_text else False).to(device)
    model.load_state_dict(torch.load(run_dir / f"{model_name}.pt",
                                     map_location=device, weights_only=True))
    return model


def run(root: Path, output_dir: Path, *, permutations: int, seed: int) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if permutations < 2:
        raise ValueError("at least two permutations required")
    if not torch.cuda.is_available():
        raise RuntimeError("V5 text permutation requires CUDA")
    manifest = root / "text_manifest/shared_text.csv.gz"
    hashes = _manifest_hashes(manifest)
    configurations = (
        ("bert", root / "bert/bioclinicalbert.float16.npy",
         root / "training_bert_aligned_run2", "structured_bert"),
        ("qwen2560", root / "qwen/qwen3_embedding.float16.npy",
         root / "training_qwen2560_aligned_run2", "structured_qwen2560"),
        ("qwen2560_balanced", root / "qwen/qwen3_embedding.float16.npy",
         root / "training_qwen2560_balanced_run3", "structured_qwen2560"),
    )
    device = torch.device("cuda")
    summary: dict[str, object] = {
        "question": "Does the frozen text branch depend on patient-specific snippet content?",
        "design": "Inference-time within-field cross-patient permutation; fields, snippet counts and missingness fixed",
        "cohort": "MIMIC-IV V5 full discharge prediction sample set",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "permutations": permutations, "seed": seed,
        "evaluation": "World A 7-365-day competing-risk integrated Brier; patient bootstrap 95% CI",
        "status": "exploratory; text extraction clinically noisy; validation used for checkpoint selection; historical test reused",
        "splits": {},
    }
    for label, vectors, run_dir, model_name in configurations:
        data, _ = preprocess(load_data(root / "structured", manifest, vectors))
        if len(hashes) != len(data.text_vectors):
            raise ValueError("manifest/vector count mismatch")
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        if report["coverage"]["admissions"] != len(data.hadm_ids):
            raise ValueError("checkpoint cohort mismatch")
        model = _load_model(run_dir, model_name, data.text_width,
                            include_text=True, device=device)
        for split in ("validation", "test"):
            sequences = split_sequences(data, split)
            selected = np.concatenate(sequences)
            key = "historical_test" if split == "test" else split
            times, events, subjects = (data.times[selected], data.events[selected],
                                       data.subject_ids[selected])
            actual_risk, _ = predict(model, data, sequences, include_text=True,
                                     device=device, batch_size=24)
            actual = metrics(times, events, actual_risk)
            expected = report[key][model_name]["integrated_brier_7_365"]
            if not np.isclose(actual["integrated_brier_7_365"], expected, atol=1e-6):
                raise ValueError(f"{label}/{key} checkpoint score did not reproduce")
            actual_error = _integrated_error_by_stay(times, events, actual_risk)
            shuffled_error = np.zeros_like(actual_error)
            null_brier: list[float] = []
            null_auc90: list[float] = []
            changed: list[float] = []
            for repetition in range(permutations):
                permuted, audit = permute_same_field_across_patients(
                    data, selected, hashes, seed=seed + repetition,
                )
                risk, _ = predict(model, permuted, sequences, include_text=True,
                                  device=device, batch_size=24)
                measured = metrics(times, events, risk)
                null_brier.append(float(measured["integrated_brier_7_365"]))
                null_auc90.append(float(measured["points"]["90"]["auc"]))
                shuffled_error += _integrated_error_by_stay(times, events, risk)
                total = sum(value["snippets"] for value in audit["fields"].values())
                different = sum(value["different_text"] for value in audit["fields"].values())
                changed.append(different / total)
                print(json.dumps({"model": label, "split": key,
                                  "permutation": repetition + 1,
                                  "integrated_brier": null_brier[-1]}), flush=True)
            shuffled_error /= permutations
            difference = actual_error - shuffled_error
            summary["splits"].setdefault(key, {})[label] = {
                "admissions": len(selected), "patients": len(sequences),
                "actual_integrated_brier": actual["integrated_brier_7_365"],
                "actual_mean_auc": actual["mean_auc_7_365"],
                "actual_auc_90": actual["points"]["90"]["auc"],
                "shuffled_integrated_brier_mean": float(np.mean(null_brier)),
                "shuffled_integrated_brier_range": [float(min(null_brier)), float(max(null_brier))],
                "shuffled_auc_90_mean": float(np.mean(null_auc90)),
                "actual_minus_shuffled_integrated_brier": float(difference.mean()),
                "actual_minus_shuffled_95ci_patient_cluster": _patient_interval(
                    subjects, difference, seed=seed,
                ),
                "fraction_with_different_text_mean": float(np.mean(changed)),
                "null_integrated_brier": null_brier,
            }
    # 结构化对照与编码器维度无关；只读同一 run 的冻结基线，且复核历史报告。
    baseline_run = root / "training_qwen2560_aligned_run2"
    baseline_report = json.loads((baseline_run / "report.json").read_text(encoding="utf-8"))
    data, _ = preprocess(load_data(root / "structured"))
    baseline_model = _load_model(baseline_run, "structured", 768,
                                 include_text=False, device=device)
    for split in ("validation", "test"):
        sequences = split_sequences(data, split)
        selected = np.concatenate(sequences)
        key = "historical_test" if split == "test" else split
        risk, _ = predict(baseline_model, data, sequences, include_text=False,
                          device=device, batch_size=24)
        measured = metrics(data.times[selected], data.events[selected], risk)
        expected = baseline_report[key]["structured"]["integrated_brier_7_365"]
        if not np.isclose(measured["integrated_brier_7_365"], expected, atol=1e-6):
            raise ValueError(f"structured baseline {key} score did not reproduce")
        summary["splits"][key]["structured"] = {
            "integrated_brier": measured["integrated_brier_7_365"],
            "mean_auc": measured["mean_auc_7_365"],
            "auc_90": measured["points"]["90"]["auc"],
        }
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=27183)
    args = parser.parse_args()
    result = run(args.root, args.output_dir,
                 permutations=args.permutations, seed=args.seed)
    print(json.dumps({"splits": result["splits"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/train.py · lines 1–384

<!-- REVIEW_FRAGMENT {"path":"training/v5/train.py","part":1,"start_line":1,"end_line":384,"source_sha256":"52ad18690e310427ea7aa01c2c514d814a0e9a7ed0905117c2b26125c64da754","fragment_sha256":"52ad18690e310427ea7aa01c2c514d814a0e9a7ed0905117c2b26125c64da754","added_terminal_newline":false} -->
````python
"""V5 首轮完整样本训练：单特征 Cox、结构化融合、结构化＋规则文本。"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from training.baselines.rank_metrics import auc_rank
from training.baselines.continuous_competing_cox import fit_competing_cox, predict_cumulative_incidence
from training.v5.data import V5Data, collate, load_data
from training.v5.model import V5Model, competing_nll, predict_cif


GRID = np.arange(366, dtype=np.float64)
REPORT_DAYS = (7, 14, 30, 60, 90, 180, 365)
CONTINUOUS = ((0, 2, 3), (), (), tuple(range(15)), tuple(range(25)), (0,))


@dataclass(frozen=True)
class FittedModel:
    model: V5Model
    history: list[dict[str, float | int]]
    best_epoch: int
    validation_predictions: np.ndarray


def preprocess(data: V5Data) -> tuple[V5Data, list[dict[str, list[float]]]]:
    """中位数和 IQR 只在训练患者的已观测值上拟合，缺失保持掩码。"""

    train = data.splits == "train"
    groups: list[np.ndarray] = []
    report: list[dict[str, list[float]]] = []
    for group_index, (values, masks) in enumerate(zip(data.group_values, data.group_masks, strict=True)):
        transformed = values.copy()
        medians = np.zeros(values.shape[1], dtype=np.float64)
        scales = np.ones(values.shape[1], dtype=np.float64)
        for column in CONTINUOUS[group_index]:
            observed = values[train & (masks[:, column] > 0), column].astype(np.float64)
            if observed.size:
                median = float(np.median(observed))
                q1, q3 = np.percentile(observed, (25, 75))
                scale = max(float(q3 - q1), 1.0)
                medians[column], scales[column] = median, scale
                present = masks[:, column] > 0
                transformed[present, column] = np.clip(
                    (values[present, column] - median) / scale, -8.0, 8.0
                )
        transformed[masks == 0] = 0.0
        if not np.isfinite(transformed).all():
            raise ValueError("nonfinite preprocessed feature")
        groups.append(transformed.astype(np.float32))
        report.append({"medians": medians.tolist(), "scales": scales.tolist()})
    return replace(data, group_values=tuple(groups)), report


def split_sequences(data: V5Data, split: str) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for sequence in data.sequences:
        labels = set(data.splits[sequence].tolist())
        if len(labels) != 1:
            raise ValueError("patient crosses data splits")
        if split in labels:
            result.append(sequence)
    if not result:
        raise ValueError(f"empty {split} partition")
    return result


def _batches(sequences: list[np.ndarray], batch_size: int) -> list[list[np.ndarray]]:
    """按序列长度局部排序降低 padding；不改变患者内时序。"""

    ordered = sorted(sequences, key=len)
    return [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]


@torch.no_grad()
def predict(model: V5Model, data: V5Data, sequences: list[np.ndarray], *,
            include_text: bool, device: torch.device, batch_size: int) -> tuple[np.ndarray, float]:
    model.eval()
    log_amplitude = np.zeros((len(data.hadm_ids), 2), dtype=np.float32)
    filled = np.zeros(len(data.hadm_ids), dtype=bool)
    losses: list[tuple[float, int]] = []
    for members in _batches(sequences, batch_size):
        batch = collate(data, members, include_text=include_text).to(device)
        output = model(batch)
        count = int(batch.valid.sum().item())
        losses.append((float(competing_nll(output, batch).item()), count))
        probabilities = output.log_amplitude.detach().cpu().numpy()
        for position, sequence in enumerate(members):
            log_amplitude[sequence] = probabilities[position, :len(sequence)]
            filled[sequence] = True
    selected = np.concatenate(sequences)
    if not filled[selected].all() or int(filled.sum()) != len(selected):
        raise ValueError("prediction index mismatch")
    curves = predict_cif(log_amplitude[selected],
                         torch.nn.functional.softplus(model.raw_shape).detach().cpu().numpy() + 0.1,
                         GRID)
    nll = sum(loss * count for loss, count in losses) / sum(count for _, count in losses)
    return curves[:, :, 1], float(nll)


def metrics(times: np.ndarray, events: np.ndarray, risk: np.ndarray) -> dict[str, object]:
    if risk.shape != (len(times), 366):
        raise ValueError("risk array must have one daily curve per admission")
    observed = ((events[:, None] == 1) & (times[:, None] <= GRID[None, :])).astype(np.float64)
    brier = np.mean((risk - observed) ** 2, axis=0)
    auc = np.full(366, np.nan, dtype=np.float64)
    cases = observed.sum(axis=0).astype(int)
    for day in range(7, 366):
        if min(int(cases[day]), len(times) - int(cases[day])) >= 20:
            auc[day] = auc_rank(risk[:, day], observed[:, day].astype(np.int8))
    points = {str(day): {"auc": float(auc[day]) if np.isfinite(auc[day]) else None,
                         "brier": float(brier[day]), "cases": int(cases[day]),
                         "mean_predicted_risk": float(risk[:, day].mean()),
                         "observed_cumulative_fraction_world_a": float(observed[:, day].mean())}
              for day in REPORT_DAYS}
    return {
        "admissions": int(len(times)), "readmissions": int((events == 1).sum()),
        "deaths": int((events == 2).sum()),
        "integrated_brier_7_365": float(np.trapezoid(brier[7:], GRID[7:]) / (365 - 7)),
        "mean_auc_7_365": float(np.nanmean(auc[7:])),
        "points": points,
        "brier_curve": brier, "auc_curve": auc, "case_curve": cases,
    }


def _public_metrics(detail: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in detail.items()
            if key not in {"brier_curve", "auc_curve", "case_curve"}}


def train_model(data: V5Data, train_sequences: list[np.ndarray],
                validation_sequences: list[np.ndarray], *, include_text: bool,
                device: torch.device, max_epochs: int, batch_size: int,
                seed: int, balance_text: bool = False) -> FittedModel:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 结构化对照与文本编码维度无关，始终使用相同参数形状。
    model = V5Model(text_width=data.text_width if include_text else 768,
                    balance_text=balance_text if include_text else False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
    rng = np.random.default_rng(seed)
    best_epoch, best_score = 0, math.inf
    best_state = copy.deepcopy(model.state_dict())
    best_predictions: np.ndarray | None = None
    history: list[dict[str, float | int]] = []
    stale = 0
    val_indices = np.concatenate(validation_sequences)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(train_sequences))
        train_loss_sum, train_count = 0.0, 0
        for start in range(0, len(order), batch_size):
            members = [train_sequences[int(index)] for index in order[start:start + batch_size]]
            batch = collate(data, members, include_text=include_text).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = competing_nll(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite V5 training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(batch.valid.sum().item())
            train_loss_sum += float(loss.detach().item()) * count
            train_count += count
        validation_risk, val_nll = predict(
            model, data, validation_sequences, include_text=include_text,
            device=device, batch_size=batch_size,
        )
        evaluated = metrics(data.times[val_indices], data.events[val_indices], validation_risk)
        score = float(evaluated["integrated_brier_7_365"])
        entry: dict[str, float | int] = {
            "epoch": epoch, "train_nll": train_loss_sum / train_count,
            "validation_nll": val_nll, "validation_integrated_brier": score,
            "validation_mean_auc": float(evaluated["mean_auc_7_365"]),
        }
        history.append(entry)
        print(json.dumps({"text": include_text, **entry}), flush=True)
        if score < best_score - 1e-5:
            best_epoch, best_score = epoch, score
            best_state = copy.deepcopy(model.state_dict())
            best_predictions = validation_risk.copy()
            stale = 0
        else:
            stale += 1
        if stale >= 7:
            break
    model.load_state_dict(best_state)
    assert best_predictions is not None
    return FittedModel(model, history, best_epoch, best_predictions)


def clustered_brier_interval(subject_ids: np.ndarray, times: np.ndarray,
                             events: np.ndarray, full: np.ndarray,
                             comparator: np.ndarray, *, seed: int = 2026) -> list[float]:
    observed = ((events[:, None] == 1) & (times[:, None] <= GRID[None, :])).astype(float)
    delta = np.trapezoid((full[:, 7:] - observed[:, 7:]) ** 2
                         - (comparator[:, 7:] - observed[:, 7:]) ** 2,
                         GRID[7:], axis=1) / (365 - 7)
    patients, inverse = np.unique(subject_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(len(patients))]
    rng = np.random.default_rng(seed)
    samples = np.empty(300, dtype=np.float64)
    for iteration in range(len(samples)):
        draw = np.concatenate([groups[index] for index in rng.integers(len(groups), size=len(groups))])
        samples[iteration] = delta[draw].mean()
    return np.quantile(samples, (0.025, 0.975)).tolist()


def _save_curves(output_dir: Path, curves: dict[str, dict[str, dict[str, object]]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output_dir / "aggregate_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "model", "day", "auc", "brier", "cumulative_cases"))
        for split, models in curves.items():
            for name, values in models.items():
                for day in range(366):
                    score = values["auc_curve"][day]
                    writer.writerow((split, name, day, "" if not np.isfinite(score) else score,
                                     values["brier_curve"][day], values["case_curve"][day]))
    for split, models in curves.items():
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for name, values in models.items():
            axes[0].plot(GRID[7:], values["auc_curve"][7:], label=name)
            axes[1].plot(GRID[7:], values["brier_curve"][7:], label=name)
        axes[0].set(xlabel="Days after discharge", ylabel="Competing-risk cumulative AUC(t)",
                    ylim=(0.4, 0.9), title=f"{split}: discrimination")
        axes[1].set(xlabel="Days after discharge", ylabel="Brier(t)",
                    title=f"{split}: probability error")
        for axis in axes:
            axis.legend()
            axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output_dir / f"{split}_auc_brier.png", dpi=160)
        plt.close(figure)


def run(structured_dir: Path, manifest: Path, vectors: Path, output_dir: Path,
        *, text_name: str = "bert", vector_width: int | None = None,
        max_epochs: int = 35, batch_size: int = 24,
        balance_text: bool = False) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("V5 full run requires CUDA")
    expected_width = {"bert": 768, "qwen2560": 2560, "qwen768": 768, "qwen128": 128}
    if text_name not in expected_width:
        raise ValueError("unsupported text encoder experiment name")
    data, preprocessing = preprocess(load_data(structured_dir, manifest, vectors,
                                              vector_width=vector_width))
    expected_encoder = "Bio_ClinicalBERT" if text_name == "bert" else "Qwen3-Embedding-4B"
    if data.text_width != expected_width[text_name] or data.coverage["text_encoder"] != expected_encoder:
        raise ValueError("experiment name and text vector dimension disagree")
    train_sequences = split_sequences(data, "train")
    validation_sequences = split_sequences(data, "validation")
    test_sequences = split_sequences(data, "test")
    index = {split: np.concatenate(sequences) for split, sequences in
             (("train", train_sequences), ("validation", validation_sequences),
              ("historical_test", test_sequences))}
    device = torch.device("cuda")
    anchor_train = index["train"]
    anchor_feature = data.group_values[0][:, 3:4].astype(np.float64)
    anchor = fit_competing_cox(anchor_feature[anchor_train], data.times[anchor_train],
                               data.events[anchor_train], l2=10.0)
    if not anchor.readmission.converged or not anchor.death.converged:
        raise RuntimeError("single-feature competing Cox baseline failed")
    fitted = {
        "structured": train_model(data, train_sequences, validation_sequences,
                                  include_text=False, device=device, max_epochs=max_epochs,
                                  batch_size=batch_size, seed=2026),
        f"structured_{text_name}": train_model(data, train_sequences, validation_sequences,
                                               include_text=True, device=device, max_epochs=max_epochs,
                                               batch_size=batch_size, seed=2026,
                                               balance_text=balance_text),
    }
    curves: dict[str, dict[str, dict[str, object]]] = {}
    for split, sequences in (("validation", validation_sequences),
                             ("historical_test", test_sequences)):
        selected = index[split]
        risks: dict[str, np.ndarray] = {
            "prior_aecopd_cox": predict_cumulative_incidence(
                anchor, anchor_feature[selected], GRID
            )[:, :, 1]
        }
        for name, result in fitted.items():
            risks[name] = (result.validation_predictions if split == "validation"
                           else predict(result.model, data, sequences,
                                        include_text=name != "structured",
                                        device=device, batch_size=batch_size)[0])
        curves[split] = {name: metrics(data.times[selected], data.events[selected], risk)
                         for name, risk in risks.items()}
    validation = curves["validation"]
    interval = clustered_brier_interval(
        data.subject_ids[index["validation"]], data.times[index["validation"]],
        data.events[index["validation"]],
        fitted[f"structured_{text_name}"].validation_predictions,
        fitted["structured"].validation_predictions,
    )
    report: dict[str, object] = {
        "model": "V5 repeated-discharge structured/text causal Transformer competing Weibull",
        "text_experiment": text_name,
        "text_input_width": data.text_width,
        "balanced_text_content": balance_text,
        "status": "exploratory internal run; text rules not gold-standard validated",
        "source": "MIMIC-IV full discharge prediction sample set",
        "coverage": data.coverage,
        "split_counts": {split: {"admissions": int(len(rows)),
                                 "patients": int(len(sequences))}
                         for split, rows, sequences in
                         (("train", index["train"], train_sequences),
                          ("validation", index["validation"], validation_sequences),
                          ("historical_test", index["historical_test"], test_sequences))},
        "split_rule": "patient md5(aecopd-model-v1:subject_id), 70/15/15; historical test reused",
        "head": "two cause-specific Weibull amplitudes and two global shapes",
        "selection": "each network checkpoint selected by validation 7-365d integrated Brier",
        "validation": {name: _public_metrics(value) for name, value in curves["validation"].items()},
        "historical_test": {name: _public_metrics(value) for name, value in curves["historical_test"].items()},
        "text_minus_structured_validation_brier_95ci_patient_cluster": interval,
        "best_epochs": {name: result.best_epoch for name, result in fitted.items()},
        "limitations": [
            "No individual 365-day follow-up completeness or outside-hospital readmission capture is proven.",
            "Discharge summaries stored after discharge are treated as discharge-time retrospective documents.",
            "Rule text fields have not been validated against an annotated gold standard.",
            "The historical test split has been used by prior project experiments and is not a fresh blind test.",
            "eMAR drug-name mapping is conservative and does not prove an indication or treatment response.",
        ],
    }
    output_dir.mkdir(parents=True)
    for name, result in fitted.items():
        torch.save(result.model.state_dict(), output_dir / f"{name}.pt")
        (output_dir / f"{name}_history.json").write_text(
            json.dumps(result.history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output_dir / "preprocessing.json").write_text(
        json.dumps(preprocessing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_curves(output_dir, curves)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--text-manifest", type=Path, required=True)
    parser.add_argument("--text-vectors", "--bert-vectors", dest="text_vectors", type=Path,
                        required=True)
    parser.add_argument("--text-name", choices=("bert", "qwen2560", "qwen768", "qwen128"),
                        default="bert")
    parser.add_argument("--vector-width", type=int)
    parser.add_argument("--balance-text", action="store_true",
                        help="Layer-normalize projected snippet content before adding domain embedding")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    if args.max_epochs < 1 or args.batch_size < 1:
        raise ValueError("max epochs and batch size must be positive")
    report = run(args.structured_dir, args.text_manifest, args.text_vectors,
                 args.output_dir, text_name=args.text_name, vector_width=args.vector_width,
                 max_epochs=args.max_epochs, batch_size=args.batch_size,
                 balance_text=args.balance_text)
    print(json.dumps({"coverage": report["coverage"],
                      "validation": report["validation"],
                      "text_minus_structured_validation_brier_95ci_patient_cluster":
                          report["text_minus_structured_validation_brier_95ci_patient_cluster"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/train_event_sequence.py · lines 1–568

<!-- REVIEW_FRAGMENT {"path":"training/v5/train_event_sequence.py","part":1,"start_line":1,"end_line":568,"source_sha256":"469711a01842b60f77e9e4abd8b3d169f8132020c90ade8ec8ec117284acde86","fragment_sha256":"469711a01842b60f77e9e4abd8b3d169f8132020c90ade8ec8ec117284acde86","added_terminal_newline":false} -->
````python
"""以原始住院内事件流取代检验／给药摘要的 V5 竞争风险实验。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from dataclasses import replace
from functools import wraps
from pathlib import Path

import numpy as np
import torch

from training.v5.event_data import EVENT_NAMES, EventData, collate_events, load_event_data
from training.v5.event_model import TemporalFusionModel
from training.v5.event_set_model import EventSetFusionModel
from training.v5.abnormality_model import AbnormalityFusionModel
from training.v5.medication_effect_model import MedicationEffectFusionModel
from training.v5.medication_effect_data import EFFECT_ABLATIONS, EFFECT_GROUPS, attach_medication_effects
from training.v5.lab_recovery_data import ABNORMALITY_ABLATIONS, attach_abnormalities, attach_recovery
from training.v5.independent_data import IndependentData, split_sequences
from training.v5.model import competing_nll, predict_cif
from training.v5.processed_event_data import load_processed_event_data
from training.v5.train import GRID, _public_metrics, _save_curves, metrics

PRIOR_COUNT_CONCEPTS = ("context:prior_coded_admissions",
                        "context:prior_strict_aecopd_admissions")
# 整组非 context 静态概念；慢性共病 token 同时含本次和历史编码，不能解释为纯本次字段。
CURRENT_STATIC_TYPES = frozenset({"chronic_comorbidity", "aecopd_index_code",
                                  "acute_diagnosis", "disposition"})


def select_active_text_fields(base: IndependentData, active_fields: tuple[str, ...]) -> IndependentData:
    """保持字段词表和参数形状，只遮蔽未启用字段的片段引用。"""

    if not active_fields or len(set(active_fields)) != len(active_fields):
        raise ValueError("invalid active text field selection")
    unknown = set(active_fields) - set(base.field_roles)
    if unknown:
        raise ValueError("unknown active text fields: " + ", ".join(sorted(unknown)))
    active_ids = {base.field_roles.index(role) for role in active_fields}
    return replace(base, text_by_admission=tuple(
        tuple(ref for ref in refs if ref[0] in active_ids)
        for refs in base.text_by_admission))


def ablate_event_inputs(data: EventData, *, disable_medications: bool = False,
                        lab_process: str = "full",
                        excluded_concepts: tuple[str, ...] = ()) -> EventData:
    """保留事件词表；单次检验过程对照按概念合为一条平均值，时间设零。"""
    if lab_process not in {"full", "no_time", "one_per_concept"}:
        raise ValueError("unknown laboratory process ablation")
    if len(set(excluded_concepts)) != len(excluded_concepts) or set(excluded_concepts) - set(data.concept_names):
        raise ValueError("invalid excluded event concepts")
    excluded_ids = np.array([data.concept_names.index(name) + 1 for name in excluded_concepts], dtype=np.int64)
    values = data.event_values.copy()
    refs_by_admission = []
    for refs in data.events_by_admission:
        refs = refs[data.event_is_lab[refs]] if disable_medications else refs.copy()
        if len(excluded_ids):
            refs = refs[~np.isin(data.event_concepts[refs], excluded_ids)]
        if lab_process == "one_per_concept":
            keep = []
            for concept in np.unique(data.event_concepts[refs[data.event_is_lab[refs]]]):
                members = refs[(data.event_concepts[refs] == concept) & data.event_is_lab[refs]]
                observed = (members if data.event_numeric_mask is None else
                            members[data.event_numeric_mask[members]])
                first = int(observed[0] if len(observed) else members[0])
                values[first, 0] = float(values[observed, 0].mean()) if len(observed) else 0.0
                values[first, 1] = 0.0
                keep.append(first)
            keep.extend(int(ref) for ref in refs if not data.event_is_lab[ref])
            refs = np.asarray(keep, dtype=np.int64)
        elif lab_process == "no_time":
            values[refs[data.event_is_lab[refs]], 1] = 0.0
        refs_by_admission.append(refs)
    return replace(data, event_values=values, events_by_admission=tuple(refs_by_admission))


def _batches(sequences: list[np.ndarray], size: int) -> list[list[np.ndarray]]:
    return [sequences[start:start + size] for start in range(0, len(sequences), size)]


def deterministic_inference(function):
    """固定 CUDA 分段加和的推理结果；不改变训练阶段的算法开关。"""
    @wraps(function)
    def wrapped(*args, **kwargs):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            return function(*args, **kwargs)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    return wrapped


@deterministic_inference
@torch.no_grad()
def predict(model: TemporalFusionModel, data: EventData, sequences: list[np.ndarray], *,
            process_only: bool, batch_size: int, device: torch.device,
            shuffle_lab_order_seed: int | None = None,
            return_all_causes: bool = False) -> tuple[np.ndarray, float]:
    model.eval()
    selected = np.concatenate(sequences)
    logits = np.zeros((len(data.base.hadm_ids), 2), dtype=np.float32)
    filled = np.zeros(len(data.base.hadm_ids), dtype=bool)
    loss_sum, count_sum = 0.0, 0
    for members in _batches(sequences, batch_size):
        batch = collate_events(data, members, include_text=True,
                               process_only=process_only,
                               shuffle_lab_order_seed=shuffle_lab_order_seed).to(device)
        output = model(batch)
        count = int(batch.base.valid.sum().item())
        loss_sum += float(competing_nll(output, batch.base).item()) * count
        count_sum += count
        scores = output.log_amplitude.cpu().numpy()
        for position, sequence in enumerate(members):
            logits[sequence] = scores[position, :len(sequence)]
            filled[sequence] = True
    if int(filled.sum()) != len(selected) or not filled[selected].all():
        raise ValueError("prediction row alignment failed")
    shape = torch.nn.functional.softplus(model.raw_shape).detach().cpu().numpy() + 0.1
    cumulative = predict_cif(logits[selected], shape, GRID)
    return (cumulative if return_all_causes else cumulative[:, :, 1]), loss_sum / count_sum


def _build_model_with_matched_text_initialization(model_class: type[TemporalFusionModel], *,
                                                  concept_count: int, type_count: int,
                                                  field_count: int, text_width: int,
                                                  shared_text_init_width: int | None,
                                                  event_arguments: dict[str, int],
                                                  device: torch.device) -> TemporalFusionModel:
    model = model_class(
        concept_count=concept_count, type_count=type_count,
        field_count=field_count,
        text_width=text_width if shared_text_init_width is None else shared_text_init_width,
        **event_arguments,
    )
    if shared_text_init_width is not None and shared_text_init_width != text_width:
        # Subclasses initialize event/lab/medication layers after the text projection.
        # Build them with the BERT-sized projection first, then replace only that layer.
        model.text_projection = torch.nn.Linear(text_width,
                                                model.text_projection.out_features)
    return model.to(device)


def fit(data: EventData, train: list[np.ndarray], validation: list[np.ndarray], *,
        process_only: bool, max_epochs: int, batch_size: int,
        device: torch.device, event_encoder: str = "gru", seed: int = 2026,
        shared_text_init_width: int | None = None,
        history_path: Path | None = None) -> tuple[TemporalFusionModel, list[dict[str, float | int]], int, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_class = {"set": EventSetFusionModel, "gru": TemporalFusionModel,
                   "abnormal": AbnormalityFusionModel,
                   "abnormal_medication": MedicationEffectFusionModel}[event_encoder]
    event_arguments = {"event_concept_count": len(data.concept_names)} if event_encoder == "set" and data.concept_names != EVENT_NAMES else {}
    if data.recovery_by_admission is not None:
        if event_encoder == "set":
            event_arguments["clinical_recovery_width"] = data.recovery_by_admission.shape[1]
        elif event_encoder in ("abnormal", "abnormal_medication"):
            event_arguments["abnormality_width"] = data.recovery_by_admission.shape[1]
        else:
            raise ValueError("clinical features require set or abnormal encoder")
    elif event_encoder in ("abnormal", "abnormal_medication"):
        raise ValueError("abnormality encoder requires derived features")
    model = _build_model_with_matched_text_initialization(
        model_class,
        concept_count=len(data.base.concept_names), type_count=len(data.base.type_names),
        field_count=len(data.base.field_roles), text_width=data.base.text_width,
        shared_text_init_width=shared_text_init_width,
        event_arguments=event_arguments, device=device,
    )
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
    rng = np.random.default_rng(seed)
    validation_indices = np.concatenate(validation)
    best_state = copy.deepcopy(model.state_dict())
    best_risk: np.ndarray | None = None
    best_epoch, best_score, stale = 0, math.inf, 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        shuffled = rng.permutation(len(train))
        loss_sum, count_sum = 0.0, 0
        for start in range(0, len(shuffled), batch_size):
            members = [train[int(index)] for index in shuffled[start:start + batch_size]]
            batch = collate_events(data, members, include_text=True,
                                   process_only=process_only).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = competing_nll(model(batch), batch.base)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite temporal training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(batch.base.valid.sum().item())
            loss_sum += float(loss.detach().item()) * count
            count_sum += count
        risk, val_nll = predict(model, data, validation, process_only=process_only,
                                batch_size=batch_size, device=device)
        evaluated = metrics(data.base.times[validation_indices],
                            data.base.events[validation_indices], risk)
        score = float(evaluated["integrated_brier_7_365"])
        record: dict[str, float | int] = {
            "epoch": epoch, "train_nll": loss_sum / count_sum,
            "validation_nll": val_nll, "validation_integrated_brier": score,
            "validation_mean_auc": float(evaluated["mean_auc_7_365"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        if history_path is not None:
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps({"process_only": process_only, **record}), flush=True)
        if score < best_score - 1e-5:
            best_state, best_risk = copy.deepcopy(model.state_dict()), risk.copy()
            best_epoch, best_score, stale = epoch, score, 0
        else:
            stale += 1
        if stale >= 7:
            break
    model.load_state_dict(best_state)
    assert best_risk is not None
    return model, history, best_epoch, best_risk


def run(aligned_dir: Path, structured_dir: Path, base_csv: Path, raw_dir: Path | None,
        field_dir: Path, vectors: Path, labels_csv: Path, output_dir: Path, *,
        process_only: bool, max_epochs: int, batch_size: int,
        event_encoder: str = "gru", processed_event_dir: Path | None = None,
        raw_ledger_dir: Path | None = None,
        exclude_prior_admission_counts: bool = False,
        disable_text: bool = False, disable_events: bool = False,
        disable_current_static: bool = False,
        disable_medications: bool = False,
        lab_process: str = "full",
        excluded_event_concepts: tuple[str, ...] = (),
        active_text_fields: tuple[str, ...] | None = None,
        clinical_recovery_dir: Path | None = None,
        clinical_recovery_mode: str = "off",
        abnormality_scale_mode: str = "none",
        medication_effect_mode: str = "off",
        shared_text_init_width: int | None = None,
        prepared_source: tuple[EventData, dict[str, object]] | None = None,
        prepared_exposure_only: np.ndarray | None = None,
        seed: int = 2026) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("full temporal experiment requires CUDA")
    if (processed_event_dir is None) == (raw_dir is None):
        raise ValueError("choose exactly one legacy raw source or processed event source")
    if processed_event_dir is not None:
        if raw_ledger_dir is None or event_encoder not in ("set", "abnormal", "abnormal_medication"):
            raise ValueError("processed events require the raw ledger and set/abnormal encoder")
        data, preprocessing = prepared_source if prepared_source is not None else load_processed_event_data(
            aligned_dir, structured_dir, base_csv, processed_event_dir, raw_ledger_dir,
            labels_csv=labels_csv, field_dir=field_dir, vectors_path=vectors,
        )
    else:
        if prepared_source is not None or prepared_exposure_only is not None:
            raise ValueError("prepared batch inputs require the checked processed source")
        data, preprocessing = load_event_data(
            aligned_dir, structured_dir, base_csv, raw_dir, labels_csv=labels_csv,
            field_dir=field_dir, vectors_path=vectors,
        )
    if clinical_recovery_mode not in ("off", "masked", "active", "measurement_only", "core9", "core_risk",
                                      *(f"without_{group}" for group in ABNORMALITY_ABLATIONS)):
        raise ValueError("unknown clinical recovery mode")
    if clinical_recovery_mode != "off":
        if clinical_recovery_dir is None or processed_event_dir is None or event_encoder not in ("set", "abnormal", "abnormal_medication"):
            raise ValueError("clinical recovery requires derived features and processed events")
        if clinical_recovery_mode in ("measurement_only", "core9", "core_risk") and event_encoder not in ("abnormal", "abnormal_medication"):
            raise ValueError("control is specific to abnormality encoder")
        data = (attach_abnormalities(data, clinical_recovery_dir, mode=clinical_recovery_mode,
                                     scale_mode=abnormality_scale_mode)
                if event_encoder in ("abnormal", "abnormal_medication") else
                attach_recovery(data, clinical_recovery_dir, mode=clinical_recovery_mode))
    elif clinical_recovery_dir is not None:
        raise ValueError("clinical recovery directory supplied while mode is off")
    if event_encoder == "abnormal_medication":
        if processed_event_dir is None or medication_effect_mode == "off":
            raise ValueError("medication-effect encoder requires accepted eMAR and explicit mode")
        if prepared_exposure_only is None:
            data = attach_medication_effects(data, processed_event_dir, mode=medication_effect_mode)
        else:
            # 批量消融只复用已核验的二元给药；不得复用含频次／时间的缓存。
            exposure = prepared_exposure_only.reshape(len(data.base.hadm_ids), len(EFFECT_GROUPS), 6)
            if (medication_effect_mode != "exposure_only" or
                    not np.isin(exposure[:, :, 0], (0., 1.)).all() or
                    np.any(exposure[:, :, 1:] != 0.)):
                raise ValueError("cached medication input must contain only binary exposures")
            data = replace(data, medication_by_admission=prepared_exposure_only)
    elif medication_effect_mode != "off":
        raise ValueError("medication-effect mode requires abnormal_medication encoder")
    if event_encoder in ("abnormal", "abnormal_medication"):
        if clinical_recovery_mode == "off" or process_only or disable_events or disable_medications or lab_process != "full" or excluded_event_concepts:
            raise ValueError("abnormality-only encoder has no raw-event ablations and requires active/masked features")
        # 原始事件仍经来源契约校验，但绝不组成模型输入或梯度路径。
        data = replace(data, events_by_admission=tuple(
            np.empty(0, dtype=np.int64) for _ in data.base.hadm_ids))
    if data.base.label_sha256 is None:
        raise ValueError("explicit validated outcome labels are required")
    if exclude_prior_admission_counts:
        if not set(PRIOR_COUNT_CONCEPTS).issubset(data.base.concept_names):
            raise ValueError("prior admission count concepts missing")
        data = replace(data, base=replace(data.base, disabled_concepts=PRIOR_COUNT_CONCEPTS))
    if disable_current_static:
        types = {i + 1 for i, name in enumerate(data.base.type_names)
                 if name in CURRENT_STATIC_TYPES}
        disabled = tuple(name for i, name in enumerate(data.base.concept_names, start=1)
                         if np.any((data.base.token_concepts == i) &
                                   np.isin(data.base.token_types, list(types))))
        if not disabled or not types:
            raise ValueError("current static concepts missing")
        data = replace(data, base=replace(data.base,
                                         disabled_concepts=data.base.disabled_concepts + disabled))
    # 保持完整模型及参数形状相同，只把指定模态的本次住院输入变为空。
    # 训练和评估均使用此数据，因此与冻结检查点遮蔽是不同实验。
    if disable_text:
        data = replace(data, base=replace(
            data.base, text_by_admission=tuple(() for _ in data.base.hadm_ids)))
    if active_text_fields is not None:
        if disable_text:
            raise ValueError("cannot select text fields while text is disabled")
        data = replace(data, base=select_active_text_fields(data.base, active_text_fields))
    if disable_events:
        data = replace(data, events_by_admission=tuple(
            np.empty(0, dtype=np.int64) for _ in data.base.hadm_ids))
    if disable_medications or lab_process != "full" or excluded_event_concepts:
        if disable_events or (process_only and lab_process != "full"):
            raise ValueError("overlapping event ablations")
        data = ablate_event_inputs(data, disable_medications=disable_medications,
                                   lab_process=lab_process,
                                   excluded_concepts=excluded_event_concepts)
    train = split_sequences(data.base, "train")
    validation = split_sequences(data.base, "validation")
    test = split_sequences(data.base, "test")
    device = torch.device("cuda")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    (output_dir / "status.json").write_text(json.dumps({"status": "training", "max_epochs": max_epochs,
                                                        "disable_text": disable_text, "disable_events": disable_events}), encoding="utf-8")
    if "contract_audit" in preprocessing:
        (output_dir / "model_data_audit.json").write_text(json.dumps(preprocessing["contract_audit"], indent=2), encoding="utf-8")
    model, history, best_epoch, val_risk = fit(
        data, train, validation, process_only=process_only,
        max_epochs=max_epochs, batch_size=batch_size, device=device,
        event_encoder=event_encoder, seed=seed,
        shared_text_init_width=shared_text_init_width,
        history_path=output_dir / "history.json",
    )
    curves: dict[str, dict[str, dict[str, object]]] = {}
    death_scores: dict[str, dict[str, object]] = {}
    for split, sequences in (("validation", validation), ("historical_test", test)):
        indices = np.concatenate(sequences)
        all_causes, _ = predict(
            model, data, sequences, process_only=process_only,
            batch_size=batch_size, device=device, return_all_causes=True)
        risk = all_causes[:, :, 1]
        if split == "validation":
            maximum_difference = float(np.max(np.abs(risk - val_risk)))
            (output_dir / "checkpoint_prediction_check.json").write_text(json.dumps({
                "maximum_absolute_difference": maximum_difference,
                "absolute_tolerance": 1e-6, "deterministic_inference": True,
                "passed": maximum_difference <= 1e-6,
            }, indent=2), encoding="utf-8")
            if maximum_difference > 1e-6:
                torch.save(model.state_dict(), output_dir / "diagnostic_best_model.pt")
                np.savez_compressed(output_dir / "diagnostic_validation_risks.npz",
                                    selected=val_risk, reproduced=risk)
                raise ValueError(f"best checkpoint risk mismatch: maximum difference={maximum_difference}")
        curves[split] = {"temporal_process_only" if process_only else "temporal_full":
                         metrics(data.base.times[indices], data.base.events[indices], risk)}
        swapped_events = np.where(data.base.events[indices] == 1, 2,
                                  np.where(data.base.events[indices] == 2, 1, 0))
        death_scores[split] = metrics(data.base.times[indices], swapped_events, all_causes[:, :, 2])
        # 留在本地供同患者配对比较，不向聊天输出患者级预测。
        np.savez_compressed(output_dir / f"{split}_predictions.npz",
                            subject_ids=data.base.subject_ids[indices],
                            hadm_ids=data.base.hadm_ids[indices],
                            times=data.base.times[indices], events=data.base.events[indices],
                            risk=risk, competing_cif=all_causes)
    report: dict[str, object] = {
        "status": "training_completed",
        "model": ("V5 static/text + abnormality and medication-effect summaries; causal patient Transformer; two-cause Weibull"
                  if event_encoder == "abnormal_medication" else
                  "V5 static/text + abnormality-only admission encoder; causal patient Transformer; two-cause Weibull"
                  if event_encoder == "abnormal" else
                  "V5 independent static and text branches + " + event_encoder + " event encoder; causal patient Transformer; two-cause Weibull"),
        "event_encoder": event_encoder,
        "self_supervised_pretraining": False,
        "primary_event": data.base.label_name,
        "ablation": (f"medication effect {medication_effect_mode}; abnormality {clinical_recovery_mode}"
                     if event_encoder == "abnormal_medication" else
                     "abnormality features masked" if event_encoder == "abnormal" and clinical_recovery_mode == "masked" else
                     "original nine axes active; thirteen added axes masked"
                     if event_encoder == "abnormal" and clinical_recovery_mode == "core9" else
                     "abnormality severities/recovery masked; measurement presence/span retained"
                     if event_encoder == "abnormal" and clinical_recovery_mode == "measurement_only" else
                     "abnormality features active; raw event sequence absent" if event_encoder == "abnormal" else
                     "event type/time/frequency without lab numeric values" if process_only else
                     "no event inputs" if disable_events else "full raw event sequence"),
        "disabled_modalities": [name for name, disabled in (("text", disable_text), ("events", disable_events)) if disabled],
        "disabled_static_concepts": data.base.disabled_concepts,
        "disabled_current_static": disable_current_static,
        "current_static_ablation_scope": (
            "All non-context static concepts, including both current and historical channels of chronic comorbidity; "
            "age, sex, explicit prior counts, gap, events and text retained."
            if disable_current_static else None),
        "disabled_medications": disable_medications,
        "excluded_event_concepts": excluded_event_concepts,
        "lab_process": lab_process,
        "ablation_scope": ("Explicit counts fully masked; patient history, gaps, diagnoses and text unchanged."
                           if exclude_prior_admission_counts else
                           "Explicit prior admission counts retained; all other inputs unchanged."),
        "cohort_admissions": int(len(data.base.hadm_ids)),
        "cohort_patients": int(len(data.base.sequences)),
        "field_manifest_sha256": data.base.manifest_sha256,
        "field_roles": data.base.field_roles,
        "active_text_fields": (() if disable_text else
                               data.base.field_roles if active_text_fields is None else active_text_fields),
        "label_sha256": data.base.label_sha256,
        "raw_source_sha256": data.source_hashes,
        "text_vector_width": data.base.text_width,
        "shared_text_init_width": (data.base.text_width if shared_text_init_width is None
                                   else shared_text_init_width),
        "deterministic_inference": True,
        "event_concept_count": len(data.concept_names),
        "event_concepts": data.concept_names,
        "raw_events_enter_model": event_encoder not in ("abnormal", "abnormal_medication") and not disable_events,
        "abnormality_feature_count": (data.recovery_by_admission.shape[1]
                                      if event_encoder in ("abnormal", "abnormal_medication") else None),
        "medication_effect_feature_count": (data.medication_by_admission.shape[1]
                                            if event_encoder == "abnormal_medication" else None),
        "preprocessed_events": processed_event_dir is not None,
        "clinical_recovery_mode": clinical_recovery_mode,
        "abnormality_scale_mode": abnormality_scale_mode,
        "abnormality_scaling": data.abnormality_scaling,
        "medication_effect_mode": medication_effect_mode,
        "medication_nonconstant_input_count": (len(EFFECT_GROUPS) if medication_effect_mode == "exposure_only"
                                                else None),
        "medication_effect_groups": tuple(EFFECT_GROUPS) if event_encoder == "abnormal_medication" else (),
        "clinical_recovery_manifest_sha256": (None if clinical_recovery_dir is None else
                                             hashlib.sha256((clinical_recovery_dir / "manifest.json").read_bytes()).hexdigest()),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": torch.cuda.get_device_name(device),
        "epochs_run": len(history),
        "max_epochs": max_epochs,
        "patience": 7,
        "seed": seed,
        "learning_rate": 8e-4,
        "weight_decay": 0.02,
        "patient_batch_size": batch_size,
        "training_and_evaluation_seconds": time.perf_counter() - started,
        "best_epoch": best_epoch,
        "selection": f"validation 7-365d integrated Brier; {max_epochs}-epoch cap and seven-epoch patience",
        "validation": {**{name: _public_metrics(detail) for name, detail in curves["validation"].items()},
                       "death_temporal_full": _public_metrics(death_scores["validation"])},
        "historical_test": {**{name: _public_metrics(detail) for name, detail in curves["historical_test"].items()},
                            "death_temporal_full": _public_metrics(death_scores["historical_test"])},
        "limitations": [
            "This is a within-MIMIC design experiment, not external validation.",
            "A qualifying COPD code denotes coded co-occurrence, not adjudicated COPD cause.",
            "Diagnosis and readmission outside MIMIC hospitals are not fully observed.",
            "The historical test split was previously examined and is not blind.",
            "365-day administrative censoring is the existing observation contract, not complete lifetime or out-of-hospital follow-up.",
        ],
    }
    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "preprocessing.json").write_text(
        json.dumps(preprocessing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_curves(output_dir, curves)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_nll"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_nll"] for row in history], label="validation")
    axes[0].set(ylabel="Negative log likelihood")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_integrated_brier"] for row in history])
    axes[1].set(ylabel="Validation integrated Brier")
    axes[2].plot(epochs, [row["validation_mean_auc"] for row in history])
    axes[2].set(ylabel="Validation mean AUC(t)")
    for axis in axes:
        axis.set(xlabel="Epoch")
        axis.axvline(best_epoch, linestyle="--", color="grey")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=160)
    plt.close(figure)
    (output_dir / "status.json").write_text(json.dumps({"status": "training_completed"}), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned_dir", "structured_dir", "base_csv", "field_dir",
                 "vectors", "labels_csv", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--processed-event-dir", type=Path)
    parser.add_argument("--raw-ledger-dir", type=Path)
    parser.add_argument("--clinical-recovery-dir", type=Path)
    parser.add_argument("--clinical-recovery-mode", choices=("off", "masked", "active", "measurement_only", "core9", "core_risk",
                                                    *(f"without_{group}" for group in ABNORMALITY_ABLATIONS)), default="off")
    parser.add_argument("--abnormality-scale-mode", choices=("none", "training_positive_p95"), default="none")
    parser.add_argument("--medication-effect-mode", choices=("off", "active", "masked", "exposure_only", "core_acute", *(f"without_{group}" for group in EFFECT_ABLATIONS)), default="off")
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument("--exclude-prior-admission-counts", action="store_true")
    parser.add_argument("--disable-text", action="store_true")
    parser.add_argument("--active-text-fields", help="Comma-separated roles; fixed full field vocabulary is retained")
    parser.add_argument("--disable-events", action="store_true")
    parser.add_argument("--disable-current-static", action="store_true")
    parser.add_argument("--disable-medications", action="store_true")
    parser.add_argument("--exclude-event-concepts", help="Comma-separated existing event concepts to omit without changing vocabulary")
    parser.add_argument("--lab-process", choices=("full", "no_time", "one_per_concept"), default="full")
    parser.add_argument("--event-encoder", choices=("gru", "set", "abnormal", "abnormal_medication"), default="gru")
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--shared-text-init-width", type=int,
                        help="Match shared-layer initialization to this reference text width")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.max_epochs, args.batch_size, args.seed) < 1:
        raise ValueError("epochs, batch size, and seed must be positive")
    if args.shared_text_init_width is not None and args.shared_text_init_width < 1:
        raise ValueError("shared text initialization width must be positive")
    report = run(args.aligned_dir, args.structured_dir, args.base_csv, args.raw_dir,
                 args.field_dir, args.vectors, args.labels_csv, args.output_dir,
                 process_only=args.process_only, max_epochs=args.max_epochs,
                 batch_size=args.batch_size, event_encoder=args.event_encoder,
                 processed_event_dir=args.processed_event_dir, raw_ledger_dir=args.raw_ledger_dir,
                 exclude_prior_admission_counts=args.exclude_prior_admission_counts,
                 disable_text=args.disable_text, disable_events=args.disable_events,
                 disable_current_static=args.disable_current_static,
                 disable_medications=args.disable_medications, lab_process=args.lab_process,
                 clinical_recovery_dir=args.clinical_recovery_dir,
                 clinical_recovery_mode=args.clinical_recovery_mode,
                 abnormality_scale_mode=args.abnormality_scale_mode,
                 medication_effect_mode=args.medication_effect_mode,
                 excluded_event_concepts=(tuple(args.exclude_event_concepts.split(","))
                                          if args.exclude_event_concepts else ()),
                 active_text_fields=(tuple(args.active_text_fields.split(","))
                                     if args.active_text_fields is not None else None),
                 shared_text_init_width=args.shared_text_init_width,
                 seed=args.seed)
    print(json.dumps({"validation": report["validation"],
                      "historical_test": report["historical_test"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/train_independent.py · lines 1–264

<!-- REVIEW_FRAGMENT {"path":"training/v5/train_independent.py","part":1,"start_line":1,"end_line":264,"source_sha256":"68a705efa4f5551dc9262ed3ebd0df079540149496954b58f92807787e0aac21","fragment_sha256":"68a705efa4f5551dc9262ed3ebd0df079540149496954b58f92807787e0aac21","added_terminal_newline":false} -->
````python
"""独立字段文本分支的 V5 增量实验；沿用既有竞争风险评价合同。"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_data import (
    IndependentData, collate, load_data, normalize_on_training_patients, split_sequences,
)
from training.v5.independent_fusion import IndependentFusionModel
from training.v5.model import competing_nll, predict_cif
from training.v5.train import GRID, _public_metrics, _save_curves, clustered_brier_interval, metrics


def _batches(sequences: list[np.ndarray], batch_size: int) -> list[list[np.ndarray]]:
    ordered = sorted(sequences, key=len)
    return [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]


@torch.no_grad()
def predict(model: IndependentFusionModel, data: IndependentData,
            sequences: list[np.ndarray], *, include_text: bool, batch_size: int,
            device: torch.device) -> tuple[np.ndarray, float]:
    model.eval()
    indices = np.concatenate(sequences)
    logits = np.zeros((len(data.hadm_ids), 2), dtype=np.float32)
    filled = np.zeros(len(data.hadm_ids), dtype=bool)
    loss_sum, count_sum = 0.0, 0
    for members in _batches(sequences, batch_size):
        batch = collate(data, members, include_text=include_text).to(device)
        output = model(batch)
        count = int(batch.valid.sum().item())
        loss_sum += float(competing_nll(output, batch).item()) * count
        count_sum += count
        values = output.log_amplitude.cpu().numpy()
        for position, sequence in enumerate(members):
            logits[sequence] = values[position, :len(sequence)]
            filled[sequence] = True
    if int(filled.sum()) != len(indices) or not filled[indices].all():
        raise ValueError("prediction row alignment failed")
    shape = torch.nn.functional.softplus(model.raw_shape).detach().cpu().numpy() + 0.1
    return predict_cif(logits[indices], shape, GRID)[:, :, 1], loss_sum / count_sum


def fit(data: IndependentData, train: list[np.ndarray], validation: list[np.ndarray],
        *, include_text: bool, max_epochs: int, batch_size: int, seed: int,
        device: torch.device) -> tuple[IndependentFusionModel, list[dict[str, float | int]], int, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = IndependentFusionModel(
        concept_count=len(data.concept_names), type_count=len(data.type_names),
        field_count=len(FIELD_LABELS), text_width=data.text_width,
    ).to(device)
    # 文本投影宽度不同会消耗不同数量的初始化随机数；重置训练随机流，
    # 确保共享层在无文本对照中的 dropout 序列和更新顺序完全一致。
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
    rng = np.random.default_rng(seed)
    val_indices = np.concatenate(validation)
    best_state = copy.deepcopy(model.state_dict())
    best_risk: np.ndarray | None = None
    best_epoch, best_score, stale = 0, math.inf, 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        shuffled = rng.permutation(len(train))
        loss_sum, count_sum = 0.0, 0
        for start in range(0, len(shuffled), batch_size):
            members = [train[int(index)] for index in shuffled[start:start + batch_size]]
            batch = collate(data, members, include_text=include_text).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = competing_nll(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(batch.valid.sum().item())
            loss_sum += float(loss.detach().item()) * count
            count_sum += count
        val_risk, val_nll = predict(model, data, validation, include_text=include_text,
                                    batch_size=batch_size, device=device)
        summary = metrics(data.times[val_indices], data.events[val_indices], val_risk)
        score = float(summary["integrated_brier_7_365"])
        record: dict[str, float | int] = {
            "epoch": epoch, "train_nll": loss_sum / count_sum,
            "validation_nll": val_nll, "validation_integrated_brier": score,
            "validation_mean_auc": float(summary["mean_auc_7_365"]),
        }
        history.append(record)
        print(json.dumps({"text": include_text, **record}), flush=True)
        if score < best_score - 1e-5:
            best_state, best_risk = copy.deepcopy(model.state_dict()), val_risk.copy()
            best_epoch, best_score, stale = epoch, score, 0
        else:
            stale += 1
        if stale >= 7:
            break
    model.load_state_dict(best_state)
    assert best_risk is not None
    return model, history, best_epoch, best_risk


def run(aligned_dir: Path, structured_dir: Path, base_csv: Path, field_dir: Path,
        vectors: Path, output_dir: Path, *, text_name: str = "bert",
        max_epochs: int, batch_size: int,
        reuse_structured_run: Path | None = None,
        labels_csv: Path | None = None) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("full independent-fusion experiment requires CUDA")
    data, preprocessing = normalize_on_training_patients(
        load_data(aligned_dir, structured_dir, base_csv, field_dir, vectors,
                  labels_csv=labels_csv)
    )
    if (text_name, data.text_width) not in (("bert", 768), ("qwen", 2560)):
        raise ValueError("text experiment name and vector width disagree")
    train = split_sequences(data, "train")
    validation = split_sequences(data, "validation")
    test = split_sequences(data, "test")
    partitions = {"validation": validation, "historical_test": test}
    device = torch.device("cuda")
    text_model_name = f"structured_field_{text_name}"
    reference_report: dict[str, object] | None = None
    if reuse_structured_run is None:
        structured = fit(data, train, validation, include_text=False,
                         max_epochs=max_epochs, batch_size=batch_size, seed=2026,
                         device=device)
    else:
        reference_report = json.loads((reuse_structured_run / "report.json").read_text(encoding="utf-8"))
        if (reference_report["field_manifest_sha256"] != data.manifest_sha256
                or reference_report.get("label_sha256") != data.label_sha256
                or reference_report["cohort_admissions"] != len(data.hadm_ids)
                or reference_report["structured_concepts"] != len(data.concept_names)
                or reference_report["structured_types"] != len(data.type_names)):
            raise ValueError("reused structured run has a different input contract")
        structured_model = IndependentFusionModel(
            concept_count=len(data.concept_names), type_count=len(data.type_names),
            field_count=len(FIELD_LABELS), text_width=int(reference_report["text_vector_width"]),
        ).to(device)
        structured_model.load_state_dict(torch.load(
            reuse_structured_run / "structured.pt", map_location=device, weights_only=True
        ))
        structured_history = json.loads(
            (reuse_structured_run / "structured_history.json").read_text(encoding="utf-8")
        )
        structured_risk, _ = predict(structured_model, data, validation,
                                      include_text=False, batch_size=batch_size, device=device)
        structured = (structured_model, structured_history,
                      int(reference_report["best_epochs"]["structured"]), structured_risk)
    fitted = {
        "structured": structured,
        text_model_name: fit(data, train, validation, include_text=True,
                             max_epochs=max_epochs, batch_size=batch_size,
                             seed=2026, device=device),
    }
    curves: dict[str, dict[str, dict[str, object]]] = {}
    for split, sequences in partitions.items():
        indices = np.concatenate(sequences)
        curves[split] = {}
        for name, (model, _, _, val_risk) in fitted.items():
            risk = val_risk if split == "validation" else predict(
                model, data, sequences, include_text=name != "structured",
                batch_size=batch_size, device=device,
            )[0]
            curves[split][name] = metrics(data.times[indices], data.events[indices], risk)
    if reference_report is not None:
        for split in partitions:
            score = float(curves[split]["structured"]["integrated_brier_7_365"])
            expected = float(reference_report[split]["structured"]["integrated_brier_7_365"])
            if abs(score - expected) > 1e-6:
                raise ValueError(f"reused structured checkpoint changed {split} predictions")
    val_indices = np.concatenate(validation)
    interval = clustered_brier_interval(
        data.subject_ids[val_indices], data.times[val_indices], data.events[val_indices],
        fitted[text_model_name][3], fitted["structured"][3],
    )
    report: dict[str, object] = {
        "model": "V5 independent sparse structured and field-text admission branches; causal patient Transformer; two-cause Weibull",
        "status": "exploratory internal comparison; clinical content not gold-standard validated",
        "cohort_admissions": int(len(data.hadm_ids)),
        "cohort_patients": int(len(data.sequences)),
        "structured_concepts": len(data.concept_names),
        "structured_types": len(data.type_names),
        "text_encoder": text_name,
        "field_manifest_sha256": data.manifest_sha256,
        "label_sha256": data.label_sha256,
        "primary_event": data.label_name or "strict AECOPD-coded readmission",
        "text_vector_width": data.text_width,
        "split_counts": {name: {"admissions": int(sum(map(len, sequences))),
                                "patients": len(sequences)}
                         for name, sequences in (("train", train), ("validation", validation),
                                                  ("historical_test", test))},
        "selection": "checkpoint by validation 7-365d integrated Brier; same patient split and shared seed",
        "structured_reference_run": str(reuse_structured_run) if reuse_structured_run else None,
        "validation": {name: _public_metrics(value) for name, value in curves["validation"].items()},
        "historical_test": {name: _public_metrics(value) for name, value in curves["historical_test"].items()},
        "text_minus_structured_validation_brier_95ci_patient_cluster": interval,
        "best_epochs": {name: result[2] for name, result in fitted.items()},
        "limitations": [
            "MIMIC-IV is a design proxy; no private-data sample or field contract was provided.",
            "The text field selection has not been checked against an annotated clinical gold standard.",
            "Outside-hospital readmission and complete 365-day follow-up are unproven.",
            "The historical test split has been used previously and is not blind.",
        ],
    }
    if reference_report is not None:
        for split in partitions:
            report[split]["structured"] = reference_report[split]["structured"]
    output_dir.mkdir(parents=True)
    for name, (model, history, _, _) in fitted.items():
        torch.save(model.state_dict(), output_dir / f"{name}.pt")
        (output_dir / f"{name}_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output_dir / "preprocessing.json").write_text(
        json.dumps(preprocessing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_curves(output_dir, curves)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--field-dir", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--text-name", choices=("bert", "qwen"), default="bert")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--reuse-structured-run", type=Path,
                        help="Reuse one fitted structured comparator across text encoders")
    parser.add_argument("--labels-csv", type=Path,
                        help="Validated broad COPD-coded outcome overlay; legacy strict labels otherwise")
    args = parser.parse_args()
    if min(args.max_epochs, args.batch_size) < 1:
        raise ValueError("epochs and batch size must be positive")
    report = run(args.aligned_dir, args.structured_dir, args.base_csv, args.field_dir,
                 args.vectors, args.output_dir, text_name=args.text_name,
                 max_epochs=args.max_epochs,
                 batch_size=args.batch_size,
                 reuse_structured_run=args.reuse_structured_run,
                 labels_csv=args.labels_csv)
    print(json.dumps({"validation": report["validation"],
                      "historical_test": report["historical_test"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/validate_field_text_baseline.py · lines 1–146

<!-- REVIEW_FRAGMENT {"path":"training/v5/validate_field_text_baseline.py","part":1,"start_line":1,"end_line":146,"source_sha256":"798bc6ff1025e8030f4b5b8e54d012630a72a05281877581e65e3061b4f23786","fragment_sha256":"798bc6ff1025e8030f4b5b8e54d012630a72a05281877581e65e3061b4f23786","added_terminal_newline":false} -->
````python
"""独立核对字段文本清单：来源、筛查、行号、原文和两种分词器长度。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

from training.v5.field_text_baseline import FIELD_LABELS, FIELDS, eligible, field_text


def _rows(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if path.name == "shared_text.csv.gz" and set(FIELDS).issubset(reader.fieldnames or ()):
            if {"event_type", "time_days"} & set(reader.fieldnames or ()):
                raise ValueError("outcome labels present in text manifest")
        yield from reader


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(
    aligned_dir: Path, candidate_dir: Path, field_dir: Path,
    bert_tokenizer: object, qwen_tokenizer: object,
) -> dict[str, object]:
    """每个保留块必须来自已校验的源清单，且前缀只能由字段名生成。"""

    summary = json.loads((field_dir / "summary.json").read_text(encoding="utf-8"))
    source_path = candidate_dir / "shared_text.csv.gz"
    output_path = field_dir / "shared_text.csv.gz"
    if _sha(source_path) != summary["source_candidate_manifest_sha256"]:
        raise ValueError("source manifest fingerprint changed")
    if _sha(output_path) != summary["manifest_sha256"]:
        raise ValueError("field manifest fingerprint changed")
    section_path = aligned_dir / "note_sections.csv.gz"
    if _sha(section_path) != summary["cohort_source_sections_sha256"]:
        raise ValueError("source section fingerprint changed")
    sections: dict[str, dict[str, str]] = {}
    for row in _rows(section_path):
        if row["section_role"] in FIELD_LABELS:
            sections[row["section_id"]] = row
    output = iter(_rows(output_path))
    next_row = next(output, None)
    counts: Counter[str] = Counter()
    admissions: set[str] = set()
    included_sections: set[str] = set()
    excluded: Counter[str] = Counter()
    for source_number, source in enumerate(_rows(source_path)):
        role = source["section_role"]
        if role not in FIELD_LABELS:
            continue
        section = sections.get(source["section_id"])
        if section is None or section["section_role"] != role:
            raise ValueError("source field section mismatch")
        keep, reason = eligible(role, section["text"], section["timestamp_relation"])
        if not keep:
            excluded[reason] += 1
            if next_row is not None and int(next_row["source_manifest_row"]) == source_number:
                raise ValueError("excluded section found in field manifest")
            continue
        if next_row is None or int(next_row["source_manifest_row"]) != source_number:
            raise ValueError("eligible source chunk omitted or out of order")
        row = next_row
        if int(row["embedding_row"]) != counts["total"]:
            raise ValueError("non-contiguous field embedding rows")
        for field, source_field in (
            ("subject_id", "subject_id"), ("hadm_id", "hadm_id"),
            ("field", "section_role"), ("section_id", "section_id"),
            ("chunk_index", "chunk_index"), ("source_start_char", "source_start_char"),
            ("source_end_char", "source_end_char"),
            ("source_section_sha256", "source_section_sha256"),
            ("timestamp_relation", "timestamp_relation"),
        ):
            if row[field] != source[source_field]:
                raise ValueError(f"field text metadata mismatch: {field}")
        raw = source["chunk_text"]
        start = int(source["source_start_char"]) - int(section["start_char"])
        end = int(source["source_end_char"]) - int(section["start_char"])
        if not 0 <= start < end <= len(section["text"]) or section["text"][start:end] != raw:
            raise ValueError("raw chunk cannot be recovered from source section")
        if row["source_chunk_sha256"] != hashlib.sha256(raw.encode("utf-8")).hexdigest():
            raise ValueError("raw chunk fingerprint mismatch")
        expected = field_text(role, raw)
        if row["chunk_text"] != expected or row["text_sha256"] != hashlib.sha256(expected.encode("utf-8")).hexdigest():
            raise ValueError("field prefix or content differs from source")
        bert_count = len(bert_tokenizer(expected, add_special_tokens=True)["input_ids"])
        qwen_count = len(qwen_tokenizer(expected, add_special_tokens=False)["input_ids"])
        if bert_count != int(row["bert_token_count"]) or qwen_count != int(row["token_count"]):
            raise ValueError("encoder length metadata mismatch")
        if bert_count > 512 or qwen_count > 8192:
            raise ValueError("encoder context exceeded")
        counts["total"] += 1
        counts[role] += 1
        admissions.add(row["hadm_id"])
        included_sections.add(row["section_id"])
        next_row = next(output, None)
    if next_row is not None or counts["total"] != summary["chunks"]:
        raise ValueError("field manifest has extra rows or count drift")
    if len(admissions) != summary["admissions_with_text"] or len(included_sections) != summary["source_sections"]:
        raise ValueError("coverage count drift")
    if {role: counts[role] for role in FIELD_LABELS} != summary["chunks_by_field"]:
        raise ValueError("field distribution drift")
    if dict(sorted(excluded.items())) != summary["excluded_chunks_by_reason"]:
        raise ValueError("screening decision drift")
    result: dict[str, object] = {
        "status": "field_text_manifest_validated",
        "manifest_sha256": summary["manifest_sha256"],
        "chunks": counts["total"], "source_sections": len(included_sections),
        "admissions_with_text": len(admissions),
        "same_rows_for_bert_and_qwen": True,
        "clinical_semantics_review_complete": False,
        "encoding_started": False,
    }
    (field_dir / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--field-dir", type=Path, required=True)
    parser.add_argument("--bert-model-dir", type=Path, default=Path("/model-cache/Bio_ClinicalBERT"))
    parser.add_argument("--qwen-model-dir", type=Path, default=Path("/model-cache/Qwen3-Embedding-4B"))
    args = parser.parse_args()
    bert = AutoTokenizer.from_pretrained(args.bert_model_dir, local_files_only=True)
    qwen = AutoTokenizer.from_pretrained(args.qwen_model_dir, local_files_only=True)
    print(json.dumps(validate(args.aligned_dir, args.candidate_dir, args.field_dir, bert, qwen), ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/validate_strict_candidate.py · lines 1–225

<!-- REVIEW_FRAGMENT {"path":"training/v5/validate_strict_candidate.py","part":1,"start_line":1,"end_line":225,"source_sha256":"9725b32450a9f5f7eda22acbece47f02cd2f1109d456c6ddc2281c11c0ca7d21","fragment_sha256":"9725b32450a9f5f7eda22acbece47f02cd2f1109d456c6ddc2281c11c0ca7d21","added_terminal_newline":false} -->
````python
"""同一严格结局与患者划分下，审计冻结 V5 和低容量连续时间参照。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from training.baselines.continuous_competing_cox import fit_competing_cox, predict_cumulative_incidence
from training.v5.event_set_model import EventSetFusionModel
from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.independent_data import split_sequences
from training.v5.processed_event_data import load_processed_event_data
from training.v5.train import GRID, _public_metrics, _save_curves, clustered_brier_interval, metrics
from training.v5.train_event_sequence import predict


def _rows(base_csv: Path, hadm_ids: np.ndarray) -> list[dict[str, str]]:
    with base_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        by_hadm = {int(row["hadm_id"]): row for row in csv.DictReader(handle)}
    if set(by_hadm) != set(hadm_ids.tolist()):
        raise ValueError("simple baseline does not match model admissions")
    return [by_hadm[int(hadm)] for hadm in hadm_ids]


def _features(rows: list[dict[str, str]], train_mask: np.ndarray) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """预固定五项出院可得特征；只使用训练集拟合年龄中心。"""
    age = np.array([float(row["age_at_admission"]) for row in rows])
    if not np.isfinite(age).all():
        raise ValueError("nonfinite age in simple baseline")
    center = float(np.median(age[train_mask]))
    gender = [row["gender"].strip().upper() for row in rows]
    if set(gender) - {"M", "F"}:
        raise ValueError("unexpected gender in simple baseline")
    prior = np.array([int(row["prior_coded_admission_count"]) for row in rows])
    prior_strict = np.array([int(row["prior_strict_aecopd_count"]) for row in rows])
    strict_values = [row["index_is_strict_aecopd"].strip().lower() for row in rows]
    if set(strict_values) - {"t", "f"}:
        raise ValueError("unexpected current strict code flag")
    strict_now = np.array([value == "t" for value in strict_values], dtype=float)
    if (prior < 0).any() or (prior_strict < 0).any():
        raise ValueError("invalid admission history or current strict code")
    x = np.column_stack((
        (age - center) / 10.0,
        np.array([value == "M" for value in gender], dtype=float),
        np.log1p(prior),
        np.log1p(prior_strict),
        strict_now,
    ))
    return x, ["age_per_10_years", "male", "log1p_prior_coded", "log1p_prior_strict", "current_strict_code"], {"age_median_train": center}


def _check_saved(path: Path, data, indices: np.ndarray, risk: np.ndarray) -> None:
    saved = np.load(path)
    for key, actual in (("subject_ids", data.base.subject_ids[indices]),
                        ("hadm_ids", data.base.hadm_ids[indices]),
                        ("events", data.base.events[indices])):
        if not np.array_equal(saved[key], actual):
            raise ValueError(f"saved model predictions differ on {key}")
    if not np.allclose(saved["times"], data.base.times[indices], atol=1e-7, rtol=0):
        raise ValueError("saved model predictions differ on event times")
    if not np.allclose(saved["risk"], risk, atol=2e-6, rtol=0):
        raise ValueError("reloaded checkpoint does not reproduce saved risks")


def _calibration(times: np.ndarray, events: np.ndarray, risk: np.ndarray) -> dict[str, object]:
    answer = {}
    for day in (30, 90, 365):
        predicted = risk[:, day]
        observed = (events == 1) & (times <= day)
        edges = np.quantile(predicted, np.linspace(0, 1, 6))
        groups = []
        for bin_index in range(5):
            selected = (predicted >= edges[bin_index]) & (
                predicted < edges[bin_index + 1] if bin_index < 4 else predicted <= edges[bin_index + 1]
            )
            groups.append({"n": int(selected.sum()),
                           "predicted": float(predicted[selected].mean()) if selected.any() else None,
                           "observed_world_a": float(observed[selected].mean()) if selected.any() else None})
        answer[str(day)] = groups
    return answer


def _subgroups(data, rows: list[dict[str, str]], indices: np.ndarray, risk: np.ndarray) -> dict[str, object]:
    prior = np.array([int(row["prior_coded_admission_count"]) for row in rows])[indices]
    strict = np.array([row["index_is_strict_aecopd"].strip().lower() == "t" for row in rows])[indices]
    groups = {
        "first_coded_admission": prior == 0,
        "one_to_two_prior": (prior >= 1) & (prior <= 2),
        "three_plus_prior": prior >= 3,
        "current_strict_code": strict == 1,
        "current_non_strict_code": strict == 0,
    }
    out = {}
    for name, mask in groups.items():
        if not mask.any():
            continue
        detail = metrics(data.base.times[indices][mask], data.base.events[indices][mask], risk[mask])
        out[name] = {"patients": int(np.unique(data.base.subject_ids[indices][mask]).size),
                     **_public_metrics(detail)}
    return out


def _masked_data(data, name: str):
    if name == "text_absent":
        return replace(data, base=replace(data.base, text_by_admission=tuple(() for _ in data.base.hadm_ids)))
    if name == "events_absent":
        return replace(data, events_by_admission=tuple(np.empty(0, dtype=np.int64) for _ in data.base.hadm_ids))
    if name == "prior_counts_absent":
        missing = ("context:prior_coded_admissions", "context:prior_strict_aecopd_admissions")
        if not set(missing).issubset(data.base.concept_names):
            raise ValueError("prior count concepts missing")
        return replace(data, base=replace(data.base, disabled_concepts=missing))
    if name == "medications_absent":
        return replace(data, events_by_admission=tuple(refs[data.event_is_lab[refs]] for refs in data.events_by_admission))
    raise ValueError(name)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    existing = json.loads((args.candidate_dir / "report.json").read_text(encoding="utf-8"))
    if existing["status"] != "training_completed" or existing["event_encoder"] != "set":
        raise ValueError("candidate is not a complete set-encoder run")
    data, _ = load_processed_event_data(
        args.aligned_dir, args.structured_dir, args.base_csv, args.processed_event_dir,
        args.raw_ledger_dir, labels_csv=args.labels_csv, field_dir=args.field_dir,
        vectors_path=args.vectors,
    )
    if (existing["label_sha256"] != data.base.label_sha256 or
            existing["field_manifest_sha256"] != data.base.manifest_sha256 or
            existing["raw_source_sha256"] != data.source_hashes or
            existing["text_vector_width"] != data.base.text_width):
        raise ValueError("candidate data contract differs from current inputs")
    model = EventSetFusionModel(
        concept_count=len(data.base.concept_names), type_count=len(data.base.type_names),
        field_count=len(FIELD_LABELS), text_width=data.base.text_width,
        event_concept_count=len(data.concept_names),
    )
    model.load_state_dict(torch.load(args.candidate_dir / "model.pt", map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rows = _rows(args.base_csv, data.base.hadm_ids)
    train_mask = data.base.splits == "train"
    x, feature_names, preprocessing = _features(rows, train_mask)
    cox = fit_competing_cox(x[train_mask], data.base.times[train_mask], data.base.events[train_mask], l2=10.0)
    if not cox.readmission.converged or not cox.death.converged:
        raise ValueError("simple Cox baseline failed to converge")
    sequences = {split: split_sequences(data.base, split) for split in ("validation", "test")}
    all_curves = {}
    results = {}
    for split, members in sequences.items():
        label = "historical_test" if split == "test" else split
        indices = np.concatenate(members)
        times, events = data.base.times[indices], data.base.events[indices]
        full = predict(model, data, members, process_only=False, batch_size=12, device=device)[0]
        _check_saved(args.candidate_dir / f"{label}_predictions.npz", data, indices, full)
        simple = predict_cumulative_incidence(cox, x[indices], GRID)[:, :, 1]
        curves = {"v5_full": metrics(times, events, full),
                  "five_feature_cox": metrics(times, events, simple)}
        variants = {"lab_values_zeroed": predict(model, data, members, process_only=True, batch_size=12, device=device)[0]}
        for name in ("text_absent", "events_absent", "medications_absent", "prior_counts_absent"):
            variants[name] = predict(model, _masked_data(data, name), members,
                                     process_only=False, batch_size=12, device=device)[0]
        for name, risk in variants.items():
            curves[name] = metrics(times, events, risk)
        all_curves[label] = curves
        subject_ids = data.base.subject_ids[indices]
        results[label] = {
            "models": {name: _public_metrics(detail) for name, detail in curves.items()},
            "subgroups_v5": _subgroups(data, rows, indices, full),
            "subgroups_five_feature_cox": _subgroups(data, rows, indices, simple),
            "calibration_v5": _calibration(times, events, full),
            "calibration_five_feature_cox": _calibration(times, events, simple),
            "paired_ibs_difference_v5_minus_comparator": {
                name: {"point": curves["v5_full"]["integrated_brier_7_365"] - detail["integrated_brier_7_365"],
                       "patient_cluster_95pct_interval": clustered_brier_interval(subject_ids, times, events, full,
                                                                                     simple if name == "five_feature_cox" else variants[name])}
                for name, detail in curves.items() if name != "v5_full"
            },
        }
        print(json.dumps({"split": label, "v5_mean_auc": curves["v5_full"]["mean_auc_7_365"],
                          "cox_mean_auc": curves["five_feature_cox"]["mean_auc_7_365"]}), flush=True)
    args.output_dir.mkdir(parents=True)
    _save_curves(args.output_dir, all_curves)
    report = {
        "status": "internal_validation_completed",
        "candidate_dir": str(args.candidate_dir),
        "label_sha256": data.base.label_sha256,
        "field_manifest_sha256": data.base.manifest_sha256,
        "source_sha256": data.source_hashes,
        "device": str(device),
        "simple_baseline": {"model": "two cause-specific Cox models with cumulative incidence",
                            "features": feature_names, "l2": 10.0, "preprocessing": preprocessing,
                            "coefficients_readmission": cox.readmission.coefficients.tolist(),
                            "coefficients_death": cox.death.coefficients.tolist(),
                            "fit_on": "training patients only"},
        "splits": results,
        "limitations": [
            "World A: no-event discharge observed to day 365; out-of-system readmissions and death unmeasured.",
            "AUC(t) uses observed strict readmission by day t versus all others, including prior competing death.",
            "Input masking is an out-of-distribution frozen-checkpoint sensitivity test, not a retraining ablation.",
            "Historical test patients were used in earlier model exploration, so this is not a fresh blind test.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned_dir", "structured_dir", "base_csv", "processed_event_dir", "raw_ledger_dir",
                 "field_dir", "vectors", "labels_csv", "candidate_dir", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/validate_vectors.py · lines 1–73

<!-- REVIEW_FRAGMENT {"path":"training/v5/validate_vectors.py","part":1,"start_line":1,"end_line":73,"source_sha256":"1d2e908a7cf0fc342583ec1a33cf9fd0dfffeab0f37f349912a62543d2075bec","fragment_sha256":"1d2e908a7cf0fc342583ec1a33cf9fd0dfffeab0f37f349912a62543d2075bec","added_terminal_newline":false} -->
````python
"""校验 V5 规则片段清单与冻结文本向量的一一对应关系。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def validate(manifest: Path, vectors_path: Path, state_path: Path) -> dict[str, object]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if state.get("status") != "complete" or state.get("manifest_sha256") != source_hash:
        raise ValueError("vector state is incomplete or belongs to a different manifest")
    vectors = np.load(vectors_path, mmap_mode="r")
    if vectors.ndim != 2 or vectors.shape[1] not in (768, 2560):
        raise ValueError("unexpected embedding matrix shape")
    if int(state.get("completed_rows", -1)) != vectors.shape[0]:
        raise ValueError("completed row count disagrees with vectors")
    seen_text: dict[str, int] = {}
    duplicates = 0
    rows = 0
    with gzip.open(manifest, "rt", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if int(row["embedding_row"]) != index:
                raise ValueError("non-contiguous manifest rows")
            digest = hashlib.sha256(row["chunk_text"].encode("utf-8")).hexdigest()
            if digest != row["text_sha256"]:
                raise ValueError("manifest text fingerprint mismatch")
            previous = seen_text.setdefault(digest, index)
            if previous != index:
                if not np.array_equal(vectors[previous], vectors[index]):
                    raise ValueError("same original text has different vectors")
                duplicates += 1
            rows += 1
    if rows != vectors.shape[0]:
        raise ValueError("manifest row count disagrees with vectors")
    norm_min, norm_max = float("inf"), float("-inf")
    for start in range(0, rows, 512):
        block = np.asarray(vectors[start:start + 512], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("non-finite text embedding")
        norms = np.linalg.norm(block, axis=1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
    if norm_min < 0.99 or norm_max > 1.01:
        raise ValueError("text vectors are not unit normalized")
    return {"passed": True, "manifest_sha256": source_hash, "rows": rows,
            "dimensions": int(vectors.shape[1]), "dtype": str(vectors.dtype),
            "unique_texts": len(seen_text), "duplicate_rows": duplicates,
            "l2_norm_min": norm_min, "l2_norm_max": norm_max}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.manifest, args.vectors, args.state)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/verify_completed_event_run.py · lines 1–77

<!-- REVIEW_FRAGMENT {"path":"training/v5/verify_completed_event_run.py","part":1,"start_line":1,"end_line":77,"source_sha256":"0f97875a51ea95f78044de657c15b6f260015f7467adbc672b227683bc4fb65d","fragment_sha256":"0f97875a51ea95f78044de657c15b6f260015f7467adbc672b227683bc4fb65d","added_terminal_newline":false} -->
````python
"""重载正式检查点并检查新字典、事件排列、药物 mask 和历史因果性。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from training.v5.event_data import EventBatch
from training.v5.event_set_model import EventSetFusionModel
from training.v5.independent_fusion import IndependentBatch


def verify(root: Path) -> dict[str, object]:
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    history = json.loads((root / "history.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "model_data_audit.json").read_text(encoding="utf-8"))
    if report["status"] != "training_completed" or len(history) != report["epochs_run"]:
        raise ValueError("正式训练报告或历史未完成")
    state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("正式检查点有非有限权重")
    configuration = {"concept_count": state["concept_embedding.weight"].shape[0] - 1,
                     "type_count": state["type_embedding.weight"].shape[0] - 1,
                     "field_count": state["field_embedding.weight"].shape[0],
                     "text_width": state["text_projection.weight"].shape[1],
                     "width": state["concept_embedding.weight"].shape[1],
                     "event_width": state["event_embedding.weight"].shape[1],
                     "event_concept_count": report["event_concept_count"]}
    model = EventSetFusionModel(**configuration).eval()
    model.load_state_dict(state, strict=True)
    if configuration["event_concept_count"] != len(audit["event_concepts"]):
        raise ValueError("模型概念字典与数据审计不一致")
    if sum(p.numel() for p in model.parameters()) != report["parameter_count"]:
        raise ValueError("模型参数数量与报告不一致")
    values = torch.zeros((1, 2, 1, 5))
    masks = torch.zeros_like(values)
    masks[..., 0] = 1
    base = IndependentBatch(values, masks, torch.ones((1, 2, 1), dtype=torch.long),
                            torch.ones((1, 2, 1), dtype=torch.long), torch.ones((1, 2, 1), dtype=torch.bool),
                            torch.ones((2, configuration["text_width"])) / configuration["text_width"]**0.5,
                            torch.tensor([[0, 0], [1, 0]]), torch.zeros((1, 2, 2)),
                            torch.ones((1, 2), dtype=torch.bool), torch.tensor([[5., 365.]]), torch.tensor([[1, 0]]))
    dictionary = report["event_concepts"]
    lab_id, med_id = dictionary.index("ph:arterial") + 1, dictionary.index("saba") + 1
    batch = EventBatch(base, torch.tensor([[0., .5, 0., 0.], [1., .5, 0., 0.], [0., .2, 0., 0.]]),
                       torch.tensor([lab_id, lab_id, med_id]), torch.tensor([[2, 1]]),
                       torch.tensor([True, True, False]), torch.tensor([True, True, False]))
    with torch.no_grad():
        before = model(batch).log_amplitude
        order = torch.tensor([1, 0, 2])
        reordered = EventBatch(base, batch.event_values[order], batch.event_concepts[order], batch.event_counts,
                               batch.event_is_lab[order], batch.event_numeric_mask[order])
        permutation_delta = float((model(reordered).log_amplitude - before).abs().max())
        batch.event_values[2, 0] = 999.
        masked_delta = float((model(batch).log_amplitude - before).abs().max())
        batch.event_values[2, 1] = 999.
        future_delta = float((model(batch).log_amplitude[0, 0] - before[0, 0]).abs().max())
    if permutation_delta > 1e-5 or masked_delta > 1e-6 or future_delta > 1e-6:
        raise ValueError("正式检查点未通过事件类型／排列／因果性检查")
    result = {"status": "checkpoint_reloaded_and_checked", "model_configuration": configuration,
              "same_timestamp_permutation_max_logit_difference": permutation_delta,
              "medication_numeric_mask_max_logit_difference": masked_delta,
              "future_event_change_previous_prediction_difference": future_delta,
              "fixture": "synthetic observations only; not a new cohort evaluation or clinical accuracy test",
              "checkpoint_sha256": hashlib.sha256((root / "model.pt").read_bytes()).hexdigest()}
    (root / "checkpoint_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir)))
````
<!-- END REVIEW_FRAGMENT -->

## training/v5/visualize.py · lines 1–189

<!-- REVIEW_FRAGMENT {"path":"training/v5/visualize.py","part":1,"start_line":1,"end_line":189,"source_sha256":"d989b5d2e452fed49fb21f852f9404869322dee90ad9aa09419a1fd0f1150cb7","fragment_sha256":"d989b5d2e452fed49fb21f852f9404869322dee90ad9aa09419a1fd0f1150cb7","added_terminal_newline":false} -->
````python
"""用已保存的 V5 权重补充患者聚类 AUC 区间、训练和校准图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.baselines.rank_metrics import auc_rank
from training.baselines.continuous_competing_cox import fit_competing_cox, predict_cumulative_incidence
from training.v5.data import load_data
from training.v5.model import V5Model
from training.v5.train import GRID, preprocess, predict, split_sequences


POINTS = (7, 14, 30, 60, 90, 180, 365)


def auc_intervals(subject_ids: np.ndarray, times: np.ndarray, events: np.ndarray,
                  risk: np.ndarray, *, bootstrap: int = 200) -> dict[str, dict[str, float | int | None]]:
    patients, inverse = np.unique(subject_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(len(patients))]
    rng = np.random.default_rng(2026)
    result: dict[str, dict[str, float | int | None]] = {}
    for day in POINTS:
        labels = ((events == 1) & (times <= day)).astype(np.int8)
        count = int(labels.sum())
        if min(count, len(labels) - count) < 20:
            result[str(day)] = {"cases": count, "auc": None, "lower": None, "upper": None}
            continue
        point = auc_rank(risk[:, day], labels)
        values: list[float] = []
        for _ in range(bootstrap):
            draw = np.concatenate([groups[index] for index in rng.integers(len(groups), size=len(groups))])
            if min(int(labels[draw].sum()), len(draw) - int(labels[draw].sum())) >= 20:
                values.append(auc_rank(risk[draw, day], labels[draw]))
        if len(values) < bootstrap // 2:
            raise ValueError("too few valid patient bootstrap samples")
        lower, upper = np.quantile(values, (0.025, 0.975))
        result[str(day)] = {"cases": count, "auc": float(point),
                            "lower": float(lower), "upper": float(upper)}
    return result


def calibration_points(times: np.ndarray, events: np.ndarray, risk: np.ndarray,
                       day: int, bins: int = 10) -> list[dict[str, float | int]]:
    order = np.argsort(risk[:, day], kind="stable")
    result: list[dict[str, float | int]] = []
    for indices in np.array_split(order, bins):
        observed = (events[indices] == 1) & (times[indices] <= day)
        result.append({"count": int(len(indices)), "predicted": float(risk[indices, day].mean()),
                       "observed_world_a": float(observed.mean())})
    return result


def run(structured_dir: Path, manifest: Path, vectors: Path,
        training_dir: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required to replay V5 checkpoints")
    training_report = json.loads((training_dir / "report.json").read_text(encoding="utf-8"))
    text_name = str(training_report.get("text_experiment", "bert"))
    text_model_name = f"structured_{text_name}"
    data, _ = preprocess(load_data(structured_dir, manifest, vectors,
                                   vector_width=int(training_report.get("text_input_width", 768))))
    training = split_sequences(data, "train")
    anchor_indices = np.concatenate(training)
    anchor_feature = data.group_values[0][:, 3:4].astype(np.float64)
    anchor = fit_competing_cox(anchor_feature[anchor_indices], data.times[anchor_indices],
                               data.events[anchor_indices], l2=10.0)
    report: dict[str, object] = {"bootstrap_replicates": 200,
                                 "confidence": "patient-cluster 95% pointwise intervals",
                                 "splits": {}}
    plot_data: dict[str, dict[str, np.ndarray]] = {}
    for split in ("validation", "historical_test"):
        sequences = split_sequences(data, "test" if split == "historical_test" else split)
        indices = np.concatenate(sequences)
        risks: dict[str, np.ndarray] = {
            "prior_aecopd_cox": predict_cumulative_incidence(
                anchor, anchor_feature[indices], GRID)[:, :, 1]
        }
        for name in ("structured", text_model_name):
            model = V5Model(text_width=data.text_width if name == text_model_name else 768,
                            balance_text=bool(training_report.get("balanced_text_content", False))
                            if name == text_model_name else False).to("cuda")
            state = torch.load(training_dir / f"{name}.pt", map_location="cuda",
                               weights_only=True)
            model.load_state_dict(state)
            risks[name] = predict(model, data, sequences, include_text=name == text_model_name,
                                  device=torch.device("cuda"), batch_size=24)[0]
        plot_data[split] = risks
        split_report: dict[str, object] = {}
        for name, risk in risks.items():
            split_report[name] = {
                "auc_points": auc_intervals(data.subject_ids[indices], data.times[indices],
                                             data.events[indices], risk),
                "calibration": {str(day): calibration_points(
                    data.times[indices], data.events[indices], risk, day
                ) for day in (30, 90, 180, 365)},
            }
        report["splits"][split] = split_report
    output_dir.mkdir(parents=True)
    (output_dir / "aggregate_uncertainty_calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plots(training_dir, output_dir, plot_data, report, text_model_name)
    return report


def _plots(training_dir: Path, output_dir: Path,
           risks: dict[str, dict[str, np.ndarray]], report: dict[str, object],
           text_model_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histories = {name: json.loads((training_dir / f"{name}_history.json").read_text(encoding="utf-8"))
                 for name in ("structured", text_model_name)}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, history in histories.items():
        axes[0].plot([row["epoch"] for row in history],
                     [row["train_nll"] for row in history], label=f"{name} train")
        axes[0].plot([row["epoch"] for row in history],
                     [row["validation_nll"] for row in history], linestyle="--",
                     label=f"{name} validation")
        axes[1].plot([row["epoch"] for row in history],
                     [row["validation_integrated_brier"] for row in history], label=name)
    axes[0].set(xlabel="Epoch", ylabel="Negative log likelihood", title="Training and validation")
    axes[1].set(xlabel="Epoch", ylabel="Integrated Brier, days 7–365",
                title="Validation checkpoint selection")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=170)
    plt.close(figure)

    colors = {"prior_aecopd_cox": "#b36b00", "structured": "#1261a0",
              text_model_name: "#258b5a"}
    for split, models in risks.items():
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for name, risk in models.items():
            points = report["splits"][split][name]["auc_points"]
            x = np.array([day for day in POINTS if points[str(day)]["auc"] is not None])
            y = np.array([points[str(day)]["auc"] for day in x])
            lower = np.array([points[str(day)]["lower"] for day in x])
            upper = np.array([points[str(day)]["upper"] for day in x])
            axes[0].errorbar(x, y, yerr=np.stack((y - lower, upper - y)),
                             marker="o", capsize=2, linewidth=1.4, markersize=3,
                             label=name, color=colors[name])
            day = 365
            calibration = report["splits"][split][name]["calibration"][str(day)]
            axes[1].plot([item["predicted"] for item in calibration],
                         [item["observed_world_a"] for item in calibration],
                         marker="o", label=name, color=colors[name])
        axes[0].set(xlabel="Days after discharge", ylabel="Cumulative competing-risk AUC(t)",
                    title=f"{split}: patient-cluster 95% CI")
        axes[1].plot([0, 0.7], [0, 0.7], color="gray", linestyle="--")
        axes[1].set(xlabel="Predicted 365-day readmission risk",
                    ylabel="Observed fraction (World A)", title=f"{split}: calibration deciles")
        for axis in axes:
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output_dir / f"{split}_auc_ci_calibration.png", dpi=170)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--text-manifest", type=Path, required=True)
    parser.add_argument("--text-vectors", "--bert-vectors", dest="text_vectors", type=Path,
                        required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.structured_dir, args.text_manifest, args.text_vectors,
                 args.training_dir, args.output_dir)
    print(json.dumps({"outputs": str(args.output_dir), "splits": list(result["splits"])},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->
