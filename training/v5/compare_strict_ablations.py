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
