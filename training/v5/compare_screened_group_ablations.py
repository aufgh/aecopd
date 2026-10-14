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
