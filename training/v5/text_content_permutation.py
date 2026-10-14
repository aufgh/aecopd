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
