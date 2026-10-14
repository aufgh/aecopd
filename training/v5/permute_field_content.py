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
