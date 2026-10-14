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
