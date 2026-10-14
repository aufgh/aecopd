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
