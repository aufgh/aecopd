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
