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
