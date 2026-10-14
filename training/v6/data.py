"""V6 数据桥接：原始历史、本次编码、独立严格标签、B 四字段冻结向量。"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import numpy as np

from patient_data.export_mimiciv import rows, sha256
from patient_data.preprocess_events import CHRONIC, ACUTE
from patient_data.v6_lab_state import AXES, MED_GROUPS
from training.v5.clinical_concepts import diagnosis_groups
from training.v5.field_text_baseline import FIELD_LABELS
from training.v5.numeric_preprocessing import robust_parameters
from training.v5.strict_endpoint_labels import checked_strict_labels, STRICT_PRIMARY_EVENT

HISTORY_NAMES = ("age", "female", "log_prior_completed_copd", "log_prior_completed_strict",
                 "log_previous_discharge_gap_days", "no_completed_history")
CURRENT_NAMES = ("current_strict_code", *("current_code:" + c for c in CHRONIC + ACUTE),
                 "disposition_home", "disposition_facility", "disposition_other")


def disposition(value: str) -> tuple[np.ndarray, np.ndarray]:
    if value in ("", "\\N"):
        return np.zeros(3), np.zeros(3, dtype=bool)
    name = value.strip().upper()
    category = 0 if "HOME" in name else 1 if any(t in name for t in ("REHAB", "SKILLED", "NURSING", "LONG TERM")) else 2
    vector = np.zeros(3); vector[category] = 1
    return vector, np.ones(3, dtype=bool)


def history_for(row: dict, previous: list[dict], patient: dict) -> np.ndarray:
    start = datetime.fromisoformat(row["admittime"])
    past = [p for p in previous if datetime.fromisoformat(p["admittime"]) < start
            and datetime.fromisoformat(p["admittime"]) < datetime.fromisoformat(p["dischtime"]) <= start]
    gender = patient["gender"]
    if gender not in ("M", "F"):
        raise ValueError("baseline gender unavailable; do not silently impute a sex")
    age = float(patient["anchor_age"]) + start.year - int(patient["anchor_year"])
    if not np.isfinite(age) or age <= 0:
        raise ValueError("invalid age")
    gap = (start - max(datetime.fromisoformat(p["dischtime"]) for p in past)).total_seconds() / 86400 if past else 0.
    return np.array([age, float(gender == "F"), np.log1p(len(past)),
                     np.log1p(sum(p["is_strict_aecopd"] == "t" for p in past)),
                     np.log1p(gap), float(not past)])


@dataclass(frozen=True)
class V6Data:
    arrays: dict[str, np.ndarray]
    history: np.ndarray
    history_raw: np.ndarray
    current: np.ndarray
    current_masks: np.ndarray
    times: np.ndarray
    events: np.ndarray
    text_vectors: np.ndarray | None
    text_refs: tuple[tuple[tuple[int, int], ...], ...]
    normalization: dict
    provenance: dict

    def collate(self, positions: np.ndarray, device='cpu'):
        import torch
        from training.v6.model import ClinicalBatch
        positions = np.asarray(positions, dtype=np.int64)
        if (positions.ndim != 1 or not len(positions) or len(set(positions.tolist())) != len(positions)
                or (positions < 0).any() or (positions >= len(self.history)).any()):
            raise ValueError("invalid admission batch positions")
        vectors, index = [], []
        for slot, position in enumerate(positions):
            for field, vector_row in self.text_refs[position]:
                vectors.append(self.text_vectors[vector_row])
                index.append((slot, field))
        width = 768 if self.text_vectors is None else self.text_vectors.shape[1]
        tensor = lambda x: torch.as_tensor(np.asarray(x), device=device)
        text = np.asarray(vectors, dtype=np.float32).reshape(-1, width)
        batch = ClinicalBatch(tensor(self.current[positions]), tensor(self.current_masks[positions]),
                              tensor(self.arrays["lab_axis_inputs"][positions]),
                              tensor(self.arrays["med_exposure"][positions]),
                              tensor(np.ones((len(positions), 7), dtype=bool)),
                              tensor(text), tensor(np.asarray(index, dtype=np.int64).reshape(-1, 2)))
        # 标签只返回给损失调用，不成为 ClinicalBatch 或 encoder 的字段。
        return tensor(self.history[positions]), batch, tensor(self.times[positions]), tensor(self.events[positions])


def load(snapshot: Path, state_dir: Path, labels_path: Path, *, field_dir: Path | None = None,
         vectors_path: Path | None = None) -> V6Data:
    if (field_dir is None) != (vectors_path is None):
        raise ValueError("text manifest and vectors must be supplied together")
    state_path = state_dir / "manifest.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["status"] != "derived_and_roundtrip_checked_not_prediction_validated":
        raise ValueError("unfinished clinical state")
    for name, digest in state["output_sha256"].items():
        if sha256(state_dir / name) != digest:
            raise ValueError("state output changed")
    with np.load(state_dir / "clinical_state.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    if tuple(arrays["axis_names"]) != AXES or tuple(arrays["med_group_names"]) != tuple(MED_GROUPS):
        raise ValueError("state vocabulary changed")
    sm_path = snapshot / "metadata/manifest.json"
    sm = json.loads(sm_path.read_text(encoding="utf-8"))
    if sm["status"] != "validated" or sha256(sm_path) != state["source_manifest_sha256"]:
        raise ValueError("state/raw manifest mismatch")
    reconstructed, raw_hashes = checked_strict_labels(snapshot)
    for name in ("cohort/admissions", "cohort/patient_splits", "raw/patients", "raw/diagnoses_icd"):
        digest = sha256(snapshot / (name + ".csv.gz"))
        if digest != sm["sources"][name]["sha256"]:
            raise ValueError("raw context source changed")
        if name in state["source_hashes"] and digest != state["source_hashes"][name]:
            raise ValueError("context/state lineage mismatch")
        raw_hashes[name] = digest
    hadms = [str(h) for h in arrays["hadm_ids"]]
    if len(hadms) != 19401 or len(set(hadms)) != len(hadms) or set(hadms) != set(reconstructed):
        raise ValueError("admission membership mismatch")
    qa = json.loads((labels_path.parent / "summary.json").read_text(encoding="utf-8"))
    label_hash = sha256(labels_path)
    if (qa["status"] != "strict_labels_reconstructed_and_checked" or qa["label_sha256"] != label_hash
            or qa["primary_event"] != STRICT_PRIMARY_EVENT):
        raise ValueError("wrong or changed endpoint")
    label_rows = list(rows(labels_path))
    labels = {r["hadm_id"]: r for r in label_rows}
    if len(labels) != len(label_rows) or set(labels) != set(hadms):
        raise ValueError("label membership mismatch")
    for h, label in labels.items():
        expected = reconstructed[h]
        if (label["subject_id"] != expected["subject_id"] or int(label["event_type"]) != expected["event_type"]
                or not np.isclose(float(label["time_days"]), expected["time_days"], atol=1e-8, rtol=0)):
            raise ValueError("strict label reconstruction mismatch")
    cohort_rows = list(rows(snapshot / "cohort/admissions.csv.gz"))
    cohort = {r["hadm_id"]: r for r in cohort_rows}
    patients = {r["subject_id"]: r for r in rows(snapshot / "raw/patients.csv.gz")}
    split_rows = list(rows(snapshot / "cohort/patient_splits.csv.gz"))
    splits = {r["subject_id"]: r["patient_split"] for r in split_rows}
    if len(splits) != len(split_rows):
        raise ValueError("duplicate patient split")
    diagnoses = defaultdict(set)
    for r in rows(snapshot / "raw/diagnoses_icd.csv.gz"):
        diagnoses[r["hadm_id"]].update(diagnosis_groups(r["icd_code"], int(r["icd_version"])))
    histories, previous = {}, defaultdict(list)
    for r in sorted(cohort_rows, key=lambda r: (int(r["subject_id"]), r["admittime"], int(r["hadm_id"]))):
        if r["hadm_id"] in reconstructed:
            histories[r["hadm_id"]] = history_for(r, previous[r["subject_id"]], patients[r["subject_id"]])
        previous[r["subject_id"]].append(r)
    history_raw = np.stack([histories[h] for h in hadms])
    history = history_raw.copy()
    normalization = {}
    for c in (0, 2, 3, 4):
        # 无历史者 gap 不可估计；不把占位0用于拟合真实间隔。
        valid = np.ones(len(hadms), dtype=bool) if c != 4 else history_raw[:, 5] == 0
        train = (arrays["patient_splits"] == "train") & valid
        p = robust_parameters(history_raw[train, c])
        history[:, c] = np.where(valid, np.clip((history[:, c] - p["median"]) / p["scale"], -8, 8), 0)
        normalization[HISTORY_NAMES[c]] = p
    current, masks = np.zeros((len(hadms), 14), dtype=np.float32), np.ones((len(hadms), 14), dtype=bool)
    for i, h in enumerate(hadms):
        r = cohort[h]
        if (int(r["subject_id"]) != arrays["subject_ids"][i]
                or splits[r["subject_id"]] != arrays["patient_splits"][i]):
            raise ValueError("patient identity/split mismatch")
        current[i, :11] = [r["is_strict_aecopd"] == "t", *(c in diagnoses[h] for c in CHRONIC + ACUTE)]
        current[i, 11:], masks[i, 11:] = disposition(r["discharge_location"])
    refs = [[] for _ in hadms]
    vectors, text_provenance = None, {"encoder": "none", "text_admissions": 0}
    if field_dir is not None:
        shared = field_dir / "shared_text.csv.gz"
        shared_hash = sha256(shared)
        summary = json.loads((field_dir / "summary.json").read_text(encoding="utf-8"))
        validation = json.loads((field_dir / "validation.json").read_text(encoding="utf-8"))
        encoding = json.loads((vectors_path.parent / "state.json").read_text(encoding="utf-8"))
        if (tuple(summary["fields"]) != tuple(FIELD_LABELS) or summary["manifest_sha256"] != shared_hash
                or validation["status"] != "field_text_manifest_validated" or validation["manifest_sha256"] != shared_hash
                or encoding["status"] != "complete" or encoding["manifest_sha256"] != shared_hash):
            raise ValueError("text source/encoding contract mismatch")
        vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
        if vectors.ndim != 2 or vectors.shape[1] not in (768, 2560):
            raise ValueError("unknown encoder dimension")
        if ((vectors.shape[1] == 768 and encoding.get("model") != "Bio_ClinicalBERT")
                or (vectors.shape[1] == 2560 and encoding.get("encoder") != "Qwen3-Embedding-4B")):
            raise ValueError("encoder identity mismatch")
        by_hadm = {h: i for i, h in enumerate(hadms)}
        fields = {f: i for i, f in enumerate(FIELD_LABELS)}
        count = 0
        for count, row in enumerate(rows(shared), start=1):
            h = row["hadm_id"]
            if (h not in by_hadm or row["subject_id"] != cohort[h]["subject_id"]
                    or int(row["embedding_row"]) != count - 1 or row["field"] not in fields):
                raise ValueError("text index mismatch")
            refs[by_hadm[h]].append((fields[row["field"]], count - 1))
        if count != len(vectors):
            raise ValueError("text row count mismatch")
        for start in range(0, len(vectors), 8192):
            if not np.isfinite(vectors[start:start + 8192]).all():
                raise ValueError("nonfinite frozen text vectors")
        text_provenance = {"encoder": encoding.get("model", encoding.get("encoder")),
                           "manifest_sha256": shared_hash, "vectors_sha256": sha256(vectors_path),
                           "vectors_path": str(vectors_path), "rows": count, "width": vectors.shape[1],
                           "text_admissions": sum(bool(r) for r in refs)}
    return V6Data(arrays, history.astype(np.float32), history_raw, current, masks,
                  np.asarray([float(labels[h]["time_days"]) for h in hadms], dtype=np.float32),
                  np.asarray([int(labels[h]["event_type"]) for h in hadms], dtype=np.int64), vectors,
                  tuple(tuple(r) for r in refs), normalization,
                  {"state_manifest_sha256": sha256(state_path), "label_sha256": label_hash,
                   "raw_context_hashes": raw_hashes, "text": text_provenance})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'state_dir', 'labels', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--field-dir', type=Path)
    parser.add_argument('--vectors-path', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = load(args.snapshot, args.state_dir, args.labels, field_dir=args.field_dir, vectors_path=args.vectors_path)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'context.npz', hadm_ids=data.arrays['hadm_ids'],
                        history_inputs=data.history, history_raw=data.history_raw, current_values=data.current,
                        current_masks=data.current_masks, history_names=np.asarray(HISTORY_NAMES),
                        current_names=np.asarray(CURRENT_NAMES))
    report = {"status": "data_contract_checked_not_trained", "admissions": len(data.history),
              "patients": len(set(data.arrays['subject_ids'].tolist())),
              "unknown_disposition_admissions": int((~data.current_masks[:, 11]).sum()),
              "no_completed_history": int(data.history_raw[:, 5].sum()),
              "labels": dict(Counter(int(e) for e in data.events)),
              "normalization": data.normalization, "provenance": data.provenance,
              "context_sha256": sha256(args.output / 'context.npz'),
              "labels_in_context_file": False,
              "med_source_mask_meaning": "mapping extraction available; binary zero means no retained eMAR evidence, not untreated",
              "history_scope": "all prior completed positive-duration retained COPD admissions; no future admissions"}
    (args.output / 'input_contract.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'admissions', 'patients', 'unknown_disposition_admissions', 'no_completed_history')}))


if __name__ == '__main__':
    main()
