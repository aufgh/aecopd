"""从已校验原始底账建立可追溯的静态输入与原始事件，不训练模型。"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# 允许直接用 python patient_data/preprocess_events.py 启动。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patient_data.export_mimiciv import NULL, patient_split, rows, sha256
from training.v5.clinical_concepts import MED_NAMES, diagnosis_groups, medication_classes
from training.v5.numeric_preprocessing import robust_parameters


LABS = {
    "50818": ("pco2", "mm Hg"), "50820": ("ph", "units"),
    "50821": ("po2", "mm Hg"), "50862": ("albumin", "g/dL"),
    "50882": ("bicarbonate", "mEq/L"), "50912": ("creatinine", "mg/dL"),
    "51222": ("hemoglobin", "g/dL"), "51301": ("wbc", "K/uL"),
}
BLOOD_GAS = {"50818", "50820", "50821"}
SPECIMENS = ("arterial", "venous", "mixed", "central_venous", "unknown", "conflict")
CHRONIC = ("heart_failure", "cardiac_other", "ckd", "diabetes", "malignancy", "chronic_respiratory")
ACUTE = ("acute_resp_failure", "pneumonia_or_aspiration", "sepsis", "aki")
ADMINISTERED = {"Administered", "Started", "Applied", "Delayed Administered"}
NON_SYSTEMIC_ROUTES = {"TP", "TOP", "TOPICAL", "OP", "OU", "OD", "OS", "OT", "AU", "AD", "AS", "NAS", "NASAL"}
EVENT_FIELDS = ("subject_id", "hadm_id", "patient_split", "kind", "concept", "specimen_type",
                "source_id", "specimen_id", "charttime", "storetime", "elapsed_days",
                "value_raw", "value_unit", "value_normalized", "numeric_mask", "clipped")
REJECT_FIELDS = ("source_table", "source_id", "reason")


@dataclass(frozen=True)
class Admission:
    subject: str
    hadm: str
    start: datetime
    end: datetime
    split: str
    strict: bool


@dataclass(frozen=True)
class Event:
    hadm: str
    kind: str
    concept: str
    specimen_type: str
    source_id: str
    specimen_id: str
    moment: datetime
    stored: datetime
    value: float | None
    unit: str


def csv_writer(path: Path, fields: tuple[str, ...]):
    handle = gzip.open(path, "wt", encoding="utf-8", newline="")
    writer = csv.writer(handle)
    writer.writerow(fields)
    return handle, writer


def parse_time(value: str) -> datetime | None:
    if value in (NULL, ""):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def temporal_reason(row: dict[str, str], admission: Admission) -> str | None:
    moment, stored = parse_time(row["charttime"]), parse_time(row["storetime"])
    if moment is None or stored is None:
        return "missing_or_invalid_timestamp"
    if not admission.start <= moment <= admission.end:
        return "outside_admission"
    if stored > admission.end:
        return "unavailable_at_discharge"
    if stored < moment:
        return "storetime_before_charttime"
    return None


def specimen_name(value: str) -> str:
    value = " ".join(value.strip().upper().rstrip(".").split())
    return {"ART": "arterial", "VEN": "venous", "MIX": "mixed",
            "CENTRAL VENOUS": "central_venous"}.get(value, "unknown")


def specimen_type(candidates: list[tuple[str, datetime, datetime]], discharge: datetime) -> str:
    # 不能从出院之后记录的标本类型回填出院输入。
    available = {name for name, moment, stored in candidates if moment <= stored <= discharge}
    if len(available) > 1:
        return "conflict"
    return next(iter(available), "unknown")


def accepted_lab(row: dict[str, str], admission: Admission,
                 specimen: str) -> tuple[Event | None, str | None]:
    reason = temporal_reason(row, admission)
    if reason:
        return None, reason
    name, unit = LABS[row["itemid"]]
    if row["valueuom"].strip() != unit:
        return None, "unit_mismatch_or_missing"
    try:
        value = float(row["valuenum"])
    except ValueError:
        return None, "missing_or_nonnumeric_value"
    if not math.isfinite(value) or value <= 0:
        return None, "nonfinite_or_nonpositive_value"
    concept = f"{name}:{specimen}" if row["itemid"] in BLOOD_GAS else name
    return Event(admission.hadm, "lab", concept, specimen, row["labevent_id"],
                 row["specimen_id"], parse_time(row["charttime"]), parse_time(row["storetime"]), value, unit), None


def fit_lab_parameters(events: list[Event], admissions: dict[str, Admission]) -> dict[str, dict[str, object]]:
    observed: dict[str, list[float]] = defaultdict(list)
    concepts = {e.concept for e in events if e.kind == "lab"}
    for event in events:
        if event.kind == "lab" and admissions[event.hadm].split == "train":
            observed[event.concept].append(event.value)
    result = {}
    for concept in sorted(concepts):
        if observed[concept]:
            result[concept] = robust_parameters(np.asarray(observed[concept]))
        else:
            # 保留观测，但无训练支持时不把验证／测试的分布用于拟合。
            result[concept] = {"median": 0.0, "scale": 1.0, "method": "no_training_support",
                               "training_observations": 0, "iqr": 0.0}
    return result


def normalized(value: float, parameters: dict[str, object]) -> tuple[float, int, int]:
    if parameters["method"] == "no_training_support":
        return 0.0, 0, 0
    z = (value - float(parameters["median"])) / float(parameters["scale"])
    return float(np.clip(z, -8.0, 8.0)), 1, int(abs(z) > 8.0)


def verify_sources(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "metadata/manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "validated":
        raise ValueError("原始底账尚未通过校验")
    used = ("cohort/admissions", "cohort/patient_splits", "raw/patients", "raw/labevents",
            "raw/diagnoses_icd", "raw/emar", "raw/emar_detail", "outcomes/readmission_death_labels")
    for name in used:
        if sha256(root / (name + ".csv.gz")) != manifest["sources"][name]["sha256"]:
            raise ValueError(f"原始底账文件变化：{name}")
    return {name: manifest["sources"][name]["sha256"] for name in used}


def static_features(root: Path, admissions: dict[str, Admission],
                    diagnoses: dict[str, set[str]], output: Path) -> dict[str, object]:
    patients = {r["subject_id"]: r for r in rows(root / "raw/patients.csv.gz")}
    grouped: dict[str, list[Admission]] = defaultdict(list)
    for admission in admissions.values():
        grouped[admission.subject].append(admission)
    names = ("age", "female", "prior_completed_copd_admissions", "prior_completed_strict_admissions",
             "current_strict_code", *(f"current_code:{c}" for c in CHRONIC + ACUTE),
             *(f"prior_code:{c}" for c in CHRONIC))
    continuous = {"age", "prior_completed_copd_admissions", "prior_completed_strict_admissions"}
    records: dict[str, tuple[list[float], list[int]]] = {}
    for patient_admissions in grouped.values():
        patient_admissions.sort(key=lambda a: (a.start, int(a.hadm)))
        previous: list[Admission] = []
        for admission in patient_admissions:
            past = [p for p in previous if p.end <= admission.start]
            history = set().union(*(diagnoses[p.hadm] & set(CHRONIC) for p in past))
            patient = patients[admission.subject]
            age = float(patient["anchor_age"]) + admission.start.year - int(patient["anchor_year"])
            gender = patient["gender"]
            values = [age, float(gender == "F"), float(len(past)), float(sum(p.strict for p in past)),
                      float(admission.strict), *(float(c in diagnoses[admission.hadm]) for c in CHRONIC + ACUTE),
                      *(float(c in history) for c in CHRONIC)]
            masks = [1, int(gender in ("M", "F")), *([1] * (len(names) - 2))]
            records[admission.hadm] = values, masks
            previous.append(admission)
    parameters = {}
    for index, name in enumerate(names):
        if name in continuous:
            observed = [values[index] for hadm, (values, masks) in records.items()
                        if admissions[hadm].split == "train" and masks[index]]
            parameters[name] = robust_parameters(np.asarray(observed))
    handle, writer = csv_writer(output / "static_features.csv.gz",
                                ("subject_id", "hadm_id", "patient_split", "feature", "value_raw", "value_normalized", "mask", "clipped"))
    try:
        for hadm in sorted(records, key=int):
            admission = admissions[hadm]
            values, masks = records[hadm]
            for index, name in enumerate(names):
                z, clipped = values[index], 0
                if name in parameters:
                    z, _, clipped = normalized(values[index], parameters[name])
                writer.writerow((admission.subject, hadm, admission.split, name, values[index],
                                 z if masks[index] else 0.0, masks[index], clipped))
    finally:
        handle.close()
    return {"feature_names": names, "continuous_parameters": parameters,
            "history_scope": "earlier eligible COPD-coded admissions completed before current admission",
            "diagnosis_zero_means": "no recorded code, not proof of no disease"}


def preprocess(root: Path, output: Path, *, medication_mapping: str = "legacy") -> dict[str, object]:
    if medication_mapping not in ("legacy", "gold_review_v2", "clinical_reviewed_2026_09"):
        raise ValueError("unknown medication mapping version")
    if output.exists():
        raise FileExistsError("拒绝覆盖已有预处理目录")
    input_hashes = verify_sources(root)
    cohort = list(rows(root / "cohort/admissions.csv.gz"))
    eligible_rows = [r for r in cohort if r["discharge_landmark_eligible"] == "t"]
    labels = {r["hadm_id"]: r["subject_id"] for r in rows(root / "outcomes/readmission_death_labels.csv.gz")}
    split_records = {r["subject_id"]: r["patient_split"] for r in rows(root / "cohort/patient_splits.csv.gz")}
    admissions = {}
    for row in eligible_rows:
        subject, hadm = row["subject_id"], row["hadm_id"]
        if labels.get(hadm) != subject or split_records[subject] != patient_split(subject):
            raise ValueError("成员、标签或患者划分不一致")
        admissions[hadm] = Admission(subject, hadm, datetime.fromisoformat(row["admittime"]),
                                    datetime.fromisoformat(row["dischtime"]), split_records[subject], row["is_strict_aecopd"] == "t")
    if set(labels) != set(admissions) or len(admissions) != len(eligible_rows):
        raise ValueError("合格出院成员有重复或标签不完整")
    output.mkdir(parents=True)
    (output / "status.json").write_text('{"status":"processing"}', encoding="utf-8")
    try:
        return _build(root, output, admissions, input_hashes, medication_mapping)
    except Exception:
        (output / "status.json").write_text('{"status":"failed_preserved_for_audit"}', encoding="utf-8")
        raise


def _build(root: Path, output: Path, admissions: dict[str, Admission],
           input_hashes: dict[str, object], medication_mapping: str) -> dict[str, object]:
    specimen_records: dict[tuple[str, str], list[tuple[str, datetime, datetime]]] = defaultdict(list)
    selected_labs: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for row in rows(root / "raw/labevents.csv.gz"):
        if row["hadm_id"] not in admissions:
            continue
        if row["itemid"] == "52033" and row["value"] not in (NULL, ""):
            moment, stored = parse_time(row["charttime"]), parse_time(row["storetime"])
            if moment is not None and stored is not None:
                specimen_records[(row["subject_id"], row["specimen_id"])].append((specimen_name(row["value"]), moment, stored))
        elif row["itemid"] in LABS:
            selected_labs.append(row)
    rejection_handle, rejection_writer = csv_writer(output / "excluded_records.csv.gz", REJECT_FIELDS)
    events: list[Event] = []
    for row in selected_labs:
        admission = admissions[row["hadm_id"]]
        specimen = specimen_type(specimen_records[(admission.subject, row["specimen_id"])], admission.end) if row["itemid"] in BLOOD_GAS else "not_applicable"
        event, reason = accepted_lab(row, admission, specimen)
        counts["selected_lab_records"] += 1
        if reason:
            counts[f"lab_excluded:{reason}"] += 1
            rejection_writer.writerow(("labevents", row["labevent_id"], reason))
        else:
            events.append(event)
    del selected_labs, specimen_records
    print(json.dumps({"stage": "labs_checked", "accepted": len(events), "counts": counts}), flush=True)
    medication_records: dict[tuple[str, str], tuple[dict[str, str], set[str]]] = {}
    for row in rows(root / "raw/emar.csv.gz"):
        if row["hadm_id"] not in admissions:
            continue
        counts["emar_records_reviewed"] += 1
        categories = (medication_classes(row["medication"], mapping_version=medication_mapping)
                      if row["medication"] != NULL else set())
        if not categories:
            counts["emar_outside_existing_categories"] += 1
            continue
        reason = temporal_reason(row, admissions[row["hadm_id"]])
        if row["event_txt"] not in ADMINISTERED:
            reason = "not_confirmed_administration"
        if reason:
            counts[f"emar_excluded:{reason}"] += 1
            rejection_writer.writerow(("emar", row["emar_id"] + ":" + row["emar_seq"], reason))
        else:
            medication_records[(row["emar_id"], row["emar_seq"])] = row, categories
    routes: dict[tuple[str, str], set[str]] = defaultdict(set)
    product_descriptions: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows(root / "raw/emar_detail.csv.gz"):
        key = row["emar_id"], row["emar_seq"]
        if key in medication_records and row["route"] not in (NULL, ""):
            routes[key].add(row["route"].strip().upper())
        if key in medication_records and row.get("product_description", NULL) not in (NULL, ""):
            product_descriptions[key].add(row["product_description"].strip())
    for key, (row, _candidate_categories) in medication_records.items():
        route_values = routes[key]
        # 途径仅用于排除明显不属于全身/吸入治疗的制剂，不入模。
        if route_values and route_values.issubset(NON_SYSTEMIC_ROUTES):
            counts["emar_excluded:non_systemic_route"] += 1
            rejection_writer.writerow(("emar", ":".join(key), "non_systemic_route"))
            continue
        categories = medication_classes(row["medication"],
                                        product_descriptions=tuple(product_descriptions[key]),
                                        routes=tuple(route_values), mapping_version=medication_mapping)
        if not categories:
            counts["emar_excluded:non_target_formulation"] += 1
            rejection_writer.writerow(("emar", ":".join(key), "non_target_formulation"))
            continue
        if not route_values:
            counts["accepted_emar_route_unknown"] += 1
        for category in sorted(categories):
            events.append(Event(row["hadm_id"], "medication", category, "not_applicable", ":".join(key), NULL,
                                parse_time(row["charttime"]), parse_time(row["storetime"]), None, "not_applicable"))
    rejection_handle.close()
    del medication_records, routes, product_descriptions
    diagnoses: dict[str, set[str]] = defaultdict(set)
    for row in rows(root / "raw/diagnoses_icd.csv.gz"):
        if row["hadm_id"] in admissions:
            diagnoses[row["hadm_id"]].update(diagnosis_groups(row["icd_code"], int(row["icd_version"])))
    static = static_features(root, admissions, diagnoses, output)
    lab_parameters = fit_lab_parameters(events, admissions)
    ordered_admissions = sorted(admissions.values(), key=lambda a: (int(a.subject), a.start, int(a.hadm)))
    admission_index = {a.hadm: index for index, a in enumerate(ordered_admissions)}
    # 排序只是存储规则，张量中无行号位置、全局 gap 或距出院时间。
    events.sort(key=lambda e: (admission_index[e.hadm], e.moment, e.concept, e.source_id))
    concepts = tuple(sorted({e.concept for e in events}))
    concept_index = {name: index + 1 for index, name in enumerate(concepts)}
    n = len(events)
    values = np.zeros(n, dtype=np.float32)
    numeric_mask = np.zeros(n, dtype=bool)
    elapsed = np.zeros(n, dtype=np.float32)
    ids = np.zeros(n, dtype=np.int32)
    per_admission = np.zeros(len(admissions), dtype=np.int64)
    coverage: dict[str, set[str]] = defaultdict(set)
    concept_counts: Counter[str] = Counter()
    clipped: Counter[str] = Counter()
    unsupported: Counter[str] = Counter()
    handle, writer = csv_writer(output / "events.csv.gz", EVENT_FIELDS)
    seen: set[tuple[str, str]] = set()
    try:
        for index, event in enumerate(events):
            key = event.concept, event.source_id
            if key in seen:
                raise ValueError("同一概念原始记录重复；不自动删掉同时间戳不同标本")
            seen.add(key)
            admission = admissions[event.hadm]
            days = (event.moment - admission.start).total_seconds() / 86400
            z, mask, is_clipped = normalized(event.value, lab_parameters[event.concept]) if event.kind == "lab" else (0.0, 0, 0)
            values[index], numeric_mask[index] = z, bool(mask)
            elapsed[index], ids[index] = math.log1p(days), concept_index[event.concept]
            per_admission[admission_index[event.hadm]] += 1
            concept_counts[event.concept] += 1
            coverage[event.kind].add(event.hadm)
            clipped[event.concept] += is_clipped
            if event.kind == "lab" and not mask:
                unsupported[event.concept] += 1
            writer.writerow((admission.subject, event.hadm, admission.split, event.kind, event.concept,
                             event.specimen_type, event.source_id, event.specimen_id,
                             event.moment.isoformat(sep=" "), event.stored.isoformat(sep=" "), days,
                             event.value if event.value is not None else NULL, event.unit, z, mask, is_clipped))
    finally:
        handle.close()
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(per_admission)))
    np.savez_compressed(output / "event_arrays.npz", hadm_ids=np.array([int(a.hadm) for a in ordered_admissions]),
                        subject_ids=np.array([int(a.subject) for a in ordered_admissions]),
                        offsets=offsets, concept_ids=ids, values=values, numeric_mask=numeric_mask,
                        elapsed_log_days=elapsed)
    normalization = {"lab": lab_parameters, "static": static, "clip_range": [-8, 8],
                     "fit_scope": "training patients only; lab parameters weighted by observed events",
                     "unknown_specimen": "separate concept, never inferred arterial",
                     "empty_event_admission": "zero-length segment, not a zero-valued observation"}
    (output / "normalization.json").write_text(json.dumps(normalization, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {"status": "preprocessed_and_checked", "patients": len({a.subject for a in admissions.values()}),
              "admissions": len(admissions), "events": n, "lab_events": sum(v for c, v in concept_counts.items() if c in lab_parameters),
              "medication_events": sum(v for c, v in concept_counts.items()
                                       if c in MED_NAMES or c == "tetracycline"),
              "admissions_with_labs": len(coverage["lab"]), "admissions_with_medications": len(coverage["medication"]),
              "admissions_without_events": int((per_admission == 0).sum()), "concept_counts": concept_counts,
              "clipped_events_by_concept": clipped, "lab_events_without_training_support": unsupported,
              "filter_counts": counts, "normalization": normalization,
              "text": "unchanged; no new extraction, encoding, or vector substitution",
              "labels": "unchanged and stored only in original outcomes directory",
              "model_integration": "not connected; new concept dictionary must be explicitly supported before training"}
    validate_outputs(output, admissions, concepts, lab_parameters)
    # 再次检查源文件，防止运行期间源被更换。
    if verify_sources(root) != input_hashes:
        raise ValueError("预处理期间原始底账变化")
    (output / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    artifacts = {path.name: sha256(path) for path in output.iterdir() if path.is_file() and path.name != "status.json"}
    manifest = {"status": report["status"], "raw_root": str(root.resolve()), "input_sha256": input_hashes,
                "medication_mapping": medication_mapping,
                "output_sha256": artifacts, "concepts": concepts, "code_sha256": sha256(Path(__file__)),
                "clinical_mapping_code_sha256": sha256(Path(__file__).resolve().parents[1] / "training/v5/clinical_concepts.py"),
                "scaling_code_sha256": sha256(Path(__file__).resolve().parents[1] / "training/v5/numeric_preprocessing.py")}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "status.json").write_text(json.dumps({"status": report["status"]}), encoding="utf-8")
    return {key: report[key] for key in ("status", "patients", "admissions", "events", "lab_events", "medication_events", "admissions_with_labs", "admissions_without_events")}


def validate_outputs(output: Path, admissions: dict[str, Admission], concepts: tuple[str, ...],
                     parameters: dict[str, dict[str, object]]) -> None:
    """从文件重新读回，验证时间、归属、尺度及 CSV／张量一致性。"""
    # NPZ 每次按键索引都会重新解压整列；一次载入，后续逐行查内存。
    with np.load(output / "event_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    offsets, hadms = arrays["offsets"], arrays["hadm_ids"]
    if len(hadms) != len(admissions) or set(map(str, hadms)) != set(admissions):
        raise ValueError("事件张量住院成员不一致")
    n = len(arrays["values"])
    if offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) < 0):
        raise ValueError("事件偏移错误")
    for field in ("values", "elapsed_log_days"):
        if not np.isfinite(arrays[field]).all():
            raise ValueError("事件张量包含非有限值")
    train_counts: Counter[str] = Counter()
    observed_counts = np.zeros(len(hadms), dtype=np.int64)
    index_by_hadm = {str(h): i for i, h in enumerate(hadms)}
    count = 0
    for index, row in enumerate(rows(output / "events.csv.gz")):
        admission = admissions[row["hadm_id"]]
        if row["subject_id"] != admission.subject or row["patient_split"] != admission.split:
            raise ValueError("事件患者或划分错误")
        if temporal_reason(row, admission):
            raise ValueError("事件时间审查未通过")
        position = index_by_hadm[row["hadm_id"]]
        if not offsets[position] <= index < offsets[position + 1]:
            raise ValueError("CSV 与住院事件偏移不匹配")
        concept = row["concept"]
        if concepts[int(arrays["concept_ids"][index]) - 1] != concept:
            raise ValueError("事件概念映射错误")
        z = float(row["value_normalized"])
        if not np.isclose(z, arrays["values"][index], atol=1e-6):
            raise ValueError("归一化 CSV 与张量不一致")
        if row["kind"] == "lab":
            expected, mask, clipped = normalized(float(row["value_raw"]), parameters[concept])
            if not math.isclose(z, expected, abs_tol=1e-9) or int(row["numeric_mask"]) != mask or int(row["clipped"]) != clipped:
                raise ValueError("归一化值与训练参数不匹配")
            if admission.split == "train":
                train_counts[concept] += 1
        elif row["numeric_mask"] != "0":
            raise ValueError("药物不应伪造检验数值")
        if bool(int(row["numeric_mask"])) != bool(arrays["numeric_mask"][index]):
            raise ValueError("数值 mask 不匹配")
        expected_time = math.log1p(float(row["elapsed_days"]))
        if not np.isclose(expected_time, arrays["elapsed_log_days"][index], atol=1e-6):
            raise ValueError("相对时间不匹配")
        observed_counts[position] += 1
        count += 1
    if count != n or not np.array_equal(observed_counts, np.diff(offsets)):
        raise ValueError("落盘事件数量不一致")
    if any(train_counts[concept] != p["training_observations"] for concept, p in parameters.items()):
        raise ValueError("归一化拟合范围不等于训练事件")
    static_hadms = set()
    for row in rows(output / "static_features.csv.gz"):
        admission = admissions[row["hadm_id"]]
        if row["subject_id"] != admission.subject or row["patient_split"] != admission.split:
            raise ValueError("静态特征归属错误")
        if not math.isfinite(float(row["value_normalized"])):
            raise ValueError("静态特征非有限")
        static_hadms.add(row["hadm_id"])
    if static_hadms != set(admissions):
        raise ValueError("静态特征成员不完整")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--medication-mapping", choices=("legacy", "gold_review_v2", "clinical_reviewed_2026_09"), default="legacy")
    args = parser.parse_args()
    print(json.dumps(preprocess(args.raw_root, args.output, medication_mapping=args.medication_mapping), ensure_ascii=False))
