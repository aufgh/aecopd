"""由冻结原始底账生成严格结局，保持当前并集实验的转院、死亡和索引规则。"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

STRICT_PRIMARY_EVENT = "first non-transfer readmission with strict 491.21/J44.1 at any diagnosis position; current all-coded transfer rule"
SOURCES = ("cohort/admissions", "raw/patients", "raw/diagnoses_icd",
           "outcomes/readmission_death_labels")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def diagnosis_key(record: dict[str, str]) -> tuple[int, str]:
    """PostgreSQL 定长 ICD 字段含尾随空格，不把空格当成代码的一部分。"""
    return int(record["icd_version"]), record["icd_code"].strip().replace(".", "").upper()


def reconstruct(cohort: list[dict[str, str]], patients: dict[str, dict[str, str]], *,
                strict_only: bool) -> dict[str, dict[str, object]]:
    """首次符合结局的非续接住院；其他住院不结束观察，死亡为竞争事件。"""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in cohort:
        groups[record["subject_id"]].append(record)
    output: dict[str, dict[str, object]] = {}
    for subject, records in groups.items():
        records.sort(key=lambda r: (r["admittime"], int(r["hadm_id"])))
        candidates = []
        previous_end = None
        for record in records:
            start, end = datetime.fromisoformat(record["admittime"]), datetime.fromisoformat(record["dischtime"])
            continuation = (record["admission_location"] == "TRANSFER FROM HOSPITAL"
                            and previous_end is not None and previous_end <= start <= previous_end + timedelta(days=2))
            qualifying = record["is_strict_aecopd"] == "t" or (not strict_only and int(record["best_copd_seq_num"]) <= 3)
            if qualifying and not continuation:
                candidates.append((start, int(record["hadm_id"]), record))
            # 不因本次结局定义而改变转院参照住院集合。
            previous_end = end
        dod_text = patients[subject]["dod"]
        dod = None if dod_text in {"", "\\N"} else datetime.fromisoformat(dod_text).date()
        for record in records:
            if record["discharge_landmark_eligible"] != "t":
                continue
            start, end = datetime.fromisoformat(record["admittime"]), datetime.fromisoformat(record["dischtime"])
            next_event = next((item for item in candidates
                               if (item[0], item[1]) > (start, int(record["hadm_id"])) and item[0] >= end), None)
            next_days = None if next_event is None else (next_event[0] - end).total_seconds() / 86400
            death_days = None if dod is None or dod < end.date() else float((dod - end.date()).days)
            if next_days is not None and next_days <= 365 and (death_days is None or next_days <= death_days):
                event, time = 1, next_days
            elif death_days is not None and death_days <= 365 and (next_days is None or death_days < next_days):
                event, time = 2, death_days
            else:
                event, time = 0, 365.
            output[record["hadm_id"]] = {"subject_id": subject, "hadm_id": record["hadm_id"],
                "admittime": record["admittime"], "dischtime": record["dischtime"],
                "event_type": event, "time_days": time,
                "next_copd_hadm_id": "" if next_event is None else next_event[1]}
    return output


def checked_strict_labels(raw_dir: Path) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    manifest = json.loads((raw_dir / "metadata/manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "validated":
        raise ValueError("raw ledger not validated")
    hashes = {name: sha(raw_dir / (name + ".csv.gz")) for name in SOURCES}
    if any(hashes[name] != manifest["sources"][name]["sha256"] for name in SOURCES):
        raise ValueError("raw endpoint source changed")
    cohort = rows(raw_dir / "cohort/admissions.csv.gz")
    patients = {r["subject_id"]: r for r in rows(raw_dir / "raw/patients.csv.gz")}
    strict_ids = {r["hadm_id"] for r in rows(raw_dir / "raw/diagnoses_icd.csv.gz")
                  if diagnosis_key(r) in {(9, "49121"), (10, "J441")}}
    if any((r["is_strict_aecopd"] == "t") != (r["hadm_id"] in strict_ids) for r in cohort):
        raise ValueError("strict flags disagree with raw diagnosis codes")
    broad = reconstruct(cohort, patients, strict_only=False)
    reference = {r["hadm_id"]: r for r in rows(raw_dir / "outcomes/readmission_death_labels.csv.gz")}
    if set(broad) != set(reference) or len(broad) != 19401:
        raise ValueError("reconstructed endpoint cohort differs")
    for hadm, label in broad.items():
        old = reference[hadm]
        if int(old["event_type"]) != label["event_type"] or not np.isclose(float(old["time_days"]), label["time_days"], atol=1e-8, rtol=0):
            raise ValueError("union label reconstruction differs from frozen ledger")
    strict = reconstruct(cohort, patients, strict_only=True)
    if any(strict[h]["event_type"] == 1 and
           (str(strict[h]["next_copd_hadm_id"]) not in strict_ids or
            broad[h]["event_type"] != 1 or float(strict[h]["time_days"]) < float(broad[h]["time_days"])) for h in strict):
        raise ValueError("strict event not a valid subset/later occurrence")
    return strict, hashes


def export(raw_dir: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    strict, hashes = checked_strict_labels(raw_dir)
    output_dir.mkdir(parents=True)
    target = output_dir / "labels.csv.gz"
    with gzip.open(target, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(strict.values()))))
        writer.writeheader()
        writer.writerows(strict.values())
    original = {r["hadm_id"]: r for r in rows(raw_dir / "outcomes/readmission_death_labels.csv.gz")}
    transitions = Counter(f"{original[h]['event_type']}->{label['event_type']}" for h, label in strict.items())
    summary = {"status": "strict_labels_reconstructed_and_checked", "primary_event": STRICT_PRIMARY_EVENT,
               "cohort_admissions": len(strict), "cohort_patients": len({r["subject_id"] for r in strict.values()}),
               "events": dict(Counter(str(r["event_type"]) for r in strict.values())),
               "event_type_transitions_from_union": dict(transitions), "label_sha256": sha(target),
               "source_sha256": hashes, "union_reconstruction_identical": True,
               "strict_flags_checked_against_diagnoses": True, "individual_followup_complete": False,
               "transfer_rule": "all coded admissions; hospital transfer within two days of preceding coded discharge",
               "code_position": "any diagnosis position"}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.raw_dir, args.output_dir)))
