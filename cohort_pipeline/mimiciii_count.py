"""只读统计 MIMIC-III 中与现有 v3 对齐的 AECOPD 队列规模。

仅输出汇总计数，不输出患者标识、原始记录或病历正文。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


AECOPD_ICD9 = "49121"
FOLLOWUP_DAYS = 365.0


@dataclass(frozen=True)
class Admission:
    subject_id: int
    hadm_id: int
    admittime: datetime
    dischtime: datetime
    admission_location: str
    hospital_expire_flag: int


def read_csv(root: Path, name: str):
    path = root / f"{name}.csv" / f"{name}.csv"
    with path.open("r", encoding="utf-8", newline="") as source:
        yield from csv.DictReader(source)


def to_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    # 与 02_build_strict_aecopd.sql 一致：一次住院取 49121 的最低诊断顺位。
    best_seq: dict[int, int] = {}
    for row in read_csv(args.root, "DIAGNOSES_ICD"):
        if row["ICD9_CODE"] != AECOPD_ICD9:
            continue
        hadm_id = int(row["HADM_ID"])
        seq = int(row["SEQ_NUM"]) if row["SEQ_NUM"] else 9999
        best_seq[hadm_id] = min(seq, best_seq.get(hadm_id, 9999))

    admissions: dict[int, Admission] = {}
    for row in read_csv(args.root, "ADMISSIONS"):
        hadm_id = int(row["HADM_ID"])
        if hadm_id not in best_seq:
            continue
        admissions[hadm_id] = Admission(
            subject_id=int(row["SUBJECT_ID"]),
            hadm_id=hadm_id,
            admittime=to_time(row["ADMITTIME"]),
            dischtime=to_time(row["DISCHTIME"]),
            admission_location=row["ADMISSION_LOCATION"],
            hospital_expire_flag=int(row["HOSPITAL_EXPIRE_FLAG"]),
        )

    seq3_subjects = {
        adm.subject_id
        for hadm_id, adm in admissions.items()
        if best_seq[hadm_id] <= 3
    }
    strict_by_subject: dict[int, list[Admission]] = defaultdict(list)
    for adm in admissions.values():
        if adm.subject_id in seq3_subjects:
            strict_by_subject[adm.subject_id].append(adm)

    dod_by_subject: dict[int, date] = {}
    for row in read_csv(args.root, "PATIENTS"):
        subject_id = int(row["SUBJECT_ID"])
        if subject_id in seq3_subjects and row["DOD"]:
            dod_by_subject[subject_id] = date.fromisoformat(row["DOD"][:10])

    outcome_counts = {"aecopd_readmission": 0, "death_before_readmission": 0, "no_event_365d": 0}
    training_outcome_counts = outcome_counts.copy()
    outcome_by_90d = 0
    outcome_by_180d = 0
    transfer_continuations = 0
    alive_index = 0
    training_eligible = 0
    nonpositive_los = 0
    for subject_id, rows in strict_by_subject.items():
        rows.sort(key=lambda adm: (adm.admittime, adm.hadm_id))
        is_continuation: list[bool] = []
        for i, adm in enumerate(rows):
            gap = (adm.admittime - rows[i - 1].dischtime).total_seconds() / 86400 if i else None
            # MIMIC-III 的类别名对应 IV 的 TRANSFER FROM HOSPITAL。
            continuation = (
                adm.admission_location == "TRANSFER FROM HOSP/EXTRAM"
                and gap is not None
                and gap <= 2
            )
            is_continuation.append(continuation)
            transfer_continuations += int(continuation)

        for i, adm in enumerate(rows):
            if adm.hospital_expire_flag != 0:
                continue
            alive_index += 1
            eligible = adm.dischtime > adm.admittime
            training_eligible += int(eligible)
            nonpositive_los += int(not eligible)

            next_real = next(
                (later for later, continuation in zip(rows[i + 1 :], is_continuation[i + 1 :]) if not continuation),
                None,
            )
            gap = (next_real.admittime - adm.dischtime).total_seconds() / 86400 if next_real else None
            readmission_days = gap if gap is not None and gap >= 0 else None
            dod = dod_by_subject.get(subject_id)
            death_days = (dod - adm.dischtime.date()).days if dod and dod >= adm.dischtime.date() else None
            if readmission_days is not None and readmission_days <= FOLLOWUP_DAYS and (
                death_days is None or readmission_days <= death_days
            ):
                outcome = "aecopd_readmission"
                outcome_by_90d += int(readmission_days <= 90)
                outcome_by_180d += int(readmission_days <= 180)
            elif death_days is not None and death_days <= FOLLOWUP_DAYS and (
                readmission_days is None or death_days < readmission_days
            ):
                outcome = "death_before_readmission"
            else:
                outcome = "no_event_365d"
            outcome_counts[outcome] += 1
            if eligible:
                training_outcome_counts[outcome] += 1

    result = {
        "source": "MIMIC-III v1.4 local CSV",
        "strict_icd9_code": AECOPD_ICD9,
        "strict_code_admissions": len(admissions),
        "strict_code_patients": len({adm.subject_id for adm in admissions.values()}),
        "seq_le_3_code_admissions": sum(best_seq[hadm_id] <= 3 for hadm_id in admissions),
        "seq_le_3_patients": len(seq3_subjects),
        "all_strict_admissions_of_seq3_patients": sum(map(len, strict_by_subject.values())),
        "alive_index_admissions": alive_index,
        "training_eligible_index_admissions": training_eligible,
        "nonpositive_los_index_admissions": nonpositive_los,
        "transfer_continuations_merged": transfer_continuations,
        "index_outcomes": outcome_counts,
        "training_eligible_index_outcomes": training_outcome_counts,
        "aecopd_readmission_by_90d_index": outcome_by_90d,
        "aecopd_readmission_by_180d_index": outcome_by_180d,
        "label_assumption": "no observed event by 365d treated as censored at 365d (World A)",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
