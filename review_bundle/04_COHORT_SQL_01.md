# 04_COHORT_SQL · 1

源文件正文或经过字段白名单提取的汇总数据，来源身份见每段元数据。

## cohort_pipeline/mimiciii_count.py · lines 1–161

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/mimiciii_count.py","part":1,"start_line":1,"end_line":161,"source_sha256":"85ccc395de79c5b9fb8fdc17bce73c0e9b18bfd2f748848274144c0c013bdbd6","fragment_sha256":"85ccc395de79c5b9fb8fdc17bce73c0e9b18bfd2f748848274144c0c013bdbd6","added_terminal_newline":false} -->
````python
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
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/official_mimiciii_postgres/postgres_add_indexes.sql · lines 1–979

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/official_mimiciii_postgres/postgres_add_indexes.sql","part":1,"start_line":1,"end_line":979,"source_sha256":"ea9195af6befa877e5b64979a76565d6364e944ea251850322f7b08e9edbcd45","fragment_sha256":"ea9195af6befa877e5b64979a76565d6364e944ea251850322f7b08e9edbcd45","added_terminal_newline":false} -->
````sql
-- ----------------------------------------------------------------
--
-- This is a script to add the MIMIC-III indexes for Postgres.
--
-- ----------------------------------------------------------------

-- If running scripts individually, you can set the schema where all tables are created as follows:
-- SET search_path TO mimiciii;

-- Restoring the search path to its default value can be accomplished as follows:
-- SET search_path TO "$user",public;

-------------
-- ADMISSIONS
-------------

DROP INDEX IF EXISTS ADMISSIONS_idx01;
CREATE INDEX ADMISSIONS_IDX01
  ON ADMISSIONS (SUBJECT_ID);

DROP INDEX IF EXISTS ADMISSIONS_idx02;
CREATE INDEX ADMISSIONS_IDX02
  ON ADMISSIONS (HADM_ID);

-- DROP INDEX IF EXISTS ADMISSIONS_idx03;
-- CREATE INDEX ADMISSIONS_IDX03
--   ON ADMISSIONS (ADMISSION_TYPE);


-----------
--CALLOUT--
-----------

DROP INDEX IF EXISTS CALLOUT_idx01;
CREATE INDEX CALLOUT_IDX01
  ON CALLOUT (SUBJECT_ID);

DROP INDEX IF EXISTS CALLOUT_idx02;
CREATE INDEX CALLOUT_IDX02
  ON CALLOUT (HADM_ID);

-- DROP INDEX IF EXISTS CALLOUT_idx03;
-- CREATE INDEX CALLOUT_IDX03
--   ON CALLOUT (CALLOUT_SERVICE);

-- DROP INDEX IF EXISTS CALLOUT_idx04;
-- CREATE INDEX CALLOUT_IDX04
--   ON CALLOUT (CURR_WARDID, CALLOUT_WARDID,
--     DISCHARGE_WARDID);

-- DROP INDEX IF EXISTS CALLOUT_idx05;
-- CREATE INDEX CALLOUT_IDX05
--   ON CALLOUT (CALLOUT_STATUS,
--     CALLOUT_OUTCOME);

-- DROP INDEX IF EXISTS CALLOUT_idx06;
-- CREATE INDEX CALLOUT_IDX06
--   ON CALLOUT (CREATETIME, UPDATETIME,
--     ACKNOWLEDGETIME, OUTCOMETIME);

---------------
-- CAREGIVERS
---------------

-- DROP INDEX IF EXISTS CAREGIVERS_idx01;
-- CREATE INDEX CAREGIVERS_IDX01
--   ON CAREGIVERS (CGID, LABEL);

---------------
-- CHARTEVENTS
---------------

-- CHARTEVENTS is built in 10 partitions which are inherited by a single mother table, "CHARTEVENTS"
-- Therefore, indices need to be added on every single inherited (or partitioned) table.

DROP INDEX IF EXISTS chartevents_1_idx01;
CREATE INDEX chartevents_1_idx01 ON chartevents_1 (itemid);
DROP INDEX IF EXISTS chartevents_2_idx01;
CREATE INDEX chartevents_2_idx01 ON chartevents_2 (itemid);
DROP INDEX IF EXISTS chartevents_3_idx01;
CREATE INDEX chartevents_3_idx01 ON chartevents_3 (itemid);
DROP INDEX IF EXISTS chartevents_4_idx01;
CREATE INDEX chartevents_4_idx01 ON chartevents_4 (itemid);
DROP INDEX IF EXISTS chartevents_5_idx01;
CREATE INDEX chartevents_5_idx01 ON chartevents_5 (itemid);
DROP INDEX IF EXISTS chartevents_6_idx01;
CREATE INDEX chartevents_6_idx01 ON chartevents_6 (itemid);
DROP INDEX IF EXISTS chartevents_7_idx01;
CREATE INDEX chartevents_7_idx01 ON chartevents_7 (itemid);
DROP INDEX IF EXISTS chartevents_8_idx01;
CREATE INDEX chartevents_8_idx01 ON chartevents_8 (itemid);
DROP INDEX IF EXISTS chartevents_9_idx01;
CREATE INDEX chartevents_9_idx01 ON chartevents_9 (itemid);
DROP INDEX IF EXISTS chartevents_10_idx01;
CREATE INDEX chartevents_10_idx01 ON chartevents_10 (itemid);
DROP INDEX IF EXISTS chartevents_11_idx01;
CREATE INDEX chartevents_11_idx01 ON chartevents_11 (itemid);
DROP INDEX IF EXISTS chartevents_12_idx01;
CREATE INDEX chartevents_12_idx01 ON chartevents_12 (itemid);
DROP INDEX IF EXISTS chartevents_13_idx01;
CREATE INDEX chartevents_13_idx01 ON chartevents_13 (itemid);
DROP INDEX IF EXISTS chartevents_14_idx01;
CREATE INDEX chartevents_14_idx01 ON chartevents_14 (itemid);
DROP INDEX IF EXISTS chartevents_15_idx01;
CREATE INDEX chartevents_15_idx01 ON chartevents_15 (itemid);
DROP INDEX IF EXISTS chartevents_16_idx01;
CREATE INDEX chartevents_16_idx01 ON chartevents_16 (itemid);
DROP INDEX IF EXISTS chartevents_17_idx01;
CREATE INDEX chartevents_17_idx01 ON chartevents_17 (itemid);

-- only create these indices if we have sufficient partitions
DO $$
BEGIN

IF EXISTS (
    SELECT 1
    FROM         pg_class c
    INNER JOIN   pg_namespace n
      ON n.oid = c.relnamespace
    WHERE  c.relname = 'chartevents_207'
  ) THEN

  DROP INDEX IF EXISTS chartevents_18_idx01;
  CREATE INDEX chartevents_18_idx01 ON chartevents_18 (itemid);
  DROP INDEX IF EXISTS chartevents_19_idx01;
  CREATE INDEX chartevents_19_idx01 ON chartevents_19 (itemid);
  DROP INDEX IF EXISTS chartevents_20_idx01;
  CREATE INDEX chartevents_20_idx01 ON chartevents_20 (itemid);
  DROP INDEX IF EXISTS chartevents_21_idx01;
  CREATE INDEX chartevents_21_idx01 ON chartevents_21 (itemid);
  DROP INDEX IF EXISTS chartevents_22_idx01;
  CREATE INDEX chartevents_22_idx01 ON chartevents_22 (itemid);
  DROP INDEX IF EXISTS chartevents_23_idx01;
  CREATE INDEX chartevents_23_idx01 ON chartevents_23 (itemid);
  DROP INDEX IF EXISTS chartevents_24_idx01;
  CREATE INDEX chartevents_24_idx01 ON chartevents_24 (itemid);
  DROP INDEX IF EXISTS chartevents_25_idx01;
  CREATE INDEX chartevents_25_idx01 ON chartevents_25 (itemid);
  DROP INDEX IF EXISTS chartevents_26_idx01;
  CREATE INDEX chartevents_26_idx01 ON chartevents_26 (itemid);
  DROP INDEX IF EXISTS chartevents_27_idx01;
  CREATE INDEX chartevents_27_idx01 ON chartevents_27 (itemid);
  DROP INDEX IF EXISTS chartevents_28_idx01;
  CREATE INDEX chartevents_28_idx01 ON chartevents_28 (itemid);
  DROP INDEX IF EXISTS chartevents_29_idx01;
  CREATE INDEX chartevents_29_idx01 ON chartevents_29 (itemid);
  DROP INDEX IF EXISTS chartevents_30_idx01;
  CREATE INDEX chartevents_30_idx01 ON chartevents_30 (itemid);
  DROP INDEX IF EXISTS chartevents_31_idx01;
  CREATE INDEX chartevents_31_idx01 ON chartevents_31 (itemid);
  DROP INDEX IF EXISTS chartevents_32_idx01;
  CREATE INDEX chartevents_32_idx01 ON chartevents_32 (itemid);
  DROP INDEX IF EXISTS chartevents_33_idx01;
  CREATE INDEX chartevents_33_idx01 ON chartevents_33 (itemid);
  DROP INDEX IF EXISTS chartevents_34_idx01;
  CREATE INDEX chartevents_34_idx01 ON chartevents_34 (itemid);
  DROP INDEX IF EXISTS chartevents_35_idx01;
  CREATE INDEX chartevents_35_idx01 ON chartevents_35 (itemid);
  DROP INDEX IF EXISTS chartevents_36_idx01;
  CREATE INDEX chartevents_36_idx01 ON chartevents_36 (itemid);
  DROP INDEX IF EXISTS chartevents_37_idx01;
  CREATE INDEX chartevents_37_idx01 ON chartevents_37 (itemid);
  DROP INDEX IF EXISTS chartevents_38_idx01;
  CREATE INDEX chartevents_38_idx01 ON chartevents_38 (itemid);
  DROP INDEX IF EXISTS chartevents_39_idx01;
  CREATE INDEX chartevents_39_idx01 ON chartevents_39 (itemid);
  DROP INDEX IF EXISTS chartevents_40_idx01;
  CREATE INDEX chartevents_40_idx01 ON chartevents_40 (itemid);
  DROP INDEX IF EXISTS chartevents_41_idx01;
  CREATE INDEX chartevents_41_idx01 ON chartevents_41 (itemid);
  DROP INDEX IF EXISTS chartevents_42_idx01;
  CREATE INDEX chartevents_42_idx01 ON chartevents_42 (itemid);
  DROP INDEX IF EXISTS chartevents_43_idx01;
  CREATE INDEX chartevents_43_idx01 ON chartevents_43 (itemid);
  DROP INDEX IF EXISTS chartevents_44_idx01;
  CREATE INDEX chartevents_44_idx01 ON chartevents_44 (itemid);
  DROP INDEX IF EXISTS chartevents_45_idx01;
  CREATE INDEX chartevents_45_idx01 ON chartevents_45 (itemid);
  DROP INDEX IF EXISTS chartevents_46_idx01;
  CREATE INDEX chartevents_46_idx01 ON chartevents_46 (itemid);
  DROP INDEX IF EXISTS chartevents_47_idx01;
  CREATE INDEX chartevents_47_idx01 ON chartevents_47 (itemid);
  DROP INDEX IF EXISTS chartevents_48_idx01;
  CREATE INDEX chartevents_48_idx01 ON chartevents_48 (itemid);
  DROP INDEX IF EXISTS chartevents_49_idx01;
  CREATE INDEX chartevents_49_idx01 ON chartevents_49 (itemid);
  DROP INDEX IF EXISTS chartevents_50_idx01;
  CREATE INDEX chartevents_50_idx01 ON chartevents_50 (itemid);
  DROP INDEX IF EXISTS chartevents_51_idx01;
  CREATE INDEX chartevents_51_idx01 ON chartevents_51 (itemid);
  DROP INDEX IF EXISTS chartevents_52_idx01;
  CREATE INDEX chartevents_52_idx01 ON chartevents_52 (itemid);
  DROP INDEX IF EXISTS chartevents_53_idx01;
  CREATE INDEX chartevents_53_idx01 ON chartevents_53 (itemid);
  DROP INDEX IF EXISTS chartevents_54_idx01;
  CREATE INDEX chartevents_54_idx01 ON chartevents_54 (itemid);
  DROP INDEX IF EXISTS chartevents_55_idx01;
  CREATE INDEX chartevents_55_idx01 ON chartevents_55 (itemid);
  DROP INDEX IF EXISTS chartevents_56_idx01;
  CREATE INDEX chartevents_56_idx01 ON chartevents_56 (itemid);
  DROP INDEX IF EXISTS chartevents_57_idx01;
  CREATE INDEX chartevents_57_idx01 ON chartevents_57 (itemid);
  DROP INDEX IF EXISTS chartevents_58_idx01;
  CREATE INDEX chartevents_58_idx01 ON chartevents_58 (itemid);
  DROP INDEX IF EXISTS chartevents_59_idx01;
  CREATE INDEX chartevents_59_idx01 ON chartevents_59 (itemid);
  DROP INDEX IF EXISTS chartevents_60_idx01;
  CREATE INDEX chartevents_60_idx01 ON chartevents_60 (itemid);
  DROP INDEX IF EXISTS chartevents_61_idx01;
  CREATE INDEX chartevents_61_idx01 ON chartevents_61 (itemid);
  DROP INDEX IF EXISTS chartevents_62_idx01;
  CREATE INDEX chartevents_62_idx01 ON chartevents_62 (itemid);
  DROP INDEX IF EXISTS chartevents_63_idx01;
  CREATE INDEX chartevents_63_idx01 ON chartevents_63 (itemid);
  DROP INDEX IF EXISTS chartevents_64_idx01;
  CREATE INDEX chartevents_64_idx01 ON chartevents_64 (itemid);
  DROP INDEX IF EXISTS chartevents_65_idx01;
  CREATE INDEX chartevents_65_idx01 ON chartevents_65 (itemid);
  DROP INDEX IF EXISTS chartevents_66_idx01;
  CREATE INDEX chartevents_66_idx01 ON chartevents_66 (itemid);
  DROP INDEX IF EXISTS chartevents_67_idx01;
  CREATE INDEX chartevents_67_idx01 ON chartevents_67 (itemid);
  DROP INDEX IF EXISTS chartevents_68_idx01;
  CREATE INDEX chartevents_68_idx01 ON chartevents_68 (itemid);
  DROP INDEX IF EXISTS chartevents_69_idx01;
  CREATE INDEX chartevents_69_idx01 ON chartevents_69 (itemid);
  DROP INDEX IF EXISTS chartevents_70_idx01;
  CREATE INDEX chartevents_70_idx01 ON chartevents_70 (itemid);
  DROP INDEX IF EXISTS chartevents_71_idx01;
  CREATE INDEX chartevents_71_idx01 ON chartevents_71 (itemid);
  DROP INDEX IF EXISTS chartevents_72_idx01;
  CREATE INDEX chartevents_72_idx01 ON chartevents_72 (itemid);
  DROP INDEX IF EXISTS chartevents_73_idx01;
  CREATE INDEX chartevents_73_idx01 ON chartevents_73 (itemid);
  DROP INDEX IF EXISTS chartevents_74_idx01;
  CREATE INDEX chartevents_74_idx01 ON chartevents_74 (itemid);
  DROP INDEX IF EXISTS chartevents_75_idx01;
  CREATE INDEX chartevents_75_idx01 ON chartevents_75 (itemid);
  DROP INDEX IF EXISTS chartevents_76_idx01;
  CREATE INDEX chartevents_76_idx01 ON chartevents_76 (itemid);
  DROP INDEX IF EXISTS chartevents_77_idx01;
  CREATE INDEX chartevents_77_idx01 ON chartevents_77 (itemid);
  DROP INDEX IF EXISTS chartevents_78_idx01;
  CREATE INDEX chartevents_78_idx01 ON chartevents_78 (itemid);
  DROP INDEX IF EXISTS chartevents_79_idx01;
  CREATE INDEX chartevents_79_idx01 ON chartevents_79 (itemid);
  DROP INDEX IF EXISTS chartevents_80_idx01;
  CREATE INDEX chartevents_80_idx01 ON chartevents_80 (itemid);
  DROP INDEX IF EXISTS chartevents_81_idx01;
  CREATE INDEX chartevents_81_idx01 ON chartevents_81 (itemid);
  DROP INDEX IF EXISTS chartevents_82_idx01;
  CREATE INDEX chartevents_82_idx01 ON chartevents_82 (itemid);
  DROP INDEX IF EXISTS chartevents_83_idx01;
  CREATE INDEX chartevents_83_idx01 ON chartevents_83 (itemid);
  DROP INDEX IF EXISTS chartevents_84_idx01;
  CREATE INDEX chartevents_84_idx01 ON chartevents_84 (itemid);
  DROP INDEX IF EXISTS chartevents_85_idx01;
  CREATE INDEX chartevents_85_idx01 ON chartevents_85 (itemid);
  DROP INDEX IF EXISTS chartevents_86_idx01;
  CREATE INDEX chartevents_86_idx01 ON chartevents_86 (itemid);
  DROP INDEX IF EXISTS chartevents_87_idx01;
  CREATE INDEX chartevents_87_idx01 ON chartevents_87 (itemid);
  DROP INDEX IF EXISTS chartevents_88_idx01;
  CREATE INDEX chartevents_88_idx01 ON chartevents_88 (itemid);
  DROP INDEX IF EXISTS chartevents_89_idx01;
  CREATE INDEX chartevents_89_idx01 ON chartevents_89 (itemid);
  DROP INDEX IF EXISTS chartevents_90_idx01;
  CREATE INDEX chartevents_90_idx01 ON chartevents_90 (itemid);
  DROP INDEX IF EXISTS chartevents_91_idx01;
  CREATE INDEX chartevents_91_idx01 ON chartevents_91 (itemid);
  DROP INDEX IF EXISTS chartevents_92_idx01;
  CREATE INDEX chartevents_92_idx01 ON chartevents_92 (itemid);
  DROP INDEX IF EXISTS chartevents_93_idx01;
  CREATE INDEX chartevents_93_idx01 ON chartevents_93 (itemid);
  DROP INDEX IF EXISTS chartevents_94_idx01;
  CREATE INDEX chartevents_94_idx01 ON chartevents_94 (itemid);
  DROP INDEX IF EXISTS chartevents_95_idx01;
  CREATE INDEX chartevents_95_idx01 ON chartevents_95 (itemid);
  DROP INDEX IF EXISTS chartevents_96_idx01;
  CREATE INDEX chartevents_96_idx01 ON chartevents_96 (itemid);
  DROP INDEX IF EXISTS chartevents_97_idx01;
  CREATE INDEX chartevents_97_idx01 ON chartevents_97 (itemid);
  DROP INDEX IF EXISTS chartevents_98_idx01;
  CREATE INDEX chartevents_98_idx01 ON chartevents_98 (itemid);
  DROP INDEX IF EXISTS chartevents_99_idx01;
  CREATE INDEX chartevents_99_idx01 ON chartevents_99 (itemid);
  DROP INDEX IF EXISTS chartevents_100_idx01;
  CREATE INDEX chartevents_100_idx01 ON chartevents_100 (itemid);
  DROP INDEX IF EXISTS chartevents_101_idx01;
  CREATE INDEX chartevents_101_idx01 ON chartevents_101 (itemid);
  DROP INDEX IF EXISTS chartevents_102_idx01;
  CREATE INDEX chartevents_102_idx01 ON chartevents_102 (itemid);
  DROP INDEX IF EXISTS chartevents_103_idx01;
  CREATE INDEX chartevents_103_idx01 ON chartevents_103 (itemid);
  DROP INDEX IF EXISTS chartevents_104_idx01;
  CREATE INDEX chartevents_104_idx01 ON chartevents_104 (itemid);
  DROP INDEX IF EXISTS chartevents_105_idx01;
  CREATE INDEX chartevents_105_idx01 ON chartevents_105 (itemid);
  DROP INDEX IF EXISTS chartevents_106_idx01;
  CREATE INDEX chartevents_106_idx01 ON chartevents_106 (itemid);
  DROP INDEX IF EXISTS chartevents_107_idx01;
  CREATE INDEX chartevents_107_idx01 ON chartevents_107 (itemid);
  DROP INDEX IF EXISTS chartevents_108_idx01;
  CREATE INDEX chartevents_108_idx01 ON chartevents_108 (itemid);
  DROP INDEX IF EXISTS chartevents_109_idx01;
  CREATE INDEX chartevents_109_idx01 ON chartevents_109 (itemid);
  DROP INDEX IF EXISTS chartevents_110_idx01;
  CREATE INDEX chartevents_110_idx01 ON chartevents_110 (itemid);
  DROP INDEX IF EXISTS chartevents_111_idx01;
  CREATE INDEX chartevents_111_idx01 ON chartevents_111 (itemid);
  DROP INDEX IF EXISTS chartevents_112_idx01;
  CREATE INDEX chartevents_112_idx01 ON chartevents_112 (itemid);
  DROP INDEX IF EXISTS chartevents_113_idx01;
  CREATE INDEX chartevents_113_idx01 ON chartevents_113 (itemid);
  DROP INDEX IF EXISTS chartevents_114_idx01;
  CREATE INDEX chartevents_114_idx01 ON chartevents_114 (itemid);
  DROP INDEX IF EXISTS chartevents_115_idx01;
  CREATE INDEX chartevents_115_idx01 ON chartevents_115 (itemid);
  DROP INDEX IF EXISTS chartevents_116_idx01;
  CREATE INDEX chartevents_116_idx01 ON chartevents_116 (itemid);
  DROP INDEX IF EXISTS chartevents_117_idx01;
  CREATE INDEX chartevents_117_idx01 ON chartevents_117 (itemid);
  DROP INDEX IF EXISTS chartevents_118_idx01;
  CREATE INDEX chartevents_118_idx01 ON chartevents_118 (itemid);
  DROP INDEX IF EXISTS chartevents_119_idx01;
  CREATE INDEX chartevents_119_idx01 ON chartevents_119 (itemid);
  DROP INDEX IF EXISTS chartevents_120_idx01;
  CREATE INDEX chartevents_120_idx01 ON chartevents_120 (itemid);
  DROP INDEX IF EXISTS chartevents_121_idx01;
  CREATE INDEX chartevents_121_idx01 ON chartevents_121 (itemid);
  DROP INDEX IF EXISTS chartevents_122_idx01;
  CREATE INDEX chartevents_122_idx01 ON chartevents_122 (itemid);
  DROP INDEX IF EXISTS chartevents_123_idx01;
  CREATE INDEX chartevents_123_idx01 ON chartevents_123 (itemid);
  DROP INDEX IF EXISTS chartevents_124_idx01;
  CREATE INDEX chartevents_124_idx01 ON chartevents_124 (itemid);
  DROP INDEX IF EXISTS chartevents_125_idx01;
  CREATE INDEX chartevents_125_idx01 ON chartevents_125 (itemid);
  DROP INDEX IF EXISTS chartevents_126_idx01;
  CREATE INDEX chartevents_126_idx01 ON chartevents_126 (itemid);
  DROP INDEX IF EXISTS chartevents_127_idx01;
  CREATE INDEX chartevents_127_idx01 ON chartevents_127 (itemid);
  DROP INDEX IF EXISTS chartevents_128_idx01;
  CREATE INDEX chartevents_128_idx01 ON chartevents_128 (itemid);
  DROP INDEX IF EXISTS chartevents_129_idx01;
  CREATE INDEX chartevents_129_idx01 ON chartevents_129 (itemid);
  DROP INDEX IF EXISTS chartevents_130_idx01;
  CREATE INDEX chartevents_130_idx01 ON chartevents_130 (itemid);
  DROP INDEX IF EXISTS chartevents_131_idx01;
  CREATE INDEX chartevents_131_idx01 ON chartevents_131 (itemid);
  DROP INDEX IF EXISTS chartevents_132_idx01;
  CREATE INDEX chartevents_132_idx01 ON chartevents_132 (itemid);
  DROP INDEX IF EXISTS chartevents_133_idx01;
  CREATE INDEX chartevents_133_idx01 ON chartevents_133 (itemid);
  DROP INDEX IF EXISTS chartevents_134_idx01;
  CREATE INDEX chartevents_134_idx01 ON chartevents_134 (itemid);
  DROP INDEX IF EXISTS chartevents_135_idx01;
  CREATE INDEX chartevents_135_idx01 ON chartevents_135 (itemid);
  DROP INDEX IF EXISTS chartevents_136_idx01;
  CREATE INDEX chartevents_136_idx01 ON chartevents_136 (itemid);
  DROP INDEX IF EXISTS chartevents_137_idx01;
  CREATE INDEX chartevents_137_idx01 ON chartevents_137 (itemid);
  DROP INDEX IF EXISTS chartevents_138_idx01;
  CREATE INDEX chartevents_138_idx01 ON chartevents_138 (itemid);
  DROP INDEX IF EXISTS chartevents_139_idx01;
  CREATE INDEX chartevents_139_idx01 ON chartevents_139 (itemid);
  DROP INDEX IF EXISTS chartevents_140_idx01;
  CREATE INDEX chartevents_140_idx01 ON chartevents_140 (itemid);
  DROP INDEX IF EXISTS chartevents_141_idx01;
  CREATE INDEX chartevents_141_idx01 ON chartevents_141 (itemid);
  DROP INDEX IF EXISTS chartevents_142_idx01;
  CREATE INDEX chartevents_142_idx01 ON chartevents_142 (itemid);
  DROP INDEX IF EXISTS chartevents_143_idx01;
  CREATE INDEX chartevents_143_idx01 ON chartevents_143 (itemid);
  DROP INDEX IF EXISTS chartevents_144_idx01;
  CREATE INDEX chartevents_144_idx01 ON chartevents_144 (itemid);
  DROP INDEX IF EXISTS chartevents_145_idx01;
  CREATE INDEX chartevents_145_idx01 ON chartevents_145 (itemid);
  DROP INDEX IF EXISTS chartevents_146_idx01;
  CREATE INDEX chartevents_146_idx01 ON chartevents_146 (itemid);
  DROP INDEX IF EXISTS chartevents_147_idx01;
  CREATE INDEX chartevents_147_idx01 ON chartevents_147 (itemid);
  DROP INDEX IF EXISTS chartevents_148_idx01;
  CREATE INDEX chartevents_148_idx01 ON chartevents_148 (itemid);
  DROP INDEX IF EXISTS chartevents_149_idx01;
  CREATE INDEX chartevents_149_idx01 ON chartevents_149 (itemid);
  DROP INDEX IF EXISTS chartevents_150_idx01;
  CREATE INDEX chartevents_150_idx01 ON chartevents_150 (itemid);
  DROP INDEX IF EXISTS chartevents_151_idx01;
  CREATE INDEX chartevents_151_idx01 ON chartevents_151 (itemid);
  DROP INDEX IF EXISTS chartevents_152_idx01;
  CREATE INDEX chartevents_152_idx01 ON chartevents_152 (itemid);
  DROP INDEX IF EXISTS chartevents_153_idx01;
  CREATE INDEX chartevents_153_idx01 ON chartevents_153 (itemid);
  DROP INDEX IF EXISTS chartevents_154_idx01;
  CREATE INDEX chartevents_154_idx01 ON chartevents_154 (itemid);
  DROP INDEX IF EXISTS chartevents_155_idx01;
  CREATE INDEX chartevents_155_idx01 ON chartevents_155 (itemid);
  DROP INDEX IF EXISTS chartevents_156_idx01;
  CREATE INDEX chartevents_156_idx01 ON chartevents_156 (itemid);
  DROP INDEX IF EXISTS chartevents_157_idx01;
  CREATE INDEX chartevents_157_idx01 ON chartevents_157 (itemid);
  DROP INDEX IF EXISTS chartevents_158_idx01;
  CREATE INDEX chartevents_158_idx01 ON chartevents_158 (itemid);
  DROP INDEX IF EXISTS chartevents_159_idx01;
  CREATE INDEX chartevents_159_idx01 ON chartevents_159 (itemid);
  DROP INDEX IF EXISTS chartevents_160_idx01;
  CREATE INDEX chartevents_160_idx01 ON chartevents_160 (itemid);
  DROP INDEX IF EXISTS chartevents_161_idx01;
  CREATE INDEX chartevents_161_idx01 ON chartevents_161 (itemid);
  DROP INDEX IF EXISTS chartevents_162_idx01;
  CREATE INDEX chartevents_162_idx01 ON chartevents_162 (itemid);
  DROP INDEX IF EXISTS chartevents_163_idx01;
  CREATE INDEX chartevents_163_idx01 ON chartevents_163 (itemid);
  DROP INDEX IF EXISTS chartevents_164_idx01;
  CREATE INDEX chartevents_164_idx01 ON chartevents_164 (itemid);
  DROP INDEX IF EXISTS chartevents_165_idx01;
  CREATE INDEX chartevents_165_idx01 ON chartevents_165 (itemid);
  DROP INDEX IF EXISTS chartevents_166_idx01;
  CREATE INDEX chartevents_166_idx01 ON chartevents_166 (itemid);
  DROP INDEX IF EXISTS chartevents_167_idx01;
  CREATE INDEX chartevents_167_idx01 ON chartevents_167 (itemid);
  DROP INDEX IF EXISTS chartevents_168_idx01;
  CREATE INDEX chartevents_168_idx01 ON chartevents_168 (itemid);
  DROP INDEX IF EXISTS chartevents_169_idx01;
  CREATE INDEX chartevents_169_idx01 ON chartevents_169 (itemid);
  DROP INDEX IF EXISTS chartevents_170_idx01;
  CREATE INDEX chartevents_170_idx01 ON chartevents_170 (itemid);
  DROP INDEX IF EXISTS chartevents_171_idx01;
  CREATE INDEX chartevents_171_idx01 ON chartevents_171 (itemid);
  DROP INDEX IF EXISTS chartevents_172_idx01;
  CREATE INDEX chartevents_172_idx01 ON chartevents_172 (itemid);
  DROP INDEX IF EXISTS chartevents_173_idx01;
  CREATE INDEX chartevents_173_idx01 ON chartevents_173 (itemid);
  DROP INDEX IF EXISTS chartevents_174_idx01;
  CREATE INDEX chartevents_174_idx01 ON chartevents_174 (itemid);
  DROP INDEX IF EXISTS chartevents_175_idx01;
  CREATE INDEX chartevents_175_idx01 ON chartevents_175 (itemid);
  DROP INDEX IF EXISTS chartevents_176_idx01;
  CREATE INDEX chartevents_176_idx01 ON chartevents_176 (itemid);
  DROP INDEX IF EXISTS chartevents_177_idx01;
  CREATE INDEX chartevents_177_idx01 ON chartevents_177 (itemid);
  DROP INDEX IF EXISTS chartevents_178_idx01;
  CREATE INDEX chartevents_178_idx01 ON chartevents_178 (itemid);
  DROP INDEX IF EXISTS chartevents_179_idx01;
  CREATE INDEX chartevents_179_idx01 ON chartevents_179 (itemid);
  DROP INDEX IF EXISTS chartevents_180_idx01;
  CREATE INDEX chartevents_180_idx01 ON chartevents_180 (itemid);
  DROP INDEX IF EXISTS chartevents_181_idx01;
  CREATE INDEX chartevents_181_idx01 ON chartevents_181 (itemid);
  DROP INDEX IF EXISTS chartevents_182_idx01;
  CREATE INDEX chartevents_182_idx01 ON chartevents_182 (itemid);
  DROP INDEX IF EXISTS chartevents_183_idx01;
  CREATE INDEX chartevents_183_idx01 ON chartevents_183 (itemid);
  DROP INDEX IF EXISTS chartevents_184_idx01;
  CREATE INDEX chartevents_184_idx01 ON chartevents_184 (itemid);
  DROP INDEX IF EXISTS chartevents_185_idx01;
  CREATE INDEX chartevents_185_idx01 ON chartevents_185 (itemid);
  DROP INDEX IF EXISTS chartevents_186_idx01;
  CREATE INDEX chartevents_186_idx01 ON chartevents_186 (itemid);
  DROP INDEX IF EXISTS chartevents_187_idx01;
  CREATE INDEX chartevents_187_idx01 ON chartevents_187 (itemid);
  DROP INDEX IF EXISTS chartevents_188_idx01;
  CREATE INDEX chartevents_188_idx01 ON chartevents_188 (itemid);
  DROP INDEX IF EXISTS chartevents_189_idx01;
  CREATE INDEX chartevents_189_idx01 ON chartevents_189 (itemid);
  DROP INDEX IF EXISTS chartevents_190_idx01;
  CREATE INDEX chartevents_190_idx01 ON chartevents_190 (itemid);
  DROP INDEX IF EXISTS chartevents_191_idx01;
  CREATE INDEX chartevents_191_idx01 ON chartevents_191 (itemid);
  DROP INDEX IF EXISTS chartevents_192_idx01;
  CREATE INDEX chartevents_192_idx01 ON chartevents_192 (itemid);
  DROP INDEX IF EXISTS chartevents_193_idx01;
  CREATE INDEX chartevents_193_idx01 ON chartevents_193 (itemid);
  DROP INDEX IF EXISTS chartevents_194_idx01;
  CREATE INDEX chartevents_194_idx01 ON chartevents_194 (itemid);
  DROP INDEX IF EXISTS chartevents_195_idx01;
  CREATE INDEX chartevents_195_idx01 ON chartevents_195 (itemid);
  DROP INDEX IF EXISTS chartevents_196_idx01;
  CREATE INDEX chartevents_196_idx01 ON chartevents_196 (itemid);
  DROP INDEX IF EXISTS chartevents_197_idx01;
  CREATE INDEX chartevents_197_idx01 ON chartevents_197 (itemid);
  DROP INDEX IF EXISTS chartevents_198_idx01;
  CREATE INDEX chartevents_198_idx01 ON chartevents_198 (itemid);
  DROP INDEX IF EXISTS chartevents_199_idx01;
  CREATE INDEX chartevents_199_idx01 ON chartevents_199 (itemid);
  DROP INDEX IF EXISTS chartevents_200_idx01;
  CREATE INDEX chartevents_200_idx01 ON chartevents_200 (itemid);
  DROP INDEX IF EXISTS chartevents_201_idx01;
  CREATE INDEX chartevents_201_idx01 ON chartevents_201 (itemid);
  DROP INDEX IF EXISTS chartevents_202_idx01;
  CREATE INDEX chartevents_202_idx01 ON chartevents_202 (itemid);
  DROP INDEX IF EXISTS chartevents_203_idx01;
  CREATE INDEX chartevents_203_idx01 ON chartevents_203 (itemid);
  DROP INDEX IF EXISTS chartevents_204_idx01;
  CREATE INDEX chartevents_204_idx01 ON chartevents_204 (itemid);
  DROP INDEX IF EXISTS chartevents_205_idx01;
  CREATE INDEX chartevents_205_idx01 ON chartevents_205 (itemid);
  DROP INDEX IF EXISTS chartevents_206_idx01;
  CREATE INDEX chartevents_206_idx01 ON chartevents_206 (itemid);
  DROP INDEX IF EXISTS chartevents_207_idx01;
  CREATE INDEX chartevents_207_idx01 ON chartevents_207 (itemid);
END IF;

END$$;
---------------
-- CPTEVENTS
---------------

DROP INDEX IF EXISTS CPTEVENTS_idx01;
CREATE INDEX CPTEVENTS_idx01
  ON CPTEVENTS (SUBJECT_ID);

DROP INDEX IF EXISTS CPTEVENTS_idx02;
CREATE INDEX CPTEVENTS_idx02
  ON CPTEVENTS (CPT_CD);

-----------
-- D_CPT
-----------

-- Table is 134 rows - doesn't need an index.

--------------------
-- D_ICD_DIAGNOSES
--------------------

DROP INDEX IF EXISTS D_ICD_DIAG_idx01;
CREATE INDEX D_ICD_DIAG_idx01
  ON D_ICD_DIAGNOSES (ICD9_CODE);

DROP INDEX IF EXISTS D_ICD_DIAG_idx02;
CREATE INDEX D_ICD_DIAG_idx02
  ON D_ICD_DIAGNOSES (LONG_TITLE);

--------------------
-- D_ICD_PROCEDURES
--------------------

DROP INDEX IF EXISTS D_ICD_PROC_idx01;
CREATE INDEX D_ICD_PROC_idx01
  ON D_ICD_PROCEDURES (ICD9_CODE);

DROP INDEX IF EXISTS D_ICD_PROC_idx02;
CREATE INDEX D_ICD_PROC_idx02
  ON D_ICD_PROCEDURES (LONG_TITLE);

-----------
-- D_ITEMS
-----------

DROP INDEX IF EXISTS D_ITEMS_idx01;
CREATE INDEX D_ITEMS_idx01
  ON D_ITEMS (ITEMID);

DROP INDEX IF EXISTS D_ITEMS_idx02;
CREATE INDEX D_ITEMS_idx02
  ON D_ITEMS (LABEL);

-- DROP INDEX IF EXISTS D_ITEMS_idx03;
-- CREATE INDEX D_ITEMS_idx03
--   ON D_ITEMS (CATEGORY);

---------------
-- D_LABITEMS
---------------

DROP INDEX IF EXISTS D_LABITEMS_idx01;
CREATE INDEX D_LABITEMS_idx01
  ON D_LABITEMS (ITEMID);

DROP INDEX IF EXISTS D_LABITEMS_idx02;
CREATE INDEX D_LABITEMS_idx02
  ON D_LABITEMS (LABEL);

DROP INDEX IF EXISTS D_LABITEMS_idx03;
CREATE INDEX D_LABITEMS_idx03
  ON D_LABITEMS (LOINC_CODE);

-------------------
-- DATETIMEEVENTS
-------------------

DROP INDEX IF EXISTS DATETIMEEVENTS_idx01;
CREATE INDEX DATETIMEEVENTS_idx01
  ON DATETIMEEVENTS (SUBJECT_ID);

DROP INDEX IF EXISTS DATETIMEEVENTS_idx02;
CREATE INDEX DATETIMEEVENTS_idx02
  ON DATETIMEEVENTS (ITEMID);

DROP INDEX IF EXISTS DATETIMEEVENTS_idx03;
CREATE INDEX DATETIMEEVENTS_idx03
  ON DATETIMEEVENTS (ICUSTAY_ID);

DROP INDEX IF EXISTS DATETIMEEVENTS_idx04;
CREATE INDEX DATETIMEEVENTS_idx04
  ON DATETIMEEVENTS (HADM_ID);

-- DROP INDEX IF EXISTS DATETIMEEVENTS_idx05;
-- CREATE INDEX DATETIMEEVENTS_idx05
--   ON DATETIMEEVENTS (VALUE);

------------------
-- DIAGNOSES_ICD
------------------

DROP INDEX IF EXISTS DIAGNOSES_ICD_idx01;
CREATE INDEX DIAGNOSES_ICD_idx01
  ON DIAGNOSES_ICD (SUBJECT_ID);

DROP INDEX IF EXISTS DIAGNOSES_ICD_idx02;
CREATE INDEX DIAGNOSES_ICD_idx02
  ON DIAGNOSES_ICD (ICD9_CODE);

DROP INDEX IF EXISTS DIAGNOSES_ICD_idx03;
CREATE INDEX DIAGNOSES_ICD_idx03
  ON DIAGNOSES_ICD (HADM_ID);

--------------
-- DRGCODES
--------------

DROP INDEX IF EXISTS DRGCODES_idx01;
CREATE INDEX DRGCODES_idx01
  ON DRGCODES (SUBJECT_ID);

DROP INDEX IF EXISTS DRGCODES_idx02;
CREATE INDEX DRGCODES_idx02
  ON DRGCODES (DRG_CODE);

DROP INDEX IF EXISTS DRGCODES_idx03;
CREATE INDEX DRGCODES_idx03
  ON DRGCODES (DESCRIPTION);

-- HADM_ID

------------------
-- ICUSTAYS
------------------

DROP INDEX IF EXISTS ICUSTAYS_idx01;
CREATE INDEX ICUSTAYS_idx01
  ON ICUSTAYS (SUBJECT_ID);

DROP INDEX IF EXISTS ICUSTAYS_idx02;
CREATE INDEX ICUSTAYS_idx02
  ON ICUSTAYS (ICUSTAY_ID);

-- DROP INDEX IF EXISTS ICUSTAYS_idx03;
-- CREATE INDEX ICUSTAYS_idx03
--   ON ICUSTAYS (LOS);

-- DROP INDEX IF EXISTS ICUSTAYS_idx04;
-- CREATE INDEX ICUSTAYS_idx04
--   ON ICUSTAYS (FIRST_CAREUNIT);

-- DROP INDEX IF EXISTS ICUSTAYS_idx05;
-- CREATE INDEX ICUSTAYS_idx05
--   ON ICUSTAYS (LAST_CAREUNIT);

DROP INDEX IF EXISTS ICUSTAYS_idx06;
CREATE INDEX ICUSTAYS_IDX06
  ON ICUSTAYS (HADM_ID);

-------------
-- INPUTEVENTS_CV
-------------

DROP INDEX IF EXISTS INPUTEVENTS_CV_idx01;
CREATE INDEX INPUTEVENTS_CV_idx01
  ON INPUTEVENTS_CV (SUBJECT_ID);

  DROP INDEX IF EXISTS INPUTEVENTS_CV_idx02;
  CREATE INDEX INPUTEVENTS_CV_idx02
    ON INPUTEVENTS_CV (HADM_ID);

DROP INDEX IF EXISTS INPUTEVENTS_CV_idx03;
CREATE INDEX INPUTEVENTS_CV_idx03
  ON INPUTEVENTS_CV (ICUSTAY_ID);

DROP INDEX IF EXISTS INPUTEVENTS_CV_idx04;
CREATE INDEX INPUTEVENTS_CV_idx04
  ON INPUTEVENTS_CV (CHARTTIME);

DROP INDEX IF EXISTS INPUTEVENTS_CV_idx05;
CREATE INDEX INPUTEVENTS_CV_idx05
  ON INPUTEVENTS_CV (ITEMID);

-- DROP INDEX IF EXISTS INPUTEVENTS_CV_idx06;
-- CREATE INDEX INPUTEVENTS_CV_idx06
--   ON INPUTEVENTS_CV (RATE);

-- DROP INDEX IF EXISTS INPUTEVENTS_CV_idx07;
-- CREATE INDEX INPUTEVENTS_CV_idx07
--   ON INPUTEVENTS_CV (AMOUNT);

-- DROP INDEX IF EXISTS INPUTEVENTS_CV_idx08;
-- CREATE INDEX INPUTEVENTS_CV_idx08
--   ON INPUTEVENTS_CV (CGID);

-- DROP INDEX IF EXISTS INPUTEVENTS_CV_idx09;
-- CREATE INDEX INPUTEVENTS_CV_idx09
--   ON INPUTEVENTS_CV (LINKORDERID, ORDERID);

-------------
-- INPUTEVENTS_MV
-------------

DROP INDEX IF EXISTS INPUTEVENTS_MV_idx01;
CREATE INDEX INPUTEVENTS_MV_idx01
  ON INPUTEVENTS_MV (SUBJECT_ID);

DROP INDEX IF EXISTS INPUTEVENTS_MV_idx02;
CREATE INDEX INPUTEVENTS_MV_idx02
  ON INPUTEVENTS_MV (HADM_ID);

DROP INDEX IF EXISTS INPUTEVENTS_MV_idx03;
CREATE INDEX INPUTEVENTS_MV_idx03
  ON INPUTEVENTS_MV (ICUSTAY_ID);

-- DROP INDEX IF EXISTS INPUTEVENTS_MV_idx04;
-- CREATE INDEX INPUTEVENTS_MV_idx04
--   ON INPUTEVENTS_MV (ENDTIME, STARTTIME);

DROP INDEX IF EXISTS INPUTEVENTS_MV_idx05;
CREATE INDEX INPUTEVENTS_MV_idx05
  ON INPUTEVENTS_MV (ITEMID);

-- DROP INDEX IF EXISTS INPUTEVENTS_MV_idx06;
-- CREATE INDEX INPUTEVENTS_MV_idx06
--   ON INPUTEVENTS_MV (RATE);

-- DROP INDEX IF EXISTS INPUTEVENTS_MV_idx07;
-- CREATE INDEX INPUTEVENTS_MV_idx07
--   ON INPUTEVENTS_MV (VOLUME);

-- DROP INDEX IF EXISTS INPUTEVENTS_MV_idx08;
-- CREATE INDEX INPUTEVENTS_MV_idx08
--   ON INPUTEVENTS_MV (CGID);

-- DROP INDEX IF EXISTS INPUTEVENTS_MV_idx09;
-- CREATE INDEX INPUTEVENTS_MV_idx09
--   ON INPUTEVENTS_MV (LINKORDERID, ORDERID);

-- DROP INDEX IF EXISTS INPUTEVENTS_MV_idx10;
-- CREATE INDEX INPUTEVENTS_MV_idx10
--   ON INPUTEVENTS_MV (ORDERCATEGORYDESCRIPTION,
--     ORDERCATEGORYNAME, SECONDARYORDERCATEGORYNAME);

-- DROP INDEX IF EXISTS INPUTEVENTS_MV_idx11;
-- CREATE INDEX INPUTEVENTS_MV_idx11
--   ON INPUTEVENTS_MV (ORDERCOMPONENTTYPEDESCRIPTION,
--     ORDERCATEGORYDESCRIPTION);


--------------
-- LABEVENTS
--------------

DROP INDEX IF EXISTS LABEVENTS_idx01;
CREATE INDEX LABEVENTS_idx01
  ON LABEVENTS (SUBJECT_ID);

DROP INDEX IF EXISTS LABEVENTS_idx02;
CREATE INDEX LABEVENTS_idx02
  ON LABEVENTS (HADM_ID);

DROP INDEX IF EXISTS LABEVENTS_idx03;
CREATE INDEX LABEVENTS_idx03
  ON LABEVENTS (ITEMID);

-- DROP INDEX IF EXISTS LABEVENTS_idx04;
-- CREATE INDEX LABEVENTS_idx04
--   ON LABEVENTS (VALUE, VALUENUM);

----------------------
-- MICROBIOLOGYEVENTS
----------------------

DROP INDEX IF EXISTS MICROBIOLOGYEVENTS_idx01;
CREATE INDEX MICROBIOLOGYEVENTS_idx01
  ON MICROBIOLOGYEVENTS (SUBJECT_ID);

DROP INDEX IF EXISTS MICROBIOLOGYEVENTS_idx02;
CREATE INDEX MICROBIOLOGYEVENTS_idx02
  ON MICROBIOLOGYEVENTS (HADM_ID);

-- DROP INDEX IF EXISTS MICROBIOLOGYEVENTS_idx03;
-- CREATE INDEX MICROBIOLOGYEVENTS_idx03
--   ON MICROBIOLOGYEVENTS (SPEC_ITEMID,
--     ORG_ITEMID, AB_ITEMID);

---------------
-- NOTEEVENTS
---------------

DROP INDEX IF EXISTS NOTEEVENTS_idx01;
CREATE INDEX NOTEEVENTS_idx01
  ON NOTEEVENTS (SUBJECT_ID);

DROP INDEX IF EXISTS NOTEEVENTS_idx02;
CREATE INDEX NOTEEVENTS_idx02
  ON NOTEEVENTS (HADM_ID);

-- DROP INDEX IF EXISTS NOTEEVENTS_idx03;
-- CREATE INDEX NOTEEVENTS_idx03
--   ON NOTEEVENTS (CGID);

-- DROP INDEX IF EXISTS NOTEEVENTS_idx04;
-- CREATE INDEX NOTEEVENTS_idx04
--   ON NOTEEVENTS (RECORD_ID);

DROP INDEX IF EXISTS NOTEEVENTS_idx05;
CREATE INDEX NOTEEVENTS_idx05
  ON NOTEEVENTS (CATEGORY);


---------------
-- OUTPUTEVENTS
---------------
DROP INDEX IF EXISTS OUTPUTEVENTS_idx01;
CREATE INDEX OUTPUTEVENTS_idx01
  ON OUTPUTEVENTS (SUBJECT_ID);


DROP INDEX IF EXISTS OUTPUTEVENTS_idx02;
CREATE INDEX OUTPUTEVENTS_idx02
  ON OUTPUTEVENTS (ITEMID);


DROP INDEX IF EXISTS OUTPUTEVENTS_idx03;
CREATE INDEX OUTPUTEVENTS_idx03
  ON OUTPUTEVENTS (ICUSTAY_ID);


DROP INDEX IF EXISTS OUTPUTEVENTS_idx04;
CREATE INDEX OUTPUTEVENTS_idx04
  ON OUTPUTEVENTS (HADM_ID);

-- Perhaps not useful to index on just value? Index just for popular subset?
-- DROP INDEX IF EXISTS OUTPUTEVENTS_idx05;
-- CREATE INDEX OUTPUTEVENTS_idx05
--   ON OUTPUTEVENTS (VALUE);


-------------
-- PATIENTS
-------------

-- Note that SUBJECT_ID is already indexed as it is unique

-- DROP INDEX IF EXISTS PATIENTS_idx01;
-- CREATE INDEX PATIENTS_idx01
--   ON PATIENTS (EXPIRE_FLAG);


------------------
-- PRESCRIPTIONS
------------------

DROP INDEX IF EXISTS PRESCRIPTIONS_idx01;
CREATE INDEX PRESCRIPTIONS_idx01
  ON PRESCRIPTIONS (SUBJECT_ID);

DROP INDEX IF EXISTS PRESCRIPTIONS_idx02;
CREATE INDEX PRESCRIPTIONS_idx02
  ON PRESCRIPTIONS (ICUSTAY_ID);

DROP INDEX IF EXISTS PRESCRIPTIONS_idx03;
CREATE INDEX PRESCRIPTIONS_idx03
  ON PRESCRIPTIONS (DRUG_TYPE);

DROP INDEX IF EXISTS PRESCRIPTIONS_idx04;
CREATE INDEX PRESCRIPTIONS_idx04
  ON PRESCRIPTIONS (DRUG);

DROP INDEX IF EXISTS PRESCRIPTIONS_idx05;
CREATE INDEX PRESCRIPTIONS_idx05
  ON PRESCRIPTIONS (HADM_ID);


---------------------
-- PROCEDUREEVENTS_MV
---------------------

DROP INDEX IF EXISTS PROCEDUREEVENTS_MV_idx01;
CREATE INDEX PROCEDUREEVENTS_MV_idx01
  ON PROCEDUREEVENTS_MV (SUBJECT_ID);

DROP INDEX IF EXISTS PROCEDUREEVENTS_MV_idx02;
CREATE INDEX PROCEDUREEVENTS_MV_idx02
  ON PROCEDUREEVENTS_MV (HADM_ID);

DROP INDEX IF EXISTS PROCEDUREEVENTS_MV_idx03;
CREATE INDEX PROCEDUREEVENTS_MV_idx03
  ON PROCEDUREEVENTS_MV (ICUSTAY_ID);

-- DROP INDEX IF EXISTS PROCEDUREEVENTS_MV_idx04;
-- CREATE INDEX PROCEDUREEVENTS_MV_idx04
--   ON PROCEDUREEVENTS_MV (ENDTIME, STARTTIME);

DROP INDEX IF EXISTS PROCEDUREEVENTS_MV_idx05;
CREATE INDEX PROCEDUREEVENTS_MV_idx05
  ON PROCEDUREEVENTS_MV (ITEMID);

-- DROP INDEX IF EXISTS PROCEDUREEVENTS_MV_idx06;
-- CREATE INDEX PROCEDUREEVENTS_MV_idx06
--   ON PROCEDUREEVENTS_MV (VALUE);

-- DROP INDEX IF EXISTS PROCEDUREEVENTS_MV_idx07;
-- CREATE INDEX PROCEDUREEVENTS_MV_idx07
--   ON PROCEDUREEVENTS_MV (CGID);

-- DROP INDEX IF EXISTS PROCEDUREEVENTS_MV_idx08;
-- CREATE INDEX PROCEDUREEVENTS_MV_idx08
--   ON PROCEDUREEVENTS_MV (LINKORDERID, ORDERID);

-- DROP INDEX IF EXISTS PROCEDUREEVENTS_MV_idx09;
-- CREATE INDEX PROCEDUREEVENTS_MV_idx09
--   ON PROCEDUREEVENTS_MV (ORDERCATEGORYDESCRIPTION,
--     ORDERCATEGORYNAME, SECONDARYORDERCATEGORYNAME);

-------------------
-- PROCEDURES_ICD
-------------------

DROP INDEX IF EXISTS PROCEDURES_ICD_idx01;
CREATE INDEX PROCEDURES_ICD_idx01
  ON PROCEDURES_ICD (SUBJECT_ID);

DROP INDEX IF EXISTS PROCEDURES_ICD_idx02;
CREATE INDEX PROCEDURES_ICD_idx02
  ON PROCEDURES_ICD (ICD9_CODE);

DROP INDEX IF EXISTS PROCEDURES_ICD_idx03;
CREATE INDEX PROCEDURES_ICD_idx03
  ON PROCEDURES_ICD (HADM_ID);


-------------
-- SERVICES
-------------

DROP INDEX IF EXISTS SERVICES_idx01;
CREATE INDEX SERVICES_idx01
  ON SERVICES (SUBJECT_ID);

DROP INDEX IF EXISTS SERVICES_idx02;
CREATE INDEX SERVICES_idx02
  ON SERVICES (HADM_ID);

-- DROP INDEX IF EXISTS SERVICES_idx03;
-- CREATE INDEX SERVICES_idx03
--   ON SERVICES (CURR_SERVICE, PREV_SERVICE);

-------------
-- TRANSFERS
-------------

DROP INDEX IF EXISTS TRANSFERS_idx01;
CREATE INDEX TRANSFERS_idx01
  ON TRANSFERS (SUBJECT_ID);

DROP INDEX IF EXISTS TRANSFERS_idx02;
CREATE INDEX TRANSFERS_idx02
  ON TRANSFERS (ICUSTAY_ID);

DROP INDEX IF EXISTS TRANSFERS_idx03;
CREATE INDEX TRANSFERS_idx03
  ON TRANSFERS (HADM_ID);

-- DROP INDEX IF EXISTS TRANSFERS_idx04;
-- CREATE INDEX TRANSFERS_idx04
--   ON TRANSFERS (INTIME, OUTTIME);

-- DROP INDEX IF EXISTS TRANSFERS_idx05;
-- CREATE INDEX TRANSFERS_idx05
--   ON TRANSFERS (LOS);
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/official_mimiciii_postgres/postgres_checks.sql · lines 1–77

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/official_mimiciii_postgres/postgres_checks.sql","part":1,"start_line":1,"end_line":77,"source_sha256":"541248b971f7617075466932136005f40c6a1ce4d7a3e6d8dd51f89cf19b64c4","fragment_sha256":"541248b971f7617075466932136005f40c6a1ce4d7a3e6d8dd51f89cf19b64c4","added_terminal_newline":false} -->
````sql
-- this query runs a few simple checks to make sure the database has loaded in OK
-- These checks are designed for MIMIC-III v1.4

-- If running scripts individually, you can set the schema where all tables are created as follows:
-- SET search_path TO mimiciii;

with expected as
(
select 'admissions' as tbl, 58976 as row_count UNION ALL
select 'callout' as tbl, 34499 as row_count UNION ALL
select 'caregivers' as tbl, 7567 as row_count UNION ALL
select 'chartevents' as tbl, 330712483 as row_count UNION ALL
select 'cptevents' as tbl, 573146 as row_count UNION ALL
select 'd_cpt' as tbl, 134 as row_count UNION ALL
select 'd_icd_diagnoses' as tbl, 14567 as row_count UNION ALL
select 'd_icd_procedures' as tbl, 3882 as row_count UNION ALL
select 'd_items' as tbl, 12487 as row_count UNION ALL
select 'd_labitems' as tbl, 753 as row_count UNION ALL
select 'datetimeevents' as tbl, 4485937 as row_count UNION ALL
select 'diagnoses_icd' as tbl, 651047 as row_count UNION ALL
select 'drgcodes' as tbl, 125557 as row_count UNION ALL
select 'icustays' as tbl, 61532 as row_count UNION ALL
select 'inputevents_cv' as tbl, 17527935 as row_count UNION ALL
select 'inputevents_mv' as tbl, 3618991 as row_count UNION ALL
select 'labevents' as tbl, 27854055 as row_count UNION ALL
select 'microbiologyevents' as tbl, 631726 as row_count UNION ALL
select 'noteevents' as tbl, 2083180 as row_count UNION ALL
select 'outputevents' as tbl, 4349218 as row_count UNION ALL
select 'patients' as tbl, 46520 as row_count UNION ALL
select 'prescriptions' as tbl, 4156450 as row_count UNION ALL
select 'procedureevents_mv' as tbl, 258066 as row_count UNION ALL
select 'procedures_icd' as tbl, 240095 as row_count UNION ALL
select 'services' as tbl, 73343 as row_count UNION ALL
select 'transfers' as tbl, 261897 as row_count
)
, observed as
(
  select 'admissions' as tbl, count(*) as row_count from admissions UNION ALL
  select 'callout' as tbl, count(*) as row_count from callout UNION ALL
  select 'caregivers' as tbl, count(*) as row_count from caregivers UNION ALL
  select 'chartevents' as tbl, count(*) as row_count from chartevents UNION ALL
  select 'cptevents' as tbl, count(*) as row_count from cptevents UNION ALL
  select 'd_cpt' as tbl, count(*) as row_count from d_cpt UNION ALL
  select 'd_icd_diagnoses' as tbl, count(*) as row_count from d_icd_diagnoses UNION ALL
  select 'd_icd_procedures' as tbl, count(*) as row_count from d_icd_procedures UNION ALL
  select 'd_items' as tbl, count(*) as row_count from d_items UNION ALL
  select 'd_labitems' as tbl, count(*) as row_count from d_labitems UNION ALL
  select 'datetimeevents' as tbl, count(*) as row_count from datetimeevents UNION ALL
  select 'diagnoses_icd' as tbl, count(*) as row_count from diagnoses_icd UNION ALL
  select 'drgcodes' as tbl, count(*) as row_count from drgcodes UNION ALL
  select 'icustays' as tbl, count(*) as row_count from icustays UNION ALL
  select 'inputevents_cv' as tbl, count(*) as row_count from inputevents_cv UNION ALL
  select 'inputevents_mv' as tbl, count(*) as row_count from inputevents_mv UNION ALL
  select 'labevents' as tbl, count(*) as row_count from labevents UNION ALL
  select 'microbiologyevents' as tbl, count(*) as row_count from microbiologyevents UNION ALL
  select 'noteevents' as tbl, count(*) as row_count from noteevents UNION ALL
  select 'outputevents' as tbl, count(*) as row_count from outputevents UNION ALL
  select 'patients' as tbl, count(*) as row_count from patients UNION ALL
  select 'prescriptions' as tbl, count(*) as row_count from prescriptions UNION ALL
  select 'procedureevents_mv' as tbl, count(*) as row_count from procedureevents_mv UNION ALL
  select 'procedures_icd' as tbl, count(*) as row_count from procedures_icd UNION ALL
  select 'services' as tbl, count(*) as row_count from services UNION ALL
  select 'transfers' as tbl, count(*) as row_count from transfers
)
select
  exp.tbl
  , exp.row_count as expected_count
  , obs.row_count as observed_count
  , case
      when exp.row_count = obs.row_count
        then 'PASSED'
      else 'FAILED'
    end as ROW_COUNT_CHECK
from expected exp
inner join observed obs
  on exp.tbl = obs.tbl
order by exp.tbl;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/official_mimiciii_postgres/postgres_create_tables.sql · lines 1–672

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/official_mimiciii_postgres/postgres_create_tables.sql","part":1,"start_line":1,"end_line":672,"source_sha256":"3fff0d431caaf75f990bde925cf412cede5db40a534dd5388fec4159b7539894","fragment_sha256":"3fff0d431caaf75f990bde925cf412cede5db40a534dd5388fec4159b7539894","added_terminal_newline":false} -->
````sql
-- -------------------------------------------------------------------------------
--
-- Create the MIMIC-III tables
--
-- -------------------------------------------------------------------------------

--------------------------------------------------------
--  File created - Thursday-November-28-2015
--------------------------------------------------------

-- If running scripts individually, you can set the schema where all tables are created as follows:
-- SET search_path TO mimiciii;

-- Restoring the search path to its default value can be accomplished as follows:
--  SET search_path TO "$user",public;

/* Set the mimic_data_dir variable to point to directory containing
   all .csv files. If using Docker, this should not be changed here.
   Rather, when running the docker container, use the -v option
   to have Docker mount a host volume to the container path /mimic_data
   as explained in the README file
*/


--------------------------------------------------------
--  DDL for Table ADMISSIONS
--------------------------------------------------------

DROP TABLE IF EXISTS ADMISSIONS CASCADE;
CREATE TABLE ADMISSIONS
(
  ROW_ID INT NOT NULL,
  SUBJECT_ID INT NOT NULL,
  HADM_ID INT NOT NULL,
  ADMITTIME TIMESTAMP(0) NOT NULL,
  DISCHTIME TIMESTAMP(0) NOT NULL,
  DEATHTIME TIMESTAMP(0),
  ADMISSION_TYPE VARCHAR(50) NOT NULL,
  ADMISSION_LOCATION VARCHAR(50) NOT NULL,
  DISCHARGE_LOCATION VARCHAR(50) NOT NULL,
  INSURANCE VARCHAR(255) NOT NULL,
  LANGUAGE VARCHAR(10),
  RELIGION VARCHAR(50),
  MARITAL_STATUS VARCHAR(50),
  ETHNICITY VARCHAR(200) NOT NULL,
  EDREGTIME TIMESTAMP(0),
  EDOUTTIME TIMESTAMP(0),
  DIAGNOSIS VARCHAR(255),
  HOSPITAL_EXPIRE_FLAG SMALLINT,
  HAS_CHARTEVENTS_DATA SMALLINT NOT NULL,
  CONSTRAINT adm_rowid_pk PRIMARY KEY (ROW_ID),
  CONSTRAINT adm_hadm_unique UNIQUE (HADM_ID)
) ;

--------------------------------------------------------
--  DDL for Table CALLOUT
--------------------------------------------------------

DROP TABLE IF EXISTS CALLOUT CASCADE;
CREATE TABLE CALLOUT
(
  ROW_ID INT NOT NULL,
  SUBJECT_ID INT NOT NULL,
  HADM_ID INT NOT NULL,
  SUBMIT_WARDID INT,
  SUBMIT_CAREUNIT VARCHAR(15),
  CURR_WARDID INT,
  CURR_CAREUNIT VARCHAR(15),
  CALLOUT_WARDID INT,
  CALLOUT_SERVICE VARCHAR(10) NOT NULL,
  REQUEST_TELE SMALLINT NOT NULL,
  REQUEST_RESP SMALLINT NOT NULL,
  REQUEST_CDIFF SMALLINT NOT NULL,
  REQUEST_MRSA SMALLINT NOT NULL,
  REQUEST_VRE SMALLINT NOT NULL,
  CALLOUT_STATUS VARCHAR(20) NOT NULL,
  CALLOUT_OUTCOME VARCHAR(20) NOT NULL,
  DISCHARGE_WARDID INT,
  ACKNOWLEDGE_STATUS VARCHAR(20) NOT NULL,
  CREATETIME TIMESTAMP(0) NOT NULL,
  UPDATETIME TIMESTAMP(0) NOT NULL,
  ACKNOWLEDGETIME TIMESTAMP(0),
  OUTCOMETIME TIMESTAMP(0) NOT NULL,
  FIRSTRESERVATIONTIME TIMESTAMP(0),
  CURRENTRESERVATIONTIME TIMESTAMP(0),
  CONSTRAINT callout_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table CAREGIVERS
--------------------------------------------------------

DROP TABLE IF EXISTS CAREGIVERS CASCADE;
CREATE TABLE CAREGIVERS
(
  ROW_ID INT NOT NULL,
	CGID INT NOT NULL,
	LABEL VARCHAR(15),
	DESCRIPTION VARCHAR(30),
	CONSTRAINT cg_rowid_pk  PRIMARY KEY (ROW_ID),
	CONSTRAINT cg_cgid_unique UNIQUE (CGID)
) ;

--------------------------------------------------------
--  DDL for Table CHARTEVENTS
--------------------------------------------------------

DROP TABLE IF EXISTS CHARTEVENTS CASCADE;
CREATE TABLE CHARTEVENTS
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT,
	ICUSTAY_ID INT,
	ITEMID INT,
	CHARTTIME TIMESTAMP(0),
	STORETIME TIMESTAMP(0),
	CGID INT,
	VALUE VARCHAR(255),
	VALUENUM DOUBLE PRECISION,
	VALUEUOM VARCHAR(50),
	WARNING INT,
	ERROR INT,
	RESULTSTATUS VARCHAR(50),
	STOPPED VARCHAR(50),
	CONSTRAINT chartevents_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  PARTITION for Table CHARTEVENTS
--------------------------------------------------------

-- CREATE CHARTEVENTS TABLE
 CREATE TABLE chartevents_1 ( CHECK ( itemid >= 0 AND itemid < 127 )) INHERITS (chartevents);
 CREATE TABLE chartevents_2 ( CHECK ( itemid >= 127 AND itemid < 210 )) INHERITS (chartevents);
 CREATE TABLE chartevents_3 ( CHECK ( itemid >= 210 AND itemid < 425 )) INHERITS (chartevents);
 CREATE TABLE chartevents_4 ( CHECK ( itemid >= 425 AND itemid < 549 )) INHERITS (chartevents);
 CREATE TABLE chartevents_5 ( CHECK ( itemid >= 549 AND itemid < 643 )) INHERITS (chartevents);
 CREATE TABLE chartevents_6 ( CHECK ( itemid >= 643 AND itemid < 741 )) INHERITS (chartevents);
 CREATE TABLE chartevents_7 ( CHECK ( itemid >= 741 AND itemid < 1483 )) INHERITS (chartevents);
 CREATE TABLE chartevents_8 ( CHECK ( itemid >= 1483 AND itemid < 3458 )) INHERITS (chartevents);
 CREATE TABLE chartevents_9 ( CHECK ( itemid >= 3458 AND itemid < 3695 )) INHERITS (chartevents);
 CREATE TABLE chartevents_10 ( CHECK ( itemid >= 3695 AND itemid < 8440 )) INHERITS (chartevents);
 CREATE TABLE chartevents_11 ( CHECK ( itemid >= 8440 AND itemid < 8553 )) INHERITS (chartevents);
 CREATE TABLE chartevents_12 ( CHECK ( itemid >= 8553 AND itemid < 220274 )) INHERITS (chartevents);
 CREATE TABLE chartevents_13 ( CHECK ( itemid >= 220274 AND itemid < 223921 )) INHERITS (chartevents);
 CREATE TABLE chartevents_14 ( CHECK ( itemid >= 223921 AND itemid < 224085 )) INHERITS (chartevents);
 CREATE TABLE chartevents_15 ( CHECK ( itemid >= 224085 AND itemid < 224859 )) INHERITS (chartevents);
 CREATE TABLE chartevents_16 ( CHECK ( itemid >= 224859 AND itemid < 227629 )) INHERITS (chartevents);
 CREATE TABLE chartevents_17 ( CHECK ( itemid >= 227629 AND itemid < 999999999 )) INHERITS (chartevents);

-- CREATE CHARTEVENTS TRIGGER
CREATE OR REPLACE FUNCTION chartevents_insert_trigger()
RETURNS TRIGGER AS $$
BEGIN
IF ( NEW.itemid >= 0 AND NEW.itemid < 127 ) THEN INSERT INTO chartevents_1 VALUES (NEW.*);
ELSIF ( NEW.itemid >= 127 AND NEW.itemid < 210 ) THEN INSERT INTO chartevents_2 VALUES (NEW.*);
ELSIF ( NEW.itemid >= 210 AND NEW.itemid < 425 ) THEN INSERT INTO chartevents_3 VALUES (NEW.*);
ELSIF ( NEW.itemid >= 425 AND NEW.itemid < 549 ) THEN INSERT INTO chartevents_4 VALUES (NEW.*);
ELSIF ( NEW.itemid >= 549 AND NEW.itemid < 643 ) THEN INSERT INTO chartevents_5 VALUES (NEW.*);
ELSIF ( NEW.itemid >= 643 AND NEW.itemid < 741 ) THEN INSERT INTO chartevents_6 VALUES (NEW.*);
ELSIF ( NEW.itemid >= 741 AND NEW.itemid < 1483 ) THEN INSERT INTO chartevents_7 VALUES (NEW.*);
ELSIF ( NEW.itemid >= 1483 AND NEW.itemid < 3458 ) THEN INSERT INTO chartevents_8 VALUES (NEW.*);
ELSIF ( NEW.itemid >= 3458 AND NEW.itemid < 3695 ) THEN INSERT INTO chartevents_9 VALUES (NEW.*);
ELSIF ( NEW.itemid >= 3695 AND NEW.itemid < 8440 ) THEN INSERT INTO chartevents_10 VALUES (NEW.*);
ELSIF ( NEW.itemid >= 8440 AND NEW.itemid < 8553 ) THEN INSERT INTO chartevents_11 VALUES (NEW.*);
ELSIF ( NEW.itemid >= 8553 AND NEW.itemid < 220274 ) THEN INSERT INTO chartevents_12 VALUES (NEW.*);
ELSIF ( NEW.itemid >= 220274 AND NEW.itemid < 223921 ) THEN INSERT INTO chartevents_13 VALUES (NEW.*);
ELSIF ( NEW.itemid >= 223921 AND NEW.itemid < 224085 ) THEN INSERT INTO chartevents_14 VALUES (NEW.*);
ELSIF ( NEW.itemid >= 224085 AND NEW.itemid < 224859 ) THEN INSERT INTO chartevents_15 VALUES (NEW.*);
ELSIF ( NEW.itemid >= 224859 AND NEW.itemid < 227629 ) THEN INSERT INTO chartevents_16 VALUES (NEW.*);
ELSIF ( NEW.itemid >= 227629 AND NEW.itemid < 999999999 ) THEN INSERT INTO chartevents_17 VALUES (NEW.*);
ELSE
	INSERT INTO chartevents_null VALUES (NEW.*);
END IF;
RETURN NULL;
END;
$$
LANGUAGE plpgsql;

CREATE TRIGGER insert_chartevents_trigger
    BEFORE INSERT ON chartevents
    FOR EACH ROW EXECUTE PROCEDURE chartevents_insert_trigger();

--------------------------------------------------------
--  DDL for Table CPTEVENTS
--------------------------------------------------------

DROP TABLE IF EXISTS CPTEVENTS CASCADE;
CREATE TABLE CPTEVENTS
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT NOT NULL,
	COSTCENTER VARCHAR(10) NOT NULL,
	CHARTDATE TIMESTAMP(0),
	CPT_CD VARCHAR(10) NOT NULL,
	CPT_NUMBER INT,
	CPT_SUFFIX VARCHAR(5),
	TICKET_ID_SEQ INT,
	SECTIONHEADER VARCHAR(50),
	SUBSECTIONHEADER VARCHAR(255),
	DESCRIPTION VARCHAR(200),
	CONSTRAINT cpt_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table DATETIMEEVENTS
--------------------------------------------------------

DROP TABLE IF EXISTS DATETIMEEVENTS CASCADE;
CREATE TABLE DATETIMEEVENTS
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT,
	ICUSTAY_ID INT,
	ITEMID INT NOT NULL,
	CHARTTIME TIMESTAMP(0) NOT NULL,
	STORETIME TIMESTAMP(0) NOT NULL,
	CGID INT NOT NULL,
	VALUE TIMESTAMP(0),
	VALUEUOM VARCHAR(50) NOT NULL,
	WARNING SMALLINT,
	ERROR SMALLINT,
	RESULTSTATUS VARCHAR(50),
	STOPPED VARCHAR(50),
	CONSTRAINT datetime_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table DIAGNOSES_ICD
--------------------------------------------------------

DROP TABLE IF EXISTS DIAGNOSES_ICD CASCADE;
CREATE TABLE DIAGNOSES_ICD
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT NOT NULL,
	SEQ_NUM INT,
	ICD9_CODE VARCHAR(10),
	CONSTRAINT diagnosesicd_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table DRGCODES
--------------------------------------------------------

DROP TABLE IF EXISTS DRGCODES CASCADE;
CREATE TABLE DRGCODES
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT NOT NULL,
	DRG_TYPE VARCHAR(20) NOT NULL,
	DRG_CODE VARCHAR(20) NOT NULL,
	DESCRIPTION VARCHAR(255),
	DRG_SEVERITY SMALLINT,
	DRG_MORTALITY SMALLINT,
	CONSTRAINT drg_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table D_CPT
--------------------------------------------------------

DROP TABLE IF EXISTS D_CPT CASCADE;
CREATE TABLE D_CPT
(
  ROW_ID INT NOT NULL,
	CATEGORY SMALLINT NOT NULL,
	SECTIONRANGE VARCHAR(100) NOT NULL,
	SECTIONHEADER VARCHAR(50) NOT NULL,
	SUBSECTIONRANGE VARCHAR(100) NOT NULL,
	SUBSECTIONHEADER VARCHAR(255) NOT NULL,
	CODESUFFIX VARCHAR(5),
	MINCODEINSUBSECTION INT NOT NULL,
	MAXCODEINSUBSECTION INT NOT NULL,
	CONSTRAINT dcpt_ssrange_unique UNIQUE (SUBSECTIONRANGE),
	CONSTRAINT dcpt_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table D_ICD_DIAGNOSES
--------------------------------------------------------

DROP TABLE IF EXISTS D_ICD_DIAGNOSES CASCADE;
CREATE TABLE D_ICD_DIAGNOSES
(
  ROW_ID INT NOT NULL,
	ICD9_CODE VARCHAR(10) NOT NULL,
	SHORT_TITLE VARCHAR(50) NOT NULL,
	LONG_TITLE VARCHAR(255) NOT NULL,
	CONSTRAINT d_icd_diag_code_unique UNIQUE (ICD9_CODE),
	CONSTRAINT d_icd_diag_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table D_ICD_PROCEDURES
--------------------------------------------------------

DROP TABLE IF EXISTS D_ICD_PROCEDURES CASCADE;
CREATE TABLE D_ICD_PROCEDURES
(
  ROW_ID INT NOT NULL,
	ICD9_CODE VARCHAR(10) NOT NULL,
	SHORT_TITLE VARCHAR(50) NOT NULL,
	LONG_TITLE VARCHAR(255) NOT NULL,
	CONSTRAINT d_icd_proc_code_unique UNIQUE (ICD9_CODE),
	CONSTRAINT d_icd_proc_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table D_ITEMS
--------------------------------------------------------

DROP TABLE IF EXISTS D_ITEMS CASCADE;
CREATE TABLE D_ITEMS
(
  ROW_ID INT NOT NULL,
	ITEMID INT NOT NULL,
	LABEL VARCHAR(200),
	ABBREVIATION VARCHAR(100),
	DBSOURCE VARCHAR(20),
	LINKSTO VARCHAR(50),
	CATEGORY VARCHAR(100),
	UNITNAME VARCHAR(100),
	PARAM_TYPE VARCHAR(30),
	CONCEPTID INT,
	CONSTRAINT ditems_itemid_unique UNIQUE (ITEMID),
	CONSTRAINT ditems_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table D_LABITEMS
--------------------------------------------------------

DROP TABLE IF EXISTS D_LABITEMS CASCADE;
CREATE TABLE D_LABITEMS
(
  ROW_ID INT NOT NULL,
	ITEMID INT NOT NULL,
	LABEL VARCHAR(100) NOT NULL,
	FLUID VARCHAR(100) NOT NULL,
	CATEGORY VARCHAR(100) NOT NULL,
	LOINC_CODE VARCHAR(100),
	CONSTRAINT dlabitems_itemid_unique UNIQUE (ITEMID),
	CONSTRAINT dlabitems_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table ICUSTAYS
--------------------------------------------------------

DROP TABLE IF EXISTS ICUSTAYS CASCADE;
CREATE TABLE ICUSTAYS
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT NOT NULL,
	ICUSTAY_ID INT NOT NULL,
	DBSOURCE VARCHAR(20) NOT NULL,
	FIRST_CAREUNIT VARCHAR(20) NOT NULL,
	LAST_CAREUNIT VARCHAR(20) NOT NULL,
	FIRST_WARDID SMALLINT NOT NULL,
	LAST_WARDID SMALLINT NOT NULL,
	INTIME TIMESTAMP(0) NOT NULL,
	OUTTIME TIMESTAMP(0),
	LOS DOUBLE PRECISION,
	CONSTRAINT icustay_icustayid_unique UNIQUE (ICUSTAY_ID),
	CONSTRAINT icustay_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table INPUTEVENTS_CV
--------------------------------------------------------

DROP TABLE IF EXISTS INPUTEVENTS_CV CASCADE;
CREATE TABLE INPUTEVENTS_CV
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT,
	ICUSTAY_ID INT,
	CHARTTIME TIMESTAMP(0),
	ITEMID INT,
	AMOUNT DOUBLE PRECISION,
	AMOUNTUOM VARCHAR(30),
	RATE DOUBLE PRECISION,
	RATEUOM VARCHAR(30),
	STORETIME TIMESTAMP(0),
	CGID INT,
	ORDERID INT,
	LINKORDERID INT,
	STOPPED VARCHAR(30),
	NEWBOTTLE INT,
	ORIGINALAMOUNT DOUBLE PRECISION,
	ORIGINALAMOUNTUOM VARCHAR(30),
	ORIGINALROUTE VARCHAR(30),
	ORIGINALRATE DOUBLE PRECISION,
	ORIGINALRATEUOM VARCHAR(30),
	ORIGINALSITE VARCHAR(30),
	CONSTRAINT inputevents_cv_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table INPUTEVENTS_MV
--------------------------------------------------------

DROP TABLE IF EXISTS INPUTEVENTS_MV CASCADE;
CREATE TABLE INPUTEVENTS_MV
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT,
	ICUSTAY_ID INT,
	STARTTIME TIMESTAMP(0),
	ENDTIME TIMESTAMP(0),
	ITEMID INT,
	AMOUNT DOUBLE PRECISION,
	AMOUNTUOM VARCHAR(30),
	RATE DOUBLE PRECISION,
	RATEUOM VARCHAR(30),
	STORETIME TIMESTAMP(0),
	CGID INT,
	ORDERID INT,
	LINKORDERID INT,
	ORDERCATEGORYNAME VARCHAR(100),
	SECONDARYORDERCATEGORYNAME VARCHAR(100),
	ORDERCOMPONENTTYPEDESCRIPTION VARCHAR(200),
	ORDERCATEGORYDESCRIPTION VARCHAR(50),
	PATIENTWEIGHT DOUBLE PRECISION,
	TOTALAMOUNT DOUBLE PRECISION,
	TOTALAMOUNTUOM VARCHAR(50),
	ISOPENBAG SMALLINT,
	CONTINUEINNEXTDEPT SMALLINT,
	CANCELREASON SMALLINT,
	STATUSDESCRIPTION VARCHAR(30),
	COMMENTS_EDITEDBY VARCHAR(30),
	COMMENTS_CANCELEDBY VARCHAR(40),
	COMMENTS_DATE TIMESTAMP(0),
	ORIGINALAMOUNT DOUBLE PRECISION,
	ORIGINALRATE DOUBLE PRECISION,
	CONSTRAINT inputevents_mv_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table LABEVENTS
--------------------------------------------------------

DROP TABLE IF EXISTS LABEVENTS CASCADE;
CREATE TABLE LABEVENTS
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT,
	ITEMID INT NOT NULL,
	CHARTTIME TIMESTAMP(0),
	VALUE VARCHAR(200),
	VALUENUM DOUBLE PRECISION,
	VALUEUOM VARCHAR(20),
	FLAG VARCHAR(20),
	CONSTRAINT labevents_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table MICROBIOLOGYEVENTS
--------------------------------------------------------

DROP TABLE IF EXISTS MICROBIOLOGYEVENTS CASCADE;
CREATE TABLE MICROBIOLOGYEVENTS
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT,
	CHARTDATE TIMESTAMP(0),
	CHARTTIME TIMESTAMP(0),
	SPEC_ITEMID INT,
	SPEC_TYPE_DESC VARCHAR(100),
	ORG_ITEMID INT,
	ORG_NAME VARCHAR(100),
	ISOLATE_NUM SMALLINT,
	AB_ITEMID INT,
	AB_NAME VARCHAR(30),
	DILUTION_TEXT VARCHAR(10),
	DILUTION_COMPARISON VARCHAR(20),
	DILUTION_VALUE DOUBLE PRECISION,
	INTERPRETATION VARCHAR(5),
	CONSTRAINT micro_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table NOTEEVENTS
--------------------------------------------------------

DROP TABLE IF EXISTS NOTEEVENTS CASCADE;
CREATE TABLE NOTEEVENTS
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT,
	CHARTDATE TIMESTAMP(0),
	CHARTTIME TIMESTAMP(0),
	STORETIME TIMESTAMP(0),
	CATEGORY VARCHAR(50),
	DESCRIPTION VARCHAR(255),
	CGID INT,
	ISERROR CHAR(1),
	TEXT TEXT,
	CONSTRAINT noteevents_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table OUTPUTEVENTS
--------------------------------------------------------

DROP TABLE IF EXISTS OUTPUTEVENTS CASCADE;
CREATE TABLE OUTPUTEVENTS
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT,
	ICUSTAY_ID INT,
	CHARTTIME TIMESTAMP(0),
	ITEMID INT,
	VALUE DOUBLE PRECISION,
	VALUEUOM VARCHAR(30),
	STORETIME TIMESTAMP(0),
	CGID INT,
	STOPPED VARCHAR(30),
	NEWBOTTLE CHAR(1),
	ISERROR INT,
	CONSTRAINT outputevents_cv_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table PATIENTS
--------------------------------------------------------

DROP TABLE IF EXISTS PATIENTS CASCADE;
CREATE TABLE PATIENTS
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	GENDER VARCHAR(5) NOT NULL,
	DOB TIMESTAMP(0) NOT NULL,
	DOD TIMESTAMP(0),
	DOD_HOSP TIMESTAMP(0),
	DOD_SSN TIMESTAMP(0),
	EXPIRE_FLAG INT NOT NULL,
	CONSTRAINT pat_subid_unique UNIQUE (SUBJECT_ID),
	CONSTRAINT pat_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table PRESCRIPTIONS
--------------------------------------------------------

DROP TABLE IF EXISTS PRESCRIPTIONS CASCADE;
CREATE TABLE PRESCRIPTIONS
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT NOT NULL,
	ICUSTAY_ID INT,
	STARTDATE TIMESTAMP(0),
	ENDDATE TIMESTAMP(0),
	DRUG_TYPE VARCHAR(100) NOT NULL,
	DRUG VARCHAR(100) NOT NULL,
	DRUG_NAME_POE VARCHAR(100),
	DRUG_NAME_GENERIC VARCHAR(100),
	FORMULARY_DRUG_CD VARCHAR(120),
	GSN VARCHAR(200),
	NDC VARCHAR(120),
	PROD_STRENGTH VARCHAR(120),
	DOSE_VAL_RX VARCHAR(120),
	DOSE_UNIT_RX VARCHAR(120),
	FORM_VAL_DISP VARCHAR(120),
	FORM_UNIT_DISP VARCHAR(120),
	ROUTE VARCHAR(120),
	CONSTRAINT prescription_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table PROCEDUREEVENTS_MV
--------------------------------------------------------

DROP TABLE IF EXISTS PROCEDUREEVENTS_MV CASCADE;
CREATE TABLE PROCEDUREEVENTS_MV
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT NOT NULL,
	ICUSTAY_ID INT,
	STARTTIME TIMESTAMP(0),
	ENDTIME TIMESTAMP(0),
	ITEMID INT,
	VALUE DOUBLE PRECISION,
	VALUEUOM VARCHAR(30),
	LOCATION VARCHAR(30),
	LOCATIONCATEGORY VARCHAR(30),
	STORETIME TIMESTAMP(0),
	CGID INT,
	ORDERID INT,
	LINKORDERID INT,
	ORDERCATEGORYNAME VARCHAR(100),
	SECONDARYORDERCATEGORYNAME VARCHAR(100),
	ORDERCATEGORYDESCRIPTION VARCHAR(50),
	ISOPENBAG SMALLINT,
	CONTINUEINNEXTDEPT SMALLINT,
	CANCELREASON SMALLINT,
	STATUSDESCRIPTION VARCHAR(30),
	COMMENTS_EDITEDBY VARCHAR(30),
	COMMENTS_CANCELEDBY VARCHAR(30),
	COMMENTS_DATE TIMESTAMP(0),
	CONSTRAINT procedureevents_mv_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table PROCEDURES_ICD
--------------------------------------------------------

DROP TABLE IF EXISTS PROCEDURES_ICD CASCADE;
CREATE TABLE PROCEDURES_ICD
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT NOT NULL,
	SEQ_NUM INT NOT NULL,
	ICD9_CODE VARCHAR(10) NOT NULL,
	CONSTRAINT proceduresicd_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table SERVICES
--------------------------------------------------------

DROP TABLE IF EXISTS SERVICES CASCADE;
CREATE TABLE SERVICES
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT NOT NULL,
	TRANSFERTIME TIMESTAMP(0) NOT NULL,
	PREV_SERVICE VARCHAR(20),
	CURR_SERVICE VARCHAR(20),
	CONSTRAINT services_rowid_pk PRIMARY KEY (ROW_ID)
) ;

--------------------------------------------------------
--  DDL for Table TRANSFERS
--------------------------------------------------------

DROP TABLE IF EXISTS TRANSFERS CASCADE;
CREATE TABLE TRANSFERS
(
  ROW_ID INT NOT NULL,
	SUBJECT_ID INT NOT NULL,
	HADM_ID INT NOT NULL,
	ICUSTAY_ID INT,
	DBSOURCE VARCHAR(20),
	EVENTTYPE VARCHAR(20),
	PREV_CAREUNIT VARCHAR(20),
	CURR_CAREUNIT VARCHAR(20),
	PREV_WARDID SMALLINT,
	CURR_WARDID SMALLINT,
	INTIME TIMESTAMP(0),
	OUTTIME TIMESTAMP(0),
	LOS DOUBLE PRECISION,
	CONSTRAINT transfers_rowid_pk PRIMARY KEY (ROW_ID)
) ;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/official_mimiciii_postgres/postgres_load_data.sql · lines 1–181

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/official_mimiciii_postgres/postgres_load_data.sql","part":1,"start_line":1,"end_line":181,"source_sha256":"5c21241561fdb380fb87f79a92a2d154a8fc2277572ecf62df90a1519a26e394","fragment_sha256":"5c21241561fdb380fb87f79a92a2d154a8fc2277572ecf62df90a1519a26e394","added_terminal_newline":false} -->
````sql
-- -------------------------------------------------------------------------------
--
-- Load data into the MIMIC-III schema
--
-- -------------------------------------------------------------------------------

--------------------------------------------------------
--  File created - Thursday-August-27-2015
--------------------------------------------------------

-- Change to the directory containing the data files
\cd :mimic_data_dir

-- If running scripts individually, you can set the schema where all tables are created as follows:
-- SET search_path TO mimiciii;

-- Restoring the search path to its default value can be accomplished as follows:
-- SET search_path TO "$user",public;

/* Set the mimic_data_dir variable to point to directory containing
   all .csv files. If using Docker, this should not be changed here.
   Rather, when running the docker container, use the -v option
   to have Docker mount a host volume to the container path /mimic_data
   as explained in the README file
*/

--------------------------------------------------------
--  Load Data for Table ADMISSIONS
--------------------------------------------------------

\copy ADMISSIONS FROM 'ADMISSIONS.csv' DELIMITER ',' CSV HEADER NULL ''

--------------------------------------------------------
--  Load Data for Table CALLOUT
--------------------------------------------------------

\copy CALLOUT from 'CALLOUT.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table CAREGIVERS
--------------------------------------------------------

\copy CAREGIVERS from 'CAREGIVERS.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table CHARTEVENTS
--------------------------------------------------------

\copy CHARTEVENTS from 'CHARTEVENTS.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table CPTEVENTS
--------------------------------------------------------

\copy CPTEVENTS from 'CPTEVENTS.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table DATETIMEEVENTS
--------------------------------------------------------

\copy DATETIMEEVENTS from 'DATETIMEEVENTS.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table DIAGNOSES_ICD
--------------------------------------------------------

\copy DIAGNOSES_ICD from 'DIAGNOSES_ICD.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table DRGCODES
--------------------------------------------------------

\copy DRGCODES from 'DRGCODES.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table D_CPT
--------------------------------------------------------

\copy D_CPT from 'D_CPT.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table D_ICD_DIAGNOSES
--------------------------------------------------------

\copy D_ICD_DIAGNOSES from 'D_ICD_DIAGNOSES.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table D_ICD_PROCEDURES
--------------------------------------------------------

\copy D_ICD_PROCEDURES from 'D_ICD_PROCEDURES.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table D_ITEMS
--------------------------------------------------------

\copy D_ITEMS from 'D_ITEMS.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table D_LABITEMS
--------------------------------------------------------

\copy D_LABITEMS from 'D_LABITEMS.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table ICUSTAYS
--------------------------------------------------------

\copy ICUSTAYS from 'ICUSTAYS.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table INPUTEVENTS_CV
--------------------------------------------------------

\copy INPUTEVENTS_CV from 'INPUTEVENTS_CV.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table INPUTEVENTS_MV
--------------------------------------------------------

\copy INPUTEVENTS_MV from 'INPUTEVENTS_MV.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table LABEVENTS
--------------------------------------------------------

\copy LABEVENTS from 'LABEVENTS.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table MICROBIOLOGYEVENTS
--------------------------------------------------------

\copy MICROBIOLOGYEVENTS from 'MICROBIOLOGYEVENTS.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table NOTEEVENTS
--------------------------------------------------------

\copy NOTEEVENTS from 'NOTEEVENTS.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table OUTPUTEVENTS
--------------------------------------------------------

\copy OUTPUTEVENTS from 'OUTPUTEVENTS.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table PATIENTS
--------------------------------------------------------

\copy PATIENTS from 'PATIENTS.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table PRESCRIPTIONS
--------------------------------------------------------

\copy PRESCRIPTIONS from 'PRESCRIPTIONS.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table PROCEDUREEVENTS_MV
--------------------------------------------------------

\copy PROCEDUREEVENTS_MV from 'PROCEDUREEVENTS_MV.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table PROCEDURES_ICD
--------------------------------------------------------

\copy PROCEDURES_ICD from 'PROCEDURES_ICD.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table SERVICES
--------------------------------------------------------

\copy SERVICES from 'SERVICES.csv' delimiter ',' csv header NULL ''

--------------------------------------------------------
--  Load Data for Table TRANSFERS
--------------------------------------------------------

\copy TRANSFERS from 'TRANSFERS.csv' delimiter ',' csv header NULL ''
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/run_pipeline.ps1 · lines 1–103

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/run_pipeline.ps1","part":1,"start_line":1,"end_line":103,"source_sha256":"e8c9d9d70f41ba4f3c2a3cf839d01c32658b543b354f0a7ac0713032706b01ae","fragment_sha256":"e8c9d9d70f41ba4f3c2a3cf839d01c32658b543b354f0a7ac0713032706b01ae","added_terminal_newline":false} -->
````powershell
param(
    [ValidateSet('All','InventoryPhenotype','CohortNote','StructuredMaster','LandmarkAudit','Verify')]
    [string]$Stage = 'All',
    [string]$Database = 'mimiciv',
    [string]$HostName = '127.0.0.1',
    [string]$Username = 'postgres',
    [string]$PsqlPath = 'psql'
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$createdPassword = $false
$executionTimes = [System.Collections.Generic.List[object]]::new()

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'outputs'))) {
    New-Item -ItemType Directory -Path (Join-Path $projectRoot 'outputs') | Out-Null
}

if ($PsqlPath -eq 'psql' -and -not (Get-Command psql -ErrorAction SilentlyContinue)) {
    $detectedPsql = @(
        'C:\Program Files\PostgreSQL\18\bin\psql.exe',
        'C:\Program Files\PostgreSQL\17\bin\psql.exe',
        'C:\Program Files\PostgreSQL\16\bin\psql.exe',
        'C:\Program Files\PostgreSQL\15\bin\psql.exe'
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($detectedPsql) {
        $PsqlPath = $detectedPsql
    }
    else {
        throw '找不到 psql。请安装 PostgreSQL 客户端，或通过 -PsqlPath 指定 psql.exe 的完整路径。'
    }
}

if (-not $env:PGPASSWORD) {
    $securePassword = Read-Host '请输入 PostgreSQL 密码' -AsSecureString
    $passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
    try {
        $env:PGPASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
        $createdPassword = $true
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
    }
}

function Invoke-ProjectSql {
    param([string]$FileName)
    Write-Host "正在执行 $FileName"
    $startedAt = Get-Date
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $status = 'passed'
    try {
        & $PsqlPath -h $HostName -U $Username -d $Database -v ON_ERROR_STOP=1 -P pager=off -f (Join-Path 'sql' $FileName)
        if ($LASTEXITCODE -ne 0) {
            $status = 'failed'
            throw "$FileName 执行失败，退出码 $LASTEXITCODE"
        }
    }
    catch {
        $status = 'failed'
        throw
    }
    finally {
        $stopwatch.Stop()
        $executionTimes.Add([PSCustomObject]@{
            script = $FileName
            started_at = $startedAt.ToString('yyyy-MM-dd HH:mm:ss')
            elapsed_seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
            status = $status
        })
    }
}

$groups = @{
    InventoryPhenotype = @('00_inventory.sql','01_icd_phenotype_audit.sql')
    CohortNote = @('02_build_strict_aecopd.sql','03_longitudinal_history.sql','04_note_coverage.sql')
    StructuredMaster = @('05_structured_field_audit.sql','06_master_cohort.sql')
    LandmarkAudit = @('07_two_landmark_availability_audit.sql')
    Verify = @('99_verify.sql')
}

Push-Location $projectRoot
try {
    $selected = switch ($Stage) {
        'All' { $groups.InventoryPhenotype + $groups.CohortNote + $groups.StructuredMaster + $groups.LandmarkAudit + $groups.Verify }
        default { $groups[$Stage] }
    }
    foreach ($script in $selected) {
        Invoke-ProjectSql -FileName $script
    }
    Write-Host '执行完成。请查看 outputs 目录和 docs\EXECUTION_REPORT.md。'
}
finally {
    if ($executionTimes.Count -gt 0) {
        $executionTimes | Export-Csv -LiteralPath (Join-Path $projectRoot 'outputs\execution_times.csv') -NoTypeInformation -Encoding utf8
    }
    Pop-Location
    if ($createdPassword) {
        Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    }
}


````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/SOURCE_SHA256SUMS.txt · lines 1–12

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/SOURCE_SHA256SUMS.txt","part":1,"start_line":1,"end_line":12,"source_sha256":"e7bc1258aedbb6970ef7b279ba5be184d85a9976d1a0f07854ece0b6abcdee3f","fragment_sha256":"e7bc1258aedbb6970ef7b279ba5be184d85a9976d1a0f07854ece0b6abcdee3f","added_terminal_newline":false} -->
````text
# 2026-09-19 从原始目录读取的源文件 SHA-256；用于溯源，不包含患者级输出。
199b759b65bb8a63b6fba55009ce1193cef5873e48251796e6966d1251af3e64  sql/00_inventory.sql
ce5cc00bed121765d12bd65279af6d54eb7b070382b7619b179b1c772bdd1ab0  sql/01_icd_phenotype_audit.sql
974dd0d13f56e0c28bdb62246f99866e11d15c51b34bef7605dadd3787ebc317  sql/02_build_strict_aecopd.sql
f689d4ba00c2ee0c40e19334eca742bdace9c59c908cc500bf114c793f2e39c5  sql/03_longitudinal_history.sql
a3969cec2a70923a6338f6d17cd04863778a82fb28ce4c7dc96588ca2da9a851  sql/04_note_coverage.sql
015787d7927239703e7e1e14269d148b8b49886686f9793276b912340a2ca83f  sql/05_structured_field_audit.sql
dd2f82d17dd25712263651e51ec161f299165037d976ba3af60eb8dfbf446d49  sql/06_master_cohort.sql
feaf87c8f63d47f1340a21e7ae900976469687003a681bcadabf5af868a43313  sql/07_two_landmark_availability_audit.sql
e2333d92dccdfdfb6c289dd8e3d1dbde260defd298b784c2d6cadbc8a23fd303  sql/99_verify.sql
1b49bd81664f71f3a66bf45928c7a15b356eeb35f5a9d83965131effaa36a0a9  run_pipeline.ps1

````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/00_inventory.sql · lines 1–19

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/00_inventory.sql","part":1,"start_line":1,"end_line":19,"source_sha256":"f3033194bf8920ca17fc7b9289c8986f1f7d1f487e847cf7f701bbf5f551892d","fragment_sha256":"f3033194bf8920ca17fc7b9289c8986f1f7d1f487e847cf7f701bbf5f551892d","added_terminal_newline":false} -->
````sql
\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';

\copy (SELECT extname, extversion FROM pg_extension ORDER BY extname) TO 'outputs/00_postgresql_extensions.csv' CSV HEADER

\copy (SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema IN ('mimiciv_hosp','mimiciv_icu','mimiciv_note','mimiciv_derived') AND table_type = 'BASE TABLE' ORDER BY table_schema, table_name) TO 'outputs/00_schema_tables.csv' CSV HEADER

\copy (SELECT table_schema, table_name, ordinal_position, column_name, data_type, is_nullable FROM information_schema.columns WHERE table_schema IN ('mimiciv_hosp','mimiciv_icu','mimiciv_note','mimiciv_derived') ORDER BY table_schema, table_name, ordinal_position) TO 'outputs/00_schema_columns.csv' CSV HEADER

\copy (SELECT n.nspname AS table_schema, c.relname AS table_name, c.reltuples::bigint AS estimated_rows, pg_total_relation_size(c.oid) AS total_bytes FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relkind IN ('r','p') AND n.nspname IN ('mimiciv_hosp','mimiciv_icu','mimiciv_note','mimiciv_derived') ORDER BY n.nspname, c.relname) TO 'outputs/00_table_row_estimates.csv' CSV HEADER

\copy (SELECT 'mimiciv_hosp.patients' AS table_name, COUNT(*) AS exact_rows FROM mimiciv_hosp.patients UNION ALL SELECT 'mimiciv_hosp.admissions', COUNT(*) FROM mimiciv_hosp.admissions UNION ALL SELECT 'mimiciv_hosp.diagnoses_icd', COUNT(*) FROM mimiciv_hosp.diagnoses_icd UNION ALL SELECT 'mimiciv_hosp.d_icd_diagnoses', COUNT(*) FROM mimiciv_hosp.d_icd_diagnoses UNION ALL SELECT 'mimiciv_note.discharge', COUNT(*) FROM mimiciv_note.discharge UNION ALL SELECT 'mimiciv_icu.icustays', COUNT(*) FROM mimiciv_icu.icustays) TO 'outputs/00_core_table_counts.csv' CSV HEADER

\copy (SELECT MIN(admittime) AS min_admittime, MAX(admittime) AS max_admittime, MIN(dischtime) AS min_dischtime, MAX(dischtime) AS max_dischtime, COUNT(*) AS admissions FROM mimiciv_hosp.admissions) TO 'outputs/00_admissions_time_range.csv' CSV HEADER

\copy (SELECT COUNT(*) AS diagnoses_rows, COUNT(*) FILTER (WHERE a.hadm_id IS NOT NULL) AS matched_admissions_rows, COUNT(*) FILTER (WHERE p.subject_id IS NOT NULL) AS matched_patients_rows, COUNT(*) FILTER (WHERE a.hadm_id IS NULL) AS unmatched_admissions_rows, COUNT(*) FILTER (WHERE p.subject_id IS NULL) AS unmatched_patients_rows FROM mimiciv_hosp.diagnoses_icd d LEFT JOIN mimiciv_hosp.admissions a ON a.subject_id=d.subject_id AND a.hadm_id=d.hadm_id LEFT JOIN mimiciv_hosp.patients p ON p.subject_id=d.subject_id) TO 'outputs/00_core_join_audit.csv' CSV HEADER



````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/01_icd_phenotype_audit.sql · lines 1–88

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/01_icd_phenotype_audit.sql","part":1,"start_line":1,"end_line":88,"source_sha256":"ec5e67b579a16a5f1410fb6879fff48bfe20a326fe6f251ec39b989f12e174bd","fragment_sha256":"ec5e67b579a16a5f1410fb6879fff48bfe20a326fe6f251ec39b989f12e174bd","added_terminal_newline":false} -->
````sql
\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;

CREATE SCHEMA IF NOT EXISTS project_aecopd;

DROP TABLE IF EXISTS project_aecopd.aecopd_codebook_audit;
CREATE TABLE project_aecopd.aecopd_codebook_audit AS
WITH dictionary_candidates AS (
  SELECT
    icd_version,
    icd_code,
    long_title,
    CASE
      WHEN (icd_version = 10 AND icd_code = 'J441')
        OR (icd_version = 9 AND icd_code = '49121') THEN 'strict'
      WHEN (icd_version = 10 AND icd_code = 'J440')
        OR (icd_version = 9 AND icd_code = '49122') THEN 'expanded_addition'
      WHEN (icd_version = 10 AND (icd_code LIKE 'J41%' OR icd_code = 'J42' OR icd_code LIKE 'J43%' OR icd_code LIKE 'J44%'))
        OR (icd_version = 9 AND (icd_code LIKE '491%' OR icd_code LIKE '492%' OR icd_code = '496')) THEN 'copd_all_candidate'
      ELSE 'dictionary_text_review'
    END AS phenotype_group,
    CASE
      WHEN (icd_version = 10 AND (icd_code LIKE 'J41%' OR icd_code = 'J42' OR icd_code LIKE 'J43%' OR icd_code LIKE 'J44%'))
        OR (icd_version = 9 AND (icd_code LIKE '491%' OR icd_code LIKE '492%' OR icd_code = '496')) THEN TRUE
      ELSE FALSE
    END AS included_in_copd_all_candidate,
    CASE
      WHEN lower(long_title) LIKE '%asthma%' THEN 'exclude from COPD-all candidate: asthma terminology requires separate clinical decision'
      WHEN lower(long_title) LIKE '%pyelonephritis%' OR lower(long_title) LIKE '%subcutaneous%' OR lower(long_title) LIKE '%perinatal%' OR lower(long_title) LIKE '%interstitial emphysema%' OR lower(long_title) LIKE '%compensatory emphysema%' OR lower(long_title) LIKE '%screening%' THEN 'exclude: non-COPD pulmonary phenotype or non-pulmonary context'
      ELSE 'candidate based on local ICD title/code family; requires clinical confirmation'
    END AS audit_note
  FROM mimiciv_hosp.d_icd_diagnoses
  WHERE icd_code IN ('J441','J440','49121','49122','496')
     OR lower(long_title) LIKE '%chronic obstructive%'
     OR lower(long_title) LIKE '%emphysema%'
     OR lower(long_title) LIKE '%chronic bronchitis%'
)
SELECT * FROM dictionary_candidates;

DROP TABLE IF EXISTS project_aecopd.phenotype_counts_audit;
CREATE TABLE project_aecopd.phenotype_counts_audit AS
WITH definitions AS (
  SELECT 'strict_any'::text AS phenotype, d.subject_id, d.hadm_id
  FROM mimiciv_hosp.diagnoses_icd d
  WHERE (d.icd_version=10 AND d.icd_code='J441') OR (d.icd_version=9 AND d.icd_code='49121')
  UNION ALL
  SELECT 'strict_seq1', d.subject_id, d.hadm_id
  FROM mimiciv_hosp.diagnoses_icd d
  WHERE d.seq_num=1 AND ((d.icd_version=10 AND d.icd_code='J441') OR (d.icd_version=9 AND d.icd_code='49121'))
  UNION ALL
  SELECT CASE WHEN d.icd_version=10 THEN 'expanded_add_J440' ELSE 'expanded_add_49122' END, d.subject_id, d.hadm_id
  FROM mimiciv_hosp.diagnoses_icd d
  WHERE (d.icd_version=10 AND d.icd_code='J440') OR (d.icd_version=9 AND d.icd_code='49122')
  UNION ALL
  SELECT 'expanded_union', d.subject_id, d.hadm_id
  FROM mimiciv_hosp.diagnoses_icd d
  WHERE (d.icd_version=10 AND d.icd_code IN ('J441','J440')) OR (d.icd_version=9 AND d.icd_code IN ('49121','49122'))
  UNION ALL
  SELECT 'copd_all_candidate', d.subject_id, d.hadm_id
  FROM mimiciv_hosp.diagnoses_icd d
  WHERE (d.icd_version=10 AND (d.icd_code LIKE 'J41%' OR d.icd_code='J42' OR d.icd_code LIKE 'J43%' OR d.icd_code LIKE 'J44%'))
     OR (d.icd_version=9 AND (d.icd_code LIKE '491%' OR d.icd_code LIKE '492%' OR d.icd_code='496'))
), dedup AS (
  SELECT DISTINCT phenotype, subject_id, hadm_id FROM definitions
), strict_admissions AS (
  SELECT subject_id, hadm_id FROM dedup WHERE phenotype='strict_any'
), strict_patients AS (
  SELECT DISTINCT subject_id FROM strict_admissions
)
SELECT
  d.phenotype,
  COUNT(DISTINCT d.subject_id) AS patients,
  COUNT(*) AS admissions,
  COUNT(DISTINCT d.subject_id) FILTER (WHERE sp.subject_id IS NULL) AS new_patients_vs_strict,
  COUNT(*) FILTER (WHERE sa.hadm_id IS NULL) AS new_admissions_vs_strict
FROM dedup d
LEFT JOIN strict_admissions sa ON sa.subject_id=d.subject_id AND sa.hadm_id=d.hadm_id
LEFT JOIN strict_patients sp ON sp.subject_id=d.subject_id
GROUP BY d.phenotype;

COMMIT;

\copy (SELECT * FROM project_aecopd.aecopd_codebook_audit ORDER BY icd_version, icd_code) TO 'outputs/01_aecopd_codebook.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.phenotype_counts_audit ORDER BY CASE phenotype WHEN 'strict_any' THEN 1 WHEN 'strict_seq1' THEN 2 WHEN 'expanded_add_J440' THEN 3 WHEN 'expanded_add_49122' THEN 4 WHEN 'expanded_union' THEN 5 ELSE 6 END) TO 'outputs/01_phenotype_counts.csv' CSV HEADER



````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/02_build_strict_aecopd.sql · lines 1–77

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/02_build_strict_aecopd.sql","part":1,"start_line":1,"end_line":77,"source_sha256":"22e6b027abf0cf0bbdddf3c17dd5848d67d565652d3f538863067ac567ff8612","fragment_sha256":"22e6b027abf0cf0bbdddf3c17dd5848d67d565652d3f538863067ac567ff8612","added_terminal_newline":false} -->
````sql
\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;
CREATE SCHEMA IF NOT EXISTS project_aecopd;

DROP TABLE IF EXISTS project_aecopd.aecopd_admissions_strict;
CREATE TABLE project_aecopd.aecopd_admissions_strict AS
WITH dx AS (
  SELECT
    subject_id,
    hadm_id,
    MIN(seq_num) AS best_seq_num,
    STRING_AGG(DISTINCT icd_code, ',' ORDER BY icd_code) AS aecopd_codes,
    STRING_AGG(DISTINCT icd_version::text, ',' ORDER BY icd_version::text) AS aecopd_icd_versions
  FROM mimiciv_hosp.diagnoses_icd
  WHERE (icd_version=10 AND icd_code='J441')
     OR (icd_version=9 AND icd_code='49121')
  GROUP BY subject_id, hadm_id
), joined AS (
  SELECT
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.dischtime,
    a.deathtime,
    a.admission_type,
    a.admission_location,
    a.discharge_location,
    a.insurance,
    a.language,
    a.marital_status,
    a.race,
    a.edregtime,
    a.edouttime,
    a.hospital_expire_flag,
    dx.best_seq_num,
    dx.aecopd_codes,
    dx.aecopd_icd_versions,
    EXTRACT(EPOCH FROM (a.dischtime-a.admittime))/86400.0 AS hospital_los_days,
    LAG(a.dischtime) OVER (PARTITION BY a.subject_id ORDER BY a.admittime,a.hadm_id) AS previous_strict_dischtime
  FROM dx
  JOIN mimiciv_hosp.admissions a USING (subject_id,hadm_id)
)
SELECT
  *,
  (admittime IS NULL OR dischtime IS NULL) AS qc_missing_admission_time,
  (dischtime <= admittime) AS qc_nonpositive_los,
  (previous_strict_dischtime IS NOT NULL AND admittime < previous_strict_dischtime) AS qc_overlap_previous_strict,
  (admission_location='TRANSFER FROM HOSPITAL') AS qc_transfer_from_hospital
FROM joined;

ALTER TABLE project_aecopd.aecopd_admissions_strict ADD CONSTRAINT aecopd_admissions_strict_pk PRIMARY KEY (hadm_id);
CREATE INDEX aecopd_admissions_strict_subject_time_idx ON project_aecopd.aecopd_admissions_strict(subject_id,admittime,hadm_id);

DROP TABLE IF EXISTS project_aecopd.aecopd_cohort_qc;
CREATE TABLE project_aecopd.aecopd_cohort_qc AS
SELECT 'rows'::text metric, COUNT(*)::text value FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'distinct_hadm_id', COUNT(DISTINCT hadm_id)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'distinct_subject_id', COUNT(DISTINCT subject_id)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'duplicate_hadm_id_groups', COUNT(*)::text FROM (SELECT hadm_id FROM project_aecopd.aecopd_admissions_strict GROUP BY hadm_id HAVING COUNT(*)>1) q
UNION ALL SELECT 'missing_admission_time', COUNT(*) FILTER (WHERE qc_missing_admission_time)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'nonpositive_los', COUNT(*) FILTER (WHERE qc_nonpositive_los)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'overlap_previous_strict', COUNT(*) FILTER (WHERE qc_overlap_previous_strict)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'transfer_from_hospital', COUNT(*) FILTER (WHERE qc_transfer_from_hospital)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'los_min_days', MIN(hospital_los_days)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'los_p25_days', percentile_cont(0.25) WITHIN GROUP (ORDER BY hospital_los_days)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'los_median_days', percentile_cont(0.50) WITHIN GROUP (ORDER BY hospital_los_days)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'los_p75_days', percentile_cont(0.75) WITHIN GROUP (ORDER BY hospital_los_days)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'los_max_days', MAX(hospital_los_days)::text FROM project_aecopd.aecopd_admissions_strict;

COMMIT;

\copy (SELECT * FROM project_aecopd.aecopd_admissions_strict ORDER BY subject_id,admittime,hadm_id) TO 'outputs/02_aecopd_admissions_strict.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.aecopd_cohort_qc) TO 'outputs/02_aecopd_cohort_qc.csv' CSV HEADER



````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/03_longitudinal_history.sql · lines 1–88

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/03_longitudinal_history.sql","part":1,"start_line":1,"end_line":88,"source_sha256":"77318659da0bd38e19b473e228bd226b9624eb7e78397b8ba93f5df1ba9a21ad","fragment_sha256":"77318659da0bd38e19b473e228bd226b9624eb7e78397b8ba93f5df1ba9a21ad","added_terminal_newline":false} -->
````sql
\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;

DROP TABLE IF EXISTS project_aecopd.aecopd_longitudinal_history;
CREATE TABLE project_aecopd.aecopd_longitudinal_history AS
WITH cohort_subjects AS (
  SELECT DISTINCT subject_id FROM project_aecopd.aecopd_admissions_strict
), all_admission_history AS (
  SELECT
    a.subject_id,
    a.hadm_id,
    ROW_NUMBER() OVER (PARTITION BY a.subject_id ORDER BY a.admittime,a.hadm_id)-1 AS prior_all_admission_count
  FROM mimiciv_hosp.admissions a
  JOIN cohort_subjects c USING (subject_id)
), strict_ordered AS (
  SELECT
    s.*,
    ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY admittime,hadm_id) AS aecopd_order,
    COUNT(*) OVER (PARTITION BY subject_id) AS aecopd_total,
    LAG(hadm_id) OVER (PARTITION BY subject_id ORDER BY admittime,hadm_id) AS prev_aecopd_hadm_id,
    LAG(dischtime) OVER (PARTITION BY subject_id ORDER BY admittime,hadm_id) AS prev_aecopd_dischtime,
    LEAD(hadm_id) OVER (PARTITION BY subject_id ORDER BY admittime,hadm_id) AS next_observed_aecopd_hadm_id,
    LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime,hadm_id) AS next_observed_aecopd_admittime
  FROM project_aecopd.aecopd_admissions_strict s
), gaps AS (
  SELECT
    s.*,
    s.aecopd_order-1 AS prior_aecopd_count,
    a.prior_all_admission_count,
    EXTRACT(EPOCH FROM (s.admittime-s.prev_aecopd_dischtime))/86400.0 AS days_since_previous_aecopd,
    EXTRACT(EPOCH FROM (s.next_observed_aecopd_admittime-s.dischtime))/86400.0 AS days_to_next_observed_aecopd
  FROM strict_ordered s
  JOIN all_admission_history a USING (subject_id,hadm_id)
)
SELECT
  *,
  (aecopd_order=1) AS is_first_observed_aecopd,
  (aecopd_order=aecopd_total) AS is_last_observed_aecopd,
  (days_to_next_observed_aecopd <= 0) AS qc_nonpositive_next_gap,
  (days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 7) AS qc_short_gap,
  (days_to_next_observed_aecopd < 0 OR qc_overlap_previous_strict) AS qc_overlap,
  qc_transfer_from_hospital AS qc_transfer
FROM gaps;

ALTER TABLE project_aecopd.aecopd_longitudinal_history ADD CONSTRAINT aecopd_longitudinal_history_pk PRIMARY KEY (hadm_id);
CREATE INDEX aecopd_longitudinal_history_subject_time_idx ON project_aecopd.aecopd_longitudinal_history(subject_id,admittime,hadm_id);

DROP TABLE IF EXISTS project_aecopd.recurrence_distribution;
CREATE TABLE project_aecopd.recurrence_distribution AS
WITH per_patient AS (
  SELECT subject_id,MAX(aecopd_total) AS aecopd_total
  FROM project_aecopd.aecopd_longitudinal_history
  GROUP BY subject_id
)
SELECT
  CASE WHEN aecopd_total>=5 THEN '5+' ELSE aecopd_total::text END AS aecopd_admissions_category,
  COUNT(*) AS patients
FROM per_patient
GROUP BY 1
ORDER BY CASE WHEN CASE WHEN aecopd_total>=5 THEN '5+' ELSE aecopd_total::text END='5+' THEN 5 ELSE (CASE WHEN aecopd_total>=5 THEN '5+' ELSE aecopd_total::text END)::int END;

DROP TABLE IF EXISTS project_aecopd.gap_summary;
CREATE TABLE project_aecopd.gap_summary AS
SELECT
  COUNT(*) FILTER (WHERE next_observed_aecopd_hadm_id IS NOT NULL) AS observed_adjacent_gaps,
  percentile_cont(0.25) WITHIN GROUP (ORDER BY days_to_next_observed_aecopd) FILTER (WHERE next_observed_aecopd_hadm_id IS NOT NULL) AS gap_p25_days,
  percentile_cont(0.50) WITHIN GROUP (ORDER BY days_to_next_observed_aecopd) FILTER (WHERE next_observed_aecopd_hadm_id IS NOT NULL) AS gap_median_days,
  percentile_cont(0.75) WITHIN GROUP (ORDER BY days_to_next_observed_aecopd) FILTER (WHERE next_observed_aecopd_hadm_id IS NOT NULL) AS gap_p75_days,
  COUNT(*) FILTER (WHERE qc_nonpositive_next_gap) AS nonpositive_gaps,
  COUNT(*) FILTER (WHERE qc_short_gap) AS short_positive_gaps_le_7d,
  COUNT(*) FILTER (WHERE qc_overlap) AS overlap_flag_rows,
  COUNT(*) FILTER (WHERE qc_transfer) AS transfer_from_hospital_rows,
  COUNT(*) FILTER (WHERE prior_aecopd_count>=2) AS landmarks_with_at_least_2_prior_aecopd,
  COUNT(*) FILTER (WHERE prior_aecopd_count>=3) AS landmarks_with_at_least_3_prior_aecopd,
  COUNT(*) FILTER (WHERE prior_aecopd_count>=4) AS landmarks_with_at_least_4_prior_aecopd,
  COUNT(*) FILTER (WHERE is_first_observed_aecopd) AS first_observed_rows,
  COUNT(*) FILTER (WHERE is_last_observed_aecopd) AS last_observed_rows
FROM project_aecopd.aecopd_longitudinal_history;

COMMIT;

\copy (SELECT * FROM project_aecopd.aecopd_longitudinal_history ORDER BY subject_id,admittime,hadm_id) TO 'outputs/03_longitudinal_history.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.recurrence_distribution) TO 'outputs/03_recurrence_distribution.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.gap_summary) TO 'outputs/03_gap_summary.csv' CSV HEADER



````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/04_note_coverage.sql · lines 1–147

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/04_note_coverage.sql","part":1,"start_line":1,"end_line":147,"source_sha256":"7fd7800a4912296596dfbf9424b00f6524a83c1c4a7eda38914726b36cda2ed4","fragment_sha256":"7fd7800a4912296596dfbf9424b00f6524a83c1c4a7eda38914726b36cda2ed4","added_terminal_newline":false} -->
````sql
\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;

DROP TABLE IF EXISTS project_aecopd.discharge_note_admission_audit;
CREATE TABLE project_aecopd.discharge_note_admission_audit AS
WITH cohort_notes AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    c.dischtime,
    n.note_id,
    n.note_seq,
    n.charttime,
    n.storetime,
    LENGTH(n.text) AS text_chars
  FROM project_aecopd.aecopd_admissions_strict c
  LEFT JOIN mimiciv_note.discharge n
    ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
), aggregated AS (
  SELECT
    subject_id,
    hadm_id,
    dischtime,
    COUNT(note_id) AS note_count,
    MIN(note_seq) FILTER (WHERE note_id IS NOT NULL) AS min_note_seq,
    MAX(note_seq) FILTER (WHERE note_id IS NOT NULL) AS max_note_seq,
    MIN(charttime) AS first_charttime,
    MAX(charttime) AS last_charttime,
    MIN(storetime) AS first_storetime,
    MAX(storetime) AS last_storetime,
    MIN(text_chars) FILTER (WHERE note_id IS NOT NULL) AS min_text_chars,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY text_chars) FILTER (WHERE note_id IS NOT NULL) AS median_text_chars,
    MAX(text_chars) FILTER (WHERE note_id IS NOT NULL) AS max_text_chars,
    COUNT(*) FILTER (WHERE note_id IS NOT NULL AND COALESCE(text_chars,0)=0) AS empty_text_count,
    COUNT(*) FILTER (WHERE note_id IS NOT NULL AND charttime<=dischtime AND storetime<=dischtime) AS notes_available_by_discharge,
    COUNT(*) FILTER (WHERE note_id IS NOT NULL AND charttime<=dischtime AND storetime>dischtime) AS notes_charted_by_but_stored_after_discharge,
    COUNT(*) FILTER (WHERE note_id IS NOT NULL AND charttime>dischtime) AS notes_charted_after_discharge,
    COUNT(*) FILTER (WHERE note_id IS NOT NULL AND (charttime IS NULL OR storetime IS NULL)) AS notes_missing_time
  FROM cohort_notes
  GROUP BY subject_id,hadm_id,dischtime
)
SELECT
  *,
  (note_count>0) AS has_discharge_note,
  (notes_available_by_discharge>0) AS has_note_available_by_discharge,
  CASE
    WHEN note_count=0 THEN 'NO_NOTE'
    WHEN notes_available_by_discharge>0 THEN 'AVAILABLE_BY_DISCHARGE'
    WHEN notes_missing_time=note_count THEN 'MISSING_TIME'
    WHEN notes_charted_after_discharge>0 THEN 'CHARTED_AFTER_DISCHARGE'
    WHEN notes_charted_by_but_stored_after_discharge>0 THEN 'CHARTED_BY_BUT_STORED_AFTER_DISCHARGE'
    ELSE 'OTHER_TIME_PATTERN'
  END AS note_time_flag
FROM aggregated;

ALTER TABLE project_aecopd.discharge_note_admission_audit ADD CONSTRAINT discharge_note_admission_audit_pk PRIMARY KEY (hadm_id);

DROP TABLE IF EXISTS project_aecopd.discharge_note_coverage_summary;
CREATE TABLE project_aecopd.discharge_note_coverage_summary AS
SELECT
  COUNT(*) AS aecopd_admissions,
  COUNT(*) FILTER (WHERE has_discharge_note) AS with_any_discharge_note,
  ROUND(100.0*COUNT(*) FILTER (WHERE has_discharge_note)/NULLIF(COUNT(*),0),2) AS any_note_coverage_pct,
  COUNT(*) FILTER (WHERE has_note_available_by_discharge) AS with_note_available_by_discharge,
  ROUND(100.0*COUNT(*) FILTER (WHERE has_note_available_by_discharge)/NULLIF(COUNT(*),0),2) AS available_by_discharge_coverage_pct,
  COUNT(*) FILTER (WHERE note_count>1) AS admissions_with_multiple_notes,
  SUM(note_count) AS discharge_notes,
  SUM(empty_text_count) AS empty_text_notes,
  ROUND(100.0*SUM(empty_text_count)/NULLIF(SUM(note_count),0),4) AS empty_text_pct
FROM project_aecopd.discharge_note_admission_audit;

DROP TABLE IF EXISTS project_aecopd.discharge_note_time_distribution;
CREATE TABLE project_aecopd.discharge_note_time_distribution AS
SELECT note_time_flag,COUNT(*) AS admissions,
       ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM project_aecopd.discharge_note_admission_audit),2) AS pct
FROM project_aecopd.discharge_note_admission_audit
GROUP BY note_time_flag
ORDER BY admissions DESC;

DROP TABLE IF EXISTS project_aecopd.discharge_note_seq_distribution;
CREATE TABLE project_aecopd.discharge_note_seq_distribution AS
SELECT n.note_seq,COUNT(*) AS notes,COUNT(DISTINCT n.hadm_id) AS admissions
FROM mimiciv_note.discharge n
JOIN project_aecopd.aecopd_admissions_strict c ON c.subject_id=n.subject_id AND c.hadm_id=n.hadm_id
GROUP BY n.note_seq
ORDER BY n.note_seq;

DROP TABLE IF EXISTS project_aecopd.discharge_note_timing_summary;
CREATE TABLE project_aecopd.discharge_note_timing_summary AS
WITH cohort_notes AS (
  SELECT c.dischtime,n.charttime,n.storetime
  FROM project_aecopd.aecopd_admissions_strict c
  JOIN mimiciv_note.discharge n ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
)
SELECT
  COUNT(*) AS notes,
  COUNT(*) FILTER (WHERE charttime<=dischtime) AS charted_on_or_before_discharge,
  COUNT(*) FILTER (WHERE storetime<=dischtime) AS stored_on_or_before_discharge,
  COUNT(*) FILTER (WHERE charttime<=dischtime AND storetime<=dischtime) AS both_on_or_before_discharge,
  COUNT(*) FILTER (WHERE charttime IS NULL) AS missing_charttime,
  COUNT(*) FILTER (WHERE storetime IS NULL) AS missing_storetime,
  percentile_cont(0.25) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (charttime-dischtime))/86400.0) AS chart_offset_p25_days,
  percentile_cont(0.50) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (charttime-dischtime))/86400.0) AS chart_offset_median_days,
  percentile_cont(0.75) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (charttime-dischtime))/86400.0) AS chart_offset_p75_days,
  percentile_cont(0.25) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (storetime-dischtime))/86400.0) AS store_offset_p25_days,
  percentile_cont(0.50) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (storetime-dischtime))/86400.0) AS store_offset_median_days,
  percentile_cont(0.75) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (storetime-dischtime))/86400.0) AS store_offset_p75_days
FROM cohort_notes;

DROP TABLE IF EXISTS project_aecopd.discharge_note_missingness_summary;
CREATE TABLE project_aecopd.discharge_note_missingness_summary AS
WITH note_subjects AS (
  SELECT DISTINCT n.subject_id
  FROM mimiciv_note.discharge n
  JOIN (SELECT DISTINCT subject_id FROM project_aecopd.aecopd_admissions_strict) c USING (subject_id)
)
SELECT
  COUNT(*) FILTER (WHERE a.has_discharge_note) AS matching_hadm_has_note,
  COUNT(*) FILTER (WHERE NOT a.has_discharge_note AND s.subject_id IS NOT NULL) AS no_matching_hadm_but_subject_has_other_note,
  COUNT(*) FILTER (WHERE NOT a.has_discharge_note AND s.subject_id IS NULL) AS subject_has_no_discharge_note
FROM project_aecopd.discharge_note_admission_audit a
LEFT JOIN note_subjects s USING (subject_id);

DROP TABLE IF EXISTS project_aecopd.discharge_note_text_summary;
CREATE TABLE project_aecopd.discharge_note_text_summary AS
SELECT
  COUNT(*) FILTER (WHERE has_discharge_note) AS admissions_with_note,
  MIN(min_text_chars) AS min_text_chars,
  percentile_cont(0.25) WITHIN GROUP (ORDER BY median_text_chars) FILTER (WHERE has_discharge_note) AS text_chars_p25,
  percentile_cont(0.50) WITHIN GROUP (ORDER BY median_text_chars) FILTER (WHERE has_discharge_note) AS text_chars_median,
  percentile_cont(0.75) WITHIN GROUP (ORDER BY median_text_chars) FILTER (WHERE has_discharge_note) AS text_chars_p75,
  MAX(max_text_chars) AS max_text_chars
FROM project_aecopd.discharge_note_admission_audit;

COMMIT;

\copy (SELECT * FROM project_aecopd.discharge_note_coverage_summary) TO 'outputs/04_note_coverage.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.discharge_note_admission_audit ORDER BY subject_id,hadm_id) TO 'outputs/04_note_admission_audit.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.discharge_note_time_distribution) TO 'outputs/04_note_time_distribution.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.discharge_note_seq_distribution) TO 'outputs/04_note_seq_distribution.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.discharge_note_timing_summary) TO 'outputs/04_note_timing_summary.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.discharge_note_missingness_summary) TO 'outputs/04_note_missingness_summary.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.discharge_note_text_summary) TO 'outputs/04_note_text_summary.csv' CSV HEADER



````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/05_structured_field_audit.sql · lines 1–273

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/05_structured_field_audit.sql","part":1,"start_line":1,"end_line":273,"source_sha256":"5b763aca4f224631f7c251fd979e5e99da5ef9c34db0e56a0743e6255a94a792","fragment_sha256":"5b763aca4f224631f7c251fd979e5e99da5ef9c34db0e56a0743e6255a94a792","added_terminal_newline":false} -->
````sql
\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;

DROP TABLE IF EXISTS project_aecopd.cohort_hadm_ids;
CREATE TABLE project_aecopd.cohort_hadm_ids AS
SELECT subject_id,hadm_id,admittime,dischtime
FROM project_aecopd.aecopd_admissions_strict;
ALTER TABLE project_aecopd.cohort_hadm_ids ADD CONSTRAINT cohort_hadm_ids_pk PRIMARY KEY(hadm_id);
CREATE INDEX cohort_hadm_ids_subject_idx ON project_aecopd.cohort_hadm_ids(subject_id);
ANALYZE project_aecopd.cohort_hadm_ids;
SELECT COUNT(*) AS cohort_n FROM project_aecopd.cohort_hadm_ids \gset

DROP TABLE IF EXISTS project_aecopd.core_field_coverage;
CREATE TABLE project_aecopd.core_field_coverage AS
WITH e AS (
  SELECT s.*,p.gender,p.anchor_age,p.anchor_year,p.anchor_year_group,p.dod
  FROM project_aecopd.aecopd_admissions_strict s
  JOIN mimiciv_hosp.patients p USING(subject_id)
), audit AS (
  SELECT 'mimiciv_hosp.patients' source_table,'gender' field_name,'character' data_type,COUNT(*) FILTER(WHERE gender IS NOT NULL) non_null_admissions,COUNT(DISTINCT gender)::bigint distinct_values,'patient-level; repeated across admissions' audit_note FROM e
  UNION ALL SELECT 'mimiciv_hosp.patients','anchor_age','smallint',COUNT(*) FILTER(WHERE anchor_age IS NOT NULL),COUNT(DISTINCT anchor_age),'deidentified anchor age; derive admission age only in later approved feature work' FROM e
  UNION ALL SELECT 'mimiciv_hosp.patients','anchor_year','smallint',COUNT(*) FILTER(WHERE anchor_year IS NOT NULL),COUNT(DISTINCT anchor_year),'deidentified anchor year' FROM e
  UNION ALL SELECT 'mimiciv_hosp.patients','anchor_year_group','varchar',COUNT(*) FILTER(WHERE anchor_year_group IS NOT NULL),COUNT(DISTINCT anchor_year_group),'deidentified year group' FROM e
  UNION ALL SELECT 'mimiciv_hosp.patients','dod','date',COUNT(*) FILTER(WHERE dod IS NOT NULL),COUNT(DISTINCT dod),'death support field; not a finalized competing-risk definition' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','admittime','timestamp',COUNT(*) FILTER(WHERE admittime IS NOT NULL),COUNT(DISTINCT admittime),'prediction history boundary' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','dischtime','timestamp',COUNT(*) FILTER(WHERE dischtime IS NOT NULL),COUNT(DISTINCT dischtime),'candidate prediction time' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','deathtime','timestamp',COUNT(*) FILTER(WHERE deathtime IS NOT NULL),COUNT(DISTINCT deathtime),'in-hospital death support field' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','admission_type','varchar',COUNT(*) FILTER(WHERE admission_type IS NOT NULL),COUNT(DISTINCT admission_type),'administrative field; audit before modeling' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','admission_location','varchar',COUNT(*) FILTER(WHERE admission_location IS NOT NULL),COUNT(DISTINCT admission_location),'administrative field; transfer QC source' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','discharge_location','varchar',COUNT(*) FILTER(WHERE discharge_location IS NOT NULL),COUNT(DISTINCT discharge_location),'may be known at discharge; administrative/clinical interpretation required' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','insurance','varchar',COUNT(*) FILTER(WHERE insurance IS NOT NULL),COUNT(DISTINCT insurance),'administrative field; fairness and temporal stability review required' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','language','varchar',COUNT(*) FILTER(WHERE language IS NOT NULL),COUNT(DISTINCT language),'administrative field; missingness audit only' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','marital_status','varchar',COUNT(*) FILTER(WHERE marital_status IS NOT NULL),COUNT(DISTINCT marital_status),'administrative field; missingness audit only' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','race','varchar',COUNT(*) FILTER(WHERE race IS NOT NULL),COUNT(DISTINCT race),'fairness-sensitive administrative field; not automatically included' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','edregtime','timestamp',COUNT(*) FILTER(WHERE edregtime IS NOT NULL),COUNT(DISTINCT edregtime),'ED timing support field' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','edouttime','timestamp',COUNT(*) FILTER(WHERE edouttime IS NOT NULL),COUNT(DISTINCT edouttime),'ED timing support field' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','hospital_expire_flag','smallint',COUNT(*) FILTER(WHERE hospital_expire_flag IS NOT NULL),COUNT(DISTINCT hospital_expire_flag),'death support field; not a finalized outcome definition' FROM e
)
SELECT *,:cohort_n::bigint AS cohort_admissions,ROUND(100.0*non_null_admissions/:cohort_n,2) AS coverage_pct
FROM audit;

DROP TABLE IF EXISTS project_aecopd.core_value_distribution;
CREATE TABLE project_aecopd.core_value_distribution AS
WITH e AS (
  SELECT s.*,p.gender,p.anchor_age,p.anchor_year_group,p.dod
  FROM project_aecopd.aecopd_admissions_strict s JOIN mimiciv_hosp.patients p USING(subject_id)
), u AS (
  SELECT 'gender' field_name,COALESCE(gender,'<NULL>')::text value FROM e
  UNION ALL SELECT 'anchor_year_group',COALESCE(anchor_year_group,'<NULL>') FROM e
  UNION ALL SELECT 'admission_type',COALESCE(admission_type,'<NULL>') FROM e
  UNION ALL SELECT 'admission_location',COALESCE(admission_location,'<NULL>') FROM e
  UNION ALL SELECT 'discharge_location',COALESCE(discharge_location,'<NULL>') FROM e
  UNION ALL SELECT 'insurance',COALESCE(insurance,'<NULL>') FROM e
  UNION ALL SELECT 'language',COALESCE(language,'<NULL>') FROM e
  UNION ALL SELECT 'marital_status',COALESCE(marital_status,'<NULL>') FROM e
  UNION ALL SELECT 'race',COALESCE(race,'<NULL>') FROM e
  UNION ALL SELECT 'hospital_expire_flag',COALESCE(hospital_expire_flag::text,'<NULL>') FROM e
)
SELECT field_name,value,COUNT(*) AS admissions,ROUND(100.0*COUNT(*)/:cohort_n,2) AS pct
FROM u GROUP BY field_name,value;

DROP TABLE IF EXISTS project_aecopd.lab_item_coverage_top100;
CREATE TABLE project_aecopd.lab_item_coverage_top100 AS
WITH cohort_labs AS MATERIALIZED (
  SELECT l.subject_id,l.hadm_id,l.itemid,l.charttime,l.valueuom,c.admittime,c.dischtime
  FROM mimiciv_hosp.labevents l
  JOIN project_aecopd.cohort_hadm_ids c ON c.hadm_id=l.hadm_id AND c.subject_id=l.subject_id
), per_admission AS (
  SELECT itemid,hadm_id,MIN(subject_id) subject_id,COUNT(*) measurements_per_admission
  FROM cohort_labs GROUP BY itemid,hadm_id
), aggregate_items AS (
  SELECT
    l.itemid,
    COUNT(*) AS measurements,
    COUNT(DISTINCT l.hadm_id) AS admissions_with_item,
    COUNT(DISTINCT l.subject_id) AS patients_with_item,
    COUNT(DISTINCT l.valueuom) AS unit_count,
    STRING_AGG(DISTINCT COALESCE(l.valueuom,'<NULL>'),' | ' ORDER BY COALESCE(l.valueuom,'<NULL>')) AS observed_units,
    COUNT(*) FILTER(WHERE l.charttime<l.admittime) AS measurements_before_admission,
    COUNT(*) FILTER(WHERE l.charttime>l.dischtime) AS measurements_after_discharge
  FROM cohort_labs l GROUP BY l.itemid
), medians AS (
  SELECT itemid,percentile_cont(0.5) WITHIN GROUP(ORDER BY measurements_per_admission) AS median_measurements_per_admission
  FROM per_admission GROUP BY itemid
)
SELECT
  a.itemid,d.label,d.fluid,d.category,a.measurements,a.admissions_with_item,a.patients_with_item,
  ROUND(100.0*a.admissions_with_item/:cohort_n,2) AS admission_coverage_pct,
  m.median_measurements_per_admission,a.unit_count,a.observed_units,
  a.measurements_before_admission,a.measurements_after_discharge,
  (lower(COALESCE(d.label,'')) ~ '(blood gas|ph$|pco2|po2|oxygen|bicarbonate|base excess|hemoglobin|hematocrit|white blood|wbc|platelet|neutrophil|lymphocyte|creatinine|urea nitrogen|sodium|potassium|chloride|calcium|magnesium|phosphate|lactate|c-reactive|procalcitonin|albumin)') AS copd_clinical_review_candidate
FROM aggregate_items a
JOIN medians m USING(itemid)
LEFT JOIN mimiciv_hosp.d_labitems d USING(itemid)
ORDER BY a.admissions_with_item DESC,a.measurements DESC,a.itemid
LIMIT 100;

DROP TABLE IF EXISTS project_aecopd.medication_coverage_top100;
CREATE TABLE project_aecopd.medication_coverage_top100 AS
SELECT
  p.drug,
  COUNT(*) AS prescription_rows,
  COUNT(DISTINCT p.hadm_id) AS admissions_with_drug,
  COUNT(DISTINCT p.subject_id) AS patients_with_drug,
  ROUND(100.0*COUNT(DISTINCT p.hadm_id)/:cohort_n,2) AS admission_coverage_pct,
  COUNT(DISTINCT p.route) AS route_count,
  STRING_AGG(DISTINCT COALESCE(p.route,'<NULL>'),' | ' ORDER BY COALESCE(p.route,'<NULL>')) AS observed_routes,
  COUNT(*) FILTER(WHERE p.starttime>c.dischtime) AS starts_after_discharge,
  COUNT(*) FILTER(WHERE p.stoptime<c.admittime) AS stops_before_admission
FROM mimiciv_hosp.prescriptions p
JOIN project_aecopd.cohort_hadm_ids c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
GROUP BY p.drug
ORDER BY admissions_with_drug DESC,prescription_rows DESC,p.drug
LIMIT 100;

DROP TABLE IF EXISTS project_aecopd.pharmacy_coverage_top100;
CREATE TABLE project_aecopd.pharmacy_coverage_top100 AS
SELECT
  p.medication,
  COUNT(*) AS pharmacy_rows,
  COUNT(DISTINCT p.hadm_id) AS admissions_with_medication,
  COUNT(DISTINCT p.subject_id) AS patients_with_medication,
  ROUND(100.0*COUNT(DISTINCT p.hadm_id)/:cohort_n,2) AS admission_coverage_pct,
  COUNT(DISTINCT p.route) AS route_count,
  STRING_AGG(DISTINCT COALESCE(p.route,'<NULL>'),' | ' ORDER BY COALESCE(p.route,'<NULL>')) AS observed_routes,
  COUNT(*) FILTER(WHERE p.starttime>c.dischtime) AS starts_after_discharge,
  COUNT(*) FILTER(WHERE p.verifiedtime>c.dischtime) AS verified_after_discharge
FROM mimiciv_hosp.pharmacy p
JOIN project_aecopd.cohort_hadm_ids c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
GROUP BY p.medication
ORDER BY admissions_with_medication DESC,pharmacy_rows DESC,p.medication
LIMIT 100;

DROP TABLE IF EXISTS project_aecopd.procedure_coverage_top100;
CREATE TABLE project_aecopd.procedure_coverage_top100 AS
SELECT
  p.icd_version,p.icd_code,d.long_title,
  COUNT(*) AS procedure_rows,
  COUNT(DISTINCT p.hadm_id) AS admissions_with_procedure,
  COUNT(DISTINCT p.subject_id) AS patients_with_procedure,
  ROUND(100.0*COUNT(DISTINCT p.hadm_id)/:cohort_n,2) AS admission_coverage_pct,
  COUNT(*) FILTER(WHERE p.chartdate<c.admittime::date) AS charted_before_admission,
  COUNT(*) FILTER(WHERE p.chartdate>c.dischtime::date) AS charted_after_discharge,
  (lower(COALESCE(d.long_title,'')) ~ '(ventilat|intubat|respirat|oxygen|trache|airway|thorac)') AS respiratory_support_review_candidate
FROM mimiciv_hosp.procedures_icd p
JOIN project_aecopd.cohort_hadm_ids c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
LEFT JOIN mimiciv_hosp.d_icd_procedures d USING(icd_version,icd_code)
GROUP BY p.icd_version,p.icd_code,d.long_title
ORDER BY admissions_with_procedure DESC,procedure_rows DESC,p.icd_version,p.icd_code
LIMIT 100;

DROP TABLE IF EXISTS project_aecopd.icu_admission_audit;
CREATE TABLE project_aecopd.icu_admission_audit AS
SELECT
  c.subject_id,c.hadm_id,
  COUNT(i.stay_id) AS icu_stay_count,
  MIN(i.intime) AS first_icu_intime,
  MAX(i.outtime) AS last_icu_outtime,
  SUM(i.los) AS total_icu_los_days,
  (COUNT(i.stay_id)>0) AS has_icu_stay,
  COUNT(i.stay_id) FILTER(WHERE i.intime<c.admittime OR i.outtime>c.dischtime) AS icu_stays_outside_hospital_time
FROM project_aecopd.cohort_hadm_ids c
LEFT JOIN mimiciv_icu.icustays i ON i.hadm_id=c.hadm_id AND i.subject_id=c.subject_id
GROUP BY c.subject_id,c.hadm_id;
ALTER TABLE project_aecopd.icu_admission_audit ADD CONSTRAINT icu_admission_audit_pk PRIMARY KEY(hadm_id);

DROP TABLE IF EXISTS project_aecopd.icu_coverage_summary;
CREATE TABLE project_aecopd.icu_coverage_summary AS
SELECT
  COUNT(*) AS cohort_admissions,
  COUNT(*) FILTER(WHERE has_icu_stay) AS admissions_with_icu,
  ROUND(100.0*COUNT(*) FILTER(WHERE has_icu_stay)/COUNT(*),2) AS icu_coverage_pct,
  SUM(icu_stay_count) AS icu_stays,
  COUNT(*) FILTER(WHERE icu_stay_count>1) AS admissions_with_multiple_icu_stays,
  SUM(icu_stays_outside_hospital_time) AS icu_stays_outside_hospital_time
FROM project_aecopd.icu_admission_audit;

DROP TABLE IF EXISTS project_aecopd.icu_selection_audit;
CREATE TABLE project_aecopd.icu_selection_audit AS
SELECT
  i.has_icu_stay,
  COUNT(*) AS admissions,
  percentile_cont(0.5) WITHIN GROUP(ORDER BY s.hospital_los_days) AS median_hospital_los_days,
  percentile_cont(0.5) WITHIN GROUP(ORDER BY p.anchor_age) AS median_anchor_age,
  COUNT(*) FILTER(WHERE s.hospital_expire_flag=1) AS in_hospital_deaths,
  ROUND(100.0*COUNT(*) FILTER(WHERE s.hospital_expire_flag=1)/COUNT(*),2) AS in_hospital_death_pct,
  COUNT(*) FILTER(WHERE n.has_discharge_note) AS with_discharge_note,
  ROUND(100.0*COUNT(*) FILTER(WHERE n.has_discharge_note)/COUNT(*),2) AS discharge_note_coverage_pct
FROM project_aecopd.icu_admission_audit i
JOIN project_aecopd.aecopd_admissions_strict s USING(subject_id,hadm_id)
JOIN mimiciv_hosp.patients p USING(subject_id)
JOIN project_aecopd.discharge_note_admission_audit n USING(subject_id,hadm_id)
GROUP BY i.has_icu_stay;

DROP TABLE IF EXISTS project_aecopd.structured_source_coverage;
CREATE TABLE project_aecopd.structured_source_coverage AS
WITH source_hadm AS (
  SELECT 'mimiciv_hosp.labevents'::text AS data_source,l.hadm_id
  FROM mimiciv_hosp.labevents l JOIN project_aecopd.cohort_hadm_ids c ON c.hadm_id=l.hadm_id AND c.subject_id=l.subject_id
  GROUP BY l.hadm_id
  UNION ALL
  SELECT 'mimiciv_hosp.prescriptions',p.hadm_id
  FROM mimiciv_hosp.prescriptions p JOIN project_aecopd.cohort_hadm_ids c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  GROUP BY p.hadm_id
  UNION ALL
  SELECT 'mimiciv_hosp.pharmacy',p.hadm_id
  FROM mimiciv_hosp.pharmacy p JOIN project_aecopd.cohort_hadm_ids c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  GROUP BY p.hadm_id
  UNION ALL
  SELECT 'mimiciv_hosp.procedures_icd',p.hadm_id
  FROM mimiciv_hosp.procedures_icd p JOIN project_aecopd.cohort_hadm_ids c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  GROUP BY p.hadm_id
)
SELECT data_source,COUNT(*) AS admissions_with_data,ROUND(100.0*COUNT(*)/:cohort_n,2) AS admission_coverage_pct
FROM source_hadm GROUP BY data_source;

DROP TABLE IF EXISTS project_aecopd.derived_inventory;
CREATE TABLE project_aecopd.derived_inventory AS
SELECT n.nspname AS table_schema,c.relname AS object_name,
       CASE c.relkind WHEN 'r' THEN 'table' WHEN 'p' THEN 'partitioned table' WHEN 'v' THEN 'view' WHEN 'm' THEN 'materialized view' WHEN 'f' THEN 'foreign table' ELSE c.relkind::text END AS object_type
FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
WHERE n.nspname='mimiciv_derived' AND c.relkind IN('r','p','v','m','f');

DROP TABLE IF EXISTS project_aecopd.hosp_field_inventory;
CREATE TABLE project_aecopd.hosp_field_inventory AS
SELECT source_table AS data_source,field_name AS field_or_item,NULL::text AS item_id,data_type,
       cohort_admissions,non_null_admissions AS admissions_with_data,NULL::bigint AS patients_with_data,
       coverage_pct,NULL::bigint AS total_records,NULL::text AS observed_units_or_routes,
       (source_table='mimiciv_hosp.admissions') AS has_hadm_id,
       CASE WHEN data_type IN('timestamp','date') THEN field_name ELSE NULL END AS time_fields,
       audit_note,
       'researcher_review_required'::text AS later_model_consideration
FROM project_aecopd.core_field_coverage
UNION ALL
SELECT 'mimiciv_hosp.labevents',COALESCE(label,'<missing dictionary label>'),itemid::text,'lab item',
       :cohort_n,admissions_with_item,patients_with_item,admission_coverage_pct,measurements,observed_units,
       TRUE,'charttime/storetime','aggregated within cohort; no imputation or unit harmonization; before/after admission flags retained',
       'candidate_after_window_and_unit_rules'
FROM project_aecopd.lab_item_coverage_top100
UNION ALL
SELECT 'mimiciv_hosp.prescriptions',COALESCE(drug,'<NULL>'),NULL,'raw drug name',
       :cohort_n,admissions_with_drug,patients_with_drug,admission_coverage_pct,prescription_rows,observed_routes,TRUE,'starttime/stoptime',
       'raw names/routes not standardized; timing anomalies counted','candidate_after_drug_normalization_and_time_rules'
FROM project_aecopd.medication_coverage_top100
UNION ALL
SELECT 'mimiciv_hosp.pharmacy',COALESCE(medication,'<NULL>'),NULL,'raw pharmacy medication',
       :cohort_n,admissions_with_medication,patients_with_medication,admission_coverage_pct,pharmacy_rows,observed_routes,TRUE,'starttime/stoptime/entertime/verifiedtime',
       'raw medication/routes/status not standardized; timing anomalies counted','candidate_after_medication_normalization_and_time_rules'
FROM project_aecopd.pharmacy_coverage_top100
UNION ALL
SELECT 'mimiciv_hosp.procedures_icd',COALESCE(long_title,'<missing dictionary title>'),icd_version::text||':'||icd_code,'ICD procedure',
       :cohort_n,admissions_with_procedure,patients_with_procedure,admission_coverage_pct,procedure_rows,NULL,TRUE,'chartdate',
       'candidate only; procedure and respiratory-support interpretation requires clinical review','candidate_after_clinical_code_review'
FROM project_aecopd.procedure_coverage_top100;

COMMIT;

\copy (SELECT * FROM project_aecopd.core_field_coverage ORDER BY source_table,field_name) TO 'outputs/05_core_field_coverage.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.core_value_distribution ORDER BY field_name,admissions DESC,value) TO 'outputs/05_core_value_distribution.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.lab_item_coverage_top100) TO 'outputs/05_lab_item_coverage_top100.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.medication_coverage_top100) TO 'outputs/05_medication_coverage_top100.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.pharmacy_coverage_top100) TO 'outputs/05_pharmacy_coverage_top100.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.procedure_coverage_top100) TO 'outputs/05_procedure_coverage_top100.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.icu_admission_audit ORDER BY subject_id,hadm_id) TO 'outputs/05_icu_admission_audit.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.icu_coverage_summary) TO 'outputs/05_icu_coverage.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.icu_selection_audit ORDER BY has_icu_stay) TO 'outputs/05_icu_selection_audit.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.structured_source_coverage ORDER BY data_source) TO 'outputs/05_structured_source_coverage.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.derived_inventory) TO 'outputs/05_derived_inventory.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.hosp_field_inventory ORDER BY data_source,coverage_pct DESC,field_or_item) TO 'outputs/05_hosp_field_inventory.csv' CSV HEADER



````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/06_master_cohort.sql · lines 1–123

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/06_master_cohort.sql","part":1,"start_line":1,"end_line":123,"source_sha256":"7e1cb1513b5e17693341380f8510fe050f783b6cf04ecc46f570753575443ce8","fragment_sha256":"7e1cb1513b5e17693341380f8510fe050f783b6cf04ecc46f570753575443ce8","added_terminal_newline":false} -->
````sql
\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;

DROP TABLE IF EXISTS project_aecopd.master_cohort;
CREATE TABLE project_aecopd.master_cohort AS
WITH allcause_observation AS (
  SELECT a.subject_id,MAX(a.dischtime) AS last_observed_allcause_dischtime
  FROM mimiciv_hosp.admissions a
  JOIN (SELECT DISTINCT subject_id FROM project_aecopd.aecopd_admissions_strict) c USING(subject_id)
  GROUP BY a.subject_id
), base AS (
  SELECT
    h.*,
    p.gender,p.anchor_age,p.anchor_year,p.anchor_year_group,p.dod,
    n.has_discharge_note,n.note_count,n.has_note_available_by_discharge,n.note_time_flag,
    n.first_charttime AS discharge_note_first_charttime,
    n.first_storetime AS discharge_note_first_storetime,
    n.median_text_chars AS discharge_note_text_chars,
    i.has_icu_stay,i.icu_stay_count,i.first_icu_intime,i.last_icu_outtime,i.total_icu_los_days,i.icu_stays_outside_hospital_time,
    o.last_observed_allcause_dischtime,
    EXTRACT(EPOCH FROM (o.last_observed_allcause_dischtime-h.dischtime))/86400.0 AS observed_allcause_span_after_discharge_days
  FROM project_aecopd.aecopd_longitudinal_history h
  JOIN mimiciv_hosp.patients p USING(subject_id)
  JOIN project_aecopd.discharge_note_admission_audit n USING(subject_id,hadm_id)
  JOIN project_aecopd.icu_admission_audit i USING(subject_id,hadm_id)
  JOIN allcause_observation o USING(subject_id)
)
SELECT
  *,
  (days_to_next_observed_aecopd>0 AND days_to_next_observed_aecopd<=90) AS next_observed_aecopd_within_90d,
  (days_to_next_observed_aecopd>0 AND days_to_next_observed_aecopd<=180) AS next_observed_aecopd_within_180d,
  (days_to_next_observed_aecopd>0 AND days_to_next_observed_aecopd<=365) AS next_observed_aecopd_within_365d,
  CASE
    WHEN days_to_next_observed_aecopd<=0 THEN 'NONPOSITIVE_NEXT_EVENT_GAP_REVIEW'
    WHEN days_to_next_observed_aecopd<=90 THEN 'OBSERVED_AECOPD_WITHIN_WINDOW'
    WHEN observed_allcause_span_after_discharge_days>=90 THEN 'NO_AECOPD_OBSERVED_WITHIN_WINDOW_ALLCAUSE_SPAN_GE_WINDOW'
    ELSE 'NO_AECOPD_OBSERVED_INSUFFICIENT_ALLCAUSE_SPAN'
  END AS candidate_followup_status_90d,
  CASE
    WHEN days_to_next_observed_aecopd<=0 THEN 'NONPOSITIVE_NEXT_EVENT_GAP_REVIEW'
    WHEN days_to_next_observed_aecopd<=180 THEN 'OBSERVED_AECOPD_WITHIN_WINDOW'
    WHEN observed_allcause_span_after_discharge_days>=180 THEN 'NO_AECOPD_OBSERVED_WITHIN_WINDOW_ALLCAUSE_SPAN_GE_WINDOW'
    ELSE 'NO_AECOPD_OBSERVED_INSUFFICIENT_ALLCAUSE_SPAN'
  END AS candidate_followup_status_180d,
  CASE
    WHEN days_to_next_observed_aecopd<=0 THEN 'NONPOSITIVE_NEXT_EVENT_GAP_REVIEW'
    WHEN days_to_next_observed_aecopd<=365 THEN 'OBSERVED_AECOPD_WITHIN_WINDOW'
    WHEN observed_allcause_span_after_discharge_days>=365 THEN 'NO_AECOPD_OBSERVED_WITHIN_WINDOW_ALLCAUSE_SPAN_GE_WINDOW'
    ELSE 'NO_AECOPD_OBSERVED_INSUFFICIENT_ALLCAUSE_SPAN'
  END AS candidate_followup_status_365d
FROM base;

ALTER TABLE project_aecopd.master_cohort ADD CONSTRAINT master_cohort_pk PRIMARY KEY(hadm_id);
CREATE INDEX master_cohort_subject_time_idx ON project_aecopd.master_cohort(subject_id,admittime,hadm_id);

DROP TABLE IF EXISTS project_aecopd.master_cohort_qc;
CREATE TABLE project_aecopd.master_cohort_qc AS
SELECT 'rows'::text metric,COUNT(*)::text value FROM project_aecopd.master_cohort
UNION ALL SELECT 'distinct_hadm_id',COUNT(DISTINCT hadm_id)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'distinct_subject_id',COUNT(DISTINCT subject_id)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'duplicate_hadm_id_groups',COUNT(*)::text FROM (SELECT hadm_id FROM project_aecopd.master_cohort GROUP BY hadm_id HAVING COUNT(*)>1) q
UNION ALL SELECT 'missing_subject_id',COUNT(*) FILTER(WHERE subject_id IS NULL)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'missing_hadm_id',COUNT(*) FILTER(WHERE hadm_id IS NULL)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'missing_admittime',COUNT(*) FILTER(WHERE admittime IS NULL)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'missing_dischtime',COUNT(*) FILTER(WHERE dischtime IS NULL)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'nonpositive_los',COUNT(*) FILTER(WHERE qc_nonpositive_los)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'with_any_discharge_note',COUNT(*) FILTER(WHERE has_discharge_note)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'with_note_available_by_discharge',COUNT(*) FILTER(WHERE has_note_available_by_discharge)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'with_icu_stay',COUNT(*) FILTER(WHERE has_icu_stay)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'observed_next_aecopd_within_90d',COUNT(*) FILTER(WHERE next_observed_aecopd_within_90d)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'observed_next_aecopd_within_180d',COUNT(*) FILTER(WHERE next_observed_aecopd_within_180d)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'observed_next_aecopd_within_365d',COUNT(*) FILTER(WHERE next_observed_aecopd_within_365d)::text FROM project_aecopd.master_cohort;

DROP TABLE IF EXISTS project_aecopd.master_cohort_data_dictionary;
CREATE TABLE project_aecopd.master_cohort_data_dictionary AS
SELECT
  c.ordinal_position,
  c.column_name,
  c.data_type,
  CASE
    WHEN c.column_name IN('subject_id','hadm_id') THEN 'identifier'
    WHEN c.column_name LIKE 'qc_%' THEN 'quality_control'
    WHEN c.column_name LIKE '%note%' THEN 'note_availability'
    WHEN c.column_name LIKE '%icu%' THEN 'icu_availability'
    WHEN c.column_name LIKE '%aecopd%' OR c.column_name IN('aecopd_order','aecopd_total') THEN 'aecopd_history_or_observed_outcome'
    WHEN c.column_name LIKE 'candidate_followup_status_%' OR c.column_name LIKE 'observed_allcause%' OR c.column_name='last_observed_allcause_dischtime' THEN 'candidate_followup_support'
    WHEN c.column_name IN('gender','anchor_age','anchor_year','anchor_year_group','dod','deathtime','hospital_expire_flag') THEN 'demographic_or_death_support'
    ELSE 'admission_context'
  END AS field_group,
  CASE c.column_name
    WHEN 'subject_id' THEN 'MIMIC patient identifier'
    WHEN 'hadm_id' THEN 'strict AECOPD hospital admission identifier; one row per hadm_id'
    WHEN 'best_seq_num' THEN 'lowest diagnosis sequence number among strict AECOPD codes'
    WHEN 'aecopd_codes' THEN 'strict AECOPD ICD codes observed for this admission'
    WHEN 'prior_aecopd_count' THEN 'number of earlier system-observed strict AECOPD admissions'
    WHEN 'prior_all_admission_count' THEN 'number of earlier system-observed all-cause admissions'
    WHEN 'next_observed_aecopd_hadm_id' THEN 'next strict AECOPD admission observed in this database; absence is not a true negative'
    WHEN 'days_to_next_observed_aecopd' THEN 'days from current discharge to next observed strict AECOPD admission'
    WHEN 'has_discharge_note' THEN 'at least one matching discharge note exists'
    WHEN 'has_note_available_by_discharge' THEN 'at least one matching discharge note has charttime and storetime no later than dischtime'
    WHEN 'note_time_flag' THEN 'audit classification of discharge note timing relative to discharge'
    WHEN 'has_icu_stay' THEN 'at least one ICU stay is linked to this hospital admission; not an inclusion criterion'
    WHEN 'observed_allcause_span_after_discharge_days' THEN 'span to the patient last observed all-cause discharge; not validated censoring time'
    WHEN 'next_observed_aecopd_within_90d' THEN 'candidate indicator that a next observed strict AECOPD occurs within 90 days; not a finalized label'
    WHEN 'next_observed_aecopd_within_180d' THEN 'candidate indicator that a next observed strict AECOPD occurs within 180 days; not a finalized label'
    WHEN 'next_observed_aecopd_within_365d' THEN 'candidate indicator that a next observed strict AECOPD occurs within 365 days; not a finalized label'
    WHEN 'candidate_followup_status_90d' THEN 'candidate 90-day event/follow-up audit status; absence of event is not confirmed negative'
    WHEN 'candidate_followup_status_180d' THEN 'candidate 180-day event/follow-up audit status; absence of event is not confirmed negative'
    WHEN 'candidate_followup_status_365d' THEN 'candidate 365-day event/follow-up audit status; absence of event is not confirmed negative'
    ELSE 'See SQL and source-field name; retained without cleaning or modeling transformation'
  END AS definition
FROM information_schema.columns c
WHERE c.table_schema='project_aecopd' AND c.table_name='master_cohort'
ORDER BY c.ordinal_position;

COMMIT;

\copy (SELECT * FROM project_aecopd.master_cohort ORDER BY subject_id,admittime,hadm_id) TO 'outputs/06_master_cohort.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.master_cohort_qc) TO 'outputs/06_master_cohort_qc.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.master_cohort_data_dictionary ORDER BY ordinal_position) TO 'outputs/06_master_cohort_data_dictionary.csv' CSV HEADER


````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/07_two_landmark_availability_audit.sql · lines 1–467

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/07_two_landmark_availability_audit.sql","part":1,"start_line":1,"end_line":467,"source_sha256":"2519f8521f093681b805f26cc2ef346ae151eb120406da619b5720fa9d0a12fd","fragment_sha256":"2519f8521f093681b805f26cc2ef346ae151eb120406da619b5720fa9d0a12fd","added_terminal_newline":false} -->
````sql
\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;

DROP TABLE IF EXISTS project_aecopd.landmark_definitions;
CREATE TABLE project_aecopd.landmark_definitions AS
SELECT
  m.subject_id,
  m.hadm_id,
  LEAST(m.admittime,COALESCE(m.edregtime,m.admittime)) AS admission_window_start,
  m.admittime + INTERVAL '24 hours' AS admission_24h_landmark,
  m.dischtime AS discharge_landmark,
  (m.dischtime > m.admittime + INTERVAL '24 hours') AS eligible_admission_24h,
  (m.dischtime > m.admittime) AS eligible_discharge,
  m.qc_nonpositive_los
FROM project_aecopd.master_cohort m;
ALTER TABLE project_aecopd.landmark_definitions ADD CONSTRAINT landmark_definitions_pk PRIMARY KEY(hadm_id);
CREATE INDEX landmark_definitions_subject_idx ON project_aecopd.landmark_definitions(subject_id);
ANALYZE project_aecopd.landmark_definitions;

DROP TABLE IF EXISTS project_aecopd.landmark_lab_event_timing;
CREATE TABLE project_aecopd.landmark_lab_event_timing AS
SELECT
  c.subject_id,c.hadm_id,l.itemid,l.charttime,l.storetime,
  (l.valuenum IS NOT NULL) AS has_numeric_value,
  l.valueuom,
  (c.eligible_admission_24h
    AND l.charttime BETWEEN c.admission_window_start AND c.admission_24h_landmark) AS occurred_by_admission_24h,
  (c.eligible_admission_24h
    AND l.charttime BETWEEN c.admission_window_start AND c.admission_24h_landmark
    AND l.storetime IS NOT NULL
    AND l.storetime <= c.admission_24h_landmark) AS available_by_admission_24h,
  (c.eligible_discharge
    AND l.charttime BETWEEN c.admission_window_start AND c.discharge_landmark) AS occurred_by_discharge,
  (c.eligible_discharge
    AND l.charttime BETWEEN c.admission_window_start AND c.discharge_landmark
    AND l.storetime IS NOT NULL
    AND l.storetime <= c.discharge_landmark) AS available_by_discharge
FROM mimiciv_hosp.labevents l
JOIN project_aecopd.landmark_definitions c
  ON c.hadm_id=l.hadm_id AND c.subject_id=l.subject_id;
CREATE INDEX landmark_lab_event_timing_hadm_idx ON project_aecopd.landmark_lab_event_timing(hadm_id);
CREATE INDEX landmark_lab_event_timing_item_idx ON project_aecopd.landmark_lab_event_timing(itemid);
ANALYZE project_aecopd.landmark_lab_event_timing;

DROP TABLE IF EXISTS project_aecopd.landmark_lab_item_coverage_top100;
CREATE TABLE project_aecopd.landmark_lab_item_coverage_top100 AS
WITH denominators AS (
  SELECT
    COUNT(*) FILTER(WHERE eligible_admission_24h) AS admission_24h_denominator,
    COUNT(*) FILTER(WHERE eligible_discharge) AS discharge_denominator
  FROM project_aecopd.landmark_definitions
), per_admission AS (
  SELECT
    itemid,hadm_id,
    COUNT(*) FILTER(WHERE available_by_admission_24h) AS admission_24h_measurements,
    COUNT(*) FILTER(WHERE available_by_discharge) AS discharge_measurements
  FROM project_aecopd.landmark_lab_event_timing
  GROUP BY itemid,hadm_id
), item_summary AS (
  SELECT
    e.itemid,
    COUNT(*) AS measurements,
    COUNT(DISTINCT e.hadm_id) AS admissions_any,
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE e.occurred_by_admission_24h) AS admissions_occurred_by_admission_24h,
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE e.available_by_admission_24h) AS admissions_available_by_admission_24h,
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE e.occurred_by_discharge) AS admissions_occurred_by_discharge,
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE e.available_by_discharge) AS admissions_available_by_discharge,
    COUNT(*) FILTER(WHERE e.occurred_by_admission_24h AND NOT e.available_by_admission_24h) AS admission_results_stored_after_landmark,
    COUNT(*) FILTER(WHERE e.occurred_by_discharge AND NOT e.available_by_discharge) AS discharge_results_stored_after_landmark,
    COUNT(DISTINCT e.valueuom) AS unit_count,
    STRING_AGG(DISTINCT COALESCE(e.valueuom,'<NULL>'),' | ' ORDER BY COALESCE(e.valueuom,'<NULL>')) AS observed_units
  FROM project_aecopd.landmark_lab_event_timing e
  GROUP BY e.itemid
), measurement_summary AS (
  SELECT
    itemid,
    percentile_cont(0.5) WITHIN GROUP(ORDER BY admission_24h_measurements)
      FILTER(WHERE admission_24h_measurements>0) AS median_measurements_admission_24h,
    percentile_cont(0.5) WITHIN GROUP(ORDER BY discharge_measurements)
      FILTER(WHERE discharge_measurements>0) AS median_measurements_by_discharge
  FROM per_admission
  GROUP BY itemid
)
SELECT
  s.itemid,d.label,d.fluid,d.category,s.measurements,s.admissions_any,
  x.admission_24h_denominator,
  s.admissions_occurred_by_admission_24h,
  s.admissions_available_by_admission_24h,
  ROUND(100.0*s.admissions_available_by_admission_24h/NULLIF(x.admission_24h_denominator,0),2) AS admission_24h_coverage_pct,
  m.median_measurements_admission_24h,
  x.discharge_denominator,
  s.admissions_occurred_by_discharge,
  s.admissions_available_by_discharge,
  ROUND(100.0*s.admissions_available_by_discharge/NULLIF(x.discharge_denominator,0),2) AS discharge_coverage_pct,
  m.median_measurements_by_discharge,
  s.admission_results_stored_after_landmark,
  s.discharge_results_stored_after_landmark,
  s.unit_count,s.observed_units
FROM item_summary s
CROSS JOIN denominators x
JOIN measurement_summary m USING(itemid)
LEFT JOIN mimiciv_hosp.d_labitems d USING(itemid)
ORDER BY s.admissions_available_by_discharge DESC,s.admissions_available_by_admission_24h DESC,s.measurements DESC
LIMIT 100;

DROP TABLE IF EXISTS project_aecopd.landmark_medication_coverage_top100;
CREATE TABLE project_aecopd.landmark_medication_coverage_top100 AS
WITH denominators AS (
  SELECT
    COUNT(*) FILTER(WHERE eligible_admission_24h) AS admission_24h_denominator,
    COUNT(*) FILTER(WHERE eligible_discharge) AS discharge_denominator
  FROM project_aecopd.landmark_definitions
), medication_rows AS (
  SELECT
    'mimiciv_hosp.prescriptions'::text AS data_source,
    COALESCE(p.drug,'<NULL>')::text AS medication,
    p.subject_id,p.hadm_id,p.starttime,p.stoptime,
    p.starttime AS availability_time,
    c.admission_window_start,c.admission_24h_landmark,c.discharge_landmark,
    c.eligible_admission_24h,c.eligible_discharge,
    'starttime proxy; source has no storetime'::text AS availability_basis
  FROM mimiciv_hosp.prescriptions p
  JOIN project_aecopd.landmark_definitions c
    ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.pharmacy',COALESCE(p.medication,'<NULL>'),
    p.subject_id,p.hadm_id,p.starttime,p.stoptime,
    COALESCE(p.verifiedtime,p.entertime,p.starttime),
    c.admission_window_start,c.admission_24h_landmark,c.discharge_landmark,
    c.eligible_admission_24h,c.eligible_discharge,
    'starttime plus verifiedtime/entertime'::text
  FROM mimiciv_hosp.pharmacy p
  JOIN project_aecopd.landmark_definitions c
    ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
), summary AS (
  SELECT
    data_source,medication,MIN(availability_basis) AS availability_basis,
    COUNT(*) AS records,
    COUNT(DISTINCT hadm_id) AS admissions_any,
    COUNT(DISTINCT hadm_id) FILTER(
      WHERE eligible_admission_24h
        AND starttime BETWEEN admission_window_start AND admission_24h_landmark
        AND availability_time<=admission_24h_landmark
    ) AS admissions_available_by_admission_24h,
    COUNT(DISTINCT hadm_id) FILTER(
      WHERE eligible_discharge
        AND starttime BETWEEN admission_window_start AND discharge_landmark
        AND availability_time<=discharge_landmark
    ) AS admissions_available_by_discharge,
    COUNT(*) FILTER(WHERE starttime>discharge_landmark) AS records_starting_after_discharge
  FROM medication_rows
  GROUP BY data_source,medication
)
SELECT
  s.*,d.admission_24h_denominator,
  ROUND(100.0*s.admissions_available_by_admission_24h/NULLIF(d.admission_24h_denominator,0),2) AS admission_24h_coverage_pct,
  d.discharge_denominator,
  ROUND(100.0*s.admissions_available_by_discharge/NULLIF(d.discharge_denominator,0),2) AS discharge_coverage_pct
FROM summary s CROSS JOIN denominators d
ORDER BY s.admissions_available_by_discharge DESC,s.admissions_available_by_admission_24h DESC,s.records DESC
LIMIT 100;

DROP TABLE IF EXISTS project_aecopd.landmark_note_coverage;
CREATE TABLE project_aecopd.landmark_note_coverage AS
WITH denominators AS (
  SELECT
    COUNT(*) FILTER(WHERE eligible_admission_24h) AS admission_24h_denominator,
    COUNT(*) FILTER(WHERE eligible_discharge) AS discharge_denominator
  FROM project_aecopd.landmark_definitions
), note_rows AS (
  SELECT 'mimiciv_note.discharge'::text data_source,n.subject_id,n.hadm_id,n.charttime,n.storetime,LENGTH(n.text) AS text_chars,
         c.admission_window_start,c.admission_24h_landmark,c.discharge_landmark,c.eligible_admission_24h,c.eligible_discharge
  FROM mimiciv_note.discharge n
  JOIN project_aecopd.landmark_definitions c ON c.hadm_id=n.hadm_id AND c.subject_id=n.subject_id
  UNION ALL
  SELECT 'mimiciv_note.radiology',n.subject_id,n.hadm_id,n.charttime,n.storetime,LENGTH(n.text),
         c.admission_window_start,c.admission_24h_landmark,c.discharge_landmark,c.eligible_admission_24h,c.eligible_discharge
  FROM mimiciv_note.radiology n
  JOIN project_aecopd.landmark_definitions c ON c.hadm_id=n.hadm_id AND c.subject_id=n.subject_id
), summary AS (
  SELECT
    data_source,
    COUNT(*) AS documents,
    COUNT(DISTINCT hadm_id) AS admissions_any,
    COUNT(DISTINCT hadm_id) FILTER(
      WHERE eligible_admission_24h AND charttime BETWEEN admission_window_start AND admission_24h_landmark
    ) AS admissions_charted_by_admission_24h,
    COUNT(DISTINCT hadm_id) FILTER(
      WHERE eligible_admission_24h AND charttime BETWEEN admission_window_start AND admission_24h_landmark
        AND storetime IS NOT NULL
        AND storetime<=admission_24h_landmark
    ) AS admissions_available_by_admission_24h,
    COUNT(DISTINCT hadm_id) FILTER(
      WHERE eligible_discharge AND charttime<=discharge_landmark
        AND (data_source='mimiciv_note.discharge' OR charttime>=admission_window_start)
    ) AS admissions_charted_by_discharge,
    COUNT(DISTINCT hadm_id) FILTER(
      WHERE eligible_discharge AND charttime<=discharge_landmark
        AND (data_source='mimiciv_note.discharge' OR charttime>=admission_window_start)
        AND storetime IS NOT NULL
        AND storetime<=discharge_landmark
    ) AS admissions_available_by_discharge,
    percentile_cont(0.5) WITHIN GROUP(ORDER BY text_chars) AS median_text_chars,
    MAX(text_chars) AS max_text_chars
  FROM note_rows GROUP BY data_source
)
SELECT
  s.*,d.admission_24h_denominator,
  ROUND(100.0*s.admissions_available_by_admission_24h/NULLIF(d.admission_24h_denominator,0),2) AS admission_24h_coverage_pct,
  d.discharge_denominator,
  ROUND(100.0*s.admissions_available_by_discharge/NULLIF(d.discharge_denominator,0),2) AS discharge_coverage_pct
FROM summary s CROSS JOIN denominators d
ORDER BY data_source;

DROP TABLE IF EXISTS project_aecopd.landmark_source_availability;
CREATE TABLE project_aecopd.landmark_source_availability AS
WITH c AS MATERIALIZED (
  SELECT * FROM project_aecopd.landmark_definitions
), d AS (
  SELECT
    COUNT(*)::bigint AS cohort_admissions,
    COUNT(*) FILTER(WHERE eligible_admission_24h)::bigint AS admission_24h_denominator,
    COUNT(*) FILTER(WHERE eligible_discharge)::bigint AS discharge_denominator
  FROM c
), medication_keys AS MATERIALIZED (
  SELECT DISTINCT p.subject_id,p.hadm_id,p.pharmacy_id,p.poe_id
  FROM mimiciv_hosp.pharmacy p JOIN c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  WHERE p.pharmacy_id IS NOT NULL OR p.poe_id IS NOT NULL
  UNION
  SELECT DISTINCT p.subject_id,p.hadm_id,p.pharmacy_id,p.poe_id
  FROM mimiciv_hosp.prescriptions p JOIN c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  WHERE p.pharmacy_id IS NOT NULL OR p.poe_id IS NOT NULL
), source_rows AS (
  SELECT
    'project_aecopd.longitudinal_history'::text AS data_source,'history'::text AS data_domain,
    'computed strictly before each index admission'::text AS timing_basis,
    d.cohort_admissions AS admissions_any,
    d.admission_24h_denominator AS occurred_by_admission_24h,
    d.admission_24h_denominator AS available_by_admission_24h,
    d.discharge_denominator AS occurred_by_discharge,
    d.discharge_denominator AS available_by_discharge,
    'real-time eligible'::text AS real_time_status,
    'prior AECOPD and prior all-cause admission history'::text AS audit_note
  FROM d
  UNION ALL
  SELECT 'mimiciv_hosp.patients','demographics','patient-level anchor fields',d.cohort_admissions,d.admission_24h_denominator,d.admission_24h_denominator,d.discharge_denominator,d.discharge_denominator,'eligible with deidentification caveats','age must be derived from anchor_age and anchor_year' FROM d
  UNION ALL
  SELECT 'mimiciv_hosp.admissions.initial','admission context','known at/near admission',d.cohort_admissions,d.admission_24h_denominator,d.admission_24h_denominator,d.discharge_denominator,d.discharge_denominator,'real-time eligible','admission type/location and ED timing; excludes discharge-only fields' FROM d
  UNION ALL
  SELECT 'mimiciv_hosp.admissions.discharge','discharge context','known at discharge',d.cohort_admissions,0,0,d.discharge_denominator,d.discharge_denominator,'discharge landmark only','LOS, discharge location and hospital outcome are unavailable to admission model' FROM d
  UNION ALL
  SELECT
    'mimiciv_hosp.labevents','laboratory','charttime plus storetime',
    COUNT(DISTINCT e.hadm_id),
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE e.occurred_by_admission_24h),
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE e.available_by_admission_24h),
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE e.occurred_by_discharge),
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE e.available_by_discharge),
    'strict availability measurable','ED-to-landmark window; result must be stored by landmark'
  FROM project_aecopd.landmark_lab_event_timing e
  UNION ALL
  SELECT
    'mimiciv_hosp.prescriptions','medication order','starttime proxy; no storetime',
    COUNT(DISTINCT p.hadm_id),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_admission_24h AND p.starttime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_admission_24h AND p.starttime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_discharge AND p.starttime BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_discharge AND p.starttime BETWEEN c.admission_window_start AND c.discharge_landmark),
    'proxy only','order availability cannot be proven because prescriptions has no storetime'
  FROM mimiciv_hosp.prescriptions p JOIN c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.pharmacy','pharmacy medication','starttime plus verifiedtime/entertime',
    COUNT(DISTINCT p.hadm_id),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_admission_24h AND p.starttime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_admission_24h AND p.starttime BETWEEN c.admission_window_start AND c.admission_24h_landmark AND COALESCE(p.verifiedtime,p.entertime,p.starttime)<=c.admission_24h_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_discharge AND p.starttime BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_discharge AND p.starttime BETWEEN c.admission_window_start AND c.discharge_landmark AND COALESCE(p.verifiedtime,p.entertime,p.starttime)<=c.discharge_landmark),
    'strict proxy measurable','verifiedtime preferred, then entertime, then starttime'
  FROM mimiciv_hosp.pharmacy p JOIN c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.emar','actual medication administration','charttime plus storetime; linked conservatively through pharmacy_id',
    COUNT(DISTINCT e.hadm_id),
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE c.eligible_admission_24h AND e.charttime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE c.eligible_admission_24h AND e.charttime BETWEEN c.admission_window_start AND c.admission_24h_landmark AND e.storetime IS NOT NULL AND e.storetime<=c.admission_24h_landmark),
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE c.eligible_discharge AND e.charttime BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE c.eligible_discharge AND e.charttime BETWEEN c.admission_window_start AND c.discharge_landmark AND e.storetime IS NOT NULL AND e.storetime<=c.discharge_landmark),
    'strict availability measurable','may miss eMAR rows without a cohort-linked pharmacy_id'
  FROM mimiciv_hosp.emar e
  JOIN (SELECT DISTINCT subject_id,hadm_id,pharmacy_id FROM medication_keys WHERE pharmacy_id IS NOT NULL) k
    ON k.pharmacy_id=e.pharmacy_id AND k.hadm_id=e.hadm_id AND k.subject_id=e.subject_id
  JOIN c ON c.hadm_id=e.hadm_id AND c.subject_id=e.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.poe','provider orders','ordertime; linked conservatively through poe_id',
    COUNT(DISTINCT p.hadm_id),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_admission_24h AND p.ordertime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_admission_24h AND p.ordertime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_discharge AND p.ordertime BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_discharge AND p.ordertime BETWEEN c.admission_window_start AND c.discharge_landmark),
    'time proxy measurable','may miss orders without a cohort-linked poe_id'
  FROM mimiciv_hosp.poe p
  JOIN (SELECT DISTINCT subject_id,hadm_id,poe_id FROM medication_keys WHERE poe_id IS NOT NULL) k
    ON k.poe_id=p.poe_id AND k.hadm_id=p.hadm_id AND k.subject_id=p.subject_id
  JOIN c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.microbiologyevents','microbiology','charttime/chartdate plus storetime/storedate',
    COUNT(DISTINCT m.hadm_id),
    COUNT(DISTINCT m.hadm_id) FILTER(WHERE c.eligible_admission_24h AND COALESCE(m.charttime,m.chartdate) BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT m.hadm_id) FILTER(WHERE c.eligible_admission_24h AND COALESCE(m.charttime,m.chartdate) BETWEEN c.admission_window_start AND c.admission_24h_landmark AND COALESCE(m.storetime,m.storedate) IS NOT NULL AND COALESCE(m.storetime,m.storedate)<=c.admission_24h_landmark),
    COUNT(DISTINCT m.hadm_id) FILTER(WHERE c.eligible_discharge AND COALESCE(m.charttime,m.chartdate) BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT m.hadm_id) FILTER(WHERE c.eligible_discharge AND COALESCE(m.charttime,m.chartdate) BETWEEN c.admission_window_start AND c.discharge_landmark AND COALESCE(m.storetime,m.storedate) IS NOT NULL AND COALESCE(m.storetime,m.storedate)<=c.discharge_landmark),
    'strict availability measurable','culture result availability can lag specimen time by days'
  FROM mimiciv_hosp.microbiologyevents m JOIN c ON c.hadm_id=m.hadm_id AND c.subject_id=m.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.transfers','location trajectory','intime/outtime',
    COUNT(DISTINCT t.hadm_id),
    COUNT(DISTINCT t.hadm_id) FILTER(WHERE c.eligible_admission_24h AND t.intime<=c.admission_24h_landmark AND COALESCE(t.outtime,c.admission_24h_landmark)>=c.admission_window_start),
    COUNT(DISTINCT t.hadm_id) FILTER(WHERE c.eligible_admission_24h AND t.intime<=c.admission_24h_landmark AND COALESCE(t.outtime,c.admission_24h_landmark)>=c.admission_window_start),
    COUNT(DISTINCT t.hadm_id) FILTER(WHERE c.eligible_discharge AND t.intime<=c.discharge_landmark AND COALESCE(t.outtime,c.discharge_landmark)>=c.admission_window_start),
    COUNT(DISTINCT t.hadm_id) FILTER(WHERE c.eligible_discharge AND t.intime<=c.discharge_landmark AND COALESCE(t.outtime,c.discharge_landmark)>=c.admission_window_start),
    'event-time proxy','source has no storetime'
  FROM mimiciv_hosp.transfers t JOIN c ON c.hadm_id=t.hadm_id AND c.subject_id=t.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.services','clinical service','transfertime',
    COUNT(DISTINCT s.hadm_id),
    COUNT(DISTINCT s.hadm_id) FILTER(WHERE c.eligible_admission_24h AND s.transfertime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT s.hadm_id) FILTER(WHERE c.eligible_admission_24h AND s.transfertime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT s.hadm_id) FILTER(WHERE c.eligible_discharge AND s.transfertime BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT s.hadm_id) FILTER(WHERE c.eligible_discharge AND s.transfertime BETWEEN c.admission_window_start AND c.discharge_landmark),
    'event-time proxy','source has no storetime'
  FROM mimiciv_hosp.services s JOIN c ON c.hadm_id=s.hadm_id AND c.subject_id=s.subject_id
  UNION ALL
  SELECT
    'mimiciv_icu.icustays','ICU stay','intime/outtime',
    COUNT(DISTINCT i.hadm_id),
    COUNT(DISTINCT i.hadm_id) FILTER(WHERE c.eligible_admission_24h AND i.intime<=c.admission_24h_landmark),
    COUNT(DISTINCT i.hadm_id) FILTER(WHERE c.eligible_admission_24h AND i.intime<=c.admission_24h_landmark),
    COUNT(DISTINCT i.hadm_id) FILTER(WHERE c.eligible_discharge AND i.intime<=c.discharge_landmark),
    COUNT(DISTINCT i.hadm_id) FILTER(WHERE c.eligible_discharge AND i.intime<=c.discharge_landmark),
    'event-time proxy','ICU status after admission landmark is an outcome for severity task, not an input'
  FROM mimiciv_icu.icustays i JOIN c ON c.hadm_id=i.hadm_id AND c.subject_id=i.subject_id
  UNION ALL
  SELECT
    'mimiciv_icu.inputevents','ICU inputs and infusions','starttime plus storetime',
    COUNT(DISTINCT x.hadm_id),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_admission_24h AND x.starttime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_admission_24h AND x.starttime BETWEEN c.admission_window_start AND c.admission_24h_landmark AND x.storetime IS NOT NULL AND x.storetime<=c.admission_24h_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_discharge AND x.starttime BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_discharge AND x.starttime BETWEEN c.admission_window_start AND c.discharge_landmark AND x.storetime IS NOT NULL AND x.storetime<=c.discharge_landmark),
    'strict availability measurable','ICU-only source; medication/fluid ingredients require item-level review'
  FROM mimiciv_icu.inputevents x JOIN c ON c.hadm_id=x.hadm_id AND c.subject_id=x.subject_id
  UNION ALL
  SELECT
    'mimiciv_icu.outputevents','ICU outputs','charttime plus storetime',
    COUNT(DISTINCT x.hadm_id),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_admission_24h AND x.charttime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_admission_24h AND x.charttime BETWEEN c.admission_window_start AND c.admission_24h_landmark AND x.storetime IS NOT NULL AND x.storetime<=c.admission_24h_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_discharge AND x.charttime BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_discharge AND x.charttime BETWEEN c.admission_window_start AND c.discharge_landmark AND x.storetime IS NOT NULL AND x.storetime<=c.discharge_landmark),
    'strict availability measurable','ICU-only source; urine and other outputs require item-level review'
  FROM mimiciv_icu.outputevents x JOIN c ON c.hadm_id=x.hadm_id AND c.subject_id=x.subject_id
  UNION ALL
  SELECT
    'mimiciv_icu.procedureevents','ICU procedures','starttime plus storetime',
    COUNT(DISTINCT x.hadm_id),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_admission_24h AND x.starttime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_admission_24h AND x.starttime BETWEEN c.admission_window_start AND c.admission_24h_landmark AND x.storetime IS NOT NULL AND x.storetime<=c.admission_24h_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_discharge AND x.starttime BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_discharge AND x.starttime BETWEEN c.admission_window_start AND c.discharge_landmark AND x.storetime IS NOT NULL AND x.storetime<=c.discharge_landmark),
    'strict availability measurable','ICU-only source; item-level clinical interpretation required'
  FROM mimiciv_icu.procedureevents x JOIN c ON c.hadm_id=x.hadm_id AND c.subject_id=x.subject_id
  UNION ALL
  SELECT
    n.data_source,'clinical text','charttime plus storetime',n.admissions_any,
    n.admissions_charted_by_admission_24h,n.admissions_available_by_admission_24h,
    n.admissions_charted_by_discharge,n.admissions_available_by_discharge,
    'strict availability measurable','text remains outside the structured matrix; only availability and length audited'
  FROM project_aecopd.landmark_note_coverage n
  UNION ALL
  SELECT
    'mimiciv_hosp.omr','outpatient measurements','subject_id plus chartdate; no hadm_id',
    COUNT(DISTINCT c.hadm_id) FILTER(WHERE o.chartdate BETWEEN c.admission_window_start::date-365 AND c.discharge_landmark::date),
    COUNT(DISTINCT c.hadm_id) FILTER(WHERE c.eligible_admission_24h AND o.chartdate BETWEEN c.admission_window_start::date-365 AND c.admission_24h_landmark::date),
    COUNT(DISTINCT c.hadm_id) FILTER(WHERE c.eligible_admission_24h AND o.chartdate BETWEEN c.admission_window_start::date-365 AND c.admission_24h_landmark::date),
    COUNT(DISTINCT c.hadm_id) FILTER(WHERE c.eligible_discharge AND o.chartdate BETWEEN c.admission_window_start::date-365 AND c.discharge_landmark::date),
    COUNT(DISTINCT c.hadm_id) FILTER(WHERE c.eligible_discharge AND o.chartdate BETWEEN c.admission_window_start::date-365 AND c.discharge_landmark::date),
    'date-level proxy','365-day lookback; temporal linkage required because OMR lacks hadm_id'
  FROM c JOIN mimiciv_hosp.omr o ON o.subject_id=c.subject_id AND o.chartdate BETWEEN c.admission_window_start::date-365 AND c.discharge_landmark::date
  UNION ALL
  SELECT
    'mimiciv_hosp.diagnoses_icd','billing diagnoses','no event timestamp',
    COUNT(DISTINCT x.hadm_id),NULL::bigint,NULL::bigint,NULL::bigint,NULL::bigint,
    'not real-time verifiable','current-stay diagnoses are post-discharge billing data; prior diagnoses remain eligible history'
  FROM mimiciv_hosp.diagnoses_icd x JOIN c ON c.hadm_id=x.hadm_id AND c.subject_id=x.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.drgcodes','billing severity group','no event timestamp',
    COUNT(DISTINCT x.hadm_id),NULL::bigint,NULL::bigint,NULL::bigint,NULL::bigint,
    'not real-time verifiable','DRG severity/mortality are retrospective billing fields'
  FROM mimiciv_hosp.drgcodes x JOIN c ON c.hadm_id=x.hadm_id AND c.subject_id=x.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.hcpcsevents','billing procedures','chartdate but coding availability unknown',
    COUNT(DISTINCT x.hadm_id),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_admission_24h AND x.chartdate<=c.admission_24h_landmark::date),NULL::bigint,
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_discharge AND x.chartdate<=c.discharge_landmark::date),NULL::bigint,
    'not real-time verifiable','event date does not establish when the billing record became available'
  FROM mimiciv_hosp.hcpcsevents x JOIN c ON c.hadm_id=x.hadm_id AND c.subject_id=x.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.procedures_icd','billing procedures','chartdate but coding availability unknown',
    COUNT(DISTINCT x.hadm_id),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_admission_24h AND x.chartdate<=c.admission_24h_landmark::date),NULL::bigint,
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_discharge AND x.chartdate<=c.discharge_landmark::date),NULL::bigint,
    'not real-time verifiable','event date does not establish when the billing code became available'
  FROM mimiciv_hosp.procedures_icd x JOIN c ON c.hadm_id=x.hadm_id AND c.subject_id=x.subject_id
  UNION ALL
  SELECT
    'mimiciv_icu.chartevents','ICU vitals and observations','charttime/storetime; data scan deferred',
    COUNT(*) FILTER(WHERE i.has_icu_stay),NULL::bigint,NULL::bigint,NULL::bigint,NULL::bigint,
    'deferred for controlled extraction','432,997,491-row source lacks a cohort-key index; must select itemids and extraction strategy before scanning'
  FROM project_aecopd.icu_admission_audit i
), final AS (
  SELECT
    s.data_source,s.data_domain,s.timing_basis,d.cohort_admissions,d.admission_24h_denominator,
    s.admissions_any,s.occurred_by_admission_24h,s.available_by_admission_24h,
    ROUND(100.0*s.available_by_admission_24h/NULLIF(d.admission_24h_denominator,0),2) AS admission_24h_coverage_pct,
    d.discharge_denominator,s.occurred_by_discharge,s.available_by_discharge,
    ROUND(100.0*s.available_by_discharge/NULLIF(d.discharge_denominator,0),2) AS discharge_coverage_pct,
    s.real_time_status,s.audit_note
  FROM source_rows s CROSS JOIN d
)
SELECT * FROM final ORDER BY data_domain,data_source;

DROP TABLE IF EXISTS project_aecopd.landmark_audit_qc;
CREATE TABLE project_aecopd.landmark_audit_qc AS
SELECT 'landmark rows'::text AS check_name,5950::bigint AS expected,COUNT(*)::bigint AS actual,(COUNT(*)=5950) AS passed FROM project_aecopd.landmark_definitions
UNION ALL SELECT 'landmark unique hadm_id',5950,COUNT(DISTINCT hadm_id),(COUNT(DISTINCT hadm_id)=5950) FROM project_aecopd.landmark_definitions
UNION ALL SELECT 'nonpositive discharge windows',1,COUNT(*) FILTER(WHERE NOT eligible_discharge),(COUNT(*) FILTER(WHERE NOT eligible_discharge)=1) FROM project_aecopd.landmark_definitions
UNION ALL SELECT 'source availability rows',24,COUNT(*),(COUNT(*)=24) FROM project_aecopd.landmark_source_availability;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM project_aecopd.landmark_audit_qc WHERE NOT passed) THEN
    RAISE EXCEPTION 'Stage 7 landmark audit QC failed. Transaction will be rolled back.';
  END IF;
END $$;

COMMIT;

\copy (SELECT * FROM project_aecopd.landmark_definitions ORDER BY subject_id,admission_24h_landmark,hadm_id) TO 'outputs/07_landmark_definitions.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.landmark_source_availability ORDER BY data_domain,data_source) TO 'outputs/07_source_availability.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.landmark_lab_item_coverage_top100) TO 'outputs/07_lab_landmark_coverage_top100.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.landmark_medication_coverage_top100) TO 'outputs/07_medication_landmark_coverage_top100.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.landmark_note_coverage ORDER BY data_source) TO 'outputs/07_note_landmark_coverage.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.landmark_audit_qc ORDER BY check_name) TO 'outputs/07_landmark_audit_qc.csv' CSV HEADER

SELECT 'STAGE_07_LANDMARK_AUDIT_PASSED' AS status;


````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08_copd_longitudinal_v1_mimiciii.sql · lines 1–66

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08_copd_longitudinal_v1_mimiciii.sql","part":1,"start_line":1,"end_line":66,"source_sha256":"9bf77605b3619e4de1e58ae6b3a7e4d3f8003489676d4e644638a9e3f7e56ffd","fragment_sha256":"9bf77605b3619e4de1e58ae6b3a7e4d3f8003489676d4e644638a9e3f7e56ffd","added_terminal_newline":false} -->
````sql
-- 在独立 mimiciii 数据库执行；源表为 public.admissions / public.diagnoses_icd。
-- 与 MIMIC-IV 版本使用相同的 ICD-9 患者入组与住院层语义；不跨库合并患者。
\set ON_ERROR_STOP on
BEGIN;

CREATE SCHEMA IF NOT EXISTS project_aecopd;
CREATE TABLE project_aecopd.copd_longitudinal_v1 AS
WITH codebook(icd9_code, code_group) AS (
  VALUES
    ('49120', 'copd_chronic'),
    ('49121', 'aecopd_strict'),
    ('49122', 'copd_acute_bronchitis'),
    ('4928',  'emphysema'),
    ('496',   'copd_chronic')
), diagnosis_flags AS (
  SELECT d.subject_id, d.hadm_id,
         bool_or(d.seq_num <= 3) AS has_top3_qualifying_code,
         bool_or(c.code_group = 'aecopd_strict') AS is_strict_aecopd,
         bool_or(c.code_group = 'copd_acute_bronchitis') AS has_49122,
         bool_or(c.code_group = 'emphysema') AS has_emphysema_code,
         bool_or(c.code_group = 'copd_chronic') AS has_chronic_copd_code,
         array_agg(DISTINCT d.icd9_code ORDER BY d.icd9_code) AS matching_codes
  FROM public.diagnoses_icd d
  JOIN codebook c USING (icd9_code)
  GROUP BY d.subject_id, d.hadm_id
), first_qualification AS (
  SELECT DISTINCT ON (a.subject_id)
         a.subject_id, a.hadm_id AS first_qualifying_hadm_id,
         a.admittime AS first_qualifying_admittime
  FROM public.admissions a
  JOIN diagnosis_flags d USING (subject_id, hadm_id)
  WHERE d.has_top3_qualifying_code
  ORDER BY a.subject_id, a.admittime, a.hadm_id
), selected_admissions AS (
  SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime,
         a.admission_type, a.admission_location, a.discharge_location,
         a.hospital_expire_flag,
         q.first_qualifying_hadm_id, q.first_qualifying_admittime,
         COALESCE(d.has_top3_qualifying_code, false) AS has_top3_qualifying_code,
         COALESCE(d.is_strict_aecopd, false) AS is_strict_aecopd,
         COALESCE(d.has_49122, false) AS has_49122,
         false AS has_j440,
         COALESCE(d.has_emphysema_code, false) AS has_emphysema_code,
         COALESCE(d.has_chronic_copd_code, false) AS has_chronic_copd_code,
         d.matching_codes,
         (a.admittime, a.hadm_id) >=
           (q.first_qualifying_admittime, q.first_qualifying_hadm_id)
           AS at_or_after_first_qualification
  FROM public.admissions a
  JOIN first_qualification q USING (subject_id)
  LEFT JOIN diagnosis_flags d USING (subject_id, hadm_id)
)
SELECT s.*,
       row_number() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) AS all_admission_order,
       row_number() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) - 1 AS prior_all_admission_count,
       coalesce(sum(is_strict_aecopd::integer) OVER (
         PARTITION BY subject_id ORDER BY admittime, hadm_id
         ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0)::integer
         AS prior_strict_aecopd_count
FROM selected_admissions s;

ALTER TABLE project_aecopd.copd_longitudinal_v1 ADD PRIMARY KEY (hadm_id);
CREATE INDEX copd_longitudinal_v1_subject_time_idx
  ON project_aecopd.copd_longitudinal_v1(subject_id, admittime, hadm_id);

COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08_copd_longitudinal_v1_mimiciv.sql · lines 1–75

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08_copd_longitudinal_v1_mimiciv.sql","part":1,"start_line":1,"end_line":75,"source_sha256":"f2f2e8741cc3b170cc9058a5efd861ff6578c746be03f9234c0c196ed378c51c","fragment_sha256":"f2f2e8741cc3b170cc9058a5efd861ff6578c746be03f9234c0c196ed378c51c","added_terminal_newline":false} -->
````sql
-- 在 mimiciv 数据库执行。创建独立纵向住院表，不重建既有 master/model 表。
-- 患者层：任一次住院前三位出现 COPD/常见肺气肿编码即可入组。
-- 住院层：保留该患者在本数据库的全部住院，包括首次入组编码之前的住院。
-- 事件层：严格 AECOPD 与 COPD+急性支气管炎、COPD+感染分列，不在此处定义预测标签。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_longitudinal_v1 AS
WITH codebook(icd_version, icd_code, code_group) AS (
  VALUES
    (9, '49120', 'copd_chronic'),
    (9, '49121', 'aecopd_strict'),
    (9, '49122', 'copd_acute_bronchitis'),
    (9, '4928',  'emphysema'),
    (9, '496',   'copd_chronic'),
    (10, 'J431', 'emphysema'),
    (10, 'J432', 'emphysema'),
    (10, 'J438', 'emphysema'),
    (10, 'J439', 'emphysema'),
    (10, 'J440', 'copd_acute_infection'),
    (10, 'J441', 'aecopd_strict'),
    (10, 'J449', 'copd_chronic')
), diagnosis_flags AS (
  SELECT d.subject_id, d.hadm_id,
         bool_or(d.seq_num <= 3) AS has_top3_qualifying_code,
         bool_or(c.code_group = 'aecopd_strict') AS is_strict_aecopd,
         bool_or(c.code_group = 'copd_acute_bronchitis') AS has_49122,
         bool_or(c.code_group = 'copd_acute_infection') AS has_j440,
         bool_or(c.code_group = 'emphysema') AS has_emphysema_code,
         bool_or(c.code_group = 'copd_chronic') AS has_chronic_copd_code,
         array_agg(DISTINCT d.icd_code ORDER BY d.icd_code) AS matching_codes
  FROM mimiciv_hosp.diagnoses_icd d
  JOIN codebook c ON c.icd_version=d.icd_version AND c.icd_code=d.icd_code
  GROUP BY d.subject_id, d.hadm_id
), first_qualification AS (
  SELECT DISTINCT ON (a.subject_id)
         a.subject_id, a.hadm_id AS first_qualifying_hadm_id,
         a.admittime AS first_qualifying_admittime
  FROM mimiciv_hosp.admissions a
  JOIN diagnosis_flags d USING (subject_id, hadm_id)
  WHERE d.has_top3_qualifying_code
  ORDER BY a.subject_id, a.admittime, a.hadm_id
), selected_admissions AS (
  SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime,
         a.admission_type, a.admission_location, a.discharge_location,
         a.hospital_expire_flag,
         q.first_qualifying_hadm_id, q.first_qualifying_admittime,
         COALESCE(d.has_top3_qualifying_code, false) AS has_top3_qualifying_code,
         COALESCE(d.is_strict_aecopd, false) AS is_strict_aecopd,
         COALESCE(d.has_49122, false) AS has_49122,
         COALESCE(d.has_j440, false) AS has_j440,
         COALESCE(d.has_emphysema_code, false) AS has_emphysema_code,
         COALESCE(d.has_chronic_copd_code, false) AS has_chronic_copd_code,
         d.matching_codes,
         (a.admittime, a.hadm_id) >=
           (q.first_qualifying_admittime, q.first_qualifying_hadm_id)
           AS at_or_after_first_qualification
  FROM mimiciv_hosp.admissions a
  JOIN first_qualification q USING (subject_id)
  LEFT JOIN diagnosis_flags d USING (subject_id, hadm_id)
)
SELECT s.*,
       row_number() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) AS all_admission_order,
       row_number() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) - 1 AS prior_all_admission_count,
       coalesce(sum(is_strict_aecopd::integer) OVER (
         PARTITION BY subject_id ORDER BY admittime, hadm_id
         ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0)::integer
         AS prior_strict_aecopd_count
FROM selected_admissions s;

ALTER TABLE project_aecopd.copd_longitudinal_v1 ADD PRIMARY KEY (hadm_id);
CREATE INDEX copd_longitudinal_v1_subject_time_idx
  ON project_aecopd.copd_longitudinal_v1(subject_id, admittime, hadm_id);

COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08b_copd_longitudinal_v2_mimiciii.sql · lines 1–41

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08b_copd_longitudinal_v2_mimiciii.sql","part":1,"start_line":1,"end_line":41,"source_sha256":"bcb5781372557616ce735df444afa2e3ea2c4809a0a5aac1cbad3be73c13978c","fragment_sha256":"bcb5781372557616ce735df444afa2e3ea2c4809a0a5aac1cbad3be73c13978c","added_terminal_newline":false} -->
````sql
-- 在独立 mimiciii 数据库执行；只创建 v2，不覆盖 v1。
-- 修订：49120、496 仅主诊断(seq_num=1)可触发患者入组；
-- 49121、49122、4928 仍允许诊断前 3 位。入组后保留患者全部住院。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_longitudinal_v2 AS
WITH qualifying_admissions AS (
  SELECT DISTINCT d.subject_id, d.hadm_id
  FROM public.diagnoses_icd d
  WHERE (d.seq_num <= 3 AND d.icd9_code IN ('49121','49122','4928'))
     OR (d.seq_num = 1 AND d.icd9_code IN ('49120','496'))
), first_qualification AS (
  SELECT DISTINCT ON (v.subject_id)
         v.subject_id, v.hadm_id AS first_qualifying_hadm_id,
         v.admittime AS first_qualifying_admittime
  FROM project_aecopd.copd_longitudinal_v1 v
  JOIN qualifying_admissions q USING (subject_id, hadm_id)
  ORDER BY v.subject_id, v.admittime, v.hadm_id
)
SELECT v.subject_id, v.hadm_id, v.admittime, v.dischtime,
       v.admission_type, v.admission_location, v.discharge_location,
       v.hospital_expire_flag,
       f.first_qualifying_hadm_id, f.first_qualifying_admittime,
       (q.hadm_id IS NOT NULL) AS has_qualifying_code,
       v.is_strict_aecopd, v.has_49122, v.has_j440,
       v.has_emphysema_code, v.has_chronic_copd_code, v.matching_codes,
       (v.admittime, v.hadm_id) >=
         (f.first_qualifying_admittime, f.first_qualifying_hadm_id)
         AS at_or_after_first_qualification,
       v.all_admission_order, v.prior_all_admission_count,
       v.prior_strict_aecopd_count
FROM project_aecopd.copd_longitudinal_v1 v
JOIN first_qualification f USING (subject_id)
LEFT JOIN qualifying_admissions q USING (subject_id, hadm_id);

ALTER TABLE project_aecopd.copd_longitudinal_v2 ADD PRIMARY KEY (hadm_id);
CREATE INDEX copd_longitudinal_v2_subject_time_idx
  ON project_aecopd.copd_longitudinal_v2(subject_id, admittime, hadm_id);

COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08b_copd_longitudinal_v2_mimiciv.sql · lines 1–48

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08b_copd_longitudinal_v2_mimiciv.sql","part":1,"start_line":1,"end_line":48,"source_sha256":"f5c95e87f93d9af2185e0c0ebe8e310f2fef16701167be78b2c543c9fa3a4de5","fragment_sha256":"f5c95e87f93d9af2185e0c0ebe8e310f2fef16701167be78b2c543c9fa3a4de5","added_terminal_newline":false} -->
````sql
-- 在 mimiciv 数据库执行；只创建 v2，不覆盖 v1 或任何模型表。
-- 修订：49120、496、J449 仅主诊断(seq_num=1)可触发患者入组；
-- 其余急性/肺气肿编码仍允许诊断前 3 位。入组后保留患者全部住院。
-- v1 是上述条件的超集，因此用其全住院资料作稳定底座；不复用 v1 的入组标记。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_longitudinal_v2 AS
WITH qualifying_admissions AS (
  SELECT DISTINCT d.subject_id, d.hadm_id
  FROM mimiciv_hosp.diagnoses_icd d
  WHERE (d.seq_num <= 3 AND (
           (d.icd_version = 9 AND d.icd_code IN ('49121','49122','4928'))
        OR (d.icd_version = 10 AND d.icd_code IN ('J431','J432','J438','J439','J440','J441'))
        ))
     OR (d.seq_num = 1 AND (
           (d.icd_version = 9 AND d.icd_code IN ('49120','496'))
        OR (d.icd_version = 10 AND d.icd_code = 'J449')
        ))
), first_qualification AS (
  SELECT DISTINCT ON (v.subject_id)
         v.subject_id, v.hadm_id AS first_qualifying_hadm_id,
         v.admittime AS first_qualifying_admittime
  FROM project_aecopd.copd_longitudinal_v1 v
  JOIN qualifying_admissions q USING (subject_id, hadm_id)
  ORDER BY v.subject_id, v.admittime, v.hadm_id
)
SELECT v.subject_id, v.hadm_id, v.admittime, v.dischtime,
       v.admission_type, v.admission_location, v.discharge_location,
       v.hospital_expire_flag,
       f.first_qualifying_hadm_id, f.first_qualifying_admittime,
       (q.hadm_id IS NOT NULL) AS has_qualifying_code,
       v.is_strict_aecopd, v.has_49122, v.has_j440,
       v.has_emphysema_code, v.has_chronic_copd_code, v.matching_codes,
       (v.admittime, v.hadm_id) >=
         (f.first_qualifying_admittime, f.first_qualifying_hadm_id)
         AS at_or_after_first_qualification,
       v.all_admission_order, v.prior_all_admission_count,
       v.prior_strict_aecopd_count
FROM project_aecopd.copd_longitudinal_v1 v
JOIN first_qualification f USING (subject_id)
LEFT JOIN qualifying_admissions q USING (subject_id, hadm_id);

ALTER TABLE project_aecopd.copd_longitudinal_v2 ADD PRIMARY KEY (hadm_id);
CREATE INDEX copd_longitudinal_v2_subject_time_idx
  ON project_aecopd.copd_longitudinal_v2(subject_id, admittime, hadm_id);

COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08c_copd_coded_admissions_mimiciii.sql · lines 1–61

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08c_copd_coded_admissions_mimiciii.sql","part":1,"start_line":1,"end_line":61,"source_sha256":"9cf614988431f983b2c341d157004f3a1abea92976edfd1c459674dae798353f","fragment_sha256":"9cf614988431f983b2c341d157004f3a1abea92976edfd1c459674dae798353f","added_terminal_newline":false} -->
````sql
-- MIMIC-III：沿用 v1 患者入组规则，仅保留本次住院诊断含 v1 编码的住院。
-- 在独立 mimiciii 数据库执行；不修改 v1/v2 或原始表。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_admissions_v1_gate AS
WITH codebook(icd9_code) AS (
  VALUES ('49120'),('49121'),('49122'),('4928'),('496')
), coded AS (
  SELECT v.subject_id, v.hadm_id, v.admittime, v.dischtime,
         v.admission_type, v.admission_location, v.discharge_location,
         v.hospital_expire_flag,
         v.first_qualifying_hadm_id, v.first_qualifying_admittime,
         v.at_or_after_first_qualification,
         v.has_top3_qualifying_code,
         v.is_strict_aecopd, v.has_49122, v.has_j440,
         v.has_emphysema_code, v.has_chronic_copd_code,
         v.matching_codes,
         min(d.seq_num) AS best_copd_seq_num
  FROM project_aecopd.copd_longitudinal_v1 v
  JOIN public.diagnoses_icd d USING (subject_id, hadm_id)
  JOIN codebook b ON b.icd9_code=d.icd9_code
  WHERE v.matching_codes IS NOT NULL
  GROUP BY v.subject_id, v.hadm_id, v.admittime, v.dischtime,
           v.admission_type, v.admission_location, v.discharge_location,
           v.hospital_expire_flag,
           v.first_qualifying_hadm_id, v.first_qualifying_admittime,
           v.at_or_after_first_qualification,
           v.has_top3_qualifying_code,
           v.is_strict_aecopd, v.has_49122, v.has_j440,
           v.has_emphysema_code, v.has_chronic_copd_code,
           v.matching_codes
)
SELECT 'MIMIC-III'::text AS source_dataset, c.*,
       row_number() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id)
         AS coded_admission_order,
       row_number() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id)-1
         AS prior_coded_admission_count,
       (hospital_expire_flag=0 AND dischtime>admittime)
         AS discharge_landmark_eligible
FROM coded c;

ALTER TABLE project_aecopd.copd_coded_admissions_v1_gate ADD PRIMARY KEY (hadm_id);
CREATE INDEX copd_coded_admissions_v1_gate_subject_time_idx
  ON project_aecopd.copd_coded_admissions_v1_gate(subject_id,admittime,hadm_id);

DO $$ BEGIN
  IF (SELECT count(DISTINCT subject_id)
      FROM project_aecopd.copd_coded_admissions_v1_gate)
     <> (SELECT count(DISTINCT subject_id)
         FROM project_aecopd.copd_longitudinal_v1)
     OR (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate)
        <> (SELECT count(*) FROM project_aecopd.copd_longitudinal_v1
            WHERE matching_codes IS NOT NULL)
     OR EXISTS (SELECT 1 FROM project_aecopd.copd_coded_admissions_v1_gate
                WHERE best_copd_seq_num IS NULL) THEN
    RAISE EXCEPTION 'COPD coded-admission cohort consistency check failed';
  END IF;
END $$;

COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08c_copd_coded_admissions_mimiciv.sql · lines 1–63

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08c_copd_coded_admissions_mimiciv.sql","part":1,"start_line":1,"end_line":63,"source_sha256":"89c77b255464deb3e80ac2be2fb937810b90adb1f50788f9f96f3cc3f7430e73","fragment_sha256":"89c77b255464deb3e80ac2be2fb937810b90adb1f50788f9f96f3cc3f7430e73","added_terminal_newline":false} -->
````sql
-- MIMIC-IV：沿用 v1 患者入组规则，仅保留本次住院诊断含 v1 编码的住院。
-- 创建独立版本；不修改 v1/v2、master_cohort 或任何模型标签。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_admissions_v1_gate AS
WITH codebook(icd_version, icd_code) AS (
  VALUES (9,'49120'),(9,'49121'),(9,'49122'),(9,'4928'),(9,'496'),
         (10,'J431'),(10,'J432'),(10,'J438'),(10,'J439'),
         (10,'J440'),(10,'J441'),(10,'J449')
), coded AS (
  SELECT v.subject_id, v.hadm_id, v.admittime, v.dischtime,
         v.admission_type, v.admission_location, v.discharge_location,
         v.hospital_expire_flag,
         v.first_qualifying_hadm_id, v.first_qualifying_admittime,
         v.at_or_after_first_qualification,
         v.has_top3_qualifying_code,
         v.is_strict_aecopd, v.has_49122, v.has_j440,
         v.has_emphysema_code, v.has_chronic_copd_code,
         v.matching_codes,
         min(d.seq_num) AS best_copd_seq_num
  FROM project_aecopd.copd_longitudinal_v1 v
  JOIN mimiciv_hosp.diagnoses_icd d USING (subject_id, hadm_id)
  JOIN codebook b ON b.icd_version=d.icd_version AND b.icd_code=d.icd_code
  WHERE v.matching_codes IS NOT NULL
  GROUP BY v.subject_id, v.hadm_id, v.admittime, v.dischtime,
           v.admission_type, v.admission_location, v.discharge_location,
           v.hospital_expire_flag,
           v.first_qualifying_hadm_id, v.first_qualifying_admittime,
           v.at_or_after_first_qualification,
           v.has_top3_qualifying_code,
           v.is_strict_aecopd, v.has_49122, v.has_j440,
           v.has_emphysema_code, v.has_chronic_copd_code,
           v.matching_codes
)
SELECT 'MIMIC-IV'::text AS source_dataset, c.*,
       row_number() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id)
         AS coded_admission_order,
       row_number() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id)-1
         AS prior_coded_admission_count,
       (hospital_expire_flag=0 AND dischtime>admittime)
         AS discharge_landmark_eligible
FROM coded c;

ALTER TABLE project_aecopd.copd_coded_admissions_v1_gate ADD PRIMARY KEY (hadm_id);
CREATE INDEX copd_coded_admissions_v1_gate_subject_time_idx
  ON project_aecopd.copd_coded_admissions_v1_gate(subject_id,admittime,hadm_id);

DO $$ BEGIN
  IF (SELECT count(DISTINCT subject_id)
      FROM project_aecopd.copd_coded_admissions_v1_gate)
     <> (SELECT count(DISTINCT subject_id)
         FROM project_aecopd.copd_longitudinal_v1)
     OR (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate)
        <> (SELECT count(*) FROM project_aecopd.copd_longitudinal_v1
            WHERE matching_codes IS NOT NULL)
     OR EXISTS (SELECT 1 FROM project_aecopd.copd_coded_admissions_v1_gate
                WHERE best_copd_seq_num IS NULL) THEN
    RAISE EXCEPTION 'COPD coded-admission cohort consistency check failed';
  END IF;
END $$;

COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08d_copd_coded_note_availability_mimiciii.sql · lines 1–46

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08d_copd_coded_note_availability_mimiciii.sql","part":1,"start_line":1,"end_line":46,"source_sha256":"d426e292fe8f415f4e03c58c42b1e3e59a822b2656b3b3a6608b744024c3fd05","fragment_sha256":"d426e292fe8f415f4e03c58c42b1e3e59a822b2656b3b3a6608b744024c3fd05","added_terminal_newline":false} -->
````sql
-- MIMIC-III：仅保存按住院汇总的文本可用性标记，不复制病历正文。
-- 严格时间判定要求 charttime 与 storetime 均不晚于出院时刻。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_note_availability AS
SELECT c.subject_id, c.hadm_id,
       count(n.row_id) AS note_rows,
       count(n.row_id) FILTER (WHERE btrim(n.category)='Discharge summary')
         AS discharge_summary_rows,
       count(n.row_id) FILTER (
         WHERE btrim(n.category)='Discharge summary'
           AND n.charttime<=c.dischtime AND n.storetime<=c.dischtime)
         AS timed_discharge_summary_rows,
       count(n.row_id) FILTER (
         WHERE btrim(n.category)='Discharge summary'
           AND n.chartdate<=c.dischtime::date)
         AS chartdated_discharge_summary_rows,
       count(n.row_id) FILTER (
         WHERE btrim(n.category)<>'Discharge summary'
           AND n.charttime<=c.dischtime AND n.storetime<=c.dischtime)
         AS timed_other_note_rows,
       count(n.row_id) FILTER (
         WHERE btrim(n.category)='Physician'
           AND n.charttime<=c.dischtime AND n.storetime<=c.dischtime)
         AS timed_physician_note_rows,
       count(n.row_id) FILTER (
         WHERE btrim(n.category) IN ('Nursing','Nursing/other')
           AND n.charttime<=c.dischtime AND n.storetime<=c.dischtime)
         AS timed_nursing_note_rows
FROM project_aecopd.copd_coded_admissions_v1_gate c
LEFT JOIN public.noteevents n
  ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
 AND coalesce(n.iserror,'0')<>'1'
GROUP BY c.subject_id,c.hadm_id;

ALTER TABLE project_aecopd.copd_coded_note_availability ADD PRIMARY KEY (hadm_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_note_availability)
     <> (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate) THEN
    RAISE EXCEPTION 'MIMIC-III note-availability row count mismatch';
  END IF;
END $$;

COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08d_copd_coded_note_availability_mimiciv.sql · lines 1–43

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08d_copd_coded_note_availability_mimiciv.sql","part":1,"start_line":1,"end_line":43,"source_sha256":"16734c798296328fa2545073bec5a84bbe0b0904f25993c1d82b0322bcdce7ad","fragment_sha256":"16734c798296328fa2545073bec5a84bbe0b0904f25993c1d82b0322bcdce7ad","added_terminal_newline":false} -->
````sql
-- MIMIC-IV：分别审计出院小结与影像报告的出院时点可用性；不复制正文。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_note_availability AS
WITH discharge_notes AS (
  SELECT c.hadm_id,
         count(n.note_id) AS discharge_summary_rows,
         count(n.note_id) FILTER (
           WHERE n.charttime<=c.dischtime AND n.storetime<=c.dischtime)
           AS timed_discharge_summary_rows
  FROM project_aecopd.copd_coded_admissions_v1_gate c
  LEFT JOIN mimiciv_note.discharge n
    ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
  GROUP BY c.hadm_id
), radiology_notes AS (
  SELECT c.hadm_id,
         count(n.note_id) AS radiology_rows,
         count(n.note_id) FILTER (
           WHERE n.charttime<=c.dischtime AND n.storetime<=c.dischtime)
           AS timed_radiology_rows
  FROM project_aecopd.copd_coded_admissions_v1_gate c
  LEFT JOIN mimiciv_note.radiology n
    ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
  GROUP BY c.hadm_id
)
SELECT c.subject_id,c.hadm_id,
       d.discharge_summary_rows,d.timed_discharge_summary_rows,
       r.radiology_rows,r.timed_radiology_rows
FROM project_aecopd.copd_coded_admissions_v1_gate c
JOIN discharge_notes d USING (hadm_id)
JOIN radiology_notes r USING (hadm_id);

ALTER TABLE project_aecopd.copd_coded_note_availability ADD PRIMARY KEY (hadm_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_note_availability)
     <> (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate) THEN
    RAISE EXCEPTION 'MIMIC-IV note-availability row count mismatch';
  END IF;
END $$;

COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08e_copd_coded_dataset_manifest_mimiciii.sql · lines 1–53

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08e_copd_coded_dataset_manifest_mimiciii.sql","part":1,"start_line":1,"end_line":53,"source_sha256":"616c50626637fa0475a848313dab6495639c1a78285cdbdec2e75efc0ac7e4f8","fragment_sha256":"616c50626637fa0475a848313dab6495639c1a78285cdbdec2e75efc0ac7e4f8","added_terminal_newline":false} -->
````sql
-- MIMIC-III：新队列的逐住院来源/时间可用性清单，不等于模型特征张量。
-- LABEVENTS 与 MICROBIOLOGYEVENTS 无结果存储时刻；结果当时可见性不可验证。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_dataset_manifest AS
SELECT c.subject_id,c.hadm_id,c.source_dataset,c.admittime,c.dischtime,
       c.best_copd_seq_num,c.is_strict_aecopd,
       c.at_or_after_first_qualification,c.discharge_landmark_eligible,
       c.coded_admission_order,c.prior_coded_admission_count,
       n.discharge_summary_rows>0 AS has_discharge_summary,
       n.timed_discharge_summary_rows>0 AS has_timed_discharge_summary,
       n.timed_other_note_rows>0 AS has_timed_other_note,
       l.lab_rows_charted_before_discharge,
       NULL::bigint AS lab_rows_result_time_verified,
       r.rx_rows_started_before_discharge,
       m.micro_rows_charted_before_discharge,
       NULL::bigint AS micro_rows_result_time_verified,
       i.icu_stay_count
FROM project_aecopd.copd_coded_admissions_v1_gate c
JOIN project_aecopd.copd_coded_note_availability n USING(hadm_id)
LEFT JOIN LATERAL (
  SELECT count(*) AS lab_rows_charted_before_discharge
  FROM public.labevents x
  WHERE x.hadm_id=c.hadm_id AND x.charttime BETWEEN c.admittime AND c.dischtime
) l ON true
LEFT JOIN LATERAL (
  SELECT count(*) AS rx_rows_started_before_discharge
  FROM public.prescriptions x
  WHERE x.hadm_id=c.hadm_id AND x.startdate BETWEEN c.admittime::date AND c.dischtime::date
) r ON true
LEFT JOIN LATERAL (
  SELECT count(*) AS micro_rows_charted_before_discharge
  FROM public.microbiologyevents x
  WHERE x.hadm_id=c.hadm_id AND x.charttime BETWEEN c.admittime AND c.dischtime
) m ON true
LEFT JOIN LATERAL (
  SELECT count(*) AS icu_stay_count
  FROM public.icustays x WHERE x.hadm_id=c.hadm_id
) i ON true;

ALTER TABLE project_aecopd.copd_coded_dataset_manifest ADD PRIMARY KEY (hadm_id);
CREATE INDEX copd_coded_dataset_manifest_subject_time_idx
  ON project_aecopd.copd_coded_dataset_manifest(subject_id,admittime,hadm_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_dataset_manifest)
     <> (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate) THEN
    RAISE EXCEPTION 'MIMIC-III dataset manifest row count mismatch';
  END IF;
END $$;

COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08e_copd_coded_dataset_manifest_mimiciv.sql · lines 1–64

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08e_copd_coded_dataset_manifest_mimiciv.sql","part":1,"start_line":1,"end_line":64,"source_sha256":"c6d7fd2073847827cfb4ce1127822f7ea37cfa35d805876a6d523f3411088807","fragment_sha256":"c6d7fd2073847827cfb4ce1127822f7ea37cfa35d805876a6d523f3411088807","added_terminal_newline":false} -->
````sql
-- MIMIC-IV：新队列的逐住院来源/时间可用性清单，不等于模型特征张量。
-- 处方行只证明医嘱存在，不证明实际给药。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_dataset_manifest AS
WITH cohort AS MATERIALIZED (
  SELECT * FROM project_aecopd.copd_coded_admissions_v1_gate
), labs AS MATERIALIZED (
  SELECT c.hadm_id,
         count(*) AS lab_rows_charted_before_discharge,
         count(*) FILTER (WHERE x.storetime<=c.dischtime) AS lab_rows_result_time_verified
  FROM cohort c JOIN mimiciv_hosp.labevents x ON x.hadm_id=c.hadm_id
  WHERE x.charttime BETWEEN c.admittime AND c.dischtime
  GROUP BY c.hadm_id
), rx AS MATERIALIZED (
  SELECT c.hadm_id,count(*) AS rx_rows_started_before_discharge
  FROM cohort c JOIN mimiciv_hosp.prescriptions x ON x.hadm_id=c.hadm_id
  WHERE x.starttime BETWEEN c.admittime AND c.dischtime
  GROUP BY c.hadm_id
), micro AS MATERIALIZED (
  SELECT c.hadm_id,
         count(*) AS micro_rows_charted_before_discharge,
         count(*) FILTER (WHERE x.storetime<=c.dischtime) AS micro_rows_result_time_verified
  FROM cohort c JOIN mimiciv_hosp.microbiologyevents x ON x.hadm_id=c.hadm_id
  WHERE x.charttime BETWEEN c.admittime AND c.dischtime
  GROUP BY c.hadm_id
), icu AS MATERIALIZED (
  SELECT c.hadm_id,count(*) AS icu_stay_count
  FROM cohort c JOIN mimiciv_icu.icustays x ON x.hadm_id=c.hadm_id
  GROUP BY c.hadm_id
)
SELECT c.subject_id,c.hadm_id,c.source_dataset,c.admittime,c.dischtime,
       c.best_copd_seq_num,c.is_strict_aecopd,
       c.at_or_after_first_qualification,c.discharge_landmark_eligible,
       c.coded_admission_order,c.prior_coded_admission_count,
       n.discharge_summary_rows>0 AS has_discharge_summary,
       n.timed_discharge_summary_rows>0 AS has_timed_discharge_summary,
       n.timed_radiology_rows>0 AS has_timed_other_note,
       COALESCE(l.lab_rows_charted_before_discharge,0) AS lab_rows_charted_before_discharge,
       COALESCE(l.lab_rows_result_time_verified,0) AS lab_rows_result_time_verified,
       COALESCE(r.rx_rows_started_before_discharge,0) AS rx_rows_started_before_discharge,
       COALESCE(m.micro_rows_charted_before_discharge,0) AS micro_rows_charted_before_discharge,
       COALESCE(m.micro_rows_result_time_verified,0) AS micro_rows_result_time_verified,
       COALESCE(i.icu_stay_count,0) AS icu_stay_count
FROM cohort c
JOIN project_aecopd.copd_coded_note_availability n USING(hadm_id)
LEFT JOIN labs l USING(hadm_id)
LEFT JOIN rx r USING(hadm_id)
LEFT JOIN micro m USING(hadm_id)
LEFT JOIN icu i USING(hadm_id);

ALTER TABLE project_aecopd.copd_coded_dataset_manifest ADD PRIMARY KEY (hadm_id);
CREATE INDEX copd_coded_dataset_manifest_subject_time_idx
  ON project_aecopd.copd_coded_dataset_manifest(subject_id,admittime,hadm_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_dataset_manifest)
     <> (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate) THEN
    RAISE EXCEPTION 'MIMIC-IV dataset manifest row count mismatch';
  END IF;
END $$;

COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08f_copd_coded_timed_notes_mimiciii.sql · lines 1–18

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08f_copd_coded_timed_notes_mimiciii.sql","part":1,"start_line":1,"end_line":18,"source_sha256":"2995bf7180bc6a74beea74e5a746d2c8b73669d9ebb1150563cbd8da23dafc55","fragment_sha256":"2995bf7180bc6a74beea74e5a746d2c8b73669d9ebb1150563cbd8da23dafc55","added_terminal_newline":false} -->
````sql
-- MIMIC-III：出院节点可证明已写入的非出院小结文本视图。
-- 出院小结的 charttime/storetime 均缺失，不能混入严格及时文本。
\set ON_ERROR_STOP on

CREATE VIEW project_aecopd.copd_coded_timed_notes_v1 AS
SELECT 'MIMIC-III'::text AS source_dataset,
       c.subject_id,c.hadm_id,
       n.row_id::text AS note_id,
       btrim(n.category)::text AS note_role,
       n.charttime,n.storetime,n.text
FROM project_aecopd.copd_coded_admissions_v1_gate c
JOIN public.noteevents n
  ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
WHERE c.discharge_landmark_eligible
  AND coalesce(n.iserror,'0')<>'1'
  AND btrim(n.category)<>'Discharge summary'
  AND n.charttime<=c.dischtime AND n.storetime<=c.dischtime
  AND n.text IS NOT NULL;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08f_copd_coded_timed_notes_mimiciv.sql · lines 1–27

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08f_copd_coded_timed_notes_mimiciv.sql","part":1,"start_line":1,"end_line":27,"source_sha256":"4db6403e95fa8aaeda147b2ca145bc1431d2cc08afb58ce5bc552fb9c534ee49","fragment_sha256":"4db6403e95fa8aaeda147b2ca145bc1431d2cc08afb58ce5bc552fb9c534ee49","added_terminal_newline":false} -->
````sql
-- MIMIC-IV：出院节点可证明已写入的出院小结／影像报告文本视图。
\set ON_ERROR_STOP on

CREATE VIEW project_aecopd.copd_coded_timed_notes_v1 AS
SELECT 'MIMIC-IV'::text AS source_dataset,
       c.subject_id,c.hadm_id,
       ('discharge:'||n.note_id)::text AS note_id,
       'Discharge summary'::text AS note_role,
       n.charttime,n.storetime,n.text
FROM project_aecopd.copd_coded_admissions_v1_gate c
JOIN mimiciv_note.discharge n
  ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
WHERE c.discharge_landmark_eligible
  AND n.charttime<=c.dischtime AND n.storetime<=c.dischtime
  AND n.text IS NOT NULL
UNION ALL
SELECT 'MIMIC-IV'::text AS source_dataset,
       c.subject_id,c.hadm_id,
       ('radiology:'||n.note_id)::text AS note_id,
       'Radiology'::text AS note_role,
       n.charttime,n.storetime,n.text
FROM project_aecopd.copd_coded_admissions_v1_gate c
JOIN mimiciv_note.radiology n
  ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
WHERE c.discharge_landmark_eligible
  AND n.charttime<=c.dischtime AND n.storetime<=c.dischtime
  AND n.text IS NOT NULL;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08g_copd_coded_strict_index_labels_mimiciii.sql · lines 1–89

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08g_copd_coded_strict_index_labels_mimiciii.sql","part":1,"start_line":1,"end_line":89,"source_sha256":"e8ded8bac0871d977d2c0192441467cf9f08e578c98955360171cca4e2e3e241","fragment_sha256":"e8ded8bac0871d977d2c0192441467cf9f08e578c98955360171cca4e2e3e241","added_terminal_newline":false} -->
````sql
-- MIMIC-III：与 MIMIC-IV 相同的索引／竞争事件语义，按本库转院编码适配。
-- 365 天是可观察结局的行政窗口，不证明逐人随访完整。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_strict_index_labels_v1 AS
WITH strict_ordered AS MATERIALIZED (
  SELECT c.subject_id,c.hadm_id,c.admittime,c.dischtime,c.admission_location,
         lag(c.dischtime) OVER (
           PARTITION BY c.subject_id ORDER BY c.admittime,c.hadm_id
         ) AS previous_strict_dischtime
  FROM project_aecopd.copd_coded_admissions_v1_gate c
  WHERE c.is_strict_aecopd
), strict_events AS MATERIALIZED (
  SELECT *,
         COALESCE((admission_location='TRANSFER FROM HOSP/EXTRAM'
          AND admittime>=previous_strict_dischtime
          AND admittime<=previous_strict_dischtime+interval '2 days'),false)
           AS is_transfer_continuation
  FROM strict_ordered
), index_events AS MATERIALIZED (
  SELECT c.*,s.is_transfer_continuation
  FROM project_aecopd.copd_coded_admissions_v1_gate c
  JOIN strict_events s USING(hadm_id)
  WHERE c.at_or_after_first_qualification
    AND c.discharge_landmark_eligible
    AND NOT s.is_transfer_continuation
), event_times AS (
  SELECT a.subject_id,a.hadm_id,a.source_dataset,a.admittime,a.dischtime,
         a.best_copd_seq_num,a.coded_admission_order,
         n.hadm_id AS next_observed_aecopd_hadm_id,
         n.admittime AS next_observed_aecopd_admittime,
         extract(epoch FROM (n.admittime-a.dischtime))/86400.0
           AS next_observed_aecopd_days,
         CASE WHEN p.dod::date>=a.dischtime::date
              THEN (p.dod::date-a.dischtime::date)::double precision END
           AS observed_death_days
  FROM index_events a
  JOIN public.patients p USING(subject_id)
  LEFT JOIN LATERAL (
    SELECT b.hadm_id,b.admittime
    FROM strict_events b
    WHERE b.subject_id=a.subject_id
      AND (b.admittime,b.hadm_id)>(a.admittime,a.hadm_id)
      AND b.admittime>=a.dischtime
      AND NOT b.is_transfer_continuation
    ORDER BY b.admittime,b.hadm_id LIMIT 1
  ) n ON true
), labels AS (
  SELECT *, CASE
    WHEN next_observed_aecopd_days<=365
      AND (observed_death_days IS NULL
           OR next_observed_aecopd_days<=observed_death_days) THEN 1
    WHEN observed_death_days<=365
      AND (next_observed_aecopd_days IS NULL
           OR observed_death_days<next_observed_aecopd_days) THEN 2
    ELSE 0 END AS event_type
  FROM event_times
)
SELECT subject_id,hadm_id,source_dataset,admittime,dischtime,
       best_copd_seq_num,coded_admission_order,
       next_observed_aecopd_hadm_id,next_observed_aecopd_admittime,
       next_observed_aecopd_days,observed_death_days,event_type,
       CASE event_type WHEN 1 THEN 'AECOPD_READMISSION'
         WHEN 2 THEN 'ALL_CAUSE_DEATH' ELSE 'CENSORED_365D' END AS event_name,
       CASE event_type WHEN 1 THEN next_observed_aecopd_days
         WHEN 2 THEN observed_death_days ELSE 365.0 END AS time_days,
       'World A: database-observed readmission; 365d administrative censoring; individual follow-up completeness unverified'::text
         AS followup_assumption
FROM labels;

ALTER TABLE project_aecopd.copd_coded_strict_index_labels_v1 ADD PRIMARY KEY(hadm_id);
CREATE INDEX copd_coded_strict_index_labels_v1_subject_idx
  ON project_aecopd.copd_coded_strict_index_labels_v1(subject_id,admittime,hadm_id);

DO $$ BEGIN
  IF EXISTS (
    SELECT 1 FROM project_aecopd.copd_coded_strict_index_labels_v1 l
    LEFT JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
    WHERE c.hadm_id IS NULL OR NOT c.is_strict_aecopd
      OR NOT c.at_or_after_first_qualification
      OR NOT c.discharge_landmark_eligible
      OR l.event_type NOT IN (0,1,2)
      OR l.time_days<0 OR l.time_days>365
  ) THEN
    RAISE EXCEPTION 'MIMIC-III strict index label consistency check failed';
  END IF;
END $$;
COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08g_copd_coded_strict_index_labels_mimiciv.sql · lines 1–89

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08g_copd_coded_strict_index_labels_mimiciv.sql","part":1,"start_line":1,"end_line":89,"source_sha256":"0a8fea3779b0a66a0f992e516e92a9b77c32477b2b2b4675bdeecce175161623","fragment_sha256":"0a8fea3779b0a66a0f992e516e92a9b77c32477b2b2b4675bdeecce175161623","added_terminal_newline":false} -->
````sql
-- MIMIC-IV：v1 患者门槛和本次 COPD 编码过滤后的严格 AECOPD 出院索引。
-- 365 天是可观察结局的行政窗口，不是固定时间段分类头，也不证明逐人随访完整。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_strict_index_labels_v1 AS
WITH strict_ordered AS MATERIALIZED (
  SELECT c.subject_id,c.hadm_id,c.admittime,c.dischtime,c.admission_location,
         lag(c.dischtime) OVER (
           PARTITION BY c.subject_id ORDER BY c.admittime,c.hadm_id
         ) AS previous_strict_dischtime
  FROM project_aecopd.copd_coded_admissions_v1_gate c
  WHERE c.is_strict_aecopd
), strict_events AS MATERIALIZED (
  SELECT *,
         COALESCE((admission_location='TRANSFER FROM HOSPITAL'
          AND admittime>=previous_strict_dischtime
          AND admittime<=previous_strict_dischtime+interval '2 days'),false)
           AS is_transfer_continuation
  FROM strict_ordered
), index_events AS MATERIALIZED (
  SELECT c.*,s.is_transfer_continuation
  FROM project_aecopd.copd_coded_admissions_v1_gate c
  JOIN strict_events s USING(hadm_id)
  WHERE c.at_or_after_first_qualification
    AND c.discharge_landmark_eligible
    AND NOT s.is_transfer_continuation
), event_times AS (
  SELECT a.subject_id,a.hadm_id,a.source_dataset,a.admittime,a.dischtime,
         a.best_copd_seq_num,a.coded_admission_order,
         n.hadm_id AS next_observed_aecopd_hadm_id,
         n.admittime AS next_observed_aecopd_admittime,
         extract(epoch FROM (n.admittime-a.dischtime))/86400.0
           AS next_observed_aecopd_days,
         CASE WHEN p.dod>=a.dischtime::date
              THEN (p.dod-a.dischtime::date)::double precision END
           AS observed_death_days
  FROM index_events a
  JOIN mimiciv_hosp.patients p USING(subject_id)
  LEFT JOIN LATERAL (
    SELECT b.hadm_id,b.admittime
    FROM strict_events b
    WHERE b.subject_id=a.subject_id
      AND (b.admittime,b.hadm_id)>(a.admittime,a.hadm_id)
      AND b.admittime>=a.dischtime
      AND NOT b.is_transfer_continuation
    ORDER BY b.admittime,b.hadm_id LIMIT 1
  ) n ON true
), labels AS (
  SELECT *, CASE
    WHEN next_observed_aecopd_days<=365
      AND (observed_death_days IS NULL
           OR next_observed_aecopd_days<=observed_death_days) THEN 1
    WHEN observed_death_days<=365
      AND (next_observed_aecopd_days IS NULL
           OR observed_death_days<next_observed_aecopd_days) THEN 2
    ELSE 0 END AS event_type
  FROM event_times
)
SELECT subject_id,hadm_id,source_dataset,admittime,dischtime,
       best_copd_seq_num,coded_admission_order,
       next_observed_aecopd_hadm_id,next_observed_aecopd_admittime,
       next_observed_aecopd_days,observed_death_days,event_type,
       CASE event_type WHEN 1 THEN 'AECOPD_READMISSION'
         WHEN 2 THEN 'ALL_CAUSE_DEATH' ELSE 'CENSORED_365D' END AS event_name,
       CASE event_type WHEN 1 THEN next_observed_aecopd_days
         WHEN 2 THEN observed_death_days ELSE 365.0 END AS time_days,
       'World A: database-observed readmission; 365d administrative censoring; individual follow-up completeness unverified'::text
         AS followup_assumption
FROM labels;

ALTER TABLE project_aecopd.copd_coded_strict_index_labels_v1 ADD PRIMARY KEY(hadm_id);
CREATE INDEX copd_coded_strict_index_labels_v1_subject_idx
  ON project_aecopd.copd_coded_strict_index_labels_v1(subject_id,admittime,hadm_id);

DO $$ BEGIN
  IF EXISTS (
    SELECT 1 FROM project_aecopd.copd_coded_strict_index_labels_v1 l
    LEFT JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
    WHERE c.hadm_id IS NULL OR NOT c.is_strict_aecopd
      OR NOT c.at_or_after_first_qualification
      OR NOT c.discharge_landmark_eligible
      OR l.event_type NOT IN (0,1,2)
      OR l.time_days<0 OR l.time_days>365
  ) THEN
    RAISE EXCEPTION 'MIMIC-IV strict index label consistency check failed';
  END IF;
END $$;
COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08h_copd_coded_model_inputs_mimiciii.sql · lines 1–58

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08h_copd_coded_model_inputs_mimiciii.sql","part":1,"start_line":1,"end_line":58,"source_sha256":"ac72fea12f398ae5f29f7d59892ad4a078bbecf7d5d3904c0b7ec141abd3f5c9","fragment_sha256":"ac72fea12f398ae5f29f7d59892ad4a078bbecf7d5d3904c0b7ec141abd3f5c9","added_terminal_newline":false} -->
````sql
-- MIMIC-III：与 IV 相同的低维结构化输入；LABEVENTS 无 storetime，
-- 只能按 charttime 限制，不能证明每个检验结果已在出院前写入。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_model_base_v1 AS
SELECT l.subject_id,l.hadm_id,l.source_dataset,l.event_type,l.time_days,
       LEAST(90,GREATEST(0,extract(epoch FROM (l.admittime-p.dob))/31557600.0))::double precision
         AS age_at_admission,
       CASE p.gender WHEN 'M' THEN 1 WHEN 'F' THEN 0 END AS male,
       c.prior_coded_admission_count,
       (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate h
        WHERE h.subject_id=l.subject_id AND h.is_strict_aecopd
          AND h.dischtime<=l.admittime) AS prior_strict_aecopd_count,
       m.has_timed_discharge_summary,m.has_timed_other_note
FROM project_aecopd.copd_coded_strict_index_labels_v1 l
JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
JOIN public.patients p ON p.subject_id=l.subject_id
JOIN project_aecopd.copd_coded_dataset_manifest m USING(hadm_id);

ALTER TABLE project_aecopd.copd_coded_model_base_v1 ADD PRIMARY KEY(hadm_id);

CREATE TABLE project_aecopd.copd_coded_model_labs_v1 AS
WITH selected AS MATERIALIZED (
  SELECT l.hadm_id,x.itemid,x.charttime,x.row_id,x.valuenum
  FROM project_aecopd.copd_coded_strict_index_labels_v1 l
  JOIN public.labevents x ON x.hadm_id=l.hadm_id
  WHERE x.itemid IN (50818,50821,51301,51222,50912)
    AND x.charttime BETWEEN l.admittime AND l.dischtime
    AND x.valuenum IS NOT NULL
    AND x.valuenum>=0
), ranked AS (
  SELECT *,
         row_number() OVER (PARTITION BY hadm_id,itemid ORDER BY charttime,row_id)
           AS first_rank,
         row_number() OVER (PARTITION BY hadm_id,itemid ORDER BY charttime DESC,row_id DESC)
           AS last_rank
  FROM selected
)
SELECT hadm_id,itemid,count(*) AS measurement_count,
       max(valuenum) FILTER (WHERE first_rank=1) AS first_value,
       max(valuenum) FILTER (WHERE last_rank=1) AS last_value,
       min(valuenum) AS min_value,max(valuenum) AS max_value
FROM ranked GROUP BY hadm_id,itemid;

ALTER TABLE project_aecopd.copd_coded_model_labs_v1 ADD PRIMARY KEY(hadm_id,itemid);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_model_base_v1)
     <> (SELECT count(*) FROM project_aecopd.copd_coded_strict_index_labels_v1)
     OR EXISTS (SELECT 1 FROM project_aecopd.copd_coded_model_base_v1
                WHERE age_at_admission IS NULL OR male IS NULL
                   OR prior_coded_admission_count<0 OR prior_strict_aecopd_count<0)
     OR EXISTS (SELECT 1 FROM project_aecopd.copd_coded_model_labs_v1
                WHERE measurement_count<1 OR first_value IS NULL OR last_value IS NULL)
  THEN RAISE EXCEPTION 'MIMIC-III model input consistency check failed'; END IF;
END $$;
COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08h_copd_coded_model_inputs_mimiciv.sql · lines 1–59

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08h_copd_coded_model_inputs_mimiciv.sql","part":1,"start_line":1,"end_line":59,"source_sha256":"88d5ab4bee12c6adc0908739f579be5b848de602366afbd3607be0fb01ecf9cb","fragment_sha256":"88d5ab4bee12c6adc0908739f579be5b848de602366afbd3607be0fb01ecf9cb","added_terminal_newline":false} -->
````sql
-- MIMIC-IV：新队列低维结构化训练输入。只使用索引出院前已可用的检验结果。
-- 处方、诊断细分类及文本留给独立增量实验，避免将可用性清单直接当作特征。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_model_base_v1 AS
SELECT l.subject_id,l.hadm_id,l.source_dataset,l.event_type,l.time_days,
       LEAST(90,GREATEST(0,p.anchor_age+extract(year FROM l.admittime)-p.anchor_year))::double precision
         AS age_at_admission,
       CASE p.gender WHEN 'M' THEN 1 WHEN 'F' THEN 0 END AS male,
       c.prior_coded_admission_count,
       (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate h
        WHERE h.subject_id=l.subject_id AND h.is_strict_aecopd
          AND h.dischtime<=l.admittime) AS prior_strict_aecopd_count,
       m.has_timed_discharge_summary,m.has_timed_other_note
FROM project_aecopd.copd_coded_strict_index_labels_v1 l
JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
JOIN mimiciv_hosp.patients p ON p.subject_id=l.subject_id
JOIN project_aecopd.copd_coded_dataset_manifest m USING(hadm_id);

ALTER TABLE project_aecopd.copd_coded_model_base_v1 ADD PRIMARY KEY(hadm_id);

CREATE TABLE project_aecopd.copd_coded_model_labs_v1 AS
WITH selected AS MATERIALIZED (
  SELECT l.hadm_id,x.itemid,x.charttime,x.labevent_id,x.valuenum
  FROM project_aecopd.copd_coded_strict_index_labels_v1 l
  JOIN mimiciv_hosp.labevents x ON x.hadm_id=l.hadm_id
  WHERE x.itemid IN (50818,50821,51301,51222,50912)
    AND x.charttime BETWEEN l.admittime AND l.dischtime
    AND x.storetime<=l.dischtime
    AND x.valuenum IS NOT NULL
    AND x.valuenum>=0
), ranked AS (
  SELECT *,
         row_number() OVER (PARTITION BY hadm_id,itemid ORDER BY charttime,labevent_id)
           AS first_rank,
         row_number() OVER (PARTITION BY hadm_id,itemid ORDER BY charttime DESC,labevent_id DESC)
           AS last_rank
  FROM selected
)
SELECT hadm_id,itemid,count(*) AS measurement_count,
       max(valuenum) FILTER (WHERE first_rank=1) AS first_value,
       max(valuenum) FILTER (WHERE last_rank=1) AS last_value,
       min(valuenum) AS min_value,max(valuenum) AS max_value
FROM ranked GROUP BY hadm_id,itemid;

ALTER TABLE project_aecopd.copd_coded_model_labs_v1 ADD PRIMARY KEY(hadm_id,itemid);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_model_base_v1)
     <> (SELECT count(*) FROM project_aecopd.copd_coded_strict_index_labels_v1)
     OR EXISTS (SELECT 1 FROM project_aecopd.copd_coded_model_base_v1
                WHERE age_at_admission IS NULL OR male IS NULL
                   OR prior_coded_admission_count<0 OR prior_strict_aecopd_count<0)
     OR EXISTS (SELECT 1 FROM project_aecopd.copd_coded_model_labs_v1
                WHERE measurement_count<1 OR first_value IS NULL OR last_value IS NULL)
  THEN RAISE EXCEPTION 'MIMIC-IV model input consistency check failed'; END IF;
END $$;
COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08i_copd_coded_all_index_labels_mimiciii.sql · lines 1–88

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08i_copd_coded_all_index_labels_mimiciii.sql","part":1,"start_line":1,"end_line":88,"source_sha256":"3779e4da2c9be6ca54244798e7e18df204f370d8aea6fe865d9dfffe5921fc31","fragment_sha256":"3779e4da2c9be6ca54244798e7e18df204f370d8aea6fe865d9dfffe5921fc31","added_terminal_newline":false} -->
````sql
-- MIMIC-III：同一 v1 患者门槛下，每次存活且正时长的 COPD 编码住院出院均可作为预测起点。
-- 包含首次前三位编码资格出现之前的 COPD 编码住院；显式保留该标记供选择偏倚审计。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_all_index_labels_v2 AS
WITH strict_ordered AS MATERIALIZED (
  SELECT c.subject_id,c.hadm_id,c.admittime,c.dischtime,c.admission_location,
         lag(c.dischtime) OVER (
           PARTITION BY c.subject_id ORDER BY c.admittime,c.hadm_id
         ) AS previous_strict_dischtime
  FROM project_aecopd.copd_coded_admissions_v1_gate c
  WHERE c.is_strict_aecopd
), strict_events AS MATERIALIZED (
  SELECT *,
         COALESCE((admission_location='TRANSFER FROM HOSP/EXTRAM'
          AND admittime>=previous_strict_dischtime
          AND admittime<=previous_strict_dischtime+interval '2 days'),false)
           AS is_transfer_continuation
  FROM strict_ordered
), event_times AS (
  SELECT a.subject_id,a.hadm_id,a.source_dataset,a.admittime,a.dischtime,
         a.best_copd_seq_num,a.coded_admission_order,
         a.is_strict_aecopd AS index_is_strict_aecopd,
         a.at_or_after_first_qualification,
         n.hadm_id AS next_observed_aecopd_hadm_id,
         n.admittime AS next_observed_aecopd_admittime,
         extract(epoch FROM (n.admittime-a.dischtime))/86400.0
           AS next_observed_aecopd_days,
         CASE WHEN p.dod::date>=a.dischtime::date
              THEN (p.dod::date-a.dischtime::date)::double precision END
           AS observed_death_days
  FROM project_aecopd.copd_coded_admissions_v1_gate a
  JOIN public.patients p USING(subject_id)
  LEFT JOIN LATERAL (
    SELECT b.hadm_id,b.admittime
    FROM strict_events b
    WHERE b.subject_id=a.subject_id
      AND (b.admittime,b.hadm_id)>(a.admittime,a.hadm_id)
      AND b.admittime>=a.dischtime
      AND NOT b.is_transfer_continuation
    ORDER BY b.admittime,b.hadm_id LIMIT 1
  ) n ON true
  WHERE a.discharge_landmark_eligible
), labels AS (
  SELECT *, CASE
    WHEN next_observed_aecopd_days<=365
      AND (observed_death_days IS NULL
           OR next_observed_aecopd_days<=observed_death_days) THEN 1
    WHEN observed_death_days<=365
      AND (next_observed_aecopd_days IS NULL
           OR observed_death_days<next_observed_aecopd_days) THEN 2
    ELSE 0 END AS event_type
  FROM event_times
)
SELECT subject_id,hadm_id,source_dataset,admittime,dischtime,
       best_copd_seq_num,coded_admission_order,index_is_strict_aecopd,
       at_or_after_first_qualification,
       next_observed_aecopd_hadm_id,next_observed_aecopd_admittime,
       next_observed_aecopd_days,observed_death_days,event_type,
       CASE event_type WHEN 1 THEN 'AECOPD_READMISSION'
         WHEN 2 THEN 'ALL_CAUSE_DEATH' ELSE 'CENSORED_365D' END AS event_name,
       CASE event_type WHEN 1 THEN next_observed_aecopd_days
         WHEN 2 THEN observed_death_days ELSE 365.0 END AS time_days,
       'World A: database-observed readmission; 365d administrative censoring; individual follow-up completeness unverified'::text
         AS followup_assumption
FROM labels;

ALTER TABLE project_aecopd.copd_coded_all_index_labels_v2 ADD PRIMARY KEY(hadm_id);
CREATE INDEX copd_coded_all_index_labels_v2_subject_idx
  ON project_aecopd.copd_coded_all_index_labels_v2(subject_id,admittime,hadm_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_all_index_labels_v2)
     <> (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate
         WHERE discharge_landmark_eligible)
     OR EXISTS (
       SELECT 1 FROM project_aecopd.copd_coded_all_index_labels_v2 l
       LEFT JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
       WHERE c.hadm_id IS NULL OR NOT c.discharge_landmark_eligible
         OR l.coded_admission_order<>c.coded_admission_order
         OR l.event_type NOT IN (0,1,2)
         OR l.time_days<0 OR l.time_days>365
     ) THEN
    RAISE EXCEPTION 'MIMIC-III all-coded index label consistency check failed';
  END IF;
END $$;
COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08i_copd_coded_all_index_labels_mimiciv.sql · lines 1–89

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08i_copd_coded_all_index_labels_mimiciv.sql","part":1,"start_line":1,"end_line":89,"source_sha256":"f59e435f09a2d2f292ac02d956de862cb0be231806dfce06bc9a4e8db00a4ab9","fragment_sha256":"f59e435f09a2d2f292ac02d956de862cb0be231806dfce06bc9a4e8db00a4ab9","added_terminal_newline":false} -->
````sql
-- MIMIC-IV：v1 患者门槛不变；每次存活且正时长的 COPD 编码住院出院均可作为预测起点。
-- 包含首次前三位编码资格出现之前的 COPD 编码住院；显式保留该标记供选择偏倚审计。
-- 结局仍为下一次非转院续接的严格 AECOPD 再入院，死亡为竞争事件。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_all_index_labels_v2 AS
WITH strict_ordered AS MATERIALIZED (
  SELECT c.subject_id,c.hadm_id,c.admittime,c.dischtime,c.admission_location,
         lag(c.dischtime) OVER (
           PARTITION BY c.subject_id ORDER BY c.admittime,c.hadm_id
         ) AS previous_strict_dischtime
  FROM project_aecopd.copd_coded_admissions_v1_gate c
  WHERE c.is_strict_aecopd
), strict_events AS MATERIALIZED (
  SELECT *,
         COALESCE((admission_location='TRANSFER FROM HOSPITAL'
          AND admittime>=previous_strict_dischtime
          AND admittime<=previous_strict_dischtime+interval '2 days'),false)
           AS is_transfer_continuation
  FROM strict_ordered
), event_times AS (
  SELECT a.subject_id,a.hadm_id,a.source_dataset,a.admittime,a.dischtime,
         a.best_copd_seq_num,a.coded_admission_order,
         a.is_strict_aecopd AS index_is_strict_aecopd,
         a.at_or_after_first_qualification,
         n.hadm_id AS next_observed_aecopd_hadm_id,
         n.admittime AS next_observed_aecopd_admittime,
         extract(epoch FROM (n.admittime-a.dischtime))/86400.0
           AS next_observed_aecopd_days,
         CASE WHEN p.dod>=a.dischtime::date
              THEN (p.dod-a.dischtime::date)::double precision END
           AS observed_death_days
  FROM project_aecopd.copd_coded_admissions_v1_gate a
  JOIN mimiciv_hosp.patients p USING(subject_id)
  LEFT JOIN LATERAL (
    SELECT b.hadm_id,b.admittime
    FROM strict_events b
    WHERE b.subject_id=a.subject_id
      AND (b.admittime,b.hadm_id)>(a.admittime,a.hadm_id)
      AND b.admittime>=a.dischtime
      AND NOT b.is_transfer_continuation
    ORDER BY b.admittime,b.hadm_id LIMIT 1
  ) n ON true
  WHERE a.discharge_landmark_eligible
), labels AS (
  SELECT *, CASE
    WHEN next_observed_aecopd_days<=365
      AND (observed_death_days IS NULL
           OR next_observed_aecopd_days<=observed_death_days) THEN 1
    WHEN observed_death_days<=365
      AND (next_observed_aecopd_days IS NULL
           OR observed_death_days<next_observed_aecopd_days) THEN 2
    ELSE 0 END AS event_type
  FROM event_times
)
SELECT subject_id,hadm_id,source_dataset,admittime,dischtime,
       best_copd_seq_num,coded_admission_order,index_is_strict_aecopd,
       at_or_after_first_qualification,
       next_observed_aecopd_hadm_id,next_observed_aecopd_admittime,
       next_observed_aecopd_days,observed_death_days,event_type,
       CASE event_type WHEN 1 THEN 'AECOPD_READMISSION'
         WHEN 2 THEN 'ALL_CAUSE_DEATH' ELSE 'CENSORED_365D' END AS event_name,
       CASE event_type WHEN 1 THEN next_observed_aecopd_days
         WHEN 2 THEN observed_death_days ELSE 365.0 END AS time_days,
       'World A: database-observed readmission; 365d administrative censoring; individual follow-up completeness unverified'::text
         AS followup_assumption
FROM labels;

ALTER TABLE project_aecopd.copd_coded_all_index_labels_v2 ADD PRIMARY KEY(hadm_id);
CREATE INDEX copd_coded_all_index_labels_v2_subject_idx
  ON project_aecopd.copd_coded_all_index_labels_v2(subject_id,admittime,hadm_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_all_index_labels_v2)
     <> (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate
         WHERE discharge_landmark_eligible)
     OR EXISTS (
       SELECT 1 FROM project_aecopd.copd_coded_all_index_labels_v2 l
       LEFT JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
       WHERE c.hadm_id IS NULL OR NOT c.discharge_landmark_eligible
         OR l.coded_admission_order<>c.coded_admission_order
         OR l.event_type NOT IN (0,1,2)
         OR l.time_days<0 OR l.time_days>365
     ) THEN
    RAISE EXCEPTION 'MIMIC-IV all-coded index label consistency check failed';
  END IF;
END $$;
COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08j_copd_coded_first_index_labels_mimiciii.sql · lines 1–28

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08j_copd_coded_first_index_labels_mimiciii.sql","part":1,"start_line":1,"end_line":28,"source_sha256":"b30d170530ed205ee73c625d509ab06656640f1cdb2ba6f6771e1f9ee2a384f8","fragment_sha256":"b30d170530ed205ee73c625d509ab06656640f1cdb2ba6f6771e1f9ee2a384f8","added_terminal_newline":false} -->
````sql
-- MIMIC-III：正式患者级首次起点；其余带 COPD 编码的住院保留在 08c 纵向表。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_first_index_labels_v2 AS
SELECT DISTINCT ON (subject_id) *,
       (coded_admission_order=1) AS is_first_observed_coded_admission
FROM project_aecopd.copd_coded_all_index_labels_v2
ORDER BY subject_id,admittime,hadm_id;

ALTER TABLE project_aecopd.copd_coded_first_index_labels_v2 ADD PRIMARY KEY(hadm_id);
CREATE UNIQUE INDEX copd_coded_first_index_labels_v2_subject_idx
  ON project_aecopd.copd_coded_first_index_labels_v2(subject_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_first_index_labels_v2)
     <> (SELECT count(DISTINCT subject_id)
         FROM project_aecopd.copd_coded_admissions_v1_gate
         WHERE discharge_landmark_eligible)
     OR EXISTS (
       SELECT 1 FROM project_aecopd.copd_coded_first_index_labels_v2 f
       JOIN project_aecopd.copd_coded_all_index_labels_v2 l USING(subject_id)
       WHERE (l.admittime,l.hadm_id)<(f.admittime,f.hadm_id)
     ) THEN
    RAISE EXCEPTION 'MIMIC-III first-coded index consistency check failed';
  END IF;
END $$;
COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/08j_copd_coded_first_index_labels_mimiciv.sql · lines 1–29

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/08j_copd_coded_first_index_labels_mimiciv.sql","part":1,"start_line":1,"end_line":29,"source_sha256":"6db348b0b24e5e38a6d369e1ffd61092dcf0a255fa6eed69081d26b81cfed995","fragment_sha256":"6db348b0b24e5e38a6d369e1ffd61092dcf0a255fa6eed69081d26b81cfed995","added_terminal_newline":false} -->
````sql
-- MIMIC-IV：历史首出院实验子集，不是当前完整纵向训练队列。
-- 从 08i 中每位患者只取最早一次符合条件的可存活出院；08c/08i 的其他住院不删除。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_first_index_labels_v2 AS
SELECT DISTINCT ON (subject_id) *,
       (coded_admission_order=1) AS is_first_observed_coded_admission
FROM project_aecopd.copd_coded_all_index_labels_v2
ORDER BY subject_id,admittime,hadm_id;

ALTER TABLE project_aecopd.copd_coded_first_index_labels_v2 ADD PRIMARY KEY(hadm_id);
CREATE UNIQUE INDEX copd_coded_first_index_labels_v2_subject_idx
  ON project_aecopd.copd_coded_first_index_labels_v2(subject_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_first_index_labels_v2)
     <> (SELECT count(DISTINCT subject_id)
         FROM project_aecopd.copd_coded_admissions_v1_gate
         WHERE discharge_landmark_eligible)
     OR EXISTS (
       SELECT 1 FROM project_aecopd.copd_coded_first_index_labels_v2 f
       JOIN project_aecopd.copd_coded_all_index_labels_v2 l USING(subject_id)
       WHERE (l.admittime,l.hadm_id)<(f.admittime,f.hadm_id)
     ) THEN
    RAISE EXCEPTION 'MIMIC-IV first-coded index consistency check failed';
  END IF;
END $$;
COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/99_verify.sql · lines 1–47

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/99_verify.sql","part":1,"start_line":1,"end_line":47,"source_sha256":"24c74abc423961bdd2785c02f88f6541042e7abc913dc5f391c2c0bee1bb0dc6","fragment_sha256":"24c74abc423961bdd2785c02f88f6541042e7abc913dc5f391c2c0bee1bb0dc6","added_terminal_newline":false} -->
````sql
\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;

DROP TABLE IF EXISTS project_aecopd.verification_results;
CREATE TABLE project_aecopd.verification_results AS
WITH checks AS (
    SELECT 'strict cohort rows' check_name,5950::bigint expected,COUNT(*)::bigint actual FROM project_aecopd.aecopd_admissions_strict
    UNION ALL SELECT 'strict distinct hadm_id',5950,COUNT(DISTINCT hadm_id) FROM project_aecopd.aecopd_admissions_strict
    UNION ALL SELECT 'strict distinct subject_id',3548,COUNT(DISTINCT subject_id) FROM project_aecopd.aecopd_admissions_strict
    UNION ALL SELECT 'longitudinal rows',5950,COUNT(*) FROM project_aecopd.aecopd_longitudinal_history
    UNION ALL SELECT 'note audit rows',5950,COUNT(*) FROM project_aecopd.discharge_note_admission_audit
    UNION ALL SELECT 'any discharge note',4556,COUNT(*) FILTER(WHERE has_discharge_note) FROM project_aecopd.discharge_note_admission_audit
    UNION ALL SELECT 'note available by discharge',717,COUNT(*) FILTER(WHERE has_note_available_by_discharge) FROM project_aecopd.discharge_note_admission_audit
    UNION ALL SELECT 'ICU admission audit rows',5950,COUNT(*) FROM project_aecopd.icu_admission_audit
    UNION ALL SELECT 'admissions with ICU',2146,COUNT(*) FILTER(WHERE has_icu_stay) FROM project_aecopd.icu_admission_audit
    UNION ALL SELECT 'master cohort rows',5950,COUNT(*) FROM project_aecopd.master_cohort
    UNION ALL SELECT 'master distinct hadm_id',5950,COUNT(DISTINCT hadm_id) FROM project_aecopd.master_cohort
    UNION ALL SELECT 'observed next AECOPD within 90d',1040,COUNT(*) FILTER(WHERE next_observed_aecopd_within_90d) FROM project_aecopd.master_cohort
    UNION ALL SELECT 'observed next AECOPD within 180d',1381,COUNT(*) FILTER(WHERE next_observed_aecopd_within_180d) FROM project_aecopd.master_cohort
    UNION ALL SELECT 'observed next AECOPD within 365d',1752,COUNT(*) FILTER(WHERE next_observed_aecopd_within_365d) FROM project_aecopd.master_cohort
    UNION ALL SELECT 'landmark definition rows',5950,COUNT(*) FROM project_aecopd.landmark_definitions
    UNION ALL SELECT 'landmark distinct hadm_id',5950,COUNT(DISTINCT hadm_id) FROM project_aecopd.landmark_definitions
    UNION ALL SELECT 'admission 24h eligible rows',5468,COUNT(*) FILTER(WHERE eligible_admission_24h) FROM project_aecopd.landmark_definitions
    UNION ALL SELECT 'landmark source availability rows',24,COUNT(*) FROM project_aecopd.landmark_source_availability
)
SELECT check_name,expected,actual,(expected=actual) AS passed FROM checks ORDER BY check_name;

COMMIT;

\copy (SELECT * FROM project_aecopd.verification_results ORDER BY check_name) TO 'outputs/99_verification_results.csv' CSV HEADER

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM project_aecopd.verification_results WHERE NOT passed) THEN
    RAISE EXCEPTION 'One or more reproducibility checks failed. Inspect outputs/99_verification_results.csv';
  END IF;
END $$;

SELECT CASE WHEN bool_and(expected=actual) THEN 'ALL_CHECKS_PASSED' ELSE 'CHECK_FAILURE' END AS verification_status
FROM (
  SELECT 5950::bigint expected,COUNT(*)::bigint actual FROM project_aecopd.master_cohort
  UNION ALL SELECT 5950,COUNT(DISTINCT hadm_id) FROM project_aecopd.master_cohort
  UNION ALL SELECT 3548,COUNT(DISTINCT subject_id) FROM project_aecopd.master_cohort
) q;


````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/copd_broad_readmission_labels_mimiciv.sql · lines 1–91

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/copd_broad_readmission_labels_mimiciv.sql","part":1,"start_line":1,"end_line":91,"source_sha256":"344ff26217fd3845e13f90a3da568222c3b9a3d6c3c487be826ffc0219f6acca","fragment_sha256":"344ff26217fd3845e13f90a3da568222c3b9a3d6c3c487be826ffc0219f6acca","added_terminal_newline":false} -->
````sql
-- 每次合格出院后首次再次发生任意清单 COPD／肺气肿编码住院；保留旧严格标签表不变。
-- 数据库只新增独立表；若目标已存在则失败，不自动覆盖。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_broad_readmission_labels_v1 AS
WITH ordered AS MATERIALIZED (
  SELECT c.subject_id,c.hadm_id,c.admittime,c.dischtime,c.admission_location,
         c.best_copd_seq_num,c.is_strict_aecopd,c.has_49122,c.has_j440,
         lag(c.dischtime) OVER (
           PARTITION BY c.subject_id ORDER BY c.admittime,c.hadm_id
         ) AS previous_copd_dischtime
  FROM project_aecopd.copd_coded_admissions_v1_gate c
), candidates AS MATERIALIZED (
  SELECT *,
         COALESCE((admission_location='TRANSFER FROM HOSPITAL'
           AND admittime>=previous_copd_dischtime
           AND admittime<=previous_copd_dischtime+interval '2 days'),false)
           AS is_transfer_continuation
  FROM ordered
), event_times AS (
  SELECT a.subject_id,a.hadm_id,a.source_dataset,a.admittime,a.dischtime,
         a.best_copd_seq_num,a.coded_admission_order,
         a.is_strict_aecopd AS index_is_strict_aecopd,
         a.at_or_after_first_qualification,
         n.hadm_id AS next_copd_hadm_id,
         n.admittime AS next_copd_admittime,
         n.best_copd_seq_num AS next_copd_best_code_seq_num,
         n.is_strict_aecopd AS next_is_strict_aecopd,
         n.has_49122 AS next_has_49122,
         n.has_j440 AS next_has_j440,
         extract(epoch FROM (n.admittime-a.dischtime))/86400.0 AS next_copd_days,
         CASE WHEN p.dod>=a.dischtime::date
              THEN (p.dod-a.dischtime::date)::double precision END
           AS observed_death_days
  FROM project_aecopd.copd_coded_admissions_v1_gate a
  JOIN mimiciv_hosp.patients p USING(subject_id)
  LEFT JOIN LATERAL (
    SELECT b.hadm_id,b.admittime,b.best_copd_seq_num,b.is_strict_aecopd,
           b.has_49122,b.has_j440
    FROM candidates b
    WHERE b.subject_id=a.subject_id
      AND (b.admittime,b.hadm_id)>(a.admittime,a.hadm_id)
      AND b.admittime>=a.dischtime
      AND NOT b.is_transfer_continuation
    ORDER BY b.admittime,b.hadm_id LIMIT 1
  ) n ON true
  WHERE a.discharge_landmark_eligible
), labels AS (
  SELECT *,CASE
    WHEN next_copd_days<=365
      AND (observed_death_days IS NULL OR next_copd_days<=observed_death_days) THEN 1
    WHEN observed_death_days<=365
      AND (next_copd_days IS NULL OR observed_death_days<next_copd_days) THEN 2
    ELSE 0 END AS event_type
  FROM event_times
)
SELECT subject_id,hadm_id,source_dataset,admittime,dischtime,
       best_copd_seq_num,coded_admission_order,index_is_strict_aecopd,
       at_or_after_first_qualification,
       next_copd_hadm_id,next_copd_admittime,next_copd_best_code_seq_num,
       next_is_strict_aecopd,next_has_49122,next_has_j440,
       next_copd_days,observed_death_days,event_type,
       CASE event_type WHEN 1 THEN 'COPD_CODED_READMISSION'
         WHEN 2 THEN 'ALL_CAUSE_DEATH' ELSE 'CENSORED_365D' END AS event_name,
       CASE event_type WHEN 1 THEN next_copd_days
         WHEN 2 THEN observed_death_days ELSE 365.0 END AS time_days,
       'World A: database-observed readmission; 365d administrative censoring; individual follow-up completeness unverified'::text
         AS followup_assumption
FROM labels;

ALTER TABLE project_aecopd.copd_broad_readmission_labels_v1 ADD PRIMARY KEY(hadm_id);
CREATE INDEX copd_broad_readmission_labels_v1_subject_idx
  ON project_aecopd.copd_broad_readmission_labels_v1(subject_id,admittime,hadm_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_broad_readmission_labels_v1)
       <> (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate
           WHERE discharge_landmark_eligible)
     OR EXISTS (
       SELECT 1 FROM project_aecopd.copd_broad_readmission_labels_v1 l
       LEFT JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
       WHERE c.hadm_id IS NULL OR NOT c.discharge_landmark_eligible
         OR l.event_type NOT IN (0,1,2)
         OR l.time_days<0 OR l.time_days>365
         OR (l.event_type=1 AND l.next_copd_hadm_id IS NULL)
     ) THEN
    RAISE EXCEPTION 'Broad COPD-coded readmission label validation failed';
  END IF;
END $$;
COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/copd_qualified_readmission_labels_mimiciv.sql · lines 1–93

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/copd_qualified_readmission_labels_mimiciv.sql","part":1,"start_line":1,"end_line":93,"source_sha256":"e216ad04b0fc995fb2d2fda231df60b12a363f7c1a09b1be95840618f75b46ea","fragment_sha256":"e216ad04b0fc995fb2d2fda231df60b12a363f7c1a09b1be95840618f75b46ea","added_terminal_newline":false} -->
````sql
-- 当前主结局：再次住院任意诊断位有严格 491.21/J44.1，或前三位有既定 COPD／肺气肿编码。
-- 保持既有患者、索引出院及转院续接排除规则，不覆盖宽口径或严格旧标签。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_qualified_readmission_labels_v1 AS
WITH ordered AS MATERIALIZED (
  SELECT c.subject_id,c.hadm_id,c.admittime,c.dischtime,c.admission_location,
         c.best_copd_seq_num,c.is_strict_aecopd,c.has_49122,c.has_j440,
         lag(c.dischtime) OVER (
           PARTITION BY c.subject_id ORDER BY c.admittime,c.hadm_id
         ) AS previous_copd_dischtime
  FROM project_aecopd.copd_coded_admissions_v1_gate c
), candidates AS MATERIALIZED (
  SELECT *,
         COALESCE((admission_location='TRANSFER FROM HOSPITAL'
           AND admittime>=previous_copd_dischtime
           AND admittime<=previous_copd_dischtime+interval '2 days'),false)
           AS is_transfer_continuation
  FROM ordered
), event_times AS (
  SELECT a.subject_id,a.hadm_id,a.source_dataset,a.admittime,a.dischtime,
         a.best_copd_seq_num,a.coded_admission_order,
         a.is_strict_aecopd AS index_is_strict_aecopd,
         a.at_or_after_first_qualification,
         n.hadm_id AS next_qualified_hadm_id,
         n.admittime AS next_qualified_admittime,
         n.best_copd_seq_num AS next_copd_best_code_seq_num,
         n.is_strict_aecopd AS next_is_strict_aecopd,
         n.has_49122 AS next_has_49122,
         n.has_j440 AS next_has_j440,
         extract(epoch FROM (n.admittime-a.dischtime))/86400.0 AS next_qualified_days,
         CASE WHEN p.dod>=a.dischtime::date
              THEN (p.dod-a.dischtime::date)::double precision END
           AS observed_death_days
  FROM project_aecopd.copd_coded_admissions_v1_gate a
  JOIN mimiciv_hosp.patients p USING(subject_id)
  LEFT JOIN LATERAL (
    SELECT b.hadm_id,b.admittime,b.best_copd_seq_num,b.is_strict_aecopd,
           b.has_49122,b.has_j440
    FROM candidates b
    WHERE b.subject_id=a.subject_id
      AND (b.admittime,b.hadm_id)>(a.admittime,a.hadm_id)
      AND b.admittime>=a.dischtime
      AND NOT b.is_transfer_continuation
      AND (b.is_strict_aecopd OR b.best_copd_seq_num<=3)
    ORDER BY b.admittime,b.hadm_id LIMIT 1
  ) n ON true
  WHERE a.discharge_landmark_eligible
), labels AS (
  SELECT *,CASE
    WHEN next_qualified_days<=365
      AND (observed_death_days IS NULL OR next_qualified_days<=observed_death_days) THEN 1
    WHEN observed_death_days<=365
      AND (next_qualified_days IS NULL OR observed_death_days<next_qualified_days) THEN 2
    ELSE 0 END AS event_type
  FROM event_times
)
SELECT subject_id,hadm_id,source_dataset,admittime,dischtime,
       best_copd_seq_num,coded_admission_order,index_is_strict_aecopd,
       at_or_after_first_qualification,
       next_qualified_hadm_id,next_qualified_admittime,next_copd_best_code_seq_num,
       next_is_strict_aecopd,next_has_49122,next_has_j440,
       next_qualified_days,observed_death_days,event_type,
       CASE event_type WHEN 1 THEN 'STRICT_AECOPD_OR_TOP3_COPD_READMISSION'
         WHEN 2 THEN 'ALL_CAUSE_DEATH' ELSE 'CENSORED_365D' END AS event_name,
       CASE event_type WHEN 1 THEN next_qualified_days
         WHEN 2 THEN observed_death_days ELSE 365.0 END AS time_days,
       'World A: database-observed readmission; 365d administrative censoring; individual follow-up completeness unverified'::text
         AS followup_assumption
FROM labels;

ALTER TABLE project_aecopd.copd_qualified_readmission_labels_v1 ADD PRIMARY KEY(hadm_id);
CREATE INDEX copd_qualified_readmission_labels_v1_subject_idx
  ON project_aecopd.copd_qualified_readmission_labels_v1(subject_id,admittime,hadm_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_qualified_readmission_labels_v1)
       <> (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate
           WHERE discharge_landmark_eligible)
     OR EXISTS (
       SELECT 1 FROM project_aecopd.copd_qualified_readmission_labels_v1 l
       LEFT JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
       WHERE c.hadm_id IS NULL OR NOT c.discharge_landmark_eligible
         OR l.event_type NOT IN (0,1,2)
         OR l.time_days<0 OR l.time_days>365
         OR (l.event_type=1 AND (l.next_qualified_hadm_id IS NULL
           OR NOT (l.next_is_strict_aecopd OR l.next_copd_best_code_seq_num<=3)))
     ) THEN
    RAISE EXCEPTION 'Qualified COPD readmission label validation failed';
  END IF;
END $$;
COMMIT;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/diagnostics/d1_followup_window_diagnosis.sql · lines 1–122

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/diagnostics/d1_followup_window_diagnosis.sql","part":1,"start_line":1,"end_line":122,"source_sha256":"8a34a05836d291514447514a596ea3267b1dd366aeb6490dc19c26a28dd91b75","fragment_sha256":"8a34a05836d291514447514a596ea3267b1dd366aeb6490dc19c26a28dd91b75","added_terminal_newline":false} -->
````sql
-- =====================================================================
-- 诊断脚本 D1：随访窗口 / 信息性删失（复核问题 1）
-- 目的：量化"365 天内无事件即判为无事件"这一合同受随访长度影响的程度。
-- 只读：仅 SELECT，不 CREATE/DROP/UPDATE，不写库，不碰测试集选择。
-- 运行：psql -h localhost -U postgres -d <db> -v ON_ERROR_STOP=1 -f d1_followup_window_diagnosis.sql
--   （切到 Claude Code 完全访问模式后可由 Claude 直接执行）
-- 前置表：project_aecopd.master_cohort（含 observed_allcause_span_after_discharge_days、
--   days_to_next_observed_aecopd、dod、dischtime）、project_aecopd.model_v1_cohort（data_split、
--   training_eligible）、project_aecopd.model_v1_labels（event_type）。
-- 说明：observed_allcause_span_after_discharge_days = 患者最后一次全因出院距本次出院天数，
--   是"数据库里这次出院之后还能观察多久"的近似上界；它不是经过验证的删失时间，但
--   足以暴露"看不到 = 判为无事件"的偏倚方向。
-- 注意：CTE(WITH) 只作用于紧随其后的单条语句，故每段各自内联一份相同的 cohort 定义。
-- =====================================================================
\pset pager off
SET client_encoding = 'UTF8';

-- -------- 1) 整体：被判为 event_type=0（365 天内无事件）者的随访窗口分布 --------
WITH cohort AS (
  SELECT
    m.subject_id, m.hadm_id, c.data_split,
    m.days_to_next_observed_aecopd AS aecopd_gap,
    m.observed_allcause_span_after_discharge_days AS obs_span,
    l.event_type,
    CASE WHEN m.dod IS NOT NULL AND m.dod >= m.dischtime::date
         THEN (m.dod - m.dischtime::date)::double precision END AS death_gap
  FROM project_aecopd.master_cohort m
  JOIN project_aecopd.model_v1_cohort c USING (subject_id, hadm_id)
  JOIN project_aecopd.model_v1_labels l USING (hadm_id)
  WHERE c.training_eligible
)
SELECT '1_censored_obs_span_distribution' AS section,
  data_split,
  count(*)                                              AS censored_n,
  count(*) FILTER (WHERE obs_span < 30)                 AS span_lt_30d,
  count(*) FILTER (WHERE obs_span < 90)                 AS span_lt_90d,
  count(*) FILTER (WHERE obs_span < 365)                AS span_lt_365d,
  round(100.0*count(*) FILTER (WHERE obs_span < 365)/nullif(count(*),0),2) AS pct_span_lt_365d,
  round(percentile_cont(0.25) WITHIN GROUP (ORDER BY obs_span)::numeric,1) AS obs_span_p25,
  round(percentile_cont(0.50) WITHIN GROUP (ORDER BY obs_span)::numeric,1) AS obs_span_median,
  round(percentile_cont(0.75) WITHIN GROUP (ORDER BY obs_span)::numeric,1) AS obs_span_p75
FROM cohort
WHERE event_type = 0
GROUP BY data_split
ORDER BY data_split;

-- -------- 2) 关键偏倚指标：判为"无事件"但随访不足对应视界的比例 --------
-- 这些是"其实没看满、却被当成阴性"的住院；比例越高，标签越不可信。
WITH cohort AS (
  SELECT
    m.subject_id, m.hadm_id, c.data_split,
    m.days_to_next_observed_aecopd AS aecopd_gap,
    m.observed_allcause_span_after_discharge_days AS obs_span,
    l.event_type,
    CASE WHEN m.dod IS NOT NULL AND m.dod >= m.dischtime::date
         THEN (m.dod - m.dischtime::date)::double precision END AS death_gap
  FROM project_aecopd.master_cohort m
  JOIN project_aecopd.model_v1_cohort c USING (subject_id, hadm_id)
  JOIN project_aecopd.model_v1_labels l USING (hadm_id)
  WHERE c.training_eligible
)
SELECT '2_insufficient_followup_by_horizon' AS section,
  data_split,
  count(*) FILTER (WHERE event_type=0)                                    AS censored_n,
  count(*) FILTER (WHERE event_type=0 AND obs_span < 90)                  AS censored_but_span_lt_90d,
  count(*) FILTER (WHERE event_type=0 AND obs_span < 365)                 AS censored_but_span_lt_365d,
  round(100.0*count(*) FILTER (WHERE event_type=0 AND obs_span < 365)
        /nullif(count(*) FILTER (WHERE event_type=0),0),2)               AS pct_insufficient_365d
FROM cohort
GROUP BY data_split
ORDER BY data_split;

-- -------- 3) 开放尾部（第 365 天仍无事件者）的随访窗口分布 --------
-- 这正是尾部 none_observed 最可能与"入院年份晚、可观察短"混杂的地方。
WITH cohort AS (
  SELECT
    m.subject_id, m.hadm_id, c.data_split,
    m.days_to_next_observed_aecopd AS aecopd_gap,
    m.observed_allcause_span_after_discharge_days AS obs_span,
    l.event_type,
    CASE WHEN m.dod IS NOT NULL AND m.dod >= m.dischtime::date
         THEN (m.dod - m.dischtime::date)::double precision END AS death_gap
  FROM project_aecopd.master_cohort m
  JOIN project_aecopd.model_v1_cohort c USING (subject_id, hadm_id)
  JOIN project_aecopd.model_v1_labels l USING (hadm_id)
  WHERE c.training_eligible
)
SELECT '3_open_tail_obs_span' AS section,
  data_split,
  count(*) FILTER (WHERE event_type=0)                        AS tail_risk_n,
  count(*) FILTER (WHERE event_type=0 AND obs_span < 365)     AS tail_span_lt_365d,
  count(*) FILTER (WHERE event_type=0 AND obs_span < 730)     AS tail_span_lt_730d,
  round(percentile_cont(0.50) WITHIN GROUP (ORDER BY obs_span)
        FILTER (WHERE event_type=0)::numeric,1)               AS tail_obs_span_median
FROM cohort
GROUP BY data_split
ORDER BY data_split;

-- -------- 4) 死亡确认口径：dod 覆盖与出院后死亡可见性 --------
-- 若 dod 只覆盖到某历法边界，晚期死亡会被系统性漏记，进一步放大问题 1。
WITH cohort AS (
  SELECT
    m.subject_id, m.hadm_id, c.data_split,
    m.days_to_next_observed_aecopd AS aecopd_gap,
    m.observed_allcause_span_after_discharge_days AS obs_span,
    l.event_type,
    CASE WHEN m.dod IS NOT NULL AND m.dod >= m.dischtime::date
         THEN (m.dod - m.dischtime::date)::double precision END AS death_gap
  FROM project_aecopd.master_cohort m
  JOIN project_aecopd.model_v1_cohort c USING (subject_id, hadm_id)
  JOIN project_aecopd.model_v1_labels l USING (hadm_id)
  WHERE c.training_eligible
)
SELECT '4_death_ascertainment' AS section,
  data_split,
  count(*)                                                     AS n,
  count(*) FILTER (WHERE death_gap IS NOT NULL)                AS has_any_death_date,
  count(*) FILTER (WHERE death_gap IS NOT NULL AND death_gap<=365) AS death_within_365d,
  count(*) FILTER (WHERE death_gap IS NOT NULL AND death_gap>365)  AS death_after_365d
FROM cohort
GROUP BY data_split
ORDER BY data_split;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/diagnostics/d2_principal_diagnosis_purity.sql · lines 1–78

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/diagnostics/d2_principal_diagnosis_purity.sql","part":1,"start_line":1,"end_line":78,"source_sha256":"383a87a95c691a102409c98af0d4ef3d7c4491937ebd18907ba0fa2115f8028e","fragment_sha256":"383a87a95c691a102409c98af0d4ef3d7c4491937ebd18907ba0fa2115f8028e","added_terminal_newline":false} -->
````sql
-- =====================================================================
-- 诊断脚本 D2：表型纯度 / 主诊断（复核问题 2）
-- 目的：strict_any 表型只要求诊断列表任意位置出现 J441/49121。用已存在的
--   best_seq_num（该住院 AECOPD 码的最小诊断序号）量化 AECOPD 究竟多常是主诊断，
--   以及"如果改为主诊断入组"队列会缩小多少。这决定入组与"再入院"结局的噪声大小。
-- 只读：仅 SELECT。运行方式同 D1。
-- 前置表：project_aecopd.aecopd_admissions_strict(best_seq_num) 或
--   project_aecopd.master_cohort(best_seq_num)。此处用 master_cohort 以对齐建模口径。
-- 注意：MIMIC-IV diagnoses_icd 中 seq_num=1 通常为主要诊断（principal diagnosis）。
--   CTE(WITH) 只作用于紧随其后的单条语句，故每段各自内联一份相同的 cohort 定义。
-- =====================================================================
\pset pager off
SET client_encoding = 'UTF8';

-- -------- 1) AECOPD 诊断序号分布：多少是主诊断(=1)、前三位、更靠后 --------
WITH cohort AS (
  SELECT
    m.subject_id, m.hadm_id, c.data_split, m.best_seq_num,
    l.event_type,
    m.days_to_next_observed_aecopd AS aecopd_gap
  FROM project_aecopd.master_cohort m
  JOIN project_aecopd.model_v1_cohort c USING (subject_id, hadm_id)
  JOIN project_aecopd.model_v1_labels l USING (hadm_id)
  WHERE c.training_eligible
)
SELECT '1_best_seq_num_distribution' AS section,
  data_split,
  count(*)                                          AS n,
  count(*) FILTER (WHERE best_seq_num = 1)          AS principal_dx_n,
  round(100.0*count(*) FILTER (WHERE best_seq_num = 1)/nullif(count(*),0),2) AS pct_principal,
  count(*) FILTER (WHERE best_seq_num BETWEEN 1 AND 3) AS top3_n,
  round(100.0*count(*) FILTER (WHERE best_seq_num BETWEEN 1 AND 3)/nullif(count(*),0),2) AS pct_top3,
  count(*) FILTER (WHERE best_seq_num > 5)          AS beyond_5_n,
  max(best_seq_num)                                 AS max_seq
FROM cohort
GROUP BY data_split
ORDER BY data_split;

-- -------- 2) 敏感性：若把入组收紧为"AECOPD 为主诊断(seq=1)"，队列规模变化 --------
WITH cohort AS (
  SELECT
    m.subject_id, m.hadm_id, c.data_split, m.best_seq_num,
    l.event_type,
    m.days_to_next_observed_aecopd AS aecopd_gap
  FROM project_aecopd.master_cohort m
  JOIN project_aecopd.model_v1_cohort c USING (subject_id, hadm_id)
  JOIN project_aecopd.model_v1_labels l USING (hadm_id)
  WHERE c.training_eligible
)
SELECT '2_cohort_shrink_if_principal_only' AS section,
  count(*)                                          AS strict_any_admissions,
  count(*) FILTER (WHERE best_seq_num = 1)          AS principal_only_admissions,
  count(DISTINCT subject_id)                        AS strict_any_patients,
  count(DISTINCT subject_id) FILTER (WHERE best_seq_num = 1) AS principal_only_patients
FROM cohort;

-- -------- 3) 表型纯度是否与结局相关（潜在偏倚检查）--------
-- 若主诊断组与非主诊断组的观察事件率差异明显，说明 strict_any 引入了结局层面的噪声。
WITH cohort AS (
  SELECT
    m.subject_id, m.hadm_id, c.data_split, m.best_seq_num,
    l.event_type,
    m.days_to_next_observed_aecopd AS aecopd_gap
  FROM project_aecopd.master_cohort m
  JOIN project_aecopd.model_v1_cohort c USING (subject_id, hadm_id)
  JOIN project_aecopd.model_v1_labels l USING (hadm_id)
  WHERE c.training_eligible
)
SELECT '3_event_rate_by_purity' AS section,
  CASE WHEN best_seq_num = 1 THEN 'principal_dx' ELSE 'secondary_dx' END AS phenotype_group,
  count(*)                                                     AS n,
  count(*) FILTER (WHERE event_type = 1)                       AS aecopd_event_n,
  round(100.0*count(*) FILTER (WHERE event_type = 1)/nullif(count(*),0),2) AS pct_aecopd_event,
  count(*) FILTER (WHERE event_type = 2)                       AS death_event_n,
  round(100.0*count(*) FILTER (WHERE event_type = 2)/nullif(count(*),0),2) AS pct_death_event
FROM cohort
GROUP BY 2
ORDER BY 2;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/diagnostics/d3_short_interval_readmission.sql · lines 1–52

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/diagnostics/d3_short_interval_readmission.sql","part":1,"start_line":1,"end_line":52,"source_sha256":"b0ac218ff9d7573b0995e1c472754c37106803b5e2f3a1eb1587689a78129bbf","fragment_sha256":"b0ac218ff9d7573b0995e1c472754c37106803b5e2f3a1eb1587689a78129bbf","added_terminal_newline":false} -->
````sql
-- =====================================================================
-- 诊断脚本 D3：超短间隔再入院 / 转院拆分（复核问题 3）
-- 目的：量化"再入院"事件中间隔过短(0~2 天)的比例，这些更可能是转院或同一次
--   临床事件被拆成多条住院记录，而非真正的新一次 AECOPD 发作。若占比高，
--   0~30 天区间的 AECOPD 阳性标签会被污染。
-- 只读：仅 SELECT。运行方式同 D1。
-- 前置表：project_aecopd.aecopd_longitudinal_history（含 days_to_next_observed_aecopd、
--   qc_nonpositive_next_gap、qc_short_gap、qc_overlap、qc_transfer、next_observed_aecopd_hadm_id）。
--   这些 QC 标志在 02/03 阶段已算好，这里只做汇总。
-- =====================================================================
\pset pager off
SET client_encoding = 'UTF8';

-- -------- 1) 相邻 AECOPD 间隔的短间隔占比（全部已观察到下一次者）--------
SELECT '1_next_gap_short_interval' AS section,
  count(*) FILTER (WHERE next_observed_aecopd_hadm_id IS NOT NULL)  AS observed_next_n,
  count(*) FILTER (WHERE qc_nonpositive_next_gap)                   AS nonpositive_gap_n,       -- <=0 天，需人工复核
  count(*) FILTER (WHERE days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 1) AS gap_le_1d,
  count(*) FILTER (WHERE days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 2) AS gap_le_2d,
  count(*) FILTER (WHERE qc_short_gap)                              AS gap_le_7d,               -- (0,7] 天
  count(*) FILTER (WHERE days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 30) AS gap_le_30d,
  round(100.0*count(*) FILTER (WHERE qc_short_gap)
        /nullif(count(*) FILTER (WHERE days_to_next_observed_aecopd > 0),0),2) AS pct_le_7d_among_positive
FROM project_aecopd.aecopd_longitudinal_history;

-- -------- 2) 转院与重叠标志：可能的同一事件拆分 --------
SELECT '2_transfer_and_overlap_flags' AS section,
  count(*)                                     AS all_index_admissions,
  count(*) FILTER (WHERE qc_transfer)          AS admitted_via_transfer_from_hospital,
  count(*) FILTER (WHERE qc_overlap)           AS overlap_with_previous_strict,
  round(100.0*count(*) FILTER (WHERE qc_transfer)/nullif(count(*),0),2) AS pct_transfer,
  round(100.0*count(*) FILTER (WHERE qc_overlap)/nullif(count(*),0),2)  AS pct_overlap
FROM project_aecopd.aecopd_longitudinal_history;

-- -------- 3) 短间隔如何落在 0~30 天区间标签上（对 0-30 区间阳性的污染估计）--------
-- 把"下一次 AECOPD 在 30 天内"的阳性拆成 <=2 天 / 3-7 天 / 8-30 天，
-- <=2 天部分最可能是转院/拆分而非真正新发作。
SELECT '3_contamination_of_0_30_positive' AS section,
  count(*) FILTER (WHERE days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 30) AS aecopd_0_30_positive,
  count(*) FILTER (WHERE days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 2)  AS of_which_le_2d,
  count(*) FILTER (WHERE days_to_next_observed_aecopd > 2 AND days_to_next_observed_aecopd <= 7)  AS of_which_3_7d,
  count(*) FILTER (WHERE days_to_next_observed_aecopd > 7 AND days_to_next_observed_aecopd <= 30) AS of_which_8_30d,
  round(100.0*count(*) FILTER (WHERE days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 2)
        /nullif(count(*) FILTER (WHERE days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 30),0),2)
        AS pct_le_2d_of_0_30
FROM project_aecopd.aecopd_longitudinal_history;

-- -------- 4) 转院来源的短间隔交叉（转院 且 <=2 天，最强的拆分嫌疑）--------
SELECT '4_transfer_and_short_gap_cross' AS section,
  count(*) FILTER (WHERE qc_transfer AND days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 2) AS transfer_and_le_2d,
  count(*) FILTER (WHERE qc_transfer AND qc_short_gap)                                                          AS transfer_and_le_7d
FROM project_aecopd.aecopd_longitudinal_history;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/mimiciii_cohort_audit.sql · lines 1–81

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/mimiciii_cohort_audit.sql","part":1,"start_line":1,"end_line":81,"source_sha256":"5a09eb2252bfe9f03975bd08c03d56c2838c36881ed69d3d9a9282130d59445e","fragment_sha256":"5a09eb2252bfe9f03975bd08c03d56c2838c36881ed69d3d9a9282130d59445e","added_terminal_newline":false} -->
````sql
-- 在独立 mimiciii 数据库执行；仅查询，不创建或修改患者级表。
-- 与 work/12b_build_model_v3.sql 的患者级 seq<=3、索引及竞争事件口径对齐。
WITH dx AS (
  SELECT subject_id, hadm_id, MIN(seq_num) AS best_seq_num
  FROM public.diagnoses_icd
  WHERE icd9_code = '49121'
  GROUP BY subject_id, hadm_id
),
strict AS (
  SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime,
         a.admission_location, a.hospital_expire_flag, d.best_seq_num
  FROM dx d
  JOIN public.admissions a USING (subject_id, hadm_id)
),
seq3_patients AS (
  SELECT DISTINCT subject_id FROM strict WHERE best_seq_num <= 3
),
ordered AS (
  SELECT s.*,
         LAG(s.dischtime) OVER (PARTITION BY s.subject_id ORDER BY s.admittime, s.hadm_id) AS previous_strict_dischtime
  FROM strict s
  JOIN seq3_patients p USING (subject_id)
),
v3_adm AS (
  SELECT o.*,
         (o.admission_location = 'TRANSFER FROM HOSP/EXTRAM'
           AND o.previous_strict_dischtime IS NOT NULL
           AND EXTRACT(EPOCH FROM (o.admittime - o.previous_strict_dischtime))/86400.0 <= 2) AS is_transfer_continuation
  FROM ordered o
),
next_real AS (
  SELECT a.hadm_id, nr.admittime AS next_real_admittime
  FROM v3_adm a
  LEFT JOIN LATERAL (
    SELECT b.admittime
    FROM v3_adm b
    WHERE b.subject_id = a.subject_id
      AND (b.admittime, b.hadm_id) > (a.admittime, a.hadm_id)
      AND NOT b.is_transfer_continuation
    ORDER BY b.admittime, b.hadm_id
    LIMIT 1
  ) nr ON true
),
base AS (
  SELECT a.hadm_id, a.subject_id, a.hospital_expire_flag,
         (a.dischtime > a.admittime) AS training_eligible,
         CASE WHEN n.next_real_admittime IS NOT NULL
                AND n.next_real_admittime >= a.dischtime
              THEN EXTRACT(EPOCH FROM (n.next_real_admittime - a.dischtime))/86400.0
         END AS aecopd_days,
         CASE WHEN p.dod IS NOT NULL AND p.dod::date >= a.dischtime::date
              THEN (p.dod::date - a.dischtime::date)::double precision
         END AS death_days
  FROM v3_adm a
  JOIN next_real n USING (hadm_id)
  JOIN public.patients p USING (subject_id)
),
labelled AS (
  SELECT *, CASE
    WHEN aecopd_days <= 365 AND (death_days IS NULL OR aecopd_days <= death_days) THEN 1
    WHEN death_days <= 365 AND (aecopd_days IS NULL OR death_days < aecopd_days) THEN 2
    ELSE 0 END AS event_type
  FROM base
  WHERE hospital_expire_flag = 0
),
metrics AS (
  SELECT 'strict_code_patients' AS metric, COUNT(DISTINCT subject_id) AS value FROM strict
  UNION ALL SELECT 'strict_code_admissions', COUNT(*) FROM strict
  UNION ALL SELECT 'seq_le_3_patients', COUNT(*) FROM seq3_patients
  UNION ALL SELECT 'seq_le_3_code_admissions', COUNT(*) FROM strict WHERE best_seq_num <= 3
  UNION ALL SELECT 'all_strict_admissions_of_seq3_patients', COUNT(*) FROM v3_adm
  UNION ALL SELECT 'transfer_continuations_merged', COUNT(*) FROM v3_adm WHERE is_transfer_continuation
  UNION ALL SELECT 'alive_index_admissions', COUNT(*) FROM labelled
  UNION ALL SELECT 'training_eligible_index_admissions', COUNT(*) FROM labelled WHERE training_eligible
  UNION ALL SELECT 'aecopd_readmission_365d', COUNT(*) FROM labelled WHERE event_type = 1
  UNION ALL SELECT 'death_before_readmission_365d', COUNT(*) FROM labelled WHERE event_type = 2
  UNION ALL SELECT 'no_event_365d', COUNT(*) FROM labelled WHERE event_type = 0
  UNION ALL SELECT 'aecopd_readmission_90d', COUNT(*) FROM labelled WHERE event_type = 1 AND aecopd_days <= 90
  UNION ALL SELECT 'aecopd_readmission_180d', COUNT(*) FROM labelled WHERE event_type = 1 AND aecopd_days <= 180
)
SELECT metric, value FROM metrics ORDER BY metric;
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/mimiciii_index_core_features_for_coded_cohort.sql · lines 1–16

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/mimiciii_index_core_features_for_coded_cohort.sql","part":1,"start_line":1,"end_line":16,"source_sha256":"d806bbe7808bd7d9786d5a98d84766fa5e7c486a7896d8f68b8c64e2d60ec113","fragment_sha256":"d806bbe7808bd7d9786d5a98d84766fa5e7c486a7896d8f68b8c64e2d60ec113","added_terminal_newline":false} -->
````sql
-- 非覆盖性索引：支持按 hadm_id 抽取新队列的原始特征。
-- 官方完整索引脚本包含 DROP INDEX；这里仅创建当前所需索引。
\set ON_ERROR_STOP on

CREATE INDEX IF NOT EXISTS mimiciii_labevents_hadm_chart_idx
  ON public.labevents(hadm_id,charttime);
CREATE INDEX IF NOT EXISTS mimiciii_prescriptions_hadm_start_idx
  ON public.prescriptions(hadm_id,startdate);
CREATE INDEX IF NOT EXISTS mimiciii_microbiologyevents_hadm_chart_idx
  ON public.microbiologyevents(hadm_id,charttime);
CREATE INDEX IF NOT EXISTS mimiciii_icustays_hadm_idx
  ON public.icustays(hadm_id);
CREATE INDEX IF NOT EXISTS mimiciii_procedures_icd_hadm_idx
  ON public.procedures_icd(hadm_id);
CREATE INDEX IF NOT EXISTS mimiciii_transfers_hadm_idx
  ON public.transfers(hadm_id);
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/mimiciii_load_core_features_for_coded_cohort.sql · lines 1–28

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/mimiciii_load_core_features_for_coded_cohort.sql","part":1,"start_line":1,"end_line":28,"source_sha256":"ca2e5a4f97e0fa35e9df2e48a5f725328ba66b33092ba0987a305ecd9c20a33e","fragment_sha256":"ca2e5a4f97e0fa35e9df2e48a5f725328ba66b33092ba0987a305ecd9c20a33e","added_terminal_newline":false} -->
````sql
-- MIMIC-III 补齐新版纵向数据集所需的基础来源表；原始表须由官方建表脚本创建且保持空表。
-- 仅加载当前需要的字典、实验室、处方、ICU、微生物和住院过程数据；
-- 33GB 的 CHARTEVENTS 不在本脚本中，避免未经容量评估就整表导入。
\set ON_ERROR_STOP on

DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM public.d_labitems)
     OR EXISTS (SELECT 1 FROM public.d_icd_diagnoses)
     OR EXISTS (SELECT 1 FROM public.d_items)
     OR EXISTS (SELECT 1 FROM public.labevents)
     OR EXISTS (SELECT 1 FROM public.prescriptions)
     OR EXISTS (SELECT 1 FROM public.icustays)
     OR EXISTS (SELECT 1 FROM public.microbiologyevents)
     OR EXISTS (SELECT 1 FROM public.procedures_icd)
     OR EXISTS (SELECT 1 FROM public.transfers) THEN
    RAISE EXCEPTION 'Core-feature import expects empty target tables; inspect before retrying';
  END IF;
END $$;

\copy public.D_LABITEMS FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/D_LABITEMS.csv' DELIMITER ',' CSV HEADER NULL ''
\copy public.D_ICD_DIAGNOSES FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/D_ICD_DIAGNOSES.csv' DELIMITER ',' CSV HEADER NULL ''
\copy public.D_ITEMS FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/D_ITEMS.csv' DELIMITER ',' CSV HEADER NULL ''
\copy public.ICUSTAYS FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/ICUSTAYS.csv' DELIMITER ',' CSV HEADER NULL ''
\copy public.PROCEDURES_ICD FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/PROCEDURES_ICD.csv' DELIMITER ',' CSV HEADER NULL ''
\copy public.TRANSFERS FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/TRANSFERS.csv' DELIMITER ',' CSV HEADER NULL ''
\copy public.MICROBIOLOGYEVENTS FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/MICROBIOLOGYEVENTS.csv' DELIMITER ',' CSV HEADER NULL ''
\copy public.PRESCRIPTIONS FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/PRESCRIPTIONS.csv' DELIMITER ',' CSV HEADER NULL ''
\copy public.LABEVENTS FROM 'D:/ywj/MIMICIV/mimiciii_flat_csv/LABEVENTS.csv' DELIMITER ',' CSV HEADER NULL ''
````
<!-- END REVIEW_FRAGMENT -->

## cohort_pipeline/sql/mimiciii_note_audit.sql · lines 1–34

<!-- REVIEW_FRAGMENT {"path":"cohort_pipeline/sql/mimiciii_note_audit.sql","part":1,"start_line":1,"end_line":34,"source_sha256":"02d650e125d312748f4906103d8966cfe51ca254b16645e007ca141e8ab4b654","fragment_sha256":"02d650e125d312748f4906103d8966cfe51ca254b16645e007ca141e8ab4b654","added_terminal_newline":false} -->
````sql
-- MIMIC-III 在当前 v3 索引住院上的出院小结覆盖；只输出汇总。
WITH dx AS (
  SELECT subject_id, hadm_id, MIN(seq_num) AS best_seq_num
  FROM public.diagnoses_icd WHERE icd9_code = '49121'
  GROUP BY subject_id, hadm_id
),
seq3_patients AS (
  SELECT DISTINCT subject_id FROM dx WHERE best_seq_num <= 3
),
cohort AS (
  SELECT a.subject_id, a.hadm_id, a.dischtime
  FROM public.admissions a
  JOIN dx d USING (subject_id, hadm_id)
  JOIN seq3_patients p USING (subject_id)
  WHERE a.hospital_expire_flag = 0 AND a.dischtime > a.admittime
),
per_admission AS (
  SELECT c.hadm_id,
         COUNT(n.row_id) AS discharge_notes,
         COUNT(n.row_id) FILTER (WHERE n.charttime IS NOT NULL AND n.storetime IS NOT NULL) AS notes_with_both_times,
         COUNT(n.row_id) FILTER (WHERE n.charttime <= c.dischtime AND n.storetime <= c.dischtime) AS notes_available_by_discharge,
         COUNT(n.row_id) FILTER (WHERE n.chartdate <= c.dischtime::date) AS notes_chartdated_by_discharge
  FROM cohort c
  LEFT JOIN public.noteevents n ON n.subject_id = c.subject_id AND n.hadm_id = c.hadm_id
                                 AND n.category = 'Discharge summary' AND COALESCE(n.iserror, '0') <> '1'
  GROUP BY c.hadm_id
)
SELECT COUNT(*) AS index_admissions,
       COUNT(*) FILTER (WHERE discharge_notes > 0) AS with_discharge_note,
       COUNT(*) FILTER (WHERE notes_with_both_times > 0) AS with_note_both_times,
       COUNT(*) FILTER (WHERE notes_available_by_discharge > 0) AS with_note_available_by_discharge,
       COUNT(*) FILTER (WHERE notes_chartdated_by_discharge > 0) AS with_note_chartdated_by_discharge,
       SUM(discharge_notes) AS discharge_note_rows
FROM per_admission;
````
<!-- END REVIEW_FRAGMENT -->
