"""只读导出 V5 出院预测样本集的结构化输入；患者级原始文件仅存私有目录。"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import subprocess
from pathlib import Path


BASE_SQL = """COPY (
  SELECT l.subject_id,l.hadm_id,l.admittime,l.dischtime,l.event_type,l.time_days,
         l.coded_admission_order,c.prior_coded_admission_count,
         l.index_is_strict_aecopd,
         LEAST(90,GREATEST(0,p.anchor_age+extract(year FROM l.admittime)-p.anchor_year))::double precision AS age_at_admission,
         p.gender,c.discharge_location,
         (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate h
          WHERE h.subject_id=l.subject_id AND h.is_strict_aecopd
            AND h.dischtime<=l.admittime) AS prior_strict_aecopd_count
  FROM project_aecopd.copd_coded_all_index_labels_v2 l
  JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
  JOIN mimiciv_hosp.patients p ON p.subject_id=l.subject_id
  ORDER BY l.subject_id,l.admittime,l.hadm_id
) TO STDOUT WITH (FORMAT CSV,HEADER TRUE)"""

LAB_SQL = """COPY (
  WITH selected AS MATERIALIZED (
    SELECT l.hadm_id,x.itemid,x.charttime,x.labevent_id,x.valuenum
    FROM project_aecopd.copd_coded_all_index_labels_v2 l
    JOIN mimiciv_hosp.labevents x ON x.hadm_id=l.hadm_id
    WHERE x.itemid IN (50818,50820,50821,50862,50882,50912,51222,51301)
      AND x.charttime BETWEEN l.admittime AND l.dischtime
      AND x.storetime<=l.dischtime AND x.valuenum IS NOT NULL
      AND x.valuenum>=0
  ), ranked AS (
    SELECT *,row_number() OVER (PARTITION BY hadm_id,itemid ORDER BY charttime,labevent_id) AS first_rank,
           row_number() OVER (PARTITION BY hadm_id,itemid ORDER BY charttime DESC,labevent_id DESC) AS last_rank
    FROM selected
  )
  SELECT hadm_id,itemid,count(*) AS measurement_count,
         max(valuenum) FILTER (WHERE first_rank=1) AS first_value,
         max(valuenum) FILTER (WHERE last_rank=1) AS last_value,
         min(valuenum) AS min_value,max(valuenum) AS max_value
  FROM ranked GROUP BY hadm_id,itemid ORDER BY hadm_id,itemid
) TO STDOUT WITH (FORMAT CSV,HEADER TRUE)"""

DIAGNOSIS_SQL = """COPY (
  SELECT d.hadm_id,d.icd_version,btrim(d.icd_code) AS icd_code,d.seq_num
  FROM project_aecopd.copd_coded_all_index_labels_v2 l
  JOIN mimiciv_hosp.diagnoses_icd d USING(hadm_id)
  ORDER BY d.hadm_id,d.seq_num,d.icd_code
) TO STDOUT WITH (FORMAT CSV,HEADER TRUE)"""

MEDICATION_SQL = """COPY (
  SELECT e.hadm_id,lower(btrim(e.medication)) AS medication,
         min(e.charttime) AS first_charttime,max(e.charttime) AS last_charttime
  FROM project_aecopd.copd_coded_all_index_labels_v2 l
  JOIN mimiciv_hosp.emar e USING(hadm_id)
  WHERE e.charttime BETWEEN l.admittime AND l.dischtime
    AND e.storetime<=l.dischtime
    AND e.event_txt IN ('Administered','Started','Applied','Delayed Administered')
    AND e.medication IS NOT NULL AND btrim(e.medication)<>''
  GROUP BY e.hadm_id,lower(btrim(e.medication))
  ORDER BY e.hadm_id,medication
) TO STDOUT WITH (FORMAT CSV,HEADER TRUE)"""


def export(output_dir: Path, psql: Path, password: str) -> dict[str, object]:
    """不修改数据库；仅导出聚合建模字段并校验主键。"""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not psql.is_file():
        raise FileNotFoundError(psql)
    output_dir.mkdir(parents=True)
    environment = os.environ.copy()
    environment["PGPASSWORD"] = password
    counts: dict[str, int] = {}
    for name, sql in (("base", BASE_SQL), ("labs", LAB_SQL),
                      ("diagnoses", DIAGNOSIS_SQL), ("medications", MEDICATION_SQL)):
        target = output_dir / f"{name}.csv"
        with target.open("wb") as handle:
            result = subprocess.run(
                [str(psql), "-X", "-q", "-w", "-h", "localhost", "-U", "postgres",
                 "-d", "mimiciv", "-v", "ON_ERROR_STOP=1", "-c", sql],
                stdout=handle, stderr=subprocess.PIPE, env=environment, check=False,
            )
        if result.returncode:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        with target.open("r", encoding="utf-8-sig", newline="") as handle:
            counts[name] = sum(1 for _ in csv.DictReader(handle))
        print(json.dumps({"exported": name, "rows": counts[name]}), flush=True)
    with (output_dir / "base.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    hadms = {int(row["hadm_id"]) for row in rows}
    patients = {int(row["subject_id"]) for row in rows}
    if len(rows) != 19401 or len(hadms) != len(rows) or len(patients) != 5351:
        raise ValueError("discharge prediction set key/count mismatch")
    report: dict[str, object] = {
        "cohort": "MIMIC-IV discharge prediction sample set",
        "patients": len(patients), "admissions": len(rows), "row_counts": counts,
        "medication_source": "eMAR observed administrations, not prescriptions",
        "lab_availability": "charttime and storetime not later than discharge",
        "patient_level_identifiers_private": True,
        "individual_365_day_followup_verified": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--psql", type=Path, default=Path("C:/Program Files/PostgreSQL/18/bin/psql.exe"))
    args = parser.parse_args()
    password = os.environ.get("PGPASSWORD") or getpass.getpass("PostgreSQL password: ")
    print(json.dumps(export(args.output_dir, args.psql, password), ensure_ascii=False))


if __name__ == "__main__":
    main()
