"""只读导出 MIMIC-IV 08j 首次编码起点的低维模型输入。"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import subprocess
from pathlib import Path


BASE_SQL = """
COPY (
  SELECT l.subject_id,l.hadm_id,l.source_dataset,l.event_type,l.time_days,
         LEAST(90,GREATEST(0,p.anchor_age+extract(year FROM l.admittime)-p.anchor_year))::double precision AS age_at_admission,
         CASE p.gender WHEN 'M' THEN 1 WHEN 'F' THEN 0 END AS male,
         c.prior_coded_admission_count,
         (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate h
          WHERE h.subject_id=l.subject_id AND h.is_strict_aecopd
            AND h.dischtime<=l.admittime) AS prior_strict_aecopd_count,
         l.at_or_after_first_qualification,l.index_is_strict_aecopd
  FROM project_aecopd.copd_coded_first_index_labels_v2 l
  JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
  JOIN mimiciv_hosp.patients p ON p.subject_id=l.subject_id
  ORDER BY l.hadm_id
) TO STDOUT WITH (FORMAT CSV, HEADER TRUE)
"""

LAB_SQL = """
COPY (
  WITH selected AS MATERIALIZED (
    SELECT l.hadm_id,x.itemid,x.charttime,x.labevent_id,x.valuenum
    FROM project_aecopd.copd_coded_first_index_labels_v2 l
    JOIN mimiciv_hosp.labevents x ON x.hadm_id=l.hadm_id
    WHERE x.itemid IN (50818,50821,51301,51222,50912)
      AND x.charttime BETWEEN l.admittime AND l.dischtime
      AND x.storetime<=l.dischtime
      AND x.valuenum IS NOT NULL AND x.valuenum>=0
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
) TO STDOUT WITH (FORMAT CSV, HEADER TRUE)
"""


def export(output_dir: Path, psql: Path, password: str) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not psql.is_file():
        raise FileNotFoundError(psql)
    output_dir.mkdir(parents=True)
    environment = os.environ.copy()
    environment["PGPASSWORD"] = password
    for name, sql in (("base", BASE_SQL), ("labs", LAB_SQL)):
        with (output_dir / f"{name}.csv").open("wb") as handle:
            process = subprocess.run(
                [str(psql), "-X", "-q", "-w", "-U", "postgres", "-d", "mimiciv",
                 "-v", "ON_ERROR_STOP=1", "-c", sql],
                stdout=handle, stderr=subprocess.PIPE, env=environment, check=False,
            )
        if process.returncode:
            raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
    with (output_dir / "base.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        base = list(csv.DictReader(handle))
    with (output_dir / "labs.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        labs = list(csv.DictReader(handle))
    hadms = {int(row["hadm_id"]) for row in base}
    if len(base) != 5351 or len(hadms) != len(base) or any(int(row["hadm_id"]) not in hadms for row in labs):
        raise ValueError("08j cohort or lab export key mismatch")
    report = {"cohort": "MIMIC-IV 08j first-coded-v2", "admissions": len(base),
              "lab_rows": len(labs), "outcome_definition": "strict AECOPD readmission vs death within 365 days",
              "individual_followup_complete": False}
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
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
