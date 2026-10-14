"""导出严格 AECOPD 任意位或既定 COPD／肺气肿前三位的并集再住院标签。"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

from training.v5.export_broad_labels import export


SQL = """COPY (
  SELECT subject_id,hadm_id,admittime,dischtime,event_type,time_days,
         next_qualified_hadm_id AS next_copd_hadm_id,
         next_copd_best_code_seq_num,next_is_strict_aecopd,
         next_has_49122,next_has_j440
  FROM project_aecopd.copd_qualified_readmission_labels_v1
  ORDER BY subject_id,admittime,hadm_id
) TO STDOUT WITH (FORMAT CSV,HEADER TRUE)"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--psql", type=Path,
                        default=Path("C:/Program Files/PostgreSQL/18/bin/psql.exe"))
    args = parser.parse_args()
    password = os.environ.get("PGPASSWORD") or getpass.getpass("PostgreSQL password: ")
    report = export(
        args.output_dir, args.psql, password, sql=SQL,
        status_name="qualified_labels_exported_and_checked",
        primary_event="first non-transfer readmission with strict 491.21/J44.1 at any position or preset COPD/emphysema code in diagnosis positions 1-3",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
