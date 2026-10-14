"""只读导出新的 COPD 编码再住院结局；不改动旧严格结局或特征。"""

from __future__ import annotations

import argparse
import csv
import getpass
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path


SQL = """COPY (
  SELECT subject_id,hadm_id,admittime,dischtime,event_type,time_days,
         next_copd_hadm_id,next_copd_best_code_seq_num,next_is_strict_aecopd,
         next_has_49122,next_has_j440
  FROM project_aecopd.copd_broad_readmission_labels_v1
  ORDER BY subject_id,admittime,hadm_id
) TO STDOUT WITH (FORMAT CSV,HEADER TRUE)"""


def export(output_dir: Path, psql: Path, password: str, *, sql: str = SQL,
           status_name: str = "broad_labels_exported_and_checked",
           primary_event: str = "first non-transfer admission with any preset COPD/emphysema code in any diagnosis position") -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not psql.is_file():
        raise FileNotFoundError(psql)
    output_dir.mkdir(parents=True)
    target = output_dir / "labels.csv.gz"
    partial = output_dir / "labels.csv.gz.partial"
    environment = os.environ.copy()
    environment["PGPASSWORD"] = password
    with gzip.open(partial, "wb", compresslevel=6) as destination:
        process = subprocess.Popen(
            [str(psql), "-X", "-q", "-w", "-h", "localhost", "-U", "postgres",
             "-d", "mimiciv", "-v", "ON_ERROR_STOP=1", "-c", sql],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
        )
        assert process.stdout is not None and process.stderr is not None
        shutil.copyfileobj(process.stdout, destination)
        error = process.stderr.read()
        status = process.wait()
    if status:
        raise RuntimeError(error.decode("utf-8", errors="replace"))
    partial.replace(target)
    counts: Counter[int] = Counter()
    subtypes: Counter[str] = Counter()
    positions: Counter[str] = Counter()
    hadms: set[str] = set()
    subjects: set[str] = set()
    with gzip.open(target, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            hadm = row["hadm_id"]
            if hadm in hadms:
                raise ValueError("duplicate label admission")
            hadms.add(hadm)
            subjects.add(row["subject_id"])
            event = int(row["event_type"])
            time = float(row["time_days"])
            if event not in (0, 1, 2) or not 0 <= time <= 365:
                raise ValueError("invalid competing event label")
            if event == 0 and time != 365:
                raise ValueError("administrative censoring time changed")
            counts[event] += 1
            if event == 1:
                sequence = int(row["next_copd_best_code_seq_num"])
                positions["principal" if sequence == 1 else
                          "secondary_2_3" if sequence <= 3 else "position_4_plus"] += 1
                subtypes["strict" if row["next_is_strict_aecopd"] == "t" else
                         "acute_infection_or_bronchitis" if row["next_has_49122"] == "t"
                         or row["next_has_j440"] == "t" else
                         "other_copd_or_emphysema"] += 1
    if len(hadms) != 19401 or len(subjects) != 5351:
        raise ValueError("broad label cohort changed")
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    report: dict[str, object] = {
        "status": status_name,
        "cohort_admissions": len(hadms), "cohort_patients": len(subjects),
        "events": {str(key): counts[key] for key in (0, 1, 2)},
        "readmission_subtypes": dict(subtypes),
        "readmission_copd_code_position": dict(positions),
        "label_sha256": sha,
        "primary_event": primary_event,
        "individual_followup_complete": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--psql", type=Path,
                        default=Path("C:/Program Files/PostgreSQL/18/bin/psql.exe"))
    args = parser.parse_args()
    password = os.environ.get("PGPASSWORD") or getpass.getpass("PostgreSQL password: ")
    print(json.dumps(export(args.output_dir, args.psql, password), ensure_ascii=False))


if __name__ == "__main__":
    main()
