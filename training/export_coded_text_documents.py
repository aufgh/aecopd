"""经本机 psql 将新队列的出院前可用文书直接导出为受控 gzip CSV。"""

from __future__ import annotations

import argparse
import csv
import getpass
import gzip
import os
import shutil
import subprocess
from pathlib import Path


SOURCE_FIELDS = (
    "document_key", "subject_id", "hadm_id", "source_hadm_id", "data_split",
    "document_role", "document_rank", "days_before_landmark", "note_type",
    "charttime", "storetime", "document_text",
)


def copy_sql(database: str) -> str:
    if database not in {"mimiciv", "mimiciii"}:
        raise ValueError("database must be mimiciv or mimiciii")
    role = (
        "CASE n.note_role WHEN 'Discharge summary' THEN 'current_discharge' "
        "ELSE 'current_radiology' END"
        if database == "mimiciv" else "'current_clinical_note'"
    )
    return f"""COPY (
      SELECT n.source_dataset||':'||n.note_id AS document_key,
             n.subject_id,n.hadm_id,n.hadm_id AS source_hadm_id,
             CASE WHEN ((('x'||substr(md5('aecopd-model-v1:'||n.subject_id::text),1,8))::bit(32)::bigint % 100)<70)
                    THEN 'train'
                  WHEN ((('x'||substr(md5('aecopd-model-v1:'||n.subject_id::text),1,8))::bit(32)::bigint % 100)<85)
                    THEN 'validation' ELSE 'test' END AS data_split,
             {role} AS document_role,
             row_number() OVER (PARTITION BY n.hadm_id,n.note_role
                                ORDER BY n.storetime DESC,n.note_id)-1 AS document_rank,
             extract(epoch FROM (c.dischtime-n.storetime))/86400.0 AS days_before_landmark,
             n.note_role AS note_type,n.charttime,n.storetime,n.text AS document_text
      FROM project_aecopd.copd_coded_timed_notes_v1 n
      JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
      WHERE n.text IS NOT NULL AND btrim(n.text)<>''
      ORDER BY n.hadm_id,n.note_role,n.storetime DESC,n.note_id
    ) TO STDOUT WITH (FORMAT CSV, HEADER TRUE)"""


def export(database: str, output: Path, *, psql: Path, password: str) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    environment = os.environ.copy()
    environment["PGPASSWORD"] = password
    try:
        with gzip.open(partial, "wb", compresslevel=6) as handle:
            process = subprocess.Popen(
                [str(psql), "-X", "-q", "-w", "-U", "postgres", "-d", database,
                 "-v", "ON_ERROR_STOP=1", "-c", copy_sql(database)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
            )
            assert process.stdout is not None
            assert process.stderr is not None
            shutil.copyfileobj(process.stdout, handle)
            stderr = process.stderr.read()
            returncode = process.wait()
        if returncode:
            raise RuntimeError(stderr.decode("utf-8", errors="replace"))
        rows = 0
        roles: dict[str, int] = {}
        with gzip.open(partial, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SOURCE_FIELDS:
                raise ValueError(f"unexpected text export fields: {reader.fieldnames}")
            for row in reader:
                rows += 1
                roles[row["document_role"]] = roles.get(row["document_role"], 0) + 1
                if float(row["days_before_landmark"]) < 0:
                    raise ValueError("future note leaked into text export")
        os.replace(partial, output)
        return {"database": database, "documents": rows, "roles": roles, "output": str(output)}
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", choices=("mimiciv", "mimiciii"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--psql", type=Path, default=Path("C:/Program Files/PostgreSQL/18/bin/psql.exe"))
    arguments = parser.parse_args()
    password = getpass.getpass("PostgreSQL password (not saved): ")
    print(export(arguments.database, arguments.output, psql=arguments.psql, password=password))


if __name__ == "__main__":
    main()
