"""导出指定 MIMIC-IV 索引住院的全部出院小结；原文仅写本机受控 gzip。"""

from __future__ import annotations

import argparse
import csv
import getpass
import gzip
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path


FIELDS = (
    "subject_id", "hadm_id", "note_id", "charttime", "storetime",
    "dischtime", "timestamp_relation", "document_text",
)
INDEX_TABLES = {
    "strict-v1": "project_aecopd.copd_coded_strict_index_labels_v1",
    "first-coded-v2": "project_aecopd.copd_coded_first_index_labels_v2",
    "discharge-prediction-set": "project_aecopd.copd_coded_all_index_labels_v2",
}


def copy_sql(index_cohort: str = "strict-v1") -> str:
    """存档时间不作为小结排除条件；不读取结局或未来住院标签。"""

    if index_cohort not in INDEX_TABLES:
        raise ValueError(f"unknown index cohort: {index_cohort}")
    return f"""COPY (
      SELECT l.subject_id,l.hadm_id,n.note_id,n.charttime,n.storetime,c.dischtime,
             CASE WHEN n.storetime IS NULL THEN 'unknown'
                  WHEN n.storetime>c.dischtime THEN 'after_discharge'
                  ELSE 'by_discharge' END AS timestamp_relation,
             n.text AS document_text
      FROM {INDEX_TABLES[index_cohort]} l
      JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
      JOIN mimiciv_note.discharge n
        ON n.subject_id=l.subject_id AND n.hadm_id=l.hadm_id
      WHERE n.text IS NOT NULL AND btrim(n.text)<>''
      ORDER BY l.hadm_id,n.note_id
    ) TO STDOUT WITH (FORMAT CSV, HEADER TRUE)"""


def export(
    output_dir: Path, *, psql: Path, password: str, index_cohort: str = "strict-v1"
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not psql.is_file():
        raise FileNotFoundError(psql)
    output_dir.mkdir(parents=True)
    target = output_dir / "source_documents.csv.gz"
    partial = output_dir / "source_documents.csv.gz.partial"
    environment = os.environ.copy()
    environment["PGPASSWORD"] = password
    with gzip.open(partial, "wb", compresslevel=6) as handle:
        process = subprocess.Popen(
            [str(psql), "-X", "-q", "-w", "-U", "postgres", "-d", "mimiciv",
             "-v", "ON_ERROR_STOP=1", "-c", copy_sql(index_cohort)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
        )
        assert process.stdout is not None and process.stderr is not None
        shutil.copyfileobj(process.stdout, handle)
        stderr = process.stderr.read()
        status = process.wait()
    if status:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))
    counts: Counter[str] = Counter()
    hadm_ids: set[int] = set()
    with gzip.open(partial, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("unexpected discharge export columns")
        for row in reader:
            hadm = int(row["hadm_id"])
            if hadm in hadm_ids:
                raise ValueError("multiple discharge summaries for one index admission")
            hadm_ids.add(hadm)
            counts[row["timestamp_relation"]] += 1
    if not hadm_ids:
        raise ValueError("empty discharge export")
    partial.replace(target)
    summary: dict[str, object] = {
        "source_dataset": "MIMIC-IV",
        "index_cohort": index_cohort,
        "index_admissions_with_note": len(hadm_ids),
        "timestamp_relation": dict(counts),
        "outcome_labels_exported": False,
        "after_discharge_storetime_is_not_automatically_future_content": True,
        "raw_text_location": str(target),
    }
    if index_cohort == "strict-v1":
        summary["strict_index_admissions_with_note"] = len(hadm_ids)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--psql", type=Path, default=Path("C:/Program Files/PostgreSQL/18/bin/psql.exe"))
    parser.add_argument("--index-cohort", choices=tuple(INDEX_TABLES), default="strict-v1")
    args = parser.parse_args()
    password = os.environ.get("PGPASSWORD") or getpass.getpass("PostgreSQL password: ")
    print(json.dumps(export(
        args.output_dir, psql=args.psql, password=password, index_cohort=args.index_cohort
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
