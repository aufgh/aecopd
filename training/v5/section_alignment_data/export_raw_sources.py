"""只读导出 V5 章节对齐所需的原始结构化事件。"""

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
from pathlib import Path


COHORT = "project_aecopd.copd_coded_all_index_labels_v2"


def _copy_sql(select: str) -> str:
    return f"COPY ({select}) TO STDOUT WITH (FORMAT CSV, HEADER TRUE)"


QUERIES: dict[str, str] = {
    "cohort": _copy_sql(f"""
        SELECT l.subject_id,l.hadm_id,l.admittime,l.dischtime,l.index_is_strict_aecopd
        FROM {COHORT} l ORDER BY l.subject_id,l.admittime,l.hadm_id
    """),
    "labs": _copy_sql(f"""
        SELECT x.subject_id,x.hadm_id,x.labevent_id,x.itemid,x.charttime,x.storetime,
               x.valuenum,x.valueuom,x.flag
        FROM {COHORT} l JOIN mimiciv_hosp.labevents x USING (hadm_id)
        WHERE x.itemid IN (50818,50820,50821,50862,50882,50912,51222,51301)
          AND x.charttime BETWEEN l.admittime AND l.dischtime
          AND x.storetime <= l.dischtime AND x.valuenum IS NOT NULL AND x.valuenum >= 0
        ORDER BY x.hadm_id,x.itemid,x.charttime,x.labevent_id
    """),
    "diagnoses": _copy_sql(f"""
        SELECT d.subject_id,d.hadm_id,d.seq_num,btrim(d.icd_code) AS icd_code,d.icd_version
        FROM {COHORT} l JOIN mimiciv_hosp.diagnoses_icd d USING (hadm_id)
        ORDER BY d.hadm_id,d.seq_num,d.icd_code
    """),
    "emar": _copy_sql(f"""
        SELECT e.subject_id,e.hadm_id,e.emar_id,e.emar_seq,e.pharmacy_id,
               e.charttime,e.storetime,e.medication,e.event_txt
        FROM {COHORT} l JOIN mimiciv_hosp.emar e USING (hadm_id)
        WHERE e.charttime BETWEEN l.admittime AND l.dischtime
          AND e.storetime <= l.dischtime
          AND e.event_txt IN ('Administered','Started','Applied','Delayed Administered')
          AND e.medication IS NOT NULL AND btrim(e.medication) <> ''
        ORDER BY e.hadm_id,e.charttime,e.emar_id
    """),
    "prescriptions": _copy_sql(f"""
        SELECT p.subject_id,p.hadm_id,p.pharmacy_id,p.poe_id,p.starttime,p.stoptime,
               p.drug_type,p.drug,p.gsn,p.route
        FROM {COHORT} l JOIN mimiciv_hosp.prescriptions p USING (hadm_id)
        ORDER BY p.hadm_id,p.pharmacy_id,p.drug_type,p.drug
    """),
    "pharmacy": _copy_sql(f"""
        SELECT p.subject_id,p.hadm_id,p.pharmacy_id,p.poe_id,p.starttime,p.stoptime,
               p.entertime,p.verifiedtime,p.medication,p.status,p.proc_type
        FROM {COHORT} l JOIN mimiciv_hosp.pharmacy p USING (hadm_id)
        ORDER BY p.hadm_id,p.pharmacy_id
    """),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_count(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def export(output_dir: Path, psql: Path, password: str, note_source: Path) -> dict[str, object]:
    """SQL 全部为只读 COPY；每份压缩文件完成后再原子改名。"""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not psql.is_file() or not note_source.is_file():
        raise FileNotFoundError("psql or source notes missing")
    if any(not sql.lstrip().upper().startswith("COPY (") for sql in QUERIES.values()):
        raise ValueError("export queries must be read-only COPY")
    output_dir.mkdir(parents=True)
    environment = os.environ.copy()
    environment["PGPASSWORD"] = password
    summary: dict[str, object] = {
        "cohort": "MIMIC-IV V5 discharge prediction sample set",
        "source_note_gzip_sha256": _sha256(note_source),
        "sources": {},
        "prescriptions_are_not_confirmed_discharge_medications": True,
    }
    for name, sql in QUERIES.items():
        target = output_dir / f"{name}.csv.gz"
        partial = output_dir / f"{name}.csv.gz.partial"
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
            raise RuntimeError(f"{name} export failed: {error.decode('utf-8', errors='replace')}")
        partial.replace(target)
        summary["sources"][name] = {
            "rows": _row_count(target), "sha256": _sha256(target),
            "bytes_compressed": target.stat().st_size,
        }
        print(json.dumps({"exported": name, **summary["sources"][name]}), flush=True)
    if summary["sources"]["cohort"]["rows"] != 19401:
        raise ValueError("cohort row count changed")
    (output_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--note-source", type=Path, required=True)
    parser.add_argument("--psql", type=Path,
                        default=Path("C:/Program Files/PostgreSQL/18/bin/psql.exe"))
    args = parser.parse_args()
    password = os.environ.get("PGPASSWORD") or getpass.getpass("PostgreSQL password: ")
    export(args.output_dir, args.psql, password, args.note_source)


if __name__ == "__main__":
    main()
