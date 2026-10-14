"""从 PostgreSQL 同一只读快照导出既定 COPD 住院的原始底账。"""

from __future__ import annotations

import argparse
import csv
import getpass
import gzip
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


COHORT = "project_aecopd.copd_coded_admissions_v1_gate"
LABELS = "project_aecopd.copd_qualified_readmission_labels_v1"
NULL = "\\N"


@dataclass(frozen=True)
class Source:
    name: str
    table: str
    select: str
    primary_key: tuple[str, ...] = ()


def sources() -> tuple[Source, ...]:
    """源表不作数值清洗、特征合并或文本编码。"""
    selected = f"SELECT hadm_id FROM {COHORT}"
    patients = f"SELECT DISTINCT subject_id FROM {COHORT}"
    result = [
        Source("cohort/admissions", COHORT, f"SELECT * FROM {COHORT} ORDER BY subject_id,admittime,hadm_id", ("hadm_id",)),
        Source("raw/patients", "mimiciv_hosp.patients", f"SELECT * FROM mimiciv_hosp.patients WHERE subject_id IN ({patients}) ORDER BY subject_id", ("subject_id",)),
    ]
    for table, key in (
        ("admissions", ("hadm_id",)), ("labevents", ("labevent_id",)),
        ("diagnoses_icd", ()), ("emar", ("emar_id", "emar_seq")),
        ("prescriptions", ()), ("pharmacy", ()),
    ):
        result.append(Source(f"raw/{table}", f"mimiciv_hosp.{table}",
                             f"SELECT * FROM mimiciv_hosp.{table} WHERE hadm_id IN ({selected})", key))
    result.append(Source("raw/emar_detail", "mimiciv_hosp.emar_detail",
                         f"SELECT d.* FROM mimiciv_hosp.emar_detail d WHERE EXISTS (SELECT 1 FROM mimiciv_hosp.emar e WHERE e.emar_id=d.emar_id AND e.emar_seq=d.emar_seq AND e.hadm_id IN ({selected}))"))
    for table in ("discharge", "radiology"):
        result.append(Source(f"raw/{table}", f"mimiciv_note.{table}",
                             f"SELECT * FROM mimiciv_note.{table} WHERE hadm_id IN ({selected})", ("note_id",)))
    result.append(Source("raw/labevents_unassigned", "mimiciv_hosp.labevents",
                         f"SELECT x.* FROM mimiciv_hosp.labevents x WHERE x.hadm_id IS NULL AND x.subject_id IN ({patients}) AND EXISTS (SELECT 1 FROM {COHORT} c WHERE c.subject_id=x.subject_id AND x.charttime BETWEEN c.admittime AND c.dischtime)", ("labevent_id",)))
    for table in ("d_labitems", "d_icd_diagnoses"):
        result.append(Source(f"raw/{table}", f"mimiciv_hosp.{table}",
                             f"SELECT * FROM mimiciv_hosp.{table}"))
    result.append(Source("outcomes/readmission_death_labels", LABELS,
                         f"SELECT * FROM {LABELS} ORDER BY subject_id,admittime,hadm_id", ("hadm_id",)))
    tables = sorted({s.table for s in result})
    conditions = " OR ".join(f"(table_schema='{t.split('.')[0]}' AND table_name='{t.split('.')[1]}')" for t in tables)
    result.append(Source("metadata/columns", "information_schema.columns",
                         "SELECT table_schema,table_name,column_name,ordinal_position,data_type,udt_name,is_nullable FROM information_schema.columns WHERE " + conditions + " ORDER BY table_schema,table_name,ordinal_position"))
    return tuple(result)


def rows(path: Path) -> Iterator[dict[str, str]]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_split(subject: str) -> str:
    bucket = int(hashlib.md5(f"aecopd-model-v1:{subject}".encode("ascii")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def validate(root: Path) -> dict[str, object]:
    """独立读取落盘文件检查成员、主键和关联；不输出患者正文。"""
    cohort = {r["hadm_id"]: r for r in rows(root / "cohort/admissions.csv.gz")}
    eligible = {h for h, r in cohort.items() if r["discharge_landmark_eligible"] == "t"}
    subjects = {r["subject_id"] for r in cohort.values()}
    if (len(cohort), len(subjects), len(eligible)) != (19774, 5390, 19401):
        raise ValueError("既定队列数量不一致，停止导出验证")
    actual_patients = {r["subject_id"] for r in rows(root / "raw/patients.csv.gz")}
    actual_admissions = {r["hadm_id"] for r in rows(root / "raw/admissions.csv.gz")}
    actual_labels = {r["hadm_id"] for r in rows(root / "outcomes/readmission_death_labels.csv.gz")}
    if actual_patients != subjects or actual_admissions != set(cohort) or actual_labels != eligible:
        raise ValueError("患者、住院或结局成员不匹配")
    emar_keys = {(r["emar_id"], r["emar_seq"]) for r in rows(root / "raw/emar.csv.gz")}
    counts: dict[str, int] = {}
    for source in sources():
        seen: set[tuple[str, ...]] = set()
        count = 0
        for row in rows(root / (source.name + ".csv.gz")):
            count += 1
            if source.primary_key:
                key = tuple(row[k] for k in source.primary_key)
                if key in seen or NULL in key:
                    raise ValueError(f"{source.name}: 原始记录主键重复或为空")
                seen.add(key)
            if source.name == "raw/emar_detail" and (row["emar_id"], row["emar_seq"]) not in emar_keys:
                raise ValueError("给药详情缺少原始给药关联")
            if source.name.startswith("raw/") and "subject_id" in row:
                if row["subject_id"] not in subjects:
                    raise ValueError(f"{source.name}: 患者超出范围")
                hadm = row.get("hadm_id", NULL)
                if hadm != NULL and (hadm not in cohort or cohort[hadm]["subject_id"] != row["subject_id"]):
                    raise ValueError(f"{source.name}: 住院与患者关联不一致")
        counts[source.name] = count
    split_path = root / "cohort/patient_splits.csv.gz"
    with gzip.open(split_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subject_id", "patient_split", "has_eligible_discharge"))
        eligible_subjects = {cohort[h]["subject_id"] for h in eligible}
        for subject in sorted(subjects, key=int):
            writer.writerow((subject, patient_split(subject), subject in eligible_subjects))
    return {"status": "validated", "patients": len(subjects), "coded_admissions": len(cohort),
            "eligible_discharges": len(eligible), "eligible_patients": len(eligible_subjects),
            "non_landmark_admissions": len(cohort) - len(eligible), "rows": counts}


def export(root: Path, psql: Path, password: str) -> dict[str, object]:
    if root.exists():
        raise FileExistsError("拒绝覆盖已有患者数据目录")
    if not psql.is_file():
        raise FileNotFoundError("找不到 PostgreSQL 导出客户端")
    root.mkdir(parents=True)
    (root / "metadata").mkdir()
    (root / "metadata/status.json").write_text(json.dumps({"status": "exporting"}), encoding="utf-8")
    env = os.environ.copy()
    env["PGPASSWORD"] = password
    env["PGOPTIONS"] = "-c default_transaction_read_only=on -c statement_timeout=600000"
    command = [str(psql), "-X", "-q", "-w", "-h", "localhost", "-U", "postgres",
               "-d", "mimiciv", "-v", "ON_ERROR_STOP=1"]
    keeper = subprocess.Popen(command + ["-A", "-t"], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=env)
    assert keeper.stdin is not None and keeper.stdout is not None
    keeper.stdin.write("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;\nSELECT pg_export_snapshot();\n")
    keeper.stdin.flush()
    snapshot = keeper.stdout.readline().strip()
    if not snapshot or any(c not in "0123456789ABCDEFabcdef-" for c in snapshot):
        keeper.terminate()
        raise RuntimeError("无法建立一致的只读导出快照")
    manifest: dict[str, object] = {"status": "exporting", "database": "mimiciv", "cohort_source": COHORT,
                                  "snapshot": snapshot, "utc_started": datetime.now(timezone.utc).isoformat(),
                                  "format": "UTF-8 PostgreSQL COPY CSV gzip", "null_token": NULL,
                                  "sources": {}}
    try:
        for source in sources():
            target = root / (source.name + ".csv.gz")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".partial")
            copy_sql = f"COPY ({source.select}) TO STDOUT WITH (FORMAT CSV, HEADER TRUE, NULL '\\N', ENCODING 'UTF8');"
            sql = f"BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY; SET TRANSACTION SNAPSHOT '{snapshot}'; {copy_sql} COMMIT;"
            # 持久化查询不包含导出结束后失效的事务快照 ID。
            (root / "metadata" / (source.name.replace("/", "_") + ".sql")).write_text(copy_sql, encoding="utf-8")
            with gzip.open(partial, "wb", compresslevel=3) as destination:
                process = subprocess.Popen(command + ["-c", sql], stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, env=env)
                assert process.stdout is not None
                for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
                    destination.write(block)
                error = process.stderr.read() if process.stderr is not None else b""
                if process.wait():
                    raise RuntimeError(f"{source.name} 导出失败: {error.decode('utf-8', errors='replace')}")
            partial.replace(target)
            manifest["sources"][source.name] = {"source_table": source.table, "sha256": sha256(target),
                                                  "compressed_bytes": target.stat().st_size}
            print(json.dumps({"exported": source.name, "compressed_bytes": target.stat().st_size}), flush=True)
        checked = validate(root)
        (root / "metadata/validation.json").write_text(json.dumps(checked, indent=2), encoding="utf-8")
        for name, count in checked["rows"].items():
            manifest["sources"][name]["rows"] = count
        manifest["sources"]["cohort/patient_splits"] = {"rows": checked["patients"], "sha256": sha256(root / "cohort/patient_splits.csv.gz")}
        manifest.update(status="validated", utc_finished=datetime.now(timezone.utc).isoformat())
        (root / "metadata/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (root / "metadata/status.json").write_text(json.dumps({"status": "validated"}), encoding="utf-8")
        return checked
    except Exception:
        (root / "metadata/status.json").write_text(json.dumps({"status": "failed_preserved_for_audit"}), encoding="utf-8")
        raise
    finally:
        keeper.communicate("ROLLBACK;\n", timeout=20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--psql", type=Path, default=Path("C:/Program Files/PostgreSQL/18/bin/psql.exe"))
    args = parser.parse_args()
    password = os.environ.get("PGPASSWORD") or getpass.getpass("PostgreSQL password: ")
    print(json.dumps(export(args.output, args.psql, password), ensure_ascii=False))


if __name__ == "__main__":
    main()
