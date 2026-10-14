"""在本机生成原文审阅包，不向终端或外部服务输出 MIMIC-IV 文书正文。"""

from __future__ import annotations

import argparse
import csv
import getpass
import html
import io
import os
import subprocess
from collections import Counter
from pathlib import Path


FIELDS = (
    "note_id", "document_type", "cohort_group", "length_band", "text_chars",
    "timestamp_relation", "document_text",
)


def sample_sql() -> str:
    """按来源和文书长度分层；不利用结局标签挑选病例。"""

    return """COPY (
      WITH discharge_one AS MATERIALIZED (
        SELECT n.note_id,n.hadm_id,n.note_seq,n.charttime,n.storetime,n.text,
               c.dischtime,l.hadm_id IS NOT NULL AS strict_index,
               row_number() OVER (PARTITION BY n.hadm_id
                                  ORDER BY n.note_seq,n.charttime,n.note_id) AS admission_note_rank
        FROM project_aecopd.copd_coded_admissions_v1_gate c
        JOIN mimiciv_note.discharge n
          ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
        LEFT JOIN project_aecopd.copd_coded_strict_index_labels_v1 l
          ON l.hadm_id=c.hadm_id
        WHERE c.discharge_landmark_eligible AND n.text IS NOT NULL
          AND btrim(n.text)<>''
      ), radiology_one AS MATERIALIZED (
        SELECT n.note_id,n.hadm_id,n.charttime,n.storetime,n.text,c.dischtime,
               row_number() OVER (PARTITION BY n.hadm_id
                                  ORDER BY n.charttime DESC,n.note_id DESC) AS admission_note_rank
        FROM project_aecopd.copd_coded_strict_index_labels_v1 l
        JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
        JOIN mimiciv_note.radiology n
          ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
        WHERE n.charttime<=c.dischtime AND n.storetime<=c.dischtime
          AND n.text IS NOT NULL AND btrim(n.text)<>''
      ), candidates AS (
        SELECT note_id,'discharge'::text AS document_type,
               CASE WHEN strict_index THEN 'strict_index' ELSE 'other_coded' END
                 AS cohort_group,
               char_length(text) AS text_chars,
               CASE WHEN storetime IS NULL THEN 'unknown'
                    WHEN storetime>dischtime THEN 'after_discharge'
                    ELSE 'by_discharge' END AS timestamp_relation,
               text AS document_text
        FROM discharge_one WHERE admission_note_rank=1
        UNION ALL
        SELECT note_id,'radiology'::text,'strict_index'::text,
               char_length(text),'by_discharge'::text,text
        FROM radiology_one WHERE admission_note_rank=1
      ), length_strata AS (
        SELECT *,ntile(3) OVER (
          PARTITION BY document_type,cohort_group ORDER BY text_chars,note_id
        ) AS length_band
        FROM candidates
      ), ranked AS (
        SELECT *,row_number() OVER (
          PARTITION BY document_type,cohort_group,length_band
          ORDER BY md5('aecopd-text-review-v1:'||note_id)
        ) AS sample_rank
        FROM length_strata
      )
      SELECT note_id,document_type,cohort_group,length_band,text_chars,
             timestamp_relation,document_text
      FROM ranked
      WHERE sample_rank<=CASE
        WHEN document_type='radiology' THEN 4
        WHEN cohort_group='strict_index' THEN 15
        ELSE 5 END
      ORDER BY document_type,cohort_group,length_band,sample_rank
    ) TO STDOUT WITH (FORMAT CSV, HEADER TRUE)"""


def fetch_sample(psql: Path, password: str) -> list[dict[str, str]]:
    environment = os.environ.copy()
    environment["PGPASSWORD"] = password
    result = subprocess.run(
        [str(psql), "-X", "-q", "-w", "-U", "postgres", "-d", "mimiciv",
         "-v", "ON_ERROR_STOP=1", "-c", sample_sql()],
        env=environment, capture_output=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    reader = csv.DictReader(io.StringIO(result.stdout.decode("utf-8-sig"), newline=""))
    if tuple(reader.fieldnames or ()) != FIELDS:
        raise ValueError(f"unexpected review export fields: {reader.fieldnames}")
    rows = list(reader)
    if len(rows) > 72 or len({row["note_id"] for row in rows}) != len(rows):
        raise ValueError("unexpected review sample size or duplicate note")
    return rows


def write_review_pack(rows: list[dict[str, str]], output_dir: Path) -> dict[str, object]:
    """原文只写入本机私有目录；索引页仅含元数据和本地链接。"""

    if not rows or output_dir.exists():
        raise ValueError("review rows must be nonempty and output must not exist")
    partial = output_dir.with_name(output_dir.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    partial.mkdir(parents=True)
    try:
        manifest_rows: list[dict[str, str]] = []
        links: list[str] = []
        for ordinal, row in enumerate(rows, start=1):
            review_id = f"R{ordinal:03d}"
            filename = review_id + ".txt"
            (partial / filename).write_bytes(row["document_text"].encode("utf-8"))
            item = {key: row[key] for key in FIELDS if key != "document_text"}
            item.update({"review_id": review_id, "filename": filename})
            manifest_rows.append(item)
            description = (
                f"{review_id} · {row['document_type']} · {row['cohort_group']} · "
                f"长度层 {row['length_band']} · {row['text_chars']} 字符"
            )
            links.append(f'<li><a href="{filename}">{html.escape(description)}</a></li>')
        with (partial / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(*FIELDS[:-1], "review_id", "filename"))
            writer.writeheader()
            writer.writerows(manifest_rows)
        index_html = (
            '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<title>MIMIC-IV 原文审阅包</title></head><body>'
            '<h1>MIMIC-IV 原文审阅包</h1>'
            '<p>仅供已获授权的本地审阅。请勿转发或上传到在线模型。'
            '出院小结按新的“出院时状态总结”假设抽样；影像报告仍限出院前。</p>'
            '<p>按长度分层；strict_index 为当前严格 AECOPD 索引住院，'
            'other_coded 为其他保留 COPD 编码住院。未用未来结局挑选病例。'
            '每个链接打开一篇未经改写的完整原文。</p><ol>'
            + "".join(links) + '</ol></body></html>'
        )
        (partial / "INDEX.html").write_text(index_html, encoding="utf-8")
        partial.rename(output_dir)
    except BaseException:
        # 保留部分目录以便检查；避免隐式删除可能已写入的受控正文。
        raise
    counts = Counter((row["document_type"], row["cohort_group"]) for row in rows)
    return {
        "documents": len(rows),
        "counts": {f"{kind}/{group}": count for (kind, group), count in sorted(counts.items())},
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--psql", type=Path, default=Path("C:/Program Files/PostgreSQL/18/bin/psql.exe"))
    args = parser.parse_args()
    password = getpass.getpass("PostgreSQL password (not saved): ")
    result = write_review_pack(fetch_sample(args.psql, password), args.output_dir)
    print(result)


if __name__ == "__main__":
    main()
