"""从全文规则结果生成本地原文审阅包，特别抽查零命中以检验漏提取。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import sqlite3
from collections import Counter
from pathlib import Path

from training.clinical_discharge_rules_v2 import FIELDS


STRATA = ("eligible", "candidate_only", "no_candidate")


def _rank(field: str, stratum: str, hadm_id: int) -> str:
    return hashlib.sha256(f"rule-review-v4:{field}:{stratum}:{hadm_id}".encode()).hexdigest()


def prepare(source: Path, database: Path, output_dir: Path, *, per_stratum: int = 4) -> dict[str, object]:
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        all_ids = {int(row[0]) for row in connection.execute("SELECT hadm_id FROM documents")}
        any_by_field = {
            field: {int(row[0]) for row in connection.execute(
                "SELECT DISTINCT hadm_id FROM mentions WHERE field=?", (field,)
            )} for field in FIELDS
        }
        eligible_by_field = {
            field: {int(row[0]) for row in connection.execute(
                "SELECT DISTINCT hadm_id FROM mentions WHERE field=? AND eligible=1", (field,)
            )} for field in FIELDS
        }
    finally:
        connection.close()
    selected: list[dict[str, object]] = []
    for field in FIELDS:
        pools = {
            "eligible": eligible_by_field[field],
            "candidate_only": any_by_field[field] - eligible_by_field[field],
            "no_candidate": all_ids - any_by_field[field],
        }
        for stratum in STRATA:
            if len(pools[stratum]) < per_stratum:
                raise ValueError(f"insufficient {field}/{stratum} review candidates")
            for hadm in sorted(pools[stratum], key=lambda number: _rank(field, stratum, number))[:per_stratum]:
                selected.append({"field": field, "stratum": stratum, "hadm_id": hadm})
    selected_ids = {int(row["hadm_id"]) for row in selected}
    notes: dict[int, str] = {}
    with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            hadm = int(row["hadm_id"])
            if hadm in selected_ids:
                notes[hadm] = row["document_text"]
    if set(notes) != selected_ids:
        raise ValueError("review admissions missing from source export")
    output_dir.mkdir(parents=True)
    note_filenames: dict[int, str] = {}
    for index, hadm in enumerate(sorted(selected_ids), 1):
        name = f"N{index:03d}.txt"
        note_filenames[hadm] = name
        (output_dir / name).write_text(
            "\n".join(f"L{line_no}: {line}" for line_no, line in enumerate(notes[hadm].splitlines(), 1)),
            encoding="utf-8",
        )
    manifest_fields = ("review_id", "field", "stratum", "note_file", "hadm_id",
                       "gold_value", "gold_evidence_lines", "gold_time", "reviewer", "review_complete")
    links: list[str] = []
    with (output_dir / "annotations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for index, row in enumerate(selected, 1):
            review_id = f"R{index:03d}"
            note_file = note_filenames[int(row["hadm_id"])]
            writer.writerow({
                "review_id": review_id, "field": row["field"], "stratum": row["stratum"],
                "note_file": note_file, "hadm_id": row["hadm_id"], "gold_value": "",
                "gold_evidence_lines": "", "gold_time": "", "reviewer": "", "review_complete": "0",
            })
            label = html.escape(f"{review_id} · {row['field']} · {row['stratum']}")
            links.append(f'<li><a href="{note_file}">{label}</a></li>')
    index = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<title>出院小结规则审阅</title></head><body><h1>出院小结规则审阅</h1>'
        '<p>本地私有原文；按八个字段 × 三种命中状态分层。'
        '尤其请核对 no_candidate 是否漏掉临床表达。'
        'gold_value 建议填 documented_positive / documented_negative / not_documented / ambiguous；'
        'gold_time 填 discharge / admission / history / hospital_course / unknown。'
        '未完成人工标注前不得报告准确率或召回率。</p><ol>'
        + "".join(links) + '</ol></body></html>'
    )
    (output_dir / "INDEX.html").write_text(index, encoding="utf-8")
    summary: dict[str, object] = {
        "review_tasks": len(selected), "unique_documents": len(selected_ids),
        "strata": dict(Counter(str(row["stratum"]) for row in selected)),
        "outcome_labels_used": False, "gold_annotation_complete": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.database, args.output_dir,
                             per_stratum=args.per_stratum), ensure_ascii=False))


if __name__ == "__main__":
    main()
