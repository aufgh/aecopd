"""核验规则证据与原文，并为两种冻结编码器制作同一份输入清单。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path


ALLOWED_TIMES = frozenset({"discharge_state", "discharge_plan"})
FUTURE_FACT = re.compile(
    r"\b(?:readmitted|re-admitted|returned to (?:the )?hospital|"
    r"died|expired)\s+(?:after|following|since)\s+discharge\b",
    re.IGNORECASE,
)


def normalized(text: str) -> str:
    return " ".join(text.split())


def prepare(source: Path, database: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        documents = {
            int(hadm): (int(subject), str(digest))
            for hadm, subject, digest in connection.execute(
                "SELECT hadm_id,subject_id,source_sha256 FROM documents"
            )
        }
        mentions = connection.execute("""
            SELECT mention_id,hadm_id,field,start_line,end_line,evidence_text,
                   temporality,assertion
            FROM mentions WHERE eligible=1
            ORDER BY hadm_id,field,start_line,end_line,mention_id
        """).fetchall()
    finally:
        connection.close()
    by_hadm: dict[int, list[tuple[object, ...]]] = {}
    for mention in mentions:
        by_hadm.setdefault(int(mention[1]), []).append(mention)
    notes_seen: set[int] = set()
    selected: list[tuple[object, ...]] = []
    excluded: Counter[str] = Counter()
    dedup: set[tuple[int, str, str]] = set()
    with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as handle:
        for note in csv.DictReader(handle):
            hadm = int(note["hadm_id"])
            if hadm in notes_seen or hadm not in documents:
                raise ValueError("duplicate or unexpected source admission")
            notes_seen.add(hadm)
            subject, digest = documents[hadm]
            raw = note["document_text"]
            if int(note["subject_id"]) != subject or hashlib.sha256(raw.encode()).hexdigest() != digest:
                raise ValueError("source note changed since rule extraction")
            lines = raw.splitlines()
            for mention_id, _, field, first, last, evidence, temporality, assertion in by_hadm.get(hadm, []):
                first, last = int(first), int(last)
                if not (1 <= first <= last <= len(lines)):
                    raise ValueError("invalid evidence line range")
                if normalized(str(evidence)) not in normalized(" ".join(lines[first - 1:last])):
                    raise ValueError("rule evidence is not verbatim source text")
                if temporality not in ALLOWED_TIMES:
                    excluded["non_discharge_time"] += 1
                    continue
                if FUTURE_FACT.search(str(evidence)):
                    excluded["future_fact"] += 1
                    continue
                key = (hadm, str(field), normalized(str(evidence)))
                if key in dedup:
                    excluded["duplicate_same_field"] += 1
                    continue
                dedup.add(key)
                selected.append((int(mention_id), subject, hadm, str(field), first, last,
                                 str(temporality), str(assertion), str(evidence)))
    if notes_seen != set(documents):
        raise ValueError("rule documents and source export have different admissions")
    if not selected:
        raise ValueError("no verified discharge evidence")
    output_dir.mkdir(parents=True)
    with gzip.open(output_dir / "shared_text.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("embedding_row", "mention_id", "subject_id", "hadm_id", "field",
                         "start_line", "end_line", "temporality", "assertion",
                         "text_sha256", "token_count", "chunk_text"))
        for row_number, (mention_id, subject, hadm, field, first, last, time, assertion, evidence) in enumerate(selected):
            # 仅供 Qwen 的批次打包预估；两模型真正分词长度在编码时检查。
            estimated_tokens = max(1, len(evidence.encode("utf-8")) // 3)
            writer.writerow((row_number, mention_id, subject, hadm, field, first, last,
                             time, assertion, hashlib.sha256(evidence.encode()).hexdigest(),
                             estimated_tokens, evidence))
    report: dict[str, object] = {
        "source_documents": len(documents), "original_eligible_mentions": len(mentions),
        "shared_text_rows": len(selected), "admissions_with_text": len({row[2] for row in selected}),
        "excluded": dict(excluded), "fields": dict(Counter(row[3] for row in selected)),
        "sha256_shared_text_file": hashlib.sha256((output_dir / "shared_text.csv.gz").read_bytes()).hexdigest(),
        "clinical_accuracy_validated": False, "outcome_labels_used": False,
        "purpose": "exploratory same-input Qwen/Bio_ClinicalBERT comparison",
    }
    (output_dir / "manifest_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.database, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
