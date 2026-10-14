"""对 MIMIC-IV 首次编码索引的完整出院小结运行规则候选抽取。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from training.clinical_discharge_rules_v2 import FIELDS, RULE_VERSION, extract_mentions, text_units


def extract(source: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    database = output_dir / "rule_evidence.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript("""
        CREATE TABLE documents (
            hadm_id INTEGER PRIMARY KEY, subject_id INTEGER NOT NULL,
            timestamp_relation TEXT NOT NULL, source_sha256 TEXT NOT NULL,
            source_chars INTEGER NOT NULL, units INTEGER NOT NULL,
            candidate_mentions INTEGER NOT NULL, eligible_mentions INTEGER NOT NULL
        );
        CREATE TABLE mentions (
            mention_id INTEGER PRIMARY KEY, hadm_id INTEGER NOT NULL REFERENCES documents(hadm_id),
            field TEXT NOT NULL, section TEXT NOT NULL, start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL, evidence_text TEXT NOT NULL,
            pattern_ids_json TEXT NOT NULL, temporality TEXT NOT NULL,
            assertion TEXT NOT NULL, eligible INTEGER NOT NULL,
            exclusion_reason TEXT NOT NULL
        );
        CREATE INDEX mentions_by_document_field ON mentions(hadm_id,field);
        CREATE INDEX mentions_by_field_eligible ON mentions(field,eligible);
    """)
    counts: Counter[str] = Counter()
    eligible_counts: Counter[str] = Counter()
    time_counts: Counter[str] = Counter()
    section_counts: Counter[str] = Counter()
    document_any: Counter[str] = Counter()
    document_eligible: Counter[str] = Counter()
    seen: set[int] = set()
    try:
        with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"subject_id", "hadm_id", "timestamp_relation", "document_text"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("source text export contract is incomplete")
            for row in reader:
                hadm = int(row["hadm_id"])
                if hadm in seen:
                    raise ValueError("duplicate admission in source export")
                seen.add(hadm)
                note = row["document_text"]
                units = text_units(note)
                mentions = extract_mentions(note)
                with connection:
                    connection.execute(
                        "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)",
                        (hadm, int(row["subject_id"]), row["timestamp_relation"],
                         hashlib.sha256(note.encode("utf-8")).hexdigest(), len(note), len(units),
                         len(mentions), sum(item.eligible for item in mentions)),
                    )
                    connection.executemany("""
                        INSERT INTO mentions (
                            hadm_id,field,section,start_line,end_line,evidence_text,
                            pattern_ids_json,temporality,assertion,eligible,exclusion_reason
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        (hadm, item.field, item.section, item.start_line, item.end_line,
                         item.text, json.dumps(item.pattern_ids), item.temporality,
                         item.assertion, int(item.eligible), item.exclusion_reason)
                        for item in mentions
                    ))
                fields_any = {item.field for item in mentions}
                fields_eligible = {item.field for item in mentions if item.eligible}
                for field in FIELDS:
                    document_any[field] += int(field in fields_any)
                    document_eligible[field] += int(field in fields_eligible)
                for item in mentions:
                    counts[item.field] += 1
                    eligible_counts[item.field] += int(item.eligible)
                    time_counts[item.temporality] += 1
                    section_counts[item.section] += 1
    finally:
        connection.close()
    summary: dict[str, object] = {
        "rule_version": RULE_VERSION,
        "source": str(source.resolve()),
        "documents": len(seen),
        "candidate_mentions": sum(counts.values()),
        "eligible_mentions": sum(eligible_counts.values()),
        "candidate_mentions_by_field": dict(counts),
        "eligible_mentions_by_field": dict(eligible_counts),
        "documents_with_any_candidate_by_field": dict(document_any),
        "documents_with_eligible_candidate_by_field": dict(document_eligible),
        "temporality": dict(time_counts),
        "top_sections": dict(section_counts.most_common(20)),
        "unmatched_means_unknown_not_negative": True,
        "clinical_recall_validated": False,
        "model_training_ready": False,
        "outcome_labels_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.source, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
