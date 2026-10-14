"""独立核对字段文本清单：来源、筛查、行号、原文和两种分词器长度。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

from training.v5.field_text_baseline import FIELD_LABELS, FIELDS, eligible, field_text


def _rows(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if path.name == "shared_text.csv.gz" and set(FIELDS).issubset(reader.fieldnames or ()):
            if {"event_type", "time_days"} & set(reader.fieldnames or ()):
                raise ValueError("outcome labels present in text manifest")
        yield from reader


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(
    aligned_dir: Path, candidate_dir: Path, field_dir: Path,
    bert_tokenizer: object, qwen_tokenizer: object,
) -> dict[str, object]:
    """每个保留块必须来自已校验的源清单，且前缀只能由字段名生成。"""

    summary = json.loads((field_dir / "summary.json").read_text(encoding="utf-8"))
    source_path = candidate_dir / "shared_text.csv.gz"
    output_path = field_dir / "shared_text.csv.gz"
    if _sha(source_path) != summary["source_candidate_manifest_sha256"]:
        raise ValueError("source manifest fingerprint changed")
    if _sha(output_path) != summary["manifest_sha256"]:
        raise ValueError("field manifest fingerprint changed")
    section_path = aligned_dir / "note_sections.csv.gz"
    if _sha(section_path) != summary["cohort_source_sections_sha256"]:
        raise ValueError("source section fingerprint changed")
    sections: dict[str, dict[str, str]] = {}
    for row in _rows(section_path):
        if row["section_role"] in FIELD_LABELS:
            sections[row["section_id"]] = row
    output = iter(_rows(output_path))
    next_row = next(output, None)
    counts: Counter[str] = Counter()
    admissions: set[str] = set()
    included_sections: set[str] = set()
    excluded: Counter[str] = Counter()
    for source_number, source in enumerate(_rows(source_path)):
        role = source["section_role"]
        if role not in FIELD_LABELS:
            continue
        section = sections.get(source["section_id"])
        if section is None or section["section_role"] != role:
            raise ValueError("source field section mismatch")
        keep, reason = eligible(role, section["text"], section["timestamp_relation"])
        if not keep:
            excluded[reason] += 1
            if next_row is not None and int(next_row["source_manifest_row"]) == source_number:
                raise ValueError("excluded section found in field manifest")
            continue
        if next_row is None or int(next_row["source_manifest_row"]) != source_number:
            raise ValueError("eligible source chunk omitted or out of order")
        row = next_row
        if int(row["embedding_row"]) != counts["total"]:
            raise ValueError("non-contiguous field embedding rows")
        for field, source_field in (
            ("subject_id", "subject_id"), ("hadm_id", "hadm_id"),
            ("field", "section_role"), ("section_id", "section_id"),
            ("chunk_index", "chunk_index"), ("source_start_char", "source_start_char"),
            ("source_end_char", "source_end_char"),
            ("source_section_sha256", "source_section_sha256"),
            ("timestamp_relation", "timestamp_relation"),
        ):
            if row[field] != source[source_field]:
                raise ValueError(f"field text metadata mismatch: {field}")
        raw = source["chunk_text"]
        start = int(source["source_start_char"]) - int(section["start_char"])
        end = int(source["source_end_char"]) - int(section["start_char"])
        if not 0 <= start < end <= len(section["text"]) or section["text"][start:end] != raw:
            raise ValueError("raw chunk cannot be recovered from source section")
        if row["source_chunk_sha256"] != hashlib.sha256(raw.encode("utf-8")).hexdigest():
            raise ValueError("raw chunk fingerprint mismatch")
        expected = field_text(role, raw)
        if row["chunk_text"] != expected or row["text_sha256"] != hashlib.sha256(expected.encode("utf-8")).hexdigest():
            raise ValueError("field prefix or content differs from source")
        bert_count = len(bert_tokenizer(expected, add_special_tokens=True)["input_ids"])
        qwen_count = len(qwen_tokenizer(expected, add_special_tokens=False)["input_ids"])
        if bert_count != int(row["bert_token_count"]) or qwen_count != int(row["token_count"]):
            raise ValueError("encoder length metadata mismatch")
        if bert_count > 512 or qwen_count > 8192:
            raise ValueError("encoder context exceeded")
        counts["total"] += 1
        counts[role] += 1
        admissions.add(row["hadm_id"])
        included_sections.add(row["section_id"])
        next_row = next(output, None)
    if next_row is not None or counts["total"] != summary["chunks"]:
        raise ValueError("field manifest has extra rows or count drift")
    if len(admissions) != summary["admissions_with_text"] or len(included_sections) != summary["source_sections"]:
        raise ValueError("coverage count drift")
    if {role: counts[role] for role in FIELD_LABELS} != summary["chunks_by_field"]:
        raise ValueError("field distribution drift")
    if dict(sorted(excluded.items())) != summary["excluded_chunks_by_reason"]:
        raise ValueError("screening decision drift")
    result: dict[str, object] = {
        "status": "field_text_manifest_validated",
        "manifest_sha256": summary["manifest_sha256"],
        "chunks": counts["total"], "source_sections": len(included_sections),
        "admissions_with_text": len(admissions),
        "same_rows_for_bert_and_qwen": True,
        "clinical_semantics_review_complete": False,
        "encoding_started": False,
    }
    (field_dir / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--field-dir", type=Path, required=True)
    parser.add_argument("--bert-model-dir", type=Path, default=Path("/model-cache/Bio_ClinicalBERT"))
    parser.add_argument("--qwen-model-dir", type=Path, default=Path("/model-cache/Qwen3-Embedding-4B"))
    args = parser.parse_args()
    bert = AutoTokenizer.from_pretrained(args.bert_model_dir, local_files_only=True)
    qwen = AutoTokenizer.from_pretrained(args.qwen_model_dir, local_files_only=True)
    print(json.dumps(validate(args.aligned_dir, args.candidate_dir, args.field_dir, bert, qwen), ensure_ascii=False))


if __name__ == "__main__":
    main()
