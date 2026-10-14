"""将已校验的小结章节切成 BERT/Qwen 共用、可回查原文位置的编码清单。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterator, Protocol, Sequence


PILOT_ROLES = (
    "history_of_present_illness", "past_medical_history", "admission_medications",
    "hospital_course", "discharge_medications", "discharge_condition",
)
FIELDS = (
    "embedding_row", "subject_id", "hadm_id", "section_role", "section_id", "chunk_index",
    "source_start_char", "source_end_char", "source_section_sha256", "timestamp_relation",
    "bert_token_count", "token_count", "text_sha256", "chunk_text",
)


class OffsetTokenizer(Protocol):
    def __call__(self, text: str, *, add_special_tokens: bool,
                 return_offsets_mapping: bool) -> dict[str, Sequence[tuple[int, int]]]: ...


def _rows(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_offsets(
    offsets: Sequence[tuple[int, int]], max_tokens: int, overlap_tokens: int,
) -> tuple[tuple[int, int, int], ...]:
    """返回原片段字符起止与块索引；字符偏移右端不包含。"""

    if max_tokens < 1 or not 0 <= overlap_tokens < max_tokens:
        raise ValueError("invalid token window")
    if not offsets:
        return ()
    chunks: list[tuple[int, int, int]] = []
    start = 0
    while start < len(offsets):
        end = min(start + max_tokens, len(offsets))
        left, right = offsets[start][0], offsets[end - 1][1]
        if not 0 <= left < right:
            raise ValueError("invalid tokenizer character offsets")
        chunks.append((left, right, len(chunks)))
        if end == len(offsets):
            break
        start = end - overlap_tokens
    return tuple(chunks)


def prepare(
    aligned_dir: Path, section_inputs_dir: Path, output_dir: Path,
    bert_tokenizer: OffsetTokenizer, qwen_tokenizer: object,
    *, roles: tuple[str, ...] = PILOT_ROLES, max_tokens: int = 384,
    overlap_tokens: int = 32,
) -> dict[str, object]:
    """同一块原文分别记精确 BERT/Qwen 长度，不做医学语义改写。"""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not roles or len(set(roles)) != len(roles):
        raise ValueError("empty or duplicate section roles")
    aligned_validation = json.loads((aligned_dir / "qa_validation.json").read_text(encoding="utf-8"))
    input_validation = json.loads((section_inputs_dir / "validation.json").read_text(encoding="utf-8"))
    if aligned_validation["status"] != "structural_checks_passed" or input_validation["status"] != "section_inputs_validated":
        raise ValueError("unvalidated source dataset")
    if _sha256(aligned_dir / "note_sections.csv.gz") != aligned_validation["output_sha256"]["note_sections.csv.gz"]:
        raise ValueError("source section file changed")
    if _sha256(section_inputs_dir / "section_slots.csv.gz") != input_validation["output_sha256"]["section_slots.csv.gz"]:
        raise ValueError("section slots changed")
    cohort = {row["hadm_id"]: row["subject_id"]
              for row in _rows(aligned_dir / "cohort_manifest.csv.gz")}
    # 只允许章节级输入实际引用过的片段，避免旁路引入未选中的文书。
    referenced: set[str] = set()
    for row in _rows(section_inputs_dir / "section_slots.csv.gz"):
        if row["section_role"] in roles:
            referenced.update(json.loads(row["text_section_ids"]))

    output_dir.mkdir(parents=True)
    counts: Counter[str] = Counter()
    unique_hadms: set[str] = set()
    max_bert = max_qwen = max_chunks = 0
    source_sections = 0
    with gzip.open(output_dir / "shared_text.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for section in _rows(aligned_dir / "note_sections.csv.gz"):
            role, sid = section["section_role"], section["section_id"]
            if role not in roles:
                continue
            counts["section_total_" + role] += 1
            if sid not in referenced:
                raise ValueError("selected section not referenced by section slot")
            if section["parse_status"] == "uncertain":
                counts["excluded_uncertain_" + role] += 1
                continue
            if section["content_status"] != "informative":
                counts["excluded_placeholder_" + role] += 1
                continue
            text = section["text"]
            tokenized = bert_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = tokenized["offset_mapping"]
            source_sections += 1
            chunks = chunk_offsets(offsets, max_tokens, overlap_tokens)
            if not chunks:
                raise ValueError("informative section tokenized to zero chunks")
            max_chunks = max(max_chunks, len(chunks))
            for left, right, chunk_index in chunks:
                chunk = text[left:right]
                global_left = int(section["start_char"]) + left
                global_right = int(section["start_char"]) + right
                if global_right > int(section["end_char"]) or not chunk.strip():
                    raise ValueError("chunk exceeds source section or is blank")
                bert_count = len(bert_tokenizer(chunk, add_special_tokens=True,
                                                return_offsets_mapping=True)["offset_mapping"])
                qwen_count = len(qwen_tokenizer(chunk, add_special_tokens=False)["input_ids"])
                if bert_count > 512 or qwen_count > 8192:
                    raise ValueError("chunk exceeds encoder context limit")
                row_number = counts["chunks"]
                writer.writerow((row_number, cohort[section["hadm_id"]], section["hadm_id"],
                                 role, sid, chunk_index, global_left, global_right,
                                 section["source_sha256"], section["timestamp_relation"],
                                 bert_count, qwen_count, hashlib.sha256(chunk.encode("utf-8")).hexdigest(), chunk))
                counts["chunks"] += 1
                counts["chunks_" + role] += 1
                unique_hadms.add(section["hadm_id"])
                max_bert = max(max_bert, bert_count)
                max_qwen = max(max_qwen, qwen_count)
            counts["included_" + role] += 1
    if counts["chunks"] == 0:
        raise ValueError("no text chunks selected")
    manifest = output_dir / "shared_text.csv.gz"
    report: dict[str, object] = {
        "status": "candidate_manifest_not_encoded", "roles": roles,
        "source_sections_included": source_sections,
        "admissions_with_text": len(unique_hadms), "chunks": counts["chunks"],
        "max_chunks_per_section": max_chunks, "max_bert_tokens": max_bert,
        "max_qwen_tokens": max_qwen,
        "section_role_counts": {role: {
            "total": counts["section_total_" + role],
            "included": counts["included_" + role],
            "excluded_placeholder": counts["excluded_placeholder_" + role],
            "excluded_uncertain": counts["excluded_uncertain_" + role],
            "chunks": counts["chunks_" + role],
        } for role in roles},
        "bert_chunk_target_tokens": max_tokens,
        "bert_overlap_tokens": overlap_tokens,
        "source_note_sections_sha256": aligned_validation["output_sha256"]["note_sections.csv.gz"],
        "source_section_slots_sha256": input_validation["output_sha256"]["section_slots.csv.gz"],
        "manifest_sha256": _sha256(manifest),
        "outcome_labels_exported": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--section-inputs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bert-model-dir", type=Path, default=Path("/model-cache/Bio_ClinicalBERT"))
    parser.add_argument("--qwen-model-dir", type=Path, default=Path("/model-cache/Qwen3-Embedding-4B"))
    parser.add_argument("--max-bert-tokens", type=int, default=384)
    parser.add_argument("--overlap-tokens", type=int, default=32)
    parser.add_argument("--roles", nargs="+", default=PILOT_ROLES,
                        help="Selected parsed section roles; omit for the original six-role pilot")
    args = parser.parse_args()
    bert = AutoTokenizer.from_pretrained(args.bert_model_dir, local_files_only=True, use_fast=True)
    qwen = AutoTokenizer.from_pretrained(args.qwen_model_dir, local_files_only=True)
    if not bert.is_fast:
        raise ValueError("BERT tokenizer must provide exact character offsets")
    result = prepare(args.aligned_dir, args.section_inputs_dir, args.output_dir, bert, qwen,
                     roles=tuple(args.roles), max_tokens=args.max_bert_tokens,
                     overlap_tokens=args.overlap_tokens)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
