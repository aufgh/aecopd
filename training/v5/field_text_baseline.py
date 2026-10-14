"""生成独立字段文本分支的共同 BERT／Qwen 输入，不构造章节—结构化配对。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterator, Protocol

from training.v5.section_alignment_data.screen_section_content import screen_flags


FIELD_LABELS = {
    "history_of_present_illness": "History of Present Illness",
    "past_medical_history": "Past Medical History",
    "hospital_course": "Hospital Course",
    "discharge_condition": "Discharge Condition",
}
FIELDS = (
    "embedding_row", "subject_id", "hadm_id", "field", "section_id", "chunk_index",
    "source_manifest_row", "source_start_char", "source_end_char", "source_section_sha256",
    "timestamp_relation", "bert_token_count", "token_count", "source_chunk_sha256",
    "text_sha256", "chunk_text",
)


class Tokenizer(Protocol):
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]: ...


def _rows(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def field_text(role: str, raw_chunk: str) -> str:
    """字段名只是文本上下文，既不声称有结构化证据，也不加入结局信息。"""

    if role not in FIELD_LABELS or not raw_chunk.strip():
        raise ValueError("unsupported or empty field text")
    return f"Clinical field: {FIELD_LABELS[role]}\n{raw_chunk}"


def eligible(role: str, original: str, timestamp_relation: str) -> tuple[bool, str]:
    """保守筛查整段来源；不把标记解释为临床真值。"""

    if role not in FIELD_LABELS:
        return False, "outside_first_baseline"
    flags = set(screen_flags(role, original, timestamp_relation))
    if role == "history_of_present_illness" and "administrative_header_first_500_chars" in flags:
        return False, "hpi_administrative_header_review"
    if role == "discharge_condition":
        if "condition_over_500_chars_review_boundary" in flags:
            return False, "long_condition_boundary_review"
        if "generic_condition_only" in flags:
            return False, "generic_condition_only"
        if "function_or_respiratory_cue" not in flags:
            return False, "condition_without_selected_cue"
    return True, "screened_candidate_not_clinically_validated"


def prepare(
    aligned_dir: Path, candidate_dir: Path, output_dir: Path,
    bert_tokenizer: Tokenizer, qwen_tokenizer: Tokenizer,
) -> dict[str, object]:
    """同一字段清单给两个冻结编码器；拒绝覆盖、超长和未校验来源。"""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    aligned_qa = json.loads((aligned_dir / "qa_validation.json").read_text(encoding="utf-8"))
    candidate_qa = json.loads((candidate_dir / "validation.json").read_text(encoding="utf-8"))
    source_sections = aligned_dir / "note_sections.csv.gz"
    source_manifest = candidate_dir / "shared_text.csv.gz"
    if aligned_qa["status"] != "structural_checks_passed" or candidate_qa["status"] != "manifest_structurally_validated":
        raise ValueError("unvalidated source data")
    if _sha(source_sections) != aligned_qa["output_sha256"][source_sections.name]:
        raise ValueError("source sections changed")
    if _sha(source_manifest) != candidate_qa["manifest_sha256"]:
        raise ValueError("source chunk manifest changed")

    section_status: dict[str, tuple[bool, str]] = {}
    for row in _rows(source_sections):
        if row["section_role"] not in FIELD_LABELS:
            continue
        section_status[row["section_id"]] = eligible(
            row["section_role"], row["text"], row["timestamp_relation"]
        )
    output_dir.mkdir(parents=True)
    counts: Counter[str] = Counter()
    covered_hadms: set[str] = set()
    emitted_sections: set[str] = set()
    max_bert = max_qwen = 0
    with gzip.open(output_dir / "shared_text.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for source_row, row in enumerate(_rows(source_manifest)):
            role = row["section_role"]
            if role not in FIELD_LABELS:
                continue
            if int(row["embedding_row"]) != source_row:
                raise ValueError("non-contiguous source manifest rows")
            status = section_status.get(row["section_id"])
            if status is None:
                raise ValueError("source chunk references unknown field section")
            if not status[0]:
                counts["excluded_" + status[1]] += 1
                continue
            raw = row["chunk_text"]
            if hashlib.sha256(raw.encode("utf-8")).hexdigest() != row["text_sha256"]:
                raise ValueError("source chunk content hash mismatch")
            text = field_text(role, raw)
            bert_count = len(bert_tokenizer(text, add_special_tokens=True)["input_ids"])
            qwen_count = len(qwen_tokenizer(text, add_special_tokens=False)["input_ids"])
            if not 1 <= bert_count <= 512 or not 1 <= qwen_count <= 8192:
                raise ValueError("field text exceeds frozen encoder context")
            writer.writerow((
                counts["chunks"], row["subject_id"], row["hadm_id"], role,
                row["section_id"], row["chunk_index"], source_row,
                row["source_start_char"], row["source_end_char"],
                row["source_section_sha256"], row["timestamp_relation"],
                bert_count, qwen_count, row["text_sha256"],
                hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
            ))
            counts["chunks"] += 1
            counts["chunks_" + role] += 1
            covered_hadms.add(row["hadm_id"])
            emitted_sections.add(row["section_id"])
            max_bert = max(max_bert, bert_count)
            max_qwen = max(max_qwen, qwen_count)
    if not counts["chunks"]:
        raise ValueError("no field text selected")
    manifest = output_dir / "shared_text.csv.gz"
    result: dict[str, object] = {
        "status": "screened_candidate_not_encoded",
        "cohort_source_sections_sha256": _sha(source_sections),
        "source_candidate_manifest_sha256": _sha(source_manifest),
        "manifest_sha256": _sha(manifest),
        "fields": list(FIELD_LABELS),
        "admissions_with_text": len(covered_hadms),
        "source_sections": len(emitted_sections),
        "chunks": counts["chunks"],
        "chunks_by_field": {role: counts["chunks_" + role] for role in FIELD_LABELS},
        "excluded_chunks_by_reason": {
            key.removeprefix("excluded_"): value
            for key, value in sorted(counts.items()) if key.startswith("excluded_")
        },
        "max_bert_tokens": max_bert,
        "max_qwen_tokens": max_qwen,
        "clinical_semantics_validated": False,
        "outcome_labels_exported": False,
        "paired_structured_facts_assumed": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bert-model-dir", type=Path, default=Path("/model-cache/Bio_ClinicalBERT"))
    parser.add_argument("--qwen-model-dir", type=Path, default=Path("/model-cache/Qwen3-Embedding-4B"))
    args = parser.parse_args()
    bert = AutoTokenizer.from_pretrained(args.bert_model_dir, local_files_only=True)
    qwen = AutoTokenizer.from_pretrained(args.qwen_model_dir, local_files_only=True)
    print(json.dumps(prepare(args.aligned_dir, args.candidate_dir, args.output_dir, bert, qwen), ensure_ascii=False))


if __name__ == "__main__":
    main()
