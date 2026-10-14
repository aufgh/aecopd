"""Exact-offset tokenization for section-level Qwen documents."""

from __future__ import annotations

import csv
import gzip
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from .section_dataset import SECTION_FIELDS


class FastTokenizer(Protocol):
    is_fast: bool

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        truncation: bool,
    ) -> dict[str, Any]: ...


DOCUMENT_METADATA_FIELDS = (
    "document_key",
    "source_document_key",
    "document_role",
    "modality",
    "section_role",
    "section_order",
    "section_instance",
    "qwen_token_count",
    "chunk_count",
    "max_chunk_tokens",
    "processing_status",
)

CHUNK_FIELDS = (
    "document_key",
    "source_document_key",
    "subject_id",
    "hadm_id",
    "source_hadm_id",
    "data_split",
    "document_role",
    "document_rank",
    "days_before_landmark",
    "note_type",
    "charttime",
    "storetime",
    "modality",
    "section_role",
    "section_order",
    "section_instance",
    "chunk_index",
    "token_start",
    "token_end",
    "char_start",
    "char_end",
    "token_count",
    "chunk_text",
)


def _text_slice(
    offsets: list[tuple[int, int]],
    start: int,
    end: int,
    text: str,
) -> tuple[int, int, str]:
    usable = [(char_start, char_end) for char_start, char_end in offsets[start:end] if char_end > char_start]
    if not usable:
        return 0, 0, ""
    char_start = usable[0][0]
    char_end = usable[-1][1]
    return char_start, char_end, text[char_start:char_end]


def _windows(
    token_count: int,
    whole_document_limit: int,
    long_document_chunk_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, int]]:
    if token_count <= whole_document_limit:
        return [(0, token_count)]
    windows: list[tuple[int, int]] = []
    start = 0
    while start < token_count:
        end = min(start + long_document_chunk_tokens, token_count)
        windows.append((start, end))
        if end == token_count:
            break
        start = end - overlap_tokens
    return windows


def tokenize_section_documents(
    *,
    source_path: Path,
    chunks_path: Path,
    document_metadata_path: Path,
    chunk_metadata_path: Path,
    summary_path: Path,
    tokenizer: FastTokenizer,
    model_name: str,
    whole_document_limit: int = 32_000,
    long_document_chunk_tokens: int = 16_000,
    overlap_tokens: int = 512,
    max_documents: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Tokenize section documents and emit chunks accepted by the frozen encoder."""

    if whole_document_limit <= 0 or long_document_chunk_tokens <= 0:
        raise ValueError("Token limits must be positive")
    if overlap_tokens < 0 or overlap_tokens >= long_document_chunk_tokens:
        raise ValueError("overlap_tokens must be non-negative and smaller than the chunk size")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive when provided")
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for exact character offsets")

    destinations = (chunks_path, document_metadata_path, chunk_metadata_path, summary_path)
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {existing[0]}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex
    partials = {
        path: path.with_name(f".{path.name}.{run_id}.partial") for path in destinations
    }
    section_documents = 0
    chunk_count = 0
    documents_over_limit = 0
    max_document_tokens = 0
    role_documents: Counter[str] = Counter()
    role_chunks: Counter[str] = Counter()
    section_roles: Counter[str] = Counter()
    modalities: Counter[str] = Counter()

    try:
        with gzip.open(source_path, "rt", encoding="utf-8-sig", newline="") as source, \
                gzip.open(partials[chunks_path], "wt", encoding="utf-8", newline="", compresslevel=6) as chunk_file, \
                partials[document_metadata_path].open("w", encoding="utf-8", newline="") as document_metadata_file, \
                partials[chunk_metadata_path].open("w", encoding="utf-8", newline="") as chunk_metadata_file:
            reader = csv.DictReader(source)
            missing = [field for field in SECTION_FIELDS if field not in (reader.fieldnames or ())]
            if missing:
                raise ValueError(f"Section document table is missing fields: {', '.join(missing)}")
            chunk_writer = csv.DictWriter(chunk_file, fieldnames=CHUNK_FIELDS)
            document_writer = csv.DictWriter(document_metadata_file, fieldnames=DOCUMENT_METADATA_FIELDS)
            chunk_metadata_fields = tuple(field for field in CHUNK_FIELDS if field != "chunk_text")
            chunk_metadata_writer = csv.DictWriter(chunk_metadata_file, fieldnames=chunk_metadata_fields)
            chunk_writer.writeheader()
            document_writer.writeheader()
            chunk_metadata_writer.writeheader()

            for row in reader:
                if max_documents is not None and section_documents >= max_documents:
                    break
                text = row["document_text"] or ""
                encoded = tokenizer(
                    text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    truncation=False,
                )
                token_ids = encoded["input_ids"]
                offsets = encoded["offset_mapping"]
                if len(token_ids) != len(offsets):
                    raise RuntimeError(f"Tokenizer returned inconsistent offsets for {row['document_key']}")
                token_count = len(token_ids)
                windows = _windows(
                    token_count,
                    whole_document_limit,
                    long_document_chunk_tokens,
                    overlap_tokens,
                )
                over_limit = token_count > whole_document_limit

                section_documents += 1
                documents_over_limit += int(over_limit)
                max_document_tokens = max(max_document_tokens, token_count)
                role_documents[row["document_role"]] += 1
                section_roles[row["section_role"]] += 1
                modalities[row["modality"]] += 1
                max_chunk_tokens = 0

                for chunk_index, (token_start, token_end) in enumerate(windows):
                    char_start, char_end, chunk_text = _text_slice(
                        offsets,
                        token_start,
                        token_end,
                        text,
                    )
                    tokens_in_chunk = token_end - token_start
                    max_chunk_tokens = max(max_chunk_tokens, tokens_in_chunk)
                    chunk_row = {
                        **{field: row[field] for field in CHUNK_FIELDS if field in row},
                        "chunk_index": chunk_index,
                        "token_start": token_start,
                        "token_end": token_end,
                        "char_start": char_start,
                        "char_end": char_end,
                        "token_count": tokens_in_chunk,
                        "chunk_text": chunk_text,
                    }
                    chunk_writer.writerow(chunk_row)
                    chunk_metadata_writer.writerow(
                        {field: chunk_row[field] for field in chunk_metadata_fields}
                    )
                    chunk_count += 1
                    role_chunks[row["document_role"]] += 1

                document_writer.writerow(
                    {
                        "document_key": row["document_key"],
                        "source_document_key": row["source_document_key"],
                        "document_role": row["document_role"],
                        "modality": row["modality"],
                        "section_role": row["section_role"],
                        "section_order": row["section_order"],
                        "section_instance": row["section_instance"],
                        "qwen_token_count": token_count,
                        "chunk_count": len(windows),
                        "max_chunk_tokens": max_chunk_tokens,
                        "processing_status": "READY_FOR_QWEN",
                    }
                )

        summary: dict[str, object] = {
            "schema_version": "section_chunks_v2",
            "model": model_name,
            "whole_document_limit": whole_document_limit,
            "long_document_chunk_tokens": long_document_chunk_tokens,
            "overlap_tokens": overlap_tokens,
            "section_documents": section_documents,
            "chunks": chunk_count,
            "documents_over_limit": documents_over_limit,
            "max_document_tokens": max_document_tokens,
            "document_roles": {
                role: {
                    "documents": role_documents[role],
                    "chunks": role_chunks[role],
                }
                for role in sorted(role_documents)
            },
            "section_roles": {key: section_roles[key] for key in sorted(section_roles)},
            "modalities": {key: modalities[key] for key in sorted(modalities)},
            "source_path": str(source_path),
            "chunks_path": str(chunks_path),
            "max_documents": max_documents,
        }
        partials[summary_path].write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for destination, partial in partials.items():
            os.replace(partial, destination)
        return summary
    finally:
        for partial in partials.values():
            partial.unlink(missing_ok=True)
