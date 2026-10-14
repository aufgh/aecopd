"""Streaming construction of section-level Qwen input documents."""

from __future__ import annotations

import csv
import gzip
import json
import os
import uuid
from collections import Counter
from pathlib import Path

from .text_sections import build_section_embedding_text, split_clinical_document


SOURCE_FIELDS = (
    "document_key",
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
    "document_text",
)

SECTION_FIELDS = (
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
    "document_text",
)


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def build_section_documents(
    source_path: Path,
    output_path: Path,
    summary_path: Path,
    *,
    max_documents: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Create a section-level document stream without loading note text in memory."""

    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive when provided")
    existing = [path for path in (output_path, summary_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {existing[0]}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    partial_output = output_path.with_name(f".{output_path.name}.{run_id}.partial")
    partial_summary = summary_path.with_name(f".{summary_path.name}.{run_id}.partial")

    source_documents = 0
    section_count = 0
    empty_documents = 0
    max_sections_per_document = 0
    role_documents: Counter[str] = Counter()
    role_sections: Counter[str] = Counter()
    section_roles: Counter[str] = Counter()
    modalities: Counter[str] = Counter()
    seen_section_keys: set[str] = set()

    try:
        with gzip.open(source_path, "rt", encoding="utf-8-sig", newline="") as source, gzip.open(
            partial_output,
            "wt",
            encoding="utf-8",
            newline="",
            compresslevel=6,
        ) as destination:
            reader = csv.DictReader(source)
            missing = [field for field in SOURCE_FIELDS if field not in (reader.fieldnames or ())]
            if missing:
                raise ValueError(f"Source document table is missing fields: {', '.join(missing)}")
            writer = csv.DictWriter(destination, fieldnames=SECTION_FIELDS)
            writer.writeheader()

            for row in reader:
                if max_documents is not None and source_documents >= max_documents:
                    break
                source_documents += 1
                role = row["document_role"]
                role_documents[role] += 1
                parsed = split_clinical_document(role, row["note_type"], row["document_text"])
                modalities[parsed.modality] += 1
                max_sections_per_document = max(max_sections_per_document, len(parsed.sections))
                if not parsed.sections:
                    empty_documents += 1
                    continue

                for section in parsed.sections:
                    section_key = f"{row['document_key']}::section::{section.section_order:03d}"
                    if section_key in seen_section_keys:
                        raise RuntimeError(f"Duplicate section key: {section_key}")
                    seen_section_keys.add(section_key)
                    output_row = {
                        "document_key": section_key,
                        "source_document_key": row["document_key"],
                        "subject_id": row["subject_id"],
                        "hadm_id": row["hadm_id"],
                        "source_hadm_id": row["source_hadm_id"],
                        "data_split": row["data_split"],
                        "document_role": role,
                        "document_rank": row["document_rank"],
                        "days_before_landmark": row["days_before_landmark"],
                        "note_type": row["note_type"],
                        "charttime": row["charttime"],
                        "storetime": row["storetime"],
                        "modality": parsed.modality,
                        "section_role": section.section_role,
                        "section_order": section.section_order,
                        "section_instance": section.section_instance,
                        "document_text": build_section_embedding_text(
                            document_role=role,
                            modality=parsed.modality,
                            section=section,
                            note_type=row["note_type"],
                        ),
                    }
                    writer.writerow(output_row)
                    section_count += 1
                    role_sections[role] += 1
                    section_roles[section.section_role] += 1

        summary: dict[str, object] = {
            "schema_version": "section_documents_v2",
            "source_documents": source_documents,
            "sections": section_count,
            "empty_documents": empty_documents,
            "max_sections_per_document": max_sections_per_document,
            "document_roles": {
                role: {
                    "documents": role_documents[role],
                    "sections": role_sections[role],
                }
                for role in sorted(role_documents)
            },
            "section_roles": _counter_dict(section_roles),
            "modalities": _counter_dict(modalities),
            "source_path": str(source_path),
            "output_path": str(output_path),
            "max_documents": max_documents,
            "temporal_policy": "Document-level charttime/storetime only; note-internal dates are retained as text.",
        }
        partial_summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(partial_output, output_path)
        os.replace(partial_summary, summary_path)
        return summary
    finally:
        partial_output.unlink(missing_ok=True)
        partial_summary.unlink(missing_ok=True)
