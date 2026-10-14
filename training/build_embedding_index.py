from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


FIELDS = [
    "embedding_row",
    "document_key",
    "chunk_index",
    "hadm_id",
    "data_split",
    "document_role",
    "document_rank",
    "days_before_landmark",
    "token_count",
    "source_document_key",
    "modality",
    "section_role",
    "section_order",
    "section_instance",
]

REQUIRED_SOURCE_FIELDS = [
    "document_key",
    "chunk_index",
    "hadm_id",
    "data_split",
    "document_role",
    "document_rank",
    "days_before_landmark",
    "token_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the stable row mapping for the Qwen NumPy embedding store.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_embedding_index(metadata: Path, output: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    rows = 0
    with metadata.open("r", encoding="utf-8-sig", newline="") as source, gzip.open(
        output, "wt", encoding="utf-8", newline="", compresslevel=6
    ) as destination:
        reader = csv.DictReader(source)
        missing = [field for field in REQUIRED_SOURCE_FIELDS if field not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"Embedding metadata is missing fields: {', '.join(missing)}")
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        for embedding_row, row in enumerate(reader):
            key = (row["document_key"], row["chunk_index"])
            if key in seen:
                raise RuntimeError(f"Duplicate document/chunk key at embedding row {embedding_row}")
            seen.add(key)
            writer.writerow({
                "embedding_row": embedding_row,
                "document_key": row["document_key"],
                "chunk_index": row["chunk_index"],
                "hadm_id": row["hadm_id"],
                "data_split": row["data_split"],
                "document_role": row["document_role"],
                "document_rank": row["document_rank"],
                "days_before_landmark": row["days_before_landmark"],
                "token_count": row["token_count"],
                "source_document_key": row.get("source_document_key", ""),
                "modality": row.get("modality", ""),
                "section_role": row.get("section_role", ""),
                "section_order": row.get("section_order", ""),
                "section_instance": row.get("section_instance", ""),
            })
            rows += 1
    return {"rows": rows, "unique_document_chunks": len(seen), "output": str(output)}


def main() -> None:
    args = parse_args()
    result = build_embedding_index(args.metadata, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
