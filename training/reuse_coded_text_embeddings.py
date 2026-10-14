"""按章节输入的精确 SHA-256 复用旧编码；仅为未命中的章节重新编码。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


INDEX_FIELDS = (
    "embedding_row", "document_key", "chunk_index", "hadm_id", "data_split",
    "document_role", "document_rank", "days_before_landmark", "token_count",
    "source_document_key", "modality", "section_role", "section_order",
    "section_instance",
)


def digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def prepare(new_chunks: Path, old_chunks: Path, old_vectors: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    vectors = np.load(old_vectors, mmap_mode="r")
    if vectors.ndim != 2 or vectors.dtype != np.float16:
        raise ValueError("old embedding store must be two-dimensional float16")
    old_by_hash: dict[bytes, int] = {}
    old_rows = 0
    with gzip.open(old_chunks, "rt", encoding="utf-8-sig", newline="") as handle:
        for old_rows, row in enumerate(csv.DictReader(handle), start=1):
            old_by_hash.setdefault(digest(row["chunk_text"]), old_rows - 1)
    if old_rows != vectors.shape[0]:
        raise ValueError("old chunks and embeddings have different row counts")

    output_dir.mkdir(parents=True, exist_ok=False)
    mapping: list[int] = []
    hits = 0
    missing = 0
    chunks_path = output_dir / "missing_chunks.csv.gz"
    index_path = output_dir / "missing_index.csv.gz"
    with gzip.open(new_chunks, "rt", encoding="utf-8-sig", newline="") as source, \
            gzip.open(chunks_path, "wt", encoding="utf-8", newline="", compresslevel=6) as missing_chunks, \
            gzip.open(index_path, "wt", encoding="utf-8", newline="", compresslevel=6) as missing_index:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "chunk_text" not in reader.fieldnames:
            raise ValueError("new chunk file is missing chunk_text")
        chunk_writer = csv.DictWriter(missing_chunks, fieldnames=reader.fieldnames)
        index_writer = csv.DictWriter(missing_index, fieldnames=INDEX_FIELDS)
        chunk_writer.writeheader()
        index_writer.writeheader()
        for row in reader:
            old_index = old_by_hash.get(digest(row["chunk_text"]))
            if old_index is not None:
                mapping.append(old_index)
                hits += 1
                continue
            mapping.append(-1 - missing)
            chunk_writer.writerow(row)
            index_writer.writerow({
                "embedding_row": missing,
                **{name: row.get(name, "") for name in INDEX_FIELDS if name != "embedding_row"},
            })
            missing += 1
    if not mapping:
        raise ValueError("new chunk file is empty")
    if max(mapping) >= vectors.shape[0]:
        raise ValueError("reuse map points outside old embedding store")
    np.save(output_dir / "mapping.int32.npy", np.asarray(mapping, dtype=np.int32))
    report = {
        "total_new_chunks": len(mapping),
        "exact_reuse_rows": hits,
        "rows_to_encode": missing,
        "old_model_rows": old_rows,
        "reuse_key": "SHA-256 of exact UTF-8 chunk_text including role and section prefix",
    }
    (output_dir / "reuse_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def assemble(mapping_path: Path, old_vectors_path: Path, new_vectors_path: Path | None, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    mapping = np.load(mapping_path, mmap_mode="r")
    old = np.load(old_vectors_path, mmap_mode="r")
    missing = int(np.sum(mapping < 0))
    new = np.load(new_vectors_path, mmap_mode="r") if new_vectors_path is not None else None
    if old.ndim != 2 or old.dtype != np.float16:
        raise ValueError("old embedding store has wrong shape or dtype")
    if missing and (new is None or new.shape != (missing, old.shape[1]) or new.dtype != old.dtype):
        raise ValueError("newly encoded rows do not align with missing-row map")
    output.parent.mkdir(parents=True, exist_ok=True)
    store = np.lib.format.open_memmap(output, mode="w+", dtype=np.float16, shape=(len(mapping), old.shape[1]))
    for start in range(0, len(mapping), 8192):
        end = min(start + 8192, len(mapping))
        block = np.asarray(mapping[start:end])
        old_mask = block >= 0
        if old_mask.any():
            store[start:end][old_mask] = old[block[old_mask]]
        if (~old_mask).any():
            assert new is not None
            store[start:end][~old_mask] = new[-1 - block[~old_mask]]
    store.flush()
    return {"rows": len(mapping), "width": old.shape[1], "reused": len(mapping) - missing, "new": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = subcommands.add_parser("prepare")
    prepare_parser.add_argument("--new-chunks", required=True, type=Path)
    prepare_parser.add_argument("--old-chunks", required=True, type=Path)
    prepare_parser.add_argument("--old-vectors", required=True, type=Path)
    prepare_parser.add_argument("--output-dir", required=True, type=Path)
    assemble_parser = subcommands.add_parser("assemble")
    assemble_parser.add_argument("--mapping", required=True, type=Path)
    assemble_parser.add_argument("--old-vectors", required=True, type=Path)
    assemble_parser.add_argument("--new-vectors", type=Path)
    assemble_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = (
        prepare(args.new_chunks, args.old_chunks, args.old_vectors, args.output_dir)
        if args.mode == "prepare"
        else assemble(args.mapping, args.old_vectors, args.new_vectors, args.output)
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
