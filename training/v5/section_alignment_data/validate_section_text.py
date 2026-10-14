"""回查章节编码清单的原文位置、来源哈希和同队列角色。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterator


def _rows(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(aligned_dir: Path, manifest_dir: Path) -> dict[str, object]:
    summary = json.loads((manifest_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = manifest_dir / "shared_text.csv.gz"
    if _sha256(manifest) != summary["manifest_sha256"]:
        raise ValueError("text manifest hash changed")
    cohort = {row["hadm_id"]: row["subject_id"]
              for row in _rows(aligned_dir / "cohort_manifest.csv.gz")}
    roles = set(summary["roles"])
    sections = {row["section_id"]: row for row in _rows(aligned_dir / "note_sections.csv.gz")
                if row["section_role"] in roles and row["content_status"] == "informative"
                and row["parse_status"] != "uncertain"}
    counts: Counter[str] = Counter()
    chunk_indices: dict[str, int] = {}
    source_hadms: set[str] = set()
    for number, row in enumerate(_rows(manifest)):
        if int(row["embedding_row"]) != number:
            raise ValueError("non-contiguous embedding row")
        sid = row["section_id"]
        original = sections.get(sid)
        if original is None:
            raise ValueError("manifest references ineligible section")
        if row["hadm_id"] != original["hadm_id"] or row["section_role"] != original["section_role"]:
            raise ValueError("section key or role mismatch")
        if row["subject_id"] != cohort[row["hadm_id"]]:
            raise ValueError("subject mismatch")
        if row["timestamp_relation"] != original["timestamp_relation"] or row["source_section_sha256"] != original["source_sha256"]:
            raise ValueError("source metadata mismatch")
        start, end = int(row["source_start_char"]), int(row["source_end_char"])
        source_start, source_end = int(original["start_char"]), int(original["end_char"])
        if not source_start <= start < end <= source_end:
            raise ValueError("chunk offset outside section")
        if original["text"][start - source_start:end - source_start] != row["chunk_text"]:
            raise ValueError("chunk text cannot be found at original offset")
        if hashlib.sha256(row["chunk_text"].encode("utf-8")).hexdigest() != row["text_sha256"]:
            raise ValueError("chunk text hash mismatch")
        if int(row["chunk_index"]) != chunk_indices.get(sid, 0):
            raise ValueError("non-contiguous chunk index within section")
        chunk_indices[sid] = int(row["chunk_index"]) + 1
        if not 1 <= int(row["bert_token_count"]) <= 512 or not 1 <= int(row["token_count"]) <= 8192:
            raise ValueError("encoder token count outside limit")
        counts[row["section_role"]] += 1
        source_hadms.add(row["hadm_id"])
    if sum(counts.values()) != summary["chunks"]:
        raise ValueError("manifest row count drift")
    if len(chunk_indices) != summary["source_sections_included"]:
        raise ValueError("section count drift")
    if len(source_hadms) != summary["admissions_with_text"]:
        raise ValueError("admission count drift")
    result: dict[str, object] = {
        "status": "manifest_structurally_validated", "chunks": sum(counts.values()),
        "source_sections": len(chunk_indices), "admissions": len(source_hadms),
        "chunks_by_role": dict(counts), "manifest_sha256": summary["manifest_sha256"],
        "clinical_semantics_review_complete": False,
        "encoding_started": False,
    }
    (manifest_dir / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.aligned_dir, args.manifest_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
