from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a completed frozen clinical text embedding store.")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scan-rows", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    vectors = np.load(args.embeddings, mmap_mode="r")
    if vectors.ndim != 2:
        raise RuntimeError("Embedding store is not a two-dimensional array.")
    total_rows = int(vectors.shape[0])
    if int(state["total_rows"]) != total_rows or int(state["completed_rows"]) != total_rows:
        raise RuntimeError("Embedding state does not mark every array row complete.")
    if state.get("status") != "complete":
        raise RuntimeError(f"Embedding state is not complete: {state.get('status')}")

    index_rows = 0
    unique_keys: set[tuple[str, str]] = set()
    with gzip.open(args.index, "rt", encoding="utf-8-sig", newline="") as handle:
        for expected, row in enumerate(csv.DictReader(handle)):
            if int(row["embedding_row"]) != expected:
                raise RuntimeError(f"Non-contiguous embedding index at row {expected}.")
            key = (row["document_key"], row["chunk_index"])
            if key in unique_keys:
                raise RuntimeError(f"Duplicate embedding key at row {expected}.")
            unique_keys.add(key)
            index_rows += 1
    if index_rows != total_rows:
        raise RuntimeError("Embedding index and array have different row counts.")

    finite_rows = 0
    norm_min = float("inf")
    norm_max = float("-inf")
    for start in range(0, total_rows, args.scan_rows):
        block = np.asarray(vectors[start : start + args.scan_rows], dtype=np.float32)
        if not np.isfinite(block).all():
            raise RuntimeError(f"Non-finite embedding found in rows {start}:{start + len(block)}.")
        norms = np.linalg.vector_norm(block, axis=1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
        finite_rows += len(block)
    if norm_min < 0.99 or norm_max > 1.01:
        raise RuntimeError(f"Embedding normalization outside tolerance: {norm_min}..{norm_max}")

    report = {
        "passed": True,
        "rows": total_rows,
        "dimensions": int(vectors.shape[1]),
        "dtype": str(vectors.dtype),
        "index_rows": index_rows,
        "unique_document_chunks": len(unique_keys),
        "finite_rows": finite_rows,
        "l2_norm_min": norm_min,
        "l2_norm_max": norm_max,
        "state_status": state["status"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
