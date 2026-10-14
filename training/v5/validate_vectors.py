"""校验 V5 规则片段清单与冻结文本向量的一一对应关系。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def validate(manifest: Path, vectors_path: Path, state_path: Path) -> dict[str, object]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if state.get("status") != "complete" or state.get("manifest_sha256") != source_hash:
        raise ValueError("vector state is incomplete or belongs to a different manifest")
    vectors = np.load(vectors_path, mmap_mode="r")
    if vectors.ndim != 2 or vectors.shape[1] not in (768, 2560):
        raise ValueError("unexpected embedding matrix shape")
    if int(state.get("completed_rows", -1)) != vectors.shape[0]:
        raise ValueError("completed row count disagrees with vectors")
    seen_text: dict[str, int] = {}
    duplicates = 0
    rows = 0
    with gzip.open(manifest, "rt", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if int(row["embedding_row"]) != index:
                raise ValueError("non-contiguous manifest rows")
            digest = hashlib.sha256(row["chunk_text"].encode("utf-8")).hexdigest()
            if digest != row["text_sha256"]:
                raise ValueError("manifest text fingerprint mismatch")
            previous = seen_text.setdefault(digest, index)
            if previous != index:
                if not np.array_equal(vectors[previous], vectors[index]):
                    raise ValueError("same original text has different vectors")
                duplicates += 1
            rows += 1
    if rows != vectors.shape[0]:
        raise ValueError("manifest row count disagrees with vectors")
    norm_min, norm_max = float("inf"), float("-inf")
    for start in range(0, rows, 512):
        block = np.asarray(vectors[start:start + 512], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("non-finite text embedding")
        norms = np.linalg.norm(block, axis=1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
    if norm_min < 0.99 or norm_max > 1.01:
        raise ValueError("text vectors are not unit normalized")
    return {"passed": True, "manifest_sha256": source_hash, "rows": rows,
            "dimensions": int(vectors.shape[1]), "dtype": str(vectors.dtype),
            "unique_texts": len(seen_text), "duplicate_rows": duplicates,
            "l2_norm_min": norm_min, "l2_norm_max": norm_max}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.manifest, args.vectors, args.state)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
