"""用本地 Bio_ClinicalBERT 在 GPU 上编码规则证据清单，可中断续跑。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path


def encode(manifest: Path, output_dir: Path, model_dir: Path, batch_size: int) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU encoding")
    source_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with gzip.open(manifest, "rt", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(int(row["embedding_row"]) != index for index, row in enumerate(rows)):
        raise ValueError("manifest row order is invalid")
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = output_dir / "bioclinicalbert.float16.npy"
    state_path = output_dir / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["manifest_sha256"] != source_hash or state["rows"] != len(rows):
            raise ValueError("resume manifest changed")
        start = int(state["completed_rows"])
        vectors = np.lib.format.open_memmap(vector_path, mode="r+", dtype=np.float16, shape=(len(rows), 768))
    else:
        if vector_path.exists():
            raise FileExistsError(vector_path)
        start = 0
        vectors = np.lib.format.open_memmap(vector_path, mode="w+", dtype=np.float16, shape=(len(rows), 768))
    if start == len(rows):
        return state
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to("cuda").eval()
    if model.config.hidden_size != 768:
        raise ValueError("unexpected Bio_ClinicalBERT dimension")
    for offset in range(start, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        encoded = tokenizer([row["chunk_text"] for row in batch], padding=True,
                            truncation=False, return_tensors="pt")
        if encoded["input_ids"].shape[1] > 512:
            raise ValueError("BERT span exceeds 512 tokens")
        encoded = {name: value.to("cuda") for name, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state.float()
            mask = encoded["attention_mask"].to(torch.bool).clone()
            lengths = mask.sum(dim=1)
            mask[:, 0] = False
            mask.scatter_(1, (lengths - 1).unsqueeze(1), False)
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        array = pooled.cpu().numpy().astype(np.float16)
        if not np.isfinite(array).all():
            raise ValueError("non-finite BERT embedding")
        vectors[offset:offset + len(batch)] = array
        vectors.flush()
        state = {"status": "complete" if offset + len(batch) == len(rows) else "in_progress",
                 "model": "Bio_ClinicalBERT", "device": "cuda", "rows": len(rows),
                 "completed_rows": offset + len(batch), "dimension": 768,
                 "manifest_sha256": source_hash}
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, state_path)
        if state["completed_rows"] % (batch_size * 100) == 0 or state["status"] == "complete":
            print(json.dumps({"completed_rows": state["completed_rows"], "rows": len(rows)}), flush=True)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/model-cache/Bio_ClinicalBERT"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(encode(args.manifest, args.output_dir, args.model_dir, args.batch_size), ensure_ascii=False))


if __name__ == "__main__":
    main()
