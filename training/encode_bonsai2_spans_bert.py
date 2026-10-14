"""以既有 Bio_ClinicalBERT 编码 Bonsai 2 核实过的原文片段，不编码整篇小结。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path


def load_spans(database: Path) -> list[tuple[int, int, str, str, int]]:
    """仅接受完整且可追溯的片段；未知字段没有向量。"""

    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute("""
            SELECT d.subject_id,s.hadm_id,s.field,s.quote,s.evidence_line
            FROM evidence_spans s JOIN documents d USING(hadm_id)
            WHERE d.state='complete'
            ORDER BY s.hadm_id,s.field
        """).fetchall()
    finally:
        connection.close()
    return [(int(subject), int(hadm), str(field), str(quote), int(line))
            for subject, hadm, field, quote, line in rows]


def encode(database: Path, output_dir: Path, *, model_dir: Path, batch_size: int,
           require_cuda: bool = False) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    rows = load_spans(database)
    if not rows:
        raise ValueError("no complete Bonsai evidence spans")

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required; refusing CPU encoding")
    model = model.to(device).eval()
    vectors = np.empty((len(rows), int(model.config.hidden_size)), dtype=np.float16)
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        encoded = tokenizer(
            [row[3] for row in batch], padding=True, truncation=False,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] > 512:
            raise ValueError("one evidence span exceeds BERT's 512-token limit")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            # 与既有章节 BERT 一致：排除 [CLS]、[SEP] 与 padding 后平均池化。
            mask = encoded["attention_mask"].to(torch.bool).clone()
            lengths = mask.sum(dim=1)
            mask[:, 0] = False
            mask.scatter_(1, (lengths - 1).unsqueeze(1), False)
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors[start:start + len(batch)] = pooled.cpu().numpy().astype(np.float16)
    if not np.isfinite(vectors).all():
        raise ValueError("non-finite Bio_ClinicalBERT embedding")

    output_dir.mkdir(parents=True)
    np.save(output_dir / "bioclinicalbert_evidence.float16.npy", vectors)
    with (output_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("embedding_row", "subject_id", "hadm_id", "field", "evidence_line", "quote_sha256"))
        for index, (subject, hadm, field, quote, line) in enumerate(rows):
            writer.writerow((index, subject, hadm, field, line, hashlib.sha256(quote.encode()).hexdigest()))
    summary: dict[str, object] = {
        "source_database": str(database.resolve()),
        "model": "emilyalsentzer/Bio_ClinicalBERT",
        "rows": len(rows),
        "unique_admissions": len({row[1] for row in rows}),
        "embedding_shape": list(vectors.shape),
        "device": device.type,
        "raw_evidence_copied_to_embedding_index": False,
        "clinical_extraction_accuracy_validated": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("training/model_cache/Bio_ClinicalBERT"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(json.dumps(encode(
        args.database, args.output_dir, model_dir=args.model_dir, batch_size=args.batch_size,
        require_cuda=args.require_cuda,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
