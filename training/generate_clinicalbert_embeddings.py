"""对现有 Qwen 章节输入生成可恢复的 Bio_ClinicalBERT 冻结向量。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from aecopd_training.clinicalbert_embeddings import (
    TokenWindow,
    masked_content_mean,
    pack_token_windows,
    split_token_windows,
)
from generate_text_embeddings import (
    atomic_write_json,
    buffered,
    completed_content_rows,
    plan_reuse,
    stream_rows,
    TextRow,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def validate_index_alignment(chunks_path: Path, index_path: Path) -> int:
    """确保 BERT 输出行与已经校验的 Qwen 索引一一对应。"""

    count = 0
    with gzip.open(chunks_path, "rt", encoding="utf-8-sig", newline="") as chunks_handle:
        with gzip.open(index_path, "rt", encoding="utf-8-sig", newline="") as index_handle:
            chunks = csv.DictReader(chunks_handle)
            index = csv.DictReader(index_handle)
            for count, (source, target) in enumerate(zip(chunks, index, strict=True), start=1):
                if (
                    int(target["embedding_row"]) != count - 1
                    or source["document_key"] != target["document_key"]
                    or source["chunk_index"] != target["chunk_index"]
                    or source["hadm_id"] != target["hadm_id"]
                    or source["document_role"] != target["document_role"]
                ):
                    raise ValueError("Qwen section input and embedding index are misaligned")
    return count


def _write_state_with_retry(path: Path, state: dict[str, object]) -> None:
    """Windows 挂载目录短暂锁住目标文件时重试原子提交。"""

    for attempt in range(40):
        try:
            atomic_write_json(path, state)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.25)


def encode_representatives(
    rows: list[TextRow],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    *,
    token_budget: int,
    max_batch_size: int,
    overlap_tokens: int,
) -> dict[int, np.ndarray]:
    """每个完整章节分窗编码，再按非重叠 token 权重汇总。"""

    all_windows: list[TokenWindow] = []
    for row in rows:
        token_ids = tokenizer.encode(row.text, add_special_tokens=False)
        all_windows.extend(
            split_token_windows(token_ids, row.row_number, overlap_tokens=overlap_tokens)
        )
    sums: dict[int, np.ndarray] = {}
    weights: dict[int, int] = {}
    for batch in pack_token_windows(
        all_windows, token_budget=token_budget, max_batch_size=max_batch_size
    ):
        ids = [
            [tokenizer.cls_token_id, *window.token_ids, tokenizer.sep_token_id]
            for window in batch
        ]
        encoded = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            hidden = model(**encoded).last_hidden_state
            vectors = masked_content_mean(hidden.float(), encoded["attention_mask"])
        vectors_np = vectors.cpu().numpy()
        for window, vector in zip(batch, vectors_np, strict=True):
            sums.setdefault(window.source_row, np.zeros_like(vector))
            sums[window.source_row] += vector * window.aggregation_weight
            weights[window.source_row] = weights.get(window.source_row, 0) + window.aggregation_weight
    results: dict[int, np.ndarray] = {}
    for row in rows:
        vector = sums[row.row_number] / weights[row.row_number]
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise RuntimeError("Bio_ClinicalBERT produced a non-finite or zero embedding")
        results[row.row_number] = (vector / norm).astype(np.float16)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/model-cache/Bio_ClinicalBERT"))
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=256)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA GPU is required")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if not 0 <= args.overlap_tokens < 510:
        raise ValueError("overlap must be in [0, 510)")

    from transformers import AutoModel, AutoTokenizer

    output_dir = args.output_dir
    output_path = output_dir / "bioclinicalbert_section_embeddings.float16.npy"
    state_path = output_dir / "encoding_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        total_rows = int(state["total_rows"])
        for key, expected in (
            ("chunks_path", str(args.chunks.resolve())),
            ("index_path", str(args.index.resolve())),
            ("token_budget", args.token_budget),
            ("max_batch_size", args.max_batch_size),
            ("overlap_tokens", args.overlap_tokens),
        ):
            if state[key] != expected:
                raise ValueError(f"resume contract changed: {key}")
        completed = int(state["completed_rows"])
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("output directory exists without a resumable state")
        total_rows = validate_index_alignment(args.chunks, args.index)
        completed = 0
        state = {
            "total_rows": total_rows,
            "completed_rows": 0,
            "completed_buffers": 0,
            "chunks_path": str(args.chunks.resolve()),
            "index_path": str(args.index.resolve()),
            "model": "emilyalsentzer/Bio_ClinicalBERT",
            "token_budget": args.token_budget,
            "max_batch_size": args.max_batch_size,
            "overlap_tokens": args.overlap_tokens,
            "encoded_unique_rows": 0,
            "reused_duplicate_rows": 0,
            "status": "initialized",
        }
    if completed == total_rows:
        print(json.dumps({"completed_rows": completed, "total_rows": total_rows, "status": "complete"}))
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to("cuda")
    model.eval()
    width = int(model.config.hidden_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    if completed:
        store = np.load(output_path, mmap_mode="r+")
        if store.shape != (total_rows, width) or store.dtype != np.float16:
            raise ValueError("existing BERT embedding store does not match model")
    else:
        store = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float16, shape=(total_rows, width)
        )
    content_rows = completed_content_rows(args.chunks, completed)
    started = time.monotonic()
    source = stream_rows(args.chunks, completed, args.max_rows)
    for rows in buffered(source, args.buffer_size):
        reuse = plan_reuse(rows, content_rows)
        for row, original in reuse.existing_copies:
            store[row.row_number] = store[original]
        representatives = [group[0] for group in reuse.new_groups.values()]
        if representatives:
            vectors = encode_representatives(
                representatives,
                tokenizer,
                model,
                token_budget=args.token_budget,
                max_batch_size=args.max_batch_size,
                overlap_tokens=args.overlap_tokens,
            )
            for row in representatives:
                store[row.row_number] = vectors[row.row_number]
        for digest, group in reuse.new_groups.items():
            representative = group[0]
            for duplicate in group[1:]:
                store[duplicate.row_number] = store[representative.row_number]
            content_rows[digest] = representative.row_number
        store.flush()
        completed = rows[-1].row_number + 1
        state.update(
            completed_rows=completed,
            completed_buffers=int(state["completed_buffers"]) + 1,
            encoded_unique_rows=int(state["encoded_unique_rows"]) + len(representatives),
            reused_duplicate_rows=int(state["reused_duplicate_rows"])
            + len(reuse.existing_copies)
            + sum(len(group) - 1 for group in reuse.new_groups.values()),
            status="complete" if completed == total_rows else "in_progress",
            elapsed_seconds_this_run=round(time.monotonic() - started, 3),
        )
        _write_state_with_retry(state_path, state)
        if int(state["completed_buffers"]) % 16 == 0 or completed == total_rows:
            print(
                json.dumps({"completed_rows": completed, "total_rows": total_rows, "status": state["status"]}),
                flush=True,
            )
    if completed != total_rows and args.max_rows is None:
        raise RuntimeError("encoding ended before the declared section count")


if __name__ == "__main__":
    main()
