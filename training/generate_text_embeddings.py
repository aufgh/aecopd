from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F


EMBEDDING_DIM = 2560


@dataclass(frozen=True)
class TextRow:
    row_number: int
    recorded_tokens: int
    text: str
    content_hash: bytes = b""


@dataclass(frozen=True)
class ReusePlan:
    existing_copies: list[tuple[TextRow, int]]
    new_groups: dict[bytes, list[TextRow]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate resumable, frozen Qwen embeddings for private text chunks.")
    parser.add_argument("--model-dir", type=Path, default=Path("/model-cache/Qwen3-Embedding-4B"))
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=256)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--overwrite", action="store_true", help="Replace an output store only when no state file is present.")
    return parser.parse_args()


def count_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def stream_rows(path: Path, start_row: int, max_rows: int | None) -> Iterator[TextRow]:
    emitted = 0
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle)):
            if row_number < start_row:
                continue
            if max_rows is not None and emitted >= max_rows:
                return
            emitted += 1
            text = row["chunk_text"]
            yield TextRow(
                row_number,
                int(row["token_count"]),
                text,
                hashlib.sha256(text.encode("utf-8")).digest(),
            )


def completed_content_rows(path: Path, completed_rows: int) -> dict[bytes, int]:
    result: dict[bytes, int] = {}
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle)):
            if row_number >= completed_rows:
                break
            digest = hashlib.sha256(row["chunk_text"].encode("utf-8")).digest()
            result.setdefault(digest, row_number)
    if completed_rows and not result:
        raise RuntimeError("Could not seed content reuse from completed rows.")
    return result


def plan_reuse(rows: list[TextRow], available: Mapping[bytes, int]) -> ReusePlan:
    existing: list[tuple[TextRow, int]] = []
    groups: dict[bytes, list[TextRow]] = {}
    for row in rows:
        digest = row.content_hash or hashlib.sha256(row.text.encode("utf-8")).digest()
        if digest in available:
            existing.append((row, available[digest]))
        else:
            groups.setdefault(digest, []).append(row)
    return ReusePlan(existing_copies=existing, new_groups=groups)


def buffered(rows: Iterable[TextRow], size: int) -> Iterator[list[TextRow]]:
    group: list[TextRow] = []
    for row in rows:
        group.append(row)
        if len(group) == size:
            yield group
            group = []
    if group:
        yield group


def pack_batches(rows: list[TextRow], token_budget: int, max_batch_size: int) -> list[list[TextRow]]:
    if token_budget <= 0 or max_batch_size <= 0:
        raise ValueError("token_budget and max_batch_size must be positive")
    batches: list[list[TextRow]] = []
    current: list[TextRow] = []
    current_max = 0
    for row in sorted(rows, key=lambda item: item.recorded_tokens):
        row_tokens = row.recorded_tokens + 1
        if row_tokens > token_budget:
            if current:
                batches.append(current)
                current = []
                current_max = 0
            batches.append([row])
            continue
        proposed_max = max(current_max, row_tokens)
        if current and (len(current) >= max_batch_size or proposed_max * (len(current) + 1) > token_budget):
            batches.append(current)
            current = []
            current_max = 0
        current.append(row)
        current_max = max(current_max, row_tokens)
    if current:
        batches.append(current)
    return batches


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return hidden[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, sequence_lengths]


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_state(path: Path, total_rows: int, manifest_sha256: str) -> dict[str, object]:
    if not path.exists():
        return {"total_rows": total_rows, "completed_rows": 0, "completed_buffers": 0,
                "manifest_sha256": manifest_sha256, "status": "initialized"}
    state = json.loads(path.read_text(encoding="utf-8"))
    if int(state["total_rows"]) != total_rows:
        raise RuntimeError("State total_rows does not match the current input file.")
    if state.get("manifest_sha256") not in (None, manifest_sha256):
        raise RuntimeError("State manifest SHA-256 does not match the current input file.")
    if state.get("manifest_sha256") is None and int(state["completed_rows"]) < total_rows:
        raise RuntimeError("Legacy in-progress state has no manifest fingerprint; unsafe to resume.")
    return state


def open_store(path: Path, total_rows: int, resume: bool, overwrite: bool) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        return np.lib.format.open_memmap(path, mode="r+", dtype=np.float16, shape=(total_rows, EMBEDDING_DIM))
    if path.exists() and not overwrite:
        raise RuntimeError(f"Output exists without a resume state: {path}")
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(total_rows, EMBEDDING_DIM))


def main() -> None:
    from transformers import AutoModel, AutoTokenizer

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    total_rows = count_rows(args.chunks)
    manifest_sha256 = hashlib.sha256(args.chunks.read_bytes()).hexdigest()
    state = load_state(args.state, total_rows, manifest_sha256)
    completed_rows = int(state["completed_rows"])
    if completed_rows >= total_rows:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    store = open_store(args.output, total_rows, resume=args.state.exists(), overwrite=args.overwrite)
    content_rows = completed_content_rows(args.chunks, completed_rows)
    encoded_unique_rows = int(state.get("encoded_unique_rows_after_dedup", 0))
    reused_duplicate_rows = int(state.get("reused_duplicate_rows_after_dedup", 0))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, padding_side="left")
    model = AutoModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()

    started = time.perf_counter()
    source = stream_rows(args.chunks, completed_rows, args.max_rows)
    for source_buffer in buffered(source, args.buffer_size):
        reuse_plan = plan_reuse(source_buffer, content_rows)
        for row, reference_row in reuse_plan.existing_copies:
            store[row.row_number] = store[reference_row]
            reused_duplicate_rows += 1

        representatives = [group[0] for group in reuse_plan.new_groups.values()]
        for batch_rows in pack_batches(representatives, args.token_budget, args.max_batch_size):
            texts = [row.text for row in batch_rows]
            encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
            device_batch = {name: tensor.to("cuda") for name, tensor in encoded.items()}
            with torch.inference_mode():
                output = model(**device_batch)
                vectors = F.normalize(last_token_pool(output.last_hidden_state, device_batch["attention_mask"]).float(), p=2, dim=1)
                torch.cuda.synchronize()
            array = vectors.cpu().numpy().astype(np.float16)
            if not np.isfinite(array).all():
                raise RuntimeError("Non-finite embedding produced.")
            for position, row in enumerate(batch_rows):
                store[row.row_number] = array[position]
                encoded_unique_rows += 1
            del output, vectors, device_batch, encoded, array

        for digest, group in reuse_plan.new_groups.items():
            representative = group[0]
            for duplicate in group[1:]:
                store[duplicate.row_number] = store[representative.row_number]
                reused_duplicate_rows += 1
            content_rows[digest] = representative.row_number
        store.flush()
        completed_rows = max(row.row_number for row in source_buffer) + 1
        state = {
            "total_rows": total_rows,
            "completed_rows": completed_rows,
            "completed_buffers": int(state["completed_buffers"]) + 1,
            "manifest_sha256": manifest_sha256,
            "status": "complete" if completed_rows == total_rows else "in_progress",
            "embedding_dim": EMBEDDING_DIM,
            "dtype": "float16",
            "encoder": "Qwen3-Embedding-4B",
            "pooling": "last_token_l2",
            "instruction": "none_document_baseline",
            "token_budget": args.token_budget,
            "max_batch_size": args.max_batch_size,
            "buffer_size": args.buffer_size,
            "elapsed_seconds_this_run": round(time.perf_counter() - started, 3),
            "content_reuse": "sha256_exact_utf8",
            "unique_content_seen": len(content_rows),
            "encoded_unique_rows_after_dedup": encoded_unique_rows,
            "reused_duplicate_rows_after_dedup": reused_duplicate_rows,
        }
        atomic_write_json(args.state, state)
        print(json.dumps({key: state[key] for key in ("completed_rows", "total_rows", "status", "elapsed_seconds_this_run")}), flush=True)

    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
