from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class Candidate:
    target_tokens: int
    recorded_tokens: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Qwen embedding VRAM on real text without logging text or IDs.")
    parser.add_argument("--model-dir", type=Path, default=Path("/model-cache/Qwen3-Embedding-4B"))
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 12000, 15983])
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def choose_candidates(path: Path, targets: list[int]) -> list[Candidate]:
    best: dict[int, Candidate] = {}
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            recorded = int(row["token_count"])
            for target in targets:
                current = best.get(target)
                if current is None or abs(recorded - target) < abs(current.recorded_tokens - target):
                    best[target] = Candidate(target, recorded, row["chunk_text"])
    missing = [target for target in targets if target not in best]
    if missing:
        raise RuntimeError(f"No candidate text found for targets: {missing}")
    return [best[target] for target in targets]


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, sequence_lengths]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the memory ladder.")
    candidates = choose_candidates(args.chunks, args.targets)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, padding_side="left")

    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    load_seconds = time.perf_counter() - load_started

    results: list[dict[str, object]] = []
    for candidate in candidates:
        encoded = tokenizer(candidate.text, return_tensors="pt", truncation=False, padding=False)
        actual_tokens = int(encoded["input_ids"].shape[1])
        batch = {name: tensor.to("cuda") for name, tensor in encoded.items()}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                output = model(**batch)
                vector = F.normalize(last_token_pool(output.last_hidden_state, batch["attention_mask"]), p=2, dim=1)
                torch.cuda.synchronize()
            norm = float(torch.linalg.vector_norm(vector.float(), dim=1).item())
            result = {
                "target_tokens": candidate.target_tokens,
                "recorded_tokens": candidate.recorded_tokens,
                "actual_model_tokens": actual_tokens,
                "status": "passed",
                "seconds": round(time.perf_counter() - started, 4),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 4),
                "embedding_dim": int(vector.shape[1]),
                "l2_norm": round(norm, 6),
            }
        except torch.OutOfMemoryError as exc:
            result = {
                "target_tokens": candidate.target_tokens,
                "recorded_tokens": candidate.recorded_tokens,
                "actual_model_tokens": actual_tokens,
                "status": "out_of_memory",
                "seconds": round(time.perf_counter() - started, 4),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 4),
                "error_type": type(exc).__name__,
            }
            results.append(result)
            del batch
            torch.cuda.empty_cache()
            break
        results.append(result)
        del batch, encoded, output, vector
        torch.cuda.empty_cache()

    passed_lengths = [int(row["actual_model_tokens"]) for row in results if row["status"] == "passed"]
    report = {
        "model": "Qwen/Qwen3-Embedding-4B",
        "model_dir": str(args.model_dir),
        "device": torch.cuda.get_device_name(0),
        "device_total_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 4),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "model_load_seconds": round(load_seconds, 4),
        "largest_passed_tokens": max(passed_lengths) if passed_lengths else None,
        "all_requested_lengths_passed": len(results) == len(candidates) and all(row["status"] == "passed" for row in results),
        "results": results,
    }
    if any(not math.isfinite(float(row.get("l2_norm", 1.0))) for row in results):
        raise RuntimeError("A non-finite embedding was produced.")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
