from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "model_loader"))
sys.path.insert(0, str(WORKSPACE / "training"))

from aecopd_loader import AecopdDataBundle  # noqa: E402
from aecopd_training import (  # noqa: E402
    AecopdRiskModel,
    AlignedModelConfig,
    BatchingConfig,
    FrozenTextEmbeddingStore,
    LengthBucketBatcher,
    ModelConfig,
    MultimodalBatchCollator,
    TimeAlignedAecopdRiskModel,
    discrete_competing_risk_nll,
    load_timeline_lengths,
)
from timeline_collator import SharedTimelineCollator, TimelineConfig  # noqa: E402


def run_smoke_test(
    data_dir: Path,
    artifacts_path: Path,
    length_index_path: Path,
    *,
    batch_size: int,
    device_name: str,
    text_embeddings_path: Path | None = None,
    text_index_path: Path | None = None,
    text_state_path: Path | None = None,
    text_data_dir: Path | None = None,
    architecture: str = "baseline",
) -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU")
    device = torch.device(device_name)
    text_paths = (text_embeddings_path, text_index_path, text_state_path)
    if any(path is not None for path in text_paths) and not all(path is not None for path in text_paths):
        raise ValueError("text embeddings, index, and state paths must be provided together")
    include_text = all(path is not None for path in text_paths)
    if architecture not in {"baseline", "time_aligned"}:
        raise ValueError("architecture must be baseline or time_aligned")
    bundle = AecopdDataBundle(
        data_dir,
        text_metadata_only=include_text,
        text_dir=text_data_dir,
    )
    artifacts = bundle.load_preprocessing(artifacts_path)
    lengths = load_timeline_lengths(length_index_path, split="train")
    sample_stream = bundle.iter_samples("train", artifacts=artifacts, include_text=include_text)
    batcher = LengthBucketBatcher(
        lengths,
        BatchingConfig(
            max_padded_tokens=8192,
            max_batch_size=batch_size,
            buffer_size=max(64, batch_size),
        ),
        shuffle=False,
    )
    samples = next(batcher.iter_batches(sample_stream))
    collator = SharedTimelineCollator(TimelineConfig(max_tokens=1280), artifacts=artifacts)
    if include_text:
        text_store = FrozenTextEmbeddingStore(
            text_embeddings_path,
            text_index_path,
            state_path=text_state_path,
        )
        batch, tensors = MultimodalBatchCollator(collator, text_store)(samples, device=device_name)
    else:
        batch = collator(samples)
        tensors = batch.to_torch(device_name)
    if architecture == "time_aligned":
        if not include_text:
            raise ValueError("time_aligned smoke test requires frozen text embeddings")
        config: ModelConfig | AlignedModelConfig = AlignedModelConfig.from_artifacts(
            artifacts_path,
            text_embedding_dim=text_store.embedding_dim,
        )
        model = TimeAlignedAecopdRiskModel(config).to(device)
    else:
        config = ModelConfig.from_artifacts(
            artifacts_path,
            text_embedding_dim=text_store.embedding_dim if include_text else 0,
        )
        model = AecopdRiskModel(config).to(device)
    model.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_context = nullcontext()

    with autocast_context:
        output = model(tensors)
        loss = discrete_competing_risk_nll(
            output.logits,
            tensors["event_types"],
            tensors["time_days"],
        )
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss: {loss.item()}")
    loss.backward()
    gradient_count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient_count += 1
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("non-finite gradient detected")
    if gradient_count == 0:
        raise RuntimeError("backward pass produced no gradients")

    risk_90 = output.risk_at(90).detach().float().cpu()
    risk_180 = output.risk_at(180).detach().float().cpu()
    risk_365 = output.risk_at(365).detach().float().cpu()
    if not all(
        math.isfinite(value)
        for tensor in (risk_90, risk_180, risk_365)
        for value in tensor.flatten().tolist()
    ):
        raise RuntimeError("non-finite cumulative incidence detected")

    gpu: dict[str, Any] | None = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": properties.name,
            "total_memory_gib": properties.total_memory / 1024**3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        }
    return {
        "passed": True,
        "scope": "one real batch; forward + competing-risk loss + backward; no optimizer step",
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu": gpu,
        "batch_size": len(samples),
        "timeline_lengths": batch.timeline_lengths,
        "text_branch_included": include_text,
        "architecture": architecture,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "loss": float(loss.detach().float().cpu()),
        "finite_gradient_tensors": gradient_count,
        "risks_are_untrained_initialization_only": True,
        "output_shapes": {
            "logits": list(output.logits.shape),
            "cumulative_incidence": list(output.cumulative_incidence.shape),
        },
        "first_sample_risk": {
            "90d": risk_90[0].tolist(),
            "180d": risk_180[0].tolist(),
            "365d": risk_365[0].tolist(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one real AECOPD batch through forward, loss, and backward."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--length-index", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--text-embeddings", type=Path)
    parser.add_argument("--text-index", type=Path)
    parser.add_argument("--text-state", type=Path)
    parser.add_argument("--text-data-dir", type=Path)
    parser.add_argument("--architecture", choices=("baseline", "time_aligned"), default="baseline")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_smoke_test(
        args.data_dir,
        args.artifacts,
        args.length_index,
        batch_size=args.batch_size,
        device_name=args.device,
        text_embeddings_path=args.text_embeddings,
        text_index_path=args.text_index,
        text_state_path=args.text_state,
        text_data_dir=args.text_data_dir,
        architecture=args.architecture,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
