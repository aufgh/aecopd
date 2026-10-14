from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "Qwen/Qwen3-Embedding-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the frozen Qwen embedding model without patient data.")
    parser.add_argument("--output-dir", type=Path, default=Path("/model-cache/Qwen3-Embedding-4B"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--source", choices=("huggingface", "modelscope"), default="huggingface")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.source == "huggingface":
        resolved = snapshot_download(repo_id=args.model_id, local_dir=args.output_dir)
    else:
        from modelscope.hub.snapshot_download import snapshot_download as modelscope_download

        resolved = modelscope_download(model_id=args.model_id, local_dir=str(args.output_dir))
    required = ["config.json", "tokenizer.json"]
    missing = [name for name in required if not (Path(resolved) / name).is_file()]
    if missing:
        raise RuntimeError(f"Downloaded snapshot is incomplete; missing: {missing}")
    weight_bytes = sum(path.stat().st_size for path in Path(resolved).glob("*.safetensors"))
    if weight_bytes == 0:
        raise RuntimeError("Downloaded snapshot contains no safetensors weights.")
    print(json.dumps({
        "model_id": args.model_id,
        "source": args.source,
        "local_dir": str(Path(resolved).resolve()),
        "safetensors_bytes": weight_bytes,
        "status": "complete",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
