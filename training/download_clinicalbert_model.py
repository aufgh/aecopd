"""下载公开的 Bio_ClinicalBERT 权重；不读取或上传任何病历。"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"
MODEL_FILES = ("config.json", "pytorch_model.bin", "vocab.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/model-cache/Bio_ClinicalBERT"))
    args = parser.parse_args()
    snapshot_download(repo_id=MODEL_ID, local_dir=args.output_dir, allow_patterns=list(MODEL_FILES))
    missing = [name for name in MODEL_FILES if not (args.output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete Bio_ClinicalBERT download: {missing}")
    print(f"Bio_ClinicalBERT ready: {args.output_dir}")


if __name__ == "__main__":
    main()
