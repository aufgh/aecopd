from __future__ import annotations

import argparse
import json
from pathlib import Path

from aecopd_loader import AecopdDataBundle

from aecopd_training.patient_sequence_cache import build_patient_sequence_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local patient-sequence training cache")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--text-data-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--sequence-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = AecopdDataBundle(
        args.data_dir,
        text_metadata_only=True,
        text_dir=args.text_data_dir,
    )
    artifacts = AecopdDataBundle.load_preprocessing(args.artifacts)
    summary = build_patient_sequence_cache(
        bundle,
        artifacts=artifacts,
        sequence_index_path=args.sequence_index,
        output_path=args.output,
        overwrite=args.overwrite,
    )
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
