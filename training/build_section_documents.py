from __future__ import annotations

import argparse
import json
from pathlib import Path

from aecopd_training.section_dataset import build_section_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split selected discharge and radiology documents into canonical section-level Qwen inputs."
    )
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_section_documents(
        source_path=args.documents,
        output_path=args.output_dir / "qwen_section_documents.csv.gz",
        summary_path=args.output_dir / "section_build_summary.json",
        max_documents=args.max_documents,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
