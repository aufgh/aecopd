from __future__ import annotations

import argparse
import json
from pathlib import Path

from aecopd_training.section_tokenization import tokenize_section_documents


MODEL_NAME = "Qwen/Qwen3-Embedding-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize section-level documents with exact offsets for frozen Qwen encoding."
    )
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--whole-document-limit", type=int, default=32_000)
    parser.add_argument("--long-document-chunk-tokens", type=int, default=16_000)
    parser.add_argument("--overlap-tokens", type=int, default=512)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        use_fast=True,
        local_files_only=True,
    )
    output_dir: Path = args.output_dir
    result = tokenize_section_documents(
        source_path=args.documents,
        chunks_path=output_dir / "qwen_section_text_chunks.csv.gz",
        document_metadata_path=output_dir / "qwen_section_document_metadata.csv",
        chunk_metadata_path=output_dir / "qwen_section_chunk_metadata.csv",
        summary_path=output_dir / "qwen_section_tokenization_summary.json",
        tokenizer=tokenizer,
        model_name=MODEL_NAME,
        whole_document_limit=args.whole_document_limit,
        long_document_chunk_tokens=args.long_document_chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        max_documents=args.max_documents,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
