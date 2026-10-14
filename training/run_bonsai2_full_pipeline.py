"""续跑全部 MIMIC-IV 出院小结抽取，随后只用 Docker CUDA 编码。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from training.run_bonsai2_discharge_extract import extract


def _status(path: Path, **values: object) -> None:
    """仅记录汇总计数，不写入文书原文。"""

    values["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


def _counts(database: Path) -> dict[str, int]:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        return {str(state): int(count) for state, count in connection.execute(
            "SELECT state,COUNT(*) FROM documents GROUP BY state"
        )}
    finally:
        connection.close()


def run(source: Path, output_dir: Path, embedding_dir: Path, *, workers: int = 2,
        retries: int = 3) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "pipeline_status.json"
    with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as handle:
        expected = sum(1 for _ in csv.DictReader(handle))
    database = output_dir / "evidence.sqlite3"
    try:
        for attempt in range(retries + 1):
            _status(status_path, stage="extracting", pass_number=attempt + 1,
                    expected_documents=expected, counts=_counts(database) if database.exists() else {})
            extract(source, output_dir, retry_failed=True, workers=workers)
            counts = _counts(database)
            _status(status_path, stage="extracting", pass_number=attempt + 1,
                    expected_documents=expected, counts=counts)
            print(json.dumps({"pass": attempt + 1, "counts": counts}), flush=True)
            if counts.get("complete", 0) == expected:
                break
        counts = _counts(database)
        if sum(counts.values()) != expected:
            raise RuntimeError("extraction database does not cover all source documents")
        _status(status_path, stage="gpu_encoding", expected_documents=expected, counts=counts,
                embedding_dir=str(embedding_dir.resolve()))
        project_root = Path(__file__).resolve().parents[1]
        container_database = "/workspace/" + database.resolve().relative_to(project_root).as_posix()
        container_output = "/workspace/" + embedding_dir.resolve().relative_to(project_root).as_posix()
        command = [
            "docker", "compose", "-f", "training/compose.yaml", "run", "--rm", "--no-deps",
            "embedder", "python", "-m", "training.encode_bonsai2_spans_bert",
            "--database", container_database, "--output-dir", container_output,
            "--model-dir", "/model-cache/Bio_ClinicalBERT", "--batch-size", "4", "--require-cuda",
        ]
        subprocess.run(command, cwd=project_root, check=True)
        embedding_summary = json.loads((embedding_dir / "summary.json").read_text(encoding="utf-8"))
        if embedding_summary.get("device") != "cuda":
            raise RuntimeError("encoder did not run on CUDA")
        _status(status_path, stage="complete", expected_documents=expected, counts=counts,
                embedding_rows=embedding_summary["rows"], embedding_device="cuda",
                embedding_dir=str(embedding_dir.resolve()))
    except Exception as exc:
        _status(status_path, stage="failed", expected_documents=expected,
                counts=_counts(database) if database.exists() else {}, error_type=type(exc).__name__)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    run(args.source, args.output_dir, args.embedding_dir, workers=args.workers,
        retries=args.retries)


if __name__ == "__main__":
    main()
