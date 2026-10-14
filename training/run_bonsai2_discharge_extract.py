"""通过仅监听本机的 Bonsai 2 API 提取出院小结原文片段；可逐篇恢复。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from training.clinical_bonsai_spans import (
    EvidenceSpan,
    FIELDS,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    parse_evidence_partial,
    select_note_context,
)
from training.clinical_text_rules import has_possible_post_discharge_fact


def _require_local_api(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("clinical note text may only be sent to a loopback HTTP endpoint")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("unexpected Bonsai API URL component")
    return base_url.rstrip("/")


def _request_json(url: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise ValueError("Bonsai API returned a non-object response")
    return data


def _initialize(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            hadm_id INTEGER PRIMARY KEY,
            subject_id INTEGER NOT NULL,
            source_sha256 TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('complete','error')),
            error_code TEXT,
            possible_future_fact INTEGER NOT NULL,
            processed_at_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS evidence_spans (
            hadm_id INTEGER NOT NULL REFERENCES documents(hadm_id),
            field TEXT NOT NULL,
            quote TEXT NOT NULL,
            evidence_line INTEGER NOT NULL,
            PRIMARY KEY (hadm_id, field)
        );
        CREATE TABLE IF NOT EXISTS rejected_fields (
            hadm_id INTEGER NOT NULL REFERENCES documents(hadm_id),
            field TEXT NOT NULL,
            reason TEXT NOT NULL,
            PRIMARY KEY (hadm_id, field)
        );
    """)


def _extract_one(
    task: tuple[dict[str, str], str, str, str],
) -> tuple[dict[str, str], str, list[EvidenceSpan], dict[str, str], str | None]:
    """并行调用本机模型；SQLite 写入只在主线程进行。"""

    row, original, model_id, base_url = task
    context = select_note_context(original)
    payload: dict[str, object] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Note excerpts (line numbers are locators only):\n" + context},
        ],
        "temperature": 0,
        "max_tokens": 1200,
        "stream": False,
    }
    try:
        answer = _request_json(f"{base_url}/v1/chat/completions", payload)
        content = answer["choices"][0]["message"]["content"]
        spans, rejected = parse_evidence_partial(str(content), original, context)
        return row, original, spans, rejected, None
    except (KeyError, IndexError, TypeError, ValueError, urllib.error.URLError) as exc:
        # 只记录错误类别，避免异常信息包含文书或模型回复。
        error_code = f"HTTPError:{exc.code}" if isinstance(exc, urllib.error.HTTPError) else type(exc).__name__
        return row, original, [], {}, error_code


def extract(
    source: Path,
    output_dir: Path,
    *,
    base_url: str = "http://127.0.0.1:8080",
    max_documents: int | None = None,
    retry_failed: bool = False,
    workers: int = 1,
) -> dict[str, object]:
    """每篇单独提交，重启时跳过已完成文书；不输出正文到终端。"""

    base_url = _require_local_api(base_url)
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive")
    if workers < 1 or workers > 4:
        raise ValueError("workers must be between 1 and 4")
    health = _request_json(f"{base_url}/health")
    if health.get("status") != "ok":
        raise RuntimeError("Bonsai 2 server is not ready")
    models = _request_json(f"{base_url}/v1/models").get("data")
    if not isinstance(models, list) or not models or not isinstance(models[0], dict):
        raise RuntimeError("Bonsai 2 model list is empty")
    model_id = str(models[0]["id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output_dir / "evidence.sqlite3")
    _initialize(connection)
    counts: Counter[str] = Counter()
    seen: set[int] = set()
    try:
        tasks: list[tuple[dict[str, str], str, str, str]] = []
        with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"subject_id", "hadm_id", "document_text"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("source note export contract is incomplete")
            for row in reader:
                if max_documents is not None and len(tasks) >= max_documents:
                    break
                hadm_id = int(row["hadm_id"])
                if hadm_id in seen:
                    raise ValueError("duplicate admission in source export")
                seen.add(hadm_id)
                original = row["document_text"]
                digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
                previous = connection.execute(
                    "SELECT source_sha256,prompt_version,state FROM documents WHERE hadm_id=?", (hadm_id,)
                ).fetchone()
                if previous is not None:
                    if previous[:2] != (digest, PROMPT_VERSION):
                        raise ValueError("source text or prompt changed during resume")
                    if previous[2] == "complete" or not retry_failed:
                        counts["already_processed"] += 1
                        continue
                    connection.execute("DELETE FROM evidence_spans WHERE hadm_id=?", (hadm_id,))
                    connection.execute("DELETE FROM rejected_fields WHERE hadm_id=?", (hadm_id,))
                    connection.execute("DELETE FROM documents WHERE hadm_id=?", (hadm_id,))
                    connection.commit()
                tasks.append((row, original, model_id, base_url))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for row, original, spans, rejected, error_code in executor.map(_extract_one, tasks):
                hadm_id = int(row["hadm_id"])
                digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
                state = "complete" if error_code is None else "error"
                with connection:
                    connection.execute(
                        "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)",
                        (
                            hadm_id, int(row["subject_id"]), digest, PROMPT_VERSION,
                            state, error_code, int(has_possible_post_discharge_fact(original)),
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    connection.executemany(
                        "INSERT INTO evidence_spans VALUES (?,?,?,?)",
                        ((hadm_id, span.field, span.quote, span.line) for span in spans),
                    )
                    connection.executemany(
                        "INSERT INTO rejected_fields VALUES (?,?,?)",
                        ((hadm_id, field, reason) for field, reason in rejected.items()),
                    )
                counts["attempted"] += 1
                counts[state] += 1
                counts["spans"] += len(spans)
                counts["rejected_fields"] += len(rejected)
                if counts["attempted"] % 100 == 0:
                    print(json.dumps({"processed": counts["attempted"], "total_this_run": len(tasks),
                                      "complete": counts["complete"], "error": counts["error"]}), flush=True)
    finally:
        connection.close()
    summary: dict[str, object] = {
        "source": str(source.resolve()),
        "prompt_version": PROMPT_VERSION,
        "field_count": len(FIELDS),
        "source_documents_seen": len(seen),
        "this_run": dict(counts),
        "raw_evidence_only_in_private_sqlite": True,
        "outcome_labels_used": False,
        "clinical_accuracy_validated": False,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(extract(
        args.source, args.output_dir, base_url=args.base_url,
        max_documents=args.max_documents, retry_failed=args.retry_failed, workers=args.workers,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
