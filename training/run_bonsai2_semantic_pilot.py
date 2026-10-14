"""在 V5 分层文书上试验 Bonsai2 语义摘要；不运行编码或模型训练。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from training.clinical_bonsai_summaries import (
    PROMPT_VERSION, SYSTEM_PROMPT, numbered_note, parse_summaries,
)
from training.clinical_discharge_rules_v2 import FIELDS
from training.run_bonsai2_discharge_extract import _request_json, _require_local_api


def select_review_admissions(annotations: Path, limit: int) -> list[tuple[int, str, str]]:
    """八领域优先各抽一篇，再覆盖候选未入选与零命中。"""

    with annotations.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    first = {(row["field"], row["stratum"]): row for row in reversed(rows)}
    selected: list[tuple[int, str, str]] = []
    seen: set[int] = set()
    for stratum in ("eligible", "candidate_only", "no_candidate"):
        for field in FIELDS:
            row = first.get((field, stratum))
            if row is None:
                raise ValueError(f"review stratum missing: {field}/{stratum}")
            hadm_id = int(row["hadm_id"])
            if hadm_id in seen:
                continue
            seen.add(hadm_id)
            selected.append((hadm_id, field, stratum))
            if len(selected) >= limit:
                return selected
    return selected


def _extract_one(task: tuple[int, str, str, str, str], *, model_id: str,
                 base_url: str) -> dict[str, object]:
    hadm_id, field, stratum, original, digest = task
    result: dict[str, object] = {
        "hadm_id": hadm_id, "audit_field": field, "audit_stratum": stratum,
        "source_sha256": digest, "prompt_version": PROMPT_VERSION,
    }
    try:
        context = numbered_note(original)
        answer = _request_json(f"{base_url}/v1/chat/completions", {
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Complete numbered discharge summary:\n" + context},
            ],
            "temperature": 0,
            "max_tokens": 1800,
            "stream": False,
        })
        content = str(answer["choices"][0]["message"]["content"])
        accepted, rejected = parse_summaries(content, original)
        result.update({
            "state": "complete", "summaries": [
                {"field": item.field, "summary": item.summary,
                 "evidence_lines": item.evidence_lines} for item in accepted
            ], "rejected_fields": rejected,
        })
    except (KeyError, IndexError, TypeError, ValueError, urllib.error.URLError) as exc:
        # 日志只记录错误类别，不回显私有文书或模型回复。
        code = f"HTTPError:{exc.code}" if isinstance(exc, urllib.error.HTTPError) else type(exc).__name__
        result.update({"state": "error", "error_code": code})
    return result


def run(source: Path, annotations: Path, output_dir: Path, *, documents: int,
        workers: int, base_url: str) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not 1 <= documents <= 24 or not 1 <= workers <= 2:
        raise ValueError("pilot supports 1-24 documents and 1-2 workers")
    base_url = _require_local_api(base_url)
    if _request_json(f"{base_url}/health").get("status") != "ok":
        raise RuntimeError("local Bonsai2 server not ready")
    models = _request_json(f"{base_url}/v1/models").get("data")
    if not isinstance(models, list) or not models:
        raise RuntimeError("local Bonsai2 model missing")
    model_id = str(models[0]["id"])
    chosen = select_review_admissions(annotations, documents)
    wanted = {hadm_id for hadm_id, _, _ in chosen}
    notes: dict[int, str] = {}
    with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            hadm_id = int(row["hadm_id"])
            if hadm_id in wanted:
                if hadm_id in notes:
                    raise ValueError("duplicate pilot admission")
                notes[hadm_id] = row["document_text"]
    if set(notes) != wanted:
        raise ValueError("pilot note missing from source")
    tasks = [(hadm_id, field, stratum, notes[hadm_id],
              hashlib.sha256(notes[hadm_id].encode()).hexdigest())
             for hadm_id, field, stratum in chosen]
    # 拒绝暗中截断；过长文书的处理留给后续分块设计。
    for _, _, _, original, _ in tasks:
        numbered_note(original)
    output_dir.mkdir(parents=True)
    counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda task: _extract_one(task, model_id=model_id,
                                                          base_url=base_url), tasks))
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            counts[str(result["state"])] += 1
            if result["state"] == "complete":
                counts["accepted_fields"] += len(result["summaries"])
                counts["rejected_fields"] += len(result["rejected_fields"])
    report: dict[str, object] = {
        "prompt_version": PROMPT_VERSION,
        "model_id": model_id,
        "source": str(source.resolve()),
        "documents": len(tasks), "counts": dict(counts),
        "summary_is_model_synthesis_not_original_quote": True,
        "evidence_line_numbers_for_internal_audit_only": True,
        "encoded": False, "clinical_truth_validated": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.annotations, args.output_dir,
                         documents=args.documents, workers=args.workers,
                         base_url=args.base_url), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
