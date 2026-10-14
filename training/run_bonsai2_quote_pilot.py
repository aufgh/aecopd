"""在 V5 分层文书上试验 Bonsai2 直接输出原文摘录；不编码、不训练。"""

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

from training.clinical_bonsai_quotes import (
    PROMPT_VERSION, SECTION_PROMPT_VERSION, SECTION_SYSTEM_PROMPT,
    SYSTEM_PROMPT, complete_note, parse_quotes,
)
from training.clinical_discharge_sections import field_contexts
from training.clinical_discharge_rules_v2 import FIELDS
from training.run_bonsai2_discharge_extract import _request_json, _require_local_api
from training.run_bonsai2_semantic_pilot import select_review_admissions


def _extract_one(task: tuple[int, str, str, str, str], *, model_id: str,
                 base_url: str, section_aware: bool, max_tokens: int,
                 reasoning_effort: str | None) -> dict[str, object]:
    hadm_id, field, stratum, original, digest = task
    prompt_version = SECTION_PROMPT_VERSION if section_aware else PROMPT_VERSION
    result: dict[str, object] = {
        "hadm_id": hadm_id, "audit_field": field, "audit_stratum": stratum,
        "source_sha256": digest, "prompt_version": prompt_version,
    }
    try:
        context, allowed = field_contexts(original) if section_aware else (complete_note(original), None)
        payload: dict[str, object] = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": SECTION_SYSTEM_PROMPT if section_aware else SYSTEM_PROMPT},
                {"role": "user", "content": "Discharge summary evidence sections:\n" + context},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if reasoning_effort is not None:
            payload["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}
        answer = _request_json(f"{base_url}/v1/chat/completions", payload)
        content = str(answer["choices"][0]["message"]["content"])
        accepted, rejected = parse_quotes(content, original, allowed)
        result.update({
            "state": "complete",
            "quotes": [{"field": item.field, "text": item.text} for item in accepted],
            "rejected_fields": rejected,
        })
    except (KeyError, IndexError, TypeError, ValueError, urllib.error.URLError) as exc:
        # 不把病历正文或模型输出写入终端和错误日志。
        code = f"HTTPError:{exc.code}" if isinstance(exc, urllib.error.HTTPError) else type(exc).__name__
        result.update({"state": "error", "error_code": code})
    return result


def run(source: Path, annotations: Path, output_dir: Path, *, documents: int,
        workers: int, base_url: str, section_aware: bool = False,
        max_tokens: int = 2800,
        reasoning_effort: str | None = None,
        target_field: str | None = None) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not 1 <= documents <= 24 or not 1 <= workers <= 2 or not 100 <= max_tokens <= 8192:
        raise ValueError("pilot supports 1-24 documents and 1-2 workers")
    if reasoning_effort not in (None, "medium"):
        raise ValueError("only the official medium reasoning probe is supported")
    if target_field is not None and target_field not in FIELDS:
        raise ValueError("unknown target field")
    base_url = _require_local_api(base_url)
    if _request_json(f"{base_url}/health").get("status") != "ok":
        raise RuntimeError("local Bonsai2 server not ready")
    models = _request_json(f"{base_url}/v1/models").get("data")
    if not isinstance(models, list) or not models:
        raise RuntimeError("local Bonsai2 model missing")
    model_id = str(models[0]["id"])
    chosen = select_review_admissions(annotations, 24) if target_field else select_review_admissions(annotations, documents)
    if target_field:
        chosen = [item for item in chosen if item[1] == target_field][:documents]
    if not chosen:
        raise ValueError("no review admission selected")
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
    tasks = [(hadm_id, field, stratum, complete_note(notes[hadm_id]),
              hashlib.sha256(notes[hadm_id].encode()).hexdigest())
             for hadm_id, field, stratum in chosen]
    output_dir.mkdir(parents=True)
    counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda task: _extract_one(task, model_id=model_id,
                                                          base_url=base_url,
                                                          section_aware=section_aware,
                                                          max_tokens=max_tokens,
                                                          reasoning_effort=reasoning_effort), tasks))
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            counts[str(result["state"])] += 1
            if result["state"] == "complete":
                counts["accepted_quotes"] += len(result["quotes"])
                counts["rejected_fields"] += len(result["rejected_fields"])
    report: dict[str, object] = {
        "prompt_version": SECTION_PROMPT_VERSION if section_aware else PROMPT_VERSION,
        "model_id": model_id,
        "source": str(source.resolve()), "documents": len(tasks),
        "counts": dict(counts), "quotes_verbatim_checked": True,
        "section_aware": section_aware,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "target_field": target_field,
        "clinical_truth_validated": False, "encoded": False,
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
    parser.add_argument("--section-aware", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=2800)
    parser.add_argument("--reasoning-effort", choices=("medium",))
    parser.add_argument("--target-field", choices=FIELDS)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.annotations, args.output_dir,
                         documents=args.documents, workers=args.workers,
                         base_url=args.base_url,
                         section_aware=args.section_aware,
                         max_tokens=args.max_tokens,
                         reasoning_effort=args.reasoning_effort,
                         target_field=args.target_field), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
