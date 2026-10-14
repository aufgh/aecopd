"""有限固定文书的速度/逐字有效率/预先指定检查点验证；不生成训练输入。"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from training.bonsai_clinical_evidence_pilot import (
    decode_content, extract_one, find_source_spans, local_api, read_jsonl, write_catalog,
)
from training.clinical_text_catalog import FIELDS


def normalize(text: str) -> str:
    return ' '.join(text.casefold().split())


def checkpoint_report(result: dict, case: dict) -> dict:
    # 不将失败文书当成全部空字段，从而虚增阴性通过率。
    usable = (result.get('finish_reason') == 'stop' and 'evidence' in result
              and all(issue['type'] in {'quote_absent', 'duplicate'}
                      for issue in result.get('validation_issues', [])))
    evidence = result.get('evidence', {})
    positive = []
    for item in case['positive']:
        texts = ' '.join(normalize(q['quote']) for q in evidence.get(item['field'], []))
        present = usable and all(normalize(a) in texts for a in item['anchors'])
        positive.append(dict(item, detected=present))
    negative = [{'field': key, 'pass': usable and not evidence.get(key), 'usable': usable}
                for key in case['empty']]
    return {'positive': positive, 'negative': negative}


def run(output: Path, sources: Path, cases_path: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(cases_path.read_text(encoding='utf-8'))
    if not 1 <= len(protocol['ids']) <= 8:
        raise ValueError('Bounded benchmark only: at most eight documents')
    rows = {r['review_id']: r for r in read_jsonl(sources)}
    # 先验证预定义检查点的所有锚词在原始文书中，不能在看到输出后修标签。
    for rid in protocol['ids']:
        for fact in protocol['cases'][rid]['positive']:
            if not all(normalize(a) in normalize(rows[rid]['text']) for a in fact['anchors']):
                raise ValueError(f'Checkpoint not grounded in original: {rid} {fact}')
    props = local_api('http://127.0.0.1:8080', '/props')
    output.mkdir(parents=True)
    write_catalog(output)
    (output/'predeclared_cases.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'predeclared_cases.sha256').write_text(hashlib.sha256(cases_path.read_bytes()).hexdigest(), encoding='ascii')
    (output/'server_props.json').write_text(json.dumps(props, ensure_ascii=False, indent=2), encoding='utf-8')
    results = []
    for index, rid in enumerate(protocol['ids']):
        # 顺序交替，减少所有一个模式永远享受热机/缓存的偏差。
        for mode in (('none', 'medium') if index % 2 == 0 else ('medium', 'none')):
            print(f'START {rid} {mode}', flush=True)
            began = time.monotonic()
            try:
                result = extract_one(rows[rid], 'http://127.0.0.1:8080', mode)
            except Exception as error:
                result = {'review_id': rid, 'reasoning': mode, 'status': 'request_failed', 'error': str(error)}
            result['end_to_end_seconds'] = round(time.monotonic()-began, 3)
            result['checks'] = checkpoint_report(result, protocol['cases'][rid])
            raw_quotes = []
            if result.get('raw_response'):
                try:
                    data = decode_content(result['raw_response']['choices'][0]['message'].get('content') or '')
                    for field, values in data.items():
                        if isinstance(values, list):
                            for number, quote in enumerate(values):
                                if isinstance(quote, str):
                                    matches = find_source_spans(quote, rows[rid]['text'])
                                    raw_quotes.append({'id': f'{rid}:{mode}:{field}:{number}', 'field': field,
                                                       'quote': quote, 'source_found': bool(matches),
                                                       'clinical_grade': 'not_reviewed'})
                except (ValueError, AttributeError):
                    pass
            result['raw_quotes_for_review'] = raw_quotes
            result['prediction_allowed'] = False
            (output/f'{rid}_{mode}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            results.append(result)
            print(f"END {rid} {mode} {result['status']} {result['end_to_end_seconds']}s quotes={len(raw_quotes)}", flush=True)
    summary = {}
    for mode in ('none', 'medium'):
        selected = [r for r in results if r['reasoning'] == mode]
        times = [r['end_to_end_seconds'] for r in selected]
        quotes = [q for r in selected for q in r['raw_quotes_for_review']]
        positive = [p for r in selected for p in r['checks']['positive']]
        negative = [p for r in selected for p in r['checks']['negative']]
        summary[mode] = {'documents': len(selected), 'mean_seconds': statistics.mean(times),
                         'median_seconds': statistics.median(times), 'min_seconds': min(times), 'max_seconds': max(times),
                         'raw_quote_count': len(quotes), 'source_found_count': sum(q['source_found'] for q in quotes),
                         'positive_checkpoints': len(positive), 'positive_detected': sum(p['detected'] for p in positive),
                         'negative_checkpoints': len(negative), 'negative_passed': sum(p['pass'] for p in negative),
                         'format_completed_documents': sum('evidence' in r and r.get('finish_reason') == 'stop' for r in selected),
                         'source_clean_documents': sum(r['status'] == 'source_verified_only' for r in selected),
                         'clinical_accuracy': 'pending_manual_review'}
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources', type=Path, default=Path('patient_data/text_semantic_review_2026-10-03/eligible_sources/source_review.jsonl'))
    parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('clinical_evidence_validation_cases.json'))
    args = parser.parse_args()
    run(args.output, args.sources, args.cases)
