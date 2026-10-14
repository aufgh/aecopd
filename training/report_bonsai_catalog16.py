"""16栏并发试验逐条复核；未标注引文不得默认为正确。"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from training.benchmark_bonsai_verbatim import normalize
from training.bonsai_clinical_evidence_pilot import find_source_spans, read_jsonl
from training.bonsai_catalog16 import FIELDS, candidate_sources

ROOT = Path('patient_data/text_semantic_review_2026-10-03')
DIRECTORIES = {'four': ROOT/'catalog16_four_workers_2026-10-04'}


def collect(arm: str) -> tuple[list, list]:
    quotes, results = [], []
    for path in sorted(DIRECTORIES[arm].glob('[0-9]*.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        results.append(result)
        for field, values in result.get('returned_quotes', {}).items():
            for i, quote in enumerate(values):
                quotes.append({'id': f'{arm}:{result["review_id"]}:{field}:{i}', 'review_id': result['review_id'],
                               'field': field, 'index': i, 'quote': quote,
                               'quote_sha256': hashlib.sha256(quote.encode()).hexdigest()})
    return quotes, results


def run(annotation: Path):
    annotations = json.loads(annotation.read_text(encoding='utf-8'))
    rows = {r['review_id']: r for r in read_jsonl(ROOT/'eligible_sources/source_review.jsonl')}
    report = {'reviewer': 'assistant_original_note_comparison_not_independent_physician_gold_standard',
              'not_survival_model_performance': True, 'arms': {}}
    used = set()
    for arm, directory in DIRECTORIES.items():
        protocol = json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
        quotes, results = collect(arm)
        if len(results) != protocol['request_count']:
            raise ValueError('Incomplete experiment; do not report it as completed')
        for q in quotes:
            entry = annotations['quotes'].get(q['id'])
            if entry is None or entry['sha256'] != q['quote_sha256']:
                raise ValueError(f'Unreviewed or changed quote {q["id"]}')
            if entry['grade'] not in {'correct', 'incorrect', 'uncertain'}:
                raise ValueError('Invalid clinical grade')
            used.add(q['id'])
            q.update(grade=entry['grade'], reason=entry['reason'], source_matched=bool(find_source_spans(q['quote'], rows[q['review_id']]['text'])))
            if not q['source_matched']:
                q.update(grade='incorrect', reason='原文定位失败')
        anchors, retrieval, empty_checks = [], [], []
        for rid, case in protocol['cases']['cases'].items():
            for point in case['positive']:
                text = normalize(' '.join(q['quote'] for q in quotes if q['review_id'] == rid and q['field'] == point['field'] and q['grade'] == 'correct'))
                anchors.append({**point, 'review_id': rid, 'detected': all(normalize(a) in text for a in point['anchors'])})
                candidates, _ = candidate_sources(rows[rid]['text'], point['field'])
                retrieval_text = normalize(' '.join(s['text'] for s in candidates))
                retrieval.append({**point, 'review_id': rid, 'detected': all(normalize(a) in retrieval_text for a in point['anchors'])})
            for key in case['empty']:
                values = [q for q in quotes if q['review_id'] == rid and q['field'] == key]
                empty_checks.append({'review_id': rid, 'field': key, 'unexpected_quote_count': len(values)})
        def quality(ids):
            selected = [q for q in quotes if q['review_id'] in ids]
            points = [p for p in anchors if p['review_id'] in ids]
            return {'quotes': len(selected), 'grades': dict(Counter(q['grade'] for q in selected)),
                    'usable_fraction': sum(q['grade'] == 'correct' for q in selected)/len(selected) if selected else None,
                    'source_matched': sum(q['source_matched'] for q in selected),
                    'qualified_anchor_coverage': sum(p['detected'] for p in points), 'anchor_total': len(points)}
        status_counts = Counter(s for r in results for s in r['field_status'].values())
        report['arms'][arm] = {'speed': json.loads((directory/'speed.json').read_text()),
                              'all': quality(protocol['ids']), 'new_only': quality(protocol['new_extraction_ids']),
                              'field_status_counts': dict(status_counts),
                              'per_field': {f.key: dict(Counter(q['grade'] for q in quotes if q['field'] == f.key)) for f in FIELDS},
                              'qualified_anchors': anchors, 'retrieval_anchors': retrieval, 'empty_checks': empty_checks,
                              'quotes': quotes, 'failed_jobs': [r for r in results if 'error' in r]}
    if used != set(annotations['quotes']):
        raise ValueError('Unused annotations; evidence version mismatch')
    paired = []
    for arm, directory in DIRECTORIES.items():
        _, results = collect(arm)
        paired.append({(r['review_id'], tuple(r['keys'])): r for r in results})
    report['paired'] = {'whole_catalog_two_worker_rerun': False,
                        'reason': 'Stop repeating known semantic errors; matched speed comparison is the smaller guarded regression set.'}
    output = ROOT/'catalog16_audited_2026-10-04.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: {x: v[x] for x in ('speed', 'all', 'new_only', 'field_status_counts', 'per_field')} for k, v in report['arms'].items()}, ensure_ascii=False, indent=2))
    print(json.dumps(report['paired'], ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--annotations', type=Path, default=ROOT/'catalog16_review_2026-10-04.json')
    args = parser.parse_args()
    if args.list:
        for arm in DIRECTORIES:
            for q in collect(arm)[0]:
                print(q['id']+'\n'+q['quote']+'\n')
    else:
        run(args.annotations)
