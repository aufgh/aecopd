"""局部回归速度和逐条审阅报告，必须与首轮独立样本分开。"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from collections import Counter
from training.bonsai_clinical_evidence_pilot import read_jsonl, find_source_spans
from training.benchmark_bonsai_verbatim import normalize
from training.bonsai_catalog16_guarded import ROOT, CASES


def read_arm(arm):
    directory = ROOT/f'catalog16_guarded_{arm}_2026-10-04'
    results = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(directory.glob('[0-9]*.json'))]
    if len(results) != len(CASES):
        raise ValueError('Incomplete regression experiment')
    return results, json.loads((directory/'speed.json').read_text())


def run():
    rows = {r['review_id']: r for r in read_jsonl(ROOT/'eligible_sources/source_review.jsonl')}
    annotations = json.loads((ROOT/'catalog16_guarded_review_2026-10-04.json').read_text(encoding='utf-8'))
    used, report, paired = set(), {'scope': 'post-error development regression, not independent validation', 'arms': {}}, []
    for arm in ('four', 'two'):
        results, speed = read_arm(arm)
        quotes = []
        for r in results:
            for field, values in r.get('returned_quotes', {}).items():
                for i, quote in enumerate(values):
                    identity = f'{arm}:{r["review_id"]}:{field}:{i}'
                    a = annotations['quotes'].get(identity)
                    if a is None or a['sha256'] != hashlib.sha256(quote.encode()).hexdigest():
                        raise ValueError(f'Unreviewed quote {identity}')
                    used.add(identity)
                    quotes.append({'id': identity, 'quote': quote, **a,
                                   'source_matched': bool(find_source_spans(quote, rows[r['review_id']]['text']))})
        counts = Counter(q['grade'] for q in quotes)
        report['arms'][arm] = {'speed': speed, 'quotes': quotes, 'grades': dict(counts),
                              'usable_fraction': counts['correct']/len(quotes) if quotes else None,
                              'source_matched': sum(q['source_matched'] for q in quotes), 'quote_total': len(quotes),
                              'empty_results': [{'review_id': r['review_id'], 'field': r['keys'][0], 'status': r['field_status'][r['keys'][0]]}
                                                for r in results if not any(r.get('returned_quotes', {}).values())]}
        paired.append({(r['review_id'], r['keys'][0]): r for r in results})
    if used != set(annotations['quotes']):
        raise ValueError('Stale annotations')
    report['paired'] = {'payload_identical': all(paired[0][k]['payload'] == paired[1][k]['payload'] for k in paired[0]),
                        'identical_quote_jobs': sum(paired[0][k]['returned_quotes'] == paired[1][k]['returned_quotes'] for k in paired[0]),
                        'jobs': len(CASES), 'same_server_4_slots': True, 'order': 'four_then_two_no_randomized_repeats'}
    report['throughput_ratio_four_vs_two'] = report['arms']['two']['speed']['seconds']/report['arms']['four']['speed']['seconds']
    (ROOT/'catalog16_guarded_audited_2026-10-04.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({a: {k: x[k] for k in ['speed', 'grades', 'usable_fraction', 'source_matched', 'quote_total', 'empty_results']}
                      for a, x in report['arms'].items()}, ensure_ascii=False, indent=2))
    print(json.dumps(report['paired']))


if __name__ == '__main__':
    run()
