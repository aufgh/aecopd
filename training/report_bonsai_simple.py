"""六栏简化抽取的逐条审计；不同字段目录不直接比总体百分比。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from training.benchmark_bonsai_verbatim import normalize
from training.bonsai_clinical_evidence_pilot import find_source_spans

ROOT = Path('patient_data/text_semantic_review_2026-10-03')


def run():
    annotations = json.loads((ROOT/'simple_fields_review_2026-10-04.json').read_text(encoding='utf-8'))
    report = {}
    for mode, directory in [('serial', 'simple_fields_serial_checked_2026-10-04'),
                            ('parallel', 'simple_fields_parallel_checked_2026-10-04'),
                            ('thinking', 'simple_fields_thinking_checked_2026-10-04')]:
        root = ROOT/directory
        protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
        quotes, evidence, failed = [], {}, []
        for path in sorted(root.glob('R*.json')):
            result = json.loads(path.read_text(encoding='utf-8'))
            rid, task = result['review_id'], result['task']
            if 'evidence' not in result:
                failed.append({'review_id': rid, 'task': task, 'error': result.get('error')})
                continue
            overrides = annotations['thinking' if mode == 'thinking' else 'none'].get(f'{rid}_{task}', {})
            seen = set()
            for key, values in result['returned_quotes'].items():
                for i, quote in enumerate(values):
                    suffix = f'{key}:{i}'
                    grade, reason = 'correct', '已逐条对照完整原文；符合此六栏小试定义。'
                    for label in ('incorrect', 'uncertain'):
                        if suffix in overrides.get(label, {}):
                            grade, reason = label, overrides[label][suffix]
                            seen.add(suffix)
                    matched = any(find_source_spans(quote, s['text']) for s in result['source_blocks'])
                    if not matched:
                        grade, reason = 'incorrect', '原文逐字溯源失败；不是缺失。'
                    quotes.append({'review_id': rid, 'task': task, 'field': key, 'index': i,
                                   'quote': quote, 'source_matched': matched, 'grade': grade, 'reason': reason})
                    if grade == 'correct':
                        evidence.setdefault(rid, {}).setdefault(key, []).append(quote)
            expected = set(overrides.get('incorrect', {})) | set(overrides.get('uncertain', {}))
            if seen != expected:
                raise ValueError(f'Unused annotations {mode} {rid} {task} {expected-seen}')
        checkpoints = []
        for rid in protocol['ids']:
            for point in protocol['predeclared_cases']['cases'][rid]['positive']:
                text = normalize(' '.join(evidence.get(rid, {}).get(point['field'], [])))
                checkpoints.append(dict(point, review_id=rid, detected=all(normalize(a) in text for a in point['anchors'])))
        def quality(ids):
            group = [q for q in quotes if q['review_id'] in ids]
            points = [c for c in checkpoints if c['review_id'] in ids]
            return {'quotes': len(group), 'grades': dict(Counter(q['grade'] for q in group)),
                    'correct_fraction': round(sum(q['grade'] == 'correct' for q in group)/len(group), 4) if group else None,
                    'source_matched': sum(q['source_matched'] for q in group),
                    'qualified_anchor_coverage': sum(c['detected'] for c in points), 'anchor_total': len(points)}
        samples = json.loads((root/'gpu_samples.json').read_text())
        gpu = [list(map(int, s['gpu'].split(','))) for s in samples if s['gpu']]
        report[mode] = {'speed': json.loads((root/'speed_summary.json').read_text()),
                        'all': quality(protocol['ids']), 'new_only': quality(['R015', 'R016']),
                        'per_field': {k: dict(Counter(q['grade'] for q in quotes if q['field'] == k))
                                      for k in {q['field'] for q in quotes}},
                        'gpu_mean_util_percent': round(sum(s[0] for s in gpu)/len(gpu), 1),
                        'gpu_peak_used_MiB': max(s[1] for s in gpu),
                        'failed_tasks': failed, 'quotes': quotes, 'checkpoints': checkpoints,
                        'reviewer': 'assistant_not_physician_gold_standard'}
    (ROOT/'simple_fields_audited_2026-10-04.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: {x: v[x] for x in ['speed', 'all', 'new_only', 'gpu_mean_util_percent', 'gpu_peak_used_MiB', 'failed_tasks']}
                      for k, v in report.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
