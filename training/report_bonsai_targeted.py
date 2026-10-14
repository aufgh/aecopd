"""新文书人工逐条复核统计；同时保留原始质量、通过率、漏检与待审。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from training.clinical_evidence_checks import assess


def run(root: Path) -> dict:
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    annotations = json.loads((root/'clinical_adjudication.json').read_text(encoding='utf-8'))
    entries, documents = [], []
    for rid in protocol['ids']:
        for arm in protocol['arms']:
            result = json.loads((root/f'{rid}_{arm}.json').read_text(encoding='utf-8'))
            grades = annotations['documents'][f'{rid}_{arm}']
            overridden = set()
            for quote in result['raw_quotes_for_review']:
                suffix = quote['id'].split(':', 2)[2]
                grade, reason = 'correct', '已对照完整原文；逐字、字段、语境及聚焦性符合本轮目录。'
                for label in ('incorrect', 'uncertain'):
                    if suffix in grades.get(label, {}):
                        grade, reason = label, grades[label][suffix]
                        overridden.add(suffix)
                if not quote['source_found']:
                    grade, reason = 'incorrect', '非连续原文或大小写/标点改变，不能进入候选表。'
                decision, check_reason = assess(quote['field'], quote['quote']) if quote['source_found'] else ('reject', 'quote_absent')
                entries.append(dict(quote, review_id=rid, arm=arm, clinical_grade=grade,
                                    review_reason_zh=reason, check_decision=decision, check_reason=check_reason))
            expected = set(grades.get('incorrect', {})) | set(grades.get('uncertain', {}))
            if expected != overridden:
                raise ValueError(f'Unused annotations {rid} {arm}: {expected-overridden}')
            positive = result.get('checks_before_gate', {}).get('positive', [])
            after = result.get('checks_after_gate', {}).get('positive', [])
            # 锚词出现只说明关键词覆盖，不证明引文正确；另列合格引文覆盖。
            correct_by_field = {}
            for q in entries:
                if q['review_id'] == rid and q['arm'] == arm and q['clinical_grade'] == 'correct' and q['check_decision'] == 'pass':
                    correct_by_field.setdefault(q['field'], []).append(q['quote'])
            from training.benchmark_bonsai_verbatim import normalize
            focused = sum(all(normalize(a) in normalize(' '.join(correct_by_field.get(p['field'], [])))
                              for a in p['anchors']) for p in protocol['cases'][rid]['positive'])
            documents.append({'review_id': rid, 'arm': arm, 'seconds': result['end_to_end_seconds'],
                              'format_completed': result.get('finish_reason') == 'stop',
                              'checkpoints': len(protocol['cases'][rid]['positive']),
                              'anchor_detected_before': sum(p['detected'] for p in positive),
                              'anchor_detected_after': sum(p['detected'] for p in after),
                              'qualified_checkpoint_coverage': focused})
    summaries = {}
    for arm in protocol['arms']:
        qs = [q for q in entries if q['arm'] == arm]
        passed = [q for q in qs if q['check_decision'] == 'pass']
        docs = [d for d in documents if d['arm'] == arm]
        summaries[arm] = {
            'documents': len(docs), 'mean_seconds': round(mean(d['seconds'] for d in docs), 2),
            'raw_quotes': len(qs), 'raw_grades': dict(Counter(q['clinical_grade'] for q in qs)),
            'raw_correct_fraction': round(sum(q['clinical_grade'] == 'correct' for q in qs)/len(qs), 4) if qs else None,
            'source_matched_quotes': sum(q['source_found'] for q in qs),
            'passed_quotes': len(passed), 'passed_grades': dict(Counter(q['clinical_grade'] for q in passed)),
            'passed_correct_fraction': round(sum(q['clinical_grade'] == 'correct' for q in passed)/len(passed), 4) if passed else None,
            'passed_yield': round(len(passed)/len(qs), 4) if qs else None,
            'withheld_or_rejected': len(qs)-len(passed),
            'checkpoint_total': sum(d['checkpoints'] for d in docs),
            'anchor_detected_before': sum(d['anchor_detected_before'] for d in docs),
            'anchor_detected_after': sum(d['anchor_detected_after'] for d in docs),
            'qualified_checkpoint_coverage': sum(d['qualified_checkpoint_coverage'] for d in docs),
            'format_completed': sum(d['format_completed'] for d in docs)}
    report = {'reviewer': annotations['reviewer'], 'not_physician_gold_standard': True,
              'no_training_or_encoding': True, 'summary': summaries, 'documents': documents, 'quote_reviews': entries}
    (root/'audited_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'summary': summaries, 'documents': documents}, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
