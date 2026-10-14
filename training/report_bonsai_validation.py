"""从冻结调用和逐条原文复核生成独立审计报告；不再调用模型。"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from training.benchmark_bonsai_verbatim import checkpoint_report, normalize
from training.bonsai_clinical_evidence_pilot import decode_content


def summarize(results: list[dict], entries: list[dict], protocol: dict, exclusions: dict,
              modes: tuple[str, ...] = ('none', 'medium')) -> dict:
    summary = {}
    for mode in modes:
        selected = [r for r in results if r['reasoning'] == mode]
        if not selected:
            continue
        quotes = [q for q in entries if q['reasoning'] == mode]
        grades = Counter(q['clinical_grade'] for q in quotes)
        positive, negative, focused_hits = [], [], 0
        for r in selected:
            case = protocol['cases'][r['review_id']]
            report = checkpoint_report(r, case)
            positive.extend(report['positive'])
            negative.extend(n for n in report['negative']
                            if n['field'] not in exclusions.get(r['review_id'], []))
            focused = dict(r)
            allowed = {(q['field'], normalize(q['quote'])) for q in quotes if q['review_id'] == r['review_id']
                       and q['clinical_grade'] == 'correct'}
            focused['evidence'] = {k: [q for q in vals if (k, normalize(q['quote'])) in allowed]
                                   for k, vals in r.get('evidence', {}).items()}
            focused_hits += sum(p['detected'] for p in checkpoint_report(focused, case)['positive'])
        times = [r['end_to_end_seconds'] for r in selected]
        total = len(quotes)
        summary[mode] = {
            'documents': len(selected), 'mean_seconds': statistics.mean(times),
            'median_seconds': statistics.median(times), 'min_seconds': min(times), 'max_seconds': max(times),
            'format_documents': sum('evidence' in r and r.get('finish_reason') == 'stop' for r in selected),
            'quotes_returned': total, 'quotes_source_found': sum(q['source_found'] for q in quotes),
            'clinical_grades': dict(grades),
            'directly_usable_precision_lower': grades['correct']/total if total else None,
            'directly_usable_precision_upper': (grades['correct']+grades['uncertain'])/total if total else None,
            'key_facts_total': len(positive), 'key_facts_present_in_source_verified_quotes': sum(p['detected'] for p in positive),
            'key_facts_present_in_directly_usable_quotes': focused_hits,
            'valid_empty_checkpoints': len(negative), 'valid_empty_passed': sum(n['pass'] for n in negative),
            'all_quotes_clinically_correct_nonempty_documents': sum(
                any(q['review_id'] == r['review_id'] for q in quotes)
                and all(q['clinical_grade'] == 'correct' for q in quotes if q['review_id'] == r['review_id'])
                for r in selected),
        }
    return summary


def run(root: Path) -> None:
    protocol = json.loads((root/'predeclared_cases.json').read_text(encoding='utf-8'))
    adjudication = json.loads((root/'clinical_adjudication.json').read_text(encoding='utf-8'))
    errata = json.loads((root/'annotation_errata_before_R039.json').read_text(encoding='utf-8'))
    exclusions = errata['exclude_empty_checkpoints']
    results, entries, used_overrides = [], [], set()
    for file in sorted(root.glob('R*.json')):
        result = json.loads(file.read_text(encoding='utf-8'))
        # 严格解码只重评已有响应，绝不改写原始调用或选择性救回坏JSON。
        try:
            decode_content(result['raw_response']['choices'][0]['message'].get('content') or '')
        except ValueError as error:
            result.pop('evidence', None)
            result['strict_decode_error'] = str(error)
        for quote in result['raw_quotes_for_review']:
            override = adjudication['overrides'].get(quote['id'])
            if override:
                used_overrides.add(quote['id'])
            grade, reason = override or (adjudication['default_reviewed_grade'], adjudication['default_reason_zh'])
            if not quote['source_found'] and grade != 'incorrect':
                raise ValueError(f'Unmatched quote cannot be directly usable: {quote["id"]}')
            entries.append(dict(quote, review_id=result['review_id'], reasoning=result['reasoning'],
                                clinical_grade=grade, review_reason_zh=reason))
        results.append(result)
    if used_overrides != set(adjudication['overrides']):
        raise ValueError('Unknown adjudication IDs')
    summary = summarize(results, entries, protocol, exclusions)
    report = {'reviewer': adjudication['reviewer'], 'criteria': adjudication['criteria'],
              'annotation_erratum': errata, 'original_results_preserved': True,
              'summary': summary, 'quote_reviews': entries,
              'documents': [{
                  'review_id': r['review_id'], 'reasoning': r['reasoning'], 'seconds': r['end_to_end_seconds'],
                  'status': r['status'], 'strict_decode_error': r.get('strict_decode_error'),
                  'grades': dict(Counter(q['clinical_grade'] for q in entries
                                         if q['review_id'] == r['review_id'] and q['reasoning'] == r['reasoning']))
              } for r in results]}
    (root/'audited_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run(Path('patient_data/text_semantic_review_2026-10-03/validation_speed_accuracy_6notes_completed'))
