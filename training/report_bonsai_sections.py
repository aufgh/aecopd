"""章节对照的原文复核报告；评分不再调用LLM。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from training.report_bonsai_validation import summarize


def run(root: Path) -> None:
    protocol = json.loads((root/'predeclared_cases.json').read_text(encoding='utf-8'))
    annotations = json.loads((root/'clinical_adjudication.json').read_text(encoding='utf-8'))
    results, entries, used = [], [], set()
    for rid in protocol['ids']:
        for arm in ('full', 'sections'):
            r = json.loads((root/f'{rid}_{arm}.json').read_text(encoding='utf-8'))
            local = annotations['documents'][f'{rid}_{arm}']
            overrides = {}
            for grade in ('incorrect', 'uncertain'):
                for suffix, reason in local.get(grade, {}).items():
                    overrides[f'{rid}:{arm}:{suffix}'] = (grade, reason)
            for q in r['raw_quotes_for_review']:
                grade, reason = overrides.get(q['id'], ('correct', '已对照原文；字段符合固定目录且原句内容聚焦。'))
                if local.get('all_incorrect'):
                    grade, reason = 'incorrect', local['all_incorrect']
                if q['field'] == 'audit_unmapped_relevant_evidence':
                    grade, reason = 'incorrect', '已复核：属于已有临床字段、正常套话或无明确关联内容，并非未覆盖语义。'
                if q['id'] in overrides:
                    used.add(q['id'])
                if not q['source_found']:
                    grade, reason = 'incorrect', '非连续原文/大小写或标点改变；自动溯源拒绝。'
                entries.append(dict(q, review_id=rid, reasoning=arm, clinical_grade=grade, review_reason_zh=reason))
            expected = set(overrides)
            actual = {q['id'] for q in r['raw_quotes_for_review']}
            if not expected.issubset(actual):
                raise ValueError(f'Unknown annotation IDs {expected-actual}')
            results.append(r)
    modes = ('full', 'sections')
    summary = summarize(results, entries, protocol, {}, modes)
    core_entries = [q for q in entries if q['field'] != 'audit_unmapped_relevant_evidence']
    core_summary = summarize(results, core_entries, protocol, {}, modes)
    subgroup = {}
    for group, ids in [('regression', protocol['regression_ids']), ('new', protocol['new_ids'])]:
        subprotocol = dict(protocol, cases={k: v for k, v in protocol['cases'].items() if k in ids})
        subgroup[group] = summarize([r for r in results if r['review_id'] in ids],
                                    [q for q in core_entries if q['review_id'] in ids], subprotocol, {}, modes)
    report = {'reviewer': annotations['reviewer'], 'summary_all_35': summary, 'summary_clinical_34': core_summary,
              'subgroups_clinical_34': subgroup, 'quote_reviews': entries,
              'documents': [{
                  'review_id': r['review_id'], 'arm': r['arm'], 'seconds': r['end_to_end_seconds'],
                  'facts_detected': sum(i['detected'] for i in r['checks']['positive']),
                  'grades': dict(Counter(q['clinical_grade'] for q in core_entries
                                         if q['review_id'] == r['review_id'] and q['reasoning'] == r['arm']))
              } for r in results]}
    (root/'audited_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'core': core_summary, 'groups': subgroup, 'documents': report['documents']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run(Path('patient_data/text_semantic_review_2026-10-03/section_validation_2026-10-04'))
