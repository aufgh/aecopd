"""两份新文书、三个配对设置；旧八份只回放检查，不重跑LLM。"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from training.benchmark_bonsai_sections import TASKS, collect_quotes, section_sources, task_sources
from training.benchmark_bonsai_verbatim import checkpoint_report, normalize
from training.bonsai_clinical_evidence_pilot import extract_one, local_api, read_jsonl
from training.clinical_evidence_checks import CHECK_VERSION, EXAMPLES, assess, gate_evidence
from training.clinical_text_catalog import FIELDS


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def regression(root: Path, output: Path) -> None:
    old = json.loads((root/'audited_results.json').read_text(encoding='utf-8'))
    entries = []
    for q in old['quote_reviews']:
        if q['reasoning'] != 'sections' or q['field'] == 'audit_unmapped_relevant_evidence':
            continue
        decision, reason = assess(q['field'], q['quote']) if q['source_found'] else ('reject', 'quote_absent')
        entries.append(dict(q, check_decision=decision, check_reason=reason))
    counts = {}
    for decision in ('pass', 'needs_review', 'reject'):
        group = [q for q in entries if q['check_decision'] == decision]
        counts[decision] = {'total': len(group), **{g: sum(q['clinical_grade'] == g for q in group)
                                                 for g in ('correct', 'incorrect', 'uncertain')}}
    save(output/'old_eight_output_replay.json', {'not_new_inference': True, 'counts': counts, 'quotes': entries})


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    protocol_path = Path('training/targeted_bonsai_validation_cases.json')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    rows = {r['review_id']: r for r in read_jsonl(Path('patient_data/text_semantic_review_2026-10-03/eligible_sources/source_review.jsonl'))}
    for rid in protocol['ids']:
        for point in protocol['cases'][rid]['positive']:
            if not all(normalize(a) in normalize(rows[rid]['text']) for a in point['anchors']):
                raise ValueError(f'Ungrounded checkpoint {rid} {point}')
    props = local_api('http://127.0.0.1:8080', '/props')
    output.mkdir(parents=True)
    protocol.update(check_version=CHECK_VERSION, examples=EXAMPLES,
                    audit_status='NOT_REQUESTED_DEFERRED_FULL_CATALOG_REVIEW',
                    arms={'baseline': 'original sections prompt; presence_penalty=1.5',
                          'repaired': 'synthetic examples + auditable checks; presence_penalty=1.5',
                          'penalty_zero': 'same repaired setup; only presence_penalty=0'})
    save(output/'protocol.json', protocol)
    (output/'protocol.sha256').write_text(hashlib.sha256((output/'protocol.json').read_bytes()).hexdigest(), encoding='ascii')
    save(output/'server_props.json', props)
    regression(Path('patient_data/text_semantic_review_2026-10-03/section_validation_2026-10-04'), output)
    for number, rid in enumerate(protocol['ids']):
        row = rows[rid]
        sources = section_sources(row['text'])
        for arm in (('baseline', 'repaired', 'penalty_zero') if number == 0 else ('penalty_zero', 'repaired', 'baseline')):
            began = time.monotonic()
            parts, raw, merged = [], [], {f.key: [] for f in FIELDS if f.group != '仅审计'}
            print(f'START {rid} {arm}', flush=True)
            for task, keys, roles in TASKS:
                fields = tuple(f for f in FIELDS if f.key in keys and f.group != '仅审计')
                try:
                    part = extract_one(row, 'http://127.0.0.1:8080', 'none', fields=fields,
                                       constrain_json=True, source_blocks=task_sources(sources, roles),
                                       prompt_addition=EXAMPLES[task] if arm != 'baseline' else '',
                                       presence_penalty=0 if arm == 'penalty_zero' else 1.5)
                except Exception as error:
                    part = {'status': 'request_failed', 'error': str(error)}
                parts.append(part)
                raw.extend(collect_quotes(part, row, arm))
                for key, quotes in part.get('evidence', {}).items():
                    merged[key].extend(quotes)
                save(output/f'{rid}_{arm}_{task}_call.json', part)
                print(f'PART {rid} {arm} {task} {part["status"]} {part.get("seconds")}s', flush=True)
            usable = all('evidence' in p and p.get('finish_reason') == 'stop'
                         and all(i['type'] in {'quote_absent', 'duplicate'} for i in p.get('validation_issues', []))
                         for p in parts)
            result = {'review_id': rid, 'arm': arm, 'finish_reason': 'stop' if usable else 'failed',
                      'status': ('source_verified_only' if all(p['status'] == 'source_verified_only' for p in parts)
                                 else 'needs_source_repair') if usable else 'document_failed',
                      'end_to_end_seconds': round(time.monotonic()-began, 3),
                      'raw_quotes_for_review': raw, 'prediction_allowed': False,
                      'audit_status': protocol['audit_status']}
            if usable:
                result['evidence'] = merged
                result['checks_before_gate'] = checkpoint_report(result, protocol['cases'][rid])
                gate = gate_evidence(merged)
                result['gate'] = gate
                # 同时显示过滤前后覆盖；拒绝/待审不会伪装成真实阴性。
                passed = dict(result, evidence=gate['passed_evidence'])
                result['checks_after_gate'] = checkpoint_report(passed, protocol['cases'][rid])
                for point in result['checks_after_gate']['negative']:
                    if gate['field_states'][point['field']] in {'withheld_pending_review', 'rejected_candidates'}:
                        point['pass'] = False
                        point['usable'] = False
            else:
                result['partial_evidence_for_audit_only'] = merged
            save(output/f'{rid}_{arm}.json', result)
            print(f'END {rid} {arm} {result["end_to_end_seconds"]}s candidates={len(raw)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
