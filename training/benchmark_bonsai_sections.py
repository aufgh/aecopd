"""有限的全文/章节任务配对对照；不编码、不训练、不读取结局。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from training.benchmark_bonsai_verbatim import checkpoint_report, normalize
from training.bonsai_clinical_evidence_pilot import (
    decode_content, extract_one, find_source_spans, local_api, read_jsonl, sources_for_note,
)
from training.clinical_text_catalog import FIELDS, catalog_dict


TASKS = (
    ('state', ('episode_diagnosis', 'dyspnea', 'cough_sputum', 'airflow_resp_distress',
               'oxygenation', 'hypercapnia_acidosis', 'respiratory_recovery',
               'oxygen_requirement_transition', 'ventilation_support'),
     ('chief_complaint', 'history_of_present_illness', 'physical_exam', 'admission_physical_exam',
      'discharge_physical_exam', 'hospital_course', 'discharge_condition', 'discharge_diagnosis',
      'discharge_instructions', 'transitional_issues', 'pertinent_results', 'imaging')),
    ('history', ('baseline_respiratory_severity', 'prior_respiratory_exacerbations',
                 'cardiac_contributors', 'thromboembolic_differential', 'other_respiratory_conditions',
                 'aspiration_dysphagia', 'functional_limitations', 'nutrition_frailty',
                 'mental_cognitive_barriers', 'smoking_exposure', 'home_support_barriers',
                 'other_relevant_comorbidity'),
     ('past_medical_history', 'social_history', 'family_history', 'history_of_present_illness',
      'hospital_course', 'physical_exam', 'discharge_condition', 'discharge_diagnosis',
      'pertinent_results', 'imaging', 'discharge_instructions', 'admission_physical_exam',
      'discharge_physical_exam')),
    ('treatment', ('maintenance_rescue_inhalers', 'systemic_steroid_course',
                   'respiratory_antiinfective_course', 'treatment_failure_intolerance'),
     ('admission_medications', 'discharge_medications', 'hospital_course', 'history_of_present_illness',
      'allergies', 'discharge_instructions', 'transitional_issues', 'major_procedure')),
    ('tests', ('respiratory_infection', 'respiratory_microbiology', 'thoracic_imaging',
               'respiratory_lab_interpretation'),
     ('pertinent_results', 'imaging', 'microbiology', 'hospital_course', 'history_of_present_illness',
      'admission_labs', 'discharge_labs', 'discharge_diagnosis', 'discharge_instructions')),
    ('management', ('inhaler_technique', 'respiratory_adherence', 'medication_equipment_access',
                    'respiratory_followup_rehab', 'care_limits_ama', 'audit_unmapped_relevant_evidence'),
     ('hospital_course', 'social_history', 'past_medical_history', 'history_of_present_illness',
      'discharge_instructions', 'followup_instructions', 'transitional_issues', 'discharge_condition',
      'discharge_disposition', 'admission_medications', 'discharge_medications')),
)


def section_sources(original: str) -> list[dict]:
    """主章节+问题小标题；补齐未识别/前导正文，所有原文位置可复核。"""
    blocks = sources_for_note(original)
    complete, cursor = [], 0
    for block in sorted(blocks, key=lambda s: s['start']):
        if original[cursor:block['start']].strip():
            complete.append({'role': 'unclassified', 'start': cursor, 'end': block['start'],
                             'text': original[cursor:block['start']]})
        complete.append(block)
        cursor = block['end']
    if original[cursor:].strip():
        complete.append({'role': 'unclassified', 'start': cursor, 'end': len(original),
                         'text': original[cursor:]})
    split = []
    for block in complete:
        # 保留全部问题，而不是按病名删除非呼吸问题；模型仍需要鉴别适应证。
        boundaries = [m.start() for m in re.finditer(r'(?m)^\s*#[^\r\n]+', block['text'])]
        starts = sorted({0, *boundaries, len(block['text'])})
        for left, right in zip(starts, starts[1:]):
            text = block['text'][left:right]
            if not text.strip():
                continue
            heading = text.lstrip().splitlines()[0] if text.lstrip().startswith('#') else ''
            split.append({'source_id': f'S{len(split)+1:03}', 'role': block['role'],
                          'problem_heading': heading, 'start': block['start']+left,
                          'end': block['start']+right, 'text': text})
    return split


def task_sources(sources: list[dict], roles: tuple[str, ...]) -> list[dict]:
    # 未识别内容提供给各任务兜底；缺失标题不能被当作没有临床内容。
    return [s for s in sources if s['role'] in roles or s['role'] == 'unclassified']


def collect_quotes(result: dict, row: dict, arm: str) -> list[dict]:
    try:
        data = decode_content(result['raw_response']['choices'][0]['message'].get('content') or '')
    except (KeyError, ValueError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    quotes = []
    for field, values in data.items():
        if not isinstance(values, list):
            continue
        for index, quote in enumerate(values):
            if isinstance(quote, str):
                matches = find_source_spans(quote, row['text'])
                quotes.append({'id': f'{row["review_id"]}:{arm}:{field}:{index}', 'field': field,
                               'quote': quote, 'source_found': bool(matches),
                               'clinical_grade': 'not_reviewed'})
    return quotes


def run(output: Path, cases_path: Path, source_path: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(cases_path.read_text(encoding='utf-8'))
    if 'base_cases' in protocol:
        base = json.loads(Path(protocol['base_cases']).read_text(encoding='utf-8'))
        protocol['cases'] = {**base['cases'], **protocol['cases']}
        for rid, keys in protocol.get('remove_empty', {}).items():
            protocol['cases'][rid]['empty'] = [k for k in protocol['cases'][rid]['empty'] if k not in keys]
    if len(protocol['ids']) > 8:
        raise ValueError('At most eight documents; this is not a batch extraction')
    assigned = [key for _, keys, _ in TASKS for key in keys]
    if len(assigned) != len(set(assigned)) or set(assigned) != {f.key for f in FIELDS}:
        raise ValueError('Each fixed field must have exactly one owner task')
    rows = {r['review_id']: r for r in read_jsonl(source_path)}
    for rid in protocol['ids']:
        for fact in protocol['cases'][rid]['positive']:
            if not all(normalize(a) in normalize(rows[rid]['text']) for a in fact['anchors']):
                raise ValueError(f'Ungrounded prespecified checkpoint {rid} {fact}')
    props = local_api('http://127.0.0.1:8080', '/props')
    output.mkdir(parents=True)
    for name, value in [('predeclared_cases.json', protocol), ('server_props.json', props),
                        ('field_catalog.json', catalog_dict()), ('task_routing.json', TASKS)]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'predeclared_cases.sha256').write_text(hashlib.sha256((output/'predeclared_cases.json').read_bytes()).hexdigest(), encoding='ascii')
    for index, rid in enumerate(protocol['ids']):
        row = rows[rid]
        sources = section_sources(row['text'])
        for arm in (('full', 'sections') if index % 2 == 0 else ('sections', 'full')):
            began = time.monotonic()
            tasks = [('full', tuple(f.key for f in FIELDS), tuple())] if arm == 'full' else TASKS
            parts, raw_quotes, merged = [], [], {f.key: [] for f in FIELDS}
            print(f'START {rid} {arm}', flush=True)
            for name, keys, roles in tasks:
                relevant = sources if arm == 'full' else task_sources(sources, roles)
                fields = tuple(f for f in FIELDS if f.key in keys)
                try:
                    result = extract_one(row, 'http://127.0.0.1:8080', 'none', fields=fields,
                                         constrain_json=True, source_blocks=relevant)
                except Exception as error:
                    result = {'review_id': rid, 'status': 'request_failed', 'error': str(error)}
                result['task'] = name
                parts.append(result)
                raw_quotes.extend(collect_quotes(result, row, arm))
                for key, vals in result.get('evidence', {}).items():
                    seen = {normalize(q['quote']) for q in merged[key]}
                    merged[key].extend(q for q in vals if normalize(q['quote']) not in seen)
                (output/f'{rid}_{arm}_{name}_call.json').write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f'PART {rid} {arm} {name} {result["status"]} {result.get("seconds")}s', flush=True)
            usable = all('evidence' in p and p.get('finish_reason') == 'stop'
                         and all(i['type'] in {'quote_absent', 'duplicate'} for i in p.get('validation_issues', []))
                         for p in parts)
            result = {'review_id': rid, 'arm': arm, 'reasoning': arm,
                      'end_to_end_seconds': round(time.monotonic()-began, 3),
                      'finish_reason': 'stop' if usable else 'failed', 'status': 'source_verified_only'
                      if usable and all(p['status'] == 'source_verified_only' for p in parts)
                      else ('needs_source_repair' if usable else 'document_failed'),
                      'task_status': {p['task']: p['status'] for p in parts},
                      'raw_quotes_for_review': raw_quotes, 'prediction_allowed': False}
            if usable:
                result['evidence'] = merged
            else:
                result['partial_evidence_for_audit_only'] = merged
            result['checks'] = checkpoint_report(result, protocol['cases'][rid])
            (output/f'{rid}_{arm}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'END {rid} {arm} {result["end_to_end_seconds"]}s quotes={len(raw_quotes)} '
                  f'facts={sum(p["detected"] for p in result["checks"]["positive"])}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, default=Path('training/clinical_section_validation_cases.json'))
    parser.add_argument('--sources', type=Path, default=Path('patient_data/text_semantic_review_2026-10-03/eligible_sources/source_review.jsonl'))
    args = parser.parse_args()
    run(args.output, args.cases, args.sources)
