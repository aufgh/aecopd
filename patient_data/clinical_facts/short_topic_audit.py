"""已完成短主题先导的独立回查与保守二次门禁；不改原始LLM输出。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from patient_data.export_mimiciv import sha256
from patient_data.clinical_facts.features import discharge_vector, DISCHARGE_NAMES
from patient_data.clinical_facts.catalog import CATALOG_SHA


def extra_guard(fact, task):
    field, value, quote = fact['field'], fact['value'], fact['quote']
    if field == 'exertional_desaturation':
        if not re.search(r'\b(?:ambulat\w*|walk\w*|exert\w*|exerc\w*|activity)\b', quote, re.I):
            return 'oxygenation_without_exertional_context'
        if not re.search(r'desatur|\b(?:SpO2|O2|sat(?:uration|ted|ting)?s?|hypox\w*|oxygen)\b', quote, re.I):
            return 'dyspnea_or_exercise_limitation_not_desaturation'
    if field in ('steroid_discharge_course', 'respiratory_antibiotic_course') and value == 'completed':
        if re.search(r'\b(?:will|remaining|to complete|your last day|completes|to finish)\b', quote, re.I):
            return 'future_completion_not_already_completed'
        if re.search(r'discontinu|stopped|not indicated', quote, re.I):
            return 'stopped_not_completed_course'
        if not re.search(r'\b(?:completed|finished|received.*course)\b', quote, re.I):
            return 'completion_requires_explicit_past_evidence'
    if field == 'steroid_discharge_course' and value == 'planned':
        if not re.search(r'\b(?:taper|continue|prescrib\w*|\d+\s*mg|remaining|for.*days?)\b|\b(?:discharg\w*|home)\b.{0,40}\b(?:on|with)\b', quote, re.I):
            return 'inpatient_response_not_discharge_steroid_plan'
    if field in ('inhaled_ics', 'inhaled_laba', 'inhaled_lama') and fact['section_role'] == 'hospital_course':
        if re.search(r'formulary|instead of home', quote, re.I) and not re.search(r'\bdischarg\w*\b', quote, re.I):
            return 'inpatient_substitution_not_discharge_regimen'
    return None


def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    status = json.loads((args.results / 'status.json').read_text())
    if status['status'] != 'completed': raise ValueError('pilot is unfinished')
    manifest = json.loads((args.tasks / 'manifest.json').read_text())
    contract = json.loads((args.results / 'contract.json').read_text())
    if contract['tasks_manifest_sha256'] != sha256(args.tasks / 'manifest.json') or contract['catalog_sha256'] != CATALOG_SHA:
        raise ValueError('pilot source contract differs')
    for name, digest in manifest['output_sha256'].items():
        if sha256(args.tasks / name) != digest: raise ValueError('pilot source changed')
    if sha256(args.results / 'results.jsonl') != status['results_sha256']: raise ValueError('pilot results changed')
    with (args.tasks / 'tasks.jsonl').open(encoding='utf-8') as f: tasks = {r['task_key']: r for r in map(json.loads, f)}
    with (args.tasks / 'sources.jsonl').open(encoding='utf-8') as f: sources = {r['note_id']: r for r in map(json.loads, f)}
    with (args.results / 'results.jsonl').open(encoding='utf-8') as f: results = {r['task_key']: r for r in map(json.loads, f)}
    if tasks.keys() != results.keys(): raise ValueError('pilot tasks incomplete')
    accepted = []; rejected = []; verified = 0
    task_seconds = []; token_counts = Counter(); reason_chars = 0; route_stats = {}
    for key, task in tasks.items():
        result = results[key]; task_seconds.append(result['seconds'])
        usage = result['response'].get('usage', {})
        token_counts.update({k: v for k, v in usage.items() if isinstance(v, (int, float))})
        for usage_key, values in usage.items():
            if isinstance(values, dict):
                token_counts.update({usage_key + '.' + k: v for k, v in values.items() if isinstance(v, (int, float))})
        reason_chars += len(result['response'].get('reasoning') or '')
        for fact in result['facts']:
            if sources[task['note_id']]['text'][fact['original_start']:fact['original_end']] != fact['quote']:
                raise ValueError('verbatim original evidence mismatch')
            verified += 1
            row = dict(fact, task_key=key, hadm_id=task['hadm_id'], note_id=task['note_id'])
            error = extra_guard(fact, task)
            if error: rejected.append(dict(row, reason=error))
            else: accepted.append(row)
        route_stats.setdefault(task['route'], Counter())
        route_stats[task['route']]['tasks'] += 1
        route_stats[task['route']]['accepted_before_postcheck'] += len(result['facts'])
    members = {k: v for k, v in np.load(args.tasks / 'members.npz').items()}
    hadms = sorted({r['hadm_id'] for r in sources.values()}, key=int)
    by_hadm = {h: [f for f in accepted if f['hadm_id'] == h] for h in hadms}
    values = np.stack([discharge_vector(by_hadm[h]) for h in hadms])
    args.output.mkdir(parents=True)
    for name, rows in (('accepted_evidence.jsonl', accepted), ('rejected_evidence.jsonl', rejected)):
        with (args.output / name).open('x', encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False) + '\n')
    np.savez_compressed(args.output / 'pilot_facts.npz', hadm_ids=np.array(hadms), values=values, names=np.array(DISCHARGE_NAMES))
    report = dict(status='completed_short_topic_pilot_source_audit_not_prediction_validated', notes=len(hadms), tasks=len(tasks),
                  thinking=contract['settings']['thinking'], parallel=contract['parallel'], accepted_before_postcheck=verified,
                  all_retained_quotes_source_checked=True, accepted_after_postcheck=len(accepted),
                  source_notes_with_any_fact=sum(bool(fs) for fs in by_hadm.values()), postcheck_reasons=dict(Counter(r['reason'] for r in rejected)),
                  by_field=dict(Counter(f['field'] for f in accepted)), by_route={k: dict(v) for k, v in route_stats.items()},
                  request_seconds_median=float(np.median(task_seconds)), request_seconds_p90=float(np.quantile(task_seconds, .9)),
                  total_request_seconds=float(sum(task_seconds)), completion_rate=status['successful'] / status['total'],
                  raw_output_rejections=status['rejection_counts'], usage=dict(token_counts), reasoning_characters=reason_chars,
                  source_manifest_sha256=sha256(args.tasks / 'manifest.json'), result_sha256=sha256(args.results / 'results.jsonl'),
                  code_sha256=sha256(Path(__file__)),
                  output_sha256={name: sha256(args.output / name) for name in ('accepted_evidence.jsonl', 'rejected_evidence.jsonl', 'pilot_facts.npz')},
                  limitations=['No expert accuracy estimate or recall gold standard', '96-note pilot cannot establish predictive benefit',
                               'Postcheck developed from error inspection and is not an independent validation accuracy',
                               'Raw source/output preserved; uncertain facts excluded, never converted to normal',
                               'Explicit normality and unmentioned remain distinct; no BERT fine-tuning'])
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, default=Path('patient_data/processed/v6_short_topics_96_2026-10-10'))
    p.add_argument('--results', type=Path, default=Path('patient_data/processed/v6_short_topics_results_96_2026-10-10_r2'))
    p.add_argument('--output', type=Path, required=True)
    build(p.parse_args())
