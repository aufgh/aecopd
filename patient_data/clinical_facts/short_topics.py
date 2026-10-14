"""小结非思考短主题先导：固定随机选择、按章节限域、保留候选片段完整偏移。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from patient_data.export_mimiciv import sha256
from patient_data.clinical_facts.catalog import ROUTES, CATALOG_SHA
from patient_data.clinical_facts.tasks import blocks
from patient_data.clinical_facts.llm import digest

PATTERNS = {
    'respiratory': r'dyspn|shortness of breath|breathless|\bSOB\b|oxygen|\bO2\b|room air|cannula|desatur|BiPAP|CPAP|NIV|NIPPV|ventilat|intubat|extubat',
    'medications': r'predn|methylpred|steroid|hydrocortisone|dexamethasone|antibiotic|azithro|levoflox|moxiflox|doxycycl|ceftriax|amoxic|augmentin|tiotrop|spiriva|salmeterol|formoterol|budesonide|fluticasone|advair|symbicort|inhaler|COPD|pneumonia|respiratory',
    'function_care': r'ADL|depend|assist|ambulat|walker|wheelchair|mobility|rehab|SNF|home|lives?|living|alone|adher|complian|afford|spacer|device|inhaler|DNR|DNI|hospice|comfort|AMA|against medical',
}


def candidates(section, route):
    """优先整段保留；这是可审计的候选筛选，不宣称语义完备或无漏检。"""
    text = section['text']
    if section['role'] not in ('hospital_course', 'discharge_instructions', 'transitional_issues'):
        return [section]
    result = []
    for match in re.finditer(r'[^\n]+(?:\n(?!\s*\n)[^\n]+)*', text):
        if re.search(PATTERNS[route], match.group(), re.I):
            result.append(dict(section, text=match.group(), start_char=section['start_char'] + match.start(),
                               end_char=section['start_char'] + match.end()))
    return result


def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    m = json.loads((args.source / 'manifest.json').read_text())
    for name, value in m['output_sha256'].items():
        if sha256(args.source / name) != value: raise ValueError('source changed')
    with np.load(args.source / 'members.npz') as f: members = {k: f[k] for k in f.files}
    with (args.source / 'sources.jsonl').open(encoding='utf-8') as f:
        sources = [r for r in map(json.loads, f) if r['channel'] == 'discharge']
    fold = {str(h): s for h, s in zip(members['hadm_ids'], members['patient_splits'], strict=True)}
    rng = np.random.default_rng(42); selected = []
    for name in ('train', 'validation', 'test'):
        eligible = [r for r in sources if fold[r['hadm_id']] == name]
        positions = rng.choice(len(eligible), min(args.per_fold, len(eligible)), replace=False) if args.per_fold else range(len(eligible))
        selected.extend(eligible[i] for i in positions)
    tasks = []; audit = Counter()
    for record in selected:
        for route in PATTERNS:
            sections = [s for s in record['sections'] if s['role'] in ROUTES[route]['roles']]
            spans = [p for s in sections for p in candidates(s, route)]
            audit[route + '_source_chars'] += sum(len(s['text']) for s in sections)
            audit[route + '_selected_chars'] += sum(len(s['text']) for s in spans)
            for b in blocks(spans, limit=2400):
                task = {k: record[k] for k in ('hadm_id', 'subject_id', 'note_id', 'charttime', 'storetime', 'original_sha256')}
                task.update(route=route, **b)
                task['task_key'] = digest(json.dumps(task, sort_keys=True) + CATALOG_SHA)
                for p in task['source_maps']:
                    if record['text'][p['original_start']:p['original_end']] != task['text'][p['input_start']:p['input_end']]:
                        raise ValueError('source offset mismatch')
                tasks.append(task)
    args.output.mkdir(parents=True)
    for name, values in (('tasks.jsonl', tasks), ('sources.jsonl', selected)):
        with (args.output / name).open('x', encoding='utf-8') as f:
            for r in values: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    np.savez_compressed(args.output / 'members.npz', **members)
    result = dict(status='llm_tasks_source_checked', catalog_sha256=CATALOG_SHA, tasks=len(tasks),
                  source_counts={'discharge': len(selected)}, task_counts=dict(Counter(t['route'] for t in tasks)),
                  selected_folds=dict(Counter(fold[r['hadm_id']] for r in selected)), audit=dict(audit),
                  parent_state_manifest_sha256=m['parent_state_manifest_sha256'],
                  source_manifest_sha256=sha256(args.source / 'manifest.json'), code_sha256=sha256(Path(__file__)),
                  per_fold=args.per_fold, policy='Non-thinking short topics; 96-note pilot by default, not a full predictive experiment; keyword preselection can miss evidence',
                  output_sha256={n: sha256(args.output / n) for n in ('tasks.jsonl', 'sources.jsonl', 'members.npz')})
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('patient_data/processed/v6_bonsai_tasks_2026-10-10_validated'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-fold', type=int, default=32)
    a = p.parse_args()
    if a.per_fold < 0: p.error('invalid per-fold count')
    build(a)
