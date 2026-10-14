"""胸片报告识别、预测时点核验和章节定位；不预测影像病理标签。"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from patient_data.export_mimiciv import rows, sha256

HEADER = re.compile(r'(?im)^\s*(REASON FOR (?:EXAMINATION|EXAM)|CLINICAL HISTORY|EXAMINATION|EXAM|TECHNIQUE|INDICATION|HISTORY|COMPARISON|FINDINGS|IMPRESSION|CONCLUSION|RECOMMENDATION(?:\(S\)|S)?|NOTIFICATION)\s*:\s*')
NON_CXR = re.compile(r'\b(CT|CTA|MDCT|HRCT|COMPUTED TOMOGRAPHY|MRI|MR|ULTRASOUND|SONOGRAM|ANGIOGRAM|BIOPSY|FLUOROSCOPY)\b', re.I)
CXR = re.compile(r'\bCHEST\b', re.I)
VIEW = re.compile(r'RADIOGRAPH|\bX[ -]?RAY\b|\bPORTABLE\b|\bPORT\.?\b|\b(?:AP|PA|LAT|LATERAL|FRONTAL|VIEW|DX)\b|LINE PLACEMENT', re.I)


def identify(text):
    """检查名称优先；不因HISTORY/COMPARISON中提到胸片或CT判当前检查。"""
    headings = list(HEADER.finditer(text))
    for i,h in enumerate(headings):
        if h.group(1).upper()=='TECHNIQUE':
            end=headings[i+1].start() if i+1<len(headings) else len(text)
            technique=' '.join(text[h.end():end].split())
            if NON_CXR.search(technique): return 'non_cxr_technique', technique
    for h in headings:
        if h.group(1).upper() in ('EXAMINATION', 'EXAM'):
            tail = text[h.end():].splitlines()
            title = next((line.strip() for line in tail if line.strip()), '')
            if NON_CXR.search(title): return 'non_cxr_exam', title
            if CXR.search(title) and VIEW.search(title): return 'cxr_exam', title
            return 'unrecognized_exam', title
    stop = min((h.start() for h in headings if h.group(1).upper() not in ('TECHNIQUE',)), default=len(text))
    for line in text[:stop].splitlines()[:12]:
        title = line.strip()
        if NON_CXR.search(title): return 'non_cxr_title', title
        if CXR.search(title) and VIEW.search(title): return 'cxr_title', title
    for h in headings:
        if h.group(1).upper() == 'TECHNIQUE':
            end = next((x.start() for x in headings if x.start() > h.start()), len(text))
            title = ' '.join(text[h.end():end].split())
            if NON_CXR.search(title): return 'non_cxr_technique', title
            if CXR.search(title) and VIEW.search(title): return 'cxr_technique', title
    return 'unrecognized', ''


def report_section(text):
    headings = list(HEADER.finditer(text))
    for names, kind in ((('IMPRESSION', 'CONCLUSION'), 'impression'), (('FINDINGS',), 'findings')):
        for i, h in enumerate(headings):
            if h.group(1).upper() in names:
                start = h.end(); end = headings[i+1].start() if i+1 < len(headings) else len(text)
                value = text[start:end].strip()
                if value:
                    start += len(text[start:end]) - len(text[start:end].lstrip())
                    end = start + len(value)
                    return value, kind, start, end
    # 没有可靠章节时保留为缺失候选，不将病史/检查指征当影像结论。
    return '', 'no_findings_or_impression', 0, 0


def usable(row, admission):
    try:
        chart = datetime.fromisoformat(row['charttime']); store = datetime.fromisoformat(row['storetime'])
        start = datetime.fromisoformat(admission['admittime']); end = datetime.fromisoformat(admission['dischtime'])
    except (ValueError, KeyError): return False, 'invalid_time'
    if row['subject_id'] != admission['subject_id']: return False, 'subject_mismatch'
    if not start <= chart <= store <= end: return False, 'not_available_during_admission'
    return True, ''


def choose(reports):
    """最新有效报告，平局以存档时点/note_id确定；无报告保持缺失。"""
    return max(reports, key=lambda r: (r['charttime'], r['storetime'], r['note_id'])) if reports else None


def select(snapshot, state_dir, output):
    if output.exists(): raise FileExistsError(output)
    manifest_path = snapshot / 'metadata/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    state_manifest = json.loads((state_dir/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'validated' or state_manifest['status'] != 'derived_and_roundtrip_checked_not_prediction_validated':
        raise ValueError('unvalidated source')
    for name in ('cohort/admissions', 'raw/radiology'):
        if sha256(snapshot/(name+'.csv.gz')) != manifest['sources'][name]['sha256']:
            raise ValueError('raw source changed')
    arrays_path = state_dir/'clinical_state.npz'
    if sha256(arrays_path) != state_manifest['output_sha256']['clinical_state.npz']:
        raise ValueError('clinical members changed')
    with np.load(arrays_path, allow_pickle=False) as f:
        identities = {key:f[key] for key in ('hadm_ids', 'subject_ids', 'patient_splits')}
    admission = {r['hadm_id']:r for r in rows(snapshot/'cohort/admissions.csv.gz') if r['discharge_landmark_eligible'] == 't'}
    if set(admission) != set(map(str, identities['hadm_ids'])): raise ValueError('eligible members differ')
    chosen, audit = {}, Counter()
    for r in rows(snapshot/'raw/radiology.csv.gz'):
        audit['raw_reports'] += 1
        if r['hadm_id'] not in admission:
            audit['not_eligible_discharge'] += 1; continue
        category, title = identify(r['text']); audit['type_'+category] += 1
        if not category.startswith('cxr_'): continue
        okay, reason = usable(r, admission[r['hadm_id']])
        if not okay:
            audit[reason] += 1; continue
        selected, section, start, end = report_section(r['text'])
        if not selected:
            audit['no_findings_or_impression'] += 1; continue
        if r['text'][start:end] != selected or not 0 <= start <= end <= len(r['text']):
            raise ValueError('section provenance mismatch')
        record = {k:r[k] for k in ('hadm_id', 'subject_id', 'note_id', 'charttime', 'storetime')}
        record.update(text=selected, section=section, start_char=start, end_char=end, examination=title,
                      original_sha256=hashlib.sha256(r['text'].encode()).hexdigest(),
                      section_sha256=hashlib.sha256(selected.encode()).hexdigest())
        previous = chosen.get(r['hadm_id'])
        chosen[r['hadm_id']] = choose([record] + ([previous] if previous else []))
        audit['qualified_reports'] += 1
    if audit['raw_reports'] != manifest['sources']['raw/radiology']['rows']: raise ValueError('radiology count changed')
    output.mkdir(parents=True)
    records = []
    for hadm, subject, split in zip(identities['hadm_ids'], identities['subject_ids'], identities['patient_splits'], strict=True):
        r = chosen.get(str(hadm))
        if r:
            if r['subject_id'] != str(subject): raise ValueError('subject alignment failed')
            r['patient_split'] = str(split)
            r['days_before_discharge'] = (datetime.fromisoformat(admission[str(hadm)]['dischtime'])-datetime.fromisoformat(r['charttime'])).total_seconds()/86400
            records.append(r); audit['selected_'+r['section']] += 1
    destination = output/'selected_reports.jsonl'
    with destination.open('x', encoding='utf-8') as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    np.savez_compressed(output/'members.npz', **identities)
    result = {'status':'selected_temporally_checked_not_labeled', 'admissions':len(admission), 'reports_selected':len(records),
              'patients_with_report':len({r['subject_id'] for r in records}), 'audit':dict(audit),
              'parent_state_manifest_sha256':sha256(state_dir/'manifest.json'), 'snapshot_manifest_sha256':sha256(manifest_path),
              'raw_radiology_sha256':manifest['sources']['raw/radiology']['sha256'], 'code_sha256':sha256(Path(__file__)),
              'output_sha256':{name:sha256(output/name) for name in ('selected_reports.jsonl','members.npz')},
              'policy':'one latest chest-radiograph report with admission <= charttime <= storetime <= discharge; Impression preferred, Findings fallback',
              'limits':['Conservative title matcher and section availability may omit real CXRs', 'Reports only, no image pixels or CT morphology']}
    (output/'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--state-dir',type=Path,default=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); select(args.snapshot,args.state_dir,args.output)
