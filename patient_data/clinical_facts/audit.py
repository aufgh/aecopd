"""直接回查原始快照与全部报告/字符/成员，不把解析率当临床准确度。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.clinical_facts.llm import digest
from patient_data.cxr_report_selection import usable
from patient_data.clinical_facts.cxr_trajectory import SELECTED

def audit_cxr(args):
    m=json.loads((args.cxr/'manifest.json').read_text())
    if m['status']!='all_cxr_labeled_and_aggregated': raise ValueError('not completed')
    for name,value in m['output_sha256'].items():
        if sha256(args.cxr/name)!=value: raise ValueError('changed CXR files')
    raw={r['note_id']:r for r in rows(args.snapshot/'raw/radiology.csv.gz')}
    admissions={r['hadm_id']:r for r in rows(args.snapshot/'cohort/admissions.csv.gz')}
    with (args.cxr/'labeled_reports.jsonl').open(encoding='utf-8') as f: reports=list(map(json.loads,f))
    grouped=defaultdict(list); repeats=0
    for r in reports:
        source=raw[r['note_id']]
        if any(r[k]!=source[k] for k in ('hadm_id','subject_id','charttime','storetime')): raise ValueError('raw identity mismatch')
        if digest(source['text'])!=r['original_sha256'] or not usable(source,admissions[r['hadm_id']])[0]: raise ValueError('raw source/time failed')
        for s in r['sections']:
            if source['text'][s['start_char']:s['end_char']]!=s['text']: raise ValueError('source span mismatch')
        if r['text']!='\n'.join(s['text'] for s in r['sections']): raise ValueError('combined report differs')
        end=0
        for chunk in r['chunks']:
            if chunk['start_char']!=end or chunk['end_char']<=end or chunk['tokens']>512: raise ValueError('lost or truncated chunk')
            end=chunk['end_char']
        if end!=len(r['text']): raise ValueError('missing report tail')
        if not all(v in (0,1,2,3) for v in r['labels14'].values()): raise ValueError('invalid labels')
        grouped[r['hadm_id']].append(r)
    with np.load(args.cxr/'cxr_trajectory.npz') as f: a={k:f[k] for k in f.files}
    for i,(h,s) in enumerate(zip(a['hadm_ids'],a['subject_ids'],strict=True)):
        rs=grouped.get(str(h),[]); expected=bool(rs)
        if a['cxr_present'][i]!=expected: raise ValueError('availability incorrectly includes absent reports')
        if not rs:
            if a['cxr_values'][i].any(): raise ValueError('absent report has nonzero input')
            continue
        if any(r['subject_id']!=str(s) or r['patient_split']!=a['patient_splits'][i] for r in rs): raise ValueError('patient/split failed')
        times=sorted({r['charttime'] for r in rs}); repeats+=len(times)>1
        x=a['cxr_values'][i].reshape(7,14)
        for j,field in enumerate(SELECTED):
            sequence=[]
            for t in times:
                states={r['labels14'][field] for r in rs if r['charttime']==t}-{0}
                code=3 if {1,2}<=states else 1 if 1 in states else 3 if 3 in states else 2 if 2 in states else 0
                sequence.append(code)
            if x[j,sequence[0]]!=1 or x[j,4+sequence[-1]]!=1 or x[j,:4].sum()!=1 or x[j,4:8].sum()!=1:
                raise ValueError('first/latest four-state vector failed')
            neg=any(sequence[k]==2 and 1 in sequence[:k] for k in range(len(sequence)))
            if x[j,10]!=neg or x[j,8]!=(1 in sequence): raise ValueError('temporal change failed')
    if len(reports)!=m['reports'] or int(a['cxr_present'].sum())!=m['admissions_with_cxr']: raise ValueError('coverage differs')
    result={'status':'all_CXR_raw_source_and_trajectory_checked','reports':len(reports),'admissions_with_cxr':len(grouped),
            'repeat_timestamp_admissions':repeats,'source_chars_time_members_masks_checked':True,
            'semantic_accuracy':'not an expert gold standard','manifest_sha256':sha256(args.cxr/'manifest.json')}
    (args.cxr/'independent_source_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)

def audit_tasks(args):
    m=json.loads((args.tasks/'manifest.json').read_text())
    for name,value in m['output_sha256'].items():
        if sha256(args.tasks/name)!=value: raise ValueError('task/source changed')
    with (args.tasks/'sources.jsonl').open(encoding='utf-8') as f: sources={r['note_id']:r for r in map(json.loads,f)}
    verified=0
    for name in ('radiology','discharge'):
        for raw in rows(args.snapshot/('raw/'+name+'.csv.gz')):
            if raw['note_id'] not in sources: continue
            source=sources[raw['note_id']]
            if any(source[k]!=raw[k] for k in ('subject_id','hadm_id','charttime','storetime','text')):
                raise ValueError('task reference is not complete original note')
            if digest(raw['text'])!=source['original_sha256']: raise ValueError('original note hash failed')
            verified+=1
    if verified!=len(sources): raise ValueError('source note missing from frozen snapshot')
    total=0; maps=0; seen=set()
    with (args.tasks/'tasks.jsonl').open(encoding='utf-8') as f:
        for task in map(json.loads,f):
            if task['task_key'] in seen: raise ValueError('duplicate task key')
            seen.add(task['task_key']); source=sources[task['note_id']]
            for mapping in task['source_maps']:
                actual=source['text'][mapping['original_start']:mapping['original_end']]
                if actual!=task['text'][mapping['input_start']:mapping['input_end']]: raise ValueError('input/original offset failed')
                maps+=1
            total+=1
    if total!=m['tasks']: raise ValueError('task count differs')
    result={'status':'all_tasks_original_notes_and_offsets_checked','original_notes':verified,'tasks':total,'mapped_spans':maps,
            'manifest_sha256':sha256(args.tasks/'manifest.json'),'limits':'Source correctness, not semantic extraction accuracy'}
    (args.tasks/'independent_source_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--cxr',type=Path,default=Path('patient_data/processed/v6_cxr_trajectory_2026-10-10_r2'))
    p.add_argument('--tasks',type=Path); args=p.parse_args()
    audit_tasks(args) if args.tasks else audit_cxr(args)
