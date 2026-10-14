"""固定成员，选择最终小结，逐章节覆盖的可续跑LLM任务清单。"""
import argparse
from collections import Counter
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.clinical_facts.catalog import ROUTES,CATALOG_SHA
from patient_data.clinical_facts.llm import digest
from training.v5.section_alignment_data.section_parser import parse_sections

def blocks(sections, limit=12000):
    """完整覆盖，不用任意首尾；输入位置映射支持代码端原文回查。"""
    text=''; maps=[]; output=[]
    for section in sections:
        source=section['text']; pos=0
        while pos<len(source):
            header='\nSECTION: '+section['role']+'\n'
            room=limit-len(text)-len(header)
            if room<100:
                output.append({'text':text,'source_maps':maps}); text=''; maps=[]; continue
            end=min(len(source),pos+room)
            if end<len(source):
                boundary=max(source.rfind('\n',pos+room//2,end),source.rfind('. ',pos+room//2,end))
                if boundary>pos: end=boundary+1
            text+=header; start=len(text); text+=source[pos:end]
            maps.append({'role':section['role'],'input_start':start,'input_end':len(text),
                         'original_start':section['start_char']+pos,'original_end':section['start_char']+end})
            pos=end
    if text: output.append({'text':text,'source_maps':maps})
    # 检验每个原文章节完整且只覆盖一次。
    for section in sections:
        spans=[m for b in output for m in b['source_maps'] if m['role']==section['role']
               and section['start_char']<=m['original_start']<section['end_char']]
        if sum(m['original_end']-m['original_start'] for m in spans)!=len(section['text']):
            raise ValueError('lost source section characters')
    return output

def make_tasks(record,sections,route):
    result=[]
    for block in blocks(sections):
        task={k:record[k] for k in ('hadm_id','subject_id','note_id','charttime','storetime','original_sha256')}
        task.update(route=route,**block)
        task['task_key']=digest(json.dumps(task,sort_keys=True)+CATALOG_SHA)
        result.append(task)
    return result

def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    raw=json.loads((args.snapshot/'metadata/manifest.json').read_text())
    selected=json.loads((args.imaging/'manifest.json').read_text())
    if sha256(args.snapshot/'raw/discharge.csv.gz')!=raw['sources']['raw/discharge']['sha256']: raise ValueError('discharge source changed')
    if sha256(args.snapshot/'raw/radiology.csv.gz')!=raw['sources']['raw/radiology']['sha256']: raise ValueError('radiology source changed')
    for name,value in selected['output_sha256'].items():
        if sha256(args.imaging/name)!=value: raise ValueError('imaging source changed')
    with np.load(args.imaging/'members.npz') as f: members={k:f[k] for k in f.files}
    identities={str(h):str(s) for h,s in zip(members['hadm_ids'],members['subject_ids'],strict=True)}
    admissions={r['hadm_id']:r for r in rows(args.snapshot/'cohort/admissions.csv.gz') if r['hadm_id'] in identities}
    final={}; audit=Counter()
    for r in rows(args.snapshot/'raw/discharge.csv.gz'):
        audit['raw_discharge']+=1
        if r['hadm_id'] not in identities: continue
        if r['subject_id']!=identities[r['hadm_id']]: raise ValueError('discharge subject differs')
        for key in ('charttime','storetime'): datetime.fromisoformat(r[key])
        if not r['text'].strip(): continue
        old=final.get(r['hadm_id'])
        if old is None or (r['charttime'],r['storetime'],r['note_id'])>(old['charttime'],old['storetime'],old['note_id']): final[r['hadm_id']]=r
    tasks=[]; references=[]; eligible=set()
    original_radiology={r['note_id']:r for r in rows(args.snapshot/'raw/radiology.csv.gz')}
    for line in (args.imaging/'chest_ct_reports.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        sections=[dict(s,role=s['kind']) for s in r['sections']]
        original=original_radiology[r['note_id']]
        if digest(original['text'])!=r['original_sha256']: raise ValueError('CT original hash differs')
        for s in sections:
            if original['text'][s['start_char']:s['end_char']]!=s['text']: raise ValueError('CT original span differs')
        tasks.extend(make_tasks(r,sections,'chest_ct'))
        # 偏移指向完整原始报告，而非合并后的Findings/Impression文本。
        references.append(dict(r,text=original['text'],channel='ct'))
    for hadm,r in sorted(final.items(),key=lambda v:int(v[0])):
        r=dict(r,original_sha256=digest(r['text']))
        spans=[dict(role=s.role,start_char=s.start_char,end_char=s.end_char,text=s.text)
               for s in parse_sections(r['text']) if s.role in ROUTES['discharge_all']['roles'] and s.content_status=='informative']
        if r['storetime']>admissions[hadm]['dischtime']: audit['final_discharge_stored_after_discharge']+=1
        audit['selected_discharge']+=1
        if spans:
            eligible.add(hadm); tasks.extend(make_tasks(r,spans,'discharge_all'))
        else: audit['no_routed_sections']+=1
        references.append(dict(r,sections=spans,channel='discharge'))
    args.output.mkdir(parents=True)
    for name,records in (('tasks.jsonl',tasks),('sources.jsonl',references)):
        with (args.output/name).open('x',encoding='utf-8') as f:
            for r in records: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    np.savez_compressed(args.output/'members.npz',**members)
    result={'status':'llm_tasks_source_checked','tasks':len(tasks),'task_counts':dict(Counter(t['route'] for t in tasks)),
            'source_counts':dict(Counter(r['channel'] for r in references)), 'discharge_routed_admissions':len(eligible),
            'catalog_sha256':CATALOG_SHA,'audit':dict(audit),'snapshot_manifest_sha256':sha256(args.snapshot/'metadata/manifest.json'),
            'imaging_manifest_sha256':sha256(args.imaging/'manifest.json'),'parent_state_manifest_sha256':selected['parent_state_manifest_sha256'],
            'code_sha256':sha256(Path(__file__)),'output_sha256':{n:sha256(args.output/n) for n in ('tasks.jsonl','sources.jsonl','members.npz')},
            'policy':'Retrospective FINAL summary as discharge data, including later storage; unsuitable for claimed prospective real-time availability'}
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--imaging',type=Path,default=Path('patient_data/processed/v6_imaging_all_2026-10-10_r2'))
    p.add_argument('--output',type=Path,required=True); build(p.parse_args())
