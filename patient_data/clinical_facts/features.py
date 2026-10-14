"""事实转类别输入：未知/阴性/冲突分离；禁止将半完成抽取送入正式训练。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from patient_data.clinical_facts.catalog import CT_FIELDS,DISCHARGE_FIELDS,CATALOG_SHA

CT_NAMES=tuple(name+':'+value for name in CT_FIELDS for value in (*CT_FIELDS[name],'observed','conflict','ever_positive'))
DISCHARGE_NAMES=tuple(name+':'+value for name in DISCHARGE_FIELDS for value in (*DISCHARGE_FIELDS[name],'observed','conflict'))

def ct_vector(facts):
    output=[]
    for field in CT_FIELDS:
        selected=[f for f in facts if f['field']==field]
        latest=max((f['charttime'] for f in selected),default=None)
        values={f['value'] for f in selected if f['charttime']==latest}
        conflict='present' in values and 'absent' in values
        # 同时刻明确相反，不能挑一份当正常。保留疑似与冲突。
        if conflict: values={'uncertain'}
        output.extend([float(v in values) for v in CT_FIELDS[field]])
        output.extend([float(bool(selected)),float(conflict),float(any(f['value']=='present' for f in selected))])
    return np.array(output,np.float32)

def discharge_vector(facts):
    output=[]
    priorities={'discharge_medications':4,'discharge_physical_exam':4,'discharge_condition':4,
                'discharge_disposition':4,'discharge_instructions':3,'transitional_issues':3,'social_history':2,'hospital_course':1}
    for field,possible in DISCHARGE_FIELDS.items():
        selected=[f for f in facts if f['field']==field]
        best=max((priorities.get(f['section_role'],0) for f in selected),default=None)
        values={f['value'] for f in selected if priorities.get(f['section_role'],0)==best}
        # DNR+DNI和残余/改善可共存，其余互斥状态不静默选一个。
        conflict=len(values)>1 and field not in ('care_limits','residual_dyspnea')
        output.extend(float(v in values) for v in possible)
        output.extend([float(bool(selected)),float(conflict)])
    return np.array(output,np.float32)

def read_results(directory,tasks):
    status=json.loads((directory/'status.json').read_text())
    contract=json.loads((directory/'contract.json').read_text())
    if status['status']!='completed' or contract['limit'] or contract['catalog_sha256']!=CATALOG_SHA:
        raise ValueError('full extraction has not finished; no partial facts in formal input')
    for name,value in contract['code_sha256'].items():
        if sha256(Path(__file__).parent/name)!=value: raise ValueError('extraction code changed during/after run')
    if sha256(directory/'results.jsonl')!=status['results_sha256']: raise ValueError('result hash differs')
    with (directory/'results.jsonl').open(encoding='utf-8') as f: results={r['task_key']:r for r in map(json.loads,f)}
    expected={t['task_key'] for t in tasks}
    if set(results)!=expected or any(r['status']=='failed' for r in results.values()): raise ValueError('incomplete task results')
    return results,contract

def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    manifest=json.loads((args.tasks/'manifest.json').read_text())
    for name,value in manifest['output_sha256'].items():
        if sha256(args.tasks/name)!=value: raise ValueError('task source changed')
    with (args.tasks/'tasks.jsonl').open(encoding='utf-8') as f: tasks=list(map(json.loads,f))
    with (args.tasks/'sources.jsonl').open(encoding='utf-8') as f: sources={r['note_id']:r for r in map(json.loads,f)}
    ct_tasks=[t for t in tasks if t['route']=='chest_ct']; ds_tasks=[t for t in tasks if t['route']=='discharge_all']
    ct,ctc=read_results(args.ct,ct_tasks); ds,dsc=read_results(args.discharge,ds_tasks)
    expected_digest=sha256(args.tasks/'manifest.json')
    if ctc['tasks_manifest_sha256']!=expected_digest or dsc['tasks_manifest_sha256']!=expected_digest: raise ValueError('wrong task contract')
    facts=defaultdict(list); evidence=[]
    for task in tasks:
        r=(ct if task['route']=='chest_ct' else ds)[task['task_key']]; source=sources[task['note_id']]
        if source['original_sha256']!=task['original_sha256']: raise ValueError('source identity mismatch')
        for fact in r['facts']:
            if source['text'][fact['original_start']:fact['original_end']]!=fact['quote']: raise ValueError('original evidence failed')
            enriched=dict(fact,hadm_id=task['hadm_id'],note_id=task['note_id'],charttime=task['charttime'],
                          channel=source['channel'],task_key=task['task_key'],original_sha256=task['original_sha256'])
            facts[task['hadm_id']].append(enriched); evidence.append(enriched)
            # 确认阳性亚型必然属于肺气肿；规则推导保存原文且标为derived。
            if fact['field'] in ('centrilobular_emphysema','paraseptal_emphysema') and fact['value']=='present':
                parent=dict(enriched,field='emphysema',derived='positive_emphysema_subtype_implies_emphysema')
                facts[task['hadm_id']].append(parent); evidence.append(parent)
    with np.load(args.tasks/'members.npz') as f: members={k:f[k] for k in f.files}
    ct_hadm={r['hadm_id'] for r in sources.values() if r['channel']=='ct'}
    ds_hadm={r['hadm_id'] for r in sources.values() if r['channel']=='discharge'}
    ct_values=[]; ds_values=[]
    for h in members['hadm_ids']:
        fs=facts[str(h)]; ct_values.append(ct_vector([f for f in fs if f['channel']=='ct']))
        ds_values.append(discharge_vector([f for f in fs if f['channel']=='discharge']))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'facts.npz',**members,ct_values=np.stack(ct_values),discharge_values=np.stack(ds_values),
                        ct_present=np.array([str(h) in ct_hadm for h in members['hadm_ids']]),
                        discharge_present=np.array([str(h) in ds_hadm for h in members['hadm_ids']]),
                        ct_names=np.array(CT_NAMES),discharge_names=np.array(DISCHARGE_NAMES))
    with (args.output/'evidence.jsonl').open('x',encoding='utf-8') as f:
        for e in evidence: f.write(json.dumps(e,ensure_ascii=False)+'\n')
    result={'status':'full_facts_source_checked_not_prediction_validated','admissions':len(ct_values),
            'ct_sources':len(ct_hadm),'discharge_sources':len(ds_hadm),'evidence_facts':len(evidence),
            'ct_columns':CT_NAMES,'discharge_columns':DISCHARGE_NAMES,'catalog_sha256':CATALOG_SHA,
            'parent_state_manifest_sha256':manifest['parent_state_manifest_sha256'],'tasks_manifest_sha256':expected_digest,
            'ct_status_sha256':sha256(args.ct/'status.json'),'discharge_status_sha256':sha256(args.discharge/'status.json'),
            'code_sha256':sha256(Path(__file__)),'output_sha256':{n:sha256(args.output/n) for n in ('facts.npz','evidence.jsonl')},
            'limits':['Programmatic evidence validation is not expert semantic accuracy','Final discharge note is retrospectively assumed available at discharge',
                      'Facts describe observations and plans, not causal treatment effects','No new embeddings; existing BERT unchanged']}
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in result.items() if 'columns' not in k}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path,default=Path('patient_data/processed/v6_bonsai_tasks_2026-10-10'))
    p.add_argument('--ct',type=Path,default=Path('patient_data/processed/v6_bonsai_ct_full_2026-10-10'))
    p.add_argument('--discharge',type=Path,default=Path('patient_data/processed/v6_bonsai_discharge_full_2026-10-10'))
    p.add_argument('--output',type=Path,required=True); build(p.parse_args())
