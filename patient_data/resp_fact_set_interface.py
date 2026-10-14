"""冻结r2语义，全原队列typed候选事实集合，原模型与源不变。"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time
import numpy as np
from patient_data.discharge_resp_rules_r2 import extract
from patient_data.discharge_resp_fact_audit_r2 import LOCAL,ROOT as PARENT,digest,token_bounds
from patient_data.export_mimiciv import rows,sha256
from patient_data.text_time_coverage import SNAP,SCREEN,STATE
from training.v6.encode_screen_candidates import ROUTES as TEXT_ROUTES
from training.v5.numeric_preprocessing import robust_parameters
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_resp_fact_set_interface_2026-10-13')
PROTOCOL=Path('docs/V6_RESP_FACT_SET_INTERFACE_PROTOCOL_2026-10-13.md')
CONCEPTS=('dyspnea','dyspnea_improved','oxygen_required','room_air_status','oxygen_weaned_room_air','oxygen_weaned_off','oxygen_failed_wean','oxygen_flow_litres_notation')
ROLES=('explicit_discharge','discharge_section','inpatient_change')
POLARITIES=('affirmed','negated')
UNITS=('not_numeric','L_oxygen_notation','L/min')
CONDITIONS=('not_applicable','unknown','rest','exertion')

def member_key(record,lookup):
    if record['role']=='unknown':return None,'unknown_time_role'
    if record['role'] not in ROLES:raise ValueError('unrecognized fixed role')
    numeric=record['value'] is not None
    if numeric and record['concept']!='oxygen_flow_litres_notation':raise ValueError('unexpected numeric concept')
    if numeric and record['activity_condition']=='ambiguous':return None,'ambiguous_flow_condition'
    unit=record['unit'] if numeric else 'not_numeric'
    condition=record['activity_condition'] if numeric else 'not_applicable'
    value=float(record['value']) if numeric else None
    if numeric and not np.isfinite(value):raise ValueError('nonfinite fact value')
    key=(lookup[int(record['hadm_id'])],CONCEPTS.index(record['concept']),ROLES.index(record['role']),
         POLARITIES.index(record['polarity']),UNITS.index(unit),CONDITIONS.index(condition),value)
    return key,None

def collect(records,lookup):
    bag={};excluded=[]
    for record in records:
        key,reason=member_key(record,lookup)
        if reason:
            excluded.append(dict(reason=reason,evidence=record));continue
        if key not in bag:
            bag[key]=dict(position=key[0],concept=CONCEPTS[key[1]],role=ROLES[key[2]],polarity=POLARITIES[key[3]],
                unit=UNITS[key[4]],condition=CONDITIONS[key[5]],value=key[6],evidence=[])
        if record not in bag[key]['evidence']:bag[key]['evidence'].append(record)
    facts=[bag[k] for k in sorted(bag)]
    for fact in facts:
        fact['evidence'].sort(key=lambda r:(r['note_id'],r['evidence_start_char'],r['evidence_end_char'],r['unit_key']))
    excluded.sort(key=lambda r:(r['evidence']['note_id'],r['evidence']['start'],r['reason']))
    return facts,excluded

def arrays_from(facts,base):
    n=len(facts);out=dict(base)
    out['positions']=np.asarray([r['position'] for r in facts],np.int64)
    for key,names in [('concepts',CONCEPTS),('roles',ROLES),('polarities',POLARITIES),('units',UNITS),('conditions',CONDITIONS)]:
        field={'concepts':'concept','roles':'role','polarities':'polarity','units':'unit','conditions':'condition'}[key]
        out[key]=np.asarray([names.index(r[field]) for r in facts],np.int64)
    out['numeric_present']=np.asarray([r['value'] is not None for r in facts],bool)
    out['raw_values']=np.asarray([r['value'] if r['value'] is not None else 0 for r in facts],np.float64)
    out['values']=np.zeros(n,np.float32);out['numeric_valid']=np.zeros(n,bool);scales={}
    for unit in ('L_oxygen_notation','L/min'):
        use=out['numeric_present']&(out['units']==UNITS.index(unit))
        training=use&(base['patient_splits'][out['positions']]=='train')
        if training.any():
            scale=robust_parameters(out['raw_values'][training]);scales[unit]=scale
            out['values'][use]=np.clip((out['raw_values'][use]-scale['median'])/scale['scale'],-8,8)
            out['numeric_valid'][use]=True
        else:scales[unit]={'method':'no_training_observation','training_observations':0}
    aggregate=np.zeros((len(base['hadm_ids']),len(ROLES)),np.float32)
    aggregate[out['positions'],out['roles']]=1
    out['aggregate']=aggregate;out['observed']=aggregate.any(axis=1)
    return out,scales

def source_records():
    from transformers import AutoTokenizer
    parent=json.loads((PARENT/'audit.json').read_text());source=dict(parent['source_sha256'])
    for p,h in source.items():
        if sha256(Path(p))!=h:raise ValueError('protected r2 source changed: '+p)
    for tag in ('a','b'):
        proof=json.loads((PARENT/f'verification_{tag}.json').read_text())
        if proof['maximum_error']!=0 or proof['audit_sha256']!=sha256(PARENT/'audit.json'):raise ValueError('formal r2 proof differs')
        for name,h in proof['output_sha256'].items():
            if sha256(PARENT/name)!=h:raise ValueError('formal r2 records changed')
    manifest=json.loads((LOCAL/'manifest.json').read_text())
    for name,h in manifest['output_sha256'].items():
        if sha256(LOCAL/name)!=h:raise ValueError('safe token interface changed')
    with np.load(STATE/'clinical_state.npz') as z:base={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits')}
    if len(base['hadm_ids'])!=19401 or (base['patient_splits']=='train').sum()!=13592:raise ValueError('fixed cohort changed')
    lookup={int(h):i for i,h in enumerate(base['hadm_ids'])}
    with gzip.open(LOCAL/'index.jsonl.gz','rt',encoding='utf-8') as f:index=list(map(json.loads,f))
    if len(index)!=9339:raise ValueError('locked full safe index changed')
    with gzip.open(LOCAL/'local_tasks.jsonl.gz','rt',encoding='utf-8') as f:tasks=list(map(json.loads,f))
    wanted={(r['note_id'],r['start_char'],r['end_char']) for r in index};units={}
    with gzip.open(SCREEN/'selected_sentences.jsonl.gz','rt',encoding='utf-8') as f:
        for r in map(json.loads,f):
            key=(r['note_id'],r['start_char'],r['end_char'])
            if key in wanted:
                if key in units:raise ValueError('duplicate safe source unit')
                units[key]=r
    if set(units)!=wanted:raise ValueError('full source units missing')
    needed={r['note_id'] for r in index};raw={r['note_id']:r for r in rows(SNAP/'raw/discharge.csv.gz') if r['note_id'] in needed}
    raw_hash={note:digest(r['text']) for note,r in raw.items()}
    tokenizer=AutoTokenizer.from_pretrained('training/model_cache/Bio_ClinicalBERT',local_files_only=True,use_fast=True)
    if not tokenizer.is_fast:raise ValueError('original offset tokenizer needed')
    records=[];train_rejected=Counter()
    for old in index:
        key=(old['note_id'],old['start_char'],old['end_char']);unit=units[key];pos=lookup[int(old['hadm_id'])]
        if int(old['subject_id'])!=int(base['subject_ids'][pos]) or old['patient_split']!=base['patient_splits'][pos]:raise ValueError('full member/fold differs')
        if any(unit[k]!=old[k] for k in ('unit_key','section','hadm_id','subject_id','patient_split','original_sha256')):raise ValueError('unit lineage differs')
        text=unit['quote'];start,end=key[1:];original=raw[old['note_id']]['text']
        if raw[old['note_id']]['hadm_id']!=old['hadm_id'] or raw[old['note_id']]['subject_id']!=old['subject_id']:
            raise ValueError('original note admission/subject differs')
        if not 0<=start<end<=len(original) or original[start:end]!=text or digest(text)!=old['quote_sha256'] or raw_hash[old['note_id']]!=old['original_sha256']:
            raise ValueError('original full quote offset/hash differs')
        encoded=tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True)
        prefix=tokenizer('Clinical topic: '+TEXT_ROUTES[old['route']]+'\n',add_special_tokens=False)['input_ids'];right=0;ids=[]
        for task in tasks[old['task_start']:old['task_end']]:
            a,b=task['token_start'],task['token_end']
            if task['sentence_row']!=old['sentence_row'] or a!=right or task['input_ids']!=[tokenizer.cls_token_id,*prefix,*encoded['input_ids'][a:b],tokenizer.sep_token_id]:
                raise ValueError('actual saved full local task differs')
            ids.extend(encoded['input_ids'][a:b]);right=b
        if ids!=encoded['input_ids'] or right!=old['local_token_count']:raise ValueError('full token tail differs')
        facts,rejected=extract(text,old['section'])
        if old['patient_split']=='train':train_rejected.update(r['reason'] for r in rejected)
        envelope={k:old[k] for k in ('hadm_id','subject_id','note_id','section','unit_key','sentence_row','quote_sha256','original_sha256','containing_rows_and_tokens')}
        envelope.update(patient_split=old['patient_split'],source_start_char=start,source_end_char=end,quote=text)
        for fact in facts:
            a,b=fact['start'],fact['end']
            records.append(dict(**envelope,**fact,evidence_start_char=start+a,evidence_end_char=start+b,
                evidence_sha256=digest(text[a:b]),local_token_span=token_bounds(encoded['offset_mapping'],a,b),measurement_time='unknown',clinically_verified=False))
    with gzip.open(PARENT/'facts.jsonl.gz','rt',encoding='utf-8') as f:prior=list(map(json.loads,f))
    if [r for r in records if r['patient_split']=='train']!=prior:raise ValueError('training semantic records changed during full interface')
    for p in (PARENT/'audit.json',PARENT/'verification_a.json',PARENT/'verification_b.json',PARENT/'facts.jsonl.gz',
              PROTOCOL,Path(__file__),Path('training/tests/test_v6_resp_fact_set_interface.py'),Path('training/v5/numeric_preprocessing.py')):
        source[str(p)]=sha256(p)
    return base,lookup,records,source,train_rejected

def derive():
    base,lookup,records,source,rejected=source_records();facts,excluded=collect(records,lookup);arrays,scales=arrays_from(facts,base)
    train=base['patient_splits']=='train';selected=train[arrays['positions']]
    report=dict(status='fixed_r2_full_cohort_candidate_set_not_clinical_gold',source_sha256=source,scales=scales,cohort=19401,
        vocabularies=dict(concepts=CONCEPTS,roles=ROLES,polarities=POLARITIES,units=UNITS,conditions=CONDITIONS),
        training=dict(members=int(selected.sum()),admissions=int((train&arrays['observed']).sum()),
            patients=len(np.unique(base['subject_ids'][train&arrays['observed']])),
            role_admissions={name:int((train&arrays['aggregate'][:,j].astype(bool)).sum()) for j,name in enumerate(ROLES)},
            per_concept={name:int((selected&(arrays['concepts']==j)).sum()) for j,name in enumerate(CONCEPTS)},
            semantic_rejections=dict(rejected),interface_exclusions=dict(Counter(r['reason'] for r in excluded if r['evidence']['patient_split']=='train'))),
        training_parent_record_replay_error=0,nontraining_outcomes_used=False,high_frequency_used=False,new_fits=0,new_vectors=0,
        limits=['Candidate surface facts not medical gold','Unknown role and ambiguous numeric condition isolated','Inpatient change not discharge terminal state',
            'FINAL retrospective policy, not prospective availability','Reference repetition not measurement frequency','No whole-report encoding or model change'])
    return arrays,facts,excluded,report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('build','verify'),required=True);parser.add_argument('--tag',choices=('a','b'),default='a')
    args=parser.parse_args();started=time.perf_counter();arrays,facts,excluded,report=derive()
    payload={'facts.jsonl.gz':facts,'excluded.jsonl.gz':excluded}
    if args.mode=='build':
        if ROOT.exists():raise FileExistsError('interface exists, inspect rather than overwrite')
        ROOT.mkdir();np.savez_compressed(ROOT/'interface.npz',**arrays)
        for name,records in payload.items():
            with gzip.open(ROOT/name,'xt',encoding='utf-8') as f:
                for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        report['output_sha256']={name:sha256(ROOT/name) for name in ('interface.npz',*payload)}
        write_json(ROOT/'manifest.json',report);write_json(ROOT/'state.json',dict(status='built_pending_two_full_source_replays',new_fits=0,new_vectors=0))
    else:
        saved=json.loads((ROOT/'manifest.json').read_text());report['output_sha256']=saved['output_sha256']
        # tuple词表只在JSON边界规范化，不改变源或数组。
        if saved!=json.loads(json.dumps(report)):raise ValueError('full interface manifest differs')
        for name,h in saved['output_sha256'].items():
            if sha256(ROOT/name)!=h:raise ValueError('full interface output changed')
        with np.load(ROOT/'interface.npz') as z:
            if set(z.files)!=set(arrays):raise ValueError('full interface keys differ')
            for key,value in arrays.items():np.testing.assert_array_equal(value,z[key])
        for name,records in payload.items():
            with gzip.open(ROOT/name,'rt',encoding='utf-8') as f:
                if list(map(json.loads,f))!=records:raise ValueError('set/evidence/isolation differs')
        write_json(ROOT/f'verification_{args.tag}.json',dict(maximum_error=0,manifest_sha256=sha256(ROOT/'manifest.json'),
            complete_original_source_and_arrays_rebuilt=True,training_scales_rebuilt=True,training_parent_record_replay_error=0,new_fits=0))
        if all((ROOT/f'verification_{tag}.json').exists() for tag in ('a','b')):
            write_json(ROOT/'state.json',dict(status='two_independent_full_source_replays_verified',new_fits=0,new_vectors=0,goal_achieved=False))
    print(json.dumps(dict(status=report['status'],training=report['training'],scales=report['scales'],seconds=time.perf_counter()-started)),flush=True)

if __name__=='__main__':main()
