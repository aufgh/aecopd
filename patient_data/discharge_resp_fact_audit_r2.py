"""独立r2限定语义修复读出；同原句同来源，原r1不改，不训练。"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from patient_data.discharge_resp_rules_r2 import extract
from patient_data.export_mimiciv import rows, sha256
from patient_data.text_time_coverage import SNAP, SCREEN, STATE, checked_sources
from training.v6.encode_screen_candidates import ROUTES
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_discharge_resp_fact_audit_2026-10-13_r2_final')
LOCAL=Path('patient_data/processed/v6_local_text_interface_2026-10-13_r2')
PROTOCOL=Path('docs/V6_DISCHARGE_RESP_R2_PROTOCOL_2026-10-13.md')

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def token_bounds(offsets, start, end):
    selected=[i for i,(a,b) in enumerate(offsets) if a<end and b>start]
    if not selected or selected!=list(range(selected[0],selected[-1]+1)):
        raise ValueError('fact token span invalid')
    return [selected[0],selected[-1]+1]

def derive():
    from transformers import AutoTokenizer
    source,_=checked_sources()
    parent=Path('patient_data/processed/v6_discharge_resp_fact_audit_2026-10-13')
    parent_report=json.loads((parent/'audit.json').read_text())
    for tag in ('a','b'):
        p=json.loads((parent/f'verification_{tag}.json').read_text())
        if p['audit_sha256']!=sha256(parent/'audit.json') or p['maximum_error']!=0:raise ValueError('r1 source proof changed')
    for path,h in parent_report['source_sha256'].items():
        if sha256(Path(path))!=h:raise ValueError('r1 protected source changed: '+path)
    for p in (parent/'audit.json',parent/'verification_a.json',parent/'verification_b.json',parent/'quality_readout.json',parent/'quality_flagged_local.json'):
        source[str(p)]=sha256(p)
    m=json.loads((LOCAL/'manifest.json').read_text());proof=json.loads((LOCAL/'source_verification.json').read_text())
    if proof['manifest_sha256']!=sha256(LOCAL/'manifest.json') or proof['source_and_arrays_max_error']!=0:
        raise ValueError('old token source proof differs')
    for p,h in m['source_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('protected old source changed: '+p)
    for name,h in m['output_sha256'].items():
        if sha256(LOCAL/name)!=h:raise ValueError('old token interface output changed')
    with np.load(STATE/'clinical_state.npz') as z:
        members={str(h):str(s) for h,s,f in zip(z['hadm_ids'],z['subject_ids'],z['patient_splits']) if f=='train'}
    if len(members)!=13592:raise ValueError('training membership changed')
    with gzip.open(LOCAL/'index.jsonl.gz','rt',encoding='utf-8') as f:
        index=[r for r in map(json.loads,f) if r['patient_split']=='train']
    if len(index)!=6613:raise ValueError('locked safe candidate set changed')
    with gzip.open(LOCAL/'local_tasks.jsonl.gz','rt',encoding='utf-8') as f:tasks=list(map(json.loads,f))
    wanted={(r['note_id'],r['start_char'],r['end_char']) for r in index}
    units={}
    with gzip.open(SCREEN/'selected_sentences.jsonl.gz','rt',encoding='utf-8') as f:
        for r in map(json.loads,f):
            key=(r['note_id'],r['start_char'],r['end_char'])
            if key in wanted:
                if key in units:raise ValueError('duplicate source unit')
                units[key]=r
    if set(units)!=wanted:raise ValueError('saved source unit missing')
    wanted_notes={r['note_id'] for r in index}
    notes={r['note_id']:r for r in rows(SNAP/'raw/discharge.csv.gz') if r['hadm_id'] in members and r['note_id'] in wanted_notes}
    tokenizer=AutoTokenizer.from_pretrained('training/model_cache/Bio_ClinicalBERT',local_files_only=True,use_fast=True)
    if not tokenizer.is_fast:raise ValueError('offset tokenizer required')
    counters=Counter();accepted=[];rejections=[];sets=defaultdict(set);examples=[];role_admissions=defaultdict(set)
    for old in index:
        key=(old['note_id'],old['start_char'],old['end_char']);unit=units[key];original=notes[old['note_id']]
        h=old['hadm_id'];s=old['subject_id'];text=unit['quote'];start,end=key[1:]
        if h not in members or members[h]!=s or unit['hadm_id']!=h or unit['subject_id']!=s or unit['patient_split']!='train':
            raise ValueError('fact fold/subject differs')
        if unit['unit_key']!=old['unit_key'] or unit['section']!=old['section']:
            raise ValueError('fact unit identity/section differs')
        if not 0<=start<end<=len(original['text']) or original['text'][start:end]!=text or digest(text)!=old['quote_sha256']:
            raise ValueError('raw quote offset/hash differs')
        if digest(original['text'])!=old['original_sha256'] or unit['original_sha256']!=old['original_sha256']:
            raise ValueError('original document differs')
        encoded=tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True)
        prefix=tokenizer('Clinical topic: '+ROUTES[old['route']]+'\n',add_special_tokens=False)['input_ids']
        local=tasks[old['task_start']:old['task_end']];right=0;rebuilt_ids=[]
        for task in local:
            a,b=task['token_start'],task['token_end']
            if task['sentence_row']!=old['sentence_row'] or a!=right or task['input_ids']!=[tokenizer.cls_token_id,*prefix,*encoded['input_ids'][a:b],tokenizer.sep_token_id]:
                raise ValueError('saved actual local token IDs differ')
            rebuilt_ids.extend(encoded['input_ids'][a:b]);right=b
        if rebuilt_ids!=encoded['input_ids'] or right!=old['local_token_count']:raise ValueError('local token tail missing')
        facts,rejected=extract(text,unit['section']);counters['checked_safe_units']+=1
        for r in rejected:
            counters['rejected:'+r['reason']]+=1
        base={k:old[k] for k in ('hadm_id','subject_id','note_id','section','unit_key','sentence_row','quote_sha256','original_sha256','containing_rows_and_tokens')}
        base.update(patient_split='train',source_start_char=start,source_end_char=end,quote=text)
        if rejected:rejections.append(dict(**base,rejections=rejected))
        for fact in facts:
            a,b=fact['start'],fact['end'];token=token_bounds(encoded['offset_mapping'],a,b)
            record=dict(**base,**fact,evidence_start_char=start+a,evidence_end_char=start+b,
                evidence_sha256=digest(text[a:b]),local_token_span=token,
                measurement_time='unknown',clinically_verified=False)
            accepted.append(record);counters['concept:'+fact['concept']]+=1;counters['role:'+fact['role']]+=1
            counters['polarity:'+fact['polarity']]+=1;role_admissions[fact['role']].add(h)
            sets[(h,fact['concept'])].add((fact['polarity'],fact['role'],fact['value'],fact['unit'],fact['activity_condition']))
    # 固定hash分层回查材料只存本地，不输出原句/标识到日志。
    groups=defaultdict(list)
    for r in accepted:groups[(r['concept'],r['role'],r['polarity'])].append(r)
    for key,items in sorted(groups.items()):
        examples.extend(sorted(items,key=lambda r:digest(r['unit_key']+str(r['start'])))[:3])
    explicit=[r for r in accepted if r['role'] in ('explicit_discharge','discharge_section')]
    repeated=defaultdict(set)
    for r in explicit:repeated[r['quote_sha256']].add(r['subject_id'])
    counters['explicit_discharge_facts']=len(explicit)
    counters['explicit_shared_quote_facts']=sum(len(repeated[r['quote_sha256']])>1 for r in explicit)
    for p in (LOCAL/'manifest.json',LOCAL/'source_verification.json',LOCAL/'index.jsonl.gz',LOCAL/'local_tasks.jsonl.gz',
              Path(__file__),Path('patient_data/discharge_resp_rules.py'),Path('patient_data/discharge_resp_rules_r2.py'),PROTOCOL,Path('training/tests/test_v6_discharge_resp_rules.py'),Path('training/tests/test_v6_discharge_resp_rules_r2.py')):
        source[str(p)]=sha256(p)
    report=dict(status='training_only_lexical_fact_audit_not_gold',source_sha256=source,counters=dict(counters),
        training_admissions=13592,safe_candidate_units=len(index),safe_candidate_admissions=len({r['hadm_id'] for r in index}),
        fact_admissions=len({r['hadm_id'] for r in accepted}),fact_patients=len({r['subject_id'] for r in accepted}),
        explicit_discharge_admissions=len({r['hadm_id'] for r in explicit}),explicit_discharge_patients=len({r['subject_id'] for r in explicit}),
        role_admissions={k:len(v) for k,v in role_admissions.items()},
        conflicting_same_admission_concepts=sum(len(v)>1 for v in sets.values()),review_sample_records=len(examples),
        new_fits=0,new_vectors=0,limits=['No clinical gold or precision/recall estimate','Unknown dates remain unknown; FINAL retrospective',
        'Source candidate membership finite, missed fact not normal','Sets not recovery sequence; discordant evidence not majority vote'])
    return report,accepted,rejections,examples

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('build','verify'),required=True)
    parser.add_argument('--tag',choices=('a','b'),default='a');args=parser.parse_args();started=time.perf_counter()
    report,facts,rejected,samples=derive();payload={'facts.jsonl.gz':facts,'rejected.jsonl.gz':rejected,'review_sample.jsonl.gz':samples}
    if args.mode=='build':
        if ROOT.exists():raise FileExistsError('audit exists, inspect rather than overwrite')
        ROOT.mkdir(parents=True)
        for name,records in payload.items():
            with gzip.open(ROOT/name,'xt',encoding='utf-8') as f:
                for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        write_json(ROOT/'audit.json',report)
        write_json(ROOT/'state.json',dict(status='built_pending_two_source_replays',new_fits=0,new_vectors=0))
    else:
        if json.loads((ROOT/'audit.json').read_text())!=report:raise ValueError('full source report reconstruction differs')
        for name,records in payload.items():
            with gzip.open(ROOT/name,'rt',encoding='utf-8') as f:
                if list(map(json.loads,f))!=records:raise ValueError('full evidence/token records differ')
        write_json(ROOT/f'verification_{args.tag}.json',dict(status='independent_full_source_replay_verified',maximum_error=0,
            audit_sha256=sha256(ROOT/'audit.json'),output_sha256={n:sha256(ROOT/n) for n in payload},new_fits=0,new_vectors=0))
        if all((ROOT/f'verification_{tag}.json').exists() for tag in ('a','b')):
            write_json(ROOT/'state.json',dict(status='two_independent_full_source_replays_verified',new_fits=0,new_vectors=0,goal_achieved=False))
    print(json.dumps(dict(counters=report['counters'],fact_admissions=report['fact_admissions'],
        explicit_discharge_admissions=report['explicit_discharge_admissions'],explicit_discharge_patients=report['explicit_discharge_patients'],
        role_admissions=report['role_admissions'],seconds=time.perf_counter()-started)),flush=True)

if __name__=='__main__':main()
