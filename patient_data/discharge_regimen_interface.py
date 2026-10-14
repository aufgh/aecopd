"""冻结r3出院方案读取，全队列类别集合与引用；不读标签/原raw或训练。"""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from patient_data.discharge_regimen_audit import STATE,TASKS,SCREEN,CLASSES,lexical,span,check_maps,covered
from patient_data.discharge_regimen_entries_checked import read_unit
from patient_data.export_mimiciv import sha256

ROOT=Path('patient_data/processed/v6_discharge_regimen_interface_2026-10-14')
PARENT=Path('training_runs/v6_discharge_regimen_entry_2026-10-14_r3')
INVENTORY=Path('training_runs/v6_discharge_regimen_audit_2026-10-14')
PROTOCOL=Path('docs/V6_DISCHARGE_REGIMEN_INTERFACE_PROTOCOL_2026-10-14.md')
ROLES=('listed','held','stopped','conditional_unknown','unresolved')


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def role(reading):
    status=reading['status']
    if status=='explicit_stop':
        action=reading['prefix_action']
        if action in ('held','hold'):return 'held'
        if action in ('stop','stopped','discontinue','discontinued'):return 'stopped'
        raise ValueError('unknown explicit action')
    if status not in ROLES:raise ValueError('unrecognized frozen status')
    return status


def member_key(record,lookup):
    value=record['reading'];name=value['canonical_identity']
    # 未知文本只作证据去重键，不进入训练产品身份词表。
    product=('canonical',name) if name else ('unknown',' '.join((value['product_name'] or record['quote']).lower().split()))
    return (lookup[record['hadm_id']],product,tuple(value['classes']),role(value),bool(value['as_needed']))


def collect(records,lookup):
    bag={}
    for record in records:
        key=member_key(record,lookup)
        if key not in bag:
            bag[key]=dict(position=key[0],canonical_identity=record['reading']['canonical_identity'],
                product_key=key[1],classes=list(key[2]),role=key[3],as_needed=key[4],evidence=[])
        if record not in bag[key]['evidence']:bag[key]['evidence'].append(record)
    output=[]
    for key in sorted(bag):
        value=bag[key];value['evidence'].sort(key=lambda r:r['entry_key']);output.append(value)
    return output


def encode(members,base,available,candidates):
    n=len(base['hadm_ids']);positions=np.asarray([r['position'] for r in members],np.int64)
    vocab=sorted({r['canonical_identity'] for r in members if r['canonical_identity'] is not None
                  and base['patient_splits'][r['position']]=='train'})
    names=['UNKNOWN',*vocab];out=dict(base)
    out.update(section_available=np.asarray(available,bool),candidate_present=np.asarray(candidates,bool),
        positions=positions,products=np.asarray([names.index(r['canonical_identity']) if r['canonical_identity'] in names else 0 for r in members],np.int64),
        roles=np.asarray([ROLES.index(r['role']) for r in members],np.int64),
        class_memberships=np.asarray([[c in r['classes'] for c in CLASSES] for r in members],bool).reshape(-1,3),
        as_needed=np.asarray([r['as_needed'] for r in members],bool))
    # 无引用数/检测次数，也不把产品编号当有序连续值。
    features=np.zeros((n,20+len(vocab)),np.float32)
    for i,r in enumerate(members):
        p=r['position'];k=ROLES.index(r['role']);features[p,15+k]=1
        for c in r['classes']:features[p,CLASSES.index(c)*5+k]=1
        product=out['products'][i]
        if product:features[p,19+product]=1
    if np.any(candidates&~available):raise ValueError('candidate without saved chapter')
    out['features']=features
    return out,dict(product_vocabulary=names,roles=list(ROLES),classes=list(CLASSES),
        feature_names=[f'{c}:{r}' for c in CLASSES for r in ROLES]+[f'role:{r}' for r in ROLES]+[f'product:{v}' for v in vocab],
        numeric_scaling='not_applicable_binary_features_no_sd_amplification',product_vocabulary_training_only=True)


def derive():
    started=time.perf_counter();sources={}
    def protect(path,expected=None):
        result=sha256(path)
        if expected is not None and result!=expected:raise ValueError('protected source changed: '+str(path))
        sources[str(path)]=result;return result
    parent=json.loads((PARENT/'audit.json').read_text());proof=json.loads((PARENT/'verification.json').read_text())
    if proof['maximum_error']!=0 or proof['audit_sha256']!=protect(PARENT/'audit.json'):raise ValueError('r3 proof stale')
    protect(PARENT/'verification.json')
    for path,h in parent['source_sha256'].items():protect(Path(path),h)
    for file,h in parent['output_sha256'].items():protect(PARENT/file,h)
    inherited=parent['inherited_original_source_hashes']
    for directory,names in ((STATE,('manifest.json','clinical_state.npz')),
        (TASKS,('manifest.json','sources.jsonl','tasks.jsonl','members.npz')),
        (SCREEN,('manifest.json','selected_sentences.jsonl.gz'))):
        for name in names:
            path=directory/name
            keys=[h for k,h in inherited.items() if Path(k).resolve()==path.resolve()]
            if len(keys)!=1:raise ValueError('missing/ambiguous inherited file')
            protect(path,keys[0])
    with np.load(STATE/'clinical_state.npz') as z:base={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits')}
    with np.load(TASKS/'members.npz') as z:
        for k,value in base.items():np.testing.assert_array_equal(value,z[k])
    if len(base['hadm_ids'])!=19401 or len(set(base['subject_ids']))!=5351 or (base['patient_splits']=='train').sum()!=13592:
        raise ValueError('cohort drift')
    lookup={str(h):i for i,h in enumerate(base['hadm_ids'])}
    if len(lookup)!=len(base['hadm_ids']):raise ValueError('duplicate admissions')
    summaries={};sections={};available=np.zeros(len(lookup),bool);candidates=np.zeros_like(available)
    with (TASKS/'sources.jsonl').open(encoding='utf-8') as stream:
        for r in map(json.loads,stream):
            if r['channel']!='discharge':continue
            if r['hadm_id'] not in lookup:raise ValueError('source outside fixed cohort')
            p=lookup[r['hadm_id']]
            if r['subject_id']!=str(base['subject_ids'][p]) or r['note_id'] in summaries:raise ValueError('source identity/duplicate')
            if digest(r['text'])!=r['original_sha256']:raise ValueError('saved note hash differs')
            ss=[s for s in r['sections'] if s['role']=='discharge_medications']
            for s in ss:span(r['text'],s['start_char'],s['end_char'],s['text'])
            summaries[r['note_id']]=r;sections[r['note_id']]=ss;available[p]|=bool(ss)
    ranges=defaultdict(list)
    with (TASKS/'tasks.jsonl').open(encoding='utf-8') as stream:
        for task in map(json.loads,stream):
            if task['route']!='discharge_all' or task['note_id'] not in summaries:continue
            source=summaries[task['note_id']]
            if any(task[k]!=source[k] for k in ('hadm_id','subject_id','original_sha256')):raise ValueError('task source identity')
            ranges[task['note_id']].extend(check_maps(source,task))
    for note,ss in sections.items():
        for s in ss:covered(s,ranges[note])
    records=[];train_units=[]
    with gzip.open(SCREEN/'selected_sentences.jsonl.gz','rt',encoding='utf-8') as stream:
        for r in map(json.loads,stream):
            if r['kind']!='discharge' or r['section']!='discharge_medications':continue
            if r['hadm_id'] not in lookup or r['note_id'] not in summaries:raise ValueError('candidate outside source')
            p=lookup[r['hadm_id']];source=summaries[r['note_id']]
            if r['subject_id']!=str(base['subject_ids'][p]) or r['patient_split']!=base['patient_splits'][p]:raise ValueError('candidate patient/fold mismatch')
            a,b=r['start_char'],r['end_char'];span(source['text'],a,b,r['quote'])
            if r['original_sha256']!=source['original_sha256'] or not any(s['start_char']<=a<b<=s['end_char'] for s in sections[r['note_id']]):raise ValueError('candidate chapter/hash mismatch')
            if not lexical(r['quote'])['classes']:continue
            candidates[p]=True
            if base['patient_splits'][p]=='train':train_units.append(r['unit_key'])
            for entry in read_unit(r['quote']):
                x,y=entry['entry_span'];quote=entry['entry_quote'];span(r['quote'],x,y,quote)
                if entry['product_span'] is not None:span(quote,*entry['product_span'],entry['product_name'])
                for ra,rb,rv in entry['route_spans']:span(quote,ra,rb,rv)
                records.append(dict(**{k:r[k] for k in ('hadm_id','subject_id','note_id','unit_key','original_sha256')},
                    entry_key=digest(r['unit_key']+':'+str(x)+':'+str(y)),source_start=a+x,source_end=a+y,quote=quote,
                    reading={k:v for k,v in entry.items() if k not in ('entry_span','entry_quote')},
                    parent_unit_source_start=a,parent_unit_source_end=b))
    if len({r['entry_key'] for r in records})!=len(records):raise ValueError('duplicate source entry key')
    with gzip.open(PARENT/'entry_evidence.jsonl.gz','rt',encoding='utf-8') as stream:old=list(map(json.loads,stream))
    train_records=[r for r in records if base['patient_splits'][lookup[r['hadm_id']]]=='train']
    if sorted(train_records,key=lambda r:r['entry_key'])!=sorted(old,key=lambda r:r['entry_key']) or len(train_units)!=9369:
        raise ValueError('training frozen entries differ')
    members=collect(records,lookup);arrays,metadata=encode(members,base,available,candidates)
    for path in (Path(__file__),PROTOCOL,Path('training/tests/test_discharge_regimen_interface.py')):protect(path)
    ti=base['patient_splits']=='train';training=[r for r in members if ti[r['position']]];counts=Counter(r['role'] for r in training)
    report=dict(status='fixed_rule_full_cohort_regimen_interface_not_model_input',source_sha256=sources,
        inherited_frozen_reference=parent['inherited_frozen_reference'],metadata=metadata,
        total_admissions=len(lookup),training_admissions=int(ti.sum()),training_member_count=len(training),
        training_source_available=int((available&ti).sum()),training_candidate_available=int((candidates&ti).sum()),
        training_role_members=dict(counts),training_entries_match_frozen_r3=True,training_entry_count=len(train_records),
        training_members_with_multiple_references=sum(len(r['evidence'])>1 for r in training),
        training_role_admissions={k:len({r['position'] for r in training if r['role']==k}) for k in ROLES},
        original_candidate_rule_unchanged=True,raw_discharge_read=False,outcomes_read=False,
        new_fits=0,new_vectors=0,actual_seconds=time.perf_counter()-started)
    return report,arrays,members


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def run():
    if ROOT.exists():raise FileExistsError('interface exists')
    ROOT.mkdir();report,arrays,members=derive();np.savez_compressed(ROOT/'interface.npz',**arrays)
    with gzip.open(ROOT/'members.jsonl.gz','wt',encoding='utf-8') as stream:
        for r in members:stream.write(json.dumps(r)+'\n')
    report['output_sha256']={f:sha256(ROOT/f) for f in ('interface.npz','members.jsonl.gz')}
    write(ROOT/'manifest.json',report);write(ROOT/'state.json',dict(status='pending_independent_rebuild',new_fits=0))
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','inherited_frozen_reference','output_sha256','metadata')}),flush=True)


def verify():
    if (ROOT/'verification.json').exists():raise FileExistsError('already verified')
    old=json.loads((ROOT/'manifest.json').read_text());new,arrays,members=derive();seconds=new.pop('actual_seconds');old.pop('actual_seconds')
    for file,h in old.pop('output_sha256').items():
        if sha256(ROOT/file)!=h:raise ValueError('output changed')
    if old!=new:raise ValueError('source/metadata/statistics differ')
    with np.load(ROOT/'interface.npz') as z:
        for k,value in arrays.items():np.testing.assert_array_equal(z[k],value)
    with gzip.open(ROOT/'members.jsonl.gz','rt',encoding='utf-8') as stream:saved=list(map(json.loads,stream))
    # JSON将tuple转list，统一到落盘表示后逐引用比较。
    if saved!=json.loads(json.dumps(members)):raise ValueError('members/references differ')
    write(ROOT/'verification.json',dict(status='one_independent_saved_source_full_interface_rebuild',maximum_error=0.,
        manifest_sha256=sha256(ROOT/'manifest.json'),actual_seconds=seconds,new_fits=0))
    write(ROOT/'state.json',dict(status='full_cohort_interface_verified_not_network_consumed',new_fits=0,goal_achieved=False))
    print(json.dumps(dict(error=0.,seconds=seconds)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=('run','verify'));mode=parser.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as error:
        if ROOT.exists():write(ROOT/f'failure_{mode}.json',dict(error=repr(error)))
        raise
