"""同保存训练候选条目核验；不重抽全文、不编码、不读结局或训练。"""
from collections import Counter,defaultdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import sha256
from patient_data.discharge_regimen_entries import read_unit, masked

ROOT=Path('training_runs/v6_discharge_regimen_entry_2026-10-14_r2')
PARENT=Path('training_runs/v6_discharge_regimen_audit_2026-10-14')
PROTOCOL=Path('docs/V6_DISCHARGE_REGIMEN_ENTRY_PROTOCOL_2026-10-14.md')
CLASSES=('inhaled_steroid','laba','lama')


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def inherited_hash(mapping,path):
    matches=[value for key,value in mapping.items() if Path(key).resolve()==path.resolve()]
    if len(matches)!=1:raise ValueError('missing or ambiguous inherited source key')
    return matches[0]


def derive():
    started=time.perf_counter();old=json.loads((PARENT/'audit.json').read_text())
    proof=json.loads((PARENT/'verification.json').read_text());sources={}
    def protect(p,expected=None):
        h=sha256(p)
        if expected is not None and h!=expected:raise ValueError('changed source: '+str(p))
        sources[str(p)]=h;return h
    if proof['maximum_error']!=0 or proof['audit_sha256']!=protect(PARENT/'audit.json'):
        raise ValueError('parent proof stale')
    protect(PARENT/'verification.json')
    for k,h in old['output_sha256'].items():protect(PARENT/k,h)
    for p in (Path('training/v5/clinical_concepts.py'),Path('patient_data/discharge_regimen_audit.py')):
        protect(p,inherited_hash(old['source_sha256'],p))
    teacher=Path('training_runs/v6_module_surgery_2026-10-12_r3/seed42/control/best.pt')
    protect(teacher,old['inherited_frozen_reference']['teacher_sha256'])
    with np.load(PARENT/'audit_arrays.npz') as z:
        metadata={k:z[k] for k in ('hadm_ids','subject_ids')}
    lookup={str(h):i for i,h in enumerate(metadata['hadm_ids'])};subjects=metadata['subject_ids']
    records=[];counts=Counter();by_status=defaultdict(set);per_class=defaultdict(lambda:defaultdict(set))
    product_keys=set();bucket=defaultdict(list);unit_count=0;multi_units=0
    with gzip.open(PARENT/'local_evidence.jsonl.gz','rt',encoding='utf-8') as f:
        for row in map(json.loads,f):
            unit_count+=1;h=row['hadm_id'];i=lookup[h]
            if row['subject_id']!=str(subjects[i]) or row['end_char']-row['start_char']!=len(row['quote']):
                raise ValueError('parent identity/offset mismatch')
            entries=read_unit(row['quote']);multi_units+=len(entries)>1
            for entry in entries:
                a,b=entry['entry_span']
                if not 0<=a<b<=len(row['quote']) or row['quote'][a:b]!=entry['entry_quote']:
                    raise ValueError('derived entry source mismatch')
                product=entry['product_span']
                if product is not None and entry['entry_quote'][slice(*product)]!=entry['product_name']:
                    raise ValueError('product source mismatch')
                for ra,rb,rv in entry['route_spans']:
                    if not 0<=ra<rb<=len(entry['entry_quote']) or entry['entry_quote'][ra:rb]!=rv:
                        raise ValueError('route source mismatch')
                key=digest(row['unit_key']+':'+str(a)+':'+str(b))
                result=dict(**{k:row[k] for k in ('hadm_id','subject_id','note_id','unit_key','original_sha256')},
                    entry_key=key,source_start=row['start_char']+a,source_end=row['start_char']+b,
                    quote=entry['entry_quote'],reading={k:v for k,v in entry.items() if k not in ('entry_span','entry_quote')},
                    parent_unit_source_start=row['start_char'],parent_unit_source_end=row['end_char'])
                records.append(result);counts['status:'+entry['status']]+=1;counts['reason:'+entry['reason']]+=1
                by_status[entry['status']].add(i)
                if entry['canonical_identity']:product_keys.add(entry['canonical_identity'])
                for group in entry['classes']:per_class[group][entry['status']].add(i)
                tags=[entry['status']]
                if entry['status']=='listed':tags+=entry['classes']
                if any(x[2].upper()=='IH' for x in entry['route_spans']):tags.append('explicit_IH')
                if len(entries)>1:tags.append('multi_entry_parent')
                if entry['product_name'] and entry['canonical_identity'] is None:tags.append('identity_unknown')
                for tag in tags:bucket[tag].append(result)
    if unit_count!=old['counts']['maintenance_mapping_candidate_units']:raise ValueError('parent candidates differ')
    samples=[];chosen=set()
    for tag,items in sorted(bucket.items()):
        for r in sorted(items,key=lambda r:r['entry_key'])[:3]:
            marker=(tag,r['entry_key'])
            if marker in chosen:continue
            chosen.add(marker);v=r['reading']
            samples.append(dict(bucket=tag,entry_key=r['entry_key'],masked_quote=masked(r['quote']),
                                status=v['status'],reason=v['reason'],classes=v['classes']))
    for p in (Path(__file__),PROTOCOL,Path('patient_data/discharge_regimen_entries.py'),
        Path('training/tests/test_discharge_regimen_entries.py'),Path('patient_data/medication_identity.py'),
        Path('patient_data/v6_expanded_candidates.py'),Path('patient_data/compact_multisource.py')):protect(p)
    cls={group:{s:dict(admissions=len(ii),patients=len(np.unique(subjects[list(ii)]))) for s,ii in statuses.items()}
         for group,statuses in per_class.items()}
    union=set().union(*(v.get('listed',set()) for v in per_class.values())) if per_class else set()
    report=dict(status='train_only_saved_regimen_entry_candidate_audit',source_sha256=sources,
        inherited_original_source_hashes=old['source_sha256'],inherited_frozen_reference=old['inherited_frozen_reference'],
        training_admissions=len(lookup),training_patients=len(np.unique(subjects)),parent_candidate_units=unit_count,
        derived_entries=len(records),multiple_entry_parent_units=multi_units,counts=dict(counts),
        status_admissions={k:len(v) for k,v in by_status.items()},by_class=cls,
        listed_maintenance_admissions=len(union),listed_maintenance_patients=len(np.unique(subjects[list(union)])),
        known_canonical_products=len(product_keys),masked_sample_records=len(samples),
        related_tests=17,new_fits=0,new_vectors=0,new_llm_calls=0,no_outcomes_read=True,
        raw_discharge_read=False,whole_inventory_rebuilt=False,actual_seconds=time.perf_counter()-started,
        quality='candidate_noise_unquantified_limited_masked_readout_pending',
        limits=['listed is document-list candidate, not confirmed actual prescribing/administration or adherence',
                'source offsets and synthetic boundary tests are not expert clinical accuracy',
                'PRN is retained as as-needed, not stopped',
                'dose/frequency strings are evidence, not standardized total-dose predictors',
                'full original source hashes inherited from verified parent, not re-read this run'])
    return report,records,samples


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def run():
    if ROOT.exists():raise FileExistsError('batch exists')
    ROOT.mkdir();r,e,s=derive()
    with gzip.open(ROOT/'entry_evidence.jsonl.gz','wt',encoding='utf-8') as f:
        for row in e:f.write(json.dumps(row)+'\n')
    write(ROOT/'masked_sample.json',s)
    r['output_sha256']={k:sha256(ROOT/k) for k in ('entry_evidence.jsonl.gz','masked_sample.json')}
    write(ROOT/'audit.json',r);write(ROOT/'state.json',dict(status='pending_independent_rebuild',new_fits=0))
    print(json.dumps({k:v for k,v in r.items() if k not in ('source_sha256','inherited_original_source_hashes','inherited_frozen_reference','output_sha256')}),flush=True)


def verify():
    if (ROOT/'verification.json').exists():raise FileExistsError('already verified')
    old=json.loads((ROOT/'audit.json').read_text());new,e,s=derive();seconds=new.pop('actual_seconds');old.pop('actual_seconds')
    for k,h in old.pop('output_sha256').items():
        if sha256(ROOT/k)!=h:raise ValueError('saved output changed')
    if old!=new:raise ValueError('source/statistics differ')
    with gzip.open(ROOT/'entry_evidence.jsonl.gz','rt',encoding='utf-8') as f:saved=list(map(json.loads,f))
    if saved!=e or json.loads((ROOT/'masked_sample.json').read_text())!=s:raise ValueError('entry/sample differs')
    write(ROOT/'verification.json',dict(status='one_independent_parent_evidence_entry_scope_statistic_rebuild',
        maximum_error=0.,actual_seconds=seconds,audit_sha256=sha256(ROOT/'audit.json'),new_fits=0,raw_discharge_read=False))
    write(ROOT/'state.json',dict(status='entry_candidates_independently_verified_readout_pending',new_fits=0,goal_achieved=False))
    print(json.dumps(dict(error=0.,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=('run','verify'));mode=p.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as error:
        if ROOT.exists():write(ROOT/f'failure_{mode}.json',dict(error=repr(error)))
        raise
