"""锁定六概念PFT全原队列集合，保守未来计划隔离和训练尺度。"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import numpy as np

from patient_data.export_mimiciv import rows,sha256
from patient_data.pft_fact_ledger import CONCEPTS,canonical,ROOT as LEDGER
from patient_data.pft_evidence import bindings,HISTORY
from patient_data.pft_inventory import sha256_text
from patient_data.pft_table_binding import line_ranges,row_values,find_header
from patient_data.text_time_coverage import SCREEN,SNAP,STATE,checked_sources
from training.v6.encode_screen_candidates import selected_groups
from training.v5.numeric_preprocessing import robust_parameters
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_pft_set_interface_2026-10-13')
PROTOCOL=Path('docs/V6_PFT_SET_INTERFACE_PROTOCOL_2026-10-13.md')
POST_DISCHARGE=re.compile(r'\b(?:(?:after|following)\s+(?:the\s+)?discharge|post[- ]?discharge|after\s+leaving\s+(?:the\s+)?hospital)\b',re.I)
PFT_NAME=r'(?:PFTs?|spirometr\w*|pulmonary\s+function|FEV\s*1|FVC)'
FUTURE_PFT=re.compile(r'\b(?:will|plan\w*|schedul\w*|recommend\w*|should|to undergo)\b.{0,70}\b'+PFT_NAME+r'\b|\b'+PFT_NAME+r'\b.{0,40}\b(?:planned|scheduled)\b',re.I|re.S)


def temporal_reason(text):
    return 'explicit_post_discharge_words' if POST_DISCHARGE.search(text) else 'future_PFT_plan_words' if FUTURE_PFT.search(text) else None


def normalized(raw,concepts,positions,folds):
    values=np.zeros(len(raw),np.float32);scales={}
    for j,name in enumerate(CONCEPTS):
        mask=(concepts==j)&(folds[positions]=='train')
        if not mask.any():raise ValueError('concept has no training observation: '+name)
        scale=robust_parameters(raw[mask]);scales[name]=scale
        use=concepts==j;values[use]=np.clip((raw[use]-scale['median'])/scale['scale'],-8,8)
    if not np.isfinite(values).all():raise ValueError('PFT standardized values nonfinite')
    return values,scales


def derive():
    sources,_=checked_sources()
    lm=json.loads((LEDGER/'manifest.json').read_text())
    for tag in ('a','b'):
        p=json.loads((LEDGER/f'verification_{tag}.json').read_text())
        if p['maximum_error'] or p['manifest_sha256']!=sha256(LEDGER/'manifest.json') or p['facts_sha256']!=sha256(LEDGER/'facts.jsonl.gz'):
            raise ValueError('formal training PFT ledger proof differs')
    for p,h in lm['source_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('PFT source changed: '+p)
    with np.load(STATE/'clinical_state.npz') as z:
        base={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits')}
    if len(base['hadm_ids'])!=19401:raise ValueError('original prediction cohort changed')
    lookup={int(h):i for i,h in enumerate(base['hadm_ids'])}
    groups,quarantine,_=selected_groups(SCREEN)
    units=[u for g in groups.values() for u in g if 'pulmonary_function' in u['topics']]
    need={u['note_id'] for u in units};raw={r['note_id']:r for r in rows(SNAP/'raw/discharge.csv.gz') if r['note_id'] in need}
    with (SCREEN/'documents.jsonl').open(encoding='utf-8') as f:docs={r['note_id']:r for r in map(json.loads,f) if r['note_id'] in need}
    bag={};train_counts=Counter();banned_spans={};line_cache={}
    for u in units:
        h=int(u['hadm_id']);pos=lookup[h]
        if int(u['subject_id'])!=int(base['subject_ids'][pos]) or u['patient_split']!=base['patient_splits'][pos]:raise ValueError('PFT member/fold differs')
        text=raw[u['note_id']]['text'];quote=u['quote']
        if sha256_text(text)!=u['original_sha256'] or text[u['start_char']:u['end_char']]!=quote:raise ValueError('PFT quote source differs')
        reason=temporal_reason(quote)
        if reason:
            banned_spans.setdefault(u['note_id'],[]).append((u['start_char'],u['end_char']))
            if u['patient_split']=='train':train_counts[reason]+=1
    def add(u,metric,kind,value,reference):
        item=canonical(metric,kind,value)
        if item is None:return
        name,v,unit=item
        if not np.isfinite(v):raise ValueError('canonical PFT value nonfinite')
        pos=lookup[int(u['hadm_id'])];key=(pos,CONCEPTS.index(name),v)
        if key not in bag:bag[key]=dict(position=pos,concept=name,value=v,unit=unit,evidence=[])
        if reference not in bag[key]['evidence']:bag[key]['evidence'].append(reference)
    for u in sorted(units,key=lambda r:(r['note_id'],r['start_char'])):
        quote=u['quote'];reason=temporal_reason(quote)
        if reason:continue
        text=raw[u['note_id']]['text'];s=docs[u['note_id']]['section_spans'][u['section_ordinal']]
        if not 0<=s['start_char']<=u['start_char']<u['end_char']<=s['end_char']<=len(text):raise ValueError('PFT section source differs')
        flags=[int(u['section']=='past_medical_history'),int(bool(HISTORY.search(quote))),1]
        for b in bindings(quote):
            if b.get('scalar_binding_eligible',False):
                for q in b['quantities']:
                    a=u['start_char']+q['value_start'];z=u['start_char']+q['value_end']
                    ea=u['start_char']+q['evidence_start'];ez=u['start_char']+q['evidence_end']
                    if float(text[a:z])!=q['value'] or sha256_text(text[ea:ez])!=q['evidence_sha256']:raise ValueError('PFT numeric/quantity span differs')
                    ref=dict(parent='lexical',note_id=u['note_id'],original_sha256=u['original_sha256'],quote_start=u['start_char'],quote_end=u['end_char'],
                             quote_sha256=sha256_text(quote),value_start=a,value_end=z,evidence_start=ea,evidence_end=ez,
                             evidence_sha256=q['evidence_sha256'],metadata=flags)
                    add(u,b['metric'],q['kind'],q['value'],ref)
            key=(u['note_id'],u['section_ordinal'])
            if key not in line_cache:line_cache[key]=line_ranges(text,s['start_char'],s['end_char'])
            ranges=line_cache[key];lines=[r[2] for r in ranges];at=u['start_char']+b['metric_start']
            index=next(i for i,(a,z,_) in enumerate(ranges) if a<=at<z);parsed=row_values(lines[index])
            if parsed is None or parsed['metric']!=b['metric']:continue
            header=find_header(lines,index)
            if header is None:continue
            hi,roles=header;rs,re_,line=ranges[index];hs,he,hline=ranges[hi]
            if parsed['trailing_percent'] and roles[-1]!='percent_predicted':continue
            if any(a<re_ and z>hs for a,z in banned_spans.get(u['note_id'],[])):continue
            for role,q in zip(roles,parsed['cells'],strict=True):
                if role=='predicted':continue
                kind=('ratio_percent_predicted' if parsed['metric']=='FEV1/FVC' else 'percent_predicted') if role=='percent_predicted' else (
                    'volume_'+parsed['unit_literal'].lower() if parsed['unit_literal'] else 'unit_unknown')
                a=rs+q['start'];z=rs+q['end']
                if float(text[a:z])!=q['value']:raise ValueError('PFT table cell differs')
                ref=dict(parent='table',note_id=u['note_id'],original_sha256=u['original_sha256'],row_start=rs,row_end=rs+len(line),
                         row_sha256=sha256_text(line),header_start=hs,header_end=hs+len(hline),header_sha256=sha256_text(hline),
                         value_start=a,value_end=z,metadata=flags)
                add(u,parsed['metric'],kind,q['value'],ref)
    facts=[bag[k] for k in sorted(bag)];positions=np.asarray([r['position'] for r in facts],np.int64)
    concepts=np.asarray([CONCEPTS.index(r['concept']) for r in facts],np.int64);raw_values=np.asarray([r['value'] for r in facts],np.float64)
    metadata=np.asarray([np.max([e['metadata'] for e in r['evidence']],axis=0) for r in facts],np.float32)
    values,scales=normalized(raw_values,concepts,positions,base['patient_splits'])
    aggregate=np.zeros((len(base['hadm_ids']),9),np.float32)
    aggregate[positions,concepts]=1
    for i in range(3):np.maximum.at(aggregate[:,6+i],positions,metadata[:,i])
    observed=aggregate[:,:6].any(axis=1)
    arrays=dict(**base,positions=positions,concepts=concepts,raw_values=raw_values,values=values,metadata=metadata,aggregate=aggregate,observed=observed)
    train=base['patient_splits']=='train';selected=train[positions]
    if train.sum()!=13592:raise ValueError('original training cohort changed')
    for r in facts:r['evidence'].sort(key=lambda e:(e['note_id'],e['value_start'],e['parent']))
    for p in (PROTOCOL,Path(__file__),Path('training/tests/test_v6_pft_set_interface.py'),LEDGER/'manifest.json',LEDGER/'facts.jsonl.gz',
              Path('patient_data/pft_table_binding.py'),Path('patient_data/pft_evidence.py'),Path('patient_data/pft_fact_ledger.py'),Path('training/v5/numeric_preprocessing.py')):
        sources[str(p)]=sha256(p)
    report=dict(status='all_original_members_fixed_PFT_rules_training_only_scaling',source_sha256=sources,scales=scales,
        training=dict(admissions=int((train&observed).sum()),patients=len(np.unique(base['subject_ids'][train&observed])),
                      members=int(selected.sum()),temporal_exclusions=dict(train_counts),
                      per_concept={name:int((selected&(concepts==j)).sum()) for j,name in enumerate(CONCEPTS)}),
        cohort=19401,quarantine=quarantine,nontraining_outcomes_used=False,new_model_fits=0,new_vectors=0,high_frequency_used=False,
        limits=['word-rule PFT not medical gold','measurement dates unknown','FINAL retrospective not real-time','no mean/latest/recovery','implicit future information may remain'])
    return arrays,facts,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify'),required=True);p.add_argument('--tag',choices=('a','b'),default='a');args=p.parse_args()
    arrays,facts,report=derive()
    if args.mode=='build':
        if ROOT.exists():raise FileExistsError('PFT interface exists; inspect')
        ROOT.mkdir();np.savez_compressed(ROOT/'interface.npz',**arrays)
        with gzip.open(ROOT/'facts.jsonl.gz','wt',encoding='utf-8') as f:
            for r in facts:f.write(json.dumps(r,sort_keys=True)+'\n')
        report['output_sha256']={name:sha256(ROOT/name) for name in ('interface.npz','facts.jsonl.gz')}
        write_json(ROOT/'manifest.json',report);write_json(ROOT/'state.json',dict(status='built_pending_source_replay',new_model_fits=0))
    else:
        saved=json.loads((ROOT/'manifest.json').read_text());report['output_sha256']=saved['output_sha256']
        if saved!=report:raise ValueError('PFT interface manifest differs')
        for name,h in saved['output_sha256'].items():
            if sha256(ROOT/name)!=h:raise ValueError('PFT interface output differs')
        with np.load(ROOT/'interface.npz') as z:
            if set(z.files)!=set(arrays):raise ValueError('PFT interface keys differ')
            for key,value in arrays.items():np.testing.assert_array_equal(value,z[key])
        with gzip.open(ROOT/'facts.jsonl.gz','rt',encoding='utf-8') as f:prior=[json.loads(s) for s in f]
        if prior!=facts:raise ValueError('PFT interface references differ')
        write_json(ROOT/f'verification_{args.tag}.json',dict(maximum_error=0,manifest_sha256=sha256(ROOT/'manifest.json'),
                  original_source_and_arrays_rebuilt=True,training_scales_rebuilt=True))
        if all((ROOT/f'verification_{t}.json').exists() for t in ('a','b')):
            write_json(ROOT/'state.json',dict(status='two_independent_full_source_replays_verified',new_model_fits=0))
    print(json.dumps(dict(status=report['status'],training=report['training'],cohort=report['cohort'])))


if __name__=='__main__':main()
