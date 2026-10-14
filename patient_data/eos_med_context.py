"""源可用性与EOS/已记录激素时间上下文；不训练，不覆盖原状态。"""
from __future__ import annotations
import argparse
import gzip
from collections import defaultdict, Counter
from datetime import datetime
import json
import math
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from patient_data.preprocess_events import ADMINISTERED, NON_SYSTEMIC_ROUTES
from training.v5.numeric_preprocessing import robust_parameters
from training.v5.clinical_concepts import medication_classes

EOS_ITEMS={'51199':'absolute','52073':'absolute','51200':'percent'}
EOS_NAMES=('absolute','percent')
VALUE_NAMES=('first','minimum','maximum','latest','first_before_recorded_steroid','latest_age_log_days')
RELATIONS=('before_recorded_steroid','after_recorded_steroid','same_time_order_unknown','steroid_timing_unknown')
# 途径是上游判定，不作为网络输入。未知不猜测为全身，也不称未给药。
SYSTEMIC_ROUTES={'PO','ORAL','IV','IV DRIP','IV PUSH','IM','INTRAMUSCULAR','INTRAVENOUS','NG','NGT','OG','PO/NG','PO/NGT','PO/OG','G TUBE','GT','PEG','J TUBE','JT','PER G TUBE','PER J TUBE','ENTERAL'}


def finite(value):
    try:
        v=float(value)
        return v if math.isfinite(v) else None
    except (ValueError,TypeError): return None


def canonical_eos(value, unit, kind):
    v=finite(value)
    if v is None or v<0: return None
    unit=unit.strip().replace('µ','u').replace('μ','u')
    if kind=='absolute':
        if unit=='K/uL': return v*1000
        if unit in ('#/uL','cells/uL'): return v
        return None
    if kind=='percent' and unit=='%' and v<=100: return v
    return None


def relation(draw, confirmed, ambiguous=()):
    if confirmed is None: return 3
    if draw==confirmed: return 2
    if draw>confirmed: return 1
    if any(t<=draw for t in ambiguous): return 3
    return 0


def summarize_eos(samples, discharge, confirmed, ambiguous=()):
    values=np.zeros(6,dtype=np.float64); masks=np.zeros(6,dtype=bool); context=np.zeros(8)
    if not samples: return values,masks,context
    by_time=defaultdict(list)
    for t,v in samples:
        if t>discharge or not math.isfinite(v) or v<0: raise ValueError('invalid EOS observation')
        by_time[t].append(v)
    ordered=[(t,float(np.median(v))) for t,v in sorted(by_time.items())]
    first,latest=ordered[0],ordered[-1]
    values[:4]=[first[1],min(v for _,v in ordered),max(v for _,v in ordered),latest[1]]
    masks[:4]=True
    pre=[(t,v) for t,v in ordered if relation(t,confirmed,ambiguous)==0]
    if pre: values[4]=pre[0][1]; masks[4]=True
    values[5]=math.log1p((discharge-latest[0]).total_seconds()/86400); masks[5]=True
    context[relation(first[0],confirmed,ambiguous)]=1
    context[4+relation(latest[0],confirmed,ambiguous)]=1
    return values,masks,context


def build(raw_root, state_dir, processed, output):
    if output.exists(): raise FileExistsError(output)
    started=time.perf_counter()
    sm=json.loads((raw_root/'metadata/manifest.json').read_text(encoding='utf-8'))
    st=json.loads((state_dir/'manifest.json').read_text(encoding='utf-8'))
    pm=json.loads((processed/'manifest.json').read_text(encoding='utf-8'))
    if sm['status']!='validated' or st['status']!='derived_and_roundtrip_checked_not_prediction_validated' or pm['medication_mapping']!='clinical_reviewed_2026_09':
        raise ValueError('unvalidated source/version')
    if sha256(state_dir/'clinical_state.npz')!=st['output_sha256']['clinical_state.npz']:
        raise ValueError('state changed')
    inputs={}
    for name in ('raw/labevents','raw/emar','raw/emar_detail','raw/prescriptions','cohort/admissions','cohort/patient_splits'):
        digest=sha256(raw_root/(name+'.csv.gz'))
        if digest!=sm['sources'][name]['sha256']: raise ValueError('raw source changed')
        inputs[name]=digest
    event_file=processed/'events.csv.gz'
    if sha256(event_file)!=pm['output_sha256']['events.csv.gz']: raise ValueError('accepted medication source changed')
    inputs['accepted_events']=sha256(event_file)
    with np.load(state_dir/'clinical_state.npz',allow_pickle=False) as archive:
        state={k:archive[k] for k in archive.files}
    hadms=[str(h) for h in state['hadm_ids']]; selected=set(hadms)
    admissions={r['hadm_id']:(datetime.fromisoformat(r['admittime']),datetime.fromisoformat(r['dischtime']))
                for r in rows(raw_root/'cohort/admissions.csv.gz') if r['hadm_id'] in selected}
    if len(hadms)!=19401 or len(admissions)!=19401: raise ValueError('member drift')
    accepted={}
    for r in rows(event_file):
        if r['kind']=='medication' and r['concept']=='systemic_steroid':
            key=r['source_id']
            if key in accepted: raise ValueError('duplicate accepted steroid source')
            accepted[key]=(r['hadm_id'],datetime.fromisoformat(r['charttime']))
    raw_any, available_any=set(),set()
    steroid_records={}
    dose_info=defaultdict(list); routes=defaultdict(set)
    audit=Counter(); ambiguous=defaultdict(list); confirmed=defaultdict(list)
    raw_count=0
    for r in rows(raw_root/'raw/emar.csv.gz'):
        raw_count+=1
        h=r['hadm_id']
        if h not in selected: continue
        raw_any.add(h)
        start,end=admissions[h]
        try:
            t,s=datetime.fromisoformat(r['charttime']),datetime.fromisoformat(r['storetime'])
            if start<=t<=s<=end: available_any.add(h)
        except ValueError: pass
        key=r['emar_id']+':'+r['emar_seq']
        if key in accepted:
            if key in steroid_records: raise ValueError('duplicate eMAR key')
            steroid_records[key]=r
    if set(steroid_records)!=set(accepted): raise ValueError('missing steroid source')
    if raw_count!=sm['sources']['raw/emar']['rows']: raise ValueError('raw eMAR count changed')
    for r in rows(raw_root/'raw/emar_detail.csv.gz'):
        key=r['emar_id']+':'+r['emar_seq']
        if key not in accepted: continue
        dose_info[key].append((finite(r['dose_given']),r['dose_given_unit'],finite(r['product_amount_given'])))
        if r['route'] not in ('','\\N'): routes[key].add(r['route'].strip().upper())
    for key,(h,t) in accepted.items():
        r=steroid_records[key]
        if r['hadm_id']!=h or datetime.fromisoformat(r['charttime'])!=t or r['event_txt'] not in ADMINISTERED:
            raise ValueError('accepted steroid timing/status mismatch')
        start,end=admissions[h]
        stored=datetime.fromisoformat(r['storetime'])
        if not start<=t<=stored<=end: raise ValueError('steroid not available by discharge')
        if routes[key] and routes[key].issubset(NON_SYSTEMIC_ROUTES):
            audit['non_systemic_routes_excluded']+=1; continue
        doses=[v for v,unit,_ in dose_info[key] if v is not None and unit not in ('','\\N')]
        if doses and not any(v>0 for v in doses):
            audit['explicit_zero_or_negative_dose_excluded']+=1; continue
        positive=any(v>0 for v in doses)
        route_known=bool(routes[key]) and routes[key].issubset(SYSTEMIC_ROUTES)
        if positive and route_known:
            confirmed[h].append(t); audit['positive_dose_known_systemic_route_records']+=1
        else:
            ambiguous[h].append(t)
            audit['dose_or_route_not_confirmed_records']+=1
            for route in routes[key]: audit['ambiguous_route:'+route]+=1
    first={h:min(ts) for h,ts in confirmed.items()}
    prescriptions=set()
    for r in rows(raw_root/'raw/prescriptions.csv.gz'):
        h=r['hadm_id']
        if h not in selected or 'systemic_steroid' not in medication_classes(r['drug'],mapping_version='clinical_reviewed_2026_09'): continue
        try:
            t=datetime.fromisoformat(r['starttime'])
            if admissions[h][0]<=t<=admissions[h][1]: prescriptions.add(h)
        except ValueError: pass
    obs=defaultdict(list); eos_audit=defaultdict(Counter); count=0
    for r in rows(raw_root/'raw/labevents.csv.gz'):
        count+=1
        h,item=r['hadm_id'],r['itemid']
        if h not in selected or item not in EOS_ITEMS: continue
        kind=EOS_ITEMS[item]; a=eos_audit[kind]; a['candidate_rows']+=1
        try:
            t,s=datetime.fromisoformat(r['charttime']),datetime.fromisoformat(r['storetime'])
            if not admissions[h][0]<=t<=s<=admissions[h][1]:
                a['time_unavailable']+=1; continue
        except ValueError:
            a['invalid_timestamp']+=1; continue
        value=canonical_eos(r['valuenum'],r['valueuom'],kind)
        if value is None:
            a['invalid_value_or_unit']+=1; continue
        obs[h,kind].append((t,value)); a['accepted_rows']+=1
        a['unit:'+r['valueuom']]+=1
    if count!=sm['sources']['raw/labevents']['rows']: raise ValueError('raw lab count changed')
    values=np.zeros((len(hadms),2,6)); masks=np.zeros_like(values,dtype=bool); context=np.zeros((len(hadms),2,8))
    med_mask=np.zeros((len(hadms),7),dtype=bool); availability=np.zeros(len(hadms),dtype=np.int8)
    evidence=[]; precoverage=Counter(); timing=defaultdict(Counter)
    for i,h in enumerate(hadms):
        med_mask[i]=h in available_any
        target=bool(state['med_exposure'][i].any())
        if target and h not in available_any: raise ValueError('exposure without valid eMAR source')
        availability[i]=(3 if target else 2 if h in available_any else 1 if h in raw_any else 0)
        for k,kind in enumerate(EOS_NAMES):
            samples=obs.get((h,kind),[])
            values[i,k],masks[i,k],context[i,k]=summarize_eos(samples,admissions[h][1],first.get(h),ambiguous.get(h,[]))
            if masks[i,k,0]:
                eos_audit[kind]['observed_admissions']+=1
                eos_audit[kind]['admissions_with_distinct_repeat_times']+=int(len({t for t,_ in samples})>=2)
                eos_audit[kind]['same_timestamp_rows_collapsed']+=len(samples)-len({t for t,_ in samples})
                timing[kind][RELATIONS[int(context[i,k,:4].argmax())]]+=1
            precoverage[kind]+=int(masks[i,k,4])
        evidence.append({'hadm_id':h,'raw_emar_present':int(h in raw_any),'available_emar_present':int(h in available_any),
                         'target_exposure_present':int(target),'first_positive_systemic_steroid_charttime':first[h].isoformat() if h in first else '',
                         'ambiguous_recorded_steroid_present':int(bool(ambiguous.get(h))),
                         'steroid_prescription_present':int(h in prescriptions),'prescription_only':int(h in prescriptions and h not in first and not ambiguous.get(h))})
    scaled=values.copy(); norm={}
    transformed=values.copy(); transformed[:,:,:5]=np.log1p(transformed[:,:,:5])
    train=state['patient_splits']=='train'
    for k,kind in enumerate(EOS_NAMES):
        for f,name in enumerate(VALUE_NAMES):
            valid=train&masks[:,k,f]
            if not valid.any():
                masks[:,k,f]=False; scaled[:,k,f]=0; norm[kind+':'+name]={'method':'no_training_observation'}; continue
            p=robust_parameters(transformed[valid,k,f]); norm[kind+':'+name]=p
            scaled[:,k,f]=np.where(masks[:,k,f],np.clip((transformed[:,k,f]-p['median'])/p['scale'],-8,8),0)
    features=np.concatenate((scaled,masks.astype(float),context),axis=2).astype(np.float32)
    for name,digest in inputs.items():
        path=event_file if name=='accepted_events' else raw_root/(name+'.csv.gz')
        if sha256(path)!=digest: raise ValueError('source changed during build')
    output.mkdir(parents=True)
    np.savez_compressed(output/'eos_med_context.npz',hadm_ids=state['hadm_ids'],subject_ids=state['subject_ids'],patient_splits=state['patient_splits'],
                        eos_values_raw=values,eos_values_scaled=scaled,eos_value_masks=masks,eos_relation_flags=context,eos_features=features,
                        eos_names=np.asarray(EOS_NAMES),med_exposure=state['med_exposure'],med_source_mask=med_mask,med_availability_state=availability)
    with gzip.open(output/'steroid_source_audit.csv.gz','wt',encoding='utf-8',newline='') as out:
        import csv
        writer=csv.DictWriter(out,fieldnames=list(evidence[0])); writer.writeheader(); writer.writerows(evidence)
    (output/'normalization.json').write_text(json.dumps(norm,indent=2),encoding='utf-8')
    with np.load(output/'eos_med_context.npz',allow_pickle=False) as archive:
        restored={k:archive[k] for k in archive.files}
    if not np.array_equal(restored['eos_features'],features) or not np.isfinite(features).all(): raise ValueError('roundtrip failed')
    report={'status':'eos_med_context_validated_not_trained','admissions':len(hadms),'input_sha256':inputs,
            'parent_clinical_state_manifest_sha256':sha256(state_dir/'manifest.json'),'code_sha256':sha256(Path(__file__)),
            'output_sha256':{n:sha256(output/n) for n in ('eos_med_context.npz','steroid_source_audit.csv.gz','normalization.json')},
            'med_availability_codes':{'0':'no raw eMAR','1':'raw eMAR but none available by discharge','2':'available eMAR without retained seven-class exposure','3':'retained target exposure'},
            'med_availability_counts':{str(i):int((availability==i).sum()) for i in range(4)},
            'eos_audit':{k:dict(v) for k,v in eos_audit.items()},'first_eos_relation_counts':{k:dict(v) for k,v in timing.items()},
            'eos_observed_patients':{kind:len(set(state['subject_ids'][masks[:,k,0]].tolist())) for k,kind in enumerate(EOS_NAMES)},
            'pre_recorded_steroid_value_coverage':dict(precoverage),'steroid_audit':dict(audit),
            'positive_systemic_steroid_admissions':len(first),'only_prescription_admissions':sum(r['prescription_only'] for r in evidence),
            'eos_feature_shape':list(features.shape),'value_names':VALUE_NAMES,'relation_names':RELATIONS,'elapsed_seconds':round(time.perf_counter()-started,2),
            'limitations':['eMAR present is partial source visibility, not proof of complete medication observation',
                           'Before recorded steroid does not prove no steroid in ED, outside hospital or before admission',
                           'First/last relation concerns confirmed positive dose and known systemic route; uncertain earlier records block pre designation',
                           'Prescriptions never establish actual administration; EOS zero is a valid observed value',
                           'EOS is not a monotonic recovery severity axis; values retained separately, no claim of treatment efficacy']}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'availability':report['med_availability_counts'],'seconds':report['elapsed_seconds']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('raw_root','state_dir','processed','output'): parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args=parser.parse_args(); build(args.raw_root,args.state_dir,args.processed,args.output)
