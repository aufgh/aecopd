"""小型呼吸支持/微生物候选派生；只读来源、不训练、不解释为因果或定植。"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import re
import time
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from training.v5.numeric_preprocessing import robust_parameters

LOWER = {'SPUTUM','BRONCHOALVEOLAR LAVAGE','BRONCHIAL WASHINGS','MINI-BAL',
         'TRACHEAL ASPIRATE','BRONCHIAL BRUSH','BRONCHIAL BRUSH - PROTECTED'}
MICRO_NAMES = ('named_lower_airway_organism','pseudomonas_aeruginosa','haemophilus_influenzae',
               'streptococcus_pneumoniae','target_organism_resistance','respiratory_virus_positive')
NUMERIC_ITEMS = {'220210':'respiratory_rate','220277':'spo2','223834':'oxygen_flow',
                 '227287':'additional_oxygen_flow','227582':'bipap_oxygen_flow',
                 '223835':'fio2_percent','227579':'bipap_epap','227580':'bipap_ipap'}
NUMERIC_SUMMARY = ('first_selected','minimum_selected','maximum_selected','latest_selected','latest_age_log_days')
SUPPORT_NAMES = ('recorded_invasive_ventilation','recorded_noninvasive_ventilation',
                 'recorded_endotracheal_tube','recorded_bipap_mask','recorded_cpap_mask',
                 'recorded_high_flow_nasal_cannula','recorded_tracheostomy_tube',
                 'recorded_extubation','recorded_repeat_intubation')


def timestamp_interval(exact, date):
    """仅日期记录代表整日，不伪造午夜为精确发生时间。"""
    try:
        if exact not in ('',r'\N'):
            t=datetime.fromisoformat(exact); return t,t
        if date not in ('',r'\N'):
            t=datetime.fromisoformat(date).replace(hour=0,minute=0,second=0,microsecond=0)
            return t,t+timedelta(days=1)-timedelta(microseconds=1)
    except ValueError: pass
    return None


def available_micro(row, start, end):
    draw=timestamp_interval(row['charttime'],row['chartdate'])
    stored=timestamp_interval(row['storetime'],row['storedate'])
    if draw is None: return False,'draw_time_missing'
    if not start<=draw[0]<=draw[1]<=end: return False,'draw_outside_or_boundary_uncertain'
    if stored is None: return False,'result_time_missing'
    if not draw[1]<=stored[0]: return False,'draw_result_order_uncertain'
    if stored[1]>end: return False,'result_after_discharge_or_boundary_uncertain'
    return True,'available_by_discharge'


def micro_scope(row):
    spec=row['spec_type_desc'].strip().upper(); test=row['test_name'].strip().upper()
    if spec in LOWER: return 'lower'
    if ('RESPIRATORY VIRAL' in spec or 'INFLUENZA A/B' in spec or 'RESPIRATORY VIRAL' in test
            or ('INFLUENZA' in test and spec not in ('BLOOD CULTURE','URINE'))): return 'resp_viral'
    if spec=='PLEURAL FLUID': return 'pleural_not_lower_airway'
    return 'other'


def invalid_micro(row):
    org=row['org_name'].strip().upper(); comment=row['comments'].upper()
    if org in ('CANCELLED','CANCELED','PENDING','NEGATIVE'): return org!='NEGATIVE'
    return (bool(re.search(r'TEST CANCEL|TEST WAS CANCEL|NOT PERFORMED|SPECIMEN REJECT',comment))
            or ('INVALID' in comment and 'CONTAMIN' in comment))


def target_organism(org):
    name=org.strip().upper()
    # 精确属种/经核验别名，避免 NON-FERMENTER, NOT PSEUDOMONAS ... 误阳性。
    if name=='PSEUDOMONAS AERUGINOSA': return 1
    if name=='HAEMOPHILUS INFLUENZAE' or name.startswith('HAEMOPHILUS INFLUENZAE,'): return 2
    if name=='STREPTOCOCCUS PNEUMONIAE': return 3
    return None


def viral_positive(org, comment):
    virus=r'INFLUENZA|RESPIRATORY SYNCYTIAL|PARAINFLUENZA|ADENOVIRUS|RHINOVIRUS|METAPNEUMOVIRUS'
    org,comment=org.upper(),comment.upper()
    named=(org not in ('',r'\N','NEGATIVE') and bool(re.search(virus,org))
           and 'NEGATIVE' not in org and 'NOT DETECTED' not in org)
    explicit=bool(re.search(r'(?:POSITIVE FOR|(?<!NOT )DETECTED:)\s*(?:'+virus+r')',comment))
    return named or explicit


def union_hours(intervals):
    if not intervals: return 0.
    total=0.; left,right=sorted(intervals)[0]
    for start,end in sorted(intervals)[1:]:
        if end<start: raise ValueError('negative duration')
        if start<=right: right=max(right,end)
        else: total+=(right-left).total_seconds()/3600; left,right=start,end
    if right<left: raise ValueError('negative duration')
    return total+(right-left).total_seconds()/3600


def canonical_resp(item, value, unit):
    try: v=float(value)
    except (TypeError,ValueError): return None
    if not math.isfinite(v): return None
    if item=='220210': return v if unit=='insp/min' and 0<v<=100 else None
    if item=='220277': return v if unit=='%' and 0<v<=100 else None
    if item=='223835':
        if unit not in ('',r'\N','%'): return None
        if .20<=v<=1: return v*100
        return v if 20<=v<=100 else None
    if item in ('227579','227580'): return v if unit=='cmH2O' and 0<=v<=60 else None
    return v if unit=='L/min' and 0<=v<=100 else None


def build(snapshot, supplement, state_dir, output):
    if output.exists(): raise FileExistsError(output)
    started=time.perf_counter()
    sm=json.loads((supplement/'metadata/manifest.json').read_text(encoding='utf-8'))
    st=json.loads((state_dir/'manifest.json').read_text(encoding='utf-8'))
    parent=json.loads((snapshot/'metadata/manifest.json').read_text(encoding='utf-8'))
    if (sm['status']!='validated_restricted_source_export_not_model_validated'
            or parent['status']!='validated' or not sm['cohort_membership_exactly_matches_parent']
            or sha256(snapshot/'metadata/manifest.json')!=sm['parent_snapshot_manifest_sha256']
            or sha256(state_dir/'clinical_state.npz')!=st['output_sha256']['clinical_state.npz']):
        raise ValueError('source lineage mismatch')
    sources={}
    for name,meta in sm['sources'].items():
        path=supplement/'raw'/(name+'.csv.gz'); digest=sha256(path)
        if digest!=meta['sha256']: raise ValueError('source modified')
        sources[name]=digest
    if sha256(snapshot/'cohort/admissions.csv.gz')!=sm['parent_cohort_sha256']: raise ValueError('cohort changed')
    with np.load(state_dir/'clinical_state.npz',allow_pickle=False) as archive:
        state={k:archive[k] for k in archive.files}
    hadms=[str(h) for h in state['hadm_ids']]; positions={h:i for i,h in enumerate(hadms)}
    if len(hadms)!=19401 or len(positions)!=19401: raise ValueError('member drift')
    members={r['hadm_id']:r for r in rows(snapshot/'cohort/admissions.csv.gz')}
    admissions={h:(datetime.fromisoformat(members[h]['admittime']),datetime.fromisoformat(members[h]['dischtime'])) for h in hadms}
    for h,i in positions.items():
        if int(members[h]['subject_id'])!=state['subject_ids'][i]: raise ValueError('identity mismatch')
    def source_rows(name):
        count=0
        for r in rows(supplement/'raw'/(name+'.csv.gz')):
            count+=1
            if r['hadm_id'] not in members or r['subject_id']!=members[r['hadm_id']]['subject_id']: raise ValueError('raw association mismatch')
            if r['hadm_id'] in positions: yield r
        if count!=sm['sources'][name]['rows']: raise ValueError('source row count changed')
    n=len(hadms); audit=defaultdict(Counter)
    micro=np.zeros((n,6),dtype=np.float32); micro_mask=np.zeros_like(micro,dtype=bool)
    micro_raw=np.zeros(n,dtype=bool); lower_available=np.zeros(n,dtype=bool); viral_available=np.zeros(n,dtype=bool)
    unique_ids=set(); isolates=set(); available_rows=0; susceptible_isolates=set(); resistant_isolates=set()
    for r in source_rows('microbiologyevents'):
        if r['microevent_id'] in unique_ids: raise ValueError('duplicate microevent_id')
        unique_ids.add(r['microevent_id'])
        scope=micro_scope(r); audit['micro_specimens'][r['spec_type_desc']]+=1
        if scope not in ('lower','resp_viral'): audit['micro']['non_target_specimen_rows']+=1; continue
        i=positions[r['hadm_id']]; micro_raw[i]=True; audit['micro']['respiratory_candidate_rows']+=1
        ok,reason=available_micro(r,*admissions[r['hadm_id']]); audit['micro'][reason]+=1
        if not ok: continue
        if invalid_micro(r): audit['micro']['cancelled_or_invalid_result_rows']+=1; continue
        # 仅培养/明确呼吸病毒检查；涂片不能替代培养病原结果。
        if scope=='lower' and 'CULTURE' not in r['test_name'].upper():
            audit['micro']['nonculture_lower_airway_rows']+=1; continue
        org=r['org_name'].strip().upper(); available_rows+=1
        key=(r['hadm_id'],r['micro_specimen_id'],r['test_seq'],r['test_itemid'],org,r['isolate_num'])
        if key in isolates: audit['micro']['rows_collapsed_same_specimen_test_isolate']+=1
        isolates.add(key)
        if scope=='lower':
            lower_available[i]=True
            if org not in ('',r'\N','NEGATIVE','MIXED BACTERIAL FLORA'):
                micro[i,0]=1
            target=target_organism(org)
            if target is not None:
                micro[i,target]=1
                if r['interpretation'] in ('S','I','R'):
                    susceptible_isolates.add(key); micro_mask[i,4]=True
                    if r['interpretation']=='R': resistant_isolates.add(key); micro[i,4]=1
        else:
            viral_available[i]=True
            # 定向检查的明确阳性，不从泛化评论或孤立 POSITIVE 推断病毒。
            if viral_positive(org,r['comments']): micro[i,5]=1
    micro_mask[:,:4]=lower_available[:,None]; micro_mask[:,5]=viral_available
    micro_state=np.where(lower_available|viral_available,2,np.where(micro_raw,1,0)).astype(np.int8)
    icu=np.zeros(n,dtype=bool); stay_to_hadm={}
    for r in source_rows('icustays'):
        if r['stay_id'] in stay_to_hadm: raise ValueError('duplicate stay_id')
        stay_to_hadm[r['stay_id']]=r['hadm_id']; icu[positions[r['hadm_id']]]=True
    support=np.zeros((n,len(SUPPORT_NAMES)),dtype=np.float32); support_mask=np.zeros_like(support,dtype=bool)
    durations=np.zeros((n,2),dtype=np.float64); duration_mask=np.zeros_like(durations,dtype=bool)
    intervals=defaultdict(list); intubations=defaultdict(set); extubations=defaultdict(set); orders=set()
    for r in source_rows('respiratory_procedureevents'):
        h=r['hadm_id']; i=positions[h]
        if stay_to_hadm.get(r['stay_id'])!=h: raise ValueError('procedure/ICU stay mismatch')
        try: start,end,stored=(datetime.fromisoformat(r[k]) for k in ('starttime','endtime','storetime'))
        except ValueError: audit['procedure']['invalid_time_rows']+=1; continue
        if not admissions[h][0]<=start<=end<=admissions[h][1] or not start<=stored<=admissions[h][1]:
            audit['procedure']['unavailable_or_outside_rows']+=1; continue
        if r['statusdescription'] not in ('FinishedRunning','Stopped'):
            audit['procedure']['unconfirmed_status_rows']+=1; continue
        key=(h,r['stay_id'],r['itemid'],r['orderid'],start,end)
        if key in orders: audit['procedure']['duplicate_rows_collapsed']+=1; continue
        orders.add(key); audit['procedure']['accepted_rows']+=1
        if r['itemid'] in ('225792','225794'):
            j=0 if r['itemid']=='225792' else 1
            support[i,j]=1; support_mask[i,j]=True; intervals[h,j].append((start,end)); duration_mask[i,j]=True
        if r['itemid']=='224385': intubations[h].add(start)
        if r['itemid'] in ('225468','225477','227194'): extubations[h].add(start); support[i,7]=1; support_mask[i,7]=True
    for h,ts in intubations.items():
        i=positions[h]; support[i,8]=len(ts)>1; support_mask[i,8]=True
    for (h,j),spans in intervals.items(): durations[positions[h],j]=union_hours(spans)
    numeric=defaultdict(list); seen=set(); respiratory_rows=np.zeros(n,dtype=bool)
    devices={'ENDOTRACHEAL TUBE':2,'BIPAP MASK':3,'CPAP MASK':4,'HIGH FLOW NASAL CANNULA':5,'TRACHEOSTOMY TUBE':6}
    for r in source_rows('respiratory_chartevents'):
        h=r['hadm_id']; i=positions[h]
        if stay_to_hadm.get(r['stay_id'])!=h: raise ValueError('chart/ICU stay mismatch')
        t,s=(datetime.fromisoformat(r[k]) for k in ('charttime','storetime'))
        if not admissions[h][0]<=t<=s<=admissions[h][1]: raise ValueError('export time restriction failed')
        key=tuple(r[k] for k in ('hadm_id','stay_id','charttime','storetime','itemid','value','valuenum','valueuom'))
        if key in seen: audit['chartevents']['identical_rows_collapsed']+=1; continue
        seen.add(key); respiratory_rows[i]=True
        if r['selection_role']=='state_record':
            audit['device_values'][r['itemid']+':'+r['value']]+=1
            if r['itemid']=='226732':
                name=r['value'].strip().upper()
                if name not in ('',r'\N','OTHER'): support_mask[i,2:7]=True
                if name in devices: support[i,devices[name]]=1
            continue
        value=canonical_resp(r['itemid'],r['valuenum'],r['valueuom'])
        if value is None: audit['chartevents']['rejected_value_or_unit:'+r['itemid']]+=1; continue
        numeric[h,r['itemid']].append((t,value)); audit['chartevents']['accepted_numeric_anchors:'+r['itemid']]+=1
    raw=np.zeros((n,8,5),dtype=np.float64); masks=np.zeros_like(raw,dtype=bool)
    for (h,item),observations in numeric.items():
        k=list(NUMERIC_ITEMS).index(item); i=positions[h]; by_time=defaultdict(list)
        for t,v in observations: by_time[t].append(v)
        obs=[(t,float(np.median(v))) for t,v in sorted(by_time.items())]
        raw[i,k]=[obs[0][1],min(v for _,v in obs),max(v for _,v in obs),obs[-1][1],
                  math.log1p((admissions[h][1]-obs[-1][0]).total_seconds()/86400)]
        masks[i,k]=True
    scaled=np.zeros_like(raw); normalization={}; train=state['patient_splits']=='train'
    for k,name in enumerate(NUMERIC_ITEMS.values()):
        for f,label in enumerate(NUMERIC_SUMMARY):
            valid=train&masks[:,k,f]
            if not valid.any(): normalization[name+':'+label]={'method':'no_training_observation'}; masks[:,k,f]=False; continue
            p=robust_parameters(raw[valid,k,f]); normalization[name+':'+label]=p
            scaled[:,k,f]=np.where(masks[:,k,f],np.clip((raw[:,k,f]-p['median'])/p['scale'],-8,8),0)
    def coverage(present):
        return {'admissions':int(present.sum()),'patients':len(set(state['subject_ids'][present].tolist())),
                'missing_admissions':int((~present).sum())}
    def detection(values, masks):
        return {'positive_admissions':int((values>0).sum()),
                'positive_patients':len(set(state['subject_ids'][values>0].tolist())),
                'source_observed_admissions':int(masks.sum()),'unknown_admissions':int((~masks).sum())}
    report={'status':'resp_micro_context_validated_not_trained','admissions':n,
            'parent_supplement_manifest_sha256':sha256(supplement/'metadata/manifest.json'),
            'parent_state_manifest_sha256':sha256(state_dir/'manifest.json'),
            'source_sha256':sources,'code_sha256':sha256(Path(__file__)),'audit':{k:dict(v) for k,v in audit.items()},
            'coverage':{'ICU_stay':coverage(icu),'respiratory_chart_source':coverage(respiratory_rows),
                        'lower_airway_culture_available':coverage(lower_available),'respiratory_viral_test_available':coverage(viral_available)},
            'micro_coverage':{name:detection(micro[:,i],micro_mask[:,i]) for i,name in enumerate(MICRO_NAMES)},
            'numeric_coverage':{name:coverage(masks[:,i,0]) for i,name in enumerate(NUMERIC_ITEMS.values())},
            'support_coverage':{name:detection(support[:,i],support_mask[:,i]) for i,name in enumerate(SUPPORT_NAMES)},
            'micro_source_state_counts':{str(i):int((micro_state==i).sum()) for i in range(3)},
            'unique_available_specimen_test_isolates':len(isolates),'available_micro_rows':available_rows,
            'target_isolates_with_SIR':len(susceptible_isolates),'target_isolates_with_R':len(resistant_isolates),
            'micro_names':MICRO_NAMES,'resp_numeric_names':list(NUMERIC_ITEMS.values()),'resp_numeric_summary_names':NUMERIC_SUMMARY,
            'support_names':SUPPORT_NAMES,'duration_names':('recorded_invasive_hours','recorded_noninvasive_hours'),
            'limitations':['Positive culture does not establish infection versus colonization or causal prognosis',
                'Result time is the last known update; late final results may exclude earlier preliminary information',
                'No linked source does not prove no test/support outside ICU or outside this linked hospitalization',
                'Support zero means no positive evidence in the retained source, not proven clinical absence',
                'Last ICU observation is not the discharge state; numeric extrema concern valid selected anchors only',
                'Procedure durations are union of confirmed recorded intervals; do not infer indication or complete duration',
                'CPAP mask is not automatically acute COPD NIV; airway tube is not automatically invasive ventilation'],
            'elapsed_seconds':round(time.perf_counter()-started,2)}
    for name,digest in sources.items():
        if sha256(supplement/'raw'/(name+'.csv.gz'))!=digest: raise ValueError('source changed during build')
    output.mkdir(parents=True)
    np.savez_compressed(output/'resp_micro_context.npz',hadm_ids=state['hadm_ids'],subject_ids=state['subject_ids'],patient_splits=state['patient_splits'],
        micro_values=micro,micro_masks=micro_mask,micro_source_state=micro_state,icu_stay_present=icu,
        resp_numeric_raw=raw,resp_numeric_scaled=scaled,resp_numeric_masks=masks,
        support_values=support,support_masks=support_mask,recorded_duration_hours=durations,recorded_duration_masks=duration_mask)
    with np.load(output/'resp_micro_context.npz',allow_pickle=False) as archive:
        if len(archive['hadm_ids'])!=n or not np.isfinite(archive['resp_numeric_scaled']).all(): raise ValueError('roundtrip failed')
        if not np.array_equal(archive['micro_values'],micro): raise ValueError('micro roundtrip failed')
    (output/'normalization.json').write_text(json.dumps(normalization,indent=2),encoding='utf-8')
    report['output_sha256']={name:sha256(output/name) for name in ('resp_micro_context.npz','normalization.json')}
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','coverage','micro_source_state_counts','elapsed_seconds')},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('snapshot','supplement','state_dir','output'): parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    args=parser.parse_args(); build(args.snapshot,args.supplement,args.state_dir,args.output)
