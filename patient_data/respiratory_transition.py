"""已保存的四呼吸项目锚点时序派生；不访问数据库/原文，不推断因果恢复。"""
from collections import Counter,defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import argparse
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.resp_micro_context import canonical_resp
from training.v5.numeric_preprocessing import robust_parameters
from training.v6.train import write_json

SNAP=Path('patient_data/mimiciv_copd/snapshot_2026-09-27')
SOURCE=Path('patient_data/mimiciv_copd/supplement_resp_micro_2026-10-09')
STATE=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2')
ROOT=Path('patient_data/processed/v6_respiratory_transition_2026-10-13')
PROTOCOL=Path('docs/V6_RESPIRATORY_TRANSITION_PROTOCOL_2026-10-13.md')
ITEMS=('220210','220277','223834','223835')
NAMES=('respiratory_rate','spo2','oxygen_flow','fio2_percent')
CHANNELS=('first_selected','directional_extreme_selected','latest_selected','latest_minus_first','latest_minus_extreme',
          'log_span_days','log_latest_age_days','log_extreme_to_latest_days','log_first_admission_offset_days')
FLAGS=('observed','different_time_retest','latest_at_directional_extreme')
PAIR_NAMES=('spo2_first_matched_oxygen_flow','spo2_latest_matched_oxygen_flow','matched_oxygen_flow_change',
            'spo2_first_matched_fio2_percent','spo2_latest_matched_fio2_percent','matched_fio2_change')


def icu_contains(stay,t):
    if any(stay[k] in ('',r'\N',r'\\N') for k in ('intime','outtime')):
        return None
    return datetime.fromisoformat(stay['intime'])<=t<=datetime.fromisoformat(stay['outtime'])


def collapse(observations):
    grouped=defaultdict(list)
    for t,v,stay in observations:grouped[t].append((v,stay))
    return [(t,float(np.median([v for v,_ in obs])), min(s for _,s in obs)) for t,obs in sorted(grouped.items())]


def summarize(obs,start,end,low=False):
    raw=np.zeros(9);mask=np.zeros(9,bool);flags=np.zeros(3)
    if not obs:return raw,mask,flags
    obs=collapse(obs);first,last=obs[0],obs[-1]
    worst=min(v for _,v,_ in obs) if low else max(v for _,v,_ in obs)
    peak=max(t for t,v,_ in obs if v==worst)
    repeated=len(obs)>1;after=peak<last[0]
    raw[:3]=[first[1],worst,last[1]];mask[:3]=True
    raw[3]=last[1]-first[1] if repeated else 0;mask[3]=repeated
    raw[4]=last[1]-worst if after else 0;mask[4]=after
    raw[5]=math.log1p((last[0]-first[0]).total_seconds()/86400);mask[5]=repeated
    raw[6]=math.log1p((end-last[0]).total_seconds()/86400);mask[6]=True
    raw[7]=math.log1p((last[0]-peak).total_seconds()/86400);mask[7]=after
    raw[8]=math.log1p((first[0]-start).total_seconds()/86400);mask[8]=True
    flags[:]=[1,repeated,peak==last[0]]
    if not np.isfinite(raw).all() or (raw[[5,6,7,8]]<0).any():raise ValueError('invalid observation time order')
    return raw,mask,flags


def nearest(target,obs):
    t,_,stay=target
    eligible=[o for o in obs if o[2]==stay and abs((o[0]-t).total_seconds())<=3600]
    return min(eligible,key=lambda o:(abs((o[0]-t).total_seconds()),o[0])) if eligible else None


def support_pairs(spo,flow,fio):
    raw=np.zeros(6);mask=np.zeros(6,bool)
    if not spo:return raw,mask
    spo=collapse(spo)
    for i,other in enumerate((flow,fio)):
        a,b=nearest(spo[0],collapse(other)),nearest(spo[-1],collapse(other))
        if a:raw[3*i]=a[1];mask[3*i]=True
        if b:raw[3*i+1]=b[1];mask[3*i+1]=True
        if a and b and spo[0][0]<spo[-1][0] and spo[0][2]==spo[-1][2]:
            raw[3*i+2]=b[1]-a[1];mask[3*i+2]=True
    return raw,mask


def normalize(raw,mask,train):
    out=np.zeros_like(raw);parameters={}
    for col in np.ndindex(raw.shape[1:]):
        key=(slice(None),)+col;valid=train&mask[key]
        if not valid.any():parameters[str(col)]={'method':'no_training_observation'};continue
        p=robust_parameters(raw[key][valid]);parameters[str(col)]=p
        out[key]=np.where(mask[key],np.clip((raw[key]-p['median'])/p['scale'],-8,8),0)
    return out,parameters


def derive():
    sm=json.loads((SOURCE/'metadata/manifest.json').read_text());st=json.loads((STATE/'manifest.json').read_text())
    parent=json.loads((SNAP/'metadata/manifest.json').read_text())
    if (sm['status']!='validated_restricted_source_export_not_model_validated' or parent['status']!='validated' or
        sha256(SNAP/'metadata/manifest.json')!=sm['parent_snapshot_manifest_sha256']):raise ValueError('parent lineage differs')
    for name in ('respiratory_chartevents','icustays','d_items'):
        if sha256(SOURCE/'raw'/f'{name}.csv.gz')!=sm['sources'][name]['sha256']:raise ValueError('source changed: '+name)
    if sha256(STATE/'clinical_state.npz')!=st['output_sha256']['clinical_state.npz']:raise ValueError('state differs')
    if sha256(SNAP/'cohort/admissions.csv.gz')!=sm['parent_cohort_sha256']:raise ValueError('cohort changed')
    with np.load(STATE/'clinical_state.npz') as z:hadms=z['hadm_ids'];people=z['subject_ids'];splits=z['patient_splits']
    cohort={r['hadm_id']:r for r in rows(SNAP/'cohort/admissions.csv.gz')};by_hadm={str(h):i for i,h in enumerate(hadms)}
    stays={r['stay_id']:r for r in rows(SOURCE/'raw/icustays.csv.gz')}
    obs=defaultdict(list);audit=Counter();units=Counter();roles=Counter();seen=set();stays_per=defaultdict(set)
    for r in rows(SOURCE/'raw/respiratory_chartevents.csv.gz'):
        if r['itemid'] not in ITEMS or r['selection_role']=='state_record':continue
        units[r['itemid']+':'+r['valueuom']]+=1
        if r['hadm_id'] not in by_hadm:audit['retained_only_raw_ineligible']+=1;continue
        row=cohort[r['hadm_id']];stay=stays[r['stay_id']]
        if row['subject_id']!=r['subject_id'] or stay['hadm_id']!=r['hadm_id'] or stay['subject_id']!=r['subject_id']:
            raise ValueError('subject/hospital/ICU mismatch')
        start,end=datetime.fromisoformat(row['admittime']),datetime.fromisoformat(row['dischtime'])
        t,s=datetime.fromisoformat(r['charttime']),datetime.fromisoformat(r['storetime'])
        if not start<=t<=s<=end:raise ValueError('unavailable observation after discharge')
        location=icu_contains(stay,t)
        if location is None:audit['unknown_icu_interval_but_hospital_time_available']+=1
        elif not location:audit['outside_icu_interval_but_within_hospital']+=1
        value=canonical_resp(r['itemid'],r['valuenum'],r['valueuom'])
        if value is None:
            audit['rejected:'+r['itemid']]+=1
            for role in r['selection_role'].split('|'):roles['rejected:'+r['itemid']+':'+role]+=1
            continue
        key=tuple(r[k] for k in ('hadm_id','stay_id','charttime','storetime','itemid','valuenum','valueuom'))
        if key in seen:audit['duplicate_source_rows']+=1;continue
        seen.add(key);obs[r['hadm_id'],r['itemid']].append((t,value,r['stay_id']));stays_per[r['hadm_id']].add(r['stay_id'])
        audit['accepted:'+r['itemid']]+=1
        if r['warning'] not in ('',r'\N','0','f'):audit['warning_retained:'+r['itemid']]+=1
    n=len(hadms);raw=np.zeros((n,4,9));masks=np.zeros_like(raw,bool);flags=np.zeros((n,4,3))
    paired=np.zeros((n,6));paired_mask=np.zeros_like(paired,bool);spans=defaultdict(list);ages=defaultdict(list);first_offsets=defaultdict(list)
    for h,i in by_hadm.items():
        r=cohort[h];start,end=datetime.fromisoformat(r['admittime']),datetime.fromisoformat(r['dischtime'])
        for j,item in enumerate(ITEMS):
            raw[i,j],masks[i,j],flags[i,j]=summarize(obs[h,item],start,end,low=item=='220277')
            if splits[i]=='train' and flags[i,j,0]:
                age=np.expm1(raw[i,j,6]);ages[item].append(age);spans[item].append(np.expm1(raw[i,j,5]));first_offsets[item].append(np.expm1(raw[i,j,8]))
        paired[i],paired_mask[i]=support_pairs(obs[h,'220277'],obs[h,'223834'],obs[h,'223835'])
    scaled,params=normalize(raw,masks,splits=='train');pair_scaled,pair_params=normalize(paired,paired_mask,splits=='train')
    feature=np.concatenate((np.concatenate((scaled,masks,flags),axis=2).reshape(n,84),pair_scaled,paired_mask),axis=1).astype(np.float32)
    availability=np.zeros_like(feature);timing=np.zeros_like(feature)
    for j in range(4):
        availability[:,j*21+18:j*21+20]=flags[:,j,:2]
        for col in (5,6,8):timing[:,j*21+col]=scaled[:,j,col];timing[:,j*21+9+col]=masks[:,j,col]
    availability[:,90:]=paired_mask;timing+=availability
    arr=dict(hadm_ids=hadms,subject_ids=people,patient_splits=splits,features=feature,availability=availability,timing=timing,
        observed=flags[:,:,0].any(1),numeric_raw=raw,numeric_masks=masks,numeric_flags=flags,support_pairs_raw=paired,support_pairs_masks=paired_mask)
    def quantile(v):return np.quantile(v,[0,.1,.5,.9,1]).tolist() if v else None
    summary={name:dict(training_observed=int(flags[splits=='train',j,0].sum()),training_retest=int(flags[splits=='train',j,1].sum()),
        latest_age_days_quantiles=quantile(ages[item]),selected_span_days_quantiles=quantile(spans[item]),first_admission_offset_days_quantiles=quantile(first_offsets[item]),
        training_latest_age_over2days=int(sum(v>2 for v in ages[item]))) for j,(item,name) in enumerate(zip(ITEMS,NAMES))}
    report=dict(source_manifest_sha256=sha256(SOURCE/'metadata/manifest.json'),parent_state_manifest_sha256=sha256(STATE/'manifest.json'),
        source_sha256={name:sm['sources'][name]['sha256'] for name in ('respiratory_chartevents','icustays','d_items')},
        normalization=params,pair_normalization=pair_params,audit=dict(audit),unit_counts=dict(units),rejected_selection_roles=dict(roles),
        coverage=summary,training_any_source=int(arr['observed'][splits=='train'].sum()),
        pairing_training_counts={name:int(paired_mask[splits=='train',j].sum()) for j,name in enumerate(PAIR_NAMES)},
        multiple_icu_stays_with_accepted_values=sum(len(v)>1 for v in stays_per.values()),
        names=list(NAMES),numeric_channels=list(CHANNELS),flags=list(FLAGS),pair_names=list(PAIR_NAMES),features=96,
        limitations=['selected anchors only; selection preceded cleaning','late ICU values are not discharge state',
            'latest_at_extreme does not mean pathologically abnormal','oxygen support missing is not room air',
            'paired support within60min does not establish causality or true recovery','availability can reflect measurement practice'])
    return arr,report


def build(*,finish_metadata=False):
    if ROOT.exists() and not finish_metadata:raise FileExistsError('derived directory exists; inspect')
    if finish_metadata and (not ROOT.exists() or {p.name for p in ROOT.iterdir()}!={'respiratory_transition.npz'}):
        raise ValueError('metadata recovery only permitted for exact preserved NPZ-only scene')
    arr,report=derive()
    if not finish_metadata:
        ROOT.mkdir(parents=True)
        np.savez_compressed(ROOT/'respiratory_transition.npz',**arr)
    else:
        with np.load(ROOT/'respiratory_transition.npz') as z:
            for key,v in arr.items():np.testing.assert_array_equal(z[key],v)
    report.update(status='derived_roundtrip_pending_independent_source_rebuild',output_sha256=sha256(ROOT/'respiratory_transition.npz'),
        protocol_sha256=sha256(PROTOCOL),code_sha256={p:sha256(Path(p)) for p in ('patient_data/respiratory_transition.py',
            'patient_data/resp_micro_context.py','training/v5/numeric_preprocessing.py','training/tests/test_v6_respiratory_transition.py')})
    with np.load(ROOT/'respiratory_transition.npz') as z:
        for key,v in arr.items():np.testing.assert_array_equal(z[key],v)
    write_json(ROOT/'manifest.json',report);print(json.dumps(dict(status=report['status'],coverage=report['coverage'],pairing=report['pairing_training_counts'],audit=report['audit'])),flush=True)


def verify():
    arr,report=derive();saved=json.loads((ROOT/'manifest.json').read_text())
    for key,v in report.items():
        if saved[key]!=v:raise ValueError('metadata/source reconstruction differs: '+key)
    for p,digest in saved['code_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('derivation source differs')
    if sha256(PROTOCOL)!=saved['protocol_sha256'] or sha256(ROOT/'respiratory_transition.npz')!=saved['output_sha256']:
        raise ValueError('derived/protocol hash differs')
    with np.load(ROOT/'respiratory_transition.npz') as z:
        for key,v in arr.items():np.testing.assert_array_equal(z[key],v)
    proof=dict(status='independent_source_rebuild_verified',maximum_array_error=0.,arrays=len(arr),output_sha256=saved['output_sha256'])
    write_json(ROOT/'independent_verification.json',proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify','finish-metadata'),required=True);a=p.parse_args()
    verify() if a.mode=='verify' else build(finish_metadata=a.mode=='finish-metadata')
