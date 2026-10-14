"""完整设备同刻集合/同stay真实转换；不推断氧浓度、撤机或因果恢复。"""
from collections import Counter,defaultdict
from datetime import datetime
import argparse
import json
import math
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.respiratory_transition import SNAP,SOURCE,STATE,icu_contains,normalize
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_oxygen_device_2026-10-13')
PROTOCOL=Path('docs/V6_OXYGEN_DEVICE_PROTOCOL_2026-10-13.md')
LABELS=('Endotracheal tube','Nasal cannula','Face tent','High flow neb','High flow nasal cannula',
        'Aerosol-cool','Bipap mask','Non-rebreather','Venti mask','Oxymizer','CPAP mask','None',
        'Tracheostomy tube','Medium conc mask','Trach mask','Other','T-piece','Ultrasonic neb')
LOOKUP={v.casefold():i for i,v in enumerate(LABELS)}
PROC=('224385','227194','225468','225477','225792','225794')
PROC_NAMES=('Intubation','Extubation','Unplanned Extubation (patient-initiated)',
            'Unplanned Extubation (non-patient initiated)','Invasive Ventilation','Non-invasive Ventilation')
TIMES=('log_recorded_time_count','log_first_admission_offset_days','log_latest_age_days',
       'log_first_latest_span_days','log_latest_identical_set_record_span_days','log_set_change_count')
AVAIL=[90,91,92,93,114]
TIMING=AVAIL+list(range(94,98))+list(range(100,104))
DIM=115


def label(value):
    if value.strip() in ('',r'\N'):return None
    key=value.strip().casefold()
    if key not in LOOKUP:raise ValueError('unknown device label: '+value)
    return LOOKUP[key]


def compress(records):
    """先集合后压缩；末时间保留。同一stay，同一时刻没有人工先后关系。"""
    grouped=defaultdict(set)
    for t,k in records:grouped[t].add(k)
    result=[]
    for t,values in sorted(grouped.items()):
        values=frozenset(values)
        if result and result[-1][2]==values:result[-1]=(result[-1][0],t,values)
        else:result.append((t,t,values))
    return result,grouped


def available(start,end,t,stored,stop=None):
    return start<=t<=stored<=end and (stop is None or t<=stop<=end)


def summarize(by_stay,start,end,procedures=()):
    hot=np.zeros((5,len(LABELS)));flags=np.zeros(4);raw=np.zeros(6);mask=np.zeros(6,bool);proc=np.zeros(9)
    sets=[];runs={};transitions=0;observed_times=set()
    for stay,records in by_stay.items():
        compressed,grouped=compress(records);runs[stay]=compressed
        sets.extend((t,stay,v) for t,v in grouped.items());observed_times.update(grouped)
        for _,values in grouped.items():hot[2,list(values)]=1
        for a,b in zip(compressed,compressed[1:]):
            hot[3,list(b[2]-a[2])]=1;hot[4,list(a[2]-b[2])]=1;transitions+=1
    if sets:
        first=min(t for t,_,_ in sets);last=max(t for t,_,_ in sets)
        for t,_,values in sets:
            if t==first:hot[0,list(values)]=1
            if t==last:hot[1,list(values)]=1
        # 同刻跨stay末记录不能任意选一个，用最短的已知末连续段跨度。
        tail=[(last-runs[s][-1][0]).total_seconds()/86400 for t,s,_ in sets if t==last]
        raw[:]=[math.log1p(len(observed_times)),math.log1p((first-start).total_seconds()/86400),
            math.log1p((end-last).total_seconds()/86400),math.log1p((last-first).total_seconds()/86400),
            math.log1p(min(tail)),math.log1p(transitions)]
        mask[:]=True;flags[:]=[1,len(observed_times)>1,any(len(v)>1 for _,_,v in sets),len(by_stay)>1]
        if not start<=first<=last<=end:raise ValueError('device time unavailable')
    for stay,item,a,b in procedures:
        proc[PROC.index(item)]=1;proc[8]=1
        if sets and item in ('225792','225794') and any(t==last and s==stay and a<=t<=b for t,s,_ in sets):
            proc[6+('225792','225794').index(item)]=1
    return hot,flags,raw,mask,proc


def derive():
    sm=json.loads((SOURCE/'metadata/manifest.json').read_text());parent=json.loads((SNAP/'metadata/manifest.json').read_text())
    state=json.loads((STATE/'manifest.json').read_text())
    if sm['status']!='validated_restricted_source_export_not_model_validated' or parent['status']!='validated':raise ValueError('source not validated')
    if sha256(SNAP/'metadata/manifest.json')!=sm['parent_snapshot_manifest_sha256'] or sha256(SNAP/'cohort/admissions.csv.gz')!=sm['parent_cohort_sha256']:
        raise ValueError('parent lineage differs')
    names=('respiratory_chartevents','icustays','d_items','respiratory_procedureevents')
    for name in names:
        if sha256(SOURCE/'raw'/f'{name}.csv.gz')!=sm['sources'][name]['sha256']:raise ValueError('source changed: '+name)
    if sha256(STATE/'clinical_state.npz')!=state['output_sha256']['clinical_state.npz']:raise ValueError('state changed')
    dictionary={r['itemid']:r for r in rows(SOURCE/'raw/d_items.csv.gz')}
    if dictionary['226732']['label']!='O2 Delivery Device(s)' or dictionary['226732']['linksto']!='chartevents':raise ValueError('device dictionary differs')
    for item,name in zip(PROC,PROC_NAMES):
        if dictionary[item]['label']!=name or dictionary[item]['linksto']!='procedureevents':raise ValueError('procedure dictionary differs')
    with np.load(STATE/'clinical_state.npz') as z:hadms=z['hadm_ids'];people=z['subject_ids'];splits=z['patient_splits']
    by_hadm={str(h):i for i,h in enumerate(hadms)};cohort={r['hadm_id']:r for r in rows(SNAP/'cohort/admissions.csv.gz')}
    stays={r['stay_id']:r for r in rows(SOURCE/'raw/icustays.csv.gz')}
    observations=defaultdict(lambda:defaultdict(list));procedures=defaultdict(list);audit=Counter();counts=Counter();seen=set()
    def bounds(r,t,stored,stop=None):
        h=r['hadm_id'];c=cohort[h];stay=stays[r['stay_id']]
        if c['subject_id']!=r['subject_id'] or stay['subject_id']!=r['subject_id'] or stay['hadm_id']!=h:raise ValueError('source subject/hospital/stay mismatch')
        start,end=datetime.fromisoformat(c['admittime']),datetime.fromisoformat(c['dischtime'])
        if not available(start,end,t,stored,stop):raise ValueError('source not available by discharge')
        contains=icu_contains(stay,t)
        if contains is None:audit['unknown_ICU_boundary_hospital_available']+=1
        elif not contains:audit['outside_ICU_boundary_hospital_available']+=1
    for r in rows(SOURCE/'raw/respiratory_chartevents.csv.gz'):
        if r['itemid']!='226732':continue
        if r['hadm_id'] not in by_hadm:audit['raw_ineligible_device_rows_retained']+=1;continue
        if r['selection_role']!='state_record':raise ValueError('device states not fully retained')
        t=datetime.fromisoformat(r['charttime']);stored=datetime.fromisoformat(r['storetime']);bounds(r,t,stored)
        k=label(r['value'])
        if k is None:audit['empty_device_label']+=1;continue
        key=(r['hadm_id'],r['stay_id'],t,stored,k)
        if key in seen:audit['duplicate_device_rows']+=1;continue
        seen.add(key);observations[r['hadm_id']][r['stay_id']].append((t,k));audit['accepted_device_rows']+=1
        if splits[by_hadm[r['hadm_id']]]=='train':counts[LABELS[k]]+=1
    seen=set()
    for r in rows(SOURCE/'raw/respiratory_procedureevents.csv.gz'):
        if r['itemid'] not in PROC:raise ValueError('unexpected procedure item')
        if r['hadm_id'] not in by_hadm:audit['raw_ineligible_procedure_rows_retained']+=1;continue
        if r['statusdescription'] not in ('FinishedRunning','Stopped'):audit['procedure_status_excluded:'+r['statusdescription']]+=1;continue
        a,b,s=(datetime.fromisoformat(r[k]) for k in ('starttime','endtime','storetime'))
        hospital=cohort[r['hadm_id']];start,end=(datetime.fromisoformat(hospital[k]) for k in ('admittime','dischtime'))
        if not available(start,end,a,s,b):audit['unavailable_or_malformed_procedure_excluded']+=1;continue
        bounds(r,a,s,b)
        key=(r['hadm_id'],r['stay_id'],r['itemid'],r['orderid'],a,b)
        if key in seen:audit['duplicate_procedures']+=1;continue
        seen.add(key);procedures[r['hadm_id']].append((r['stay_id'],r['itemid'],a,b));audit['accepted_procedures:'+r['itemid']]+=1
    n=len(hadms);hot=np.zeros((n,5,len(LABELS)));flags=np.zeros((n,4));raw=np.zeros((n,6));mask=np.zeros((n,6),bool);proc=np.zeros((n,9))
    compressed_counts=[];multi_times=0
    for h,i in by_hadm.items():
        r=cohort[h];a,b=(datetime.fromisoformat(r[k]) for k in ('admittime','dischtime'))
        hot[i],flags[i],raw[i],mask[i],proc[i]=summarize(observations[h],a,b,procedures[h])
        if splits[i]=='train':
            for records in observations[h].values():
                runs,grouped=compress(records);compressed_counts.append(len(runs));multi_times+=sum(len(v)>1 for v in grouped.values())
    scaled,parameters=normalize(raw,mask,splits=='train')
    features=np.concatenate((hot.reshape(n,5*len(LABELS)),flags,scaled,mask,proc),1).astype(np.float32)
    availability=np.zeros_like(features);availability[:,AVAIL]=features[:,AVAIL]
    timing=np.zeros_like(features);timing[:,TIMING]=features[:,TIMING]
    arr=dict(hadm_ids=hadms,subject_ids=people,patient_splits=splits,features=features,availability=availability,timing=timing,
        observed=flags[:,0].astype(bool),device_multihot=hot,device_flags=flags,time_raw=raw,time_masks=mask,procedure_evidence=proc)
    training=splits=='train';observed=training&arr['observed']
    report=dict(source_manifest_sha256=sha256(SOURCE/'metadata/manifest.json'),parent_state_manifest_sha256=sha256(STATE/'manifest.json'),
        source_sha256={name:sm['sources'][name]['sha256'] for name in names},dictionary={k:dictionary[k] for k in ('226732',*PROC)},
        normalization=parameters,audit=dict(audit),training_device_label_counts=dict(counts),labels=list(LABELS),time_channels=list(TIMES),features=DIM,
        training_observed=int(observed.sum()),training_procedure_observed=int(proc[training,8].sum()),
        training_multilabel_timestamps=int(multi_times),compressed_state_count_quantiles=np.quantile(compressed_counts,[0,.1,.5,.9,1]).tolist(),
        training_latest_age_days_quantiles=np.quantile(np.expm1(raw[observed,2]),[0,.1,.5,.9,1]).tolist(),
        training_latest_procedure_match_counts=proc[training,6:8].sum(0).astype(int).tolist(),
        training_ever_device_admissions=hot[training,2].sum(0).astype(int).tolist(),
        limitations=['device None not verified room air','multi-labels can be valid simultaneous devices',
            'only within-stay transitions','no interpolation or causal recovery','latest ICU record not discharge state',
            'absence of procedure evidence is not clinical non-use','new branch gated by device observation'])
    return arr,report


def build():
    if ROOT.exists():raise FileExistsError('new derived directory exists; inspect')
    arr,report=derive();json.dumps(report,allow_nan=False);ROOT.mkdir(parents=True)
    np.savez_compressed(ROOT/'oxygen_device.npz',**arr)
    report.update(status='derived_roundtrip_pending_source_rebuild',output_sha256=sha256(ROOT/'oxygen_device.npz'),protocol_sha256=sha256(PROTOCOL),
        code_sha256={p:sha256(Path(p)) for p in ('patient_data/oxygen_device.py','patient_data/respiratory_transition.py',
            'patient_data/resp_micro_context.py','training/v5/numeric_preprocessing.py','training/tests/test_v6_oxygen_device.py')})
    with np.load(ROOT/'oxygen_device.npz') as z:
        for k,v in arr.items():np.testing.assert_array_equal(v,z[k])
    write_json(ROOT/'manifest.json',report);print(json.dumps(report),flush=True)


def verify():
    arr,report=derive();m=json.loads((ROOT/'manifest.json').read_text())
    for k,v in report.items():
        if m[k]!=v:raise ValueError('source metadata differs: '+k)
    for p,h in m['code_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('source code differs')
    if sha256(PROTOCOL)!=m['protocol_sha256'] or sha256(ROOT/'oxygen_device.npz')!=m['output_sha256']:raise ValueError('protocol/data differs')
    with np.load(ROOT/'oxygen_device.npz') as z:
        for k,v in arr.items():np.testing.assert_array_equal(v,z[k])
    proof=dict(status='independent_source_rebuild_verified',maximum_array_error=0.,arrays=len(arr),output_sha256=m['output_sha256'])
    write_json(ROOT/'independent_verification.json',proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify'),required=True);a=p.parse_args()
    build() if a.mode=='build' else verify()
