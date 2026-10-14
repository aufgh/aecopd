"""五核心检验原值与既有异常摘要：训练折无结局审计，不改变模型输入。"""
from collections import Counter,defaultdict
from datetime import datetime
import argparse
import gzip
import json
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.preprocess_events import LABS
from patient_data.lab_recovery_features import AXIS_CONCEPT,abnormal_severity
from patient_data.v6_lab_state import summarize

ROOT=Path('training_runs/v6_core_lab_value_audit_2026-10-14')
PROTOCOL=Path('docs/V6_CORE_LAB_VALUE_AUDIT_PROTOCOL_2026-10-14.md')
SNAP=Path('patient_data/mimiciv_copd/snapshot_2026-09-27')
STATE=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2')
EVENTS=Path('patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29')
AXES=('arterial_low_ph','arterial_high_pco2','hemoglobin_low','albumin_low','creatinine_high')

def accepted(row,boundary,unit,arterial_ids=None):
    try:
        chart,stored=(datetime.fromisoformat(row[k]) for k in ('charttime','storetime'))
        values=tuple(float(row[k]) for k in ('valuenum','ref_range_lower','ref_range_upper'))
        value,lower,upper=values
        if not boundary[0]<=chart<=stored<=boundary[1]:return None,'not_hospital_visible'
        if not all(np.isfinite(values)) or value<=0 or lower>=upper:return None,'invalid_numeric_reference'
    except (ValueError,TypeError,KeyError):return None,'invalid_time_numeric_reference'
    if row['valueuom'].strip()!=unit:return None,'unit_mismatch'
    if arterial_ids is not None and row['labevent_id'] not in arterial_ids:return None,'not_old_verified_arterial'
    return (chart,value,lower,upper),None

def readout(samples,end):
    """原异常摘要与物理值首末并列；不把引用/同刻行数当独立复测。"""
    if not samples:return np.zeros(9),np.zeros(2),np.zeros(2),dict(observed=False)
    grouped=defaultdict(list)
    for day,value,lower,upper,severity in samples:grouped[day].append((value,lower,upper,severity))
    ordered=sorted(grouped);physical=[];positions=[]
    for day in (ordered[0],ordered[-1]):
        physical.append(float(np.median([x[0] for x in grouped[day]])))
        positions.append(float(np.median([(x[0]-x[1])/(x[2]-x[1]) for x in grouped[day]])))
    nine,_=summarize([(x[0],x[4]) for x in samples],end)
    return nine,np.asarray(physical),np.asarray(positions),dict(observed=True,
        timestamps=len(ordered),raw_rows=len(samples),same_time_multi_rows=sum(len(v)>1 for v in grouped.values()),
        entirely_zero_direction=all(x[4]==0 for x in samples),
        original_value_varies=len({x[1] for x in samples})>1,
        first_last_value_diff=physical[0]!=physical[1],latest_age=end-ordered[-1])

def derive():
    start=time.perf_counter();sm=json.loads((SNAP/'metadata/manifest.json').read_text())
    st=json.loads((STATE/'manifest.json').read_text());em=json.loads((EVENTS/'manifest.json').read_text())
    if sm['status']!='validated' or em['status']!='preprocessed_and_checked':raise ValueError('unvalidated source')
    if sha256(EVENTS/'manifest.json')!=st['processed_manifest_sha256'] or sha256(EVENTS/'events.csv.gz')!=st['source_hashes']['processed_events']:
        raise ValueError('arterial event source differs')
    if sha256(STATE/'clinical_state.npz')!=st['output_sha256']['clinical_state.npz']:raise ValueError('state differs')
    sources={}
    for name in ('cohort/admissions','raw/labevents'):
        p=SNAP/(name+'.csv.gz');h=sha256(p)
        if h!=sm['sources'][name]['sha256'] or h!=st['source_hashes'][name]:raise ValueError('source differs')
        sources[str(p)]=h
    for p in (Path(__file__),PROTOCOL,Path('training/tests/test_v6_core_lab_value_audit.py'),
              SNAP/'metadata/manifest.json',STATE/'manifest.json',STATE/'clinical_state.npz',
              EVENTS/'manifest.json',EVENTS/'events.csv.gz',Path('patient_data/preprocess_events.py'),
              Path('patient_data/lab_recovery_features.py'),Path('patient_data/v6_lab_state.py')):sources[str(p)]=sha256(p)
    with np.load(STATE/'clinical_state.npz') as z:
        state={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits','axis_names','lab_axis_raw')}
    indices=np.flatnonzero(state['patient_splits']=='train');hadms=state['hadm_ids'][indices];subjects=state['subject_ids'][indices]
    lookup={str(h):i for i,h in enumerate(hadms)};bounds={}
    for row in rows(SNAP/'cohort/admissions.csv.gz'):
        if row['hadm_id'] not in lookup:continue
        i=lookup[row['hadm_id']]
        if int(row['subject_id'])!=subjects[i] or row['discharge_landmark_eligible']!='t':raise ValueError('training membership differs')
        bounds[row['hadm_id']]=tuple(datetime.fromisoformat(row[k]) for k in ('admittime','dischtime'))
    if len(bounds)!=13592:raise ValueError('training cohort differs')
    arterial=set()
    for row in rows(EVENTS/'events.csv.gz'):
        if row['hadm_id'] in lookup and row['kind']=='lab' and row['concept'] in ('ph:arterial','pco2:arterial'):
            arterial.add(row['source_id'])
    item_axes={}
    for j,axis in enumerate(AXES):
        concept=AXIS_CONCEPT[axis].split(':')[0]
        item,unit=next((item,unit) for item,(c,unit) in LABS.items() if c==concept)
        item_axes[item]=(j,axis,unit)
    observations=defaultdict(list);evidence=[];reasons=defaultdict(Counter);total=0
    for row in rows(SNAP/'raw/labevents.csv.gz'):
        total+=1;h=row['hadm_id'];item=row['itemid']
        if h not in lookup or item not in item_axes:continue
        j,axis,unit=item_axes[item]
        value,reason=accepted(row,bounds[h],unit,arterial if axis.startswith('arterial_') else None)
        reasons[axis]['raw_rows']+=1
        if reason:reasons[axis][reason]+=1;continue
        chart,physical,lower,upper=value;day=(chart-bounds[h][0]).total_seconds()/86400
        severity=abnormal_severity(axis,physical,lower,upper,severity_cap=None)
        observations[h,j].append((day,physical,lower,upper,severity));reasons[axis]['accepted_rows']+=1
        evidence.append(dict(hadm_id=h,labevent_id=row['labevent_id'],specimen_id=row['specimen_id'],
            axis=axis,charttime=row['charttime'],storetime=row['storetime'],unit=unit,
            value=physical,lower=lower,upper=upper,severity=severity))
    if total!=sm['sources']['raw/labevents']['rows']:raise ValueError('lab count differs')
    nine=np.zeros((len(indices),5,9));physical=np.zeros((len(indices),5,2));position=np.zeros_like(physical)
    observed=np.zeros((len(indices),5),bool);stats={};max_error=0.
    for j,axis in enumerate(AXES):
        detail=[];signatures=defaultdict(list);old_j=list(state['axis_names']).index(axis)
        for i,h in enumerate(hadms):
            key=str(h);length=(bounds[key][1]-bounds[key][0]).total_seconds()/86400
            nine[i,j],physical[i,j],position[i,j],info=readout(observations.get((key,j),[]),length)
            observed[i,j]=info['observed']
            if info['observed']:detail.append(info);signatures[tuple(nine[i,j])].append(i)
        error=float(np.abs(nine[:,j]-state['lab_axis_raw'][indices,old_j]).max());max_error=max(max_error,error)
        if error>1e-12:raise ValueError('old axis replay differs: '+axis)
        shared=[ix for ix in signatures.values() if len(set(subjects[ix]))>1 and len(set(physical[ix,j,1]))>1]
        ages=np.asarray([x['latest_age'] for x in detail]);use=observed[:,j]
        stats[axis]=dict(coverage=int(use.sum()),patients=len(np.unique(subjects[use])),
            repeated_timestamps_admissions=sum(x['timestamps']>=2 for x in detail),
            all_zero_direction_admissions=sum(x['entirely_zero_direction'] for x in detail),
            all_zero_with_original_value_variation=sum(x['entirely_zero_direction'] and x['original_value_varies'] for x in detail),
            first_last_physical_difference=sum(x['first_last_value_diff'] for x in detail),
            same_time_multirow_admissions=sum(x['same_time_multi_rows']>0 for x in detail),
            cross_patient_exact_nine_groups_with_different_latest_value=len(shared),
            discharges_in_those_groups=sum(len(ix) for ix in shared),
            latest_age_median_days=float(np.median(ages)) if len(ages) else None,
            latest_physical_quantiles=np.quantile(physical[use,j,1],[.1,.5,.9]).tolist() if use.any() else [],
            original_axis_replay_error=error)
    arrays=dict(hadm_ids=hadms,subject_ids=subjects,axis_names=np.asarray(AXES),
        original_nine=nine,physical_first_latest=physical,reference_position_first_latest=position,observed=observed)
    report=dict(status='train_only_core_lab_value_audit_not_model_interface',source_sha256=sources,
        training_admissions=len(indices),training_patients=len(np.unique(subjects)),
        assigned_lab_source_rows_scanned=total,axis_statistics=stats,acceptance_audit={k:dict(v) for k,v in reasons.items()},
        maximum_original_axis_replay_error=max_error,new_fits=0,new_vectors=0,no_outcomes_read=True,
        actual_seconds=time.perf_counter()-start,
        limits=['zero direction is not globally normal clinical state',
                'same one-axis signature does not imply same full model patient input',
                'same-time physical median is an audit readout, all source rows preserved',
                'latest observed is not necessarily discharge measurement; no prognostic gain evaluated'])
    return report,arrays,evidence

def run():
    if ROOT.exists():raise FileExistsError('audit exists')
    ROOT.mkdir()
    try:
        r,a,e=derive();np.savez_compressed(ROOT/'training_audit.npz',**a)
        with gzip.open(ROOT/'local_evidence.jsonl.gz','wt',encoding='utf-8') as f:
            for row in e:f.write(json.dumps(row)+'\n')
        r['output_sha256']={p:sha256(ROOT/p) for p in ('training_audit.npz','local_evidence.jsonl.gz')}
        (ROOT/'audit.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (ROOT/'state.json').write_text('{"status":"audit_pending_independent_replay","new_fits":0}\n')
        print(json.dumps(dict(status=r['status'],stats=r['axis_statistics'],max_error=r['maximum_original_axis_replay_error'],seconds=r['actual_seconds'])),flush=True)
    except Exception as error:
        (ROOT/'failure.json').write_text(json.dumps(dict(error=repr(error)))+'\n');raise

def verify():
    if (ROOT/'verification.json').exists():raise FileExistsError('already independently verified')
    old=json.loads((ROOT/'audit.json').read_text());new,a,e=derive()
    for k,h in old['output_sha256'].items():
        if sha256(ROOT/k)!=h:raise ValueError('saved output changed')
    seconds=new.pop('actual_seconds');old.pop('actual_seconds');old.pop('output_sha256')
    if old!=new:raise ValueError('source/statistics differ')
    with np.load(ROOT/'training_audit.npz') as z:
        for k,v in a.items():np.testing.assert_array_equal(z[k],v)
    with gzip.open(ROOT/'local_evidence.jsonl.gz','rt',encoding='utf-8') as f:
        if [json.loads(line) for line in f]!=e:raise ValueError('source rows differ')
    proof=dict(status='one_independent_full_source_array_statistic_replay',maximum_error=0.,
        original_axis_replay_error=new['maximum_original_axis_replay_error'],actual_rebuild_seconds=seconds,
        audit_sha256=sha256(ROOT/'audit.json'),new_fits=0)
    (ROOT/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    (ROOT/'state.json').write_text('{"status":"core_value_audit_independently_verified","new_fits":0,"goal_achieved":false}\n')
    print(json.dumps(proof),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('run','verify'),required=True)
    run() if p.parse_args().mode=='run' else verify()
