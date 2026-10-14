"""训练折同标本血气信息审计；复用父接受行，不读取原raw检验或结局。"""
from collections import Counter, defaultdict
from datetime import datetime
import argparse
import gzip
import json
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from patient_data.lab_recovery_features import abnormal_severity, PAIRED_AXIS
from patient_data.v6_lab_state import summarize
from patient_data.preprocess_events import LABS

ROOT=Path('training_runs/v6_same_specimen_audit_2026-10-14_r2')
PARENT=Path('training_runs/v6_core_lab_value_audit_2026-10-14')
STATE=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2')
EVENTS=Path('patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29')
SNAP=Path('patient_data/mimiciv_copd/snapshot_2026-09-27')
PROTOCOL=Path('docs/V6_SAME_SPECIMEN_AUDIT_PROTOCOL_2026-10-14.md')
AXES=('arterial_low_ph','arterial_high_pco2')
CONCEPTS=('ph:arterial','pco2:arterial')
# 复用真实既定字典，pH在本源明确使用units，不以无量纲推断为空。
UNITS=(LABS['50820'][1], LABS['50818'][1])


def bind(evidence, event, bounds, subject):
    """精确来源绑定，父接受规则与标本/单位/时间边界不可放宽。"""
    j=AXES.index(evidence['axis']);start,end=bounds
    if (event['kind']!='lab' or event['concept']!=CONCEPTS[j] or event['specimen_type']!='arterial'
            or event['patient_split']!='train' or event['subject_id']!=str(subject)
            or event['hadm_id']!=evidence['hadm_id'] or event['source_id']!=evidence['labevent_id']
            or event['specimen_id']!=evidence['specimen_id']
            or event['value_unit']!=UNITS[j] or evidence['unit']!=UNITS[j]):
        raise ValueError('source/specimen/type/unit mismatch')
    ec,es=(datetime.fromisoformat(event[k]) for k in ('charttime','storetime'))
    pc,ps=(datetime.fromisoformat(evidence[k]) for k in ('charttime','storetime'))
    value,lo,hi=(evidence[k] for k in ('value','lower','upper'))
    if (ec!=pc or es!=ps or not start<=pc<=ps<=end
            or float(event['value_raw'])!=value or not all(np.isfinite((value,lo,hi)))
            or value<=0 or not lo<hi):raise ValueError('numeric/time mismatch')
    day=(pc-start).total_seconds()/86400
    if abs(day-float(event['elapsed_days']))>1e-12:raise ValueError('elapsed time differs')
    severity=abnormal_severity(AXES[j],value,lo,hi,severity_cap=None)
    if abs(severity-evidence['severity'])>1e-12:raise ValueError('parent severity differs')
    return dict(evidence,day=day,subject_id=str(subject),concept=CONCEPTS[j],specimen_type='arterial')


def pair_slot(records):
    """同住院同标本集合；重复不平均，标本缺失不按时间补归属。"""
    if any(r['specimen_id'] in ('','\\N') for r in records):return None,'missing_specimen'
    if len({(r['hadm_id'],r['specimen_id']) for r in records})!=1:return None,'different_admission_or_specimen'
    slots={a:[r for r in records if r['axis']==a] for a in AXES}
    if any(len(v)>1 for v in slots.values()):return None,'duplicate_component'
    if any(len(v)==0 for v in slots.values()):return None,'missing_component'
    p,c=(slots[a][0] for a in AXES)
    if abs((datetime.fromisoformat(p['charttime'])-datetime.fromisoformat(c['charttime'])).total_seconds())>3600:
        return None,'chart_gap_over_one_hour'
    return dict(hadm_id=p['hadm_id'],specimen_id=p['specimen_id'],subject_id=p['subject_id'],
                values=[p['value'],c['value']],days=[p['day'],c['day']],
                day=max(p['day'],c['day']),severity=min(p['severity'],c['severity']),
                components=[p,c]),'paired'


def latest_sets(pairs,hadms):
    by=defaultdict(list)
    for p in pairs:by[p['hadm_id']].append(p)
    out=[]
    for h in hadms:
        pp=by[str(h)]
        if not pp:out.append(());continue
        last=max(p['day'] for p in pp)
        out.append(tuple(sorted({tuple(p['values']) for p in pp if p['day']==last})))
    return out


def collisions(raw,physical,subjects):
    groups=defaultdict(list)
    for i,v in enumerate(physical):
        if v:groups[tuple(raw[i].reshape(-1))].append(i)
    shared=[ix for ix in groups.values() if len({int(subjects[i]) for i in ix})>1
            and len({physical[i] for i in ix})>1]
    return dict(groups=len(shared),admissions=sum(len(ix) for ix in shared),
                patients=len({int(subjects[i]) for ix in shared for i in ix})),shared


def derive():
    started=time.perf_counter();pa=json.loads((PARENT/'audit.json').read_text())
    proof=json.loads((PARENT/'verification.json').read_text());sm=json.loads((STATE/'manifest.json').read_text())
    em=json.loads((EVENTS/'manifest.json').read_text())
    if (proof['maximum_error']!=0 or proof['audit_sha256']!=sha256(PARENT/'audit.json')
            or em['status']!='preprocessed_and_checked' or sm['processed_manifest_sha256']!=sha256(EVENTS/'manifest.json')):
        raise ValueError('parent lineage proof differs')
    for name,h in pa['output_sha256'].items():
        if sha256(PARENT/name)!=h:raise ValueError('parent accepted evidence changed')
    for p in ('patient_data/preprocess_events.py','patient_data/lab_recovery_features.py','patient_data/v6_lab_state.py'):
        if sha256(Path(p))!=pa['source_sha256'][p]:raise ValueError('inherited acceptance code changed')
    if sha256(STATE/'clinical_state.npz')!=sm['output_sha256']['clinical_state.npz']:raise ValueError('state changed')
    if sha256(EVENTS/'events.csv.gz')!=em['output_sha256']['events.csv.gz'] or em['output_sha256']['events.csv.gz']!=sm['source_hashes']['processed_events']:
        raise ValueError('accepted events changed')
    admission_path=SNAP/'cohort/admissions.csv.gz'
    if sha256(admission_path)!=sm['source_hashes']['cohort/admissions']:raise ValueError('cohort changed')
    with np.load(STATE/'clinical_state.npz',allow_pickle=False) as z:
        state={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits','axis_names','lab_axis_raw')}
    if len(state['hadm_ids'])!=19401 or len(np.unique(state['subject_ids']))!=5351:raise ValueError('cohort drift')
    ti=np.flatnonzero(state['patient_splits']=='train');hadms=state['hadm_ids'][ti];subjects=state['subject_ids'][ti]
    if len(ti)!=13592:raise ValueError('train drift')
    lookup={str(h):i for i,h in enumerate(hadms)};bounds={}
    for r in rows(admission_path):
        h=r['hadm_id']
        if h not in lookup:continue
        if r['subject_id']!=str(subjects[lookup[h]]) or r['discharge_landmark_eligible']!='t':raise ValueError('train membership differs')
        bounds[h]=tuple(datetime.fromisoformat(r[k]) for k in ('admittime','dischtime'))
    if len(bounds)!=len(ti):raise ValueError('missing boundaries')
    selected={}
    with gzip.open(PARENT/'local_evidence.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['axis'] not in AXES:continue
            if r['hadm_id'] not in lookup or r['labevent_id'] in selected:raise ValueError('parent accepted membership duplicate')
            selected[r['labevent_id']]=r
    slots=defaultdict(list);bound=[];bicarbonate=[];counts=Counter();seen=set()
    for r in rows(EVENTS/'events.csv.gz'):
        h=r['hadm_id']
        if h not in lookup or r['kind']!='lab':continue
        if r['concept']=='bicarbonate':bicarbonate.append(r);continue
        if r['concept'] not in CONCEPTS:continue
        counts['processed_train_arterial_rows']+=1
        if r['source_id'] not in selected:counts['not_parent_reference_validated']+=1;continue
        if r['source_id'] in seen:raise ValueError('accepted source duplicate')
        seen.add(r['source_id']);v=bind(selected[r['source_id']],r,bounds[h],subjects[lookup[h]])
        bound.append(v)
        if v['specimen_id'] in ('','\\N'):counts['accepted_missing_specimen_rows']+=1
        else:slots[h,v['specimen_id']].append(v)
    if seen!=set(selected):raise ValueError('parent gas source not bound completely')
    pairs=[];rejected=[];pc=Counter()
    for key,rr in sorted(slots.items()):
        p,why=pair_slot(rr);pc[why]+=1
        if p:pairs.append(p)
        else:rejected.append(dict(hadm_id=key[0],specimen_id=key[1],reason=why,components=rr))
    # 重建三个既有9通道，不从配对中把单轴缺测补成正常。
    samples=defaultdict(list)
    for r in bound:samples[r['hadm_id'],r['axis']].append((r['day'],r['severity']))
    for p in pairs:samples[p['hadm_id'],PAIRED_AXIS].append((p['day'],p['severity']))
    axis_names=(*AXES,PAIRED_AXIS);raw=np.zeros((len(ti),3,9));errors={}
    for j,axis in enumerate(axis_names):
        for i,h in enumerate(hadms):
            duration=(bounds[str(h)][1]-bounds[str(h)][0]).total_seconds()/86400
            raw[i,j],_=summarize(samples.get((str(h),axis),[]),duration)
        old=list(state['axis_names']).index(axis)
        error=float(np.abs(raw[:,j]-state['lab_axis_raw'][ti,old]).max());errors[axis]=error
        if error>1e-12:raise ValueError('existing nine-channel replay differs: '+axis)
    physical=latest_sets(pairs,hadms);joint_coll,jc=collisions(raw[:,2],physical,subjects)
    triple_coll,tc=collisions(raw,physical,subjects)
    groups=defaultdict(set)
    for p in pairs:groups[p['severity']].add(tuple(p['values']))
    charts=defaultdict(set);zero=defaultdict(set)
    for p in pairs:
        charts[p['hadm_id']].add(p['day'])
        if p['severity']==0:zero[p['hadm_id']].add(tuple(p['values']))
    pair_keys={(p['hadm_id'],p['specimen_id']) for p in pairs};bc=Counter()
    for r in bicarbonate:
        if r['specimen_id'] in ('','\\N'):bc['missing_specimen_rows']+=1;continue
        if (r['hadm_id'],r['specimen_id']) in pair_keys:
            bc['exact_same_specimen_rows']+=1
            bc['unit:'+r['value_unit']]+=1;bc['role:'+r['specimen_type']]+=1
    valid=raw[:,2,6]>0
    sourcepaths=(Path(__file__),PROTOCOL,Path('training/tests/test_same_specimen_audit.py'),
        PARENT/'audit.json',PARENT/'verification.json',PARENT/'local_evidence.jsonl.gz',PARENT/'training_audit.npz',
        STATE/'manifest.json',STATE/'clinical_state.npz',EVENTS/'manifest.json',EVENTS/'events.csv.gz',admission_path,
        Path('patient_data/v6_lab_state.py'),Path('patient_data/lab_recovery_features.py'),Path('patient_data/preprocess_events.py'))
    sources={str(p):sha256(p) for p in sourcepaths}
    report=dict(status='training_only_same_specimen_information_audit',source_sha256=sources,
        inherited_raw_lab_sha256=sm['source_hashes']['raw/labevents'],raw_lab_reparsed=False,
        training_admissions=len(ti),training_patients=len(np.unique(subjects)),gas_row_audit=dict(counts),
        accepted_parent_gas_rows=len(bound),specimen_audit=dict(pc),pairs=len(pairs),
        paired_admissions=int(valid.sum()),paired_patients=len(np.unique(subjects[valid])),
        admissions_with_multiple_pair_times=sum(len(v)>1 for v in charts.values()),
        same_pair_time_multiple_specimen_admissions=sum(len([p for p in pairs if p['hadm_id']==h])>len(v) for h,v in charts.items()),
        old_min_zero_pairs=sum(p['severity']==0 for p in pairs),
        old_min_zero_admissions=len(zero),zero_min_multiple_physical_pairs_admissions=sum(len(v)>1 for v in zero.values()),
        exact_min_groups_with_distinct_physical_pairs=sum(len(v)>1 for v in groups.values()),
        same_joint_nine_different_latest_physical_sets=joint_coll,
        same_three_axes_27_different_latest_physical_sets=triple_coll,
        original_axis_replay_errors=errors,bicarbonate_audit=dict(bc),
        bicarbonate_processed_train_rows=len(bicarbonate),bicarbonate_not_arterial_by_mapping=True,
        latest_pair_age_median_days=float(np.median(np.expm1(raw[valid,2,4]))) if valid.any() else None,
        latest_pair_multi_value_set_admissions=sum(len(v)>1 for v in physical),
        new_fits=0,new_vectors=0,no_outcomes_read=True,actual_seconds=time.perf_counter()-started,
        limitations=['exact signatures are not complete patient inputs or outcome information',
            'zero min is not globally normal; no new disease labels or effect estimate',
            'same specimen components remain distinct actual chart/store times',
            'latest sets are not confirmed discharge-day status; no physical pair averaging'])
    arrays=dict(hadm_ids=hadms,subject_ids=subjects,old_three_axis_raw=raw,
                pair_admission_indices=np.asarray([lookup[p['hadm_id']] for p in pairs],np.int64),
                pair_values=np.asarray([p['values'] for p in pairs],np.float64).reshape(-1,2),
                pair_chart_days=np.asarray([p['days'] for p in pairs],np.float64).reshape(-1,2),
                pair_joint_severity=np.asarray([p['severity'] for p in pairs],np.float64))
    evidence=dict(pairs=pairs,rejected=rejected,collision_indices=dict(joint=jc,three_axes=tc))
    return report,arrays,evidence


def run():
    if ROOT.exists():raise FileExistsError('audit exists; do not repeat')
    ROOT.mkdir()
    report,a,e=derive();np.savez_compressed(ROOT/'audit_arrays.npz',**a)
    with gzip.open(ROOT/'local_evidence.json.gz','wt',encoding='utf-8') as f:json.dump(e,f)
    report['output_sha256']={k:sha256(ROOT/k) for k in ('audit_arrays.npz','local_evidence.json.gz')}
    write(ROOT/'audit.json',report);write(ROOT/'state.json',dict(status='pending_independent_rebuild',new_fits=0))
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','output_sha256')}),flush=True)


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def verify():
    if (ROOT/'verification.json').exists():raise FileExistsError('already verified')
    old=json.loads((ROOT/'audit.json').read_text());new,a,e=derive();seconds=new.pop('actual_seconds');old.pop('actual_seconds')
    for k,h in old.pop('output_sha256').items():
        if sha256(ROOT/k)!=h:raise ValueError('saved arrays/evidence changed')
    if old!=new:raise ValueError('statistics/source differ')
    with np.load(ROOT/'audit_arrays.npz') as z:
        for k,v in a.items():np.testing.assert_array_equal(z[k],v)
    with gzip.open(ROOT/'local_evidence.json.gz','rt',encoding='utf-8') as f:saved=json.load(f)
    # JSON converts tuples in collision lists, but all persisted evidence uses lists.
    if saved!=e:raise ValueError('local evidence differs')
    write(ROOT/'verification.json',dict(status='one_independent_accepted_source_parent_row_axis_array_statistic_rebuild',
        maximum_error=0.,audit_sha256=sha256(ROOT/'audit.json'),actual_seconds=seconds,
        raw_lab_reparsed=False,new_fits=0,new_vectors=0,no_outcomes_read=True))
    write(ROOT/'state.json',dict(status='training_audit_independently_verified',new_fits=0,goal_achieved=False))
    print(json.dumps(dict(error=0.,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('run','verify'),required=True);mode=p.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as error:
        if ROOT.exists():write(ROOT/f'failure_{mode}.json',dict(error=repr(error)))
        raise
