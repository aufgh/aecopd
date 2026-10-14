"""无结局训练折同标本住院锚点审计；不自动填住院或构建模型输入。"""
from collections import Counter,defaultdict
from datetime import datetime
import gzip
import json
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import rows,sha256

SNAPSHOT=Path('patient_data/mimiciv_copd/snapshot_2026-09-27')
ROOT=Path('training_runs/v6_unassigned_lab_audit_2026-10-14')
PARENT=Path('training_runs/v6_admission_context_2026-10-14')
PROTOCOL=Path('docs/V6_UNASSIGNED_LAB_PROTOCOL_2026-10-14.md')
NULLS=('',r'\N',None)

def key(row):
    return None if row.get('subject_id') in NULLS or row.get('specimen_id') in NULLS else (row['subject_id'],row['specimen_id'])

def visible(row,boundary):
    try:
        chart,stored=(datetime.fromisoformat(row[k]) for k in ('charttime','storetime'))
        return boundary[0]<=chart<=stored<=boundary[1]
    except (ValueError,TypeError,KeyError):return False

def attribution(row,anchor,admissions,eligible,duplicate=False):
    if duplicate:return None,'duplicate_labevent_id'
    if key(row) is None:return None,'missing_patient_or_specimen'
    if not anchor:return None,'no_assigned_specimen_anchor'
    if len(anchor)!=1:return None,'multiple_hospital_anchors'
    h,info=next(iter(anchor.items()))
    if h not in eligible:return None,'anchor_not_eligible_training_admission'
    if not visible(row,admissions[h]):return None,'candidate_not_hospital_visible'
    if row['charttime'] not in info['visible_times']:return None,'no_visible_same_time_anchor'
    return h,'same_specimen_hospital_evidence'

def derive():
    started=time.perf_counter();sm_path=SNAPSHOT/'metadata/manifest.json';sm=json.loads(sm_path.read_text())
    if sm['status']!='validated':raise ValueError('snapshot not validated')
    manifest=json.loads((PARENT/'interface_manifest.json').read_text())
    proof=json.loads((PARENT/'verification.json').read_text())
    if (manifest['sha256']!=sha256(PARENT/'interface.npz') or proof['maximum_error']!=0
            or proof['screening_sha256']!=sha256(PARENT/'screening.json')):raise ValueError('parent proof changed')
    sources={str(sm_path):sha256(sm_path),str(PARENT/'interface.npz'):sha256(PARENT/'interface.npz')}
    for name in ('cohort/admissions','raw/labevents','raw/labevents_unassigned','raw/d_labitems'):
        p=SNAPSHOT/(name+'.csv.gz');h=sha256(p)
        if h!=sm['sources'][name]['sha256']:raise ValueError('source changed: '+name)
        sources[str(p)]=h
    for p in (Path(__file__),PROTOCOL,Path('training/tests/test_v6_unassigned_lab_audit.py')):sources[str(p)]=sha256(p)
    with np.load(PARENT/'interface.npz') as z:
        eligible={str(h):str(s) for h,s,f in zip(z['hadm_ids'],z['subject_ids'],z['patient_splits']) if f=='train'}
    if len(eligible)!=13592:raise ValueError('training membership differs')
    people=set(eligible.values());admissions={};owners={}
    for row in rows(SNAPSHOT/'cohort/admissions.csv.gz'):
        if row['subject_id'] not in people:continue
        h=row['hadm_id']
        if h in admissions:raise ValueError('duplicate cohort')
        admissions[h]=tuple(datetime.fromisoformat(row[k]) for k in ('admittime','dischtime'));owners[h]=row['subject_id']
        if h in eligible and (eligible[h]!=row['subject_id'] or row['discharge_landmark_eligible']!='t'):
            raise ValueError('training qualification differs')
    if not set(eligible)<=set(admissions):raise ValueError('training source missing')
    candidate=[];total=0;ids=Counter();items=Counter();specimens=set();hadm_present=0
    for row in rows(SNAPSHOT/'raw/labevents_unassigned.csv.gz'):
        total+=1
        if row['hadm_id'] not in NULLS:hadm_present+=1
        if row['subject_id'] not in people:continue
        candidate.append(row);ids[row['labevent_id']]+=1;items[row['itemid']]+=1
        if key(row) is not None:specimens.add(key(row))
    if total!=sm['sources']['raw/labevents_unassigned']['rows'] or hadm_present:
        raise ValueError('unassigned source scope differs')
    anchors=defaultdict(lambda:defaultdict(lambda:dict(visible_times=set(),ids=set())))
    raw_total=0;relevant=0;cross_ids=set();seen_assigned=Counter()
    for row in rows(SNAPSHOT/'raw/labevents.csv.gz'):
        raw_total+=1
        if row['labevent_id'] in ids:cross_ids.add(row['labevent_id'])
        k=key(row)
        if k not in specimens:continue
        relevant+=1;seen_assigned[row['labevent_id']]+=1;h=row['hadm_id']
        if h not in admissions or owners[h]!=row['subject_id']:raise ValueError('assigned anchor member conflict')
        slot=anchors[k][h];slot['ids'].add(row['labevent_id'])
        if visible(row,admissions[h]):slot['visible_times'].add(row['charttime'])
    if raw_total!=sm['sources']['raw/labevents']['rows']:raise ValueError('assigned source count differs')
    duplicate_anchor_ids={k for k,v in seen_assigned.items() if v>1}
    records=[];reasons=Counter();bound_items=Counter();bound_hadms=set()
    for row in candidate:
        anchor=anchors.get(key(row),{})
        bad_anchor=any(slot['ids']&duplicate_anchor_ids for slot in anchor.values())
        h,reason=attribution(row,anchor,admissions,eligible,
            duplicate=ids[row['labevent_id']]>1 or row['labevent_id'] in cross_ids or bad_anchor)
        reasons[reason]+=1
        if h is not None:bound_items[row['itemid']]+=1;bound_hadms.add(h)
        records.append(dict(labevent_id=row['labevent_id'],subject_id=row['subject_id'],specimen_id=row['specimen_id'],
            itemid=row['itemid'],attributed_hadm_id=h,reason=reason,
            anchor_hadm_ids=sorted(anchor),anchor_count=sum(len(s['ids']) for s in anchor.values())))
    labels={r['itemid']:r['label'] for r in rows(SNAPSHOT/'raw/d_labitems.csv.gz')}
    result=dict(status='train_only_specimen_anchor_source_audit',source_sha256=sources,
        training_admissions=len(eligible),training_patients=len(people),
        total_unassigned_rows=total,training_patient_candidate_rows=len(candidate),
        training_candidate_specimens=len(specimens),training_candidate_items=len(items),
        assigned_source_rows_scanned=raw_total,assigned_rows_with_candidate_specimens=relevant,
        candidate_specimens_with_anchor=len(anchors),multiple_hospital_anchor_specimens=sum(len(v)>1 for v in anchors.values()),
        duplicate_candidate_ids=sum(v>1 for v in ids.values()),duplicate_relevant_anchor_ids=len(duplicate_anchor_ids),
        overlapping_labevent_ids=len(cross_ids),attribution_reasons=dict(reasons),
        bound_rows=sum(bound_items.values()),bound_training_admissions=len(bound_hadms),
        bound_items={k:dict(label=labels.get(k,'unknown'),rows=v) for k,v in bound_items.items()},
        top_training_candidate_items=[dict(itemid=k,label=labels.get(k,'unknown'),rows=v) for k,v in items.most_common(15)],
        actual_seconds=time.perf_counter()-started,new_fits=0,new_vectors=0,no_outcome_data_read=True,
        next_stage='bound_numeric_specimen_unit_feasibility_needed' if bound_items else 'stop_current_same_specimen_recipe_no_anchors',
        limits=['only preserved COPD hospital source available, not complete allcause hospital inventory',
                'same specimen attribution evidence not clinical gold; time-only assignment not accepted',
                'no clinical validity or incremental performance measured'])
    return result,records

def run():
    if ROOT.exists():raise FileExistsError('audit exists; do not repeat without reason')
    ROOT.mkdir();(ROOT/'state.json').write_text('{"status":"source_audit_running","new_fits":0}\n')
    try:
        result,records=derive()
        with gzip.open(ROOT/'local_evidence.jsonl.gz','wt',encoding='utf-8') as f:
            for row in records:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        result['local_evidence_sha256']=sha256(ROOT/'local_evidence.jsonl.gz')
        (ROOT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (ROOT/'state.json').write_text(json.dumps(dict(status='source_audit_complete',next_stage=result['next_stage'],new_fits=0,goal_achieved=False))+'\n')
        print(json.dumps({k:result[k] for k in ('status','training_patient_candidate_rows','assigned_rows_with_candidate_specimens','bound_rows','attribution_reasons','next_stage','actual_seconds')}),flush=True)
    except Exception as error:
        (ROOT/'failure.json').write_text(json.dumps(dict(error=repr(error)))+'\n');raise

if __name__=='__main__':run()
