"""训练折原去向与医院资格来源QC；不读取下游结局、不改队列。"""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows,sha256

PARENT=Path('training_runs/v6_admission_context_2026-10-14')
ROOT=Path('training_runs/v6_disposition_audit_2026-10-14')
PROTOCOL=Path('docs/V6_DISPOSITION_PROTOCOL_2026-10-14.md')
SQL=Path('cohort_pipeline/sql/08c_copd_coded_admissions_mimiciv.sql')

def eligibility(row):
    try:
        return int(row['hospital_expire_flag'])==0 and datetime.fromisoformat(row['dischtime'])>datetime.fromisoformat(row['admittime'])
    except (KeyError,ValueError,TypeError):
        raise ValueError('unreadable hospital qualification') from None

def time_role(value,row):
    if value in ('',r'\N',None):return 'unrecorded'
    value=datetime.fromisoformat(value)
    a,b=(datetime.fromisoformat(row[k]) for k in ('admittime','dischtime'))
    return 'before_admission' if value<a else 'within_hospital' if value<=b else 'after_discharge'

def derive():
    m=json.loads((PARENT/'interface_manifest.json').read_text());proof=json.loads((PARENT/'verification.json').read_text())
    if m['sha256']!=sha256(PARENT/'interface.npz') or proof['maximum_error']!=0 or proof['screening_sha256']!=sha256(PARENT/'screening.json'):
        raise ValueError('parent proof changed')
    for p,h in m['sources_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('source changed')
    with np.load(PARENT/'interface.npz') as z:
        members={str(h):int(s) for h,s,f in zip(z['hadm_ids'],z['subject_ids'],z['patient_splits']) if f=='train'}
    paths={key:next(Path(p) for p in m['sources_sha256'] if p.endswith(suffix)) for key,suffix in
           (('cohort','/cohort/admissions.csv.gz'),('raw','/raw/admissions.csv.gz'))}
    data={}
    for name,path in paths.items():
        found={}
        for row in rows(path):
            h=row['hadm_id']
            if h not in members:continue
            if h in found or int(row['subject_id'])!=members[h]:raise ValueError('training identity conflict')
            found[h]=row
        if set(found)!=set(members):raise ValueError('training source missing')
        data[name]=found
    targets=[]; conflicts=Counter();categories=Counter()
    for h,cohort in data['cohort'].items():
        raw=data['raw'][h];categories[cohort['discharge_location']]+=1
        for key in ('subject_id','admittime','dischtime','discharge_location','hospital_expire_flag'):
            if cohort[key]!=raw[key]:conflicts[key]+=1
        if cohort['discharge_landmark_eligible']!='t' or not eligibility(raw):conflicts['eligibility']+=1
        if cohort['discharge_location']=='DIED':
            targets.append(dict(hadm_id=h,subject_id=members[h],
                cohort_eligible=cohort['discharge_landmark_eligible']=='t',
                hospital_expire_flag=int(raw['hospital_expire_flag']),
                positive_hospital_duration=datetime.fromisoformat(raw['dischtime'])>datetime.fromisoformat(raw['admittime']),
                deathtime_role=time_role(raw.get('deathtime'),raw)))
    if len(targets)!=2 or len(members)!=13592:raise ValueError('prior training audit differs')
    result=dict(status='train_only_hospital_source_qc',training_admissions=len(members),
        source_conflicts=dict(conflicts),died_records=len(targets),
        died_expire_flags=dict(Counter(str(r['hospital_expire_flag']) for r in targets)),
        died_deathtime_roles=dict(Counter(r['deathtime_role'] for r in targets)),
        all_died_match_original_eligibility=all(r['cohort_eligible'] and r['hospital_expire_flag']==0 and r['positive_hospital_duration'] for r in targets),
        raw_categories=dict(categories),safe_to_research_non_death_categories=not bool(conflicts),
        died_new_input_policy='unusable_source_mask_false_original_risk_unchanged_not_removed',
        no_downstream_outcomes=True,no_new_fits=True,no_new_vectors=True,
        source_sha256={**{str(p):sha256(p) for p in paths.values()},str(SQL):sha256(SQL),str(PROTOCOL):sha256(PROTOCOL),
            str(PARENT/'interface.npz'):sha256(PARENT/'interface.npz'),str(Path(__file__)):sha256(Path(__file__))})
    return result,targets

def run():
    if ROOT.exists():raise FileExistsError('audit exists; do not overwrite')
    result,targets=derive();ROOT.mkdir()
    (ROOT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # 身份只在本地私有QC底账；不打印原标识或日期。
    (ROOT/'local_qc_records.json').write_text(json.dumps(targets,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','source_conflicts','died_records','died_expire_flags','died_deathtime_roles','all_died_match_original_eligibility','safe_to_research_non_death_categories')}),flush=True)

if __name__=='__main__':run()
