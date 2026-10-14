"""训练折常规入院上下文可用性，不读结局/文本/高频记录，不拟合模型。"""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from training.v6.train import write_json

ROOT=Path('training_runs/v6_admission_context_audit_2026-10-14')

def interval(start,end):
    try:a,b=datetime.fromisoformat(start),datetime.fromisoformat(end)
    except (ValueError,TypeError):return None,'missing_or_invalid_time'
    if b<a:return None,'reverse_time'
    return (b-a).total_seconds()/86400,None

def run():
    if ROOT.exists():raise FileExistsError('existing audit; do not repeat')
    cfg_path=Path('training_runs/v6_original_interaction_2026-10-12_r3/config.json');cfg=json.loads(cfg_path.read_text())
    ref_path=Path(cfg['reference'])/'config.json';ref=json.loads(ref_path.read_text());snapshot=Path(ref['snapshot']);state=Path(ref['state_dir'])
    sm=json.loads((snapshot/'metadata/manifest.json').read_text());st=json.loads((state/'manifest.json').read_text())
    files={snapshot/'metadata/manifest.json':st['source_manifest_sha256'],state/'clinical_state.npz':st['output_sha256']['clinical_state.npz']}
    for name in ('cohort/admissions','raw/admissions'):
        files[snapshot/(name+'.csv.gz')]=sm['sources'][name]['sha256']
    for path,d in files.items():
        if sha256(path)!=d:raise ValueError('source changed: '+str(path))
    with np.load(state/'clinical_state.npz') as z:
        hadms=z['hadm_ids'];subjects=z['subject_ids'];folds=z['patient_splits']
    if len(hadms)!=19401 or (folds=='train').sum()!=13592:raise ValueError('cohort drift')
    lookup={str(h):i for i,h in enumerate(hadms)};cohort={r['hadm_id']:r for r in rows(snapshot/'cohort/admissions.csv.gz') if r['hadm_id'] in lookup}
    admissions={};columns=None
    for r in rows(snapshot/'raw/admissions.csv.gz'):
        columns=tuple(r)
        if r['hadm_id'] in lookup:
            if r['hadm_id'] in admissions:raise ValueError('duplicate admission')
            admissions[r['hadm_id']]=r
    if set(admissions)!=set(lookup) or set(cohort)!=set(lookup):raise ValueError('missing source admission')
    los=[];ed=[];types=Counter();locations=Counter();issues=Counter();n=0
    for h,p in lookup.items():
        c=cohort[h];r=admissions[h]
        if int(c['subject_id'])!=int(subjects[p]) or r['subject_id']!=c['subject_id']:raise ValueError('patient identity mismatch')
        if r['admittime']!=c['admittime'] or r['dischtime']!=c['dischtime']:raise ValueError('cohort hospital boundary differs')
        if folds[p]!='train':continue
        n+=1;value,reason=interval(r['admittime'],r['dischtime'])
        if reason:issues['hospital_'+reason]+=1
        else:los.append(value)
        for key,d in (('admission_type',types),('admission_location',locations)):
            value=r.get(key,'');d[value if value not in ('','\\N') else '<UNKNOWN>']+=1
        value,reason=interval(r.get('edregtime'),r.get('edouttime'))
        if reason:issues['ed_'+reason]+=1
        elif datetime.fromisoformat(r['edregtime'])>datetime.fromisoformat(r['dischtime']) or datetime.fromisoformat(r['edouttime'])>datetime.fromisoformat(r['dischtime']):
            issues['ed_after_discharge']+=1
        else:ed.append(value*24)
    quant=lambda x:dict(observed=len(x),quantile_levels=[.1,.5,.9,.99],quantiles=np.quantile(x,[.1,.5,.9,.99]).tolist()) if x else dict(observed=0)
    report=dict(status='training_only_source_validated_not_model_input',training_admissions=n,source_columns=list(columns),
        hospital_length_days=quant(los),ed_recorded_interval_hours=quant(ed),admission_type=dict(types),admission_location=dict(locations),issues=dict(issues),
        sources_sha256={str(p):d for p,d in files.items()},config_sha256={str(p):sha256(p) for p in (cfg_path,ref_path)},audit_code_sha256=sha256(Path(__file__)),
        no_model_fit=True,no_outcome_read=True,no_high_frequency_read=True,
        cautions=['Recorded ED interval is not necessarily waiting time or acuity; may overlap hospital admission.',
                  'Administrative source categories may encode care pathway and era; no causal inference.',
                  'Insurance/language/race/marital status/expiration flags were not proposed or analyzed.'])
    ROOT.mkdir();write_json(ROOT/'audit.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('sources_sha256','config_sha256','source_columns')}),flush=True)

if __name__=='__main__':run()
