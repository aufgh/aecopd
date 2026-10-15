"""一次解析常规检验，保留物理轨迹摘要；内折编码只拟合内折训练尺度。"""
from collections import defaultdict
from datetime import datetime
import argparse
import json
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.core_lab_value_audit import SNAP,EVENTS,accepted
from patient_data.eos_med_context import canonical_eos
from training.v5.numeric_preprocessing import robust_parameters
from training.v6.train import write_json

ROOT=Path('training_runs/v7_sequential_research_2026-10-15/features')
CORE=Path('training_runs/v6_core_lab_continuous_2026-10-14/interface.npz')
EOS=Path('patient_data/processed/v6_eos_med_context_2026-10-09_r3/eos_med_context.npz')
REGIMEN=Path('patient_data/processed/v6_discharge_regimen_interface_2026-10-14/interface.npz')
SPECS=(('arterial_ph','50820','units',-1),('arterial_pco2','50818','mm Hg',1),
       ('hemoglobin','51222','g/dL',-1),('albumin','50862','g/dL',-1),
       ('creatinine','50912','mg/dL',1),('bicarbonate','50882','mEq/L',1),
       ('wbc','51301','K/uL',1),('ntprobnp','50963','pg/mL',1),
       ('eos_absolute','51199','cells/uL',1))
CHANNELS=('first','latest','minimum','maximum','latest_age_days','observed_span_days',
          'distinct_sample_times','physical_recovery_velocity','last_three_curvature','sampling_density')
WIDTH=288
CLINICAL={0:(7.2,7.3,7.35,7.4,7.45),1:(45.,52.,65.),3:(3.,3.5),8:(100.,300.)}


def trajectory(samples,los,direction):
    if not np.isfinite(los) or los<=0 or direction not in (-1,1):raise ValueError('invalid duration/direction')
    values=np.zeros(10);valid=np.zeros(10,bool)
    if not samples:return values,valid
    grouped=defaultdict(list)
    for t,v in samples:
        if not np.isfinite([t,v]).all() or not 0<=t<=los:raise ValueError('invalid trajectory')
        grouped[t].append(v)
    t=np.array(sorted(grouped));v=np.array([np.median(grouped[x]) for x in t])
    values[:7]=[v[0],v[-1],v.min(),v.max(),los-t[-1],t[-1]-t[0],len(t)];valid[:7]=True
    extreme=int(np.argmin(v) if direction<0 else np.argmax(v))
    if t[-1]>t[extreme]:
        values[7]=-direction*(v[-1]-v[extreme])/(t[-1]-t[extreme]);valid[7]=True
    if len(t)>=3:
        slopes=np.diff(v[-3:])/np.diff(t[-3:])
        values[8]=-direction*(slopes[1]-slopes[0])/((t[-1]-t[-3])/2);valid[8]=True
    values[9]=np.log1p(len(t)/max(los,1e-6));valid[9]=True
    return values,valid


def derive():
    started=time.perf_counter()
    sm=json.loads((SNAP/'metadata/manifest.json').read_text())
    if sm['status']!='validated':raise ValueError('source unvalidated')
    paths=[SNAP/'cohort/admissions.csv.gz',SNAP/'raw/labevents.csv.gz',EVENTS/'events.csv.gz',CORE,EOS,REGIMEN]
    sources={str(p):sha256(p) for p in paths}
    for name in ('cohort/admissions','raw/labevents'):
        if sources[str(SNAP/(name+'.csv.gz'))]!=sm['sources'][name]['sha256']:raise ValueError('raw source changed')
    with np.load(CORE) as z:ids={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits')};old_raw=z['physical_raw'];old_observed=z['concept_observed']
    n=len(ids['hadm_ids']);order={str(h):i for i,h in enumerate(ids['hadm_ids'])};bounds={}
    for r in rows(SNAP/'cohort/admissions.csv.gz'):
        if r['hadm_id'] in order:
            i=order[r['hadm_id']]
            if int(r['subject_id'])!=ids['subject_ids'][i] or r['discharge_landmark_eligible']!='t':raise ValueError('cohort mismatch')
            bounds[r['hadm_id']]=tuple(datetime.fromisoformat(r[k]) for k in ('admittime','dischtime'))
    if len(bounds)!=n:raise ValueError('missing clinical boundaries')
    arterial={r['source_id'] for r in rows(EVENTS/'events.csv.gz') if r['hadm_id'] in order
              and r['kind']=='lab' and r['concept'] in ('ph:arterial','pco2:arterial')}
    lookup={spec[1]:j for j,spec in enumerate(SPECS)};lookup['52073']=8
    observations=defaultdict(list);count=0
    for r in rows(SNAP/'raw/labevents.csv.gz'):
        count+=1;h=r['hadm_id'];item=r['itemid']
        if h not in order or item not in lookup:continue
        j=lookup[item]
        if j==8:
            try:
                t,s=(datetime.fromisoformat(r[k]) for k in ('charttime','storetime'))
                if not bounds[h][0]<=t<=s<=bounds[h][1]:continue
            except ValueError:continue
            v=canonical_eos(r['valuenum'],r['valueuom'],'absolute')
            if v is None:continue
        else:
            accepted_value,reason=accepted(r,bounds[h],SPECS[j][2],arterial if j<2 else None)
            if reason:continue
            t,v,_,_=accepted_value
        day=(t-bounds[h][0]).total_seconds()/86400
        observations[order[h],j].append((day,v))
    if count!=sm['sources']['raw/labevents']['rows']:raise ValueError('raw row count differs')
    raw=np.zeros((n,9,10));mask=np.zeros_like(raw,bool);los=np.empty(n)
    for h,i in order.items():
        los[i]=(bounds[h][1]-bounds[h][0]).total_seconds()/86400
        for j,spec in enumerate(SPECS):raw[i,j],mask[i,j]=trajectory(observations.get((i,j),[]),los[i],spec[3])
    np.testing.assert_array_equal(raw[:,:5,:2],old_raw)
    np.testing.assert_array_equal(mask[:,:5,0],old_observed)
    with np.load(EOS) as z:
        for k in ids:np.testing.assert_array_equal(ids[k],z[k])
        np.testing.assert_array_equal(raw[:,8,0],z['eos_values_raw'][:,0,0])
        np.testing.assert_array_equal(raw[:,8,1],z['eos_values_raw'][:,0,3])
        eos_max=z['eos_values_raw'][:,0,2];eos_observed=z['eos_value_masks'][:,0,2]
    with np.load(REGIMEN) as z:
        for k in ids:np.testing.assert_array_equal(ids[k],z[k])
        section=z['section_available'];positions=z['positions'];roles=z['roles'];classes=z['class_memberships']
        ics=np.zeros(n,bool);uncertain=np.zeros(n,bool)
        for i,role,c in zip(positions,roles,classes):
            if role==0 and c[0]:ics[i]=True
            if role in (3,4):uncertain[i]=True
    # 未记录ICS表示文书证据，不声称确认未治疗；未知另有mask。
    pair=eos_observed&section&~uncertain
    high=eos_max>=300
    clinical=np.stack((eos_observed,section,uncertain,ics,high&pair,high&~ics&pair),1).astype(np.float32)
    train=ids['patient_splits']=='train'
    arrays=dict(**ids,raw=raw,valid=mask,los=los,clinical=clinical,clinical_pair=pair)
    meta=dict(source_sha256=sources,raw_rows=count,specs=SPECS,channels=CHANNELS,
        train_coverage=mask[train,:,0].sum(0).tolist(),train_velocity=mask[train,:,7].sum(0).tolist(),
        train_curvature=mask[train,:,8].sum(0).tolist(),train_eos_ics_pair=int(pair[train].sum()),
        eos_ics_is_documentation_not_treatment_quality=True,old_five_axis_and_eos_replay_error=0,
        no_labels_used=True,seconds=time.perf_counter()-started)
    return arrays,meta


def scaled(values,present,fit):
    result=np.zeros_like(values,dtype=np.float32);parameters=[]
    for j in range(values.shape[1]):
        use=present[:,j]&fit
        if not use.any():parameters.append(None);continue
        p=robust_parameters(values[use,j]);parameters.append(p)
        result[:,j]=np.where(present[:,j],np.clip((values[:,j]-p['median'])/p['scale'],-8,8),0)
    return result,parameters


def encode(a,fit_indices,view):
    if view not in ('availability','linear','ple','velocity','density','curvature','clinical_main','clinical_interaction'):
        raise ValueError('unknown research feature view')
    raw,valid=a['raw'],a['valid'];n=len(raw);fit=np.zeros(n,bool);fit[fit_indices]=True
    x=np.zeros((n,WIDTH),np.float32);observed=valid[:,:,0]
    age,_=scaled(np.log1p(raw[:,:,4]),valid[:,:,4],fit)
    x[:,216:225]=observed;x[:,225:234]=age
    parameters={}
    if view not in ('availability','clinical_main','clinical_interaction'):
        for j in range(9):
            for c in (0,1):
                values=raw[:,j,c];present=valid[:,j,c];use=fit&present;offset=(2*j+c)*12
                if not use.any():continue
                if view=='linear':
                    z,p=scaled(values[:,None],present[:,None],fit)
                    x[:,offset:offset+12]=z/np.sqrt(12);parameters[f'{j}/{c}']=p
                else:
                    edges=np.unique(np.r_[np.quantile(values[use],[0,.2,.4,.6,.8,1]),CLINICAL.get(j,())])
                    if len(edges)<2:continue
                    if len(edges)-1>12:raise ValueError('too many finite PLE bins')
                    basis=np.clip((values[:,None]-edges[:-1])/(edges[1:]-edges[:-1]),0,1)
                    x[:,offset:offset+len(edges)-1]=np.where(present[:,None],basis,0)
                    parameters[f'{j}/{c}']=edges.tolist()
    if view in ('velocity','density','curvature'):
        channel={'velocity':7,'density':9,'curvature':8}[view]
        z,p=scaled(raw[:,:,channel],valid[:,:,channel],fit)
        x[:,234:243]=z;x[:,243:252]=valid[:,:,channel];parameters['dynamic']=p
    if view in ('clinical_main','clinical_interaction'):
        x[:,:]=0;x[:,:5]=a['clinical'][:,:5]
        if view=='clinical_interaction':x[:,5]=a['clinical'][:,5]
    if not np.isfinite(x).all():raise ValueError('nonfinite encoded features')
    return x,parameters


def main(mode):
    if mode=='build' and ROOT.exists():raise FileExistsError('features exist; no overwrite')
    a,meta=derive()
    if mode=='build':
        ROOT.mkdir(parents=True);np.savez_compressed(ROOT/'interface.npz',**a)
        meta['interface_sha256']=sha256(ROOT/'interface.npz')
        meta['code_sha256']=sha256(Path(__file__));write_json(ROOT/'manifest.json',meta)
        print(json.dumps({k:v for k,v in meta.items() if k not in ('source_sha256','specs','channels') }),flush=True)
    else:
        saved=json.loads((ROOT/'manifest.json').read_text())
        if saved['source_sha256']!=meta['source_sha256'] or saved['code_sha256']!=sha256(Path(__file__)):
            raise ValueError('feature source changed')
        if saved['interface_sha256']!=sha256(ROOT/'interface.npz'):raise ValueError('feature archive changed')
        with np.load(ROOT/'interface.npz') as z:
            for k,v in a.items():np.testing.assert_array_equal(z[k],v)
        write_json(ROOT/'verification.json',dict(maximum_error=0,manifest_sha256=sha256(ROOT/'manifest.json')))
        print(json.dumps(dict(feature_replay_error=0,seconds=meta['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify'),required=True)
    main(p.parse_args().mode)
