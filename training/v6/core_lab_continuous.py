"""五核心连续原值与同源旧摘要的最小配对，原V6不变。"""
import argparse
from collections import defaultdict
from dataclasses import asdict,replace
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import rows,sha256
from patient_data.core_lab_value_audit import ROOT as AUDIT,STATE,EVENTS,SNAP,AXES,accepted,readout
from patient_data.lab_recovery_features import AXIS_CONCEPT,abnormal_severity
from patient_data.preprocess_events import LABS
from training.v5.numeric_preprocessing import robust_parameters
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.fast_multiview import Batch,Correction,fit,digest
from training.v6.curve_adaptation import caches,Correction as Zero,prediction as original_prediction
from training.v6.resp_fact_set_experiment import prediction
from training.v6.module_surgery_experiment import summary
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import competing_nll
from training.v6.train import write_json

ROOT=Path('training_runs/v6_core_lab_continuous_2026-10-14')
PROTOCOL=Path('docs/V6_CORE_LAB_CONTINUOUS_PROTOCOL_2026-10-14.md')
ARMS=('summary','continuous')

def scaled_values(raw,observed,training):
    x=np.zeros_like(raw);scales={}
    for j,axis in enumerate(AXES):
        use=observed[:,j];train=use&training
        for k in range(2):
            if not train.any():raise ValueError('no training value')
            p=robust_parameters(raw[train,j,k]);scales[f'{axis}/{k}']=p
            x[use,j,k]=np.clip((raw[use,j,k]-p['median'])/p['scale'],-8,8)
    return x.astype(np.float32),scales

def feature_view(x,arm):
    if arm not in ARMS:raise ValueError('unknown arm')
    v=x.copy()
    if arm=='summary':v[:,:10]=0
    return v

def derive(d):
    audit=json.loads((AUDIT/'audit.json').read_text());proof=json.loads((AUDIT/'verification.json').read_text())
    if proof['maximum_error'] or proof['audit_sha256']!=sha256(AUDIT/'audit.json'):raise ValueError('audit proof stale')
    for p,h in audit['source_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('audit source changed')
    order={str(h):i for i,h in enumerate(d.arrays['hadm_ids'])};bounds={}
    for row in rows(SNAP/'cohort/admissions.csv.gz'):
        if row['hadm_id'] not in order:continue
        i=order[row['hadm_id']]
        if int(row['subject_id'])!=d.arrays['subject_ids'][i] or row['discharge_landmark_eligible']!='t':raise ValueError('alignment differs')
        bounds[row['hadm_id']]=tuple(datetime.fromisoformat(row[k]) for k in ('admittime','dischtime'))
    if len(bounds)!=19401:raise ValueError('cohort differs')
    arterial={row['source_id'] for row in rows(EVENTS/'events.csv.gz') if row['hadm_id'] in order
              and row['kind']=='lab' and row['concept'] in ('ph:arterial','pco2:arterial')}
    items={}
    for j,axis in enumerate(AXES):
        item,unit=next((i,u) for i,(c,u) in LABS.items() if c==AXIS_CONCEPT[axis].split(':')[0]);items[item]=(j,axis,unit)
    observations=defaultdict(list);count=0
    for row in rows(SNAP/'raw/labevents.csv.gz'):
        count+=1;h=row['hadm_id']
        if h not in order or row['itemid'] not in items:continue
        j,axis,unit=items[row['itemid']];value,reason=accepted(row,bounds[h],unit,arterial if axis.startswith('arterial_') else None)
        if reason:continue
        t,v,lo,hi=value;day=(t-bounds[h][0]).total_seconds()/86400
        observations[h,j].append((day,v,lo,hi,abnormal_severity(axis,v,lo,hi,severity_cap=None)))
    if count!=audit['assigned_lab_source_rows_scanned']:raise ValueError('source count differs')
    n=len(order);raw=np.zeros((n,5,2));obs=np.zeros((n,5),bool);nine=np.zeros((n,5,9));x=np.zeros((n,32),np.float32)
    for i,h in enumerate(d.arrays['hadm_ids']):
        h=str(h);length=(bounds[h][1]-bounds[h][0]).total_seconds()/86400
        for j in range(5):
            nine[i,j],raw[i,j],_,info=readout(observations.get((h,j),[]),length);obs[i,j]=info['observed']
    indices=[list(d.arrays['axis_names']).index(a) for a in AXES]
    error=float(np.abs(nine-d.arrays['lab_axis_raw'][:,indices]).max())
    if error>1e-12:raise ValueError('original allfold axis replay differs')
    training=d.arrays['patient_splits']=='train'
    with np.load(AUDIT/'training_audit.npz') as z:
        np.testing.assert_array_equal(d.arrays['hadm_ids'][training],z['hadm_ids'])
        np.testing.assert_array_equal(raw[training],z['physical_first_latest']);np.testing.assert_array_equal(obs[training],z['observed'])
    scaled,scales=scaled_values(raw,obs,training);x[:,:10]=scaled.reshape(n,10)
    old=d.arrays['lab_axis_inputs'][:,indices]
    x[:,10:20]=old[:,:,[0,2]].reshape(n,10);x[:,20:25]=obs;x[:,25:30]=old[:,:,4]
    arrays=dict(features=x,physical_raw=raw,concept_observed=obs,observed=obs.any(1),
        **{k:d.arrays[k] for k in ('hadm_ids','subject_ids','patient_splits')})
    return arrays,dict(scales=scales,axes=list(AXES),original_replay_error=error,
        train_concept_counts=obs[training].sum(0).tolist(),raw_source_rows=count,
        fixed_columns='physical10/old_severity10/observed5/old_age5/padding2')

def batches(frozen,a,ix,arm):
    x=feature_view(a['features'],arm);output=[];right=0
    for f in frozen:
        use=ix[right:right+len(f.hidden)];right+=len(use)
        mask=np.stack((a['observed'][use],np.ones(len(use),bool)),1)
        output.append(Batch(f,torch.as_tensor(x[use],device='cuda'),torch.as_tensor(mask,device='cuda',dtype=torch.float32)))
    if right!=len(ix):raise ValueError('cache alignment differs')
    return output

def setup():
    data=checked();d=data[0];train=np.flatnonzero(d.arrays['patient_splits']=='train');a,meta=derive(d)
    sources={**data[8]['source_sha256'],**data[9]['source_sha256'],**json.loads((AUDIT/'audit.json').read_text())['source_sha256']}
    for p in (Path(__file__),PROTOCOL,AUDIT/'audit.json',AUDIT/'verification.json',AUDIT/'training_audit.npz',
              Path('training/tests/test_v6_core_lab_continuous.py'),Path('training/v6/fast_multiview.py'),
              Path('training/v6/curve_adaptation.py'),Path('training/v6/early_stopping.py'),Path('training/v6/resp_fact_set_experiment.py')):sources[str(p)]=sha256(p)
    contract=dict(source_sha256=sources,data_provenance=d.provenance,compact_provenance=data[1].provenance,
        interface_sha256=digest(a),metadata=meta,seed=42,arms=list(ARMS),new_parameters=1140,
        teacher_sha256=sha256(REFERENCE/'seed42/control/best.pt'),base_sha256=data[8]['frozen_baseline_sha256'],
        no_high_frequency=True,confirmation_test_used=False)
    return data,train,a,contract

def lock(r):
    scores={k:r['fits'][k]['metrics']['readmission']['integrated_brier_7_365'] for k in ARMS}
    scores['original']=r['reference']['readmission']['integrated_brier_7_365'];chosen=min(scores,key=scores.get)
    return dict(scores=scores,selected=chosen,nominated=['continuous'] if chosen=='continuous'
        and scores['continuous']<min(scores['summary'],scores['original']) else [],goal_achieved=False)

def run():
    if ROOT.exists():raise FileExistsError('batch exists')
    start=time.perf_counter();data,train,a,contract=setup();d=data[0];tune=data[6]
    ROOT.mkdir();write_json(ROOT/'contract.json',contract);np.savez_compressed(ROOT/'interface.npz',**a)
    write_json(ROOT/'interface_manifest.json',dict(sha256=sha256(ROOT/'interface.npz'),**contract['metadata']))
    tc,tp=caches(data,train,42);vc,vp=caches(data,tune,42);_,_,ref,nll=original_prediction(Zero('clinical_nll').cuda(),vc)
    for arm in ARMS:
        probe=batches(tc,a,train,arm)[0];m=Correction().cuda();out=m(probe);old=probe.frozen.raw.reshape(-1,2,2)
        torch.testing.assert_close(out.log_amplitude,(probe.frozen.prior+old[:,:,0]).clamp(-10,6),atol=0,rtol=0)
        torch.testing.assert_close(out.shape,torch.nn.functional.softplus(probe.frozen.shape_raw+old[:,:,1])+.1,atol=0,rtol=0)
        opt=torch.optim.AdamW(m.parameters(),lr=.001);weight=m.head[-1].weight.detach().clone()
        competing_nll(out,probe.frozen.times,probe.frozen.events).backward();opt.step()
        if torch.equal(weight,m.head[-1].weight):raise ValueError('no update')
        missing=m(replace(probe,masks=torch.zeros_like(probe.masks)))
        torch.testing.assert_close(missing.log_amplitude,(probe.frozen.prior+old[:,:,0]).clamp(-10,6),atol=0,rtol=0)
        np.testing.assert_array_equal(prediction(Correction().cuda(),batches(vc,a,tune,arm))[2],ref)
    write_json(ROOT/'preflight.json',dict(initial_parameters_and_daily_risk_error=0.,actual_update=True,
        missing_gate=True,related_tests=6))
    r=dict(fits={},reference=summary(ref,d,tune,nll),teacher_caches={'train':tp,'tune':vp})
    for arm in ARMS:
        write_json(ROOT/'state.json',dict(status='risk_training',arm=arm,goal_achieved=False))
        r['fits'][arm]=fit(batches(tc,a,train,arm),batches(vc,a,tune,arm),d,tune,ROOT/arm)
    r['lock']=lock(r);r['seconds']=time.perf_counter()-start
    write_json(ROOT/'screening.json',r);write_json(ROOT/'nomination_lock.json',r['lock'])
    write_json(ROOT/'state.json',dict(status='pending_independent_replay',goal_achieved=False));print(json.dumps(r['lock']),flush=True)

def verify():
    data,train,a,contract=setup();d=data[0];tune=data[6]
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract differs')
    manifest=json.loads((ROOT/'interface_manifest.json').read_text())
    if manifest['sha256']!=sha256(ROOT/'interface.npz'):raise ValueError('archive changed')
    with np.load(ROOT/'interface.npz') as z:
        for k,v in a.items():np.testing.assert_array_equal(z[k],v)
    tc,tp=caches(data,train,42);vc,vp=caches(data,tune,42);r=json.loads((ROOT/'screening.json').read_text());errors={}
    if r['teacher_caches']!={'train':tp,'tune':vp}:raise ValueError('teacher cache changed')
    _,_,ref,nll=original_prediction(Zero('clinical_nll').cuda(),vc)
    if summary(ref,d,tune,nll)!=r['reference']:raise ValueError('reference differs')
    for arm in ARMS:
        folder=ROOT/arm;info=r['fits'][arm];hist=json.loads((folder/'history.json').read_text());early=EarlyStopping(7,10,1e-5);first=None
        for row in hist:
            _,stop=early.observe(row['epoch'],row['tuning_ibs'])
            if stop and first is None:first=row['epoch']
        if asdict(early)!=info['early'] or len(hist)-1!=info['completed_epochs']:raise ValueError('early differs')
        if info['stop_reason']=='patience_exhausted' and first!=info['completed_epochs']:raise ValueError('stop delayed')
        saved=torch.load(folder/'best.pt',map_location='cuda',weights_only=True)
        if saved['epoch']!=early.best_epoch or sha256(folder/'best.pt')!=info['checkpoint_sha256']:raise ValueError('weight differs')
        m=Correction().cuda();m.load_state_dict(saved['state_dict']);aa,k,curve,nll=prediction(m,batches(vc,a,tune,arm))
        with np.load(folder/'tuning_predictions.npz') as z:
            np.testing.assert_array_equal(z['times'],d.times[tune]);np.testing.assert_array_equal(z['events'],d.events[tune])
            error=max(float(np.abs(aa-z['log_amplitude']).max()),float(np.abs(k-z['shape']).max()),float(np.abs(curve-z['curve']).max()))
        if error or summary(curve,d,tune,nll)!=info['metrics']:raise ValueError('prediction differs')
        np.testing.assert_array_equal(curve[~a['observed'][tune]],ref[~a['observed'][tune]]);errors[arm]=error
    if lock(r)!=r['lock'] or r['lock']!=json.loads((ROOT/'nomination_lock.json').read_text()):raise ValueError('lock differs')
    write_json(ROOT/'verification.json',dict(status='one_independent_full_interface_scale_cache_prediction_replay',
        maximum_error=max(errors.values()),errors=errors,screening_sha256=sha256(ROOT/'screening.json'),no_refitting=True))
    write_json(ROOT/'state.json',dict(status='single_seed_independently_verified',goal_achieved=False));print(json.dumps(errors),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('run','verify'),required=True);mode=p.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as e:
        if ROOT.exists():write_json(ROOT/f'failure_{mode}.json',dict(error=repr(e)))
        raise
