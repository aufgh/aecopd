"""固定原V6，常规去向类别细化单seed配对；不读死亡标志作输入。"""
import argparse
from collections import Counter
from dataclasses import asdict,replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import rows,sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.fast_multiview import Correction,Batch,fit,digest
from training.v6.curve_adaptation import caches,Correction as Zero,prediction as original_prediction
from training.v6.resp_fact_set_experiment import prediction
from training.v6.module_surgery_experiment import summary
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import competing_nll
from training.v6.train import write_json

ROOT=Path('training_runs/v6_disposition_context_2026-10-14')
PROTOCOL=Path('docs/V6_DISPOSITION_PROTOCOL_2026-10-14.md')
QC=Path('training_runs/v6_disposition_audit_2026-10-14/audit.json')
ARMS=('availability','content')
CATEGORIES=('HOME','HOME HEALTH CARE','SKILLED NURSING FACILITY','REHAB',
            'CHRONIC/LONG TERM ACUTE CARE','HOSPICE','AGAINST ADVICE','OTHER_KNOWN')
OTHER=('PSYCH FACILITY','OTHER FACILITY','ACUTE HOSPITAL','ASSISTED LIVING')

def classify(value):
    name=(value or '').strip().upper()
    if name in CATEGORIES[:-1]:return CATEGORIES.index(name)
    if name in OTHER:return 7
    return None  # 未知、未核验新类别和DIED不补正常、不建死亡类别。

def encode(names):
    x=np.zeros((len(names),32),np.float32);present=np.zeros(len(names),bool)
    for i,name in enumerate(names):
        code=classify(name)
        if code is not None:x[i,code]=1;present[i]=True
    return dict(features=x,observed=present)

def feature_view(x,arm):
    if arm not in ARMS:raise ValueError('unknown arm')
    return x.copy() if arm=='content' else np.zeros_like(x)

def derive(d):
    qc=json.loads(QC.read_text())
    if not qc['safe_to_research_non_death_categories'] or qc['source_conflicts']:
        raise ValueError('hospital QC unresolved')
    for p,h in qc['source_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('QC source changed')
    path=next(Path(p) for p in qc['source_sha256'] if p.endswith('/cohort/admissions.csv.gz'))
    order={str(h):i for i,h in enumerate(d.arrays['hadm_ids'])};found={}
    for row in rows(path):
        h=row['hadm_id']
        if h not in order:continue
        if h in found or int(row['subject_id'])!=int(d.arrays['subject_ids'][order[h]]):raise ValueError('member alignment changed')
        if row['discharge_landmark_eligible']!='t':raise ValueError('eligibility changed')
        found[h]=row['discharge_location']
    if set(found)!=set(order):raise ValueError('source missing')
    names=[found[str(h)] for h in d.arrays['hadm_ids']];arrays=encode(names)
    arrays.update({k:d.arrays[k] for k in ('hadm_ids','subject_ids','patient_splits')})
    train=d.arrays['patient_splits']=='train';counts=Counter(n for n,t in zip(names,train) if t)
    if counts!=Counter(qc['raw_categories']):raise ValueError('training QC changed')
    meta=dict(categories=list(CATEGORIES),known_other=list(OTHER),padding=24,
        training_content_counts=arrays['features'][train,:8].sum(0).astype(int).tolist(),
        training_present=int(arrays['observed'][train].sum()),training_unknown=int((~arrays['observed'][train]).sum()),
        training_died_unusable=counts['DIED'],no_numeric_or_outcome_scaling=True,
        role='routine discharge destination, not physiological state or causal care effect')
    return arrays,meta

def batches(frozen,arrays,indices,arm):
    x=feature_view(arrays['features'],arm);cache=[];right=0
    for f in frozen:
        ix=indices[right:right+len(f.hidden)];right+=len(ix)
        masks=np.stack((arrays['observed'][ix],np.ones(len(ix),bool)),1)
        cache.append(Batch(f,torch.as_tensor(x[ix],device='cuda'),torch.as_tensor(masks,device='cuda',dtype=torch.float32)))
    if right!=len(indices):raise ValueError('cache index mismatch')
    return cache

def setup():
    data=checked();d=data[0];train=np.flatnonzero(d.arrays['patient_splits']=='train');arrays,meta=derive(d)
    sources={**data[8]['source_sha256'],**data[9]['source_sha256']}
    for p in (Path(__file__),PROTOCOL,QC,Path('patient_data/disposition_audit.py'),
        Path('training/tests/test_v6_disposition_context.py'),Path('training/v6/fast_multiview.py'),
        Path('training/v6/curve_adaptation.py'),Path('training/v6/early_stopping.py'),Path('training/v6/resp_fact_set_experiment.py')):
        sources[str(p)]=sha256(p)
    sources.update(json.loads(QC.read_text())['source_sha256'])
    contract=dict(source_sha256=sources,data_provenance=d.provenance,compact_provenance=data[1].provenance,
        interface_sha256=digest(arrays),metadata=meta,seed=42,arms=list(ARMS),new_parameters=1140,
        teacher_sha256=sha256(REFERENCE/'seed42/control/best.pt'),base_sha256=data[8]['frozen_baseline_sha256'],
        optimizer={'name':'AdamW','lr':.001,'weight_decay':1e-4,'delta_penalty':1e-4,'clip':1.},
        early={'min_epochs':10,'patience':7,'min_delta':1e-5,'max_epochs':45},
        no_high_frequency=True,confirmation_test_used=False,cohort_labels_unchanged=True)
    return data,train,arrays,contract

def lock(r):
    values={k:r['fits'][k]['metrics']['readmission']['integrated_brier_7_365'] for k in ARMS}
    values['original']=r['reference']['readmission']['integrated_brier_7_365'];chosen=min(values,key=values.get)
    return dict(scores=values,selected=chosen,
        nominated=['content'] if chosen=='content' and values['content']<min(values['availability'],values['original']) else [],
        goal_achieved=False)

def preflight(tc,vc,arrays,train,tune,ref):
    errors={}
    for arm in ARMS:
        probe=batches(tc,arrays,train,arm)[0];m=Correction().cuda();out=m(probe);old=probe.frozen.raw.reshape(-1,2,2)
        torch.testing.assert_close(out.log_amplitude,(probe.frozen.prior+old[:,:,0]).clamp(-10,6),rtol=0,atol=0)
        torch.testing.assert_close(out.shape,torch.nn.functional.softplus(probe.frozen.shape_raw+old[:,:,1])+.1,rtol=0,atol=0)
        old_weight=m.head[-1].weight.detach().clone();opt=torch.optim.AdamW(m.parameters(),lr=.001)
        loss=competing_nll(out,probe.frozen.times,probe.frozen.events);loss.backward();opt.step()
        if torch.equal(old_weight,m.head[-1].weight):raise ValueError('no actual update')
        missing=replace(probe,masks=torch.zeros_like(probe.masks));output=m(missing)
        torch.testing.assert_close(output.log_amplitude,(probe.frozen.prior+old[:,:,0]).clamp(-10,6),rtol=0,atol=0)
        torch.testing.assert_close(output.shape,torch.nn.functional.softplus(probe.frozen.shape_raw+old[:,:,1])+.1,rtol=0,atol=0)
        np.testing.assert_array_equal(prediction(Correction().cuda(),batches(vc,arrays,tune,arm))[2],ref)
        errors[arm]=0.
    return dict(initial_parameter_and_daily_risk_errors=errors,actual_update=True,
        synthetic_missing_gate=True,related_tests=8,original_weight_updates=False)

def run():
    if ROOT.exists():raise FileExistsError('batch exists; do not repeat')
    start=time.perf_counter();data,train,arrays,contract=setup();d=data[0];tune=data[6]
    ROOT.mkdir();write_json(ROOT/'contract.json',contract)
    np.savez_compressed(ROOT/'interface.npz',**arrays)
    write_json(ROOT/'interface_manifest.json',dict(sha256=sha256(ROOT/'interface.npz'),**contract['metadata']))
    tc,tp=caches(data,train,42);vc,vp=caches(data,tune,42);_,_,ref,nll=original_prediction(Zero('clinical_nll').cuda(),vc)
    write_json(ROOT/'preflight.json',preflight(tc,vc,arrays,train,tune,ref))
    r=dict(fits={},reference=summary(ref,d,tune,nll),teacher_caches={'train':tp,'tune':vp},
        source_tuning_admissions=int(arrays['observed'][tune].sum()))
    for arm in ARMS:
        write_json(ROOT/'state.json',dict(status='risk_training',arm=arm,goal_achieved=False))
        r['fits'][arm]=fit(batches(tc,arrays,train,arm),batches(vc,arrays,tune,arm),d,tune,ROOT/arm)
    r['lock']=lock(r);r['seconds']=time.perf_counter()-start
    write_json(ROOT/'screening.json',r);write_json(ROOT/'nomination_lock.json',r['lock'])
    write_json(ROOT/'state.json',dict(status='complete_pending_independent_replay',goal_achieved=False))
    print(json.dumps(r['lock']),flush=True)

def verify():
    data,train,arrays,contract=setup();d=data[0];tune=data[6]
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    manifest=json.loads((ROOT/'interface_manifest.json').read_text())
    if manifest['sha256']!=sha256(ROOT/'interface.npz'):raise ValueError('archive changed')
    with np.load(ROOT/'interface.npz') as z:
        for key,v in arrays.items():np.testing.assert_array_equal(z[key],v)
    tc,tp=caches(data,train,42);vc,vp=caches(data,tune,42);r=json.loads((ROOT/'screening.json').read_text());errors={}
    if r['teacher_caches']!={'train':tp,'tune':vp}:raise ValueError('teacher cache changed')
    _,_,ref,nll=original_prediction(Zero('clinical_nll').cuda(),vc)
    if summary(ref,d,tune,nll)!=r['reference']:raise ValueError('reference changed')
    for arm in ARMS:
        info=r['fits'][arm];folder=ROOT/arm;hist=json.loads((folder/'history.json').read_text());early=EarlyStopping(7,10,1e-5);first_stop=None
        for row in hist:
            _,stop=early.observe(row['epoch'],row['tuning_ibs'])
            if stop and first_stop is None:first_stop=row['epoch']
        if asdict(early)!=info['early'] or len(hist)-1!=info['completed_epochs']:raise ValueError('early state differs')
        if info['stop_reason']=='patience_exhausted' and first_stop!=info['completed_epochs']:raise ValueError('delayed stop')
        saved=torch.load(folder/'best.pt',map_location='cuda',weights_only=True)
        if saved['epoch']!=early.best_epoch or sha256(folder/'best.pt')!=info['checkpoint_sha256']:raise ValueError('checkpoint changed')
        m=Correction().cuda();m.load_state_dict(saved['state_dict']);a,k,curve,nll=prediction(m,batches(vc,arrays,tune,arm))
        with np.load(folder/'tuning_predictions.npz') as z:
            np.testing.assert_array_equal(z['times'],d.times[tune]);np.testing.assert_array_equal(z['events'],d.events[tune])
            error=max(float(np.abs(a-z['log_amplitude']).max()),float(np.abs(k-z['shape']).max()),float(np.abs(curve-z['curve']).max()))
        if error!=0 or summary(curve,d,tune,nll)!=info['metrics']:raise ValueError('risk replay differs')
        np.testing.assert_array_equal(curve[~arrays['observed'][tune]],ref[~arrays['observed'][tune]])
        errors[arm]=error
    if lock(r)!=r['lock'] or r['lock']!=json.loads((ROOT/'nomination_lock.json').read_text()):raise ValueError('lock changed')
    write_json(ROOT/'verification.json',dict(status='independent_new_interface_cache_risk_early_lock_replay',errors=errors,
        maximum_error=max(errors.values()),screening_sha256=sha256(ROOT/'screening.json'),no_refitting=True))
    write_json(ROOT/'state.json',dict(status='single_seed_independently_verified',goal_achieved=False));print(json.dumps(errors),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('run','verify'),required=True);mode=p.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as error:
        if ROOT.exists():write_json(ROOT/f'failure_{mode}.json',dict(error=repr(error)))
        raise
