"""低频常规住院上下文接口及单seed配对，不改变原模型。"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch
from dataclasses import replace,asdict
from patient_data.export_mimiciv import rows,sha256
from training.v5.numeric_preprocessing import robust_parameters
from training.v6.admission_context_audit import interval
from training.v6.fast_multiview import Correction,Batch,fit,digest
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.curve_adaptation import caches,Correction as Zero,prediction as original_prediction
from training.v6.resp_fact_set_experiment import prediction
from training.v6.module_surgery_experiment import summary
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import competing_nll
from training.v6.train import write_json

ROOT=Path('training_runs/v6_admission_context_2026-10-14')
PROTOCOL=Path('docs/V6_ADMISSION_CONTEXT_PROTOCOL_2026-10-14.md')
AUDIT=Path('training_runs/v6_admission_context_audit_2026-10-14/audit.json')
ARMS=('availability','content')

def category(value):
    value=(value or '').strip()
    return None if value in ('','\\N','INFORMATION NOT AVAILABLE') else value

def raw_row(row):
    raw=np.zeros(2,np.float64);mask=np.zeros(4,bool);reasons=[]
    los,error=interval(row.get('admittime'),row.get('dischtime'))
    if error:reasons.append('los_'+error)
    else:raw[0]=los;mask[0]=True
    ed,error=interval(row.get('edregtime'),row.get('edouttime'))
    if error:reasons.append('ed_'+error)
    else:
        try:discharge=datetime.fromisoformat(row.get('dischtime'))
        except (ValueError,TypeError):discharge=None
        if discharge is None:reasons.append('ed_hospital_boundary_unknown')
        elif datetime.fromisoformat(row['edregtime'])>discharge or datetime.fromisoformat(row['edouttime'])>discharge:reasons.append('ed_after_discharge')
        else:raw[1]=ed*24;mask[1]=True
    cats=(category(row.get('admission_type')),category(row.get('admission_location')));mask[2:]=[c is not None for c in cats]
    return raw,mask,cats,reasons

def encode(raw,masks,categories,folds):
    n=len(raw);x=np.zeros((n,32),np.float32);vocab={};scales={};codes=np.zeros((n,2),np.int64)
    if raw.shape!=(n,2) or masks.shape!=(n,4) or len(categories)!=n or len(folds)!=n or not np.isfinite(raw).all() or (raw<0).any():raise ValueError('invalid input interface')
    x[:,2:6]=masks;names=['hospital_length_log_days','ed_interval_log_hours',*[f'{k}_present' for k in ('los','ed','type','location')]]
    for j,key in enumerate(names[:2]):
        use=masks[:,j];train=use&(folds=='train')
        if train.any():
            p=robust_parameters(np.log1p(raw[train,j]));x[use,j]=np.clip((np.log1p(raw[use,j])-p['median'])/p['scale'],-8,8);scales[key]=p
        else:scales[key]={'method':'no_training_observation','training_observations':0}
    offset=6
    for j,key in enumerate(('type','location')):
        counts=Counter(c[j] for i,c in enumerate(categories) if folds[i]=='train' and c[j] is not None)
        words=['UNKNOWN',*sorted(k for k,v in counts.items() if v>=100)];vocab[key]=words;lookup={k:v for v,k in enumerate(words)}
        if offset+len(words)>32:raise ValueError('category width exceeds predeclared capacity')
        for i,c in enumerate(categories):
            codes[i,j]=lookup.get(c[j],0)
            if masks[i,j+2]:x[i,offset+codes[i,j]]=1
        names.extend(key+':'+w for w in words);offset+=len(words)
    names.extend(['unused']*(32-len(names)))
    return dict(features=x,raw_values=raw,masks=masks,codes=codes,observed=masks.any(1)),dict(scales=scales,vocab=vocab,columns=names,used_columns=offset)

def derive(data):
    d=data[0];a=json.loads(AUDIT.read_text())
    for collection in ('sources_sha256','config_sha256'):
        for p,h in a[collection].items():
            if sha256(Path(p))!=h:raise ValueError('audited source changed '+p)
    path=next(Path(p) for p in a['sources_sha256'] if p.endswith('/raw/admissions.csv.gz'))
    cohortpath=next(Path(p) for p in a['sources_sha256'] if p.endswith('/cohort/admissions.csv.gz'))
    lookup={str(h):i for i,h in enumerate(d.arrays['hadm_ids'])};source={};cohort={}
    for target,sourcepath in ((source,path),(cohort,cohortpath)):
        for row in rows(sourcepath):
            h=row['hadm_id']
            if h in lookup:
                if h in target:raise ValueError('duplicate source admission')
                target[h]=row
        if set(target)!=set(lookup):raise ValueError('source admission missing')
    raw=np.zeros((len(lookup),2),np.float64);masks=np.zeros((len(lookup),4),bool);cats=[];rejected=Counter()
    for h,i in lookup.items():
        r=source[h];c=cohort[h]
        if int(r['subject_id'])!=int(d.arrays['subject_ids'][i]) or r['subject_id']!=c['subject_id']:raise ValueError('patient alignment differs')
        if any(r[k]!=c[k] for k in ('admittime','dischtime')):raise ValueError('hospital boundaries differ')
        for k in ('admission_type','admission_location'):
            if k in c and r[k]!=c[k]:raise ValueError('cohort source differs')
        raw[i],masks[i],category_values,reason=raw_row(r);cats.append(category_values)
        if d.arrays['patient_splits'][i]=='train':rejected.update(reason)
    arrays,metadata=encode(raw,masks,cats,d.arrays['patient_splits'])
    arrays.update({k:d.arrays[k] for k in ('hadm_ids','subject_ids','patient_splits')})
    training=d.arrays['patient_splits']=='train'
    metadata.update(training_mask_counts=masks[training].sum(0).tolist(),training_rejections=dict(rejected),
        all_admissions=len(lookup),audit_sha256=sha256(AUDIT),sources_sha256=a['sources_sha256'])
    if metadata['training_mask_counts'][:2]!=[13592,10326] or rejected['ed_after_discharge']!=54:raise ValueError('audited aggregate differs')
    return arrays,metadata

def feature_view(x,arm):
    if arm not in ARMS:raise ValueError('unknown condition')
    result=x.copy()
    if arm=='availability':result[:,:2]=0;result[:,6:]=0
    return result

def batches(frozen,arrays,indices,arm):
    features=feature_view(arrays['features'],arm);cache=[];right=0
    for f in frozen:
        use=indices[right:right+len(f.hidden)];right+=len(use)
        masks=np.stack((arrays['observed'][use],np.ones(len(use),bool)),1)
        cache.append(Batch(f,torch.as_tensor(features[use],device='cuda'),torch.as_tensor(masks,device='cuda',dtype=torch.float32)))
    if right!=len(indices):raise ValueError('cache alignment')
    return cache

def setup():
    data=checked();train=np.flatnonzero(data[0].arrays['patient_splits']=='train');tune=data[6];arrays,metadata=derive(data)
    src={**data[8]['source_sha256'],**data[9]['source_sha256'],**metadata['sources_sha256']}
    for p in (Path(__file__),PROTOCOL,AUDIT,Path('training/tests/test_v6_admission_context.py'),
        Path('training/v6/admission_context_audit.py'),Path('training/v6/fast_multiview.py'),Path('training/v6/curve_adaptation.py'),Path('training/v6/early_stopping.py')):src[str(p)]=sha256(p)
    contract=dict(source_sha256=src,data_provenance=data[0].provenance,compact_provenance=data[1].provenance,interface_sha256=digest(arrays),
        metadata=metadata,protocol_sha256=sha256(PROTOCOL),seed=42,arms=list(ARMS),new_parameters=1140,
        teacher_sha256=sha256(REFERENCE/'seed42/control/best.pt'),no_high_frequency=True,confirmation_test_used=False)
    return data,train,tune,arrays,contract

def lock(r):
    scores={a:r['fits'][a]['metrics']['readmission']['integrated_brier_7_365'] for a in ARMS}
    scores['original']=r['reference']['readmission']['integrated_brier_7_365'];chosen=min(scores,key=scores.get)
    return dict(scores=scores,selected=chosen,nominated=['content'] if chosen=='content' and scores['content']<min(scores['original'],scores['availability']) else [],goal_achieved=False)

def run():
    if ROOT.exists():raise FileExistsError('existing batch; inspect')
    start=time.perf_counter();data,train,tune,arrays,contract=setup();d=data[0]
    ROOT.mkdir();write_json(ROOT/'contract.json',contract);np.savez_compressed(ROOT/'interface.npz',**arrays);write_json(ROOT/'interface_manifest.json',dict(sha256=sha256(ROOT/'interface.npz'),**contract['metadata']))
    tc,tp=caches(data,train,42);vc,vp=caches(data,tune,42);_,_,ref,nll=original_prediction(Zero('clinical_nll').cuda(),vc)
    probe=batches(tc,arrays,train,'content')[0];model=Correction().cuda();out=model(probe);old=probe.frozen.raw.reshape(-1,2,2)
    torch.testing.assert_close(out.log_amplitude,(probe.frozen.prior+old[:,:,0]).clamp(-10,6),atol=0,rtol=0)
    torch.testing.assert_close(out.shape,torch.nn.functional.softplus(probe.frozen.shape_raw+old[:,:,1])+.1,atol=0,rtol=0)
    opt=torch.optim.AdamW(model.parameters(),lr=.001);loss=competing_nll(out,probe.frozen.times,probe.frozen.events);loss.backward();opt.step()
    if not model.head[-1].weight.ne(0).any():raise ValueError('no real update')
    missing=replace(probe,masks=torch.zeros_like(probe.masks));missing_out=model(missing)
    torch.testing.assert_close(missing_out.log_amplitude,out.log_amplitude,atol=0,rtol=0);torch.testing.assert_close(missing_out.shape,out.shape,atol=0,rtol=0)
    control=batches(tc,arrays,train,'availability')[0]
    mutated=arrays['features'].copy();mutated[:,:2]+=10;mutated[:,6:]=1-mutated[:,6:]
    altered=batches(tc,dict(arrays,features=mutated),train,'availability')[0]
    torch.testing.assert_close(model(control).log_amplitude,model(altered).log_amplitude,atol=0,rtol=0)
    write_json(ROOT/'preflight.json',dict(status='passed',initial_risk_error=0.,actual_update=True,missing_gate=True,content_hidden=True,gpu=torch.cuda.get_device_name()))
    r=dict(fits={},teacher_caches={'train':tp,'tune':vp},reference=summary(ref,d,tune,nll),single_seed_only=True)
    for arm in ARMS:
        tb=batches(tc,arrays,train,arm);vb=batches(vc,arrays,tune,arm)
        np.testing.assert_array_equal(prediction(Correction().cuda(),vb)[2],ref)
        write_json(ROOT/'state.json',dict(status='risk_training',arm=arm,goal_achieved=False));r['fits'][arm]=fit(tb,vb,d,tune,ROOT/arm)
    r['lock']=lock(r);r['seconds']=time.perf_counter()-start;write_json(ROOT/'screening.json',r);write_json(ROOT/'nomination_lock.json',r['lock'])
    write_json(ROOT/'state.json',dict(status='complete_pending_independent_replay',goal_achieved=False));print(json.dumps(r['lock']),flush=True)

def verify():
    data,train,tune,arrays,contract=setup();d=data[0]
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    manifest=json.loads((ROOT/'interface_manifest.json').read_text())
    if manifest['sha256']!=sha256(ROOT/'interface.npz'):raise ValueError('archive changed')
    with np.load(ROOT/'interface.npz') as saved:
        for key,value in arrays.items():np.testing.assert_array_equal(saved[key],value)
    tc,tp=caches(data,train,42);vc,vp=caches(data,tune,42);r=json.loads((ROOT/'screening.json').read_text());errors={}
    if r['teacher_caches']!={'train':tp,'tune':vp}:raise ValueError('teacher source changed')
    _,_,ref,nll=original_prediction(Zero('clinical_nll').cuda(),vc)
    if summary(ref,d,tune,nll)!=r['reference']:raise ValueError('reference differs')
    for arm in ARMS:
        folder=ROOT/arm;info=r['fits'][arm];hist=json.loads((folder/'history.json').read_text());early=EarlyStopping(7,10,1e-5);first_stop=None
        for row in hist:
            _,stop=early.observe(row['epoch'],row['tuning_ibs'])
            if stop and first_stop is None:first_stop=row['epoch']
        if asdict(early)!=info['early'] or info['completed_epochs']!=len(hist)-1:raise ValueError('early state differs')
        if info['stop_reason']=='patience_exhausted' and first_stop!=info['completed_epochs']:raise ValueError('delayed stop')
        saved=torch.load(folder/'best.pt',map_location='cuda',weights_only=True)
        if saved['epoch']!=early.best_epoch or sha256(folder/'best.pt')!=info['checkpoint_sha256']:raise ValueError('checkpoint selection changed')
        model=Correction().cuda();model.load_state_dict(saved['state_dict']);a,k,curve,nll=prediction(model,batches(vc,arrays,tune,arm))
        with np.load(folder/'tuning_predictions.npz') as old:
            np.testing.assert_array_equal(old['times'],d.times[tune]);np.testing.assert_array_equal(old['events'],d.events[tune])
            error=max(float(np.abs(a-old['log_amplitude']).max()),float(np.abs(k-old['shape']).max()),float(np.abs(curve-old['curve']).max()))
        if error!=0 or summary(curve,d,tune,nll)!=info['metrics']:raise ValueError('risk/metrics differs')
        missing=~arrays['observed'][tune]
        if missing.any():np.testing.assert_array_equal(curve[missing],ref[missing])
        errors[arm]=error
    if lock(r)!=r['lock'] or r['lock']!=json.loads((ROOT/'nomination_lock.json').read_text()):raise ValueError('lock changed')
    write_json(ROOT/'verification.json',dict(status='independent_full_interface_and_prediction_replay',maximum_error=max(errors.values()),errors=errors,
        screening_sha256=sha256(ROOT/'screening.json'),interface_rebuilt=True,early_and_lock_replayed=True))
    write_json(ROOT/'state.json',dict(status='single_seed_independently_verified',goal_achieved=False));print(json.dumps(errors),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('run','verify'),required=True);mode=p.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as e:
        if ROOT.exists():write_json(ROOT/f'failure_{mode}.json',{'error':repr(e)})
        raise
