"""冻结原V6，固定出院方案接口单seed最小配对；不重新读取病历。"""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.fast_multiview import Batch,digest
from training.v6.curve_adaptation import caches,Correction as Zero,prediction as original_prediction
from training.v6.resp_fact_set_experiment import prediction
from training.v6.module_surgery_experiment import summary,save
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import RiskOutput,competing_nll
from training.v6.train import write_json

ROOT=Path('training_runs/v6_discharge_regimen_screen_2026-10-14')
DATA=Path('patient_data/processed/v6_discharge_regimen_interface_2026-10-14')
PROTOCOL=Path('docs/V6_DISCHARGE_REGIMEN_SCREEN_PROTOCOL_2026-10-14.md')
ARMS=('availability','content')


class Correction(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42);self.head=nn.Sequential(nn.Linear(68,16),nn.GELU(),nn.Linear(16,4))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,b):
        f=b.frozen;gate=b.masks.bool().all(1)
        raw=self.head(torch.cat((f.hidden,b.features,b.masks),1)).reshape(-1,2,2)*gate[:,None,None]
        delta=torch.stack((raw[:,:,0],.25*torch.tanh(raw[:,:,1])),2);old=f.raw.reshape(-1,2,2)
        return RiskOutput((f.prior+old[:,:,0]+delta[:,:,0]).clamp(-10,6),
            F.softplus(f.shape_raw+old[:,:,1]+delta[:,:,1])+.1,delta)


def feature_view(x,arm):
    if arm not in ARMS:raise ValueError('unknown arm')
    if x.ndim!=2 or x.shape[1]!=34 or not np.isfinite(x).all():raise ValueError('frozen 34 features required')
    return x.copy() if arm=='content' else np.zeros_like(x)


def batches(frozen,arrays,indices,arm):
    x=feature_view(arrays['features'],arm);output=[];right=0
    for f in frozen:
        ix=indices[right:right+len(f.hidden)];right+=len(ix)
        mask=np.stack((arrays['section_available'][ix],np.ones(len(ix),bool)),1)
        output.append(Batch(f,torch.as_tensor(x[ix],device='cuda'),torch.as_tensor(mask,device='cuda',dtype=torch.float32)))
    if right!=len(indices):raise ValueError('cache member alignment')
    return output


def setup():
    data=checked();d=data[0];m=json.loads((DATA/'manifest.json').read_text());p=json.loads((DATA/'verification.json').read_text())
    if p['maximum_error']!=0 or p['manifest_sha256']!=sha256(DATA/'manifest.json'):raise ValueError('interface proof stale')
    for file,h in m['source_sha256'].items():
        if sha256(Path(file))!=h:raise ValueError('protected interface source changed')
    for file,h in m['output_sha256'].items():
        if sha256(DATA/file)!=h:raise ValueError('interface output changed')
    with np.load(DATA/'interface.npz') as z:
        arrays={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits','features','section_available')}
    for k in ('hadm_ids','subject_ids','patient_splits'):np.testing.assert_array_equal(arrays[k],d.arrays[k])
    if arrays['features'].shape!=(19401,34):raise ValueError('feature contract changed')
    train=np.flatnonzero(d.arrays['patient_splits']=='train');tune=data[6]
    if len(train)!=13592 or len(tune)!=1531:raise ValueError('split changed')
    sources={**data[8]['source_sha256'],**data[9]['source_sha256'],**m['source_sha256']}
    for path in (Path(__file__),PROTOCOL,DATA/'manifest.json',DATA/'verification.json',
        Path('training/tests/test_v6_discharge_regimen_screen.py'),Path('training/v6/curve_adaptation.py'),
        Path('training/v6/early_stopping.py'),Path('training/v6/resp_fact_set_experiment.py'),Path('training/v6/fast_multiview.py')):
        sources[str(path)]=sha256(path)
    contract=dict(source_sha256=sources,data_provenance=d.provenance,compact_provenance=data[1].provenance,
        interface_sha256=digest(arrays),interface_metadata=m['metadata'],data_sha256=m['output_sha256']['interface.npz'],
        teacher_sha256=sha256(REFERENCE/'seed42/control/best.pt'),base_sha256=data[8]['frozen_baseline_sha256'],
        label_sha256=m['inherited_frozen_reference']['label_array_sha256'],seed=42,arms=list(ARMS),new_parameters=1172,
        optimizer=dict(name='AdamW',lr=.001,weight_decay=1e-4,delta_penalty=1e-4,clip=1.),
        early=dict(min_epochs=10,patience=7,min_delta=1e-5,max_epochs=45),
        execution_image='aecopd-embedding:v1',isolated_dependencies='training/runtime/public_tabular_2026-10-12',
        no_high_frequency=True,confirmation_test_used=False,original_inputs_and_model_unchanged=True)
    if contract['teacher_sha256']!=m['inherited_frozen_reference']['teacher_sha256']:raise ValueError('teacher mismatch')
    return data,train,tune,arrays,contract


def preflight(tc,vc,arrays,train,tune,ref):
    for arm in ARMS:
        rows=batches(tc,arrays,train,arm);probe=next(b for b in rows if b.masks[:,0].any())
        model=Correction().cuda();out=model(probe);old=probe.frozen.raw.reshape(-1,2,2)
        torch.testing.assert_close(out.log_amplitude,(probe.frozen.prior+old[:,:,0]).clamp(-10,6),rtol=0,atol=0)
        torch.testing.assert_close(out.shape,F.softplus(probe.frozen.shape_raw+old[:,:,1])+.1,rtol=0,atol=0)
        opt=torch.optim.AdamW(model.parameters(),lr=.001);loss=competing_nll(out,probe.frozen.times,probe.frozen.events)
        loss.backward();opt.step()
        if not model.head[-1].weight.ne(0).any():raise ValueError('no actual update')
        missing=replace(probe,masks=torch.zeros_like(probe.masks));missing_out=model(missing)
        torch.testing.assert_close(missing_out.log_amplitude,out.log_amplitude,rtol=0,atol=0)
        torch.testing.assert_close(missing_out.shape,out.shape,rtol=0,atol=0)
        np.testing.assert_array_equal(prediction(Correction().cuda(),batches(vc,arrays,tune,arm))[2],ref)
    # 非零修正头下屏蔽核验，不能仅以零初始天然相同蒙混通过。
    model=Correction().cuda()
    with torch.no_grad():model.head[-1].weight.fill_(.03)
    a=batches(tc,arrays,train,'availability')[0];mutated=dict(arrays,features=1-arrays['features'])
    b=batches(tc,mutated,train,'availability')[0]
    torch.testing.assert_close(model(a).log_amplitude,model(b).log_amplitude,rtol=0,atol=0)
    return dict(status='passed',initial_risk_error=0.,actual_update=True,missing_gate=True,content_hidden_nonzero_head=True,
        gpu=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in model.parameters()))


def fit(tc,vc,d,tune,folder):
    folder.mkdir();model=Correction().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    rng=np.random.default_rng(42);early=EarlyStopping(7,10,1e-5);history=[];reason='max_epochs'
    for epoch in range(46):
        online=None
        if epoch:
            total=0.;n=0
            for j in rng.permutation(len(tc)):
                b=tc[j];opt.zero_grad(set_to_none=True);out=model(b);nll=competing_nll(out,b.frozen.times,b.frozen.events)
                loss=nll+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                total+=float(nll.detach())*len(b.frozen.times);n+=len(b.frozen.times)
            online=total/n
        a,k,curve,nll=prediction(model,vc);metrics=summary(curve,d,tune,nll);ibs=metrics['readmission']['integrated_brier_7_365']
        best,stop=early.observe(epoch,ibs);history.append(dict(epoch=epoch,tuning_ibs=ibs,train_nll=online))
        if best:torch.save(dict(state_dict=model.state_dict(),epoch=epoch),folder/'best.pt')
        if epoch%5==0:print(json.dumps(dict(arm=folder.name,epoch=epoch,ibs=ibs,best=early.best_epoch)),flush=True)
        if stop:reason='patience_exhausted';break
    model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
    a,k,curve,nll=prediction(model,vc);save(folder/'tuning_predictions.npz',d,tune,a,k,curve)
    info=dict(completed_epochs=history[-1]['epoch'],best_epoch=early.best_epoch,early=asdict(early),stop_reason=reason,
        metrics=summary(curve,d,tune,nll),checkpoint_sha256=sha256(folder/'best.pt'))
    write_json(folder/'history.json',history);write_json(folder/'training_result.json',info);return info


def lock(r):
    scores={a:r['fits'][a]['metrics']['readmission']['integrated_brier_7_365'] for a in ARMS}
    scores['original']=r['reference']['readmission']['integrated_brier_7_365'];selected=min(scores,key=scores.get)
    return dict(scores=scores,selected=selected,nominated=['content'] if selected=='content' and scores['content']<min(scores['original'],scores['availability']) else [],goal_achieved=False)


def run():
    if ROOT.exists():raise FileExistsError('batch exists')
    started=time.perf_counter();data,train,tune,arrays,contract=setup();d=data[0]
    ROOT.mkdir();write_json(ROOT/'contract.json',contract)
    tc,tp=caches(data,train,42);vc,vp=caches(data,tune,42);_,_,ref,nll=original_prediction(Zero('clinical_nll').cuda(),vc)
    write_json(ROOT/'preflight.json',preflight(tc,vc,arrays,train,tune,ref))
    r=dict(reference=summary(ref,d,tune,nll),fits={},teacher_caches={'train':tp,'tune':vp},
        source_tuning_admissions=int(arrays['section_available'][tune].sum()),single_seed_only=True)
    for arm in ARMS:
        write_json(ROOT/'state.json',dict(status='risk_training',arm=arm,goal_achieved=False))
        r['fits'][arm]=fit(batches(tc,arrays,train,arm),batches(vc,arrays,tune,arm),d,tune,ROOT/arm)
    r['lock']=lock(r);r['actual_seconds']=time.perf_counter()-started
    write_json(ROOT/'screening.json',r);write_json(ROOT/'nomination_lock.json',r['lock'])
    write_json(ROOT/'state.json',dict(status='complete_pending_independent_replay',goal_achieved=False));print(json.dumps(r['lock']),flush=True)


def verify():
    data,train,tune,arrays,contract=setup();d=data[0]
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    tc,tp=caches(data,train,42);vc,vp=caches(data,tune,42);r=json.loads((ROOT/'screening.json').read_text());errors={}
    if r['teacher_caches']!={'train':tp,'tune':vp}:raise ValueError('teacher cache changed')
    _,_,ref,nll=original_prediction(Zero('clinical_nll').cuda(),vc)
    if summary(ref,d,tune,nll)!=r['reference']:raise ValueError('reference changed')
    for arm in ARMS:
        folder=ROOT/arm;info=r['fits'][arm];history=json.loads((folder/'history.json').read_text());early=EarlyStopping(7,10,1e-5);first_stop=None
        for row in history:
            _,stop=early.observe(row['epoch'],row['tuning_ibs'])
            if stop and first_stop is None:first_stop=row['epoch']
        if asdict(early)!=info['early'] or len(history)-1!=info['completed_epochs']:raise ValueError('early state differs')
        if info['stop_reason']=='patience_exhausted' and first_stop!=info['completed_epochs']:raise ValueError('delayed stop')
        saved=torch.load(folder/'best.pt',map_location='cuda',weights_only=True)
        if saved['epoch']!=early.best_epoch or sha256(folder/'best.pt')!=info['checkpoint_sha256']:raise ValueError('selected weights changed')
        model=Correction().cuda();model.load_state_dict(saved['state_dict']);a,k,curve,nll=prediction(model,batches(vc,arrays,tune,arm))
        with np.load(folder/'tuning_predictions.npz') as z:
            np.testing.assert_array_equal(z['times'],d.times[tune]);np.testing.assert_array_equal(z['events'],d.events[tune])
            error=max(float(np.abs(a-z['log_amplitude']).max()),float(np.abs(k-z['shape']).max()),float(np.abs(curve-z['curve']).max()))
        if error!=0 or summary(curve,d,tune,nll)!=info['metrics']:raise ValueError('predictions/metrics differ')
        np.testing.assert_array_equal(curve[~arrays['section_available'][tune]],ref[~arrays['section_available'][tune]])
        errors[arm]=error
    if lock(r)!=r['lock'] or r['lock']!=json.loads((ROOT/'nomination_lock.json').read_text()):raise ValueError('lock changed')
    write_json(ROOT/'verification.json',dict(status='independent_new_interface_cache_weights_daily_risk_early_lock_replay',
        maximum_error=max(errors.values()),errors=errors,screening_sha256=sha256(ROOT/'screening.json'),no_refitting=True))
    write_json(ROOT/'state.json',dict(status='single_seed_independently_verified',goal_achieved=False));print(json.dumps(errors),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=('run','verify'));mode=parser.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as error:
        if ROOT.exists():write_json(ROOT/f'failure_{mode}.json',dict(error=repr(error)))
        raise
