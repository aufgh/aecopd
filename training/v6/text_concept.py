"""训练患者内masked检验概念辅助监督；冻结course BERT，单seed快筛。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from patient_data.export_mimiciv import sha256
from patient_data.v6_lab_state import NUMERIC,FLAGS
from training.v6.fast_multiview import digest,Correction,Batch,fit,r_curve
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.curve_adaptation import caches
from training.v6.resp_fact_set_experiment import prediction
from training.v6.module_surgery_experiment import summary
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import competing_nll
from training.v6.train import write_json
from training.baselines.rank_metrics import auc_rank

ROOT=Path('training_runs/v6_text_concept_2026-10-14')
PROTOCOL=Path('docs/V6_TEXT_CONCEPT_PROTOCOL_2026-10-14.md')
AXES=('arterial_low_ph','arterial_high_pco2','wbc_high','albumin_low','hemoglobin_low')
HEADS=tuple(f'{a}/{role}' for a in AXES for role in ('peak','latest_observed'))
ARMS=('random','concept')

def targets(arrays):
    if tuple(arrays['feature_names'])!=NUMERIC+FLAGS:raise ValueError('summary fields changed')
    raw=arrays['lab_axis_raw'];names=list(arrays['axis_names']);slots=[names.index(a) for a in AXES]
    if raw.ndim!=3 or raw.shape[2]!=9 or not np.isfinite(raw).all():raise ValueError('invalid raw summary')
    observed=raw[:,slots,6].astype(bool)
    mask=np.repeat(observed,2,axis=1)
    y=(raw[:,slots,1:3]>0).reshape(len(raw),10).astype(np.float32)
    # Unknown target slots remain masked, not forced normal evidence.
    return y,mask,raw[:,slots,4].copy()

def course(d):
    text=np.zeros((len(d.history),768),np.float32);present=np.zeros(len(text),bool)
    for p,refs in enumerate(d.text_refs):
        rows=[row for field,row in refs if field==2]
        if rows:text[p]=np.asarray(d.text_vectors[rows],np.float32).mean(0);present[p]=True
    return text,present

def masked_bce(logits,y,mask):
    if logits.shape!=y.shape or mask.shape!=y.shape or mask.dtype!=torch.bool:raise ValueError('target mask differs')
    losses=F.binary_cross_entropy_with_logits(logits,y,reduction='none');counts=mask.sum(0);active=counts>0
    means=(losses*mask).sum(0)/counts.clamp_min(1)
    return means[active].mean() if active.any() else logits.sum()*0

class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.encoder=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,64),nn.GELU(),nn.Linear(64,32))
            self.decoder=nn.Linear(32,10)
    def forward(self,x):
        h=self.encoder(x);return h,self.decoder(h)

def task_split(subjects,train,present,mask,y):
    use=train[present[train]&mask[train].any(1)];fit_ix=use[subjects[use]%5!=0];hold=use[subjects[use]%5==0]
    if set(subjects[fit_ix])&set(subjects[hold]):raise ValueError('aux patient conflict')
    count=mask[fit_ix].sum(0);positive=(y[fit_ix]*mask[fit_ix]).sum(0);negative=count-positive
    active=(count>=100)&(positive>=20)&(negative>=20)
    if not active.any() or len(hold)<100:raise ValueError('insufficient valid auxiliary targets')
    return fit_ix,hold,active

def readout(logits,y,mask,prevalence):
    probabilities=1/(1+np.exp(-np.clip(logits,-30,30)));results={}
    for j,name in enumerate(HEADS):
        use=mask[:,j];yy=y[use,j];pp=np.clip(probabilities[use,j],1e-7,1-1e-7);q=float(np.clip(prevalence[j],1e-7,1-1e-7))
        results[name]=dict(observed=int(use.sum()),positive=int(yy.sum()),
            bce=float(-(yy*np.log(pp)+(1-yy)*np.log(1-pp)).mean()) if len(yy) else None,
            training_prevalence_constant_bce=float(-(yy*np.log(q)+(1-yy)*np.log(1-q)).mean()) if len(yy) else None,
            auc=float(auc_rank(pp,yy.astype(np.int8))) if 0<yy.sum()<len(yy) else None)
    return results

def auxiliary(d,text,present,y,mask,train):
    fit_ix,hold,active=task_split(d.arrays['subject_ids'],train,present,mask,y)
    used=mask&active;tensor=lambda a:torch.as_tensor(a,device='cuda')
    x=tensor(text);yy=tensor(y);mm=tensor(used);m=Reader().cuda();rng=np.random.default_rng(42)
    groups=[fit_ix[d.arrays['subject_ids'][fit_ix]==s] for s in np.unique(d.arrays['subject_ids'][fit_ix])]
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=1e-4);early=EarlyStopping(4,5,1e-4);history=[];reason='max_epochs'
    for epoch in range(21):
        if epoch:
            indices=np.asarray([rng.choice(g) for g in groups]);rng.shuffle(indices)
            for start in range(0,len(indices),256):
                ix=indices[start:start+256];opt.zero_grad(set_to_none=True);_,logits=m(x[ix]);loss=masked_bce(logits,yy[ix],mm[ix])
                if not torch.isfinite(loss):raise ValueError('nonfinite auxiliary loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        with torch.no_grad():_,logits=m(x[hold]);value=float(masked_bce(logits,yy[hold],mm[hold]))
        best,stop=early.observe(epoch,value);history.append(dict(epoch=epoch,internal_holdout_bce=value))
        if best:torch.save(m.state_dict(),ROOT/'reader_best.pt')
        if stop:reason='patience_exhausted';break
    m.load_state_dict(torch.load(ROOT/'reader_best.pt',map_location='cuda',weights_only=True));m.eval()
    counts=used[fit_ix].sum(0);prevalence=(y[fit_ix]*used[fit_ix]).sum(0)/np.maximum(counts,1)
    with torch.no_grad():_,logits=m(x[hold])
    detail=dict(history=history,early=asdict(early),actual_epochs=history[-1]['epoch'],stop_reason=reason,active_heads=active.tolist(),
        fit_admissions=len(fit_ix),holdout_admissions=len(hold),fit_patients=len(groups),fit_indices_sha256=digest({'fit':fit_ix,'hold':hold}),
        prevalence=prevalence.tolist(),readout=readout(logits.cpu().numpy(),y[hold],used[hold],prevalence),
        reader_sha256=sha256(ROOT/'reader_best.pt'),no_outer_validation_used=True,no_readmission_death_targets=True)
    write_json(ROOT/'auxiliary.json',detail)

def representation(text,present,train,arm):
    m=Reader().cuda().eval()
    if arm=='concept':m.load_state_dict(torch.load(ROOT/'reader_best.pt',map_location='cuda',weights_only=True))
    result=[]
    with torch.no_grad():
        for start in range(0,len(text),512):result.append(m(torch.as_tensor(text[start:start+512],device='cuda'))[0].cpu().numpy())
    z=np.concatenate(result);use=train[present[train]];mean=z[use].mean(0);scale=np.maximum(z[use].std(0),.001)
    z=np.clip((z-mean)/scale,-8,8);z[~present]=0
    return z.astype(np.float32),dict(mean=mean.tolist(),scale=scale.tolist(),training_records=len(use))

def batches(frozen,z,present,ix):
    result=[];offset=0
    for f in frozen:
        use=ix[offset:offset+len(f.hidden)];offset+=len(use)
        result.append(Batch(f,torch.as_tensor(z[use],device='cuda'),torch.as_tensor(np.stack((present[use],np.ones(len(use),bool)),1),device='cuda',dtype=torch.float32)))
    if offset!=len(ix):raise ValueError('frozen cache alignment changed')
    return result

def setup():
    data=checked();d=data[0];train=np.flatnonzero(d.arrays['patient_splits']=='train');tune=data[6]
    text,present=course(d);y,mask,age=targets(d.arrays)
    sources={**data[8]['source_sha256'],**data[9]['source_sha256']}
    for p in (Path(__file__),PROTOCOL,Path('training/tests/test_v6_text_concept.py'),Path('training/v6/fast_multiview.py'),
              Path('training/v6/curve_adaptation.py'),Path('training/v6/early_stopping.py'),Path('patient_data/v6_lab_state.py')):sources[str(p)]=sha256(p)
    contract=dict(source_sha256=sources,data_provenance=d.provenance,compact_provenance=data[1].provenance,
        teacher_sha256=sha256(REFERENCE/'seed42/control/best.pt'),inputs_sha256=digest(dict(text=text,present=present,y=y,mask=mask,age=age)),
        protocol_sha256=sha256(PROTOCOL),heads=list(HEADS),seed=42,arms=list(ARMS),high_frequency_used=False,confirmation_test_used=False)
    return data,train,tune,text,present,y,mask,age,contract

def lock(r):
    scores={a:r['fits'][a]['metrics']['readmission']['integrated_brier_7_365'] for a in ARMS}
    scores['original']=r['reference']['readmission']['integrated_brier_7_365'];chosen=min(scores,key=scores.get)
    return dict(scores=scores,selected=chosen,nominated=['concept'] if chosen=='concept' and scores['concept']<min(scores['random'],scores['original']) else [],goal_achieved=False)

def run():
    if ROOT.exists():raise FileExistsError('existing batch; inspect')
    started=time.perf_counter();data,train,tune,text,present,y,mask,age,contract=setup();d=data[0]
    fit_ix,hold,active=task_split(d.arrays['subject_ids'],train,present,mask,y);ROOT.mkdir();write_json(ROOT/'contract.json',contract)
    audit={}
    for j,name in enumerate(HEADS):
        use=train[present[train]&mask[train,j]];a=age[use,j//2]
        audit[name]=dict(training_observed_with_course=len(use),positive=int(y[use,j].sum()),negative=int(len(use)-y[use,j].sum()),
            latest_measurement_age_median_days=float(np.median(np.expm1(a))) if len(a) else None,role='whole_admission_peak' if j%2==0 else 'latest_observed_not_discharge')
    write_json(ROOT/'target_audit.json',dict(heads=audit,active_heads=active.tolist(),training_text_admissions=int(present[train].sum()),
        no_target_for_unobserved=True,no_original_database_rescan=True))
    tc,tp=caches(data,train,42);vc,vp=caches(data,tune,42)
    from training.v6.curve_adaptation import Correction as Zero,prediction as original_prediction
    _,_,ref,nll=original_prediction(Zero('clinical_nll').cuda(),vc)
    z,_=representation(text,present,train,'random');cache=batches(tc,z,present,train)
    observed=np.flatnonzero(present[train]);slot=int(observed[0]);bi,ri=divmod(slot,512);probe=cache[bi]
    m=Correction().cuda();out=m(probe);old=probe.frozen.raw.reshape(-1,2,2)
    torch.testing.assert_close(out.log_amplitude,(probe.frozen.prior+old[:,:,0]).clamp(-10,6),atol=0,rtol=0)
    torch.testing.assert_close(out.shape,F.softplus(probe.frozen.shape_raw+old[:,:,1])+.1,atol=0,rtol=0)
    opt=torch.optim.AdamW(m.parameters(),lr=.001);loss=competing_nll(out,probe.frozen.times,probe.frozen.events);loss.backward();opt.step()
    if not m.head[-1].weight.ne(0).any():raise ValueError('no real update')
    missing=~probe.masks.bool().all(1);updated=m(probe)
    if missing.any():torch.testing.assert_close(updated.log_amplitude[missing],out.log_amplitude[missing],atol=0,rtol=0)
    write_json(ROOT/'preflight.json',dict(status='passed',initial_risk_error=0.,real_update=True,missing_gate=True,gpu=torch.cuda.get_device_name()))
    write_json(ROOT/'state.json',dict(status='auxiliary_training',goal_achieved=False));auxiliary(d,text,present,y,mask,train)
    r=dict(fits={},features={},teacher_caches={'train':tp,'tune':vp},reference=summary(ref,d,tune,nll),single_seed_only=True)
    for arm in ARMS:
        z,scale=representation(text,present,train,arm);r['features'][arm]={'sha256':digest({'z':z}),'scale':scale}
        tb=batches(tc,z,present,train);vb=batches(vc,z,present,tune)
        np.testing.assert_array_equal(prediction(Correction().cuda(),vb)[2],ref)
        write_json(ROOT/'state.json',dict(status='risk_training',arm=arm,goal_achieved=False));r['fits'][arm]=fit(tb,vb,d,tune,ROOT/arm)
    r['lock']=lock(r);r['seconds']=time.perf_counter()-started;write_json(ROOT/'screening.json',r);write_json(ROOT/'nomination_lock.json',r['lock'])
    write_json(ROOT/'state.json',dict(status='complete_pending_independent_replay',goal_achieved=False));print(json.dumps(r['lock']),flush=True)

def replay_early(history,info,aux=False):
    early=EarlyStopping(4,5,1e-4) if aux else EarlyStopping(7,10,1e-5);first_stop=None
    for row in history:
        _,stop=early.observe(row['epoch'],row['internal_holdout_bce'] if aux else row['tuning_ibs'])
        if stop and first_stop is None:first_stop=row['epoch']
    expected=info['actual_epochs'] if aux else info['completed_epochs']
    if asdict(early)!=info['early'] or expected!=len(history)-1:raise ValueError('early state differs')
    if info['stop_reason']=='patience_exhausted' and first_stop!=expected:raise ValueError('stopping delayed')
    return early

def verify():
    data,train,tune,text,present,y,mask,age,contract=setup();d=data[0]
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract/source differs')
    aux=json.loads((ROOT/'auxiliary.json').read_text());replay_early(aux['history'],aux,True)
    fi,ho,active=task_split(d.arrays['subject_ids'],train,present,mask,y)
    if digest({'fit':fi,'hold':ho})!=aux['fit_indices_sha256'] or active.tolist()!=aux['active_heads'] or sha256(ROOT/'reader_best.pt')!=aux['reader_sha256']:raise ValueError('aux source/model changed')
    m=Reader().cuda().eval();m.load_state_dict(torch.load(ROOT/'reader_best.pt',map_location='cuda',weights_only=True))
    # Re-establish the same strict FP32 policy as all frozen risk caches.
    torch.backends.cuda.matmul.allow_tf32=False
    with torch.no_grad():_,logits=m(torch.as_tensor(text[ho],device='cuda'))
    if readout(logits.cpu().numpy(),y[ho],mask[ho]&active,np.asarray(aux['prevalence']))!=aux['readout']:raise ValueError('auxiliary prediction differs')
    tc,tp=caches(data,train,42);vc,vp=caches(data,tune,42);r=json.loads((ROOT/'screening.json').read_text());errors={}
    if r['teacher_caches']!={'train':tp,'tune':vp}:raise ValueError('teacher cache differs')
    ref=r_curve(data,vc)
    for arm in ARMS:
        z,scale=representation(text,present,train,arm)
        if digest({'z':z})!=r['features'][arm]['sha256'] or scale!=r['features'][arm]['scale']:raise ValueError('representation differs')
        folder=ROOT/arm;info=r['fits'][arm];early=replay_early(json.loads((folder/'history.json').read_text()),info)
        state=torch.load(folder/'best.pt',map_location='cuda',weights_only=True)
        if sha256(folder/'best.pt')!=info['checkpoint_sha256'] or state['epoch']!=early.best_epoch:raise ValueError('saved selection differs')
        model=Correction().cuda();model.load_state_dict(state['state_dict']);a,k,curve,nll=prediction(model,batches(vc,z,present,tune))
        with np.load(folder/'tuning_predictions.npz') as zold:
            np.testing.assert_array_equal(zold['times'],d.times[tune]);np.testing.assert_array_equal(zold['events'],d.events[tune])
            error=max(float(np.abs(a-zold['log_amplitude']).max()),float(np.abs(k-zold['shape']).max()),float(np.abs(curve-zold['curve']).max()))
        if error!=0 or summary(curve,d,tune,nll)!=info['metrics']:raise ValueError('risk metrics differs')
        np.testing.assert_array_equal(curve[~present[tune]],ref[~present[tune]]);errors[arm]=error
    if lock(r)!=r['lock'] or r['lock']!=json.loads((ROOT/'nomination_lock.json').read_text()):raise ValueError('nomination differs')
    write_json(ROOT/'verification.json',dict(status='independent_replay_passed',maximum_error=max(errors.values()),errors=errors,
        auxiliary_replayed=True,full_source_targets_features_scales_teacher_caches_rebuilt=True,screening_sha256=sha256(ROOT/'screening.json')))
    write_json(ROOT/'state.json',dict(status='single_seed_independently_verified',goal_achieved=False));print(json.dumps(errors),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('run','verify'),required=True);mode=p.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as e:
        if ROOT.exists():write_json(ROOT/f'failure_{mode}.json',{'error':repr(e)})
        raise
