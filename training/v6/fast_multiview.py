"""训练患者内跨模态无负样本表示；单种子快速配对，不改原V6。"""
import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.regularization_audit import checked, REFERENCE
from training.v6.curve_adaptation import caches
from training.v6.resp_fact_set_experiment import prediction
from training.v6.module_surgery_experiment import summary, save
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import RiskOutput, competing_nll
from training.v6.train import write_json
from patient_data.export_mimiciv import sha256

ROOT=Path('training_runs/v6_fast_multiview_2026-10-14')
PROTOCOL=Path('docs/V6_FAST_MULTIVIEW_PROTOCOL_2026-10-14.md')
ARMS=('random','aligned')

def digest(arrays):
    h=hashlib.sha256()
    for key,v in sorted(arrays.items()):
        h.update(key.encode());h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()

def views(data):
    d,c,_,ext,*_=data;n=len(d.history)
    table=np.concatenate((d.arrays['lab_axis_inputs'].reshape(n,-1),c.arrays['med_features'].reshape(n,-1),
        c.arrays['micro_values'],c.arrays['micro_masks'],c.arrays['support_values'],c.arrays['support_masks'],
        ext['new_labs'].reshape(n,-1),ext['eos'].reshape(n,-1)),1).astype(np.float32)
    text=np.zeros((n,768),np.float32);tp=np.zeros(n,bool)
    for p,refs in enumerate(d.text_refs):
        rows=[r for f,r in refs if f==2]
        if rows:text[p]=np.asarray(d.text_vectors[rows],np.float32).mean(0);tp[p]=True
    tab=(d.arrays['lab_axis_inputs'][:,:,6].astype(bool).any(1)|c.arrays['med_features'][:,:,2:].astype(bool).any((1,2))|
         c.arrays['micro_masks'].any(1)|c.arrays['support_masks'].any(1)|ext['new_labs'][:,:,5:].astype(bool).any((1,2))|
         ext['eos'][:,:,6:12].astype(bool).any((1,2)))
    if table.shape!=(n,332) or not np.isfinite(table).all() or not np.isfinite(text).all():raise ValueError('view shape/nonfinite')
    return dict(table=table,text=text,table_present=tab,text_present=tp,paired=tab&tp)

class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.tab=nn.Sequential(nn.Linear(332,64),nn.GELU(),nn.Linear(64,16))
            self.text=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,64),nn.GELU(),nn.Linear(64,16))
    def forward(self,t,x):return self.tab(t),self.text(x)

def vicreg(x,y):
    if x.shape!=y.shape or x.ndim!=2 or len(x)<2:raise ValueError('paired batch must have at least two rows')
    rep=F.mse_loss(x,y);a=x-x.mean(0);b=y-y.mean(0)
    std=(F.relu(1-torch.sqrt(a.var(0)+1e-4)).mean()+F.relu(1-torch.sqrt(b.var(0)+1e-4)).mean())/2
    width=x.shape[1];mask=~torch.eye(width,dtype=torch.bool,device=x.device)
    cov=sum(((v.T@v)/(len(v)-1))[mask].square().sum()/width for v in (a,b))
    return 25*rep+25*std+cov

def eligible_split(subjects,train,paired):
    use=train[paired[train]];hold=use[(subjects[use]%5)==0];fit=use[(subjects[use]%5)!=0]
    if set(subjects[hold])&set(subjects[fit]) or len(hold)<256 or len(fit)<512:raise ValueError('internal view split insufficient/conflicted')
    return fit,hold

def pretrain(v,d,train):
    fit,hold=eligible_split(d.arrays['subject_ids'],train,v['paired']);m=Reader().cuda();rng=np.random.default_rng(42)
    tensor=lambda x:torch.as_tensor(x,device='cuda');t=tensor(v['table']);x=tensor(v['text'])
    groups=[fit[d.arrays['subject_ids'][fit]==s] for s in np.unique(d.arrays['subject_ids'][fit])]
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=1e-4);early=EarlyStopping(3,3,.001);hist=[]
    def score():
        with torch.no_grad():
            a,b=m(t[hold],x[hold]);return float(vicreg(a,b))
    for epoch in range(13):
        online=None
        if epoch:
            rows=np.asarray([rng.choice(g) for g in groups]);rng.shuffle(rows);total=0.;n=0
            for start in range(0,len(rows),256):
                ix=rows[start:start+256]
                if len(ix)<2:continue
                opt.zero_grad(set_to_none=True);a,b=m(t[ix],x[ix]);loss=vicreg(a,b)
                if not torch.isfinite(loss):raise ValueError('nonfinite self supervision')
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();total+=float(loss.detach())*len(ix);n+=len(ix)
            online=total/n
        value=score();best,stop=early.observe(epoch,value);hist.append(dict(epoch=epoch,internal_holdout_loss=value,train_loss=online))
        if best:torch.save(m.state_dict(),ROOT/'reader_best.pt')
        if stop:break
    m.load_state_dict(torch.load(ROOT/'reader_best.pt',map_location='cuda',weights_only=True));m.requires_grad_(False)
    with torch.no_grad():
        a,b=m(t[hold],x[hold]);stats=dict(table_std_mean=float(a.std(0).mean()),text_std_mean=float(b.std(0).mean()),
            table_effective_rank=int(torch.linalg.matrix_rank(a-a.mean(0))),text_effective_rank=int(torch.linalg.matrix_rank(b-b.mean(0))))
    write_json(ROOT/'self_supervision.json',dict(history=hist,early=asdict(early),actual_epochs=hist[-1]['epoch'],
        fit_admissions=len(fit),holdout_admissions=len(hold),fit_patients=len(groups),no_outer_validation=True,no_outcome_targets=True,
        one_admission_per_patient_per_training_epoch=True,diagnostics=stats,weights_sha256=sha256(ROOT/'reader_best.pt')))

@dataclass
class Batch:
    frozen:object
    features:torch.Tensor
    masks:torch.Tensor

class Correction(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42);self.head=nn.Sequential(nn.Linear(66,16),nn.GELU(),nn.Linear(16,4))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,b):
        f=b.frozen;gate=b.masks.bool().all(1)
        raw=self.head(torch.cat((f.hidden,b.features,b.masks),1)).reshape(-1,2,2)*gate[:,None,None]
        delta=torch.stack((raw[:,:,0],.25*torch.tanh(raw[:,:,1])),2);old=f.raw.reshape(-1,2,2)
        return RiskOutput((f.prior+old[:,:,0]+delta[:,:,0]).clamp(-10,6),F.softplus(f.shape_raw+old[:,:,1]+delta[:,:,1])+.1,delta)

def features(v,train,arm):
    m=Reader().cuda()
    if arm=='aligned':m.load_state_dict(torch.load(ROOT/'reader_best.pt',map_location='cuda',weights_only=True))
    m.eval().requires_grad_(False);zs=[]
    with torch.no_grad():
        for start in range(0,len(v['table']),512):
            a,b=m(torch.as_tensor(v['table'][start:start+512],device='cuda'),torch.as_tensor(v['text'][start:start+512],device='cuda'))
            zs.append(torch.cat((a,b),1).cpu().numpy())
    z=np.concatenate(zs);use=train[v['paired'][train]];mean=z[use].mean(0);scale=np.maximum(z[use].std(0),.001)
    z=np.clip((z-mean)/scale,-8,8);z[~v['paired']]=0
    return z.astype(np.float32),dict(mean=mean.tolist(),scale=scale.tolist(),training_records=len(use))

def batches(frozen,z,v,ix):
    out=[];right=0
    for f in frozen:
        use=ix[right:right+len(f.hidden)];right+=len(use)
        out.append(Batch(f,torch.as_tensor(z[use],device='cuda'),torch.as_tensor(np.stack((v['table_present'][use],v['text_present'][use]),1),device='cuda',dtype=torch.float32)))
    if right!=len(ix):raise ValueError('cache index mismatch')
    return out

def fit(tc,vc,d,tune,folder):
    folder.mkdir();m=Correction().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=1e-4)
    rng=np.random.default_rng(42);early=EarlyStopping(7,10,1e-5);hist=[];reason='max_epochs'
    for epoch in range(46):
        online=None
        if epoch:
            total=0.;n=0
            for j in rng.permutation(len(tc)):
                b=tc[j];opt.zero_grad(set_to_none=True);out=m(b);nll=competing_nll(out,b.frozen.times,b.frozen.events)
                loss=nll+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite risk fit')
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();total+=float(nll.detach())*len(b.frozen.times);n+=len(b.frozen.times)
            online=total/n
        a,k,curve,nll=prediction(m,vc);metrics=summary(curve,d,tune,nll);ibs=metrics['readmission']['integrated_brier_7_365']
        best,stop=early.observe(epoch,ibs);hist.append(dict(epoch=epoch,tuning_ibs=ibs,train_nll=online))
        if best:torch.save(dict(state_dict=m.state_dict(),epoch=epoch),folder/'best.pt')
        if epoch%5==0:print(json.dumps(dict(arm=folder.name,epoch=epoch,ibs=ibs,best=early.best_epoch)),flush=True)
        if stop:reason='patience_exhausted';break
    m.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True)['state_dict']);a,k,curve,nll=prediction(m,vc)
    save(folder/'tuning_predictions.npz',d,tune,a,k,curve)
    info=dict(completed_epochs=hist[-1]['epoch'],best_epoch=early.best_epoch,early=asdict(early),stop_reason=reason,
        metrics=summary(curve,d,tune,nll),checkpoint_sha256=sha256(folder/'best.pt'))
    write_json(folder/'history.json',hist);write_json(folder/'training_result.json',info);return info

def setup():
    data=checked();d=data[0];train=np.flatnonzero(d.arrays['patient_splits']=='train');tune=data[6]
    v=views(data)
    src={**data[8]['source_sha256'],**data[9]['source_sha256']}
    for p in (Path(__file__),PROTOCOL,Path('training/tests/test_v6_fast_multiview.py'),Path('training/v6/resp_fact_set_experiment.py'),
              Path('training/v6/curve_adaptation.py'),Path('training/v6/early_stopping.py')):src[str(p)]=sha256(p)
    contract=dict(source_sha256=src,data_provenance=d.provenance,compact_provenance=data[1].provenance,
        teacher_sha256=sha256(REFERENCE/'seed42/control/best.pt'),view_sha256=digest(v),seed=42,arms=list(ARMS),
        protocol_sha256=sha256(PROTOCOL),single_seed_screening_only=True,confirmation_test_evaluated=False,high_frequency_used=False)
    return data,train,tune,v,contract

def verify():
    data,train,tune,v,contract=setup();d=data[0]
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('source or contract differs')
    tc,tp=caches(data,train,42);vc,vp=caches(data,tune,42);r=json.loads((ROOT/'screening.json').read_text());errors={}
    if r['teacher_caches']!={'train':tp,'tune':vp}:raise ValueError('teacher cache differs')
    for arm in ARMS:
        z,scale=features(v,train,arm)
        if digest({'z':z})!=r['features'][arm]['sha256'] or scale!=r['features'][arm]['scale']:raise ValueError('representation/scales differ')
        folder=ROOT/arm;info=r['fits'][arm];hist=json.loads((folder/'history.json').read_text());early=EarlyStopping(7,10,1e-5)
        for row in hist:early.observe(row['epoch'],row['tuning_ibs'])
        if asdict(early)!=info['early'] or len(hist)-1!=info['completed_epochs']:raise ValueError('early stop differs')
        if info['stop_reason']=='patience_exhausted' and not (early.last_epoch>=10 and early.stale>=7):raise ValueError('invalid early stop')
        if sha256(folder/'best.pt')!=info['checkpoint_sha256']:raise ValueError('weight hash differs')
        state=torch.load(folder/'best.pt',map_location='cuda',weights_only=True)
        if state['epoch']!=early.best_epoch:raise ValueError('checkpoint selection differs')
        m=Correction().cuda();m.load_state_dict(state['state_dict']);cache=batches(vc,z,v,tune);a,k,curve,nll=prediction(m,cache)
        with np.load(folder/'tuning_predictions.npz') as old:
            np.testing.assert_array_equal(old['times'],d.times[tune]);np.testing.assert_array_equal(old['events'],d.events[tune])
            error=max(float(np.abs(a-old['log_amplitude']).max()),float(np.abs(k-old['shape']).max()),float(np.abs(curve-old['curve']).max()))
        if error!=0 or summary(curve,d,tune,nll)!=info['metrics']:raise ValueError('risk/metric replay differs')
        if np.any(curve[~v['paired'][tune]]!=r_curve(data,vc)[~v['paired'][tune]]):raise ValueError('missing views changed original')
        errors[arm]=error
    write_json(ROOT/'verification.json',dict(status='independent_full_source_representation_risk_replay_passed',errors=errors,
        maximum_error=max(errors.values()),screening_sha256=sha256(ROOT/'screening.json'),single_seed_only=True))
    write_json(ROOT/'state.json',dict(status='single_seed_screening_independently_verified',goal_achieved=False))
    print(json.dumps(errors),flush=True)

def r_curve(data,vc):
    from training.v6.curve_adaptation import Correction as Zero
    from training.v6.curve_adaptation import prediction as old_prediction
    return old_prediction(Zero('clinical_nll').cuda(),vc)[2]

def run():
    if ROOT.exists():raise FileExistsError('batch already exists; inspect, do not overwrite')
    started=time.perf_counter();data,train,tune,v,contract=setup();d=data[0];eligible_split(d.arrays['subject_ids'],train,v['paired'])
    ROOT.mkdir();write_json(ROOT/'contract.json',contract)
    coverage={fold:dict(admissions=int((d.arrays['patient_splits']==fold).sum()),paired=int(v['paired'][d.arrays['patient_splits']==fold].sum())) for fold in ('train','validation','test')}
    write_json(ROOT/'coverage.json',dict(coverage=coverage,table_dim=332,text_dim=768,source_values_not_labels=True))
    tc,tp=caches(data,train,42);vc,vp=caches(data,tune,42)
    from training.v6.curve_adaptation import Correction as Zero, prediction as original_prediction
    _,_,ref,ref_nll=original_prediction(Zero('clinical_nll').cuda(),vc)
    z,_=features(v,train,'random');probe=batches(tc,z,v,train);m=Correction().cuda();out=m(probe[0]);original=probe[0].frozen.raw.reshape(-1,2,2)
    torch.testing.assert_close(out.log_amplitude,(probe[0].frozen.prior+original[:,:,0]).clamp(-10,6),atol=0,rtol=0)
    loss=competing_nll(out,probe[0].frozen.times,probe[0].frozen.events);loss.backward();opt=torch.optim.AdamW(m.parameters(),lr=.001);opt.step()
    if not m.head[-1].weight.ne(0).any():raise ValueError('no real GPU update')
    write_json(ROOT/'preflight.json',dict(status='passed',initial_risk_error=0.,gpu=torch.cuda.get_device_name(),actual_update=True))
    write_json(ROOT/'state.json',dict(status='training_internal_view_pretraining',goal_achieved=False));pretrain(v,d,train)
    result=dict(fits={},features={},teacher_caches={'train':tp,'tune':vp},reference=summary(ref,d,tune,ref_nll),single_seed_only=True)
    for arm in ARMS:
        z,scale=features(v,train,arm);result['features'][arm]={'sha256':digest({'z':z}),'scale':scale}
        train_cache=batches(tc,z,v,train);tune_cache=batches(vc,z,v,tune)
        np.testing.assert_array_equal(prediction(Correction().cuda(),tune_cache)[2],ref)
        write_json(ROOT/'state.json',dict(status='single_seed_risk_training',arm=arm,goal_achieved=False))
        result['fits'][arm]=fit(train_cache,tune_cache,d,tune,ROOT/arm)
    scores={a:result['fits'][a]['metrics']['readmission']['integrated_brier_7_365'] for a in ARMS}
    scores['original']=result['reference']['readmission']['integrated_brier_7_365'];chosen=min(scores,key=scores.get)
    result['lock']=dict(scores=scores,selected=chosen,nominated=['aligned'] if chosen=='aligned' and scores['aligned']<min(scores['random'],scores['original']) else [],goal_achieved=False)
    result['seconds']=time.perf_counter()-started;write_json(ROOT/'screening.json',result);write_json(ROOT/'nomination_lock.json',result['lock'])
    write_json(ROOT/'state.json',dict(status='single_seed_complete_pending_independent_replay',goal_achieved=False));print(json.dumps(result['lock']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('run','verify'),required=True);mode=p.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as e:
        if ROOT.exists():write_json(ROOT/f'failure_{mode}.json',{'error':repr(e)})
        raise
