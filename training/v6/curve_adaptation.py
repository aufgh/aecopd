"""冻结原表示的小容量全年评分对照；原V6文件/权重不变。"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.module_surgery_experiment import batches
from training.v6.warm_residual_audit import teacher
from training.v6.warm_residual import BASE_SEEDS
from training.v6.horizon_objective import restore
from training.v6.risk import RiskOutput,competing_nll
from training.v6.curve_score import daily_cif,curve_scores
from training.v6.module_surgery import probability_curve
from training.v6.evaluation import metrics
from training.v6.train import write_json

ROOT=Path('training_runs/v6_curve_adaptation_2026-10-13')
PROTOCOL=Path('docs/V6_CURVE_ADAPTATION_PROTOCOL_2026-10-13.md')
SEEDS=(42,7,101)
ARMS=('clinical_nll','clinical_curve','global_curve')
EPOCHS=100
SOURCES=('training/v6/curve_adaptation.py','training/v6/curve_adaptation_audit.py',
         'training/v6/curve_score.py','training/tests/test_v6_curve_adaptation.py',
         'training/tests/test_v6_curve_score.py',str(PROTOCOL))


@dataclass
class FrozenBatch:
    hidden:torch.Tensor
    raw:torch.Tensor
    prior:torch.Tensor
    shape_raw:torch.Tensor
    times:torch.Tensor
    events:torch.Tensor


class Correction(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError('unknown arm')
        self.arm=arm
        if arm=='global_curve':self.global_raw=nn.Parameter(torch.zeros(4))
        else:
            self.linear=nn.Linear(32,4)
            nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)

    def forward(self,b):
        new=self.global_raw.expand(len(b.raw),4) if self.arm=='global_curve' else self.linear(b.hidden)
        raw=(b.raw+new).reshape(-1,2,2)
        return RiskOutput((b.prior+raw[:,:,0]).clamp(-10,6),F.softplus(b.shape_raw+raw[:,:,1])+.1,new.reshape(-1,2,2))


def new_model(arm,seed):
    model=Correction(arm).cuda();torch.manual_seed(seed)
    return model


@torch.no_grad()
def extract(d,c,ext,ix,model):
    result=[];digest=hashlib.sha256(ix.tobytes());model.eval();error=0.
    for _,h,b,t,e in batches(d,c,ext,ix):
        state,_=model.encoder(b.base);hidden=model.delta_head[:-1](state);raw=model.delta_head[-1](hidden)
        prior=model.baseline(h)
        row=FrozenBatch(hidden.detach(),raw.detach(),prior.log_amplitude.detach(),model.baseline.raw_shape.detach(),t,e)
        expected=model(h,b)
        zero=raw.reshape(-1,2,2)
        a=(row.prior+zero[:,:,0]).clamp(-10,6);k=F.softplus(row.shape_raw+zero[:,:,1])+.1
        error=max(error,float((a-expected.log_amplitude).abs().max()),float((k-expected.shape).abs().max()))
        for value in (row.hidden,row.raw,row.prior,row.shape_raw,row.times,row.events):
            digest.update(value.cpu().numpy().tobytes())
        result.append(row)
    if error!=0:raise ValueError('teacher extraction differs')
    return result,dict(records=len(ix),cache_sha256=digest.hexdigest(),initial_parameter_max_error=error,
                       index_sha256=hashlib.sha256(ix.tobytes()).hexdigest())


def setup_inputs():
    data=checked();d,c,h,ext,_,bins,tune,confirm,search,_=data
    train=np.flatnonzero(d.arrays['patient_splits']=='train')
    if not len(train) or set(d.arrays['subject_ids'][train])&set(d.arrays['subject_ids'][tune]):
        raise ValueError('patient fold conflict')
    return data,train


def source_contract(data):
    d,c,h,ext,_,bins,tune,confirm,search,_=data
    return dict(source_sha256={p:sha256(Path(p)) for p in SOURCES},
        data_provenance=d.provenance,compact_provenance=c.provenance,input_config=search['input_config'],
        search_contract_sha256=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')),
        teacher_sha256={str(s):sha256(REFERENCE/f'seed{s}/control/best.pt') for s in (42,7)},
        baseline_sha256={str(s):sha256(Path(search['input_config']['parent'])/f'seed{s}/survival_base/best.pt') for s in (42,7)},
        seeds=list(SEEDS),arms=list(ARMS),epochs=EPOCHS,protocol_sha256=sha256(PROTOCOL),
        optimizer=dict(kind='AdamW',lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=1e-4,
                       clipping=1.,correction_square_penalty=1e-4,warmup_epochs=5),
        tuning_records=len(tune),confirmation_records=len(confirm),goal_achieved=False)


def assert_contract(data):
    saved=json.loads((ROOT/'contract.json').read_text())
    current=source_contract(data)
    for key,value in current.items():
        if saved[key]!=value:raise ValueError('curve adaptation contract changed: '+key)
    return saved


def caches(data,ix,seed):
    d,c,h,ext,_,bins,tune,confirm,search,_=data
    # 每次重新抽取均恢复原教师前向精度，不能继承小头的FP32设置。
    torch.backends.cuda.matmul.allow_tf32=True
    src=BASE_SEEDS[seed]
    base=restore(Path(search['input_config']['parent']),src,'survival_base',h,.1)
    original=teacher(base,seed,bins).eval().requires_grad_(False)
    frozen={k:v.detach().clone() for k,v in original.state_dict().items()}
    cached,proof=extract(d,c,ext,ix,original)
    if any(not torch.equal(v,original.state_dict()[k]) for k,v in frozen.items()):raise ValueError('teacher changed')
    del original
    torch.backends.cuda.matmul.allow_tf32=False
    return cached,proof


@torch.no_grad()
def prediction(model,cache):
    a=[];k=[];nll=0.;count=0
    for b in cache:
        out=model(b);a.append(out.log_amplitude.cpu().numpy());k.append(out.shape.cpu().numpy())
        nll+=float(competing_nll(out,b.times,b.events))*len(b.times);count+=len(b.times)
    aa,kk=np.concatenate(a),np.concatenate(k)
    return aa,kk,probability_curve(aa,kk),nll/count


def score(model,cache):
    a,k,curve,nll=prediction(model,cache)
    t=np.concatenate([b.times.cpu().numpy() for b in cache]);e=np.concatenate([b.events.cpu().numpy() for b in cache])
    detail=metrics(t,e,curve[:,:,1]);death=metrics(t,e,curve[:,:,2],cause=2)
    return dict(nll=nll,ibs=detail['integrated_brier_7_365'],death_ibs=death['integrated_brier_7_365'],
                auc90=detail['points']['90']['auc']),a,k,curve


def objective(model,b):
    out=model(b)
    primary=competing_nll(out,b.times,b.events) if model.arm=='clinical_nll' else curve_scores(daily_cif(out),b.times,b.events)['joint'].mean()
    return primary+1e-4*out.delta.square().mean()


def preflight():
    if (ROOT/'contract.json').exists():raise RuntimeError('preflight already exists; inspect instead of overwrite')
    data,ix=setup_inputs();ROOT.mkdir(exist_ok=True)
    contract=source_contract(data);proof={}
    for seed in SEEDS:
        cache,cp=caches(data,ix[:256],seed);row=cache[0];reference=None;rows={}
        for arm in ARMS:
            m=new_model(arm,seed);out=m(row);curve=daily_cif(out).detach()
            if reference is None:reference=curve
            torch.testing.assert_close(curve,reference,atol=0,rtol=0)
            if any(torch.count_nonzero(p) for p in m.parameters()):raise ValueError('initial correction nonzero')
            opt=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=1e-4)
            loss=objective(m,row);loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()):raise ValueError('invalid first gradients')
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            if not any(torch.count_nonzero(p) for p in m.parameters()):raise ValueError('first step did not update')
            rows[arm]=dict(parameters=sum(p.numel() for p in m.parameters()),initial_curve_error=0.,
                           first_loss=float(loss.detach()),first_step_finite=True)
        proof[str(seed)]=dict(cache=cp,arms=rows)
    contract['preflight']=proof;write_json(ROOT/'contract.json',contract)
    write_json(ROOT/'state.json',dict(status='preflight_passed',goal_achieved=False))
    print(json.dumps(dict(status='preflight_passed',proof=proof)),flush=True)


def fit(arm,seed,train,tune,path):
    if path.exists():raise RuntimeError('training output exists; do not silently retrain')
    path.mkdir(parents=True);m=new_model(arm,seed)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
    rng=np.random.default_rng(seed);history=[];best=math.inf;chosen=0;start=time.perf_counter()
    def record(epoch,online=None):
        nonlocal best,chosen
        info,_,_,_=score(m,tune)
        history.append(dict(epoch=epoch,train_online_objective=online,**{'tuning_'+k:v for k,v in info.items()}))
        if info['ibs']<best:
            best,chosen=info['ibs'],epoch
            torch.save(dict(state_dict=m.state_dict(),epoch=epoch,arm=arm,seed=seed,
                            contract_sha256=sha256(ROOT/'contract.json')),path/'best.pt')
        if epoch%10==0:
            write_json(path/'history.json',history)
            print(json.dumps(dict(seed=seed,arm=arm,epoch=epoch,best_epoch=chosen,ibs=info['ibs'],
                                  auc90=info['auc90'],seconds=round(time.perf_counter()-start,1))),flush=True)
    record(0)
    for epoch in range(1,EPOCHS+1):
        lr=1e-4*min(1.,epoch/5)*(.1+.45*(1+math.cos(math.pi*max(0,epoch-5)/(EPOCHS-5))))
        for group in opt.param_groups:group['lr']=lr
        total=0.;n=0
        for j in rng.permutation(len(train)):
            b=train[j];opt.zero_grad(set_to_none=True);loss=objective(m,b)
            if not torch.isfinite(loss):raise ValueError('nonfinite objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
            opt.step();total+=float(loss.detach())*len(b.times);n+=len(b.times)
        record(epoch,total/n)
    write_json(path/'history.json',history)
    result=dict(arm=arm,seed=seed,completed_epochs=EPOCHS,best_epoch=chosen,
                checkpoint_sha256=sha256(path/'best.pt'),elapsed_seconds=time.perf_counter()-start)
    write_json(path/'training_result.json',result)


def train():
    data,ix=setup_inputs();contract=assert_contract(data)
    if json.loads((ROOT/'state.json').read_text())['status']!='preflight_passed':raise RuntimeError('not a fresh passed run')
    for seed in SEEDS:
        train_cache,tp=caches(data,ix,seed);tune_cache,vp=caches(data,data[6],seed)
        write_json(ROOT/f'cache_seed{seed}.json',dict(train=tp,tuning=vp))
        for arm in ARMS:
            assert_contract(data)
            write_json(ROOT/'state.json',dict(status='running',seed=seed,arm=arm,goal_achieved=False))
            fit(arm,seed,train_cache,tune_cache,ROOT/f'seed{seed}'/arm)
        del train_cache,tune_cache
    write_json(ROOT/'state.json',dict(status='training_complete',new_fits=9,epochs_per_fit=100,goal_achieved=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('preflight','train'),required=True);args=p.parse_args()
    try:preflight() if args.mode=='preflight' else train()
    except Exception as error:
        if ROOT.exists():write_json(ROOT/'failure.json',dict(mode=args.mode,type=type(error).__name__,error=str(error),goal_achieved=False))
        raise
