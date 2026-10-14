"""冻结全年风险的时钟训练：H/C/HC/乘积四输入对照。"""
import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.warm_residual_experiment import ROOT as PRIOR
from training.v6.warm_residual import BASE_SEEDS
from training.v6.module_surgery import SurgeryModel,probability_curve
from training.v6.module_surgery_experiment import batches,summary
from training.v6.horizon_objective import restore
from training.v6.risk import RiskOutput,predict_cif
from training.v6.anchored_clock import clock_nll,clock_curve,anchor_dense_to_daily
from training.v6.train import write_json

ROOT=Path('training_runs/v6_anchored_clock_2026-10-13_r2')
SEEDS=(42,7,101)
ARMS=('H_clock','C_clock','HC_clock','product_clock')


class ClockHead(nn.Module):
    def __init__(self,width):
        super().__init__();self.net=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,2))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return .25*torch.tanh(self.net(x))


def features(cache,arm):
    h,c=cache['history'],cache['clinical']
    if arm=='H_clock':return h
    if arm=='C_clock':return c
    if arm=='HC_clock':return torch.cat((h,c),1)
    if arm=='product_clock':
        context=torch.tanh(cache['base_amplitude']/4.)
        return torch.cat((c,c*context[:,0:1],c*context[:,1:2]),1)
    raise ValueError('unknown clock arm')


@torch.no_grad()
def extract(model,cache):
    model.eval();out={k:[] for k in ('history','clinical','base_amplitude','amplitude','shape','times','events')}
    for _,h,b,t,e in cache:
        prior=model(h,b);c,_=model.encoder(b.base);a=model.baseline(h).log_amplitude
        for key,value in zip(out,(h,c,a,prior.log_amplitude,prior.shape,t,e)):out[key].append(value.detach())
    return {k:torch.cat(v,0) for k,v in out.items()}


def dense_teacher(cache):
    a,k=cache['amplitude'].cpu().numpy(),cache['shape'].cpu().numpy()
    daily=probability_curve(a,k)
    grid=np.unique(np.r_[np.arange(0,365.001,.25),np.geomspace(1e-10,.5,80)])
    dense=predict_cif(a,k,grid,step_days=.25)
    positions=np.searchsorted(grid,np.arange(366))
    difference=float(np.abs(dense[:,positions]-daily).max())
    if difference>1e-4:raise ValueError('teacher dense quadrature disagrees with official daily curve')
    # 不直接覆盖整数节点，否则两种积分步长的微差会产生锯齿下降。
    dense=anchor_dense_to_daily(dense,grid,daily)
    return grid,dense,daily,difference


def validate(model,x,cache,grid,dense):
    model.eval()
    with torch.no_grad():log_clock=model(x).cpu().numpy()
    curve=clock_curve(dense,grid,log_clock,np.arange(366))
    t,e=cache['times'].cpu().numpy(),cache['events'].cpu().numpy()
    y=((e[:,None]==1)&(t[:,None]<=np.arange(366))).astype(float)
    brier=np.mean((curve[:,:,1]-y)**2,0)
    from training.baselines.rank_metrics import auc_rank
    return dict(ibs=float(np.trapezoid(brier[7:],np.arange(7,366))/358),brier90=float(brier[90]),
                auc90=float(auc_rank(curve[:,90,1],y[:,90].astype(np.int8)))),log_clock,curve


def fit(seed,arm,train,tune,grid,dense,path):
    x,tx=features(train,arm),features(tune,arm);torch.manual_seed(seed)
    model=ClockHead(x.shape[1]).cuda();torch.manual_seed(seed);rng=np.random.default_rng(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    rows=[];best=math.inf;chosen=0;start=time.perf_counter()
    def record(epoch,nll=None):
        nonlocal best,chosen
        score,_,_=validate(model,tx,tune,grid,dense)
        rows.append(dict(epoch=epoch,train_nll=nll,**{'tuning_'+key:value for key,value in score.items()}))
        if score['ibs']<best:
            best,chosen=score['ibs'],epoch
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,arm=arm,width=x.shape[1]),path/'best.pt')
        if epoch%20==0:
            write_json(path/'history.json',rows)
            print(json.dumps(dict(seed=seed,arm=arm,epoch=epoch,best_epoch=chosen,tuning_ibs=score['ibs'],seconds=round(time.perf_counter()-start,1))),flush=True)
    path.mkdir(parents=True,exist_ok=True);record(0)
    for epoch in range(1,101):
        model.train();rate=1e-3*min(1.,epoch/5)*(.1+.45*(1+math.cos(math.pi*max(0,epoch-5)/95)))
        for g in optimizer.param_groups:g['lr']=rate
        total=0.
        for start_index in range(0,len(x),256):
            # 每轮一次随机排列，不引入患者标签作为特征。
            if start_index==0:order=rng.permutation(len(x))
            ix=torch.as_tensor(order[start_index:start_index+256],device='cuda')
            optimizer.zero_grad(set_to_none=True);log_clock=model(x[ix])
            loss=clock_nll(RiskOutput(train['amplitude'][ix],train['shape'][ix]),log_clock,train['times'][ix],train['events'][ix])
            objective=loss+.01*log_clock.square().mean()
            if not torch.isfinite(objective):raise ValueError('nonfinite clock objective')
            objective.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite clock gradient')
            optimizer.step();total+=float(loss.detach())*len(ix)
        record(epoch,total/len(x))
    write_json(path/'history.json',rows)
    return dict(seed=seed,arm=arm,width=x.shape[1],completed_epochs=100,best_epoch=chosen,tuning_ibs=best,
                checkpoint_sha256=sha256(path/'best.pt'),elapsed_seconds=time.perf_counter()-start)


def run():
    proof=json.loads((PRIOR/'tuning_verification.json').read_text())
    if proof['status']!='tuning_independently_verified' or proof['models']!=15:raise ValueError('warm batch not verified')
    d,c,h,ext,_,bins,tune,_,search,_=checked()
    protected=('training/v6/clock_experiment.py','training/v6/anchored_clock.py','training/v6/warm_residual.py',
               'training/tests/test_v6_anchored_clock.py','docs/V6_CLOCK_TRAINING_PROTOCOL_2026-10-13.md')
    contract=dict(source_sha256={p:sha256(Path(p)) for p in protected},search_contract_sha256=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')),
        seeds=list(SEEDS),arms=list(ARMS),epochs=100,selection='three-seed tuning IBS; original teacher included',
        teachers={str(s):sha256(REFERENCE/f'seed{BASE_SEEDS[s]}/control/best.pt') for s in SEEDS},
        frozen_annual_probabilities=True,confirmation_not_evaluated=True,test_not_evaluated=True)
    if (ROOT/'contract.json').exists():
        if json.loads((ROOT/'contract.json').read_text())!=contract:raise ValueError('clock resume contract changed')
    else:write_json(ROOT/'contract.json',contract)
    cached={name:batches(d,c,ext,ix) for name,ix in dict(train=np.flatnonzero(d.arrays['patient_splits']=='train'),tune=tune).items()}
    results={};references={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        teacher=SurgeryModel(base,BASE_SEEDS[seed],'control',bins).cuda()
        teacher.load_state_dict(torch.load(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt',map_location='cuda',weights_only=True)['state_dict'])
        data={f:extract(teacher,cache) for f,cache in cached.items()};grid,dense,daily,error=dense_teacher(data['tune']);del teacher
        prior=RiskOutput(data['tune']['amplitude'],data['tune']['shape'])
        from training.v6.risk import competing_nll
        torch.testing.assert_close(clock_nll(prior,torch.zeros_like(prior.shape),data['tune']['times'],data['tune']['events']),
                                   competing_nll(prior,data['tune']['times'],data['tune']['events']),atol=1e-6,rtol=1e-6)
        score,_,identity=validate(ClockHead(34).cuda(),features(data['tune'],'H_clock'),data['tune'],grid,dense)
        np.testing.assert_allclose(identity,daily,atol=1e-12,rtol=0)
        references[str(seed)]=dict(tuning=score,dense_teacher_maximum_difference=error,newly_trained=False)
        results[str(seed)]={}
        for arm in ARMS:
            path=ROOT/f'seed{seed}'/arm
            if (path/'training_result.json').exists():
                info=json.loads((path/'training_result.json').read_text())
                if sha256(path/'best.pt')!=info['checkpoint_sha256'] or info['completed_epochs']!=100:raise ValueError('completed clock model changed')
            else:
                if path.exists() and any((path/p).exists() for p in ('best.pt','history.json')):
                    import shutil
                    backup=path/'recovery_snapshots'/str(time.time_ns());backup.mkdir(parents=True)
                    for p in ('best.pt','history.json'):
                        if (path/p).exists():shutil.copy2(path/p,backup/p)
                write_json(ROOT/'state.json',dict(status='running',seed=seed,arm=arm,completed=results,goal_achieved=False))
                info=fit(seed,arm,data['train'],data['tune'],grid,dense,path);write_json(path/'training_result.json',info)
            results[str(seed)][arm]=info
        del data,dense
    scores={'original_teacher':float(np.mean([references[str(s)]['tuning']['ibs'] for s in SEEDS]))}
    scores.update({arm:float(np.mean([results[str(s)][arm]['tuning_ibs'] for s in SEEDS])) for arm in ARMS})
    selected=min(scores,key=scores.get)
    write_json(ROOT/'selected_before_confirmation.json',dict(results=results,references=references,tuning_scores=scores,selected=selected,
        all_100_candidate_epochs_complete=True,confirmation_not_evaluated=True,test_not_evaluated=True))
    write_json(ROOT/'state.json',dict(status='training_complete_pending_grid_and_replay_audit',selected=selected,goal_achieved=False))
    print(json.dumps(dict(status='training_complete_pending_grid_and_replay_audit',selected=selected)),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();ROOT.mkdir(exist_ok=True);lock=ROOT/'.running.lock'
    with lock.open('x',encoding='utf-8') as owner:owner.write(json.dumps(dict(pid=os.getpid(),started=time.time())))
    try:run()
    except Exception as e:write_json(ROOT/'state.json',dict(status='failed',error_type=type(e).__name__,error=str(e),goal_achieved=False));raise
    finally:lock.unlink()
