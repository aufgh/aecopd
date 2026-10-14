"""长期探索首批：固定清单，仅验证调参半折；不访问测试指标。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_model import TRIALS,make_model,probability_consistency,auxiliary_brier
from training.v6.module_surgery import member_nll,probability_curve
from training.v6.module_surgery_experiment import inputs,batches,parameters,validation,setup
from training.v6.horizon_objective import restore
from training.v6.train import write_json

ROOT=Path('training_runs/v6_regularization_search_2026-10-12')
PLAN=Path('docs/V6_LONG_TERM_OPTIMIZATION_PLAN_2026-10-12.md')
SEED=42
EPOCHS=60


def validation_parts(subjects,splits):
    if subjects.shape!=splits.shape:raise ValueError('patient/split mismatch')
    patients=np.unique(subjects[splits=='validation'])
    choose={p:int(hashlib.sha256(f'v6-regularization-confirm-20261012:{p}'.encode()).hexdigest()[:8],16)%2 for p in patients}
    selected=np.array([splits[i]=='validation' and choose[subjects[i]]==0 for i in range(len(subjects))])
    confirm=(splits=='validation') & ~selected
    if not selected.any() or not confirm.any():raise ValueError('empty validation subset')
    if set(subjects[selected]) & set(subjects[confirm]):raise ValueError('patient split overlap')
    return np.flatnonzero(selected),np.flatnonzero(confirm)


def build(base,trial,bins):
    m=make_model(base,SEED,trial,bins).cuda();torch.manual_seed(SEED)
    return m


def train_one(base,trial,bins,cache,times,events,path):
    path.mkdir(parents=True,exist_ok=True)
    m=build(base,trial,bins)
    torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=trial.lr,weight_decay=trial.weight_decay)
    frozen={k:v.detach().clone() for k,v in m.baseline.state_dict().items()}
    history,best,chosen=[],math.inf,0
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def record(epoch,online=None,penalty=None,lr=0.):
        nonlocal best,chosen
        info=validation(m,cache['tune'],times,events)
        _,_,fixed=parameters(m,cache['train'][:8])
        history.append(dict(epoch=epoch,train_online_nll=online,train_fixed2048_nll=fixed,
            auxiliary_penalty=penalty,lr=lr,**{'tuning_'+k:v for k,v in info.items()}))
        if info['ibs']<best:
            best,chosen=info['ibs'],epoch
            torch.save(dict(state_dict=m.state_dict(),epoch=epoch,tuning_ibs=best,trial=asdict(trial)),path/'best.pt')
        if epoch%10==0:
            write_json(path/'history.json',history)
            print(json.dumps(dict(trial=trial.name,epoch=epoch,best_epoch=chosen,tuning_ibs=info['ibs'],
                tuning_auc90=info['auc90'],seconds=round(time.perf_counter()-start,1))),flush=True)
    record(0)
    for epoch in range(1,EPOCHS+1):
        m.train()
        lr=trial.lr*min(1.,epoch/5)*(.1+.45*(1+math.cos(math.pi*max(0,epoch-5)/(EPOCHS-5))))
        for g in opt.param_groups:g['lr']=lr
        total,count,regularization=0.,0,0.
        for j in rng.permutation(len(cache['train'])):
            _,h,b,t,e=cache['train'][j];opt.zero_grad(set_to_none=True)
            out=m(h,b);nll=member_nll(out,t,e)
            penalty=trial.delta_penalty*out.delta.square().mean()
            if trial.consistency:
                other=m(h,b);nll=(nll+member_nll(other,t,e))*.5
                penalty+=trial.consistency*probability_consistency(out,other)
            if trial.brier:penalty+=trial.brier*auxiliary_brier(out,t,e)
            loss=nll+penalty
            if not torch.isfinite(loss):raise ValueError('nonfinite regularized loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite gradients')
            if any(p.grad is not None for p in m.baseline.parameters()):raise ValueError('baseline gradient leakage')
            opt.step();total+=float(nll.detach())*len(t);regularization+=float(penalty.detach())*len(t);count+=len(t)
        record(epoch,total/count,regularization/count,lr)
    if any(not torch.equal(v,m.baseline.state_dict()[k]) for k,v in frozen.items()):raise ValueError('baseline changed')
    chosen_state=torch.load(path/'best.pt',map_location='cuda',weights_only=True);m.load_state_dict(chosen_state['state_dict'])
    a,k,nll=parameters(m,cache['tune']);curve=probability_curve(a,k)
    np.savez_compressed(path/'tuning_predictions.npz',log_amplitude=a,shape=k,curve=curve,times=times,events=events)
    info=validation(m,cache['tune'],times,events)
    if abs(info['ibs']-best)>1e-12:raise ValueError('best checkpoint replay failed')
    result=dict(trial=asdict(trial),completed_epochs=EPOCHS,best_epoch=chosen,tuning=info,
        checkpoint_sha256=sha256(path/'best.pt'),frozen_baseline_verified=True,
        trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
        elapsed_seconds=time.perf_counter()-start,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    write_json(path/'history.json',history);write_json(path/'training_result.json',result)
    del m;torch.cuda.empty_cache()
    return result


def run():
    setup();torch.backends.mha.set_fastpath_enabled(False)
    d,c,h,ext,config,old,prov=inputs()
    tune,confirm=validation_parts(d.arrays['subject_ids'],d.arrays['patient_splits'])
    train=np.flatnonzero(d.arrays['patient_splits']=='train')
    prep=json.loads(Path('training_runs/v6_module_surgery_2026-10-12_r3/preprocessing.json').read_text())
    protected=dict(old['contract']['code_sha256'])
    for p in ('training/v6/regularization_model.py','training/v6/regularization_search.py',
              'training/v6/deep_interaction.py','training/tests/test_v6_regularization.py'):
        protected[p]=sha256(Path(p))
    contract=dict(source_sha256=protected,plan_sha256=sha256(PLAN),data_provenance=d.provenance,
        compact_provenance=c.provenance,extensions=prov,input_config=config,seed=SEED,epochs=EPOCHS,
        trials=[asdict(x) for x in TRIALS],selection='tuning-half validation readmission IBS7-365',
        admissions=dict(train=len(train),tune=len(tune),confirmation=len(confirm)),
        validation_partition='patient SHA256 modulo2; original validation previously exposed, not blind',
        frozen_baseline_sha256=sha256(Path(config['parent'])/f'seed{SEED}/survival_base/best.pt'),
        preprocessing_sha256=sha256(Path('training_runs/v6_module_surgery_2026-10-12_r3/preprocessing.json')),
        endpoint='strict491.21/J44.1; competing death; WorldA365',
        no_test_evaluation=True,confirmation_not_used_for_search=True)
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'contract.json').exists():
        if json.loads((ROOT/'contract.json').read_text())!=contract:raise ValueError('resume contract changed')
    else:
        write_json(ROOT/'contract.json',contract)
        np.savez_compressed(ROOT/'validation_partition.npz',tuning_indices=tune,confirmation_indices=confirm,
                            subject_ids=d.arrays['subject_ids'])
    cache=dict(train=batches(d,c,ext,train),tune=batches(d,c,ext,tune))
    base=restore(Path(config['parent']),SEED,'survival_base',h,.1)
    completed={}
    for trial in TRIALS:
        path=ROOT/trial.name
        if (path/'training_result.json').exists():
            result=json.loads((path/'training_result.json').read_text())
            if result['trial']!=asdict(trial) or result['checkpoint_sha256']!=sha256(path/'best.pt'):
                raise ValueError('saved completed trial differs')
            completed[trial.name]=result
            continue
        if path.exists() and any((path/name).exists() for name in ('best.pt','history.json')):
            backup=path/'recovery_snapshots'/str(time.time_ns())
            backup.mkdir(parents=True)
            for name in ('best.pt','history.json'):
                if (path/name).exists():shutil.copy2(path/name,backup/name)
        write_json(ROOT/'state.json',dict(status='running',current_trial=trial.name,completed_trials=list(completed),
            total_trials=len(TRIALS),next_phase='lock top candidates, do not evaluate test',goal_achieved=False))
        completed[trial.name]=train_one(base,trial,prep['bins'],cache,d.times[tune],d.events[tune],path)
    ranked=sorted(completed,key=lambda x:completed[x]['tuning']['ibs'])
    # 至多三个、不同机制的候选；阈值按调参折筛，不看确认/测试。
    nominated=[x for x in ranked if x!='v6_control' and completed[x]['tuning']['ibs']<completed['v6_control']['tuning']['ibs']][:3]
    write_json(ROOT/'screening_report.json',dict(status='screening_complete_not_confirmed',contract=contract,
        results=completed,ranked_by_tuning_ibs=ranked,nominated_before_confirmation=nominated,
        original_reference_checkpoint='prior official V6 retained; no test scores consumed',goal_achieved=False))
    write_json(ROOT/'state.json',dict(status='screening_complete',completed_trials=list(completed),total_trials=len(TRIALS),
        nominated=nominated,next_phase='fresh-process replay then locked candidate confirmation/multiseed',goal_achieved=False))
    print(json.dumps(dict(status='screening_complete',nominated=nominated,goal_achieved=False)),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    ROOT.mkdir(exist_ok=True)
    lock=ROOT/'.running.lock'
    with lock.open('x',encoding='utf-8') as owner:
        owner.write(json.dumps(dict(pid=os.getpid(),started_unix=time.time(),purpose='single regularization batch')))
    try:
        run()
    except Exception as error:
        write_json(ROOT/'state.json',dict(status='failed',error_type=type(error).__name__,error=str(error),
            next_phase='inspect exact logs before retry; preserve partial checkpoints',goal_achieved=False))
        raise
    finally:
        lock.unlink()
