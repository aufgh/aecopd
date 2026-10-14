"""患者均衡×编码器可塑性有限搜索；固定逐出院验证，原V6不覆盖。"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.warm_residual_experiment import ROOT as OLD
from training.v6.warm_residual_audit import candidate,teacher,inputs as old_inputs
from training.v6.warm_residual import BASE_SEEDS,WarmTrial
from training.v6.patient_balance_model import TRIALS,build
from training.v6.patient_weight_loss import training_weights,weighted_nll
from training.v6.module_surgery_experiment import batches,validation
from training.v6.horizon_objective import restore
from training.v6.module_surgery import member_nll
from training.v6.train import write_json

ROOT=Path('training_runs/v6_patient_balance_2026-10-13')
SEEDS=(42,7,101)
PROTOCOL=Path('docs/V6_PATIENT_BALANCE_PROTOCOL_2026-10-13.md')


def fit(base,seed,trial,bins,cache,weights,d,tune,path):
    model=build(base,seed,trial,bins);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=trial.lr,weight_decay=trial.weight_decay)
    frozen={key:value.detach().clone() for key,value in model.state_dict().items()
        if key.startswith('baseline.') or (trial.head_only and key.startswith('encoder.'))}
    rows=[];best=math.inf;chosen=0;started=time.perf_counter()
    def record(epoch,nll=None,wnll=None,rate=0.):
        nonlocal best,chosen
        detail=validation(model,cache['tune'],d.times[tune],d.events[tune])
        rows.append(dict(epoch=epoch,train_nll=nll,train_weighted_nll=wnll,lr=rate,
            **{'tuning_'+key:value for key,value in detail.items()}))
        if detail['ibs']<best:
            best,chosen=detail['ibs'],epoch
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,tuning_ibs=best,trial=asdict(trial)),path/'best.pt')
        if epoch%20==0:
            write_json(path/'history.json',rows)
            print(json.dumps(dict(seed=seed,trial=trial.name,epoch=epoch,best_epoch=chosen,tuning_ibs=detail['ibs'],
                seconds=round(time.perf_counter()-started,1))),flush=True)
    path.mkdir(parents=True,exist_ok=True);record(0)
    for epoch in range(1,101):
        model.train();rate=trial.lr*min(1.,epoch/5)*(.1+.45*(1+math.cos(math.pi*max(0,epoch-5)/95)))
        for group in optimizer.param_groups:group['lr']=rate
        total=weighted_total=count=0.
        for j in rng.permutation(len(cache['train'])):
            pos,h,b,t,e=cache['train'][j];optimizer.zero_grad(set_to_none=True)
            out=model(h,b);nll=member_nll(out,t,e)
            wnll=weighted_nll(out,t,e,weights[torch.as_tensor(pos,device='cuda')])
            objective=wnll+trial.delta_penalty*out.delta.square().mean()
            if not torch.isfinite(objective):raise ValueError('nonfinite weighted clinical objective')
            objective.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite weighted clinical gradient')
            optimizer.step();total+=float(nll.detach())*len(t);weighted_total+=float(wnll.detach())*len(t);count+=len(t)
        record(epoch,total/count,weighted_total/count,rate)
    for key,value in frozen.items():
        if not torch.equal(value,model.state_dict()[key]):raise ValueError('frozen state changed during training: '+key)
    best_state=torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict']
    for key,value in frozen.items():
        if not torch.equal(value,best_state[key]):raise ValueError('frozen selected state changed: '+key)
    write_json(path/'history.json',rows)
    return dict(seed=seed,baseline_seed=BASE_SEEDS[seed],trial=asdict(trial),completed_epochs=100,best_epoch=chosen,
        tuning_ibs=best,checkpoint_sha256=sha256(path/'best.pt'),elapsed_seconds=time.perf_counter()-started,
        frozen_end_and_best_verified=True,newly_trained=True)


def run():
    (d,c,h,ext,_,bins,tune,_,search,_),_=old_inputs()
    preflight=json.loads((ROOT/'preflight.json').read_text())
    if preflight['status']!='gpu_initialization_and_ordinary_step_verified':raise ValueError('GPU preflight not completed')
    for p,digest in preflight['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('preflight source changed')
    if preflight['old_control_verification_sha256']!=sha256(OLD/'tuning_verification.json'):
        raise ValueError('ordinary reused control proof changed')
    source=('training/v6/patient_balance_model.py','training/v6/patient_weight_loss.py','training/v6/patient_balance_experiment.py',
        'training/v6/patient_balance_preflight.py','training/tests/test_v6_patient_weights.py',str(PROTOCOL),
        'training/v6/warm_residual.py','training/v6/regularization_multiseed.py')
    contract=dict(source_sha256={p:sha256(Path(p)) for p in source},preflight_sha256=sha256(ROOT/'preflight.json'),
        search_contract_sha256=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')),
        old_warm_contract_sha256=sha256(OLD/'contract.json'),trials=[asdict(t) for t in TRIALS],seeds=list(SEEDS),epochs=100,
        teachers={str(s):sha256(REFERENCE/f'seed{BASE_SEEDS[s]}/control/best.pt') for s in SEEDS},
        full_uniform_controls={str(s):sha256(OLD/f'seed{s}/warm_finetune/best.pt') for s in SEEDS},
        selection='unweighted admission tuning IBS three-seed mean',confirmation_not_evaluated=True,test_not_evaluated=True)
    train=np.flatnonzero(d.arrays['patient_splits']=='train')
    weights={}
    arrays={}
    for power in (0.,.5,1.):
        w=np.zeros(len(d.history),np.float32);w[train]=training_weights(d.arrays['subject_ids'][train],power)
        arrays[str(power)]=w;weights[power]=torch.as_tensor(w,device='cuda')
    wp=ROOT/'training_weights.npz'
    if wp.exists():
        with np.load(wp) as z:
            for key,value in arrays.items():np.testing.assert_array_equal(z[key],value)
    else:np.savez_compressed(wp,**arrays)
    contract['training_weights_sha256']=sha256(wp)
    if (ROOT/'contract.json').exists():
        if json.loads((ROOT/'contract.json').read_text())!=contract:raise ValueError('balance resume contract differs')
    else:write_json(ROOT/'contract.json',contract)
    cache=dict(train=batches(d,c,ext,train),tune=batches(d,c,ext,tune));fits={};references={}
    old_trial=WarmTrial('warm_finetune',mode='finetune',lr=2e-5)
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        references[str(seed)]={}
        for name in ('original_teacher','full_uniform'):
            if name=='original_teacher':model=teacher(base,seed,bins);info=None
            else:model,info,_=candidate(base,seed,old_trial,bins)
            references[str(seed)][name]=dict(tuning=validation(model,cache['tune'],d.times[tune],d.events[tune]),
                newly_trained=False,reused_training=info,reused_from=str(OLD/f'seed{seed}/warm_finetune') if info else None)
            del model
        fits[str(seed)]={}
        for trial in TRIALS:
            path=ROOT/f'seed{seed}'/trial.name
            if (path/'training_result.json').exists():
                info=json.loads((path/'training_result.json').read_text())
                if info['trial']!=asdict(trial) or info['completed_epochs']!=100 or info['checkpoint_sha256']!=sha256(path/'best.pt'):
                    raise ValueError('completed balance fit differs')
            else:
                if path.exists() and any((path/p).exists() for p in ('history.json','best.pt')):
                    raise RuntimeError('partial balance fit exists; preserve scene, no automatic restart')
                write_json(ROOT/'state.json',dict(status='running',seed=seed,trial=trial.name,completed=fits,goal_achieved=False))
                info=fit(base,seed,trial,bins,cache,weights[trial.power],d,tune,path)
                write_json(path/'training_result.json',info)
            fits[str(seed)][trial.name]=info
    scores={name:float(np.mean([references[str(s)][name]['tuning']['ibs'] for s in SEEDS])) for name in ('original_teacher','full_uniform')}
    scores.update({t.name:float(np.mean([fits[str(s)][t.name]['tuning_ibs'] for s in SEEDS])) for t in TRIALS})
    chosen=min(scores,key=scores.get)
    write_json(ROOT/'selected_before_confirmation.json',dict(fits=fits,references=references,tuning_scores=scores,selected=chosen,
        newly_trained_models=15,old_full_uniform_controls_reused=3,confirmation_not_evaluated=True,test_not_evaluated=True))
    write_json(ROOT/'state.json',dict(status='training_complete_pending_independent_replay',selected=chosen,goal_achieved=False))
    print(json.dumps(dict(status='balance_training_complete_pending_replay',selected=chosen)),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();ROOT.mkdir(exist_ok=True);lock=ROOT/'.running.lock'
    with lock.open('x',encoding='utf-8') as owner:owner.write(json.dumps(dict(pid=os.getpid(),started=time.time())))
    try:run()
    except Exception as error:
        write_json(ROOT/'state.json',dict(status='failed',error_type=type(error).__name__,error=str(error),goal_achieved=False));raise
    finally:lock.unlink()
