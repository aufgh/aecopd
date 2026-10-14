"""锁定正则候选五种子确认：同基线配对，继续不用测试选择。"""
from __future__ import annotations
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
from training.v6.regularization_model import make_model,Trial,probability_consistency,auxiliary_brier
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.module_surgery_experiment import batches,parameters,validation,summary,save
from training.v6.module_surgery import member_nll,probability_curve,SurgeryModel
from training.v6.horizon_objective import restore
from training.v6.evaluation import paired_interval
from training.v6.train import write_json

SEARCH=Path('training_runs/v6_regularization_search_2026-10-12')
ROOT=Path('training_runs/v6_regularization_multiseed_2026-10-13')
PROTOCOL=Path('docs/V6_REGULARIZATION_CONFIRMATION_PROTOCOL_2026-10-13.md')
SEEDS=(42,7,101,202,303)
BASE_SEEDS={42:42,7:7,101:42,202:7,303:42}
EPOCHS=100


def build(base,seed,trial,bins):
    model=make_model(base,seed,trial,bins).cuda();torch.manual_seed(seed)
    return model


def fit(base,seed,trial,bins,cache,times,events,path,*,model_factory=None,backward=None):
    path.mkdir(parents=True,exist_ok=True)
    model=(model_factory or build)(base,seed,trial,bins);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=trial.lr,weight_decay=trial.weight_decay)
    frozen={k:v.detach().clone() for k,v in model.baseline.state_dict().items()}
    rows,best,chosen=[],math.inf,0;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def record(epoch,online=None,regular=None,lr=0.):
        nonlocal best,chosen
        score=validation(model,cache['tune'],times,events)
        # 小样本诊断不足10病例/对照时AUC不可计算；不能写NaN或冒充0。
        if not math.isfinite(score['auc90']):score['auc90']=None
        rows.append(dict(epoch=epoch,train_nll=online,penalty=regular,lr=lr,
                         **{'tuning_'+k:v for k,v in score.items()}))
        if score['ibs']<best:
            best,chosen=score['ibs'],epoch
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,tuning_ibs=best,trial=asdict(trial)),path/'best.pt')
        if epoch%20==0:
            write_json(path/'history.json',rows)
            print(json.dumps(dict(seed=seed,trial=trial.name,epoch=epoch,best_epoch=chosen,tuning_ibs=score['ibs'],
                seconds=round(time.perf_counter()-start,1))),flush=True)
    record(0)
    for epoch in range(1,EPOCHS+1):
        model.train()
        lr=trial.lr*min(1.,epoch/5)*(.1+.45*(1+math.cos(math.pi*max(0,epoch-5)/(EPOCHS-5))))
        for g in opt.param_groups:g['lr']=lr
        loss_sum,reg_sum,count=0.,0.,0
        for j in rng.permutation(len(cache['train'])):
            _,h,b,t,e=cache['train'][j];opt.zero_grad(set_to_none=True)
            out=model(h,b);nll=member_nll(out,t,e);penalty=trial.delta_penalty*out.delta.square().mean()
            if trial.consistency:
                other=model(h,b);nll=(nll+member_nll(other,t,e))*.5
                penalty+=trial.consistency*probability_consistency(out,other)
            if trial.brier:penalty+=trial.brier*auxiliary_brier(out,t,e)
            loss=nll+penalty
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            if backward is None:loss.backward()
            else:backward(model,out,nll,penalty,t,e,trial)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
            opt.step();loss_sum+=float(nll.detach())*len(t);reg_sum+=float(penalty.detach())*len(t);count+=len(t)
        record(epoch,loss_sum/count,reg_sum/count,lr)
    if any(not torch.equal(v,model.baseline.state_dict()[k]) for k,v in frozen.items()):raise ValueError('baseline changed')
    info=dict(seed=seed,baseline_seed=BASE_SEEDS[seed],trial=asdict(trial),completed_epochs=EPOCHS,best_epoch=chosen,
        tuning_ibs=best,checkpoint_sha256=sha256(path/'best.pt'),elapsed_seconds=time.perf_counter()-start,
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,frozen_verified=True)
    write_json(path/'history.json',rows);write_json(path/'training_result.json',info)
    del model;torch.cuda.empty_cache()
    return info


def run():
    # 用确认分数只决定是否值得继续，不重新挑/替换候选。三个锁定候选均确认。
    prior=json.loads((SEARCH/'confirmation_report.json').read_text())
    lock=json.loads((SEARCH/'nomination_lock.json').read_text())
    nominated=lock['nominated']
    if not nominated:raise ValueError('no nominated candidates; next hypothesis needed')
    d,c,h,ext,_,bins,tune,confirm,search_contract,_=checked()
    screen=json.loads((SEARCH/'screening_report.json').read_text())
    if sha256(SEARCH/'screening_report.json')!=lock['screening_report_sha256']:raise ValueError('nomination stale')
    trials=[Trial(**screen['results'][name]['trial']) for name in ['v6_control',*nominated]]
    ROOT.mkdir(exist_ok=True)
    frozen={str(s):sha256(Path(search_contract['input_config']['parent'])/f'seed{BASE_SEEDS[s]}/survival_base/best.pt') for s in SEEDS}
    contract=dict(search_contract_sha256=sha256(SEARCH/'contract.json'),search_verification_sha256=sha256(SEARCH/'screening_verification.json'),
        nomination_lock_sha256=sha256(SEARCH/'nomination_lock.json'),trials=[asdict(t) for t in trials],seeds=list(SEEDS),
        baseline_seed_mapping=BASE_SEEDS,frozen_baselines=frozen,epochs=EPOCHS,selection='same tuning-half validation IBS only',
        source_sha256={str(Path(__file__)):sha256(Path(__file__)),'training/v6/regularization_model.py':sha256(Path('training/v6/regularization_model.py')),
            str(PROTOCOL):sha256(PROTOCOL),'training/v6/regularization_multiseed_verify.py':sha256(Path('training/v6/regularization_multiseed_verify.py'))},
        no_test_evaluation=True,limits=['Repeated two fixed H34 sources, five clinical seeds; not five independently refit cohorts.',
        'Validation subsets previously exposed, not blind.','Candidates locked before confirmation.'])
    if (ROOT/'contract.json').exists():
        if json.loads((ROOT/'contract.json').read_text())!=json.loads(json.dumps(contract)):
            raise ValueError('multiseed resume contract differs')
    else:write_json(ROOT/'contract.json',contract)
    cache=dict(train=batches(d,c,ext,np.flatnonzero(d.arrays['patient_splits']=='train')),tune=batches(d,c,ext,tune))
    fits={}
    for seed in SEEDS:
        base=restore(Path(search_contract['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        fits[str(seed)]={}
        for trial in trials:
            path=ROOT/f'seed{seed}'/trial.name
            if (path/'training_result.json').exists():
                completed=json.loads((path/'training_result.json').read_text())
                if (completed['trial']!=asdict(trial) or completed['completed_epochs']!=EPOCHS or
                        sha256(path/'best.pt')!=completed['checkpoint_sha256']):
                    raise ValueError('completed training result differs')
                fits[str(seed)][trial.name]=completed
                continue
            if path.exists() and any((path/name).exists() for name in ('best.pt','history.json')):
                import shutil
                backup=path/'recovery_snapshots'/str(time.time_ns());backup.mkdir(parents=True)
                for name in ('best.pt','history.json'):
                    if (path/name).exists():shutil.copy2(path/name,backup/name)
            write_json(ROOT/'state.json',dict(status='running',seed=seed,trial=trial.name,completed=fits,goal_achieved=False))
            fits[str(seed)][trial.name]=fit(base,seed,trial,bins,cache,d.times[tune],d.events[tune],path)
    write_json(ROOT/'selected_before_confirmation.json',dict(fits=fits,epochs_complete=True,no_confirmation_used_for_checkpoint_selection=True))
    del cache['train'];del cache['tune'];torch.cuda.empty_cache()
    cache=batches(d,c,ext,confirm)
    results,comparisons={},{}
    for seed in SEEDS:
        base=restore(Path(search_contract['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        results[str(seed)],curves={},{}
        for trial in trials:
            path=ROOT/f'seed{seed}'/trial.name;m=build(base,seed,trial,bins)
            m.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k)
            results[str(seed)][trial.name]=dict(training=fits[str(seed)][trial.name],metrics=summary(curve,d,confirm,nll))
            save(path/'confirmation_predictions.npz',d,confirm,a,k,curve);curves[trial.name]=curve;del m
        old=SurgeryModel(base,BASE_SEEDS[seed],'control',bins).cuda()
        old.load_state_dict(torch.load(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt',map_location='cuda',weights_only=True)['state_dict'])
        a,k,nll=parameters(old,cache);curves['official_V6']=probability_curve(a,k)
        folder=ROOT/f'seed{seed}'/'official_V6';folder.mkdir(parents=True,exist_ok=True)
        save(folder/'confirmation_predictions.npz',d,confirm,a,k,curves['official_V6'])
        results[str(seed)]['official_V6']=dict(metrics=summary(curves['official_V6'],d,confirm,nll));del old
        comparisons[str(seed)]={f'{name} minus {ref}':paired_interval(d.arrays['subject_ids'][confirm],d.times[confirm],d.events[confirm],
            curves[name][:,:,1],curves[ref][:,:,1],replicates=200,seed=seed) for name in nominated for ref in ('v6_control','official_V6')}
    write_json(ROOT/'report.json',dict(status='multiseed_confirmation_pending_replay',contract=contract,models=results,
        comparisons=comparisons,goal_achieved=False,test_not_evaluated=True))
    write_json(ROOT/'state.json',dict(status='training_and_confirmation_complete',next_phase='fresh-process replay and audit criteria before any test evaluation',goal_achieved=False))
    print(json.dumps(dict(status='multiseed_confirmation_pending_replay',goal_achieved=False)),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    ROOT.mkdir(exist_ok=True)
    lock=ROOT/'.running.lock'
    with lock.open('x',encoding='utf-8') as owner:
        owner.write(json.dumps(dict(pid=os.getpid(),started_unix=time.time(),purpose='single locked multiseed batch')))
    try:run()
    except Exception as error:
        write_json(ROOT/'state.json',dict(status='failed',error_type=type(error).__name__,error=str(error),
            next_phase='inspect logs before a bounded recovery',goal_achieved=False))
        raise
    finally:lock.unlink()
