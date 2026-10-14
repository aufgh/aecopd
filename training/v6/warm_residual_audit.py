"""第四批独立重放与一次锁定确认；教师只重载，不伪造新训练。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.warm_residual_experiment import ROOT,SEEDS
from training.v6.warm_residual import TRIALS,BASE_SEEDS,build
from training.v6.task_optimization_audit import load,check_curve,check_saved
from training.v6.module_surgery import SurgeryModel,probability_curve
from training.v6.module_surgery_experiment import batches,parameters,validation,summary,save
from training.v6.horizon_objective import restore
from training.v6.evaluation import paired_interval
from training.v6.train import write_json


def inputs():
    data=checked();contract=json.loads((ROOT/'contract.json').read_text())
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('warm source changed: '+p)
    if contract['search_contract_sha256']!=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')):
        raise ValueError('warm source data contract changed')
    for seed,digest in contract['teacher_checkpoint_sha256'].items():
        if sha256(REFERENCE/f'seed{seed}/control/best.pt')!=digest:raise ValueError('teacher checkpoint changed')
    return data,contract


def teacher(base,seed,bins):
    m=SurgeryModel(base,BASE_SEEDS[seed],'control',bins).cuda()
    m.load_state_dict(torch.load(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt',map_location='cuda',weights_only=True)['state_dict'])
    return m


def candidate(base,seed,trial,bins):
    m,info,best=load(base,seed,trial,bins,root=ROOT,model_factory=build,baseline_seeds=BASE_SEEDS)
    if trial.mode!='finetune':
        reference=teacher(base,seed,bins)
        for k,v in reference.state_dict().items():torch.testing.assert_close(v,m.teacher.state_dict()[k],atol=0,rtol=0)
        if any(p.requires_grad for p in m.teacher.parameters()):raise ValueError('teacher not frozen')
        del reference
    return m,info,best


def verify():
    (d,c,h,ext,_,bins,ix,_,search,_),contract=inputs();cache=batches(d,c,ext,ix);models={};errors={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1);models[str(seed)]={}
        for name in ['original_teacher',*[t.name for t in TRIALS]]:
            if name=='original_teacher':m=teacher(base,seed,bins);info=None
            else:
                trial=next(t for t in TRIALS if t.name==name);m,info,best=candidate(base,seed,trial,bins)
                score=validation(m,cache,d.times[ix],d.events[ix])
                for key,value in score.items():
                    if best['tuning_'+key]!=value:raise ValueError('warm training metric differs')
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k);check_curve(curve)
            path=ROOT/f'seed{seed}'/name/'tuning_predictions.npz'
            if path.exists():error=check_saved(path,d,ix,a,k,curve)
            else:save(path,d,ix,a,k,curve);error=None
            errors[f'{seed}/{name}']=error
            models[str(seed)][name]=dict(training=info,metrics=summary(curve,d,ix,nll));del m
    selected=json.loads((ROOT/'selected_before_confirmation.json').read_text())
    scores={name:float(np.mean([models[str(s)][name]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS]))
            for name in ['original_teacher',*[t.name for t in TRIALS]]}
    if scores!=selected['tuning_scores'] or min(scores,key=scores.get)!=selected['selected']:raise ValueError('warm selection changed')
    write_json(ROOT/'tuning_report.json',dict(models=models,tuning_scores=scores,selected=selected['selected'],goal_achieved=False))
    complete=all(x is not None for x in errors.values())
    write_json(ROOT/'tuning_verification.json',dict(status='tuning_independently_verified' if complete else 'tuning_predictions_saved_pending_second_process',
        models=len(errors),errors=errors,maximum_error=max(errors.values()) if complete else None,code_sha256=sha256(Path(__file__)),
        selection_sha256=sha256(ROOT/'selected_before_confirmation.json'),report_sha256=sha256(ROOT/'tuning_report.json'),test_not_evaluated=True))
    print(json.dumps(dict(status='tuning_independently_verified' if complete else 'tuning_predictions_saved_pending_second_process',models=len(errors))),flush=True)


def confirm():
    proof=json.loads((ROOT/'tuning_verification.json').read_text())
    if proof['status']!='tuning_independently_verified' or proof['code_sha256']!=sha256(Path(__file__)):
        raise ValueError('warm complete independent proof required')
    if proof['selection_sha256']!=sha256(ROOT/'selected_before_confirmation.json'):raise ValueError('selection proof stale')
    selected=json.loads((ROOT/'selected_before_confirmation.json').read_text())['selected']
    target=ROOT/'nomination_lock.json'
    if target.exists():raise FileExistsError('warm confirmation already locked; do not reselect')
    write_json(target,dict(selected=selected,nominated=[] if selected=='original_teacher' else [selected],selection_sha256=proof['selection_sha256']))
    if selected=='original_teacher':
        write_json(ROOT/'state.json',dict(status='original_teacher_selected',next_phase='no confirmed improvement; next bounded hypothesis',goal_achieved=False));return
    (d,c,h,ext,_,bins,_,ix,search,_),_=inputs();cache=batches(d,c,ext,ix);models={};pairs={}
    trial=next(t for t in TRIALS if t.name==selected)
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1);models[str(seed)]={};curves={}
        for name in ('original_teacher',selected):
            if name=='original_teacher':m=teacher(base,seed,bins);digest=sha256(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt')
            else:m,info,_=candidate(base,seed,trial,bins);digest=info['checkpoint_sha256']
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k);check_curve(curve)
            save(ROOT/f'seed{seed}'/name/'confirmation_predictions.npz',d,ix,a,k,curve)
            models[str(seed)][name]=dict(metrics=summary(curve,d,ix,nll),checkpoint_sha256=digest);curves[name]=curve;del m
        pairs[str(seed)]=paired_interval(d.arrays['subject_ids'][ix],d.times[ix],d.events[ix],curves[selected][:,:,1],
                                       curves['original_teacher'][:,:,1],replicates=200,seed=seed)
    write_json(ROOT/'confirmation_report.json',dict(models=models,comparisons=pairs,selected=selected,
        lock_sha256=sha256(target),test_not_evaluated=True,goal_achieved=False))
    write_json(ROOT/'state.json',dict(status='confirmation_complete_pending_replay',goal_achieved=False))


def verify_confirmation():
    r=json.loads((ROOT/'confirmation_report.json').read_text())
    if r['lock_sha256']!=sha256(ROOT/'nomination_lock.json'):raise ValueError('lock changed')
    (d,c,h,ext,_,bins,_,ix,search,_),_=inputs();cache=batches(d,c,ext,ix);errors={}
    trial=next(t for t in TRIALS if t.name==r['selected'])
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        for name in ('original_teacher',r['selected']):
            if name=='original_teacher':m=teacher(base,seed,bins);digest=sha256(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt')
            else:m,info,_=candidate(base,seed,trial,bins);digest=info['checkpoint_sha256']
            if digest!=r['models'][str(seed)][name]['checkpoint_sha256']:raise ValueError('confirmation weights differ')
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k)
            errors[f'{seed}/{name}']=check_saved(ROOT/f'seed{seed}'/name/'confirmation_predictions.npz',d,ix,a,k,curve)
            if summary(curve,d,ix,nll)!=r['models'][str(seed)][name]['metrics']:raise ValueError('confirmation metrics differ')
            del m
    write_json(ROOT/'confirmation_verification.json',dict(status='warm_confirmation_independently_verified',errors=errors,
        maximum_error=max(errors.values()),report_sha256=sha256(ROOT/'confirmation_report.json'),test_not_evaluated=True,goal_achieved=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('verify','confirm','verify-confirmation'),required=True)
    mode=p.parse_args().mode
    if mode=='verify':verify()
    elif mode=='confirm':confirm()
    else:verify_confirmation()
