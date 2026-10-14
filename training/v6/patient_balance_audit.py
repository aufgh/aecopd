"""患者均衡独立重放与一次确认；评价从不使用训练权重。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.patient_balance_experiment import ROOT,SEEDS,OLD
from training.v6.patient_balance_model import TRIALS,build
from training.v6.patient_weight_loss import training_weights
from training.v6.warm_residual import BASE_SEEDS,WarmTrial
from training.v6.warm_residual_audit import inputs as old_inputs,teacher,candidate as old_candidate
from training.v6.task_optimization_audit import load,check_curve,check_saved
from training.v6.module_surgery_experiment import batches,parameters,validation,summary,save
from training.v6.module_surgery import probability_curve
from training.v6.horizon_objective import restore
from training.v6.evaluation import paired_interval
from training.v6.task_optimization_diagnostics import analyze
from training.v6.regularization_diagnostics import history_counts
from training.v6.train import write_json


def inputs():
    result,_=old_inputs();contract=json.loads((ROOT/'contract.json').read_text())
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('balance protected source changed: '+p)
    for key,path in (('old_warm_contract_sha256',OLD/'contract.json'),('preflight_sha256',ROOT/'preflight.json'),
        ('training_weights_sha256',ROOT/'training_weights.npz'),('search_contract_sha256',Path('training_runs/v6_regularization_search_2026-10-12/contract.json'))):
        if sha256(path)!=contract[key]:raise ValueError('balance upstream proof changed')
    d=result[0];mask=d.arrays['patient_splits']=='train'
    with np.load(ROOT/'training_weights.npz') as z:
        for power in (0.,.5,1.):
            expected=np.zeros(len(mask),np.float32);expected[mask]=training_weights(d.arrays['subject_ids'][mask],power)
            np.testing.assert_array_equal(z[str(power)],expected)
    return result,contract


def model_for(base,seed,name,bins):
    if name=='original_teacher':return teacher(base,seed,bins),None,None
    if name=='full_uniform':
        m,info,best=old_candidate(base,seed,WarmTrial('warm_finetune',mode='finetune',lr=2e-5),bins)
        return m,dict(newly_trained=False,reused_from=str(OLD/f'seed{seed}/warm_finetune'),source_training=info),best
    trial=next(t for t in TRIALS if t.name==name)
    m,info,best=load(base,seed,trial,bins,root=ROOT,model_factory=build,baseline_seeds=BASE_SEEDS)
    if not info['frozen_end_and_best_verified']:raise ValueError('frozen-state training proof absent')
    if trial.head_only:
        reference=teacher(base,seed,bins)
        for key,value in reference.encoder.state_dict().items():torch.testing.assert_close(value,m.encoder.state_dict()[key],atol=0,rtol=0)
        if any(p.requires_grad for p in m.encoder.parameters()):raise ValueError('encoder not frozen after replay')
        del reference
    return m,info,best


def verify():
    if (ROOT/'.running.lock').exists():raise RuntimeError('balance training is running')
    (d,c,h,ext,_,bins,ix,_,search,_),contract=inputs();cache=batches(d,c,ext,ix)
    selected=json.loads((ROOT/'selected_before_confirmation.json').read_text())
    names=('original_teacher','full_uniform',*[t.name for t in TRIALS]);models={};errors={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1);models[str(seed)]={}
        for name in names:
            m,info,best=model_for(base,seed,name,bins)
            score=validation(m,cache,d.times[ix],d.events[ix])
            if best is not None:
                if any(score[k]!=best['tuning_'+k] for k in score):raise ValueError('balance best metric replay differs')
            elif score!=selected['references'][str(seed)][name]['tuning']:raise ValueError('balance reference replay differs')
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k);check_curve(curve)
            path=ROOT/f'seed{seed}'/name/'tuning_predictions.npz';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():error=check_saved(path,d,ix,a,k,curve)
            else:save(path,d,ix,a,k,curve);error=None
            errors[f'{seed}/{name}']=error;models[str(seed)][name]=dict(metrics=summary(curve,d,ix,nll),training=info);del m
    scores={name:float(np.mean([models[str(s)][name]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS])) for name in names}
    if scores!=selected['tuning_scores'] or min(scores,key=scores.get)!=selected['selected']:raise ValueError('balance choice differs')
    write_json(ROOT/'tuning_report.json',dict(models=models,tuning_scores=scores,selected=selected['selected'],
        confirmation_not_evaluated=True,test_not_evaluated=True,goal_achieved=False))
    replay=all(x is not None for x in errors.values())
    proof=dict(status='balance_tuning_independently_verified' if replay else 'balance_tuning_saved_pending_second_process',
        models=len(errors),errors=errors,maximum_error=max(errors.values()) if replay else None,
        report_sha256=sha256(ROOT/'tuning_report.json'),selection_sha256=sha256(ROOT/'selected_before_confirmation.json'),
        verifier_sha256=sha256(Path(__file__)),training_weights_recomputed=True,test_not_evaluated=True,goal_achieved=False)
    write_json(ROOT/'tuning_verification.json',proof)
    print(json.dumps(dict(status=proof['status'],models=len(errors),maximum_error=proof['maximum_error'])),flush=True)


def confirm():
    proof=json.loads((ROOT/'tuning_verification.json').read_text())
    if (proof['status']!='balance_tuning_independently_verified' or proof['models']!=21 or
        proof['verifier_sha256']!=sha256(Path(__file__)) or proof['report_sha256']!=sha256(ROOT/'tuning_report.json') or
        proof['selection_sha256']!=sha256(ROOT/'selected_before_confirmation.json')):
        raise ValueError('balance full independent proof required')
    selected=json.loads((ROOT/'selected_before_confirmation.json').read_text())['selected']
    target=ROOT/'nomination_lock.json'
    if target.exists():raise FileExistsError('balance confirmation already locked')
    ordinary=('original_teacher','full_uniform','head_uniform')
    reference='head_uniform' if selected.startswith('head_') else 'full_uniform'
    lock=dict(selected=selected,nominated=[] if selected in ordinary else [selected],matching_uniform=reference,
        selection_sha256=proof['selection_sha256'],verifier_sha256=proof['verifier_sha256'])
    write_json(target,lock)
    if not lock['nominated']:
        write_json(ROOT/'state.json',dict(status='verified_ordinary_reference_selected',selected=selected,
            no_patient_weight_candidate=True,goal_achieved=False));return
    evaluate_confirmation(lock,replay=False)


def evaluate_confirmation(lock,*,replay):
    if lock['selection_sha256']!=sha256(ROOT/'selected_before_confirmation.json') or lock['verifier_sha256']!=sha256(Path(__file__)):
        raise ValueError('balance nomination changed')
    (d,c,h,ext,_,bins,_,ix,search,_),_=inputs();cache=batches(d,c,ext,ix)
    names=('original_teacher',lock['matching_uniform'],lock['selected']);models={};errors={};pairs={};diagnostics={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1);models[str(seed)]={};curves={}
        for name in names:
            m,info,_=model_for(base,seed,name,bins);a,k,nll=parameters(m,cache);curve=probability_curve(a,k);check_curve(curve)
            path=ROOT/f'seed{seed}'/name/'confirmation_predictions.npz';path.parent.mkdir(parents=True,exist_ok=True)
            if replay:
                if not path.exists():raise FileNotFoundError('balance confirmation missing')
                errors[f'{seed}/{name}']=check_saved(path,d,ix,a,k,curve)
            else:
                if path.exists():raise FileExistsError('partial balance confirmation exists; inspect scene')
                save(path,d,ix,a,k,curve)
            models[str(seed)][name]=dict(metrics=summary(curve,d,ix,nll),training=info);curves[name]=curve;del m
        if not replay:
            pairs[str(seed)]={name:paired_interval(d.arrays['subject_ids'][ix],d.times[ix],d.events[ix],
                curves[lock['selected']][:,:,1],curves[name][:,:,1],replicates=200,seed=seed) for name in names[:2]}
            diagnostics[str(seed)],strata=analyze(d.times[ix],d.events[ix],d.arrays['subject_ids'][ix],
                history_counts(d.history_raw[ix]),curves,seed=seed,comparators=names[:2])
    if replay:
        prior=json.loads((ROOT/'confirmation_report.json').read_text())
        if models!=prior['models'] or prior['lock']!=lock:raise ValueError('balance confirmation report differs')
        write_json(ROOT/'confirmation_verification.json',dict(status='balance_confirmation_independently_verified',errors=errors,
            maximum_error=max(errors.values()),report_sha256=sha256(ROOT/'confirmation_report.json'),test_not_evaluated=True,goal_achieved=False))
        from training.v6.task_optimization_finish import material_signal
        decision=material_signal(models,lock['selected'],comparators=names[:2])
        write_json(ROOT/'continuation_decision.json',decision)
        write_json(ROOT/'state.json',dict(status='verified_balance_confirmation_complete',decision=decision,goal_achieved=False))
    else:
        write_json(ROOT/'confirmation_report.json',dict(models=models,comparisons=pairs,lock=lock,selected=lock['selected'],
            test_not_evaluated=True,goal_achieved=False))
        write_json(ROOT/'calibration_and_history_strata.json',dict(models=diagnostics,strata_admissions=strata,
            confirmation_report_sha256=sha256(ROOT/'confirmation_report.json'),goal_achieved=False))
        write_json(ROOT/'state.json',dict(status='balance_confirmation_pending_replay',goal_achieved=False))
    print(json.dumps(dict(status='balance_confirmation_replayed' if replay else 'balance_confirmation_written')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('verify','confirm','verify-confirmation'),required=True)
    mode=p.parse_args().mode
    if mode=='verify':verify()
    elif mode=='confirm':confirm()
    else:evaluate_confirmation(json.loads((ROOT/'nomination_lock.json').read_text()),replay=True)
