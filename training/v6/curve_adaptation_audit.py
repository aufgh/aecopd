"""独立重建固定表示/重放小头，锁定一次确认；不读取历史测试。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.curve_adaptation import (ROOT,SEEDS,ARMS,EPOCHS,setup_inputs,assert_contract,caches,
                                        new_model,score,prediction)
from training.v6.module_surgery_experiment import summary,save
from training.v6.task_optimization_audit import check_curve,check_saved
from training.v6.task_optimization_finish import material_signal
from training.v6.task_optimization_diagnostics import analyze
from training.v6.regularization_diagnostics import history_counts
from training.v6.evaluation import paired_interval
from training.v6.train import write_json

NAMES=('original_teacher',*ARMS)


def choose(scores):
    if set(scores)!=set(NAMES) or not all(np.isfinite(v) for v in scores.values()):
        raise ValueError('all finite comparator scores required')
    # 固定次序处理相等值：共同epoch0不得因字典顺序而提名。
    selected=min(NAMES,key=lambda n:scores[n])
    return selected,[selected] if selected in ('clinical_nll','clinical_curve') else []


def load(arm,seed):
    path=ROOT/f'seed{seed}'/arm
    info=json.loads((path/'training_result.json').read_text())
    hist=json.loads((path/'history.json').read_text())
    state=torch.load(path/'best.pt',map_location='cuda',weights_only=True)
    if ([r['epoch'] for r in hist]!=list(range(EPOCHS+1)) or info['completed_epochs']!=EPOCHS
            or min(hist,key=lambda r:r['tuning_ibs'])['epoch']!=state['epoch'] or info['best_epoch']!=state['epoch']
            or info['checkpoint_sha256']!=sha256(path/'best.pt') or state['contract_sha256']!=sha256(ROOT/'contract.json')
            or state['arm']!=arm or state['seed']!=seed):raise ValueError('training/selection/checkpoint proof differs')
    m=new_model(arm,seed);m.load_state_dict(state['state_dict'])
    return m,info,hist[state['epoch']]


def evaluate(data,ix,names,stage):
    d=data[0];results={};errors={}
    for seed in SEEDS:
        cache,cp=caches(data,ix,seed)
        if stage=='tuning':
            expected=json.loads((ROOT/f'cache_seed{seed}.json').read_text())['tuning']
            if expected!=cp:raise ValueError('tuning cache differs')
            train_ix=np.flatnonzero(d.arrays['patient_splits']=='train')
            train,trainproof=caches(data,train_ix,seed)
            if trainproof!=json.loads((ROOT/f'cache_seed{seed}.json').read_text())['train']:raise ValueError('training cache differs')
            del train
        results[str(seed)]={}
        for name in names:
            info=None
            if name=='original_teacher':m=new_model('clinical_nll',seed)
            else:m,info,best=load(name,seed)
            values,a,k,curve=score(m,cache);check_curve(curve)
            if stage=='tuning' and name!='original_teacher':
                for key,value in values.items():
                    if best['tuning_'+key]!=value:raise ValueError('tuning metric replay differs: '+key)
            path=ROOT/f'seed{seed}'/name;path.mkdir(exist_ok=True)
            target=path/f'{stage}_predictions.npz'
            if target.exists():error=check_saved(target,d,ix,a,k,curve)
            else:save(target,d,ix,a,k,curve);error=None
            errors[f'{seed}/{name}']=error
            results[str(seed)][name]=dict(training=info,metrics=summary(curve,d,ix,values['nll']))
        del cache
    return results,errors


def verify():
    data,_=setup_inputs();assert_contract(data)
    if json.loads((ROOT/'state.json').read_text())['status']!='training_complete':raise RuntimeError('training incomplete')
    models,errors=evaluate(data,data[6],NAMES,'tuning')
    scores={n:float(np.mean([models[str(s)][n]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS])) for n in NAMES}
    report=dict(models=models,tuning_scores=scores,selected=choose(scores)[0],goal_achieved=False)
    path=ROOT/'tuning_report.json'
    if path.exists() and json.loads(path.read_text())!=report:raise ValueError('independent tuning report differs')
    write_json(path,report)
    complete=all(v is not None for v in errors.values())
    write_json(ROOT/'tuning_verification.json',dict(status='independently_replayed' if complete else 'saved_pending_replay',
        errors=errors,models=len(errors),maximum_error=max(errors.values()) if complete else None,report_sha256=sha256(path)))
    print(json.dumps(dict(status='tuning_checked',complete=complete,scores=scores,selected=report['selected'])),flush=True)


def lock_selection():
    proof=json.loads((ROOT/'tuning_verification.json').read_text())
    if proof['status']!='independently_replayed' or proof['report_sha256']!=sha256(ROOT/'tuning_report.json'):
        raise RuntimeError('two independent tuning passes required')
    report=json.loads((ROOT/'tuning_report.json').read_text());selected,nominated=choose(report['tuning_scores'])
    if report['selected']!=selected:raise ValueError('saved nomination winner differs')
    lock=dict(selected=selected,nominated=nominated,
        tuning_report_sha256=sha256(ROOT/'tuning_report.json'),contract_sha256=sha256(ROOT/'contract.json'),
        rule='lowest mean three-seed tuning IBS; original/global winners are not clinical nominees',goal_achieved=False)
    path=ROOT/'nomination_lock.json'
    if path.exists():
        if json.loads(path.read_text())!=lock:raise RuntimeError('existing lock differs; cannot reselect')
    else:write_json(path,lock)
    return lock


def confirm(verify_only=False):
    data,_=setup_inputs();assert_contract(data)
    lock=lock_selection()
    if lock['tuning_report_sha256']!=sha256(ROOT/'tuning_report.json') or lock['contract_sha256']!=sha256(ROOT/'contract.json'):
        raise RuntimeError('locked selection stale')
    if not lock['nominated']:return
    target=ROOT/'confirmation_report.json'
    if not verify_only and target.exists():raise RuntimeError('confirmation already run; use verify-confirmation')
    if verify_only and not target.exists():raise RuntimeError('confirmation not yet run')
    names=list(dict.fromkeys(['original_teacher','clinical_nll','global_curve',lock['selected']]))
    models,errors=evaluate(data,data[7],names,'confirmation')
    d=data[0];ix=data[7];comparators=('original_teacher','global_curve')
    paired={};strata={}
    for seed in SEEDS:
        curves={}
        for name in names:
            with np.load(ROOT/f'seed{seed}'/name/'confirmation_predictions.npz') as z:curves[name]=z['curve'].copy()
        paired[str(seed)]={ref:paired_interval(d.arrays['subject_ids'][ix],d.times[ix],d.events[ix],
            curves[lock['selected']][:,:,1],curves[ref][:,:,1],seed=seed) for ref in dict.fromkeys([*comparators,'clinical_nll']) if ref!=lock['selected']}
        strata[str(seed)],counts=analyze(d.times[ix],d.events[ix],d.arrays['subject_ids'][ix],history_counts(d.history_raw[ix]),
            curves,seed=seed,comparators=comparators)
    report=dict(lock=lock,models=models,paired=paired,calibration_and_strata=strata,strata_admissions=counts,
                signal=material_signal(models,lock['selected'],comparators=comparators),goal_achieved=False)
    if verify_only:
        if json.loads(target.read_text())!=report:raise ValueError('confirmation report replay differs')
        if not all(v is not None for v in errors.values()):raise ValueError('confirmation predictions absent')
        write_json(ROOT/'confirmation_verification.json',dict(errors=errors,maximum_error=max(errors.values()),report_sha256=sha256(target)))
    else:write_json(target,report)
    print(json.dumps(dict(status='confirmation_replayed' if verify_only else 'confirmation_saved',signal=report['signal'])),flush=True)


def finish():
    if (ROOT/'postprocess_state.json').exists():raise RuntimeError('postprocessing exists; inspect instead of automatic repeat')
    def step(mode):subprocess.run([sys.executable,'-u','-m','training.v6.curve_adaptation_audit','--mode',mode],check=True)
    write_json(ROOT/'postprocess_state.json',dict(status='running',goal_achieved=False))
    try:
        step('verify');step('verify');lock=lock_selection()
        if lock['nominated']:step('confirm');step('verify-confirmation')
        write_json(ROOT/'postprocess_state.json',dict(status='complete',selected=lock['selected'],nominated=lock['nominated'],goal_achieved=False))
        print(json.dumps(dict(status='eighth_batch_complete',selected=lock['selected'],nominated=lock['nominated'])),flush=True)
    except Exception as error:
        write_json(ROOT/'postprocess_state.json',dict(status='failed',error=str(error),goal_achieved=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('verify','confirm','verify-confirmation','finish'),required=True)
    args=p.parse_args()
    if args.mode=='verify':verify()
    elif args.mode=='confirm':confirm()
    elif args.mode=='verify-confirmation':confirm(True)
    else:finish()
