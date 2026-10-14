"""第三候选：条件化临床修正固定配对训练/核验；当前批次结束后才启动。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.regularization_multiseed import fit
from training.v6.contextual_residual import TRIALS,build
from training.v6.task_optimization_audit import load,check_curve,check_saved
from training.v6.module_surgery_experiment import batches,parameters,validation,summary,save
from training.v6.module_surgery import SurgeryModel,probability_curve
from training.v6.horizon_objective import restore
from training.v6.evaluation import paired_interval
from training.v6.train import write_json

ROOT=Path('training_runs/v6_contextual_residual_2026-10-13')
PRIOR=Path('training_runs/v6_task_optimization_2026-10-13')
SEEDS=(42,7,101)
BASE_SEEDS={42:42,7:7,101:42}
SOURCES=('training/v6/contextual_residual.py','training/v6/contextual_residual_experiment.py',
         'training/v6/regularization_multiseed.py','training/v6/task_optimization_audit.py',
         'training/v6/task_optimization_diagnostics.py','training/v6/contextual_residual_report.py',
         'training/tests/test_v6_contextual_residual.py','training/tests/test_v6_contextual_pipeline.py',
         'docs/V6_CONTEXTUAL_RESIDUAL_HYPOTHESIS_2026-10-13.md')


def clinical_nomination(scores):
    """先固定IBS均值优胜者；基础再校准胜出不冒称临床增量。"""
    selected=min(scores,key=scores.get)
    return selected,[] if selected in ('control','baseline_only') else [selected]


def inputs():
    result=checked();d,c,h,ext,_,bins,tune,confirm,search,prior=result
    contract=json.loads((ROOT/'contract.json').read_text())
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('context source changed: '+p)
    if contract['search_contract_sha256']!=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')):
        raise ValueError('source data contract changed')
    if contract['trials']!=[asdict(t) for t in TRIALS] or contract['seeds']!=list(SEEDS):raise ValueError('context trial list differs')
    if contract['baseline_seed_mapping']!={str(k):v for k,v in BASE_SEEDS.items()}:raise ValueError('baseline seed map differs')
    for s in set(BASE_SEEDS.values()):
        p=Path(search['input_config']['parent'])/f'seed{s}/survival_base/best.pt'
        if sha256(p)!=prior['frozen_baselines'][str(s)]:raise ValueError('frozen source moved')
    official=json.loads((REFERENCE/'report.json').read_text())
    for s in set(BASE_SEEDS.values()):
        if sha256(REFERENCE/f'seed{s}/control/best.pt')!=official['models'][str(s)]['control']['checkpoint_sha256']:
            raise ValueError('official source moved')
    return result,contract


def model(base,seed,trial,bins):
    return load(base,seed,trial,bins,root=ROOT,model_factory=build,baseline_seeds=BASE_SEEDS)


def train():
    # 当前批次未核验/确认时绝不自动抢跑下一批；文件存在不是确认完成证据。
    previous=json.loads((PRIOR/'tuning_verification.json').read_text())
    if previous['status']!='tuning_independently_verified' or previous['models']!=24:
        raise ValueError('previous tuning verification incomplete')
    if previous['selected_before_confirmation_sha256']!=sha256(PRIOR/'selected_before_confirmation.json'):
        raise ValueError('previous selection proof stale')
    lock=json.loads((PRIOR/'nomination_lock.json').read_text())
    if lock['selected_sha256']!=sha256(PRIOR/'selected_before_confirmation.json'):raise ValueError('previous lock stale')
    if lock['nominated']:
        proof=json.loads((PRIOR/'confirmation_verification.json').read_text())
        if proof['report_sha256']!=sha256(PRIOR/'confirmation_report.json'):raise ValueError('previous confirmation proof stale')
    d,c,h,ext,_,bins,tune,_,search,prior=checked()
    contract=dict(source_sha256={p:sha256(Path(p)) for p in SOURCES},
        search_contract_sha256=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')),
        previous_tuning_proof_sha256=sha256(PRIOR/'tuning_verification.json'),
        previous_nomination_sha256=sha256(PRIOR/'nomination_lock.json'),trials=[asdict(t) for t in TRIALS],
        seeds=list(SEEDS),baseline_seed_mapping=BASE_SEEDS,epochs=100,
        selection='three-seed tuning IBS mean; clinical candidate must beat control and baseline-only',test_not_evaluated=True)
    if (ROOT/'contract.json').exists():
        if json.loads((ROOT/'contract.json').read_text())!=json.loads(json.dumps(contract)):raise ValueError('resume contract changed')
    else:write_json(ROOT/'contract.json',contract)
    for s in set(BASE_SEEDS.values()):
        if sha256(Path(search['input_config']['parent'])/f'seed{s}/survival_base/best.pt')!=prior['frozen_baselines'][str(s)]:
            raise ValueError('frozen baseline source differs')
    cache=dict(train=batches(d,c,ext,np.flatnonzero(d.arrays['patient_splits']=='train')),tune=batches(d,c,ext,tune));results={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        results[str(seed)]={};control=build(base,seed,TRIALS[0],bins)
        for trial in TRIALS:
            path=ROOT/f'seed{seed}'/trial.name
            if (path/'training_result.json').exists():
                m,info,_=model(base,seed,trial,bins);results[str(seed)][trial.name]=info;del m;continue
            if path.exists() and any((path/p).exists() for p in ('best.pt','history.json')):
                import shutil
                backup=path/'recovery_snapshots'/str(time.time_ns());backup.mkdir(parents=True)
                for p in ('best.pt','history.json'):
                    if (path/p).exists():shutil.copy2(path/p,backup/p)
            probe=build(base,seed,trial,bins).eval()
            for k,v in control.encoder.state_dict().items():
                if not torch.equal(v,probe.encoder.state_dict()[k]):raise ValueError('encoder initialization differs')
            with torch.no_grad():
                _,hist,b,_,_=cache['tune'][0];actual,expected=probe(hist,b),base(hist)
                if not torch.equal(actual.log_amplitude,expected.log_amplitude) or not torch.equal(actual.shape,expected.shape):
                    raise ValueError('epoch0 risk differs')
            del probe
            write_json(ROOT/'state.json',dict(status='running',seed=seed,trial=trial.name,completed=results,goal_achieved=False))
            results[str(seed)][trial.name]=fit(base,seed,trial,bins,cache,d.times[tune],d.events[tune],path,model_factory=build)
        del control
    scores={t.name:float(np.mean([results[str(s)][t.name]['tuning_ibs'] for s in SEEDS])) for t in TRIALS}
    selected,nominated=clinical_nomination(scores)
    write_json(ROOT/'selected_before_confirmation.json',dict(results=results,tuning_scores=scores,selected=selected,
               nominated=nominated,all_100_epochs_complete=True,confirmation_not_evaluated=True,test_not_evaluated=True))
    write_json(ROOT/'state.json',dict(status='training_complete_pending_replay',selected=selected,goal_achieved=False))
    print(json.dumps(dict(status='training_complete_pending_replay',selected=selected)),flush=True)


def verify():
    (d,c,h,ext,_,bins,tune,_,search,_),contract=inputs();cache=batches(d,c,ext,tune);results={};errors={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1);results[str(seed)]={}
        for trial in TRIALS:
            m,info,best=model(base,seed,trial,bins);score=validation(m,cache,d.times[tune],d.events[tune])
            for key,value in score.items():
                if best['tuning_'+key]!=value:raise ValueError('tuning metric differs')
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k);check_curve(curve)
            target=ROOT/f'seed{seed}'/trial.name/'tuning_predictions.npz'
            if target.exists():error=check_saved(target,d,tune,a,k,curve)
            else:save(target,d,tune,a,k,curve);error=None
            errors[f'{seed}/{trial.name}']=error
            results[str(seed)][trial.name]=dict(training=info,metrics=summary(curve,d,tune,nll));del m
    selected=json.loads((ROOT/'selected_before_confirmation.json').read_text())
    scores={t.name:float(np.mean([results[str(s)][t.name]['training']['tuning_ibs'] for s in SEEDS])) for t in TRIALS}
    chosen,nominated=clinical_nomination(scores)
    if chosen!=selected['selected'] or scores!=selected['tuning_scores'] or nominated!=selected['nominated']:raise ValueError('nomination differs')
    replay=all(x is not None for x in errors.values())
    write_json(ROOT/'tuning_report.json',dict(models=results,tuning_scores=scores,selected=chosen,nominated=nominated,goal_achieved=False))
    write_json(ROOT/'tuning_verification.json',dict(status='tuning_independently_verified' if replay else 'tuning_predictions_saved_pending_second_process',
        models=len(errors),replay_errors=errors,maximum_error=max(errors.values()) if replay else None,
        verifier_sha256=sha256(Path(__file__)),contract_sha256=sha256(ROOT/'contract.json'),
        selected_sha256=sha256(ROOT/'selected_before_confirmation.json'),tuning_report_sha256=sha256(ROOT/'tuning_report.json'),
        test_not_evaluated=True,goal_achieved=False))
    print(json.dumps(dict(status='tuning_independently_verified' if replay else 'tuning_predictions_saved_pending_second_process',models=len(errors))),flush=True)


def confirm():
    proof=json.loads((ROOT/'tuning_verification.json').read_text())
    if proof['status']!='tuning_independently_verified' or proof['verifier_sha256']!=sha256(Path(__file__)):
        raise ValueError('complete fresh-process proof needed')
    r=json.loads((ROOT/'selected_before_confirmation.json').read_text())
    if proof['selected_sha256']!=sha256(ROOT/'selected_before_confirmation.json'):raise ValueError('selection changed')
    target=ROOT/'nomination_lock.json'
    if target.exists():raise FileExistsError('confirmation already locked')
    write_json(target,dict(selected=r['selected'],nominated=r['nominated'],selected_sha256=proof['selected_sha256'],test_not_evaluated=True))
    if not r['nominated']:
        write_json(ROOT/'state.json',dict(status='no_clinical_candidate_selected',selected=r['selected'],
            next_phase='next bounded hypothesis; baseline-only winner is not evidence of clinical increment',goal_achieved=False))
        return
    (d,c,h,ext,_,bins,_,ix,search,_),_=inputs();cache=batches(d,c,ext,ix);models={};pairs={};trials={t.name:t for t in TRIALS}
    chosen=r['nominated'][0]
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1);models[str(seed)]={};curves={}
        for name in ['control','baseline_only',chosen,'official_V6']:
            if name=='official_V6':
                m=SurgeryModel(base,BASE_SEEDS[seed],'control',bins).cuda();p=REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt'
                m.load_state_dict(torch.load(p,map_location='cuda',weights_only=True)['state_dict']);digest=sha256(p)
            else:m,info,_=model(base,seed,trials[name],bins);digest=info['checkpoint_sha256']
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k);check_curve(curve)
            folder=ROOT/f'seed{seed}'/name;folder.mkdir(exist_ok=True);save(folder/'confirmation_predictions.npz',d,ix,a,k,curve)
            models[str(seed)][name]=dict(metrics=summary(curve,d,ix,nll),checkpoint_sha256=digest);curves[name]=curve;del m
        pairs[str(seed)]={ref:paired_interval(d.arrays['subject_ids'][ix],d.times[ix],d.events[ix],curves[chosen][:,:,1],
            curves[ref][:,:,1],replicates=200,seed=seed) for ref in ('control','baseline_only','official_V6')}
    write_json(ROOT/'confirmation_report.json',dict(models=models,comparisons=pairs,selected=chosen,
        nomination_lock_sha256=sha256(target),test_not_evaluated=True,goal_achieved=False))
    write_json(ROOT/'state.json',dict(status='confirmation_complete_pending_replay',goal_achieved=False))


def verify_confirmation():
    r=json.loads((ROOT/'confirmation_report.json').read_text())
    if r['nomination_lock_sha256']!=sha256(ROOT/'nomination_lock.json'):raise ValueError('confirmation lock differs')
    (d,c,h,ext,_,bins,_,ix,search,prior),_=inputs();cache=batches(d,c,ext,ix);errors={};trials={t.name:t for t in TRIALS}
    official=json.loads((REFERENCE/'report.json').read_text())
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        for name in ['control','baseline_only',r['selected'],'official_V6']:
            if name=='official_V6':
                m=SurgeryModel(base,BASE_SEEDS[seed],'control',bins).cuda();p=REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt'
                digest=sha256(p)
                if digest!=official['models'][str(BASE_SEEDS[seed])]['control']['checkpoint_sha256']:raise ValueError('official source changed')
                m.load_state_dict(torch.load(p,map_location='cuda',weights_only=True)['state_dict'])
            else:m,info,_=model(base,seed,trials[name],bins);digest=info['checkpoint_sha256']
            if digest!=r['models'][str(seed)][name]['checkpoint_sha256']:raise ValueError('confirmation model changed')
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k)
            errors[f'{seed}/{name}']=check_saved(ROOT/f'seed{seed}'/name/'confirmation_predictions.npz',d,ix,a,k,curve)
            if summary(curve,d,ix,nll)!=r['models'][str(seed)][name]['metrics']:raise ValueError('confirmation metrics differ')
            del m
    write_json(ROOT/'confirmation_verification.json',dict(status='three_seed_confirmation_independently_verified',
        models=len(errors),maximum_error=max(errors.values()),errors=errors,report_sha256=sha256(ROOT/'confirmation_report.json'),
        test_not_evaluated=True,goal_achieved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('train','verify','confirm','verify-confirmation'),required=True)
    mode=parser.parse_args().mode
    if mode=='train':
        ROOT.mkdir(exist_ok=True);lock=ROOT/'.running.lock'
        with lock.open('x',encoding='utf-8') as owner:owner.write(json.dumps(dict(pid=os.getpid(),started=time.time())))
        try:train()
        except Exception as e:write_json(ROOT/'state.json',dict(status='failed',error_type=type(e).__name__,error=str(e),goal_achieved=False));raise
        finally:lock.unlink()
    elif mode=='verify':verify()
    elif mode=='confirm':confirm()
    else:verify_confirmation()
