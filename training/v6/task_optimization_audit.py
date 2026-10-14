"""第二批独立重放、调参锁定及一次确认；不加载历史测试预测。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.task_optimization_experiment import ROOT,SEEDS,BASE_SEEDS
from training.v6.task_optimization import TRIALS,build
from training.v6.module_surgery_experiment import batches,parameters,validation,summary,save
from training.v6.module_surgery import SurgeryModel,probability_curve
from training.v6.token_model import daily_probabilities
from training.v6.horizon_objective import restore
from training.v6.evaluation import paired_interval
from training.v6.train import write_json


def inputs():
    result=checked()
    d,c,h,ext,base,bins,tune,confirm,search,prior=result
    contract=json.loads((ROOT/'contract.json').read_text())
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('task training source changed: '+p)
    if sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json'))!=contract['search_contract_sha256']:
        raise ValueError('source contract changed')
    if contract['seeds']!=list(SEEDS) or contract['baseline_seed_mapping']!={str(k):v for k,v in BASE_SEEDS.items()}:
        raise ValueError('seed map differs')
    if contract['trials']!=[asdict(t) for t in TRIALS]:raise ValueError('trial list changed')
    # 原正式合同已保存两个基础来源的hash，不以检查时hash冒充训练快照。
    for seed in set(BASE_SEEDS.values()):
        path=Path(search['input_config']['parent'])/f'seed{seed}/survival_base/best.pt'
        if sha256(path)!=prior['frozen_baselines'][str(seed)]:raise ValueError('baseline source changed')
    official=json.loads((REFERENCE/'report.json').read_text())
    for seed in set(BASE_SEEDS.values()):
        path=REFERENCE/f'seed{seed}/control/best.pt'
        if sha256(path)!=official['models'][str(seed)]['control']['checkpoint_sha256']:
            raise ValueError('official comparator checkpoint changed')
    return result,contract


def load(base,seed,trial,bins,*,root=None,model_factory=None,baseline_seeds=None):
    """公共检查点核验；显式注入目录/工厂，不改模块全局状态。"""
    root=ROOT if root is None else root
    model_factory=build if model_factory is None else model_factory
    baseline_seeds=BASE_SEEDS if baseline_seeds is None else baseline_seeds
    path=root/f'seed{seed}'/trial.name
    info=json.loads((path/'training_result.json').read_text())
    rows=json.loads((path/'history.json').read_text())
    state=torch.load(path/'best.pt',map_location='cuda',weights_only=True)
    if info['trial']!=asdict(trial) or state['trial']!=asdict(trial):raise ValueError('trial mismatch')
    if info['seed']!=seed or info['baseline_seed']!=baseline_seeds[seed]:raise ValueError('seed mismatch')
    if info['completed_epochs']!=100 or [x['epoch'] for x in rows]!=list(range(101)):
        raise ValueError('incomplete 100-epoch record')
    best=min(rows,key=lambda x:x['tuning_ibs'])
    if (state['epoch']!=best['epoch'] or info['best_epoch']!=best['epoch'] or
            info['tuning_ibs']!=best['tuning_ibs'] or state['tuning_ibs']!=best['tuning_ibs']):
        raise ValueError('checkpoint not selected by tuning IBS')
    if sha256(path/'best.pt')!=info['checkpoint_sha256']:raise ValueError('weight hash mismatch')
    m=model_factory(base,seed,trial,bins);m.load_state_dict(state['state_dict'])
    if any(not torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):
        raise ValueError('frozen baseline moved')
    return m,info,best


def check_curve(curve):
    if not np.isfinite(curve).all() or np.any((curve<0)|(curve>1)):raise ValueError('invalid probability')
    np.testing.assert_allclose(curve.sum(2),1,atol=1e-9,rtol=0)
    if np.any(np.diff(curve[:,:,1:],axis=1)<-1e-10):raise ValueError('CIF decreases')


def check_saved(path,d,ix,a,k,curve):
    daily,conditional=daily_probabilities(curve)
    with np.load(path) as z:
        for key,expected in (('hadm_ids',d.arrays['hadm_ids'][ix]),('subject_ids',d.arrays['subject_ids'][ix]),
                             ('times',d.times[ix]),('events',d.events[ix])):
            np.testing.assert_array_equal(z[key],expected)
        for key,expected in (('log_amplitude',a),('shape',k),('curve',curve),
                             ('daily_event_probability',daily),('conditional_daily_event_probability',conditional)):
            np.testing.assert_allclose(z[key],expected,atol=3e-7,rtol=0)
        np.testing.assert_array_equal(z['conditional_daily_valid'],curve[:,:-1,0]>1e-12)
        return float(np.abs(z['curve']-curve).max())


def verify(*,partial=False):
    (d,c,h,ext,_,bins,tune,_,search,_),contract=inputs()
    cache=batches(d,c,ext,tune);results={};errors={};models=0
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        results[str(seed)]={}
        for trial in TRIALS:
            path=ROOT/f'seed{seed}'/trial.name
            if not (path/'training_result.json').exists():
                if not partial:raise ValueError('task training incomplete')
                continue
            m,info,best=load(base,seed,trial,bins)
            # 训练日志与独立新进程计算必须相同；报告全时间点而非只看AUC90。
            score=validation(m,cache,d.times[tune],d.events[tune])
            for key,value in score.items():
                if best['tuning_'+key]!=value:raise ValueError('training metric replay mismatch: '+key)
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k);check_curve(curve)
            target=path/'tuning_predictions.npz'
            if target.exists():error=check_saved(target,d,tune,a,k,curve)
            else:save(target,d,tune,a,k,curve);error=None
            # 首次写盘再读盘并不叫独立预测重放；第二次新进程才有数值误差证据。
            errors[f'{seed}/{trial.name}']=error
            results[str(seed)][trial.name]=dict(training=info,metrics=summary(curve,d,tune,nll))
            models+=1;del m
    proof=dict(status='partial_training_metrics_replayed' if partial else 'training_metrics_replayed',models=models,
        prediction_replay_errors=errors,confirmation_not_evaluated=True,test_not_evaluated=True,goal_achieved=False,
        contract_sha256=sha256(ROOT/'contract.json'),verifier_sha256=sha256(Path(__file__)))
    if not partial:
        selected=json.loads((ROOT/'selected_before_confirmation.json').read_text())
        scores={t.name:float(np.mean([results[str(s)][t.name]['training']['tuning_ibs'] for s in SEEDS])) for t in TRIALS}
        if scores!=selected['tuning_scores'] or min(scores,key=scores.get)!=selected['selected']:
            raise ValueError('tuning nomination differs')
        proof['selected_before_confirmation_sha256']=sha256(ROOT/'selected_before_confirmation.json')
        replayed=all(x is not None for x in errors.values())
        proof['status']='tuning_independently_verified' if replayed else 'tuning_predictions_saved_pending_second_process'
        proof['maximum_error']=max(errors.values()) if replayed else None
        report=dict(models=results,tuning_scores=scores,selected=selected['selected'],
                    confirmation_not_evaluated=True,test_not_evaluated=True,goal_achieved=False)
        write_json(ROOT/'tuning_report.json',report)
        proof['tuning_report_sha256']=sha256(ROOT/'tuning_report.json')
    write_json(ROOT/('partial_verification.json' if partial else 'tuning_verification.json'),proof)
    print(json.dumps(proof),flush=True)


def confirm():
    proof=json.loads((ROOT/'tuning_verification.json').read_text())
    if proof['status']!='tuning_independently_verified' or proof['models']!=len(SEEDS)*len(TRIALS):
        raise ValueError('independent full replay needed')
    if proof['verifier_sha256']!=sha256(Path(__file__)) or proof['selected_before_confirmation_sha256']!=sha256(ROOT/'selected_before_confirmation.json'):
        raise ValueError('proof changed')
    chosen=json.loads((ROOT/'selected_before_confirmation.json').read_text())['selected']
    nominated=[] if chosen=='control' else [chosen]
    lock=dict(selected=chosen,nominated=nominated,selection='three-seed mean tuning IBS',
              selected_sha256=sha256(ROOT/'selected_before_confirmation.json'),test_not_evaluated=True)
    lockpath=ROOT/'nomination_lock.json'
    if lockpath.exists():raise FileExistsError('confirmation already locked; do not substitute nominees')
    write_json(lockpath,lock)
    if not nominated:
        write_json(ROOT/'state.json',dict(status='verified_control_selected',next_phase='another bounded hypothesis; no candidate confirmation needed',goal_achieved=False))
        print(json.dumps(dict(status='verified_control_selected',goal_achieved=False)),flush=True);return
    (d,c,h,ext,_,bins,_,ix,search,_),_=inputs();cache=batches(d,c,ext,ix)
    trials={x.name:x for x in TRIALS};models={};pairs={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        models[str(seed)]={};curves={}
        for name in ['control',*nominated,'official_V6']:
            if name=='official_V6':
                m=SurgeryModel(base,BASE_SEEDS[seed],'control',bins).cuda()
                weight=REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt'
                m.load_state_dict(torch.load(weight,map_location='cuda',weights_only=True)['state_dict']);digest=sha256(weight)
            else:
                m,info,_=load(base,seed,trials[name],bins);digest=info['checkpoint_sha256']
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k);check_curve(curve)
            models[str(seed)][name]=dict(metrics=summary(curve,d,ix,nll),checkpoint_sha256=digest)
            path=ROOT/f'seed{seed}'/name;path.mkdir(exist_ok=True)
            save(path/'confirmation_predictions.npz',d,ix,a,k,curve);curves[name]=curve;del m
        pairs[str(seed)]={ref:paired_interval(d.arrays['subject_ids'][ix],d.times[ix],d.events[ix],
            curves[chosen][:,:,1],curves[ref][:,:,1],replicates=200,seed=seed) for ref in ('control','official_V6')}
    write_json(ROOT/'confirmation_report.json',dict(status='three_seed_confirmation_pending_replay',models=models,comparisons=pairs,
        lock=lock,patients=len(np.unique(d.arrays['subject_ids'][ix])),admissions=len(ix),
        test_not_evaluated=True,goal_achieved=False,limitations=['Previously exposed validation, not blind.',
        'Three clinical seeds reuse two fixed baseline sources.', 'Not the final five-seed stability gate.']))
    write_json(ROOT/'state.json',dict(status='three_seed_confirmation_complete_pending_replay',next_phase='fresh confirmation replay, then decide if five-seed confirmation justified',goal_achieved=False))
    print(json.dumps(dict(status='three_seed_confirmation_complete_pending_replay',selected=chosen)),flush=True)


def verify_confirmation():
    r=json.loads((ROOT/'confirmation_report.json').read_text());lock=json.loads((ROOT/'nomination_lock.json').read_text())
    if r['lock']!=lock:raise ValueError('nomination changed')
    (d,c,h,ext,_,bins,_,ix,search,_),_=inputs();cache=batches(d,c,ext,ix);trials={x.name:x for x in TRIALS};errors={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        for name in ['control',*lock['nominated'],'official_V6']:
            path=ROOT/f'seed{seed}'/name
            if name=='official_V6':
                m=SurgeryModel(base,BASE_SEEDS[seed],'control',bins).cuda()
                weight=REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt'
                m.load_state_dict(torch.load(weight,map_location='cuda',weights_only=True)['state_dict']);digest=sha256(weight)
            else:m,info,_=load(base,seed,trials[name],bins);digest=info['checkpoint_sha256']
            if digest!=r['models'][str(seed)][name]['checkpoint_sha256']:raise ValueError('confirmation weight changed')
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k)
            errors[f'{seed}/{name}']=check_saved(path/'confirmation_predictions.npz',d,ix,a,k,curve)
            if summary(curve,d,ix,nll)!=r['models'][str(seed)][name]['metrics']:raise ValueError('confirmation metrics differ')
            del m
    write_json(ROOT/'confirmation_verification.json',dict(status='three_seed_confirmation_independently_verified',models=len(errors),
        maximum_error=max(errors.values()),replay_errors=errors,report_sha256=sha256(ROOT/'confirmation_report.json'),
        verifier_sha256=sha256(Path(__file__)),test_not_evaluated=True,goal_achieved=False))
    print(json.dumps(dict(status='three_seed_confirmation_independently_verified',maximum_error=max(errors.values()))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('partial','verify','confirm','verify-confirmation'),required=True)
    mode=p.parse_args().mode
    if mode in ('partial','verify'):verify(partial=mode=='partial')
    elif mode=='confirm':confirm()
    else:verify_confirmation()
