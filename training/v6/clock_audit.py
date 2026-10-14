"""固定全年概率时钟的独立重放、网格收敛和一次锁定确认。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked, REFERENCE
from training.v6.clock_experiment import ROOT, SEEDS, ARMS, ClockHead, features, extract, dense_teacher, validate
from training.v6.warm_residual import BASE_SEEDS
from training.v6.warm_residual_audit import teacher
from training.v6.module_surgery_experiment import batches, summary
from training.v6.horizon_objective import restore
from training.v6.anchored_clock import clock_curve, clock_nll, anchor_dense_to_daily
from training.v6.risk import RiskOutput, predict_cif
from training.v6.task_optimization_audit import check_curve, check_saved
from training.v6.task_optimization_diagnostics import analyze
from training.v6.regularization_diagnostics import history_counts
from training.v6.task_optimization_finish import material_signal
from training.v6.evaluation import paired_interval
from training.v6.train import write_json


def inputs():
    data = checked()
    contract = json.loads((ROOT/'contract.json').read_text())
    for p, digest in contract['source_sha256'].items():
        if sha256(Path(p)) != digest:
            raise ValueError('clock protected source changed: '+p)
    if contract['search_contract_sha256'] != sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')):
        raise ValueError('clock upstream contract changed')
    if contract['seeds'] != list(SEEDS) or contract['arms'] != list(ARMS) or contract['epochs'] != 100:
        raise ValueError('clock training specification differs')
    for seed in SEEDS:
        if contract['teachers'][str(seed)] != sha256(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt'):
            raise ValueError('teacher weights changed')
    return data, contract


def load_head(seed, arm):
    path = ROOT/f'seed{seed}'/arm
    info = json.loads((path/'training_result.json').read_text())
    rows = json.loads((path/'history.json').read_text())
    state = torch.load(path/'best.pt', map_location='cuda', weights_only=True)
    if info['seed'] != seed or info['arm'] != arm or info['completed_epochs'] != 100:
        raise ValueError('clock identity/training count differs')
    if [r['epoch'] for r in rows] != list(range(101)):
        raise ValueError('clock epoch record incomplete')
    best = min(rows, key=lambda r:r['tuning_ibs'])
    if (best['epoch'] != info['best_epoch'] or state['epoch'] != best['epoch'] or
            state['arm'] != arm or state['width'] != info['width'] or best['tuning_ibs'] != info['tuning_ibs']):
        raise ValueError('clock selected checkpoint differs')
    if info['checkpoint_sha256'] != sha256(path/'best.pt'):
        raise ValueError('clock checkpoint hash differs')
    model = ClockHead(info['width']).cuda()
    model.load_state_dict(state['state_dict'])
    return model, info, best


def refined_teacher(cache, daily, *, fine=False):
    """只在核验读出细化，训练源码/权重/选点保持原样；首周增加曲率采样。"""
    step = .0625 if fine else .125
    early = 1/128 if fine else 1/64
    grid = np.unique(np.r_[np.arange(0,365.001,step),np.arange(0,7.001,early),np.geomspace(1e-10,.5,80)])
    raw = predict_cif(cache['amplitude'].cpu().numpy(),cache['shape'].cpu().numpy(),grid,step_days=step)
    return grid, anchor_dense_to_daily(raw,grid,daily)


def save_or_check(path, d, ix, cache, log_clock, curve):
    """旧a/k为冻结教师参数，不冒充时钟模型的完整参数；另存log_clock。"""
    a, k = cache['amplitude'].cpu().numpy(), cache['shape'].cpu().numpy()
    if path.exists():
        error = check_saved(path, d, ix, a, k, curve)
        with np.load(path) as z:
            np.testing.assert_array_equal(z['log_clock'], log_clock)
        return error
    path.parent.mkdir(parents=True, exist_ok=True)
    from training.v6.token_model import daily_probabilities
    daily, conditional = daily_probabilities(curve)
    np.savez_compressed(path, hadm_ids=d.arrays['hadm_ids'][ix], subject_ids=d.arrays['subject_ids'][ix],
        times=d.times[ix], events=d.events[ix], log_amplitude=a, shape=k, log_clock=log_clock, curve=curve,
        daily_event_probability=daily, conditional_daily_event_probability=conditional,
        conditional_daily_valid=curve[:,:-1,0]>1e-12)
    return None


def infer(cache, grid, dense, daily, seed, name):
    if name == 'original_teacher':
        log_clock = np.zeros((len(daily),2), dtype=np.float32)
        curve, info, best = daily, None, None
    else:
        model, info, best = load_head(seed, name)
        score, log_clock, curve = validate(model, features(cache, name), cache, grid, dense)
        del model
    check_curve(curve)
    np.testing.assert_array_equal(curve[:,-1], daily[:,-1])
    prior = RiskOutput(cache['amplitude'], cache['shape'])
    with torch.no_grad():
        nll = float(clock_nll(prior, torch.as_tensor(log_clock, device='cuda'), cache['times'], cache['events']))
    return log_clock, curve, info, best, nll


def verify():
    (d,c,h,ext,_,bins,ix,_,search,_), contract = inputs()
    cache = batches(d,c,ext,ix)
    selected = json.loads((ROOT/'selected_before_confirmation.json').read_text())
    models, errors, convergence, training_scores = {}, {}, {}, {}
    for seed in SEEDS:
        base = restore(Path(search['input_config']['parent']), BASE_SEEDS[seed], 'survival_base', h, .1)
        model = teacher(base,seed,bins)
        data = extract(model,cache)
        del model
        grid,dense,daily,_ = dense_teacher(data)
        eval_grid, evaluation = refined_teacher(data,daily)
        fine_grid, fine = refined_teacher(data,daily,fine=True)
        models[str(seed)] = {}
        training_scores[str(seed)] = {}
        for name in ('original_teacher',*ARMS):
            log_clock,curve,info,best,nll = infer(data,grid,dense,daily,seed,name)
            detail = summary(curve,d,ix,nll)
            if best is not None:
                actual = dict(ibs=detail['readmission']['integrated_brier_7_365'],
                    auc90=detail['readmission']['points']['90']['auc'],brier90=detail['readmission']['points']['90']['brier'])
                if any(actual[k] != best['tuning_'+k] for k in actual):
                    raise ValueError('clock training metric replay differs')
            training_scores[str(seed)][name] = detail['readmission']['integrated_brier_7_365']
            original_curve = curve
            curve = clock_curve(evaluation,eval_grid,log_clock,np.arange(366))
            check_curve(curve)
            np.testing.assert_array_equal(curve[:,-1],daily[:,-1])
            detail = summary(curve,d,ix,nll)
            fine_curve = clock_curve(fine,fine_grid,log_clock,np.arange(366))
            fine_score = summary(fine_curve,d,ix,nll)
            error = float(np.abs(fine_curve-curve).max())
            ibs_delta = float(fine_score['readmission']['integrated_brier_7_365']-detail['readmission']['integrated_brier_7_365'])
            auc_delta = float(fine_score['readmission']['points']['90']['auc']-detail['readmission']['points']['90']['auc'])
            # 数值误差预算事前固定，不因改善大小放宽。
            convergence[f'{seed}/{name}'] = dict(maximum_probability_error=error,ibs_difference=ibs_delta,auc90_difference=auc_delta,
                original_training_grid_probability_difference=float(np.abs(original_curve-curve).max()))
            if error > 1e-4 or abs(ibs_delta) > 1e-6 or abs(auc_delta) > 1e-4:
                write_json(ROOT/'grid_convergence_failure.json',dict(seed=seed,name=name,comparisons=convergence,
                    thresholds=dict(probability=1e-4,ibs=1e-6,auc90=1e-4),goal_achieved=False))
                raise ValueError('clock grid convergence failed: '+json.dumps(convergence[f'{seed}/{name}']))
            errors[f'{seed}/{name}'] = save_or_check(ROOT/f'seed{seed}'/name/'tuning_predictions.npz',d,ix,data,log_clock,curve)
            models[str(seed)][name] = dict(metrics=detail,training=info)
        del data,dense,fine,evaluation
    scores = {name:float(np.mean([training_scores[str(s)][name] for s in SEEDS]))
        for name in ('original_teacher',*ARMS)}
    if scores != selected['tuning_scores'] or min(scores,key=scores.get) != selected['selected']:
        raise ValueError('clock nomination differs')
    refined_scores = {name:float(np.mean([models[str(s)][name]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS]))
        for name in ('original_teacher',*ARMS)}
    if min(refined_scores,key=refined_scores.get)!=selected['selected']:
        raise ValueError('accurate grid reverses nomination; no reselection permitted')
    write_json(ROOT/'tuning_report.json',dict(models=models,selected=selected['selected'],tuning_scores=scores,
        refined_tuning_scores=refined_scores,selection_unchanged=True,goal_achieved=False))
    write_json(ROOT/'grid_convergence.json',dict(comparisons=convergence,grids_days=[.125,.0625],
        first_week_grid_days=[1/64,1/128],daily_endpoints_anchored=True,training_grid_unchanged=True))
    complete = all(e is not None for e in errors.values())
    proof = dict(status='tuning_independently_verified' if complete else 'tuning_saved_pending_second_process',
        models=len(errors),errors=errors,maximum_error=max(errors.values()) if complete else None,
        auditor_sha256=sha256(Path(__file__)),selection_sha256=sha256(ROOT/'selected_before_confirmation.json'),
        contract_sha256=sha256(ROOT/'contract.json'),report_sha256=sha256(ROOT/'tuning_report.json'),
        grid_proof_sha256=sha256(ROOT/'grid_convergence.json'),test_not_evaluated=True)
    write_json(ROOT/'tuning_verification.json',proof)
    print(json.dumps(dict(status=proof['status'],models=proof['models'],maximum_error=proof['maximum_error'])),flush=True)


def confirm():
    proof = json.loads((ROOT/'tuning_verification.json').read_text())
    if (proof['status'] != 'tuning_independently_verified' or proof['models'] != 15 or
            proof['auditor_sha256'] != sha256(Path(__file__))):
        raise ValueError('complete independent clock proof required')
    for key,file in (('selection_sha256','selected_before_confirmation.json'),('contract_sha256','contract.json'),
            ('report_sha256','tuning_report.json'),('grid_proof_sha256','grid_convergence.json')):
        if proof[key] != sha256(ROOT/file):raise ValueError('clock proof stale: '+file)
    chosen = json.loads((ROOT/'selected_before_confirmation.json').read_text())['selected']
    if (ROOT/'nomination_lock.json').exists():raise FileExistsError('clock confirmation already locked')
    write_json(ROOT/'nomination_lock.json',dict(selected=chosen,selection_sha256=proof['selection_sha256'],
        auditor_sha256=proof['auditor_sha256'],nominated=[] if chosen=='original_teacher' else [chosen]))
    if chosen=='original_teacher':
        write_json(ROOT/'state.json',dict(status='verified_teacher_selected',goal_achieved=False));return
    evaluate_confirmation(chosen, replay=False)


def evaluate_confirmation(chosen, *, replay):
    lock = json.loads((ROOT/'nomination_lock.json').read_text())
    if lock['selected']!=chosen or lock['auditor_sha256']!=sha256(Path(__file__)):
        raise ValueError('clock confirmation lock differs')
    if lock['selection_sha256']!=sha256(ROOT/'selected_before_confirmation.json'):
        raise ValueError('clock selection changed after lock')
    (d,c,h,ext,_,bins,_,ix,search,_),_ = inputs()
    cache=batches(d,c,ext,ix);models={};pairs={};diagnostics={};errors={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        model=teacher(base,seed,bins);data=extract(model,cache);del model
        _,_,daily,_=dense_teacher(data)
        grid,dense=refined_teacher(data,daily);models[str(seed)]={};curves={}
        for name in ('original_teacher',chosen):
            log_clock,curve,info,_,nll=infer(data,grid,dense,daily,seed,name)
            digest=sha256(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt') if info is None else info['checkpoint_sha256']
            path=ROOT/f'seed{seed}'/name/'confirmation_predictions.npz'
            if replay and not path.exists():raise FileNotFoundError('confirmation predictions absent')
            if not replay and path.exists():raise FileExistsError('confirmation exists without report; inspect first')
            errors[f'{seed}/{name}']=save_or_check(path,d,ix,data,log_clock,curve)
            models[str(seed)][name]=dict(metrics=summary(curve,d,ix,nll),checkpoint_sha256=digest);curves[name]=curve
        if not replay:
            pairs[str(seed)]=paired_interval(d.arrays['subject_ids'][ix],d.times[ix],d.events[ix],curves[chosen][:,:,1],
                curves['original_teacher'][:,:,1],replicates=200,seed=seed)
            diagnostics[str(seed)],strata=analyze(d.times[ix],d.events[ix],d.arrays['subject_ids'][ix],
                history_counts(d.history_raw[ix]),curves,seed=seed,comparators=('original_teacher',))
        del data,dense
    if replay:
        report=json.loads((ROOT/'confirmation_report.json').read_text())
        if models!=report['models'] or report['lock_sha256']!=sha256(ROOT/'nomination_lock.json'):
            raise ValueError('clock confirmation report replay differs')
        write_json(ROOT/'confirmation_verification.json',dict(status='clock_confirmation_independently_verified',
            models=len(errors),errors=errors,maximum_error=max(errors.values()),report_sha256=sha256(ROOT/'confirmation_report.json'),
            test_not_evaluated=True,goal_achieved=False))
        signal=material_signal(models,chosen,comparators=('original_teacher',))
        write_json(ROOT/'continuation_decision.json',signal)
        write_json(ROOT/'state.json',dict(status='verified_confirmation_complete',selected=chosen,decision=signal,goal_achieved=False))
    else:
        write_json(ROOT/'confirmation_report.json',dict(models=models,comparisons=pairs,selected=chosen,
            lock_sha256=sha256(ROOT/'nomination_lock.json'),test_not_evaluated=True,goal_achieved=False))
        write_json(ROOT/'calibration_and_history_strata.json',dict(models=diagnostics,strata_admissions=strata,
            confirmation_report_sha256=sha256(ROOT/'confirmation_report.json'),goal_achieved=False))
        write_json(ROOT/'state.json',dict(status='confirmation_complete_pending_replay',selected=chosen,goal_achieved=False))
    print(json.dumps(dict(status='confirmation_replayed' if replay else 'confirmation_written',models=len(errors))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('verify','confirm','verify-confirmation'),required=True)
    mode=p.parse_args().mode
    if mode=='verify':verify()
    elif mode=='confirm':confirm()
    else:
        r=json.loads((ROOT/'confirmation_report.json').read_text())
        evaluate_confirmation(r['selected'],replay=True)
