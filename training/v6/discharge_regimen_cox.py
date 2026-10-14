"""固定出院方案Cox条件诊断；绝对风险/校准与AUC分开，原V6不改。"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import sha256
from training.baselines.continuous_competing_cox import fit_competing_cox,predict_cumulative_incidence
from training.v6.admission_cox_diagnostic import SETTINGS,digest,save_model,load_model,scores
from training.v6.evaluation import GRID,calibration_bins,paired_interval
from training.v6.train import write_json

ROOT=Path('training_runs/v6_discharge_regimen_cox_2026-10-14')
OLD=Path('training_runs/v6_admission_cox_2026-10-14')
PARENT=Path('training_runs/v6_discharge_regimen_screen_2026-10-14')
PROTOCOL=Path('docs/V6_DISCHARGE_REGIMEN_COX_PROTOCOL_2026-10-14.md')
ARMS=('availability','content')


def views(h34,features,source):
    n=len(h34)
    if (h34.shape!=(n,34) or features.shape!=(n,34) or source.shape!=(n,) or source.dtype!=bool
            or not np.isfinite(h34).all() or not np.isfinite(features).all()
            or not np.isin(features,(0,1)).all() or np.any(features[~source]!=0)):
        raise ValueError('invalid frozen regimen/H34 view')
    common=np.column_stack((h34,source)).astype(np.float64)
    return dict(availability=common,content=np.concatenate((common,features),1).astype(np.float64))


def probability_readout(times,events,curve):
    # scores先验证每日CIF边界/守恒；校准只作描述，不回拟概率。
    metrics=scores(times,events,curve);result={}
    for name,cause in (('readmission',1),('death',2)):
        result[name]={}
        for day in (30,90,365):
            groups=calibration_bins(times,events,curve[:,:,cause],day=day,cause=cause)
            result[name][str(day)]=dict(brier=metrics[name]['points'][str(day)]['brier'],
                mean_predicted=float(curve[:,day,cause].mean()),
                observed_fraction=float(((events==cause)&(times<=day)).mean()),bins=groups,
                binned_weighted_absolute_gap=sum(g['n']*abs(g['predicted']-g['observed']) for g in groups)/len(times))
    return result


def lock(results,h34):
    c=results['content']['metrics'];others=(results['availability']['metrics'],h34['metrics'])
    signal=all(c['readmission']['integrated_brier_7_365']<r['readmission']['integrated_brier_7_365'] for r in others)
    material=signal and all(c['readmission']['integrated_brier_7_365']<=.99*r['readmission']['integrated_brier_7_365']
        and c['readmission']['points']['90']['auc']>=r['readmission']['points']['90']['auc']+.005
        and c['death']['integrated_brier_7_365']<=r['death']['integrated_brier_7_365'] for r in others)
    return dict(conditional_signal='content' if signal else None,material_signal=material,
        rule='vs availability and H34: IBS -1%, AUC90 +.005, death IBS not worse; calibration still required',
        default_changed=False,goal_achieved=False)


def setup():
    from training.v6.discharge_regimen_screen import setup as interface_setup
    from training.v6.exploration_model import expanded_inputs
    import torch
    data,train,tune,a,parent=interface_setup();d,c=data[:2]
    if parent!=json.loads((PARENT/'contract.json').read_text()):raise ValueError('verified parent contract differs')
    pp=json.loads((PARENT/'verification.json').read_text())
    if pp['maximum_error']!=0 or not pp['no_refitting'] or pp['screening_sha256']!=sha256(PARENT/'screening.json'):
        raise ValueError('parent screen proof differs')
    old=json.loads((OLD/'contract.json').read_text());proof=json.loads((OLD/'verification.json').read_text())
    if proof['maximum_error']!=0 or not proof['no_refitting'] or proof['report_sha256']!=sha256(OLD/'report.json'):
        raise ValueError('old H34 proof differs')
    if old['solver']!=SETTINGS or old['inherited_data_provenance']!=d.provenance:raise ValueError('old settings/data differ')
    sources={**parent['source_sha256'],**old['source_sha256']}
    for path,h in sources.items():
        if sha256(Path(path))!=h:raise ValueError('protected source changed: '+path)
    h34=np.concatenate((d.history,np.where(d.current_masks,d.current,0),d.current_masks.astype(d.history.dtype)),1)
    ix=np.unique(np.linspace(0,len(h34)-1,64,dtype=int))
    for device in ('cpu','cuda'):
        with torch.no_grad():h,b,_,_=c.collate(ix,device=device);actual=expanded_inputs(h,b).cpu().numpy()
        np.testing.assert_array_equal(actual,h34[ix])
    v=views(h34,a['features'],a['section_available'])
    np.testing.assert_array_equal(v['availability'],v['content'][:,:35])
    inherited=dict(h34=digest(h34.astype(np.float64)),train=digest(train),tune=digest(tune),times=digest(d.times),events=digest(d.events))
    if any(old['array_sha256'][k]!=h for k,h in inherited.items()):raise ValueError('H34/indices/labels differ')
    oldresult=json.loads((OLD/'report.json').read_text())['results']['h34']
    if sha256(OLD/'h34/model.npz')!=oldresult['model_sha256']:raise ValueError('old H34 model changed')
    model=load_model(OLD/'h34/model.npz')
    if not(model.readmission.converged and model.death.converged):raise ValueError('old model not converged')
    curve=predict_cumulative_incidence(model,h34[tune].astype(np.float64),GRID)
    with np.load(OLD/'h34/tuning_predictions.npz') as z:
        for k,value in (('curve',curve),('indices',tune),('times',d.times[tune]),('events',d.events[tune])):np.testing.assert_array_equal(z[k],value)
    if scores(d.times[tune],d.events[tune],curve)!=oldresult['metrics']:raise ValueError('H34 scores differ')
    oldresult={**oldresult,'probability_readout':probability_readout(d.times[tune],d.events[tune],curve)}
    reference=json.loads((PARENT/'screening.json').read_text())['reference']
    for path in (Path(__file__),PROTOCOL,Path('training/tests/test_v6_discharge_regimen_cox.py'),
        Path('training/v6/admission_cox_diagnostic.py'),Path('training/baselines/continuous_competing_cox.py'),
        Path('training/v6/evaluation.py'),PARENT/'contract.json',PARENT/'verification.json',PARENT/'screening.json',
        OLD/'contract.json',OLD/'report.json',OLD/'verification.json',OLD/'h34/model.npz',OLD/'h34/tuning_predictions.npz'):
        sources[str(path)]=sha256(path)
    contract=dict(source_sha256=sources,data_provenance=d.provenance,solver=SETTINGS,
        parent_teacher_sha256=parent['teacher_sha256'],parent_base_sha256=parent['base_sha256'],
        array_sha256={**inherited,**{k:digest(x) for k,x in v.items()}},widths={k:x.shape[1] for k,x in v.items()},
        fixed_interface_sha256=parent['data_sha256'],new_scaling_fitted=False,h34_reused=True,h34_replay_error=0.,
        common_expanded_inputs_errors=dict(cpu=0.,cuda=0.),no_high_frequency=True,confirmation_test_used=False,
        probability_readout='WorldA descriptive 30/90/365; no calibration refitting or clinical accuracy claim')
    return d,train,tune,v,contract,oldresult,reference


def run():
    if ROOT.exists():raise FileExistsError('batch exists; no repeated fitting')
    d,train,tune,v,contract,h34,reference=setup();ROOT.mkdir();write_json(ROOT/'contract.json',contract)
    write_json(ROOT/'preflight.json',dict(status='passed',related_tests=11,common_input_errors={'cpu':0.,'cuda':0.},
        reused_h34_risk_error=0.,new_scaling=False,deterministic_cpu64_fitting=True))
    results={}
    for arm in ARMS:
        write_json(ROOT/'state.json',dict(status='fixed_cox_fitting',arm=arm,goal_achieved=False))
        started=time.perf_counter();folder=ROOT/arm;folder.mkdir()
        model=fit_competing_cox(v[arm][train],d.times[train],d.events[train],**SETTINGS);save_model(folder/'model.npz',model)
        if not(model.readmission.converged and model.death.converged):raise RuntimeError('fixed solver not converged: '+arm)
        curve=predict_cumulative_incidence(model,v[arm][tune],GRID)
        result=dict(feature_width=v[arm].shape[1],metrics=scores(d.times[tune],d.events[tune],curve),
            probability_readout=probability_readout(d.times[tune],d.events[tune],curve),
            fitting_and_prediction_seconds=time.perf_counter()-started,readmission_converged=True,death_converged=True,
            actual_iteration_count_available=False,max_iterations_per_cause=40,
            stop_reason='accepted_Newton_step_below_tolerance',model_sha256=sha256(folder/'model.npz'))
        np.savez_compressed(folder/'tuning_predictions.npz',curve=curve,indices=tune,times=d.times[tune],events=d.events[tune])
        results[arm]=result;write_json(folder/'result.json',result)
        print(json.dumps(dict(arm=arm,seconds=result['fitting_and_prediction_seconds'],
            ibs=result['metrics']['readmission']['integrated_brier_7_365'],auc90=result['metrics']['readmission']['points']['90']['auc'],
            death_ibs=result['metrics']['death']['integrated_brier_7_365'])),flush=True)
    chosen=lock(results,h34);report=dict(results=results,h34_reference=h34,full_v6_reference=reference,lock=chosen,
        train_admissions=len(train),tuning_admissions=len(tune),new_fits=2,
        limits=['partial input Cox, not complete V6 ranking','exposed tuning, WorldA incomplete out-of-hospital followup',
                'death is first-event death before strict readmission, not all later mortality'])
    if chosen['material_signal']:
        with np.load(ROOT/'content/tuning_predictions.npz') as z:full=z['curve']
        with np.load(ROOT/'availability/tuning_predictions.npz') as z:other=z['curve']
        report['paired']=paired_interval(d.arrays['subject_ids'][tune],d.times[tune],d.events[tune],full[:,:,1],other[:,:,1],replicates=200,seed=42)
    write_json(ROOT/'report.json',report);write_json(ROOT/'nomination_lock.json',chosen)
    write_json(ROOT/'state.json',dict(status='pending_independent_replay',goal_achieved=False));print(json.dumps(chosen),flush=True)


def verify():
    d,train,tune,v,contract,h34,reference=setup()
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    report=json.loads((ROOT/'report.json').read_text());errors={}
    if report['h34_reference']!=h34 or report['full_v6_reference']!=reference:raise ValueError('reference changed')
    for arm in ARMS:
        folder=ROOT/arm;result=report['results'][arm]
        if sha256(folder/'model.npz')!=result['model_sha256']:raise ValueError('new model changed')
        model=load_model(folder/'model.npz')
        if model.l2!=SETTINGS['l2'] or model.horizon_days!=365 or not(model.readmission.converged and model.death.converged):raise ValueError('model settings/convergence differ')
        curve=predict_cumulative_incidence(model,v[arm][tune],GRID)
        with np.load(folder/'tuning_predictions.npz') as z:
            for k,value in (('indices',tune),('times',d.times[tune]),('events',d.events[tune])):np.testing.assert_array_equal(z[k],value)
            errors[arm]=float(np.abs(z['curve']-curve).max())
        if errors[arm]!=0 or scores(d.times[tune],d.events[tune],curve)!=result['metrics'] or probability_readout(d.times[tune],d.events[tune],curve)!=result['probability_readout']:
            raise ValueError('daily risk/metrics/calibration differ')
    if lock(report['results'],h34)!=report['lock'] or report['lock']!=json.loads((ROOT/'nomination_lock.json').read_text()):raise ValueError('lock differs')
    write_json(ROOT/'verification.json',dict(status='independent_fixed_interface_view_model_daily_probability_calibration_lock_replay',
        maximum_error=max(errors.values()),errors=errors,no_refitting=True,no_raw_or_saved_text_reparse=True,
        h34_reused_prediction_error=0.,report_sha256=sha256(ROOT/'report.json')))
    write_json(ROOT/'state.json',dict(status='fixed_cox_independently_verified',goal_achieved=False));print(json.dumps(errors),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=('run','verify'));mode=parser.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as error:
        if ROOT.exists():write_json(ROOT/f'failure_{mode}.json',dict(error=repr(error)))
        raise
