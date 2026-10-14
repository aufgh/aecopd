"""复用核验接口的54/64列Cox诊断；不重扫检验、不改变正式V6。"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import sha256
from training.baselines.continuous_competing_cox import fit_competing_cox, predict_cumulative_incidence
from training.v6.admission_cox_diagnostic import (
    SETTINGS, digest, save_model, load_model, scores,
)
from training.v6.evaluation import GRID, paired_interval
from training.v6.train import write_json

ROOT = Path('training_runs/v6_core_lab_cox_2026-10-14')
PARENT = Path('training_runs/v6_core_lab_continuous_2026-10-14')
OLD = Path('training_runs/v6_admission_cox_2026-10-14')
PROTOCOL = Path('docs/V6_CORE_LAB_COX_PROTOCOL_2026-10-14.md')
ARMS = ('summary', 'continuous')


def views(h34, features, observed):
    n = len(h34)
    if (h34.shape != (n,34) or features.shape != (n,32) or observed.shape != (n,5)
            or observed.dtype != bool or not np.isfinite(h34).all()
            or not np.isfinite(features).all() or np.any(features[:,30:] != 0)
            or not np.array_equal(features[:,20:25], observed)):
        raise ValueError('invalid verified core interface')
    if np.any(features[:,:10].reshape(n,5,2)[~observed] != 0):
        raise ValueError('missing raw value is not masked')
    return {k:np.concatenate((h34,features[:,10:30] if k=='summary' else features[:,:30]),1).astype(np.float64)
            for k in ARMS}


def lock(results):
    a = results['summary']['metrics']; b = results['continuous']['metrics']
    ra, rb = a['readmission'], b['readmission']
    signal = rb['integrated_brier_7_365'] < ra['integrated_brier_7_365']
    material = (signal and rb['integrated_brier_7_365'] <= .99*ra['integrated_brier_7_365']
                and rb['points']['90']['auc'] >= ra['points']['90']['auc']+.005
                and b['death']['integrated_brier_7_365'] <= a['death']['integrated_brier_7_365'])
    return dict(conditional_signal='continuous' if signal else None, material_signal=material,
                rule='vs matched summary: IBS -1%, AUC90 +.005, death IBS not worse',
                replace_v6=False, goal_achieved=False)


def setup():
    from training.v6.regularization_audit import checked
    from training.v6.exploration_model import expanded_inputs
    from training.v6.fast_multiview import digest as array_digest
    import torch
    data=checked(); d,c=data[:2]
    parent=json.loads((PARENT/'contract.json').read_text())
    manifest=json.loads((PARENT/'interface_manifest.json').read_text())
    proof=json.loads((PARENT/'verification.json').read_text())
    if (proof['maximum_error'] != 0 or not proof['no_refitting']
            or proof['screening_sha256'] != sha256(PARENT/'screening.json')
            or manifest['sha256'] != sha256(PARENT/'interface.npz')):
        raise ValueError('parent proof stale')
    old=json.loads((OLD/'contract.json').read_text()); oldproof=json.loads((OLD/'verification.json').read_text())
    if (oldproof['maximum_error'] != 0 or not oldproof['no_refitting']
            or oldproof['report_sha256'] != sha256(OLD/'report.json') or old['solver'] != SETTINGS
            or old['inherited_data_provenance'] != d.provenance):
        raise ValueError('old H34 reuse proof/settings differ')
    sources={**parent['source_sha256'],**old['source_sha256']}
    for p,h in sources.items():
        if sha256(Path(p)) != h: raise ValueError('protected source changed: '+p)
    with np.load(PARENT/'interface.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    if array_digest(a) != parent['interface_sha256']:raise ValueError('parent arrays differ')
    for key in ('hadm_ids','subject_ids','patient_splits'):np.testing.assert_array_equal(a[key],d.arrays[key])
    h34=np.concatenate((d.history,np.where(d.current_masks,d.current,0),d.current_masks.astype(d.history.dtype)),1)
    ix=np.unique(np.linspace(0,len(h34)-1,64,dtype=int))
    for device in ('cpu','cuda'):
        with torch.no_grad():
            h,b,_,_=c.collate(ix,device=device);actual=expanded_inputs(h,b).cpu().numpy()
        np.testing.assert_array_equal(actual,h34[ix])
    train=np.flatnonzero(d.arrays['patient_splits']=='train'); tune=data[6]
    if (len(h34),len(train),len(tune)) != (19401,13592,1531):raise ValueError('cohort differs')
    v=views(h34,a['features'],a['concept_observed'])
    np.testing.assert_array_equal(v['summary'][:,34:],v['continuous'][:,44:])
    inherited=dict(h34=digest(h34.astype(np.float64)),train=digest(train),tune=digest(tune),
                   times=digest(d.times),events=digest(d.events))
    if any(old['array_sha256'][k] != x for k,x in inherited.items()):raise ValueError('H34 reuse inputs differ')
    oldresult=json.loads((OLD/'report.json').read_text())['results']['h34']
    if sha256(OLD/'h34/model.npz') != oldresult['model_sha256']:raise ValueError('old H34 model changed')
    base=load_model(OLD/'h34/model.npz')
    if not(base.readmission.converged and base.death.converged):raise ValueError('old not converged')
    curve=predict_cumulative_incidence(base,h34[tune].astype(np.float64),GRID)
    with np.load(OLD/'h34/tuning_predictions.npz') as z:
        for k,x in (('curve',curve),('indices',tune),('times',d.times[tune]),('events',d.events[tune])):
            np.testing.assert_array_equal(z[k],x)
    if scores(d.times[tune],d.events[tune],curve) != oldresult['metrics']:raise ValueError('old metrics differ')
    for p in (Path(__file__),PROTOCOL,Path('training/tests/test_v6_core_lab_cox.py'),
              Path('training/v6/admission_cox_diagnostic.py'),PARENT/'interface.npz',PARENT/'interface_manifest.json',
              PARENT/'contract.json',PARENT/'verification.json',PARENT/'screening.json',
              OLD/'contract.json',OLD/'verification.json',OLD/'report.json',OLD/'h34/model.npz',OLD/'h34/tuning_predictions.npz'):
        sources[str(p)]=sha256(p)
    contract=dict(source_sha256=sources,data_provenance=d.provenance,solver=SETTINGS,
        parent_teacher_sha256=parent['teacher_sha256'],parent_base_sha256=parent['base_sha256'],
        array_sha256={**inherited,**{k:digest(x) for k,x in v.items()}},
        axes=manifest['axes'],inherited_scales=manifest['scales'],new_scaling_fitted=False,
        features=dict(summary='H34 + old first/latest severity10 + observed5 + old age5',
                      continuous='H34 + physical first/latest10 + same old20'),
        widths={k:x.shape[1] for k,x in v.items()},h34_reference_reused=True,h34_prediction_replay_error=0.,
        common_expanded_inputs_errors=dict(cpu=0.,cuda=0.),no_high_frequency=True,confirmation_test_used=False)
    reference=json.loads((PARENT/'screening.json').read_text())['reference']
    return d,train,tune,v,contract,oldresult,reference


def run():
    if ROOT.exists():raise FileExistsError('batch exists; do not repeat fitting')
    d,train,tune,v,contract,oldresult,reference=setup()
    ROOT.mkdir();write_json(ROOT/'contract.json',contract)
    write_json(ROOT/'preflight.json',dict(common_expanded_inputs_errors=contract['common_expanded_inputs_errors'],
        parent_source_proof_reused=True,h34_reused_prediction_error=0.,related_tests=7,
        cpu64_fixed_solver=True,not_gpu_neural_training=True))
    results={}
    for arm in ARMS:
        write_json(ROOT/'state.json',dict(status='fixed_cox_fitting',arm=arm,goal_achieved=False))
        start=time.perf_counter();folder=ROOT/arm;folder.mkdir()
        model=fit_competing_cox(v[arm][train],d.times[train],d.events[train],**SETTINGS)
        save_model(folder/'model.npz',model)
        if not(model.readmission.converged and model.death.converged):raise RuntimeError('fixed solver did not converge: '+arm)
        curve=predict_cumulative_incidence(model,v[arm][tune],GRID)
        result=dict(feature_width=v[arm].shape[1],metrics=scores(d.times[tune],d.events[tune],curve),
            fitting_and_prediction_seconds=time.perf_counter()-start,
            readmission_converged=True,death_converged=True,actual_iteration_count_available=False,
            max_iterations_per_cause=40,stop_reason='accepted_Newton_step_below_tolerance',
            model_sha256=sha256(folder/'model.npz'))
        np.savez_compressed(folder/'tuning_predictions.npz',curve=curve,indices=tune,times=d.times[tune],events=d.events[tune])
        write_json(folder/'result.json',result);results[arm]=result
        print(json.dumps(dict(arm=arm,seconds=result['fitting_and_prediction_seconds'],metrics=result['metrics'])),flush=True)
    chosen=lock(results)
    report=dict(results=results,h34_reference=oldresult,full_v6_reference=reference,lock=chosen,
        train_admissions=len(train),tuning_admissions=len(tune),
        limits=['minimal conditional Cox diagnostic, not complete V6 inputs','exposed tuning, not blind',
                'deterministic CPU convergence; no seed or epoch grid'])
    if chosen['material_signal']:
        with np.load(ROOT/'continuous/tuning_predictions.npz') as z:full=z['curve']
        with np.load(ROOT/'summary/tuning_predictions.npz') as z:other=z['curve']
        report['paired']=paired_interval(d.arrays['subject_ids'][tune],d.times[tune],d.events[tune],
            full[:,:,1],other[:,:,1],replicates=200,seed=42)
    write_json(ROOT/'report.json',report);write_json(ROOT/'nomination_lock.json',chosen)
    write_json(ROOT/'state.json',dict(status='pending_independent_replay',goal_achieved=False))
    print(json.dumps(chosen),flush=True)


def verify():
    d,train,tune,v,contract,oldresult,reference=setup()
    if contract != json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract differs')
    report=json.loads((ROOT/'report.json').read_text());errors={}
    if report['h34_reference'] != oldresult or report['full_v6_reference'] != reference:raise ValueError('references changed')
    for arm in ARMS:
        folder=ROOT/arm;result=report['results'][arm]
        if sha256(folder/'model.npz') != result['model_sha256']:raise ValueError('new model changed')
        model=load_model(folder/'model.npz')
        if (model.horizon_days != SETTINGS['horizon_days'] or model.l2 != SETTINGS['l2']
                or not(model.readmission.converged and model.death.converged)):raise ValueError('solver differs')
        curve=predict_cumulative_incidence(model,v[arm][tune],GRID)
        with np.load(folder/'tuning_predictions.npz') as z:
            for k,x in (('indices',tune),('times',d.times[tune]),('events',d.events[tune])):np.testing.assert_array_equal(z[k],x)
            errors[arm]=float(np.abs(z['curve']-curve).max())
        if errors[arm] != 0 or scores(d.times[tune],d.events[tune],curve) != result['metrics']:raise ValueError('replay differs')
    decision=lock(report['results'])
    if decision != report['lock'] or decision != json.loads((ROOT/'nomination_lock.json').read_text()):raise ValueError('lock differs')
    write_json(ROOT/'verification.json',dict(status='independent_verified_interface_view_model_daily_metric_replay',
        maximum_error=max(errors.values()),errors=errors,no_refitting=True,no_raw_csv_reparse=True,
        h34_reused_prediction_error=0.,report_sha256=sha256(ROOT/'report.json')))
    write_json(ROOT/'state.json',dict(status='fixed_cox_independently_verified',goal_achieved=False))
    print(json.dumps(dict(errors=errors,lock=decision)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=('run','verify'))
    mode=p.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as error:
        if ROOT.exists():write_json(ROOT/f'failure_{mode}.json',dict(error=repr(error)))
        raise
