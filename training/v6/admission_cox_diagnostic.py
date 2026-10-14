"""已核验常规接口的固定原因别Cox诊断；不修改正式V6。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np

from patient_data.export_mimiciv import sha256
from training.baselines.continuous_competing_cox import (
    CauseSpecificCox, CompetingCoxModel, fit_competing_cox,
    predict_cumulative_incidence,
)
from training.v6.data import HISTORY_NAMES, CURRENT_NAMES
from training.v6.evaluation import metrics, public_metrics, GRID, paired_interval
from training.v6.train import write_json

ROOT = Path('training_runs/v6_admission_cox_2026-10-14')
PARENT = Path('training_runs/v6_admission_context_2026-10-14')
PROTOCOL = Path('docs/V6_ADMISSION_COX_PROTOCOL_2026-10-14.md')
ARMS = ('h34', 'availability', 'content')
SETTINGS = dict(horizon_days=365., l2=10., max_iter=40, tolerance=1e-6)


def digest(value):
    import hashlib
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def input_view(h34, context, masks, arm):
    h34, context, masks = map(np.asarray, (h34, context, masks))
    n = len(h34)
    if (h34.shape != (n, 34) or context.shape != (n, 32) or masks.shape != (n, 4)
            or masks.dtype != bool or not np.isfinite(h34).all()
            or not np.isfinite(context).all() or np.any(context[:, 25:] != 0)
            or not np.array_equal(context[:, 2:6], masks)):
        raise ValueError('invalid audited feature interface')
    if arm == 'h34': result = h34
    elif arm == 'availability': result = np.concatenate((h34, masks), axis=1)
    elif arm == 'content': result = np.concatenate((h34, context[:, :25]), axis=1)
    else: raise ValueError('unknown condition')
    return result.astype(np.float64)


def save_model(path, model):
    arrays = dict(horizon=np.asarray(model.horizon_days), l2=np.asarray(model.l2))
    for name, cause in (('readmission', model.readmission), ('death', model.death)):
        for key in ('coefficients', 'event_times', 'baseline_hazard_increments', 'converged'):
            arrays[name+'_'+key] = np.asarray(getattr(cause, key))
    np.savez_compressed(path, **arrays)


def load_model(path):
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    def cause(name):
        return CauseSpecificCox(*(a[name+'_'+k] for k in
            ('coefficients', 'event_times', 'baseline_hazard_increments')),
            bool(a[name+'_converged']))
    return CompetingCoxModel(cause('readmission'), cause('death'), float(a['horizon']), float(a['l2']))


def validate_curve(curve):
    if (curve.ndim != 3 or curve.shape[1:] != (366, 3) or not np.isfinite(curve).all()
            or np.any((curve < 0) | (curve > 1)) or np.any(np.diff(curve[:, :, 1:], axis=1) < -1e-12)):
        raise ValueError('invalid daily competing probabilities')
    if not np.allclose(curve.sum(2), 1., atol=1e-12, rtol=0):
        raise ValueError('probabilities do not conserve mass')
    if not np.array_equal(curve[:, 0], np.tile([1., 0., 0.], (len(curve), 1))):
        raise ValueError('invalid zero-time probabilities')


def scores(times, events, curve):
    validate_curve(curve)
    return dict(readmission=public_metrics(metrics(times, events, curve[:, :, 1])),
                death=public_metrics(metrics(times, events, curve[:, :, 2], cause=2)))


def decision(results):
    r = results['content']['metrics']; ibis = r['readmission']['integrated_brier_7_365']
    others = [results[k]['metrics'] for k in ARMS[:2]]
    signal = all(ibis < x['readmission']['integrated_brier_7_365'] for x in others)
    material = signal and all(
        ibis <= .99*x['readmission']['integrated_brier_7_365']
        and r['readmission']['points']['90']['auc'] >= x['readmission']['points']['90']['auc']+.005
        and r['death']['integrated_brier_7_365'] <= x['death']['integrated_brier_7_365']
        for x in others)
    return dict(conditional_signal='content' if signal else None, material_signal=material,
        rule='content IBS strictly beats both; material >=1% IBS and >=.005 AUC90, death IBS not worse',
        replace_v6=False, goal_achieved=False)


def setup():
    # 复用既定核心队列/标签检查，不重新抽取新源或编码文本。
    from training.v6.regularization_audit import checked
    from training.v6.exploration_model import expanded_inputs
    import torch
    data = checked(); d, c = data[:2]
    manifest = json.loads((PARENT/'interface_manifest.json').read_text())
    proof = json.loads((PARENT/'verification.json').read_text())
    if (proof['maximum_error'] != 0 or not proof['interface_rebuilt']
            or proof['screening_sha256'] != sha256(PARENT/'screening.json')):
        raise ValueError('parent proof stale')
    if manifest['sha256'] != sha256(PARENT/'interface.npz'):
        raise ValueError('parent archive changed')
    parent = json.loads((PARENT/'contract.json').read_text())
    for p, h in parent['source_sha256'].items():
        if sha256(Path(p)) != h: raise ValueError('protected source changed: '+p)
    with np.load(PARENT/'interface.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    for key in ('hadm_ids', 'subject_ids', 'patient_splits'):
        np.testing.assert_array_equal(arrays[key], d.arrays[key])
    h34 = np.concatenate((d.history, np.where(d.current_masks, d.current, 0),
                          d.current_masks.astype(d.history.dtype)), axis=1)
    ix = np.unique(np.linspace(0, len(h34)-1, 64, dtype=int))
    common_error = {}
    for device in ('cpu', 'cuda'):
        with torch.no_grad():
            h, batch, _, _ = c.collate(ix, device=device)
            actual = expanded_inputs(h, batch).cpu().numpy()
        np.testing.assert_array_equal(actual, h34[ix]); common_error[device] = 0.
    train = np.flatnonzero(d.arrays['patient_splits'] == 'train'); tune = data[6]
    if (len(train), len(tune), len(h34)) != (13592, 1531, 19401): raise ValueError('cohort differs')
    views = {arm: input_view(h34, arrays['features'], arrays['masks'], arm) for arm in ARMS}
    names = [*HISTORY_NAMES, *CURRENT_NAMES, *(k+'_valid' for k in CURRENT_NAMES)]
    columns = dict(h34=names, availability=names+manifest['columns'][2:6],
                   content=names+manifest['columns'][:25])
    sources = dict(parent['source_sha256'])
    for p in (Path(__file__), PROTOCOL, Path('training/tests/test_v6_admission_cox.py'),
              Path('training/baselines/continuous_competing_cox.py'), Path('training/v6/exploration_model.py'),
              PARENT/'interface.npz', PARENT/'interface_manifest.json', PARENT/'contract.json',
              PARENT/'verification.json', PARENT/'screening.json'):
        sources[str(p)] = sha256(p)
    contract = dict(source_sha256=sources, inherited_data_provenance=d.provenance,
        parent_teacher_sha256=parent['teacher_sha256'],
        parent_base_sha256=data[8]['frozen_baseline_sha256'], solver=SETTINGS,
        features=columns, inherited_train_preprocessing=True, new_scaling_fitted=False,
        array_sha256={**{k:digest(v) for k,v in views.items()},
            'train':digest(train), 'tune':digest(tune), 'times':digest(d.times), 'events':digest(d.events)},
        common_expanded_inputs_errors=common_error, no_high_frequency=True,
        confirmation_test_used=False, deterministic_from_zero=True)
    return d, train, tune, views, contract


def run():
    if ROOT.exists(): raise FileExistsError('diagnostic already exists; inspect, do not repeat fit')
    d, train, tune, views, contract = setup()
    ROOT.mkdir(); write_json(ROOT/'contract.json', contract)
    write_json(ROOT/'preflight.json', dict(related_tests=9,
        common_expanded_inputs_errors=contract['common_expanded_inputs_errors'],
        toy_solver_convergence_and_serialization_checked=True,
        parent_source_proof_reused=True, cpu_solver_not_gpu_neural_training=True))
    write_json(ROOT/'state.json', dict(status='fixed_cox_fitting', goal_achieved=False))
    results = {}
    for arm in ARMS:
        start = time.perf_counter(); folder = ROOT/arm; folder.mkdir()
        model = fit_competing_cox(views[arm][train], d.times[train], d.events[train], **SETTINGS)
        save_model(folder/'model.npz', model)
        if not (model.readmission.converged and model.death.converged):
            raise RuntimeError('Cox did not converge in fixed budget: '+arm)
        curve = predict_cumulative_incidence(model, views[arm][tune], GRID)
        result = dict(feature_width=views[arm].shape[1], metrics=scores(d.times[tune], d.events[tune], curve),
            fitting_and_prediction_seconds=time.perf_counter()-start,
            readmission_converged=model.readmission.converged, death_converged=model.death.converged,
            max_iterations_per_cause=40, actual_iteration_count_available=False,
            stop_reason='accepted_Newton_step_below_tolerance', model_sha256=sha256(folder/'model.npz'))
        np.savez_compressed(folder/'tuning_predictions.npz', curve=curve, indices=tune,
                            times=d.times[tune], events=d.events[tune])
        write_json(folder/'result.json', result); results[arm] = result
        print(json.dumps(dict(arm=arm, seconds=result['fitting_and_prediction_seconds'],
            auc90=result['metrics']['readmission']['points']['90']['auc'],
            ibs=result['metrics']['readmission']['integrated_brier_7_365'])), flush=True)
    lock = decision(results); write_json(ROOT/'nomination_lock.json', lock)
    report = dict(results=results, lock=lock, train_admissions=len(train), tuning_admissions=len(tune),
        limits=['minimal H34 comparator, not full clinical/BERT inputs',
                'exposed tuning, not blind', 'fixed CPU solver, no random seed grid'])
    if lock['material_signal']:
        with np.load(ROOT/'content/tuning_predictions.npz') as z: full = z['curve']
        report['paired'] = {}
        for arm in ARMS[:2]:
            with np.load(ROOT/arm/'tuning_predictions.npz') as z: other = z['curve']
            report['paired'][arm] = paired_interval(d.arrays['subject_ids'][tune], d.times[tune],
                d.events[tune], full[:, :, 1], other[:, :, 1], replicates=200, seed=42)
    write_json(ROOT/'report.json', report)
    write_json(ROOT/'state.json', dict(status='fixed_cox_fitted_awaiting_independent_reload', goal_achieved=False))


def verify():
    d, train, tune, views, contract = setup()
    if contract != json.loads((ROOT/'contract.json').read_text()): raise ValueError('contract changed')
    report = json.loads((ROOT/'report.json').read_text()); errors = {}
    for arm in ARMS:
        folder = ROOT/arm; info = report['results'][arm]
        if sha256(folder/'model.npz') != info['model_sha256']: raise ValueError('model changed')
        model = load_model(folder/'model.npz')
        if not (model.readmission.converged and model.death.converged): raise ValueError('not converged')
        if model.horizon_days != SETTINGS['horizon_days'] or model.l2 != SETTINGS['l2']: raise ValueError('solver differs')
        curve = predict_cumulative_incidence(model, views[arm][tune], GRID)
        with np.load(folder/'tuning_predictions.npz') as z:
            for k, value in (('indices',tune), ('times',d.times[tune]), ('events',d.events[tune])):
                np.testing.assert_array_equal(z[k], value)
            errors[arm] = float(np.abs(curve-z['curve']).max())
        if errors[arm] != 0 or scores(d.times[tune], d.events[tune], curve) != info['metrics']:
            raise ValueError('prediction/metric replay differs')
    lock = decision(report['results'])
    if lock != report['lock'] or lock != json.loads((ROOT/'nomination_lock.json').read_text()):
        raise ValueError('selection changed')
    write_json(ROOT/'verification.json', dict(status='independent_view_model_daily_prediction_metric_replay',
        maximum_error=max(errors.values()), errors=errors, report_sha256=sha256(ROOT/'report.json'),
        no_refitting=True, parent_source_proof_reused_with_unchanged_hash=True))
    write_json(ROOT/'state.json', dict(status='fixed_cox_independently_verified', goal_achieved=False))
    print(json.dumps(dict(errors=errors, lock=lock)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=('run','verify'), required=True)
    mode = p.parse_args().mode
    try: run() if mode == 'run' else verify()
    except Exception as error:
        if ROOT.exists(): write_json(ROOT/f'failure_{mode}.json', {'error':repr(error)})
        raise
