"""原V6局部替换完整训练；共同输入/基线/头，验证选择后再看测试。"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

from patient_data.export_mimiciv import sha256
from training.v6.token_data import load_inputs
from training.v6.token_experiment import legacy_batches
from training.v6.original_interaction import ExpandedBatch
from training.v6.horizon_objective import restore
from training.v6.module_surgery import (ARMS, SurgeryModel, fit_lab_bins, member_nll,
    probability_curve, shared_initial_errors)
from training.v6.token_model import daily_probabilities
from training.v6.evaluation import metrics, public_metrics, paired_interval, calibration_bins
from training.v6.train import write_json

ROOT = Path('training_runs/v6_module_surgery_2026-10-12_r3')
PROTOCOL = Path('docs/V6_MODULE_SURGERY_PROTOCOL_2026-10-12.md')
REFERENCE = Path('training_runs/v6_original_interaction_2026-10-12_r3')
SEEDS = (42, 7)


def inputs():
    config = json.loads((REFERENCE/'config.json').read_text())
    old = json.loads((REFERENCE/'report.json').read_text())
    proof = json.loads((REFERENCE/'independent_verification.json').read_text())
    if proof['report_sha256'] != sha256(REFERENCE/'report.json'):
        raise ValueError('old reference proof stale')
    for file, digest in old['contract']['code_sha256'].items():
        if sha256(Path(file)) != digest:
            raise ValueError('old source changed: ' + file)
    d, c, h, ext, _, prov = load_inputs(*(Path(config[k]) for k in
        ('reference', 'compact_dir', 'extension', 'medication')))
    if d.provenance != old['contract']['data_provenance'] or c.provenance != old['contract']['compact_provenance']:
        raise ValueError('original input provenance differs')
    return d, c, h, ext, config, old, prov


def batches(d, c, ext, indices):
    cached = legacy_batches(d, c, ext, indices)
    output = []
    for pos, h, b, t, e in cached:
        index = b.base.clinical.snippet_index.cpu().numpy()
        counts, slots = {}, []
        for admission, _ in index:
            slots.append(counts.get(admission, 0)); counts[admission] = counts.get(admission, 0) + 1
        wrapped = ExpandedBatch(b, torch.zeros(len(pos), 10, 10, device='cuda'),
            torch.zeros(len(pos), 38, 23, device='cuda'), torch.zeros(len(pos), 38, dtype=torch.bool, device='cuda'),
            torch.as_tensor(slots, dtype=torch.long, device='cuda'))
        output.append((pos, h, wrapped, t, e))
    return output


@torch.no_grad()
def parameters(model, cache):
    model.eval(); aa, kk, loss, count = [], [], 0., 0
    for _, h, b, t, e in cache:
        out = model(h, b)
        aa.append(out.log_amplitude.cpu().numpy()); kk.append(out.shape.cpu().numpy())
        loss += float(member_nll(out, t, e)) * len(t); count += len(t)
    return np.concatenate(aa), np.concatenate(kk), loss/count


def validation(model, cache, times, events):
    a, k, loss = parameters(model, cache); curve = probability_curve(a, k)
    y = ((events[:, None] == 1) & (times[:, None] <= np.arange(366))).astype(float)
    brier = ((curve[:, :, 1] - y)**2).mean(0)
    from training.baselines.rank_metrics import auc_rank
    return dict(nll=loss, ibs=float(np.trapezoid(brier[7:], np.arange(7, 366))/358),
        brier90=float(brier[90]), auc90=float(auc_rank(curve[:, 90, 1], y[:, 90].astype(np.int8))),
        log_amplitude_quantiles=np.quantile(a.reshape(-1, 2), [.01, .5, .99], axis=0).tolist(),
        shape_quantiles=np.quantile(k.reshape(-1, 2), [.01, .5, .99], axis=0).tolist())


def fit(model, train, val, times, events, path, *, seed, epochs=100, lr=1e-4):
    path.mkdir(parents=True); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    frozen = {k: v.detach().cpu().clone() for k, v in model.baseline.state_dict().items()}
    history, best, chosen = [], math.inf, 0; start = time.perf_counter()
    def record(epoch, online=None, gradient=None, current_lr=0.):
        nonlocal best, chosen
        detail = validation(model, val, times, events)
        _, _, train_nll = parameters(model, train[:8])
        history.append(dict(epoch=epoch, train_online_nll=online, train_fixed2048_nll=train_nll,
            gradient_norm_before_clip=gradient, lr=current_lr, **{'validation_'+k: v for k, v in detail.items()}))
        if detail['ibs'] < best:
            best, chosen = detail['ibs'], epoch
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch, validation_ibs=best), path/'best.pt')
        if epoch % 10 == 0 or epoch == epochs:
            write_json(path/'history.json', history)
            print(json.dumps(dict(arm=model.arm, seed=seed, epoch=epoch, best_epoch=chosen,
                ibs=detail['ibs'], auc90=detail['auc90'], seconds=round(time.perf_counter()-start, 1))), flush=True)
    record(0)
    for epoch in range(1, epochs+1):
        model.train(); warm = min(1., epoch/5)
        cosine = .1 + .9 * .5 * (1 + math.cos(math.pi*max(0, epoch-5)/max(1, epochs-5)))
        current_lr = lr * warm * cosine
        for group in opt.param_groups:
            group['lr'] = current_lr
        total, count, norms = 0., 0, []
        for j in rng.permutation(len(train)):
            _, h, b, t, e = train[j]; opt.zero_grad(set_to_none=True)
            out = model(h, b); nll = member_nll(out, t, e); objective = nll + 1e-4*out.delta.square().mean()
            if not torch.isfinite(objective):
                raise ValueError('nonfinite objective')
            objective.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError('nonfinite gradients')
            opt.step(); total += float(nll.detach())*len(t); count += len(t); norms.append(float(norm))
        record(epoch, total/count, float(np.mean(norms)), current_lr)
    if not all(torch.equal(v, model.baseline.state_dict()[k].cpu()) for k, v in frozen.items()):
        raise ValueError('frozen H34 changed')
    write_json(path/'history.json', history)
    result = dict(best_epoch=chosen, completed_epochs=epochs, validation_ibs=best, lr=lr,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        checkpoint_sha256=sha256(path/'best.pt'), elapsed_seconds=time.perf_counter()-start,
        frozen_baseline_verified=True)
    write_json(path/'training_result.json', result)
    return result


def summary(curve, d, indices, nll):
    return dict(nll=nll,
        readmission=public_metrics(metrics(d.times[indices], d.events[indices], curve[:, :, 1], ordering=True)),
        death=public_metrics(metrics(d.times[indices], d.events[indices], curve[:, :, 2], cause=2, ordering=True)),
        calibration={str(day): calibration_bins(d.times[indices], d.events[indices], curve[:, :, 1], day=day)
                     for day in (30, 90, 180, 365)})


def save(path, d, ix, a, k, curve):
    daily, conditional = daily_probabilities(curve)
    np.savez_compressed(path, hadm_ids=d.arrays['hadm_ids'][ix], subject_ids=d.arrays['subject_ids'][ix],
        times=d.times[ix], events=d.events[ix], log_amplitude=a, shape=k, curve=curve,
        daily_event_probability=daily, conditional_daily_event_probability=conditional,
        conditional_daily_valid=curve[:, :-1, 0] > 1e-12)


def model_for(base, seed, arm, preprocessing):
    m = SurgeryModel(base, seed, arm, preprocessing['bins']).cuda()
    torch.manual_seed(seed)
    return m


def setup():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(True)


def original_replay(base, seed, cache, fold):
    """旧权重精确重放与重新训练的指标近似复现，是不同验收层。"""
    from training.v6.expanded_original_experiment import new_model
    model = new_model(base, seed, 'control')
    old_path = REFERENCE/f'seed{seed}/control'
    model.load_state_dict(torch.load(old_path/'best.pt', map_location='cuda', weights_only=True)['state_dict'])
    a, k, _ = parameters(model, cache)
    curve = probability_curve(a, k)
    with np.load(old_path/(fold+'_predictions.npz')) as z:
        error = float(np.abs(z['curve']-curve).max())
    if error > 3e-7:
        raise ValueError('old checkpoint exact replay failed')
    return curve, error


def run():
    setup(); start = time.perf_counter()
    if ROOT.exists():
        raise FileExistsError(ROOT)
    d, c, h, ext, config, old, prov = inputs()
    ix = {f: np.flatnonzero(d.arrays['patient_splits'] == f) for f in ('train', 'validation', 'test')}
    bins = fit_lab_bins(d.arrays['lab_axis_inputs'], d.arrays['patient_splits'] == 'train')
    ROOT.mkdir(); write_json(ROOT/'preprocessing.json', bins)
    # 旧文件保护 + 本轮代码，验证进程再次核对。
    protected = dict(old['contract']['code_sha256'])
    for p in ('training/v6/model.py', 'training/v6/risk.py', 'training/v6/evaluation.py',
              'training/v6/token_data.py', 'training/v6/horizon_objective.py',
              'training/v6/module_surgery.py', 'training/v6/module_surgery_experiment.py',
              'training/tests/test_v6_module_surgery.py'):
        protected[p] = sha256(Path(p))
    for p in (REFERENCE/'report.json', REFERENCE/'independent_verification.json', PROTOCOL):
        protected[str(p)] = sha256(p)
    contract = dict(source_sha256=protected, data_provenance=d.provenance, compact_provenance=c.provenance,
        extensions=prov, seeds=list(SEEDS), arms=list(ARMS), epochs=100, lr=1e-4, batch_size=256,
        selection='validation readmission IBS7-365', frozen_baselines={str(s): sha256(Path(config['parent'])/f'seed{s}/survival_base/best.pt') for s in SEEDS},
        input_config=config, preprocessing_sha256=sha256(ROOT/'preprocessing.json'),
        fold_admissions={f: len(v) for f, v in ix.items()}, patients=len(np.unique(d.arrays['subject_ids'])),
        text='unchanged full cached B-four-field BERT fragments, no PCA',
        endpoint='strict491.21/J44.1; competing death; WorldA365', device=torch.cuda.get_device_name(0),
        packages={'tabm': '0.0.3', 'rtdl-revisiting-models': '0.0.2', 'rtdl-num-embeddings': '0.0.12'},
        deterministic_training_and_inference=True,
        original_control_checkpoints={str(s): sha256(REFERENCE/f'seed{s}/control/best.pt') for s in SEEDS},
        limits=['Reused test, not blind; no multiplicity adjustment.', 'Architecture capacity differs.',
                'TabM member probability mixture; not a single Weibull.', 'No causal treatment interpretation or complete outside followup.'])
    write_json(ROOT/'contract.json', contract)
    cache = {f: batches(d, c, ext, ix[f]) for f in ('train', 'validation')}
    fits, shared, initial, prior_checks = {}, {}, {}, {}
    for seed in SEEDS:
        base = restore(Path(config['parent']), seed, 'survival_base', h, .1)
        control = model_for(base, seed, 'control', bins)
        fits[str(seed)], shared[str(seed)], initial[str(seed)] = {}, {}, {}
        for arm in ARMS:
            m = model_for(base, seed, arm, bins)
            shared[str(seed)][arm] = shared_initial_errors(m, control)
            a, k, _ = parameters(m, cache['validation'][:2])
            aa, kk, _ = parameters(control, cache['validation'][:2])
            if a.ndim == 3:
                aa, kk = aa[:, None], kk[:, None]
            error = float(max(np.abs(a-aa).max(), np.abs(k-kk).max()))
            if error > 1e-7:
                raise ValueError('epoch0 differs from frozen common risk')
            initial[str(seed)][arm] = error
            fits[str(seed)][arm] = fit(m, cache['train'], cache['validation'], d.times[ix['validation']],
                d.events[ix['validation']], ROOT/f'seed{seed}'/arm, seed=seed)
            if arm == 'control':
                checkpoint = torch.load(ROOT/f'seed{seed}'/arm/'best.pt', map_location='cuda', weights_only=True)
                m.load_state_dict(checkpoint['state_dict'])
                aa, kk, _ = parameters(m, cache['validation'])
                current_curve = probability_curve(aa, kk)
                prior_curve, replay_error = original_replay(base, seed, cache['validation'], 'validation')
                before = metrics(d.times[ix['validation']], d.events[ix['validation']], prior_curve[:, :, 1])
                after = metrics(d.times[ix['validation']], d.events[ix['validation']], current_curve[:, :, 1])
                ibs_delta = after['integrated_brier_7_365']-before['integrated_brier_7_365']
                auc_delta = after['points']['90']['auc']-before['points']['90']['auc']
                prior_checks[str(seed)] = dict(old_checkpoint_exact_replay_error=replay_error,
                    retrained_control_ibs_difference=ibs_delta, retrained_control_auc90_difference=auc_delta)
                if abs(ibs_delta) > 1e-6 or abs(auc_delta) > 1e-4:
                    raise ValueError(f'retrained control metrics materially differ: {prior_checks[str(seed)]}')
            del m; torch.cuda.empty_cache()
        del control
    scores = {arm: float(np.mean([fits[str(s)][arm]['validation_ibs'] for s in SEEDS])) for arm in ARMS}
    choice = min(scores, key=scores.get)
    write_json(ROOT/'selected_before_test.json', dict(fits=fits, validation_scores=scores, selected_arm=choice,
        all_training_complete=True))
    del cache['train']; torch.cuda.empty_cache(); cache['test'] = batches(d, c, ext, ix['test'])
    models, curves, old_errors = {}, {}, {}
    for seed in SEEDS:
        base = restore(Path(config['parent']), seed, 'survival_base', h, .1)
        models[str(seed)], curves[seed] = {}, {}
        for arm in ARMS:
            path = ROOT/f'seed{seed}'/arm; m = model_for(base, seed, arm, bins)
            m.load_state_dict(torch.load(path/'best.pt', map_location='cuda', weights_only=True)['state_dict'])
            models[str(seed)][arm] = dict(**fits[str(seed)][arm], folds={}); curves[seed][arm] = {}
            for fold in ('validation', 'test'):
                a, k, nll = parameters(m, cache[fold]); curve = probability_curve(a, k)
                models[str(seed)][arm]['folds'][fold] = summary(curve, d, ix[fold], nll)
                save(path/(fold+'_predictions.npz'), d, ix[fold], a, k, curve); curves[seed][arm][fold] = curve
                if arm == 'control':
                    with np.load(REFERENCE/f'seed{seed}/control'/(fold+'_predictions.npz')) as z:
                        for key, expected in (('hadm_ids', d.arrays['hadm_ids'][ix[fold]]), ('times', d.times[ix[fold]]), ('events', d.events[ix[fold]])):
                            if not np.array_equal(z[key], expected):
                                raise ValueError('old control member/outcome differs')
                        error = float(np.abs(z['curve']-curve).max())
                previous_curve, replay_error = original_replay(base, seed, cache[fold], fold) if arm == 'control' else (None, None)
                if arm == 'control':
                    before = metrics(d.times[ix[fold]], d.events[ix[fold]], previous_curve[:, :, 1])
                    after = models[str(seed)][arm]['folds'][fold]['readmission']
                    old_errors[f'{seed}/{fold}'] = dict(old_checkpoint_exact_replay_error=replay_error,
                        retrained_control_max_probability_difference=error,
                        retrained_control_ibs_difference=after['integrated_brier_7_365']-before['integrated_brier_7_365'],
                        retrained_control_auc90_difference=after['points']['90']['auc']-before['points']['90']['auc'])
            del m
    comparisons = {}
    pairs = [(arm, 'control') for arm in ARMS[1:]] + [('interaction_ft', 'interaction_torch')]
    for seed in SEEDS:
        comparisons[str(seed)] = {f: {f'{a} minus {b}': paired_interval(d.arrays['subject_ids'][ix[f]],
            d.times[ix[f]], d.events[ix[f]], curves[seed][a][f][:, :, 1], curves[seed][b][f][:, :, 1], seed=seed)
            for a, b in pairs} for f in ('validation', 'test')}
    report = dict(status='training_completed_pending_independent_verification', contract=contract, models=models,
        comparisons=comparisons, validation_scores=scores, validation_selected_arm=choice,
        shared_initial_checks=shared, epoch0_errors=initial, original_control_reproduction=old_errors,
        validation_reference_checks_before_candidates=prior_checks,
        elapsed_seconds=time.perf_counter()-start, peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
    write_json(ROOT/'report.json', report)
    print(json.dumps(dict(status=report['status'], selected=choice, seconds=report['elapsed_seconds'])), flush=True)


def verify():
    setup(); d, c, h, ext, config, _, _ = inputs()
    report = json.loads((ROOT/'report.json').read_text()); contract = report['contract']
    for p, digest in contract['source_sha256'].items():
        if sha256(Path(p)) != digest:
            raise ValueError('protected source changed: ' + p)
    if sha256(ROOT/'preprocessing.json') != contract['preprocessing_sha256']:
        raise ValueError('preprocessing changed')
    bins = json.loads((ROOT/'preprocessing.json').read_text())
    if bins != fit_lab_bins(d.arrays['lab_axis_inputs'], d.arrays['patient_splits'] == 'train'):
        raise ValueError('training-only bins not reproducible')
    results, count = {}, 0
    for fold in ('validation', 'test'):
        ix = np.flatnonzero(d.arrays['patient_splits'] == fold); cache = batches(d, c, ext, ix)
        for seed in SEEDS:
            base = restore(Path(config['parent']), seed, 'survival_base', h, .1)
            if sha256(Path(config['parent'])/f'seed{seed}/survival_base/best.pt') != contract['frozen_baselines'][str(seed)]:
                raise ValueError('baseline source changed')
            for arm in ARMS:
                path = ROOT/f'seed{seed}'/arm; state = torch.load(path/'best.pt', map_location='cuda', weights_only=True)
                info = report['models'][str(seed)][arm]
                history = json.loads((path/'history.json').read_text())
                if len(history) != 101 or [r['epoch'] for r in history] != list(range(101)):
                    raise ValueError('incomplete epochs')
                if min(history, key=lambda r: r['validation_ibs'])['epoch'] != state['epoch'] or state['epoch'] != info['best_epoch']:
                    raise ValueError('wrong validation selection')
                if sha256(path/'best.pt') != info['checkpoint_sha256']:
                    raise ValueError('checkpoint changed')
                m = model_for(base, seed, arm, bins); m.load_state_dict(state['state_dict'])
                if arm == 'control':
                    if sha256(REFERENCE/f'seed{seed}/control/best.pt') != contract['original_control_checkpoints'][str(seed)]:
                        raise ValueError('old checkpoint source changed')
                    original_replay(base, seed, cache, fold)
                if not all(torch.equal(v, m.baseline.state_dict()[k]) for k, v in base.state_dict().items()):
                    raise ValueError('checkpoint baseline is not frozen original')
                a, k, nll = parameters(m, cache); curve = probability_curve(a, k)
                with np.load(path/(fold+'_predictions.npz')) as z:
                    for key, expected in (('hadm_ids', d.arrays['hadm_ids'][ix]), ('subject_ids', d.arrays['subject_ids'][ix]), ('times', d.times[ix]), ('events', d.events[ix])):
                        if not np.array_equal(z[key], expected):
                            raise ValueError('member/split/outcome mismatch')
                    error = float(np.abs(curve-z['curve']).max())
                    if not np.allclose(a, z['log_amplitude'], atol=1e-7, rtol=0) or not np.allclose(k, z['shape'], atol=1e-7, rtol=0):
                        raise ValueError('replayed parameter difference')
                    daily, conditional = daily_probabilities(curve)
                    np.testing.assert_allclose(daily, z['daily_event_probability'], atol=3e-7, rtol=0)
                    np.testing.assert_allclose(conditional, z['conditional_daily_event_probability'], atol=3e-7, rtol=0)
                if error > 3e-7 or not np.isfinite(curve).all() or np.any((curve < 0) | (curve > 1)):
                    raise ValueError('replay or probability bounds failed')
                np.testing.assert_allclose(curve.sum(2), 1, atol=1e-9)
                if np.any(np.diff(curve[:, :, 1:], axis=1) < -1e-10) or np.any(np.diff(curve[:, :, 0], axis=1) > 1e-10):
                    raise ValueError('nonmonotonic cumulative probability')
                recomputed = summary(curve, d, ix, nll)
                if recomputed != info['folds'][fold]:
                    raise ValueError('metric replay differs')
                results[f'{seed}/{arm}/{fold}'] = error; count += 1; del m
        del cache
    report['status'] = 'completed_and_independently_verified'
    write_json(ROOT/'report.json', report)
    write_json(ROOT/'independent_verification.json', dict(status='fresh_process_28_fold_replays_verified',
        folds=count, models=14, replay_errors=results, maximum_replay_error=max(results.values()),
        report_sha256=sha256(ROOT/'report.json'), source_hashes_verified=len(contract['source_sha256']),
        training_bins_recomputed=True, frozen_baselines_verified=True, daily_probability_verified=True,
        all_100_epochs_verified=True, original_control_reproduction=report['original_control_reproduction']))
    print(json.dumps(dict(status=report['status'], folds=count, maximum_error=max(results.values()))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=('train', 'verify'), required=True)
    mode = p.parse_args().mode
    run() if mode == 'train' else verify()
