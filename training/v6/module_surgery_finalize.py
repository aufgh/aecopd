"""只汇总完整检查点；保留训练源码合同，修复NPZ关闭后的汇总读取。"""
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.module_surgery import ARMS, probability_curve
from training.v6.module_surgery_experiment import (ROOT, REFERENCE, SEEDS, setup, inputs,
    batches, model_for, parameters, summary, save, original_replay)
from training.v6.horizon_objective import restore
from training.v6.evaluation import metrics, paired_interval
from training.v6.train import write_json


def finalize():
    setup(); start = time.perf_counter()
    if (ROOT/'report.json').exists():
        raise FileExistsError('report already present; do not replace it')
    contract = json.loads((ROOT/'contract.json').read_text())
    driver = 'training/v6/module_surgery_experiment.py'
    snapshot = ROOT/'code_snapshot/module_surgery_experiment.py'
    for p, digest in contract['source_sha256'].items():
        target = snapshot if p == driver else Path(p)
        if sha256(target) != digest:
            raise ValueError('actual training source differs: ' + p)
    # 训练函数和模型不改；原实际执行源码/合同作为不可变证据保留。
    original_contract = ROOT/'contract_training_original.json'
    if original_contract.exists():
        raise FileExistsError('finalization previously started; inspect rather than overwrite')
    write_json(original_contract, contract)
    sources = dict(contract['source_sha256']); sources[str(snapshot)] = sources[driver]
    sources[driver] = sha256(Path(driver))
    sources['training/v6/module_surgery_finalize.py'] = sha256(Path(__file__))
    contract = dict(contract, source_sha256=sources,
        actual_training_driver_snapshot=str(snapshot), original_training_contract_sha256=sha256(original_contract),
        finalization_correction='Move NPZ probability read inside open context; no training/model/selection change.')
    write_json(ROOT/'contract.json', contract)
    selected = json.loads((ROOT/'selected_before_test.json').read_text())
    if not selected['all_training_complete']:
        raise ValueError('unfinished training')
    fits = selected['fits']
    for seed in SEEDS:
        for arm in ARMS:
            path = ROOT/f'seed{seed}'/arm; h = json.loads((path/'history.json').read_text())
            cp = torch.load(path/'best.pt', map_location='cpu', weights_only=True)
            if len(h) != 101 or min(h, key=lambda row: row['validation_ibs'])['epoch'] != cp['epoch']:
                raise ValueError('incomplete/wrong selection')
            if sha256(path/'best.pt') != fits[str(seed)][arm]['checkpoint_sha256']:
                raise ValueError('checkpoint changed')
    scores = {arm: float(np.mean([fits[str(s)][arm]['validation_ibs'] for s in SEEDS])) for arm in ARMS}
    if scores != selected['validation_scores'] or min(scores, key=scores.get) != selected['selected_arm']:
        raise ValueError('validation choice changed')
    d, c, h, ext, config, _, _ = inputs()
    if d.provenance != contract['data_provenance'] or c.provenance != contract['compact_provenance']:
        raise ValueError('input changed')
    prep = json.loads((ROOT/'preprocessing.json').read_text())
    ix = {f: np.flatnonzero(d.arrays['patient_splits'] == f) for f in ('validation', 'test')}
    cache = {f: batches(d, c, ext, ix[f]) for f in ix}
    models, curves, previous_checks = {}, {}, {}
    for seed in SEEDS:
        base = restore(Path(config['parent']), seed, 'survival_base', h, .1)
        models[str(seed)], curves[seed] = {}, {}
        for arm in ARMS:
            path = ROOT/f'seed{seed}'/arm; m = model_for(base, seed, arm, prep)
            m.load_state_dict(torch.load(path/'best.pt', map_location='cuda', weights_only=True)['state_dict'])
            if not all(torch.equal(v, m.baseline.state_dict()[k]) for k, v in base.state_dict().items()):
                raise ValueError('frozen baseline differs')
            models[str(seed)][arm] = dict(**fits[str(seed)][arm], folds={},
                interaction_gate=float(torch.tanh(m.encoder.interaction_gate).detach()) if arm.startswith('interaction_') else None,
                correction_head_last_weight_norm=float(m.delta_head[-1].weight.detach().norm()))
            curves[seed][arm] = {}
            for fold in ix:
                a, k, nll = parameters(m, cache[fold]); curve = probability_curve(a, k)
                models[str(seed)][arm]['folds'][fold] = summary(curve, d, ix[fold], nll)
                save(path/(fold+'_predictions.npz'), d, ix[fold], a, k, curve); curves[seed][arm][fold] = curve
                if arm == 'control':
                    old_curve, replay = original_replay(base, seed, cache[fold], fold)
                    before = metrics(d.times[ix[fold]], d.events[ix[fold]], old_curve[:, :, 1])
                    after = models[str(seed)][arm]['folds'][fold]['readmission']
                    previous_checks[f'{seed}/{fold}'] = dict(old_checkpoint_exact_replay_error=replay,
                        retrained_control_max_probability_difference=float(np.abs(curve-old_curve).max()),
                        retrained_control_ibs_difference=after['integrated_brier_7_365']-before['integrated_brier_7_365'],
                        retrained_control_auc90_difference=after['points']['90']['auc']-before['points']['90']['auc'])
            del m
    pairs = [(arm, 'control') for arm in ARMS[1:]] + [('interaction_ft', 'interaction_torch')]
    comparisons = {str(seed): {f: {f'{a} minus {b}': paired_interval(d.arrays['subject_ids'][ix[f]],
        d.times[ix[f]], d.events[ix[f]], curves[seed][a][f][:, :, 1], curves[seed][b][f][:, :, 1], seed=seed)
        for a, b in pairs} for f in ix} for seed in SEEDS}
    # 共同初值证据在训练时由硬性检查通过；新进程重新逐项核验并保存数目。
    from training.v6.module_surgery import shared_initial_errors
    shared, initial = {}, {}
    for seed in SEEDS:
        base = restore(Path(config['parent']), seed, 'survival_base', h, .1)
        control = model_for(base, seed, 'control', prep); aa, kk, _ = parameters(control, cache['validation'][:2])
        shared[str(seed)], initial[str(seed)] = {}, {}
        for arm in ARMS:
            m = model_for(base, seed, arm, prep); shared[str(seed)][arm] = shared_initial_errors(m, control)
            a, k, _ = parameters(m, cache['validation'][:2])
            expect_a, expect_k = (aa[:, None], kk[:, None]) if a.ndim == 3 else (aa, kk)
            error = float(max(np.abs(a-expect_a).max(), np.abs(k-expect_k).max()))
            if error != 0:
                raise ValueError('common initial risk differs')
            initial[str(seed)][arm] = error; del m
        del control
    report = dict(status='training_completed_pending_independent_verification', contract=contract, models=models,
        comparisons=comparisons, validation_scores=scores, validation_selected_arm=selected['selected_arm'],
        shared_initial_checks=shared, epoch0_errors=initial, original_control_reproduction=previous_checks,
        elapsed_seconds=sum(fits[str(s)][a]['elapsed_seconds'] for s in SEEDS for a in ARMS)+time.perf_counter()-start,
        finalization_seconds=time.perf_counter()-start, finalization_correction=contract['finalization_correction'],
        peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
        peak_gpu_memory_scope='finalization inference only; training peak was not persisted')
    write_json(ROOT/'report.json', report)
    print(json.dumps(dict(status=report['status'], selected=selected['selected_arm'], seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    finalize()
