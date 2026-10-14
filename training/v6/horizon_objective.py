"""固定输入/严格终点，比较90天BCE与双事件连续时间NLL；不改旧训练器。"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from patient_data.export_mimiciv import sha256
from training.v6.architecture_experiment import batches
from training.v6.compact_ablation import inputs
from training.v6.compact_model import CompactAnchoredModel
from training.v6.exploration_model import ExpandedBaseline
from training.v6.model import ConstantCorrection
from training.v6.risk import competing_nll, predict_cif
from training.v6.evaluation import GRID
from training.v6.train import write_json

ARMS = ('survival_base', 'survival_constant', 'survival_full',
        'binary_base', 'binary_constant', 'binary_full')
PAIRS = (('binary_full', 'survival_full'), ('binary_base', 'survival_base'),
         ('survival_full', 'survival_constant'), ('binary_full', 'binary_constant'),
         ('survival_constant', 'survival_base'), ('binary_constant', 'binary_base'))


def labels90(times, events):
    """先死亡属于未发生目标事件；只接受现有无提前删失World A合同。"""
    t, e = np.asarray(times), np.asarray(events)
    if (t.ndim != 1 or not len(t) or e.shape != t.shape or not np.isfinite(t).all()
            or np.any((t < 0) | (t > 365)) or not np.isin(e, (0, 1, 2)).all()
            or np.any((e == 0) & (t != 365))):
        raise ValueError('invalid World A labels; early censoring needs another contract')
    return ((e == 1) & (t <= 90)).astype(np.int8)


def ranked_curves(y, p):
    """按不同阈值整组处理同分分数，AP采用阶梯积分，不用梯形PR面积。"""
    y, p = np.asarray(y), np.asarray(p)
    if (y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isin(y, (0, 1)).all()
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError('invalid binary outcomes/probabilities')
    order = np.argsort(-p, kind='stable')
    score, target = p[order], y[order]
    end = np.r_[np.flatnonzero(score[1:] != score[:-1]), len(y)-1]
    tp = np.cumsum(target)[end].astype(float)
    fp = end + 1 - tp
    positives, negatives = float(y.sum()), float(len(y)-y.sum())
    recall = tp/positives if positives else np.zeros_like(tp)
    precision = tp/(end+1)
    fpr = fp/negatives if negatives else np.zeros_like(fp)
    ap = float(np.sum(np.diff(np.r_[0., recall])*precision)) if positives else None
    return dict(fpr=np.r_[0., fpr], tpr=np.r_[0., recall],
                recall=np.r_[0., recall], precision=np.r_[1., precision], ap=ap,
                auc=float(np.trapezoid(np.r_[0., recall], np.r_[0., fpr])) if positives and negatives else None)


def calibration_fit(y, p):
    """评价性无惩罚Logistic截距/斜率，不将测试拟合结果用于预测再校准。"""
    if len(np.unique(y)) < 2 or np.ptp(p) < 1e-12:
        return dict(intercept=None, slope=None, reason='one class or constant scores')
    q = np.clip(p, 1e-8, 1-1e-8)
    x = np.column_stack((np.ones(len(y)), np.log(q/(1-q))))
    theta = np.array([0., 1.])
    def loss(v):
        z = x@v
        return float(np.mean(np.logaddexp(0, z)-y*z))
    for _ in range(80):
        z = x@theta
        prob = 1/(1+np.exp(-np.clip(z, -40, 40)))
        grad = x.T@(prob-y)/len(y)
        hess = (x.T*(prob*(1-prob)))@x/len(y)
        if np.linalg.cond(hess) > 1e12:
            return dict(intercept=None, slope=None, reason='singular calibration fit')
        step = np.linalg.solve(hess, grad)
        scale = 1.
        while scale > 1e-7 and loss(theta-scale*step) > loss(theta):
            scale *= .5
        theta -= scale*step
        if np.max(np.abs(scale*step)) < 1e-7:
            return dict(intercept=float(theta[0]), slope=float(theta[1]), converged=True)
    return dict(intercept=None, slope=None, reason='calibration fit did not converge')


def binary_metrics(y, p):
    curves = ranked_curves(y, p)
    y, p = np.asarray(y), np.asarray(p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    group = np.searchsorted(edges[1:-1], p, side='right')
    bins = [dict(n=int((group == g).sum()), predicted=float(p[group == g].mean()),
                 observed=float(y[group == g].mean()))
            for g in range(max(1, len(edges)-1)) if (group == g).any()]
    q = np.clip(p, 1e-8, 1-1e-8)
    return dict(admissions=len(y), positives=int(y.sum()), observed_fraction=float(y.mean()),
                auc90=curves['auc'],
                average_precision=curves['ap'], brier90=float(np.mean((p-y)**2)),
                logloss90=float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q))),
                mean_predicted=float(p.mean()), calibration_bins=bins,
                calibration_logistic=calibration_fit(y, p),
                ece10=float(sum(r['n']*abs(r['predicted']-r['observed']) for r in bins)/len(y)))


def paired90(subjects, y, candidate, comparator, *, replicates=200, seed=42):
    if len(subjects) != len(y) or replicates < 20:
        raise ValueError('invalid cluster/bootstrap contract')
    left, right = binary_metrics(y, candidate), binary_metrics(y, comparator)
    names = ('auc90', 'average_precision', 'brier90', 'logloss90')
    people, inverse = np.unique(subjects, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(people))]
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in names}
    for _ in range(replicates):
        ix = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        a, b, target = candidate[ix], comparator[ix], y[ix]
        samples['brier90'].append(float(np.mean((a-target)**2-(b-target)**2)))
        ca, cb = np.clip(a, 1e-8, 1-1e-8), np.clip(b, 1e-8, 1-1e-8)
        samples['logloss90'].append(float(np.mean(-target*np.log(ca/cb)-(1-target)*np.log((1-ca)/(1-cb)))))
        if 0 < target.sum() < len(target):
            curve_a, curve_b = ranked_curves(target, a), ranked_curves(target, b)
            samples['auc90'].append(curve_a['auc']-curve_b['auc'])
            samples['average_precision'].append(curve_a['ap']-curve_b['ap'])
    return dict(definition='candidate minus comparator', patients=len(people), replicates=replicates,
                metrics={k:dict(difference=left[k]-right[k] if left[k] is not None and right[k] is not None else None,
                                ci95=np.quantile(samples[k], [.025, .975]).tolist() if samples[k] else [None, None],
                                valid_replicates=len(samples[k])) for k in names},
                limits='conditional on fitted models; reused test; no multiplicity correction or retraining variance')


@dataclass(frozen=True)
class BinaryOutput:
    logits: torch.Tensor
    delta: torch.Tensor | None = None


class BinaryBaseline(nn.Module):
    def __init__(self, prevalence=.1):
        super().__init__()
        if not 0 < prevalence < 1:
            raise ValueError('training prevalence must be in (0,1)')
        self.logit = nn.Linear(34, 1)
        nn.init.zeros_(self.logit.weight)
        nn.init.constant_(self.logit.bias, math.log(prevalence/(1-prevalence)))

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != 34 or not torch.isfinite(x).all():
            raise ValueError('binary baseline requires 34 finite inputs')
        return BinaryOutput(self.logit(x).squeeze(-1))


class BinaryConstant(nn.Module):
    def __init__(self, baseline):
        super().__init__()
        self.baseline = baseline.requires_grad_(False).eval()
        self.correction = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        with torch.no_grad():
            base = self.baseline(x)
        delta = self.correction.expand(len(x))
        return BinaryOutput(base.logits+delta, delta)


class BinaryResidual(CompactAnchoredModel):
    """同临床编码器/初值；保留4输出槽位但分类只使用第一个Logit修正。"""
    def forward(self, history, batch):
        with torch.no_grad():
            base = self.baseline(history)
        hidden, _ = self.encoder(batch)
        delta = self.delta_head(hidden)[:, 0]
        return BinaryOutput(base.logits+delta, delta)


def make_model(arm, base, width, seed):
    torch.manual_seed(seed)
    b = copy.deepcopy(base)
    if arm.endswith('_base'):
        return b.cuda()
    if arm.endswith('_constant'):
        return (BinaryConstant(b) if arm.startswith('binary') else ConstantCorrection(b)).cuda()
    return (BinaryResidual(b, width) if arm.startswith('binary') else CompactAnchoredModel(b, width)).cuda()


def invoke(model, h, b):
    return model(h, b) if hasattr(model, 'encoder') else model(h)


def loss_for(output, t, e):
    if isinstance(output, BinaryOutput):
        target = ((e == 1) & (t <= 90)).to(output.logits.dtype)
        return F.binary_cross_entropy_with_logits(output.logits, target)
    return competing_nll(output, t, e)


@torch.no_grad()
def predict90(model, prepared):
    model.eval()
    probabilities, parameters, shapes, deltas = [], [], [], []
    total_loss, total = 0., 0
    for pos, h, b, t, e in prepared:
        out = invoke(model, h, b)
        total_loss += float(loss_for(out, t, e))*len(pos)
        total += len(pos)
        if isinstance(out, BinaryOutput):
            probabilities.append(torch.sigmoid(out.logits).cpu().numpy().astype(np.float64))
            parameters.append(out.logits.cpu().numpy())
        else:
            parameters.append(out.log_amplitude.cpu().numpy())
            shapes.append(out.shape.cpu().numpy())
        if out.delta is not None:
            deltas.append(out.delta.cpu().numpy())
    parameters = np.concatenate(parameters)
    result = dict(parameters=parameters, loss=total_loss/total)
    if shapes:
        result['shape'] = np.concatenate(shapes)
        result['risk90'] = predict_cif(parameters, result['shape'], np.array([0., 90.]))[:, 1, 1]
    else:
        result['risk90'] = np.concatenate(probabilities)
    if deltas:
        result['delta'] = np.concatenate(deltas)
    return result


def fit90(model, prepared_train, prepared_val, yval, directory, *, epochs, lr, seed, patience, warmup=0):
    directory.mkdir()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    frozen = {k:v.detach().cpu().clone() for k,v in model.baseline.state_dict().items()} if hasattr(model, 'baseline') else None
    rows, best_state, best_score, best_epoch, stale = [], None, math.inf, -1, 0
    start = time.perf_counter()
    for epoch in range(epochs+1):
        training_loss = None
        if epoch:
            rate = lr*epoch/warmup if warmup and epoch <= warmup else (
                lr*(.1+.9*.5*(1+math.cos(math.pi*(epoch-warmup)/max(1, epochs-warmup)))) if warmup else lr)
            for group in optimizer.param_groups:
                group['lr'] = rate
            model.train()
            summed, n = 0., 0
            for j in rng.permutation(len(prepared_train)):
                pos, h, b, t, e = prepared_train[j]
                optimizer.zero_grad(set_to_none=True)
                out = invoke(model, h, b)
                core = loss_for(out, t, e)
                penalty = out.delta.square().mean()*1e-4 if out.delta is not None else core.new_zeros(())
                loss = core+penalty
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                loss.backward()
                if frozen is not None and any(p.grad is not None for p in model.baseline.parameters()):
                    raise ValueError('frozen baseline gradient leak')
                torch.nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step()
                summed += float(core.detach())*len(pos)
                n += len(pos)
            training_loss = summed/n
        val = predict90(model, prepared_val)
        detail = binary_metrics(yval, val['risk90'])
        score = detail['brier90']
        rows.append(dict(epoch=epoch, train_objective=training_loss, validation_objective=val['loss'],
                         validation_brier90=score, validation_auc90=detail['auc90'],
                         validation_ap=detail['average_precision'], seconds=round(time.perf_counter()-start, 2)))
        write_json(directory/'history.json', rows)
        if score < best_score-1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        elif epoch:
            stale += 1
        if epoch % 10 == 0 or stale >= patience:
            print(json.dumps(dict(arm=directory.name, epoch=epoch, brier90=score, best_epoch=best_epoch)), flush=True)
        if epoch and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError('no selected checkpoint')
    model.load_state_dict(best_state)
    if frozen is not None and not all(torch.equal(v.detach().cpu(), frozen[k]) for k,v in model.baseline.state_dict().items()):
        raise ValueError('frozen baseline changed')
    checkpoint = directory/'best.pt'
    torch.save(dict(state_dict=best_state, seed=seed, best_epoch=best_epoch), checkpoint)
    return dict(best_epoch=best_epoch, epochs_trained=len(rows)-1, best_validation_brier90=best_score,
                checkpoint_sha256=sha256(checkpoint), parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in params))


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; no CPU fallback')
    torch.set_num_threads(2)
    started = time.perf_counter()
    data, compact, hist, checkpoint = inputs(args.reference, args.compact_dir)
    ix = {f:np.flatnonzero(data.arrays['patient_splits'] == f) for f in ('train', 'validation', 'test')}
    y = labels90(data.times, data.events)
    args.output.mkdir(parents=True)
    config = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    write_json(args.output/'config.json', config)
    contract = dict(data_provenance=data.provenance, compact_provenance=compact.provenance,
                    old_H_sha256=sha256(checkpoint), seeds=args.seeds, arms=list(ARMS),
                    selection='validation Brier90 only for every newly fitted arm',
                    endpoint='strict 491.21/J44.1 before competing death within90d; World A',
                    folds={f:dict(admissions=len(idx), patients=len(np.unique(data.arrays['subject_ids'][idx])),
                                  positives=int(y[idx].sum())) for f,idx in ix.items()},
                    backbone='same CompactClinicalEncoder, width128, existing B4 Bio_ClinicalBERT',
                    device=torch.cuda.get_device_name(0),
                    baseline='H6+D/P14 values+14 masks; residual D/P values and masks removed',
                    objective='Weibull joint readmission/death NLL vs one-output unweighted BCE90',
                    protocol_sha256=sha256(Path('docs/V6_90DAY_OBJECTIVE_PROTOCOL_2026-10-11.md')),
                    code_sha256={str(p):sha256(p) for p in [Path('training/v6')/f for f in
                        ('horizon_objective.py', 'architecture_experiment.py', 'compact_ablation.py',
                         'exploration_model.py', 'compact_model.py', 'compact_context_data.py', 'model.py',
                         'risk.py', 'evaluation.py', 'data.py')]},
                    limits=['reused test, not blind; no multiplicity adjustment',
                            'objective and output family/capacity/initial baseline differ; not loss-only causal attribution',
                            'no outside-hospital follow-up guarantee; text may reconstruct history',
                            'seeds share previously trained H source; not new cohort splits',
                            'not full reproduction of all-cause90d COPD literature'])
    write_json(args.output/'contract.json', contract)
    cached = {f:batches(data, compact, ix[f], args.batch_size, expanded=True, text='bert') for f in ('train', 'validation')}
    fitted = {}
    for seed in args.seeds:
        directory = args.output/f'seed{seed}'
        directory.mkdir()
        bases = {}
        for kind in ('survival', 'binary'):
            torch.manual_seed(seed)
            base = ExpandedBaseline(hist).cuda() if kind == 'survival' else BinaryBaseline(float(y[ix['train']].mean())).cuda()
            write_json(args.output/'status.json', dict(status='training', seed=seed, arm=kind+'_base'))
            fitted[seed, kind+'_base'] = fit90(base, cached['train'], cached['validation'], y[ix['validation']],
                directory/(kind+'_base'), epochs=args.baseline_epochs, lr=.005, seed=seed, patience=20)
            bases[kind] = base
        common = None
        for arm in (a for a in ARMS if not a.endswith('_base')):
            kind = arm.split('_')[0]
            model = make_model(arm, bases[kind], 768, seed)
            if hasattr(model, 'encoder'):
                state = {k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()}
                if common is not None and not all(torch.equal(v, common[k]) for k,v in state.items()):
                    raise ValueError('survival/binary encoder initialization differs')
                common = state
                initial = predict90(model, cached['validation'])['risk90']
                expected = predict90(bases[kind], cached['validation'])['risk90']
                if not np.allclose(initial, expected, atol=1e-7, rtol=1e-6):
                    raise ValueError('clinical residual did not start at zero')
            write_json(args.output/'status.json', dict(status='training', seed=seed, arm=arm))
            full = arm.endswith('_full')
            fitted[seed, arm] = fit90(model, cached['train'], cached['validation'], y[ix['validation']], directory/arm,
                epochs=args.epochs, lr=3e-4 if full else .005, seed=seed, patience=10, warmup=5 if full else 0)
            del model
        bases.clear()
        torch.cuda.empty_cache()
    write_json(args.output/'status.json', dict(status='all_seeds_all_checkpoints_selected_before_test'))
    results, predictions, comparisons = {}, {}, {}
    for seed in args.seeds:
        results[str(seed)], comparisons[str(seed)] = {}, {}
        for arm in ARMS:
            model = restore(args.output, seed, arm, hist, y[ix['train']].mean())
            results[str(seed)][arm] = {**fitted[seed, arm], 'folds':{}}
            for fold in ('validation', 'test'):
                idx = ix[fold]
                prepared = cached[fold] if fold in cached else batches(data, compact, idx, args.batch_size, expanded=True, text='bert')
                pred = predict90(model, prepared)
                detail = binary_metrics(y[idx], pred['risk90'])
                prior = np.expm1(data.history_raw[idx, 2]).round().astype(int)
                detail['history_strata'] = {name:binary_metrics(y[idx][keep], pred['risk90'][keep]) for name,keep in
                    (('prior_0', prior == 0), ('prior_1_2', (prior >= 1)&(prior <= 2)), ('prior_3_plus', prior >= 3)) if keep.any()}
                if 'shape' in pred:
                    curve = predict_cif(pred['parameters'], pred['shape'], GRID)
                    fine = predict_cif(pred['parameters'][:64], pred['shape'][:64], GRID, step_days=.125)
                    error = float(np.max(np.abs(fine-curve[:64])))
                    if error > 2e-4 or not np.allclose(curve[:, 90, 1], pred['risk90'], atol=1e-8):
                        raise ValueError('survival CIF integration convergence failed')
                    detail['CIF_step_max_error_first64'] = error
                    detail['max_probability_sum_error'] = float(np.abs(curve.sum(2)-1).max())
                    pred['curve'] = curve
                predictions[seed, arm, fold] = pred['risk90']
                np.savez_compressed(args.output/f'seed{seed}'/arm/(fold+'_predictions.npz'),
                    hadm_ids=data.arrays['hadm_ids'][idx], subject_ids=data.arrays['subject_ids'][idx],
                    times=data.times[idx], events=data.events[idx], y90=y[idx], history_counts=prior,
                    **{k:v for k,v in pred.items() if k != 'loss'})
                results[str(seed)][arm]['folds'][fold] = detail
                del prepared
            del model
        for fold in ('validation', 'test'):
            idx = ix[fold]
            comparisons[str(seed)][fold] = {a+'_minus_'+b:paired90(data.arrays['subject_ids'][idx], y[idx],
                predictions[seed, a, fold], predictions[seed, b, fold], replicates=args.bootstrap, seed=seed) for a,b in PAIRS}
    report = dict(status='completed_two_objectives_fixed_strict_endpoint', contract=contract, models=results,
                  comparisons=comparisons, elapsed_seconds=time.perf_counter()-started)
    write_json(args.output/'report.json', report)
    render(args.output, report)
    write_json(args.output/'status.json', dict(status=report['status'], seconds=report['elapsed_seconds']))
    print(json.dumps(dict(status=report['status'], seconds=report['elapsed_seconds'])), flush=True)


def restore(root, seed, arm, hist, prevalence):
    kind = arm.split('_')[0]
    base = ExpandedBaseline(hist).cuda() if kind == 'survival' else BinaryBaseline(float(prevalence)).cuda()
    base_path = root/f'seed{seed}'/(kind+'_base')/'best.pt'
    base.load_state_dict(torch.load(base_path, map_location='cuda', weights_only=True)['state_dict'])
    model = make_model(arm, base, 768, seed)
    model.load_state_dict(torch.load(root/f'seed{seed}'/arm/'best.pt', map_location='cuda', weights_only=True)['state_dict'])
    if hasattr(model, 'baseline') and not all(torch.equal(v, base.state_dict()[k]) for k,v in model.baseline.state_dict().items()):
        raise ValueError('residual checkpoint frozen base differs')
    return model


def verify(args):
    report = json.loads((args.output/'report.json').read_text())
    c = report['contract']
    for p,digest in c['code_sha256'].items():
        if sha256(Path(p)) != digest:
            raise ValueError('code changed: '+p)
    if sha256(Path('docs/V6_90DAY_OBJECTIVE_PROTOCOL_2026-10-11.md')) != c['protocol_sha256']:
        raise ValueError('protocol changed')
    data, compact, hist, checkpoint = inputs(args.reference, args.compact_dir)
    if data.provenance != c['data_provenance'] or compact.provenance != c['compact_provenance'] or sha256(checkpoint) != c['old_H_sha256']:
        raise ValueError('source changed')
    y = labels90(data.times, data.events)
    errors = {}
    for seed in c['seeds']:
        for arm in ARMS:
            if sha256(args.output/f'seed{seed}'/arm/'best.pt') != report['models'][str(seed)][arm]['checkpoint_sha256']:
                raise ValueError('checkpoint changed')
            model = restore(args.output, seed, arm, hist, y[data.arrays['patient_splits'] == 'train'].mean())
            for fold in ('validation', 'test'):
                idx = np.flatnonzero(data.arrays['patient_splits'] == fold)
                pred = predict90(model, batches(data, compact, idx, args.batch_size, expanded=True, text='bert'))
                with np.load(args.output/f'seed{seed}'/arm/(fold+'_predictions.npz')) as z:
                    for key,value in (('hadm_ids', data.arrays['hadm_ids'][idx]), ('subject_ids', data.arrays['subject_ids'][idx]),
                                      ('times', data.times[idx]), ('events', data.events[idx]), ('y90', y[idx])):
                        if not np.array_equal(z[key], value):
                            raise ValueError('saved labels or identity differ')
                    for key,value in pred.items():
                        if key != 'loss' and not np.allclose(z[key], value, atol=2e-6, rtol=1e-5):
                            raise ValueError('prediction reload mismatch: '+arm+'/'+key)
                    if 'shape' in pred and not np.allclose(z['curve'], predict_cif(pred['parameters'], pred['shape'], GRID), atol=2e-6, rtol=1e-5):
                        raise ValueError('full curve reload mismatch')
                    errors[f'{seed}/{arm}/{fold}'] = float(np.abs(z['risk90']-pred['risk90']).max())
                actual = binary_metrics(y[idx], pred['risk90'])
                recorded = report['models'][str(seed)][arm]['folds'][fold]
                for key in ('auc90', 'average_precision', 'brier90', 'logloss90'):
                    if abs(actual[key]-recorded[key]) > 1e-7:
                        raise ValueError('recomputed metric mismatch')
            del model
    write_json(args.output/'independent_verification.json', dict(status='all12_models_all24_folds_gpu_reloaded',
        errors=errors, baseline_unchanged=True, report_sha256=sha256(args.output/'report.json')))
    print('all12_models_all24_folds_gpu_reloaded', flush=True)


def render(root, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = root/'figures'
    folder.mkdir()
    colors = dict(survival_base='#729fcf', survival_constant='#3465a4', survival_full='#08306b',
                  binary_base='#fdae6b', binary_constant='#e6550d', binary_full='#a63603')
    for seed in report['contract']['seeds']:
        for fold in ('validation', 'test'):
            fig, axes = plt.subplots(1, 3, figsize=(17, 5))
            for arm in ARMS:
                with np.load(root/f'seed{seed}'/arm/(fold+'_predictions.npz')) as z:
                    curves = ranked_curves(z['y90'], z['risk90'])
                m = report['models'][str(seed)][arm]['folds'][fold]
                label = f'{arm}: AUC={m["auc90"]:.3f}'
                axes[0].plot(curves['fpr'], curves['tpr'], color=colors[arm], label=label, alpha=.9)
                axes[1].step(curves['recall'], curves['precision'], where='pre', color=colors[arm], label=f'{arm}: AP={m["average_precision"]:.3f}')
                bins = m['calibration_bins']
                axes[2].plot([b['predicted'] for b in bins], [b['observed'] for b in bins], 'o-', color=colors[arm], label=arm)
            axes[0].plot([0, 1], [0, 1], 'k--', alpha=.4)
            axes[1].axhline(m['observed_fraction'], color='k', linestyle='--', alpha=.4)
            axes[2].plot([0, 1], [0, 1], 'k--', alpha=.4)
            for ax, title, xlabel, ylabel in zip(axes, ('90-day ROC', '90-day Precision-Recall (AP)', '90-day Calibration'),
                    ('False positive rate', 'Recall', 'Mean predicted risk'), ('True positive rate', 'Precision', 'Observed fraction')):
                ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
                ax.legend(fontsize=7); ax.grid(alpha=.2)
            axes[2].set_xlim(0, .6); axes[2].set_ylim(0, .6)
            fig.suptitle(f'{fold}, seed{seed}: fixed strict AECOPD; all arms selected by validation Brier90; reused test')
            fig.tight_layout(); fig.savefig(folder/f'{fold}_seed{seed}_roc_pr_calibration.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    selected = PAIRS[:4]
    for j, (left, right) in enumerate(selected):
        for i, seed in enumerate(report['contract']['seeds']):
            comparison = report['comparisons'][str(seed)]['test'][left+'_minus_'+right]['metrics']
            for ax,key in zip(axes, ('brier90', 'auc90')):
                d = comparison[key]; value = d['difference']; lo, hi = d['ci95']
                color = ('#3465a4', '#e6550d')[i % 2]
                ax.plot([lo, hi], [j+(i-.5)*.16]*2, color=color, alpha=.6)
                ax.scatter(value, j+(i-.5)*.16, color=color, label=f'seed{seed}' if j == 0 else None)
    for ax in axes:
        ax.axvline(0, color='black', linewidth=1); ax.grid(axis='x', alpha=.2)
        ax.set_yticks(range(len(selected)), [a+' minus '+b for a,b in selected]); ax.legend()
    axes[0].invert_yaxis(); axes[0].set_xlabel('Paired Brier90 difference (lower better)')
    axes[1].set_xlabel('Paired AUC90 difference (higher better)')
    fig.suptitle('Reused test: patient-clustered 95% intervals, 200 replicates; no multiplicity adjustment')
    fig.tight_layout(); fig.savefig(folder/'test_paired_two_seeds.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col, seed in enumerate(report['contract']['seeds']):
        for arm in ARMS:
            history = json.loads((root/f'seed{seed}'/arm/'history.json').read_text())
            axes[0, col].plot([r['epoch'] for r in history], [r['validation_brier90'] for r in history], color=colors[arm], label=arm)
            axes[1, col].plot([r['epoch'] for r in history], [r['validation_auc90'] for r in history], color=colors[arm])
        axes[0, col].set(title=f'Seed{seed}: validation Brier90', ylabel='Brier90 (lower better)')
        axes[1, col].set(title=f'Seed{seed}: validation AUC90', xlabel='Epoch', ylabel='AUC90 (higher better)')
        axes[0, col].legend(fontsize=8)
        for row in range(2): axes[row, col].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder/'training_validation_history.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    parser.add_argument('--compact-dir', type=Path, default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 7])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--baseline-epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.seeds != [42, 7] or min(args.epochs, args.baseline_epochs, args.batch_size) < 1 or args.bootstrap < 20:
        parser.error('formal recipe requires seeds42/7 and positive budget')
    (verify if args.verify else run)(args)


if __name__ == '__main__':
    main()
