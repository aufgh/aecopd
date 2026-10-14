"""固定单种子四臂完整训练；验证选模后才统一读取历史测试预测。"""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from patient_data.export_mimiciv import sha256
from training.v6.data import load
from training.v6.model import HistoryBaseline, ConstantCorrection, AnchoredModel
from training.v6.risk import competing_nll, predict_cif, RiskOutput
from training.v6.evaluation import GRID, metrics, public_metrics, paired_interval

ARMS = ('history', 'constant', 'clinical_no_text', 'clinical_bert')


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def forward(model, history, batch):
    return model(history, batch) if isinstance(model, AnchoredModel) else model(history)


def make_batches(data, indices, size, device, *, include_text):
    result = []
    for start in range(0, len(indices), size):
        positions = indices[start:start + size]
        h, b, t, e = data.collate(positions, device)
        if not include_text:
            b = replace(b, snippet_vectors=b.snippet_vectors[:0], snippet_index=b.snippet_index[:0])
        result.append((positions, h, b, t, e))
    return result


@torch.no_grad()
def predict(model, batches):
    model.eval()
    aa, kk, losses, deltas = [], [], [], []
    for positions, h, b, t, e in batches:
        output = forward(model, h, b)
        aa.append(output.log_amplitude.cpu().numpy()); kk.append(output.shape.cpu().numpy())
        losses.append((float(competing_nll(output, t, e)), len(positions)))
        if output.delta is not None:
            deltas.append(output.delta.cpu().numpy())
    a, k = np.concatenate(aa), np.concatenate(kk)
    nll = sum(x*n for x, n in losses) / sum(n for _, n in losses)
    delta = np.concatenate(deltas) if deltas else np.zeros((len(a), 2, 2))
    return a, k, nll, delta


def gradient_diagnostics(model):
    report = {}
    modules = [('current', model.encoder.current), ('lab', model.encoder.lab),
                         ('medication', model.encoder.med), ('text', model.encoder.text_projection),
                         ('fusion', model.encoder.fusion)]
    for name in ('microbiology', 'support'):
        if hasattr(model.encoder, name):
            modules.append((name, getattr(model.encoder, name)))
    for name, module in modules:
        params = list(module.parameters())
        count = sum(p.numel() for p in params)
        grads = [p.grad for p in params if p.grad is not None]
        report[name] = {"gradient_rms_last_batch": math.sqrt(sum(float(g.square().sum()) for g in grads) / count),
                        "parameter_rms": math.sqrt(sum(float(p.detach().square().sum()) for p in params) / count)}
    return report


def fit(model, data, train_batches, val_batches, validation_indices, directory, *, epochs, lr,
        seed, patience, warmup=0, weight_decay=1e-4):
    directory.mkdir()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    # 构造/设备迁移后恢复共同随机流；BERT/no-text共同架构与相同非文本初值。
    torch.manual_seed(seed)
    # 默认冻结臂行为不变；独立匹配解冻候选允许历史参数收到梯度。
    frozen = ({k: v.detach().cpu().clone() for k, v in model.baseline.state_dict().items()}
              if hasattr(model, 'baseline') and not any(p.requires_grad for p in model.baseline.parameters()) else None)
    history, best_score, best_epoch, stale = [], math.inf, -1, 0
    best_state = None
    started = time.perf_counter()
    for epoch in range(epochs + 1):
        train_nll, regularization, diagnostics = None, None, {}
        if epoch > 0:
            if warmup and epoch <= warmup:
                rate = lr * epoch / warmup
            elif warmup:
                rate = lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup))))
            else:
                rate = lr
            for group in optimizer.param_groups:
                group['lr'] = rate
            model.train()
            summed, penalty_sum, total = 0., 0., 0
            for j in rng.permutation(len(train_batches)):
                positions, h, b, t, e = train_batches[j]
                optimizer.zero_grad(set_to_none=True)
                output = forward(model, h, b)
                nll = competing_nll(output, t, e)
                penalty = output.delta.square().mean() * 1e-4 if output.delta is not None else nll.new_zeros(())
                loss = nll + penalty
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                loss.backward()
                if frozen is not None and any(p.grad is not None for p in model.baseline.parameters()):
                    raise ValueError('baseline gradient leakage')
                torch.nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step()
                summed += float(nll.detach()) * len(positions)
                penalty_sum += float(penalty.detach()) * len(positions)
                total += len(positions)
            train_nll, regularization = summed / total, penalty_sum / total
            if isinstance(model, AnchoredModel):
                diagnostics = gradient_diagnostics(model)
        a, k, val_nll, delta = predict(model, val_batches)
        curves = predict_cif(a, k, GRID, step_days=.25)
        val = metrics(data.times[validation_indices], data.events[validation_indices], curves[:, :, 1])
        score = val['integrated_brier_7_365']
        row = {'epoch': epoch, 'train_nll': train_nll, 'regularization': regularization,
               'validation_nll': val_nll, 'validation_integrated_brier': score,
               'validation_mean_auc': val['mean_auc_7_365'], 'diagnostics': diagnostics,
               'delta_mean': delta.mean(axis=0).tolist(), 'delta_std': delta.std(axis=0).tolist(),
               'shape_min': float(k.min()), 'shape_max': float(k.max()),
               'elapsed_seconds': round(time.perf_counter() - started, 2)}
        history.append(row)
        write_json(directory / 'history.json', history)
        if score < best_score - 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: v.detach().cpu().clone() for key, v in model.state_dict().items()}
        elif epoch > 0:
            stale += 1
        if epoch % 5 == 0 or stale >= patience:
            print(json.dumps({'arm': directory.name, 'epoch': epoch, 'validation_IBS': score,
                              'best_epoch': best_epoch, 'seconds': row['elapsed_seconds']}), flush=True)
        if epoch > 0 and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError('no best checkpoint')
    model.load_state_dict(best_state)
    if frozen is not None:
        if not all(torch.equal(v.detach().cpu(), frozen[key]) for key, v in model.baseline.state_dict().items()):
            raise ValueError('frozen baseline changed')
    checkpoint = directory / 'best.pt'
    torch.save({'state_dict': best_state, 'best_epoch': best_epoch, 'seed': seed}, checkpoint)
    # 验证真正从磁盘检查点重载而非只保留内存模型。
    restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(restored['state_dict'])
    a1, k1, _, _ = predict(model, val_batches)
    model.load_state_dict(best_state)
    a2, k2, _, _ = predict(model, val_batches)
    if not np.allclose(a1, a2, atol=2e-6, rtol=1e-5) or not np.allclose(k1, k2, atol=2e-6, rtol=1e-5):
        raise ValueError('checkpoint prediction reproduction failed')
    return {'best_epoch': best_epoch, 'epochs_trained': len(history) - 1, 'best_validation_IBS': best_score,
            'history': history, 'checkpoint_sha256': sha256(checkpoint), 'checkpoint_reloaded': True}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required, no CPU fallback')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True)
    config = vars(args).copy()
    write_json(args.output / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in config.items()})
    write_json(args.output / 'status.json', {'status': 'loading_validated_data'})
    data = load(args.snapshot, args.state_dir, args.labels, field_dir=args.field_dir, vectors_path=args.vectors_path)
    indices = {fold: np.flatnonzero(data.arrays['patient_splits'] == fold) for fold in ('train', 'validation', 'test')}
    if any(not len(ix) for ix in indices.values()):
        raise ValueError('empty fold')
    # 验证/训练预加载；此时不构造或计算历史测试预测。
    cached = {text: {f: make_batches(data, indices[f], args.batch_size, 'cuda', include_text=text)
                      for f in ('train', 'validation')} for text in (False, True)}
    common = {'data_provenance': data.provenance, 'history_normalization': data.normalization,
              'seed': args.seed, 'selection': 'validation readmission integrated Brier 7-365 daily grid',
              'optimizer': 'AdamW; NLL plus 1e-4 delta mean square; weight decay 1e-4; gradient clip 1',
              'sampling': 'admission equal weight; fixed admission minibatches with shuffled batch order',
              'folds': {f: {'admissions': len(ix), 'patients': len(set(data.arrays['subject_ids'][ix].tolist()))} for f, ix in indices.items()},
              'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'code_sha256': {f: sha256(Path(__file__).parent / f) for f in ('train.py', 'model.py', 'risk.py', 'data.py', 'evaluation.py')}}
    write_json(args.output / 'contract.json', common)
    torch.manual_seed(args.seed)
    baseline = HistoryBaseline().cuda()
    fitted, models = {}, {'history': baseline}
    fitted['history'] = fit(baseline, data, cached[False]['train'], cached[False]['validation'], indices['validation'],
                             args.output / 'history', epochs=args.baseline_epochs, lr=.01, seed=args.seed, patience=20)
    for arm in ARMS[1:]:
        write_json(args.output / 'status.json', {'status': 'training', 'arm': arm})
        torch.manual_seed(args.seed)
        base_copy = copy.deepcopy(baseline).cuda()
        model = (ConstantCorrection(base_copy) if arm == 'constant'
                 else AnchoredModel(base_copy, data.text_vectors.shape[1])).cuda()
        models[arm] = model
        text = arm == 'clinical_bert'
        fitted[arm] = fit(model, data, cached[text]['train'], cached[text]['validation'], indices['validation'],
                          args.output / arm, epochs=args.epochs, lr=.005 if arm == 'constant' else 3e-4,
                          seed=args.seed, patience=10, warmup=0 if arm == 'constant' else 5)
    write_json(args.output / 'status.json', {'status': 'all_models_selected_before_historical_test_evaluation'})
    results, predictions = {}, {}
    for arm in ARMS:
        results[arm] = {k: v for k, v in fitted[arm].items() if k != 'history'}
        results[arm]['folds'] = {}
        for fold in ('validation', 'test'):
            batches = (cached[arm == 'clinical_bert']['validation'] if fold == 'validation' else
                       make_batches(data, indices[fold], args.batch_size, 'cuda', include_text=arm == 'clinical_bert'))
            a, k, nll, delta = predict(models[arm], batches)
            curve = predict_cif(a, k, GRID, step_days=.25)
            fine = predict_cif(a[:64], k[:64], GRID, step_days=.125)
            error = float(np.max(np.abs(fine - curve[:64])))
            if error > 2e-4:
                raise ValueError(f'CIF step convergence insufficient {arm}/{fold}: {error}')
            ix = indices[fold]
            readmission = metrics(data.times[ix], data.events[ix], curve[:, :, 1], ordering=True)
            death = metrics(data.times[ix], data.events[ix], curve[:, :, 2], cause=2, ordering=True)
            stratified = {}
            prior = np.expm1(data.history_raw[ix, 2]).round().astype(int)
            for label, keep in (('prior_0', prior == 0), ('prior_1_2', (prior >= 1) & (prior <= 2)), ('prior_3_plus', prior >= 3)):
                if keep.any():
                    stratified[label] = public_metrics(metrics(data.times[ix][keep], data.events[ix][keep], curve[keep, :, 1], ordering=True))
            results[arm]['folds'][fold] = {'readmission': public_metrics(readmission), 'death': public_metrics(death),
                                          'nll': nll, 'history_strata': stratified, 'CIF_step_convergence_max_error_first64': error,
                                          'maximum_probability_sum_error': float(np.abs(curve.sum(2) - 1).max())}
            predictions[arm, fold] = curve
            np.savez_compressed(args.output / arm / (fold + '_predictions.npz'),
                                hadm_ids=data.arrays['hadm_ids'][ix], subject_ids=data.arrays['subject_ids'][ix],
                                times=data.times[ix], events=data.events[ix], history_counts=prior,
                                curve=curve, log_amplitude=a, shape=k, delta=delta)
    comparisons = {}
    for fold in ('validation', 'test'):
        ix = indices[fold]
        comparisons[fold] = {}
        for candidate, comparator in (('constant', 'history'), ('clinical_no_text', 'constant'),
                                       ('clinical_bert', 'clinical_no_text'), ('clinical_bert', 'constant')):
            label = candidate + '_minus_' + comparator
            comparisons[fold][label] = paired_interval(data.arrays['subject_ids'][ix], data.times[ix], data.events[ix],
                                                       predictions[candidate, fold][:, :, 1], predictions[comparator, fold][:, :, 1],
                                                       replicates=args.bootstrap, seed=args.seed)
    report = {'status': 'completed_single_seed_exploratory', 'contract': common, 'models': results, 'comparisons': comparisons,
              'limits': ['World A record-observed endpoint, external readmissions unknown', 'Historical test reused, not a new blind test',
                         'One training seed; cluster intervals do not include retraining variance',
                         'No new encoder, drug-lab cycle or extra predictor introduced', 'C-index explicitly project fixed-horizon CIF score definition']}
    write_json(args.output / 'report.json', report)
    from training.v6.plot import render
    render(args.output)
    write_json(args.output / 'status.json', {'status': report['status']})
    print(json.dumps({'status': report['status'], 'output': str(args.output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'state_dir', 'labels', 'output', 'field_dir', 'vectors_path'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--baseline-epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--bootstrap', type=int, default=200)
    args = parser.parse_args()
    if args.epochs < 1 or args.baseline_epochs < 1 or args.batch_size < 1 or args.bootstrap < 20:
        parser.error('invalid training budget')
    run(args)


if __name__ == '__main__':
    main()
