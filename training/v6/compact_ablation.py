"""新接口五臂配对重训；冻结同一个历史H，既有BERT在各臂保持不变。"""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.data import load
from training.v6.compact_context_data import attach
from training.v6.compact_model import ARMS, CompactAnchoredModel, mask_group
from training.v6.model import HistoryBaseline
from training.v6.train import fit, predict, write_json
from training.v6.group_ablation import evaluate
from training.v6.risk import predict_cif
from training.v6.evaluation import GRID, metrics, paired_interval, calibration_bins

LABELS = {'full': 'Full compact clinical + existing BERT', 'without_medications': 'Without multi-source medications',
          'without_microbiology': 'Without microbiology', 'without_support': 'Without respiratory support',
          'without_labs': 'Without all 22 lab axes'}


def prepared(data, indices, size, arm, *, include_text):
    from dataclasses import replace
    result = []
    for start in range(0, len(indices), size):
        pos = indices[start:start + size]
        h, batch, t, e = data.collate(pos, device='cuda')
        if not include_text:
            b = batch.clinical
            batch = replace(batch, clinical=replace(b, snippet_vectors=b.snippet_vectors[:0], snippet_index=b.snippet_index[:0]))
        result.append((pos, h, mask_group(batch, arm), t, e))
    return result


def build(base, width, seed):
    torch.manual_seed(seed)
    return CompactAnchoredModel(copy.deepcopy(base), width).cuda()


def inputs(reference, compact_dir):
    config = json.loads((reference / 'config.json').read_text(encoding='utf-8'))
    report = json.loads((reference / 'report.json').read_text(encoding='utf-8'))
    data = load(Path(config['snapshot']), Path(config['state_dir']), Path(config['labels']),
                field_dir=Path(config['field_dir']), vectors_path=Path(config['vectors_path']))
    if data.provenance != report['contract']['data_provenance']:
        raise ValueError('reference data provenance mismatch')
    checkpoint = reference / 'history/best.pt'
    if sha256(checkpoint) != report['models']['history']['checkpoint_sha256']:
        raise ValueError('history checkpoint hash mismatch')
    base = HistoryBaseline().cuda()
    base.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True)['state_dict'])
    return data, attach(data, compact_dir), base, checkpoint


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.seed != 42:
        raise ValueError('this matched pilot reuses the seed42 H; other seeds need an explicit baseline contract')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    started = time.perf_counter()
    data, compact, base, checkpoint = inputs(args.reference, args.compact_dir)
    indices = {f: np.flatnonzero(data.arrays['patient_splits'] == f) for f in ('train', 'validation', 'test')}
    args.output.mkdir(parents=True)
    write_json(args.output / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    contract = {'seed': args.seed, 'max_epochs': args.epochs, 'batch_size': args.batch_size, 'patience': 10,
                'lr': 3e-4, 'warmup': 5, 'bootstrap': args.bootstrap, 'arms': list(ARMS),
                'data_provenance': data.provenance, 'compact_provenance': compact.provenance,
                'baseline_checkpoint_sha256': sha256(checkpoint), 'baseline_frozen_for_every_arm': True,
                'text': 'existing B four-field Bio_ClinicalBERT, unchanged in every arm' if args.text == 'bert' else 'none for every arm',
                'whole_group_omission': 'content, masks and presence zeroed; all modules/slots retained',
                'selection': 'validation readmission integrated Brier 7-365; no test-based model selection',
                'endpoint': 'strict 491.21/J44.1; joint readmission/death; World A 365-day observation contract',
                'device': torch.cuda.get_device_name(0), 'fold_admissions': {f: len(ix) for f, ix in indices.items()},
                'code_sha256': {f: sha256(Path(__file__).parent / f) for f in
                                ('compact_ablation.py', 'compact_model.py', 'compact_context_data.py', 'train.py', 'model.py', 'data.py', 'risk.py', 'evaluation.py')},
                'limits': ['One training seed; reused historical test is not a blind test',
                           'Patient-bootstrap CI does not include retraining variance; no multiplicity adjustment',
                           'Omission includes measurement availability; no causal treatment effect',
                           'Comparisons with old V6 change input and architecture and are descriptive only']}
    write_json(args.output / 'contract.json', contract)
    initial, fitted = None, {}
    for arm in ARMS:
        write_json(args.output / 'status.json', {'status': 'training', 'arm': arm})
        model = build(base, data.text_vectors.shape[1], args.seed)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if initial is None:
            initial = state
        elif state.keys() != initial.keys() or not all(torch.equal(v, initial[k]) for k, v in state.items()):
            raise ValueError('unmatched initial parameters')
        train = prepared(compact, indices['train'], args.batch_size, arm, include_text=args.text == 'bert')
        val = prepared(compact, indices['validation'], args.batch_size, arm, include_text=args.text == 'bert')
        fitted[arm] = fit(model, data, train, val, indices['validation'], args.output / arm,
                          epochs=args.epochs, lr=3e-4, seed=args.seed, patience=10, warmup=5)
        del train, val, model
        torch.cuda.empty_cache()
    write_json(args.output / 'status.json', {'status': 'all_models_selected_before_test'})
    results, predictions = {}, {}
    for arm in ARMS:
        model = build(base, data.text_vectors.shape[1], args.seed)
        model.load_state_dict(torch.load(args.output / arm / 'best.pt', map_location='cuda', weights_only=True)['state_dict'])
        if not all(torch.equal(v, base.state_dict()[k]) for k, v in model.baseline.state_dict().items()):
            raise ValueError('frozen baseline changed')
        results[arm] = {k: v for k, v in fitted[arm].items() if k != 'history'}
        results[arm]['folds'] = {}
        for fold in ('validation', 'test'):
            batches = prepared(compact, indices[fold], args.batch_size, arm, include_text=args.text == 'bert')
            curve, detail = evaluate(data, indices[fold], model, batches, args.output / arm, fold)
            results[arm]['folds'][fold] = detail
            predictions[arm, fold] = curve
            del batches
    comparisons = {}
    for fold in ('validation', 'test'):
        ix = indices[fold]
        comparisons[fold] = {}
        for arm in ARMS[1:]:
            detail = paired_interval(data.arrays['subject_ids'][ix], data.times[ix], data.events[ix],
                                     predictions[arm, fold][:, :, 1], predictions['full', fold][:, :, 1],
                                     replicates=args.bootstrap, seed=args.seed)
            detail['mean_auc_difference'] = (results[arm]['folds'][fold]['readmission']['mean_auc_7_365']
                                             - results['full']['folds'][fold]['readmission']['mean_auc_7_365'])
            comparisons[fold][arm] = detail
    report = {'status': 'completed_compact_group_ablation_single_seed', 'contract': contract,
              'initial_state_identical_across_arms': True, 'models': results, 'comparisons': comparisons,
              'elapsed_seconds': round(time.perf_counter() - started, 2), 'limits': contract['limits']}
    write_json(args.output / 'report.json', report)
    render(args.output, report)
    write_json(args.output / 'status.json', {'status': report['status']})
    print(json.dumps({'status': report['status'], 'seconds': report['elapsed_seconds']}), flush=True)


def verify(args):
    """新进程全折磁盘重载；核对配方、成员、输入/代码/检查点哈希。"""
    report = json.loads((args.output / 'report.json').read_text(encoding='utf-8'))
    config = json.loads((args.output / 'config.json').read_text(encoding='utf-8'))
    c = report['contract']
    for f, expected in c['code_sha256'].items():
        if sha256(Path(__file__).parent / f) != expected:
            raise ValueError('run code changed: ' + f)
    data, compact, base, checkpoint = inputs(Path(config['reference']), Path(config['compact_dir']))
    if (data.provenance != c['data_provenance'] or compact.provenance != c['compact_provenance']
            or sha256(checkpoint) != c['baseline_checkpoint_sha256']):
        raise ValueError('run inputs changed')
    errors = {}
    for arm in ARMS:
        path = args.output / arm / 'best.pt'
        if sha256(path) != report['models'][arm]['checkpoint_sha256']:
            raise ValueError('checkpoint changed')
        model = build(base, data.text_vectors.shape[1], c['seed'])
        model.load_state_dict(torch.load(path, map_location='cuda', weights_only=True)['state_dict'])
        if not all(torch.equal(v, base.state_dict()[k]) for k, v in model.baseline.state_dict().items()):
            raise ValueError('history changed')
        for fold in ('validation', 'test'):
            ix = np.flatnonzero(data.arrays['patient_splits'] == fold)
            batches = prepared(compact, ix, c['batch_size'], arm, include_text=config['text'] == 'bert')
            a, k, nll, delta = predict(model, batches)
            curve = predict_cif(a, k, GRID, step_days=.25)
            with np.load(args.output / arm / (fold + '_predictions.npz')) as saved:
                for key, value in (('hadm_ids', data.arrays['hadm_ids'][ix]), ('subject_ids', data.arrays['subject_ids'][ix]),
                                   ('events', data.events[ix]), ('times', data.times[ix])):
                    if not np.array_equal(saved[key], value):
                        raise ValueError('prediction members/outcomes mismatch')
                for key, value in (('log_amplitude', a), ('shape', k), ('curve', curve), ('delta', delta)):
                    if not np.allclose(saved[key], value, atol=2e-6, rtol=1e-5):
                        raise ValueError('prediction reproduction failed: ' + arm + '/' + key)
                errors[arm + '/' + fold] = float(np.abs(saved['curve'] - curve).max())
            actual = metrics(data.times[ix], data.events[ix], curve[:, :, 1])
            for metric in ('integrated_brier_7_365', 'mean_auc_7_365'):
                if abs(actual[metric] - report['models'][arm]['folds'][fold]['readmission'][metric]) > 1e-7:
                    raise ValueError('report metric mismatch')
    write_json(args.output / 'independent_verification.json', {'status': 'all_five_arms_gpu_reloaded_verified',
               'report_sha256': sha256(args.output / 'report.json'), 'max_curve_errors': errors,
               'baseline_unchanged': True, 'folds': ['validation', 'test']})
    print('all_five_arms_gpu_reloaded_verified', flush=True)


def render(root, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = root / 'figures'; folder.mkdir()
    for fold in ('validation', 'test'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
        for i, arm in enumerate(ARMS[1:]):
            c = report['comparisons'][fold][arm]
            axes[0].plot(c['integrated_brier_95ci'], [i, i], color='#2474b7')
            axes[0].scatter(c['integrated_brier_difference'], i, marker='D', color='#d95f02')
            axes[1].scatter(c['mean_auc_difference'], i, marker='D', color='#d95f02')
        for ax in axes:
            ax.axvline(0, color='gray'); ax.set_yticks(range(4), [LABELS[a] for a in ARMS[1:]], fontsize=9)
            ax.invert_yaxis() if not ax.yaxis_inverted() else None
            ax.grid(axis='x', alpha=.2)
        axes[0].set_title('Integrated Brier: removed - full (lower better)')
        axes[1].set_title('Mean AUC(t): removed - full (higher better)')
        axes[0].set_xlabel('Paired difference; 200 patient-bootstrap replicates')
        axes[1].set_xlabel('Paired difference; single-seed point estimate (no CI)')
        fig.suptitle(f'Strict AECOPD {fold}; each group retrained; one seed; no multiplicity adjustment')
        fig.tight_layout(); fig.savefig(folder / (fold + '_paired_differences.png'), dpi=180); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig_cal, cal = plt.subplots(figsize=(6, 5))
        cal.plot([0, 1], [0, 1], '--', color='gray')
        for arm in ARMS:
            with np.load(root / arm / (fold + '_predictions.npz')) as saved:
                m = metrics(saved['times'], saved['events'], saved['curve'][:, :, 1])
                bins = calibration_bins(saved['times'], saved['events'], saved['curve'][:, :, 1])
            axes[0].plot(GRID[7:], m['auc_curve'][7:], label=LABELS[arm])
            axes[1].plot(GRID[7:], m['brier_curve'][7:])
            cal.plot([b['predicted'] for b in bins], [b['observed'] for b in bins], 'o-', label=LABELS[arm])
        axes[0].set_title('Strict readmission AUC(t)'); axes[1].set_title('Strict readmission Brier(t)')
        axes[0].legend(fontsize=7)
        for ax in axes: ax.set_xlabel('Days after discharge'); ax.grid(alpha=.2)
        fig.suptitle(f'{fold}: fixed patient split; World A; existing BERT held constant')
        fig.tight_layout(); fig.savefig(folder / (fold + '_auc_brier.png'), dpi=180); plt.close(fig)
        cal.set(xlabel='Predicted 90-day cumulative incidence', ylabel='Observed World A fraction', title=f'{fold}: 90-day calibration')
        cal.legend(fontsize=7); cal.grid(alpha=.2)
        fig_cal.tight_layout(); fig_cal.savefig(folder / (fold + '_calibration.png'), dpi=180); plt.close(fig_cal)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for arm in ARMS:
        history = json.loads((root / arm / 'history.json').read_text())
        axes[0].plot([h['epoch'] for h in history], [h['validation_integrated_brier'] for h in history], label=LABELS[arm])
        axes[1].plot([h['epoch'] for h in history[1:]], [h['train_nll'] for h in history[1:]])
    axes[0].set_title('Validation integrated Brier (checkpoint selection)'); axes[1].set_title('Training joint NLL')
    axes[0].legend(fontsize=7)
    for ax in axes: ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder / 'learning_curves.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    parser.add_argument('--compact-dir', type=Path, default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--text', choices=('bert', 'none'), default='bert')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.bootstrap < 20:
        parser.error('invalid budget')
    verify(args) if args.verify else run(args)
