"""冻结RadBERT/RadGraph影像输入，同容量同初始化、同H、同队列配对重训。"""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from patient_data.export_mimiciv import sha256
from training.v6.compact_ablation import inputs
from training.v6.clinical_facts_context import FactsBatch, FactsData, FactsEncoder, checked
from training.v6.model import AnchoredModel
from training.v6.compact_model import CompactClinicalEncoder
from training.v6.train import fit, predict, write_json
from training.v6.group_ablation import evaluate
from training.v6.evaluation import GRID, metrics, paired_interval, public_metrics
from training.v6.risk import predict_cif
from training.v6 import facts_experiment

ARMS = ('without_new', 'availability_only', 'radbert', 'radgraph')
WIDTHS = (1536, 1536, 69)
LABELS = {'without_new': 'No new imaging', 'availability_only': 'Imaging availability only',
          'radbert': 'Frozen RadBERT: CXR + CT', 'radgraph': 'RadGraph-XL facts: CXR + CT'}


class ImagingEncoder(FactsEncoder):
    def __init__(self, text_width, width=128):
        nn.Module.__init__(self)
        self.fact_branches = nn.ModuleList([nn.Sequential(nn.Linear(v, 32), nn.GELU(), nn.Linear(32, 64), nn.LayerNorm(64)) for v in WIDTHS])
        self.fusion = nn.Sequential(nn.Linear(width + 3 * 64 + 3, width), nn.GELU(), nn.Dropout(.1), nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))
        self.core = CompactClinicalEncoder(text_width, width=width)

    def forward(self, batch, *, mask_pmh=False):
        if not isinstance(batch, FactsBatch): raise TypeError('explicit FactsBatch required')
        n = len(batch.compact.clinical.current_values)
        if batch.present.shape != (n, 3) or batch.present.dtype != torch.bool: raise ValueError('invalid source mask')
        reps = []
        for j, (value, width, branch) in enumerate(zip(batch.values, WIDTHS, self.fact_branches, strict=True)):
            if value.shape != (n, width) or not torch.isfinite(value).all(): raise ValueError('invalid imaging tensor')
            reps.append(branch(value) * batch.present[:, j, None])
        core, branches = self.core(batch.compact, mask_pmh=mask_pmh)
        return self.fusion(torch.cat([core, *reps, batch.present.to(core.dtype)], 1)), dict(branches, cxr=reps[0], ct=reps[1], discharge=reps[2])


class RadiologyAnchoredModel(AnchoredModel):
    def __init__(self, baseline, text_width, width=128):
        nn.Module.__init__(self)
        self.baseline = baseline.requires_grad_(False).eval()
        self.delta_head = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 4))
        nn.init.zeros_(self.delta_head[-1].weight); nn.init.zeros_(self.delta_head[-1].bias)
        self.encoder = ImagingEncoder(text_width, width); self.patient_shape = True


def model_for(base, width):
    torch.manual_seed(42)
    return RadiologyAnchoredModel(copy.deepcopy(base), width).cuda()


def load_sources(compact, radbert, radgraph):
    arrays = {}; provenance = {}; source_manifests = []
    for name, directory in (('radbert', radbert), ('radgraph', radgraph)):
        a, digest = checked(directory, 'imaging.npz', 'full_radiology_processed', compact)
        n = len(a['hadm_ids'])
        if a['values'].shape != (n, 2, 1536) or a['present'].shape != (n, 2): raise ValueError('imaging shape differs')
        if not np.isfinite(a['values']).all() or (a['values'][~a['present']] != 0).any(): raise ValueError('invalid imaging mask')
        arrays[name] = a; provenance[name] = digest
        m = json.loads((directory / 'manifest.json').read_text())
        if m['kind'] != name: raise ValueError('radiology model kind differs')
        source_manifests.append(m['source_manifest_sha256'])
    if len(set(source_manifests)) != 1: raise ValueError('radiology report source differs')
    if not np.array_equal(arrays['radbert']['present'], arrays['radgraph']['present']): raise ValueError('source populations differ')
    for directory in (radbert, radgraph):
        m = json.loads((directory / 'manifest.json').read_text())
        if m['reports'] != 14464 or not m['encoder_frozen']: raise ValueError('not full frozen imaging')
    return arrays, provenance


def attached(compact, sources, arm):
    a = sources['radgraph' if arm == 'radgraph' else 'radbert']; n = len(a['hadm_ids'])
    result = {k: a[k] for k in ('hadm_ids', 'subject_ids', 'patient_splits')}
    for j, key in enumerate(('cxr', 'ct')):
        result[key + '_values'] = a['values'][:, j] if arm in ('radbert', 'radgraph') else np.zeros((n, 1536), np.float32)
        result[key + '_present'] = a['present'][:, j] if arm != 'without_new' else np.zeros(n, bool)
    result['discharge_values'] = np.zeros((n, 69), np.float32); result['discharge_present'] = np.zeros(n, bool)
    return FactsData(compact, result, {}, ('cxr', 'ct'))


def prepared(data, ix):
    result = []
    for start in range(0, len(ix), 512):
        pos = ix[start:start + 512]; h, b, t, e = data.collate(pos, device='cuda')
        result.append((pos, h, b, t, e))
    return result


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2); started = time.perf_counter()
    data, compact, base, checkpoint = inputs(args.reference, args.compact_dir)
    sources, provenance = load_sources(compact, args.radbert, args.radgraph)
    ix = {f: np.flatnonzero(data.arrays['patient_splits'] == f) for f in ('train', 'validation', 'test')}
    args.output.mkdir(parents=True)
    write_json(args.output / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    contract = dict(seed=42, epochs=args.epochs, arms=ARMS, bootstrap=args.bootstrap, baseline_frozen=True,
                    zero_initialized_delta=True, batch_size=512, lr=3e-4, warmup=5, patience=10,
                    baseline_checkpoint_sha256=sha256(checkpoint), data_provenance=data.provenance,
                    compact_provenance=compact.provenance, facts_provenance=provenance, image_widths=WIDTHS,
                    text='Same existing four-field BERT; no fine-tuning', selection='Validation readmission integrated Brier 7-365',
                    endpoint='strict 491.21/J44.1, joint first readmission or death, World A, 365 days',
                    code_sha256={n: sha256(Path(__file__).parent / n) for n in ('radiology_experiment.py', 'train.py', 'risk.py', 'evaluation.py', 'compact_model.py')},
                    limits=['Single seed; historical test reused; no multiplicity correction', 'Representation comparison, not diagnostic gold-standard accuracy',
                            'Patient bootstrap does not include training variance', 'CXR/CT report availability differs; subgroup results are exploratory'])
    write_json(args.output / 'contract.json', contract); fitted = {}; initial = None
    for arm in ARMS:
        write_json(args.output / 'status.json', dict(status='training', arm=arm))
        model = model_for(base, data.text_vectors.shape[1]); state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if initial is None: initial = state
        elif not all(torch.equal(v, initial[k]) for k, v in state.items()): raise ValueError('initialization differs')
        source = attached(compact, sources, arm)
        fitted[arm] = fit(model, data, prepared(source, ix['train']), prepared(source, ix['validation']), ix['validation'], args.output / arm,
                          epochs=args.epochs, lr=3e-4, seed=42, patience=10, warmup=5)
        del model; torch.cuda.empty_cache()
    write_json(args.output / 'status.json', {'status': 'all_models_selected_before_test'})
    results = {}; predictions = {}
    for arm in ARMS:
        model = model_for(base, data.text_vectors.shape[1]); model.load_state_dict(torch.load(args.output / arm / 'best.pt', map_location='cuda', weights_only=True)['state_dict'])
        if not all(torch.equal(v, base.state_dict()[k]) for k, v in model.baseline.state_dict().items()): raise ValueError('history changed')
        source = attached(compact, sources, arm); results[arm] = {k: v for k, v in fitted[arm].items() if k != 'history'}; results[arm]['folds'] = {}
        for fold in ('validation', 'test'):
            curve, detail = evaluate(data, ix[fold], model, prepared(source, ix[fold]), args.output / arm, fold)
            results[arm]['folds'][fold] = detail; predictions[arm, fold] = curve
            for subgroup, mask in [('with_any_imaging', sources['radbert']['present'].any(1)), ('with_ct', sources['radbert']['present'][:, 1])]:
                subset = mask[ix[fold]]
                results[arm]['folds'][fold][subgroup] = public_metrics(metrics(data.times[ix[fold]][subset], data.events[ix[fold]][subset], curve[subset, :, 1], ordering=True))
    comparisons = {}
    for fold in ('validation', 'test'):
        selected = ix[fold]; comparisons[fold] = {}
        for left, right in [(a, 'without_new') for a in ARMS[1:]] + [('radbert', 'availability_only'), ('radgraph', 'availability_only'), ('radgraph', 'radbert')]:
            delta = paired_interval(data.arrays['subject_ids'][selected], data.times[selected], data.events[selected], predictions[left, fold][:, :, 1], predictions[right, fold][:, :, 1], replicates=args.bootstrap, seed=42)
            delta['mean_auc_difference'] = results[left]['folds'][fold]['readmission']['mean_auc_7_365'] - results[right]['folds'][fold]['readmission']['mean_auc_7_365']
            comparisons[fold][left + '_minus_' + right] = delta
    report = dict(status='completed_full_radiology_paired', initial_weights_identical=True, contract=contract,
                  models=results, comparisons=comparisons, elapsed_seconds=time.perf_counter() - started)
    write_json(args.output / 'report.json', report)
    facts_experiment.LABELS.update(LABELS); facts_experiment.render(args.output, report)
    write_json(args.output / 'status.json', {'status': report['status']})
    print(json.dumps({'status': report['status'], 'seconds': report['elapsed_seconds']}), flush=True)


def verify(args):
    r = json.loads((args.output / 'report.json').read_text()); c = r['contract']
    config = json.loads((args.output / 'config.json').read_text())
    for name, digest in c['code_sha256'].items():
        if sha256(Path(__file__).parent / name) != digest: raise ValueError('experiment code changed')
    data, compact, base, checkpoint = inputs(Path(config['reference']), Path(config['compact_dir']))
    sources, provenance = load_sources(compact, Path(config['radbert']), Path(config['radgraph']))
    if provenance != c['facts_provenance'] or sha256(checkpoint) != c['baseline_checkpoint_sha256']: raise ValueError('input changed')
    errors = {}
    for arm in ARMS:
        path = args.output / arm / 'best.pt'
        if sha256(path) != r['models'][arm]['checkpoint_sha256']: raise ValueError('checkpoint changed')
        model = model_for(base, data.text_vectors.shape[1]); model.load_state_dict(torch.load(path, map_location='cuda', weights_only=True)['state_dict'])
        if not all(torch.equal(v, base.state_dict()[k]) for k, v in model.baseline.state_dict().items()): raise ValueError('history changed')
        source = attached(compact, sources, arm)
        for fold in ('validation', 'test'):
            selected = np.flatnonzero(data.arrays['patient_splits'] == fold)
            a, k, _, delta = predict(model, prepared(source, selected)); curve = predict_cif(a, k, GRID)
            with np.load(args.output / arm / (fold + '_predictions.npz')) as saved:
                if not np.array_equal(saved['hadm_ids'], data.arrays['hadm_ids'][selected]) or not np.array_equal(saved['events'], data.events[selected]): raise ValueError('members or labels changed')
                if not np.allclose(curve, saved['curve'], atol=2e-6, rtol=1e-5): raise ValueError('prediction differs')
                errors[arm + '/' + fold] = float(np.max(np.abs(curve - saved['curve'])))
            actual = metrics(data.times[selected], data.events[selected], curve[:, :, 1])
            if any(abs(actual[x] - r['models'][arm]['folds'][fold]['readmission'][x]) > 1e-7 for x in ('integrated_brier_7_365', 'mean_auc_7_365')): raise ValueError('metrics differ')
    write_json(args.output / 'independent_verification.json', dict(status='all_arms_gpu_reloaded', history_unchanged=True, max_curve_errors=errors))
    print('all_arms_gpu_reloaded', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--compact-dir', type=Path, default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--radbert', type=Path, default=Path('patient_data/processed/v6_radbert_imaging_2026-10-10_r2'))
    p.add_argument('--radgraph', type=Path, default=Path('patient_data/processed/v6_radgraph_imaging_source_checked_2026-10-10'))
    p.add_argument('--output', type=Path, required=True); p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--bootstrap', type=int, default=200); p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    verify(args) if args.verify else run(args)
