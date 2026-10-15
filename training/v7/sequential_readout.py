"""只读汇总本批已锁定实验；患者配对重采样不重新训练或选择模型。"""
import argparse
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json

ROOT = Path('training_runs/v7_sequential_research_2026-10-15')
OUT = ROOT/'readout.json'
FIGURE = Path('docs/figures/v7_sequential_research_2026-10-15.png')
OLD = Path('training_runs/v7_first_2026-10-15_r2')
STRONG = Path('training_runs/v6_symmetric_context_2026-10-14')
FOLDERS = ('phase1', 'cached', 'cached_followup', 'drug_screen', 'strong_anchor',
           'query_replication', 'residual_controls', 'strong_replication', 'residual_followup',
           'two_fold', 'full_refit')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def json_read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def summarize(metrics):
    return dict(f1=metrics['annual']['f1'], threshold=metrics['annual']['threshold'],
                ibs=metrics['readmission']['integrated_brier_7_365'],
                death_ibs=metrics['death']['integrated_brier_7_365'],
                mean_auc=metrics['readmission']['mean_auc_7_365'])


def outcome_statistics(curve, times, events, threshold):
    y = (events == 1) & (times <= 365)
    flag = curve[:, 365, 1] >= threshold
    tp, fp, fn = (np.asarray(v, dtype=np.float64) for v in (flag & y, flag & ~y, ~flag & y))
    targets = (events[:, None, None] == np.array([1, 2])[None, None, :]) & (times[:, None, None] <= np.arange(366)[None, :, None])
    errors = np.trapezoid((curve[:, 7:, 1:] - targets[:, 7:])**2, np.arange(7, 366), axis=1)/358
    denom = 2*tp.sum()+fp.sum()+fn.sum()
    return dict(f1=float(2*tp.sum()/denom) if denom else 0., ibs=float(errors[:, 0].mean()),
                death_ibs=float(errors[:, 1].mean())), (tp, fp, fn, errors[:, 0])


def paired(a, b, weights):
    def bootstrap(values):
        tp, fp, fn, error = values
        denominator = weights @ (2*tp+fp+fn)
        f1 = np.divide(2*(weights @ tp), denominator, out=np.zeros(len(weights)), where=denominator > 0)
        return f1, (weights @ error)/weights.sum(1)
    av, bv = bootstrap(a), bootstrap(b)
    return dict(f1_95ci=np.quantile(av[0]-bv[0], [.025, .975]).tolist(),
                ibs_95ci=np.quantile(av[1]-bv[1], [.025, .975]).tolist())


def calculate():
    data = load(ROOT/'cached/cache.npz'); tune = data['tune']
    times, events, subjects = data['times'][tune], data['events'][tune], data['subject_ids'][tune]
    models, statistics, sources = {}, {}, {}

    def remember(path):
        sources[str(path)] = sha256(path)

    def register(name, folder, info, prediction='best_predictions.npz', curve=None, metadata=None):
        metrics = info.get('metrics', info)
        entry = summarize(metrics)
        entry.update(metadata or {})
        pred_path = folder/prediction
        remember(pred_path)
        z = load(pred_path)
        if curve is None:
            curve = z['curve']
        for key, expected in (('subject_ids', subjects), ('hadm_ids', data['hadm_ids'][tune]),
                              ('times', times), ('events', events), ('positions', tune), ('tune', tune)):
            if key in z and not (key == 'positions' and z[key].shape != tune.shape):
                np.testing.assert_array_equal(z[key], expected)
        if 'predictions_sha256' in info and info['predictions_sha256'] != sha256(pred_path):
            raise ValueError('prediction hash differs: '+name)
        actual, values = outcome_statistics(curve, times, events, entry['threshold'])
        for key in actual:
            if abs(actual[key]-entry[key]) > 1e-12:
                raise ValueError('point metric differs: '+name+'/'+key)
        if curve.shape != (len(tune), 366, 3) or not np.isfinite(curve).all():
            raise ValueError('invalid curve '+name)
        np.testing.assert_allclose(curve.sum(2), 1., atol=1e-12, rtol=0)
        if np.min(np.diff(curve[:, :, 1:], axis=1)) < -1e-12:
            raise ValueError('nonmonotone CIF '+name)
        models[name] = entry; statistics[name] = values

    remember(ROOT/'cached/cache_contract.json'); remember(Path(__file__))
    for root in (STRONG, OLD):
        proof = json_read(root/'verification.json')
        if proof['report_sha256'] != sha256(root/'report.json'):
            raise ValueError('reference replay stale')
        remember(root/'report.json'); remember(root/'verification.json')
    strong = json_read(STRONG/'report.json')['results']['symmetric']
    register('reference/strong_v6', STRONG/'symmetric', strong, 'tuning_predictions.npz',
             metadata=dict(epoch=strong['best_epoch'], reused=True))
    old = json_read(OLD/'report.json')
    for name, info in old['results'].items():
        register('reference/v7_'+name, OLD/name, info, 'tuning_predictions.npz',
                 metadata=dict(epoch=info['best_epoch'], reused=True))

    proofs = {}
    def collect(value, folder, name):
        if 'outputs' in value:
            b = value['outputs']['best']
            register(name, folder, b, metadata=dict(epoch=b['epoch'], completed_epochs=value['completed_epochs'],
                     seconds=value.get('seconds', value.get('fit_seconds')), reused=False))
            remember(folder/'history.json')
            return
        for key, child in value.items():
            if isinstance(child, dict):
                collect(child, folder/('seed'+key if key.isdigit() else key), name+'/'+key)

    for category in FOLDERS:
        folder = ROOT/category; report = json_read(folder/'report.json'); proof = json_read(folder/'verification.json')
        if proof['report_sha256'] != sha256(folder/'report.json'):
            raise ValueError('stale independent replay '+category)
        error = proof.get('maximum_error', proof.get('maximum_prediction_error'))
        if error != 0:
            raise ValueError('nonzero or unknown replay '+category)
        proofs[category] = dict(maximum_error=error, report_sha256=proof['report_sha256'])
        remember(folder/'report.json'); remember(folder/'verification.json')
        if category == 'strong_anchor':
            collect(report, folder, category)
        elif category == 'strong_replication':
            collect(report['result'], folder/'model', category)
        elif category in ('two_fold', 'full_refit'):
            z = load(folder/'predictions.npz')
            for i, family in enumerate(('cox', 'tree', 'neural', 'equal', 'weighted', 'weighted_temperature')):
                if category == 'full_refit':
                    curve = z[family]
                else:
                    curve = z['outer'][i] if i < 3 else z['outer'].mean(0) if family == 'equal' else z['weighted' if family == 'weighted' else 'calibrated']
                register(category+'/'+family, folder, report['metrics'][family], 'predictions.npz', curve=curve,
                         metadata=dict(reused=False, ensemble=family in ('equal', 'weighted', 'weighted_temperature')))
        else:
            collect(report['results'], folder, category)

    people, inverse = np.unique(subjects, return_inverse=True)
    rng = np.random.default_rng(42)
    counts = np.array([np.bincount(rng.integers(len(people), size=len(people)), minlength=len(people)) for _ in range(200)])
    weights = counts[:, inverse].astype(np.float64)
    pairs = [('cached/availability', 'reference/strong_v6'), ('cached/availability', 'cached/global'),
             ('cached/linear', 'cached/availability'), ('cached/clinical_interaction', 'cached/clinical_main'),
             ('cached/routing_prior', 'cached/routing_uniform'), ('drug_screen/name_semantic', 'drug_screen/name_random'),
             ('drug_screen/dynamics_slope', 'drug_screen/dynamics_source'),
             ('phase1/without_cross_query', 'reference/v7_nll_control'),
             ('query_replication/without_cross_query', 'query_replication/control'),
             ('strong_anchor', 'reference/strong_v6'), ('strong_replication', 'reference/strong_v6'),
             ('residual_controls/fixed_projection', 'residual_controls/scaled_v7'),
             ('residual_controls/fixed_projection', 'reference/strong_v6'),
             ('residual_controls/zero_gate', 'residual_controls/scaled_v7'),
             ('two_fold/weighted_temperature', 'two_fold/cox'),
             ('full_refit/weighted_temperature', 'full_refit/cox'),
             ('full_refit/weighted_temperature', 'reference/strong_v6')]
    for seed in (7, 101):
        pairs.append((f'residual_followup/{seed}/fixed_projection', f'residual_followup/{seed}/scaled_v7'))
    paired_results = []
    for a, b in pairs:
        paired_results.append(dict(candidate=a, comparator=b, delta_f1=models[a]['f1']-models[b]['f1'],
            delta_ibs=models[a]['ibs']-models[b]['ibs'], **paired(statistics[a], statistics[b], weights)))
    seed_groups = {name: ['cached/'+name]+[f'cached_followup/{seed}/{name}' for seed in (7, 101)]
                   for name in ('global', 'availability', 'linear', 'v7_frozen', 'routing_prior')}
    seed_groups.update({name: ['residual_controls/'+name]+[f'residual_followup/{seed}/{name}' for seed in (7, 101)]
                        for name in ('scaled_v7', 'fixed_projection')})
    seed_summary = {name: {metric: dict(values=[models[k][metric] for k in keys],
        mean=float(np.mean([models[k][metric] for k in keys])), sd=float(np.std([models[k][metric] for k in keys], ddof=1)))
        for metric in ('f1', 'ibs')} for name, keys in seed_groups.items()}
    new_neural = [m for m in models.values() if 'completed_epochs' in m]
    return dict(models=models, seed_summary=seed_summary, paired=paired_results, proofs=proofs,
        source_sha256=sources, bootstrap=dict(replicates=200, seed=42, patients=len(people), admissions=len(tune),
        thresholds='frozen at saved model selection; no bootstrap threshold reoptimization',
        limits='conditional on fitted models and exposed tuning; no multiplicity adjustment; not an independent validation'),
        execution=dict(new_neural_fits=len(new_neural), other_member_fits=9,
            total_new_model_fits=len(new_neural)+9, completed_neural_epochs=sum(m['completed_epochs'] for m in new_neural),
            selected_initialization=sum(m['epoch'] == 0 for m in new_neural)), default_replaced=False)


def plot(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.0), gridspec_kw={'width_ratios': [1.2, 1.2, 1.5]})
    names = ('global', 'availability', 'linear', 'v7_frozen', 'routing_prior', 'scaled_v7', 'fixed_projection')
    labels = ('Global calibration', 'Lab availability/time', 'Lab physical values', 'Frozen V7 (unscaled)',
              'Soft clinical routing', 'Frozen V7 (scaled)', 'Fixed ridge residual')
    y = np.arange(len(names))
    for axis, metric in zip(axes[:2], ('f1', 'ibs')):
        means = [report['seed_summary'][n][metric]['mean'] for n in names]
        sd = [report['seed_summary'][n][metric]['sd'] for n in names]
        axis.errorbar(means, y, xerr=sd, fmt='o', color='#24618e', capsize=4)
        axis.axvline(report['models']['reference/strong_v6'][metric], color='#bb5839', ls='--', label='Frozen full V6')
        axis.set_yticks(y, labels if metric == 'f1' else ['']*len(names)); axis.invert_yaxis()
        axis.set_xlabel('Annual F1 (higher better)' if metric == 'f1' else 'Readmission IBS (lower better)')
        axis.set_title('3 reader seeds: mean +/- SD'); axis.grid(axis='x', alpha=.2)
    keys = [('cached/availability', 'reference/strong_v6'), ('cached/linear', 'cached/availability'),
            ('cached/routing_prior', 'cached/routing_uniform'), ('drug_screen/name_semantic', 'drug_screen/name_random'),
            ('strong_anchor', 'reference/strong_v6'), ('strong_replication', 'reference/strong_v6'),
            ('residual_controls/fixed_projection', 'residual_controls/scaled_v7'),
            ('residual_followup/7/fixed_projection', 'residual_followup/7/scaled_v7'),
            ('residual_followup/101/fixed_projection', 'residual_followup/101/scaled_v7')]
    pair_labels = ('Lab availability - V6', 'Lab values - availability', 'Prior - uniform routing',
                   'Drug names - random', 'Full V7 correction s42 - V6', 'Full V7 correction s7 - V6',
                   'Projection - scaled s42', 'Projection - scaled s7', 'Projection - scaled s101')
    lookup = {(p['candidate'], p['comparator']): p for p in report['paired']}
    ax = axes[2]
    for i, key in enumerate(keys):
        p = lookup[key]; low, high = p['f1_95ci']
        ax.plot([low, high], [i, i], color='#24618e'); ax.plot(p['delta_f1'], i, 'o', color='#24618e')
    ax.axvline(0, color='#bb5839', ls='--'); ax.set_yticks(np.arange(len(keys)), pair_labels)
    ax.invert_yaxis(); ax.set_xlabel('Paired annual F1 difference'); ax.set_title('200 patient bootstrap: 95% intervals')
    ax.grid(axis='x', alpha=.2)
    fig.suptitle('V7 finite research queue: exposed tuning, no default replacement', fontsize=14)
    fig.text(.5, .02, 'Same frozen V6 seed 42. Reader-seed variation is not full-model replication. Intervals do not adjust for model selection.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95)); FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=170); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=('calculate', 'verify'), required=True)
    mode = parser.parse_args().mode
    result = calculate()
    if mode == 'calculate':
        if OUT.exists():
            raise FileExistsError('readout exists; preserve it')
        write_json(OUT, result); plot(result)
        print(json.dumps(dict(execution=result['execution'], seed_summary=result['seed_summary'])), flush=True)
    else:
        if result != json_read(OUT):
            raise ValueError('readout independent recomputation differs')
        write_json(ROOT/'readout_verification.json', dict(maximum_error=0, readout_sha256=sha256(OUT),
            source_sha256=sha256(Path(__file__)), figure_sha256=sha256(FIGURE)))
        print('readout independently recomputed: error 0', flush=True)
