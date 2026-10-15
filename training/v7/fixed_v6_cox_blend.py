"""一次固定比例概率融合；不训练成员、不学习或扫描权重。"""
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.evaluation import metrics, public_metrics
from training.v6.unified_longitudinal import best_f1, f1_at
from training.v7.sequential_readout import outcome_statistics, paired

ROOT = Path('training_runs/v7_fixed_v6_cox_blend_2026-10-15')
PARENT = Path('training_runs/v7_sequential_research_2026-10-15')
ANCHOR = Path('training_runs/v6_symmetric_context_2026-10-14/symmetric/tuning_predictions.npz')
PROTOCOL = Path('docs/V7_FIXED_V6_COX_BLEND_PROTOCOL_2026-10-15.md')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def calculate():
    with np.load(ANCHOR) as z:
        anchor = {k: z[k] for k in ('hadm_ids', 'subject_ids', 'times', 'events', 'curve')}
    with np.load(PARENT/'cached/cache.npz') as z:
        tune = z['tune']
        for key in ('hadm_ids', 'subject_ids', 'times', 'events'):
            np.testing.assert_array_equal(anchor[key], z[key][tune])
    with np.load(PARENT/'full_refit/predictions.npz') as z:
        np.testing.assert_array_equal(z['tune'], tune)
        cox = z['cox']
    full = PARENT/'full_refit'
    proof = read(full/'verification.json')
    if proof['maximum_error'] != 0 or proof['report_sha256'] != sha256(full/'report.json'):
        raise ValueError('Cox full-refit proof stale')
    reference_metrics = read(Path('training_runs/v7_first_2026-10-15_r2/report.json'))['reference_metrics']['v6_symmetric']
    fixed_threshold = reference_metrics['annual']['threshold']
    curves = dict(full_v6=anchor['curve'], cox=cox, fixed_blend=.9*anchor['curve']+.1*cox)
    label = (anchor['events'] == 1) & (anchor['times'] <= 365)
    results = {}; values = {}
    for name, curve in curves.items():
        if (not np.isfinite(curve).all() or (curve < 0).any() or (curve > 1).any()
                or np.any(np.diff(curve[:, :, 1:], axis=1) < -1e-10)):
            raise ValueError('invalid mixture curve')
        np.testing.assert_allclose(curve.sum(2), 1., rtol=0, atol=1e-9)
        annual = best_f1(label, curve[:, 365, 1])
        _, values[name] = outcome_statistics(curve, anchor['times'], anchor['events'], annual['threshold'])
        results[name] = dict(annual=annual, fixed_anchor_threshold=f1_at(label, curve[:, 365, 1], fixed_threshold),
            readmission=public_metrics(metrics(anchor['times'], anchor['events'], curve[:, :, 1], cause=1, ordering=True)),
            death=public_metrics(metrics(anchor['times'], anchor['events'], curve[:, :, 2], cause=2, ordering=True)))
    if results['full_v6']['annual'] != reference_metrics['annual']:
        raise ValueError('anchor classification differs')
    for cause in ('readmission', 'death'):
        if results['full_v6'][cause] != reference_metrics[cause]:
            raise ValueError('anchor daily metrics differ')
    people, inverse = np.unique(anchor['subject_ids'], return_inverse=True)
    rng = np.random.default_rng(42)
    weights = np.array([np.bincount(rng.integers(len(people), size=len(people)), minlength=len(people))[inverse]
                        for _ in range(200)], np.float64)
    targets = (anchor['events'][:, None] == 2) & (anchor['times'][:, None] <= np.arange(366))
    death_difference = np.trapezoid(((curves['fixed_blend'][:, :, 2]-targets)**2
        -(curves['full_v6'][:, :, 2]-targets)**2)[:, 7:], dx=1., axis=1)/358
    a, b = results['fixed_blend'], results['full_v6']
    comparison = dict(delta_f1=a['annual']['f1']-b['annual']['f1'],
        delta_ibs=a['readmission']['integrated_brier_7_365']-b['readmission']['integrated_brier_7_365'],
        delta_death_ibs=float(death_difference.mean()),
        **paired(values['fixed_blend'], values['full_v6'], weights),
        death_ibs_95ci=np.quantile((weights@death_difference)/weights.sum(1), [.025, .975]).tolist())
    paths = (Path(__file__), PROTOCOL, ANCHOR, PARENT/'cached/cache.npz', full/'predictions.npz',
             full/'report.json', full/'verification.json', Path('training/v7/sequential_readout.py'),
             Path('training/v6/evaluation.py'), Path('training/v6/unified_longitudinal.py'))
    contract = dict(source_sha256={str(p): sha256(p) for p in paths}, weights=dict(full_v6=.9, cox=.1),
        discharges=len(tune), patients=len(people), new_model_fits=0, weight_search=False,
        threshold='tuning best F1 and fixed old anchor threshold both reported',
        not_oof_weight_fitting=True, confirmation_test_evaluated=False)
    report = dict(models=results, fixed_blend_minus_full_v6=comparison,
        limits='one prespecified ratio; exposed tune; fixed-model patient bootstrap 200; no multiplicity adjustment',
        default_replaced=False)
    return contract, report, anchor, curves['fixed_blend']


def main():
    contract, report, anchor, blend = calculate()
    if ROOT.exists():
        if read(ROOT/'contract.json') != contract or read(ROOT/'report.json') != report:
            raise ValueError('independent fixed-blend recalculation differs')
        with np.load(ROOT/'predictions.npz') as z:
            np.testing.assert_array_equal(z['curve'], blend)
            for key in ('hadm_ids', 'subject_ids', 'times', 'events'):
                np.testing.assert_array_equal(z[key], anchor[key])
        (ROOT/'verification.json').write_text(json.dumps(dict(maximum_error=0,
            report_sha256=sha256(ROOT/'report.json'), contract_sha256=sha256(ROOT/'contract.json'),
            predictions_sha256=sha256(ROOT/'predictions.npz')), indent=2)+'\n', encoding='utf-8')
    else:
        ROOT.mkdir()
        for name, value in (('contract', contract), ('report', report)):
            (ROOT/f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        np.savez_compressed(ROOT/'predictions.npz', **{k: v for k, v in anchor.items() if k != 'curve'}, curve=blend)
    print(json.dumps(dict(comparison=report['fixed_blend_minus_full_v6'],
        models={name: dict(f1=v['annual']['f1'], fixed_threshold_f1=v['fixed_anchor_threshold']['f1'],
            ibs=v['readmission']['integrated_brier_7_365'], death_ibs=v['death']['integrated_brier_7_365'],
            mean_auc=v['readmission']['mean_auc_7_365']) for name,v in report['models'].items()}), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
