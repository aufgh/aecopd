"""既有共同调参预测的固定阈值描述；不加载模型、不训练或选择阈值。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('training_runs/v6_fixed_threshold_readout_2026-10-14')
JOINT = Path('training_runs/v6_regimen_joint_fusion_2026-10-14')
BASE = Path('training_runs/v6_baseline_content_diagnostic_2026-10-13')
CONTEXT = Path('patient_data/processed/v6_model_inputs_bert_2026-10-09/context.npz')
PROTOCOL = Path('docs/V6_FIXED_THRESHOLD_READOUT_PROTOCOL_2026-10-14.md')
DAYS = (30, 90, 365)
THRESHOLDS = (.1, .2, .3)
EDGES = (0., .1, .2, .3, .5, 1.)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def validate(a):
    n = len(a['times'])
    if (n == 0 or a['times'].shape != (n,) or a['events'].shape != (n,)
            or not np.isfinite(a['times']).all() or np.any((a['times'] < 0) | (a['times'] > 365))
            or not np.isin(a['events'], (0, 1, 2)).all()
            or np.any((a['events'] == 0) & (a['times'] != 365))
            or a['hadm_ids'].shape != (n,) or a['subject_ids'].shape != (n,)
            or len(np.unique(a['hadm_ids'])) != n):
        raise ValueError('invalid WorldA labels/alignment')
    c = a['curve']
    if (c.shape != (n, 366, 3) or not np.isfinite(c).all()
            or np.any((c < -1e-12) | (c > 1+1e-12))
            or np.max(np.abs(c.sum(2)-1)) > 1e-9
            or np.any(np.diff(c[:, :, 0], axis=1) > 1e-10)
            or np.any(np.diff(c[:, :, 1:], axis=1) < -1e-10)
            or np.max(np.abs(c[:, 0]-np.array([1., 0., 0.]))) > 1e-9):
        raise ValueError('invalid full daily joint probability')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in ('hadm_ids', 'subject_ids', 'times', 'events', 'curve')}
    validate(a)
    return a


def aligned(a, b, *, same_curve=False):
    for key in ('hadm_ids', 'subject_ids', 'times', 'events'):
        np.testing.assert_array_equal(a[key], b[key], err_msg='prediction alignment: '+key)
    if same_curve:
        np.testing.assert_array_equal(a['curve'], b['curve'])


def fraction(n, d):
    return float(n / d) if d else None


def threshold_stats(y, score, threshold, competing):
    y, score, competing = np.asarray(y), np.asarray(score), np.asarray(competing)
    if (y.dtype != bool or competing.dtype != bool or score.shape != y.shape or competing.shape != y.shape
            or y.ndim != 1 or not len(y) or not np.isfinite(score).all()
            or np.any((score < 0) | (score > 1)) or not 0 < threshold < 1 or np.any(y & competing)):
        raise ValueError('invalid threshold input')
    flagged = score >= threshold
    n = len(y)
    tp, fp = int((flagged & y).sum()), int((flagged & ~y).sum())
    fn, tn = int((~flagged & y).sum()), int((~flagged & ~y).sum())
    sensitivity, specificity = fraction(tp, tp+fn), fraction(tn, tn+fp)
    return dict(threshold=threshold, n=n, events=int(y.sum()), alerts=tp+fp,
                tp=tp, fp=fp, fn=fn, tn=tn, sensitivity=sensitivity, specificity=specificity,
                ppv=fraction(tp, tp+fp), npv=fraction(tn, tn+fn), accuracy=(tp+tn)/n,
                balanced_accuracy=(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
                alerts_per_100=100*(tp+fp)/n, true_alerts_per_100=100*tp/n,
                false_alerts_per_100=100*fp/n, missed_events_per_100=100*fn/n,
                competing_event_false_alerts=int((flagged & competing).sum()),
                net_benefit=tp/n-fp/n*threshold/(1-threshold),
                alert_all_net_benefit=float(y.mean()-(1-y.mean())*threshold/(1-threshold)),
                alert_none_net_benefit=0.)


def point(y, score, competing):
    groups = np.searchsorted(np.asarray(EDGES)[1:-1], score, side='right')
    bins = []
    for k in range(len(EDGES)-1):
        mask = groups == k
        if mask.any():
            bins.append(dict(lower=EDGES[k], upper=EDGES[k+1], n=int(mask.sum()),
                             events=int(y[mask].sum()), predicted=float(score[mask].mean()),
                             observed=float(y[mask].mean())))
    return dict(n=len(y), events=int(y.sum()), mean_predicted=float(score.mean()), observed=float(y.mean()),
                brier=float(np.mean((score-y)**2)), no_event_constant_brier=float(y.mean()),
                no_event_constant_accuracy=float(1-y.mean()), fixed_bins=bins,
                fixed_bin_weighted_absolute_gap=sum(b['n']*abs(b['predicted']-b['observed']) for b in bins)/len(y),
                thresholds=[threshold_stats(y, score, t, competing) for t in THRESHOLDS])


def summarize(a, counts):
    masks = dict(all=np.ones(len(counts), dtype=bool), prior_0=counts == 0,
                 prior_1_2=(counts >= 1) & (counts <= 2), prior_3_plus=counts >= 3)
    result = {}
    for group, mask in masks.items():
        entry = dict(admissions=int(mask.sum()), patients=len(np.unique(a['subject_ids'][mask])), outcomes={})
        if mask.any():
            for name, cause in (('readmission', 1), ('death_before_readmission', 2)):
                entry['outcomes'][name] = {}
                for day in DAYS:
                    y = (a['events'][mask] == cause) & (a['times'][mask] <= day)
                    competing = (a['events'][mask] == 3-cause) & (a['times'][mask] <= day)
                    entry['outcomes'][name][str(day)] = point(y, a['curve'][mask, day, cause], competing)
        result[group] = entry
    return result


def prepare():
    jp, bp = read(JOINT/'verification.json'), read(BASE/'verification.json')
    if (jp['maximum_error'] != 0 or not jp['no_refitting']
            or jp['screening_sha256'] != sha(JOINT/'screening.json')
            or bp['maximum_error'] != 0 or bp['report_sha256'] != sha(BASE/'report.json')):
        raise ValueError('parent independent proof stale')
    parent = read(JOINT/'contract.json')
    for path, expected in parent['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('protected source changed: '+path)
    source_paths = [JOINT/'original/tune_predictions.npz', BASE/'seed42/original_V6/tuning_predictions.npz',
                    BASE/'seed42/H34/tuning_predictions.npz', CONTEXT, JOINT/'contract.json',
                    JOINT/'verification.json', JOINT/'screening.json', BASE/'verification.json', BASE/'report.json',
                    PROTOCOL, Path(__file__), Path('training/tests/test_v6_fixed_threshold_readout.py')]
    teacher_path = Path('training_runs/v6_module_surgery_2026-10-12_r3/seed42/control/best.pt')
    if sha(teacher_path) != parent['teacher_sha256']:
        raise ValueError('teacher changed')
    source_paths.append(teacher_path)
    old_report = read(BASE/'report.json')
    expected_teacher = old_report['models']['42']['teacher_checkpoint_sha256']
    if expected_teacher != parent['teacher_sha256']:
        raise ValueError('different original teacher')
    v6 = load(source_paths[0]); old = load(source_paths[1]); h34 = load(source_paths[2])
    aligned(v6, old, same_curve=True); aligned(v6, h34)
    if len(v6['times']) != 1531:
        raise ValueError('not specified tuning fold')
    context_contract = CONTEXT.parent/'input_contract.json'
    cc = read(context_contract)
    if cc['context_sha256'] != sha(CONTEXT):
        raise ValueError('original context checksum differs')
    for key in ('state_manifest_sha256', 'label_sha256', 'raw_context_hashes'):
        if cc['provenance'][key] != parent['data_provenance'][key]:
            raise ValueError('original history provenance differs')
    source_paths.append(context_contract)
    with np.load(CONTEXT, allow_pickle=False) as z:
        ids, history = z['hadm_ids'], z['history_raw']
    lookup = {int(h): i for i, h in enumerate(ids)}
    if len(lookup) != len(ids) or len(ids) != 19401:
        raise ValueError('invalid original context')
    ix = np.array([lookup[int(h)] for h in v6['hadm_ids']])
    count_raw = np.expm1(history[ix, 2]); counts = np.rint(count_raw).astype(int)
    if np.any(counts < 0) or np.max(np.abs(count_raw-counts)) > 1e-6:
        raise ValueError('invalid raw history count')
    # 旧描述已独立核验同原数据；逐层病例数/概率/Brier再核，防止错用旧context。
    prior = read(JOINT/'descriptive_report.json')['layers']
    result = {'V6': summarize(v6, counts), 'H34': summarize(h34, counts)}
    for name, legacy in (('prior_0', 'first'), ('prior_1_2', 'one_two'), ('prior_3_plus', 'three_plus')):
        r = result['V6'][name]
        if r['admissions'] != prior[legacy]['admissions'] or r['patients'] != prior[legacy]['patients']:
            raise ValueError('history context differs')
        for outcome, old_name in (('readmission', 'readmission'), ('death_before_readmission', 'death')):
            for day in DAYS:
                old_point = prior[legacy]['models']['original'][old_name]['points'][str(day)]
                p = r['outcomes'][outcome][str(day)]
                if abs(p['brier']-old_point['brier']) > 1e-12 or p['events'] != old_point['cases']:
                    raise ValueError('history/context label replay differs')
    source_paths.append(JOINT/'descriptive_report.json')
    contract = dict(source_sha256={str(p): sha(p) for p in source_paths},
                    parent_code_data_sources=parent['source_sha256'], label_sha256=parent['label_sha256'],
                    teacher_sha256=parent['teacher_sha256'], base_sha256=parent['base_sha256'],
                    data_provenance=parent['data_provenance'], days=list(DAYS), thresholds=list(THRESHOLDS),
                    fixed_bin_edges=list(EDGES), training=False, calibration_refit=False,
                    confirmation_test_used=False, default_changed=False)
    report = dict(models=result, matched_admissions=1531, patients=len(np.unique(v6['subject_ids'])),
                  seed=42, trained_models=0, threshold_selection=False, goal_achieved=False,
                  limits=['Previously exposed tuning fold; not external/blind validation.',
                          'WorldA observed in-system events; unknown out-of-system events.',
                          'Retrospective FINAL availability; not real-time deployment.',
                          'Repeated discharges; no confidence intervals or significance claims.',
                          'Net benefit assumes action costs; not intervention efficacy.',
                          'Death is competing before readmission, not all-cause mortality.'])
    return contract, report


def run(mode):
    contract, report = prepare()
    if mode == 'build':
        ROOT.mkdir(exist_ok=False)
        write(ROOT/'contract.json', contract); write(ROOT/'report.json', report)
        write(ROOT/'state.json', dict(status='built_pending_independent_replay', goal_achieved=False))
    else:
        if contract != read(ROOT/'contract.json') or report != read(ROOT/'report.json'):
            raise ValueError('independent readout replay differs')
        write(ROOT/'verification.json', dict(status='independent_saved_prediction_readout_replay', maximum_error=0.,
                                             report_sha256=sha(ROOT/'report.json'), no_training=True))
        write(ROOT/'state.json', dict(status='independently_verified_readout_only', goal_achieved=False))
    print(json.dumps(dict(mode=mode, matched_admissions=report['matched_admissions'], trained_models=0)), flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = read(ROOT/'report.json'); proof = read(ROOT/'verification.json')
    if proof['report_sha256'] != sha(ROOT/'report.json'):
        raise ValueError('stale readout')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for name in ('H34', 'V6'):
        rows = r['models'][name]['all']['outcomes']['readmission']['90']['thresholds']
        for ax, key in zip(axes, ('alerts_per_100', 'sensitivity', 'ppv'), strict=True):
            values = [v[key] if key == 'alerts_per_100' else 100*v[key] if v[key] is not None else np.nan for v in rows]
            ax.plot([10, 20, 30], values, 'o-', label=name)
    for ax, title in zip(axes, ('Alerts per 100 discharges', 'Detected share of events (%)', 'Events among alerts (%)'), strict=True):
        ax.set(title=title, xlabel='Fixed 90-day risk threshold (%)', xticks=[10, 20, 30], ylim=(0, 100))
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Same 1,531 tuning discharges; World A; descriptive, not a clinical threshold recommendation')
    fig.tight_layout(); fig.savefig(ROOT/'threshold_readout.png', dpi=140); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', required=True, choices=('build', 'verify', 'plot'))
    mode = p.parse_args().mode
    plot() if mode == 'plot' else run(mode)
