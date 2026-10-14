"""锁定checkpoint的多时间点读出；阈值探索不重新选择模型或训练。"""
import argparse
from pathlib import Path
import numpy as np
from training.v6.fixed_threshold_readout import load, aligned, point, read, write, sha
from training.v6.unified_longitudinal import best_f1, f1_at
from training.v6.evaluation import metrics, public_metrics, REPORT_DAYS

PARENT = Path('training_runs/v6_unified_longitudinal_f1_2026-10-14_r2')
ROOT = Path('training_runs/v6_multihorizon_readout_2026-10-14')
PROTOCOL = Path('docs/V6_MULTIHORIZON_READOUT_PROTOCOL_2026-10-14.md')
TEST = Path('training/tests/test_v6_multihorizon_readout.py')
NAMES = ('original', 'current_original', 'current_unified', 'history_original', 'history_unified')
WINDOWS = ((7, 30), (31, 90), (91, 365))


def decision(y, score, threshold):
    result = f1_at(y, score, threshold)
    result['accuracy'] = (result['tp'] + result['tn']) / len(y)
    result['alerts_per_100'] = 100 * result['alerts'] / len(y)
    return result


def horizon_readout(a, day, cause, annual_threshold):
    if day not in REPORT_DAYS or cause not in (1, 2):
        raise ValueError('unsupported horizon/cause')
    y = (a['events'] == cause) & (a['times'] <= day)
    competing = (a['events'] == 3-cause) & (a['times'] <= day)
    score = a['curve'][:, day, cause]
    optimal = best_f1(y, score)
    result = point(y, score, competing)
    result['tuning_best_f1'] = decision(y, score, optimal['threshold'])
    result['annual_threshold_readout'] = decision(y, score, annual_threshold)
    for row in result['thresholds']:
        row['f1'] = f1_at(y, score, row['threshold'])['f1']
    return result


def segmented_auc(curve):
    result = {}
    for start, end in WINDOWS:
        values = curve[start:end+1]
        good = np.isfinite(values)
        result[f'{start}_{end}'] = dict(valid_days=int(good.sum()),
            mean_auc=float(values[good].mean()) if good.any() else None)
    return result


def prepare():
    parent = read(PARENT/'report.json')
    proof = read(PARENT/'verification.json')
    if proof['maximum_error'] != 0 or proof['report_sha256'] != sha(PARENT/'report.json'):
        raise ValueError('parent proof stale')
    paths = [PARENT/'report.json', PARENT/'verification.json', PARENT/'contract.json',
             PARENT/'nomination_lock.json', Path(__file__), TEST, PROTOCOL,
             Path('training/v6/evaluation.py'), Path('training/v6/fixed_threshold_readout.py'),
             Path('training/v6/unified_longitudinal.py')]
    arrays = {}; reference = None
    for name in NAMES:
        p = PARENT/name/'tuning_predictions.npz'
        a = load(p); paths.append(p)
        if reference is None: reference = a
        aligned(reference, a)
        if name != 'original':
            checkpoint = PARENT/name/'best.pt'
            if sha(checkpoint) != parent['results'][name]['checkpoint_sha256']:
                raise ValueError('parent checkpoint changed')
            paths.append(checkpoint)
        arrays[name] = a
    if len(reference['times']) != 1531:
        raise ValueError('not specified tuning fold')
    contract = dict(source_sha256={str(p):sha(p) for p in paths},
        parent_labels_sha256=read(PARENT/'contract.json')['labels_sha256'],
        horizon_days=list(REPORT_DAYS), models=list(NAMES), new_fits=0,
        selected_checkpoints_unchanged=True, tuning_thresholds_not_independent_evaluation=True,
        confirmation_or_test_read=False, automation_resumed=False)
    return parent, arrays, contract


def calculate():
    parent, arrays, contract = prepare(); results = {}
    for name, a in arrays.items():
        expected = parent['reference'] if name == 'original' else parent['results'][name]['metrics']
        result = dict(admissions=len(a['times']), patients=len(np.unique(a['subject_ids'])), outcomes={})
        for label, cause in (('readmission',1), ('death',2)):
            m = metrics(a['times'], a['events'], a['curve'][:,:,cause], cause=cause, ordering=True)
            # 既有保存模型全部每日指标逐项复现，不重算/替换其原选点。
            if public_metrics(m) != expected[label]:
                raise ValueError('parent daily metrics differ: '+name+'/'+label)
            annual_y = (a['events'] == cause) & (a['times'] <= 365)
            annual = (expected['annual'] if cause == 1 else
                      best_f1(annual_y, a['curve'][:,365,cause]))
            points = {str(day):horizon_readout(a,day,cause,annual['threshold']) for day in REPORT_DAYS}
            if cause == 1:
                actual = f1_at(annual_y, a['curve'][:,365,1], annual['threshold'])
                if actual != expected['annual']:
                    raise ValueError('parent annual decision differs')
            result['outcomes'][label] = dict(daily_metrics=public_metrics(m),
                segmented_auc=segmented_auc(m['auc_curve']), annual_threshold=annual['threshold'], points=points)
        results[name] = result
    return contract, dict(models=results, new_fits=0, no_model_reselection=True,
        fold='same exposed tuning fold', threshold_policy='tuning-best per horizon + annual transfer + common fixed',
        calibration_bins='fixed 0/.1/.2/.3/.5/1; descriptive, not individual truth')


def build():
    if ROOT.exists(): raise FileExistsError('readout exists; use verify, do not rebuild')
    contract, report = calculate()
    ROOT.mkdir(); write(ROOT/'contract.json',contract); write(ROOT/'report.json',report)
    print({n:{d:x['tuning_best_f1']['f1'] for d,x in r['outcomes']['readmission']['points'].items()}
           for n,r in report['models'].items()},flush=True)


def verify():
    contract, report = calculate()
    if contract != read(ROOT/'contract.json') or report != read(ROOT/'report.json'):
        raise ValueError('independent readout differs')
    write(ROOT/'verification.json',dict(maximum_error=0,independent_new_process=True,
        no_refitting=True,report_sha256=sha(ROOT/'report.json'),contract_sha256=sha(ROOT/'contract.json')))
    print('all five multi-horizon readouts replay exactly; no refitting',flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    proof = read(ROOT/'verification.json')
    if proof['report_sha256'] != sha(ROOT/'report.json'): raise ValueError('readout proof stale')
    report = read(ROOT/'report.json'); days=np.array(REPORT_DAYS)
    labels=('Retained V6','Current / original','Current / unified','History / original','History / unified')
    fig, axes=plt.subplots(2,2,figsize=(12,8))
    for name,label in zip(NAMES,labels,strict=True):
        r=report['models'][name]['outcomes']['readmission']; p=r['points'];dm=r['daily_metrics']
        axes[0,0].plot(days,[dm['points'][str(d)]['auc'] for d in days],marker='o',label=label)
        axes[0,1].plot(days,[p[str(d)]['tuning_best_f1']['f1'] for d in days],marker='o',label=label)
        axes[1,0].plot(days,[p[str(d)]['brier'] for d in days],marker='o',label=label)
        axes[1,1].plot(days,[p[str(d)]['tuning_best_f1']['threshold'] for d in days],marker='o',label=label)
    titles=('AUC(t), threshold-independent','F1 at separately tuning-selected thresholds',
            'Brier(t), lower is better','Tuning-selected threshold at each horizon')
    for ax,title in zip(axes.flat,titles,strict=True):
        ax.set(title=title,xlabel='Days since discharge');ax.set_xscale('log');ax.set_xticks(days)
        ax.set_xticklabels(days);ax.grid(alpha=.2)
    axes[0,0].set_ylim(.5,1);axes[0,1].set_ylim(0,1);axes[1,1].set_ylim(0,1)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Same 1,531 exposed tuning discharges; annual-selected checkpoints locked; zero new fits')
    fig.tight_layout();fig.savefig(ROOT/'multihorizon.png',dpi=140);plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=('build','verify','plot'))
    {'build':build,'verify':verify,'plot':plot}[parser.parse_args().mode]()
