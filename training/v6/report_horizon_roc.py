"""只读取已有预测，展示明确时点ROC/AUC；不重训、不改变选模口径。"""
import argparse
import json
from pathlib import Path
import numpy as np


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    report = json.loads((root / 'report.json').read_text())
    folder = root / 'horizon_reporting'
    if folder.exists(): raise FileExistsError(folder)
    folder.mkdir()
    labels = {'without_new': 'No new imaging', 'availability_only': 'Imaging availability only',
              'radbert': 'Frozen RadBERT', 'radgraph': 'RadGraph-XL facts',
              'history': 'History baseline', 'constant': 'History + constant correction',
              'clinical_no_text': 'History + current clinical state',
              'clinical_bert': 'History + current state + BERT'}
    arms = report['contract'].get('arms', list(report['models']))
    all_values = {}
    for fold in ('validation', 'test'):
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        all_values[fold] = {}
        for arm in arms:
            with np.load(root / arm / (fold + '_predictions.npz')) as a:
                times, events, curves = a['times'], a['events'], a['curve']
            all_values[fold][arm] = {}
            for ax, day in zip(axes.flat, (30, 90, 180, 365), strict=True):
                y = ((events == 1) & (times <= day)).astype(int)
                score = curves[:, day, 1]
                fpr, tpr, _ = roc_curve(y, score); auc = roc_auc_score(y, score)
                recorded = report['models'][arm]['folds'][fold]['readmission']['points'][str(day)]['auc']
                if abs(auc - recorded) > 1e-10: raise ValueError('horizon AUC disagrees with existing evaluation')
                ax.plot(fpr, tpr, label=f'{labels[arm]}: {auc:.4f}')
                all_values[fold][arm][str(day)] = dict(auc=auc, cases=int(y.sum()), controls=int((1-y).sum()))
        for ax, day in zip(axes.flat, (30, 90, 180, 365), strict=True):
            ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
            ax.set(xlabel='False positive rate', ylabel='True positive rate', title=f'{day}-day cumulative readmission ROC')
            ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.suptitle(f'{fold}: existing continuous-risk predictions; strict endpoint; World A')
        fig.tight_layout(); fig.savefig(folder / (fold + '_horizon_roc.png'), dpi=160); plt.close(fig)
    (folder / 'report.json').write_text(json.dumps(dict(status='existing_predictions_horizon_auc_reproduced',
        values=all_values, limits=['No model training or selection change',
        'Competing death is control for cumulative readmission incidence',
        'World A recorded outcomes, not verified complete external follow-up',
        'No per-horizon confidence intervals added here']), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    render(parser.parse_args().run)
