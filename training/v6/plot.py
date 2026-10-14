"""四臂实测图；英文标签，不将探索区间当显著性结论。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from training.v6.evaluation import metrics, calibration_bins, GRID

LABELS = {'history': 'History baseline', 'constant': 'History + constant calibration',
          'clinical_no_text': 'History + clinical (no text)', 'clinical_bert': 'History + clinical + BERT'}
COLORS = dict(zip(LABELS, ('#777777', '#9c6a36', '#2474b7', '#d95f02')))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def roc(scores, y):
    order = np.argsort(-scores, kind='stable')
    s, y = scores[order], y[order]
    endings = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    positives, negatives = y.sum(), (1-y).sum()
    if positives == 0 or negatives == 0:
        return np.array([0., 1.]), np.array([np.nan, np.nan])
    tpr = np.r_[0., np.cumsum(y)[endings] / positives]
    fpr = np.r_[0., np.cumsum(1-y)[endings] / negatives]
    return fpr, tpr


def render(root: Path):
    report = json.loads((root / 'report.json').read_text(encoding='utf-8'))
    figures = root / 'figures'; figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for arm, label in LABELS.items():
        history = json.loads((root / arm / 'history.json').read_text())
        epoch = [r['epoch'] for r in history]
        axes[0].plot(epoch, [r['validation_nll'] for r in history], color=COLORS[arm], label=label)
        axes[1].plot(epoch, [r['validation_integrated_brier'] for r in history], color=COLORS[arm])
        axes[2].plot(epoch, [r['validation_mean_auc'] for r in history], color=COLORS[arm])
        best = report['models'][arm]['best_epoch']
        for ax in axes:
            ax.axvline(best, color=COLORS[arm], alpha=.2, linestyle=':')
    for ax, title in zip(axes, ('Validation NLL', 'Validation integrated Brier (lower better)', 'Validation mean AUC(t)')):
        ax.set_title(title); ax.set_xlabel('Epoch (0 = initial)'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig, figures / 'training_validation.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for arm, label in LABELS.items():
        h = json.loads((root / arm / 'history.json').read_text())
        rows = [r for r in h if r['train_nll'] is not None]
        axes[0].plot([r['epoch'] for r in rows], [r['train_nll'] for r in rows], label=label, color=COLORS[arm])
        axes[1].plot([r['epoch'] for r in rows], [r['validation_nll'] for r in rows], color=COLORS[arm])
    axes[0].set_title('Training NLL'); axes[1].set_title('Validation NLL'); axes[0].legend(fontsize=8)
    for ax in axes: ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    save(fig, figures / 'training_losses.png')
    for fold in ('validation', 'test'):
        predictions = {}
        for arm in LABELS:
            with np.load(root / arm / (fold + '_predictions.npz')) as archive:
                predictions[arm] = {k: archive[k] for k in archive.files}
        reference = predictions['history']
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for cause in (1, 2):
            for arm, label in LABELS.items():
                p = predictions[arm]
                m = metrics(p['times'], p['events'], p['curve'][:, :, cause], cause=cause)
                axes[cause-1, 0].plot(GRID[7:], m['auc_curve'][7:], color=COLORS[arm], label=label)
                axes[cause-1, 1].plot(GRID[7:], m['brier_curve'][7:], color=COLORS[arm])
            axes[cause-1, 0].set_title(('Strict readmission' if cause == 1 else 'Death before readmission') + ': AUC(t)')
            axes[cause-1, 1].set_title('Brier(t) (lower better)')
        axes[0, 0].legend(fontsize=8)
        for ax in axes.flat: ax.set_xlabel('Days after discharge'); ax.grid(alpha=.2)
        fig.suptitle(f'{fold}: record-observed competing endpoints (World A)')
        save(fig, figures / (fold + '_auc_brier.png'))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, day in zip(axes, (30, 90, 365)):
            for arm, label in LABELS.items():
                p = predictions[arm]
                y = ((p['events'] == 1) & (p['times'] <= day)).astype(int)
                x, yy = roc(p['curve'][:, day, 1], y)
                auc = report['models'][arm]['folds'][fold]['readmission']['points'][str(day)]['auc']
                ax.plot(x, yy, label=f'{label} ({auc:.3f})' if auc is not None else label, color=COLORS[arm])
            ax.plot([0, 1], [0, 1], '--', color='gray'); ax.set_title(f'Time-dependent ROC: {day} days')
            ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); ax.legend(fontsize=7)
        save(fig, figures / (fold + '_roc.png'))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, day in zip(axes, (30, 90, 365)):
            for arm, label in LABELS.items():
                p = predictions[arm]
                bins = calibration_bins(p['times'], p['events'], p['curve'][:, :, 1], day=day)
                ax.plot([b['predicted'] for b in bins], [b['observed'] for b in bins], 'o-', label=label, color=COLORS[arm], ms=4)
            ax.plot([0, 1], [0, 1], '--', color='gray'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f'Calibration: {day} days'); ax.set_xlabel('Predicted risk'); ax.set_ylabel('Observed fraction (World A)')
        axes[0].legend(fontsize=7)
        save(fig, figures / (fold + '_calibration.png'))
        pairs = report['comparisons'][fold]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for i, (label, r) in enumerate(pairs.items()):
            for ax, point, bounds in zip(axes, ('integrated_brier_difference', 'auc90_difference'), ('integrated_brier_95ci', 'auc90_95ci')):
                if None not in r[bounds]:
                    ax.plot(r[bounds], [i, i], color='#2474b7', lw=2)
                ax.scatter(r[point], i, marker='D', color='#d95f02', zorder=3)
        for ax, title in zip(axes, ('Integrated Brier difference (lower better)', 'AUC(90) difference (higher better)')):
            ax.axvline(0, color='gray'); ax.set_title(title); ax.set_yticks(range(len(pairs)), [k.replace('_minus_', ' - ').replace('_', ' ') for k in pairs], fontsize=8)
            ax.set_xlabel('Candidate - comparator'); ax.grid(axis='x', alpha=.2)
        fig.suptitle(f'{fold}: paired patient bootstrap 95% intervals; one seed; no multiplicity adjustment')
        save(fig, figures / (fold + '_paired_differences.png'))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for cause, ax in zip((1, 2), axes):
            observed = np.mean((reference['events'][:, None] == cause) & (reference['times'][:, None] <= GRID), axis=0)
            ax.plot(GRID, observed, '--', color='black', label='Observed record fraction')
            for arm, label in LABELS.items():
                ax.plot(GRID, predictions[arm]['curve'][:, :, cause].mean(0), label=label, color=COLORS[arm])
            ax.set_title('Readmission' if cause == 1 else 'Death before readmission'); ax.set_xlabel('Days after discharge'); ax.set_ylabel('Mean cumulative incidence'); ax.legend(fontsize=7)
        save(fig, figures / (fold + '_cumulative_incidence.png'))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, day in zip(axes, (30, 90)):
            y = ((reference['events'] == 1) & (reference['times'] <= day)).astype(bool)
            thresholds = np.linspace(.02, .5, 80)
            for arm, label in LABELS.items():
                score = predictions[arm]['curve'][:, day, 1]
                benefit = [np.mean((score >= pt) & y) - np.mean((score >= pt) & ~y)*pt/(1-pt) for pt in thresholds]
                ax.plot(thresholds, benefit, color=COLORS[arm], label=label)
            ax.plot(thresholds, y.mean() - (1-y.mean())*thresholds/(1-thresholds), '--', color='gray', label='Follow all')
            ax.axhline(0, color='black', linestyle=':', label='Follow none'); ax.set_title(f'Exploratory follow-up DCA: {day} days')
            ax.set_xlabel('Risk threshold (assumed harm / benefit trade-off)'); ax.set_ylabel('Net benefit'); ax.legend(fontsize=7); ax.set_ylim(-.03, max(.03, y.mean()))
        fig.suptitle('Not evidence of intervention efficacy; no selected clinical threshold')
        save(fig, figures / (fold + '_exploratory_dca.png'))
        arms = list(LABELS)
        values = np.array([[report['models'][a]['folds'][fold]['history_strata'][s]['mean_auc_7_365'] or np.nan for s in ('prior_0', 'prior_1_2', 'prior_3_plus')] for a in arms])
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(values, vmin=.5, vmax=.85, cmap='YlGnBu')
        ax.set_xticks(range(3), ['Prior 0', 'Prior 1-2', 'Prior >=3']); ax.set_yticks(range(4), [LABELS[a] for a in arms]); ax.set_title(f'{fold}: mean AUC(t) within history strata')
        for i in range(4):
            for j in range(3): ax.text(j, i, f'{values[i,j]:.3f}', ha='center', va='center')
        fig.colorbar(im, ax=ax)
        save(fig, figures / (fold + '_history_strata.png'))
