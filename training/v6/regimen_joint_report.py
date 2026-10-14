"""已锁定联合融合配对的描述读出与图，不重新拟合。"""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path('training_runs/v6_regimen_joint_fusion_2026-10-14')


def build():
    from patient_data.export_mimiciv import sha256
    from training.v6.regimen_joint_experiment import setup
    from training.v6.evaluation import metrics, public_metrics
    from training.v6.train import write_json
    proof = json.loads((ROOT/'verification.json').read_text())
    if proof['maximum_error'] != 0 or proof['screening_sha256'] != sha256(ROOT/'screening.json'):
        raise ValueError('independent replay not current')
    data, _, tune, arrays, _ = setup(); d = data[0]
    counts = np.rint(np.expm1(d.history_raw[tune, 2])).astype(int)
    masks = dict(first=counts == 0, one_two=(counts >= 1) & (counts <= 2), three_plus=counts >= 3,
                 chapter_available=arrays['section_available'][tune], chapter_missing=~arrays['section_available'][tune])
    with np.load(ROOT/'original/tune_predictions.npz') as z:
        ref = z['curve']
    report = dict(screening_sha256=sha256(ROOT/'screening.json'), layers={}, selected_max_risk_difference={},
                  identical_predictions_no_bootstrap=True, no_refitting=True,
                  limits='WorldA retrospective descriptive tuning calibration; no individual accuracy or new blind test')
    curves = {'original': ref}
    for arm in ('availability', 'content'):
        with np.load(ROOT/arm/'tuning_predictions.npz') as z:
            curves[arm] = z['curve']
        delta = float(np.abs(curves[arm]-ref).max())
        report['selected_max_risk_difference'][arm] = delta
        if delta != 0:
            raise ValueError('do not assume identical predictions for a changed batch')
    for group, mask in masks.items():
        ix = tune[mask]
        report['layers'][group] = dict(admissions=int(mask.sum()), patients=int(len(np.unique(d.arrays['subject_ids'][ix]))),
            models={name: {label: public_metrics(metrics(d.times[ix], d.events[ix], value[mask, :, cause], cause=cause))
                           for label, cause in (('readmission', 1), ('death', 2))} for name, value in curves.items()})
    write_json(ROOT/'descriptive_report.json', report)
    print(json.dumps(dict(maximum_selected_difference=report['selected_max_risk_difference'],
                         layer_admissions={k: v['admissions'] for k, v in report['layers'].items()})), flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads((ROOT/'screening.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ref = report['reference']['metrics']['readmission']['integrated_brier_7_365']
    for arm in ('availability', 'content'):
        rows = json.loads((ROOT/arm/'history.json').read_text())
        axes[0].plot([x['epoch'] for x in rows[1:]], [x['train_nll'] for x in rows[1:]], label=arm)
        axes[1].plot([x['epoch'] for x in rows], [x['tuning_ibs'] for x in rows], label=arm)
    axes[1].axhline(ref, color='black', linestyle='--', label='Original V6 / selected epoch 0')
    for ax, title, ylabel in zip(axes, ('Actual training: all clinical branches learn', 'Specified tuning: no gain over epoch 0'),
                                ('Online training NLL (lower better)', 'Readmission IBS7-365 (lower better)'), strict=True):
        ax.set(title=title, xlabel='Epoch', ylabel=ylabel); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Seed 42; H34 frozen; pre-GELU regimen injection; 10-epoch early stopping')
    fig.tight_layout(); fig.savefig(ROOT/'training.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for i, cause in enumerate(('readmission', 'death')):
        for j, day in enumerate(('30', '90', '365')):
            bins = report['reference']['probability_readout'][cause][day]['bins']
            ax = axes[i, j]
            ax.plot([b['predicted'] for b in bins], [b['observed'] for b in bins], 'o-', label='All three selected curves coincide')
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set(title=f'{cause.title()}: day {day}', xlabel='Mean predicted cumulative probability',
                   ylabel='Observed World-A fraction', xlim=(0, 1), ylim=(0, 1)); ax.grid(alpha=.2)
    fig.suptitle('Descriptive risk-bin calibration; not individual correctness or external validation')
    axes[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(ROOT/'calibration.png', dpi=140); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', required=True, choices=('build', 'plot'))
    build() if parser.parse_args().mode == 'build' else plot()
