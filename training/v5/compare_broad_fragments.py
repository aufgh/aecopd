"""同患者/严格结局配对指标与英文科研图；单种子，探索性区间。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from training.bonsai_broad_fragments import atomic_json
from training.baselines.rank_metrics import auc_rank
from training.v5.compare_screened_group_ablations import _patient_draws, _predictions, _summary_for_cause
from training.v5.train import GRID, metrics

ARMS = ('no_text', 'old_fields_bert', 'fragments_bert', 'fragments_qwen')
LABELS = ('No text', 'Old four fields: BERT', 'Clinical fragments: BERT', 'Clinical fragments: Qwen')
PAIRS = (('fragments_bert', 'no_text'), ('fragments_qwen', 'no_text'),
         ('fragments_bert', 'old_fields_bert'), ('fragments_qwen', 'fragments_bert'))


def interval(times, events, subjects, candidate, reference, cause, *, draws=100):
    labels = ((events[:, None] == cause) & (times[:, None] <= GRID[None, :])).astype(float)
    error = np.trapezoid((candidate[:, 7:, cause]-labels[:, 7:])**2-
                         (reference[:, 7:, cause]-labels[:, 7:])**2, GRID[7:], axis=1)/358
    sampled = _patient_draws(subjects, draws, 2126)
    ibs = np.array([error[ix].mean() for ix in sampled])
    auc = []
    if min(int(labels[:, 90].sum()), len(times)-int(labels[:, 90].sum())) >= 20:
        for ix in sampled:
            y = labels[ix, 90].astype(int)
            if min(int(y.sum()), len(y)-int(y.sum())) >= 20:
                auc.append(auc_rank(candidate[ix, 90, cause], y)-auc_rank(reference[ix, 90, cause], y))
    return {'bootstrap_patient_cluster_draws': draws, 'delta_integrated_brier_95ci': np.quantile(ibs, (.025, .975)).tolist(),
        'delta_auc90_95ci': np.quantile(auc, (.025, .975)).tolist() if len(auc) >= draws//2 else None,
        'valid_auc90_draws': len(auc), 'interval_scope': 'patient sampling for one trained seed; not training randomness'}


def compare(root: Path) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = root/'paired_analysis'
    output.mkdir(parents=True, exist_ok=False)
    reports = {arm: json.loads((root/arm/'report.json').read_text(encoding='utf-8')) for arm in ARMS}
    for arm, report in reports.items():
        if report['status'] != 'training_completed':
            raise ValueError('Incomplete comparison arm')
        for key in ('member_sha256', 'label_sha256', 'primary_event', 'seed', 'shared_initialization_sha256',
                    'split_counts', 'medication_effect_mode', 'clinical_recovery_axes', 'field_roles',
                    'max_epochs', 'learning_rate', 'weight_decay', 'patient_batch_size'):
            if report[key] != reports['no_text'][key]:
                raise ValueError('Pairing contract differs: '+key)
    records, detailed, absolute = [], {}, {}
    for split in ('validation', 'historical_test'):
        predictions = {arm: _predictions(root/arm, split) for arm in ARMS}
        full = predictions['no_text']
        for arm, prediction in predictions.items():
            for key in ('subject_ids', 'hadm_ids', 'times', 'events', 'note_member'):
                if not np.array_equal(full[key], prediction[key]):
                    raise ValueError('Prediction pairing differs: '+key)
        absolute[split] = {arm: {name: _summary_for_cause(p['times'], p['events'], p['competing_cif'], cause)
                          for cause, name in ((1, 'readmission'), (2, 'death'))} for arm, p in predictions.items()}
        detailed[split] = []
        for added, reference in PAIRS:
            for cause, name in ((1, 'readmission'), (2, 'death')):
                a, b = absolute[split][added][name], absolute[split][reference][name]
                row = {'split': split, 'added': added, 'reference': reference, 'cause': name,
                    'delta_ibs': a['integrated_brier_7_365']-b['integrated_brier_7_365'],
                    'delta_mean_auc': a['mean_auc_7_365']-b['mean_auc_7_365'],
                    'delta_c_index365': a['c_index_365']-b['c_index_365'],
                    'delta_population_curve_gap': a['mean_absolute_population_curve_gap_7_365']-b['mean_absolute_population_curve_gap_7_365'],
                    'delta_calibration_gap90': a['points']['90']['absolute_mean_calibration_gap']-b['points']['90']['absolute_mean_calibration_gap']}
                records.append(row)
                detailed[split].append({**row, **interval(full['times'], full['events'], full['subject_ids'],
                    predictions[added]['competing_cif'], predictions[reference]['competing_cif'], cause)})
        figure, axes = plt.subplots(2, 2, figsize=(13, 9))
        for arm, label in zip(ARMS, LABELS, strict=True):
            p = predictions[arm]
            score = metrics(p['times'], p['events'], p['risk'])
            axes[0, 0].plot(GRID[7:], score['auc_curve'][7:], label=label)
            axes[0, 1].plot(GRID[7:], score['brier_curve'][7:], label=label)
            history = json.loads((root/arm/'history.json').read_text(encoding='utf-8'))
            axes[1, 0].plot([r['epoch'] for r in history], [r['validation_integrated_brier'] for r in history], label=label)
            axes[1, 1].plot(GRID, p['risk'].mean(axis=0), label=label)
        observed = ((full['events'][:, None] == 1) & (full['times'][:, None] <= GRID[None, :])).mean(axis=0)
        axes[1, 1].plot(GRID, observed, 'k--', label='Observed cumulative fraction (World A)')
        axes[0, 0].set(xlabel='Days after discharge', ylabel='AUC(t)', ylim=(0, 1), title='Strict readmission discrimination')
        axes[0, 1].set(xlabel='Days after discharge', ylabel='Brier(t)', title='Probability error (lower is better)')
        axes[1, 0].set(xlabel='Epoch', ylabel='Validation integrated Brier', title='Training checkpoint selection')
        axes[1, 1].set(xlabel='Days after discharge', ylabel='Cumulative probability', title='Population risk curves')
        for axis in axes.flat:
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
        figure.suptitle(f'{split}: matched patient histories; one seed; exploratory', fontsize=13)
        figure.tight_layout()
        figure.savefig(output/f'{split}_overview.png', dpi=160)
        plt.close(figure)
        readmission = [r for r in detailed[split] if r['cause'] == 'readmission']
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        labels = [f"{r['added']} - {r['reference']}" for r in readmission]
        for i, row in enumerate(readmission):
            lo, hi = row['delta_integrated_brier_95ci']
            axes[0].plot([lo, hi], [i, i], color='tab:blue', lw=2)
            axes[0].plot(row['delta_ibs'], i, 'D', color='tab:blue')
            axes[1].plot(row['delta_mean_auc'], i, 'D', color='tab:orange')
        axes[0].set_yticks(range(len(labels)), labels)
        axes[0].invert_yaxis()
        axes[0].set(xlabel='Added - reference', title='Integrated Brier difference (negative is better)')
        axes[1].set(xlabel='Added - reference', title='Mean AUC(t) difference (positive is better)')
        for axis in axes:
            axis.axvline(0, color='grey')
            axis.grid(alpha=.2)
        figure.suptitle(f'{split}: one seed; Brier CI = 100 patient-cluster draws; AUC = point estimate')
        figure.tight_layout()
        figure.savefig(output/f'{split}_paired_differences.png', dpi=160)
        plt.close(figure)
    with (output/'summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = {'status': 'paired_analysis_completed', 'absolute': absolute, 'paired': detailed,
        'primary_decision': 'jointly review readmission/death IBS, supported AUC(t), C-index and calibration; no automatic winner',
        'bootstrap_draws': 100, 'no_multiplicity_adjustment': True, 'single_seed': True,
        'not_comparable_to_old_full_cohort_auc': True}
    atomic_json(output/'paired_results.json', result)
    return result
