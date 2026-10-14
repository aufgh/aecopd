"""只读完成报告，绘制有患者聚类区间的影像配对结果；不训练或选模。"""
import argparse
import json
from pathlib import Path


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    report = json.loads((root / 'report.json').read_text())
    if report['status'] != 'completed_full_radiology_paired':
        raise ValueError('completed comparison required')
    names = ('availability_only', 'radbert', 'radgraph')
    labels = ('Imaging availability only', 'Frozen RadBERT: CXR + CT', 'RadGraph-XL facts: CXR + CT')
    for fold in ('validation', 'test'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
        for i, name in enumerate(names):
            result = report['comparisons'][fold][name + '_minus_without_new']
            for ax, key, ci in zip(axes, ('integrated_brier_difference', 'auc90_difference'),
                                  ('integrated_brier_95ci', 'auc90_95ci'), strict=True):
                lo, hi = result[ci]
                ax.plot([lo, hi], [i, i], color='C' + str(i), linewidth=2)
                ax.scatter(result[key], i, marker='D', color='C' + str(i), zorder=3)
        for ax in axes:
            ax.axvline(0, color='gray', linewidth=1)
            ax.set_yticks(range(3), labels)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_powerlimits((-3, 3))
            ax.xaxis.set_major_formatter(formatter)
            ax.grid(alpha=.2)
            ax.set_xlabel('Candidate - no new imaging; patient bootstrap 95% CI')
            ax.set_ylim(-.5, 2.5)
        axes[0].set_title('Integrated Brier, days 7-365 (lower better)')
        axes[1].set_title('90-day AUC(t) (higher better)')
        fig.suptitle(f'{fold}: one training seed; 200 patient-cluster resamples; no multiplicity adjustment')
        fig.tight_layout()
        fig.savefig(root / 'figures' / (fold + '_paired_ci.png'), dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    render(parser.parse_args().output)
