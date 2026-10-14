"""只读已有日志，比较同种训练目标的验证轨迹；不加载患者资料或训练模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path) -> tuple[dict, dict]:
    report = json.loads((root / 'report.json').read_text(encoding='utf-8'))
    contract = report['contract']
    summaries, histories, hashes = [], {}, {}
    for seed in contract['seeds']:
        for arm in contract['arms']:
            path = root / f'seed{seed}' / arm / 'history.json'
            rows = json.loads(path.read_text(encoding='utf-8'))
            if [r['epoch'] for r in rows] != list(range(len(rows))):
                raise ValueError(f'non-contiguous epoch: {path}')
            for row in rows:
                for key in ('validation_objective', 'validation_brier90'):
                    if not math.isfinite(row[key]):
                        raise ValueError(f'non-finite {key}: {path}')
                if row['epoch'] and not math.isfinite(row['train_objective']):
                    raise ValueError(f'non-finite train objective: {path}')
            epoch = report['models'][str(seed)][arm]['best_epoch']
            chosen, last = rows[epoch], rows[-1]
            if report['models'][str(seed)][arm]['epochs_trained'] != last['epoch']:
                raise ValueError(f'training length differs: {path}')
            key = f'seed{seed}/{arm}'
            histories[key] = rows
            hashes[str(path.relative_to(root))] = digest(path)
            summaries.append(dict(seed=seed, arm=arm, best_epoch=epoch,
                epochs_trained=last['epoch'],
                train_nll_at_best=chosen['train_objective'], train_nll_final=last['train_objective'],
                validation_nll_at_best=chosen['validation_objective'], validation_nll_final=last['validation_objective'],
                validation_brier90_at_best=chosen['validation_brier90'], validation_brier90_final=last['validation_brier90'],
                later_validation_nll_change=last['validation_objective']-chosen['validation_objective'],
                later_validation_brier90_change=last['validation_brier90']-chosen['validation_brier90']))
    return dict(source=str(root), source_report_sha256=digest(root/'report.json'),
        history_sha256=hashes, selection=contract['selection'], seeds=contract['seeds'], arms=contract['arms'],
        status='read_only_saved_history_audit_no_new_training', rows=summaries,
        limitations=['Training NLL is online batch mean; validation NLL is evaluated at epoch end.',
                     'No gradient/update norms were recorded; instability cause is not uniquely identified.',
                     'A later validation loss increase does not prove every new feature is harmful.']), histories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result, histories = audit(args.source)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(result['seeds']), 2, figsize=(13, 8), squeeze=False)
    colors = dict(control='tab:blue', new_labs='tab:orange', med_details='tab:green',
                  both='tab:red', availability='tab:purple')
    for row, seed in enumerate(result['seeds']):
        for col, metric in enumerate(('validation_objective', 'validation_brier90')):
            ax = axes[row, col]
            for arm in result['arms']:
                history = histories[f'seed{seed}/{arm}']
                ax.plot([x['epoch'] for x in history], [x[metric] for x in history],
                        label=arm, color=colors.get(arm))
                selected = next(x for x in result['rows'] if x['seed']==seed and x['arm']==arm)
                epoch = selected['best_epoch']
                ax.scatter(epoch, history[epoch][metric], marker='D', s=22, color=colors.get(arm))
            ax.set_title(f'Seed {seed}: '+('Validation joint survival NLL' if col==0 else 'Validation Brier90')+' (lower better)')
            ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
            if row==0:
                ax.legend(fontsize=9)
    fig.suptitle('Saved validation logs only; no new training; diamonds = selected checkpoints')
    fig.tight_layout()
    args.output.mkdir(parents=True)
    (args.output/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    fig.savefig(args.output/'validation_nll_and_brier90.png', dpi=170)
    plt.close(fig)
    for name, expected in result['history_sha256'].items():
        if digest(args.source/name) != expected:
            raise ValueError('source history changed during audit')
    if digest(args.source/'report.json') != result['source_report_sha256']:
        raise ValueError('source report changed during audit')
    print(json.dumps(dict(status=result['status'], histories=len(result['rows']), output=str(args.output))))


if __name__ == '__main__':
    main()
