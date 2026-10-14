"""读取既有训练日志绘图，不重训/不修改检查点。"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def plot(root):
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for row,seed in enumerate((42,7)):
        for arm in ('control','new_labs','med_details','both','availability'):
            h=json.loads((root/f'seed{seed}'/arm/'history.json').read_text())
            train=[r for r in h if r['train_objective'] is not None]
            axes[row,0].plot([r['epoch'] for r in train],[r['train_objective'] for r in train],label=arm)
            axes[row,1].plot([r['epoch'] for r in h],[r['validation_brier90'] for r in h],label=arm)
        for col,title in enumerate(('Training joint survival NLL','Validation Brier90')):
            axes[row,col].set_title(f'Seed {seed}: {title} (lower better)')
            axes[row,col].set_xlabel('Epoch'); axes[row,col].grid(alpha=.2); axes[row,col].legend(fontsize=8)
    fig.suptitle('Fixed split; frozen baseline; checkpoints selected by validation Brier90')
    fig.tight_layout(); fig.savefig(root/'figures/learning_curves.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    plot(p.parse_args().output)
