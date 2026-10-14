"""补充配对增量/校准图，不改变训练、检查点或指标。"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(root):
    r=json.loads((root/'report.json').read_text());seeds=r['contract']['seeds']
    pairs=(('pool_fine minus pool_coarse','Drug identity + actual dose/frequency'),
           ('interaction_fine minus pool_fine','Interaction before pooling'),
           ('interaction_ssl minus interaction_fine','Masked reconstruction pretraining'),
           ('flat_mlp minus pool_fine','Flat MLP versus token pooling'),
           ('legacy_lr0.0001 minus legacy_lr0.0003','Lower LR: 1e-4 versus 3e-4'),
           ('legacy_lr3e-05 minus legacy_lr0.0003','Lower LR: 3e-5 versus 3e-4'))
    fig,ax=plt.subplots(1,2,figsize=(14,6),sharey=True)
    for row,(key,label) in enumerate(pairs):
        for j,seed in enumerate(seeds):
            m=r['comparisons'][str(seed)]['test'][key];y=row+(j-.5)*.16
            for col,(point,ci) in enumerate((('integrated_brier_difference','integrated_brier_95ci'),('auc90_difference','auc90_95ci'))):
                lo,hi=m[ci];ax[col].plot([lo,hi],[y,y],color=f'C{j}',alpha=.5)
                ax[col].scatter(m[point],y,color=f'C{j}',s=32,label=f'Seed {seed}' if row==0 else None)
    ax[0].set_yticks(np.arange(len(pairs)),[label for _,label in pairs]);ax[0].invert_yaxis()
    ax[0].set_title('IBS 7-365: candidate minus comparator (lower better)')
    ax[1].set_title('AUC90: candidate minus comparator (higher better)')
    for a in ax:a.axvline(0,color='black',linewidth=1);a.grid(axis='x',alpha=.2);a.legend();a.set_xlabel('Paired difference; patient-cluster bootstrap 200; 95% CI')
    fig.suptitle('Reused-test exploratory comparisons; no multiplicity adjustment');fig.tight_layout()
    fig.savefig(root/'paired_increment.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    for col,day in enumerate((30,90,365)):
        for name in ('pool_coarse','pool_fine','interaction_fine','interaction_ssl','flat_mlp'):
            bins=r['models'][str(seeds[0])][name]['folds']['test']['calibration'][str(day)]
            ax[col].plot([b['predicted'] for b in bins],[b['observed'] for b in bins],marker='o',label=name)
        ax[col].plot([0,1],[0,1],'k--',linewidth=1);ax[col].set_title(f'Seed {seeds[0]}: day {day} calibration')
        ax[col].set_xlabel('Predicted cumulative incidence');ax[col].set_ylabel('Observed World A fraction');ax[col].grid(alpha=.2)
    ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'calibration.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'));render(p.parse_args().root)
