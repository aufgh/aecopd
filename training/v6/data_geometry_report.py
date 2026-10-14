"""已独立核验的早停/文本表示结果图与患者配对；不改候选锁。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patient_data.export_mimiciv import sha256
from training.v6.data_geometry_experiment import ROOT,SEEDS,ARMS
from training.v6.evaluation import paired_interval
from training.v6.train import write_json


def main():
    report=json.loads((ROOT/'screening.json').read_text())
    for tag in ('a','b'):
        proof=json.loads((ROOT/f'verification_{tag}.json').read_text())
        if proof['report_sha256']!=sha256(ROOT/'screening.json') or proof['maximum_error']!=0:
            raise ValueError('unverified report')
    fig,axs=plt.subplots(1,3,figsize=(15,4.4));colors={'raw':'#1f77b4','centered':'#ff7f0e','whitened':'#2ca02c'}
    pairs={};values={arm:[] for arm in ('centered','whitened')}
    for j,seed in enumerate(SEEDS):
        for arm in ARMS:
            history=json.loads((ROOT/f'seed{seed}'/arm/'history.json').read_text())
            axs[j].plot([r['epoch'] for r in history],[r['tuning_ibs'] for r in history],label=arm,color=colors[arm])
            info=report['fits'][str(seed)][arm];best=info['best_epoch']
            axs[j].scatter(best,history[best]['tuning_ibs'],marker='D',s=35,color=colors[arm])
        axs[j].set(title=f'Seed {seed}: early stopping active',xlabel='Actual epoch',ylabel='Tuning IBS7-365 (lower better)')
        axs[j].grid(alpha=.2);axs[j].legend()
        with np.load(ROOT/f'seed{seed}/raw/tuning_predictions.npz') as z:raw={k:z[k] for k in ('subject_ids','times','events','curve')}
        for arm in ('centered','whitened'):
            with np.load(ROOT/f'seed{seed}'/arm/'tuning_predictions.npz') as z:curve=z['curve']
            p=paired_interval(raw['subject_ids'],raw['times'],raw['events'],curve[:,:,1],raw['curve'][:,:,1],replicates=200,seed=seed)
            pairs[f'{seed}/{arm} minus raw']=p
            values[arm].append((p['integrated_brier_difference'],p['auc90_difference']))
    fig.suptitle('Only frozen text representation changed; matched V6; checkpoints selected by tuning IBS',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(ROOT/'early_stopping.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,3.8))
    for j,arm in enumerate(('centered','whitened')):
        v=np.asarray(values[arm])
        for k,ax in enumerate(axs):
            ax.scatter(v[:,k],j+np.array([-.12,0,.12]),s=35,color=colors[arm],alpha=.6)
            ax.scatter(v[:,k].mean(),j,marker='D',s=75,color=colors[arm])
    for ax,title in zip(axs,('IBS: candidate minus raw (lower better)','AUC90: candidate minus raw (higher better)'),strict=True):
        ax.axvline(0,color='black',linewidth=1);ax.set_yticks([0,1],['centered','whitened']);ax.set_title(title);ax.set_xlabel('Paired difference across three seeds');ax.grid(alpha=.2)
    fig.suptitle('Specified tuning set; retrained from matched initialization; no multiplicity adjustment',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(ROOT/'paired_differences.png',dpi=150);plt.close(fig)
    write_json(ROOT/'paired_diagnostic.json',dict(pairs=pairs,replicates=200,limits='conditional on fitted models; tuning exposed; no multiplicity adjustment',
        screening_sha256=sha256(ROOT/'screening.json'),no_confirmation_or_test=True))


if __name__=='__main__':main()
