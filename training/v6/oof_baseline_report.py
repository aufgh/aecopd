"""OOF基线有限诊断图；不选择模型或更新拟合来源。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patient_data.export_mimiciv import sha256
from training.v6.oof_baseline_diagnostic import ROOT
from training.v6.module_surgery import probability_curve


def main():
    report=json.loads((ROOT/'diagnostic.json').read_text())
    for tag in ('a','b'):
        proof=json.loads((ROOT/f'verification_{tag}.json').read_text())
        if proof['report_sha256']!=sha256(ROOT/'diagnostic.json') or proof['maximum_replay_error']!=0:
            raise ValueError('unverified result')
    with np.load(ROOT/'oof_risk_interface.npz') as z:
        indices=z['indices']; a=z['log_amplitude']; k=z['shape']
    oof=probability_curve(a,k)[:,90,1]
    with np.load(ROOT/'full/predictions.npz') as z:
        np.testing.assert_array_equal(indices,z['indices'][:len(indices)])
        full=probability_curve(z['log_amplitude'][:len(indices)],z['shape'][:len(indices)])[:,90,1]
    delta=(oof-full)*100
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    for name,info in report['fit_information'].items():
        history=info['history']
        axs[0].plot([r['update'] for r in history],[r['fixed_train_nll'] for r in history],label=name,
                    linewidth=2.5 if name=='full' else 1.3)
    axs[0].set(title='Fresh H34: fixed training NLL',xlabel='Full-batch updates',ylabel='Joint survival NLL (lower better)')
    axs[0].legend(ncol=2);axs[0].grid(alpha=.2)
    ordered=np.sort(delta)
    axs[1].plot(ordered,np.arange(1,len(delta)+1)/len(delta),color='#1f77b4',linewidth=2)
    axs[1].axvline(0,color='black',linewidth=1)
    axs[1].set(title='Same training records: OOF minus in-sample risk',
               xlabel='90-day readmission probability difference (percentage points)',ylabel='Cumulative fraction of records',
               xlim=(-2.5,2.5),ylim=(0,1))
    outside=int((np.abs(delta)>2.5).sum())
    axs[1].text(.04,.82,f'Mean absolute difference: {np.abs(delta).mean():.3f} pp\n'
        f'10th / 90th percentiles: {np.quantile(delta,.1):.3f} / {np.quantile(delta,.9):.3f} pp\n'
        f'{outside} records outside displayed x-range',transform=axs[1].transAxes,fontsize=10)
    axs[1].grid(alpha=.2)
    fig.suptitle('Patient-level 5-fold baseline diagnostic; one seed; no clinical network retraining',fontsize=13)
    fig.text(.5,.015,'80% inner-fold fits vs 100% full fit: sample-size / composition effects are not isolated; not a clinical gain.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(ROOT/'oof_baseline_diagnostic.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
