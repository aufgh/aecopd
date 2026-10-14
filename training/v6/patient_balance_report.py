"""核验后的患者加权学习曲线与配对图；不根据图表重选。"""
import argparse
import json
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.patient_balance_experiment import ROOT,SEEDS,OLD
from training.v6.patient_balance_model import TRIALS
from training.v6.task_optimization_report import averages,prepare_plot
from training.v6.train import write_json


def run(mode):
    file='tuning_report.json' if mode=='tuning' else 'confirmation_report.json'
    p=json.loads((ROOT/('tuning_verification.json' if mode=='tuning' else 'confirmation_verification.json')).read_text())
    if p['report_sha256']!=sha256(ROOT/file):raise ValueError('balance report proof stale')
    r=json.loads((ROOT/file).read_text());plt=prepare_plot()
    if mode=='tuning':
        if p['status']!='balance_tuning_independently_verified':raise ValueError('independent tuning replay required')
        names=('original_teacher','full_uniform',*[t.name for t in TRIALS])
        fig,axes=plt.subplots(3,2,figsize=(12,10))
        for row,seed in enumerate(SEEDS):
            reference=r['models'][str(seed)]['original_teacher']['metrics']['readmission']['integrated_brier_7_365']
            for index,name in enumerate(names[1:]):
                path=OLD/f'seed{seed}/warm_finetune/history.json' if name=='full_uniform' else ROOT/f'seed{seed}'/name/'history.json'
                history=json.loads(path.read_text());style='--' if name=='full_uniform' else '-'
                axes[row,0].plot([x['epoch'] for x in history[1:]],[x['train_nll'] for x in history[1:]],ls=style,label=name,lw=1.)
                axes[row,1].plot([x['epoch'] for x in history],[x['tuning_ibs']-reference for x in history],ls=style,label=name,lw=1.)
            axes[row,1].axhline(0,color='black',ls=':',label='Original V6')
            axes[row,0].set_title(f'Seed {seed}: unweighted online training NLL',fontsize=10)
            axes[row,1].set_title(f'Seed {seed}: tuning IBS minus original V6 (negative better)',fontsize=10)
            for ax in axes[row]:ax.grid(alpha=.2);ax.set_xlabel('Epoch')
        fig.suptitle('Patient-balanced loss / encoder plasticity; fixed admission-level evaluation\n'
            '15 new complete 100-epoch fits; full_uniform is a verified reused control')
        fig.legend(*axes[0,1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.935),ncol=4)
        fig.tight_layout(rect=(0,0,1,.87));fig.savefig(ROOT/'training_curves.png',dpi=160);plt.close(fig)
    else:
        names=('original_teacher',r['lock']['matching_uniform'],r['selected'])
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for index,reference in enumerate(names[:2]):
            for row,seed in enumerate(SEEDS):
                result=r['comparisons'][str(seed)][reference]
                for ax,value,ci in zip(axes,(result['integrated_brier_difference'],result['auc90_difference']),
                    (result['integrated_brier_95ci'],result['auc90_95ci'])):
                    y=row+(index-.5)*.15;ax.plot(ci,[y,y],color=f'C{index}',label=reference if row==0 else None)
                    ax.scatter(value,y,color=f'C{index}',marker='D',s=25)
        for ax in axes:ax.axvline(0,color='gray');ax.set_yticks(range(3),[f'Seed {s}' for s in SEEDS]);ax.grid(axis='x',alpha=.2)
        axes[0].set_xlabel('IBS7-365: candidate minus comparator (negative better)')
        axes[1].set_xlabel('AUC90: candidate minus comparator (positive better)')
        fig.suptitle('Locked patient-balanced candidate; admission-level confirmation validation\n200 patient-cluster resamples; no multiplicity adjustment')
        axes[0].legend();fig.tight_layout();fig.savefig(ROOT/'confirmation_increment.png',dpi=180);plt.close(fig)
    stats=averages(r['models'],names)
    write_json(ROOT/(mode+'_readout.json'),dict(means=stats,selected=r['selected'],source_report_sha256=sha256(ROOT/file),
        goal_achieved=False,evaluation_unweighted=True))
    print(json.dumps(dict(status='verified_balance_figures_written',mode=mode,selected=r['selected'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('tuning','confirmation'),required=True)
    run(p.parse_args().mode)
