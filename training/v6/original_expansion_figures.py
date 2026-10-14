"""已保存预测的按日排序/校准和历史层分析；不选择检查点、不训练。"""
import argparse
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.expanded_original_experiment import ARMS,checked
from training.v6.evaluation import metrics,public_metrics,paired_interval
from training.v6.train import write_json


def render(root):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((root/'report.json').read_text());cfg=json.loads((root/'config.json').read_text())
    proof=json.loads((root/'fresh_process_verification.json').read_text())
    if proof['report_sha256']!=sha256(root/'report.json'):raise ValueError('independent proof stale')
    args=argparse.Namespace(**{k:Path(v) for k,v in cfg.items()});d,*_=checked(args)
    idx=np.flatnonzero(d.arrays['patient_splits']=='test');prior=np.expm1(d.history_raw[idx,2]).round().astype(int)
    strata={};curves={};selected=report['validation_selected_arm']
    labels={'control':'Original V6','more_labs':'Additional labs','more_meds':'Drug identity + dose/frequency',
       'both':'Additional labs + drug details','interaction_only':'Original V6 pre-pooling interaction','both_interaction':'Expanded inputs + interaction'}
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for column,seed in enumerate((42,7)):
        strata[str(seed)]={};curves[seed]={}
        for arm in ARMS:
            with np.load(root/f'seed{seed}/{arm}/test_predictions.npz') as z:a={k:z[k] for k in ('hadm_ids','subject_ids','times','events','curve')}
            if not np.array_equal(a['hadm_ids'],d.arrays['hadm_ids'][idx]):raise ValueError('saved members mismatch')
            curves[seed][arm]=a
            detail=metrics(a['times'],a['events'],a['curve'][:,:,1])
            axes[column].plot(np.arange(7,366),detail['auc_curve'][7:],label=labels[arm],lw=1.3)
            strata[str(seed)][arm]={}
            for name,keep in (('prior_0',prior==0),('prior_1_2',(prior>=1)&(prior<=2)),('prior_3_plus',prior>=3)):
                strata[str(seed)][arm][name]=public_metrics(metrics(a['times'][keep],a['events'][keep],a['curve'][keep,:,1]))
        axes[column].set_title(f'Seed {seed}: reused-test AUC(t)');axes[column].set_xlabel('Days after discharge');axes[column].set_ylabel('AUC(t)')
        axes[column].legend(fontsize=7);axes[column].grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'auc_by_day.png',dpi=180);plt.close(fig)
    comparisons={}
    for seed in (42,7):
        comparisons[str(seed)]={};control=curves[seed]['control']
        for arm in ('more_labs','interaction_only'):
            a=curves[seed][arm];comparisons[str(seed)][arm]={}
            for name,keep in (('prior_0',prior==0),('prior_1_2',(prior>=1)&(prior<=2)),('prior_3_plus',prior>=3)):
                comparisons[str(seed)][arm][name]=paired_interval(a['subject_ids'][keep],a['times'][keep],a['events'][keep],
                     a['curve'][keep,:,1],control['curve'][keep,:,1],seed=seed)
    write_json(root/'history_strata.json',dict(metrics=strata,comparisons=comparisons,
       definition='prior completed COPD-coded stays from fixed original history; no new cohort selection',
       limits='reused test; exploratory subgroup/multiple comparisons; low cases can prevent stable AUC'))
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    for row,seed in enumerate((42,7)):
        for column,day in enumerate((90,365)):
            ax=axes[row,column]
            for arm in dict.fromkeys(('control',selected,'interaction_only')):
                bins=report['models'][str(seed)][arm]['folds']['test']['calibration'][str(day)]
                ax.plot([b['predicted'] for b in bins],[b['observed'] for b in bins],'o-',label=labels[arm])
            ax.plot([0,1],[0,1],'k--',lw=.8);ax.set_title(f'Seed {seed}: day {day}');ax.set_xlabel('Predicted cumulative risk');ax.set_ylabel('Observed World-A fraction')
            ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Reused-test calibration; no post-hoc test recalibration');fig.tight_layout();fig.savefig(root/'calibration.png',dpi=170);plt.close(fig)
    # 训练报告的数值不改，仅重画更易读的英文标签和刻度，并补同架构21/29药物对照。
    pairs=[(labels[a],f'{a} minus control') for a in ARMS[1:]]+[
       ('Interaction on expanded inputs','both_interaction minus both')]
    width_file=root/'molecule_width_control/report.json'
    width=json.loads(width_file.read_text()) if width_file.exists() else None
    if width:pairs.append(('29 versus 21 drug identities',None))
    fig,axes=plt.subplots(1,2,figsize=(15,7),sharey=True)
    from matplotlib.ticker import MaxNLocator
    for j,(_,key) in enumerate(pairs):
        for seed,color,offset in ((42,'C0',-.08),(7,'C1',.08)):
            rr=report['comparisons'][str(seed)]['test'][key] if key else width['comparisons'][str(seed)]['test']
            for ax,value,ci in ((axes[0],'integrated_brier_difference','integrated_brier_95ci'),(axes[1],'auc90_difference','auc90_95ci')):
                ax.plot(rr[ci],[j+offset]*2,color=color,alpha=.6);ax.scatter(rr[value],j+offset,color=color,label=f'Seed {seed}' if j==0 else None)
    for ax in axes:
        ax.axvline(0,color='black',lw=1);ax.set_yticks(range(len(pairs)),[n for n,_ in pairs]);ax.grid(axis='x',alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.legend(fontsize=9)
    axes[0].invert_yaxis();axes[0].ticklabel_format(axis='x',style='sci',scilimits=(-3,-3))
    axes[0].set_xlabel('IBS7-365 difference (lower better)');axes[1].set_xlabel('AUC90 difference (higher better)')
    fig.suptitle('Original V6; candidate minus comparator; reused test\nPatient-clustered bootstrap 200; 95% intervals; no multiplicity adjustment')
    fig.tight_layout();fig.savefig(root/'paired_increment_readable.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_original_interaction_2026-10-12'));render(p.parse_args().root)
