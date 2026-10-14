"""核验后的纵向读取器学习曲线；不重新选择候选。"""
import json
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.clinical_history_experiment import ROOT,SEEDS
from training.v6.clinical_history_model import ARMS
from training.v6.task_optimization_report import prepare_plot,averages
from training.v6.train import write_json


def run():
    proof=json.loads((ROOT/'tuning_verification.json').read_text())
    if proof['status']!='history_tuning_independently_verified' or proof['report_sha256']!=sha256(ROOT/'tuning_report.json'):
        raise ValueError('history independent proof required')
    r=json.loads((ROOT/'tuning_report.json').read_text());plt=prepare_plot()
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    for row,seed in enumerate(SEEDS):
        baseline=r['models'][str(seed)]['original_teacher']['metrics']['readmission']['integrated_brier_7_365']
        for arm in ARMS:
            records=json.loads((ROOT/f'seed{seed}'/arm/'history.json').read_text())
            axes[row,0].plot([x['epoch'] for x in records[1:]],[x['train_nll'] for x in records[1:]],label=arm,lw=1.1)
            axes[row,1].plot([x['epoch'] for x in records],[x['tuning_ibs']-baseline for x in records],label=arm,lw=1.1)
        axes[row,1].axhline(0,color='black',ls='--',label='Original V6')
        axes[row,0].set_title(f'Seed {seed}: online training joint NLL (lower better)',fontsize=10)
        axes[row,1].set_title(f'Seed {seed}: tuning IBS minus original V6 (negative better)',fontsize=10)
        for ax in axes[row]:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.legend(*axes[0,1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.935),ncol=3)
    fig.suptitle('Historical clinical-state reading; 15 complete 100-epoch fits\nFrozen original V6 teacher; checkpoints selected on tuning IBS')
    fig.tight_layout(rect=(0,0,1,.87));fig.savefig(ROOT/'training_curves.png',dpi=160);plt.close(fig)
    summary=averages(r['models'],('original_teacher',*ARMS))
    write_json(ROOT/'readout.json',dict(selected=r['selected'],means=summary,all_best_epochs_zero=all(
        r['models'][str(s)][a]['training']['best_epoch']==0 for s in SEEDS for a in ARMS),
        independent_replay_maximum_error=proof['maximum_error'],goal_achieved=False))
    print(json.dumps(dict(status='verified_history_report_written',selected=r['selected'])),flush=True)


if __name__=='__main__':run()
