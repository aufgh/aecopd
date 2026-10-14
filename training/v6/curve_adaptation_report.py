"""只展示已经独立核验的第八批结果；不据图重新选模。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.curve_adaptation import ROOT,SEEDS,ARMS
from training.v6.train import write_json


def run():
    status=json.loads((ROOT/'postprocess_state.json').read_text())
    if status['status']!='complete':raise ValueError('postprocessing not complete')
    report=json.loads((ROOT/'tuning_report.json').read_text())
    proof=json.loads((ROOT/'tuning_verification.json').read_text())
    if proof['status']!='independently_replayed' or proof['report_sha256']!=sha256(ROOT/'tuning_report.json'):
        raise ValueError('tuning proof stale')
    fig,axes=plt.subplots(3,2,figsize=(13,11),layout='constrained')
    for row,seed in enumerate(SEEDS):
        original=report['models'][str(seed)]['original_teacher']['metrics']['readmission']
        for arm in ARMS:
            history=json.loads((ROOT/f'seed{seed}'/arm/'history.json').read_text())
            x=[r['epoch'] for r in history]
            axes[row,0].plot(x,[r['tuning_ibs'] for r in history],label=arm)
            axes[row,1].plot(x,[r['tuning_auc90'] for r in history],label=arm)
        axes[row,0].axhline(original['integrated_brier_7_365'],color='black',ls='--',label='Original V6')
        axes[row,1].axhline(original['points']['90']['auc'],color='black',ls='--',label='Original V6')
        for col,title in enumerate(('Tuning readmission IBS (lower better)','Tuning AUC90 (higher better)')):
            axes[row,col].set_title(f'Seed {seed}: {title}');axes[row,col].set_xlabel('Epoch')
            axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=8)
    fig.suptitle('Fixed original V6 representations; three objectives; 100 epochs; checkpoints selected by tuning IBS')
    fig.savefig(ROOT/'tuning_training_curves.png',dpi=160);plt.close(fig)
    stage='confirmation' if (ROOT/'confirmation_report.json').exists() else 'tuning'
    if stage=='confirmation':
        report=json.loads((ROOT/'confirmation_report.json').read_text())
        if json.loads((ROOT/'confirmation_verification.json').read_text())['report_sha256']!=sha256(ROOT/'confirmation_report.json'):
            raise ValueError('confirmation proof stale')
    names=[n for n in ARMS if n in report['models']['42']]
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for j,name in enumerate(names):
        changes=[]
        for seed in SEEDS:
            m=report['models'][str(seed)]
            a=m[name]['metrics']['readmission'];b=m['original_teacher']['metrics']['readmission']
            changes.append([a['integrated_brier_7_365']-b['integrated_brier_7_365'],a['points']['90']['auc']-b['points']['90']['auc']])
        changes=np.array(changes)
        for col in range(2):
            axes[col].scatter(changes[:,col],np.full(3,j),alpha=.6)
            axes[col].scatter(changes[:,col].mean(),j,marker='D',s=65)
    for col,title in enumerate(('IBS: adapted - original (lower better)','AUC90: adapted - original (higher better)')):
        axes[col].set_title(title);axes[col].set_yticks(range(len(names)),names if col==0 else ['']*len(names))
        axes[col].axvline(0,color='black',lw=1);axes[col].grid(axis='x',alpha=.2)
        axes[col].set_xlabel('Dots: three seeds; diamond: mean; no multiplicity adjustment')
        # 固定实用量级，避免将1e-7的IBS波动拉满画布造成明显改善错觉。
        axes[col].set_xlim((-.001,.001) if col==0 else (-.005,.005))
        axes[col].ticklabel_format(axis='x',style='plain',useOffset=False)
        axes[col].xaxis.set_major_locator(MaxNLocator(5))
    fig.suptitle(f'{stage.capitalize()}: frozen original V6; tiny global recalibration is not clinical increment')
    fig.savefig(ROOT/f'{stage}_differences.png',dpi=160);plt.close(fig)
    write_json(ROOT/'figure_verification.json',dict(status='figures_generated_from_verified_reports',
        source_sha256=sha256(Path(__file__)),report_sha256=sha256(ROOT/f'{stage}_report.json'),
        stage=stage,goal_achieved=False))
    print(json.dumps(dict(status='verified_figures_saved',stage=stage,goal_achieved=False)),flush=True)


if __name__=='__main__':run()
