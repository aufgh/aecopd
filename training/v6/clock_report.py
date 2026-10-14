"""第五批真实核验结果的英文图与中文报告；不重选模型。"""
import json
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.clock_experiment import ROOT,SEEDS,ARMS
from training.v6.task_optimization_report import averages,prepare_plot
from training.v6.evaluation import metrics
from training.v6.train import write_json


def run():
    r=json.loads((ROOT/'confirmation_report.json').read_text())
    proof=json.loads((ROOT/'confirmation_verification.json').read_text())
    if proof['report_sha256']!=sha256(ROOT/'confirmation_report.json'):
        raise ValueError('clock confirmation proof stale')
    chosen=r['selected'];names=('original_teacher',chosen);stats=averages(r['models'],names)
    plt=prepare_plot();fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for i,seed in enumerate(SEEDS):
        p=r['comparisons'][str(seed)]
        for ax,value,ci in zip(axes,(p['integrated_brier_difference'],p['auc90_difference']),
                (p['integrated_brier_95ci'],p['auc90_95ci'])):
            ax.plot(ci,[i,i],color=f'C{i}',lw=2);ax.scatter(value,i,color=f'C{i}',marker='D',s=45)
    for ax in axes:
        ax.axvline(0,color='gray');ax.set_yticks(range(3),[f'Seed {s}' for s in SEEDS]);ax.grid(axis='x',alpha=.25)
    axes[0].set_xlabel('IBS7-365: candidate minus original V6 (negative better)')
    axes[1].set_xlabel('AUC90: candidate minus original V6 (positive better)')
    fig.suptitle('Locked clock correction; previously exposed confirmation validation\n'
        '200 patient-cluster paired resamples per seed; no multiplicity adjustment')
    fig.tight_layout();fig.savefig(ROOT/'confirmation_increment.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for j,name in enumerate(names):
        details=[]
        for seed in SEEDS:
            with np.load(ROOT/f'seed{seed}'/name/'confirmation_predictions.npz') as z:
                details.append([metrics(z['times'],z['events'],z['curve'][:,:,c],cause=c) for c in (1,2)])
        for cause in (1,2):
            auc=np.array([x[cause-1]['auc_curve'] for x in details]);brier=np.array([x[cause-1]['brier_curve'] for x in details])
            valid=np.isfinite(auc[:,7:]).sum(0)
            mean_auc=np.divide(np.nansum(auc[:,7:],axis=0),valid,out=np.full(359,np.nan),where=valid>0)
            axes[cause-1,0].plot(np.arange(7,366),mean_auc,label=name,lw=1.5)
            axes[cause-1,1].plot(np.arange(7,366),brier[:,7:].mean(0),label=name,lw=1.5)
    for row,label in enumerate(('Strict AECOPD readmission','Competing death')):
        for ax in axes[row]:ax.set_xlabel('Days after discharge');ax.grid(alpha=.25);ax.legend()
        axes[row,0].set_title(label+': time-dependent AUC (higher better)')
        axes[row,1].set_title(label+': Brier score (lower better)')
    fig.suptitle('Three clinical-seed means; frozen annual probabilities; confirmation validation only')
    fig.tight_layout();fig.savefig(ROOT/'daily_evaluation.png',dpi=160);plt.close(fig)
    tuning=json.loads((ROOT/'tuning_report.json').read_text());ts=averages(tuning['models'],('original_teacher',*ARMS))
    decision=json.loads((ROOT/'continuation_decision.json').read_text())
    grid=json.loads((ROOT/'grid_convergence.json').read_text())
    result=dict(confirmation_means=stats,tuning_means=ts,decision=decision,
        maximum_grid_probability_error=max(x['maximum_probability_error'] for x in grid['comparisons'].values()),
        maximum_grid_ibs_difference=max(abs(x['ibs_difference']) for x in grid['comparisons'].values()),
        confirmation_report_sha256=sha256(ROOT/'confirmation_report.json'),goal_achieved=False)
    write_json(ROOT/'readout.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':run()
