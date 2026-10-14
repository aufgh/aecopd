"""从已独立验收的预测重算历史分层增量，不训练或挑选新模型。"""
import argparse
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.horizon_objective import binary_metrics,paired90,labels90
from training.v6.train import write_json


def read(root,seed,arm,fold):
    with np.load(root/f'seed{seed}'/arm/(fold+'_predictions.npz')) as z: return {k:z[k] for k in z.files}


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    report=json.loads((args.parent/'report.json').read_text())
    proof=json.loads((args.parent/'independent_verification.json').read_text())
    if proof['report_sha256']!=sha256(args.parent/'report.json'): raise ValueError('parent unverified')
    for p,digest in report['contract']['code_sha256'].items():
        if sha256(Path(p))!=digest: raise ValueError('parent source changed')
    results={}; hashes={}
    for seed in (42,7):
        results[str(seed)]={}
        for fold in ('validation','test'):
            a,b=(read(args.parent,seed,arm,fold) for arm in ('survival_base','survival_full'))
            for key in ('hadm_ids','subject_ids','times','events','y90','history_counts'):
                if not np.array_equal(a[key],b[key]): raise ValueError('paired identity/label mismatch')
            if not np.array_equal(a['y90'],labels90(a['times'],a['events'])): raise ValueError('90-day label mismatch')
            for arm in ('survival_base','survival_full'):
                hashes[f'{seed}/{arm}/{fold}']=sha256(args.parent/f'seed{seed}'/arm/(fold+'_predictions.npz'))
                checkpoint=args.parent/f'seed{seed}'/arm/'best.pt'
                if sha256(checkpoint)!=report['models'][str(seed)][arm]['checkpoint_sha256']: raise ValueError('checkpoint changed')
            count=a['history_counts']
            groups={'all':np.ones(len(count),bool),'prior_0':count==0,'prior_1_2':(count>=1)&(count<=2),'prior_3_plus':count>=3}
            results[str(seed)][fold]={}
            for name,keep in groups.items():
                base=binary_metrics(a['y90'][keep],a['risk90'][keep]); full=binary_metrics(b['y90'][keep],b['risk90'][keep])
                comparison=paired90(a['subject_ids'][keep],a['y90'][keep],b['risk90'][keep],a['risk90'][keep],seed=seed)
                results[str(seed)][fold][name]=dict(baseline=base,full=full,comparison=comparison,
                    absolute_risk_change_mean=float((b['risk90'][keep]-a['risk90'][keep]).mean()),
                    absolute_risk_change_median=float(np.median(np.abs(b['risk90'][keep]-a['risk90'][keep]))))
                expected_base=report['models'][str(seed)]['survival_base']['folds'][fold]
                expected_full=report['models'][str(seed)]['survival_full']['folds'][fold]
                if name!='all': expected_base=expected_base['history_strata'][name]; expected_full=expected_full['history_strata'][name]
                for actual,expected in ((base,expected_base),(full,expected_full)):
                    for key in ('auc90','brier90','average_precision','logloss90'):
                        if abs(actual[key]-expected[key])>1e-10: raise ValueError('stored summary metric mismatch')
    args.output.mkdir(parents=True)
    summary=dict(status='recomputed_verified_same_model_history_strata_not_new_training',
        parent_report_sha256=sha256(args.parent/'report.json'),prediction_sha256=hashes,results=results,
        code_sha256=sha256(Path(__file__)),
        limits=['Reused test, not blind; post-hoc subgroup analysis; no multiplicity correction.',
            'Only40 test positives in recorded-prior0; intervals conditional on fitted models, not retraining variance.',
            'Recorded-prior0 is not lifetime first disease or no admissions outside this database.',
            'Clinical+BERT combination, not proof that any one lab/med/text field causes the gain.',
            'Text may contain additional historical utilization; no claim of causal disentanglement.'])
    write_json(args.output/'report.json',summary); plot(args.output,summary)
    print(json.dumps({s:{name:dict(delta_auc=x['comparison']['metrics']['auc90'],delta_brier=x['comparison']['metrics']['brier90'])
                         for name,x in result['test'].items()} for s,result in results.items()}),flush=True)


def plot(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=('all','prior_0','prior_1_2','prior_3_plus')
    labels=('All eligible discharges','No recorded prior COPD stays','1-2 recorded prior COPD stays','3+ recorded prior COPD stays')
    folder=root/'figures'; folder.mkdir()
    fig,axes=plt.subplots(1,2,figsize=(15,6),sharey=True)
    for j,name in enumerate(groups):
        for i,seed in enumerate((42,7)):
            d=report['results'][str(seed)]['test'][name]['comparison']['metrics']
            for ax,key in zip(axes,('brier90','auc90')):
                r=d[key]; yy=j+(i-.5)*.17; color=('#2674ae','#cf7722')[i]
                ax.plot(r['ci95'],[yy,yy],color=color,lw=2)
                ax.scatter(r['difference'],yy,color=color,label=f'seed{seed}' if j==0 else None)
    for ax in axes:
        ax.axvline(0,color='k',lw=1); ax.set_yticks(range(4),labels); ax.grid(axis='x',alpha=.2); ax.legend()
    axes[0].invert_yaxis(); axes[0].set(title='Probability error',xlabel='Brier90: clinical+BERT minus frozen baseline (lower better)')
    axes[1].set(title='Risk ordering',xlabel='AUC90: clinical+BERT minus frozen baseline (higher better)')
    fig.suptitle('Within-history clinical increment; same patients and endpoint; reused test, not blind')
    fig.text(.5,.01,'200 patient-clustered 95% intervals; no multiplicity adjustment; prior0 has only40 positive events',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.045,1,.95)); fig.savefig(folder/'history_stratified_increment.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,5),sharey=True)
    for ax,seed in zip(axes,(42,7)):
        base=[report['results'][str(seed)]['test'][g]['baseline']['auc90'] for g in groups]
        full=[report['results'][str(seed)]['test'][g]['full']['auc90'] for g in groups]
        x=np.arange(4)
        ax.bar(x-.18,base,.36,label='Frozen history + diagnosis + disposition',color='#99bad5')
        ax.bar(x+.18,full,.36,label='Plus clinical state and BERT',color='#2674ae')
        for xx,v in zip(x-.18,base): ax.text(xx,v+.005,f'{v:.3f}',ha='center',fontsize=10)
        for xx,v in zip(x+.18,full): ax.text(xx,v+.005,f'{v:.3f}',ha='center',fontsize=10)
        ax.set(xticks=x,xticklabels=('All','Prior0','Prior1-2','Prior3+'),ylim=(0,.9),title=f'Seed{seed}',ylabel='90-day AUC')
        ax.grid(axis='y',alpha=.2); ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('Existing validated V6: absolute performance, not a new tuned model; reused test')
    fig.tight_layout(); fig.savefig(folder/'absolute_auc_by_history.png',dpi=170); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,default=Path('training_runs/v6_90day_objective_2026-10-11'))
    p.add_argument('--output',type=Path,default=Path('training_runs/v6_clinical_increment_evidence_2026-10-11'))
    run(p.parse_args())
