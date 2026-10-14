"""一次有限的低学习率/早停敏感性检查；原90天主试验不覆盖。"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.architecture_experiment import batches
from training.v6.compact_ablation import inputs
from training.v6.horizon_objective import (labels90, binary_metrics, paired90, ranked_curves,
    make_model, restore, fit90, predict90)
from training.v6.risk import predict_cif
from training.v6.evaluation import GRID
from training.v6.train import write_json


def loaded(root, seed, arm, fold):
    with np.load(root/f'seed{seed}'/arm/(fold+'_predictions.npz')) as z:
        return {k:z[k] for k in z.files}


def new_model(args, seed, arm, hist, prevalence):
    previous = restore(args.parent, seed, arm, hist, prevalence)
    return make_model(arm, previous.baseline, 768, seed)


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2); start=time.perf_counter()
    primary=json.loads((args.parent/'report.json').read_text())
    proof=json.loads((args.parent/'independent_verification.json').read_text())
    if proof['report_sha256'] != sha256(args.parent/'report.json'): raise ValueError('unverified primary report')
    data, compact, hist, _=inputs(args.reference,args.compact_dir)
    if data.provenance != primary['contract']['data_provenance'] or compact.provenance != primary['contract']['compact_provenance']:
        raise ValueError('primary input mismatch')
    ix={f:np.flatnonzero(data.arrays['patient_splits']==f) for f in ('train','validation','test')}
    y=labels90(data.times,data.events); prevalence=float(y[ix['train']].mean())
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    contract=dict(parent=str(args.parent),parent_report_sha256=sha256(args.parent/'report.json'),
        data_provenance=data.provenance,compact_provenance=compact.provenance,seeds=[42,7],
        lr=1e-4,patience=20,epochs=60,warmup=5,selection='validation Brier90 only',
        trigger='binary seed7 selected epoch0; validation AUC improved but Brier deteriorated',
        limits='post-hoc after primary test inspected; no blind test or multiplicity adjustment',
        protocol_sha256=sha256(Path('docs/V6_90DAY_OPTIMIZATION_CHECK_2026-10-11.md')),
        code_sha256={**primary['contract']['code_sha256'],str(Path(__file__)):sha256(Path(__file__))})
    write_json(args.output/'contract.json',contract)
    cached={f:batches(data,compact,ix[f],512,expanded=True,text='bert') for f in ('train','validation')}
    fitted={}
    for seed in (42,7):
        (args.output/f'seed{seed}').mkdir()
        for arm in ('survival_full','binary_full'):
            model=new_model(args,seed,arm,hist,prevalence)
            write_json(args.output/'status.json',dict(status='training',seed=seed,arm=arm))
            fitted[seed,arm]=fit90(model,cached['train'],cached['validation'],y[ix['validation']],
                args.output/f'seed{seed}'/arm,epochs=60,lr=1e-4,seed=seed,patience=20,warmup=5)
            del model
    write_json(args.output/'status.json',dict(status='all_checkpoints_selected_before_supplementary_test'))
    models,comparisons,selected={},{},{}
    for seed in (42,7):
        models[str(seed)]={}; comparisons[str(seed)]={}; selected[str(seed)]={}
        for arm in ('survival_full','binary_full'):
            m=new_model(args,seed,arm,hist,prevalence)
            m.load_state_dict(torch.load(args.output/f'seed{seed}'/arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            models[str(seed)][arm]={**fitted[seed,arm],'folds':{}}
            for fold in ('validation','test'):
                idx=ix[fold]
                prepared=cached[fold] if fold in cached else batches(data,compact,idx,512,expanded=True,text='bert')
                pred=predict90(m,prepared); ref=loaded(args.parent,seed,arm,fold)
                if not np.array_equal(ref['hadm_ids'],data.arrays['hadm_ids'][idx]) or not np.array_equal(ref['y90'],y[idx]):
                    raise ValueError('primary identity/target mismatch')
                detail=binary_metrics(y[idx],pred['risk90'])
                prior=np.expm1(data.history_raw[idx,2]).round().astype(int)
                detail['history_strata']={name:binary_metrics(y[idx][keep],pred['risk90'][keep]) for name,keep in
                    (('prior_0',prior==0),('prior_1_2',(prior>=1)&(prior<=2)),('prior_3_plus',prior>=3)) if keep.any()}
                if 'shape' in pred: pred['curve']=predict_cif(pred['parameters'],pred['shape'],GRID)
                np.savez_compressed(args.output/f'seed{seed}'/arm/(fold+'_predictions.npz'),
                    hadm_ids=data.arrays['hadm_ids'][idx],subject_ids=data.arrays['subject_ids'][idx],
                    y90=y[idx],times=data.times[idx],events=data.events[idx],history_counts=prior,
                    **{k:v for k,v in pred.items() if k!='loss'})
                models[str(seed)][arm]['folds'][fold]=detail
                comparisons[str(seed)].setdefault(fold,{})[arm+'_lower_lr_minus_primary']=paired90(
                    data.arrays['subject_ids'][idx],y[idx],pred['risk90'],ref['risk90'],replicates=200,seed=seed)
            primary_score=primary['models'][str(seed)][arm]['best_validation_brier90']
            selected[str(seed)][arm]='check' if fitted[seed,arm]['best_validation_brier90'] < primary_score else 'primary'
            del m
        for fold in ('validation','test'):
            idx=ix[fold]
            chosen={arm:loaded(args.output if selected[str(seed)][arm]=='check' else args.parent,seed,arm,fold)['risk90']
                    for arm in ('survival_full','binary_full')}
            comparisons[str(seed)][fold]['validation_selected_binary_minus_survival']=paired90(
                data.arrays['subject_ids'][idx],y[idx],chosen['binary_full'],chosen['survival_full'],replicates=200,seed=seed)
            for arm in chosen:
                constant=loaded(args.parent,seed,arm.replace('_full','_constant'),fold)['risk90']
                comparisons[str(seed)][fold]['validation_selected_'+arm+'_minus_constant']=paired90(
                    data.arrays['subject_ids'][idx],y[idx],chosen[arm],constant,replicates=200,seed=seed)
    report=dict(status='completed_limited_optimization_check',contract=contract,models=models,
                comparisons=comparisons,validation_selected_source=selected,elapsed_seconds=time.perf_counter()-start)
    write_json(args.output/'report.json',report); render(args,report,primary)
    write_json(args.output/'status.json',dict(status=report['status'],seconds=report['elapsed_seconds']))
    print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'])),flush=True)


def verify(args):
    report=json.loads((args.output/'report.json').read_text()); contract=report['contract']
    if sha256(args.parent/'report.json') != contract['parent_report_sha256']: raise ValueError('primary changed')
    for p,digest in contract['code_sha256'].items():
        if sha256(Path(p)) != digest: raise ValueError('code changed: '+p)
    if sha256(Path('docs/V6_90DAY_OPTIMIZATION_CHECK_2026-10-11.md')) != contract['protocol_sha256']: raise ValueError('protocol changed')
    data,compact,hist,_=inputs(args.reference,args.compact_dir)
    if data.provenance != contract['data_provenance'] or compact.provenance != contract['compact_provenance']: raise ValueError('input changed')
    y=labels90(data.times,data.events); errors={}
    for seed in (42,7):
        for arm in ('survival_full','binary_full'):
            path=args.output/f'seed{seed}'/arm/'best.pt'
            if sha256(path) != report['models'][str(seed)][arm]['checkpoint_sha256']: raise ValueError('checkpoint changed')
            model=new_model(args,seed,arm,hist,y[data.arrays['patient_splits']=='train'].mean())
            frozen={k:v.clone() for k,v in model.baseline.state_dict().items()}
            model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
            if not all(torch.equal(v,model.baseline.state_dict()[k]) for k,v in frozen.items()): raise ValueError('frozen base changed')
            for fold in ('validation','test'):
                idx=np.flatnonzero(data.arrays['patient_splits']==fold)
                pred=predict90(model,batches(data,compact,idx,512,expanded=True,text='bert'))
                saved=loaded(args.output,seed,arm,fold)
                for key,value in (('hadm_ids',data.arrays['hadm_ids'][idx]),('subject_ids',data.arrays['subject_ids'][idx]),
                                  ('times',data.times[idx]),('events',data.events[idx]),('y90',y[idx])):
                    if not np.array_equal(saved[key],value): raise ValueError('identity/label mismatch')
                for key,value in pred.items():
                    if key!='loss' and not np.allclose(saved[key],value,atol=2e-6,rtol=1e-5): raise ValueError('reload mismatch')
                if 'shape' in pred and not np.allclose(saved['curve'],predict_cif(pred['parameters'],pred['shape'],GRID),atol=2e-6,rtol=1e-5):
                    raise ValueError('curve reload mismatch')
                actual=binary_metrics(y[idx],pred['risk90']); expected=report['models'][str(seed)][arm]['folds'][fold]
                for key in ('auc90','average_precision','brier90','logloss90'):
                    if abs(actual[key]-expected[key])>1e-7: raise ValueError('metric mismatch')
                errors[f'{seed}/{arm}/{fold}']=float(np.abs(saved['risk90']-pred['risk90']).max())
    write_json(args.output/'independent_verification.json',dict(status='all4_models_all8_folds_gpu_reloaded',
        errors=errors,baseline_unchanged=True,report_sha256=sha256(args.output/'report.json')))
    print('all4_models_all8_folds_gpu_reloaded',flush=True)


def render(args,report,primary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=args.output/'figures'; folder.mkdir()
    for seed in (42,7):
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        for arm,color in (('survival_base','#729fcf'),('survival_full','#08306b'),('binary_base','#fdae6b'),('binary_full','#a63603')):
            source='primary' if arm.endswith('_base') else report['validation_selected_source'][str(seed)][arm]
            root=args.output if source=='check' else args.parent
            p=loaded(root,seed,arm,'test'); m=(report if source=='check' else primary)['models'][str(seed)][arm]['folds']['test']
            c=ranked_curves(p['y90'],p['risk90'])
            axes[0].plot(c['fpr'],c['tpr'],color=color,label=f'{arm} ({source}), AUC={m["auc90"]:.3f}')
            axes[1].step(c['recall'],c['precision'],where='pre',color=color,label=f'{arm}, AP={m["average_precision"]:.3f}')
            bins=m['calibration_bins']
            axes[2].plot([b['predicted'] for b in bins],[b['observed'] for b in bins],'o-',color=color,label=arm)
        axes[0].plot([0,1],[0,1],'k--',alpha=.4); axes[1].axhline(m['observed_fraction'],color='k',linestyle='--',alpha=.4)
        axes[2].plot([0,1],[0,1],'k--',alpha=.4); axes[2].set_xlim(0,.6); axes[2].set_ylim(0,.6)
        for ax,title,xlabel,ylabel in zip(axes,('90-day ROC','90-day Precision-Recall','90-day Calibration'),
                ('False positive rate','Recall','Mean predicted risk'),('True positive rate','Precision','Observed fraction')):
            ax.set(title=title,xlabel=xlabel,ylabel=ylabel); ax.legend(fontsize=7); ax.grid(alpha=.2)
        fig.suptitle(f'Seed{seed}: validation-selected from two recipes; reused test; post-hoc sensitivity')
        fig.tight_layout(); fig.savefig(folder/f'test_seed{seed}_selected.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
    names=('validation_selected_binary_minus_survival','validation_selected_survival_full_minus_constant','validation_selected_binary_full_minus_constant')
    for j,name in enumerate(names):
        for i,seed in enumerate((42,7)):
            d=report['comparisons'][str(seed)]['test'][name]['metrics']
            for ax,key in zip(axes,('brier90','auc90')):
                x=d[key]; lo,hi=x['ci95']; yy=j+(i-.5)*.16; color=('#3465a4','#e6550d')[i]
                ax.plot([lo,hi],[yy,yy],color=color,alpha=.6)
                ax.scatter(x['difference'],yy,color=color,label=f'seed{seed}' if j==0 else None)
    for ax in axes:
        ax.axvline(0,color='black',linewidth=1); ax.grid(axis='x',alpha=.2); ax.legend()
        ax.set_yticks(range(3),['Binary full minus survival full','Survival clinical increment','Binary clinical increment'])
    axes[0].invert_yaxis(); axes[0].set_xlabel('Paired Brier90 difference (lower better)')
    axes[1].set_xlabel('Paired AUC90 difference (higher better)')
    fig.suptitle('Post-hoc limited optimization: 200 patient-clustered 95% intervals; reused test; unadjusted')
    fig.tight_layout(); fig.savefig(folder/'selected_test_paired.png',dpi=170); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--compact-dir',type=Path,default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--parent',type=Path,default=Path('training_runs/v6_90day_objective_2026-10-11'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--verify',action='store_true')
    args=p.parse_args(); (verify if args.verify else run)(args)


if __name__=='__main__': main()
