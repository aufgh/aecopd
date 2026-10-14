"""先验证扩展冻结基线，再同一基线下探索架构/组合/候选原文。"""
import argparse
import copy
from dataclasses import replace
import gzip
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.compact_ablation import inputs
from training.v6.compact_model import CompactAnchoredModel
from training.v6.exploration_model import ExpandedBaseline, expanded_inputs, ExplorationBatch, ExplorationModel
from training.v6.model import ConstantCorrection
from training.v6.train import fit, predict, write_json
from training.v6.group_ablation import evaluate
from training.v6.risk import predict_cif
from training.v6.evaluation import GRID, paired_interval, calibration_bins, metrics


def omit_current(batch):
    b = batch.clinical
    return replace(batch, clinical=replace(b, current_values=torch.zeros_like(b.current_values),
                                          current_masks=torch.zeros_like(b.current_masks)))


def load_extra(data, combinations, candidate=None):
    m = json.loads((combinations/'manifest.json').read_text())
    if m['status'] != 'combinations_source_checked' or sha256(combinations/'combinations.npz') != m['output_sha256']:
        raise ValueError('combination lineage changed')
    with np.load(combinations/'combinations.npz') as z: a = {k:z[k] for k in z.files}
    for k in ('hadm_ids','subject_ids','patient_splits'):
        if not np.array_equal(a[k], data.arrays[k]): raise ValueError('combination member/split mismatch')
    if a['values'].shape != (19401,5,9) or not np.isfinite(a['values']).all(): raise ValueError('invalid combination shape')
    refs = [[] for _ in data.history]; vectors = None
    provenance = dict(combinations_manifest_sha256=sha256(combinations/'manifest.json'))
    if candidate:
        cm = json.loads((candidate/'manifest.json').read_text())
        if cm['status'] != 'candidate_bert_complete' or cm['model'] != 'Bio_ClinicalBERT': raise ValueError('candidate not complete')
        for k,v in cm['output_sha256'].items():
            if sha256(candidate/k) != v: raise ValueError('candidate encoding changed')
        vectors = np.load(candidate/'vectors.float16.npy', mmap_mode='r')
        if vectors.shape != (cm['rows'],768) or not np.isfinite(vectors).all(): raise ValueError('candidate vectors invalid')
        by_h = {str(h):i for i,h in enumerate(data.arrays['hadm_ids'])}
        count=0
        with gzip.open(candidate/'chunks.jsonl.gz','rt',encoding='utf-8') as f:
            for count,line in enumerate(f, start=1):
                r=json.loads(line); position=by_h[r['hadm_id']]
                if (int(r['subject_id']) != data.arrays['subject_ids'][position] or r['patient_split'] != data.arrays['patient_splits'][position]
                        or r['embedding_row'] != count-1 or not 0 <= r['field'] < 4): raise ValueError('candidate index mismatch')
                refs[position].append((r['field'], count-1))
        if count != len(vectors): raise ValueError('candidate row mismatch')
        provenance['candidate_manifest_sha256']=sha256(candidate/'manifest.json')
    return a, vectors, refs, provenance


def knots(data):
    labs=data.arrays['lab_axis_inputs']; train=data.arrays['patient_splits']=='train'
    result=np.zeros((22,6,7), np.float32)
    for a in range(22):
        for j in range(6):
            valid=train&(labs[:,a,6]>0)
            if j in (3,5): valid &= labs[:,a,7]>0
            if valid.any(): result[a,j]=np.quantile(labs[valid,a,j],np.linspace(0,1,7))
    return torch.tensor(result)


def batches(data, compact, indices, size, *, expanded=False, text='bert', combination='none', extra=None):
    result=[]
    for start in range(0,len(indices),size):
        pos=indices[start:start+size]; h,b,t,e=compact.collate(pos,device='cuda')
        if expanded: h=expanded_inputs(h,b); b=omit_current(b)
        if text not in ('bert','fused'):
            c=b.clinical; b=replace(b,clinical=replace(c,snippet_vectors=c.snippet_vectors[:0],snippet_index=c.snippet_index[:0]))
        if extra is not None:
            a,vectors,refs,_=extra
            values=a['values'][pos].copy()
            if combination=='none': values[:]=0
            elif combination=='components': values[:,3:]=0
            elif combination=='availability': values[:,:,:6]=0; values[:,:,7:]=0
            elif combination!='ratios': raise ValueError('unknown combination arm')
            cv,ci=[],[]
            if text in ('screen','fused'):
                if vectors is None: raise ValueError('candidate missing')
                for slot, p in enumerate(pos):
                    for field,row in refs[p]: cv.append(vectors[row]); ci.append((slot,field))
            b=ExplorationBatch(b,torch.as_tensor(values,device='cuda'),
                               torch.as_tensor(np.asarray(cv,np.float32).reshape(-1,768),device='cuda'),
                               torch.as_tensor(np.asarray(ci,np.int64).reshape(-1,2),device='cuda'))
        result.append((pos,h,b,t,e))
    return result


def configurations(phase):
    if phase=='baseline':
        return {'legacy_full':dict(expanded=False), 'expanded_only':dict(expanded=True,kind='baseline'),
                'expanded_constant':dict(expanded=True,kind='constant'), 'expanded_full':dict(expanded=True)}
    configs={'pool128':{}, 'pool64':dict(width=64), 'pool256':dict(width=256), 'hidden64':dict(hidden=64),
             'retained':dict(mode='retained'), 'set4':dict(mode='set'), 'piecewise':dict(mode='piecewise'),
             'availability':dict(combination='availability'), 'components':dict(combination='components'),
             'nlr_sii':dict(combination='ratios'), 'no_text':dict(text='none'),
             'screen_only':dict(text='screen'), 'bert_screen':dict(text='fused')}
    # 对不同骨干给共同两种学习率，验证折选，测试不参与选优。
    for name in ('pool128','retained','set4','piecewise'):
        configs[name+'_lr1e4']={**configs[name], 'lr':1e-4}
    return configs


def model_for(config, base, text_width, seed, quantiles=None, phase='baseline'):
    torch.manual_seed(seed)
    if config.get('kind')=='baseline': return copy.deepcopy(base).cuda()
    if config.get('kind')=='constant': return ConstantCorrection(copy.deepcopy(base)).cuda()
    if phase=='baseline': return CompactAnchoredModel(copy.deepcopy(base),text_width).cuda()
    return ExplorationModel(copy.deepcopy(base),text_width,width=config.get('width',128),hidden=config.get('hidden',32),
                            mode=config.get('mode','pooled'),knots=quantiles if config.get('mode')=='piecewise' else None).cuda()


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2); started=time.perf_counter()
    data,compact,hist,checkpoint=inputs(args.reference,args.compact_dir)
    ix={f:np.flatnonzero(data.arrays['patient_splits']==f) for f in ('train','validation','test')}
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    expanded=ExpandedBaseline(hist).cuda(); baseline_fit=None; extra=None; quantiles=None
    if args.phase=='baseline':
        baseline_fit=fit(expanded,data,batches(data,compact,ix['train'],args.batch_size,expanded=True,text='none'),
                         batches(data,compact,ix['validation'],args.batch_size,expanded=True,text='none'),ix['validation'],
                         args.output/'expanded_baseline_fit',epochs=120,lr=.005,seed=args.seed,patience=20)
    else:
        br=json.loads((args.baseline_run/'report.json').read_text())
        if br['contract']['data_provenance']!=data.provenance or br['contract']['compact_provenance']!=compact.provenance:
            raise ValueError('baseline round lineage differs')
        path=args.baseline_run/'expanded_baseline_fit/best.pt'
        if sha256(path)!=br['expanded_baseline_fit']['checkpoint_sha256']: raise ValueError('expanded baseline changed')
        expanded.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
        extra=load_extra(data,args.combinations,args.candidate); quantiles=knots(data)
    configs=configurations(args.phase)
    contract=dict(phase=args.phase,seed=args.seed,epochs=args.epochs,batch_size=args.batch_size,bootstrap=args.bootstrap,
                  data_provenance=data.provenance,compact_provenance=compact.provenance,
                  baseline_input='H6 + D/P14 values + D/P14 masks; residual masks D/P',
                  selection='validation readmission IBS 7-365 only; repeated test not blind',
                  expanded_baseline_source=str(args.baseline_run) if args.phase=='explore' else 'trained this round from old H',
                  extra_provenance=extra[3] if extra else {},arms=configs,
                  code_sha256={str(p):sha256(p) for p in [Path('training/v6')/f for f in
                               ('architecture_experiment.py','exploration_model.py','train.py','compact_model.py','risk.py','evaluation.py','data.py')]},
                  limits=['Single-seed exploratory; many candidates, no multiplicity adjustment',
                          'World A observed-record risk, not full external follow-up',
                          'Different architectures/capacities change initialization; same recipe is not exhaustive tuning',
                          'Candidate source audit not semantic gold; matching diagnosis proxies may remain in text'])
    write_json(args.output/'contract.json',contract); fitted={}; initial={}
    for arm,cfg in configs.items():
        is_expanded=cfg.get('expanded',True)
        base=expanded if is_expanded else hist
        model=model_for(cfg,base,768,args.seed,quantiles,args.phase)
        key=(cfg.get('width',128),cfg.get('hidden',32),cfg.get('mode','pooled'),is_expanded,cfg.get('kind'))
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if key in initial:
            if state.keys()!=initial[key].keys() or not all(torch.equal(v,initial[key][k]) for k,v in state.items()):
                raise ValueError('same-architecture initial state mismatch')
        else: initial[key]=state
        write_json(args.output/'status.json',dict(status='training',arm=arm))
        def prep(fold): return batches(data,compact,ix[fold],args.batch_size,expanded=is_expanded,
                                      text=cfg.get('text','bert'),combination=cfg.get('combination','none'),extra=extra)
        train,val=prep('train'),prep('validation')
        if cfg.get('kind')=='baseline':
            # 已选择的B再落盘，不重复训练。
            directory=args.output/arm; directory.mkdir()
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(dict(state_dict=state,best_epoch=baseline_fit['best_epoch'],seed=args.seed),directory/'best.pt')
            fitted[arm]={k:v for k,v in baseline_fit.items() if k!='history'}
            fitted[arm]['checkpoint_sha256']=sha256(directory/'best.pt')
        else:
            fitted[arm]=fit(model,data,train,val,ix['validation'],args.output/arm,epochs=args.epochs,
                            lr=.005 if cfg.get('kind')=='constant' else cfg.get('lr',3e-4),seed=args.seed,
                            patience=10,warmup=0 if cfg.get('kind')=='constant' else 5)
        fitted[arm]['parameters_total']=sum(p.numel() for p in model.parameters())
        fitted[arm]['parameters_trainable']=sum(p.numel() for p in model.parameters() if p.requires_grad)
        del train,val,model; torch.cuda.empty_cache()
    # 此处只根据验证IBS记录赢家；下方才统一看历史测试。
    chosen=min(fitted,key=lambda a:fitted[a]['best_validation_IBS'])
    write_json(args.output/'validation_selection.json',dict(selected_arm=chosen,criterion=contract['selection']))
    write_json(args.output/'status.json',dict(status='all_models_selected_before_test',validation_selected=chosen))
    results={}; predictions={}
    for arm,cfg in configs.items():
        is_expanded=cfg.get('expanded',True); base=expanded if is_expanded else hist
        model=model_for(cfg,base,768,args.seed,quantiles,args.phase)
        model.load_state_dict(torch.load(args.output/arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
        if hasattr(model,'baseline') and not all(torch.equal(v,base.state_dict()[k]) for k,v in model.baseline.state_dict().items()):
            raise ValueError('frozen baseline changed')
        results[arm]={k:v for k,v in fitted[arm].items() if k!='history'}; results[arm]['folds']={}
        for fold in ('validation','test'):
            bs=batches(data,compact,ix[fold],args.batch_size,expanded=is_expanded,text=cfg.get('text','bert'),
                       combination=cfg.get('combination','none'),extra=extra)
            curve,detail=evaluate(data,ix[fold],model,bs,args.output/arm,fold)
            detail['calibration90']=calibration_bins(data.times[ix[fold]],data.events[ix[fold]],curve[:,:,1])
            if extra:
                keep=extra[0]['values'][ix[fold],3,6]>0
                detail['nlr_observed_subgroup']= {k:v for k,v in metrics(data.times[ix[fold]][keep],data.events[ix[fold]][keep],curve[keep,:,1],ordering=True).items() if not k.endswith('_curve')}
            results[arm]['folds'][fold]=detail; predictions[arm,fold]=curve
            del bs
    pairs= [('expanded_full','legacy_full'),('expanded_full','expanded_constant'),('expanded_constant','expanded_only')] if args.phase=='baseline' else [(a,'pool128') for a in configs if a!='pool128']+[('nlr_sii','components'),('components','availability'),('bert_screen','screen_only')]
    comparisons={}
    for fold in ('validation','test'):
        comparisons[fold]={}; idx=ix[fold]
        for left,right in pairs:
            comparisons[fold][left+'_minus_'+right]=paired_interval(data.arrays['subject_ids'][idx],data.times[idx],data.events[idx],
                predictions[left,fold][:,:,1],predictions[right,fold][:,:,1],replicates=args.bootstrap,seed=args.seed)
    report=dict(status='completed_'+args.phase+'_exploration_single_seed',contract=contract,models=results,comparisons=comparisons,
                validation_selected=chosen,expanded_baseline_fit={k:v for k,v in baseline_fit.items() if k!='history'} if baseline_fit else None,
                elapsed_seconds=time.perf_counter()-started)
    write_json(args.output/'report.json',report); render(args.output,report)
    write_json(args.output/'status.json',dict(status=report['status'],seconds=report['elapsed_seconds']))
    print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'],validation_selected=chosen)),flush=True)


def verify(args):
    report=json.loads((args.output/'report.json').read_text()); config=json.loads((args.output/'config.json').read_text())
    for path,digest in report['contract']['code_sha256'].items():
        if sha256(Path(path))!=digest: raise ValueError('code changed: '+path)
    data,compact,hist,_=inputs(Path(config['reference']),Path(config['compact_dir']))
    if data.provenance!=report['contract']['data_provenance'] or compact.provenance!=report['contract']['compact_provenance']: raise ValueError('inputs changed')
    base=ExpandedBaseline(hist).cuda(); directory=args.output if config['phase']=='baseline' else Path(config['baseline_run'])
    base.load_state_dict(torch.load(directory/'expanded_baseline_fit/best.pt',map_location='cuda',weights_only=True)['state_dict'])
    extra=load_extra(data,Path(config['combinations']),Path(config['candidate'])) if config['phase']=='explore' else None
    if extra and extra[3]!=report['contract']['extra_provenance']: raise ValueError('extra input changed')
    q=knots(data); errors={}
    for arm,cfg in report['contract']['arms'].items():
        b=base if cfg.get('expanded',True) else hist
        model=model_for(cfg,b,768,config['seed'],q,config['phase'])
        path=args.output/arm/'best.pt'
        if sha256(path)!=report['models'][arm]['checkpoint_sha256']: raise ValueError('checkpoint changed')
        model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
        if hasattr(model,'baseline') and not all(torch.equal(v,b.state_dict()[k]) for k,v in model.baseline.state_dict().items()): raise ValueError('baseline changed')
        for fold in ('validation','test'):
            ix=np.flatnonzero(data.arrays['patient_splits']==fold)
            a,k,_,delta=predict(model,batches(data,compact,ix,config['batch_size'],expanded=cfg.get('expanded',True),
                       text=cfg.get('text','bert'),combination=cfg.get('combination','none'),extra=extra))
            curve=predict_cif(a,k,GRID,step_days=.25)
            with np.load(args.output/arm/(fold+'_predictions.npz')) as z:
                for key,v in (('hadm_ids',data.arrays['hadm_ids'][ix]),('subject_ids',data.arrays['subject_ids'][ix]),('events',data.events[ix]),('times',data.times[ix])):
                    if not np.array_equal(z[key],v): raise ValueError('saved identity/labels differ')
                for key,v in (('log_amplitude',a),('shape',k),('curve',curve),('delta',delta)):
                    if not np.allclose(z[key],v,atol=2e-6,rtol=1e-5): raise ValueError('reload failure: '+arm+'/'+key)
                errors[arm+'/'+fold]=float(np.abs(z['curve']-curve).max())
            actual=metrics(data.times[ix],data.events[ix],curve[:,:,1])
            for key in ('integrated_brier_7_365','mean_auc_7_365'):
                if abs(actual[key]-report['models'][arm]['folds'][fold]['readmission'][key])>1e-7: raise ValueError('metric mismatch')
    write_json(args.output/'independent_verification.json',dict(status='all_arms_gpu_reloaded',errors=errors,baseline_unchanged=True,report_sha256=sha256(args.output/'report.json')))
    print('all_arms_gpu_reloaded',flush=True)


def render(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures'; folder.mkdir()
    arms=list(report['models'])
    for fold in ('validation','test'):
        fig,axes=plt.subplots(1,2,figsize=(15, max(4,len(arms)*.3)),sharey=True)
        for i,arm in enumerate(arms):
            m=report['models'][arm]['folds'][fold]['readmission']
            axes[0].scatter(m['integrated_brier_7_365'],i)
            axes[1].scatter(m['points']['90']['auc'],i)
        for ax in axes: ax.set_yticks(range(len(arms)),arms,fontsize=8); ax.grid(axis='x',alpha=.2)
        axes[0].invert_yaxis(); axes[0].set_xlabel('Integrated Brier (lower better)'); axes[1].set_xlabel('90-day AUC (higher better)')
        fig.suptitle(f'{fold}: fixed strict AECOPD endpoint; single seed; reused test'); fig.tight_layout()
        fig.savefig(folder/(fold+'_overview.png'),dpi=160); plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(14,4))
        comparator='expanded_full' if 'expanded_full' in arms else 'pool128'
        show=list(dict.fromkeys([comparator,report['validation_selected']]+(['nlr_sii','components','screen_only','bert_screen'] if 'pool128' in arms else ['legacy_full','expanded_only'])))
        for arm in show:
            with np.load(root/arm/(fold+'_predictions.npz')) as z:
                curve=z['curve'][:,:,1]; detail=metrics(z['times'],z['events'],curve)
                axes[0].plot(GRID[7:],detail['auc_curve'][7:],label=arm)
                bins=calibration_bins(z['times'],z['events'],curve)
                axes[1].plot([r['predicted'] for r in bins],[r['observed'] for r in bins],marker='o',label=arm)
        axes[0].set(xlabel='Days after discharge',ylabel='AUC(t)'); axes[1].plot([0,1],[0,1],'k--')
        axes[1].set(xlabel='Predicted 90-day risk',ylabel='Observed record fraction',xlim=(0,.6),ylim=(0,.6))
        for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(folder/(fold+'_auc_calibration.png'),dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for arm in arms:
        path=root/arm/'history.json'
        if path.exists():
            h=json.loads(path.read_text()); ax.plot([r['epoch'] for r in h],[r['validation_integrated_brier'] for r in h],label=arm)
    ax.set(xlabel='Epoch',ylabel='Validation integrated Brier'); ax.legend(fontsize=6,ncol=3); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder/'training_validation.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('baseline','explore'),default='baseline')
    p.add_argument('--reference',type=Path,default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--compact-dir',type=Path,default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--baseline-run',type=Path,default=Path('training_runs/v6_expanded_baseline_seed42_2026-10-10'))
    p.add_argument('--combinations',type=Path,default=Path('patient_data/processed/v6_cbc_combinations_2026-10-10'))
    p.add_argument('--candidate',type=Path,default=Path('patient_data/processed/v6_screen_candidate_bert_2026-10-10'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=60); p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--bootstrap',type=int,default=200); p.add_argument('--verify',action='store_true')
    a=p.parse_args()
    if a.seed!=42 or a.epochs<1 or a.batch_size<1 or a.bootstrap<20: p.error('pilot uses explicit seed42 H and positive budget')
    verify(a) if a.verify else run(a)
