"""三条临床事实通道同容量配对重训；全量未完成不得训练完整臂。"""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.compact_ablation import inputs
from training.v6.clinical_facts_context import attach,mask,FactsAnchoredModel,CHANNELS
from training.v6.train import fit,predict,write_json
from training.v6.group_ablation import evaluate
from training.v6.evaluation import GRID,metrics,paired_interval,calibration_bins
from training.v6.risk import predict_cif

ALL_ARMS=('without_new','availability_only','cxr_only','ct_only','discharge_only','full')
LABELS={'without_new':'No new clinical facts','availability_only':'Source availability only','cxr_only':'All CXR trajectories',
        'ct_only':'Chest CT morphology facts','discharge_only':'Discharge clinical facts','full':'All three fact channels'}

def prepared(data,ix,size,arm):
    result=[]
    for start in range(0,len(ix),size):
        pos=ix[start:start+size]; h,b,t,e=data.collate(pos,device='cuda')
        result.append((pos,h,mask(b,arm),t,e))
    return result

def model_for(base,width):
    torch.manual_seed(42); return FactsAnchoredModel(copy.deepcopy(base),width).cuda()

def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2); started=time.perf_counter()
    data,compact,base,checkpoint=inputs(args.reference,args.compact_dir)
    facts=attach(compact,args.cxr_dir,args.facts_dir)
    arms=ALL_ARMS if args.facts_dir else ALL_ARMS[:3]
    ix={f:np.flatnonzero(data.arrays['patient_splits']==f) for f in ('train','validation','test')}
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    contract={'seed':42,'epochs':args.epochs,'batch_size':512,'lr':3e-4,'warmup':5,'patience':10,'bootstrap':args.bootstrap,
              'arms':arms,'available_channels':facts.available_channels,'baseline_frozen':True,'zero_initialized_delta':True,
              'baseline_checkpoint_sha256':sha256(checkpoint),'data_provenance':data.provenance,
              'compact_provenance':compact.provenance,'facts_provenance':facts.provenance,
              'text':'same existing four-field BERT in all arms; no new text encoding',
              'endpoint':'strict 491.21/J44.1, joint first readmission or death, World A, 365 days',
              'selection':'validation readmission integrated Brier 7-365',
              'code_sha256':{n:sha256(Path(__file__).parent/n) for n in ('facts_experiment.py','clinical_facts_context.py','compact_model.py','train.py','data.py','risk.py','evaluation.py')},
              'limits':['Single seed, reused historical test, no multiplicity correction',
                        'Patient bootstrap omits training variance','Source masks are not normality',
                        'Automated facts not causal treatment effects or expert diagnostic gold standard',
                        'Full final-summary features retrospective, not proven available in real time']}
    write_json(args.output/'contract.json',contract); fitted={}; initial=None
    for arm in arms:
        write_json(args.output/'status.json',{'status':'training','arm':arm})
        model=model_for(base,data.text_vectors.shape[1])
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if initial is None: initial=state
        elif not all(torch.equal(v,initial[k]) for k,v in state.items()): raise ValueError('initial weights differ')
        train=prepared(facts,ix['train'],512,arm); val=prepared(facts,ix['validation'],512,arm)
        fitted[arm]=fit(model,data,train,val,ix['validation'],args.output/arm,epochs=args.epochs,lr=3e-4,seed=42,patience=10,warmup=5)
        fitted[arm]['fact_branch_max_weight_changes']={name:max(float((v.detach().cpu()-initial[k]).abs().max()) for k,v in model.state_dict().items()
                                                               if k.startswith('encoder.fact_branches.'+str(j)+'.')) for j,name in enumerate(CHANNELS)}
        del model,train,val; torch.cuda.empty_cache()
    write_json(args.output/'status.json',{'status':'all_models_selected_before_test'})
    results={}; predictions={}
    for arm in arms:
        model=model_for(base,data.text_vectors.shape[1]); model.load_state_dict(torch.load(args.output/arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
        if not all(torch.equal(v,base.state_dict()[k]) for k,v in model.baseline.state_dict().items()): raise ValueError('H changed')
        results[arm]={k:v for k,v in fitted[arm].items() if k!='history'}; results[arm]['folds']={}
        for fold in ('validation','test'):
            curve,detail=evaluate(data,ix[fold],model,prepared(facts,ix[fold],512,arm),args.output/arm,fold)
            results[arm]['folds'][fold]=detail; predictions[arm,fold]=curve
    comparisons={}
    for fold in ('validation','test'):
        selected=ix[fold]; comparisons[fold]={}
        for candidate in arms[1:]:
            detail=paired_interval(data.arrays['subject_ids'][selected],data.times[selected],data.events[selected],
                                   predictions[candidate,fold][:,:,1],predictions['without_new',fold][:,:,1],replicates=args.bootstrap,seed=42)
            detail['mean_auc_difference']=results[candidate]['folds'][fold]['readmission']['mean_auc_7_365']-results['without_new']['folds'][fold]['readmission']['mean_auc_7_365']
            comparisons[fold][candidate+'_minus_without_new']=detail
        if 'full' in arms:
            comparisons[fold]['full_minus_availability_only']=paired_interval(data.arrays['subject_ids'][selected],data.times[selected],data.events[selected],
                predictions['full',fold][:,:,1],predictions['availability_only',fold][:,:,1],replicates=args.bootstrap,seed=42)
    report={'status':'completed_three_channel_paired' if args.facts_dir else 'completed_cxr_trajectory_only_paired',
            'initial_weights_identical':True,'contract':contract,'models':results,'comparisons':comparisons,
            'elapsed_seconds':time.perf_counter()-started}
    write_json(args.output/'report.json',report); render(args.output,report)
    write_json(args.output/'status.json',{'status':report['status']}); print(json.dumps({'status':report['status'],'seconds':report['elapsed_seconds']}),flush=True)

def verify(args):
    config=json.loads((args.output/'config.json').read_text()); report=json.loads((args.output/'report.json').read_text()); c=report['contract']
    data,compact,base,checkpoint=inputs(Path(config['reference']),Path(config['compact_dir']))
    facts=attach(compact,Path(config['cxr_dir']),Path(config['facts_dir']) if config['facts_dir'] else None)
    if facts.provenance!=c['facts_provenance'] or sha256(checkpoint)!=c['baseline_checkpoint_sha256']: raise ValueError('input changed')
    errors={}
    for arm in c['arms']:
        path=args.output/arm/'best.pt'
        if sha256(path)!=report['models'][arm]['checkpoint_sha256']: raise ValueError('checkpoint changed')
        model=model_for(base,data.text_vectors.shape[1]); model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
        if not all(torch.equal(v,base.state_dict()[k]) for k,v in model.baseline.state_dict().items()): raise ValueError('H changed')
        for fold in ('validation','test'):
            selected=np.flatnonzero(data.arrays['patient_splits']==fold)
            a,k,_,delta=predict(model,prepared(facts,selected,512,arm)); curve=predict_cif(a,k,GRID)
            with np.load(args.output/arm/(fold+'_predictions.npz')) as saved:
                if not np.array_equal(saved['hadm_ids'],data.arrays['hadm_ids'][selected]) or not np.array_equal(saved['events'],data.events[selected]): raise ValueError('members or labels differ')
                if not np.allclose(saved['curve'],curve,atol=2e-6,rtol=1e-5): raise ValueError('GPU prediction mismatch')
                errors[arm+'/'+fold]=float(np.max(np.abs(saved['curve']-curve)))
            actual=metrics(data.times[selected],data.events[selected],curve[:,:,1])
            for key in ('integrated_brier_7_365','mean_auc_7_365'):
                if abs(actual[key]-report['models'][arm]['folds'][fold]['readmission'][key])>1e-7: raise ValueError('metrics differ')
    write_json(args.output/'independent_verification.json',{'status':'all_arms_gpu_reloaded','history_unchanged':True,'max_curve_errors':errors})
    print('all_arms_gpu_reloaded',flush=True)

def render(root,report):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures'; folder.mkdir()
    arms=report['contract']['arms']
    for fold in ('validation','test'):
        fig,axes=plt.subplots(1,2,figsize=(13,4)); calfig,cal=plt.subplots(figsize=(6,5)); cal.plot([0,1],[0,1],'--',color='gray')
        for arm in arms:
            with np.load(root/arm/(fold+'_predictions.npz')) as s:
                m=metrics(s['times'],s['events'],s['curve'][:,:,1]); bins=calibration_bins(s['times'],s['events'],s['curve'][:,:,1])
            axes[0].plot(GRID[7:],m['auc_curve'][7:],label=LABELS[arm]); axes[1].plot(GRID[7:],m['brier_curve'][7:])
            cal.plot([b['predicted'] for b in bins],[b['observed'] for b in bins],'o-',label=LABELS[arm])
        for ax in axes: ax.set_xlabel('Days after discharge'); ax.grid(alpha=.2)
        axes[0].set_title('Strict AECOPD readmission AUC(t)'); axes[1].set_title('Readmission Brier(t)'); axes[0].legend(fontsize=8)
        fig.suptitle(f'{fold}: matched retraining, one seed, World A'); fig.tight_layout(); fig.savefig(folder/(fold+'_auc_brier.png'),dpi=180); plt.close(fig)
        cal.set(xlabel='Predicted 90-day incidence',ylabel='Observed World A fraction',title=f'{fold}: calibration')
        cal.legend(fontsize=7); cal.grid(alpha=.2); calfig.tight_layout(); calfig.savefig(folder/(fold+'_calibration.png'),dpi=180); plt.close(calfig)
    fig,axes=plt.subplots(1,2,figsize=(12,4)); base=report['models']['without_new']['folds']['validation']['readmission']
    for i,arm in enumerate(arms[1:]):
        m=report['models'][arm]['folds']['validation']['readmission']
        axes[0].scatter(m['integrated_brier_7_365']-base['integrated_brier_7_365'],i,marker='D')
        axes[1].scatter(m['mean_auc_7_365']-base['mean_auc_7_365'],i,marker='D')
    for ax in axes: ax.axvline(0,color='gray'); ax.set_yticks(range(len(arms)-1),[LABELS[a] for a in arms[1:]]); ax.grid(alpha=.2)
    axes[0].set_title('Integrated Brier: added - no new (lower better)'); axes[1].set_title('Mean AUC(t): added - no new (higher better)')
    fig.suptitle('Validation paired differences; single seed; no multiplicity adjustment'); fig.tight_layout(); fig.savefig(folder/'paired_validation.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for arm in arms:
        history=json.loads((root/arm/'history.json').read_text()); ax.plot([h['epoch'] for h in history],[h['validation_integrated_brier'] for h in history],label=LABELS[arm])
    ax.set(xlabel='Epoch',ylabel='Validation integrated Brier',title='Validation-only checkpoint selection'); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder/'learning_curves.png',dpi=180); plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--compact-dir',type=Path,default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--cxr-dir',type=Path,default=Path('patient_data/processed/v6_cxr_trajectory_2026-10-10_r2'))
    p.add_argument('--facts-dir',type=Path); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=60); p.add_argument('--bootstrap',type=int,default=200); p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    if args.epochs<1 or args.bootstrap<20: p.error('invalid budget')
    verify(args) if args.verify else run(args)
