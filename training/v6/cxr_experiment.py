"""同H/同临床/同BERT，CheXbert内容与仅报告可用性三臂配对重训。"""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.compact_ablation import inputs
from training.v6.cxr_context import attach, mask_cxr, CXRAnchoredModel
from training.v6.train import fit, predict, write_json
from training.v6.group_ablation import evaluate
from training.v6.evaluation import GRID, metrics, public_metrics, paired_interval, calibration_bins
from training.v6.risk import predict_cif

ARMS=('without_cxr','availability_only','with_cxr')
LABELS={'without_cxr':'No CXR information','availability_only':'Report availability only','with_cxr':'CheXbert seven four-state findings'}


def prepared(data,ix,size,arm):
    from dataclasses import replace
    if arm not in ARMS: raise ValueError('unknown CXR arm')
    result=[]
    for start in range(0,len(ix),size):
        positions=ix[start:start+size]
        h,b,t,e=data.collate(positions,device='cuda')
        if arm=='without_cxr': b=mask_cxr(b,arm)
        if arm=='availability_only':
            # 固定类别仅用于容量匹配；无任何报告内容，仍只保留同一个报告可用标记。
            constant=torch.zeros_like(b.cxr_onehot).reshape(-1,7,4)
            constant[:,:,0]=b.cxr_present[:,None].to(constant.dtype)
            b=replace(b,cxr_onehot=constant.reshape(-1,28))
        result.append((positions,h,b,t,e))
    return result


def model_for(base,text_width,seed):
    torch.manual_seed(seed)
    return CXRAnchoredModel(copy.deepcopy(base),text_width).cuda()


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if args.seed!=42: raise ValueError('matched pilot uses seed42 H')
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2); started=time.perf_counter()
    data,compact,base,checkpoint=inputs(args.reference,args.compact_dir)
    cxr=attach(compact,args.cxr_dir)
    ix={f:np.flatnonzero(data.arrays['patient_splits']==f) for f in ('train','validation','test')}
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    contract={'seed':args.seed,'epochs':args.epochs,'batch_size':args.batch_size,'lr':3e-4,'warmup':5,'patience':10,
              'bootstrap':args.bootstrap,'arms':ARMS,'text':'same precomputed B four-field Bio_ClinicalBERT in every arm',
              'endpoint':'strict 491.21/J44.1, joint readmission/death, World A 365 days',
              'selection':'validation readmission integrated Brier 7-365','data_provenance':data.provenance,
              'compact_provenance':compact.provenance,'cxr_provenance':cxr.provenance,'baseline_checkpoint_sha256':sha256(checkpoint),
              'baseline_frozen':True,'CXR':'7 x four categorical states; 28 onehot, report mask; no report age or count inputs',
              'device':torch.cuda.get_device_name(0),'folds':{f:len(v) for f,v in ix.items()},
              'code_sha256':{name:sha256(Path(__file__).parent/name) for name in ('cxr_experiment.py','cxr_context.py','compact_model.py','compact_context_data.py','train.py','risk.py','data.py','evaluation.py')},
              'limits':['Single seed; historical test reused','Bootstrap omits retraining variance; no multiplicity adjustment',
                        'CheXbert training may overlap MIMIC-CXR; not independent external imaging representation',
                        'Matched no-image arm has new fusion architecture; old compact full is descriptive only',
                        'Labels are automatic observations, not clinician-confirmed diagnoses or causal explanations']}
    write_json(args.output/'contract.json',contract)
    fitted={}; initial=None
    for arm in ARMS:
        write_json(args.output/'status.json',{'status':'training','arm':arm})
        model=model_for(base,data.text_vectors.shape[1],args.seed)
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if initial is None: initial=state
        elif state.keys()!=initial.keys() or not all(torch.equal(v,initial[k]) for k,v in state.items()): raise ValueError('initial weights differ')
        train=prepared(cxr,ix['train'],args.batch_size,arm); val=prepared(cxr,ix['validation'],args.batch_size,arm)
        fitted[arm]=fit(model,data,train,val,ix['validation'],args.output/arm,epochs=args.epochs,lr=3e-4,
                        seed=args.seed,patience=10,warmup=5)
        gradients=[p.grad for p in model.encoder.xray.parameters() if p.grad is not None]
        fitted[arm]['xray_gradient_rms_last_batch']=float(torch.sqrt(sum(g.square().sum() for g in gradients)/sum(p.numel() for p in model.encoder.xray.parameters())))
        fitted[arm]['xray_max_weight_change']=max(float((v.detach().cpu()-initial[k]).abs().max()) for k,v in model.state_dict().items() if k.startswith('encoder.xray.'))
        del model,train,val
        torch.cuda.empty_cache()
    write_json(args.output/'status.json',{'status':'all_models_selected_before_test'})
    predictions={}; results={}
    for arm in ARMS:
        model=model_for(base,data.text_vectors.shape[1],args.seed)
        model.load_state_dict(torch.load(args.output/arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
        if not all(torch.equal(v,base.state_dict()[k]) for k,v in model.baseline.state_dict().items()): raise ValueError('H changed')
        results[arm]={k:v for k,v in fitted[arm].items() if k!='history'}; results[arm]['folds']={}
        for fold in ('validation','test'):
            batches=prepared(cxr,ix[fold],args.batch_size,arm)
            curve,detail=evaluate(data,ix[fold],model,batches,args.output/arm,fold)
            covered=cxr.arrays['cxr_present'][ix[fold]]
            detail['cxr_coverage_admissions']=int(covered.sum())
            detail['cxr_covered_readmission']=public_metrics(metrics(data.times[ix[fold]][covered],data.events[ix[fold]][covered],curve[covered,:,1],ordering=True))
            results[arm]['folds'][fold]=detail; predictions[arm,fold]=curve
    comparisons={}
    for fold in ('validation','test'):
        comparisons[fold]={}; selected=ix[fold]
        for candidate,reference in (('with_cxr','without_cxr'),('availability_only','without_cxr'),('with_cxr','availability_only')):
            label=candidate+'_minus_'+reference
            detail=paired_interval(data.arrays['subject_ids'][selected],data.times[selected],data.events[selected],
                                   predictions[candidate,fold][:,:,1],predictions[reference,fold][:,:,1],replicates=args.bootstrap,seed=args.seed)
            detail['mean_auc_difference']=results[candidate]['folds'][fold]['readmission']['mean_auc_7_365']-results[reference]['folds'][fold]['readmission']['mean_auc_7_365']
            comparisons[fold][label]=detail
    report={'status':'completed_chexbert_cxr_single_seed_paired','contract':contract,'initial_state_identical':True,
            'models':results,'comparisons':comparisons,'elapsed_seconds':time.perf_counter()-started,'limits':contract['limits']}
    write_json(args.output/'report.json',report); render(args.output,report)
    write_json(args.output/'status.json',{'status':report['status']})
    print(json.dumps({'status':report['status'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)


def verify(args):
    report=json.loads((args.output/'report.json').read_text(encoding='utf-8'))
    config=json.loads((args.output/'config.json').read_text(encoding='utf-8')); c=report['contract']
    for name,digest in c['code_sha256'].items():
        if sha256(Path(__file__).parent/name)!=digest: raise ValueError('run code changed')
    data,compact,base,checkpoint=inputs(Path(config['reference']),Path(config['compact_dir']))
    cxr=attach(compact,Path(config['cxr_dir']))
    if data.provenance!=c['data_provenance'] or compact.provenance!=c['compact_provenance'] or cxr.provenance!=c['cxr_provenance'] or sha256(checkpoint)!=c['baseline_checkpoint_sha256']:
        raise ValueError('input provenance mismatch')
    errors={}
    for arm in ARMS:
        path=args.output/arm/'best.pt'
        if sha256(path)!=report['models'][arm]['checkpoint_sha256']: raise ValueError('checkpoint hash changed')
        model=model_for(base,data.text_vectors.shape[1],c['seed'])
        model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
        if not all(torch.equal(v,base.state_dict()[k]) for k,v in model.baseline.state_dict().items()): raise ValueError('history changed')
        for fold in ('validation','test'):
            ix=np.flatnonzero(data.arrays['patient_splits']==fold)
            a,k,_,delta=predict(model,prepared(cxr,ix,c['batch_size'],arm)); curve=predict_cif(a,k,GRID)
            with np.load(args.output/arm/(fold+'_predictions.npz')) as saved:
                for key,val in (('hadm_ids',data.arrays['hadm_ids'][ix]),('subject_ids',data.arrays['subject_ids'][ix]),('events',data.events[ix]),('times',data.times[ix])):
                    if not np.array_equal(saved[key],val): raise ValueError('members/outcomes mismatch')
                for key,val in (('log_amplitude',a),('shape',k),('delta',delta),('curve',curve)):
                    if not np.allclose(saved[key],val,atol=2e-6,rtol=1e-5): raise ValueError('prediction mismatch')
                errors[arm+'/'+fold]=float(np.abs(saved['curve']-curve).max())
            actual=metrics(data.times[ix],data.events[ix],curve[:,:,1])
            for key in ('integrated_brier_7_365','mean_auc_7_365'):
                if abs(actual[key]-report['models'][arm]['folds'][fold]['readmission'][key])>1e-7: raise ValueError('metric mismatch')
    write_json(args.output/'independent_verification.json',{'status':'three_arms_independently_gpu_reloaded','max_curve_errors':errors,'history_unchanged':True,'report_sha256':sha256(args.output/'report.json')})
    print('three_arms_independently_gpu_reloaded',flush=True)


def render(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures'; folder.mkdir()
    for fold in ('validation','test'):
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        calfig,cal=plt.subplots(figsize=(6,5)); cal.plot([0,1],[0,1],'--',color='gray')
        for arm in ARMS:
            with np.load(root/arm/(fold+'_predictions.npz')) as saved:
                m=metrics(saved['times'],saved['events'],saved['curve'][:,:,1]); bins=calibration_bins(saved['times'],saved['events'],saved['curve'][:,:,1])
            axes[0].plot(GRID[7:],m['auc_curve'][7:],label=LABELS[arm]); axes[1].plot(GRID[7:],m['brier_curve'][7:])
            cal.plot([b['predicted'] for b in bins],[b['observed'] for b in bins],'o-',label=LABELS[arm])
        axes[0].set_title('Strict readmission AUC(t)'); axes[1].set_title('Readmission Brier(t)')
        for ax in axes: ax.set_xlabel('Days after discharge'); ax.grid(alpha=.2)
        axes[0].legend(fontsize=8); fig.suptitle(f'{fold}: true CheXbert, matched retraining, one seed, World A')
        fig.tight_layout(); fig.savefig(folder/(fold+'_auc_brier.png'),dpi=180); plt.close(fig)
        cal.set(xlabel='Predicted 90-day cumulative incidence',ylabel='Observed World A fraction',title=f'{fold}: 90-day calibration')
        cal.legend(fontsize=7); cal.grid(alpha=.2); calfig.tight_layout(); calfig.savefig(folder/(fold+'_calibration.png'),dpi=180); plt.close(calfig)
    fig,ax=plt.subplots(figsize=(9,4))
    for arm in ARMS:
        history=json.loads((root/arm/'history.json').read_text())
        ax.plot([h['epoch'] for h in history],[h['validation_integrated_brier'] for h in history],label=LABELS[arm])
    ax.set(xlabel='Epoch',ylabel='Validation integrated Brier',title='Validation-only checkpoint selection')
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(folder/'learning_curves.png',dpi=180); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--compact-dir',type=Path,default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--cxr-dir',type=Path,default=Path('patient_data/processed/v6_chexbert_cxr_2026-10-09'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--batch-size',type=int,default=512); p.add_argument('--bootstrap',type=int,default=200)
    p.add_argument('--verify',action='store_true'); args=p.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.bootstrap<20: p.error('invalid budget')
    verify(args) if args.verify else run(args)
