"""相同B四字段冻结向量的BERT/Qwen配对重训；既有源代码不改。"""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from patient_data.v6_lab_state import AXES, NUMERIC, FLAGS, MED_GROUPS
from patient_data.compact_multisource import MED_CHANNELS, MICRO_NAMES, SUPPORT_NAMES
from training.v6.data import load, HISTORY_NAMES, CURRENT_NAMES
from training.v6.compact_context_data import attach
from training.v6.compact_ablation import inputs
from training.v6.architecture_experiment import batches
from training.v6.horizon_objective import (labels90, binary_metrics, paired90, ranked_curves,
    make_model, restore, fit90, predict90)
from training.v6.evaluation import GRID, metrics, public_metrics, paired_interval
from training.v6.risk import predict_cif
from training.v6.train import write_json

ARMS = ('bert', 'qwen', 'none_768', 'none_2560')
PAIRS = (('qwen','bert'), ('bert','none_768'), ('qwen','none_768'))
PROTOCOL = Path('docs/V6_QWEN_COMPARISON_PROTOCOL_2026-10-11.md')


def shared_parameters(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()
            if not k.startswith('encoder.text_projection.')}


def checked_inputs(args):
    parent=json.loads((args.parent/'report.json').read_text())
    proof=json.loads((args.parent/'independent_verification.json').read_text())
    if proof['report_sha256'] != sha256(args.parent/'report.json'):
        raise ValueError('parent report not verified')
    for p,digest in parent['contract']['code_sha256'].items():
        if sha256(Path(p)) != digest: raise ValueError('parent source changed: '+p)
    data,compact,hist,_=inputs(args.reference,args.compact_dir)
    if data.provenance != parent['contract']['data_provenance'] or compact.provenance != parent['contract']['compact_provenance']:
        raise ValueError('parent input changed')
    cfg=json.loads((args.reference/'config.json').read_text())
    field_dir=Path(cfg['field_dir'])
    qwen=load(Path(cfg['snapshot']),Path(cfg['state_dir']),Path(cfg['labels']),
              field_dir=field_dir,vectors_path=field_dir/'qwen/qwen3_embedding.float16.npy')
    left,right=copy.deepcopy(data.provenance),copy.deepcopy(qwen.provenance)
    left.pop('text'); right.pop('text')
    if (left != right or data.text_refs != qwen.text_refs
            or data.text_vectors.shape != (76974,768) or qwen.text_vectors.shape != (76974,2560)):
        raise ValueError('encoder content/index/input mismatch')
    for key in ('manifest_sha256','rows','text_admissions'):
        if data.provenance['text'][key] != qwen.provenance['text'][key]:
            raise ValueError('shared text mismatch')
    audit={}
    for name,item in (('bert',data),('qwen',qwen)):
        norms=np.concatenate([np.linalg.norm(np.asarray(item.text_vectors[j:j+4096],np.float32),axis=1)
                              for j in range(0,len(item.text_vectors),4096)])
        if not np.isfinite(norms).all() or not np.all((norms>.99)&(norms<1.01)):
            raise ValueError('expected complete L2 vectors')
        audit[name]=dict(**item.provenance['text'],norm_min=float(norms.min()),norm_max=float(norms.max()))
    return {'bert':data,'qwen':qwen}, {'bert':compact,'qwen':attach(qwen,args.compact_dir)}, hist,parent,audit


def prepared(data,compact,idx,size,arm):
    return batches(data,compact,idx,size,expanded=True,text='none' if arm.startswith('none') else 'bert')


def arm_source(arm):
    return 'qwen' if arm in ('qwen','none_2560') else 'bert'


def new_model(args,seed,arm,hist,prevalence):
    base=restore(args.parent,seed,'survival_base',hist,prevalence)
    return make_model('survival_full',base,2560 if arm_source(arm)=='qwen' else 768,seed)


def read_prediction(root,seed,arm,fold):
    with np.load(root/f'seed{seed}'/arm/(fold+'_predictions.npz')) as z:
        return {k:z[k] for k in z.files}


def assert_empty_equal(a,b):
    keys=('epoch','train_objective','validation_objective','validation_brier90','validation_auc90','validation_ap')
    if len(a)!=len(b): raise ValueError('empty-text training length differs')
    for x,y in zip(a,b,strict=True):
        if any(x[k]!=y[k] for k in keys): raise ValueError('empty-text RNG/optimization differs')


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2); started=time.perf_counter()
    data,compact,hist,parent,audit=checked_inputs(args)
    d=data['bert']; y=labels90(d.times,d.events)
    ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('train','validation','test')}
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    code={**parent['contract']['code_sha256'],'training/v6/text_encoder_comparison.py':sha256(Path(__file__))}
    contract=dict(parent_report_sha256=sha256(args.parent/'report.json'),text_audit=audit,
        data_provenance=d.provenance,compact_provenance=compact['bert'].provenance,
        protocol_sha256=sha256(PROTOCOL),code_sha256=code,seeds=[42,7],arms=list(ARMS),
        selection='validation Brier90',lr=3e-4,epochs=60,patience=10,warmup=5,batch_size=512,
        device=torch.cuda.get_device_name(0),endpoint='strict491.21/J44.1; World A; competing death',
        limits=['reused test; no multiplicity adjustment; not blind',
                'text projection parameter counts differ: 768 vs2560; not identity-only attribution',
                'expanded baseline is a mechanism reference, not replacement of old default H/B',
                'text may reconstruct history; no causal claim or complete outside follow-up'])
    write_json(args.output/'contract.json',contract)
    catalog=dict(history=list(HISTORY_NAMES),current=list(CURRENT_NAMES),lab_axes=list(AXES),
        lab_channels=list(NUMERIC+FLAGS),med_groups=list(MED_GROUPS),med_channels=list(MED_CHANNELS),
        micro=list(MICRO_NAMES),support=list(SUPPORT_NAMES),text=audit,
        source_coverage=json.loads((args.compact_dir/'manifest.json').read_text()),
        lab_coverage=json.loads((Path(json.loads((args.reference/'config.json').read_text())['state_dir'])/'manifest.json').read_text()))
    write_json(args.output/'input_catalog.json',catalog)
    cached={key:{f:prepared(data[key],compact[key],ix[f],512,key) for f in ('train','validation')}
            for key in ('bert','qwen')}
    fitted={}
    for seed in (42,7):
        (args.output/f'seed{seed}').mkdir(); common=None
        for arm in ARMS:
            source=arm_source(arm); m=new_model(args,seed,arm,hist,y[ix['train']].mean())
            initial=shared_parameters(m)
            if common is None: common=initial
            elif common.keys()!=initial.keys() or not all(torch.equal(v,common[k]) for k,v in initial.items()):
                raise ValueError('non-text initial state differs')
            cache=cached[source]
            if arm.startswith('none'):
                cache={f:prepared(data[source],compact[source],ix[f],512,arm) for f in ('train','validation')}
            write_json(args.output/'status.json',dict(status='training',seed=seed,arm=arm))
            fitted[seed,arm]=fit90(m,cache['train'],cache['validation'],y[ix['validation']],
                args.output/f'seed{seed}'/arm,epochs=60,lr=3e-4,seed=seed,patience=10,warmup=5)
            del m
        histories=[json.loads((args.output/f'seed{seed}'/a/'history.json').read_text()) for a in ('none_768','none_2560')]
        assert_empty_equal(*histories)
    write_json(args.output/'status.json',dict(status='all_eight_checkpoints_selected_before_test'))
    results,comparisons,parent_errors={},{},{}
    for seed in (42,7):
        results[str(seed)]={}; comparisons[str(seed)]={}
        for arm in ARMS:
            source=arm_source(arm); m=new_model(args,seed,arm,hist,y[ix['train']].mean())
            m.load_state_dict(torch.load(args.output/f'seed{seed}'/arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            results[str(seed)][arm]={**fitted[seed,arm],'folds':{}}
            for fold in ('validation','test'):
                idx=ix[fold]; pred=predict90(m,prepared(data[source],compact[source],idx,512,arm))
                pred['curve']=predict_cif(pred['parameters'],pred['shape'],GRID)
                if not np.allclose(pred['curve'].sum(2),1.,atol=1e-9): raise ValueError('mass conservation')
                fine=predict_cif(pred['parameters'][:64],pred['shape'][:64],GRID,step_days=.125)
                if np.abs(fine-pred['curve'][:64]).max()>2e-4: raise ValueError('CIF step convergence failed')
                if not np.allclose(pred['risk90'],pred['curve'][:,90,1],atol=2e-6,rtol=1e-5): raise ValueError('90-day CIF query differs')
                detail=binary_metrics(y[idx],pred['risk90'])
                detail['survival_readmission']=public_metrics(metrics(d.times[idx],d.events[idx],pred['curve'][:,:,1],ordering=True))
                detail['survival_death']=public_metrics(metrics(d.times[idx],d.events[idx],pred['curve'][:,:,2],cause=2,ordering=True))
                results[str(seed)][arm]['folds'][fold]=detail
                np.savez_compressed(args.output/f'seed{seed}'/arm/(fold+'_predictions.npz'),
                    hadm_ids=d.arrays['hadm_ids'][idx],subject_ids=d.arrays['subject_ids'][idx],times=d.times[idx],events=d.events[idx],y90=y[idx],
                    **{k:v for k,v in pred.items() if k!='loss'})
                if arm=='bert':
                    previous=read_prediction(args.parent,seed,'survival_full',fold)
                    if not np.array_equal(previous['hadm_ids'],d.arrays['hadm_ids'][idx]) or not np.allclose(previous['risk90'],pred['risk90'],atol=2e-6,rtol=1e-5):
                        raise ValueError('BERT retraining does not reproduce parent recipe')
                    parent_errors[f'{seed}/{fold}']=float(np.abs(previous['risk90']-pred['risk90']).max())
            del m
        for fold in ('validation','test'):
            idx=ix[fold]; comparisons[str(seed)][fold]={}
            a,b=(read_prediction(args.output,seed,arm,fold) for arm in ('none_768','none_2560'))
            if not np.array_equal(a['risk90'],b['risk90']) or not np.array_equal(a['curve'],b['curve']):
                raise ValueError('empty-text prediction differs across encoder widths')
            for left,right in PAIRS:
                a,b=(read_prediction(args.output,seed,arm,fold) for arm in (left,right))
                comparison=paired90(d.arrays['subject_ids'][idx],y[idx],a['risk90'],b['risk90'],replicates=200,seed=seed)
                comparison['integrated']=paired_interval(d.arrays['subject_ids'][idx],d.times[idx],d.events[idx],a['curve'][:,:,1],b['curve'][:,:,1],replicates=200,seed=seed)
                comparisons[str(seed)][fold][left+'_minus_'+right]=comparison
    report=dict(status='completed_same_text_bert_qwen_two_seeds',contract=contract,models=results,
        comparisons=comparisons,nontext_initial_states_equal=True,empty_text_histories_and_predictions_identical=True,
        bert_parent_retraining_errors=parent_errors,elapsed_seconds=time.perf_counter()-started)
    write_json(args.output/'report.json',report); render(args.output,report)
    write_json(args.output/'status.json',dict(status=report['status'],seconds=report['elapsed_seconds']))
    print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'])),flush=True)


def verify(args):
    report=json.loads((args.output/'report.json').read_text()); c=report['contract']
    for path,digest in c['code_sha256'].items():
        if sha256(Path(path))!=digest: raise ValueError('source changed: '+path)
    if sha256(PROTOCOL)!=c['protocol_sha256']: raise ValueError('protocol changed')
    data,compact,hist,parent,audit=checked_inputs(args)
    if audit!=c['text_audit'] or sha256(args.parent/'report.json')!=c['parent_report_sha256']: raise ValueError('asset changed')
    d=data['bert']; y=labels90(d.times,d.events); errors={}
    for seed in (42,7):
        for arm in ARMS:
            path=args.output/f'seed{seed}'/arm/'best.pt'
            if sha256(path)!=report['models'][str(seed)][arm]['checkpoint_sha256']: raise ValueError('checkpoint changed')
            m=new_model(args,seed,arm,hist,y[d.arrays['patient_splits']=='train'].mean())
            frozen=copy.deepcopy(m.baseline.state_dict())
            m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
            if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in frozen.items()): raise ValueError('baseline changed')
            source=arm_source(arm)
            for fold in ('validation','test'):
                idx=np.flatnonzero(d.arrays['patient_splits']==fold)
                p=predict90(m,prepared(data[source],compact[source],idx,512,arm))
                p['curve']=predict_cif(p['parameters'],p['shape'],GRID)
                saved=read_prediction(args.output,seed,arm,fold)
                for key,value in (('hadm_ids',d.arrays['hadm_ids'][idx]),('subject_ids',d.arrays['subject_ids'][idx]),('times',d.times[idx]),('events',d.events[idx]),('y90',y[idx])):
                    if not np.array_equal(saved[key],value): raise ValueError('identity/label mismatch')
                for key,value in p.items():
                    if key!='loss' and not np.allclose(saved[key],value,atol=2e-6,rtol=1e-5): raise ValueError('reload mismatch')
                detail=binary_metrics(y[idx],p['risk90'])
                for key in ('auc90','average_precision','brier90','logloss90'):
                    if abs(detail[key]-report['models'][str(seed)][arm]['folds'][fold][key])>1e-7: raise ValueError('metric mismatch')
                for name,cause in (('survival_readmission',1),('survival_death',2)):
                    actual=public_metrics(metrics(d.times[idx],d.events[idx],p['curve'][:,:,cause],cause=cause,ordering=True))
                    expected=report['models'][str(seed)][arm]['folds'][fold][name]
                    if actual!=expected: raise ValueError('survival metric mismatch')
                errors[f'{seed}/{arm}/{fold}']=float(np.abs(saved['risk90']-p['risk90']).max())
            del m
    write_json(args.output/'independent_verification.json',dict(status='all8_models_all16_folds_gpu_reloaded',
        report_sha256=sha256(args.output/'report.json'),baseline_unchanged=True,errors=errors))
    print('all8_models_all16_folds_gpu_reloaded',flush=True)


def render(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures'; folder.mkdir()
    names=('none_768','bert','qwen'); labels=('No text','Bio_ClinicalBERT','Qwen3-Embedding-4B')
    for seed in (42,7):
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        for arm,label in zip(names,labels,strict=True):
            p=read_prediction(root,seed,arm,'test'); m=report['models'][str(seed)][arm]['folds']['test']
            curves=ranked_curves(p['y90'],p['risk90'])
            axes[0].plot(curves['fpr'],curves['tpr'],label=f'{label}: AUC90={m["auc90"]:.3f}')
            actual=metrics(p['times'],p['events'],p['curve'][:,:,1])
            axes[1].plot(GRID[7:],actual['auc_curve'][7:],label=label)
            b=m['calibration_bins']; axes[2].plot([x['predicted'] for x in b],[x['observed'] for x in b],'o-',label=label)
        axes[0].plot([0,1],[0,1],'k--',alpha=.3); axes[2].plot([0,1],[0,1],'k--',alpha=.3)
        axes[2].set_xlim(0,.6); axes[2].set_ylim(0,.6)
        for ax,title,x,y in zip(axes,('90-day ROC','Time-dependent AUC(t)','90-day Calibration'),
            ('False positive rate','Days after discharge','Mean predicted risk'),('True positive rate','AUC(t)','Observed fraction')):
            ax.set(title=title,xlabel=x,ylabel=y); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle(f'Same four fields; frozen expanded baseline; seed{seed}; reused test')
        fig.tight_layout(); fig.savefig(folder/f'test_seed{seed}_curves.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
    for j,(left,right) in enumerate(PAIRS):
        for i,seed in enumerate((42,7)):
            d=report['comparisons'][str(seed)]['test'][left+'_minus_'+right]['metrics']
            for ax,key in zip(axes,('brier90','auc90')):
                r=d[key]; yy=j+(i-.5)*.16; color=('#3465a4','#e6550d')[i]
                ax.plot(r['ci95'],[yy,yy],color=color,alpha=.6); ax.scatter(r['difference'],yy,color=color,label=f'seed{seed}' if j==0 else None)
    for ax in axes:
        ax.axvline(0,color='black',linewidth=1); ax.grid(axis='x',alpha=.2); ax.legend()
        ax.set_yticks(range(3),['Qwen minus BERT','BERT minus no text','Qwen minus no text'])
    axes[0].invert_yaxis(); axes[0].set_xlabel('Paired Brier90 difference (lower better)'); axes[1].set_xlabel('Paired AUC90 difference (higher better)')
    fig.suptitle('Reused test; 200 patient-clustered 95% intervals; no multiplicity adjustment')
    fig.tight_layout(); fig.savefig(folder/'test_paired_two_seeds.png',dpi=170); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--compact-dir',type=Path,default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--parent',type=Path,default=Path('training_runs/v6_90day_objective_2026-10-11'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--verify',action='store_true')
    args=p.parse_args(); (verify if args.verify else run)(args)


if __name__=='__main__': main()
