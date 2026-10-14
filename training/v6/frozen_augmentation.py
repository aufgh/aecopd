"""冻结已验收完整模型，仅学习小容量的新字段幅度修正。"""
import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from patient_data.export_mimiciv import sha256
from training.v6.detail_extension_experiment import load_inputs
from training.v6.text_encoder_comparison import new_model, read_prediction
from training.v6.architecture_experiment import batches
from training.v6.horizon_objective import labels90, fit90, predict90, binary_metrics, paired90
from training.v6.risk import RiskOutput, predict_cif
from training.v6.evaluation import GRID, metrics, public_metrics
from training.v6.train import write_json

ARMS=('constant','new_labs','all_details')
RATES=(1e-4,1e-3)
PROTOCOL=Path('docs/V6_FROZEN_AUGMENTATION_PROTOCOL_2026-10-11.md')


@dataclass(frozen=True)
class AugmentBatch:
    amplitude: torch.Tensor
    shape: torch.Tensor
    features: torch.Tensor


class FrozenAugmentation(nn.Module):
    def __init__(self,baseline):
        super().__init__()
        self.baseline=baseline.requires_grad_(False).eval()
        self.encoder=nn.Identity()  # 兼容既有invoke的双输入接口，不参与计算。
        self.correction=nn.Linear(283,2,bias=False)
        nn.init.zeros_(self.correction.weight)

    def train(self,mode=True):
        super().train(mode); self.baseline.eval(); return self

    def forward(self,history,batch):
        n=len(history)
        if (batch.amplitude.shape!=(n,2) or batch.shape.shape!=(n,2) or batch.features.shape!=(n,283)
            or not torch.isfinite(batch.features).all() or batch.amplitude.requires_grad or batch.shape.requires_grad):
            raise ValueError('invalid frozen augmentation cache')
        delta=self.correction(batch.features).clamp(-.35,.35)
        return RiskOutput((batch.amplitude+delta).clamp(-10,6),batch.shape,delta)


def baseline_for(args,seed,hist):
    r=json.loads((args.text_parent/'report.json').read_text())
    proof=json.loads((args.text_parent/'independent_verification.json').read_text())
    if proof['report_sha256']!=sha256(args.text_parent/'report.json'): raise ValueError('unverified text parent')
    path=args.text_parent/f'seed{seed}/bert/best.pt'
    if sha256(path)!=r['models'][str(seed)]['bert']['checkpoint_sha256']: raise ValueError('parent checkpoint changed')
    m=new_model(args,seed,'bert',hist,.1)
    m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
    return m.requires_grad_(False).eval()


@torch.no_grad()
def cached(data,compact,a,ix,baseline):
    result=[]
    for pos,h,b,t,e in batches(data,compact,ix,512,expanded=True,text='bert'):
        out=baseline(h,b)
        x=np.concatenate((a['new_labs'][pos].reshape(len(pos),-1),a['eos'][pos].reshape(len(pos),-1),
                          a['med_details'][pos].reshape(len(pos),-1),np.ones((len(pos),1),np.float32)),1)
        result.append((pos,h,AugmentBatch(out.log_amplitude.detach(),out.shape.detach(),torch.as_tensor(x,device='cuda')),t,e))
    return result


def mask(cache,arm):
    result=[]
    for pos,h,b,t,e in cache:
        x=b.features.clone()
        if arm=='constant': x[:,:282]=0
        elif arm=='new_labs': x[:,90:282]=0
        elif arm!='all_details': raise ValueError('unknown augmentation arm')
        result.append((pos,h,AugmentBatch(b.amplitude,b.shape,x),t,e))
    return result


def construct(args,seed,hist):
    torch.manual_seed(seed)
    return FrozenAugmentation(baseline_for(args,seed,hist)).cuda()


def checkpoint(root,seed,arm,rate):
    return root/f'seed{seed}'/arm/f'lr{rate:g}'


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2); started=time.perf_counter()
    d,c,hist,parent,a,_=load_inputs(args); y=labels90(d.times,d.events)
    ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('train','validation','test')}
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    contract=dict(data_provenance=d.provenance,compact_provenance=c.provenance,
        extension_manifest_sha256=sha256(args.extension/'manifest.json'),
        text_parent_report_sha256=sha256(args.text_parent/'report.json'),
        protocol_sha256=sha256(PROTOCOL),code_sha256={**parent['contract']['code_sha256'],
            'training/v6/frozen_augmentation.py':sha256(Path(__file__)),
            'training/v6/detail_extension_experiment.py':sha256(Path('training/v6/detail_extension_experiment.py')),
            'training/v6/text_encoder_comparison.py':sha256(Path('training/v6/text_encoder_comparison.py'))},
        seeds=[42,7],arms=list(ARMS),learning_rates=list(RATES),selection='validation Brier90',
        correction_bound=.35,trainable_parameters=566,frozen='entire same-seed clinical+BERT model',
        device=torch.cuda.get_device_name(0))
    write_json(args.output/'contract.json',contract)
    fits,selection={},{}
    for seed in (42,7):
        base=baseline_for(args,seed,hist); cache={f:cached(d,c,a,ix[f],base) for f in ('train','validation')}
        selection[str(seed)]={}; fits[str(seed)]={}
        for arm in ARMS:
            fit_results={}; tr,va=mask(cache['train'],arm),mask(cache['validation'],arm)
            for rate in RATES:
                directory=checkpoint(args.output,seed,arm,rate); directory.parent.mkdir(parents=True,exist_ok=True)
                model=construct(args,seed,hist)
                if sum(p.numel() for p in model.parameters() if p.requires_grad)!=566: raise ValueError('capacity changed')
                write_json(args.output/'status.json',dict(status='training',seed=seed,arm=arm,lr=rate))
                fit_results[str(rate)]=fit90(model,tr,va,y[ix['validation']],directory,epochs=80,lr=rate,seed=seed,patience=15,warmup=5)
                del model
            chosen=min(RATES,key=lambda r:(fit_results[str(r)]['best_validation_brier90'],r))
            selection[str(seed)][arm]=dict(lr=chosen,**fit_results[str(chosen)])
            fits[str(seed)][arm]=fit_results
            del tr,va
        del base,cache
    write_json(args.output/'selection.json',selection)
    write_json(args.output/'status.json',dict(status='all12_candidates_selected_before_test'))
    models,comparisons={},{ }
    for seed in (42,7):
        models[str(seed)]={}; comparisons[str(seed)]={}
        baseline=baseline_for(args,seed,hist)
        for f in ('validation','test'):
            cache=cached(d,c,a,ix[f],baseline); idx=ix[f]
            for arm in ('original',*ARMS):
                m=construct(args,seed,hist)
                if arm!='original':
                    m.load_state_dict(torch.load(checkpoint(args.output,seed,arm,selection[str(seed)][arm]['lr'])/'best.pt',
                        map_location='cuda',weights_only=True)['state_dict'])
                if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in baseline.state_dict().items()):
                    raise ValueError('entire original model changed')
                p=predict90(m,mask(cache,'constant' if arm=='original' else arm))
                p['curve']=predict_cif(p['parameters'],p['shape'],GRID)
                if not np.allclose(p['curve'].sum(2),1,atol=1e-9): raise ValueError('probability mass failed')
                if arm=='original':
                    old=read_prediction(args.text_parent,seed,'bert',f)
                    if not np.array_equal(old['hadm_ids'],d.arrays['hadm_ids'][idx]) or not np.allclose(old['risk90'],p['risk90'],atol=1e-7):
                        raise ValueError('original reference differs')
                detail=binary_metrics(y[idx],p['risk90'])
                detail['survival_readmission']=public_metrics(metrics(d.times[idx],d.events[idx],p['curve'][:,:,1],ordering=True))
                detail['survival_death']=public_metrics(metrics(d.times[idx],d.events[idx],p['curve'][:,:,2],cause=2,ordering=True))
                models[str(seed)].setdefault(arm,dict(folds={}))['folds'][f]=detail
                folder=args.output/f'seed{seed}'/arm; folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/(f+'_predictions.npz'),hadm_ids=d.arrays['hadm_ids'][idx],subject_ids=d.arrays['subject_ids'][idx],
                    times=d.times[idx],events=d.events[idx],y90=y[idx],**{k:v for k,v in p.items() if k!='loss'})
                del m
            comparisons[str(seed)][f]={}
            for arm in ARMS:
                p=read_prediction(args.output,seed,arm,f)
                for other in ('original','constant'):
                    if arm==other: continue
                    q=read_prediction(args.output,seed,other,f)
                    comparisons[str(seed)][f][arm+'_minus_'+other]=paired90(p['subject_ids'],p['y90'],p['risk90'],q['risk90'],seed=seed)
            del cache
        del baseline
    report=dict(status='completed_frozen_full_model_augmentation',contract=contract,fits=fits,selection=selection,
        models=models,comparisons=comparisons,elapsed_seconds=time.perf_counter()-started,
        limits=['Reused test; not blind; no multiplicity correction.',
                'Amplitude-only bounded linear incremental prediction, not causal treatment effects.',
                'No clinical validity claim for every extra dose/frequency measurement.'])
    write_json(args.output/'report.json',report); render(args.output,report)
    write_json(args.output/'status.json',dict(status=report['status'],seconds=report['elapsed_seconds']))
    print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'])),flush=True)


def verify(args):
    report=json.loads((args.output/'report.json').read_text()); contract=report['contract']
    for p,digest in contract['code_sha256'].items():
        if sha256(Path(p))!=digest: raise ValueError('source changed')
    if sha256(PROTOCOL)!=contract['protocol_sha256']: raise ValueError('protocol changed')
    d,c,hist,parent,a,_=load_inputs(args)
    if (d.provenance!=contract['data_provenance'] or c.provenance!=contract['compact_provenance']
        or sha256(args.extension/'manifest.json')!=contract['extension_manifest_sha256']
        or sha256(args.text_parent/'report.json')!=contract['text_parent_report_sha256']): raise ValueError('input changed')
    errors={}; y=labels90(d.times,d.events)
    for seed in (42,7):
        baseline=baseline_for(args,seed,hist)
        for f in ('validation','test'):
            idx=np.flatnonzero(d.arrays['patient_splits']==f); cache=cached(d,c,a,idx,baseline)
            for arm in ('original',*ARMS):
                m=construct(args,seed,hist)
                if arm!='original':
                    chosen=report['selection'][str(seed)][arm]
                    path=checkpoint(args.output,seed,arm,chosen['lr'])/'best.pt'
                    if sha256(path)!=chosen['checkpoint_sha256']: raise ValueError('checkpoint changed')
                    m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
                if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in baseline.state_dict().items()):
                    raise ValueError('frozen whole model changed')
                p=predict90(m,mask(cache,'constant' if arm=='original' else arm))
                saved=read_prediction(args.output,seed,arm,f)
                for k,val in (('hadm_ids',d.arrays['hadm_ids'][idx]),('subject_ids',d.arrays['subject_ids'][idx]),
                              ('times',d.times[idx]),('events',d.events[idx]),('y90',y[idx])):
                    if not np.array_equal(saved[k],val): raise ValueError('identity/label changed')
                for k,val in p.items():
                    if k!='loss' and not np.allclose(saved[k],val,atol=1e-7): raise ValueError('replay mismatch')
                curve=predict_cif(p['parameters'],p['shape'],GRID)
                if not np.allclose(saved['curve'],curve,atol=1e-7): raise ValueError('curve mismatch')
                fine=predict_cif(p['parameters'][:64],p['shape'][:64],GRID,step_days=.125)
                if np.abs(fine-curve[:64]).max()>2e-4: raise ValueError('CIF integration not converged')
                actual=binary_metrics(y[idx],p['risk90']); expected=report['models'][str(seed)][arm]['folds'][f]
                for k in ('auc90','average_precision','brier90','logloss90'):
                    if abs(actual[k]-expected[k])>1e-7: raise ValueError('metric replay differs')
                errors[f'{seed}/{arm}/{f}']=float(np.abs(saved['risk90']-p['risk90']).max())
                del m
            del cache
        del baseline
    write_json(args.output/'independent_verification.json',dict(status='selected6_corrections_and2_references_all16_folds_reloaded',
        report_sha256=sha256(args.output/'report.json'),original_full_models_unchanged=True,errors=errors))
    print('selected6_corrections_and2_references_all16_folds_reloaded',flush=True)


def render(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures'; folder.mkdir()
    names=('constant_minus_original','new_labs_minus_constant','all_details_minus_constant')
    fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
    for j,name in enumerate(names):
        for i,seed in enumerate((42,7)):
            m=report['comparisons'][str(seed)]['test'][name]['metrics']
            for ax,k in zip(axes,('brier90','auc90')):
                r=m[k]; yy=j+(i-.5)*.15; color=('#2674ae','#cf7722')[i]
                ax.plot(r['ci95'],[yy,yy],color=color); ax.scatter(r['difference'],yy,color=color,label=f'seed{seed}' if j==0 else None)
    for ax in axes:
        ax.axvline(0,color='k',lw=1); ax.grid(axis='x',alpha=.2); ax.legend(); ax.set_yticks(range(3),names)
    axes[0].invert_yaxis(); axes[0].set_xlabel('Brier90 paired difference (lower better)')
    axes[1].set_xlabel('AUC90 paired difference (higher better)')
    fig.suptitle('Entire clinical+BERT model frozen; 566 new weights; reused test; 200 patient-clustered intervals')
    fig.tight_layout(); fig.savefig(folder/'test_paired.png',dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--compact-dir',type=Path,default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--parent',type=Path,default=Path('training_runs/v6_90day_objective_2026-10-11'))
    p.add_argument('--text-parent',type=Path,default=Path('training_runs/v6_bert_qwen_same_fields_2026-10-11'))
    p.add_argument('--extension',type=Path,default=Path('patient_data/processed/v6_detail_extension_2026-10-11_r2'))
    p.add_argument('--output',type=Path,default=Path('training_runs/v6_frozen_augmentation_2026-10-11'))
    p.add_argument('--verify',action='store_true')
    args=p.parse_args(); (verify if args.verify else run)(args)


if __name__=='__main__': main()
