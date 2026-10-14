"""未利用数据的独立增量试验，不改已有模型/检查点/默认输入。"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from patient_data.export_mimiciv import sha256
from training.v6.text_encoder_comparison import checked_inputs, new_model, read_prediction
from training.v6.architecture_experiment import batches
from training.v6.horizon_objective import labels90, fit90, predict90, binary_metrics, paired90
from training.v6.evaluation import GRID, metrics, public_metrics
from training.v6.risk import predict_cif
from training.v6.train import write_json

ARMS=('control','new_labs','med_details','both','availability')


@dataclass(frozen=True)
class DetailBatch:
    base: object
    labs: torch.Tensor
    medications: torch.Tensor


class DetailEncoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base=base
        self.extra_labs=nn.Sequential(nn.Linear(90,32),nn.GELU(),nn.Linear(32,128),nn.LayerNorm(128))
        self.extra_meds=nn.Sequential(nn.Linear(192,32),nn.GELU(),nn.Linear(32,128),nn.LayerNorm(128))

    def forward(self,batch,*,mask_pmh=False):
        h,branches=self.base(batch.base,mask_pmh=mask_pmh)
        additions=[]
        for x,width,model in ((batch.labs,90,self.extra_labs),(batch.medications,192,self.extra_meds)):
            if x.shape!=(len(h),width) or not torch.isfinite(x).all(): raise ValueError('invalid extension')
            active=x.ne(0).any(1)
            additions.append(model(x)*active[:,None])
        return h+additions[0]+additions[1],dict(branches,extra_labs=additions[0],extra_meds=additions[1])


def model_for(args,seed,hist):
    m=new_model(args,seed,'bert',hist,.1)
    m.encoder=DetailEncoder(m.encoder).cuda()
    return m


def prepared(data,compact,a,idx,arm):
    cache=batches(data,compact,idx,512,expanded=True,text='bert')
    result=[]
    for pos,h,b,t,e in cache:
        lab=a['new_labs'][pos].copy(); eos=a['eos'][pos].copy(); med=a['med_details'][pos].copy()
        if arm not in ('new_labs','both','availability'): lab[:]=0; eos[:]=0
        if arm not in ('med_details','both','availability'): med[:]=0
        if arm=='availability': lab[:,:,:5]=0; eos[:,:,:6]=0; eos[:,:,12:]=0; med[:,:,:7]=0
        labs=np.concatenate((lab.reshape(len(pos),-1),eos.reshape(len(pos),-1)),1)
        detail=DetailBatch(b,torch.as_tensor(labs,device='cuda'),torch.as_tensor(med.reshape(len(pos),-1),device='cuda'))
        result.append((pos,h,detail,t,e))
    return result


def load_inputs(args):
    data,compact,hist,parent,_=checked_inputs(args)
    data,compact=data['bert'],compact['bert']
    m=json.loads((args.extension/'manifest.json').read_text())
    if (m['status']!='source_checked_extension_not_trained' or
        m['parent_state_manifest_sha256']!=data.provenance['state_manifest_sha256'] or
        sha256(args.extension/'extension.npz')!=m['output_sha256']): raise ValueError('extension lineage differs')
    with np.load(args.extension/'extension.npz') as z: a={k:z[k] for k in z.files}
    for k in ('hadm_ids','subject_ids','patient_splits'):
        if not np.array_equal(a[k],data.arrays[k]): raise ValueError('extension member/split differs')
    for k,shape in (('new_labs',(19401,5,10)),('eos',(19401,2,20)),('med_details',(19401,12,16))):
        if a[k].shape!=shape or not np.isfinite(a[k]).all(): raise ValueError('invalid extension shape')
    return data,compact,hist,parent,a,m


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2); start=time.perf_counter()
    d,c,hist,parent,a,manifest=load_inputs(args)
    ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('train','validation','test')}
    y=labels90(d.times,d.events); args.output.mkdir(parents=True)
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    code={**parent['contract']['code_sha256'],__file__.replace('/workspace/',''):sha256(Path(__file__)),
          'patient_data/v6_detail_extension.py':sha256(Path('patient_data/v6_detail_extension.py'))}
    contract=dict(code_sha256=code,data_provenance=d.provenance,compact_provenance=c.provenance,
        extension_manifest_sha256=sha256(args.extension/'manifest.json'),
        protocol_sha256=sha256(Path('docs/V6_DETAIL_EXTENSION_PROTOCOL_2026-10-11.md')),
        seeds=[42,7],arms=list(ARMS),selection='validation Brier90',epochs=60,lr=3e-4,
        baseline='same expanded frozen 34 as text comparison',bootstrap=200,device=torch.cuda.get_device_name(0))
    write_json(args.output/'contract.json',contract)
    cached={arm:{f:prepared(d,c,a,ix[f],arm) for f in ('train','validation')} for arm in ARMS}
    fit={}; initial_errors={}
    for seed in (42,7):
        (args.output/f'seed{seed}').mkdir(); common=None
        for arm in ARMS:
            m=model_for(args,seed,hist)
            state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            if common is None: common=state
            elif not all(torch.equal(v,common[k]) for k,v in state.items()): raise ValueError('initial state differs')
            fit[seed,arm]=fit90(m,cached[arm]['train'],cached[arm]['validation'],y[ix['validation']],
                args.output/f'seed{seed}'/arm,epochs=60,lr=3e-4,seed=seed,patience=10,warmup=5)
            del m
    write_json(args.output/'status.json',dict(status='all_checkpoints_selected_before_test'))
    results={}; comparisons={}; verification={}
    for seed in (42,7):
        results[str(seed)]={}; comparisons[str(seed)]={}
        for arm in ARMS:
            m=model_for(args,seed,hist)
            m.load_state_dict(torch.load(args.output/f'seed{seed}'/arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            results[str(seed)][arm]=dict(**fit[seed,arm],folds={})
            for f in ('validation','test'):
                idx=ix[f]; batch=prepared(d,c,a,idx,arm); pred=predict90(m,batch)
                pred['curve']=predict_cif(pred['parameters'],pred['shape'],GRID)
                if not np.allclose(pred['curve'].sum(2),1,atol=1e-9): raise ValueError('mass failed')
                detail=binary_metrics(y[idx],pred['risk90'])
                detail['survival_readmission']=public_metrics(metrics(d.times[idx],d.events[idx],pred['curve'][:,:,1],ordering=True))
                detail['survival_death']=public_metrics(metrics(d.times[idx],d.events[idx],pred['curve'][:,:,2],cause=2,ordering=True))
                results[str(seed)][arm]['folds'][f]=detail
                np.savez_compressed(args.output/f'seed{seed}'/arm/(f+'_predictions.npz'),
                    hadm_ids=d.arrays['hadm_ids'][idx],subject_ids=d.arrays['subject_ids'][idx],
                    times=d.times[idx],events=d.events[idx],y90=y[idx],**{k:v for k,v in pred.items() if k!='loss'})
                # 磁盘checkpoint重建网络，独立重新加载后推理，而非复用训练实例。
                again=model_for(args,seed,hist)
                again.load_state_dict(torch.load(args.output/f'seed{seed}'/arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
                replay=predict90(again,batch)
                err=float(np.abs(replay['risk90']-pred['risk90']).max())
                if err>1e-7: raise ValueError('reload differs')
                verification[f'{seed}/{arm}/{f}']=err
                del again
                if arm=='control':
                    previous=read_prediction(args.text_parent,seed,'bert',f)
                    err=float(np.abs(previous['risk90']-pred['risk90']).max())
                    if err>2e-6: raise ValueError('control does not reproduce prior BERT')
                    initial_errors[f'{seed}/{f}']=err
            del m
        for f in ('validation','test'):
            comparisons[str(seed)][f]={}
            control=read_prediction(args.output,seed,'control',f)
            for arm in ARMS[1:]:
                p=read_prediction(args.output,seed,arm,f)
                comparisons[str(seed)][f][arm]=paired90(p['subject_ids'],p['y90'],p['risk90'],control['risk90'],seed=seed)
    report=dict(status='completed_extension_two_seeds',contract=contract,coverage=manifest['coverage'],models=results,
        comparisons=comparisons,reload_max_errors=verification,control_parent_errors=initial_errors,
        elapsed_seconds=time.perf_counter()-start,limits=['Reused test, not blind; no multiplicity correction.',
            'Availability/control contrasts do not establish causality or individual variable contributions.'])
    write_json(args.output/'report.json',report); plot(args.output,report)
    print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'])),flush=True)


def plot(root,report):
    import matplotlib.pyplot as plt
    folder=root/'figures'; folder.mkdir()
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for j,arm in enumerate(ARMS[1:]):
        for i,seed in enumerate((42,7)):
            for ax,key in zip(axes,('brier90','auc90')):
                r=report['comparisons'][str(seed)]['test'][arm]['metrics'][key]
                yy=j+(i-.5)*.15; color=('#3465a4','#e6550d')[i]
                ax.plot(r['ci95'],[yy,yy],color=color,alpha=.6)
                ax.scatter(r['difference'],yy,color=color,label=f'seed {seed}' if j==0 else None)
    for ax in axes:
        ax.axvline(0,color='black',linewidth=1); ax.grid(axis='x',alpha=.2); ax.legend()
        ax.set_yticks(range(4),['Additional labs','Molecule dose / frequency','Both groups','Availability only'])
    axes[0].invert_yaxis(); axes[0].set_xlabel('Candidate minus control: Brier90 (lower better)')
    axes[1].set_xlabel('Candidate minus control: AUC90 (higher better)')
    fig.suptitle('Reused test; 200 patient-clustered 95% intervals; two seeds; no multiplicity adjustment')
    fig.tight_layout(); fig.savefig(folder/'extension_paired.png',dpi=170); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--compact-dir',type=Path,default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--parent',type=Path,default=Path('training_runs/v6_90day_objective_2026-10-11'))
    p.add_argument('--text-parent',type=Path,default=Path('training_runs/v6_bert_qwen_same_fields_2026-10-11'))
    p.add_argument('--extension',type=Path,default=Path('patient_data/processed/v6_detail_extension_2026-10-11_r2'))
    p.add_argument('--output',type=Path,default=Path('training_runs/v6_detail_extension_2026-10-11'))
    run(p.parse_args())
