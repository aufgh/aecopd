"""在原V6结构内独立测试新数据和池化前交互，旧实验源文件不改。"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.token_data import load_inputs
from training.v6.token_experiment import legacy_batches,make_model as old_model,fit,parameters
from training.v6.horizon_objective import restore
from training.v6.original_interaction import ExpandedBatch,OriginalInteractionEncoder
from training.v6.token_model import fast_cif,daily_probabilities
from training.v6.evaluation import metrics,public_metrics,paired_interval,calibration_bins
from training.v6.train import write_json

ARMS=('control','more_labs','more_meds','both','interaction_only','both_interaction')
PROTOCOL=Path('docs/V6_ORIGINAL_INTERACTION_PROTOCOL_2026-10-12.md')


def new_model(base,seed,arm):
    m=old_model(base,seed,'legacy')
    m.encoder=OriginalInteractionEncoder(m.encoder,labs=arm in ('more_labs','both','both_interaction'),
         medications=arm in ('more_meds','both','both_interaction'),interaction=arm in ('interaction_only','both_interaction')).cuda()
    # 固定同槽位初始化；原unused extra_meds旧保留，其他不修改。
    torch.manual_seed(seed)
    return m


def checked(args):
    d,c,h,ext,med,prov=load_inputs(args.reference,args.compact_dir,args.extension,args.medication)
    additions=[];active={};norm_checks={}
    from training.v5.numeric_preprocessing import robust_parameters
    train=d.arrays['patient_splits']=='train'
    for folder,file,key in ((args.expanded/'labs','expanded_labs.npz','lab'),(args.expanded/'medications','medication_identity.npz','drug')):
        manifest=json.loads((folder/'manifest.json').read_text());norm=json.loads((folder/'normalization.json').read_text())
        if manifest['parent_state_manifest_sha256']!=d.provenance['state_manifest_sha256'] or sha256(folder/file)!=manifest['output_sha256']:raise ValueError('new data lineage')
        for name,digest in manifest['sources_sha256'].items():
            if sha256(args.snapshot/(name+'.csv.gz'))!=digest:raise ValueError('source changed')
        with np.load(folder/file) as z:a={k:z[k] for k in z.files}
        for name in ('hadm_ids','subject_ids','patient_splits'):
            if not np.array_equal(a[name],d.arrays[name]):raise ValueError('new data members differ')
        if not np.isfinite(a['features']).all() or not np.isfinite(a['values_raw']).all():raise ValueError('nonfinite new features')
        names=manifest['names'] if key=='drug' else list(manifest['catalog'])
        available=a['present'] if key=='drug' else a['value_masks'][:,:,0]
        keep=np.array([len(np.unique(a['subject_ids'][train&available[:,j]]))>=20 or (key=='drug' and j<21) for j in range(len(names))])
        active[key]={name:dict(in_model=bool(keep[j]),train_patients=int(len(np.unique(a['subject_ids'][train&available[:,j]]))),
                        admissions=int(available[:,j].sum())) for j,name in enumerate(names)}
        # 独立重算训练尺度，药物剂量按单位分层；缺失数值不进入拟合。
        error=0.;raw=a['values_raw'];mask=a['value_masks']
        from patient_data.medication_identity import UNITS,CHANNELS
        for j,name in enumerate(names):
            for k in range(raw.shape[2]):
                buckets=range(len(UNITS)) if key=='drug' and k in (4,5) else (-1,)
                for unit in buckets:
                    valid=mask[:,j,k]&(a['dose_unit_id'][:,j]==unit if unit>=0 else True)
                    if not valid.any():continue
                    x=np.log1p(raw[:,j,k]) if key=='drug' or k<4 else raw[:,j,k]
                    p=robust_parameters(x[valid&train]);n=f'{name}/{CHANNELS[k]}/{UNITS[unit] if unit>=0 else "count"}' if key=='drug' else f'{name}/{k}'
                    if any(abs(p[v]-norm[n][v])>1e-7 for v in ('median','scale')):raise ValueError('train-only scale differs')
                    target=np.clip((x[valid]-p['median'])/p['scale'],-8,8)
                    if key=='drug':target*=available[valid,j]
                    error=max(error,float(np.abs(target-a['features'][valid,j,k]).max()))
        if error>2e-6:raise ValueError('normalization mismatch')
        norm_checks[key]=error
        if key=='drug':
            executed=raw[:,:,0]>0
            np.testing.assert_allclose(raw[:,:,3][executed],raw[:,:,0][executed]/raw[:,:,1][executed],rtol=1e-6)
            if np.any(a['features'][~available]!=0):raise ValueError('empty drug token')
            complete=mask[:,:,4]
            if np.any(a['dose_unit_id'][complete]==0) or np.any(a['dose_unit_id'][complete]==UNITS.index('mixed')):raise ValueError('mixed dose')
            a['present']=available&keep[None];a['features']*=a['present'][:,:,None]
        else:
            if np.any(raw[:,:,1][available]>raw[:,:,2][available]):raise ValueError('lab min/max')
            a['features']*=keep[None,:,None]
        additions.append(a)
        prov[file]=sha256(folder/'manifest.json')
    return d,c,h,ext,additions[0],additions[1],prov,active,norm_checks


def prepare(d,c,ext,labs,med,idx):
    cache=legacy_batches(d,c,ext,idx)
    result=[]
    for pos,h,b,t,e in cache:
        index=b.base.clinical.snippet_index.cpu().numpy();counts={};slots=[]
        for admission,_ in index:slots.append(counts.get(admission,0));counts[admission]=counts.get(admission,0)+1
        tensor=lambda x:torch.as_tensor(x,device='cuda')
        result.append((pos,h,ExpandedBatch(b,tensor(labs['features'][pos]),tensor(med['features'][pos]),
                 tensor(med['present'][pos]),tensor(np.asarray(slots,np.int64))),t,e))
    return result


def evaluate(m,cache,d,idx,path):
    a,k,nll=parameters(m,cache);curve=fast_cif(a,k,device='cuda');daily,conditional=daily_probabilities(curve)
    detail=dict(nll=nll,readmission=public_metrics(metrics(d.times[idx],d.events[idx],curve[:,:,1],ordering=True)),
      death=public_metrics(metrics(d.times[idx],d.events[idx],curve[:,:,2],cause=2,ordering=True)),
      calibration={str(day):calibration_bins(d.times[idx],d.events[idx],curve[:,:,1],day=day) for day in (30,90,180,365)})
    np.savez_compressed(path,hadm_ids=d.arrays['hadm_ids'][idx],subject_ids=d.arrays['subject_ids'][idx],times=d.times[idx],events=d.events[idx],
       log_amplitude=a,shape=k,curve=curve,daily_event_probability=daily,conditional_daily_event_probability=conditional,conditional_daily_valid=curve[:,:-1,0]>1e-12)
    return detail,curve,a,k


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;start=time.perf_counter()
    oldproof=json.loads((args.comparator/'independent_verification.json').read_text())
    if oldproof['report_sha256']!=sha256(args.comparator/'report.json'):raise ValueError('comparator stale')
    d,c,h,ext,lab,med,prov,active,norm_checks=checked(args);ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('train','validation','test')}
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    files=('training/v6/expanded_original_experiment.py','training/v6/original_interaction.py','patient_data/v6_expanded_candidates.py',
      'training/v6/token_experiment.py','training/v6/compact_model.py','training/v6/detail_extension_experiment.py')
    contract=dict(protocol_sha256=sha256(PROTOCOL),code_sha256={p:sha256(Path(p)) for p in files},
      data_provenance=d.provenance,compact_provenance=c.provenance,new_sources=prov,active=active,normalization_errors=norm_checks,
      arms=ARMS,seeds=[42,7],epochs=100,lr=1e-4,selection='validation IBS7-365',batch_size=256,
      endpoint='strict491.21/J44.1; competing death; WorldA365',device=torch.cuda.get_device_name(0),
      frozen_baseline_sha256={str(s):sha256(args.parent/f'seed{s}/survival_base/best.pt') for s in (42,7)},
      comparator_report_sha256=sha256(args.comparator/'report.json'),limits=['Reused test; not blind; no multiplicity correction.',
      'Interaction capacity differs; dose associations not causal; no complete outside followup.'])
    write_json(args.output/'contract.json',contract)
    cache={f:prepare(d,c,ext,lab,med,ix[f]) for f in ('train','validation')};fitted={};initial={}
    for seed in (42,7):
        base=restore(args.parent,seed,'survival_base',h,.1)
        for arm in ARMS:
            m=new_model(base,seed,arm)
            if arm=='interaction_only':
                ref=old_model(base,seed,'legacy');aa,kk,_=parameters(ref,[(p,hi,b.base,t,e) for p,hi,b,t,e in cache['validation'][:2]])
                a,k,_=parameters(m,cache['validation'][:2]);err=float(max(np.abs(a-aa).max(),np.abs(k-kk).max()))
                if err>1e-7:raise ValueError('epoch0 changed frozen reference')
                initial[str(seed)]=err;del ref
            fitted[seed,arm]=fit(m,cache['train'],cache['validation'],d.times[ix['validation']],d.events[ix['validation']],
                args.output/f'seed{seed}'/arm,seed=seed,lr=1e-4,epochs=100)
            del m
    write_json(args.output/'status.json',dict(status='all_checkpoints_selected_before_test'))
    del cache['train'];torch.cuda.empty_cache();cache['test']=prepare(d,c,ext,lab,med,ix['test'])
    models={};curves={};replay={};control_error={}
    for seed in (42,7):
        base=restore(args.parent,seed,'survival_base',h,.1);models[str(seed)]={};curves[seed]={}
        for arm in ARMS:
            path=args.output/f'seed{seed}'/arm;m=new_model(base,seed,arm)
            state=torch.load(path/'best.pt',map_location='cuda',weights_only=True)
            m.load_state_dict(state['state_dict']);models[str(seed)][arm]=dict(**fitted[seed,arm],folds={},interaction_gate=float(torch.tanh(m.encoder.interaction_gate).detach()))
            log=json.loads((path/'history.json').read_text());chosen=min(log,key=lambda r:r['validation_ibs'])
            if len(log)!=101 or chosen['epoch']!=state['epoch']:raise ValueError('incomplete or wrong selection')
            if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):raise ValueError('frozen base changed')
            curves[seed][arm]={}
            for f in ('validation','test'):
                detail,curve,a,k=evaluate(m,cache[f],d,ix[f],path/(f+'_predictions.npz'))
                fresh=new_model(base,seed,arm);fresh.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
                aa,kk,_=parameters(fresh,cache[f]);err=float(np.abs(fast_cif(aa,kk,device='cuda')-curve).max());del fresh
                if err>3e-7:raise ValueError('fresh GPU replay differs')
                replay[f'{seed}/{arm}/{f}']=err;models[str(seed)][arm]['folds'][f]=detail;curves[seed][arm][f]=curve
                if arm=='control':
                    with np.load(args.comparator/f'seed{seed}/legacy_lr0.0001'/(f+'_predictions.npz')) as z:previous=z['curve']
                    control_error[f'{seed}/{f}']=float(np.abs(previous-curve).max())
                    if control_error[f'{seed}/{f}']>3e-6:raise ValueError('control failed previous exact protocol reproduction')
            del m
    comparisons={}
    pairs=[(a,'control') for a in ARMS[1:]]+[('both_interaction','both'),('both_interaction','interaction_only')]
    for seed in (42,7):
        comparisons[str(seed)]={f:{f'{a} minus {b}':paired_interval(d.arrays['subject_ids'][ix[f]],d.times[ix[f]],d.events[ix[f]],
             curves[seed][a][f][:,:,1],curves[seed][b][f][:,:,1],seed=seed) for a,b in pairs} for f in ('validation','test')}
    scores={arm:sum(models[str(s)][arm]['validation_ibs'] for s in (42,7))/2 for arm in ARMS}
    report=dict(status='completed_original_v6_expansion',contract=contract,models=models,comparisons=comparisons,
       initial_errors=initial,reload_errors=replay,control_previous_errors=control_error,validation_selected_arm=min(scores,key=scores.get),
       validation_scores=scores,elapsed_seconds=time.perf_counter()-start,peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
    write_json(args.output/'report.json',report);write_json(args.output/'independent_verification.json',dict(status='fresh_gpu_reloads_verified',
        report_sha256=sha256(args.output/'report.json'),reload_errors=replay,control_previous_errors=control_error,
        normalization_errors=norm_checks,completed_models=12,frozen_checked=True))
    render(args.output,report);print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'],selected=report['validation_selected_arm'])),flush=True)


def render(root,report):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
    names=list(ARMS[1:])+['interaction increment on expanded data']
    for j,name in enumerate(names):
        pair=f'{name} minus control' if j<5 else 'both_interaction minus both'
        for seed,color,offset in ((42,'C0',-.08),(7,'C1',.08)):
            r=report['comparisons'][str(seed)]['test'][pair]
            for ax,key,ci in ((axes[0],'integrated_brier_difference','integrated_brier_95ci'),(axes[1],'auc90_difference','auc90_95ci')):
                ax.plot(r[ci],[j+offset]*2,color=color,alpha=.6);ax.scatter(r[key],j+offset,color=color,label=f'Seed {seed}' if j==0 else None)
    for ax in axes:ax.axvline(0,color='black',lw=1);ax.set_yticks(range(len(names)),names);ax.grid(axis='x',alpha=.2);ax.legend()
    axes[0].invert_yaxis();axes[0].set_xlabel('IBS7-365: candidate minus comparator (lower better)');axes[1].set_xlabel('AUC90 difference (higher better)')
    fig.suptitle('Original V6; reused test; patient-clustered bootstrap 200; no multiplicity adjustment');fig.tight_layout();fig.savefig(root/'paired_increment.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for i,seed in enumerate((42,7)):
        for arm in ARMS:
            history=json.loads((root/f'seed{seed}'/arm/'history.json').read_text());x=[r['epoch'] for r in history]
            axes[i,0].plot(x,[r['train_fixed2048_nll'] for r in history],label=arm)
            axes[i,1].plot(x,[r['validation_ibs'] for r in history],label=arm)
        axes[i,0].set_title(f'Seed {seed}: training survival NLL');axes[i,1].set_title(f'Seed {seed}: validation IBS7-365')
        for ax in axes[i]:ax.legend(fontsize=8);ax.grid(alpha=.2);ax.set_xlabel('Epoch')
    fig.tight_layout();fig.savefig(root/'training.png',dpi=170);plt.close(fig)


def parser():
    p=argparse.ArgumentParser()
    defaults=dict(reference='training_runs/v6_history_anchored_bert_seed42_2026-10-09',parent='training_runs/v6_90day_objective_2026-10-11',
      compact_dir='patient_data/processed/v6_compact_multisource_2026-10-09',extension='patient_data/processed/v6_detail_extension_2026-10-11_r2',
      medication='patient_data/processed/v6_medication_identity_2026-10-12',expanded='patient_data/processed/v6_expanded_candidates_2026-10-12_r2',
      comparator='training_runs/v6_coarse_fine_token_2026-10-12_r2',snapshot='patient_data/mimiciv_copd/snapshot_2026-09-27',
      output='training_runs/v6_original_interaction_2026-10-12')
    for k,v in defaults.items():p.add_argument('--'+k.replace('_','-'),type=Path,default=Path(v))
    return p


if __name__=='__main__':run(parser().parse_args())
