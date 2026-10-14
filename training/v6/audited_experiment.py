"""审计后固定配方实验；完成后才运行简单对照，不修改旧实验源。"""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.expanded_original_experiment import checked, parser as original_parser
from training.v6.horizon_objective import restore
from training.v6.exploration_model import expanded_inputs
from training.v6.audited_model import AuditedBatch, AuditedModel, flat_features
from training.v6.token_experiment import fit, parameters
from training.v6.token_model import fast_cif, daily_probabilities
from training.v6.evaluation import metrics, public_metrics, paired_interval, calibration_bins
from training.v6.train import write_json

ARMS=('full_pool','compact_pool','compact_interaction_pool','compact_interaction_cls')
INPUT=Path('patient_data/processed/v6_audited_coarse_fine_2026-10-12_r2')
OUTPUT=Path('training_runs/v6_audited_coarse_fine_2026-10-12')


def load():
    cfg=original_parser().parse_args([]);cfg.expanded=Path('patient_data/processed/v6_expanded_candidates_2026-10-12_r3')
    d,c,h,ext,lab,med,provenance,active,errors=checked(cfg)
    audit=json.loads((INPUT/'audit.json').read_text())
    if audit['interface_sha256']!=sha256(INPUT/'interface.npz'):raise ValueError('audit interface changed')
    with np.load(INPUT/'interface.npz') as z:a={k:z[k] for k in z.files}
    for key in ('hadm_ids','subject_ids','patient_splits'):
        if not np.array_equal(a[key],d.arrays[key]):raise ValueError('member mismatch')
    return cfg,d,c,h,a,audit


def prepare(d,c,a,idx,size=256):
    cache=[]
    for start in range(0,len(idx),size):
        pos=idx[start:start+size];h,compact,t,e=c.collate(pos,'cuda');h=expanded_inputs(h,compact)
        length=max(1,max(len(d.text_refs[p]) for p in pos));text=np.zeros((len(pos),length,768),np.float32)
        fields=np.zeros((len(pos),length),np.int64);present=np.zeros((len(pos),length),bool)
        for slot,p in enumerate(pos):
            for j,(f,r) in enumerate(d.text_refs[p]):text[slot,j]=d.text_vectors[r];fields[slot,j]=f;present[slot,j]=True
        tensor=lambda x:torch.as_tensor(x,device='cuda')
        b=AuditedBatch(*(tensor(a[key][pos]) for key in ('labs','lab_present','medications','med_present','coarse','micro','micro_present','support','support_present')),
                       tensor(text),tensor(fields),tensor(present))
        cache.append((pos,h,b,t,e))
    return cache


def restrict(cache,a,compact):
    if not compact:return cache
    lk=torch.as_tensor(a['lab_keep'],device='cuda');mk=torch.as_tensor(a['med_keep'],device='cuda')
    return [(p,h,replace(b,labs=b.labs*lk[None,:,None],lab_present=b.lab_present&lk[None],
                         medications=b.medications*mk[None,:,None],med_present=b.med_present&mk[None]),t,e) for p,h,b,t,e in cache]


def new_model(base,seed,arm,a):
    torch.manual_seed(seed)
    mode={'full_pool':'pool','compact_pool':'pool','compact_interaction_pool':'interaction_pool','compact_interaction_cls':'interaction_cls','flat':'flat'}[arm]
    model=AuditedModel(copy.deepcopy(base),a['lab_family'],a['drug_parent'],mode=mode).cuda()
    torch.manual_seed(seed);return model


def summary(curve,d,idx):
    return dict(readmission=public_metrics(metrics(d.times[idx],d.events[idx],curve[:,:,1],ordering=True)),
      death=public_metrics(metrics(d.times[idx],d.events[idx],curve[:,:,2],cause=2,ordering=True)),
      calibration={str(day):calibration_bins(d.times[idx],d.events[idx],curve[:,:,1],day=day) for day in (30,90,180,365)})


def save_curve(path,curve,d,idx,**extra):
    daily,conditional=daily_probabilities(curve)
    np.savez_compressed(path,hadm_ids=d.arrays['hadm_ids'][idx],subject_ids=d.arrays['subject_ids'][idx],times=d.times[idx],events=d.events[idx],
      curve=curve,daily_event_probability=daily,conditional_daily_event_probability=conditional,**extra)


def run(root,epochs=100):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;begin=time.perf_counter()
    cfg,d,c,h,a,audit=load();idx={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('train','validation','test')}
    root.mkdir(parents=True)
    sources=('training/v6/audited_experiment.py','training/v6/audited_model.py','patient_data/v6_input_audit_compact.py','training/v6/token_experiment.py')
    contract=dict(arms=ARMS,seeds=[42,7],epochs=epochs,lr=1e-4,selection='validation IBS7-365',
      input_audit_sha256=sha256(INPUT/'audit.json'),protocol_sha256=sha256(Path('docs/V6_AUDITED_COARSE_FINE_PROTOCOL_2026-10-12.md')),
      source_sha256={file:sha256(Path(file)) for file in sources},data=d.provenance,
      frozen_baseline_sha256={str(s):sha256(cfg.parent/f'seed{s}/survival_base/best.pt') for s in (42,7)},
      counts={f:len(idx[f]) for f in idx},gpu=torch.cuda.get_device_name(0),reused_test=True,
      limits=['No multiplicity adjustment; hierarchy/representation/capacity differ from old V6.',
              'Simple comparators run after neural selection; XGB/Cox are refitted end-to-end, not frozen history.'])
    write_json(root/'contract.json',contract)
    cache={f:prepare(d,c,a,idx[f]) for f in ('train','validation')};fitted={};initial={}
    for seed in (42,7):
        base=restore(cfg.parent,seed,'survival_base',h,.1)
        states=None
        for arm in ARMS:
            model=new_model(base,seed,arm,a)
            initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if states is None:states=initial_state
            elif not all(torch.equal(states[k],v) for k,v in initial_state.items()):raise ValueError('paired initialization differs')
            batches=restrict(cache['validation'],a,arm!='full_pool');aa,kk,_=parameters(model,batches[:1])
            with torch.no_grad():expected=base(batches[0][1])
            err=float(max(np.abs(aa-expected.log_amplitude.cpu().numpy()).max(),np.abs(kk-expected.shape.cpu().numpy()).max()))
            if err>1e-7:raise ValueError('nonzero initial correction')
            initial[f'{seed}/{arm}']=err
            fitted[seed,arm]=fit(model,restrict(cache['train'],a,arm!='full_pool'),batches,d.times[idx['validation']],d.events[idx['validation']],
              root/f'seed{seed}'/arm,seed=seed,lr=1e-4,epochs=epochs);del model
    scores={arm:sum(fitted[s,arm]['validation_ibs'] for s in (42,7))/2 for arm in ARMS};selected=min(scores,key=scores.get)
    write_json(root/'status.json',dict(status='neural_trainings_complete_before_simple_controls',selected=selected,validation_scores=scores))
    # 简单MLP在相同神经选择的输入配方上从头训练；不用测试决定配方。
    compact=selected!='full_pool'
    for seed in (42,7):
        base=restore(cfg.parent,seed,'survival_base',h,.1);model=new_model(base,seed,'flat',a)
        fitted[seed,'flat']=fit(model,restrict(cache['train'],a,compact),restrict(cache['validation'],a,compact),
            d.times[idx['validation']],d.events[idx['validation']],root/f'seed{seed}'/'flat',seed=seed,lr=1e-4,epochs=epochs);del model
    del cache['train'];torch.cuda.empty_cache();cache['test']=prepare(d,c,a,idx['test'])
    torch.use_deterministic_algorithms(True);models={};curves={};replay={}
    for seed in (42,7):
        base=restore(cfg.parent,seed,'survival_base',h,.1);models[str(seed)]={}
        for arm in (*ARMS,'flat'):
            path=root/f'seed{seed}'/arm;model=new_model(base,seed,arm,a)
            state=torch.load(path/'best.pt',map_location='cuda',weights_only=True);model.load_state_dict(state['state_dict'])
            history=json.loads((path/'history.json').read_text())
            if len(history)!=epochs+1 or min(history,key=lambda r:r['validation_ibs'])['epoch']!=state['epoch']:raise ValueError('selection mismatch')
            if not all(torch.equal(v,model.baseline.state_dict()[k]) for k,v in base.state_dict().items()):raise ValueError('frozen reference changed')
            models[str(seed)][arm]=dict(**fitted[seed,arm],folds={});curves[seed,arm]={}
            for f in ('validation','test'):
                batches=restrict(cache[f],a,compact if arm=='flat' else arm!='full_pool')
                aa,kk,nll=parameters(model,batches);curve=fast_cif(aa,kk,device='cuda')
                models[str(seed)][arm]['folds'][f]=dict(nll=nll,**summary(curve,d,idx[f]));curves[seed,arm][f]=curve
                save_curve(path/(f+'_predictions.npz'),curve,d,idx[f],log_amplitude=aa,shape=kk)
                fresh=new_model(base,seed,arm,a);fresh.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
                aaa,kkk,_=parameters(fresh,batches);err=float(np.abs(fast_cif(aaa,kkk,device='cuda')-curve).max());del fresh
                if err>1e-9:raise ValueError('fresh replay differs')
                replay[f'{seed}/{arm}/{f}']=err
            del model
    comparisons={};pairs=[('compact_pool','full_pool'),('compact_interaction_pool','compact_pool'),('compact_interaction_cls','compact_interaction_pool'),('flat',selected)]
    for s in (42,7):
        comparisons[str(s)]={f:{f'{arm} minus {ref}':paired_interval(d.arrays['subject_ids'][idx[f]],d.times[idx[f]],d.events[idx[f]],
          curves[s,arm][f][:,:,1],curves[s,ref][f][:,:,1],seed=s) for arm,ref in pairs} for f in ('validation','test')}
    report=dict(status='neural_completed',contract=contract,models=models,comparisons=comparisons,selected_arm=selected,
      validation_scores=scores,initial_errors=initial,reload_errors=replay,elapsed_seconds=time.perf_counter()-begin,
      peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
    write_json(root/'report.json',report)
    from training.v6.audited_simple_controls import run as simple_run
    simple_run(root,cfg,d,c,h,a,idx,compact,selected)
    report['status']='completed_neural_and_simple_controls';report['elapsed_seconds']=time.perf_counter()-begin
    write_json(root/'report.json',report);write_json(root/'independent_verification.json',dict(status='neural_exact_replay_and_frozen_verified',
      completed_models=10,report_sha256=sha256(root/'report.json'),reload_errors=replay,all_curves_probability_checked=True))
    render(root,report);print(json.dumps(dict(status=report['status'],selected=selected,seconds=report['elapsed_seconds'])),flush=True)


def render(root,r):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    names=list(r['comparisons']['42']['test'])
    for j,name in enumerate(names):
        for s,col,dy in ((42,'C0',-.1),(7,'C1',.1)):
            row=r['comparisons'][str(s)]['test'][name]
            for ax,key,ci in ((axes[0],'integrated_brier_difference','integrated_brier_95ci'),(axes[1],'auc90_difference','auc90_95ci')):
                ax.plot(row[ci],[j+dy]*2,color=col,alpha=.65);ax.scatter(row[key],j+dy,color=col,label=f'Seed {s}' if j==0 else None)
    for ax in axes:ax.axvline(0,color='black',lw=1);ax.set_yticks(range(len(names)),names);ax.grid(axis='x',alpha=.2);ax.legend()
    axes[0].invert_yaxis();axes[0].set_xlabel('IBS7-365 difference (lower better)');axes[1].set_xlabel('AUC90 difference (higher better)')
    fig.suptitle('Strict AECOPD; paired retraining; patient bootstrap200; reused test');fig.tight_layout();fig.savefig(root/'paired_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,s in enumerate((42,7)):
        for arm in (*ARMS,'flat'):
            rows=json.loads((root/f'seed{s}'/arm/'history.json').read_text());x=[v['epoch'] for v in rows]
            axes[i,0].plot(x,[v['train_fixed2048_nll'] for v in rows],label=arm)
            axes[i,1].plot(x,[v['validation_ibs'] for v in rows],label=arm)
        for ax in axes[i]:ax.legend(fontsize=7);ax.grid(alpha=.2);ax.set_xlabel('Epoch')
        axes[i,0].set_title(f'Seed{s}: training survival NLL');axes[i,1].set_title(f'Seed{s}: validation IBS7-365')
    fig.tight_layout();fig.savefig(root/'training.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--epochs',type=int,default=100)
    args=p.parse_args();run(args.output,args.epochs)
