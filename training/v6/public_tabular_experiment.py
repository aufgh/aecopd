"""公开骨干的固定预算完整训练；验证选择在测试预测之前落盘。"""
import argparse
import json
import math
from pathlib import Path
import time
import importlib.metadata
import warnings
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.public_tabular_data import build, read
from training.v6.public_tabular_model import ARMS, PublicModel, initial_bias, ensemble_nll, ensemble_curve
from training.v6.audited_experiment import save_curve, summary, OUTPUT as AUDITED
from training.v6.evaluation import paired_interval
from training.v6.train import write_json

ROOT=Path('training_runs/v6_public_tabular_2026-10-12')
LRS=(3e-4,1e-3)
SEEDS=(42,7)
SOURCES=('training/v6/public_tabular_data.py','training/v6/public_tabular_model.py',
         'training/v6/public_tabular_experiment.py','training/v6/public_tabular_verify.py',
         'training/v6/risk.py','training/v6/token_model.py','training/v6/evaluation.py')


def setup():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.use_deterministic_algorithms(True)
    warnings.filterwarnings('ignore',message=r'The .*feature has just two bin edges.*',category=UserWarning)


def model_for(arm,seed,data,prep):
    torch.manual_seed(seed)
    model=PublicModel(arm,data['x_num'].shape[1],data['x_binary'].shape[1],
        [torch.as_tensor(b) for b in prep['bins']],initial_bias(data['events'][data['patient_splits']=='train'])).cuda()
    torch.manual_seed(seed)  # 构造模块消耗的随机数不改变训练采样/dropout起点。
    return model


def tensors(data,fold):
    ix=np.flatnonzero(data['patient_splits']==fold)
    return ix,tuple(torch.as_tensor(data[key][ix],device='cuda',dtype=dtype)
        for key,dtype in (('x_num',torch.float32),('x_binary',torch.float32),('times',torch.float32),('events',torch.long)))


@torch.no_grad()
def parameters(model,batch,size=128):
    model.eval();xn,xb,t,e=batch;a=[];k=[];total=0.
    for start in range(0,len(t),size):
        end=min(start+size,len(t))
        # 推理固定FP32，避免低精度改变细小排序差；训练统一BF16混合精度。
        aa,kk=model(xn[start:end],xb[start:end])
        total+=float(ensemble_nll(aa,kk,t[start:end],e[start:end]))*(end-start)
        a.append(aa.cpu().numpy());k.append(kk.cpu().numpy())
    return np.concatenate(a),np.concatenate(k),total/len(t)


def validation(model,batch,data,ix):
    a,k,nll=parameters(model,batch)
    curve=ensemble_curve(a,k)
    y=((data['events'][ix,None]==1)&(data['times'][ix,None]<=np.arange(366))).astype(float)
    brier=((curve[:,:,1]-y)**2).mean(0)
    from training.baselines.rank_metrics import auc_rank
    return dict(nll=nll,ibs=float(np.trapezoid(brier[7:],np.arange(7,366))/358),
        brier90=float(brier[90]),auc90=float(auc_rank(curve[:,90,1],y[:,90].astype(np.int8))))


def fit(model,train,val,data,vix,path,seed,lr,epochs=100):
    path.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    generator=torch.Generator(device='cuda').manual_seed(seed)
    xn,xb,t,e=train;rows=[];best=math.inf;best_epoch=0
    def record(epoch,online=None,rate=0.):
        nonlocal best,best_epoch
        stats=validation(model,val,data,vix)
        _,_,fixed=parameters(model,tuple(x[:2048] for x in train))
        rows.append(dict(epoch=epoch,training_nll_fixed2048=fixed,training_online_nll=online,
            lr=rate,**{'validation_'+k:v for k,v in stats.items()}))
        if stats['ibs']<best:
            best=stats['ibs'];best_epoch=epoch
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,validation_ibs=best),path/'best.pt')
        if epoch%10==0 or epoch==epochs:
            write_json(path/'history.json',rows)
            print(json.dumps(dict(arm=model.arm,seed=seed,lr=lr,epoch=epoch,validation_ibs=stats['ibs'],
                best_epoch=best_epoch,seconds=round(time.perf_counter()-start,1))),flush=True)
        write_json(path/'progress.json',dict(epoch=epoch,epochs=epochs,best_epoch=best_epoch,validation_ibs=best,
            elapsed_seconds=time.perf_counter()-start))
    record(0)
    for epoch in range(1,epochs+1):
        rate=lr*(epoch/5 if epoch<=5 else .1+.9*.5*(1+math.cos(math.pi*(epoch-5)/max(1,epochs-5))))
        for group in optimizer.param_groups:group['lr']=rate
        model.train();permutation=torch.randperm(len(t),device='cuda',generator=generator);total=0.
        # 同一有效batch256；每个物理batch128的损失按实际人数加权。
        for block in range(0,len(t),256):
            indices=permutation[block:block+256];optimizer.zero_grad(set_to_none=True)
            for micro in indices.split(128):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    a,k=model(xn[micro],xb[micro]);loss=ensemble_nll(a,k,t[micro],e[micro])
                if not torch.isfinite(loss):raise ValueError('nonfinite objective')
                (loss*len(micro)/len(indices)).backward()
                total+=float(loss.detach())*len(micro)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
            optimizer.step()
        record(epoch,total/len(t),rate)
    result=dict(arm=model.arm,seed=seed,lr=lr,completed_epochs=epochs,best_epoch=best_epoch,
        validation_ibs=best,parameters=sum(p.numel() for p in model.parameters()),
        peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,elapsed_seconds=time.perf_counter()-start,
        checkpoint_sha256=sha256(path/'best.pt'))
    write_json(path/'training_result.json',result)
    return result


def profile(root):
    setup();root.mkdir(parents=True,exist_ok=True)
    if not (root/'matrix.npz').exists():print(json.dumps(build(root)),flush=True)
    data,prep=read(root);_,batch=tensors(data,'train');vix,val=tensors(data,'validation')
    profiles={};initial=[]
    for arm in ARMS:
        model=model_for(arm,42,data,prep);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
        with torch.no_grad():
            model.eval();aa,kk=model(batch[0][:128],batch[1][:128])
            initial.append((aa.cpu().numpy()[:,0],kk.cpu().numpy()[:,0]))
        model.train()
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        for step in range(5):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                a,k=model(batch[0][:128],batch[1][:128]);loss=ensemble_nll(a,k,batch[2][:128],batch[3][:128])
            loss.backward();optimizer.step()
        torch.cuda.synchronize();step_seconds=(time.perf_counter()-start)/5
        start=time.perf_counter();stats=validation(model,val,data,vix)
        profiles[arm]=dict(step_seconds=step_seconds,validation_seconds=time.perf_counter()-start,
            parameters=sum(p.numel() for p in model.parameters()),peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
            **stats)
        print(json.dumps(dict(profile=arm,**profiles[arm])),flush=True)
        del model,optimizer;torch.cuda.empty_cache()
    if not all(np.array_equal(initial[0][0],a) and np.array_equal(initial[0][1],k) for a,k in initial):
        raise ValueError('initial risk differs across backbones')
    write_json(root/'profile.json',dict(models=profiles,initial_risk_identical=True,labels_not_used_for_model_choice=True))


def run(root,epochs=100):
    setup();begin=time.perf_counter()
    if not (root/'profile.json').exists():raise ValueError('profile all backbones first')
    if (root/'contract.json').exists():raise FileExistsError('new experiment root required')
    data,prep=read(root);trix,train=tensors(data,'train');vix,val=tensors(data,'validation')
    contract=dict(arms=ARMS,seeds=SEEDS,learning_rates=LRS,epochs=epochs,effective_batch=256,micro_batch=128,
        optimizer='AdamW',weight_decay=1e-4,selection='validation IBS7-365; LR by two-seed mean',
        end_to_end=True,frozen_history=False,mixed_precision_training='BF16',inference='FP32',
        source_sha256={p:sha256(Path(p)) for p in SOURCES},
        package_versions={p:importlib.metadata.version(p) for p in ('tabm','rtdl-revisiting-models','rtdl-num-embeddings','torch')},
        package_source_sha256={str(p):sha256(p) for p in Path('training/runtime/public_tabular_2026-10-12').glob('*.py')},
        protocol_sha256=sha256(Path('docs/V6_PUBLIC_TABULAR_PROTOCOL_2026-10-12.md')),
        data=json.loads((root/'data_contract.json').read_text()),gpu=torch.cuda.get_device_name(0),reused_test=True)
    write_json(root/'contract.json',contract);fits={}
    for arm in ARMS:
        fits[arm]={}
        for lr in LRS:
            fits[arm][str(lr)]={}
            for seed in SEEDS:
                model=model_for(arm,seed,data,prep)
                path=root/arm/f'lr{lr}'/f'seed{seed}'
                fits[arm][str(lr)][str(seed)]=fit(model,train,val,data,vix,path,seed,lr,epochs)
                del model;torch.cuda.empty_cache()
    choices={arm:min(LRS,key=lambda lr:np.mean([fits[arm][str(lr)][str(s)]['validation_ibs'] for s in SEEDS])) for arm in ARMS}
    write_json(root/'selected_before_test.json',dict(fits=fits,learning_rates=choices,all_training_complete=True))
    del train;torch.cuda.empty_cache();tix,test=tensors(data,'test')
    models={};comparisons={}
    for arm in ARMS:
        models[arm]={};comparisons[arm]={}
        for seed in SEEDS:
            lr=choices[arm];path=root/arm/f'lr{lr}'/f'seed{seed}';model=model_for(arm,seed,data,prep)
            checkpoint=torch.load(path/'best.pt',map_location='cuda',weights_only=True);model.load_state_dict(checkpoint['state_dict'])
            models[arm][str(seed)]=dict(**fits[arm][str(lr)][str(seed)],folds={});comparisons[arm][str(seed)]={}
            for fold,ix,batch in (('validation',vix,val),('test',tix,test)):
                a,k,nll=parameters(model,batch);curve=ensemble_curve(a,k)
                # summary/save_curve只依赖这些属性，不重新读取整个原始底账。
                from types import SimpleNamespace
                holder=SimpleNamespace(times=data['times'],events=data['events'],arrays=data)
                save_curve(path/(fold+'_predictions.npz'),curve,holder,ix,log_amplitude_members=a,shape_members=k)
                models[arm][str(seed)]['folds'][fold]=dict(nll=nll,**summary(curve,holder,ix))
                ref=AUDITED/'simple_controls'/f'xgb{seed}_{fold}_predictions.npz'
                with np.load(ref) as z:
                    if not np.array_equal(z['hadm_ids'],data['hadm_ids'][ix]):raise ValueError('reference members differ')
                    comparator=z['curve']
                comparisons[arm][str(seed)][fold]=paired_interval(data['subject_ids'][ix],data['times'][ix],data['events'][ix],
                    curve[:,:,1],comparator[:,:,1],seed=seed)
            del model
    report=dict(status='completed_training_and_prediction_pending_independent_verification',models=models,fits=fits,
        learning_rates=choices,comparisons_to_same_input_xgb=comparisons,elapsed_seconds=time.perf_counter()-begin,contract=contract)
    write_json(root/'report.json',report)
    render(root,report)
    print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'])),flush=True)


def render(root,r):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for i,arm in enumerate(ARMS):
        for s,col,dy in ((42,'C0',-.1),(7,'C1',.1)):
            row=r['comparisons_to_same_input_xgb'][arm][str(s)]['test']
            for ax,key,ci in ((axes[0],'integrated_brier_difference','integrated_brier_95ci'),(axes[1],'auc90_difference','auc90_95ci')):
                ax.plot(row[ci],[i+dy]*2,color=col);ax.scatter(row[key],i+dy,color=col,label=f'Seed {s}' if i==0 else None)
    for ax in axes:
        ax.axvline(0,color='black',lw=1);ax.set_yticks(range(len(ARMS)),ARMS);ax.grid(axis='x',alpha=.2);ax.legend()
    axes[0].invert_yaxis();axes[0].set_xlabel('IBS7-365: candidate minus XGBoost (lower better)')
    axes[1].set_xlabel('AUC90: candidate minus XGBoost (higher better)')
    fig.suptitle('Official backbones; matched information; patient bootstrap200; reused test')
    fig.tight_layout();fig.savefig(root/'paired_xgb.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,s in enumerate(SEEDS):
        for arm in ARMS:
            lr=r['learning_rates'][arm];rows=json.loads((root/arm/f'lr{lr}'/f'seed{s}'/'history.json').read_text())
            axes[i,0].plot([v['epoch'] for v in rows],[v['training_nll_fixed2048'] for v in rows],label=arm)
            axes[i,1].plot([v['epoch'] for v in rows],[v['validation_ibs'] for v in rows],label=arm)
        for ax in axes[i]:ax.grid(alpha=.2);ax.set_xlabel('Epoch');ax.legend(fontsize=8)
        axes[i,0].set_title(f'Seed{s}: training joint NLL');axes[i,1].set_title(f'Seed{s}: validation IBS7-365')
    fig.tight_layout();fig.savefig(root/'training.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--mode',choices=('profile','train'),required=True);p.add_argument('--epochs',type=int,default=100)
    args=p.parse_args();(profile(args.output) if args.mode=='profile' else run(args.output,args.epochs))
