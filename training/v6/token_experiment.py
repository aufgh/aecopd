"""四项授权目标的独立实验入口：固定严格结局、每日风险、配对神经臂。"""
import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from patient_data.export_mimiciv import sha256
from training.v6.token_data import load_inputs,prepare
from training.v6.token_model import TokenModel,fast_cif,daily_probabilities,masked_reconstruction_batch
from training.v6.horizon_objective import restore
from training.v6.detail_extension_experiment import DetailBatch,DetailEncoder
from training.v6.compact_model import CompactAnchoredModel
from training.v6.architecture_experiment import omit_current
from training.v6.exploration_model import expanded_inputs
from training.v6.risk import competing_nll
from training.v6.evaluation import metrics,public_metrics,paired_interval,calibration_bins
from training.v6.train import write_json

PROTOCOL=Path('docs/V6_COARSE_FINE_TOKEN_PROTOCOL_2026-10-12.md')
ARMS=('pool_coarse','pool_fine','interaction_fine','interaction_ssl','flat_mlp')


def legacy_batches(d,c,ext,idx,size=256):
    result=[]
    for start in range(0,len(idx),size):
        pos=idx[start:start+size];h,b,t,e=c.collate(pos,'cuda');h=expanded_inputs(h,b);b=omit_current(b)
        labs=np.concatenate((ext['new_labs'][pos].reshape(len(pos),-1),ext['eos'][pos].reshape(len(pos),-1)),1)
        result.append((pos,h,DetailBatch(b,torch.as_tensor(labs,device='cuda'),torch.zeros(len(pos),192,device='cuda')),t,e))
    return result


def arm_batches(cache,arm):
    if arm!='pool_coarse':return cache
    return [(p,h,replace(b,named=b.named*0,named_present=b.named_present&False),t,e) for p,h,b,t,e in cache]


def make_model(base,seed,arm):
    torch.manual_seed(seed)
    if arm=='legacy':
        m=CompactAnchoredModel(copy.deepcopy(base),768)
        m.encoder=DetailEncoder(m.encoder)
    else:m=TokenModel(copy.deepcopy(base),interaction=arm.startswith('interaction'),flat=arm=='flat_mlp')
    m=m.cuda();torch.manual_seed(seed)  # 初始化耗用随机数不应改变后续dropout流。
    return m


@torch.no_grad()
def parameters(model,prepared):
    model.eval();a=[];k=[];total=0.;count=0
    for _,h,b,t,e in prepared:
        out=model(h,b) if hasattr(model,'encoder') else model(h)
        a.append(out.log_amplitude.cpu().numpy());k.append(out.shape.cpu().numpy())
        total+=float(competing_nll(out,t,e))*len(t);count+=len(t)
    return np.concatenate(a),np.concatenate(k),total/count


def validation(model,prepared,times,events):
    a,k,nll=parameters(model,prepared);curve=fast_cif(a,k,device='cuda')
    y=((events[:,None]==1)&(times[:,None]<=np.arange(366))).astype(float)
    brier=((curve[:,:,1]-y)**2).mean(0)
    from training.baselines.rank_metrics import auc_rank
    return dict(nll=nll,ibs=float(np.trapezoid(brier[7:],np.arange(7,366))/358),
                brier90=float(brier[90]),auc90=float(auc_rank(curve[:,90,1],y[:,90].astype(np.int8))),
                log_amplitude_quantiles=np.quantile(a,[.01,.5,.99],axis=0).tolist(),
                shape_quantiles=np.quantile(k,[.01,.5,.99],axis=0).tolist())


def fit(model,train,val,times,events,path,*,seed,lr,epochs):
    path.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4)
    frozen={k:v.detach().cpu().clone() for k,v in model.baseline.state_dict().items()}
    history=[];best=math.inf;selected=0;begin=time.perf_counter()
    def record(epoch,online=None,gradient=None,lr_value=0.):
        nonlocal best,selected
        detail=validation(model,val,times,events)
        # 固定训练住院样本，epoch末评估；不与在线批次均值混淆。
        _,_,train_nll=parameters(model,train[:8])
        row=dict(epoch=epoch,train_online_nll=online,train_fixed2048_nll=train_nll,
                 gradient_norm_before_clip=gradient,lr=lr_value,**{'validation_'+k:v for k,v in detail.items()})
        history.append(row)
        if detail['ibs']<best:
            best=detail['ibs'];selected=epoch
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,validation_ibs=best),path/'best.pt')
        if epoch%10==0 or epoch==epochs:
            write_json(path/'history.json',history)
            print(json.dumps(dict(arm=str(path),epoch=epoch,best_epoch=selected,ibs=detail['ibs'],auc90=detail['auc90'],
                                  seconds=round(time.perf_counter()-begin,1))),flush=True)
    record(0)
    for epoch in range(1,epochs+1):
        model.train();warm=min(1.,epoch/5);cosine=.1+.9*.5*(1+math.cos(math.pi*max(0,epoch-5)/max(1,epochs-5)))
        current_lr=lr*warm*cosine
        for g in opt.param_groups:g['lr']=current_lr
        total=0.;count=0;grad=[]
        for j in rng.permutation(len(train)):
            _,h,b,t,e=train[j];opt.zero_grad(set_to_none=True);out=model(h,b);nll=competing_nll(out,t,e)
            loss=nll+1e-4*out.delta.square().mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite training objective')
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(gn):raise ValueError('nonfinite gradients')
            opt.step();total+=float(nll.detach())*len(t);count+=len(t);grad.append(float(gn))
        record(epoch,total/count,float(np.mean(grad)),current_lr)
    if not all(torch.equal(v,model.baseline.state_dict()[k].cpu()) for k,v in frozen.items()):raise ValueError('frozen H34 changed')
    write_json(path/'history.json',history)
    return dict(best_epoch=selected,completed_epochs=epochs,lr=lr,validation_ibs=best,
        checkpoint_sha256=sha256(path/'best.pt'),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        elapsed_seconds=time.perf_counter()-begin,frozen_baseline_verified=True)


def pretrain(model,train,path,seed,epochs=20):
    """训练折重构：临床标签不进入优化；重构头丢弃后风险头仍严格为0。"""
    torch.manual_seed(seed);decoder=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,20)).cuda()
    opt=torch.optim.AdamW(list(model.encoder.parameters())+list(decoder.parameters()),lr=2e-4,weight_decay=1e-4)
    history=[];start=time.perf_counter()
    for epoch in range(epochs+1):
        model.encoder.train(epoch>0);decoder.train(epoch>0);total=0.;count=0;zero=0.
        iterator=train[:8] if epoch==0 else train
        for j,(_,_,b,_,_) in enumerate(iterator):
            masked,target=masked_reconstruction_batch(b,seed+epoch*10000+j)
            if not target.any():continue
            with torch.set_grad_enabled(epoch>0):
                h,states=model.encoder(masked);x=states['tokens'][:,:29]
                pred=decoder(torch.cat((x,h[:,None].expand(-1,29,-1)),2))
                errors=(pred-b.labs).square()[target];loss=errors.mean()
                if epoch:
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters())+list(decoder.parameters()),1.)
                    opt.step()
            total+=float(errors.detach().sum());count+=len(errors);zero+=float(b.labs.square()[target].sum())
        history.append(dict(epoch=epoch,reconstruction_mse=total/max(count,1),zero_normalized_prediction_mse=zero/max(count,1),observed_targets=count))
    if any(torch.count_nonzero(p) for p in model.delta_head[-1].parameters()):raise ValueError('SSL changed risk zero initialization')
    write_json(path/'self_supervision.json',dict(history=history,epochs=epochs,training_fold_only=True,
        diagnostic='training-fold masked reconstruction, not held-out clinical accuracy',seconds=time.perf_counter()-start))
    return history[-1]


def evaluate_model(model,cache,d,indices,path):
    a,k,nll=parameters(model,cache);curve=fast_cif(a,k,device='cuda');daily,conditional=daily_probabilities(curve)
    reread=torch.load(path/'best.pt',map_location='cuda',weights_only=True)
    model.load_state_dict(reread['state_dict']);aa,kk,_=parameters(model,cache)
    if not np.array_equal(a,aa) or not np.array_equal(k,kk):raise ValueError('checkpoint replay differs')
    return curve,dict(nll=nll,readmission=public_metrics(metrics(d.times[indices],d.events[indices],curve[:,:,1],ordering=True)),
                      death=public_metrics(metrics(d.times[indices],d.events[indices],curve[:,:,2],cause=2,ordering=True)),
                      calibration={str(day):calibration_bins(d.times[indices],d.events[indices],curve[:,:,1],day=day)
                                   for day in (30,90,180,365)}),dict(log_amplitude=a,shape=k,curve=curve,
                      daily_event_probability=daily,conditional_daily_event_probability=conditional)


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;start=time.perf_counter()
    proof=json.loads((args.parent/'independent_verification.json').read_text())
    if proof['report_sha256']!=sha256(args.parent/'report.json'):raise ValueError('parent report verification stale')
    d,c,hist,ext,med,prov=load_inputs(args.reference,args.compact_dir,args.extension,args.medication)
    ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('train','validation','test')}
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    code_paths=('training/v6/token_model.py','training/v6/token_data.py','training/v6/token_experiment.py','patient_data/medication_identity.py')
    contract=dict(protocol_sha256=sha256(PROTOCOL),code_sha256={p:sha256(Path(p)) for p in code_paths},
        data_provenance=d.provenance,compact_provenance=c.provenance,extensions=prov,seeds=args.seeds,
        epochs=args.epochs,self_supervised_epochs=args.ssl_epochs,selection='validation readmission IBS 7-365',
        baseline_parent_report_sha256=sha256(args.parent/'report.json'),
        frozen_baselines={str(s):sha256(args.parent/f'seed{s}/survival_base/best.pt') for s in args.seeds},
        endpoint='strict491.21/J44.1; competing death; WorldA365',fold_admissions={f:len(v) for f,v in ix.items()},
        device=torch.cuda.get_device_name(0),batch_size=256,arms=list(ARMS),lr_sweep=[3e-4,1e-4,3e-5],
        text='unchanged B four fields; all saved Bio_ClinicalBERT fragments; not raw/full-note comparison',
        limits=['Reused test, not blind; no multiplicity adjustment.',
                'Frozen H34 was selected by prior Brier90, not refit here.',
                'Medication/test interactions are associations, not treatment causal effects.'])
    write_json(args.output/'contract.json',contract)
    cache={f:prepare(d,c,ext,med,ix[f]) for f in ('train','validation')}
    legacy={f:legacy_batches(d,c,ext,ix[f]) for f in ('train','validation')}
    fitted={};recon={}
    for seed in args.seeds:
        base=restore(args.parent,seed,'survival_base',hist,.1)
        for lr in (3e-4,1e-4,3e-5):
            name=f'legacy_lr{lr:g}';m=make_model(base,seed,'legacy');path=args.output/f'seed{seed}'/name
            fitted[seed,name]=fit(m,legacy['train'],legacy['validation'],d.times[ix['validation']],d.events[ix['validation']],path,
                                 seed=seed,lr=lr,epochs=args.epochs);del m
        for arm in ARMS:
            m=make_model(base,seed,arm);path=args.output/f'seed{seed}'/arm
            if arm=='interaction_ssl':
                path.mkdir(parents=True);recon[str(seed)]=pretrain(m,cache['train'],path,seed,args.ssl_epochs)
                # fit负责创建目录，保留该预训练文件。
            fitted[seed,arm]=fit(m,arm_batches(cache['train'],arm),arm_batches(cache['validation'],arm),
                d.times[ix['validation']],d.events[ix['validation']],path,seed=seed,lr=1e-4,epochs=args.epochs);del m
    write_json(args.output/'status.json',dict(status='all_neural_checkpoints_selected_before_test'))
    # 所有选择结束后才首次为本次实验生成测试输入。
    cache['test']=prepare(d,c,ext,med,ix['test']);legacy['test']=legacy_batches(d,c,ext,ix['test'])
    models={};curves={};comparisons={}
    for seed in args.seeds:
        base=restore(args.parent,seed,'survival_base',hist,.1);models[str(seed)]={};curves[seed]={}
        for name in [f'legacy_lr{lr:g}' for lr in (3e-4,1e-4,3e-5)]+list(ARMS):
            m=make_model(base,seed,'legacy' if name.startswith('legacy') else name);path=args.output/f'seed{seed}'/name
            m.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            models[str(seed)][name]=dict(**fitted[seed,name],folds={});curves[seed][name]={}
            for f in ('validation','test'):
                batches=legacy[f] if name.startswith('legacy') else arm_batches(cache[f],name)
                curve,detail,pred=evaluate_model(m,batches,d,ix[f],path);models[str(seed)][name]['folds'][f]=detail;curves[seed][name][f]=curve
                np.savez_compressed(path/(f+'_predictions.npz'),hadm_ids=d.arrays['hadm_ids'][ix[f]],
                    subject_ids=d.arrays['subject_ids'][ix[f]],times=d.times[ix[f]],events=d.events[ix[f]],**pred)
            del m
        comparisons[str(seed)]={}
        pairs=(('pool_fine','pool_coarse'),('interaction_fine','pool_fine'),('interaction_ssl','interaction_fine'),
               ('flat_mlp','pool_fine'),('legacy_lr0.0001','legacy_lr0.0003'),('legacy_lr3e-05','legacy_lr0.0003'))
        for f in ('validation','test'):
            comparisons[str(seed)][f]={f'{a} minus {b}':paired_interval(d.arrays['subject_ids'][ix[f]],d.times[ix[f]],d.events[ix[f]],
                curves[seed][a][f][:,:,1],curves[seed][b][f][:,:,1],seed=seed) for a,b in pairs}
    report=dict(status='neural_completed',contract=contract,models=models,comparisons=comparisons,reconstruction=recon,
                elapsed_seconds=time.perf_counter()-start,peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
    write_json(args.output/'report.json',report);render(args.output,report)
    print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'])),flush=True)


def render(root,report):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'legacy_lr0.0003':'C0','legacy_lr0.0001':'C1','legacy_lr3e-05':'C2',
            'pool_coarse':'C3','pool_fine':'C4','interaction_fine':'C5','interaction_ssl':'C6','flat_mlp':'C7'}
    for group,names in (('learning_rate',list(colors)[:3]),('token_interaction',list(ARMS))):
        fig,ax=plt.subplots(2,2,figsize=(14,9))
        for row,seed in enumerate(report['contract']['seeds']):
            for name in names:
                hist=json.loads((root/f'seed{seed}'/name/'history.json').read_text());epochs=[x['epoch'] for x in hist]
                ax[row,0].plot(epochs,[x['train_fixed2048_nll'] for x in hist],label=name,color=colors[name])
                ax[row,1].plot(epochs,[x['validation_ibs'] for x in hist],label=name,color=colors[name])
            ax[row,0].set_title(f'Seed {seed}: fixed training sample survival NLL');ax[row,1].set_title(f'Seed {seed}: validation IBS 7-365')
            for a in ax[row]:a.set_xlabel('Epoch');a.grid(alpha=.25);a.legend(fontsize=8)
        fig.suptitle('100-epoch diagnostic; frozen H34; checkpoint selected by validation IBS');fig.tight_layout()
        fig.savefig(root/(group+'_training.png'),dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(14,5))
    for col,seed in enumerate(report['contract']['seeds']):
        for name in ('pool_coarse','pool_fine','interaction_fine','interaction_ssl','flat_mlp'):
            with np.load(root/f'seed{seed}'/name/'test_predictions.npz') as z:
                m=metrics(z['times'],z['events'],z['curve'][:,:,1]);ax[col].plot(np.arange(366),m['auc_curve'],label=name,color=colors[name])
        ax[col].set_title(f'Seed {seed}: reused-test AUC(t)');ax[col].set_xlabel('Days after discharge');ax[col].set_ylabel('AUC(t)');ax[col].legend(fontsize=8);ax[col].grid(alpha=.25)
    fig.tight_layout();fig.savefig(root/'time_dependent_auc.png',dpi=180);plt.close(fig)
    seed=report['contract']['seeds'][0];chosen=min(ARMS,key=lambda n:report['models'][str(seed)][n]['validation_ibs'])
    with np.load(root/f'seed{seed}'/chosen/'test_predictions.npz') as z:
        risk=z['curve'];daily=z['daily_event_probability'];conditional=z['conditional_daily_event_probability']
    # 固定前三条测试记录，无按结局/最高效果挑病例；不展示患者身份。
    fig,ax=plt.subplots(3,3,figsize=(14,11))
    for i in range(3):
        ax[i,0].plot(np.arange(366),risk[i,:,1],label='Readmission CIF');ax[i,0].plot(np.arange(366),risk[i,:,2],label='Death CIF')
        ax[i,1].plot(np.arange(1,366),daily[i,:,0],label='Unconditional daily readmission')
        ax[i,2].plot(np.arange(1,366),conditional[i,:,0],label='Daily readmission given event-free yesterday')
        for a in ax[i]:a.set_xlabel('Days after discharge');a.legend(fontsize=7);a.grid(alpha=.25)
    fig.suptitle(f'Daily probability interface: {chosen}; first 3 held records, not outcome-selected')
    fig.tight_layout();fig.savefig(root/'daily_probability_examples.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--parent',type=Path,default=Path('training_runs/v6_90day_objective_2026-10-11'))
    p.add_argument('--compact-dir',type=Path,default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--extension',type=Path,default=Path('patient_data/processed/v6_detail_extension_2026-10-11_r2'))
    p.add_argument('--medication',type=Path,default=Path('patient_data/processed/v6_medication_identity_2026-10-12'))
    p.add_argument('--output',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'))
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--ssl-epochs',type=int,default=20);p.add_argument('--seeds',type=int,nargs='+',default=[42,7])
    run(p.parse_args())


if __name__=='__main__':main()
