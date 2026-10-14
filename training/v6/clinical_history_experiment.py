"""冻结原V6、读取过去临床状态的五条件三种子完整训练。"""
import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.warm_residual import BASE_SEEDS
from training.v6.warm_residual_audit import teacher
from training.v6.clock_experiment import extract
from training.v6.clinical_history_model import HistoryReader,ARMS
from training.v6.clinical_history_audit import ROOT as INDEX_ROOT
from training.v6.module_surgery_experiment import batches
from training.v6.module_surgery import probability_curve
from training.v6.horizon_objective import restore
from training.v6.risk import competing_nll
from training.v6.train import write_json

ROOT=Path('training_runs/v6_clinical_history_2026-10-13')
SEEDS=(42,7,101)


def layout(d,positions):
    proof=json.loads((INDEX_ROOT/'feasibility.json').read_text())
    file=INDEX_ROOT/'past_clinical_indices.npz'
    if sha256(file)!=proof['input_sha256'] or proof['strict_labels_sha256']!=d.provenance['label_sha256']:
        raise ValueError('historical index/label source changed')
    with np.load(file) as z:
        for key in ('hadm_ids','subject_ids','patient_splits'):np.testing.assert_array_equal(z[key],d.arrays[key])
        neighbors,offsets,gaps=z['previous_positions'],z['offsets'],z['days_from_previous_discharge_to_current_admission']
    mapping=np.full(len(d.history),-1,np.int64);mapping[positions]=np.arange(len(positions))
    count=np.diff(offsets)[positions];width=max(1,int(count.max()))
    index=np.zeros((len(positions),width),np.int64);valid=np.zeros_like(index,bool);distance=np.zeros_like(index,np.float32)
    for slot,p in enumerate(positions):
        selected=neighbors[offsets[p]:offsets[p+1]]
        local=mapping[selected]
        if (local<0).any() or np.any(d.arrays['patient_splits'][selected]!=d.arrays['patient_splits'][p]):
            raise ValueError('historical clinical row not available within patient fold')
        if np.any(d.arrays['subject_ids'][selected]!=d.arrays['subject_ids'][p]):raise ValueError('cross-patient clinical history')
        size=len(local);index[slot,:size]=local;valid[slot,:size]=True;distance[slot,:size]=gaps[offsets[p]:offsets[p+1]]
    return {key:torch.as_tensor(value,device='cuda') for key,value in dict(index=index,present=valid,gaps=distance).items()}


def call(model,data,history,ix):
    return model(data['clinical'][ix],data['clinical'][history['index'][ix]],history['gaps'][ix],history['present'][ix],
        data['amplitude'][ix],data['shape'][ix])


@torch.no_grad()
def score(model,data,history,indices):
    model.eval();amps=[];shapes=[];loss=0.
    for start in range(0,len(indices),512):
        ix=indices[start:start+512];out=call(model,data,history,ix)
        amps.append(out.log_amplitude.cpu().numpy());shapes.append(out.shape.cpu().numpy())
        loss+=float(competing_nll(out,data['times'][ix],data['events'][ix]))*len(ix)
    a,k=np.concatenate(amps),np.concatenate(shapes);curve=probability_curve(a,k)
    t,e=data['times'][indices].cpu().numpy(),data['events'][indices].cpu().numpy()
    y=((e[:,None]==1)&(t[:,None]<=np.arange(366))).astype(float)
    brier=((curve[:,:,1]-y)**2).mean(0)
    from training.baselines.rank_metrics import auc_rank
    return dict(ibs=float(np.trapezoid(brier[7:],np.arange(7,366))/358),brier90=float(brier[90]),
        auc90=float(auc_rank(curve[:,90,1],y[:,90].astype(np.int8))),nll=loss/len(indices)),a,k,curve


def fit(seed,arm,data,history,train,tune,path):
    torch.manual_seed(seed);model=HistoryReader(arm).cuda();torch.manual_seed(seed)
    rng=np.random.default_rng(seed);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    path.mkdir(parents=True,exist_ok=True);rows=[];best=math.inf;chosen=0;started=time.perf_counter()
    # 实际GPU核验Epoch0恢复完整教师；不是H34-only初始化。
    with torch.no_grad():
        out=call(model.eval(),data,history,tune[:512])
        torch.testing.assert_close(out.log_amplitude,data['amplitude'][tune[:512]],atol=0,rtol=0)
        torch.testing.assert_close(out.shape,data['shape'][tune[:512]],atol=0,rtol=0)
    def record(epoch,nll=None,rate=0.):
        nonlocal best,chosen
        detail,_,_,_=score(model,data,history,tune)
        rows.append(dict(epoch=epoch,train_nll=nll,lr=rate,**{'tuning_'+key:value for key,value in detail.items()}))
        if detail['ibs']<best:
            best,chosen=detail['ibs'],epoch
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,arm=arm,tuning_ibs=best),path/'best.pt')
        if epoch%20==0:
            write_json(path/'history.json',rows)
            print(json.dumps(dict(seed=seed,arm=arm,epoch=epoch,best_epoch=chosen,tuning_ibs=detail['ibs'],
                seconds=round(time.perf_counter()-started,1))),flush=True)
    record(0)
    for epoch in range(1,101):
        rate=1e-3*min(1.,epoch/5)*(.1+.45*(1+math.cos(math.pi*max(0,epoch-5)/95)))
        for group in opt.param_groups:group['lr']=rate
        model.train();order=rng.permutation(train.cpu().numpy());total=0.
        for start in range(0,len(order),512):
            ix=torch.as_tensor(order[start:start+512],device='cuda');opt.zero_grad(set_to_none=True)
            out=call(model,data,history,ix);nll=competing_nll(out,data['times'][ix],data['events'][ix])
            objective=nll+.01*out.delta.square().mean()
            if not torch.isfinite(objective):raise ValueError('nonfinite longitudinal objective')
            objective.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite longitudinal gradient')
            opt.step();total+=float(nll.detach())*len(ix)
        record(epoch,total/len(train),rate)
    write_json(path/'history.json',rows)
    return dict(seed=seed,arm=arm,completed_epochs=100,best_epoch=chosen,tuning_ibs=best,
        checkpoint_sha256=sha256(path/'best.pt'),elapsed_seconds=time.perf_counter()-started,
        trainable_parameters=sum(p.numel() for p in model.parameters()),teacher_frozen=True)


def cache_data(seed,d,c,h,ext,bins,search,positions,train_positions,tune_positions):
    base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
    model=teacher(base,seed,bins)
    # 分别按原train/tune分批，以免拼接改变教师矩阵形状/舍入路径。
    first=extract(model,batches(d,c,ext,train_positions));second=extract(model,batches(d,c,ext,tune_positions))
    data={key:torch.cat((first[key],second[key]),0).detach() for key in first}
    if any(x.requires_grad for x in data.values()):raise ValueError('teacher feature cache is not detached')
    del model,first,second
    path=ROOT/f'teacher_seed{BASE_SEEDS[seed]}_train_tune_cache.npz'
    payload={key:data[key].cpu().numpy() for key in ('clinical','amplitude','shape')}
    payload['positions']=positions
    if path.exists():
        with np.load(path) as z:
            for key,value in payload.items():np.testing.assert_array_equal(z[key],value)
    else:np.savez_compressed(path,**payload)
    return data,sha256(path)


def run():
    previous=Path('training_runs/v6_anchored_clock_2026-10-13_r2')
    proof=json.loads((previous/'confirmation_verification.json').read_text())
    if proof['report_sha256']!=sha256(previous/'confirmation_report.json'):raise ValueError('previous batch not verified')
    d,c,h,ext,_,bins,tune_positions,_,search,_=checked()
    train_positions=np.flatnonzero(d.arrays['patient_splits']=='train');positions=np.r_[train_positions,tune_positions]
    history=layout(d,positions)
    train=torch.arange(len(train_positions),device='cuda');tune=torch.arange(len(train_positions),len(positions),device='cuda')
    protected=('training/v6/clinical_history_model.py','training/v6/clinical_history_experiment.py',
        'training/v6/clinical_history_audit.py','training/tests/test_v6_clinical_history_model.py',
        'training/tests/test_v6_clinical_history_audit.py','docs/V6_CLINICAL_HISTORY_PROTOCOL_2026-10-13.md',
        'training/v6/clock_experiment.py','training/v6/warm_residual_audit.py','training/v6/warm_residual.py',
        'training/v6/regularization_audit.py')
    contract=dict(source_sha256={p:sha256(Path(p)) for p in protected},seeds=list(SEEDS),arms=list(ARMS),epochs=100,
        history_index_sha256=sha256(INDEX_ROOT/'past_clinical_indices.npz'),search_contract_sha256=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')),
        teachers={str(s):sha256(REFERENCE/f'seed{BASE_SEEDS[s]}/control/best.pt') for s in SEEDS},
        frozen_base_sources={str(s):sha256(Path(search['input_config']['parent'])/f'seed{BASE_SEEDS[s]}/survival_base/best.pt') for s in SEEDS},
        selection='three-seed tuning IBS with original_teacher included',confirmation_not_evaluated=True,test_not_evaluated=True)
    if (ROOT/'contract.json').exists():
        if json.loads((ROOT/'contract.json').read_text())!=contract:raise ValueError('history resume contract differs')
    else:write_json(ROOT/'contract.json',contract)
    results={};references={};cache_hashes={}
    for seed in SEEDS:
        data,digest=cache_data(seed,d,c,h,ext,bins,search,positions,train_positions,tune_positions)
        cache_hashes[str(seed)]=digest
        torch.manual_seed(seed);identity=HistoryReader('current_only').cuda()
        references[str(seed)]=dict(tuning=score(identity,data,history,tune)[0],newly_trained=False);del identity
        results[str(seed)]={}
        for arm in ARMS:
            path=ROOT/f'seed{seed}'/arm
            if (path/'training_result.json').exists():
                info=json.loads((path/'training_result.json').read_text())
                if info['completed_epochs']!=100 or info['checkpoint_sha256']!=sha256(path/'best.pt'):
                    raise ValueError('completed history result changed')
            else:
                if path.exists() and any((path/p).exists() for p in ('best.pt','history.json')):
                    raise RuntimeError('partial history fit exists; preserve and inspect, no automatic restart')
                write_json(ROOT/'state.json',dict(status='running',seed=seed,arm=arm,completed=results,goal_achieved=False))
                info=fit(seed,arm,data,history,train,tune,path);write_json(path/'training_result.json',info)
            results[str(seed)][arm]=info
        del data
    scores={'original_teacher':float(np.mean([references[str(s)]['tuning']['ibs'] for s in SEEDS]))}
    scores.update({arm:float(np.mean([results[str(s)][arm]['tuning_ibs'] for s in SEEDS])) for arm in ARMS})
    selected=min(scores,key=scores.get)
    write_json(ROOT/'selected_before_confirmation.json',dict(results=results,references=references,tuning_scores=scores,
        selected=selected,cache_sha256=cache_hashes,all_100_epochs_complete=True,confirmation_not_evaluated=True,test_not_evaluated=True))
    write_json(ROOT/'state.json',dict(status='training_complete_pending_independent_audit',selected=selected,goal_achieved=False))
    print(json.dumps(dict(status='history_training_complete_pending_audit',selected=selected)),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();ROOT.mkdir(exist_ok=True)
    lock=ROOT/'.running.lock'
    with lock.open('x',encoding='utf-8') as owner:owner.write(json.dumps(dict(pid=os.getpid(),started=time.time())))
    try:run()
    except Exception as error:
        write_json(ROOT/'state.json',dict(status='failed',error_type=type(error).__name__,error=str(error),goal_achieved=False));raise
    finally:lock.unlink()
