"""冻结r2候选事实集合有限早停配对；保留原完整V6风险。"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from patient_data.resp_fact_set_interface import ROOT as DATA,derive
from training.v6.resp_fact_set_model import FactCorrection,FactBatch,ARMS
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.curve_adaptation import FrozenBatch,caches
from training.v6.module_surgery import probability_curve
from training.v6.module_surgery_experiment import summary,save
from training.v6.risk import competing_nll
from training.v6.evaluation import metrics
from training.v6.early_stopping import EarlyStopping
from training.v6.train import write_json

ROOT=Path('training_runs/v6_resp_fact_set_2026-10-13_r2')
PROTOCOL=Path('docs/V6_RESP_FACT_SET_TRAINING_PROTOCOL_2026-10-13.md')
SEEDS=(42,7,101);CAP=60



def inputs():
    data=checked();d,c,h,ext,_,bins,tune,confirm,search,prior=data
    m=json.loads((DATA/'manifest.json').read_text())
    for tag in ('a','b'):
        p=json.loads((DATA/f'verification_{tag}.json').read_text())
        if p['maximum_error'] or p['manifest_sha256']!=sha256(DATA/'manifest.json'):
            raise ValueError('PFT interface source proof differs')
    for name,digest in m['source_sha256'].items():
        if sha256(Path(name))!=digest:raise ValueError('PFT source changed: '+name)
    for name,digest in m['output_sha256'].items():
        if sha256(DATA/name)!=digest:raise ValueError('PFT interface output changed')
    with np.load(DATA/'interface.npz') as z:arrays={k:z[k] for k in z.files}
    for key in ('hadm_ids','subject_ids','patient_splits'):np.testing.assert_array_equal(arrays[key],d.arrays[key])
    train=np.flatnonzero(d.arrays['patient_splits']=='train')
    if len(train)!=13592 or len(tune)!=1531:raise ValueError('outer split changed')
    sources={**search['source_sha256'],**prior['source_sha256'],**m['source_sha256']}
    for name in ('training/v6/resp_fact_set_model.py','training/v6/resp_fact_set_experiment.py','training/v6/curve_adaptation.py',
                 'training/v6/warm_residual_audit.py','training/v6/early_stopping.py','training/v6/pft_set_model.py','training/tests/test_v6_resp_fact_set_model.py',
                 'training/tests/test_v6_resp_fact_set_experiment.py'):
        sources[name]=sha256(Path(name))
    contract=dict(source_sha256=sources,protocol_sha256=sha256(PROTOCOL),data_manifest_sha256=sha256(DATA/'manifest.json'),
        data_sha256=m['output_sha256']['interface.npz'],fact_sha256=m['output_sha256']['facts.jsonl.gz'],
        scales=m['scales'],data_provenance=d.provenance,compact_provenance=c.provenance,
        teacher_hashes={str(s):sha256(REFERENCE/f'seed{s}/control/best.pt') for s in (42,7)},
        baseline_hashes={str(s):sha256(Path(search['input_config']['parent'])/f'seed{s}/survival_base/best.pt') for s in (42,7)},
        seeds=list(SEEDS),source_mapping={'42':42,'7':7,'101':42},arms=list(ARMS),new_parameters=998,
        optimizer={'name':'AdamW','lr':.001,'weight_decay':1e-4,'delta_penalty':1e-4,'clip':1.,'batch':512},
        early_stopping={'patience':10,'min_epochs':15,'min_delta':1e-5,'max_epochs':60},
        selection='mean tuning IBS; content must beat original and availability for new typed-fact content nomination',
        high_frequency_trajectories_used=False,old_static_support_retained_by_user=True,
        fact_role='explicit discharge/section and separately marked inpatient change; unknown role isolated',
        execution_image='aecopd-embedding:v1',torch_runtime='2.12.0+cu130',
        confirmation_test_evaluated=False,goal_achieved=False)
    return data,train,arrays,contract


def augmented(data,indices,seed,arrays):
    frozen,proof=caches(data,indices,seed)
    x={key:torch.cat([getattr(b,key) for b in frozen]) for key in ('hidden','raw','prior','times','events')}
    base=frozen[0].shape_raw;result=[];hashes=hashlib.sha256(indices.tobytes())
    for value in (*x.values(),base):hashes.update(value.cpu().numpy().tobytes())
    for offset in range(0,len(indices),512):
        sl=slice(offset,offset+512);use=indices[sl];owner=np.full(len(arrays['hadm_ids']),-1,np.int64);owner[use]=np.arange(len(use))
        source=owner[arrays['positions']];rows=np.flatnonzero(source>=0)
        f=FrozenBatch(x['hidden'][sl],x['raw'][sl],x['prior'][sl],base,x['times'][sl],x['events'][sl])
        tensor=lambda v:torch.as_tensor(v,device='cuda')
        b=FactBatch(f,*[tensor(arrays[key][rows]) for key in ('values','concepts','roles','polarities','units','conditions','numeric_present','numeric_valid')],
            tensor(source[rows]),tensor(arrays['aggregate'][use]),tensor(arrays['observed'][use]))
        for value in (b.values,b.concepts,b.roles,b.polarities,b.units,b.conditions,b.numeric_present,b.numeric_valid,b.owners,b.aggregate,b.observed):hashes.update(value.cpu().numpy().tobytes())
        result.append(b)
    return result,dict(teacher_cache=proof,augmented_cache_sha256=hashes.hexdigest(),records=len(indices))


def rebuild_data(arrays):
    rebuilt,_,_,_=derive()
    for key,value in rebuilt.items():np.testing.assert_array_equal(value,arrays[key])


def new_model(arm,seed):
    m=FactCorrection(arm,seed).cuda();torch.manual_seed(seed);return m


@torch.no_grad()
def prediction(model,cache):
    a,k=[],[];loss,n=0.,0
    for b in cache:
        out=model(b);a.append(out.log_amplitude.cpu().numpy());k.append(out.shape.cpu().numpy())
        loss+=float(competing_nll(out,b.frozen.times,b.frozen.events))*len(b.frozen.times);n+=len(b.frozen.times)
    aa,kk=np.concatenate(a),np.concatenate(k)
    return aa,kk,probability_curve(aa,kk),loss/n


def score(model,cache):
    a,k,curve,nll=prediction(model,cache)
    t=np.concatenate([b.frozen.times.cpu().numpy() for b in cache]);e=np.concatenate([b.frozen.events.cpu().numpy() for b in cache])
    r=metrics(t,e,curve[:,:,1]);death=metrics(t,e,curve[:,:,2],cause=2)
    return dict(nll=nll,ibs=r['integrated_brier_7_365'],auc90=r['points']['90']['auc'],death_ibs=death['integrated_brier_7_365'])


def preflight():
    data,train,arrays,contract=inputs()
    if ROOT.exists():raise FileExistsError('preflight already exists; inspect')
    ROOT.mkdir();write_json(ROOT/'contract.json',contract);rebuild_data(arrays)
    cached,proof=augmented(data,train[:128],42,arrays);b=cached[0]
    prior=None
    for arm in ARMS:
        m=new_model(arm,42);out=m(b)
        if prior is None:prior=out
        else:
            torch.testing.assert_close(out.log_amplitude,prior.log_amplitude,atol=0,rtol=0)
            torch.testing.assert_close(out.shape,prior.shape,atol=0,rtol=0)
        # 用真实训练来源存在者做单步，不能只拿全缺失批次宣称梯度启动。
        del m
    use=train[np.asarray(arrays['observed'][train],bool)][:128]
    observed,_=augmented(data,use,42,arrays);probe=new_model('content',42)
    opt=torch.optim.AdamW(probe.parameters(),lr=.001);before={k:v.detach().clone() for k,v in probe.state_dict().items()}
    out=probe(observed[0]);loss=competing_nll(out,observed[0].frozen.times,observed[0].frozen.events)
    loss.backward();torch.nn.utils.clip_grad_norm_(probe.parameters(),1.);opt.step()
    if torch.equal(before['head.2.weight'],probe.state_dict()['head.2.weight']):raise ValueError('real observed source cannot update')
    # 非零修正下检查真实GPU来源控制、numeric_valid屏蔽、集合排列和缺失gate。
    control=new_model('availability',42)
    with torch.no_grad():control.head[-1].weight.fill_(.01)
    b2=observed[0]
    changed=replace(b2,values=b2.values+50,concepts=(b2.concepts+1)%8,roles=(b2.roles+1)%3,
        polarities=1-b2.polarities,units=(b2.units+1)%3,conditions=(b2.conditions+1)%4,
        numeric_present=torch.zeros_like(b2.numeric_present),numeric_valid=torch.zeros_like(b2.numeric_valid))
    torch.testing.assert_close(control(b2).log_amplitude,control(changed).log_amplitude,atol=0,rtol=0)
    torch.testing.assert_close(control(b2).shape,control(changed).shape,atol=0,rtol=0)
    ix=torch.arange(len(b2.values)-1,-1,-1,device='cuda')
    permuted=replace(b2,**{key:getattr(b2,key)[ix] for key in
        ('values','concepts','roles','polarities','units','conditions','numeric_present','numeric_valid','owners')})
    torch.testing.assert_close(probe(b2).log_amplitude,probe(permuted).log_amplitude,atol=1e-6,rtol=1e-6)
    masked=replace(b2,values=b2.values+50*(~b2.numeric_valid))
    torch.testing.assert_close(probe(b2).log_amplitude,probe(masked).log_amplitude,atol=0,rtol=0)
    missing=~b.observed
    if not missing.any():raise ValueError('missing-source preflight has no actual rows')
    torch.testing.assert_close(probe(b).log_amplitude[missing],prior.log_amplitude[missing],atol=0,rtol=0)
    torch.testing.assert_close(probe(b).shape[missing],prior.shape[missing],atol=0,rtol=0)
    if sum(p.numel() for p in probe.parameters())!=998:raise ValueError('parameter budget changed')
    write_json(ROOT/'preflight.json',dict(status='passed',common_initial_risk_error=0.,original_cache_error=proof['teacher_cache']['initial_parameter_max_error'],
        source_rebuilt=True,availability_all_typed_content_hidden=True,real_observed_GPU_update=True,gpu=torch.cuda.get_device_name(0)))
    write_json(ROOT/'state.json',dict(status='preflight_passed',fits=0,goal_achieved=False))
    print(json.dumps(dict(status='preflight_passed',original_risk_error=0.)),flush=True)


def fit(model,train_cache,tune_cache,d,tune,folder,seed,arm):
    folder.mkdir(parents=True);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);early=EarlyStopping();history=[];start=time.perf_counter();reason='max_epochs'
    for epoch in range(CAP+1):
        online=None
        if epoch:
            total,n=0.,0;model.train()
            for j in rng.permutation(len(train_cache)):
                b=train_cache[j];opt.zero_grad(set_to_none=True);out=model(b)
                nll=competing_nll(out,b.frozen.times,b.frozen.events);loss=nll+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite PFT training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('nonfinite PFT gradient')
                opt.step();total+=float(nll.detach())*len(b.frozen.times);n+=len(b.frozen.times)
            online=total/n
        detail=score(model,tune_cache);save_best,stop=early.observe(epoch,detail['ibs'])
        history.append(dict(epoch=epoch,train_nll=online,**{'tuning_'+k:v for k,v in detail.items()},early_stale=early.stale,early_anchor=early.anchor))
        if save_best:torch.save(dict(state_dict=model.state_dict(),epoch=epoch,seed=seed,arm=arm),folder/'best.pt')
        write_json(folder/'history.json',history)
        if epoch%5==0 or stop:print(json.dumps(dict(seed=seed,arm=arm,epoch=epoch,best=early.best_epoch,ibs=detail['ibs'],early_stop=stop)),flush=True)
        if stop:reason='patience_exhausted';break
    model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
    a,k,curve,nll=prediction(model,tune_cache);save(folder/'tuning_predictions.npz',d,tune,a,k,curve)
    info=dict(arm=arm,seed=seed,completed_epochs=history[-1]['epoch'],best_epoch=early.best_epoch,stop_reason=reason,early_state=asdict(early),
        seconds=time.perf_counter()-start,checkpoint_sha256=sha256(folder/'best.pt'),metrics=summary(curve,d,tune,nll))
    write_json(folder/'training_result.json',info);return info


def lock(report):
    scores={arm:float(np.mean([report['fits'][str(s)][arm]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS])) for arm in ARMS}
    scores['original_V6']=float(np.mean([report['references'][str(s)]['readmission']['integrated_brier_7_365'] for s in SEEDS]))
    selected=min(scores,key=scores.get)
    return dict(selected=selected,scores=scores,nominated=['content'] if selected=='content' and scores['content']<min(v for k,v in scores.items() if k!='content') else [],
        selection='predeclared3-seed tuning IBS; availability gain not typed-fact content gain',goal_achieved=False)


def run():
    data,train,arrays,contract=inputs();d,_,_,_,_,_,tune,_,_,_=data
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    if json.loads((ROOT/'state.json').read_text())['status']!='preflight_passed':raise ValueError('already started; inspect')
    report=dict(fits={},references={},cache_hashes={},confirmation_test_not_evaluated=True,goal_achieved=False)
    for seed in SEEDS:
        tc,tp=augmented(data,train,seed,arrays);vc,vp=augmented(data,tune,seed,arrays)
        report['cache_hashes'][str(seed)]={'train':tp,'tune':vp}
        ref=new_model('content',seed);a,k,curve,nll=prediction(ref,vc)
        report['references'][str(seed)]=summary(curve,d,tune,nll);del ref
        report['fits'][str(seed)]={}
        for arm in ARMS:
            write_json(ROOT/'state.json',dict(status='training_with_early_stop',seed=seed,arm=arm,goal_achieved=False))
            m=new_model(arm,seed);report['fits'][str(seed)][arm]=fit(m,tc,vc,d,tune,ROOT/f'seed{seed}'/arm,seed,arm);del m
        del tc,vc;torch.cuda.empty_cache()
    write_json(ROOT/'screening.json',report);write_json(ROOT/'nomination_lock.json',lock(report))
    write_json(ROOT/'state.json',dict(status='training_complete_pending_independent_replay',fits=6,goal_achieved=False))
    print(json.dumps(lock(report)),flush=True)


def verify(tag):
    data,train,arrays,contract=inputs();d,_,_,_,_,_,tune,_,_,_=data
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    rebuild_data(arrays);report=json.loads((ROOT/'screening.json').read_text());errors={}
    for seed in SEEDS:
        tc,tp=augmented(data,train,seed,arrays);vc,vp=augmented(data,tune,seed,arrays)
        if report['cache_hashes'][str(seed)]!={'train':tp,'tune':vp}:raise ValueError('teacher/current source cache differs')
        ref=new_model('content',seed);a,k,curve,nll=prediction(ref,vc)
        if summary(curve,d,tune,nll)!=report['references'][str(seed)]:raise ValueError('reference differs')
        del ref,tc
        for arm in ARMS:
            folder=ROOT/f'seed{seed}'/arm;info=json.loads((folder/'training_result.json').read_text())
            if info!=report['fits'][str(seed)][arm] or sha256(folder/'best.pt')!=info['checkpoint_sha256']:raise ValueError('checkpoint differs')
            history=json.loads((folder/'history.json').read_text());early=EarlyStopping();first_stop=None
            for row in history:
                _,stop=early.observe(row['epoch'],row['tuning_ibs'])
                if stop and first_stop is None:first_stop=row['epoch']
            if asdict(early)!=info['early_state'] or early.best_epoch!=info['best_epoch'] or info['completed_epochs']!=len(history)-1:
                raise ValueError('early stop state/selection differs')
            if info['stop_reason']=='patience_exhausted' and first_stop!=info['completed_epochs']:raise ValueError('early stop bypassed')
            m=new_model(arm,seed);stored=torch.load(folder/'best.pt',map_location='cuda',weights_only=True)
            if stored['epoch']!=early.best_epoch:raise ValueError('wrong saved epoch')
            m.load_state_dict(stored['state_dict']);a,k,curve,nll=prediction(m,vc)
            with np.load(folder/'tuning_predictions.npz') as z:
                np.testing.assert_array_equal(z['times'],d.times[tune]);np.testing.assert_array_equal(z['events'],d.events[tune])
                error=max(float(np.abs(a-z['log_amplitude']).max()),float(np.abs(k-z['shape']).max()),float(np.abs(curve-z['curve']).max()))
            if error>3e-7 or summary(curve,d,tune,nll)!=info['metrics']:raise ValueError('prediction/metric differs')
            errors[f'{seed}/{arm}']=error;del m
        del vc
    if lock(report)!=json.loads((ROOT/'nomination_lock.json').read_text()):raise ValueError('selection lock differs')
    proof=dict(status='independently_verified',models=6,maximum_error=max(errors.values()),replay_errors=errors,
        report_sha256=sha256(ROOT/'screening.json'),contract_sha256=sha256(ROOT/'contract.json'),
        source_inputs_and_full_teacher_caches_rebuilt=True,early_stop_replayed=True,confirmation_test_not_evaluated=True)
    write_json(ROOT/f'verification_{tag}.json',proof)
    if all((ROOT/f'verification_{t}.json').exists() for t in ('a','b')):
        for t in ('a','b'):
            if json.loads((ROOT/f'verification_{t}.json').read_text())!=proof:raise ValueError('independent proof differs')
        write_json(ROOT/'state.json',dict(status='screening_complete_independently_verified',fits=6,nominated=lock(report)['nominated'],goal_achieved=False))
    print(json.dumps(proof),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('preflight','run','verify'),required=True);p.add_argument('--tag',choices=('a','b'),default='a');args=p.parse_args()
    try:
        if args.mode=='preflight':preflight()
        elif args.mode=='run':run()
        else:verify(args.tag)
    except Exception as e:
        if ROOT.exists():write_json(ROOT/f'failure_{args.mode}_{args.tag}.json',dict(error=repr(e),goal_achieved=False))
        raise


if __name__=='__main__':main()
