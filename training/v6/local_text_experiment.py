"""同原句块级/局部文本有限早停配对；不使用新高频轨迹。"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from patient_data.local_text_interface import ROOT as DATA,derive
from training.v6.local_text_model import LocalTextCorrection,LocalTextBatch,ARMS
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.curve_adaptation import FrozenBatch,caches
from training.v6.module_surgery import probability_curve
from training.v6.module_surgery_experiment import summary,save
from training.v6.risk import competing_nll
from training.v6.evaluation import metrics
from training.v6.early_stopping import EarlyStopping
from training.v6.train import write_json

ROOT=Path('training_runs/v6_local_text_2026-10-13')
PROTOCOL=Path('docs/V6_LOCAL_TEXT_TRAINING_PROTOCOL_2026-10-13.md')
SEEDS=(42,7,101);CAP=60



def inputs():
    from training.v5.numeric_preprocessing import robust_parameters
    data=checked();d,c,h,ext,_,bins,tune,confirm,search,prior=data
    m=json.loads((DATA/'manifest.json').read_text());proof=json.loads((DATA/'source_verification.json').read_text())
    enc=json.loads((DATA/'encoding.json').read_text())
    if proof['manifest_sha256']!=sha256(DATA/'manifest.json') or enc['source_manifest_sha256']!=sha256(DATA/'manifest.json'):
        raise ValueError('interface source proof differs')
    for tag in ('a','b'):
        p=json.loads((DATA/f'encoding_verification_{tag}.json').read_text())
        if p['manifest_sha256']!=sha256(DATA/'manifest.json') or p['vector_sha256']!=enc['output_sha256']:
            raise ValueError('local encoding proof differs')
        if p['code_sha256']!=sha256(Path('training/v6/local_text_encoding_verify.py')):raise ValueError('encoding verifier source differs')
    for p,digest in m['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('interface source changed')
    for name,digest in m['output_sha256'].items():
        if sha256(DATA/name)!=digest:raise ValueError('interface output changed')
    if sha256(DATA/'local_vectors.float16.npy')!=enc['output_sha256']:raise ValueError('local vectors changed')
    with np.load(DATA/'interface.npz') as z:arrays={k:z[k] for k in z.files}
    arrays['local_vectors']=np.load(DATA/'local_vectors.float16.npy')
    for key in ('hadm_ids','subject_ids','patient_splits'):np.testing.assert_array_equal(arrays[key],d.arrays[key])
    n=len(d.times);count=np.bincount(arrays['sentence_positions'],minlength=n);obs=count>0
    meta=np.zeros((n,10),np.float32);np.add.at(meta,arrays['sentence_positions'],arrays['metadata'])
    meta/=np.maximum(count,1)[:,None]
    train=np.flatnonzero(d.arrays['patient_splits']=='train')
    scale=robust_parameters(np.log1p(count[train][obs[train]]))
    number=np.where(obs,np.clip((np.log1p(count)-scale['median'])/scale['scale'],-8,8),0)
    arrays['aggregate']=np.column_stack((meta,number)).astype(np.float32);arrays['observed']=obs
    if len(train)!=13592 or len(tune)!=1531:raise ValueError('outer split changed')
    sources={**search['source_sha256'],**prior['source_sha256'],**m['source_sha256']}
    for p in ('training/v6/local_text_model.py','training/v6/local_text_experiment.py',
              'training/v6/curve_adaptation.py','training/v6/warm_residual_audit.py','training/v6/early_stopping.py',
              'training/v6/local_text_encoding_verify.py',
              'training/tests/test_v6_local_text_model.py'):
        sources[p]=sha256(Path(p))
    contract=dict(source_sha256=sources,protocol_sha256=sha256(PROTOCOL),data_manifest_sha256=sha256(DATA/'manifest.json'),
        data_sha256=m['output_sha256']['interface.npz'],local_vectors_sha256=enc['output_sha256'],
        count_training_normalization=scale,data_provenance=d.provenance,compact_provenance=c.provenance,
        teacher_hashes={str(s):sha256(REFERENCE/f'seed{s}/control/best.pt') for s in (42,7)},
        baseline_hashes={str(s):sha256(Path(search['input_config']['parent'])/f'seed{s}/survival_base/best.pt') for s in (42,7)},
        seeds=list(SEEDS),source_mapping={'42':42,'7':7,'101':42},arms=list(ARMS),new_parameters=13557,
        optimizer={'name':'AdamW','lr':.001,'weight_decay':1e-4,'delta_penalty':1e-4,'clip':1.,'batch':512},
        early_stopping={'patience':10,'min_epochs':15,'min_delta':1e-5,'max_epochs':60},
        selection='mean tuning IBS; local must beat original/availability/containing for local-representation nomination',
        high_frequency_trajectories_used=False,old_static_support_retained_by_user=True,
        confirmation_test_evaluated=False,goal_achieved=False)
    return data,train,arrays,contract


def augmented(data,indices,seed,arrays):
    frozen,proof=caches(data,indices,seed)
    x={key:torch.cat([getattr(b,key) for b in frozen]) for key in ('hidden','raw','prior','times','events')}
    base=frozen[0].shape_raw;result=[];hashes=hashlib.sha256(indices.tobytes())
    for value in (*x.values(),base):hashes.update(value.cpu().numpy().tobytes())
    for offset in range(0,len(indices),512):
        sl=slice(offset,offset+512);use=indices[sl];owner=np.full(len(arrays['hadm_ids']),-1,np.int64);owner[use]=np.arange(len(use))
        source=owner[arrays['sentence_positions']];rows=np.flatnonzero(source>=0)
        f=FrozenBatch(x['hidden'][sl],x['raw'][sl],x['prior'][sl],base,x['times'][sl],x['events'][sl])
        tensor=lambda v:torch.as_tensor(v,device='cuda')
        b=LocalTextBatch(f,tensor(arrays['containing_vectors'][rows].astype(np.float32)),tensor(arrays['metadata'][rows]),
            tensor(source[rows]),tensor(arrays['aggregate'][use]),tensor(arrays['observed'][use]),tensor(arrays['local_vectors'][rows].astype(np.float32)))
        for v in (b.vectors,b.metadata,b.owners,b.aggregate,b.observed,b.local_vectors):hashes.update(v.cpu().numpy().tobytes())
        result.append(b)
    return result,dict(teacher_cache=proof,augmented_cache_sha256=hashes.hexdigest(),records=len(indices))


def rebuild_data(arrays):
    rebuilt,_,_,_=derive()
    for key,value in rebuilt.items():np.testing.assert_array_equal(value,arrays[key])


def new_model(arm,seed):
    m=LocalTextCorrection(arm,seed).cuda();torch.manual_seed(seed);return m


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
    observed,_=augmented(data,use,42,arrays);probe=new_model('local',42)
    opt=torch.optim.AdamW(probe.parameters(),lr=.001);before={k:v.detach().clone() for k,v in probe.state_dict().items()}
    out=probe(observed[0]);loss=competing_nll(out,observed[0].frozen.times,observed[0].frozen.events)
    loss.backward();torch.nn.utils.clip_grad_norm_(probe.parameters(),1.);opt.step()
    if torch.equal(before['head.2.weight'],probe.state_dict()['head.2.weight']):raise ValueError('real observed source cannot update')
    write_json(ROOT/'preflight.json',dict(status='passed',common_initial_risk_error=0.,original_cache_error=proof['teacher_cache']['initial_parameter_max_error'],
        source_rebuilt=True,same_sentence_source_views_verified=True,real_observed_GPU_update=True,gpu=torch.cuda.get_device_name(0)))
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
                if not torch.isfinite(loss):raise ValueError('nonfinite local text training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('nonfinite local text gradient')
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
    return dict(selected=selected,scores=scores,nominated=['local'] if selected=='local' and scores['local']<min(v for k,v in scores.items() if k!='local') else [],
        selection='predeclared3-seed tuning IBS; availability/containing gain not local-representation gain',goal_achieved=False)


def run():
    data,train,arrays,contract=inputs();d,_,_,_,_,_,tune,_,_,_=data
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    if json.loads((ROOT/'state.json').read_text())['status']!='preflight_passed':raise ValueError('already started; inspect')
    report=dict(fits={},references={},cache_hashes={},confirmation_test_not_evaluated=True,goal_achieved=False)
    for seed in SEEDS:
        tc,tp=augmented(data,train,seed,arrays);vc,vp=augmented(data,tune,seed,arrays)
        report['cache_hashes'][str(seed)]={'train':tp,'tune':vp}
        ref=new_model('local',seed);a,k,curve,nll=prediction(ref,vc)
        report['references'][str(seed)]=summary(curve,d,tune,nll);del ref
        report['fits'][str(seed)]={}
        for arm in ARMS:
            write_json(ROOT/'state.json',dict(status='training_with_early_stop',seed=seed,arm=arm,goal_achieved=False))
            m=new_model(arm,seed);report['fits'][str(seed)][arm]=fit(m,tc,vc,d,tune,ROOT/f'seed{seed}'/arm,seed,arm);del m
        del tc,vc;torch.cuda.empty_cache()
    write_json(ROOT/'screening.json',report);write_json(ROOT/'nomination_lock.json',lock(report))
    write_json(ROOT/'state.json',dict(status='training_complete_pending_independent_replay',fits=9,goal_achieved=False))
    print(json.dumps(lock(report)),flush=True)


def verify(tag):
    data,train,arrays,contract=inputs();d,_,_,_,_,_,tune,_,_,_=data
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    rebuild_data(arrays);report=json.loads((ROOT/'screening.json').read_text());errors={}
    for seed in SEEDS:
        tc,tp=augmented(data,train,seed,arrays);vc,vp=augmented(data,tune,seed,arrays)
        if report['cache_hashes'][str(seed)]!={'train':tp,'tune':vp}:raise ValueError('teacher/current source cache differs')
        ref=new_model('local',seed);a,k,curve,nll=prediction(ref,vc)
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
    proof=dict(status='independently_verified',models=9,maximum_error=max(errors.values()),replay_errors=errors,
        report_sha256=sha256(ROOT/'screening.json'),contract_sha256=sha256(ROOT/'contract.json'),
        source_inputs_and_full_teacher_caches_rebuilt=True,early_stop_replayed=True,confirmation_test_not_evaluated=True)
    write_json(ROOT/f'verification_{tag}.json',proof)
    if all((ROOT/f'verification_{t}.json').exists() for t in ('a','b')):
        for t in ('a','b'):
            if json.loads((ROOT/f'verification_{t}.json').read_text())!=proof:raise ValueError('independent proof differs')
        write_json(ROOT/'state.json',dict(status='screening_complete_independently_verified',fits=9,nominated=lock(report)['nominated'],goal_achieved=False))
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




