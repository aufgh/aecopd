"""原V6仅更换冻结向量数据表示；共同早停，指定调参，独立重放。"""
import argparse
from dataclasses import replace,asdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.data_geometry_audit import ROOT as DATA
from training.v6.text_geometry import fit_geometry,transform_rows
from training.v6.early_stopping import EarlyStopping
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.warm_residual import BASE_SEEDS
from training.v6.horizon_objective import restore
from training.v6.module_surgery import SurgeryModel,probability_curve,member_nll
from training.v6.module_surgery_experiment import batches,parameters,validation,summary,save
from training.v6.train import write_json

ROOT=Path('training_runs/v6_text_geometry_2026-10-13')
ARMS=('raw','centered','whitened');SEEDS=(42,7,101)
PLAN=Path('docs/V6_DATA_ORGANIZATION_PLAN_2026-10-13.md')
CAP=60


def inputs():
    d,c,h,ext,_,bins,tune,_,search,prior=checked()
    manifest=json.loads((DATA/'manifest.json').read_text())
    if manifest['data_provenance']!=d.provenance:raise ValueError('geometry data mismatch')
    for p,digest in manifest['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('geometry source changed: '+p)
    if sha256(DATA/'transform.npz')!=manifest['transform_sha256']:raise ValueError('transform differs')
    representations={'raw':d.text_vectors}
    for name,item in manifest['derivatives'].items():
        if sha256(Path(item['path']))!=item['sha256']:raise ValueError('derivative changed')
        representations[name]=np.load(item['path'],mmap_mode='r',allow_pickle=False)
    train=np.flatnonzero(d.arrays['patient_splits']=='train')
    cache={}
    for name,vector in representations.items():
        nd=replace(d,text_vectors=vector);nc=replace(c,base=nd)
        cache[name]={fold:batches(nd,nc,ext,ix) for fold,ix in [('train',train),('tune',tune)]}
    protected={str(REFERENCE/f'seed{s}/control/best.pt'):sha256(REFERENCE/f'seed{s}/control/best.pt') for s in (42,7)}
    parent=Path(search['input_config']['parent'])
    protected.update({str(parent/f'seed{s}/survival_base/best.pt'):sha256(parent/f'seed{s}/survival_base/best.pt') for s in (42,7)})
    sources={**search['source_sha256'],**prior['source_sha256'],**manifest['source_sha256']}
    for p in ('training/v6/data_geometry_experiment.py','training/v6/early_stopping.py'):
        sources[p]=sha256(Path(p))
    contract=dict(source_sha256=sources,plan_sha256=sha256(PLAN),data_manifest_sha256=sha256(DATA/'manifest.json'),
        protected_weights=protected,source_mapping={str(s):BASE_SEEDS[s] for s in SEEDS},seeds=list(SEEDS),arms=list(ARMS),
        early_stop=dict(patience=10,min_epochs=15,min_delta=1e-5,max_epochs=CAP,monitor='specified1531 tuning IBS7-365'),
        optimizer=dict(name='AdamW',lr=1e-4,weight_decay=1e-4,delta_penalty=1e-4,clip=1.,warmup=5,
            scheduler='fixed100-reference cosine; same across arms; cap60 and early stopping active'),
        training_only_geometry=True,train_admissions=len(train),tuning_admissions=len(tune),
        confirmation_test_evaluated=False,selection='mean tuning IBS across3 clinical seeds; compare matched raw and old V6',
        limits=['old V6 already selected on exposed validation','3 clinical seeds reuse2 H34 sources',
            'geometric separation is not clinical prognostic increment','no clinical facts created or old vectors overwritten'])
    return d,h,bins,search,cache,tune,contract


def new_model(h,bins,search,seed):
    base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
    m=SurgeryModel(base,seed,'control',bins).cuda();torch.manual_seed(seed)
    return m


def verify_geometry(d):
    rows,fields,weights=[],[],[]
    from collections import Counter
    for i,refs in enumerate(d.text_refs):
        if d.arrays['patient_splits'][i]!='train':continue
        count=Counter(f for f,_ in refs)
        for f,row in refs:rows.append(row);fields.append(f);weights.append(1/count[f])
    order=np.argsort(rows);ix=np.array(rows)[order];fs=np.array(fields)[order];w=np.array(weights,np.float32)[order]
    mean,mat,_=fit_geometry(torch.as_tensor(np.asarray(d.text_vectors[ix],np.float32),device='cuda'),
        torch.as_tensor(fs,device='cuda'),torch.as_tensor(w,device='cuda'))
    with np.load(DATA/'transform.npz') as z:
        np.testing.assert_allclose(mean.cpu().numpy(),z['means'],atol=1e-7,rtol=0)
        np.testing.assert_allclose(mat.cpu().numpy(),z['whitening'],atol=1e-4,rtol=0)
        all_fields=z['field_ids']
    raw=torch.as_tensor(np.asarray(d.text_vectors,np.float32),device='cuda');f=torch.as_tensor(all_fields,device='cuda')
    errors={}
    for name,k in [('centered',None),('whitened',mat)]:
        rebuilt=transform_rows(raw,f,mean,k).cpu().numpy().astype(np.float16)
        saved=np.load(DATA/f'{name}.float16.npy',mmap_mode='r')
        error=float(np.abs(rebuilt.astype(np.float32)-saved.astype(np.float32)).max())
        if error>1e-3:raise ValueError('geometry rebuild differs')
        errors[name]=error
    return errors


def preflight():
    d,h,bins,search,cache,tune,contract=inputs()
    if ROOT.exists():raise FileExistsError('preflight already exists; inspect')
    ROOT.mkdir()
    write_json(ROOT/'contract.json',contract)
    geometry_errors=verify_geometry(d)
    reference=None;reference_out=None
    for arm in ARMS:
        m=new_model(h,bins,search,42).eval();state={k:v.detach().clone() for k,v in m.state_dict().items()}
        pos,hist,b,t,e=cache[arm]['train'][0]
        with torch.no_grad():out=m(hist,b)
        if reference is None:reference,reference_out=state,out
        else:
            if any(not torch.equal(v,reference[k]) for k,v in state.items()):raise ValueError('initial weights differ')
            if not torch.equal(out.log_amplitude,reference_out.log_amplitude) or not torch.equal(out.shape,reference_out.shape):
                raise ValueError('epoch0 risks differ')
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
        m.train();opt.zero_grad();loss=member_nll(m(hist,b),t,e);loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        if torch.equal(m.delta_head[-1].weight,state['delta_head.2.weight']):raise ValueError('no clinical update')
        del m
    write_json(ROOT/'preflight.json',dict(status='passed',geometry_rebuild_errors=geometry_errors,
        three_arm_identical_initial_weights_and_risks=True,real_gpu_single_update=True,gpu=torch.cuda.get_device_name(0)))
    write_json(ROOT/'state.json',dict(status='preflight_passed',completed_fits=0,goal_achieved=False))
    print(json.dumps(dict(status='preflight_passed',geometry_errors=geometry_errors)),flush=True)


def fit(m,cache,d,tune,folder,seed,arm):
    folder.mkdir(parents=True);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    frozen={k:v.detach().clone() for k,v in m.baseline.state_dict().items()};early=EarlyStopping()
    rows=[];start=time.perf_counter();reason='max_epochs'
    for epoch in range(CAP+1):
        train_loss=None
        if epoch:
            m.train();rate=1e-4*min(1.,epoch/5)*(.1+.45*(1+math.cos(math.pi*max(0,epoch-5)/95)))
            for g in opt.param_groups:g['lr']=rate
            total,n=0.,0
            for j in rng.permutation(len(cache['train'])):
                _,history,b,t,e=cache['train'][j];opt.zero_grad(set_to_none=True)
                out=m(history,b);nll=member_nll(out,t,e);loss=nll+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
                opt.step();total+=float(nll.detach())*len(t);n+=len(t)
            train_loss=total/n
        score=validation(m,cache['tune'],d.times[tune],d.events[tune]);store,stop=early.observe(epoch,score['ibs'])
        rows.append(dict(epoch=epoch,train_nll=train_loss,**{'tuning_'+k:v for k,v in score.items()},
            early_stale=early.stale,early_anchor=early.anchor))
        if store:torch.save(dict(state_dict=m.state_dict(),epoch=epoch,arm=arm,seed=seed,tuning_ibs=score['ibs']),folder/'best.pt')
        write_json(folder/'history.json',rows)
        if epoch%5==0 or stop:print(json.dumps(dict(seed=seed,arm=arm,epoch=epoch,best_epoch=early.best_epoch,ibs=score['ibs'],early_stop=stop)),flush=True)
        if stop:reason='patience_exhausted';break
    if any(not torch.equal(v,m.baseline.state_dict()[k]) for k,v in frozen.items()):raise ValueError('H34 changed')
    m.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
    a,k,nll=parameters(m,cache['tune']);curve=probability_curve(a,k)
    save(folder/'tuning_predictions.npz',d,tune,a,k,curve)
    info=dict(seed=seed,arm=arm,completed_epochs=rows[-1]['epoch'],best_epoch=early.best_epoch,stop_reason=reason,
        early_state=asdict(early),elapsed_seconds=time.perf_counter()-start,checkpoint_sha256=sha256(folder/'best.pt'),
        metrics=summary(curve,d,tune,nll),frozen_baseline_verified=True)
    write_json(folder/'training_result.json',info)
    return info


def lock(report):
    scores={arm:float(np.mean([report['fits'][str(s)][arm]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS])) for arm in ARMS}
    scores['official_V6']=float(np.mean([report['references'][str(s)]['readmission']['integrated_brier_7_365'] for s in SEEDS]))
    selected=min(scores,key=scores.get)
    nominated=[]
    if selected in ('centered','whitened') and scores[selected]<min(scores['raw'],scores['official_V6']):nominated=[selected]
    return dict(selected=selected,nominated=nominated,scores=scores,criterion='predeclared three-seed mean tuning IBS',
                no_confirmation_test_evaluation=True,goal_achieved=False)


def run():
    d,h,bins,search,cache,tune,contract=inputs()
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    if json.loads((ROOT/'state.json').read_text())['status']!='preflight_passed':raise ValueError('already started; inspect instead of repeated fits')
    results,references={},{}
    for seed in SEEDS:
        original=new_model(h,bins,search,seed)
        original.load_state_dict(torch.load(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt',map_location='cuda',weights_only=True)['state_dict'])
        a,k,nll=parameters(original,cache['raw']['tune']);curve=probability_curve(a,k)
        references[str(seed)]=summary(curve,d,tune,nll);del original
        results[str(seed)]={}
        for arm in ARMS:
            write_json(ROOT/'state.json',dict(status='training_with_early_stop',seed=seed,arm=arm,goal_achieved=False))
            m=new_model(h,bins,search,seed)
            results[str(seed)][arm]=fit(m,cache[arm],d,tune,ROOT/f'seed{seed}'/arm,seed,arm)
            del m;torch.cuda.empty_cache()
    report=dict(fits=results,references=references,goal_achieved=False,confirmation_test_not_evaluated=True)
    write_json(ROOT/'screening.json',report)
    write_json(ROOT/'nomination_lock.json',lock(report))
    write_json(ROOT/'state.json',dict(status='training_complete_pending_independent_replay',fits=9,goal_achieved=False))
    print(json.dumps(lock(report)),flush=True)


def verify(tag):
    d,h,bins,search,cache,tune,contract=inputs()
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    geometry=verify_geometry(d);errors={}
    report=json.loads((ROOT/'screening.json').read_text())
    for seed in SEEDS:
        for arm in ARMS:
            folder=ROOT/f'seed{seed}'/arm;info=json.loads((folder/'training_result.json').read_text())
            if info!=report['fits'][str(seed)][arm] or sha256(folder/'best.pt')!=info['checkpoint_sha256']:raise ValueError('checkpoint differs')
            history=json.loads((folder/'history.json').read_text());early=EarlyStopping();first_stop=None
            for row in history:
                _,stop=early.observe(row['epoch'],row['tuning_ibs'])
                if stop and first_stop is None:first_stop=row['epoch']
            if asdict(early)!=info['early_state'] or early.best_epoch!=info['best_epoch'] or info['completed_epochs']!=len(history)-1:
                raise ValueError('early stop replay differs')
            if info['stop_reason']=='patience_exhausted' and first_stop!=info['completed_epochs']:raise ValueError('stop condition bypassed')
            m=new_model(h,bins,search,seed);state=torch.load(folder/'best.pt',map_location='cuda',weights_only=True)
            m.load_state_dict(state['state_dict']);a,k,nll=parameters(m,cache[arm]['tune']);curve=probability_curve(a,k)
            with np.load(folder/'tuning_predictions.npz') as z:
                np.testing.assert_array_equal(z['times'],d.times[tune]);np.testing.assert_array_equal(z['events'],d.events[tune])
                error=float(np.abs(curve-z['curve']).max())
            if error>3e-7 or summary(curve,d,tune,nll)!=info['metrics']:raise ValueError('prediction/metric differs')
            errors[f'{seed}/{arm}']=error;del m
    if lock(report)!=json.loads((ROOT/'nomination_lock.json').read_text()):raise ValueError('lock differs')
    proof=dict(status='independently_verified',maximum_error=max(errors.values()),models=9,geometry_rebuild_errors=geometry,
        early_stopping_replayed=True,report_sha256=sha256(ROOT/'screening.json'),contract_sha256=sha256(ROOT/'contract.json'))
    write_json(ROOT/f'verification_{tag}.json',proof)
    if all((ROOT/f'verification_{t}.json').exists() for t in ('a','b')):
        write_json(ROOT/'state.json',dict(status='screening_complete_independently_verified',fits=9,
            nominated=lock(report)['nominated'],goal_achieved=False))
    print(json.dumps(proof),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('preflight','run','verify'),required=True)
    p.add_argument('--tag',choices=('a','b'),default='a');args=p.parse_args()
    try:
        if args.mode=='preflight':preflight()
        elif args.mode=='run':run()
        else:verify(args.tag)
    except Exception as error:
        if ROOT.exists():write_json(ROOT/f'failure_{args.mode}_{args.tag}.json',dict(error=repr(error),goal_achieved=False))
        raise


if __name__=='__main__':main()
