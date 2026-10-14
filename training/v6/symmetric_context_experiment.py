"""单seed匹配背景校正；父实验与受保护模型不修改。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6 import unified_longitudinal_experiment as parent
from training.v6.symmetric_context import SymmetricContextModel
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import competing_nll
from training.v6.train import write_json
from training.v6.multihorizon_readout import horizon_readout, segmented_auc
from training.v6.evaluation import metrics, REPORT_DAYS

ROOT = Path('training_runs/v6_symmetric_context_2026-10-14')
PROTOCOL = Path('docs/V6_SYMMETRIC_CONTEXT_PROTOCOL_2026-10-14.md')
CONDITIONS = ('asymmetric', 'symmetric')


def setup():
    data = parent.setup()
    contract = dict(parent_contract=data[-2], conditions=list(CONDITIONS), no_new_encodings=True,
        source_sha256={str(p):sha256(p) for p in (Path(__file__),
            Path('training/v6/symmetric_context.py'), Path('training/tests/test_v6_symmetric_context.py'),
            PROTOCOL, parent.ROOT/'report.json', parent.ROOT/'verification.json', parent.ROOT/'contract.json',
            parent.ROOT/'history_unified/best.pt', parent.ROOT/'history_unified/tuning_predictions.npz')})
    proof=json.loads((parent.ROOT/'verification.json').read_text())
    if proof['maximum_error'] != 0 or proof['report_sha256'] != sha256(parent.ROOT/'report.json'):
        raise ValueError('parent proof changed')
    return data, contract


def readout(d, indices, curve, annual_threshold):
    a = dict(times=d.times[indices], events=d.events[indices], curve=curve)
    result = {}
    for label, cause in (('readmission', 1), ('death', 2)):
        m = metrics(a['times'], a['events'], curve[:, :, cause], cause=cause)
        y = ((a['events'][:, None] == cause) & (a['times'][:, None] <= np.arange(366)))
        brier = ((curve[:, :, cause] - y) ** 2).mean(0)
        early = {}
        for day in (1, 3):
            from training.v6.unified_longitudinal import best_f1
            early[str(day)] = dict(cases=int(y[:, day].sum()), brier=float(brier[day]),
                tuning_best_f1=best_f1(y[:, day], curve[:, day, cause]), auc=None,
                reason='fewer than 20 cases or controls')
        threshold = annual_threshold if cause == 1 else .2
        result[label] = dict(integrated_brier_1_365=float(np.trapezoid(brier[1:], np.arange(1,366))/364),
            segmented_auc=segmented_auc(m['auc_curve']), early=early,
            points={str(day):horizon_readout(a, day, cause, threshold) for day in REPORT_DAYS})
    return result


def preflight(fresh, b, d, positions):
    models={n:SymmetricContextModel(fresh,n).cuda().eval() for n in CONDITIONS}
    for p,q in zip(models['asymmetric'].parameters(),models['symmetric'].parameters(),strict=True):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    initial={n:m(b) for n,m in models.items()}
    for field in ('log_amplitude','shape'):
        torch.testing.assert_close(getattr(initial['asymmetric'],field),getattr(initial['symmetric'],field),rtol=0,atol=0)
    model=models['symmetric']; baseline={n:p.detach().clone() for n,p in model.original.baseline.named_parameters()}
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss=competing_nll(model(b),torch.as_tensor(d.times[positions],device='cuda'),
                          torch.as_tensor(d.events[positions],device='cuda'))
        loss.backward(); opt.step()
    if any(p.grad is not None or not torch.equal(p,baseline[n]) for n,p in model.original.baseline.named_parameters()):
        raise ValueError('H34 updated')
    changed={prefix:any(n.startswith(prefix) and not torch.equal(p,before[n]) for n,p in model.named_parameters() if n in before)
        for prefix in ('past_context.','temporal.','history_head.','original.encoder.base.lab.','original.delta_head.')}
    if not all(changed.values()): raise ValueError('active layer did not update')
    # nonzero fitted history head allows a real check of current-background sensitivity.
    altered=list(b); altered[4]=b[4].clone(); altered[4][b[5],0]+=.5
    # Both paths change final H34: compare only the delta, which excludes baseline output.
    delta_change=float((model(tuple(altered)).delta-model(b).delta).abs().max().detach())
    if delta_change<=0: raise ValueError('current background not consumed before baseline')
    return dict(initial_parameter_error=0, h34_frozen=True, actual_updates=changed,
                current_background_delta_sensitivity=delta_change)


def fit(fresh, condition, tr, va, d, tune):
    folder=ROOT/condition; folder.mkdir()
    model=SymmetricContextModel(fresh,condition).cuda();torch.manual_seed(42);rng=np.random.default_rng(42)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4)
    early=EarlyStopping(7,10,1e-4);best=(-np.inf,np.inf);history=[];steps=0;started=time.perf_counter()
    frozen={n:p.detach().clone() for n,p in model.original.baseline.named_parameters()}
    for epoch in range(46):
        train_nll=None
        if epoch:
            model.train(); total=0;count=0
            for j in rng.permutation(len(tr)):
                p,b=tr[j];opt.zero_grad(set_to_none=True);out=model(b)
                nll=competing_nll(out,torch.as_tensor(d.times[p],device='cuda'),torch.as_tensor(d.events[p],device='cuda'))
                loss=nll+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
                opt.step();steps+=1;total+=float(nll.detach())*len(p);count+=len(p)
            train_nll=total/count
        a,k,curve,nll=parent.predict(model,va,d);m=parent.detail(d,tune,curve,nll)
        f1=m['annual']['f1'];ibs=m['readmission']['integrated_brier_7_365']
        _,stop=early.observe(epoch,-f1)
        if f1>best[0]+1e-12 or (abs(f1-best[0])<=1e-12 and ibs<best[1]-1e-12):
            best=(f1,ibs);torch.save(dict(state_dict=model.state_dict(),epoch=epoch,threshold=m['annual']['threshold']),folder/'best.pt')
        history.append(dict(epoch=epoch,train_nll=train_nll,annual=m['annual'],tuning_ibs=ibs))
        write_json(folder/'history.json',history)
        if epoch%5==0:print(json.dumps(dict(condition=condition,epoch=epoch,f1=f1,ibs=ibs)),flush=True)
        if stop:break
    if any(not torch.equal(p,frozen[n]) for n,p in model.original.baseline.named_parameters()):raise ValueError('base changed')
    torch.save(dict(state_dict=model.state_dict(),epoch=epoch),folder/'last.pt')
    chosen=torch.load(folder/'best.pt',map_location='cuda',weights_only=True);model.load_state_dict(chosen['state_dict'])
    a,k,curve,nll=parent.predict(model,va,d);m=parent.detail(d,tune,curve,nll,chosen['threshold'])
    parent.save(folder/'tuning_predictions.npz',d,tune,a,k,curve)
    info=dict(best_epoch=chosen['epoch'],completed_epochs=epoch,early=asdict(early),
        stop_reason='patience_exhausted' if stop else 'max_epochs',optimizer_steps=steps,
        checkpoint_sha256=sha256(folder/'best.pt'),metrics=m,
        multihorizon=readout(d,tune,curve,chosen['threshold']),seconds=time.perf_counter()-started)
    write_json(folder/'result.json',info);return info


def run():
    if ROOT.exists():raise FileExistsError('experiment exists; no duplicate run')
    data,contract=setup();d,c,ext,lab,rel,train,tune,_,fresh,_,_,source=data
    ROOT.mkdir();write_json(ROOT/'contract.json',contract);write_json(ROOT/'source_report.json',source)
    selected=train[np.argsort(-np.diff(rel['offsets'])[train])[:16]]
    b=parent.cache(c,ext,lab,rel,selected,True)[0][1]
    write_json(ROOT/'preflight.json',preflight(fresh,b,d,selected))
    tr=parent.cache(c,ext,lab,rel,train,True);va=parent.cache(c,ext,lab,rel,tune,True)
    results={n:fit(fresh,n,tr,va,d,tune) for n in CONDITIONS}
    with np.load(parent.ROOT/'history_unified/tuning_predictions.npz') as old, np.load(ROOT/'asymmetric/tuning_predictions.npz') as new:
        error=float(np.abs(old['curve']-new['curve']).max())
    if error!=0:raise ValueError('matched control did not reproduce parent')
    selected=max(CONDITIONS,key=lambda n:(results[n]['metrics']['annual']['f1'],-results[n]['metrics']['readmission']['integrated_brier_7_365']))
    write_json(ROOT/'report.json',dict(results=results,matched_control_parent_error=error,selected=selected,
        default_changed=False,single_seed=True,confirmation_or_test_used=False))
    write_json(ROOT/'state.json',dict(status='trained_pending_replay'))
    print(json.dumps(dict(selected=selected,matched_control_parent_error=error)),flush=True)


def verify():
    data,contract=setup()
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    d,c,ext,lab,rel,_,tune,_,fresh,_,_,_=data;va=parent.cache(c,ext,lab,rel,tune,True)
    report=json.loads((ROOT/'report.json').read_text());errors={}
    for n in CONDITIONS:
        info=report['results'][n];model=SymmetricContextModel(fresh,n).cuda()
        chosen=torch.load(ROOT/n/'best.pt',map_location='cuda',weights_only=True)
        if sha256(ROOT/n/'best.pt')!=info['checkpoint_sha256']:raise ValueError('checkpoint changed')
        model.load_state_dict(chosen['state_dict']);a,k,curve,nll=parent.predict(model,va,d)
        with np.load(ROOT/n/'tuning_predictions.npz') as saved:
            for key,actual in (('hadm_ids',d.arrays['hadm_ids'][tune]),('subject_ids',d.arrays['subject_ids'][tune]),('events',d.events[tune]),('times',d.times[tune])):
                np.testing.assert_array_equal(saved[key],actual)
            error=max(float(np.abs(saved['curve']-curve).max()),float(np.abs(saved['log_amplitude']-a).max()),float(np.abs(saved['shape']-k).max()))
        if error!=0:raise ValueError('prediction replay differs')
        if parent.detail(d,tune,curve,nll,chosen['threshold'])!=info['metrics']:raise ValueError('metric replay differs')
        if readout(d,tune,curve,chosen['threshold'])!=info['multihorizon']:raise ValueError('multi-horizon replay differs')
        history=json.loads((ROOT/n/'history.json').read_text());early=EarlyStopping(7,10,1e-4)
        for r in history:early.observe(r['epoch'],-r['annual']['f1'])
        if asdict(early)!=info['early']:raise ValueError('early stop differs')
        best=min(history,key=lambda r:(-r['annual']['f1'],r['tuning_ibs'],r['epoch']))
        if best['epoch']!=chosen['epoch'] or best['annual']['threshold']!=chosen['threshold']:raise ValueError('selection differs')
        model.load_state_dict(torch.load(ROOT/n/'last.pt',map_location='cuda',weights_only=True)['state_dict'])
        _,_,last_curve,last_nll=parent.predict(model,va,d);last=parent.detail(d,tune,last_curve,last_nll)
        if last['annual']!=history[-1]['annual'] or last['readmission']['integrated_brier_7_365']!=history[-1]['tuning_ibs']:
            raise ValueError('last checkpoint differs')
        errors[n]=error
    write_json(ROOT/'verification.json',dict(maximum_error=0,errors=errors,
        report_sha256=sha256(ROOT/'report.json'),new_process=True))
    write_json(ROOT/'state.json',dict(status='independently_verified'))
    print(json.dumps(errors),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('train','verify'),required=True)
    try:{'train':run,'verify':verify}[p.parse_args().mode]()
    except Exception as e:
        if ROOT.exists():write_json(ROOT/'failure.json',dict(type=type(e).__name__,message=str(e)))
        raise
