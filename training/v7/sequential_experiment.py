"""顺序第一批：缓存复用、有限早停、独立best/last重放。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.early_stopping import EarlyStopping
from training.v6.train import write_json
from training.v7 import experiment as first
from training.v7.data import cache, make_v7_batch
from training.v7.sequential_variants import VARIANTS, SequentialV7, loss_terms

ROOT = Path('training_runs/v7_sequential_research_2026-10-15/phase1')
PROTOCOL = Path('docs/V7_SEQUENTIAL_RESEARCH_PRD_2026-10-15.md')
MAX_EPOCHS = 16


def setup():
    data = first.setup()
    previous = json.loads((first.ROOT/'contract.json').read_text())
    for p, digest in previous['source_sha256'].items():
        if sha256(Path(p)) != digest:
            raise ValueError('locked previous source changed: ' + p)
    paths = (Path(__file__), Path('training/v7/sequential_variants.py'),
             Path('training/tests/test_v7_sequential.py'), PROTOCOL, first.ROOT/'contract.json')
    return data, dict(parent=previous, variants=[asdict(v) for v in VARIANTS],
        source_sha256={str(p):sha256(p) for p in paths}, seed=42,
        early=dict(patience=6,min_epochs=8,min_delta=1e-4,max_epochs=MAX_EPOCHS),
        selection='annual F1; tie lower readmission IBS; tie earlier epoch',
        new_encodings=False, confirmation_or_test_used=False)


def tensor_labels(d, positions):
    return (torch.as_tensor(d.times[positions],device='cuda'),
            torch.as_tensor(d.events[positions],device='cuda'))


def preflight(base, drug, b, d, positions):
    results = {}
    for v in VARIANTS:
        m = SequentialV7(base,drug['drug_parent'],v).cuda().eval()
        before = {n:p.detach().clone() for n,p in m.named_parameters()}
        out = m(b); expected = m.baseline(b.context[b.current])
        for key in ('log_amplitude','shape'):
            torch.testing.assert_close(getattr(out,key),getattr(expected,key),atol=0,rtol=0)
        groups,_=m.optimizer_groups(); opt=torch.optim.AdamW(groups,weight_decay=.01)
        m.train()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            loss,_=loss_terms(m(b),*tensor_labels(d,positions),v)
            if not torch.isfinite(loss):raise ValueError('nonfinite preflight')
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        disconnected=[n for n,p in m.named_parameters() if p.requires_grad and p.grad is None]
        if disconnected:raise ValueError('unexpected disconnected parameter: '+repr(disconnected))
        if not any(not torch.equal(p,before[n]) for n,p in m.named_parameters() if n.startswith('delta_head.')):
            raise ValueError('risk path did not update')
        if any(not torch.equal(p,before[n]) for n,p in m.named_parameters() if n.startswith('baseline.')):
            raise ValueError('base changed')
        results[v.name]=dict(initial_error=0,actual_three_step_update=True,unexpected_disconnected=disconnected,
            trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad))
        del m,opt;torch.cuda.empty_cache()
    return results


def fit(base,drug,v,tr,va,d,tune):
    folder=ROOT/v.name;folder.mkdir()
    m=SequentialV7(base,drug['drug_parent'],v).cuda()
    groups,_=m.optimizer_groups();opt=torch.optim.AdamW(groups,weight_decay=.01)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);rng=np.random.default_rng(42)
    early=EarlyStopping(6,8,1e-4);hist=[];best=None;steps=0;stop=False
    start=first.synchronized_time()
    for epoch in range(MAX_EPOCHS+1):
        totals=None;train_seconds=0.
        if epoch:
            tick=first.synchronized_time();m.train();totals={k:0. for k in ('total','nll','penalty','focal','brier')};count=0
            for j in rng.permutation(len(tr)):
                p,cpu=tr[j];b=cpu.to('cuda');opt.zero_grad(set_to_none=True)
                total,parts=loss_terms(m(b),*tensor_labels(d,p),v)
                if not torch.isfinite(total):raise ValueError('nonfinite fit loss')
                total.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
                opt.step();steps+=1;count+=len(p);totals['total']+=float(total.detach())*len(p)
                for key,value in parts.items():totals[key]+=float(value.detach())*len(p)
            totals={k:x/count for k,x in totals.items()};train_seconds=first.synchronized_time()-tick
        a,k,curve,nll=first.predict(m,va,d);metrics=first.parent.detail(d,tune,curve,nll)
        row=dict(epoch=epoch,annual=metrics['annual'],tuning_ibs=metrics['readmission']['integrated_brier_7_365'],
            tuning_death_ibs=metrics['death']['integrated_brier_7_365'],nll=nll,online=totals,train_seconds=train_seconds)
        hist.append(row);_,stop=early.observe(epoch,-row['annual']['f1']);key=first.selection_key(row)
        if best is None or key>best:
            best=key;torch.save(dict(state_dict=m.state_dict(),epoch=epoch,threshold=row['annual']['threshold']),folder/'best.pt')
        torch.save(dict(state_dict=m.state_dict(),epoch=epoch,steps=steps),folder/'last.pt')
        write_json(folder/'history.json',hist)
        print(json.dumps(dict(variant=v.name,epoch=epoch,f1=row['annual']['f1'],ibs=row['tuning_ibs'],seconds=train_seconds)),flush=True)
        if stop:break
    fit_seconds=first.synchronized_time()-start
    outputs={}
    for which in ('last','best'):
        saved=torch.load(folder/f'{which}.pt',map_location='cuda',weights_only=True);m.load_state_dict(saved['state_dict'])
        a,k,curve,nll=first.predict(m,va,d)
        threshold=saved.get('threshold',hist[-1]['annual']['threshold'])
        metrics=first.parent.detail(d,tune,curve,nll,threshold)
        first.parent.save(folder/f'{which}_predictions.npz',d,tune,a,k,curve)
        outputs[which]=dict(epoch=saved['epoch'],metrics=metrics,threshold=threshold,
            checkpoint_sha256=sha256(folder/f'{which}.pt'),predictions_sha256=sha256(folder/f'{which}_predictions.npz'))
    info=dict(variant=asdict(v),outputs=outputs,completed_epochs=epoch,steps=steps,early=asdict(early),
        stop_reason='patience' if stop else 'budget_max_epochs',fit_seconds=fit_seconds,
        training_seconds=sum(r['train_seconds'] for r in hist))
    write_json(folder/'result.json',info)
    del m,opt;torch.cuda.empty_cache()
    return info


def run():
    if ROOT.exists():raise FileExistsError('phase1 exists; keep scene, no implicit rerun')
    started=time.perf_counter();data,contract=setup()
    d,c,ext,lab,rel,train,tune,base,drug,rad,*_=data
    ROOT.mkdir(parents=True);write_json(ROOT/'contract.json',contract)
    rich=train[np.argsort(-np.diff(rel['offsets'])[train])[:16]]
    cross=train[rad['present'][train].any(1)&drug['med_present'][train].any(1)][:16]
    p=np.unique(np.r_[rich,cross]);b=make_v7_batch(c,ext,lab,rel,drug,rad,p).to('cuda')
    write_json(ROOT/'preflight.json',preflight(base,drug,b,d,p));del b
    tr=cache(c,ext,lab,rel,drug,rad,train);va=cache(c,ext,lab,rel,drug,rad,tune)
    results={}
    for v in VARIANTS:
        results[v.name]=fit(base,drug,v,tr,va,d,tune)
        write_json(ROOT/'progress.json',dict(completed=list(results),remaining=[x.name for x in VARIANTS if x.name not in results]))
    write_json(ROOT/'report.json',dict(results=results,seconds=time.perf_counter()-started,
        default_replaced=False,confirmation_or_test_used=False,seed=42))


def verify():
    started=time.perf_counter();data,current=setup();saved=json.loads((ROOT/'contract.json').read_text())
    if current!=saved:raise ValueError('phase1 source reconstruction differs')
    d,c,ext,lab,rel,_,tune,base,drug,rad,*_=data
    va=cache(c,ext,lab,rel,drug,rad,tune);report=json.loads((ROOT/'report.json').read_text());proof={}
    for v in VARIANTS:
        folder=ROOT/v.name;info=report['results'][v.name]
        m=SequentialV7(base,drug['drug_parent'],v).cuda()
        for which,expected in info['outputs'].items():
            if sha256(folder/f'{which}.pt')!=expected['checkpoint_sha256'] or sha256(folder/f'{which}_predictions.npz')!=expected['predictions_sha256']:
                raise ValueError('artifact changed')
            ck=torch.load(folder/f'{which}.pt',map_location='cuda',weights_only=True);m.load_state_dict(ck['state_dict'])
            a,k,curve,nll=first.predict(m,va,d)
            with np.load(folder/f'{which}_predictions.npz') as z:
                for key,value in (('hadm_ids',d.arrays['hadm_ids'][tune]),('subject_ids',d.arrays['subject_ids'][tune]),('times',d.times[tune]),('events',d.events[tune])):
                    np.testing.assert_array_equal(z[key],value)
                for key,value in (('log_amplitude',a),('shape',k),('curve',curve)):
                    np.testing.assert_array_equal(z[key],value)
            if first.parent.detail(d,tune,curve,nll,expected['threshold'])!=expected['metrics']:
                raise ValueError('replay metrics differ')
            for key,value in m.baseline.state_dict().items():
                torch.testing.assert_close(value.cpu(),base.state_dict()[key],atol=0,rtol=0)
            if np.max(np.abs(curve.sum(2)-1))>1e-10 or (np.diff(curve[:,:,1:],axis=1)<-1e-12).any():
                raise ValueError('invalid probability curve')
        hist=json.loads((folder/'history.json').read_text());chosen=max(hist,key=first.selection_key)
        if chosen['epoch']!=info['outputs']['best']['epoch']:raise ValueError('selection differs')
        early=EarlyStopping(6,8,1e-4)
        for i,row in enumerate(hist):
            _,stop=early.observe(i,-row['annual']['f1'])
            if stop and i!=len(hist)-1:raise ValueError('trained after stop')
        if asdict(early)!=info['early']:raise ValueError('early state differs')
        proof[v.name]=dict(best_and_last_error=0,metrics_equal=True,selection_and_early_equal=True)
        print(json.dumps(dict(verified=v.name)),flush=True)
        del m;torch.cuda.empty_cache()
    write_json(ROOT/'verification.json',dict(results=proof,maximum_error=0,source_contract_equal=True,
        report_sha256=sha256(ROOT/'report.json'),seconds=time.perf_counter()-started))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('train','verify'),required=True)
    args=parser.parse_args();run() if args.mode=='train' else verify()
