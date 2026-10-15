"""五个已选读取条件的两个追加种子；同冻结强基座。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import competing_nll
from training.v6.train import write_json
from training.v7 import cached_research as old

ROOT=Path('training_runs/v7_sequential_research_2026-10-15/cached_followup')
PROTOCOL=Path('docs/V7_CACHED_FOLLOWUP_PROTOCOL_2026-10-15.md')
VIEWS=('global','availability','linear','v7_frozen','routing_prior')
SEEDS=(7,101)


def source():
    a,f,_=old.checked();proof=json.loads((old.ROOT/'verification.json').read_text())
    if proof['maximum_error']!=0 or proof['report_sha256']!=sha256(old.ROOT/'report.json'):raise ValueError('cached proof stale')
    paths=(Path(__file__),PROTOCOL,old.ROOT/'report.json',old.ROOT/'verification.json',old.ROOT/'cache_contract.json')
    return a,f,dict(source_sha256={str(p):sha256(p) for p in paths},seeds=SEEDS,views=VIEWS,frozen_anchor_seed=42)


def metrics(a,tune,curve,nll,threshold=None):
    d=type('Data',(),dict(times=a['times'],events=a['events']))()
    return old.first.parent.detail(d,tune,curve,nll,threshold)


def fit(a,f,view,seed):
    folder=ROOT/f'seed{seed}'/view;folder.mkdir(parents=True)
    tr,va=a['train'],a['tune'];x=torch.as_tensor(old.views(a,f,tr,view),device='cuda')
    m=old.Reader(view,seed).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
    torch.manual_seed(seed);rng=np.random.default_rng(seed);early=EarlyStopping(6,8,1e-4);hist=[];best=None;started=time.perf_counter()
    for epoch in range(25):
        if epoch:
            m.train();order=rng.permutation(tr)
            for start in range(0,len(order),256):
                p=order[start:start+256];opt.zero_grad(set_to_none=True)
                out=m(x[p],torch.as_tensor(a['anchor_a'][p],device='cuda'),torch.as_tensor(a['anchor_k'][p],device='cuda'))
                loss=competing_nll(out,*old.labels(a,p))+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite followup loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        aa,kk,curve,nll=old.predict(m,x,a,va);score=metrics(a,va,curve,nll)
        row=dict(epoch=epoch,annual=score['annual'],tuning_ibs=score['readmission']['integrated_brier_7_365'])
        hist.append(row);_,stop=early.observe(epoch,-row['annual']['f1']);key=old.first.selection_key(row)
        if best is None or key>best:
            best=key;torch.save(dict(state_dict=m.state_dict(),epoch=epoch,threshold=row['annual']['threshold']),folder/'best.pt')
        torch.save(dict(state_dict=m.state_dict(),epoch=epoch,threshold=row['annual']['threshold']),folder/'last.pt')
        if stop:break
    write_json(folder/'history.json',hist);outputs={}
    for which in ('best','last'):
        ck=torch.load(folder/f'{which}.pt',map_location='cuda',weights_only=True);m.load_state_dict(ck['state_dict'])
        aa,kk,curve,nll=old.predict(m,x,a,va);score=metrics(a,va,curve,nll,ck['threshold'])
        np.savez_compressed(folder/f'{which}_predictions.npz',curve=curve,log_amplitude=aa,shape=kk)
        outputs[which]=dict(epoch=ck['epoch'],threshold=ck['threshold'],metrics=score,
            checkpoint_sha256=sha256(folder/f'{which}.pt'),predictions_sha256=sha256(folder/f'{which}_predictions.npz'))
    info=dict(outputs=outputs,early=asdict(early),completed_epochs=epoch,seconds=time.perf_counter()-started)
    write_json(folder/'result.json',info)
    print(json.dumps(dict(seed=seed,view=view,best=outputs['best']['epoch'],f1=outputs['best']['metrics']['annual']['f1'],
        ibs=outputs['best']['metrics']['readmission']['integrated_brier_7_365'],seconds=info['seconds'])),flush=True)
    del m,x;torch.cuda.empty_cache();return info


def train():
    if ROOT.exists():raise FileExistsError('followup exists')
    a,f,contract=source();ROOT.mkdir();write_json(ROOT/'contract.json',contract);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    results={str(seed):{view:fit(a,f,view,seed) for view in VIEWS} for seed in SEEDS}
    write_json(ROOT/'report.json',dict(results=results,default_replaced=False,frozen_anchor_seed=42))


def verify():
    a,f,current=source();saved=json.loads((ROOT/'contract.json').read_text())
    if json.loads(json.dumps(current))!=saved:raise ValueError('followup source differs')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    report=json.loads((ROOT/'report.json').read_text());va=a['tune'];count=0
    for seed in SEEDS:
        for view in VIEWS:
            folder=ROOT/f'seed{seed}'/view;info=report['results'][str(seed)][view]
            m=old.Reader(view,seed).cuda();x=torch.as_tensor(old.views(a,f,a['train'],view),device='cuda')
            for which,expected in info['outputs'].items():
                if sha256(folder/f'{which}.pt')!=expected['checkpoint_sha256'] or sha256(folder/f'{which}_predictions.npz')!=expected['predictions_sha256']:raise ValueError('followup artifact changed')
                ck=torch.load(folder/f'{which}.pt',map_location='cuda',weights_only=True);m.load_state_dict(ck['state_dict'])
                aa,kk,curve,nll=old.predict(m,x,a,va)
                with np.load(folder/f'{which}_predictions.npz') as z:
                    for k,value in (('curve',curve),('log_amplitude',aa),('shape',kk)):np.testing.assert_array_equal(z[k],value)
                if metrics(a,va,curve,nll,expected['threshold'])!=expected['metrics']:raise ValueError('followup metrics differ')
            hist=json.loads((folder/'history.json').read_text());early=EarlyStopping(6,8,1e-4)
            for row in hist:early.observe(row['epoch'],-row['annual']['f1'])
            if asdict(early)!=info['early'] or max(hist,key=old.first.selection_key)['epoch']!=info['outputs']['best']['epoch']:raise ValueError('followup selection differs')
            count+=1;del m,x
    write_json(ROOT/'verification.json',dict(maximum_error=0,models=count,report_sha256=sha256(ROOT/'report.json')))
    print(json.dumps(dict(followup_verified=count,maximum_error=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('train','verify'),required=True)
    train() if p.parse_args().mode=='train' else verify()
