"""药名语义/同分子记录动态：同冻结基座小容量配对筛查。"""
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
from training.v7 import cached_research as base
from training.v7.drug_context_features import ROOT as DATA,view as feature_view,PROTOCOL

ROOT=Path('training_runs/v7_sequential_research_2026-10-15/drug_screen')
VIEWS=('name_source','name_identity','name_random','name_semantic',
       'dynamics_source','dynamics_main','dynamics_slope','dynamics_route')


def setup():
    a,_,_=base.checked();features=base.load_npz(DATA/'interface.npz');manifest=json.loads((DATA/'manifest.json').read_text())
    proof=json.loads((DATA/'verification.json').read_text())
    if proof['manifest_sha256']!=sha256(DATA/'manifest.json') or proof['maximum_error']!=0:raise ValueError('drug feature proof stale')
    for p,h in manifest['source_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('drug feature source changed')
    if sha256(DATA/'interface.npz')!=manifest['output_sha256']:raise ValueError('drug feature array changed')
    for k in ('hadm_ids','subject_ids','patient_splits'):np.testing.assert_array_equal(features[k],a[k])
    paths=(Path(__file__),PROTOCOL,DATA/'manifest.json',DATA/'verification.json',DATA/'interface.npz',base.ROOT/'cache_contract.json')
    return a,features,dict(source_sha256={str(p):sha256(p) for p in paths},views=VIEWS,seed=42)


def score(a,ix,curve,nll,threshold=None):
    return base.first.parent.detail(type('Data',(),dict(times=a['times'],events=a['events']))(),ix,curve,nll,threshold)


def fit(a,features,view):
    folder=ROOT/view;folder.mkdir();tr,va=a['train'],a['tune'];x=torch.as_tensor(feature_view(features,tr,view),device='cuda')
    m=base.Reader(view).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
    torch.manual_seed(42);rng=np.random.default_rng(42);early=EarlyStopping(6,8,1e-4);hist=[];best=None;started=time.perf_counter()
    for epoch in range(25):
        if epoch:
            m.train();order=rng.permutation(tr)
            for start in range(0,len(order),256):
                p=order[start:start+256];opt.zero_grad(set_to_none=True)
                out=m(x[p],torch.as_tensor(a['anchor_a'][p],device='cuda'),torch.as_tensor(a['anchor_k'][p],device='cuda'))
                loss=competing_nll(out,*base.labels(a,p))+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite drug fit')
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        aa,kk,curve,nll=base.predict(m,x,a,va);metrics=score(a,va,curve,nll)
        row=dict(epoch=epoch,annual=metrics['annual'],tuning_ibs=metrics['readmission']['integrated_brier_7_365'])
        hist.append(row);_,stop=early.observe(epoch,-row['annual']['f1']);key=base.first.selection_key(row)
        if best is None or key>best:
            best=key;torch.save(dict(state_dict=m.state_dict(),epoch=epoch,threshold=row['annual']['threshold']),folder/'best.pt')
        torch.save(dict(state_dict=m.state_dict(),epoch=epoch,threshold=row['annual']['threshold']),folder/'last.pt')
        if stop:break
    write_json(folder/'history.json',hist);outputs={}
    for which in ('best','last'):
        ck=torch.load(folder/f'{which}.pt',map_location='cuda',weights_only=True);m.load_state_dict(ck['state_dict'])
        aa,kk,curve,nll=base.predict(m,x,a,va)
        np.savez_compressed(folder/f'{which}_predictions.npz',log_amplitude=aa,shape=kk,curve=curve)
        outputs[which]=dict(epoch=ck['epoch'],threshold=ck['threshold'],metrics=score(a,va,curve,nll,ck['threshold']),
            checkpoint_sha256=sha256(folder/f'{which}.pt'),predictions_sha256=sha256(folder/f'{which}_predictions.npz'))
    info=dict(outputs=outputs,completed_epochs=epoch,early=asdict(early),seconds=time.perf_counter()-started)
    write_json(folder/'result.json',info);print(json.dumps(dict(view=view,epoch=outputs['best']['epoch'],
        f1=outputs['best']['metrics']['annual']['f1'],ibs=outputs['best']['metrics']['readmission']['integrated_brier_7_365'],seconds=info['seconds'])),flush=True)
    del m,x;torch.cuda.empty_cache();return info


def train():
    if ROOT.exists():raise FileExistsError('drug screen exists')
    a,f,contract=setup();ROOT.mkdir();write_json(ROOT/'contract.json',contract)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    results={name:fit(a,f,name) for name in VIEWS}
    write_json(ROOT/'report.json',dict(results=results,default_replaced=False,
        unavailable=['completed source-checked full clinical facts NPZ','confirmed home NIV absence','complete heart failure regimen'],
        limits=['drug-name embeddings only','recorded administration changes, not equivalent clinical escalation']))


def verify():
    a,f,current=setup()
    if json.loads(json.dumps(current))!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('drug screen contract changed')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;report=json.loads((ROOT/'report.json').read_text());va=a['tune']
    for name,info in report['results'].items():
        folder=ROOT/name;x=torch.as_tensor(feature_view(f,a['train'],name),device='cuda');m=base.Reader(name).cuda()
        for which,out in info['outputs'].items():
            if sha256(folder/f'{which}.pt')!=out['checkpoint_sha256'] or sha256(folder/f'{which}_predictions.npz')!=out['predictions_sha256']:raise ValueError('drug fit artifact changed')
            ck=torch.load(folder/f'{which}.pt',map_location='cuda',weights_only=True);m.load_state_dict(ck['state_dict'])
            aa,kk,curve,nll=base.predict(m,x,a,va)
            with np.load(folder/f'{which}_predictions.npz') as z:
                for k,v in (('log_amplitude',aa),('shape',kk),('curve',curve)):np.testing.assert_array_equal(z[k],v)
            if score(a,va,curve,nll,out['threshold'])!=out['metrics']:raise ValueError('drug screen metrics differ')
        hist=json.loads((folder/'history.json').read_text());early=EarlyStopping(6,8,1e-4)
        for row in hist:early.observe(row['epoch'],-row['annual']['f1'])
        if asdict(early)!=info['early'] or max(hist,key=base.first.selection_key)['epoch']!=info['outputs']['best']['epoch']:raise ValueError('drug selection differs')
        del m,x
    write_json(ROOT/'verification.json',dict(maximum_error=0,models_replayed=len(VIEWS),report_sha256=sha256(ROOT/'report.json')))
    print(json.dumps(dict(drug_models_verified=len(VIEWS),maximum_error=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('train','verify'),required=True)
    train() if p.parse_args().mode=='train' else verify()
