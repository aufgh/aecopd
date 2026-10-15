"""完整V6预测冻结，V7仍端到端训练；不以冻结表示冒充整网比较。"""
import argparse
from dataclasses import dataclass,asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import RiskOutput,competing_nll
from training.v6.train import write_json
from patient_data.export_mimiciv import sha256
from training.v7 import experiment as first
from training.v7.data import cache
from training.v7.sequential_variants import SequentialV7,Variant
from training.v7.cached_research import checked as checked_cache,ROOT as CACHE_ROOT

ROOT=Path('training_runs/v7_sequential_research_2026-10-15/strong_anchor')
PROTOCOL=Path('docs/V7_STRONG_ANCHOR_PROTOCOL_2026-10-15.md')


@dataclass(frozen=True)
class StrongBatch:
    original:object
    positions:torch.Tensor
    def to(self,device):return StrongBatch(self.original.to(device),self.positions.to(device))


class StrongV7(SequentialV7):
    def __init__(self,base,parent,a,seed=42):
        super().__init__(base,parent,Variant('strong_anchor'),seed)
        self.register_buffer('anchor_a',torch.as_tensor(a['anchor_a']))
        self.register_buffer('anchor_k',torch.as_tensor(a['anchor_k']))
        self.register_buffer('anchor_valid',torch.as_tensor(a['valid']))

    def forward(self,b):
        if not self.anchor_valid[b.positions].all():raise ValueError('anchor unavailable')
        out=super().forward(b.original);delta=out.delta
        return RiskOutput((self.anchor_a[b.positions]+delta[:,:,0]).clamp(-10,6),
            self.anchor_k[b.positions]*torch.exp(.25*torch.tanh(delta[:,:,1])),delta)


def setup():
    data=first.setup();a,_,_=checked_cache();d=data[0]
    for k in ('hadm_ids','subject_ids','patient_splits'):np.testing.assert_array_equal(d.arrays[k],a[k])
    for k in ('times','events'):np.testing.assert_array_equal(getattr(d,k)[np.r_[a['train'],a['tune']]],a[k][np.r_[a['train'],a['tune']]])
    paths=(Path(__file__),Path('training/tests/test_v7_strong_anchor.py'),PROTOCOL,
           Path('training/v7/sequential_variants.py'),
           CACHE_ROOT/'cache_contract.json',CACHE_ROOT/'cache.npz')
    contract=dict(source_sha256={str(p):sha256(p) for p in paths},seed=42,max_epochs=16,
        parent_contract=json.loads((first.ROOT/'contract.json').read_text()),entire_v7_trainable=True)
    return data,a,contract


def wrapped(batches):return [(p,StrongBatch(b,torch.as_tensor(p))) for p,b in batches]


def run():
    if ROOT.exists():raise FileExistsError('strong anchor exists; no overwrite')
    started=time.perf_counter();data,a,contract=setup();d,c,ext,lab,rel,train,tune,base,drug,rad,*_=data
    ROOT.mkdir();write_json(ROOT/'contract.json',contract)
    tr=wrapped(cache(c,ext,lab,rel,drug,rad,train));va=wrapped(cache(c,ext,lab,rel,drug,rad,tune))
    m=StrongV7(base,drug['drug_parent'],a).cuda();groups,_=m.optimizer_groups();opt=torch.optim.AdamW(groups,weight_decay=.01)
    torch.manual_seed(42);rng=np.random.default_rng(42);early=EarlyStopping(6,8,1e-4);hist=[];best=None
    for epoch in range(17):
        tick=first.synchronized_time();online=None
        if epoch:
            m.train();total=0.;count=0
            for j in rng.permutation(len(tr)):
                p,cpu=tr[j];b=cpu.to('cuda');opt.zero_grad(set_to_none=True);out=m(b)
                loss=competing_nll(out,torch.as_tensor(d.times[p],device='cuda'),torch.as_tensor(d.events[p],device='cuda'))+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite strong loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('nonfinite strong gradient')
                opt.step();total+=float(loss.detach())*len(p);count+=len(p)
            online=total/count
        aa,kk,curve,nll=first.predict(m,va,d)
        if epoch==0:
            np.testing.assert_array_equal(aa,a['anchor_a'][tune]);np.testing.assert_array_equal(kk,a['anchor_k'][tune])
        metrics=first.parent.detail(d,tune,curve,nll)
        row=dict(epoch=epoch,annual=metrics['annual'],tuning_ibs=metrics['readmission']['integrated_brier_7_365'],
            tuning_death_ibs=metrics['death']['integrated_brier_7_365'],online_nll=online,seconds=first.synchronized_time()-tick)
        hist.append(row);_,stop=early.observe(epoch,-row['annual']['f1']);key=first.selection_key(row)
        if best is None or key>best:
            best=key;torch.save(dict(state_dict=m.state_dict(),epoch=epoch,threshold=row['annual']['threshold']),ROOT/'best.pt')
        torch.save(dict(state_dict=m.state_dict(),epoch=epoch,threshold=row['annual']['threshold']),ROOT/'last.pt')
        write_json(ROOT/'history.json',hist);print(json.dumps({k:v for k,v in row.items() if k!='annual'}|dict(f1=row['annual']['f1'])),flush=True)
        if stop:break
    outputs={}
    for which in ('best','last'):
        ck=torch.load(ROOT/f'{which}.pt',map_location='cuda',weights_only=True);m.load_state_dict(ck['state_dict'])
        aa,kk,curve,nll=first.predict(m,va,d);first.parent.save(ROOT/f'{which}_predictions.npz',d,tune,aa,kk,curve)
        outputs[which]=dict(epoch=ck['epoch'],threshold=ck['threshold'],metrics=first.parent.detail(d,tune,curve,nll,ck['threshold']),
            checkpoint_sha256=sha256(ROOT/f'{which}.pt'),predictions_sha256=sha256(ROOT/f'{which}_predictions.npz'))
    write_json(ROOT/'report.json',dict(outputs=outputs,early=asdict(early),completed_epochs=epoch,
        initial_anchor_error=0,seconds=time.perf_counter()-started,default_replaced=False))


def verify():
    data,a,contract=setup()
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('strong contract differs')
    d,c,ext,lab,rel,_,tune,base,drug,rad,*_=data
    va=wrapped(cache(c,ext,lab,rel,drug,rad,tune));m=StrongV7(base,drug['drug_parent'],a).cuda()
    r=json.loads((ROOT/'report.json').read_text())
    for which,info in r['outputs'].items():
        if sha256(ROOT/f'{which}.pt')!=info['checkpoint_sha256'] or sha256(ROOT/f'{which}_predictions.npz')!=info['predictions_sha256']:raise ValueError('strong artifact changed')
        ck=torch.load(ROOT/f'{which}.pt',map_location='cuda',weights_only=True);m.load_state_dict(ck['state_dict'])
        aa,kk,curve,nll=first.predict(m,va,d)
        with np.load(ROOT/f'{which}_predictions.npz') as z:
            for key,val in (('log_amplitude',aa),('shape',kk),('curve',curve)):np.testing.assert_array_equal(z[key],val)
        if first.parent.detail(d,tune,curve,nll,info['threshold'])!=info['metrics']:raise ValueError('strong metrics differ')
        torch.testing.assert_close(m.anchor_a.cpu(),torch.as_tensor(a['anchor_a']),rtol=0,atol=0)
        torch.testing.assert_close(m.anchor_k.cpu(),torch.as_tensor(a['anchor_k']),rtol=0,atol=0)
    hist=json.loads((ROOT/'history.json').read_text());early=EarlyStopping(6,8,1e-4)
    for row in hist:early.observe(row['epoch'],-row['annual']['f1'])
    if asdict(early)!=r['early'] or max(hist,key=first.selection_key)['epoch']!=r['outputs']['best']['epoch']:raise ValueError('strong selection differs')
    write_json(ROOT/'verification.json',dict(maximum_error=0,best_last_replayed=True,report_sha256=sha256(ROOT/'report.json')))
    print('strong anchor best/last replay error 0',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('train','verify'),required=True)
    run() if p.parse_args().mode=='train' else verify()
