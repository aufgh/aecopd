"""固定训练映射替代批次投影；零门配非零末层的独立快筛。"""
from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from patient_data.export_mimiciv import sha256
from training.v6.risk import RiskOutput
from training.v6.train import write_json
from training.v7 import cached_research as cache
from training.v7.query_replication import fit,verify_result
from training.v7.research_features import scaled
from training.v7.sequential_variants import Variant

ROOT=Path('training_runs/v7_sequential_research_2026-10-15/residual_controls')
PROTOCOL=Path('docs/V7_RESIDUAL_CONTROLS_PROTOCOL_2026-10-15.md')
VIEWS=('scaled_v7','fixed_projection','zero_gate')


def fixed_projection(x,h,fit):
    design=np.column_stack((np.ones(len(x)),x)).astype(np.float64)
    xt=design[fit];ht=h[fit].astype(np.float64);penalty=np.eye(xt.shape[1]);penalty[0,0]=0
    coefficients=np.linalg.solve(xt.T@xt+penalty,xt.T@ht)
    return h-design@coefficients,coefficients


def features(a):
    fit=a['train'];n=len(a['times']);use=np.zeros(n,bool);use[fit]=True
    original=a['v7_hidden'];residual,b=fixed_projection(a['context'],original,fit)
    views={}
    for name,value in (('scaled_v7',original),('fixed_projection',residual)):
        x=np.zeros((n,512),np.float32);x[:,:256],_=scaled(value,np.ones_like(value,bool),use);views[name]=x
    views['zero_gate']=views['scaled_v7'].copy()
    return views,b


@dataclass(frozen=True)
class DenseBatch:
    x:torch.Tensor
    a:torch.Tensor
    k:torch.Tensor
    def to(self,device):return DenseBatch(self.x.to(device),self.a.to(device),self.k.to(device))


class Model(nn.Module):
    def __init__(self,view,seed=42):
        super().__init__();self.reader=cache.Reader('scaled_v7',seed);self.view=view
        if view=='zero_gate':
            self.alpha=nn.Parameter(torch.zeros(2))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                nn.init.normal_(self.reader.head[-1].weight,std=.02)
                nn.init.zeros_(self.reader.head[-1].bias)

    def optimizer_groups(self):return [dict(params=self.parameters(),lr=.001)],{}

    def forward(self,b):
        if self.view!='zero_gate':return self.reader(b.x,b.a,b.k)
        raw=self.reader.head(b.x).reshape(-1,2,2)*torch.tanh(self.alpha)[None,:,None]
        return RiskOutput((b.a+raw[:,:,0]).clamp(-10,6),b.k*torch.exp(.25*torch.tanh(raw[:,:,1])),raw)


def batches(a,x,ix):
    return [(p,DenseBatch(torch.as_tensor(x[p]),torch.as_tensor(a['anchor_a'][p]),torch.as_tensor(a['anchor_k'][p])))
            for start in range(0,len(ix),256) for p in [ix[start:start+256]]]


def setup():
    a,_,_=cache.checked();d=type('Data',(),dict(times=a['times'],events=a['events'],arrays=a))()
    paths=(Path(__file__),PROTOCOL,Path('training/v7/query_replication.py'),
           Path('training/tests/test_v7_residual_controls.py'),cache.ROOT/'cache_contract.json')
    return a,d,dict(source_sha256={str(p):sha256(p) for p in paths},views=VIEWS,seed=42,frozen_anchor_seed=42)


def train():
    if ROOT.exists():raise FileExistsError('residual controls exist')
    a,d,contract=setup();ROOT.mkdir();write_json(ROOT/'contract.json',contract);views,b=features(a)
    np.savez_compressed(ROOT/'features.npz',**views,projection=b)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    results={}
    for name in VIEWS:
        factory=lambda base,parent,variant,seed:Model(variant.name,seed)
        results[name]=fit(None,dict(drug_parent=None),Variant(name),batches(a,views[name],a['train']),
            batches(a,views[name],a['tune']),d,a['tune'],model_factory=factory,folder=ROOT/name,
            seed=42,max_epochs=24,min_epochs=8,patience=6)
    write_json(ROOT/'report.json',dict(results=results,feature_sha256=sha256(ROOT/'features.npz'),default_replaced=False))


def verify():
    a,d,current=setup()
    if json.loads(json.dumps(current))!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('residual source differs')
    report=json.loads((ROOT/'report.json').read_text())
    if sha256(ROOT/'features.npz')!=report['feature_sha256']:raise ValueError('residual inputs changed')
    actual,b=features(a);saved=cache.load_npz(ROOT/'features.npz');np.testing.assert_allclose(b,saved['projection'],atol=1e-10,rtol=0)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    for name in VIEWS:
        np.testing.assert_allclose(actual[name],saved[name],atol=1e-6,rtol=0)
        m=Model(name).cuda();verify_result(m,batches(a,saved[name],a['tune']),d,a['tune'],ROOT/name,report['results'][name],patience=6,min_epochs=8)
    write_json(ROOT/'verification.json',dict(maximum_prediction_error=0,feature_tolerance=1e-6,
        models_replayed=3,report_sha256=sha256(ROOT/'report.json')));print('three residual controls independently replayed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('train','verify'),required=True)
    train() if p.parse_args().mode=='train' else verify()
