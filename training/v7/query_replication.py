"""单个有明显F1线索的结构消融，追加一对同预算seed7。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json
from training.v6.risk import competing_nll
from training.v6.early_stopping import EarlyStopping
from training.v7 import sequential_experiment as phase
from training.v7.sequential_variants import Variant,SequentialV7
from training.v7.data import cache

ROOT=Path('training_runs/v7_sequential_research_2026-10-15/query_replication')
PROTOCOL=Path('docs/V7_QUERY_REPLICATION_PROTOCOL_2026-10-15.md')
VARIANTS=(Variant('control'),Variant('without_cross_query',cross=False))


def setup():
    data,parent=phase.setup()
    if parent!=json.loads((phase.ROOT/'contract.json').read_text()):raise ValueError('phase1 reconstruction changed')
    proof=json.loads((phase.ROOT/'verification.json').read_text())
    if proof['maximum_error']!=0 or proof['report_sha256']!=sha256(phase.ROOT/'report.json'):raise ValueError('phase1 replay stale')
    paths=(Path(__file__),PROTOCOL,phase.ROOT/'contract.json',phase.ROOT/'report.json',phase.ROOT/'verification.json')
    return data,dict(source_sha256={str(p):sha256(p) for p in paths},seed=7,max_epochs=12,min_epochs=6,patience=4,
        variants=[asdict(v) for v in VARIANTS],frozen_h34_seed=42)


def fit(base,drug,variant,tr,va,d,tune,*,model_factory=SequentialV7,folder=None,seed=7,max_epochs=12,min_epochs=6,patience=4):
    folder=ROOT/variant.name if folder is None else folder
    folder.mkdir();m=model_factory(base,drug['drug_parent'],variant,seed=seed).cuda()
    groups,_=m.optimizer_groups();opt=torch.optim.AdamW(groups,weight_decay=.01);rng=np.random.default_rng(seed);torch.manual_seed(seed)
    early=EarlyStopping(patience,min_epochs,1e-4);hist=[];best=None;started=time.perf_counter()
    for epoch in range(max_epochs+1):
        if epoch:
            m.train()
            for j in rng.permutation(len(tr)):
                p,b=tr[j];opt.zero_grad(set_to_none=True);out=m(b.to('cuda'))
                loss=competing_nll(out,*phase.tensor_labels(d,p))+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('query replication nonfinite')
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        aa,kk,curve,nll=phase.first.predict(m,va,d);metrics=phase.first.parent.detail(d,tune,curve,nll)
        row=dict(epoch=epoch,annual=metrics['annual'],tuning_ibs=metrics['readmission']['integrated_brier_7_365'])
        hist.append(row);_,stop=early.observe(epoch,-row['annual']['f1']);key=phase.first.selection_key(row)
        if best is None or key>best:
            best=key;torch.save(dict(state_dict=m.state_dict(),epoch=epoch,threshold=row['annual']['threshold']),folder/'best.pt')
        torch.save(dict(state_dict=m.state_dict(),epoch=epoch,threshold=row['annual']['threshold']),folder/'last.pt')
        write_json(folder/'history.json',hist)
        print(json.dumps(dict(variant=variant.name,seed=seed,epoch=epoch,f1=row['annual']['f1'],ibs=row['tuning_ibs'])),flush=True)
        if stop:break
    outputs={}
    for which in ('best','last'):
        ck=torch.load(folder/f'{which}.pt',map_location='cuda',weights_only=True);m.load_state_dict(ck['state_dict'])
        aa,kk,curve,nll=phase.first.predict(m,va,d);phase.first.parent.save(folder/f'{which}_predictions.npz',d,tune,aa,kk,curve)
        outputs[which]=dict(epoch=ck['epoch'],threshold=ck['threshold'],metrics=phase.first.parent.detail(d,tune,curve,nll,ck['threshold']),
            checkpoint_sha256=sha256(folder/f'{which}.pt'),predictions_sha256=sha256(folder/f'{which}_predictions.npz'))
    info=dict(outputs=outputs,completed_epochs=epoch,early=asdict(early),seconds=time.perf_counter()-started,
        stop_reason='patience' if stop else 'budget_max_epochs')
    write_json(folder/'result.json',info);return info


def verify_result(m,va,d,tune,folder,info,*,patience=4,min_epochs=6):
    for which,out in info['outputs'].items():
        if sha256(folder/f'{which}.pt')!=out['checkpoint_sha256'] or sha256(folder/f'{which}_predictions.npz')!=out['predictions_sha256']:raise ValueError('replication artifact changed')
        ck=torch.load(folder/f'{which}.pt',map_location='cuda',weights_only=True);m.load_state_dict(ck['state_dict'])
        aa,kk,curve,nll=phase.first.predict(m,va,d)
        with np.load(folder/f'{which}_predictions.npz') as z:
            for key,val in (('hadm_ids',d.arrays['hadm_ids'][tune]),('subject_ids',d.arrays['subject_ids'][tune]),
                ('times',d.times[tune]),('events',d.events[tune]),('log_amplitude',aa),('shape',kk),('curve',curve)):
                np.testing.assert_array_equal(z[key],val)
        if phase.first.parent.detail(d,tune,curve,nll,out['threshold'])!=out['metrics']:raise ValueError('replication metrics differ')
    hist=json.loads((folder/'history.json').read_text());early=EarlyStopping(patience,min_epochs,1e-4)
    for row in hist:early.observe(row['epoch'],-row['annual']['f1'])
    if asdict(early)!=info['early'] or max(hist,key=phase.first.selection_key)['epoch']!=info['outputs']['best']['epoch']:raise ValueError('replication selection differs')


def train():
    if ROOT.exists():raise FileExistsError('query replication exists')
    data,contract=setup();d,c,ext,lab,rel,train,tune,base,drug,rad,*_=data
    ROOT.mkdir();write_json(ROOT/'contract.json',contract)
    tr=cache(c,ext,lab,rel,drug,rad,train);va=cache(c,ext,lab,rel,drug,rad,tune)
    results={v.name:fit(base,drug,v,tr,va,d,tune) for v in VARIANTS}
    write_json(ROOT/'report.json',dict(results=results,default_replaced=False))


def verify():
    data,current=setup()
    if current!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('query replication source differs')
    d,c,ext,lab,rel,_,tune,base,drug,rad,*_=data;va=cache(c,ext,lab,rel,drug,rad,tune)
    report=json.loads((ROOT/'report.json').read_text())
    for v in VARIANTS:
        folder=ROOT/v.name;info=report['results'][v.name];m=SequentialV7(base,drug['drug_parent'],v,seed=7).cuda()
        verify_result(m,va,d,tune,folder,info)
        del m;torch.cuda.empty_cache()
    write_json(ROOT/'verification.json',dict(maximum_error=0,models_replayed=2,report_sha256=sha256(ROOT/'report.json')))
    print('query seed7 matched pair replay error 0',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('train','verify'),required=True)
    train() if p.parse_args().mode=='train' else verify()
