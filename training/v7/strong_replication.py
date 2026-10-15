"""强基座端到端方案只追加一个表示种子，复用有限训练函数。"""
import argparse
import json
from pathlib import Path
import torch
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json
from training.v7 import strong_anchor_experiment as first
from training.v7.query_replication import fit,verify_result
from training.v7.sequential_variants import Variant
from training.v7.data import cache

ROOT=Path('training_runs/v7_sequential_research_2026-10-15/strong_replication')
PROTOCOL=Path('docs/V7_STRONG_REPLICATION_PROTOCOL_2026-10-15.md')


def setup():
    data,a,parent=first.setup();proof=json.loads((first.ROOT/'verification.json').read_text())
    if proof['maximum_error']!=0 or proof['report_sha256']!=sha256(first.ROOT/'report.json'):raise ValueError('strong proof stale')
    paths=(Path(__file__),PROTOCOL,Path('training/v7/query_replication.py'),first.ROOT/'contract.json',first.ROOT/'report.json',first.ROOT/'verification.json')
    return data,a,dict(source_sha256={str(p):sha256(p) for p in paths},seed=7,parent=parent,frozen_anchor_seed=42)


def train():
    if ROOT.exists():raise FileExistsError('strong replication exists')
    data,a,contract=setup();d,c,ext,lab,rel,tr,va,base,drug,rad,*_=data;ROOT.mkdir();write_json(ROOT/'contract.json',contract)
    train_batches=first.wrapped(cache(c,ext,lab,rel,drug,rad,tr));val_batches=first.wrapped(cache(c,ext,lab,rel,drug,rad,va))
    factory=lambda base,parent,variant,seed:first.StrongV7(base,parent,a,seed=seed)
    result=fit(base,drug,Variant('strong_seed7'),train_batches,val_batches,d,va,model_factory=factory,
        folder=ROOT/'model',seed=7,max_epochs=16,min_epochs=8,patience=6)
    hist=json.loads((ROOT/'model/history.json').read_text());old=json.loads((first.ROOT/'history.json').read_text())
    if hist[0]['annual']!=old[0]['annual'] or hist[0]['tuning_ibs']!=old[0]['tuning_ibs']:raise ValueError('strong seed7 initialization differs')
    write_json(ROOT/'report.json',dict(result=result,initial_anchor_error=0,default_replaced=False))


def verify():
    data,a,current=setup()
    if current!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('strong replication source differs')
    d,c,ext,lab,rel,_,va,base,drug,rad,*_=data;val=first.wrapped(cache(c,ext,lab,rel,drug,rad,va))
    m=first.StrongV7(base,drug['drug_parent'],a,seed=7).cuda();r=json.loads((ROOT/'report.json').read_text())
    verify_result(m,val,d,va,ROOT/'model',r['result'],patience=6,min_epochs=8)
    write_json(ROOT/'verification.json',dict(maximum_error=0,best_last_replayed=True,report_sha256=sha256(ROOT/'report.json')))
    print('strong seed7 best/last replay error 0',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('train','verify'),required=True)
    train() if p.parse_args().mode=='train' else verify()
