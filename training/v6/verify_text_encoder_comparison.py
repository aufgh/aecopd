"""独立重载验收；指标浮点递归容差，训练源码与原合同不改。"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.text_encoder_comparison import (ARMS, PROTOCOL, checked_inputs, new_model,
    prepared, arm_source, read_prediction)
from training.v6.horizon_objective import labels90, predict90, binary_metrics
from training.v6.risk import predict_cif
from training.v6.evaluation import GRID, metrics, public_metrics
from training.v6.train import write_json


def metric_difference(actual,expected,path='root'):
    """仅数值浮点允许1e-7；结构、整数计数、状态和文本须精确。"""
    if isinstance(actual,dict):
        if not isinstance(expected,dict) or actual.keys()!=expected.keys(): raise ValueError('metric structure: '+path)
        return max([metric_difference(v,expected[k],path+'/'+k) for k,v in actual.items()]+[0.])
    if isinstance(actual,list):
        if not isinstance(expected,list) or len(actual)!=len(expected): raise ValueError('metric list: '+path)
        return max([metric_difference(a,b,path) for a,b in zip(actual,expected,strict=True)]+[0.])
    if isinstance(actual,(float,np.floating)):
        if not isinstance(expected,(float,int)) or not np.isfinite(actual) or not np.isfinite(expected): raise ValueError('metric nonfinite: '+path)
        error=abs(float(actual)-expected)
        if error>1e-7: raise ValueError(f'metric differs {path}: {actual} != {expected}')
        return error
    if actual!=expected: raise ValueError('metric differs: '+path)
    return 0.


def verify(args):
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    report=json.loads((args.output/'report.json').read_text()); c=report['contract']
    for path,digest in c['code_sha256'].items():
        if sha256(Path(path))!=digest: raise ValueError('training code changed: '+path)
    if sha256(PROTOCOL)!=c['protocol_sha256']: raise ValueError('protocol changed')
    data,compact,hist,_,audit=checked_inputs(args)
    if audit!=c['text_audit'] or sha256(args.parent/'report.json')!=c['parent_report_sha256']: raise ValueError('asset changed')
    d=data['bert']; y=labels90(d.times,d.events); errors={}; metric_errors={}
    for seed in (42,7):
        for arm in ARMS:
            path=args.output/f'seed{seed}'/arm/'best.pt'
            if sha256(path)!=report['models'][str(seed)][arm]['checkpoint_sha256']: raise ValueError('checkpoint changed')
            m=new_model(args,seed,arm,hist,y[d.arrays['patient_splits']=='train'].mean())
            frozen=copy.deepcopy(m.baseline.state_dict())
            m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
            if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in frozen.items()): raise ValueError('baseline changed')
            source=arm_source(arm)
            for fold in ('validation','test'):
                idx=np.flatnonzero(d.arrays['patient_splits']==fold)
                p=predict90(m,prepared(data[source],compact[source],idx,512,arm))
                p['curve']=predict_cif(p['parameters'],p['shape'],GRID)
                saved=read_prediction(args.output,seed,arm,fold)
                for key,value in (('hadm_ids',d.arrays['hadm_ids'][idx]),('subject_ids',d.arrays['subject_ids'][idx]),
                                  ('times',d.times[idx]),('events',d.events[idx]),('y90',y[idx])):
                    if not np.array_equal(saved[key],value): raise ValueError('identity/label mismatch')
                for key,value in p.items():
                    if key!='loss' and not np.allclose(saved[key],value,atol=2e-6,rtol=1e-5): raise ValueError('reload mismatch: '+key)
                detail=binary_metrics(y[idx],p['risk90'])
                detail['survival_readmission']=public_metrics(metrics(d.times[idx],d.events[idx],p['curve'][:,:,1],ordering=True))
                detail['survival_death']=public_metrics(metrics(d.times[idx],d.events[idx],p['curve'][:,:,2],cause=2,ordering=True))
                key=f'{seed}/{arm}/{fold}'
                metric_errors[key]=metric_difference(detail,report['models'][str(seed)][arm]['folds'][fold])
                errors[key]=float(np.abs(saved['risk90']-p['risk90']).max())
            del m
    write_json(args.output/'independent_verification.json',dict(status='all8_models_all16_folds_gpu_reloaded',
        report_sha256=sha256(args.output/'report.json'),baseline_unchanged=True,errors=errors,metric_errors=metric_errors,
        verifier_sha256=sha256(Path(__file__)),training_source_and_contract_unchanged=True,
        original_verifier_limitation='exact dict equality rejected tiny floating-point metric differences; independent verifier uses tested1e-7 tolerance'))
    print(json.dumps(dict(status='all8_models_all16_folds_gpu_reloaded',max_risk_error=max(errors.values()),
                         max_metric_error=max(metric_errors.values()))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--compact-dir',type=Path,default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--parent',type=Path,default=Path('training_runs/v6_90day_objective_2026-10-11'))
    p.add_argument('--output',type=Path,required=True)
    verify(p.parse_args())


if __name__=='__main__': main()
