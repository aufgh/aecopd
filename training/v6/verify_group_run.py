"""组级实验独立GPU重载验收，恢复正确的屏蔽和解冻配方。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.data import load
from training.v6.model import HistoryBaseline
from training.v6.group_ablation import ARMS, build_model, prepared
from training.v6.train import predict, write_json
from training.v6.risk import predict_cif
from training.v6.evaluation import GRID


def verify(root):
    report=json.loads((root/'report.json').read_text(encoding='utf-8'))
    contract=report['contract']
    if report['status']!='completed_matched_group_ablation_single_seed':
        raise ValueError('incomplete run')
    for name,digest in contract['code_sha256'].items():
        if sha256(Path(__file__).parent/name)!=digest:
            raise ValueError('group experiment code changed')
    reference=Path(contract['reference_run'])
    if sha256(reference/'report.json')!=contract['reference_report_sha256']:
        raise ValueError('reference changed')
    config=json.loads((reference/'config.json').read_text(encoding='utf-8'))
    data=load(Path(config['snapshot']),Path(config['state_dir']),Path(config['labels']),
              field_dir=Path(config['field_dir']),vectors_path=Path(config['vectors_path']))
    if data.provenance!=contract['data_provenance']:
        raise ValueError('data changed')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    base=HistoryBaseline().cuda()
    checkpoint=reference/'history/best.pt'
    if sha256(checkpoint)!=contract['baseline_checkpoint_sha256']:
        raise ValueError('baseline changed')
    base.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['state_dict'])
    checks={}
    for arm in ARMS:
        model=build_model(base,data.text_vectors.shape[1],arm,contract['seed'])
        cp=root/arm/'best.pt'
        if sha256(cp)!=report['models'][arm]['checkpoint_sha256']:
            raise ValueError('checkpoint changed')
        model.load_state_dict(torch.load(cp,map_location='cuda',weights_only=True)['state_dict'])
        checks[arm]={}
        for fold in ('validation','test'):
            ix=np.flatnonzero(data.arrays['patient_splits']==fold)
            batches=prepared(data,ix,contract['batch_size'],arm)
            a,k,_,_=predict(model,batches)
            curve=predict_cif(a,k,GRID,step_days=.25)
            with np.load(root/arm/(fold+'_predictions.npz')) as archive:
                ids=archive['hadm_ids']; saved=archive['curve']
            error=float(np.abs(curve-saved).max())
            if not np.array_equal(ids,data.arrays['hadm_ids'][ix]) or error>5e-6:
                raise ValueError('group prediction reproduction failed')
            if not np.allclose(curve.sum(2),1,atol=1e-10) or np.any(np.diff(curve[:,:,0],axis=1)>1e-10) or np.any(np.diff(curve[:,:,1:],axis=1)<-1e-10):
                raise ValueError('invalid probability curve')
            checks[arm][fold]={'admissions':len(ix),'maximum_curve_error':error,'probability_checks_passed':True}
    write_json(root/'verification.json',{'status':'independent_group_gpu_predictions_verified','checks':checks,
                                         'optimizer_steps':0,'verifier_sha256':sha256(Path(__file__))})
    print(json.dumps({'status':'independent_group_gpu_predictions_verified'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    if (args.run/'verification.json').exists(): raise FileExistsError(args.run/'verification.json')
    verify(args.run)
