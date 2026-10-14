"""训练结束后的独立GPU重载验收；不修改预测/检查点。"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.detail_extension_experiment import load_inputs,model_for,prepared,ARMS
from training.v6.horizon_objective import predict90,binary_metrics
from training.v6.risk import predict_cif


def verify(root):
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    report=json.loads((root/'report.json').read_text()); contract=report['contract']
    for p,digest in contract['code_sha256'].items():
        if sha256(Path(p))!=digest: raise ValueError('source changed: '+p)
    if sha256(Path('docs/V6_DETAIL_EXTENSION_PROTOCOL_2026-10-11.md'))!=contract['protocol_sha256']:
        raise ValueError('protocol changed')
    config=json.loads((root/'config.json').read_text()); args=SimpleNamespace(**{k:Path(v) for k,v in config.items()})
    d,c,hist,parent,a,_=load_inputs(args)
    if sha256(args.extension/'manifest.json')!=contract['extension_manifest_sha256']: raise ValueError('extension changed')
    evidence={}
    for seed in (42,7):
        for arm in ARMS:
            path=root/f'seed{seed}'/arm/'best.pt'
            if sha256(path)!=report['models'][str(seed)][arm]['checkpoint_sha256']: raise ValueError('checkpoint changed')
            m=model_for(args,seed,hist)
            m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
            for fold in ('validation','test'):
                idx=np.flatnonzero(d.arrays['patient_splits']==fold)
                with np.load(root/f'seed{seed}'/arm/(fold+'_predictions.npz')) as z: saved={k:z[k] for k in z.files}
                for k,actual in (('hadm_ids',d.arrays['hadm_ids'][idx]),('subject_ids',d.arrays['subject_ids'][idx]),
                                 ('times',d.times[idx]),('events',d.events[idx])):
                    if not np.array_equal(saved[k],actual): raise ValueError('saved population/labels differ')
                replay=predict90(m,prepared(d,c,a,idx,arm)); err=float(np.abs(saved['risk90']-replay['risk90']).max())
                if err>1e-7: raise ValueError('risk reload differs')
                mm=binary_metrics(saved['y90'],replay['risk90'])
                expected=report['models'][str(seed)][arm]['folds'][fold]
                errors={k:abs(mm[k]-expected[k]) for k in ('auc90','brier90','average_precision','logloss90')}
                if max(errors.values())>1e-7: raise ValueError('metrics differ')
                fine=predict_cif(replay['parameters'][:64],replay['shape'][:64],np.arange(366.),step_days=.125)
                curve_error=float(np.abs(fine-saved['curve'][:64]).max())
                if curve_error>2e-4 or not np.allclose(saved['curve'].sum(2),1,atol=1e-9): raise ValueError('integration failed')
                evidence[f'{seed}/{arm}/{fold}']=dict(risk_max_error=err,metric_errors=errors,fine_curve_error=curve_error)
            del m
    result=dict(status='independently_verified_20_folds',report_sha256=sha256(root/'report.json'),
        verifier_sha256=sha256(Path(__file__)),device=torch.cuda.get_device_name(0),folds=evidence)
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result['status'],maximum_risk_error=max(x['risk_max_error'] for x in evidence.values()))))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    verify(p.parse_args().output)
