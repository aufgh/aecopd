"""新进程独立核验哈希/选轮/冻结及逐住院每日预测；不重训。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.audited_experiment import ARMS,INPUT,OUTPUT,load,prepare,restrict,new_model
from training.v6.horizon_objective import restore
from training.v6.token_experiment import parameters
from training.v6.token_model import fast_cif,daily_probabilities
from training.v6.train import write_json


def verify(root):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.use_deterministic_algorithms(True)
    report=json.loads((root/'report.json').read_text());contract=report['contract']
    if report['status']!='completed_neural_and_simple_controls':raise ValueError('incomplete run')
    for name,digest in contract['source_sha256'].items():
        if sha256(Path(name))!=digest:raise ValueError('training source changed')
    if sha256(INPUT/'audit.json')!=contract['input_audit_sha256']:raise ValueError('audit changed')
    cfg,d,c,h,a,audit=load()
    if d.provenance!=contract['data']:raise ValueError('data lineage changed')
    snapshot=json.loads((cfg.snapshot/'metadata/manifest.json').read_text())
    raw_verified={}
    for name,item in snapshot['sources'].items():
        digest=sha256(cfg.snapshot/(name+'.csv.gz'))
        if digest!=item['sha256']:raise ValueError('raw snapshot changed')
        raw_verified[name]=digest
    idx={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('validation','test')}
    cache={f:prepare(d,c,a,idx[f]) for f in idx};compact=report['selected_arm']!='full_pool';errors={};metric_errors={}
    for seed in (42,7):
        if sha256(cfg.parent/f'seed{seed}/survival_base/best.pt')!=contract['frozen_baseline_sha256'][str(seed)]:raise ValueError('base file changed')
        base=restore(cfg.parent,seed,'survival_base',h,.1)
        for arm in (*ARMS,'flat'):
            path=root/f'seed{seed}'/arm;info=report['models'][str(seed)][arm]
            if sha256(path/'best.pt')!=info['checkpoint_sha256']:raise ValueError('checkpoint changed')
            history=json.loads((path/'history.json').read_text());state=torch.load(path/'best.pt',map_location='cuda',weights_only=True)
            if len(history)!=contract['epochs']+1 or min(history,key=lambda r:r['validation_ibs'])['epoch']!=state['epoch']:raise ValueError('selection wrong')
            model=new_model(base,seed,arm,a);model.load_state_dict(state['state_dict'])
            if not all(torch.equal(value,model.baseline.state_dict()[name]) for name,value in base.state_dict().items()):raise ValueError('frozen changed')
            for fold in idx:
                aa,kk,nll=parameters(model,restrict(cache[fold],a,compact if arm=='flat' else arm!='full_pool'))
                curve=fast_cif(aa,kk,device='cuda');daily,conditional=daily_probabilities(curve)
                with np.load(path/(fold+'_predictions.npz')) as z:previous={k:z[k] for k in z.files}
                if not np.array_equal(previous['hadm_ids'],d.arrays['hadm_ids'][idx[fold]]):raise ValueError('prediction members wrong')
                err=float(max(np.abs(curve-previous['curve']).max(),np.abs(daily-previous['daily_event_probability']).max(),
                  np.abs(conditional-previous['conditional_daily_event_probability']).max()))
                if err>1e-9:raise ValueError('new process predictions differ')
                errors[f'{seed}/{arm}/{fold}']=err
                from training.v6.evaluation import metrics
                detail=metrics(d.times[idx[fold]],d.events[idx[fold]],curve[:,:,1])
                delta=abs(detail['integrated_brier_7_365']-info['folds'][fold]['readmission']['integrated_brier_7_365'])
                if delta>1e-12:raise ValueError('metrics differ')
                metric_errors[f'{seed}/{arm}/{fold}']=delta
            del model
    simple=json.loads((root/'simple_controls/report.json').read_text())
    if simple['status']!='completed_simple_controls' or max(simple['reload_errors'].values())>1e-9:raise ValueError('simple verification incomplete')
    if sha256(Path('training/v6/audited_simple_controls.py'))!=simple['source_sha256']:raise ValueError('simple source changed')
    if sha256(Path('training/v6/finalize_audited_controls.py'))!=simple['completion_code_sha256']:raise ValueError('recovery source changed')
    write_json(root/'independent_verification.json',dict(status='new_process_exact_gpu_prediction_replay',report_sha256=sha256(root/'report.json'),
      simple_report_sha256=sha256(root/'simple_controls/report.json'),completed_neural_models=10,verified_folds=20,reload_errors=errors,
      metric_errors=metric_errors,maximum_probability_error=max(errors.values()),frozen_baseline_unchanged=True,raw_snapshot_sha256=raw_verified,
      limits=['Neural new process replay verified; simple model save/load checked by training runner.',
              'Engineering success is not clinical validation or a blind test.']))
    print(json.dumps(dict(status='independently_verified',models=10,folds=20,max_error=max(errors.values()))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=OUTPUT);verify(p.parse_args().root)
