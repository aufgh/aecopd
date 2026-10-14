"""新进程独立重载全部检查点，验证逐日接口；不训练、不修改旧结果。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.token_data import load_inputs,prepare
from training.v6.token_experiment import legacy_batches,make_model,arm_batches,parameters
from training.v6.token_model import fast_cif,daily_probabilities
from training.v6.horizon_objective import restore
from training.v6.risk import predict_cif
from training.v6.train import write_json


def verify(root):
    report=json.loads((root/'report.json').read_text());contract=report['contract'];cfg=json.loads((root/'config.json').read_text())
    for p,digest in contract['code_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('source changed '+p)
    d,c,h,ext,med,provenance=load_inputs(*(Path(cfg[k]) for k in ('reference','compact_dir','extension','medication')))
    if d.provenance!=contract['data_provenance'] or c.provenance!=contract['compact_provenance'] or provenance!=contract['extensions']:
        raise ValueError('independent data lineage differs')
    errors={};max_integration_error=0.;initial={}
    for seed,models in report['models'].items():
        seed=int(seed);basepath=Path(cfg['parent'])/f'seed{seed}/survival_base/best.pt'
        if sha256(basepath)!=contract['frozen_baselines'][str(seed)]:raise ValueError('base checkpoint changed')
        base=restore(Path(cfg['parent']),seed,'survival_base',h,.1)
        for fold in ('validation','test'):
            ix=np.flatnonzero(d.arrays['patient_splits']==fold);token=prepare(d,c,ext,med,ix);legacy=legacy_batches(d,c,ext,ix)
            for name,item in models.items():
                path=root/f'seed{seed}'/name
                if sha256(path/'best.pt')!=item['checkpoint_sha256']:raise ValueError('checkpoint changed')
                m=make_model(base,seed,'legacy' if name.startswith('legacy') else name)
                m.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
                if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):raise ValueError('frozen baseline differs')
                cache=legacy if name.startswith('legacy') else arm_batches(token,name)
                a,k,nll=parameters(m,cache);curve=fast_cif(a,k,device='cuda');daily,conditional=daily_probabilities(curve)
                with np.load(path/(fold+'_predictions.npz')) as z:
                    if not np.array_equal(z['hadm_ids'],d.arrays['hadm_ids'][ix]) or not np.array_equal(z['times'],d.times[ix]) or not np.array_equal(z['events'],d.events[ix]):
                        raise ValueError('prediction indices/labels changed')
                    err=float(np.max(np.abs(z['curve']-curve)));de=float(np.max(np.abs(z['daily_event_probability']-daily)))
                    ce=float(np.max(np.abs(z['conditional_daily_event_probability']-conditional)))
                if max(err,de,ce)>2e-7:raise ValueError('replayed probabilities exceed GPU float tolerance')
                if (daily< -1e-9).any() or (conditional.sum(2)>1+1e-8).any():raise ValueError('invalid daily event probability')
                slow=predict_cif(a[:5],k[:5],np.arange(366));max_integration_error=max(max_integration_error,float(np.max(np.abs(slow-curve[:5]))))
                errors[f'{seed}/{name}/{fold}']=dict(curve=err,daily=de,conditional=ce,nll=nll)
                history=json.loads((path/'history.json').read_text())
                if len(history)!=contract['epochs']+1 or history[-1]['epoch']!=contract['epochs']:raise ValueError('incomplete long training')
                selected=min(history,key=lambda x:x['validation_ibs'])
                if selected['epoch']!=item['best_epoch']:raise ValueError('checkpoint not validation IBS selected')
                initial[f'{seed}/{name}']=history[0]['validation_ibs'];del m
            del token,legacy
    for seed in contract['seeds']:
        values=[v for name,v in initial.items() if name.startswith(f'{seed}/')]
        if max(values)-min(values)>1e-10:raise ValueError('zero-start risk differs')
    if max_integration_error>1e-10:raise ValueError('fast/reference integration differs')
    proof=dict(status='independently_verified_neural',report_sha256=sha256(root/'report.json'),code_sha256=contract['code_sha256'],
        max_reference_integration_error=max_integration_error,reload_errors=errors,epoch0_risks_equal=True,
        complete_long_training=True,selected_by_validation_ibs=True,daily_interface='CIF; unconditional day mass; conditional day mass',
        limits=['Independent verification is engineering correctness, not clinical utility or causal validation.'])
    write_json(root/'independent_verification.json',proof);print(json.dumps(dict(status=proof['status'],checkpoints=len(errors),integration_error=max_integration_error)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'));torch.set_num_threads(2);verify(p.parse_args().root)
