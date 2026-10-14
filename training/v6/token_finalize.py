"""只用已完成检查点汇总；容忍GPU浮点归约误差，不续训或重选测试。"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.token_data import load_inputs,prepare
from training.v6.token_experiment import ARMS,legacy_batches,make_model,arm_batches,parameters,render
from training.v6.token_model import fast_cif,daily_probabilities
from training.v6.horizon_objective import restore
from training.v6.evaluation import metrics,public_metrics,paired_interval,calibration_bins
from training.v6.train import write_json


def finalize(root):
    started=time.perf_counter();contract=json.loads((root/'contract.json').read_text());cfg=json.loads((root/'config.json').read_text())
    names=[f'legacy_lr{lr:g}' for lr in (3e-4,1e-4,3e-5)]+list(ARMS)
    summaries={}
    # 先核对全部100轮日志，未完成不读取测试集。
    for seed in cfg['seeds']:
        for name in names:
            path=root/f'seed{seed}'/name;history=json.loads((path/'history.json').read_text())
            if len(history)!=cfg['epochs']+1 or history[-1]['epoch']!=cfg['epochs']:raise ValueError('unfinished training '+str(path))
            chosen=min(history,key=lambda r:r['validation_ibs']);best=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
            if chosen['epoch']!=best['epoch'] or abs(chosen['validation_ibs']-best['validation_ibs'])>1e-12:
                raise ValueError('checkpoint selection differs')
            summaries[seed,name]=dict(best_epoch=best['epoch'],completed_epochs=cfg['epochs'],validation_ibs=best['validation_ibs'],
                lr=1e-4 if not name.startswith('legacy') else float(name.split('lr')[1]),checkpoint_sha256=sha256(path/'best.pt'),
                fit_summary_source='recovered from saved complete epoch histories and checkpoint metadata; no new training')
    for p,digest in contract['code_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('training source changed '+p)
    d,c,h,ext,med,provenance=load_inputs(*(Path(cfg[k]) for k in ('reference','compact_dir','extension','medication')))
    if d.provenance!=contract['data_provenance'] or c.provenance!=contract['compact_provenance'] or provenance!=contract['extensions']:
        raise ValueError('input lineage changed')
    ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('validation','test')}
    token={f:prepare(d,c,ext,med,ix[f]) for f in ix};legacy={f:legacy_batches(d,c,ext,ix[f]) for f in ix}
    models={};curves={};comparisons={};recon={};errors={}
    for seed in cfg['seeds']:
        base=restore(Path(cfg['parent']),seed,'survival_base',h,.1);models[str(seed)]={};curves[seed]={}
        for name in names:
            m=make_model(base,seed,'legacy' if name.startswith('legacy') else name);path=root/f'seed{seed}'/name
            m.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):raise ValueError('frozen baseline changed')
            models[str(seed)][name]=dict(**summaries[seed,name],trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
                                       frozen_baseline_verified=True,folds={});curves[seed][name]={}
            for f in ix:
                batch=legacy[f] if name.startswith('legacy') else arm_batches(token[f],name)
                a,k,nll=parameters(m,batch);curve=fast_cif(a,k,device='cuda');daily,conditional=daily_probabilities(curve)
                fresh=make_model(base,seed,'legacy' if name.startswith('legacy') else name)
                fresh.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
                aa,kk,_=parameters(fresh,batch);replay=fast_cif(aa,kk,device='cuda')
                parameter_error=float(max(np.abs(a-aa).max(),np.abs(k-kk).max()));curve_error=float(np.abs(curve-replay).max())
                if parameter_error>2e-6 or curve_error>2e-7:raise ValueError('replay exceeds float tolerance')
                errors[f'{seed}/{name}/{f}']=dict(parameters=parameter_error,curve=curve_error)
                detail=dict(nll=nll,readmission=public_metrics(metrics(d.times[ix[f]],d.events[ix[f]],curve[:,:,1],ordering=True)),
                            death=public_metrics(metrics(d.times[ix[f]],d.events[ix[f]],curve[:,:,2],cause=2,ordering=True)),
                            calibration={str(day):calibration_bins(d.times[ix[f]],d.events[ix[f]],curve[:,:,1],day=day) for day in (30,90,180,365)})
                models[str(seed)][name]['folds'][f]=detail;curves[seed][name][f]=curve
                np.savez_compressed(path/(f+'_predictions.npz'),hadm_ids=d.arrays['hadm_ids'][ix[f]],subject_ids=d.arrays['subject_ids'][ix[f]],
                    times=d.times[ix[f]],events=d.events[ix[f]],log_amplitude=a,shape=k,curve=curve,daily_event_probability=daily,
                    conditional_daily_event_probability=conditional,conditional_daily_valid=curve[:,:-1,0]>1e-12)
                del fresh
            del m
        ssl=json.loads((root/f'seed{seed}/interaction_ssl/self_supervision.json').read_text());recon[str(seed)]=ssl
        comparisons[str(seed)]={}
        pairs=(('pool_fine','pool_coarse'),('interaction_fine','pool_fine'),('interaction_ssl','interaction_fine'),
               ('flat_mlp','pool_fine'),('legacy_lr0.0001','legacy_lr0.0003'),('legacy_lr3e-05','legacy_lr0.0003'),
               ('pool_coarse','legacy_lr0.0001'),('interaction_fine','legacy_lr0.0001'),('interaction_ssl','legacy_lr0.0001'))
        for f in ix:
            comparisons[str(seed)][f]={f'{a} minus {b}':paired_interval(d.arrays['subject_ids'][ix[f]],d.times[ix[f]],d.events[ix[f]],
                curves[seed][a][f][:,:,1],curves[seed][b][f][:,:,1],seed=seed) for a,b in pairs}
    report=dict(status='completed_long_training_and_token_experiments',contract=contract,models=models,comparisons=comparisons,
        reconstruction=recon,reload_errors=errors,finalization_seconds=time.perf_counter()-started,
        numerical_tolerance=dict(parameters=2e-6,probabilities=2e-7),
        finalizer_sha256=sha256(Path(__file__)),limits=contract['limits']+[
        'Original training completed all arms; overly strict bitwise GPU replay was replaced with explicit floating-point tolerance.',
        'No training rerun or test-selected checkpoints in this finalizer.'])
    write_json(root/'report.json',report);render(root,report)
    print(json.dumps(dict(status=report['status'],seconds=report['finalization_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'));torch.set_num_threads(2);finalize(p.parse_args().root)
