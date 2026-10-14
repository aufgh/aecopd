"""只汇总已完成100轮；确定性GPU重放，不改训练源/权重或用测试选模。"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.expanded_original_experiment import checked,prepare,new_model,evaluate,render,ARMS
from training.v6.horizon_objective import restore
from training.v6.token_experiment import parameters
from training.v6.token_model import fast_cif
from training.v6.evaluation import metrics,public_metrics,paired_interval
from training.v6.train import write_json


def finalize(root):
    if (root/'report.json').exists():raise FileExistsError(root/'report.json')
    begin=time.perf_counter();cfg=json.loads((root/'config.json').read_text());contract=json.loads((root/'contract.json').read_text())
    for file,digest in contract['code_sha256'].items():
        if sha256(Path(file))!=digest:raise ValueError('hashed training source changed')
    summaries={}
    for seed in (42,7):
        for arm in ARMS:
            path=root/f'seed{seed}/{arm}';history=json.loads((path/'history.json').read_text());cp=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
            best=min(history,key=lambda r:r['validation_ibs'])
            if len(history)!=101 or history[-1]['epoch']!=100 or best['epoch']!=cp['epoch'] or abs(best['validation_ibs']-cp['validation_ibs'])>1e-12:raise ValueError('unfinished or wrong checkpoint selection')
            summaries[seed,arm]=dict(best_epoch=cp['epoch'],completed_epochs=100,lr=1e-4,validation_ibs=best['validation_ibs'],
                 checkpoint_sha256=sha256(path/'best.pt'),fit_summary_source='completed100 epoch histories; no retraining')
    args=argparse.Namespace(**{k:Path(v) for k,v in cfg.items()});d,c,h,ext,lab,med,prov,active,norm=checked(args)
    if prov!=contract['new_sources'] or d.provenance!=contract['data_provenance']:raise ValueError('source changed')
    ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('validation','test')};cache={f:prepare(d,c,ext,lab,med,ix[f]) for f in ix}
    models={};curves={};replay={};prior_errors={};init={}
    for seed in (42,7):
        base=restore(args.parent,seed,'survival_base',h,.1);models[str(seed)]={};curves[seed]={}
        for arm in ARMS:
            m=new_model(base,seed,arm);path=root/f'seed{seed}/{arm}'
            if arm=='interaction_only':
                a,k,_=parameters(m,cache['validation'][:2]);expect=base(cache['validation'][0][1])
                init[str(seed)]=float(np.abs(a[:len(expect.log_amplitude)]-expect.log_amplitude.detach().cpu().numpy()).max())
                if init[str(seed)]>0:raise ValueError('zero head not base')
            m.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            if not all(torch.equal(v,m.baseline.state_dict()[key]) for key,v in base.state_dict().items()):raise ValueError('frozen baseline changed')
            models[str(seed)][arm]=dict(**summaries[seed,arm],trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
                 frozen_baseline_verified=True,interaction_gate=float(torch.tanh(m.encoder.interaction_gate).detach()),folds={});curves[seed][arm]={}
            for fold in ix:
                detail,curve,a,k=evaluate(m,cache[fold],d,ix[fold],path/(fold+'_predictions.npz'))
                fresh=new_model(base,seed,arm);fresh.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
                if not all(torch.equal(v,fresh.state_dict()[key]) for key,v in m.state_dict().items()):raise ValueError('loaded states unequal')
                aa,kk,_=parameters(fresh,cache[fold]);err=float(np.abs(fast_cif(aa,kk,device='cuda')-curve).max());del fresh
                if err>2e-7:raise ValueError('deterministic fresh replay differs')
                replay[f'{seed}/{arm}/{fold}']=err;models[str(seed)][arm]['folds'][fold]=detail;curves[seed][arm][fold]=curve
                if arm=='control':
                    with np.load(args.comparator/f'seed{seed}/legacy_lr0.0001'/(fold+'_predictions.npz')) as z:old=z['curve']
                    previous=public_metrics(metrics(d.times[ix[fold]],d.events[ix[fold]],old[:,:,1]))
                    current=detail['readmission'];prior_errors[f'{seed}/{fold}']=dict(max_probability_error=float(np.abs(old-curve).max()),
                        mean_probability_error=float(np.abs(old-curve).mean()),ibs_difference=current['integrated_brier_7_365']-previous['integrated_brier_7_365'],
                        auc90_difference=current['points']['90']['auc']-previous['points']['90']['auc'])
                    # 本次对照是重新训练、并非同一个旧checkpoint。明确记录轨迹差异，不冒充逐位重放。
                    if abs(prior_errors[f'{seed}/{fold}']['ibs_difference'])>1e-6 or abs(prior_errors[f'{seed}/{fold}']['auc90_difference'])>1e-4:
                        raise ValueError('new control metrics materially differ; stop comparison')
            del m
    pairs=[(a,'control') for a in ARMS[1:]]+[('both_interaction','both'),('both_interaction','interaction_only')]
    comparisons={str(seed):{f:{f'{a} minus {b}':paired_interval(d.arrays['subject_ids'][ix[f]],d.times[ix[f]],d.events[ix[f]],
              curves[seed][a][f][:,:,1],curves[seed][b][f][:,:,1],seed=seed) for a,b in pairs} for f in ix} for seed in (42,7)}
    scores={arm:sum(models[str(s)][arm]['validation_ibs'] for s in (42,7))/2 for arm in ARMS}
    report=dict(status='completed_original_v6_expansion',contract=contract,models=models,comparisons=comparisons,
        initial_errors=init,reload_errors=replay,control_previous_errors=prior_errors,validation_selected_arm=min(scores,key=scores.get),validation_scores=scores,
        finalizer_sha256=sha256(Path(__file__)),finalization_seconds=time.perf_counter()-begin,
        numerical_notes=['All12 trainings completed100 epochs before original nondeterministic GPU replay threshold failure.',
         'Same state GPU text index_add atomic reduction reproduced a7.4e-7 curve drift; deterministic GPU inference gives exact replay.',
         'Prior comparison is independently retrained control, not same-checkpoint replay; its measured trajectory differences are disclosed.',
         'No retraining/reselection, no modified hashed sources.'],peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
    write_json(root/'report.json',report);write_json(root/'independent_verification.json',dict(status='deterministic_fresh_gpu_reloads_verified',
        report_sha256=sha256(root/'report.json'),reload_errors=replay,normalization_errors=norm,control_previous_errors=prior_errors,
        completed_models=12,frozen_checked=True,deterministic_gpu=True))
    render(root,report);print(json.dumps(dict(status=report['status'],selected=report['validation_selected_arm'],replay_max=max(replay.values()),prior_control=prior_errors)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_original_interaction_2026-10-12'))
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.use_deterministic_algorithms(True);finalize(a.root)
