"""可重复完整入口：旧已哈希训练代码不改，显式注入数值容差核验适配器。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6 import token_experiment as engine
from training.v6.token_model import fast_cif,daily_probabilities
from training.v6.evaluation import metrics,public_metrics,calibration_bins


def tolerant_evaluation(model,cache,data,indices,path):
    a,k,nll=engine.parameters(model,cache);curve=fast_cif(a,k,device='cuda');daily,conditional=daily_probabilities(curve)
    model.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
    aa,kk,_=engine.parameters(model,cache)
    if max(np.abs(a-aa).max(),np.abs(k-kk).max())>2e-6 or np.abs(curve-fast_cif(aa,kk,device='cuda')).max()>2e-7:
        raise ValueError('checkpoint replay exceeds explicit GPU float tolerance')
    detail=dict(nll=nll,readmission=public_metrics(metrics(data.times[indices],data.events[indices],curve[:,:,1],ordering=True)),
        death=public_metrics(metrics(data.times[indices],data.events[indices],curve[:,:,2],cause=2,ordering=True)),
        calibration={str(day):calibration_bins(data.times[indices],data.events[indices],curve[:,:,1],day=day) for day in (30,90,180,365)})
    return curve,detail,dict(log_amplitude=a,shape=k,curve=curve,daily_event_probability=daily,
        conditional_daily_event_probability=conditional,conditional_daily_valid=curve[:,:-1,0]>1e-12)


def main():
    p=argparse.ArgumentParser()
    defaults=dict(reference='training_runs/v6_history_anchored_bert_seed42_2026-10-09',parent='training_runs/v6_90day_objective_2026-10-11',
        compact_dir='patient_data/processed/v6_compact_multisource_2026-10-09',extension='patient_data/processed/v6_detail_extension_2026-10-11_r2',
        medication='patient_data/processed/v6_medication_identity_2026-10-12')
    for name,value in defaults.items():p.add_argument('--'+name.replace('_','-'),type=Path,default=Path(value))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=100);p.add_argument('--ssl-epochs',type=int,default=20)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,7]);a=p.parse_args()
    if a.seeds!=[42,7] or a.epochs!=100:raise ValueError('complete suite currently contracted to two seeds and 100 epochs')
    original=engine.evaluate_model
    # 这是显式兼容接口，不热改训练、模型或历史源文件；仅替换逐位相等验收。
    engine.evaluate_model=tolerant_evaluation
    try:engine.run(a)
    finally:engine.evaluate_model=original
    manifest=dict(runner_sha256=sha256(Path(__file__)),engine_sha256=sha256(Path(engine.__file__)),
        evaluation_adapter='parameter atol2e-6; probability atol2e-7; no training changes')
    engine.write_json(a.output/'execution_adapter.json',manifest)
    from training.v6.token_finalize import finalize
    from training.v6.token_verify import verify
    from training.v6.token_cox_control import run as cox
    from training.v6.token_cox_verify import verify as verify_cox
    from training.v6.reconstruction_controls import run as reconstruction
    from training.v6.token_choose_reference import choose
    from training.v6.token_result_figures import render
    finalize(a.output);verify(a.output);cox(a.output);verify_cox(a.output);reconstruction(a.output);choose(a.output);render(a.output)
    print(json.dumps(dict(status='complete_verified_suite',output=str(a.output))))


if __name__=='__main__':main()
