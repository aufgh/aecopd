"""第七批启动前GPU权重/初始化/旧控制首步匹配；不训练100轮。"""
import json
from pathlib import Path
import torch
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.warm_residual_audit import inputs as old_inputs
from training.v6.horizon_objective import restore
from training.v6.module_surgery_experiment import batches
from training.v6.patient_balance_model import TRIALS,build,uniform_step_proof
from training.v6.warm_residual_audit import teacher
from training.v6.warm_residual import BASE_SEEDS
from training.v6.train import write_json

ROOT=Path('training_runs/v6_patient_balance_2026-10-13')


def run():
    (d,c,h,ext,_,bins,_,_,search,_),_=old_inputs()
    old=Path('training_runs/v6_warm_residual_2026-10-13')
    p=json.loads((old/'tuning_verification.json').read_text())
    if p['status']!='tuning_independently_verified' or p['report_sha256']!=sha256(old/'tuning_report.json'):
        raise ValueError('old ordinary finetuning not independently verified')
    train=batches(d,c,ext,np.flatnonzero(d.arrays['patient_splits']=='train'));proof={}
    for seed in (42,7,101):
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        proof[str(seed)]=dict(uniform_step=uniform_step_proof(base,seed,bins,train),initial={})
        previous=teacher(base,seed,bins).eval()
        with torch.no_grad():
            _,hist,b,_,_=train[0];expected=previous(hist,b)
        for trial in TRIALS:
            m=build(base,seed,trial,bins).eval()
            with torch.no_grad():out=m(hist,b)
            torch.testing.assert_close(out.log_amplitude,expected.log_amplitude,atol=0,rtol=0)
            torch.testing.assert_close(out.shape,expected.shape,atol=0,rtol=0)
            if any(p.requires_grad for p in m.baseline.parameters()):raise ValueError('H34 not frozen')
            if trial.head_only and any(p.requires_grad for p in m.encoder.parameters()):raise ValueError('encoder not frozen')
            m.train()
            if not m.encoder.training:raise ValueError('head/full conditions differ in encoder dropout mode')
            proof[str(seed)]['initial'][trial.name]=dict(maximum_error=0.,trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad))
            del m
        del previous
    ROOT.mkdir(exist_ok=True)
    write_json(ROOT/'preflight.json',dict(status='gpu_initialization_and_ordinary_step_verified',seeds=proof,
        source_sha256={x:sha256(Path(x)) for x in ('training/v6/patient_balance_model.py','training/v6/patient_weight_loss.py','training/v6/patient_balance_preflight.py')},
        old_control_verification_sha256=sha256(old/'tuning_verification.json'),goal_achieved=False))
    print(json.dumps(dict(status='gpu_preflight_passed',seeds=3,initial_models=15,gradient_update_error=0.)),flush=True)


if __name__=='__main__':run()
