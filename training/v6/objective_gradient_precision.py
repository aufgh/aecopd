"""真实首个训练batch的最小精度差分；只切换反向TF32，不改变前向。"""
import json
from pathlib import Path
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked
from training.v6.module_surgery_experiment import batches
from training.v6.warm_residual_audit import teacher
from training.v6.horizon_objective import restore
from training.v6.objective_direction_diagnostic import output
from training.v6.patient_weight_loss import per_record_nll
from training.v6.train import write_json
import numpy as np


def run():
    d,c,h,ext,_,bins,_,_,search,_=checked()
    ix=np.flatnonzero(d.arrays['patient_splits']=='train')[:256]
    _,history,batch,t,e=batches(d,c,ext,ix)[0]
    base=restore(Path(search['input_config']['parent']),42,'survival_base',h,.1)
    model=teacher(base,42,bins).eval()
    with torch.no_grad():state,_=model.encoder(batch.base);hidden=model.delta_head[:-1](state);prior=model.baseline(history)
    layer=model.delta_head[-1];raw=layer(hidden);out=output(raw,prior.log_amplitude,model.baseline.raw_shape.detach())
    loss=per_record_nll(out,t,e).sum()/13592
    raw_grad=torch.autograd.grad(loss,raw,retain_graph=True)[0]
    reference=raw_grad.double().T@hidden.double();result={}
    original=torch.backends.cuda.matmul.allow_tf32
    try:
        for enabled in (True,False):
            torch.backends.cuda.matmul.allow_tf32=enabled
            actual=torch.autograd.grad(loss,layer.weight,retain_graph=True)[0]
            difference=(actual.double()-reference).abs()
            result[str(enabled)]=dict(max_error=float(difference.max()),relative_norm_error=float(difference.norm()/reference.norm()))
    finally:torch.backends.cuda.matmul.allow_tf32=original
    root=Path('training_runs/v6_objective_network_2026-10-13');root.mkdir(exist_ok=True)
    report=dict(status='real_batch_backward_precision_probe',results=result,
        source_sha256=sha256(Path(__file__)),forward_unchanged=True)
    write_json(root/'precision_probe.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':run()
