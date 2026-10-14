"""补充真正的训练折每概念均值重构对照；不伪称归一化零值就是均值。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.v6.token_data import load_inputs,prepare
from training.v6.token_model import masked_reconstruction_batch
from training.v6.train import write_json


def run(root):
    cfg=json.loads((root/'config.json').read_text());d,c,h,ext,med,_=load_inputs(*(Path(cfg[k]) for k in ('reference','compact_dir','extension','medication')))
    cache=prepare(d,c,ext,med,np.flatnonzero(d.arrays['patient_splits']=='train'),device='cuda')
    total=torch.zeros(29,20,device='cuda');count=torch.zeros(29,20,device='cuda')
    for _,_,b,_,_ in cache:total+=(b.labs*b.numeric_valid).sum(0);count+=b.numeric_valid.sum(0)
    means=total/count.clamp_min(1);results={}
    for seed in cfg['seeds']:
        mean_loss=0.;zero_loss=0.;targets=0
        for j,(_,_,b,_,_) in enumerate(cache):
            _,mask=masked_reconstruction_batch(b,seed+cfg['ssl_epochs']*10000+j)
            mean_loss+=float(((b.labs-means).square()[mask]).sum());zero_loss+=float(b.labs.square()[mask].sum());targets+=int(mask.sum())
        results[str(seed)]=dict(mean_predictor_mse=mean_loss/targets,zero_normalized_predictor_mse=zero_loss/targets,targets=targets,
            diagnostic='training fold; same CUDA masking seeds and batches as pretraining epoch20')
    write_json(root/'self_supervised_controls.json',dict(status='completed_training_mean_reconstruction_control',mean_fit='observed numeric channels in training fold only',
        results=results,limits=['Do not infer readmission gain from reconstruction gain.','No validation/test examples used.']))
    print(json.dumps(results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'));torch.set_num_threads(2);run(p.parse_args().root)
