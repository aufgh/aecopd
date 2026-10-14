"""最后风险投影的均值/关联分解；只读，不用验证梯度更新。"""
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.module_surgery_experiment import batches
from training.v6.horizon_objective import restore
from training.v6.warm_residual_audit import teacher
from training.v6.objective_direction_diagnostic import NAMES,cosine,compare
from training.v6.generalization_direction import ROOT as PARENT
from training.v6.train import write_json

ROOT=Path('training_runs/v6_gradient_calibration_2026-10-13')
SOURCE=('training/v6/gradient_calibration_decomposition.py','training/tests/test_v6_gradient_calibration.py',
        'docs/V6_GRADIENT_CALIBRATION_PROTOCOL_2026-10-13.md')


def components(weight,bias,mean):
    if weight.shape!=(4,32) or bias.shape!=(4,) or mean.shape!=(32,):raise ValueError('final projection shapes required')
    global_weight=bias[:,None]*mean[None,:]
    return dict(mean=np.r_[global_weight.ravel(),bias],
                covariance=np.r_[(weight-global_weight).ravel(),np.zeros(4)],
                raw=np.r_[weight.ravel(),bias])


def common_center_gradient(weight,bias,training_mean):
    return weight-bias[:,None]*training_mean[None,:]


@torch.no_grad()
def hidden_mean(model,cache,count):
    total=np.zeros(32,np.float64);seen=0
    for _,h,b,t,e in cache:
        state,_=model.encoder(b.base);hidden=model.delta_head[:-1](state)
        total+=hidden.double().sum(0).cpu().numpy();seen+=len(t)
    if seen!=count:raise ValueError('hidden mean coverage differs')
    return total/count


def final_gradient(vector,layout):
    rows=[r for r in layout if r['last_projection']]
    weight=next(r for r in rows if r['name'].endswith('.weight'))
    bias=next(r for r in rows if r['name'].endswith('.bias'))
    if len(rows)!=2 or weight['end']-weight['start']!=128 or bias['end']-bias['start']!=4:
        raise ValueError('final projection layout differs')
    return vector[weight['start']:weight['end']].reshape(4,32),vector[bias['start']:bias['end']]


def run():
    start=time.perf_counter();d,c,h,ext,_,bins,tune,_,search,_=checked()
    parent=json.loads((PARENT/'diagnostic.json').read_text());proof=json.loads((PARENT/'independent_verification.json').read_text())
    if proof['report_sha256']!=sha256(PARENT/'diagnostic.json') or proof['maximum_gradient_error']>1e-10:
        raise ValueError('parent gradient proof stale')
    for p,digest in json.loads((PARENT/'contract.json').read_text())['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('parent dependency changed')
    train=np.flatnonzero(d.arrays['patient_splits']=='train');cache={f:batches(d,c,ext,ix) for f,ix in [('train',train),('tuning',tune)]}
    contract=dict(source_sha256={p:sha256(Path(p)) for p in SOURCE},parent_report_sha256=sha256(PARENT/'diagnostic.json'),
        parent_contract_sha256=sha256(PARENT/'contract.json'),
        aggregate_gradient_sha256={str(s):sha256(PARENT/f'seed{s}_aggregate_gradients.npz') for s in (42,7)})
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'contract.json').exists():
        if json.loads((ROOT/'contract.json').read_text())!=contract:raise ValueError('decomposition contract changed')
    else:write_json(ROOT/'contract.json',contract)
    results={}
    for seed in (42,7):
        base=restore(Path(search['input_config']['parent']),seed,'survival_base',h,.1);model=teacher(base,seed,bins).eval()
        frozen={k:v.detach().clone() for k,v in model.state_dict().items()}
        means={f:hidden_mean(model,cache[f],parent['folds'][f]['records']) for f in ('train','tuning')}
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()):raise ValueError('model changed')
        with np.load(PARENT/f'seed{seed}_aggregate_gradients.npz') as z:gradients={k:z[k].copy() for k in z.files}
        layout=parent['models'][str(seed)]['parameter_layout'];parts={};common={};biases={};reconstruction=0.
        for f in ('train','tuning'):
            parts[f]={};common[f]={};biases[f]={}
            for name in NAMES:
                weight,bias=final_gradient(gradients[f+'_'+name],layout)
                part=components(weight,bias,means[f]);parts[f][name]=part
                reconstruction=max(reconstruction,float(np.abs(part['mean']+part['covariance']-part['raw']).max()))
                common[f][name]=common_center_gradient(weight,bias,means['train']).ravel();biases[f][name]=bias
        if reconstruction>1e-15:raise ValueError('mean covariance reconstruction differs')
        rows={}
        for name in NAMES:
            rows[name]=dict(raw_same_objective=cosine(parts['train'][name]['raw'],parts['tuning'][name]['raw']),
                bias_same_objective=cosine(biases['train'][name],biases['tuning'][name]),
                covariance_within_fold_same_objective=cosine(parts['train'][name]['covariance'],parts['tuning'][name]['covariance']),
                common_training_mean_centered_same_objective=cosine(common['train'][name],common['tuning'][name]))
        cross={a:{b:cosine(common['train'][a],common['tuning'][b]) for b in NAMES} for a in NAMES}
        results[str(seed)]=dict(hidden_means={f:m.tolist() for f,m in means.items()},same_objective=rows,
            common_center_cross_objective=cross,reconstruction_max_error=reconstruction,original_weights_unchanged=True)
        print(json.dumps(dict(seed=seed,same_objective=rows,center_cross=cross,seconds=round(time.perf_counter()-start,1))),flush=True)
        del model,frozen
    report=dict(models=results,contract_sha256=sha256(ROOT/'contract.json'),model_fits=0,goal_achieved=False,
        scope='Exploratory fixed-state head gradient decomposition; shared center from training only; no parameter update.')
    path=ROOT/'diagnostic.json'
    if path.exists():
        error=compare(json.loads(path.read_text()),report)
        write_json(ROOT/'independent_verification.json',dict(status='independently_replayed',maximum_error=error,
            report_sha256=sha256(path),seconds=time.perf_counter()-start))
    else:write_json(path,report)


if __name__=='__main__':run()
