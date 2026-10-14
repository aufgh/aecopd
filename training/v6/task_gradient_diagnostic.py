"""训练折上双原因梯度方向诊断；不训练/选模，不读取确认或测试分数。"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.module_surgery import SurgeryModel
from training.v6.module_surgery_experiment import batches
from training.v6.risk import competing_nll
from training.v6.train import write_json

ROOT=Path('training_runs/v6_task_gradient_diagnostic_2026-10-13')


def cause_losses(out,times,events):
    scaled=times.clamp_min(1e-3)[:,None]/365.
    cumulative=torch.exp(out.log_amplitude)*scaled.pow(out.shape)
    log_hazard=out.log_amplitude+torch.log(out.shape)-math.log(365.)+(out.shape-1)*torch.log(scaled)
    return torch.stack([(cumulative[:,c]-torch.where(events==c+1,log_hazard[:,c],0.)).mean() for c in (0,1)])


def gradient_summary(left,right):
    aa=sum(float(x.detach().square().sum()) for x in left if x is not None)
    bb=sum(float(x.detach().square().sum()) for x in right if x is not None)
    dot=sum(float((a.detach()*b.detach()).sum()) for a,b in zip(left,right) if a is not None and b is not None)
    return dict(readmission_norm=math.sqrt(aa),death_norm=math.sqrt(bb),
        cosine=dot/math.sqrt(aa*bb) if aa*bb>1e-24 else None,
        death_to_readmission_norm_ratio=math.sqrt(bb/aa) if aa>1e-24 else None)


def run():
    if ROOT.exists():raise FileExistsError(ROOT)
    d,c,h,ext,base,bins,_,_,contract,_=checked()
    train=np.flatnonzero(d.arrays['patient_splits']=='train')
    cache=batches(d,c,ext,train)
    model=SurgeryModel(base,42,'control',bins).cuda().eval()
    checkpoint=REFERENCE/'seed42/control/best.pt'
    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['state_dict'])
    core=model.detail.base
    groups=dict(shared_head=model.delta_head[0],lab=core.lab,text=core.text_projection,fusion=core.fusion)
    rows=[]
    selected=np.sort(np.random.default_rng(20261013).choice(len(cache),min(16,len(cache)),replace=False))
    parameters=[p for module in groups.values() for p in module.parameters() if p.requires_grad]
    lengths=[sum(p.requires_grad for p in module.parameters()) for module in groups.values()]
    for j in selected:
        _,history,b,t,e=cache[j];out=model(history,b);loss=cause_losses(out,t,e)
        torch.testing.assert_close(loss.sum(),competing_nll(out,t,e),atol=1e-6,rtol=1e-6)
        left=torch.autograd.grad(loss[0],parameters,retain_graph=True,allow_unused=True)
        right=torch.autograd.grad(loss[1],parameters,allow_unused=True)
        start=0;result={}
        for name,length in zip(groups,lengths):
            result[name]=gradient_summary(left[start:start+length],right[start:start+length]);start+=length
        rows.append(dict(training_batch=int(j),admissions=len(t),groups=result))
    aggregate={}
    for name in groups:
        values=[x['groups'][name]['cosine'] for x in rows if x['groups'][name]['cosine'] is not None]
        ratios=[x['groups'][name]['death_to_readmission_norm_ratio'] for x in rows if x['groups'][name]['death_to_readmission_norm_ratio'] is not None]
        aggregate[name]=dict(valid_batches=len(values),negative_cosine_batches=sum(x<0 for x in values),
            mean_cosine=float(np.mean(values)) if values else None,
            median_death_to_readmission_norm_ratio=float(np.median(ratios)) if ratios else None)
    ROOT.mkdir()
    write_json(ROOT/'report.json',dict(status='training_only_gradient_diagnostic',aggregate=aggregate,batches=rows,
        checkpoint_sha256=sha256(checkpoint),source_sha256=sha256(Path(__file__)),
        data_provenance=contract['data_provenance'],test_not_evaluated=True,goal_achieved=False,
        definition='cause NLL terms sum exactly to original joint NLL; eval dropout off; fixed trained V6; no optimizer update',
        limits=['Negative cosine is not proof of a causal performance bottleneck.','Gradient scale depends on parameterization.','16 training batches only.']))
    print(json.dumps(dict(status='training_only_gradient_diagnostic',aggregate=aggregate)),flush=True)


if __name__=='__main__':run()
