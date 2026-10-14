"""原V6风险头最后投影的真实逐记录梯度；上游固定eval，不概括全网络训练。"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.module_surgery_experiment import batches
from training.v6.warm_residual_audit import teacher
from training.v6.horizon_objective import restore
from training.v6.risk import RiskOutput
from training.v6.patient_weight_loss import per_record_nll,training_weights
from training.v6.train import write_json

ROOT=Path('training_runs/v6_patient_gradient_diagnostic_2026-10-13')


def aggregate(subjects,gradients):
    people,inverse,count=np.unique(subjects,return_inverse=True,return_counts=True)
    patient_grad=np.zeros((len(people),gradients.shape[1]),np.float64)
    np.add.at(patient_grad,inverse,gradients)
    norms=np.linalg.norm(gradients,axis=1)
    patient_norm_sum=np.bincount(inverse,weights=norms)
    net_norm=np.linalg.norm(patient_grad,axis=1)
    order=np.argsort(-count,kind='stable');groups={}
    for fraction in (1,5,10,20):
        n=max(1,int(np.ceil(len(people)*fraction/100)));mask=np.zeros(len(people),bool);mask[order[:n]]=True
        total=patient_grad.sum(0)
        groups[str(fraction)]=dict(patients=n,admission_share=float(count[mask].sum()/count.sum()),
            individual_gradient_norm_mass_share=float(patient_norm_sum[mask].sum()/patient_norm_sum.sum()),
            patient_net_gradient_norm_mass_share=float(net_norm[mask].sum()/net_norm.sum()),
            projection_on_total_gradient=float(patient_grad[mask].sum(0)@total/(total@total)) if total@total>1e-18 else None)
    schemes={}
    for power in (0.,.5,1.):
        w=training_weights(subjects,power)
        weighted=gradients*w[:,None];pg=np.zeros_like(patient_grad);np.add.at(pg,inverse,weighted)
        total=weighted.mean(0);unweighted=gradients.mean(0)
        denom=np.linalg.norm(total)*np.linalg.norm(unweighted)
        schemes[str(power)]=dict(mean_gradient_norm=float(np.linalg.norm(total)),
            cosine_with_unweighted=float(total@unweighted/denom) if denom>1e-18 else None,
            top10_patient_net_norm_share=float(np.linalg.norm(pg[order[:max(1,int(np.ceil(len(people)*.1)))]],axis=1).sum()/np.linalg.norm(pg,axis=1).sum()))
    return dict(top_patient_groups=groups,weight_schemes=schemes)


def run():
    d,c,h,ext,_,bins,_,_,search,_=checked();ix=np.flatnonzero(d.arrays['patient_splits']=='train')
    cache=batches(d,c,ext,ix);result={}
    for seed in (42,7):
        base=restore(Path(search['input_config']['parent']),seed,'survival_base',h,.1)
        model=teacher(base,seed,bins).eval();vectors=[]
        for _,history,batch,t,e in cache:
            with torch.no_grad():
                cstate,_=model.encoder(batch.base);hidden=model.delta_head[:-1](cstate)
                raw=model.delta_head[-1](hidden);prior=model.baseline(history);expected=model(history,batch)
            raw=raw.detach().requires_grad_(True);delta=raw.reshape(-1,2,2)
            out=RiskOutput((prior.log_amplitude+delta[:,:,0]).clamp(-10,6),
                F.softplus(model.baseline.raw_shape.detach()+delta[:,:,1])+.1)
            torch.testing.assert_close(out.log_amplitude,expected.log_amplitude,atol=0,rtol=0)
            torch.testing.assert_close(out.shape,expected.shape,atol=0,rtol=0)
            grad=torch.autograd.grad(per_record_nll(out,t,e).sum(),raw)[0]
            augmented=torch.cat((hidden,torch.ones(len(t),1,device='cuda')),1)
            vector=(grad[:,:,None]*augmented[:,None,:]).flatten(1)
            vectors.append(vector.detach().cpu().numpy())
        vectors=np.concatenate(vectors).astype(np.float64)
        if vectors.shape!=(len(ix),132) or not np.isfinite(vectors).all():raise ValueError('invalid gradient interface')
        result[str(seed)]=aggregate(d.arrays['subject_ids'][ix],vectors)
        result[str(seed)]['teacher_checkpoint_sha256']=sha256(REFERENCE/f'seed{seed}/control/best.pt')
        del model
    ROOT.mkdir(exist_ok=True)
    report=dict(status='last_projection_gradient_measured',models=result,patients=len(np.unique(d.arrays['subject_ids'][ix])),
        admissions=len(ix),source_sha256={p:sha256(Path(p)) for p in ('training/v6/patient_gradient_diagnostic.py','training/v6/patient_weight_loss.py')},
        label_sha256=d.provenance['label_sha256'],goal_achieved=False,
        limits=['Eval-mode upstream fixed; final risk projection only, not full-network training trajectory.',
            'Gradient directions can cancel; norm mass and projection are distinct.',
            'Only training labels analyzed; weighting does not prove better validation performance.'])
    write_json(ROOT/'diagnostic.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':run()
