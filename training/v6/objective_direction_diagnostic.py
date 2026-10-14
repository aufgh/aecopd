"""原V6同训练记录、同状态的风险头方向诊断；没有训练或验证选择。"""
import hashlib
import json
import time
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
from training.v6.patient_weight_loss import per_record_nll
from training.v6.curve_score import daily_cif,curve_scores
from training.v6.token_model import fast_cif
from training.v6.train import write_json

ROOT=Path('training_runs/v6_objective_direction_2026-10-13')
NAMES=('nll','readmission','death','joint')
SOURCES=('training/v6/objective_direction_diagnostic.py','training/v6/curve_score.py',
         'docs/V6_OBJECTIVE_DIRECTION_PROTOCOL_2026-10-13.md')


def output(raw,prior,base_shape):
    delta=raw.reshape(-1,2,2)
    return RiskOutput((prior+delta[:,:,0]).clamp(-10,6),F.softplus(base_shape+delta[:,:,1])+.1)


def losses(out,t,e):
    curve=daily_cif(out)
    return dict(nll=per_record_nll(out,t,e),**curve_scores(curve,t,e)),curve


def cosine(a,b):
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a.flatten()@b.flatten()/norm) if norm>1e-18 else None


def compare(saved,current,path=''):
    if isinstance(current,dict):
        if current.keys()!=saved.keys():raise ValueError('replay keys differ '+path)
        return max([compare(saved[k],v,path+'/'+k) for k,v in current.items()]+[0.])
    if isinstance(current,list):
        if len(current)!=len(saved):raise ValueError('replay list differs '+path)
        return max([compare(x,y,path) for x,y in zip(saved,current)]+[0.])
    if isinstance(current,(float,int)) and not isinstance(current,bool):
        error=abs(current-saved)
        if error>1e-10:raise ValueError('replay numeric differs '+path)
        return error
    if saved!=current:raise ValueError('replay differs '+path)
    return 0.


def run():
    start=time.perf_counter()
    d,c,h,ext,_,bins,_,_,search,_=checked()
    ix=np.flatnonzero(d.arrays['patient_splits']=='train')
    cache=batches(d,c,ext,ix);results={}
    for seed in (42,7):
        base=restore(Path(search['input_config']['parent']),seed,'survival_base',h,.1)
        model=teacher(base,seed,bins).eval();records=[]
        initial_error=0.;forward_error=0.
        with torch.no_grad():
            for _,history,batch,t,e in cache:
                state,_=model.encoder(batch.base);hidden=model.delta_head[:-1](state)
                raw=model.delta_head[-1](hidden);prior=model.baseline(history)
                predicted=output(raw,prior.log_amplitude,model.baseline.raw_shape)
                expected=model(history,batch)
                for a,b in [(predicted.log_amplitude,expected.log_amplitude),(predicted.shape,expected.shape)]:
                    initial_error=max(initial_error,float((a-b).abs().max()))
                if initial_error!=0:raise ValueError('teacher parameter reconstruction differs')
                ours=daily_cif(predicted).cpu().numpy()
                official=fast_cif(expected.log_amplitude.cpu().numpy(),expected.shape.cpu().numpy(),device='cuda')
                forward_error=max(forward_error,float(np.max(np.abs(ours-official))))
                if forward_error>1e-12:raise ValueError('official daily forward differs')
                records.append((torch.cat((hidden,torch.ones(len(t),1,device='cuda')),1),raw,
                                prior.log_amplitude,t,e))
        gradients={name:np.zeros((4,33),np.float64) for name in NAMES}
        values={name:0. for name in NAMES};count=0
        for hidden,raw,prior,t,e in records:
            raw=raw.detach().requires_grad_(True)
            scored,_=losses(output(raw,prior,model.baseline.raw_shape.detach()),t,e)
            for j,name in enumerate(NAMES):
                values[name]+=float(scored[name].detach().sum())/len(ix)
                g=torch.autograd.grad(scored[name].sum(),raw,retain_graph=j<len(NAMES)-1)[0]
                gradients[name]+=(g.double().T@hidden.double()).detach().cpu().numpy()/len(ix)
            count+=len(t)
        if count!=len(ix) or not all(np.isfinite(g).all() for g in gradients.values()):
            raise ValueError('training coverage/gradient invalid')
        pairwise={a:{b:cosine(gradients[a],gradients[b]) for b in NAMES} for a in NAMES}
        component={str(i):{name:cosine(gradients['nll'][i],gradients[name][i]) for name in NAMES[1:]}
                   for i in range(4)}
        steps={}
        for direction,g in gradients.items():
            tensor=torch.as_tensor(g,dtype=torch.float64,device='cuda')
            projected=[hidden.double()@tensor.T for hidden,*_ in records]
            rms=float(torch.sqrt(sum(p.square().sum() for p in projected)/(len(ix)*4)))
            if rms<1e-15:raise ValueError('cannot normalize zero direction')
            rows=[]
            for size in (.001,.01):
                total={name:0. for name in NAMES}
                with torch.no_grad():
                    for (hidden,raw,prior,t,e),p in zip(records,projected):
                        adjusted=raw-(p*(size/rms)).to(raw.dtype)
                        scored,_=losses(output(adjusted,prior,model.baseline.raw_shape.detach()),t,e)
                        for name in NAMES:total[name]+=float(scored[name].sum())/len(ix)
                rows.append(dict(raw_delta_rms=size,losses=total,
                    changes={name:total[name]-values[name] for name in NAMES},
                    first_order_changes={name:float(-size/rms*np.sum(gradients[name]*g)) for name in NAMES}))
            steps[direction]=rows
        results[str(seed)]=dict(losses=values,gradient_norms={k:float(np.linalg.norm(v)) for k,v in gradients.items()},
            gradient_cosines=pairwise,component_cosines=component,
            component_order=['readmission_amplitude','readmission_raw_shape','death_amplitude','death_raw_shape'],
            local_descent=steps,original_parameter_max_error=initial_error,official_curve_max_error=forward_error,
            teacher_sha256=sha256(REFERENCE/f'seed{seed}/control/best.pt'))
        print(json.dumps(dict(seed=seed,losses=values,gradient_cosines=pairwise,
                             seconds=round(time.perf_counter()-start,1))),flush=True)
        del model
    report=dict(models=results,admissions=len(ix),patients=len(np.unique(d.arrays['subject_ids'][ix])),
        training_index_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),
        label_sha256=d.provenance['label_sha256'],source_sha256={p:sha256(Path(p)) for p in SOURCES},
        search_contract_sha256=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')),
        scope='All training discharges; fixed eval upstream; final linear projection only; no fits or validation selection.',
        goal_achieved=False)
    ROOT.mkdir(exist_ok=True)
    path=ROOT/'diagnostic.json'
    if path.exists():
        error=compare(json.loads(path.read_text()),report)
        write_json(ROOT/'independent_verification.json',dict(status='independently_replayed',maximum_error=error,
            report_sha256=sha256(path),seconds=time.perf_counter()-start,models=2))
    else:write_json(path,report)
    print(json.dumps(dict(status='completed_training_only_diagnostic',seconds=round(time.perf_counter()-start,1))),flush=True)


if __name__=='__main__':run()
