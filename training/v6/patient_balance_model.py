"""仅损失权重/编码器可塑性的受控适配，不修改原V6网络。"""
from dataclasses import dataclass
import numpy as np
import torch
from training.v6.warm_residual import WarmTrial,build as warm_build


@dataclass(frozen=True)
class BalanceTrial(WarmTrial):
    mode:str='finetune'
    lr:float=2e-5
    power:float=0.
    head_only:bool=False


TRIALS=(BalanceTrial('head_uniform',head_only=True),BalanceTrial('head_sqrt',power=.5,head_only=True),
    BalanceTrial('head_inverse',power=1.,head_only=True),BalanceTrial('full_sqrt',power=.5),BalanceTrial('full_inverse',power=1.))
UNIFORM=BalanceTrial('full_uniform')


def build(base,seed,trial,bins):
    model=warm_build(base,seed,trial,bins)
    if trial.head_only:model.encoder.requires_grad_(False)
    # 不把冻结encoder强制eval，保持匹配训练dropout，仅改可塑性。
    return model


def uniform_step_proof(base,seed,bins,cache):
    """普通新损失与旧warm控制的实际GPU损失/梯度/更新对照。"""
    from training.v6.patient_weight_loss import weighted_nll
    from training.v6.module_surgery import member_nll
    previous=warm_build(base,seed,WarmTrial('warm_finetune',mode='finetune',lr=2e-5),bins)
    candidate=build(base,seed,UNIFORM,bins)
    _,history,batch,t,e=cache[np.random.default_rng(seed).permutation(len(cache))[0]]
    old_state=previous.state_dict();new_state=candidate.state_dict()
    for key,value in old_state.items():torch.testing.assert_close(value,new_state[key],atol=0,rtol=0)
    values=[];gradients=[]
    for index,model in enumerate((previous,candidate)):
        torch.manual_seed(seed);model.train()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5*.2,weight_decay=1e-4)
        out=model(history,batch)
        nll=member_nll(out,t,e) if index==0 else weighted_nll(out,t,e,torch.ones_like(t))
        objective=nll+1e-4*out.delta.square().mean();objective.backward()
        gradients.append({k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None})
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();values.append(float(objective.detach()))
    if values[0]!=values[1] or gradients[0].keys()!=gradients[1].keys():raise ValueError('ordinary balance/old warm step differs')
    for key,value in gradients[0].items():torch.testing.assert_close(value,gradients[1][key],atol=0,rtol=0)
    for key,value in previous.state_dict().items():torch.testing.assert_close(value,candidate.state_dict()[key],atol=0,rtol=0)
    return dict(loss=values[0],gradient_tensors=len(gradients[0]),maximum_gradient_error=0.,maximum_updated_weight_error=0.,
        limits='first-step and recipe equivalence; not a newly executed full 100-epoch trajectory')
