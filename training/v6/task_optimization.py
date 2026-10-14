"""V6双原因模块/共享梯度替换；原编码与概率族不改。"""
import copy
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.regularization_model import Trial
from training.v6.module_surgery import SurgeryModel
from training.v6.risk import RiskOutput
from training.v6.task_gradient_diagnostic import cause_losses


@dataclass(frozen=True)
class TaskTrial(Trial):
    head: str = 'shared32'
    pcgrad: bool = False
    balance: bool = False
    independent_encoders: bool = False


TRIALS=(
    TaskTrial('control'),
    TaskTrial('shared64',head='shared64'),
    TaskTrial('specific32',head='specific32'),
    TaskTrial('specific64',head='specific64'),
    TaskTrial('independent32',head='specific32',independent_encoders=True),
    TaskTrial('pcgrad_shared32',pcgrad=True),
    TaskTrial('pcgrad_specific32',head='specific32',pcgrad=True),
    TaskTrial('balanced_specific32',head='specific32',pcgrad=True,balance=True),
)


class SpecificHead(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.heads=nn.ModuleList(nn.Sequential(nn.Linear(128,width),nn.GELU(),nn.Linear(width,2)) for _ in range(2))
        for h in self.heads:
            nn.init.zeros_(h[-1].weight);nn.init.zeros_(h[-1].bias)

    def forward(self,x):return torch.cat([h(x) for h in self.heads],1)


class IndependentCauses(nn.Module):
    def __init__(self,original,width=32):
        super().__init__();self.baseline=original.baseline
        self.encoders=nn.ModuleList([original.encoder,copy.deepcopy(original.encoder)])
        self.head=SpecificHead(width)

    def train(self,mode=True):
        super().train(mode);self.baseline.eval();return self

    def forward(self,history,batch):
        with torch.no_grad():base=self.baseline(history)
        states=[encoder(batch.base)[0] for encoder in self.encoders]
        delta=torch.stack([h(x) for h,x in zip(self.head.heads,states)],1)
        return RiskOutput((base.log_amplitude+delta[:,:,0]).clamp(-10,6),
            F.softplus(self.baseline.raw_shape.detach()+delta[:,:,1])+.1,delta)


def build(base,seed,trial,bins,*,text_width=768,device='cuda'):
    model=SurgeryModel(base,seed,'control',bins,text_width=text_width)
    if trial.independent_encoders:model=IndependentCauses(model,32)
    elif trial.head.startswith('specific'):model.delta_head=SpecificHead(int(trial.head.removeprefix('specific')))
    elif trial.head=='shared64':
        model.delta_head=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,4))
        nn.init.zeros_(model.delta_head[-1].weight);nn.init.zeros_(model.delta_head[-1].bias)
    model.arm=trial.name
    model=model.to(device);torch.manual_seed(seed)
    return model


def projected_sum(left,right,*,balance=False):
    """双任务对称投影，只用于共享参数；保留原联合损失的sum尺度。"""
    aa=sum(g.square().sum() for g in left);bb=sum(g.square().sum() for g in right)
    if balance and aa>1e-24 and bb>1e-24:
        mean=(aa.sqrt()+bb.sqrt())*.5
        left=[g*(mean/aa.sqrt()).clamp(.25,4.) for g in left]
        right=[g*(mean/bb.sqrt()).clamp(.25,4.) for g in right]
        aa=sum(g.square().sum() for g in left);bb=sum(g.square().sum() for g in right)
    dot=sum((a*b).sum() for a,b in zip(left,right))
    if dot<0:
        return [(a-dot/bb.clamp_min(1e-24)*b)+(b-dot/aa.clamp_min(1e-24)*a) for a,b in zip(left,right)]
    return [a+b for a,b in zip(left,right)]


def backward(model,out,nll,penalty,t,e,trial):
    if not trial.pcgrad:
        (nll+penalty).backward();return
    shared=[p for p in model.encoder.parameters() if p.requires_grad]
    if not isinstance(model.delta_head,SpecificHead):shared.extend(model.delta_head[0].parameters())
    terms=cause_losses(out,t,e)
    left=torch.autograd.grad(terms[0],shared,retain_graph=True,allow_unused=True)
    right=torch.autograd.grad(terms[1],shared,retain_graph=True,allow_unused=True)
    extra=torch.autograd.grad(penalty,shared,retain_graph=True,allow_unused=True)
    clean=lambda gs:[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(shared,gs)]
    corrected=projected_sum(clean(left),clean(right),balance=trial.balance)
    (nll+penalty).backward()  # 专用头保持普通联合似然梯度，不投影成对方任务。
    with torch.no_grad():
        for p,g,reg in zip(shared,corrected,extra):p.grad=g+(reg.detach() if reg is not None else 0.)
