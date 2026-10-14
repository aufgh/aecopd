"""已训练V6原位小型修正/微调，保留原预测，不重置临床表示。"""
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.module_surgery import SurgeryModel
from training.v6.regularization_model import Trial
from training.v6.risk import RiskOutput

REFERENCE=Path('training_runs/v6_module_surgery_2026-10-12_r3')
BASE_SEEDS={42:42,7:7,101:42,202:7,303:42}


@dataclass(frozen=True)
class WarmTrial(Trial):
    mode: str = 'amp'


TRIALS=(WarmTrial('warm_amp'),WarmTrial('warm_product_amp',mode='product'),
        WarmTrial('warm_product_shape',mode='bounded_shape'),WarmTrial('warm_finetune',mode='finetune',lr=2e-5))


class WarmAdapter(nn.Module):
    def __init__(self,teacher,mode):
        super().__init__();self.teacher=teacher.requires_grad_(False).eval();self.mode=mode
        width=128 if mode=='amp' else 384
        self.head=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,2 if mode!='bounded_shape' else 4))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    @property
    def baseline(self):return self.teacher.baseline

    def train(self,mode=True):
        super().train(mode);self.teacher.eval();return self

    def forward(self,history,batch):
        with torch.no_grad():
            prior=self.teacher(history,batch)
            clinical,_=self.teacher.encoder(batch.base)
            context=torch.tanh(self.teacher.baseline(history).log_amplitude/4.)
        x=clinical if self.mode=='amp' else torch.cat((clinical,clinical*context[:,0:1],clinical*context[:,1:2]),1)
        raw=self.head(x)
        if self.mode=='bounded_shape':
            raw=raw.reshape(-1,2,2)
            delta=torch.stack((raw[:,:,0],.25*torch.tanh(raw[:,:,1])),2)
            shape=F.softplus(self.baseline.raw_shape.detach()+prior.delta[:,:,1]+delta[:,:,1])+.1
        else:
            delta=torch.stack((raw,torch.zeros_like(raw)),2);shape=prior.shape
        return RiskOutput((prior.log_amplitude+delta[:,:,0]).clamp(-10,6),shape,delta)


def build(base,seed,trial,bins,*,text_width=768,device='cuda',teacher_state=None):
    teacher=SurgeryModel(base,seed,'control',bins,text_width=text_width)
    state=teacher_state if teacher_state is not None else torch.load(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt',map_location=device,weights_only=True)['state_dict']
    teacher.load_state_dict(state)
    if trial.mode=='finetune':model=teacher
    else:model=WarmAdapter(teacher,trial.mode)
    model.arm=trial.name;model=model.to(device);torch.manual_seed(seed)
    return model
