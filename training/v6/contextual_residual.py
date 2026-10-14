"""V6原位条件化风险修正；历史基础冻结，不宣称因果解耦。"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.module_surgery import SurgeryModel
from training.v6.regularization_model import Trial
from training.v6.risk import RiskOutput


@dataclass(frozen=True)
class ContextTrial(Trial):
    mode: str = 'control'


TRIALS=(ContextTrial('control'),ContextTrial('baseline_only',mode='baseline'),
        ContextTrial('context_concat',mode='concat'),ContextTrial('context_film',mode='film'),
        ContextTrial('context_product',mode='product'),
        ContextTrial('context_product_reg',mode='product',dropout=.3,weight_decay=.01))


class ContextualModel(nn.Module):
    def __init__(self,original,mode,dropout):
        super().__init__()
        if mode not in ('baseline','concat','film','product'):raise ValueError('unknown contextual mode')
        self.baseline,self.encoder=original.baseline,original.encoder
        self.mode=mode
        if mode=='film':
            self.modulation=nn.Sequential(nn.Linear(2,32),nn.GELU(),nn.Linear(32,256))
            # 初始为恒等调制，不额外扭曲原临床表示。
            nn.init.zeros_(self.modulation[-1].weight);nn.init.zeros_(self.modulation[-1].bias)
        width={'baseline':2,'concat':130,'film':128,'product':384}[mode]
        self.delta_head=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Dropout(dropout if dropout>.1 else 0.),nn.Linear(32,4))
        nn.init.zeros_(self.delta_head[-1].weight);nn.init.zeros_(self.delta_head[-1].bias)

    def train(self,mode=True):
        super().train(mode);self.baseline.eval();return self

    def forward(self,history,batch):
        with torch.no_grad():base=self.baseline(history)
        # 固定、平滑、有界变换，不在验证或确认集拟合尺度。
        context=torch.tanh(base.log_amplitude.detach()/4.)
        if self.mode=='baseline':x=context
        else:
            clinical,_=self.encoder(batch.base)
            if self.mode=='concat':x=torch.cat((clinical,context),1)
            elif self.mode=='film':
                gamma,beta=self.modulation(context).chunk(2,1)
                x=(1+torch.tanh(gamma))*clinical+torch.tanh(beta)
            else:x=torch.cat((clinical,clinical*context[:,0:1],clinical*context[:,1:2]),1)
        delta=self.delta_head(x).reshape(-1,2,2)
        return RiskOutput((base.log_amplitude+delta[:,:,0]).clamp(-10,6),
                          F.softplus(self.baseline.raw_shape.detach()+delta[:,:,1])+.1,delta)


def build(base,seed,trial,bins,*,text_width=768,device='cuda'):
    original=SurgeryModel(base,seed,'control',bins,text_width=text_width)
    model=original if trial.mode=='control' else ContextualModel(original,trial.mode,trial.dropout)
    model.arm=trial.name;model=model.to(device);torch.manual_seed(seed)
    return model
