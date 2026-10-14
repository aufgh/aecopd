"""独立粗细层级候选：细粒度保留至最后，统一治疗记录，不把来源作预测捷径。"""
from dataclasses import dataclass, replace
import torch
from torch import nn
from training.v6.model import AnchoredModel, masked_pool


@dataclass(frozen=True)
class AuditedBatch:
    labs: torch.Tensor
    lab_present: torch.Tensor
    medications: torch.Tensor
    med_present: torch.Tensor
    coarse: torch.Tensor
    micro: torch.Tensor
    micro_present: torch.Tensor
    support: torch.Tensor
    support_present: torch.Tensor
    text: torch.Tensor
    fields: torch.Tensor
    text_present: torch.Tensor


def field_means(b):
    outputs=[];flags=[]
    for field in range(4):
        mask=b.text_present&(b.fields==field)
        outputs.append((b.text*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]);flags.append(mask.any(1))
    return torch.stack(outputs,1),torch.stack(flags,1)


def flat_features(b):
    text,flags=field_means(b)
    return torch.cat((b.labs.flatten(1),b.lab_present.float(),b.medications.flatten(1),b.med_present.float(),
      b.coarse.flatten(1),b.micro,b.support,text.flatten(1),flags.float()),1)


class AuditedEncoder(nn.Module):
    def __init__(self,lab_family,drug_parent,*,mode='pool',width=128):
        super().__init__();self.mode=mode;self.width=width
        lf=torch.as_tensor(lab_family,dtype=torch.long);dp=torch.as_tensor(drug_parent,dtype=torch.float32)
        self.register_buffer('lab_family',lf);self.register_buffer('drug_parent',dp)
        self.lab=nn.Sequential(nn.Linear(24,32),nn.GELU(),nn.Linear(32,width))
        self.drug=nn.Sequential(nn.Linear(11,32),nn.GELU(),nn.Linear(32,width))
        self.coarse=nn.Linear(2,width);self.micro=nn.Linear(8,width);self.support=nn.Linear(6,width)
        self.lab_ids=nn.Embedding(len(lf),width);self.drug_ids=nn.Embedding(len(dp),width)
        self.lab_parent_ids=nn.Embedding(7,width);self.med_parent_ids=nn.Embedding(10,width)
        self.types=nn.Embedding(5,width);self.fields=nn.Embedding(4,width)
        self.local=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,width))
        layer=nn.TransformerEncoderLayer(width,4,256,.1,batch_first=True,norm_first=True)
        self.cross=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.cross.requires_grad_(mode in ('interaction_pool','interaction_cls'))
        self.cls=nn.Parameter(torch.zeros(1,1,width));self.cls.requires_grad_(mode=='interaction_cls')
        self.score=nn.Linear(width,1);self.norm=nn.LayerNorm(width)
        self.text_projection=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,width),nn.GELU())

    def tokens(self,b):
        n=len(b.labs);dev=b.labs.device
        if b.labs.shape[1:]!=(len(self.lab_family),24) or b.medications.shape[1:]!=(len(self.drug_parent),11):raise ValueError('audit interface mismatch')
        lab=self.lab(b.labs)+self.lab_ids(torch.arange(len(self.lab_family),device=dev))+self.lab_parent_ids(self.lab_family)+self.types.weight[0]
        # 复方继承多个药效父概念，平均而非把同一剂量重复加到不同药效。
        parent=self.drug_parent@self.med_parent_ids.weight/self.drug_parent.sum(1).clamp_min(1)[:,None]
        drug=self.drug(b.medications)+self.drug_ids(torch.arange(len(parent),device=dev))+parent+self.types.weight[1]
        coarse=self.coarse(b.coarse)+self.med_parent_ids.weight+self.types.weight[1]
        # 检验父概念汇聚仅作新增上下文；子概念仍原样保留，故不是先丢细节。
        parents=[];parent_present=[]
        for g in range(7):
            mask=b.lab_present&(self.lab_family[None]==g)
            parents.append((lab*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]+self.lab_parent_ids.weight[g])
            parent_present.append(mask.any(1))
        text=self.text_projection(b.text)+self.fields(b.fields)+self.types.weight[4]
        parts=(lab,drug,coarse,torch.stack(parents,1),
          (self.micro(b.micro)+self.types.weight[2])[:,None],(self.support(b.support)+self.types.weight[3])[:,None],text)
        present=torch.cat((b.lab_present,b.med_present,b.coarse[:,:,1]>0,torch.stack(parent_present,1),
                           b.micro_present[:,None],b.support_present[:,None],b.text_present),1)
        tokens=torch.cat(parts,1);tokens=torch.where(present[:,:,None],tokens,0.)
        tokens=tokens+self.local(tokens)
        return tokens,present

    def forward(self,b,*,mask_pmh=False):
        if mask_pmh:b=replace(b,text_present=b.text_present&(b.fields!=1))
        x,present=self.tokens(b)
        if self.mode=='interaction_cls':
            # 一个最终读出token：不对检验/药物/文本分别提前池化。
            x=torch.cat((self.cls.expand(len(x),-1,-1),x),1)
            padding=torch.cat((torch.zeros(len(x),1,dtype=torch.bool,device=x.device),~present),1)
            h=self.cross(x,src_key_padding_mask=padding)[:,0]
        else:
            safe=present.clone();safe[~safe.any(1),0]=True
            if self.mode=='interaction_pool':x=self.cross(x,src_key_padding_mask=~safe)
            h=masked_pool(x,present,self.score)
        return self.norm(h),dict(tokens=x,present=present)


class FlatEncoder(nn.Module):
    def __init__(self,width):
        super().__init__();self.net=nn.Sequential(nn.Linear(width,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,128),nn.GELU(),nn.LayerNorm(128))
    def forward(self,b,*,mask_pmh=False):
        if mask_pmh:b=replace(b,text_present=b.text_present&(b.fields!=1))
        return self.net(flat_features(b)),{}


class AuditedModel(AnchoredModel):
    def __init__(self,baseline,lab_family,drug_parent,*,mode='pool'):
        nn.Module.__init__(self);self.baseline=baseline.requires_grad_(False).eval();self.patient_shape=True
        self.delta_head=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,4))
        nn.init.zeros_(self.delta_head[-1].weight);nn.init.zeros_(self.delta_head[-1].bias)
        if mode=='flat':
            size=len(lab_family)*25+len(drug_parent)*12+20+8+6+4*768+4
            self.encoder=FlatEncoder(size)
        else:self.encoder=AuditedEncoder(lab_family,drug_parent,mode=mode)
