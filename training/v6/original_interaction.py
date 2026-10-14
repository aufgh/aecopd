"""保留原V6融合和旧投影，只在原池化前加入可控残差交互。"""
from dataclasses import dataclass
import torch
from torch import nn
from training.v6.model import masked_pool


@dataclass(frozen=True)
class ExpandedBatch:
    base: object
    extra_labs: torch.Tensor
    medication_details: torch.Tensor
    medication_present: torch.Tensor
    text_slots: torch.Tensor | None = None


def field_pool(content,index,n,score):
    if not len(content):return content.new_zeros(n,content.shape[-1]),torch.zeros(n,device=content.device,dtype=torch.bool)
    group=4*index[:,0]+index[:,1];logits=score(content).squeeze(-1)
    maximum=logits.new_full((4*n,),-torch.inf);maximum.scatter_reduce_(0,group,logits,reduce='amax',include_self=True)
    weights=torch.exp(logits-maximum[group]);total=weights.new_zeros(4*n).scatter_add_(0,group,weights)
    pooled=content.new_zeros(4*n,content.shape[-1]).index_add_(0,group,weights[:,None]/total[group,None]*content)
    present=(total>0).reshape(n,4)
    return pooled.reshape(n,4,-1).sum(1)/present.sum(1).clamp_min(1)[:,None],present.any(1)


class OriginalInteractionEncoder(nn.Module):
    def __init__(self,original,*,labs=False,medications=False,interaction=False,drug_count=38):
        super().__init__();self.original=original;self.use_labs=labs;self.use_meds=medications;self.interaction=interaction
        self.lab_extra=nn.Sequential(nn.Linear(10,32),nn.GELU(),nn.Linear(32,128),nn.LayerNorm(128))
        self.lab_ids=nn.Embedding(10,128);self.lab_score=nn.Linear(128,1)
        self.drug_extra=nn.Sequential(nn.Linear(23,32),nn.GELU(),nn.Linear(32,128),nn.LayerNorm(128))
        self.drug_ids=nn.Embedding(drug_count,128);self.drug_score=nn.Linear(128,1)
        self.old_lab_extra=nn.ModuleList((nn.Linear(10,128),nn.Linear(20,128)))
        self.old_ids=nn.Embedding(7,128)
        self.type_ids=nn.Embedding(7,128)
        self.attention=nn.TransformerEncoderLayer(128,4,256,dropout=.1,batch_first=True,norm_first=True)
        self.interaction_gate=nn.Parameter(torch.zeros(()))
        # 所有臂具有相同槽位，但不优化本臂不用的模块。
        for module in (self.lab_extra,self.lab_ids,self.lab_score):module.requires_grad_(labs)
        for module in (self.drug_extra,self.drug_ids,self.drug_score):module.requires_grad_(medications)
        for module in (self.old_lab_extra,self.old_ids,self.type_ids,self.attention):module.requires_grad_(interaction)
        self.interaction_gate.requires_grad_(interaction)

    def forward(self,batch,*,mask_pmh=False):
        if not isinstance(batch,ExpandedBatch):raise TypeError('ExpandedBatch required')
        if not (self.use_labs or self.use_meds or self.interaction):return self.original(batch.base,mask_pmh=mask_pmh)
        detail=batch.base;compact=detail.base;b=compact.clinical;n=len(b.current_values);core=self.original.base
        obs=b.lab_axis_inputs[:,:,6]>0
        lab=core.lab(torch.where(obs[:,:,None],b.lab_axis_inputs,0.))+core.lab_ids(torch.arange(22,device=obs.device))
        index=b.snippet_index;vectors=b.snippet_vectors
        if mask_pmh:
            keep=index[:,1]!=1;index=index[keep];vectors=vectors[keep]
        text=core.text_projection(vectors)+core.field_ids(index[:,1])
        text_length=max(1,max(torch.bincount(index[:,0],minlength=n).tolist()) if len(index) else 1)
        text_dense=lab.new_zeros(n,text_length,128);text_mask=torch.zeros(n,text_length,device=lab.device,dtype=torch.bool)
        # 原有片段没有时间序列位置信息；填槽仅为批处理，顺序不进入注意力。
        if batch.text_slots is None or mask_pmh:
            slots=torch.zeros(len(index),device=lab.device,dtype=torch.long)
            for i in range(n):
                r=torch.where(index[:,0]==i)[0]
                if len(r):slots[r]=torch.arange(len(r),device=lab.device)
        else:slots=batch.text_slots
        text_dense[index[:,0],slots]=text;text_mask[index[:,0],slots]=True
        oldlab=torch.cat((self.old_lab_extra[0](detail.labs[:,:50].reshape(n,5,10)),
                          self.old_lab_extra[1](detail.labs[:,50:].reshape(n,2,20))),1)+self.old_ids(torch.arange(7,device=lab.device))
        oldmask=torch.cat((detail.labs[:,:50].reshape(n,5,10)[:,:,5:].bool().any(2),
                           detail.labs[:,50:].reshape(n,2,20)[:,:,6:12].bool().any(2)),1)
        newlab=self.lab_extra(batch.extra_labs)+self.lab_ids(torch.arange(10,device=lab.device))
        newmask=batch.extra_labs[:,:,5:].bool().any(2)&self.use_labs
        named=self.drug_extra(batch.medication_details)+self.drug_ids(torch.arange(batch.medication_details.shape[1],device=lab.device))
        namedmask=batch.medication_present&self.use_meds
        med=core.med(compact.med_features.flatten(1));mp=compact.med_features[:,:,2:].bool().any((1,2))
        micro=core.microbiology(torch.cat((compact.micro_values,compact.micro_masks.float()),1));up=compact.micro_masks.any(1)
        support=core.support(torch.cat((compact.support_values,compact.support_masks.float()),1));vp=compact.support_masks.any(1)
        groups=(lab,text_dense,oldlab,newlab,named,torch.stack((med,micro),1),support[:,None])
        masks=(obs,text_mask,oldmask,newmask,namedmask,torch.stack((mp,up),1),vp[:,None])
        states=torch.cat(groups,1);present=torch.cat(masks,1);lengths=[x.shape[1] for x in groups]
        if self.interaction:
            types=torch.cat([torch.full((length,),i,device=lab.device,dtype=torch.long) for i,length in enumerate(lengths)])
            safe=present.clone();safe[~safe.any(1),0]=True
            # 明确残差门控为0时，原V6已有投影/池化输出保持不变。
            proposal=self.attention(states+self.type_ids(types),src_key_padding_mask=~safe)
            states=states+torch.tanh(self.interaction_gate)*proposal*present[:,:,None]
        lab,text_dense,updatedold,newlab,named,global_pair,support=states.split(lengths,1)
        l=masked_pool(lab,obs,core.lab_score)
        text,tp=field_pool(text_dense[index[:,0],slots],index,n,core.text_score)
        current=torch.where(b.current_masks,b.current_values,0.)
        s=masked_pool(core.current(torch.stack((current,b.current_masks.float()),-1))+
                      core.current_ids(torch.arange(14,device=lab.device)),b.current_masks,core.current_score)
        flags=(b.current_masks.any(1),obs.any(1),mp,up,vp)
        branches=[norm(x)*p[:,None] for norm,x,p in zip(core.branch_norms,(s,l,global_pair[:,0],global_pair[:,1],support[:,0]),flags)]
        presence=torch.stack((*flags,tp),1).float()
        hidden=core.fusion(torch.cat((*branches,text,presence),1))
        old_add=self.original.extra_labs(detail.labs)*detail.labs.ne(0).any(1)[:,None]
        if self.interaction:
            old_add=old_add+((updatedold-oldlab)*oldmask[:,:,None]).sum(1)/oldmask.sum(1).clamp_min(1)[:,None]
        additions=hidden.new_zeros(n,128)
        if self.use_labs:additions=additions+masked_pool(newlab,newmask,self.lab_score)
        if self.use_meds:additions=additions+masked_pool(named,namedmask,self.drug_score)
        return hidden+old_add+additions,dict(lab=l,text=text,presence=presence,extra=additions,
          interaction_gate=torch.tanh(self.interaction_gate))
