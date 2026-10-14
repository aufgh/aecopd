"""V6显式接收胸片变化、CT事实和小结事实；保留旧BERT和冻结H。"""
from dataclasses import dataclass,replace
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from training.v6.compact_model import CompactClinicalEncoder
from training.v6.model import AnchoredModel
from patient_data.export_mimiciv import sha256
from patient_data.clinical_facts.features import CT_NAMES,DISCHARGE_NAMES
from patient_data.clinical_facts.cxr_trajectory import FEATURES,SELECTED

CHANNELS=('cxr','ct','discharge')
WIDTHS=(98,len(CT_NAMES),len(DISCHARGE_NAMES))

@dataclass(frozen=True)
class FactsBatch:
    compact: object
    values: tuple
    present: torch.Tensor  # [N,3]，来源可用性而非疾病阳性

@dataclass(frozen=True)
class FactsData:
    compact: object
    arrays: dict
    provenance: dict
    available_channels: tuple

    def collate(self,positions,device='cpu'):
        h,b,t,e=self.compact.collate(positions,device=device)
        values=tuple(torch.as_tensor(self.arrays[k+'_values'][positions],device=device) for k in CHANNELS)
        p=torch.as_tensor(np.stack([self.arrays[k+'_present'][positions] for k in CHANNELS],1),device=device)
        return h,FactsBatch(b,values,p),t,e

def checked(directory,filename,status,compact):
    m=json.loads((directory/'manifest.json').read_text())
    if m['status']!=status or m['parent_state_manifest_sha256']!=compact.base.provenance['state_manifest_sha256']:
        raise ValueError('input status or parent mismatch')
    for name,value in m['output_sha256'].items():
        if sha256(directory/name)!=value: raise ValueError('input source changed')
    with np.load(directory/filename) as f: a={k:f[k] for k in f.files}
    for key in ('hadm_ids','subject_ids','patient_splits'):
        if not np.array_equal(a[key],compact.base.arrays[key]): raise ValueError('member/split differs')
    return a,sha256(directory/'manifest.json')

def attach(compact,cxr_dir,facts_dir=None):
    a,digest=checked(cxr_dir,'cxr_trajectory.npz','all_cxr_labeled_and_aggregated',compact)
    if tuple(a['feature_names'])!=FEATURES or tuple(a['condition_names'])!=SELECTED: raise ValueError('CXR dictionary differs')
    n=len(a['hadm_ids']); provenance={'cxr_trajectory_manifest_sha256':digest}
    if facts_dir is not None:
        f,digest=checked(facts_dir,'facts.npz','full_facts_source_checked_not_prediction_validated',compact)
        if tuple(f['ct_names'])!=CT_NAMES or tuple(f['discharge_names'])!=DISCHARGE_NAMES: raise ValueError('fact dictionary differs')
        for key in ('ct_values','ct_present','discharge_values','discharge_present'): a[key]=f[key]
        provenance['facts_manifest_sha256']=digest; channels=CHANNELS
    else:
        # 第一条通道独立验证，不将部分LLM抽取当作阴性；另外两支整支关闭。
        for key,width in zip(CHANNELS[1:],WIDTHS[1:],strict=True):
            a[key+'_values']=np.zeros((n,width),np.float32); a[key+'_present']=np.zeros(n,bool)
        channels=('cxr',)
    for key,width in zip(CHANNELS,WIDTHS,strict=True):
        x,p=a[key+'_values'],a[key+'_present']
        if x.shape!=(n,width) or p.shape!=(n,) or p.dtype!=bool or not np.isin(x,(0,1)).all() or (x[~p]!=0).any():
            raise ValueError('invalid clinical fact array')
    if not np.array_equal(a['cxr_present'],a['cxr_values'].reshape(n,7,14)[:,:,:4].sum((1,2))==7):
        raise ValueError('CXR observation mask mismatch')
    return FactsData(compact,a,provenance,channels)

def mask(batch,arm):
    arms={'without_new':(), 'availability_only':CHANNELS, 'cxr_only':('cxr',), 'ct_only':('ct',),
          'discharge_only':('discharge',),'full':CHANNELS}
    if arm not in arms: raise ValueError('unknown arm')
    present=batch.present.clone(); values=list(batch.values)
    for j,key in enumerate(CHANNELS):
        if key not in arms[arm]: present[:,j]=False; values[j]=torch.zeros_like(values[j])
        elif arm=='availability_only': values[j]=torch.zeros_like(values[j])
    return replace(batch,values=tuple(values),present=present)

class FactsEncoder(nn.Module):
    def __init__(self,text_width,width=128):
        super().__init__()
        self.fact_branches=nn.ModuleList([nn.Sequential(nn.Linear(v,32),nn.GELU(),nn.Linear(32,64),nn.LayerNorm(64)) for v in WIDTHS])
        self.fusion=nn.Sequential(nn.Linear(width+3*64+3,width),nn.GELU(),nn.Dropout(.1),nn.Linear(width,width),nn.GELU(),nn.LayerNorm(width))
        self.core=CompactClinicalEncoder(text_width,width=width)
    @property
    def current(self): return self.core.current
    @property
    def lab(self): return self.core.lab
    @property
    def med(self): return self.core.med
    @property
    def microbiology(self): return self.core.microbiology
    @property
    def support(self): return self.core.support
    @property
    def text_projection(self): return self.core.text_projection

    def forward(self,batch,*,mask_pmh=False):
        if not isinstance(batch,FactsBatch): raise TypeError('explicit FactsBatch required')
        n=len(batch.compact.clinical.current_values)
        if batch.present.shape!=(n,3) or batch.present.dtype!=torch.bool: raise ValueError('invalid source mask')
        representations=[]
        for j,(value,width,branch) in enumerate(zip(batch.values,WIDTHS,self.fact_branches,strict=True)):
            if value.shape!=(n,width) or not torch.isfinite(value).all() or not ((value==0)|(value==1)).all():
                raise ValueError('invalid clinical fact tensor')
            representations.append(branch(value)*batch.present[:,j,None])
        current,branches=self.core(batch.compact,mask_pmh=mask_pmh)
        hidden=self.fusion(torch.cat([current,*representations,batch.present.to(current.dtype)],1))
        return hidden,dict(branches,**{k:v for k,v in zip(CHANNELS,representations,strict=True)})

class FactsAnchoredModel(AnchoredModel):
    def __init__(self,baseline,text_width,width=128):
        nn.Module.__init__(self)
        self.baseline=baseline.requires_grad_(False).eval()
        self.delta_head=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,4))
        nn.init.zeros_(self.delta_head[-1].weight); nn.init.zeros_(self.delta_head[-1].bias)
        self.encoder=FactsEncoder(text_width,width); self.patient_shape=True
