"""CheXbert四状态的显式V6接收接口；影像时间仅审计，不作时间输入。"""
from dataclasses import dataclass, replace
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from training.v6.compact_context_data import CompactContextBatch
from training.v6.compact_model import CompactClinicalEncoder
from training.v6.model import AnchoredModel
from patient_data.export_mimiciv import sha256

SELECTED = ('Pneumonia','Consolidation','Edema','Cardiomegaly','Pleural Effusion','Pneumothorax','Atelectasis')
STATES = ('Blank','Positive','Negative','Uncertain')


@dataclass(frozen=True)
class CXRBatch:
    compact: CompactContextBatch
    cxr_onehot: torch.Tensor  # [N,28]，类别不能当有序数值
    cxr_present: torch.Tensor # [N] bool，缺报告和报告内未提及分开


@dataclass(frozen=True)
class CXRData:
    compact: object
    arrays: dict
    provenance: dict

    def collate(self, positions, device='cpu'):
        h,b,t,e=self.compact.collate(positions,device=device)
        positions=np.asarray(positions,dtype=np.int64)
        return h,CXRBatch(b,torch.as_tensor(self.arrays['cxr_onehot'][positions],device=device),
                          torch.as_tensor(self.arrays['cxr_present'][positions],device=device)),t,e


def attach(compact, directory:Path):
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='chexbert_labeled_temporally_checked_not_prediction_validated': raise ValueError('unfinished CXR input')
    if manifest['parent_state_manifest_sha256']!=compact.base.provenance['state_manifest_sha256']: raise ValueError('CXR parent differs')
    for name,digest in manifest['output_sha256'].items():
        if sha256(directory/name)!=digest: raise ValueError('CXR source changed')
    with np.load(directory/'cxr_context.npz',allow_pickle=False) as archive: arrays={k:archive[k] for k in archive.files}
    for key in ('hadm_ids','subject_ids','patient_splits'):
        if not np.array_equal(arrays[key],compact.base.arrays[key]): raise ValueError('CXR member/split differs')
    n=len(arrays['hadm_ids']); present=arrays['cxr_present']; codes=arrays['cxr_states']; onehot=arrays['cxr_onehot']
    if tuple(arrays['condition_names'])!=SELECTED or tuple(arrays['state_names'])!=STATES: raise ValueError('CXR dictionary differs')
    if present.shape!=(n,) or present.dtype!=bool or codes.shape!=(n,7) or not np.isin(codes,(0,1,2,3)).all():
        raise ValueError('invalid CXR codes')
    expected=(np.eye(4,dtype=np.float32)[codes]*present[:,None,None]).reshape(n,28)
    if onehot.shape!=(n,28) or not np.array_equal(onehot,expected): raise ValueError('CXR onehot differs')
    if int(present.sum())!=manifest['reports']: raise ValueError('CXR coverage differs')
    return CXRData(compact,arrays,{'cxr_manifest_sha256':sha256(directory/'manifest.json')})


def mask_cxr(batch, arm):
    if arm not in ('with_cxr','without_cxr'): raise ValueError('unknown CXR arm')
    return batch if arm=='with_cxr' else replace(batch,cxr_onehot=torch.zeros_like(batch.cxr_onehot),
                                                cxr_present=torch.zeros_like(batch.cxr_present))


class CXREncoder(nn.Module):
    def __init__(self,text_width,width=128):
        super().__init__()
        self.xray=nn.Sequential(nn.Linear(28,32),nn.GELU(),nn.Linear(32,64),nn.LayerNorm(64))
        self.fusion=nn.Sequential(nn.Linear(width+64+1,width),nn.GELU(),nn.Dropout(.1),
                                  nn.Linear(width,width),nn.GELU(),nn.LayerNorm(width))
        # 原紧凑编码器最后建立，其中维度依赖的文本投影仍最后初始化。
        self.core=CompactClinicalEncoder(text_width,width=width)

    # 兼容既有训练诊断接口；没有注册重复模块或修改旧encoder。
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
        if not isinstance(batch,CXRBatch): raise TypeError('explicit CXRBatch required')
        x,p=batch.cxr_onehot,batch.cxr_present; n=len(batch.compact.clinical.current_values)
        if (x.shape!=(n,28) or p.shape!=(n,) or p.dtype!=torch.bool or not torch.isfinite(x).all()
                or not ((x==0)|(x==1)).all()): raise ValueError('invalid CXR tensor')
        sums=x.reshape(n,7,4).sum(2)
        if not torch.equal(sums,p[:,None].expand(-1,7).to(sums.dtype)): raise ValueError('CXR states/missingness invalid')
        current,branches=self.core(batch.compact,mask_pmh=mask_pmh)
        image=self.xray(x)*p[:,None]
        hidden=self.fusion(torch.cat((current,image,p[:,None].to(current.dtype)),1))
        return hidden,dict(branches,xray=image,cxr_present=p)


class CXRAnchoredModel(AnchoredModel):
    def __init__(self,baseline,text_width,*,width=128):
        nn.Module.__init__(self)
        self.baseline=baseline.requires_grad_(False).eval()
        self.delta_head=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,4))
        nn.init.zeros_(self.delta_head[-1].weight); nn.init.zeros_(self.delta_head[-1].bias)
        self.encoder=CXREncoder(text_width,width=width)
        self.patient_shape=True
