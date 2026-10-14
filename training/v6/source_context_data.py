"""显式选择补充来源的 V6 输入接口；不静默修改旧实验配方、不构造标签。"""
from __future__ import annotations
from dataclasses import dataclass, replace
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256


@dataclass(frozen=True)
class SourceContextBatch:
    clinical: object               # 原 ClinicalBatch，仅给药 source_mask 修正
    eos_features: object           # [N,2,20]，6值+6mask+首次/末次四类时间上下文
    micro_values: object           # [N,6]，未观察不等于阴性
    micro_masks: object
    resp_numeric_scaled: object    # [N,8,5]，不是连续时序
    resp_numeric_masks: object
    support_values: object         # [N,9]，记录证据，不推断急性加重病因
    support_masks: object
    recorded_duration_hours: object # [N,2] 审计值，未归一化，不默认送网络
    recorded_duration_masks: object


def checked_arrays(directory, status, filename, base):
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!=status: raise ValueError('supplement unfinished')
    parent=manifest.get('parent_clinical_state_manifest_sha256',manifest.get('parent_state_manifest_sha256'))
    if parent!=base.provenance['state_manifest_sha256']: raise ValueError('supplement/base lineage differs')
    for name,digest in manifest['output_sha256'].items():
        if sha256(directory/name)!=digest: raise ValueError('supplement output changed')
    with np.load(directory/filename,allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    for key in ('hadm_ids','subject_ids','patient_splits'):
        if not np.array_equal(arrays[key],base.arrays[key]): raise ValueError('member, order or patient split differs')
    return arrays,manifest


@dataclass(frozen=True)
class SourceContextData:
    base: object
    eos: dict
    resp_micro: dict
    provenance: dict

    def collate(self,positions,device='cpu'):
        import torch
        h,clinical,t,e=self.base.collate(positions,device=device)
        positions=np.asarray(positions,dtype=np.int64)
        tensor=lambda array:torch.as_tensor(np.asarray(array)[positions],device=device)
        clinical=replace(clinical,med_source_mask=tensor(self.eos['med_source_mask']))
        batch=SourceContextBatch(clinical,tensor(self.eos['eos_features']),
            tensor(self.resp_micro['micro_values']),tensor(self.resp_micro['micro_masks']),
            tensor(self.resp_micro['resp_numeric_scaled']).float(),tensor(self.resp_micro['resp_numeric_masks']),
            tensor(self.resp_micro['support_values']),tensor(self.resp_micro['support_masks']),
            tensor(self.resp_micro['recorded_duration_hours']),tensor(self.resp_micro['recorded_duration_masks']))
        return h,batch,t,e


def attach(base,eos_dir:Path,resp_micro_dir:Path):
    eos,em=checked_arrays(eos_dir,'eos_med_context_validated_not_trained','eos_med_context.npz',base)
    rm,mm=checked_arrays(resp_micro_dir,'resp_micro_context_validated_not_trained','resp_micro_context.npz',base)
    n=len(base.arrays['hadm_ids'])
    shapes={'eos_features':(n,2,20),'med_source_mask':(n,7),'micro_values':(n,6),'micro_masks':(n,6),
            'resp_numeric_scaled':(n,8,5),'resp_numeric_masks':(n,8,5),'support_values':(n,9),
            'support_masks':(n,9),'recorded_duration_hours':(n,2),'recorded_duration_masks':(n,2)}
    for key,shape in shapes.items():
        value=(eos if key in eos else rm)[key]
        if value.shape!=shape or not np.isfinite(value).all(): raise ValueError('invalid supplementary tensor')
        if 'mask' in key and value.dtype!=bool: raise ValueError('invalid mask type')
    if not np.array_equal(eos['med_exposure'],base.arrays['med_exposure']): raise ValueError('medication classes changed')
    if ((eos['med_exposure']>0)&~eos['med_source_mask']).any(): raise ValueError('positive medication without source')
    for values,masks in (('micro_values','micro_masks'),('support_values','support_masks')):
        if not np.isin(rm[values],(0,1)).all() or ((rm[values]>0)&~rm[masks]).any(): raise ValueError('positive flag without source')
    return SourceContextData(base,eos,rm,{'eos_manifest_sha256':sha256(eos_dir/'manifest.json'),
                                        'resp_micro_manifest_sha256':sha256(resp_micro_dir/'manifest.json')})
