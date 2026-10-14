"""冻结数据合同的token桥接；标签只交给损失，不进入TokenBatch。"""
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.compact_ablation import inputs
from training.v6.exploration_model import expanded_inputs
from training.v6.token_model import TokenBatch


def load_inputs(reference,compact_dir,extension,medication):
    d,c,hist,_=inputs(reference,compact_dir)
    result=[]; provenance={}
    for folder,file,status in ((extension,'extension.npz','source_checked_extension_not_trained'),
                               (medication,'medication_identity.npz','validated_medication_identity')):
        m=json.loads((folder/'manifest.json').read_text())
        if m['status']!=status or m['parent_state_manifest_sha256']!=d.provenance['state_manifest_sha256'] or sha256(folder/file)!=m['output_sha256']:
            raise ValueError('token data lineage mismatch')
        with np.load(folder/file) as z: a={k:z[k] for k in z.files}
        for key in ('hadm_ids','subject_ids','patient_splits'):
            if not np.array_equal(a[key],d.arrays[key]): raise ValueError('member/split mismatch')
        result.append(a); provenance[file]=sha256(folder/'manifest.json')
    return d,c,hist,*result,provenance


def prepare(d,c,ext,med,idx,*,size=256,device='cuda'):
    cache=[]
    for start in range(0,len(idx),size):
        pos=idx[start:start+size]; h,compact,t,e=c.collate(pos,device); b=compact.clinical
        h=expanded_inputs(h,compact)
        lab=np.zeros((len(pos),29,20),np.float32); valid=np.zeros_like(lab,bool)
        lab[:,:22,:9]=d.arrays['lab_axis_inputs'][pos]
        present=lab[:,:22,6]>0
        valid[:,:22,:6]=present[:,:,None];valid[:,:22,3]&=lab[:,:22,7]>0;valid[:,:22,5]&=lab[:,:22,7]>0
        lab[:,22:27,:10]=ext['new_labs'][pos];valid[:,22:27,:5]=ext['new_labs'][pos,:,5:].astype(bool)
        lab[:,27:]=ext['eos'][pos];valid[:,27:,:6]=ext['eos'][pos,:,6:12].astype(bool)
        observed=np.concatenate((present,valid[:,22:27,:5].any(2),valid[:,27:,:6].any(2)),1)
        length=max(1,max(len(d.text_refs[p]) for p in pos));vectors=np.zeros((len(pos),length,768),np.float32)
        fields=np.zeros((len(pos),length),np.int64);tm=np.zeros((len(pos),length),bool)
        for slot,p in enumerate(pos):
            for j,(field,row) in enumerate(d.text_refs[p]):
                vectors[slot,j]=d.text_vectors[row];fields[slot,j]=field;tm[slot,j]=True
        tensor=lambda a:torch.as_tensor(a,device=device)
        other=torch.cat((torch.stack((compact.micro_values,compact.micro_masks.float()),2),
                         torch.stack((compact.support_values,compact.support_masks.float()),2)),1)
        batch=TokenBatch(tensor(lab),tensor(valid),tensor(observed),compact.med_features,
                         tensor(med['features'][pos]),tensor(med['present'][pos]),other,
                         tensor(vectors),tensor(fields),tensor(tm),tensor(np.zeros_like(observed)))
        cache.append((pos,h,batch,t,e))
    return cache
