"""补充精确角度和重复向量审计；不改正在训练的受保护源码/派生向量。"""
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked
from training.v6.data_geometry_audit import ROOT
from training.v6.train import write_json


def cosine_pairs(x,people):
    rng=np.random.default_rng(42);a=rng.integers(len(x),size=20000);b=rng.integers(len(x),size=20000)
    keep=people[a]!=people[b];a,b=a[keep][:10000],b[keep][:10000]
    left,right=x[a].astype(np.float64),x[b].astype(np.float64)
    denominator=np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1)
    valid=denominator>1e-12
    values=np.sum(left[valid]*right[valid],axis=1)/denominator[valid]
    if np.any(np.abs(values)>1+1e-10):raise ValueError('invalid cosine')
    values=np.clip(values,-1,1)
    return dict(valid_pairs=len(values),zero_norm_pairs=int((~valid).sum()),mean=float(values.mean()),
        quantiles=np.quantile(values,[.1,.5,.9]).tolist())


def main():
    d,_,_,_,_,_,_,_,_,_=checked()
    with np.load(ROOT/'transform.npz') as z:fields=z['field_ids']
    train=np.zeros(len(fields),bool);subjects=np.zeros(len(fields),np.int64)
    for i,refs in enumerate(d.text_refs):
        for _,row in refs:train[row]=d.arrays['patient_splits'][i]=='train';subjects[row]=d.arrays['subject_ids'][i]
    arrays={'raw':d.text_vectors,**{k:np.load(ROOT/f'{k}.float16.npy',mmap_mode='r') for k in ('centered','whitened')}}
    results={}
    for field in range(4):
        mask=train&(fields==field);raw=np.asarray(d.text_vectors[mask],dtype=np.float16)
        u,inverse,counts=np.unique(raw,axis=0,return_inverse=True,return_counts=True)
        cross=0
        for j in np.flatnonzero(counts>1):
            rows=inverse==j
            if len(np.unique(subjects[mask][rows]))>1:cross+=int(rows.sum())
        results[str(field)]=dict(train_vectors=len(raw),unique_exact_vectors=len(u),
            vectors_sharing_exact_embedding_across_patients=cross,
            exact_cosine={name:cosine_pairs(np.asarray(value[mask],np.float32),subjects[mask]) for name,value in arrays.items()})
    r=dict(status='exact_angle_and_duplicate_audit',fields=results,
        code_sha256=sha256(Path(__file__)),parent_manifest_sha256=sha256(ROOT/'manifest.json'),
        limitation='parent pair_cosine used dot products of near-unit float16 vectors; this supplement divides by exact norms, not an input/fit change')
    write_json(ROOT/'exact_angle_audit.json',r)
    print(json.dumps(r),flush=True)


if __name__=='__main__':main()
