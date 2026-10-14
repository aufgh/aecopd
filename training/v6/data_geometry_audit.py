"""训练输入差异及冻结BERT后处理派生审计；不输出原始文书。"""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked
from training.v6.text_geometry import fit_geometry,transform_rows
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_text_geometry_2026-10-13')
RESP=Path('patient_data/processed/v6_resp_micro_context_2026-10-09_r2')


def random_pair_cos(x,people):
    rng=np.random.default_rng(42);a=rng.integers(len(x),size=20000);b=rng.integers(len(x),size=20000)
    keep=people[a]!=people[b];a,b=a[keep][:10000],b[keep][:10000]
    values=(x[a]*x[b]).sum(1)
    return dict(pairs=len(a),mean=float(values.mean()),quantiles=np.quantile(values,[.1,.5,.9]).tolist())


def main():
    d,c,_,_,_,_,tune,_,source,_=checked()
    if ROOT.exists():raise FileExistsError('inspect existing derivation, do not overwrite')
    ROOT.mkdir(parents=True)
    n=len(d.text_vectors);fields=np.empty(n,np.int64);people=np.empty(n,np.int64);train=np.zeros(n,bool);weight=np.ones(n,np.float32)
    for i,refs in enumerate(d.text_refs):
        counts=Counter(f for f,_ in refs)
        for f,row in refs:
            fields[row]=f;people[row]=d.arrays['subject_ids'][i]
            train[row]=d.arrays['patient_splits'][i]=='train';weight[row]=1/counts[f]
    x=torch.as_tensor(np.asarray(d.text_vectors,dtype=np.float32),device='cuda')
    f=torch.as_tensor(fields,device='cuda');tr=torch.as_tensor(train,device='cuda');w=torch.as_tensor(weight,device='cuda')
    means,whitening,geometry=fit_geometry(x[tr],f[tr],w[tr])
    np.savez_compressed(ROOT/'transform.npz',means=means.cpu().numpy(),whitening=whitening.cpu().numpy(),field_ids=fields)
    derived={}
    for name,mat in [('centered',None),('whitened',whitening)]:
        out=transform_rows(x,f,means,mat).cpu().numpy().astype(np.float16)
        np.save(ROOT/f'{name}.float16.npy',out,allow_pickle=False)
        derived[name]=dict(path=str(ROOT/f'{name}.float16.npy'),sha256=sha256(ROOT/f'{name}.float16.npy'),shape=list(out.shape))
    pair={}
    arrays={'raw':np.asarray(d.text_vectors,np.float32),**{k:np.asarray(np.load(v['path'],mmap_mode='r'),np.float32) for k,v in derived.items()}}
    for field in range(4):
        use=train&(fields==field)
        pair[str(field)]={name:random_pair_cos(value[use],people[use]) for name,value in arrays.items()}
    ix=np.flatnonzero(d.arrays['patient_splits']=='train')
    clinical=np.concatenate([d.arrays['lab_axis_inputs'][ix].reshape(len(ix),-1),c.arrays['med_features'][ix].reshape(len(ix),-1),
        c.arrays['micro_values'][ix],c.arrays['micro_masks'][ix],c.arrays['support_values'][ix],c.arrays['support_masks'][ix]],1)
    signatures=np.round(clinical,3).astype(np.float32)
    unique,inv,counts=np.unique(signatures,axis=0,return_inverse=True,return_counts=True)
    repeated=counts[inv]>1
    cross=0
    for group in np.flatnonzero(counts>1):
        mask=inv==group
        if len(np.unique(d.arrays['subject_ids'][ix][mask]))>1:cross+=int(mask.sum())
    resp=json.loads((RESP/'manifest.json').read_text())
    with np.load(RESP/'resp_micro_context.npz') as z:
        hadms=z['hadm_ids'];raw=z['resp_numeric_raw'];masks=z['resp_numeric_masks']
    np.testing.assert_array_equal(hadms,d.arrays['hadm_ids'])
    low={name:dict(train_observed=int(masks[ix,j,0].sum()),train_missing=int((~masks[ix,j,0]).sum()),
        train_first_last_different=int(((raw[ix,j,0]!=raw[ix,j,3])&masks[ix,j,0]).sum()),
        input_current_best=False,warning='selected ICU anchors; not dense time series or discharge measurement')
        for j,name in enumerate(resp['resp_numeric_names'])}
    proof=dict(status='training_geometry_audited_derivatives_not_predictively_validated',data_provenance=d.provenance,
        source_sha256={p:sha256(Path(p)) for p in ('training/v6/data_geometry_audit.py','training/v6/text_geometry.py',
            'training/v6/early_stopping.py','training/tests/test_v6_data_geometry.py','docs/V6_DATA_ORGANIZATION_PLAN_2026-10-13.md')},
        transform_sha256=sha256(ROOT/'transform.npz'),derivatives=derived,geometry=geometry,pair_cosine=pair,
        training_vectors=int(train.sum()),training_admissions=len(ix),training_only_fit=True,
        weighting='equal total weight per observed admission-field; no outcome labels used',
        clinical_signature=dict(rounding_decimals=3,dimensions=clinical.shape[1],unique_signatures=len(unique),
            records_in_repeated_signatures=int(repeated.sum()),records_sharing_signature_across_patients=cross,
            excludes='H34, extra labs and text: this is not complete model input identity'),
        existing_low_frequency=low,resp_npz_sha256=sha256(RESP/'resp_micro_context.npz'),
        database_inventory=dict(status='not_verified_no_process_credentials',error='fe_sendauth: no password supplied'),
        confirmation_test_labels_not_analyzed=True)
    write_json(ROOT/'manifest.json',proof)
    write_json(ROOT/'README.md.json',dict(summary='Original B preserved; derived vectors same76974x768; train-only field transforms; not clinical validity proof'))
    print(json.dumps(dict(status=proof['status'],training_vectors=proof['training_vectors'],geometry=geometry,
        raw_pair_cosine={f:v['raw']['mean'] for f,v in pair.items()},clinical_signature=proof['clinical_signature'])),flush=True)


if __name__=='__main__':main()
