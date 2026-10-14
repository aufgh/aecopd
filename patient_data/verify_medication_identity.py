"""独立验证药物频率、剂量单位、成员/训练折尺度与落盘接口。"""
import argparse
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from patient_data.medication_identity import NAMES,UNITS,CHANNELS
from training.v5.numeric_preprocessing import robust_parameters


def verify(directory,state_dir):
    m=json.loads((directory/'manifest.json').read_text());norm=json.loads((directory/'normalization.json').read_text())
    if m['status']!='validated_medication_identity' or m['parent_state_manifest_sha256']!=sha256(state_dir/'manifest.json') or m['output_sha256']!=sha256(directory/'medication_identity.npz'):
        raise ValueError('source proof changed')
    with np.load(directory/'medication_identity.npz') as z:a={k:z[k] for k in z.files}
    with np.load(state_dir/'clinical_state.npz') as z:
        for key in ('hadm_ids','subject_ids','patient_splits'):
            if not np.array_equal(a[key],z[key]):raise ValueError('member/split differs')
    v=a['values_raw'];mask=a['value_masks'];f=a['features'];unit=a['dose_unit_id'];present=a['present'];train=a['patient_splits']=='train'
    if f.shape!=(19401,21,23) or not np.isfinite(f).all() or not np.isfinite(v).all() or np.any(v<0):raise ValueError('invalid medication tensor')
    if np.any(f[~present]!=0):raise ValueError('empty identity token not masked')
    executed=v[:,:,0]>0
    if np.any(v[:,:,1][executed]>v[:,:,0][executed]) or np.any(v[:,:,2][executed]>v[:,:,0][executed]):raise ValueError('count/day arithmetic impossible')
    np.testing.assert_allclose(v[:,:,3][executed],v[:,:,0][executed]/v[:,:,1][executed],rtol=1e-6)
    complete=mask[:,:,4]
    if np.any(unit[complete]==0) or np.any(unit[complete]==UNITS.index('mixed')) or np.any(v[:,:,4][complete]<v[:,:,5][complete]):
        raise ValueError('unknown/mixed/corrupt cumulative dose admitted')
    max_error=0.
    for j,drug in enumerate(NAMES):
        for k,name in enumerate(CHANNELS):
            for u in (range(len(UNITS)) if k in (4,5) else (-1,)):
                valid=mask[:,j,k] & (unit[:,j]==u if u>=0 else True)
                key=f'{drug}/{name}/{UNITS[u] if u>=0 else "count"}'
                if not valid.any():continue
                p=robust_parameters(np.log1p(v[train & valid,j,k]))
                if any(abs(p[n]-norm[key][n])>1e-7 for n in ('median','scale')):raise ValueError('normalization not train-only')
                expected=np.clip((np.log1p(v[valid,j,k])-p['median'])/p['scale'],-8,8)*present[valid,j]
                max_error=max(max_error,float(np.max(np.abs(expected-f[valid,j,k]))))
    if max_error>2e-6:raise ValueError('scaled values differ')
    proof=dict(status='independently_verified_medication_interface',manifest_sha256=sha256(directory/'manifest.json'),
        npz_sha256=sha256(directory/'medication_identity.npz'),raw_drug_name_dictionary_sha256=sha256(directory/'raw_drug_names.json'),
        normalization_sha256=sha256(directory/'normalization.json'),max_scaled_value_error=max_error,
        train_only_normalization_reproduced=True,frequency_arithmetic_checked=True,mixed_unit_doses_excluded=True,
        limits=['This verifies numerical/source rules, not physician-adjudicated drug mapping sensitivity.'])
    (directory/'independent_verification.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path('patient_data/processed/v6_medication_identity_2026-10-12'))
    p.add_argument('--state-dir',type=Path,default=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2'));a=p.parse_args();verify(a.directory,a.state_dir)
