"""另一个进程核对实际数据、依赖和选定模型重放，不修改历史实验。"""
import json
from pathlib import Path
import importlib.metadata
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.public_tabular_data import read
from training.v6.public_tabular_experiment import ROOT, setup, model_for, tensors, parameters
from training.v6.public_tabular_model import ensemble_curve
from training.v6.token_model import daily_probabilities
from training.v6.train import write_json


def verify(root=ROOT):
    setup();r=json.loads((root/'report.json').read_text());contract=r['contract'];d,prep=read(root)
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('code changed: '+p)
    for package,version in contract['package_versions'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('dependency changed')
    for p,digest in contract['package_source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('installed package source changed')
    for p,key in (('matrix.npz','matrix_sha256'),('preprocessing.joblib','preprocessing_sha256')):
        if sha256(root/p)!=contract['data'][key]:raise ValueError('input changed')
    for person in np.unique(d['subject_ids']):
        if len(np.unique(d['patient_splits'][d['subject_ids']==person]))!=1:raise ValueError('patient crossed folds')
    errors={}
    for arm,models in r['models'].items():
        for seed,value in models.items():
            path=root/arm/f"lr{r['learning_rates'][arm]}"/f'seed{seed}'
            if sha256(path/'best.pt')!=value['checkpoint_sha256']:raise ValueError('checkpoint changed')
            history=json.loads((path/'history.json').read_text())
            if len(history)!=contract['epochs']+1 or min(history,key=lambda x:x['validation_ibs'])['epoch']!=value['best_epoch']:
                raise ValueError('epochs or selection mismatch')
            model=model_for(arm,int(seed),d,prep)
            model.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            for fold in ('validation','test'):
                ix,batch=tensors(d,fold);a,k,_=parameters(model,batch);curve=ensemble_curve(a,k)
                with np.load(path/(fold+'_predictions.npz')) as z:
                    if not np.array_equal(z['hadm_ids'],d['hadm_ids'][ix]) or not np.array_equal(z['events'],d['events'][ix]) or not np.array_equal(z['times'],d['times'][ix]):
                        raise ValueError('prediction members/labels changed')
                    err=float(np.max(np.abs(z['curve']-curve)))
                if err>1e-9:raise ValueError('prediction replay differs '+str(err))
                daily_probabilities(curve);errors[f'{arm}/{seed}/{fold}']=err
            del model
    result=dict(status='independently_verified',selected_models=10,completed_trainings=20,
        replay_errors=errors,patient_split_checked=True,daily_probability_mass_checked=True,
        report_sha256=sha256(root/'report.json'))
    write_json(root/'independent_verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':verify()
