"""用XGBoost原生Booster重放，绕开2.1.3/sklearn1.6的标签接口冲突。"""
import json
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json
from training.v7 import two_fold_ensemble as run


def native_tree(model,x):
    probabilities=np.stack([model.predict(xgb.DMatrix(np.column_stack((x,np.full(len(x),j))))) for j in range(7)],1)
    return run.interval_cif(probabilities)


def verify():
    a,f,_=run.checked();r=run.ROOT;contract=json.loads((r/'contract.json').read_text())
    for p,h in contract['source_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('original training source changed')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    folds=run.patient_folds(a['subject_ids'],a['train']);train,tune=a['train'],a['tune'];saved=run.load_npz(r/'predictions.npz')
    outer=np.zeros_like(saved['outer']);maximum=0.
    for fold in (0,1):
        fit=train[folds[train]!=fold];hold=train[folds[train]==fold];folder=r/f'fold{fold}'
        if set(a['subject_ids'][fit])&set(a['subject_ids'][hold]):raise ValueError('patient leakage')
        prep=torch.load(folder/'preprocessing.pt',map_location='cpu',weights_only=False)
        x,_=run.preprocess(a,f,fit,saved=prep)
        cox=run.load_cox(folder/'cox.npz');tree=xgb.Booster();tree.load_model(folder/'tree.ubj')
        ck=torch.load(folder/'neural.pt',map_location='cuda',weights_only=True)
        neural=run.CompactNeural(ck['width']).cuda();neural.load_state_dict(ck['state_dict'])
        functions=(lambda v:run.predict_cumulative_incidence(cox,v,np.arange(366)),
                   lambda v:native_tree(tree,v),lambda v:run.neural_predict(neural,v))
        for j,name in enumerate(run.FAMILIES):
            source=run.load_npz(folder/f'{name}_predictions.npz')
            np.testing.assert_array_equal(source['hold'],hold)
            for key,positions in (('hold_curve',hold),('tune_curve',tune)):
                actual=functions[j](x[name][positions]);error=float(np.max(np.abs(actual-source[key])));maximum=max(maximum,error)
                np.testing.assert_allclose(actual,source[key],atol=1e-10,rtol=0)
                if key=='tune_curve':outer[j]+=actual/2
            np.testing.assert_array_equal(saved['oof'][j,np.searchsorted(train,hold)],source['hold_curve'])
        print(json.dumps(dict(fold_replayed=fold)),flush=True);del neural;torch.cuda.empty_cache()
    weights,value=run.common_weights(saved['oof'],a['times'][train],a['events'][train])
    np.testing.assert_allclose(weights,saved['weights'],atol=1e-10,rtol=0)
    np.testing.assert_allclose(outer,saved['outer'],atol=1e-10,rtol=0)
    weighted=np.einsum('m,mntk->ntk',weights,outer)
    np.testing.assert_allclose(weighted,saved['weighted'],atol=1e-10,rtol=0)
    np.testing.assert_allclose(run.temper(weighted,float(saved['temperature'])),saved['calibrated'],atol=1e-10,rtol=0)
    report=json.loads((r/'report.json').read_text());curves={name:saved['outer'][j] for j,name in enumerate(run.FAMILIES)}
    curves.update(equal=saved['outer'].mean(0),weighted=saved['weighted'],weighted_temperature=saved['calibrated'])
    for name,curve in curves.items():
        if run.score(a,tune,curve)!=report['metrics'][name]:raise ValueError('ensemble metrics differ')
        if np.max(np.abs(curve.sum(2)-1))>1e-8 or (np.diff(curve[:,:,1:],axis=1)<-1e-12).any():raise ValueError('invalid ensemble curve')
    write_json(r/'verification.json',dict(maximum_error=maximum,models_replayed=6,metrics_equal=True,
        weights_replayed=True,calibration_replayed=True,source_contract_equal=True,
        compatibility_reader='native XGBoost Booster; sklearn load error preserved',
        verification_source_sha256=sha256(Path(__file__)),report_sha256=sha256(r/'report.json')))
    print(json.dumps(dict(two_fold_models_verified=6,maximum_error=maximum)),flush=True)


if __name__=='__main__':verify()
