"""固定OOF权重，仅把三个成员重拟合到完整训练规模。"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import xgboost as xgb
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json
from training.v7 import two_fold_ensemble as base
from training.v7.two_fold_replay import native_tree

ROOT=Path('training_runs/v7_sequential_research_2026-10-15/full_refit')
PROTOCOL=Path('docs/V7_ENSEMBLE_FULL_REFIT_PROTOCOL_2026-10-15.md')


def setup():
    a,f,_=base.checked();r=json.loads((base.ROOT/'report.json').read_text());proof=json.loads((base.ROOT/'verification.json').read_text())
    if proof['report_sha256']!=sha256(base.ROOT/'report.json'):raise ValueError('twofold proof stale')
    paths=(Path(__file__),PROTOCOL,base.ROOT/'contract.json',base.ROOT/'report.json',base.ROOT/'verification.json',
           Path('training/v7/two_fold_ensemble.py'),Path('training/v7/two_fold_replay.py'))
    contract=dict(source_sha256={str(p):sha256(p) for p in paths},weights=r['weights'],temperature=r['temperature'],
        single_full_training_refit=True,hyperparameters_unchanged=True)
    return a,f,r,contract


def train():
    if ROOT.exists():raise FileExistsError('full refit exists')
    started=time.perf_counter();a,f,parent,contract=setup();ROOT.mkdir();write_json(ROOT/'contract.json',contract)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    train,tune=a['train'],a['tune'];x,prep=base.preprocess(a,f,train);torch.save(prep,ROOT/'preprocessing.pt')
    tick=time.perf_counter();cox=base.fit_competing_cox(x['cox'][train],a['times'][train],a['events'][train])
    if not cox.readmission.converged or not cox.death.converged:raise ValueError('full Cox did not converge')
    base.save_cox(cox,ROOT/'cox.npz');curves=[base.predict_cumulative_incidence(cox,x['cox'][tune],np.arange(366))]
    print(json.dumps(dict(full_family='cox',seconds=time.perf_counter()-tick)),flush=True)
    tick=time.perf_counter();xx,yy=base.period_rows(x['tree'],a['times'],a['events'],train)
    tree=base.XGBClassifier(n_estimators=100,max_depth=3,learning_rate=.05,min_child_weight=10,reg_lambda=10,
        objective='multi:softprob',num_class=3,tree_method='hist',n_jobs=4,random_state=42)
    tree.fit(xx,yy);tree.save_model(ROOT/'tree.ubj');curves.append(base.tree_predict(tree,x['tree'][tune]))
    print(json.dumps(dict(full_family='tree',seconds=time.perf_counter()-tick)),flush=True)
    tick=time.perf_counter();neural=base.fit_neural(x['neural'],a,train)
    torch.save(dict(state_dict=neural.state_dict(),width=x['neural'].shape[1]),ROOT/'neural.pt');curves.append(base.neural_predict(neural,x['neural'][tune]))
    print(json.dumps(dict(full_family='neural',seconds=time.perf_counter()-tick)),flush=True)
    curves=np.stack(curves);weights=np.asarray(parent['weights']);weighted=np.einsum('m,mntk->ntk',weights,curves)
    calibrated=base.temper(weighted,parent['temperature'])
    outputs={name:curves[j] for j,name in enumerate(base.FAMILIES)}
    outputs.update(equal=curves.mean(0),weighted=weighted,weighted_temperature=calibrated)
    scores={name:base.score(a,tune,curve) for name,curve in outputs.items()}
    np.savez_compressed(ROOT/'predictions.npz',tune=tune,**outputs)
    write_json(ROOT/'report.json',dict(metrics=scores,weights=parent['weights'],temperature=parent['temperature'],
        seconds=time.perf_counter()-started,default_replaced=False,
        artifact_sha256={p.name:sha256(p) for p in ROOT.iterdir() if p.suffix in ('.pt','.npz','.ubj')}))


def verify():
    a,f,parent,current=setup();saved=json.loads((ROOT/'contract.json').read_text())
    if current!=saved:raise ValueError('full refit source differs')
    report=json.loads((ROOT/'report.json').read_text())
    for name,h in report['artifact_sha256'].items():
        if sha256(ROOT/name)!=h:raise ValueError('full refit artifact changed')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    prep=torch.load(ROOT/'preprocessing.pt',map_location='cpu',weights_only=False);x,_=base.preprocess(a,f,a['train'],saved=prep);tune=a['tune']
    cox=base.load_cox(ROOT/'cox.npz');tree=xgb.Booster();tree.load_model(ROOT/'tree.ubj')
    ck=torch.load(ROOT/'neural.pt',map_location='cuda',weights_only=True);neural=base.CompactNeural(ck['width']).cuda();neural.load_state_dict(ck['state_dict'])
    curves=np.stack((base.predict_cumulative_incidence(cox,x['cox'][tune],np.arange(366)),native_tree(tree,x['tree'][tune]),base.neural_predict(neural,x['neural'][tune])))
    weighted=np.einsum('m,mntk->ntk',np.asarray(parent['weights']),curves)
    outputs={name:curves[j] for j,name in enumerate(base.FAMILIES)}
    outputs.update(equal=curves.mean(0),weighted=weighted,weighted_temperature=base.temper(weighted,parent['temperature']))
    saved_predictions=base.load_npz(ROOT/'predictions.npz');maximum=0.
    for name,curve in outputs.items():
        maximum=max(maximum,float(np.max(np.abs(curve-saved_predictions[name]))))
        np.testing.assert_allclose(curve,saved_predictions[name],atol=1e-10,rtol=0)
        if base.score(a,tune,curve)!=report['metrics'][name]:raise ValueError('full refit metric differs')
    write_json(ROOT/'verification.json',dict(maximum_error=maximum,models_replayed=3,metrics_equal=True,report_sha256=sha256(ROOT/'report.json')))
    print(json.dumps(dict(full_refit_replayed=True,maximum_error=maximum)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('train','verify'),required=True)
    train() if p.parse_args().mode=='train' else verify()
