"""已锁定 V7/旧参照的 AP、校准与患者配对 F1 读出；0新拟合。"""
import json
from pathlib import Path

import numpy as np

from patient_data.export_mimiciv import sha256
from training.v6.horizon_objective import binary_metrics, ranked_curves
from training.v6.train import write_json
from training.v6.unified_longitudinal import best_f1, f1_at
from training.v7.losses import HORIZONS

ROOT = Path('training_runs/v7_first_2026-10-15_r2')


def calculate():
    proof = json.loads((ROOT/'verification.json').read_text())
    if proof['maximum_error'] != 0 or proof['report_sha256'] != sha256(ROOT/'report.json'):
        raise ValueError('V7 replay proof changed')
    paths = {'nll_control': ROOT/'nll_control/tuning_predictions.npz',
             'recommended': ROOT/'recommended/tuning_predictions.npz',
             'v6_original': Path('training_runs/v6_unified_longitudinal_f1_2026-10-14_r2/original/tuning_predictions.npz'),
             'v6_history_unified': Path('training_runs/v6_unified_longitudinal_f1_2026-10-14_r2/history_unified/tuning_predictions.npz'),
             'v6_symmetric': Path('training_runs/v6_symmetric_context_2026-10-14/symmetric/tuning_predictions.npz')}
    arrays = {}
    for name,path in paths.items():
        with np.load(path) as z:
            arrays[name] = {k:z[k] for k in ('subject_ids','hadm_ids','times','events','curve')}
        if len(arrays)>1:
            for k in ('subject_ids','hadm_ids','times','events'):
                np.testing.assert_array_equal(arrays[name][k],arrays['nll_control'][k])
    metrics = {}
    thresholds = {}
    for name,a in arrays.items():
        metrics[name]={}
        for label,cause in (('readmission',1),('death',2)):
            points={}
            for day in HORIZONS:
                y = ((a['events']==cause)&(a['times']<=day)).astype(np.int8)
                p=a['curve'][:,day,cause]
                point=binary_metrics(y,p)
                # 复用统计函数，但不沿用其固定90日的字段名。
                for old,new in (('auc90','auc'),('brier90','brier'),('logloss90','logloss')):
                    point[new]=point.pop(old)
                point['tuning_best_f1']=best_f1(y.astype(bool),p)
                points[str(day)]=point
            metrics[name][label]=points
        thresholds[name]=metrics[name]['readmission']['365']['tuning_best_f1']['threshold']
    common=arrays['nll_control'];subjects=common['subject_ids']
    people,inverse=np.unique(subjects,return_inverse=True)
    groups=[np.flatnonzero(inverse==i) for i in range(len(people))]
    annual=((common['events']==1)&(common['times']<=365))
    comparisons={}
    pairs=[(arm,ref) for arm in ('nll_control','recommended') for ref in ('v6_original','v6_history_unified','v6_symmetric')]
    pairs.append(('recommended','nll_control'))
    for left,right in pairs:
        pa,pb=arrays[left]['curve'][:,365,1],arrays[right]['curve'][:,365,1]
        def difference(ix):
            y=annual[ix]
            return (f1_at(y,pa[ix],thresholds[left])['f1']-f1_at(y,pb[ix],thresholds[right])['f1'],
                    ranked_curves(y.astype(np.int8),pa[ix])['ap']-ranked_curves(y.astype(np.int8),pb[ix])['ap'])
        rng=np.random.default_rng(42);samples=[]
        for _ in range(200):
            ix=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
            samples.append(difference(ix))
        values=np.asarray(samples)
        point=difference(np.arange(len(annual)))
        comparisons[left+'_minus_'+right]=dict(annual_f1_difference=point[0],annual_f1_95ci=np.quantile(values[:,0],[.025,.975]).tolist(),
            annual_ap_difference=point[1],annual_ap_95ci=np.quantile(values[:,1],[.025,.975]).tolist(),
            patients=len(people),replicates=200,thresholds_fixed_per_selected_model=True,
            limits='conditional on tuning-selected checkpoints/thresholds; single seed; exposed fold; not unbiased validation')
    source_paths=[Path(__file__),Path('training/v6/horizon_objective.py'),Path('training/v6/unified_longitudinal.py'),
                  ROOT/'report.json',ROOT/'verification.json',*paths.values()]
    contract=dict(source_sha256={str(p):sha256(p) for p in source_paths},new_training=0,new_risk_calibration=0,
                  calibration_slope_for_descriptive_evaluation_only=True,confirmation_or_test_used=False)
    return dict(contract=contract,metrics=metrics,annual_patient_paired=comparisons)


if __name__=='__main__':
    result=calculate();target=ROOT/'readout.json'
    if target.exists():
        saved=json.loads(target.read_text())
        if saved!=result:raise ValueError('independent readout differs')
        write_json(ROOT/'readout_verification.json',dict(maximum_error=0,source_metrics_equal=True,
            readout_sha256=sha256(target),new_training=0,new_risk_calibration=0))
        print('V7 readout replay exact',flush=True)
    else:
        write_json(target,result);print('V7 readout saved',flush=True)
