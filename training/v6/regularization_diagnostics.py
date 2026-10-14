"""确认集校准与既往次数层内增量；不在层内再次选择候选。"""
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked
from training.v6.regularization_multiseed import ROOT,SEEDS
from training.v6.evaluation import metrics,public_metrics,paired_interval,calibration_bins
from training.v6.horizon_objective import calibration_fit
from training.v6.train import write_json


def history_counts(raw):
    if raw.ndim!=2 or raw.shape[1]!=6 or not np.isfinite(raw).all():raise ValueError('six raw history fields required')
    x=np.expm1(raw[:,2])
    if np.any(x<0) or not np.allclose(x,np.rint(x),atol=1e-4,rtol=0):raise ValueError('not raw log1p prior admissions')
    return np.rint(x).astype(np.int64)


def calibration(times,events,curve,cause):
    result={}
    for day in (30,90,180,365):
        risk=curve[:,:,cause]
        bins=calibration_bins(times,events,risk,day=day,cause=cause)
        y=((events==cause)&(times<=day)).astype(float)
        result[str(day)]=dict(observed=float(y.mean()),mean_predicted=float(risk[:,day].mean()),
            absolute_mean_error=float(abs(risk[:,day].mean()-y.mean())),
            ece10=float(sum(x['n']*abs(x['predicted']-x['observed']) for x in bins)/len(times)),
            logistic=calibration_fit(y,risk[:,day]),bins=bins)
    return result


def run():
    report=json.loads((ROOT/'report.json').read_text())
    proof=json.loads((ROOT/'independent_verification.json').read_text())
    if proof['report_sha256']!=sha256(ROOT/'report.json'):raise ValueError('replay proof stale')
    d,_,_,_,_,_,_,confirm,_,_=checked()
    counts=history_counts(d.history_raw[confirm])
    strata=dict(prior0=counts==0,prior1_2=(counts>=1)&(counts<=2),prior3plus=counts>=3)
    candidates=[x['name'] for x in report['contract']['trials'] if x['name']!='v6_control']
    models={}
    for seed in SEEDS:
        models[str(seed)]={};curves={}
        for name in ['v6_control',*candidates,'official_V6']:
            with np.load(ROOT/f'seed{seed}'/name/'confirmation_predictions.npz') as z:
                for key,expected in (('hadm_ids',d.arrays['hadm_ids'][confirm]),('times',d.times[confirm]),('events',d.events[confirm])):
                    np.testing.assert_array_equal(z[key],expected)
                curve=z['curve'].copy()
            curves[name]=curve
            groups={}
            for label,mask in strata.items():
                if not mask.any():groups[label]=dict(status='empty');continue
                groups[label]=public_metrics(metrics(d.times[confirm][mask],d.events[confirm][mask],curve[mask,:,1],ordering=True))
            models[str(seed)][name]=dict(calibration={str(c):calibration(d.times[confirm],d.events[confirm],curve,c) for c in (1,2)},strata=groups)
        for name in candidates:
            pairs={}
            for reference in ('v6_control','official_V6'):
                per_group={}
                for label,mask in strata.items():
                    y=(d.events[confirm][mask]==1)&(d.times[confirm][mask]<=90)
                    if not mask.any() or min(int(y.sum()),len(y)-int(y.sum()))<10:
                        per_group[label]=dict(status='insufficient cases/controls for paired AUC; not zero effect')
                        continue
                    per_group[label]=paired_interval(d.arrays['subject_ids'][confirm][mask],d.times[confirm][mask],d.events[confirm][mask],
                        curves[name][mask,:,1],curves[reference][mask,:,1],seed=seed,replicates=200)
                pairs[reference]=per_group
            models[str(seed)][name]['paired_strata']=pairs
    write_json(ROOT/'calibration_and_history_strata.json',dict(report_sha256=sha256(ROOT/'report.json'),models=models,
        definition='prior completed retained COPD-coded admissions, raw log1p field2, not strict count and not normalized values',
        strata_admissions={k:int(v.sum()) for k,v in strata.items()},test_not_evaluated=True,goal_achieved=False,
        limitations=['Sparse strata AUC may be unavailable.','Group differences not a causal interpretation.','No candidate reselection by strata.']))
    print(json.dumps(dict(status='confirmation_calibration_and_history_strata_written',goal_achieved=False)),flush=True)


if __name__=='__main__':run()
