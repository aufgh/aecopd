"""第二批锁定候选的确认校准与次数层内增量；不再次选模。"""
import json
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.task_optimization_experiment import ROOT,SEEDS
from training.v6.task_optimization_audit import inputs
from training.v6.regularization_diagnostics import history_counts,calibration
from training.v6.evaluation import metrics,public_metrics,paired_interval
from training.v6.train import write_json


def analyze(times,events,subjects,counts,curves,*,seed,comparators=('control','official_V6')):
    if counts.shape!=times.shape or subjects.shape!=times.shape:raise ValueError('history/patient rows differ')
    strata=dict(prior0=counts==0,prior1_2=(counts>=1)&(counts<=2),prior3plus=counts>=3)
    models={}
    for name,curve in curves.items():
        groups={}
        for label,mask in strata.items():
            if not mask.any():groups[label]=dict(status='empty');continue
            groups[label]=public_metrics(metrics(times[mask],events[mask],curve[mask,:,1],ordering=True))
        models[name]=dict(calibration={str(c):calibration(times,events,curve,c) for c in (1,2)},strata=groups)
    if not set(comparators).issubset(curves):raise ValueError('required comparator absent')
    for name in curves.keys()-set(comparators):
        models[name]['paired_strata']={}
        for ref in comparators:
            groups={}
            for label,mask in strata.items():
                y=(events[mask]==1)&(times[mask]<=90)
                if not mask.any() or min(int(y.sum()),len(y)-int(y.sum()))<10:
                    groups[label]=dict(status='insufficient cases/controls; not zero effect')
                else:groups[label]=paired_interval(subjects[mask],times[mask],events[mask],curves[name][mask,:,1],
                    curves[ref][mask,:,1],replicates=200,seed=seed)
            models[name]['paired_strata'][ref]=groups
    return models,{label:int(mask.sum()) for label,mask in strata.items()}


def run():
    r=json.loads((ROOT/'confirmation_report.json').read_text())
    proof=json.loads((ROOT/'confirmation_verification.json').read_text())
    if proof['report_sha256']!=sha256(ROOT/'confirmation_report.json'):raise ValueError('confirmation proof stale')
    (d,_,_,_,_,_,_,ix,_,_),_=inputs()
    counts=history_counts(d.history_raw[ix]);result={}
    for seed in SEEDS:
        curves={}
        for name in ['control',*r['lock']['nominated'],'official_V6']:
            with np.load(ROOT/f'seed{seed}'/name/'confirmation_predictions.npz') as z:
                for key,expected in (('hadm_ids',d.arrays['hadm_ids'][ix]),('subject_ids',d.arrays['subject_ids'][ix]),
                    ('times',d.times[ix]),('events',d.events[ix])):
                    np.testing.assert_array_equal(z[key],expected)
                curves[name]=z['curve'].copy()
        result[str(seed)],strata_counts=analyze(d.times[ix],d.events[ix],d.arrays['subject_ids'][ix],counts,curves,seed=seed)
    write_json(ROOT/'calibration_and_history_strata.json',dict(models=result,strata_admissions=strata_counts,
        confirmation_report_sha256=sha256(ROOT/'confirmation_report.json'),goal_achieved=False,test_not_evaluated=True,
        definition='previous completed retained COPD-coded admissions; raw log1p field2; no normalized-value threshold',
        limits=['Previously exposed validation, not blind.','Sparse strata may have unavailable AUC.',
        'No reselection based on strata/calibration.','Three seeds reuse two fixed baseline sources.']))
    print(json.dumps(dict(status='locked_confirmation_diagnostics_written',goal_achieved=False)),flush=True)


if __name__=='__main__':run()
