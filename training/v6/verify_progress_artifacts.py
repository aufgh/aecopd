"""本轮两项试验的全曲线/排序/概率评价复算，不重训、不改报告。"""
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.evaluation import metrics,public_metrics
from training.v6.horizon_objective import labels90,binary_metrics
from training.v6.train import write_json


def difference(a,b):
    if isinstance(a,dict):
        if a.keys()!=b.keys(): raise ValueError('metric keys differ')
        return max((difference(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,list):
        if len(a)!=len(b): raise ValueError('metric lengths differ')
        return max((difference(x,y) for x,y in zip(a,b,strict=True)),default=0.)
    if a is None or isinstance(a,(str,bool)):
        if a!=b: raise ValueError('metric metadata differs')
        return 0.
    return abs(a-b)


def verify():
    result={}
    for name in ('v6_saved_section_comparison_2026-10-11','v6_frozen_augmentation_2026-10-11'):
        root=Path('training_runs')/name
        report=json.loads((root/'report.json').read_text())
        proof=json.loads((root/'independent_verification.json').read_text())
        if proof['report_sha256']!=sha256(root/'report.json'): raise ValueError('stale independent proof')
        if 'selection' in report and report['selection']!=json.loads((root/'selection.json').read_text()):
            raise ValueError('selection record changed')
        folds=0; max_error=0.; max_mass=0.
        for seed,models in report['models'].items():
            for arm,model in models.items():
                for f,recorded in model['folds'].items():
                    with np.load(root/f'seed{seed}'/arm/(f+'_predictions.npz')) as z:
                        y=labels90(z['times'],z['events']); curve=z['curve']
                        if not np.array_equal(z['y90'],y): raise ValueError('strict label encoding differs')
                        if (not np.isfinite(curve).all() or (curve<-1e-10).any() or (curve>1+1e-10).any()
                            or (np.diff(curve[:,:,1:],axis=1)<-1e-10).any() or (np.diff(curve[:,:,0],axis=1)>1e-10).any()
                            or not np.array_equal(curve[:,0],np.tile([1.,0.,0.],(len(curve),1)))
                            or not np.allclose(curve[:,90,1],z['risk90'],atol=1e-8)):
                            raise ValueError('curve boundary/monotonicity failed')
                        max_mass=max(max_mass,float(np.abs(curve.sum(2)-1).max()))
                        actual=binary_metrics(y,z['risk90'])
                        for k,v in actual.items(): max_error=max(max_error,difference(v,recorded[k]))
                        for key,cause in (('survival_readmission',1),('survival_death',2)):
                            actual=public_metrics(metrics(z['times'],z['events'],curve[:,:,cause],cause=cause,ordering=True))
                            max_error=max(max_error,difference(actual,recorded[key]))
                        folds+=1
        if folds!=16 or max_error>1e-7 or max_mass>1e-9: raise ValueError('full report recomputation failed')
        result[name]=dict(folds=folds,max_metric_error=max_error,max_probability_sum_error=max_mass,
            report_sha256=sha256(root/'report.json'),proof_sha256=sha256(root/'independent_verification.json'))
    output=Path('training_runs/v6_clinical_increment_evidence_2026-10-11/progress_artifact_verification.json')
    if output.exists(): raise FileExistsError(output)
    write_json(output,dict(status='all32_new_trial_folds_complete_metrics_and_curves_recomputed',checks=result,
        code_sha256=sha256(Path(__file__))))
    print(json.dumps(result),flush=True)


if __name__=='__main__': verify()
