"""简化数学例子：分类训练偏好的分数，不自动等于真实发生率。"""
import json
from pathlib import Path
import numpy as np

from patient_data.export_mimiciv import sha256

ROOT=Path('training_runs/v7_loss_comparison_2026-10-15')


def expected_focal(p,rate,alpha=.75,gamma=2.,smoothing=.05):
    q=(1-smoothing)*rate+smoothing/2
    return -alpha*q*(1-p)**gamma*np.log(p)-(1-alpha)*(1-q)*p**gamma*np.log1p(-p)


def minimum(rate,alpha=.75,gamma=2.,smoothing=.05):
    low,high=1e-9,1-1e-9
    for _ in range(100):
        left=low+(high-low)/3;right=high-(high-low)/3
        if expected_focal(left,rate,alpha,gamma,smoothing)<expected_focal(right,rate,alpha,gamma,smoothing):
            high=right
        else:
            low=left
    return (low+high)/2


def calculate():
    if abs(minimum(.5,.5,0.,0.)-.5)>1e-7 or abs(minimum(.1,.5,0.,0.)-.1)>1e-7:
        raise ValueError('scalar optimizer sanity check failed')
    rows=[]
    for rate in (0.,.01,.05,.1,.2,.5):
        p=minimum(rate)
        rows.append(dict(true_rate=rate,ordinary_binary_cross_entropy_optimum=rate,
            focal_gamma2_alpha075_smoothing005_optimum=p,expected_focal_at_optimum=float(expected_focal(p,rate)),
            expected_focal_at_true_rate=float(expected_focal(np.clip(rate,1e-9,1-1e-9),rate))))
    return dict(rows=rows,formula='q=.95*rate+.025; E(loss)=-.75*q*(1-p)^2*log(p)-.25*(1-q)*p^2*log(1-p)',
        source_sha256={str(Path(__file__)):sha256(Path(__file__))},new_patient_model_training=0,
        scope='synthetic constant-score binary example; not patient predictions; not proof that a neural model reaches these optima',
        implication='class weighting, focal modulation and smoothing alter the optimum score; risk calibration must be evaluated separately')


if __name__=='__main__':
    result=calculate();target=ROOT/'probability_probe.json'
    if target.exists():
        if json.loads(target.read_text())!=result:raise ValueError('probability probe replay differs')
        (ROOT/'probability_probe_verification.json').write_text(json.dumps(dict(maximum_error=0,
            probe_sha256=sha256(target),new_patient_model_training=0),indent=2),encoding='utf-8')
        print('概率数学例子独立重算一致')
    else:
        target.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result['rows'],ensure_ascii=False))
