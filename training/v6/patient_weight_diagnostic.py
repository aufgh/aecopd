"""只用训练患者住院数量诊断损失权重；不读取验证/测试结局，不训练。"""
import json
from pathlib import Path
import numpy as np
from training.v6.regularization_audit import checked
from training.v6.train import write_json

ROOT=Path('training_runs/v6_patient_weight_diagnostic_2026-10-13')


def run():
    d,_,_,_,_,_,_,_,_,_=checked()
    selected=d.arrays['patient_splits']=='train'
    people,count=np.unique(d.arrays['subject_ids'][selected],return_counts=True)
    descending=np.sort(count)[::-1]
    shares={}
    for percentile in (1,5,10,20):
        k=max(1,int(np.ceil(len(count)*percentile/100)))
        shares[str(percentile)]=dict(patients=k,admissions=int(descending[:k].sum()),
            admission_loss_share=float(descending[:k].sum()/count.sum()))
    schemes={}
    for power in (0.,.5,1.):
        mass=count.astype(float)**(1-power)
        schemes[str(power)]=dict(patient_mass_ratio_max_min=float(mass.max()/mass.min()),
            effective_patients=float(mass.sum()**2/(mass**2).sum()),
            definition=f'per-admission loss weight proportional to n_patient^(-{power})')
    ROOT.mkdir(exist_ok=True)
    r=dict(status='training_input_weight_diagnostic_not_model_results',patients=len(people),admissions=int(count.sum()),
        admission_count_quantiles=np.quantile(count,[0,.5,.9,.99,1]).tolist(),top_patient_shares=shares,
        candidate_weight_schemes=schemes,no_validation_or_test_labels_analyzed=True,no_model_training=True,goal_achieved=False)
    write_json(ROOT/'diagnostic.json',r);print(json.dumps(r),flush=True)


if __name__=='__main__':run()
