"""只读定位时钟插值误差，不重新训练、不重选候选或放宽容差。"""
import json
import numpy as np
from pathlib import Path
from training.v6.clock_audit import inputs,load_head
from training.v6.clock_experiment import ROOT,features,extract,dense_teacher,validate
from training.v6.warm_residual_audit import teacher
from training.v6.module_surgery_experiment import batches
from training.v6.horizon_objective import restore
from training.v6.anchored_clock import clock_curve,anchor_dense_to_daily
from training.v6.risk import predict_cif
from training.v6.train import write_json


def run():
    (d,c,h,ext,_,bins,ix,_,search,_),_=inputs()
    base=restore(Path(search['input_config']['parent']),42,'survival_base',h,.1)
    model=teacher(base,42,bins);data=extract(model,batches(d,c,ext,ix));del model
    grid,dense,daily,_=dense_teacher(data)
    head,_,_=load_head(42,'product_clock')
    _,clocks,coarse=validate(head,features(data,'product_clock'),data,grid,dense)
    a,k=data['amplitude'].cpu().numpy(),data['shape'].cpu().numpy()
    def curve(step,indices):
        g=np.unique(np.r_[np.arange(0,365.001,step),np.geomspace(1e-10,.5,80)])
        raw=predict_cif(a[indices],k[indices],g,step_days=step)
        aligned=anchor_dense_to_daily(raw,g,daily[indices])
        return clock_curve(aligned,g,clocks[indices],np.arange(366))
    fine=curve(.125,np.arange(len(ix)))
    error=np.abs(coarse-fine);point=np.unravel_index(np.argmax(error),error.shape)
    row,day,cause=map(int,point)
    subset=np.unique(np.r_[row,np.argmax(clocks,axis=0),np.argmin(clocks,axis=0)])
    comparisons=[];previous=fine[subset]
    for step in (.0625,.03125,.015625):
        current=curve(step,subset)
        comparisons.append(dict(step=step,maximum_probability_difference=float(np.abs(current-previous).max())))
        previous=current
    report=dict(status='diagnostic_only',largest_error_point=dict(tuning_row=row,day=day,cause=cause,
        error=float(error[point]),teacher_shape=k[row].tolist(),teacher_amplitude=a[row].tolist(),log_clock=clocks[row].tolist()),
        refined_subset_rows=subset.tolist(),successive_refinement=comparisons,selection_unchanged=True,goal_achieved=False)
    write_json(ROOT/'grid_diagnostic.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':run()
