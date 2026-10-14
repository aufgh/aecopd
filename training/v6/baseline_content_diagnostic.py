"""固定调参折上的H34内容核算与原V6临床增量；不训练/改队列或选新模型。"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.module_surgery_experiment import batches,parameters,summary,save
from training.v6.module_surgery import probability_curve
from training.v6.horizon_objective import restore
from training.v6.warm_residual_audit import teacher
from training.v6.data import CURRENT_NAMES
from training.v6.evaluation import paired_interval
from training.v6.task_optimization_audit import check_curve,check_saved
from training.v6.train import write_json

ROOT=Path('training_runs/v6_baseline_content_diagnostic_2026-10-13')


class BaselineView(nn.Module):
    def __init__(self,baseline):super().__init__();self.baseline=baseline
    def forward(self,history,batch):return self.baseline(history)


def run():
    d,c,h,ext,_,bins,ix,_,search,_=checked();cache=batches(d,c,ext,ix)
    inputs=torch.cat([batch[1] for batch in cache]).cpu().numpy()
    np.testing.assert_array_equal(inputs[:,:6],d.history[ix])
    np.testing.assert_array_equal(inputs[:,6:20],np.where(d.current_masks[ix],d.current[ix],0))
    np.testing.assert_array_equal(inputs[:,20:],d.current_masks[ix])
    groups=dict(demographics=[0,1],past_frequency_and_gap=[2,3,4,5],
        current_diagnoses=[6+i for i,k in enumerate(CURRENT_NAMES) if not k.startswith('disposition_')],
        current_disposition=[6+i for i,k in enumerate(CURRENT_NAMES) if k.startswith('disposition_')],
        current_availability=list(range(20,34)))
    if sorted(sum(groups.values(),[]))!=list(range(34)):raise ValueError('H34 decomposition omitted/duplicated a field')
    ROOT.mkdir(exist_ok=True);results={};errors={}
    for seed in (42,7):
        base=restore(Path(search['input_config']['parent']),seed,'survival_base',h,.1)
        model=teacher(base,seed,bins).eval();views=dict(H34=BaselineView(model.baseline),original_V6=model);curves={};metrics={}
        for name,view in views.items():
            a,k,nll=parameters(view,cache);curve=probability_curve(a,k);check_curve(curve)
            path=ROOT/f'seed{seed}'/name/'tuning_predictions.npz';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():errors[f'{seed}/{name}']=check_saved(path,d,ix,a,k,curve)
            else:save(path,d,ix,a,k,curve);errors[f'{seed}/{name}']=None
            curves[name]=curve;metrics[name]=summary(curve,d,ix,nll)
        weight=model.baseline.amplitude.weight.detach().cpu().numpy().astype(float)
        bias=model.baseline.amplitude.bias.detach().cpu().numpy().astype(float)
        components={name:inputs[:,columns].astype(float)@weight[:,columns].T for name,columns in groups.items()}
        reconstructed=bias+sum(components.values())
        with np.load(ROOT/f'seed{seed}/H34/tuning_predictions.npz') as z:
            np.testing.assert_allclose(np.clip(reconstructed,-10,6),z['log_amplitude'],atol=2e-5,rtol=1e-6)
        contribution={name:dict(log_amplitude_std=value.std(0).tolist(),
            log_amplitude_quantiles=np.quantile(value,[.1,.5,.9],axis=0).tolist()) for name,value in components.items()}
        results[str(seed)]=dict(metrics=metrics,linear_components=contribution,
            clinical_paired_increment=paired_interval(d.arrays['subject_ids'][ix],d.times[ix],d.events[ix],
                curves['original_V6'][:,:,1],curves['H34'][:,:,1],replicates=200,seed=seed),
            teacher_checkpoint_sha256=sha256(REFERENCE/f'seed{seed}/control/best.pt'))
        del model
    report=dict(status='existing_baseline_content_measured',models=results,fields_by_group=groups,admissions=len(ix),
        fold='fixed previously exposed tuning validation only',goal_achieved=False,
        limits=['H34 is history plus current diagnoses/disposition, not pure history.',
            'Correlated group contribution variances do not sum and are not causal attribution.',
            'Only two actual teacher sources, not three independent cohorts.',
            'No new model selection or confirmation/test evaluation.'])
    write_json(ROOT/'report.json',report)
    complete=all(x is not None for x in errors.values())
    write_json(ROOT/'verification.json',dict(status='baseline_content_independently_replayed' if complete else 'baseline_content_saved_pending_replay',
        errors=errors,maximum_error=max(errors.values()) if complete else None,report_sha256=sha256(ROOT/'report.json')))
    print(json.dumps(dict(status=report['status'],summary={s:dict(auc90={name:x['readmission']['points']['90']['auc'] for name,x in r['metrics'].items()},
        ibs={name:x['readmission']['integrated_brier_7_365'] for name,x in r['metrics'].items()},
        increment=r['clinical_paired_increment']) for s,r in results.items()})),flush=True)


if __name__=='__main__':run()
