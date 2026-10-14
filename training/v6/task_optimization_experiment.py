"""第二批：三种子原因特异/独立临床分支/梯度替换，只用调参半折。"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked
from training.v6.regularization_multiseed import fit
from training.v6.task_optimization import TRIALS,build,backward
from training.v6.module_surgery_experiment import batches,parameters,validation
from training.v6.module_surgery import probability_curve
from training.v6.horizon_objective import restore
from training.v6.train import write_json

ROOT=Path('training_runs/v6_task_optimization_2026-10-13')
PROTOCOL=Path('docs/V6_TASK_OPTIMIZATION_PROTOCOL_2026-10-13.md')
SEEDS=(42,7,101)
BASE_SEEDS={42:42,7:7,101:42}


def run():
    d,c,h,ext,_,bins,tune,_,search,_=checked()
    protected={p:sha256(Path(p)) for p in ('training/v6/task_optimization.py','training/v6/task_optimization_experiment.py',
        'training/v6/regularization_multiseed.py','training/v6/task_gradient_diagnostic.py','training/tests/test_v6_task_optimization.py',str(PROTOCOL))}
    contract=dict(source_sha256=protected,search_contract_sha256=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')),
        seeds=list(SEEDS),baseline_seed_mapping=BASE_SEEDS,trials=[asdict(x) for x in TRIALS],epochs=100,
        frozen_baselines={str(s):sha256(Path(search['input_config']['parent'])/f'seed{BASE_SEEDS[s]}/survival_base/best.pt') for s in SEEDS},
        input_provenance=search['data_provenance'],selection='three-seed tuning-half readmission IBS mean',
        confirmation_not_evaluated=True,test_not_evaluated=True)
    if (ROOT/'contract.json').exists():
        if json.loads((ROOT/'contract.json').read_text())!=json.loads(json.dumps(contract)):
            raise ValueError('task-optimization resume contract changed')
    else:write_json(ROOT/'contract.json',contract)
    cache=dict(train=batches(d,c,ext,np.flatnonzero(d.arrays['patient_splits']=='train')),tune=batches(d,c,ext,tune))
    results={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        results[str(seed)]={}
        control=build(base,seed,TRIALS[0],bins)
        for trial in TRIALS:
            path=ROOT/f'seed{seed}'/trial.name
            if (path/'training_result.json').exists():
                info=json.loads((path/'training_result.json').read_text())
                if info['trial']!=asdict(trial) or sha256(path/'best.pt')!=info['checkpoint_sha256']:raise ValueError('completed model changed')
                results[str(seed)][trial.name]=info;continue
            if path.exists() and any((path/p).exists() for p in ('best.pt','history.json')):
                import shutil
                backup=path/'recovery_snapshots'/str(time.time_ns());backup.mkdir(parents=True)
                for p in ('best.pt','history.json'):
                    if (path/p).exists():shutil.copy2(path/p,backup/p)
            probe=build(base,seed,trial,bins).eval()
            encoders=probe.encoders if trial.independent_encoders else [probe.encoder]
            for encoder in encoders:
                if any(not torch.equal(v,encoder.state_dict()[k]) for k,v in control.encoder.state_dict().items()):raise ValueError('encoder initial weights differ')
            with torch.no_grad():
                _,hist,b,_,_=cache['tune'][0];out=probe(hist,b);expected=base(hist)
                if not torch.equal(out.log_amplitude,expected.log_amplitude) or not torch.equal(out.shape,expected.shape):raise ValueError('epoch0 risk differs')
            del probe
            write_json(ROOT/'state.json',dict(status='running',seed=seed,trial=trial.name,completed=results,goal_achieved=False))
            info=fit(base,seed,trial,bins,cache,d.times[tune],d.events[tune],path,model_factory=build,backward=backward)
            results[str(seed)][trial.name]=info
        del control
    score={trial.name:float(np.mean([results[str(s)][trial.name]['tuning_ibs'] for s in SEEDS])) for trial in TRIALS}
    chosen=min(score,key=score.get)
    write_json(ROOT/'selected_before_confirmation.json',dict(results=results,tuning_scores=score,selected=chosen,
        all_100_epochs_complete=True,confirmation_not_evaluated=True,test_not_evaluated=True))
    write_json(ROOT/'state.json',dict(status='training_complete_pending_replay',selected=chosen,
        next_phase='verify all 24 models, lock nominees before confirmation',goal_achieved=False))
    print(json.dumps(dict(status='training_complete_pending_replay',selected=chosen,goal_achieved=False)),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    ROOT.mkdir(exist_ok=True);lock=ROOT/'.running.lock'
    with lock.open('x',encoding='utf-8') as owner:owner.write(json.dumps(dict(pid=os.getpid(),started_unix=time.time())))
    try:run()
    except Exception as e:
        write_json(ROOT/'state.json',dict(status='failed',error_type=type(e).__name__,error=str(e),goal_achieved=False));raise
    finally:lock.unlink()
