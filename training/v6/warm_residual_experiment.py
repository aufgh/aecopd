"""从强V6起点作四类修正；教师仅重载，不冒称新训练。"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.regularization_multiseed import fit
from training.v6.warm_residual import TRIALS,BASE_SEEDS,build
from training.v6.module_surgery import SurgeryModel,probability_curve
from training.v6.module_surgery_experiment import batches,parameters,validation,summary,save
from training.v6.horizon_objective import restore
from training.v6.train import write_json

ROOT=Path('training_runs/v6_warm_residual_2026-10-13')
PRIOR=Path('training_runs/v6_contextual_residual_2026-10-13')
SEEDS=(42,7,101)
SOURCES=('training/v6/warm_residual.py','training/v6/warm_residual_experiment.py',
         'training/v6/regularization_multiseed.py','training/tests/test_v6_warm_residual.py',
         'docs/V6_WARM_START_PROTOCOL_2026-10-13.md')


def run():
    proof=json.loads((PRIOR/'tuning_verification.json').read_text())
    if proof['status']!='tuning_independently_verified' or proof['models']!=18:raise ValueError('context batch not verified')
    d,c,h,ext,_,bins,tune,_,search,prior=checked()
    official=json.loads((REFERENCE/'report.json').read_text())
    weights={str(s):sha256(REFERENCE/f'seed{s}/control/best.pt') for s in set(BASE_SEEDS[x] for x in SEEDS)}
    for s,digest in weights.items():
        if digest!=official['models'][s]['control']['checkpoint_sha256']:raise ValueError('teacher weight changed')
    contract=dict(source_sha256={p:sha256(Path(p)) for p in SOURCES},
        search_contract_sha256=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')),
        teacher_checkpoint_sha256=weights,trials=[asdict(t) for t in TRIALS],seeds=list(SEEDS),
        teacher_seed_mapping={str(s):BASE_SEEDS[s] for s in SEEDS},epochs=100,
        selection='three clinical-seed mean tuning IBS; original teacher included as comparator',
        confirmation_not_evaluated=True,test_not_evaluated=True,
        limits=['Teacher selected in previously exposed full validation.','Three clinical seeds reuse two teacher sources.'])
    if (ROOT/'contract.json').exists():
        if json.loads((ROOT/'contract.json').read_text())!=contract:raise ValueError('warm resume contract changed')
    else:write_json(ROOT/'contract.json',contract)
    cache=dict(train=batches(d,c,ext,np.flatnonzero(d.arrays['patient_splits']=='train')),tune=batches(d,c,ext,tune));results={};references={}
    for seed in SEEDS:
        source=BASE_SEEDS[seed];base=restore(Path(search['input_config']['parent']),source,'survival_base',h,.1)
        teacher=SurgeryModel(base,source,'control',bins).cuda().eval()
        teacher.load_state_dict(torch.load(REFERENCE/f'seed{source}/control/best.pt',map_location='cuda',weights_only=True)['state_dict'])
        a,k,nll=parameters(teacher,cache['tune']);curve=probability_curve(a,k)
        folder=ROOT/f'seed{seed}/original_teacher';folder.mkdir(parents=True,exist_ok=True)
        save(folder/'tuning_predictions.npz',d,tune,a,k,curve)
        references[str(seed)]=dict(tuning=validation(teacher,cache['tune'],d.times[tune],d.events[tune]),metrics=summary(curve,d,tune,nll),
                                 teacher_source=source,teacher_checkpoint_sha256=weights[str(source)],newly_trained=False)
        results[str(seed)]={}
        for trial in TRIALS:
            path=ROOT/f'seed{seed}'/trial.name
            if (path/'training_result.json').exists():
                info=json.loads((path/'training_result.json').read_text())
                if info['trial']!=asdict(trial) or info['completed_epochs']!=100 or sha256(path/'best.pt')!=info['checkpoint_sha256']:
                    raise ValueError('completed warm model changed')
                results[str(seed)][trial.name]=info;continue
            if path.exists() and any((path/p).exists() for p in ('best.pt','history.json')):
                import shutil
                backup=path/'recovery_snapshots'/str(time.time_ns());backup.mkdir(parents=True)
                for p in ('best.pt','history.json'):
                    if (path/p).exists():shutil.copy2(path/p,backup/p)
            probe=build(base,seed,trial,bins).eval()
            with torch.no_grad():
                _,hist,b,_,_=cache['tune'][0];out=probe(hist,b);expected=teacher(hist,b)
                if not torch.equal(out.log_amplitude,expected.log_amplitude) or not torch.equal(out.shape,expected.shape):
                    raise ValueError('warm epoch0 differs from trained V6')
            del probe
            write_json(ROOT/'state.json',dict(status='running',seed=seed,trial=trial.name,completed=results,goal_achieved=False))
            info=fit(base,seed,trial,bins,cache,d.times[tune],d.events[tune],path,model_factory=build)
            stored=torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict']
            if trial.mode!='finetune':
                for key,value in teacher.state_dict().items():
                    if not torch.equal(value,stored['teacher.'+key]):raise ValueError('full frozen teacher changed')
                info['full_teacher_frozen_verified']=True
            else:info['full_teacher_frozen_verified']=False
            info['initial_teacher_checkpoint_sha256']=weights[str(source)]
            write_json(path/'training_result.json',info);results[str(seed)][trial.name]=info
        del teacher
    scores={'original_teacher':float(np.mean([references[str(s)]['tuning']['ibs'] for s in SEEDS]))}
    scores.update({t.name:float(np.mean([results[str(s)][t.name]['tuning_ibs'] for s in SEEDS])) for t in TRIALS})
    selected=min(scores,key=scores.get)
    write_json(ROOT/'selected_before_confirmation.json',dict(results=results,references=references,tuning_scores=scores,selected=selected,
        all_100_candidate_epochs_complete=True,original_teacher_reloaded_not_retrained=True,confirmation_not_evaluated=True,test_not_evaluated=True))
    write_json(ROOT/'state.json',dict(status='training_complete_pending_replay',selected=selected,goal_achieved=False))
    print(json.dumps(dict(status='training_complete_pending_replay',selected=selected)),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();ROOT.mkdir(exist_ok=True)
    lock=ROOT/'.running.lock'
    with lock.open('x',encoding='utf-8') as owner:owner.write(json.dumps(dict(pid=os.getpid(),started=time.time())))
    try:run()
    except Exception as e:
        write_json(ROOT/'state.json',dict(status='failed',error_type=type(e).__name__,error=str(e),goal_achieved=False));raise
    finally:lock.unlink()
