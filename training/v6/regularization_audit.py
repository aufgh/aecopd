"""正则搜索独立核验/锁定确认；无测试折加载或候选改选。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_model import TRIALS,make_model
from training.v6.regularization_search import ROOT,SEED,EPOCHS,validation_parts
from training.v6.module_surgery_experiment import inputs,batches,parameters,summary,save
from training.v6.module_surgery import probability_curve,SurgeryModel
from training.v6.horizon_objective import restore
from training.v6.deep_interaction_experiment import runtime_setup
from training.v6.evaluation import paired_interval
from training.v6.train import write_json

REFERENCE=Path('training_runs/v6_module_surgery_2026-10-12_r3')


def checked():
    runtime_setup()
    contract=json.loads((ROOT/'contract.json').read_text())
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('search source changed: '+p)
    # 补充完整旧依赖的既定哈希审计，不把检查时的新哈希冒充训练快照。
    prior=json.loads((REFERENCE/'contract.json').read_text())
    for p,digest in prior['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('official V6 dependency changed: '+p)
    d,c,h,ext,config,_,_=inputs()
    if (d.provenance!=contract['data_provenance'] or c.provenance!=contract['compact_provenance'] or
            config!=contract['input_config']):raise ValueError('input provenance differs')
    tune,confirm=validation_parts(d.arrays['subject_ids'],d.arrays['patient_splits'])
    with np.load(ROOT/'validation_partition.npz') as z:
        np.testing.assert_array_equal(tune,z['tuning_indices'])
        np.testing.assert_array_equal(confirm,z['confirmation_indices'])
        np.testing.assert_array_equal(d.arrays['subject_ids'],z['subject_ids'])
    if sha256(Path(config['parent'])/f'seed{SEED}/survival_base/best.pt')!=contract['frozen_baseline_sha256']:
        raise ValueError('baseline source differs')
    if sha256(REFERENCE/'preprocessing.json')!=contract['preprocessing_sha256']:
        raise ValueError('preprocessing source differs')
    base=restore(Path(config['parent']),SEED,'survival_base',h,.1)
    bins=json.loads((REFERENCE/'preprocessing.json').read_text())['bins']
    return d,c,h,ext,base,bins,tune,confirm,contract,prior


def candidate(base,trial,bins):
    m=make_model(base,SEED,trial,bins).cuda();torch.manual_seed(SEED)
    path=ROOT/trial.name
    info=json.loads((path/'training_result.json').read_text())
    state=torch.load(path/'best.pt',map_location='cuda',weights_only=True)
    history=json.loads((path/'history.json').read_text())
    if info['trial']!=asdict(trial) or state['trial']!=asdict(trial):raise ValueError('trial contract differs')
    if info['completed_epochs']!=EPOCHS or [x['epoch'] for x in history]!=list(range(EPOCHS+1)):
        raise ValueError('incomplete epoch history')
    if min(history,key=lambda x:x['tuning_ibs'])['epoch']!=state['epoch'] or info['best_epoch']!=state['epoch']:
        raise ValueError('checkpoint was not selected by tuning IBS')
    if sha256(path/'best.pt')!=info['checkpoint_sha256']:raise ValueError('checkpoint hash differs')
    m.load_state_dict(state['state_dict'])
    if any(not torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):
        raise ValueError('frozen baseline changed')
    return m,info


def verify(*,partial=False):
    d,c,h,ext,base,bins,tune,_,contract,prior=checked()
    cache=batches(d,c,ext,tune)
    trials=[x for x in TRIALS if (ROOT/x.name/'training_result.json').exists()]
    if not partial and len(trials)!=len(TRIALS):raise ValueError('search not finished')
    errors={}
    for trial in trials:
        m,info=candidate(base,trial,bins)
        a,k,nll=parameters(m,cache);curve=probability_curve(a,k)
        with np.load(ROOT/trial.name/'tuning_predictions.npz') as z:
            np.testing.assert_array_equal(z['times'],d.times[tune]);np.testing.assert_array_equal(z['events'],d.events[tune])
            error=float(np.abs(curve-z['curve']).max())
            np.testing.assert_allclose(a,z['log_amplitude'],atol=1e-7,rtol=0)
            np.testing.assert_allclose(k,z['shape'],atol=1e-7,rtol=0)
        if error>3e-7 or not np.isfinite(curve).all() or np.any((curve<0)|(curve>1)):
            raise ValueError('probability replay failed')
        np.testing.assert_allclose(curve.sum(2),1,atol=1e-9)
        if np.any(np.diff(curve[:,:,1:],axis=1)<-1e-10):raise ValueError('nonmonotonic probabilities')
        from training.v6.module_surgery_experiment import validation
        if validation(m,cache,d.times[tune],d.events[tune])!=info['tuning']:raise ValueError('metric replay differs')
        errors[trial.name]=error;del m
    proof=dict(status='partial_completed_models_verified' if partial else 'screening_independently_verified',
        models=len(trials),maximum_error=max(errors.values()) if errors else None,replay_errors=errors,
        source_hashes_verified=len(contract['source_sha256']),official_dependency_hashes_verified=len(prior['source_sha256']),
        validation_partition_recomputed=True,confirmation_not_evaluated=True,test_not_evaluated=True)
    proof['audit_code_sha256']=sha256(Path(__file__))
    if not partial:
        r=json.loads((ROOT/'screening_report.json').read_text())
        ranked=sorted(r['results'],key=lambda x:r['results'][x]['tuning']['ibs'])
        if ranked!=r['ranked_by_tuning_ibs']:raise ValueError('ranking differs')
        nominated=[x for x in ranked if x!='v6_control' and r['results'][x]['tuning']['ibs']<r['results']['v6_control']['tuning']['ibs']][:3]
        if nominated!=r['nominated_before_confirmation']:raise ValueError('nomination changed')
        proof['screening_report_sha256']=sha256(ROOT/'screening_report.json')
    write_json(ROOT/('partial_verification.json' if partial else 'screening_verification.json'),proof)
    print(json.dumps(proof),flush=True)


def confirm():
    proof=json.loads((ROOT/'screening_verification.json').read_text())
    if proof['screening_report_sha256']!=sha256(ROOT/'screening_report.json'):raise ValueError('screening proof stale')
    if proof['audit_code_sha256']!=sha256(Path(__file__)):raise ValueError('verification code changed')
    r=json.loads((ROOT/'screening_report.json').read_text())
    nominated=r['nominated_before_confirmation']
    lock=dict(screening_report_sha256=sha256(ROOT/'screening_report.json'),nominated=nominated,
        comparators=['v6_control','official_V6'],no_test_evaluation=True,
        rule='confirm these locked nominees once; no substitution based on confirmation scores')
    if (ROOT/'nomination_lock.json').exists():raise FileExistsError('confirmation already locked; inspect existing run')
    write_json(ROOT/'nomination_lock.json',lock)
    d,c,h,ext,base,bins,tune,confirm_indices,contract,_=checked()
    cache=batches(d,c,ext,confirm_indices)
    trials={x.name:x for x in TRIALS}
    outputs,curves={},{ }
    for name in ['v6_control',*nominated,'official_V6']:
        if name=='official_V6':
            m=SurgeryModel(base,SEED,'control',bins).cuda()
            path=REFERENCE/f'seed{SEED}/control/best.pt'
            m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
            digest=sha256(path)
        else:
            m,info=candidate(base,trials[name],bins);digest=info['checkpoint_sha256']
        a,k,nll=parameters(m,cache);curve=probability_curve(a,k)
        outputs[name]=dict(checkpoint_sha256=digest,metrics=summary(curve,d,confirm_indices,nll))
        folder=ROOT/'confirmation'/name;folder.mkdir(parents=True,exist_ok=True)
        save(folder/'predictions.npz',d,confirm_indices,a,k,curve)
        curves[name]=curve;del m
    pairs={}
    for name in nominated:
        for reference in ('v6_control','official_V6'):
            pairs[f'{name} minus {reference}']=paired_interval(d.arrays['subject_ids'][confirm_indices],
                d.times[confirm_indices],d.events[confirm_indices],curves[name][:,:,1],curves[reference][:,:,1],replicates=200,seed=SEED)
    report=dict(status='single_seed_confirmation_not_stability_proof',lock=lock,models=outputs,comparisons=pairs,
        admissions=len(confirm_indices),patients=len(np.unique(d.arrays['subject_ids'][confirm_indices])),
        test_not_evaluated=True,goal_achieved=False,
        limitations=['Confirmation subset already exposed in prior project, not blind.','Single seed and selected candidates; not proof of stable gains.'])
    write_json(ROOT/'confirmation_report.json',report)
    write_json(ROOT/'state.json',dict(status='single_seed_confirmation_complete',completed_trials=list(r['results']),
        nominated=nominated,next_phase='predeclare multiseed confirmation without changing candidate settings',goal_achieved=False))
    print(json.dumps(dict(status=report['status'],nominated=nominated,goal_achieved=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('partial','verify','confirm'),required=True)
    mode=p.parse_args().mode
    verify(partial=mode=='partial') if mode!='confirm' else confirm()
