"""五种子确认独立重放与预定增益门槛审计；不读取测试预测。"""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.regularization_multiseed import ROOT,SEARCH,SEEDS,BASE_SEEDS,EPOCHS,build
from training.v6.regularization_model import Trial
from training.v6.module_surgery_experiment import batches,parameters,summary
from training.v6.module_surgery import SurgeryModel,probability_curve
from training.v6.token_model import daily_probabilities
from training.v6.horizon_objective import restore
from training.v6.train import write_json


def audit_gate(models,names):
    """仅生成证据与达标状态；不能自动把内部确认称盲测成功。"""
    result={}
    for name in names:
        comparisons={}
        for comparator in ('v6_control','official_V6'):
            aa,bb,dd,unsafe=[],[],[],[]
            for seed in SEEDS:
                left=models[str(seed)][name]['metrics'];right=models[str(seed)][comparator]['metrics']
                l,r=left['readmission'],right['readmission']
                left_auc,right_auc=l['points']['90']['auc'],r['points']['90']['auc']
                aa.append(left_auc-right_auc if left_auc is not None and right_auc is not None else float('nan'))
                bb.append(l['integrated_brier_7_365']-r['integrated_brier_7_365'])
                dd.append(left['death']['integrated_brier_7_365']-right['death']['integrated_brier_7_365'])
                # 多时间点不应明显退化；阈值预先固定为AUC降0.005/Brier增0.001。
                unsafe.append(any(l['points'][str(day)]['auc'] is not None and r['points'][str(day)]['auc'] is not None and
                    l['points'][str(day)]['auc']<r['points'][str(day)]['auc']-.005 for day in (30,90,180,365)) or
                    any(l['points'][str(day)]['brier']>r['points'][str(day)]['brier']+.001 for day in (30,90,180,365)))
            baseline=np.mean([models[str(s)][comparator]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS])
            passed=(np.isfinite(aa).all() and sum(a>0 and b<0 for a,b in zip(aa,bb))>=4 and np.mean(aa)>=.005 and
                    -np.mean(bb)/baseline>=.01 and sum(unsafe)<=1 and np.mean(dd)<=.001)
            comparisons[comparator]=dict(auc90_differences=[x if np.isfinite(x) else None for x in aa],ibs_differences=bb,death_ibs_differences=dd,
                joint_improved_seeds=sum(a>0 and b<0 for a,b in zip(aa,bb)),mean_auc90_difference=float(np.mean(aa)) if np.isfinite(aa).all() else None,
                mean_ibs_difference=float(np.mean(bb)),relative_ibs_reduction=float(-np.mean(bb)/baseline),
                unsafe_timepoint_seeds=sum(unsafe),predeclared_basic_gate_passed=bool(passed))
        result[name]=dict(comparisons=comparisons,basic_gate_passed=all(x['predeclared_basic_gate_passed'] for x in comparisons.values()),
            stable_superiority_proven=False,
            remaining_requirements=['calibration and history-strata audit','locked historical test comparison (not blind)','paired uncertainty across seeds'])
    return result


def verify():
    r=json.loads((ROOT/'report.json').read_text());contract=r['contract']
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('multiseed source changed: '+p)
    for file,key in (('contract.json','search_contract_sha256'),('screening_verification.json','search_verification_sha256'),('nomination_lock.json','nomination_lock_sha256')):
        if sha256(SEARCH/file)!=contract[key]:raise ValueError('locked search changed')
    d,c,h,ext,_,bins,tune,confirm,search,_=checked()
    cache=batches(d,c,ext,confirm)
    trials=[Trial(**x) for x in contract['trials']]
    errors={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        if sha256(Path(search['input_config']['parent'])/f'seed{BASE_SEEDS[seed]}/survival_base/best.pt')!=contract['frozen_baselines'][str(seed)]:
            raise ValueError('foundation checkpoint changed')
        for trial in [*trials,None]:
            name=trial.name if trial else 'official_V6';path=ROOT/f'seed{seed}'/name
            if trial:
                m=build(base,seed,trial,bins)
                state=torch.load(path/'best.pt',map_location='cuda',weights_only=True)
                history=json.loads((path/'history.json').read_text())
                fit=r['models'][str(seed)][name]['training']
                if ([x['epoch'] for x in history]!=list(range(EPOCHS+1)) or
                    min(history,key=lambda x:x['tuning_ibs'])['epoch']!=state['epoch'] or
                    sha256(path/'best.pt')!=fit['checkpoint_sha256'] or state['trial']!=asdict(trial)):
                    raise ValueError('training contract/selection differs')
                m.load_state_dict(state['state_dict'])
            else:
                m=SurgeryModel(base,BASE_SEEDS[seed],'control',bins).cuda()
                m.load_state_dict(torch.load(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt',map_location='cuda',weights_only=True)['state_dict'])
            if any(not torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):raise ValueError('foundation not frozen')
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k)
            with np.load(path/'confirmation_predictions.npz') as z:
                for key,expected in (('hadm_ids',d.arrays['hadm_ids'][confirm]),('subject_ids',d.arrays['subject_ids'][confirm]),('times',d.times[confirm]),('events',d.events[confirm])):
                    np.testing.assert_array_equal(z[key],expected)
                error=float(np.abs(curve-z['curve']).max())
                daily,conditional=daily_probabilities(curve)
                np.testing.assert_allclose(daily,z['daily_event_probability'],atol=3e-7,rtol=0)
                np.testing.assert_allclose(conditional,z['conditional_daily_event_probability'],atol=3e-7,rtol=0)
            if error>3e-7 or not np.isfinite(curve).all():raise ValueError('probability replay failed')
            np.testing.assert_allclose(curve.sum(2),1,atol=1e-9)
            if np.any(np.diff(curve[:,:,1:],axis=1)<-1e-10):raise ValueError('nonmonotonic CIF')
            if summary(curve,d,confirm,nll)!=r['models'][str(seed)][name]['metrics']:raise ValueError('metric replay differs')
            errors[f'{seed}/{name}']=error;del m
    r['status']='multiseed_confirmation_independently_verified'
    write_json(ROOT/'report.json',r)
    write_json(ROOT/'independent_verification.json',dict(status=r['status'],models=len(errors),maximum_error=max(errors.values()),
        replay_errors=errors,report_sha256=sha256(ROOT/'report.json'),verifier_sha256=sha256(Path(__file__)),test_not_evaluated=True))
    gate=audit_gate(r['models'],[x.name for x in trials if x.name!='v6_control'])
    write_json(ROOT/'basic_gate_audit.json',dict(candidate_gates=gate,goal_achieved=False,
        explanation='basic thresholds alone do not complete calibration/strata/test requirements'))
    print(json.dumps(dict(status=r['status'],models=len(errors),maximum_error=max(errors.values()),gates=gate)),flush=True)


if __name__=='__main__':verify()
