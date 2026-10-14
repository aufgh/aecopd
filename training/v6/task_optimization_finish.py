"""第二批结束后的单次后处理队列；每个核验步骤为独立Python进程。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.task_optimization_experiment import ROOT,SEEDS
from training.v6.train import write_json


def material_signal(models,chosen,comparators=('control','official_V6')):
    """仅决定是否值得五种子确认，不是长期成功门槛。"""
    result={}
    for ref in comparators:
        auc=[];ibs=[];death=[];joint=0
        for seed in SEEDS:
            x=models[str(seed)][chosen]['metrics'];b=models[str(seed)][ref]['metrics']
            a=x['readmission']['points']['90']['auc'];ba=b['readmission']['points']['90']['auc']
            ad=a-ba if a is not None and ba is not None else float('nan')
            bd=x['readmission']['integrated_brier_7_365']-b['readmission']['integrated_brier_7_365']
            auc.append(ad);ibs.append(bd);death.append(x['death']['integrated_brier_7_365']-b['death']['integrated_brier_7_365'])
            joint+=int(ad>0 and bd<0)
        denominator=np.mean([models[str(s)][ref]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS])
        ma=float(np.mean(auc)) if np.isfinite(auc).all() else None
        improvement=float(-np.mean(ibs)/denominator)
        result[ref]=dict(mean_auc90_gain=ma,relative_ibs_reduction=improvement,joint_improved_seeds=joint,
            death_ibs_change=float(np.mean(death)),worth_five_seed_confirmation=bool(ma is not None and ma>=.003 and
                improvement>=.005 and joint>=2 and np.mean(death)<=.001))
    return dict(comparators=result,worth_five_seed_confirmation=all(v['worth_five_seed_confirmation'] for v in result.values()),
                goal_achieved=False,limits='screening threshold only; final five-seed/timepoint/calibration/strata/test gate unchanged')


def run():
    state=json.loads((ROOT/'state.json').read_text())
    if state['status']=='running' or (ROOT/'.running.lock').exists():raise RuntimeError('training still running')
    if not (ROOT/'selected_before_confirmation.json').exists():raise RuntimeError('complete training selection missing')
    def step(module,mode=None):
        command=[sys.executable,'-u','-m',module]+([] if mode is None else ['--mode',mode])
        write_json(ROOT/'postprocess_state.json',dict(status='running',command=command,goal_achieved=False))
        subprocess.run(command,check=True)
    step('training.v6.task_optimization_audit','verify')
    step('training.v6.task_optimization_audit','verify')
    step('training.v6.task_optimization_report','tuning')
    lockpath=ROOT/'nomination_lock.json'
    if not lockpath.exists():step('training.v6.task_optimization_audit','confirm')
    lock=json.loads(lockpath.read_text())
    if lock['selected_sha256']!=sha256(ROOT/'selected_before_confirmation.json'):raise RuntimeError('existing lock stale')
    if lock['nominated']:
        if not (ROOT/'confirmation_report.json').exists():raise RuntimeError('locked but incomplete confirmation; inspect, do not reselect')
        step('training.v6.task_optimization_audit','verify-confirmation')
        step('training.v6.task_optimization_report','confirmation')
        step('training.v6.task_optimization_diagnostics')
        r=json.loads((ROOT/'confirmation_report.json').read_text());signal=material_signal(r['models'],lock['selected'])
    else:signal=dict(worth_five_seed_confirmation=False,reason='matched control selected on tuning',goal_achieved=False)
    write_json(ROOT/'continuation_decision.json',signal)
    write_json(ROOT/'postprocess_state.json',dict(status='complete',selected=lock['selected'],
        next_phase='inspect calibrated/stratified evidence; five seeds if material signal, otherwise next bounded hypothesis',goal_achieved=False))
    print(json.dumps(dict(status='second_batch_verified_and_review_ready',selected=lock['selected'],decision=signal)),flush=True)


if __name__=='__main__':
    lock=ROOT/'.postprocess.lock'
    with lock.open('x',encoding='utf-8') as owner:owner.write(json.dumps(dict(pid=os.getpid(),started=time.time())))
    try:run()
    except Exception as e:
        write_json(ROOT/'postprocess_state.json',dict(status='failed',error_type=type(e).__name__,error=str(e),goal_achieved=False));raise
    finally:lock.unlink()
