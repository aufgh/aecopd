"""第三批单次后处理；保持既定临床名单，不重挑确认候选。"""
import json
import os
import subprocess
import sys
import time
from patient_data.export_mimiciv import sha256
from training.v6.contextual_residual_experiment import ROOT
from training.v6.task_optimization_finish import material_signal
from training.v6.train import write_json


def run():
    state=json.loads((ROOT/'state.json').read_text())
    if state['status']=='running' or (ROOT/'.running.lock').exists():raise RuntimeError('context training still running')
    if not (ROOT/'selected_before_confirmation.json').exists():raise RuntimeError('context selection missing')
    def step(module,mode=None):
        command=[sys.executable,'-u','-m',module]+([] if mode is None else ['--mode',mode])
        write_json(ROOT/'postprocess_state.json',dict(status='running',command=command,goal_achieved=False))
        subprocess.run(command,check=True)
    step('training.v6.contextual_control_replay')
    step('training.v6.contextual_residual_experiment','verify')
    step('training.v6.contextual_residual_experiment','verify')
    step('training.v6.contextual_residual_report','tuning')
    if not (ROOT/'nomination_lock.json').exists():step('training.v6.contextual_residual_experiment','confirm')
    lock=json.loads((ROOT/'nomination_lock.json').read_text())
    if lock['selected_sha256']!=sha256(ROOT/'selected_before_confirmation.json'):raise RuntimeError('context lock stale')
    if lock['nominated']:
        if not (ROOT/'confirmation_report.json').exists():raise RuntimeError('locked but incomplete context confirmation; inspect, do not reselect')
        step('training.v6.contextual_residual_experiment','verify-confirmation')
        step('training.v6.contextual_residual_report','confirmation')
        step('training.v6.contextual_residual_report','diagnostics')
        r=json.loads((ROOT/'confirmation_report.json').read_text())
        decision=material_signal(r['models'],lock['selected'],comparators=('control','baseline_only','official_V6'))
    else:
        decision=dict(worth_five_seed_confirmation=False,selected=lock['selected'],goal_achieved=False,
            reason='no clinical candidate selected; do not substitute from confirmation',
            next_phase='if baseline_only wins, separately assess model-performance signal without claiming clinical increment; otherwise another bounded hypothesis')
    write_json(ROOT/'continuation_decision.json',decision)
    write_json(ROOT/'postprocess_state.json',dict(status='complete',selected=lock['selected'],goal_achieved=False))
    print(json.dumps(dict(status='third_batch_verified_review_ready',selected=lock['selected'],decision=decision)),flush=True)


if __name__=='__main__':
    lock=ROOT/'.postprocess.lock'
    with lock.open('x',encoding='utf-8') as owner:owner.write(json.dumps(dict(pid=os.getpid(),started=time.time())))
    try:run()
    except Exception as e:
        write_json(ROOT/'postprocess_state.json',dict(status='failed',error_type=type(e).__name__,error=str(e),goal_achieved=False));raise
    finally:lock.unlink()
