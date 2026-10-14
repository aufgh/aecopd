"""第七批完成后独立进程核验/锁定一次确认；不重复训练。"""
import json
import subprocess
import sys
from patient_data.export_mimiciv import sha256
from training.v6.patient_balance_experiment import ROOT
from training.v6.train import write_json


def run():
    if (ROOT/'.running.lock').exists():raise RuntimeError('patient balance training still running')
    if not (ROOT/'selected_before_confirmation.json').exists():raise RuntimeError('complete balance choice missing')
    def step(mode):
        cmd=[sys.executable,'-u','-m','training.v6.patient_balance_audit','--mode',mode]
        write_json(ROOT/'postprocess_state.json',dict(status='running',command=cmd,goal_achieved=False))
        subprocess.run(cmd,check=True)
    step('verify');step('verify')
    subprocess.run([sys.executable,'-u','-m','training.v6.patient_balance_report','--mode','tuning'],check=True)
    lockpath=ROOT/'nomination_lock.json'
    if not lockpath.exists():step('confirm')
    lock=json.loads(lockpath.read_text())
    if lock['selection_sha256']!=sha256(ROOT/'selected_before_confirmation.json'):raise ValueError('existing balance lock stale')
    if lock['nominated']:
        if not (ROOT/'confirmation_report.json').exists():raise RuntimeError('locked incomplete confirmation; inspect scene')
        step('verify-confirmation')
        subprocess.run([sys.executable,'-u','-m','training.v6.patient_balance_report','--mode','confirmation'],check=True)
    write_json(ROOT/'postprocess_state.json',dict(status='complete',selected=lock['selected'],nominated=lock['nominated'],goal_achieved=False))
    print(json.dumps(dict(status='balance_postprocess_complete',selected=lock['selected'])),flush=True)


if __name__=='__main__':
    lock=ROOT/'.postprocess.lock'
    with lock.open('x',encoding='utf-8') as owner:owner.write('patient_balance_finish')
    try:run()
    except Exception as error:
        write_json(ROOT/'postprocess_state.json',dict(status='failed',error_type=type(error).__name__,error=str(error),goal_achieved=False));raise
    finally:lock.unlink()
