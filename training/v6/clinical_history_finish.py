"""第六批两个独立重放进程；原教师胜出时不另挑候选确认。"""
import json
import subprocess
import sys
from patient_data.export_mimiciv import sha256
from training.v6.clinical_history_experiment import ROOT
from training.v6.train import write_json


def run():
    if (ROOT/'.running.lock').exists():raise RuntimeError('history training still running')
    for number in (1,2):
        write_json(ROOT/'postprocess_state.json',dict(status='running',independent_process=number))
        subprocess.run([sys.executable,'-u','-m','training.v6.clinical_history_verify'],check=True)
    proof=json.loads((ROOT/'tuning_verification.json').read_text())
    if proof['status']!='history_tuning_independently_verified' or proof['models']!=18:
        raise ValueError('complete history replay proof required')
    chosen=json.loads((ROOT/'selected_before_confirmation.json').read_text())['selected']
    target=ROOT/'nomination_lock.json'
    lock=dict(selected=chosen,selection_sha256=sha256(ROOT/'selected_before_confirmation.json'),
        nominated=[] if chosen in ('original_teacher','current_only','times_only') else [chosen])
    if target.exists():
        if json.loads(target.read_text())!=lock:raise ValueError('existing history lock differs')
    else:write_json(target,lock)
    write_json(ROOT/'postprocess_state.json',dict(status='complete',nominated=lock['nominated'],goal_achieved=False))
    print(json.dumps(dict(status='history_replay_and_nomination_complete',selected=chosen,nominated=lock['nominated'])),flush=True)


if __name__=='__main__':
    lock=ROOT/'.postprocess.lock'
    with lock.open('x',encoding='utf-8') as owner:owner.write('clinical_history_finish')
    try:run()
    except Exception as error:
        write_json(ROOT/'postprocess_state.json',dict(status='failed',error_type=type(error).__name__,error=str(error)));raise
    finally:lock.unlink()
