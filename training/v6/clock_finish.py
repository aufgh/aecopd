"""第五批单次后处理，每一步在独立Python进程运行，不重训。"""
import json
import subprocess
import sys
from training.v6.clock_experiment import ROOT
from training.v6.train import write_json


def run():
    if (ROOT/'.running.lock').exists():raise RuntimeError('clock training is running')
    for mode in ('verify','confirm','verify-confirmation'):
        if mode=='confirm' and (ROOT/'nomination_lock.json').exists():
            if not (ROOT/'confirmation_report.json').exists():
                raise RuntimeError('locked incomplete confirmation; inspect scene')
            continue
        command=[sys.executable,'-u','-m','training.v6.clock_audit','--mode',mode]
        write_json(ROOT/'postprocess_state.json',dict(status='running',command=command))
        subprocess.run(command,check=True)
    write_json(ROOT/'postprocess_state.json',dict(status='complete',goal_achieved=False))


if __name__=='__main__':
    lock=ROOT/'.postprocess.lock'
    with lock.open('x',encoding='utf-8') as owner:owner.write('clock_finish')
    try:run()
    except Exception as error:
        write_json(ROOT/'postprocess_state.json',dict(status='failed',error_type=type(error).__name__,error=str(error)))
        raise
    finally:lock.unlink()
