"""第三批已完成control精确复现第二批；不评估未完成候选。"""
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.contextual_residual_experiment import ROOT,PRIOR,SEEDS,BASE_SEEDS,TRIALS,inputs,model
from training.v6.module_surgery_experiment import batches,parameters
from training.v6.module_surgery import probability_curve
from training.v6.horizon_objective import restore
from training.v6.train import write_json


def run():
    (d,c,h,ext,_,bins,ix,_,search,_),_=inputs();cache=batches(d,c,ext,ix);checks={}
    for seed in SEEDS:
        path=ROOT/f'seed{seed}/control'
        if not (path/'training_result.json').exists():continue
        m,_,_=model(restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1),seed,TRIALS[0],bins)
        before=torch.load(PRIOR/f'seed{seed}/control/best.pt',map_location='cuda',weights_only=True)
        for k,v in m.state_dict().items():torch.testing.assert_close(v,before['state_dict'][k],atol=0,rtol=0)
        if json.loads((path/'history.json').read_text())!=json.loads((PRIOR/f'seed{seed}/control/history.json').read_text()):
            raise ValueError('matched control training history changed')
        a,k,_=parameters(m,cache);curve=probability_curve(a,k)
        with np.load(PRIOR/f'seed{seed}/control/tuning_predictions.npz') as z:
            np.testing.assert_array_equal(z['hadm_ids'],d.arrays['hadm_ids'][ix])
            np.testing.assert_array_equal(z['times'],d.times[ix]);np.testing.assert_array_equal(z['events'],d.events[ix])
            error=float(np.abs(z['curve']-curve).max())
        if error>3e-7:raise ValueError('matched control risk changed')
        checks[str(seed)]=dict(maximum_error=error,full_100_epoch_history_identical=True,all_weight_tensors_identical=True);del m
    write_json(ROOT/'control_replay.json',dict(completed_controls=checks,code_sha256=sha256(Path(__file__)),
        comparison='third batch control vs second batch control, same tuning subset',goal_achieved=False))
    print(json.dumps(dict(status='completed_controls_replayed',controls=checks)),flush=True)


if __name__=='__main__':run()
