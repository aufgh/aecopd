"""单种子确认的独立重放；不再次锁定或替换候选。"""
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,candidate,REFERENCE
from training.v6.regularization_search import ROOT,SEED
from training.v6.regularization_model import TRIALS
from training.v6.module_surgery import SurgeryModel,probability_curve
from training.v6.module_surgery_experiment import batches,parameters,summary
from training.v6.token_model import daily_probabilities
from training.v6.train import write_json


def run():
    r=json.loads((ROOT/'confirmation_report.json').read_text())
    lock=json.loads((ROOT/'nomination_lock.json').read_text())
    if r['lock']!=lock:raise ValueError('nomination differs')
    d,c,h,ext,base,bins,_,confirm,_,_=checked()
    cache=batches(d,c,ext,confirm);trials={x.name:x for x in TRIALS};errors={}
    for name in ['v6_control',*lock['nominated'],'official_V6']:
        if name=='official_V6':
            m=SurgeryModel(base,SEED,'control',bins).cuda();path=REFERENCE/f'seed{SEED}/control/best.pt'
            m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
            digest=sha256(path)
        else:m,info=candidate(base,trials[name],bins);digest=info['checkpoint_sha256']
        if digest!=r['models'][name]['checkpoint_sha256']:raise ValueError('weight changed')
        a,k,nll=parameters(m,cache);curve=probability_curve(a,k)
        with np.load(ROOT/'confirmation'/name/'predictions.npz') as z:
            for key,expected in (('hadm_ids',d.arrays['hadm_ids'][confirm]),('subject_ids',d.arrays['subject_ids'][confirm]),('times',d.times[confirm]),('events',d.events[confirm])):
                np.testing.assert_array_equal(z[key],expected)
            error=float(np.abs(curve-z['curve']).max());daily,conditional=daily_probabilities(curve)
            np.testing.assert_allclose(daily,z['daily_event_probability'],atol=3e-7,rtol=0)
            np.testing.assert_allclose(conditional,z['conditional_daily_event_probability'],atol=3e-7,rtol=0)
        if error>3e-7:raise ValueError('confirmation replay differs')
        if summary(curve,d,confirm,nll)!=r['models'][name]['metrics']:raise ValueError('metric replay differs')
        errors[name]=error;del m
    write_json(ROOT/'confirmation_verification.json',dict(status='single_seed_confirmation_independently_verified',models=len(errors),
        maximum_error=max(errors.values()),replay_errors=errors,confirmation_report_sha256=sha256(ROOT/'confirmation_report.json'),
        verifier_sha256=sha256(Path(__file__)),test_not_evaluated=True,goal_achieved=False))
    print(json.dumps(dict(status='single_seed_confirmation_independently_verified',models=len(errors),maximum_error=max(errors.values()))),flush=True)


if __name__=='__main__':run()
