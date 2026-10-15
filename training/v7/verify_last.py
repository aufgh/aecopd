"""新进程完整重放 last，逐项对照末轮记录，并二次核对每日曲线。"""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from patient_data.export_mimiciv import sha256
from training.v6 import unified_longitudinal_experiment as parent
from training.v6.train import write_json
from training.v7.data import cache
from training.v7.experiment import setup, predict, ROOT, ARMS
from training.v7.model import V7Model


def calculate():
    started=time.perf_counter()
    proof=json.loads((ROOT/'verification.json').read_text())
    if proof['maximum_error']!=0 or proof['report_sha256']!=sha256(ROOT/'report.json'):
        raise ValueError('best proof invalid')
    d,c,ext,lab,relation,train,tune,baseline,drug,rad,contract,source,_=setup()
    saved=json.loads((ROOT/'contract.json').read_text())
    saved.pop('source_setup_seconds');contract.pop('source_setup_seconds')
    if contract!=saved:raise ValueError('last reconstruction contract differs')
    batches=cache(c,ext,lab,relation,drug,rad,tune)
    result={};artifacts={};maximum=0.
    for arm in ARMS:
        folder=ROOT/arm
        last=torch.load(folder/'last.pt',map_location='cuda',weights_only=True)
        model=V7Model(baseline,drug['drug_parent']).cuda()
        model.load_state_dict(last['state_dict'])
        for n,p in model.baseline.state_dict().items():
            torch.testing.assert_close(p.cpu(),baseline.state_dict()[n],rtol=0,atol=0)
        a,k,curve,nll=predict(model,batches,d)
        m=parent.detail(d,tune,curve,nll)
        tail=json.loads((folder/'history.json').read_text())[-1]
        if (tail['epoch']!=last['epoch'] or m['annual']!=tail['annual'] or nll!=tail['tuning_nll']
            or m['readmission']['integrated_brier_7_365']!=tail['tuning_ibs']
            or m['death']['integrated_brier_7_365']!=tail['tuning_death_ibs']):
            raise ValueError('last metrics differ from actual terminal epoch')
        target=folder/'last_replay.npz'
        errors={}
        if target.exists():
            with np.load(target) as z:
                for name,value in (('hadm_ids',d.arrays['hadm_ids'][tune]),('subject_ids',d.arrays['subject_ids'][tune]),
                                   ('times',d.times[tune]),('events',d.events[tune])):
                    np.testing.assert_array_equal(z[name],value)
                for name,value in (('log_amplitude',a),('shape',k),('curve',curve)):
                    errors[name]=float(np.abs(z[name]-value).max())
            maximum=max(maximum,max(errors.values()))
            if maximum!=0:raise ValueError('last independent curve replay differs')
        else:
            parent.save(target,d,tune,a,k,curve)
        if np.max(np.abs(curve.sum(2)-1))>1e-10 or (np.diff(curve[:,:,1:],axis=1)<-1e-12).any():
            raise ValueError('last curve conservation/monotonicity')
        result[arm]=dict(epoch=last['epoch'],metrics_equal_to_terminal_epoch=True,h34_equal=True,
            readmission_ibs=m['readmission']['integrated_brier_7_365'],death_ibs=m['death']['integrated_brier_7_365'],
            prediction_errors=errors)
        artifacts[str(folder/'last.pt')]=sha256(folder/'last.pt')
        artifacts[str(target)]=sha256(target)
        del model;torch.cuda.empty_cache()
    return dict(arms=result,maximum_error=maximum,all_sources_reconstructed=True,
                report_sha256=sha256(ROOT/'report.json'),artifact_sha256=artifacts,
                source_sha256={str(Path(__file__)):sha256(Path(__file__))},seconds=time.perf_counter()-started)


if __name__=='__main__':
    result=calculate()
    target=ROOT/('last_verification.json' if all(v['prediction_errors'] for v in result['arms'].values()) else 'last_first_replay.json')
    write_json(target,result)
    print(json.dumps(dict(maximum_error=result['maximum_error'],proof=str(target))),flush=True)
