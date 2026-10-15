"""锁定 best 后的内容敏感性诊断，0新训练，不作为重新选点依据。"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from patient_data.export_mimiciv import sha256
from training.v6.train import write_json
from training.v7.experiment import ROOT, ARMS, setup, predict
from training.v7.data import cache
from training.v7.model import V7Model


def altered(b, source):
    if source=='text':
        return replace(b,text_vectors=b.text_vectors[:0],text_index=b.text_index[:0],text_slot=b.text_slot[:0])
    if source=='drugs':
        return replace(b,med_present=torch.zeros_like(b.med_present),coarse_meds=torch.zeros_like(b.coarse_meds))
    if source=='labs':
        return replace(b,labs=torch.zeros_like(b.labs),eos=torch.zeros_like(b.eos))
    if source=='radiology':
        return replace(b,rad_present=torch.zeros_like(b.rad_present))
    if source=='history':
        return replace(b,past_present=torch.zeros_like(b.past_present))
    raise ValueError(source)


@torch.no_grad()
def calculate():
    proof=json.loads((ROOT/'verification.json').read_text())
    if proof['maximum_error']!=0 or proof['report_sha256']!=sha256(ROOT/'report.json'):
        raise ValueError('best verification changed')
    d,c,ext,lab,relation,train,tune,baseline,drug,rad,contract,source,_=setup()
    batches=cache(c,ext,lab,relation,drug,rad,tune)
    result={}
    for arm in ARMS:
        model=V7Model(baseline,drug['drug_parent']).cuda().eval()
        checkpoint=torch.load(ROOT/arm/'best.pt',map_location='cuda',weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        _,_,curve,_=predict(model,batches,d)
        with np.load(ROOT/arm/'tuning_predictions.npz') as z:
            np.testing.assert_array_equal(curve,z['curve'])
        deltas=[];representations={n:[] for n in ('text','lab','drug','rad')}
        for _,b in batches:
            out,reps,masks=model(b.to('cuda'),auxiliary=True)
            deltas.append(out.delta.cpu().numpy())
            for n in representations:
                representations[n].append(reps[n][masks[n]].cpu().numpy())
        delta=np.concatenate(deltas)
        reps={n:np.concatenate(v) for n,v in representations.items()}
        sensitivity={}
        for source in ('text','drugs','labs','radiology','history'):
            modified=[(p,altered(b,source)) for p,b in batches]
            _,_,other,_=predict(model,modified,d)
            difference=other[:,:,1:]-curve[:,:,1:]
            sensitivity[source]=dict(annual_readmission_max_absolute_change=float(np.abs(difference[:,365,0]).max()),
                annual_readmission_mean_absolute_change=float(np.abs(difference[:,365,0]).mean()),
                annual_death_mean_absolute_change=float(np.abs(difference[:,365,1]).mean()),
                all_horizons_causes_max_absolute_change=float(np.abs(difference).max()))
        result[arm]=dict(best_epoch=checkpoint['epoch'],delta_mean=delta.mean(0).tolist(),delta_std=delta.std(0).tolist(),
            normalized_source_embedding_mean_coordinate_std={n:float(v.std(0).mean()) if len(v) else None for n,v in reps.items()},
            ablation_sensitivity=sensitivity)
        del model;torch.cuda.empty_cache()
    return dict(results=result,source_sha256={str(Path(__file__)):sha256(Path(__file__)),
        str(ROOT/'report.json'):sha256(ROOT/'report.json'),str(ROOT/'verification.json'):sha256(ROOT/'verification.json')},
        new_training=0,scope='mask selected source in all current/past visits, or remove past relations; frozen H34 unchanged',
        limits='post-selection content sensitivity, not retrained causal/incremental-value ablation; not used for selection')


if __name__=='__main__':
    result=calculate();target=ROOT/'diagnostics.json'
    if target.exists():
        if json.loads(target.read_text())!=result:raise ValueError('diagnostics replay differs')
        write_json(ROOT/'diagnostics_verification.json',dict(maximum_error=0,diagnostics_sha256=sha256(target),new_training=0))
        print('V7 diagnostics replay exact')
    else:
        write_json(target,result);print('V7 diagnostics saved')
