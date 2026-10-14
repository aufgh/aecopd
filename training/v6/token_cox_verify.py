"""重载保存的PCA/尺度/双原因Cox参数，不重新拟合，对照预测逐折复现。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.token_data import load_inputs,prepare
from training.v6.token_model import TokenEncoder,daily_probabilities
from training.v6.train import write_json
from training.baselines.continuous_competing_cox import CauseSpecificCox,CompetingCoxModel,predict_cumulative_incidence


def verify(root):
    folder=root/'cox_control';report=json.loads((folder/'report.json').read_text());cfg=json.loads((root/'config.json').read_text())
    if sha256(Path('training/v6/token_cox_control.py'))!=report['source_sha256']:raise ValueError('Cox source changed')
    with np.load(folder/'model.npz') as z:a={k:z[k] for k in z.files}
    causes=[CauseSpecificCox(a[n+'_coefficients'],a[n+'_times'],a[n+'_increments'],True) for n in ('readmission','death')]
    model=CompetingCoxModel(*causes,365.,report['selected_l2'])
    d,c,h,ext,med,provenance=load_inputs(*(Path(cfg[k]) for k in ('reference','compact_dir','extension','medication')))
    if d.provenance!=report['data_provenance'] or provenance!=report['extensions']:raise ValueError('Cox data lineage differs')
    encoder=TokenEncoder(flat=True).cpu().eval();boundary=29*20+7*4+21*23+8*2;errors={}
    for f in ('validation','test'):
        ix=np.flatnonzero(d.arrays['patient_splits']==f);cache=prepare(d,c,ext,med,ix,device='cpu')
        with torch.no_grad():raw=np.concatenate([encoder.flat_features(b).numpy() for _,_,b,_,_ in cache])
        context=np.concatenate([h.numpy() for _,h,_,_,_ in cache]);del cache
        blocks=[(raw[:,:boundary]-a['pca0_mean'])@a['pca0_components'].T,
                (raw[:,boundary:]-a['pca1_mean'])@a['pca1_components'].T]
        x=np.clip((np.concatenate((context,*blocks),1)-a['center'])/a['scale'],-10,10)
        curve=predict_cumulative_incidence(model,x,np.arange(366));daily,conditional=daily_probabilities(curve)
        with np.load(folder/(f+'_predictions.npz')) as z:
            if not np.array_equal(z['hadm_ids'],d.arrays['hadm_ids'][ix]) or not np.array_equal(z['events'],d.events[ix]):raise ValueError('Cox indices differ')
            err=float(np.max(np.abs(curve-z['curve'])));de=float(np.max(np.abs(daily-z['daily_event_probability'])))
        if max(err,de)>2e-6:raise ValueError('Cox/PCA replay differs')
        errors[f]=dict(curve_error=err,daily_error=de)
    proof=dict(status='independently_verified_cox',report_sha256=sha256(folder/'report.json'),model_sha256=sha256(folder/'model.npz'),
        saved_transform_reloaded_without_fitting=True,reload_errors=errors)
    write_json(folder/'independent_verification.json',proof);print(json.dumps(proof))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'));torch.set_num_threads(2);verify(p.parse_args().root)
