"""相同严格终点的成熟双原因Cox算法对照，降维只拟合训练患者。"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.decomposition import PCA
from patient_data.export_mimiciv import sha256
from training.v6.token_data import load_inputs,prepare
from training.v6.token_model import TokenEncoder,daily_probabilities
from training.v6.evaluation import metrics,public_metrics,paired_interval,calibration_bins
from training.v6.train import write_json
from training.baselines.continuous_competing_cox import fit_competing_cox,predict_cumulative_incidence


def run(root):
    output=root/'cox_control'
    if (output/'report.json').exists():raise FileExistsError(output)
    if not all((root/f'seed{seed}/flat_mlp/{fold}_predictions.npz').exists() for seed in (42,7) for fold in ('validation','test')):
        raise RuntimeError('Complete neural finalization before Cox paired comparison')
    # 初次并行运行仅留下未完成预测，允许重生成自己的临时文件；完整报告不可覆盖。
    output.mkdir(exist_ok=True)
    cfg=json.loads((root/'config.json').read_text());start=time.perf_counter()
    d,c,h,ext,med,provenance=load_inputs(*(Path(cfg[k]) for k in ('reference','compact_dir','extension','medication')))
    ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('train','validation','test')}
    # 不复用神经网络的可学习投影；全部原表格输入和四字段冻结向量均值。
    encoder=TokenEncoder(flat=True).cpu().eval();raw={};context={}
    for f in ('train','validation'):  # 选模前不读取测试输入
        cache=prepare(d,c,ext,med,ix[f],size=256,device='cpu')
        with torch.no_grad():raw[f]=np.concatenate([encoder.flat_features(b).numpy() for _,_,b,_,_ in cache])
        context[f]=np.concatenate([h.numpy() for _,h,_,_,_ in cache]);del cache
    boundary=29*20+7*4+21*23+8*2
    pcas=[PCA(16,svd_solver='randomized',random_state=42),PCA(16,svd_solver='randomized',random_state=42)]
    pcas[0].fit(raw['train'][:,:boundary]);pcas[1].fit(raw['train'][:,boundary:])
    def transform(x,h):return np.concatenate((h,pcas[0].transform(x[:,:boundary]),pcas[1].transform(x[:,boundary:])),1)
    x={f:transform(raw[f],context[f]) for f in ('train','validation')}
    center=x['train'].mean(0);scale=x['train'].std(0);scale[scale<1e-7]=1
    for f in x:x[f]=np.clip((x[f]-center)/scale,-10,10)
    candidates={};fitted={};curves={}
    for l2 in (10.,100.):
        fitted[l2]=fit_competing_cox(x['train'],d.times[ix['train']],d.events[ix['train']],horizon_days=365.,l2=l2,max_iter=80)
        curve=predict_cumulative_incidence(fitted[l2],x['validation'],np.arange(366))
        detail=public_metrics(metrics(d.times[ix['validation']],d.events[ix['validation']],curve[:,:,1]))
        candidates[str(l2)]=dict(validation=detail,readmission_converged=fitted[l2].readmission.converged,death_converged=fitted[l2].death.converged)
        curves[l2]=curve;print(json.dumps(dict(cox_l2=l2,validation_ibs=detail['integrated_brier_7_365'],converged=[fitted[l2].readmission.converged,fitted[l2].death.converged])),flush=True)
    best=min(fitted,key=lambda l2:candidates[str(l2)]['validation']['integrated_brier_7_365']);model=fitted[best]
    cache=prepare(d,c,ext,med,ix['test'],size=256,device='cpu')
    with torch.no_grad():raw['test']=np.concatenate([encoder.flat_features(b).numpy() for _,_,b,_,_ in cache])
    context['test']=np.concatenate([h.numpy() for _,h,_,_,_ in cache]);del cache
    x['test']=np.clip((transform(raw['test'],context['test'])-center)/scale,-10,10)
    curves[best]=None;folds={};comparison={}
    for f in ('validation','test'):
        curve=predict_cumulative_incidence(model,x[f],np.arange(366));daily,conditional=daily_probabilities(curve)
        np.savez_compressed(output/(f+'_predictions.npz'),hadm_ids=d.arrays['hadm_ids'][ix[f]],subject_ids=d.arrays['subject_ids'][ix[f]],
            times=d.times[ix[f]],events=d.events[ix[f]],curve=curve,daily_event_probability=daily,conditional_daily_event_probability=conditional)
        folds[f]=dict(readmission=public_metrics(metrics(d.times[ix[f]],d.events[ix[f]],curve[:,:,1],ordering=True)),
                      death=public_metrics(metrics(d.times[ix[f]],d.events[ix[f]],curve[:,:,2],cause=2,ordering=True)),
                      calibration={str(day):calibration_bins(d.times[ix[f]],d.events[ix[f]],curve[:,:,1],day=day) for day in (30,90,180,365)})
        comparison[f]={}
        for seed in cfg['seeds']:
            with np.load(root/f'seed{seed}/flat_mlp/{f}_predictions.npz') as z:reference=z['curve']
            comparison[f][str(seed)]=paired_interval(d.arrays['subject_ids'][ix[f]],d.times[ix[f]],d.events[ix[f]],curve[:,:,1],reference[:,:,1],seed=seed)
    components=dict(center=center,scale=scale)
    for j,pca in enumerate(pcas):components[f'pca{j}_mean']=pca.mean_;components[f'pca{j}_components']=pca.components_
    for event,cause in (('readmission',model.readmission),('death',model.death)):
        components[event+'_coefficients']=cause.coefficients;components[event+'_times']=cause.event_times;components[event+'_increments']=cause.baseline_hazard_increments
    np.savez_compressed(output/'model.npz',**components)
    report=dict(status='completed_cox_control',selection='validation readmission IBS7-365',selected_l2=best,candidates=candidates,folds=folds,
        comparisons_to_flat_mlp=comparison,features=66,clinical_pca=16,text_pca=16,context=34,train_only_pca=True,
        elapsed_seconds=time.perf_counter()-start,source_sha256=sha256(Path(__file__)),data_provenance=d.provenance,extensions=provenance,
        limits=['Mature cause-specific Cox algorithm adapted, not an external paper replication.',
                'PCA representation differs from token encoder; no causality or blind test claim.',
                'Deterministic Cox fitted once, not duplicated as two training seeds.'])
    write_json(output/'report.json',report);print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'));run(p.parse_args().root)
