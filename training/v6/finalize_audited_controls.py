"""神经已完成后的简单对照恢复；原生Booster避开sklearn tags版本不兼容。"""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import torch
import xgboost as xgb
from patient_data.export_mimiciv import sha256
from training.v6.audited_experiment import OUTPUT,load,prepare,restrict,summary,save_curve,render
from training.v6.audited_model import flat_features
from training.v6.audited_simple_controls import breslow,tree_curve
from training.baselines.continuous_competing_cox import predict_cumulative_incidence
from training.v6.evaluation import metrics,public_metrics,paired_interval
from training.v6.train import write_json


class NativeTree:
    def __init__(self,booster):
        self.booster=booster;self.booster.set_param({'device':'cpu'})
        self.end=int(booster.attr('best_iteration'))+1
    def predict(self,x,output_margin=False):
        return self.booster.predict(xgb.DMatrix(x),output_margin=output_margin,iteration_range=(0,self.end))


def run(root):
    begin=time.perf_counter();report=json.loads((root/'report.json').read_text())
    if report['status']!='neural_completed':raise ValueError('not a pending neural-completed run')
    if (root/'simple_controls/report.json').exists():raise FileExistsError('complete controls already exist')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.use_deterministic_algorithms(True)
    cfg,d,c,h,a,audit=load();idx={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('train','validation','test')}
    selected=report['selected_arm'];compact=selected!='full_pool';output=root/'simple_controls'
    previous=joblib.load(output/'cox.joblib')
    if previous['cox'].l2!=100.:raise ValueError('previous completed Cox selection differs')
    raw={};history={}
    for fold in ('train','validation'):
        batches=restrict(prepare(d,c,a,idx[fold]),a,compact)
        raw[fold]=np.concatenate([flat_features(b).cpu().numpy() for _,_,b,_,_ in batches]);history[fold]=np.concatenate([hi.cpu().numpy() for _,hi,_,_,_ in batches]);del batches
    boundary=a['labs'].shape[1]*25+a['medications'].shape[1]*12+20+8+6
    def features(f,kind):
        clinical=previous['clinical_pca'].transform(raw[f][:,:boundary]) if kind=='cox' else raw[f][:,:boundary]
        return np.concatenate((history[f],clinical,previous['text_pca'].transform(raw[f][:,boundary:])),1)
    tx={f:features(f,'tree').astype(np.float32) for f in raw};candidates={};trees={};hazards={}
    for seed in (42,7):
        candidates[str(seed)]={}
        for depth in (2,3):
            fitted=[];hz=[]
            for cause in (1,2):
                yt=np.maximum(d.times[idx['train']],1e-6)*np.where(d.events[idx['train']]==cause,1.,-1.)
                yv=np.maximum(d.times[idx['validation']],1e-6)*np.where(d.events[idx['validation']]==cause,1.,-1.)
                model=xgb.XGBRegressor(objective='survival:cox',n_estimators=300,max_depth=depth,learning_rate=.03,
                  min_child_weight=10,reg_lambda=10,subsample=.8,colsample_bytree=.8,tree_method='hist',device='cuda',
                  random_state=seed,n_jobs=2,early_stopping_rounds=30)
                model.fit(tx['train'],yt,eval_set=[(tx['validation'],yv)],verbose=False)
                file=output/f'native_seed{seed}_depth{depth}_cause{cause}.ubj';model.get_booster().save_model(file)
                native=xgb.Booster();native.load_model(file);wrapper=NativeTree(native)
                risk=np.exp(np.clip(wrapper.predict(tx['train'],output_margin=True),-20,20))
                fitted.append(wrapper);hz.append(breslow(d.times[idx['train']],d.events[idx['train']],risk,cause))
            curve=tree_curve(fitted,hz,tx['validation'])
            score=public_metrics(metrics(d.times[idx['validation']],d.events[idx['validation']],curve[:,:,1]))['integrated_brier_7_365']
            candidates[str(seed)][str(depth)]=dict(validation_ibs=score,best_iterations=[m.end-1 for m in fitted])
            trees[seed,depth]=fitted;hazards[seed,depth]=hz
            print(json.dumps(dict(recovered_tree_seed=seed,depth=depth,validation_ibs=score)),flush=True)
    choices={s:min((2,3),key=lambda k:candidates[str(s)][str(k)]['validation_ibs']) for s in (42,7)}
    write_json(output/'selected_before_test.json',dict(selected_depth=choices,candidates=candidates))
    batches=restrict(prepare(d,c,a,idx['test']),a,compact)
    raw['test']=np.concatenate([flat_features(b).cpu().numpy() for _,_,b,_,_ in batches]);history['test']=np.concatenate([hi.cpu().numpy() for _,hi,_,_,_ in batches]);del batches
    tx['test']=features('test','tree').astype(np.float32);results={};comparisons={};errors={}
    for name in ('cox','xgb42','xgb7'):
        results[name]={};comparisons[name]={}
        if name!='cox':
            s=int(name[3:]);depth=choices[s];fitted=trees[s,depth];hz=hazards[s,depth];reload=[]
            for cause in (1,2):
                booster=xgb.Booster();booster.load_model(output/f'native_seed{s}_depth{depth}_cause{cause}.ubj');reload.append(NativeTree(booster))
            np.savez_compressed(output/f'{name}_baseline.npz',**{f'{j}_{key}':v for j,pair in enumerate(hz) for key,v in zip(('time','increment'),pair)})
        for fold in ('validation','test'):
            if name=='cox':
                x=np.clip((features(fold,'cox')-previous['center'])/previous['scale'],-10,10)
                curve=predict_cumulative_incidence(previous['cox'],x,np.arange(366))
                again=predict_cumulative_incidence(joblib.load(output/'cox.joblib')['cox'],x,np.arange(366))
            else:curve=tree_curve(fitted,hz,tx[fold]);again=tree_curve(reload,hz,tx[fold])
            err=float(np.max(np.abs(curve-again)))
            if err>1e-12:raise ValueError('native replay differs')
            errors[f'{name}/{fold}']=err;results[name][fold]=summary(curve,d,idx[fold]);save_curve(output/f'{name}_{fold}_predictions.npz',curve,d,idx[fold])
            comparisons[name][fold]={}
            for s in (42,7):
                with np.load(root/f'seed{s}'/selected/(fold+'_predictions.npz')) as z:reference=z['curve']
                comparisons[name][fold][str(s)]=paired_interval(d.arrays['subject_ids'][idx[fold]],d.times[idx[fold]],d.events[idx[fold]],curve[:,:,1],reference[:,:,1],seed=s)
    joblib.dump(dict(clinical_pca=previous['clinical_pca'],text_pca=previous['text_pca'],center=previous['center'],scale=previous['scale'],boundary=boundary,compact=compact),output/'preprocessing.joblib')
    controls=dict(status='completed_simple_controls',neural_selected_recipe=selected,compact_recipe=compact,
      cox_selected_l2=100.,cox_candidates_from_original_training_log={'10':.09549750101355556,'100':.09548488470392937},
      xgb_selected_depth=choices,xgb_candidates=candidates,folds=results,comparisons_to_neural_selected=comparisons,reload_errors=errors,
      cox_features=66,xgb_features=tx['train'].shape[1],clinical_pca=16,text_pca=16,train_only_pca=True,xgboost_version=xgb.__version__,
      source_sha256=sha256(Path('training/v6/audited_simple_controls.py')),completion_code_sha256=sha256(Path(__file__)),
      elapsed_seconds=time.perf_counter()-begin,repair='sklearn wrapper reload incompatible tags; native Booster UBJ save/reload verified; reused fitted Cox and PCAs; only trees refitted',
      limits=['Refitted H34; not frozen baseline for Cox/XGB.','Reused test and no multiplicity adjustment; not external paper replication.'])
    write_json(output/'report.json',controls)
    report.update(status='completed_neural_and_simple_controls',simple_recovery_seconds=controls['elapsed_seconds'],completion_code_sha256=sha256(Path(__file__)),
      numerical_notes=['All10 neural trainings and saved predictions preserved; no neural retraining/reselection.','Native XGBoost serialization avoids sklearn1.6 estimator tags incompatibility; simple tree candidates refitted, validation selection unchanged policy.'])
    write_json(root/'report.json',report);render(root,report)
    print(json.dumps(dict(status=report['status'],seconds=controls['elapsed_seconds'],selected=selected)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=OUTPUT);run(p.parse_args().root)
