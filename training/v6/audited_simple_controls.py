"""固定同结局的成熟Cox/XGBoost-Cox；数值PCA不是重新编码患者原文。"""
import json
from pathlib import Path
import time
import joblib
import numpy as np
import torch
import xgboost as xgb
from sklearn.decomposition import PCA
from patient_data.export_mimiciv import sha256
from training.baselines.continuous_competing_cox import fit_competing_cox, predict_cumulative_incidence
from training.v6.evaluation import metrics,public_metrics,paired_interval
from training.v6.train import write_json


def breslow(times,events,risk,cause):
    t=np.maximum(times,1e-6);event_times=np.unique(t[events==cause]);increments=[]
    for day in event_times:increments.append(float(((events==cause)&(t==day)).sum()/risk[t>=day].sum()))
    return event_times,np.asarray(increments)


def tree_curve(models,hazards,features):
    """共同两原因生存量；按Breslow跳跃的危险增量分配事件概率质量。"""
    predictions=[np.exp(np.clip(model.predict(features,output_margin=True),-20,20)) for model in models]
    grid=np.unique(np.r_[0.,hazards[0][0],hazards[1][0],np.arange(366)])
    inc=np.zeros((len(grid),2))
    for e,(times,delta) in enumerate(hazards):inc[np.searchsorted(grid,times),e]=delta
    output=[];day_index=np.searchsorted(grid,np.arange(366))
    for start in range(0,len(features),512):
        risk=np.stack([p[start:start+512] for p in predictions],1)
        increments=risk[:,None,:]*inc[None];total=increments.sum(2);cumulative=np.cumsum(total,1)
        previous=np.concatenate((np.zeros_like(total[:,:1]),cumulative[:,:-1]),1)
        mass=np.exp(-previous[:,:,None])*(-np.expm1(-total))[:,:,None]*increments/np.maximum(total[:,:,None],1e-300)
        output.append(np.concatenate((np.exp(-cumulative)[:,:,None],np.cumsum(mass,1)),2)[:,day_index])
    return np.concatenate(output)


def run(root,cfg,d,c,h,a,idx,compact,selected):
    from training.v6.audited_experiment import prepare,restrict,summary,save_curve
    from training.v6.audited_model import flat_features
    begin=time.perf_counter();output=root/'simple_controls';output.mkdir()
    raw={};history={}
    for fold in ('train','validation'):
        batches=restrict(prepare(d,c,a,idx[fold]),a,compact)
        raw[fold]=np.concatenate([flat_features(b).cpu().numpy() for _,_,b,_,_ in batches])
        history[fold]=np.concatenate([hi.cpu().numpy() for _,hi,_,_,_ in batches]);del batches
    boundary=a['labs'].shape[1]*25+a['medications'].shape[1]*12+20+8+6
    # 四字段的均值仍只是简单对照的文本表示，不冒充无早期池化。
    pca_lab=PCA(16,svd_solver='randomized',random_state=42).fit(raw['train'][:,:boundary])
    pca_text=PCA(16,svd_solver='randomized',random_state=42).fit(raw['train'][:,boundary:])
    def features(f,kind):
        clinical=pca_lab.transform(raw[f][:,:boundary]) if kind=='cox' else raw[f][:,:boundary]
        return np.concatenate((history[f],clinical,pca_text.transform(raw[f][:,boundary:])),1)
    x={f:features(f,'cox') for f in ('train','validation')}
    center=x['train'].mean(0);scale=x['train'].std(0);scale[scale<1e-7]=1
    x={f:np.clip((value-center)/scale,-10,10) for f,value in x.items()}
    candidates={};models={}
    for l2 in (10.,100.):
        model=fit_competing_cox(x['train'],d.times[idx['train']],d.events[idx['train']],horizon_days=365.,l2=l2,max_iter=80)
        curve=predict_cumulative_incidence(model,x['validation'],np.arange(366))
        score=public_metrics(metrics(d.times[idx['validation']],d.events[idx['validation']],curve[:,:,1]))['integrated_brier_7_365']
        models[l2]=model;candidates[str(l2)]=dict(validation_ibs=score,converged=[model.readmission.converged,model.death.converged])
        print(json.dumps(dict(simple='Cox',l2=l2,validation_ibs=score)),flush=True)
    choice=min(models,key=lambda k:candidates[str(k)]['validation_ibs']);cox=models[choice]
    tree_models={};tree_hazards={};tree_candidates={}
    tx={f:features(f,'xgb').astype(np.float32) for f in ('train','validation')}
    for seed in (42,7):
        tree_candidates[str(seed)]={}
        for depth in (2,3):
            fitted=[];hazards=[]
            for cause in (1,2):
                yt=np.maximum(d.times[idx['train']],1e-6)*np.where(d.events[idx['train']]==cause,1.,-1.)
                yv=np.maximum(d.times[idx['validation']],1e-6)*np.where(d.events[idx['validation']]==cause,1.,-1.)
                model=xgb.XGBRegressor(objective='survival:cox',n_estimators=300,max_depth=depth,learning_rate=.03,
                  min_child_weight=10,reg_lambda=10,subsample=.8,colsample_bytree=.8,tree_method='hist',device='cuda',
                  random_state=seed,n_jobs=2,early_stopping_rounds=30)
                model.fit(tx['train'],yt,eval_set=[(tx['validation'],yv)],verbose=False)
                risk=np.exp(np.clip(model.predict(tx['train'],output_margin=True),-20,20))
                fitted.append(model);hazards.append(breslow(d.times[idx['train']],d.events[idx['train']],risk,cause))
            curve=tree_curve(fitted,hazards,tx['validation'])
            score=public_metrics(metrics(d.times[idx['validation']],d.events[idx['validation']],curve[:,:,1]))['integrated_brier_7_365']
            tree_models[seed,depth]=fitted;tree_hazards[seed,depth]=hazards
            tree_candidates[str(seed)][str(depth)]=dict(validation_ibs=score,best_iterations=[int(m.best_iteration) for m in fitted])
            print(json.dumps(dict(simple='XGBoost-Cox',seed=seed,depth=depth,validation_ibs=score)),flush=True)
    choices={seed:min((2,3),key=lambda dep:tree_candidates[str(seed)][str(dep)]['validation_ibs']) for seed in (42,7)}
    # 所有超参数选定后才读取测试输入。
    batches=restrict(prepare(d,c,a,idx['test']),a,compact)
    raw['test']=np.concatenate([flat_features(b).cpu().numpy() for _,_,b,_,_ in batches])
    history['test']=np.concatenate([hi.cpu().numpy() for _,hi,_,_,_ in batches]);del batches
    x['test']=np.clip((features('test','cox')-center)/scale,-10,10);tx['test']=features('test','xgb').astype(np.float32)
    joblib.dump(dict(cox=cox,center=center,scale=scale,clinical_pca=pca_lab,text_pca=pca_text),output/'cox.joblib')
    reload=joblib.load(output/'cox.joblib');results={};comparisons={};errors={}
    for name in ('cox','xgb42','xgb7'):
        results[name]={};comparisons[name]={}
        if name!='cox':
            seed=int(name[3:]);depth=choices[seed];fitted=tree_models[seed,depth];hazards=tree_hazards[seed,depth]
            reloaded=[]
            for j,m in enumerate(fitted):
                m.save_model(output/f'{name}_cause{j+1}.json');fresh=xgb.XGBRegressor();fresh.load_model(output/f'{name}_cause{j+1}.json');reloaded.append(fresh)
            np.savez_compressed(output/f'{name}_baseline.npz',**{f'{j}_{kind}':v for j,pair in enumerate(hazards) for kind,v in zip(('time','increment'),pair)})
        for fold in ('validation','test'):
            if name=='cox':
                curve=predict_cumulative_incidence(cox,x[fold],np.arange(366));again=predict_cumulative_incidence(reload['cox'],x[fold],np.arange(366))
            else:
                curve=tree_curve(fitted,hazards,tx[fold]);again=tree_curve(reloaded,hazards,tx[fold])
            error=float(np.max(np.abs(curve-again)))
            if error>1e-9:raise ValueError('simple model reload differs')
            errors[f'{name}/{fold}']=error;results[name][fold]=summary(curve,d,idx[fold]);save_curve(output/f'{name}_{fold}_predictions.npz',curve,d,idx[fold])
            comparisons[name][fold]={}
            for seed in (42,7):
                with np.load(root/f'seed{seed}'/selected/(fold+'_predictions.npz')) as z:reference=z['curve']
                comparisons[name][fold][str(seed)]=paired_interval(d.arrays['subject_ids'][idx[fold]],d.times[idx[fold]],d.events[idx[fold]],
                     curve[:,:,1],reference[:,:,1],seed=seed)
    joblib.dump(dict(clinical_pca=pca_lab,text_pca=pca_text,center=center,scale=scale,boundary=boundary,compact=compact),output/'preprocessing.joblib')
    report=dict(status='completed_simple_controls',neural_selected_recipe=selected,compact_recipe=compact,
      cox_selected_l2=choice,cox_candidates=candidates,xgb_selected_depth=choices,xgb_candidates=tree_candidates,folds=results,
      comparisons_to_neural_selected=comparisons,reload_errors=errors,cox_features=x['train'].shape[1],xgb_features=tx['train'].shape[1],
      clinical_pca=16,text_pca=16,train_only_pca=True,xgboost_version=xgb.__version__,elapsed_seconds=time.perf_counter()-begin,
      source_sha256=sha256(Path(__file__)),limits=['Refitted H34; mature algorithm adaptation not exact external reproduction.',
       'Breslow risks integrated jointly for competing events; outside followup unknown.',
       'Repeated discharges remain in same patient fold; reused test and no multiplicity adjustment.'])
    write_json(output/'report.json',report)
    print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'])),flush=True)
