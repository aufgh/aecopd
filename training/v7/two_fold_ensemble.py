"""真正患者隔离的两折：不复用监督基座，统一每日竞争概率。"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import PCA
from scipy.optimize import minimize,minimize_scalar
from xgboost import XGBClassifier
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json
from training.v6.risk import RiskOutput,competing_nll
from training.baselines.continuous_competing_cox import (
    fit_competing_cox,predict_cumulative_incidence,CompetingCoxModel,CauseSpecificCox)
from training.v7.cached_research import checked,ROOT as CACHE_ROOT,load_npz
from training.v7.research_features import scaled
from training.v7 import experiment as first

ROOT=Path('training_runs/v7_sequential_research_2026-10-15/two_fold')
PROTOCOL=Path('docs/V7_TWO_FOLD_ENSEMBLE_PROTOCOL_2026-10-15.md')
BOUNDARIES=np.array([0,7,14,30,60,90,180,365])
FAMILIES=('cox','tree','neural')


def patient_folds(subjects,train):
    people=np.unique(subjects[train]);people=np.random.default_rng(42).permutation(people)
    allocation={int(s):i%2 for i,s in enumerate(people)}
    folds=np.full(len(subjects),-1,np.int8)
    folds[train]=[allocation[int(s)] for s in subjects[train]]
    return folds


def preprocess(a,f,fit,saved=None):
    use=np.zeros(len(a['times']),bool);use[fit]=True
    raw=a['context_raw'].copy();present=np.ones_like(raw,bool);present[:,4]=raw[:,5]==0
    if saved is None:
        from training.v5.numeric_preprocessing import robust_parameters
        context_parameters={str(j):robust_parameters(raw[use&present[:,j],j]) for j in (0,2,3,4)}
        lab,lab_parameters=scaled(f['raw'][:,:,1],f['valid'][:,:,1],use)
        age,age_parameters=scaled(np.log1p(f['raw'][:,:,4]),f['valid'][:,:,4],use)
        pc=[]
        for source in ('text','history_text'):
            model=PCA(n_components=16,svd_solver='randomized',random_state=42,iterated_power=3)
            model.fit(a[source][fit]);v=model.transform(a[source][fit]);scale=np.maximum(v.std(0),1e-6)
            pc.append(dict(source=source,components=model.components_,mean=model.mean_,scale=scale))
        saved=dict(context=context_parameters,lab=lab_parameters,age=age_parameters,pca=pc)
    else:
        lab=np.zeros((len(raw),9),np.float32);age=np.zeros_like(lab)
        for j in range(9):
            for target,values,mask,ps in ((lab,f['raw'][:,:,1],f['valid'][:,:,1],saved['lab']),
                                         (age,np.log1p(f['raw'][:,:,4]),f['valid'][:,:,4],saved['age'])):
                p=ps[j]
                if p is not None:target[:,j]=np.where(mask[:,j],np.clip((values[:,j]-p['median'])/p['scale'],-8,8),0)
    for key,p in saved['context'].items():
        j=int(key);raw[:,j]=np.where(present[:,j],np.clip((raw[:,j]-p['median'])/p['scale'],-8,8),0)
    core=np.column_stack((raw,lab,f['valid'][:,:,1],age,f['clinical'],np.log1p(f['los']))).astype(np.float32)
    table=np.column_stack((core,a['regimen'],a['pft'],a['cxr'],a['support'])).astype(np.float32)
    embedded=[]
    for p in saved['pca']:
        values=a[p['source']];z=((values-p['mean'])@p['components'].T)/p['scale']
        observed=np.linalg.norm(values,axis=1)>0;z=np.where(observed[:,None],np.clip(z,-8,8),0)
        embedded.extend((z,observed[:,None]))
    neural=np.column_stack((table,*embedded)).astype(np.float32)
    if not all(np.isfinite(x).all() for x in (core,table,neural)):raise ValueError('invalid inner preprocessing')
    return dict(cox=core,tree=table,neural=neural),saved


def period_rows(x,times,events,indices):
    xx=[];yy=[]
    for j,(left,right) in enumerate(zip(BOUNDARIES[:-1],BOUNDARIES[1:])):
        ix=indices[(times[indices]>left)|((times[indices]==0)&(left==0))]
        if not len(ix):continue
        outcome=np.where((events[ix]>0)&(times[ix]<=right),events[ix],0)
        xx.append(np.column_stack((x[ix],np.full(len(ix),j))));yy.append(outcome)
    return np.concatenate(xx).astype(np.float32),np.concatenate(yy)


def interval_cif(probabilities):
    p=np.asarray(probabilities,dtype=np.float64)
    if p.ndim!=3 or p.shape[1:]!=(7,3) or (p<0).any():raise ValueError('invalid period probability')
    p=p/p.sum(2,keepdims=True);n=len(p);curve=np.zeros((n,366,3));curve[:,0,0]=1.
    survival=np.ones(n);cif=np.zeros((n,2))
    for j,(left,right) in enumerate(zip(BOUNDARIES[:-1],BOUNDARIES[1:])):
        daily_survival=np.clip(p[:,j,0],1e-12,1.)**(1/(right-left))
        share=np.divide(p[:,j,1:],p[:,j,1:].sum(1,keepdims=True),out=np.zeros((n,2)),where=p[:,j,1:].sum(1,keepdims=True)>0)
        for day in range(left+1,right+1):
            cif+=survival[:,None]*(1-daily_survival)[:,None]*share;survival*=daily_survival
            curve[:,day]=np.column_stack((survival,cif))
    return curve


def tree_predict(model,x):
    p=np.stack([model.predict_proba(np.column_stack((x,np.full(len(x),j)))) for j in range(7)],1)
    return interval_cif(p)


class CompactNeural(nn.Module):
    def __init__(self,width):
        super().__init__();self.network=nn.Sequential(nn.Linear(width,256),nn.GELU(),nn.Dropout(.3),
            nn.Linear(256,32),nn.GELU(),nn.Dropout(.3),nn.Linear(32,4))
        with torch.no_grad():self.network[-1].bias[:2].fill_(-2.);self.network[-1].bias[2:].zero_()
    def forward(self,x):
        raw=self.network(x)
        return RiskOutput(raw[:,:2].clamp(-10,6),F.softplus(raw[:,2:])+.1)


@torch.no_grad()
def neural_predict(model,x):
    model.eval();aa=[];kk=[]
    for start in range(0,len(x),512):
        out=model(torch.as_tensor(x[start:start+512],device='cuda'));aa.append(out.log_amplitude.cpu().numpy());kk.append(out.shape.cpu().numpy())
    return first.parent.probability_curve(np.concatenate(aa),np.concatenate(kk))


def fit_neural(x,a,fit):
    torch.manual_seed(42);m=CompactNeural(x.shape[1]).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
    rng=np.random.default_rng(42);tensor=torch.as_tensor(x,device='cuda');m.train()
    for epoch in range(24):
        ix=rng.permutation(fit)
        for start in range(0,len(ix),256):
            p=ix[start:start+256];opt.zero_grad(set_to_none=True);out=m(tensor[p])
            loss=competing_nll(out,torch.as_tensor(a['times'][p],device='cuda'),torch.as_tensor(a['events'][p],device='cuda'))
            if not torch.isfinite(loss):raise ValueError('nonfinite inner neural fit')
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    return m


def save_cox(model,path):
    values={}
    for name,cause in (('readmission',model.readmission),('death',model.death)):
        for key in ('coefficients','event_times','baseline_hazard_increments','converged'):values[name+'_'+key]=getattr(cause,key)
    np.savez_compressed(path,**values)


def load_cox(path):
    z=load_npz(path)
    return CompetingCoxModel(*(CauseSpecificCox(*(z[name+'_'+k] for k in ('coefficients','event_times','baseline_hazard_increments','converged')))
        for name in ('readmission','death')),horizon_days=365.,l2=10.)


def common_weights(curves,times,events):
    days=np.array([30,90,180,365]);causes=np.array([1,2]);target=(events[:,None,None]==causes[None,None,:])&(times[:,None,None]<=days[None,:,None])
    values=curves[:,:,days,1:].reshape(3,-1).T;y=target.reshape(-1)
    def fun(w):return float(np.mean((values@w-y)**2))
    result=minimize(fun,np.ones(3)/3,method='SLSQP',bounds=[(0.,1.)]*3,
        constraints=[dict(type='eq',fun=lambda w:w.sum()-1)],options=dict(ftol=1e-12,maxiter=100))
    if not result.success:raise ValueError('convex weight optimization failed')
    return result.x,fun(result.x)


def temper(curve,temperature):
    survival=curve[:,:-1,0];daily=np.maximum(0,np.diff(curve[:,:,1:],axis=1))
    events=np.divide(daily,survival[:,:,None],out=np.zeros_like(daily),where=survival[:,:,None]>1e-12)
    q=np.concatenate((np.maximum(0,1-events.sum(2))[:,:,None],events),2)
    logits=np.log(np.clip(q,1e-12,1))/temperature;logits-=logits.max(2,keepdims=True)
    p=np.exp(logits);p/=p.sum(2,keepdims=True)
    new=np.zeros_like(curve);new[:,0,0]=1.;s=np.ones(len(curve));cif=np.zeros((len(curve),2))
    for day in range(1,366):
        cif+=s[:,None]*p[:,day-1,1:];s*=p[:,day-1,0];new[:,day]=np.column_stack((s,cif))
    return new


def score(a,ix,curve):
    d=type('Data',(),dict(times=a['times'],events=a['events']))()
    result=first.parent.detail(d,ix,curve,0.);result.pop('nll');return result


def train():
    if ROOT.exists():raise FileExistsError('two-fold exists; preserve scene')
    started=time.perf_counter();a,f,cache_contract=checked();train,tune=a['train'],a['tune'];folds=patient_folds(a['subject_ids'],train)
    ROOT.mkdir();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    paths=(Path(__file__),PROTOCOL,Path('training/tests/test_v7_two_fold.py'),CACHE_ROOT/'cache_contract.json',CACHE_ROOT/'cache.npz')
    write_json(ROOT/'contract.json',dict(source_sha256={str(p):sha256(p) for p in paths},
        folds=2,strict_patient_oof=True,supervised_v7_hidden_used=False,neural_fixed_epochs=24,
        deploy='average two inner fold models',confirmation_test_used=False))
    np.savez_compressed(ROOT/'folds.npz',positions=train,subject_ids=a['subject_ids'][train],fold=folds[train])
    oof=np.empty((3,len(train),366,3));outer=np.zeros((3,len(tune),366,3));details=[]
    for fold in (0,1):
        fit=train[folds[train]!=fold];hold=train[folds[train]==fold];folder=ROOT/f'fold{fold}';folder.mkdir()
        if set(a['subject_ids'][fit])&set(a['subject_ids'][hold]):raise ValueError('patient leakage')
        x,prep=preprocess(a,f,fit)
        # 自有预处理仅含数值数组/尺度；保存后重放，不反序列化外部文件。
        torch.save(prep,folder/'preprocessing.pt');indices=np.searchsorted(train,hold);tick=time.perf_counter()
        model=fit_competing_cox(x['cox'][fit],a['times'][fit],a['events'][fit],l2=10.,max_iter=40,tolerance=1e-6)
        if not model.readmission.converged or not model.death.converged:raise ValueError('inner Cox did not converge')
        save_cox(model,folder/'cox.npz');oof[0,indices]=predict_cumulative_incidence(model,x['cox'][hold],np.arange(366))
        prediction=predict_cumulative_incidence(model,x['cox'][tune],np.arange(366));outer[0]+=prediction/2
        np.savez_compressed(folder/'cox_predictions.npz',hold=hold,hold_curve=oof[0,indices],tune_curve=prediction)
        print(json.dumps(dict(fold=fold,family='cox',seconds=time.perf_counter()-tick)),flush=True)
        tick=time.perf_counter();xx,yy=period_rows(x['tree'],a['times'],a['events'],fit)
        tree=XGBClassifier(n_estimators=100,max_depth=3,learning_rate=.05,min_child_weight=10,reg_lambda=10,
            objective='multi:softprob',num_class=3,tree_method='hist',n_jobs=4,random_state=42)
        tree.fit(xx,yy);tree.save_model(folder/'tree.ubj');oof[1,indices]=tree_predict(tree,x['tree'][hold])
        prediction=tree_predict(tree,x['tree'][tune]);outer[1]+=prediction/2
        np.savez_compressed(folder/'tree_predictions.npz',hold=hold,hold_curve=oof[1,indices],tune_curve=prediction)
        print(json.dumps(dict(fold=fold,family='tree',seconds=time.perf_counter()-tick)),flush=True)
        tick=time.perf_counter();neural=fit_neural(x['neural'],a,fit)
        torch.save(dict(state_dict=neural.state_dict(),width=x['neural'].shape[1]),folder/'neural.pt')
        oof[2,indices]=neural_predict(neural,x['neural'][hold]);prediction=neural_predict(neural,x['neural'][tune]);outer[2]+=prediction/2
        np.savez_compressed(folder/'neural_predictions.npz',hold=hold,hold_curve=oof[2,indices],tune_curve=prediction)
        details.append(dict(fold=fold,fit_admissions=len(fit),hold_admissions=len(hold),
            fit_patients=len(np.unique(a['subject_ids'][fit])),hold_patients=len(np.unique(a['subject_ids'][hold]))))
        print(json.dumps(dict(fold=fold,family='neural',seconds=time.perf_counter()-tick)),flush=True)
        del neural;torch.cuda.empty_cache()
    weights,value=common_weights(oof,a['times'][train],a['events'][train]);weighted=np.einsum('m,mntk->ntk',weights,outer)
    metrics={name:score(a,tune,outer[j]) for j,name in enumerate(FAMILIES)}
    metrics['equal']=score(a,tune,outer.mean(0));metrics['weighted']=score(a,tune,weighted)
    # 温度仅在训练OOF上估计，目标仍是已预定四时点Brier。
    oof_mix=np.einsum('m,mntk->ntk',weights,oof);days=np.array([30,90,180,365])
    y=(a['events'][train,None,None]==np.array([1,2])[None,None,:])&(a['times'][train,None,None]<=days[None,:,None])
    objective=lambda logt:float(np.mean((temper(oof_mix,np.exp(logt))[:,days,1:]-y)**2))
    temperature=minimize_scalar(objective,bounds=(-.3,.3),method='bounded',options=dict(xatol=1e-4))
    if not temperature.success:raise ValueError('OOF temperature fit failed')
    calibrated=temper(weighted,float(np.exp(temperature.x)));metrics['weighted_temperature']=score(a,tune,calibrated)
    np.savez_compressed(ROOT/'predictions.npz',positions=train,tune=tune,oof=oof,outer=outer,
        weighted=weighted,calibrated=calibrated,weights=weights,temperature=np.exp(temperature.x))
    write_json(ROOT/'report.json',dict(metrics=metrics,weights=weights.tolist(),oof_brier=value,
        temperature=float(np.exp(temperature.x)),fold_details=details,seconds=time.perf_counter()-started,
        limits=['two folds use half-training per model','compact neural member, not full V7','outer tuning exposed'],default_replaced=False))


def verify():
    a,f,_=checked();contract=json.loads((ROOT/'contract.json').read_text())
    for p,h in contract['source_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('twofold source changed')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    folds=patient_folds(a['subject_ids'],a['train']);train,tune=a['train'],a['tune'];saved=load_npz(ROOT/'predictions.npz')
    outer=np.zeros_like(saved['outer'])
    for fold in (0,1):
        fit=train[folds[train]!=fold];hold=train[folds[train]==fold];folder=ROOT/f'fold{fold}'
        prep=torch.load(folder/'preprocessing.pt',map_location='cpu',weights_only=False)
        x,_=preprocess(a,f,fit,saved=prep)
        models=[load_cox(folder/'cox.npz'),XGBClassifier(),None];models[1].load_model(folder/'tree.ubj')
        ck=torch.load(folder/'neural.pt',map_location='cuda',weights_only=True)
        models[2]=CompactNeural(ck['width']).cuda();models[2].load_state_dict(ck['state_dict'])
        for j,name in enumerate(FAMILIES):
            predict=lambda values:predict_cumulative_incidence(models[j],values,np.arange(366)) if j==0 else tree_predict(models[j],values) if j==1 else neural_predict(models[j],values)
            source=load_npz(folder/f'{name}_predictions.npz')
            for key,positions in (('hold_curve',hold),('tune_curve',tune)):
                actual=predict(x[name][positions]);np.testing.assert_allclose(actual,source[key],atol=1e-10,rtol=0)
                if key=='tune_curve':outer[j]+=actual/2
            np.testing.assert_allclose(saved['oof'][j,np.searchsorted(train,hold)],source['hold_curve'],atol=0,rtol=0)
        del models;torch.cuda.empty_cache()
    weights,value=common_weights(saved['oof'],a['times'][train],a['events'][train])
    np.testing.assert_allclose(weights,saved['weights'],atol=1e-10,rtol=0)
    np.testing.assert_allclose(outer,saved['outer'],atol=1e-10,rtol=0)
    weighted=np.einsum('m,mntk->ntk',weights,outer)
    np.testing.assert_allclose(weighted,saved['weighted'],atol=1e-10,rtol=0)
    np.testing.assert_allclose(temper(weighted,float(saved['temperature'])),saved['calibrated'],atol=1e-10,rtol=0)
    report=json.loads((ROOT/'report.json').read_text());curves={name:saved['outer'][j] for j,name in enumerate(FAMILIES)}
    curves.update(equal=saved['outer'].mean(0),weighted=saved['weighted'],weighted_temperature=saved['calibrated'])
    for name,curve in curves.items():
        if score(a,tune,curve)!=report['metrics'][name]:raise ValueError('ensemble metrics differ')
        if np.max(np.abs(curve.sum(2)-1))>1e-8 or (np.diff(curve[:,:,1:],axis=1)<-1e-12).any():raise ValueError('invalid ensemble curve')
    write_json(ROOT/'verification.json',dict(maximum_error_tolerance=1e-10,models_replayed=6,metrics_equal=True,
        report_sha256=sha256(ROOT/'report.json')));print('two-fold six models independently replayed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('train','verify'),required=True)
    train() if p.parse_args().mode=='train' else verify()
