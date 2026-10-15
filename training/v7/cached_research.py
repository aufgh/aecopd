"""冻结强风险与候选表示；小容量机制筛查和独立重放。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from patient_data.export_mimiciv import sha256
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import RiskOutput,competing_nll
from training.v6.train import write_json
from training.v6.module_surgery import SurgeryModel
from training.v6.symmetric_context import SymmetricContextModel
from training.v7 import experiment as first
from training.v7.data import cache
from training.v7.model import V7Model
from training.v7.research_features import ROOT as FEATURE_ROOT,encode,scaled,REGIMEN

ROOT=Path('training_runs/v7_sequential_research_2026-10-15/cached')
PROTOCOL=Path('docs/V7_CACHED_RESEARCH_PROTOCOL_2026-10-15.md')
ANCHOR=Path('training_runs/v6_symmetric_context_2026-10-14/symmetric')
VIEWS=('global','v7_frozen','availability','linear','ple','velocity','density','curvature',
       'clinical_main','clinical_interaction','routing_uniform','routing_prior')
PFT=Path('patient_data/processed/v6_pft_set_interface_2026-10-13/interface.npz')
CXR=Path('patient_data/processed/v6_chexbert_cxr_2026-10-09/cxr_context.npz')


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def prepare():
    if ROOT.exists():raise FileExistsError('cached experiment exists; preserve scene')
    start=time.perf_counter();data=first.setup();d,c,ext,lab,rel,train,tune,base,drug,rad,*_=data
    ROOT.mkdir(parents=True)
    fresh=SurgeryModel(base,42,'control',[])
    strong=SymmetricContextModel(fresh,'symmetric').cuda().eval().requires_grad_(False)
    strong.load_state_dict(torch.load(ANCHOR/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
    n=len(d.times);aa=np.zeros((n,2),np.float32);kk=np.ones_like(aa);context=np.zeros((n,34),np.float32)
    active=np.r_[train,tune];valid=np.zeros(n,bool);valid[active]=True
    with torch.no_grad():
        for start_at in range(0,len(active),128):
            ix=active[start_at:start_at+128]
            b=first.parent.make_batch(c,ext,lab,ix,rel['neighbors'],rel['offsets'],rel['gaps'],rel['note_times'],rel['admit'],history=True)
            out=strong(b);aa[ix]=out.log_amplitude.cpu();kk[ix]=out.shape.cpu();context[ix]=b[4][b[5]].cpu()
    # 同一个batch边界是旧风险重放合同的一部分；train与tune分开再核对调参。
    with torch.no_grad():
        for start_at in range(0,len(tune),128):
            ix=tune[start_at:start_at+128]
            b=first.parent.make_batch(c,ext,lab,ix,rel['neighbors'],rel['offsets'],rel['gaps'],rel['note_times'],rel['admit'],history=True)
            out=strong(b);aa[ix]=out.log_amplitude.cpu();kk[ix]=out.shape.cpu()
    old=load_npz(ANCHOR/'tuning_predictions.npz')
    np.testing.assert_array_equal(aa[tune],old['log_amplitude']);np.testing.assert_array_equal(kk[tune],old['shape'])
    np.testing.assert_array_equal(first.parent.probability_curve(aa[tune],kk[tune]),old['curve'])
    del strong,fresh;torch.cuda.empty_cache()
    m=V7Model(base,drug['drug_parent']).cuda().eval().requires_grad_(False)
    checkpoint=first.ROOT/'recommended/best.pt'
    m.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['state_dict'])
    embedded=np.zeros((n,256),np.float32);captured=[]
    handle=m.delta_head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().numpy()))
    with torch.no_grad():
        for ix,b in cache(c,ext,lab,rel,drug,rad,active):
            m(b.to('cuda'));embedded[ix]=captured.pop()
    handle.remove();del m;torch.cuda.empty_cache()
    # 无监督冻结BERT摘要用于最后的严格内折模型，不复用监督V7表示。
    text=np.zeros((n,768),np.float32);history_text=np.zeros_like(text)
    text_present=np.zeros(n,bool)
    for i,refs in enumerate(d.text_refs):
        if refs:text[i]=np.asarray(d.text_vectors[[r for _,r in refs]],np.float32).mean(0);text_present[i]=True
    for i in active:
        previous=rel['neighbors'][rel['offsets'][i]:rel['offsets'][i+1]]
        use=previous[(rel['note_times'][previous]<=rel['admit'][i])&text_present[previous]]
        if len(use):history_text[i]=text[use].mean(0)
    rx,pft,cxr=(load_npz(p) for p in (REGIMEN,PFT,CXR))
    for a in (rx,pft,cxr):
        for k in ('hadm_ids','subject_ids','patient_splits'):np.testing.assert_array_equal(a[k],d.arrays[k])
    times=np.full(n,np.nan);events=np.full(n,-1,np.int64);times[active]=d.times[active];events[active]=d.events[active]
    arrays=dict(**{k:d.arrays[k] for k in ('hadm_ids','subject_ids','patient_splits')},train=train,tune=tune,
        valid=valid,times=times,events=events,anchor_a=aa,anchor_k=kk,context=context,
        context_raw=np.column_stack((d.history_raw,d.current,d.current_masks)).astype(np.float32),
        v7_hidden=embedded,text=text,history_text=history_text,regimen=rx['features'],
        pft=pft['aggregate'],cxr=cxr['cxr_onehot'],
        support=np.concatenate((c.arrays['support_values'],c.arrays['support_masks']),1).astype(np.float32))
    np.savez_compressed(ROOT/'cache.npz',**arrays)
    paths=(Path(__file__),PROTOCOL,Path('training/v7/research_features.py'),
        Path('training/tests/test_v7_cached_research.py'),ANCHOR/'best.pt',ANCHOR/'tuning_predictions.npz',checkpoint,
        FEATURE_ROOT/'manifest.json',FEATURE_ROOT/'verification.json',FEATURE_ROOT/'interface.npz',REGIMEN,PFT,CXR)
    write_json(ROOT/'cache_contract.json',dict(source_sha256={str(p):sha256(p) for p in paths},
        parent=data[-3],cache_sha256=sha256(ROOT/'cache.npz'),anchor_replay_error=0,
        no_confirmation_test_labels_cached=True,seconds=time.perf_counter()-start))
    print(json.dumps(dict(cache_ready=True,anchor_error=0,seconds=time.perf_counter()-start)),flush=True)


def checked():
    contract=json.loads((ROOT/'cache_contract.json').read_text())
    for p,h in contract['source_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('cached source changed: '+p)
    if sha256(ROOT/'cache.npz')!=contract['cache_sha256']:raise ValueError('cache changed')
    return load_npz(ROOT/'cache.npz'),load_npz(FEATURE_ROOT/'interface.npz'),contract


def views(a,f,fit,view):
    x=np.zeros((len(a['times']),512),np.float32)
    if view=='v7_frozen':x[:,:256]=a['v7_hidden']
    elif view!='global':
        source='clinical_interaction' if view.startswith('routing_') else view
        if view.startswith('routing_'):source='ple'
        x[:,:288],_=encode(f,fit,source)
        if view.startswith('routing_'):
            x[:,288:322]=a['context'];x[:,322:356]=a['regimen'];x[:,356:365]=a['pft']
            x[:,365:393]=a['cxr'];x[:,393:399]=a['support'];x[:,399:405]=f['clinical']
            z,_=scaled(np.log1p(f['los'])[:,None],np.ones((len(x),1),bool),np.isin(np.arange(len(x)),fit));x[:,405]=z[:,0]
    return x


def group_ids():
    groups=np.full(512,9,np.int64)
    for j in range(9):
        groups[j*24:(j+1)*24]=j;groups[216+j]=j;groups[225+j]=j
    groups[322:356]=10;groups[356:365]=11;groups[365:393]=12;groups[393:399]=13;groups[399:405]=14
    return torch.as_tensor(groups)


class Reader(nn.Module):
    def __init__(self,view,seed=42):
        super().__init__();self.view=view
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if view=='global':self.global_raw=nn.Parameter(torch.zeros(4))
            elif view.startswith('routing_'):
                self.shared=nn.Sequential(nn.Linear(35,16),nn.GELU(),nn.Linear(16,8))
                self.heads=nn.ModuleList(nn.Sequential(nn.Linear(520,16),nn.GELU(),nn.Dropout(.3),nn.Linear(16,2)) for _ in range(2))
                self.route=nn.Parameter(torch.zeros(2,15));self.register_buffer('groups',group_ids())
                if view=='routing_prior':
                    with torch.no_grad():
                        self.route[0,[0,1,5,8,9,10,11,13]]=.5
                        self.route[1,[2,3,4,7,9,12]]=.5
                for head in self.heads:nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
            else:
                self.head=nn.Sequential(nn.Linear(512,16),nn.GELU(),nn.Dropout(.3),nn.Linear(16,4))
                nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,x,anchor_a,anchor_k):
        if self.view=='global':raw=self.global_raw.expand(len(x),4).reshape(-1,2,2)
        elif self.view.startswith('routing_'):
            shared=self.shared(torch.cat((x[:,288:322],x[:,405:406]),1))
            weight=15*torch.softmax(self.route,1)
            raw=torch.stack([head(torch.cat((x*weight[k,self.groups],shared),1)) for k,head in enumerate(self.heads)],1)
        else:raw=self.head(x).reshape(-1,2,2)
        return RiskOutput((anchor_a+raw[:,:,0]).clamp(-10,6),anchor_k*torch.exp(.25*torch.tanh(raw[:,:,1])),raw)


def labels(a,ix):return torch.as_tensor(a['times'][ix],device='cuda'),torch.as_tensor(a['events'][ix],device='cuda')


@torch.no_grad()
def predict(m,x,a,ix):
    m.eval();aa=[];kk=[];nll=0.
    for start in range(0,len(ix),512):
        p=ix[start:start+512];out=m(x[p],torch.as_tensor(a['anchor_a'][p],device='cuda'),torch.as_tensor(a['anchor_k'][p],device='cuda'))
        aa.append(out.log_amplitude.cpu().numpy());kk.append(out.shape.cpu().numpy());nll+=float(competing_nll(out,*labels(a,p)))*len(p)
    aa,kk=np.concatenate(aa),np.concatenate(kk)
    return aa,kk,first.parent.probability_curve(aa,kk),nll/len(ix)


def fit(a,f,view):
    folder=ROOT/view;folder.mkdir();train,tune=a['train'],a['tune']
    x=torch.as_tensor(views(a,f,train,view),device='cuda');m=Reader(view).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
    rng=np.random.default_rng(42);torch.manual_seed(42);early=EarlyStopping(6,8,1e-4);hist=[];best=None;started=time.perf_counter()
    for epoch in range(25):
        if epoch:
            m.train()
            for start in range(0,len(train),256):
                # 一次每轮打乱，而非每个batch重新抽样。
                if start==0:order=rng.permutation(train)
                p=order[start:start+256];opt.zero_grad(set_to_none=True)
                out=m(x[p],torch.as_tensor(a['anchor_a'][p],device='cuda'),torch.as_tensor(a['anchor_k'][p],device='cuda'))
                loss=competing_nll(out,*labels(a,p))+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite cached loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        aa,kk,curve,nll=predict(m,x,a,tune)
        metrics=first.parent.detail(type('Data',(),dict(times=a['times'],events=a['events']))(),tune,curve,nll)
        row=dict(epoch=epoch,annual=metrics['annual'],tuning_ibs=metrics['readmission']['integrated_brier_7_365'],
                 tuning_death_ibs=metrics['death']['integrated_brier_7_365'])
        hist.append(row);_,stop=early.observe(epoch,-row['annual']['f1']);key=first.selection_key(row)
        if best is None or key>best:
            best=key;torch.save(dict(state_dict=m.state_dict(),epoch=epoch,threshold=row['annual']['threshold']),folder/'best.pt')
        torch.save(dict(state_dict=m.state_dict(),epoch=epoch,threshold=row['annual']['threshold']),folder/'last.pt')
        write_json(folder/'history.json',hist)
        if stop:break
    outputs={}
    for which in ('best','last'):
        ck=torch.load(folder/f'{which}.pt',map_location='cuda',weights_only=True);m.load_state_dict(ck['state_dict'])
        aa,kk,curve,nll=predict(m,x,a,tune)
        metrics=first.parent.detail(type('Data',(),dict(times=a['times'],events=a['events']))(),tune,curve,nll,ck['threshold'])
        np.savez_compressed(folder/f'{which}_predictions.npz',positions=tune,log_amplitude=aa,shape=kk,curve=curve)
        outputs[which]=dict(epoch=ck['epoch'],threshold=ck['threshold'],metrics=metrics,
            checkpoint_sha256=sha256(folder/f'{which}.pt'),predictions_sha256=sha256(folder/f'{which}_predictions.npz'))
    result=dict(outputs=outputs,early=asdict(early),completed_epochs=epoch,seconds=time.perf_counter()-started,
        parameters=sum(p.numel() for p in m.parameters()),stop_reason='patience' if stop else 'max_epochs')
    write_json(folder/'result.json',result);print(json.dumps(dict(view=view,epoch=outputs['best']['epoch'],
        f1=outputs['best']['metrics']['annual']['f1'],ibs=outputs['best']['metrics']['readmission']['integrated_brier_7_365'],seconds=result['seconds'])),flush=True)
    del m,x;torch.cuda.empty_cache();return result


def train():
    if (ROOT/'report.json').exists():raise FileExistsError('cached report exists')
    a,f,_=checked();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    results={view:fit(a,f,view) for view in VIEWS}
    write_json(ROOT/'report.json',dict(results=results,default_replaced=False,confirmation_test_used=False))


def verify():
    a,f,_=checked();report=json.loads((ROOT/'report.json').read_text());proof={};tune=a['tune'];d=type('Data',(),dict(times=a['times'],events=a['events']))()
    torch.set_num_threads(4)
    for view,info in report['results'].items():
        x=torch.as_tensor(views(a,f,a['train'],view),device='cuda');m=Reader(view).cuda()
        for which,out in info['outputs'].items():
            folder=ROOT/view
            if sha256(folder/f'{which}.pt')!=out['checkpoint_sha256'] or sha256(folder/f'{which}_predictions.npz')!=out['predictions_sha256']:raise ValueError('cached fit artifact changed')
            ck=torch.load(folder/f'{which}.pt',map_location='cuda',weights_only=True);m.load_state_dict(ck['state_dict'])
            aa,kk,curve,nll=predict(m,x,a,tune)
            with np.load(folder/f'{which}_predictions.npz') as z:
                for key,value in (('positions',tune),('log_amplitude',aa),('shape',kk),('curve',curve)):np.testing.assert_array_equal(z[key],value)
            if first.parent.detail(d,tune,curve,nll,out['threshold'])!=out['metrics']:raise ValueError('cached metrics differ')
        hist=json.loads((ROOT/view/'history.json').read_text());early=EarlyStopping(6,8,1e-4)
        for row in hist:early.observe(row['epoch'],-row['annual']['f1'])
        if asdict(early)!=info['early'] or max(hist,key=first.selection_key)['epoch']!=info['outputs']['best']['epoch']:raise ValueError('cached selection differs')
        proof[view]=dict(best_last_error=0,metrics_equal=True,selection_equal=True);del m,x
    write_json(ROOT/'verification.json',dict(results=proof,maximum_error=0,report_sha256=sha256(ROOT/'report.json')))
    print(json.dumps(dict(cached_fits_verified=len(proof),maximum_error=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('prepare','train','verify'),required=True)
    globals()[p.parse_args().mode]()
