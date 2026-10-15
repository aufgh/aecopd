"""复用原用药真实性规则，派生同分子记录动态，不输出个体记录。"""
from collections import defaultdict
from datetime import datetime
import argparse
import json
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.v6_detail_extension import MOLECULES,molecule_matches,dose_mg
from patient_data.compact_multisource import INHALED_ROUTES,order_available
from patient_data.eos_med_context import SYSTEMIC_ROUTES
from patient_data.preprocess_events import ADMINISTERED
from training.v6.train import write_json
from training.v7.research_features import SNAP,ROOT as FEATURES,scaled

ROOT=Path('training_runs/v7_sequential_research_2026-10-15/drug_context')
DRUG=Path('patient_data/processed/v6_audited_coarse_fine_2026-10-12_r2/interface.npz')
NAMES=Path('training_runs/v7_sequential_research_2026-10-15/drug_names/names.npz')
PROTOCOL=Path('docs/V7_DRUG_CONTEXT_PROTOCOL_2026-10-15.md')


def route_kind(routes):
    kinds=set()
    for r in routes:
        if r in ('IV','IV DRIP','IV PUSH','INTRAVENOUS'):kinds.add('iv')
        elif r in ('IM','INTRAMUSCULAR'):kinds.add('im')
        elif r in INHALED_ROUTES:kinds.add('inhaled')
        elif r in SYSTEMIC_ROUTES:kinds.add('enteral')
        else:return None
    return next(iter(kinds)) if len(kinds)==1 else None


def summary(samples,los):
    x=np.zeros(12);valid=np.zeros(12,bool)
    if not samples:return x,valid
    ordered=sorted(samples,key=lambda r:r[0]);t=np.array([r[0] for r in ordered])
    x[:5]=[1,np.log1p(len(t)),np.log1p(t[-1]-t[0]),np.log1p((t<1).sum()),np.log1p((t>=los-1).sum())];valid[:5]=True
    first=[s for s in ordered if s[0]<1];last=[s for s in ordered if s[0]>=los-1]
    if los>=2 and first and last:
        gap=los-1;x[5]=(len(last)-len(first))/gap;valid[5]=True
        if all(s[1] is not None for s in first+last):
            x[6]=(sum(s[1] for s in last)-sum(s[1] for s in first))/gap;valid[6]=True
    start_route,end_route=ordered[0][2],ordered[-1][2]
    if t[-1]>t[0] and start_route is not None and end_route is not None:
        x[7]=start_route!=end_route;x[8]=start_route=='iv' and end_route=='enteral';valid[7:9]=True
    return x,valid


def derive():
    started=time.perf_counter();sm=json.loads((SNAP/'metadata/manifest.json').read_text())
    with np.load(FEATURES/'interface.npz') as z:ids={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits')};los=z['los']
    order={str(h):i for i,h in enumerate(ids['hadm_ids'])};bounds={}
    for r in rows(SNAP/'cohort/admissions.csv.gz'):
        if r['hadm_id'] in order:bounds[r['hadm_id']]=tuple(datetime.fromisoformat(r[k]) for k in ('admittime','dischtime'))
    sources={};records={};pharmacy={};source_present=np.zeros(len(order),bool)
    for name in ('cohort/admissions','raw/emar','raw/emar_detail','raw/pharmacy'):
        p=SNAP/(name+'.csv.gz');sources[str(p)]=sha256(p)
        if sources[str(p)]!=sm['sources'][name]['sha256']:raise ValueError('drug raw source changed')
    for r in rows(SNAP/'raw/pharmacy.csv.gz'):
        h=r['hadm_id']
        if h in order and order_available(r,bounds[h])[0]:pharmacy[h,r['pharmacy_id']]=(r['route'].strip().upper(),r['starttime'],r['stoptime'])
    for r in rows(SNAP/'raw/emar.csv.gz'):
        h=r['hadm_id']
        if h not in order:continue
        try:t,s=(datetime.fromisoformat(r[k]) for k in ('charttime','storetime'))
        except ValueError:continue
        if not bounds[h][0]<=t<=s<=bounds[h][1]:continue
        source_present[order[h]]=True;molecules=molecule_matches(r['medication'])
        if not molecules or r['event_txt'] not in ADMINISTERED:continue
        key=r['emar_id'],r['emar_seq']
        if key in records:raise ValueError('duplicate execution parent')
        records[key]=(h,t,molecules,r['pharmacy_id'])
    details=defaultdict(list)
    for r in rows(SNAP/'raw/emar_detail.csv.gz'):
        key=r['emar_id'],r['emar_seq']
        if key in records:details[key].append(r)
    samples=defaultdict(list)
    for key,(h,t,molecules,pid) in records.items():
        detail=details[key];routes={r['route'].strip().upper() for r in detail if r['route'] not in ('',r'\N')}
        if not routes and (h,pid) in pharmacy:
            route,first,last=pharmacy[h,pid]
            if datetime.fromisoformat(first)<=t and (last in ('',r'\N') or t<=datetime.fromisoformat(last)):routes={route}
        for molecule in molecules:
            allowed=INHALED_ROUTES if molecule in ('albuterol','ipratropium') else SYSTEMIC_ROUTES
            if not routes or not routes.issubset(allowed):continue
            if any(r['complete_dose_not_given'].strip().upper() in ('Y','YES','TRUE','1') for r in detail):continue
            dose,_=dose_mg(detail,molecule) if len(molecules)==1 else (None,'compound')
            if dose==0:continue
            day=(t-bounds[h][0]).total_seconds()/86400
            samples[order[h],MOLECULES.index(molecule)].append((day,dose,route_kind(routes)))
    raw=np.zeros((len(order),12,12));valid=np.zeros_like(raw,bool)
    for (i,j),values in samples.items():raw[i,j],valid[i,j]=summary(values,los[i])
    with np.load(DRUG) as z:
        for k in ids:np.testing.assert_array_equal(ids[k],z[k])
        names=z['med_names'];present=z['med_present']
    with np.load(NAMES) as z:np.testing.assert_array_equal(names,z['names']);vectors=z['vectors']
    projection=np.random.default_rng(42).normal(size=(768,256))/np.sqrt(768)
    random=np.random.default_rng(42).normal(size=vectors.shape);random/=np.linalg.norm(random,axis=1,keepdims=True)
    count=present.sum(1,keepdims=True).clip(1)
    semantic=(present@vectors/count)@projection;random_view=(present@random/count)@projection
    a=dict(**ids,raw=raw,valid=valid,los=los,source_present=source_present,fine_present=present,
        semantic=semantic.astype(np.float32),random=random_view.astype(np.float32))
    for p in (Path(__file__),PROTOCOL,Path('training/tests/test_v7_drug_context.py'),DRUG,NAMES,
              NAMES.parent/'manifest.json',FEATURES/'interface.npz',Path('patient_data/v6_detail_extension.py')):sources[str(p)]=sha256(p)
    train=ids['patient_splits']=='train'
    return a,dict(source_sha256=sources,train_frequency_valid=int(valid[train,:,5].sum()),
        train_dose_valid=int(valid[train,:,6].sum()),train_route_valid=int(valid[train,:,7].sum()),
        source_visibility_is_partial=True,recorded_span_is_not_complete_duration=True,seconds=time.perf_counter()-started)


def view(a,fit_indices,name):
    x=np.zeros((len(a['los']),512),np.float32);fit=np.zeros(len(x),bool);fit[fit_indices]=True
    if name.startswith('name_'):
        p=a['fine_present'];x[:,500]=p.any(1);x[:,501]=np.log1p(p.sum(1))
        if name=='name_identity':x[:,:38]=p
        if name=='name_random':x[:,:256]=a['random']
        if name=='name_semantic':x[:,:256]=a['semantic']
    else:
        raw,valid=a['raw'],a['valid'];x[:,:12]=raw[:,:,0];x[:,12:24]=valid[:,:,5];x[:,24:36]=valid[:,:,6]
        x[:,36:48]=valid[:,:,7];x[:,500]=a['source_present']
        if name!='dynamics_source':
            values=raw[:,:,1:5].reshape(len(x),-1);mask=valid[:,:,1:5].reshape(len(x),-1)
            x[:,48:96],_=scaled(values,mask,fit)
        if name=='dynamics_slope':x[:,96:120],_=scaled(raw[:,:,5:7].reshape(len(x),-1),valid[:,:,5:7].reshape(len(x),-1),fit)
        if name=='dynamics_route':x[:,120:144]=raw[:,:,7:9].reshape(len(x),-1)
        duration,_=scaled(np.log1p(a['los'])[:,None],np.ones((len(x),1),bool),fit)
        x[:,501]=duration[:,0]
    return x


def main(mode):
    if mode=='build' and ROOT.exists():raise FileExistsError('drug context exists')
    a,meta=derive()
    if mode=='build':
        ROOT.mkdir();np.savez_compressed(ROOT/'interface.npz',**a);meta['output_sha256']=sha256(ROOT/'interface.npz')
        write_json(ROOT/'manifest.json',meta);print(json.dumps({k:v for k,v in meta.items() if k!='source_sha256'}),flush=True)
    else:
        saved=json.loads((ROOT/'manifest.json').read_text())
        if meta['source_sha256']!=saved['source_sha256'] or sha256(ROOT/'interface.npz')!=saved['output_sha256']:raise ValueError('drug source differs')
        with np.load(ROOT/'interface.npz') as z:
            for k,v in a.items():np.testing.assert_array_equal(z[k],v)
        write_json(ROOT/'verification.json',dict(maximum_error=0,manifest_sha256=sha256(ROOT/'manifest.json')))
        print('drug context source replay error 0',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify'),required=True)
    main(p.parse_args().mode)
