"""保留原药效映射的独立药物身份/实际剂量接口；不覆盖历史派生数据。"""
from collections import Counter, defaultdict
from datetime import datetime
import argparse
import json
from pathlib import Path
import re
import time
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from patient_data.preprocess_events import ADMINISTERED
from patient_data.eos_med_context import finite
from patient_data.compact_multisource import INHALED_ROUTES, ORDER_SYSTEMIC_ROUTES, order_available
from patient_data.v6_detail_extension import MOLECULES, molecule_matches
from training.v5.numeric_preprocessing import robust_parameters

NAMES = MOLECULES + ('salmeterol', 'formoterol', 'tiotropium', 'fluticasone', 'budesonide',
                    'albuterol_ipratropium', 'fluticasone_salmeterol', 'budesonide_formoterol',
                    'umeclidinium_vilanterol')
UNITS = ('unknown', 'mg', 'puff', 'neb', 'inh', 'ml', 'mixed')
CHANNELS = ('executed_records', 'executed_days', 'maximum_executions_per_day',
            'mean_executions_per_recorded_day', 'complete_cumulative_given_dose',
            'maximum_given_dose', 'planned_doses_per_24h')
ALIASES = {'albuterol_ipratropium': ('duoneb', 'combivent'),
           'fluticasone_salmeterol': ('advair',), 'budesonide_formoterol': ('symbicort',),
           'umeclidinium_vilanterol': ('anoro',), 'tiotropium': ('spiriva',),
           'fluticasone': ('flovent',), 'budesonide': ('pulmicort',),
           'salmeterol': ('serevent',), 'formoterol': ('foradil', 'perforomist')}
INHALED = {'albuterol','ipratropium',*NAMES[12:]}


def identity(name):
    s = str(name).lower()
    for combo, constituents in (('albuterol_ipratropium', ('albuterol','ipratropium')),
                                ('fluticasone_salmeterol', ('fluticasone','salmeterol')),
                                ('budesonide_formoterol', ('budesonide','formoterol')),
                                ('umeclidinium_vilanterol', ('umeclidinium','vilanterol'))):
        if all(c in s for c in constituents) or any(a in s for a in ALIASES[combo]): return combo
    hits = list(molecule_matches(s))
    hits += [n for n in NAMES[12:17] if re.search(r'\b'+n+r'\b', s) or any(a in s for a in ALIASES[n])]
    return hits[0] if len(hits) == 1 else None


def component_dose(detail, drug):
    """返回数值、原单位类别、质量原因；不使用dose_due或浓度猜测。"""
    components = {}
    for r in detail:
        ordinal = r.get('parent_field_ordinal', '')
        if ordinal in ('', r'\N'): continue
        product = r.get('product_description','')
        if product not in ('',r'\N') and identity(product) != drug: return None, 'unknown', 'product_identity_ambiguous'
        value = finite(r.get('dose_given')); unit = r.get('dose_given_unit','').strip().lower()
        factors = {'mg':1., 'mcg':.001, 'ug':.001, 'g':1000., 'gm':1000.}
        aliases = {'puff':'puff','puffs':'puff','neb':'neb','nebs':'neb','inh':'inh','ml':'ml'}
        if value is None or value < 0 or unit not in factors and unit not in aliases:
            return None, 'unknown', 'unknown_given_dose_or_unit'
        pair = (value*factors[unit], 'mg') if unit in factors else (value, aliases[unit])
        if ordinal in components and components[ordinal] != pair: return None, 'unknown', 'ordinal_conflict'
        components[ordinal] = pair
    if not components: return None, 'unknown', 'no_scanned_component'
    units = {u for _,u in components.values()}
    if len(units) != 1: return None, 'mixed', 'component_units_mixed'
    return sum(v for v,_ in components.values()), units.pop(), 'verified_given_components'


def build(snapshot, state_dir, output):
    if output.exists(): raise FileExistsError(output)
    started=time.perf_counter(); sm=json.loads((snapshot/'metadata/manifest.json').read_text())
    st=json.loads((state_dir/'manifest.json').read_text())
    if sm['status']!='validated' or sha256(state_dir/'clinical_state.npz')!=st['output_sha256']['clinical_state.npz']:
        raise ValueError('unvalidated parent')
    with np.load(state_dir/'clinical_state.npz') as z: state={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits')}
    ids=list(map(str,state['hadm_ids'])); ix={h:i for i,h in enumerate(ids)}
    admissions={r['hadm_id']:(datetime.fromisoformat(r['admittime']),datetime.fromisoformat(r['dischtime']))
                for r in rows(snapshot/'cohort/admissions.csv.gz') if r['hadm_id'] in ix}
    hashes={}
    for name in ('cohort/admissions','raw/emar','raw/emar_detail','raw/pharmacy'):
        hashes[name]=sha256(snapshot/(name+'.csv.gz'))
        if hashes[name]!=sm['sources'][name]['sha256']: raise ValueError('raw changed '+name)
    orders=defaultdict(list); routes={}; planned=set(); executed=set(); raw_names=defaultdict(Counter); audit=Counter()
    for r in rows(snapshot/'raw/pharmacy.csv.gz'):
        h=r['hadm_id']
        if h not in ix or not order_available(r,admissions[h])[0]: continue
        planned.add(h); routes[h,r['pharmacy_id']]=(r['route'].strip().upper(),r['starttime'],r['stoptime'])
        drug=identity(r['medication'])
        if not drug: continue
        raw_names[drug][r['medication']]+=1
        allowed=INHALED_ROUTES if drug in INHALED else ORDER_SYSTEMIC_ROUTES
        if r['route'].strip().upper() not in allowed: continue
        f=finite(r['doses_per_24_hrs'])
        if f is not None and 0<f<=48: orders[h,drug].append(f)
    parents={}
    for r in rows(snapshot/'raw/emar.csv.gz'):
        h=r['hadm_id']
        if h not in ix: continue
        try: t,s=datetime.fromisoformat(r['charttime']),datetime.fromisoformat(r['storetime'])
        except ValueError: continue
        if not admissions[h][0]<=t<=s<=admissions[h][1]: continue
        executed.add(h); drug=identity(r['medication'])
        if not drug or r['event_txt'] not in ADMINISTERED: continue
        raw_names[drug][r['medication']]+=1
        key=r['emar_id'],r['emar_seq']
        if key in parents: raise ValueError('duplicate eMAR parent')
        parents[key]=(h,t,drug,r['pharmacy_id'])
    details=defaultdict(list)
    for r in rows(snapshot/'raw/emar_detail.csv.gz'):
        key=r['emar_id'],r['emar_seq']
        if key in parents: details[key].append(r)
    observations=defaultdict(list)
    for key,(h,t,drug,pid) in parents.items():
        ds=details[key]; route={r['route'].strip().upper() for r in ds if r['route'] not in ('',r'\N')}
        if not route and (h,pid) in routes:
            u,first,last=routes[h,pid]
            if datetime.fromisoformat(first)<=t and (last in ('',r'\N') or t<=datetime.fromisoformat(last)):
                route={u}; audit['route_from_linked_visible_order']+=1
        allowed=INHALED_ROUTES if drug in INHALED else ORDER_SYSTEMIC_ROUTES
        if not route or not route.issubset(allowed): audit['unverified_route']+=1; continue
        if any(r['complete_dose_not_given'].strip().upper() in ('Y','YES','1','TRUE') for r in ds):
            audit['incomplete_execution']+=1; continue
        dose,unit,reason=component_dose(ds,drug); audit[reason]+=1
        if dose==0: audit['zero_dose_execution_excluded']+=1; continue
        observations[h,drug].append((t,dose,unit))
    values=np.zeros((len(ids),len(NAMES),7),np.float32); masks=np.zeros_like(values,bool)
    units=np.zeros((len(ids),len(NAMES)),np.int64); source=np.zeros((len(ids),len(NAMES),2),bool)
    present=np.zeros((len(ids),len(NAMES)),bool); coverage={}
    for h,i in ix.items():
        for j,drug in enumerate(NAMES):
            source[i,j]=h in executed,h in planned
            if h in executed: masks[i,j,:4]=True
            obs=observations.get((h,drug),[]); order=orders.get((h,drug),[])
            present[i,j]=bool(obs or order)
            if obs:
                days=Counter(t.date() for t,_,_ in obs)
                values[i,j,:4]=len(obs),len(days),max(days.values()),len(obs)/len(days)
                known={u for _,d,u in obs if d is not None}
                units[i,j]=UNITS.index(next(iter(known))) if len(known)==1 else UNITS.index('mixed') if known else 0
                if all(d is not None for _,d,_ in obs) and len(known)==1:
                    doses=[d for _,d,_ in obs]; values[i,j,4:6]=sum(doses),max(doses); masks[i,j,4:6]=True
            if order: values[i,j,6]=np.median(order); masks[i,j,6]=True
    train=state['patient_splits']=='train'; scaled=np.zeros_like(values); norm={}
    for j,drug in enumerate(NAMES):
        for k in range(7):
            buckets=range(len(UNITS)) if k in (4,5) else (-1,)
            for u in buckets:
                valid=masks[:,j,k] & (units[:,j]==u if u>=0 else True); fitted=valid & train
                if not fitted.any(): masks[valid,j,k]=False; continue
                x=np.log1p(values[:,j,k]); p=robust_parameters(x[fitted])
                scaled[valid,j,k]=np.clip((x[valid]-p['median'])/p['scale'],-8,8)
                norm[f'{drug}/{CHANNELS[k]}/{UNITS[u] if u>=0 else "count"}']=p
        coverage[drug]={'execution_admissions':int((values[:,j,0]>0).sum()),
                        'complete_dose_admissions':int(masks[:,j,4].sum()),'present_admissions':int(present[:,j].sum()),
                        'units':dict(Counter(UNITS[u] for u in units[present[:,j],j]))}
    unit_onehot=np.eye(len(UNITS),dtype=np.float32)[units]
    features=np.concatenate((scaled,masks.astype(np.float32),unit_onehot,source.astype(np.float32)),2)
    # 缺乏本药内容的token彻底屏蔽，来源不等同本药存在。
    features*=present[:,:,None]
    output.mkdir(parents=True)
    np.savez_compressed(output/'medication_identity.npz',**state,features=features,present=present,
                        values_raw=values,value_masks=masks,dose_unit_id=units,drug_names=np.asarray(NAMES))
    (output/'normalization.json').write_text(json.dumps(norm,indent=2),encoding='utf-8')
    (output/'raw_drug_names.json').write_text(json.dumps({k:dict(v) for k,v in raw_names.items()},indent=2),encoding='utf-8')
    manifest=dict(status='validated_medication_identity',parent_state_manifest_sha256=sha256(state_dir/'manifest.json'),
        sources_sha256=hashes,output_sha256=sha256(output/'medication_identity.npz'),
        names=NAMES,units=UNITS,channels=CHANNELS,feature_width=23,coverage=coverage,audit=dict(audit),
        elapsed_seconds=time.perf_counter()-started,limits=['Dose absence is unknown, not zero.',
        'Actual frequency per recorded day is not prescription frequency or complete outside exposure.',
        'Raw name aliases are audited; canonical identity preserves explicit fixed combinations.'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'],seconds=manifest['elapsed_seconds'],coverage=coverage)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--state-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.snapshot,a.state_dir,a.output)
