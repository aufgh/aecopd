"""V6 增量候选：分子级给药计数/剂量和额外检验，原始文件只读。"""
from collections import Counter, defaultdict
from datetime import datetime
import argparse
import json
import math
from pathlib import Path
import re
import time
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from patient_data.preprocess_events import ADMINISTERED
from patient_data.eos_med_context import SYSTEMIC_ROUTES, finite
from patient_data.compact_multisource import INHALED_ROUTES, order_available
from training.v5.numeric_preprocessing import robust_parameters

LABS = {
    'neutrophils_percent': ('51256', '%'),
    'lymphocytes_percent': ('51244', '%'),
    'rdw': ('51277', '%'),
    'anion_gap': ('50868', 'mEq/L'),
    'chloride': ('50902', 'mEq/L'),
}
# 原料/复方层次固定，不把不同分子的 mg 相加或当作等效药效。
MOLECULES = ('albuterol', 'ipratropium', 'prednisone', 'methylprednisolone',
    'hydrocortisone', 'dexamethasone', 'azithromycin', 'ceftriaxone',
    'levofloxacin', 'doxycycline', 'piperacillin_tazobactam', 'vancomycin')
LAB_CHANNELS = ('first', 'minimum', 'maximum', 'latest', 'latest_age_log_days')
MED_CHANNELS = ('recorded_administrations', 'recorded_calendar_days',
    'maximum_administrations_per_calendar_day', 'administrations_per_recorded_day',
    'complete_recorded_dose_mg', 'maximum_recorded_single_dose_mg', 'ordered_doses_per_24h')


def molecule_matches(name):
    name = name.lower()
    result = []
    for molecule in MOLECULES:
        if molecule == 'piperacillin_tazobactam':
            hit = ('piperacillin' in name and 'tazobactam' in name) or 'zosyn' in name
        elif molecule == 'albuterol':
            hit = bool(re.search(r'\balbuterol\b|\bsalbutamol\b', name)) or 'duoneb' in name
        elif molecule == 'ipratropium':
            hit = 'ipratropium' in name or 'duoneb' in name
        else:
            hit = bool(re.search(r'\b'+molecule+r'\b', name))
        if hit: result.append(molecule)
    return tuple(result)


def dose_mg(detail, molecule):
    """只累加扫描子行。相同 ordinal 重复去重、冲突/缺单位/不同分子均拒绝。"""
    components = {}
    for r in detail:
        ordinal = r['parent_field_ordinal']
        if ordinal in ('', r'\N'): continue
        product = r['product_description']
        if product not in ('', r'\N') and molecule_matches(product) != (molecule,): return None, 'ambiguous_product'
        value = finite(r['dose_given'])
        factor = {'mg': 1., 'mcg': .001, 'ug': .001, 'g': 1000., 'gm': 1000.}.get(r['dose_given_unit'].strip().lower())
        if value is None or value < 0 or factor is None: return None, 'unknown_dose_or_unit'
        dose = value * factor
        if ordinal in components and components[ordinal] != dose: return None, 'conflicting_ordinal'
        components[ordinal] = dose
    if not components: return None, 'no_formulary_components'
    return sum(components.values()), 'verified_component_sum'


def scale(values, masks, train):
    transformed = np.log1p(values)
    result = np.zeros_like(values, dtype=np.float32)
    normalization = {}
    for j in range(values.shape[1]):
        for k in range(values.shape[2]):
            valid = train & masks[:, j, k]
            if not valid.any():
                masks[:, j, k] = False
                normalization[f'{j}/{k}'] = {'method': 'no_training_observation'}
                continue
            p = robust_parameters(transformed[valid, j, k])
            result[:, j, k] = np.where(masks[:, j, k], np.clip(
                (transformed[:, j, k]-p['median']) / p['scale'], -8, 8), 0)
            normalization[f'{j}/{k}'] = p
    return result, normalization


def build(snapshot, state_dir, eos_dir, output):
    if output.exists(): raise FileExistsError(output)
    started = time.perf_counter()
    sm = json.loads((snapshot/'metadata/manifest.json').read_text())
    st = json.loads((state_dir/'manifest.json').read_text())
    em = json.loads((eos_dir/'manifest.json').read_text())
    if sm['status'] != 'validated' or em['status'] != 'eos_med_context_validated_not_trained':
        raise ValueError('unvalidated parents')
    if sha256(state_dir/'clinical_state.npz') != st['output_sha256']['clinical_state.npz']:
        raise ValueError('state hash mismatch')
    if em['parent_clinical_state_manifest_sha256'] != sha256(state_dir/'manifest.json'):
        raise ValueError('EOS state lineage differs')
    if sha256(eos_dir/'eos_med_context.npz') != em['output_sha256']['eos_med_context.npz']:
        raise ValueError('EOS output changed')
    with np.load(state_dir/'clinical_state.npz') as z: state = {k:z[k] for k in z.files}
    with np.load(eos_dir/'eos_med_context.npz') as z: eos = {k:z[k] for k in z.files}
    for k in ('hadm_ids', 'subject_ids', 'patient_splits'):
        if not np.array_equal(state[k], eos[k]): raise ValueError('EOS members/splits differ')
    ids = [str(x) for x in state['hadm_ids']]; index = {h:i for i,h in enumerate(ids)}
    if len(index) != 19401: raise ValueError('member drift')
    admissions = {r['hadm_id']: (datetime.fromisoformat(r['admittime']), datetime.fromisoformat(r['dischtime']))
        for r in rows(snapshot/'cohort/admissions.csv.gz') if r['hadm_id'] in index}
    sources = ('raw/labevents', 'raw/emar', 'raw/emar_detail', 'raw/pharmacy', 'raw/d_labitems', 'cohort/admissions')
    hashes = {}
    for name in sources:
        hashes[name] = sha256(snapshot/(name+'.csv.gz'))
        if hashes[name] != sm['sources'][name]['sha256']: raise ValueError('raw source changed '+name)
    dictionary = {r['itemid']:r for r in rows(snapshot/'raw/d_labitems.csv.gz')}
    audit = Counter(); observations = defaultdict(list)
    lab_map = {item:(j,unit) for j,(item,unit) in enumerate(LABS.values())}
    for r in rows(snapshot/'raw/labevents.csv.gz'):
        if r['hadm_id'] not in index or r['itemid'] not in lab_map: continue
        j,unit = lab_map[r['itemid']]; key = list(LABS)[j]
        audit['lab:'+key+':candidate'] += 1
        try:
            t,s = datetime.fromisoformat(r['charttime']),datetime.fromisoformat(r['storetime'])
        except ValueError:
            audit['lab:'+key+':invalid_time'] += 1; continue
        start,end = admissions[r['hadm_id']]; value = finite(r['valuenum'])
        if not start <= t <= s <= end:
            audit['lab:'+key+':outside_visible_interval'] += 1; continue
        if value is None or value < 0 or (unit == '%' and value > 100) or r['valueuom'] != unit:
            audit['lab:'+key+':invalid_value_or_unit'] += 1; continue
        observations[r['hadm_id'],j].append((t,value))
    print('lab sources audited', flush=True)
    # eMAR 父记录只数一次。保存候选键，子行从 detail 补给药途径和可核验剂量。
    records = {}; execution_source = set()
    pharmacy_routes = {}; orders = defaultdict(list); order_source = set()
    for r in rows(snapshot/'raw/pharmacy.csv.gz'):
        h = r['hadm_id']
        if h not in index or not order_available(r,admissions[h])[0]: continue
        order_source.add(h)
        pharmacy_routes[h,r['pharmacy_id']] = (r['route'].strip().upper(),r['starttime'],r['stoptime'])
        f = finite(r['doses_per_24_hrs'])
        if f is None or not 0 < f <= 48: continue
        for molecule in molecule_matches(r['medication']):
            allowed = INHALED_ROUTES if molecule in ('albuterol','ipratropium') else SYSTEMIC_ROUTES
            if r['route'].strip().upper() in allowed: orders[h,molecule].append(f)
    for r in rows(snapshot/'raw/emar.csv.gz'):
        h = r['hadm_id']
        if h not in index: continue
        try: t,s = datetime.fromisoformat(r['charttime']),datetime.fromisoformat(r['storetime'])
        except ValueError: continue
        if not admissions[h][0] <= t <= s <= admissions[h][1]: continue
        execution_source.add(h)
        molecules = molecule_matches(r['medication'])
        if not molecules or r['event_txt'] not in ADMINISTERED: continue
        key = (r['emar_id'], r['emar_seq'])
        if key in records: raise ValueError('duplicate eMAR parent')
        records[key] = (h,t,molecules,r['medication'],r['pharmacy_id'])
    details = defaultdict(list)
    for r in rows(snapshot/'raw/emar_detail.csv.gz'):
        key = (r['emar_id'], r['emar_seq'])
        if key in records: details[key].append(r)
    administered = defaultdict(list)
    for key,(h,t,molecules,name,pid) in records.items():
        detail = details[key]
        routes = {r['route'].strip().upper() for r in detail if r['route'] not in ('', r'\N')}
        if not routes and (h,pid) in pharmacy_routes:
            route,first,last = pharmacy_routes[h,pid]
            if datetime.fromisoformat(first) <= t and (last in ('',r'\N') or t <= datetime.fromisoformat(last)):
                routes = {route}
                audit['route_from_linked_visible_order'] += 1
        for molecule in molecules:
            allowed = INHALED_ROUTES if molecule in ('albuterol','ipratropium') else SYSTEMIC_ROUTES
            if not routes or not routes.issubset(allowed):
                audit['med:'+molecule+':route_not_verified'] += 1; continue
            if any(r['complete_dose_not_given'].strip().upper() in ('Y','YES','TRUE','1') for r in detail):
                audit['med:'+molecule+':incomplete_dose'] += 1; continue
            dose,reason = dose_mg(detail,molecule) if len(molecules)==1 else (None,'combination_dose_not_allocated')
            audit['med:'+molecule+':'+reason] += 1
            # 明确零剂量不计作执行；无法解析剂量仍保留执行记录，但剂量mask关闭。
            if dose == 0: audit['med:'+molecule+':zero_dose'] += 1; continue
            administered[h,molecule].append((t,dose))
    print('execution and formulary components audited', flush=True)
    lv = np.zeros((len(ids),len(LABS),5)); lm = np.zeros_like(lv,dtype=bool)
    mv = np.zeros((len(ids),len(MOLECULES),7)); mm = np.zeros_like(mv,dtype=bool)
    med_source = np.zeros((len(ids),len(MOLECULES),2),dtype=bool)
    coverage = {'labs':{}, 'medications':{}}
    for h,i in index.items():
        for j in range(len(LABS)):
            samples = observations.get((h,j),[])
            if not samples: continue
            by_time = defaultdict(list)
            for t,v in samples: by_time[t].append(v)
            samples = [(t,float(np.median(v))) for t,v in sorted(by_time.items())]
            lv[i,j] = [samples[0][1],min(v for _,v in samples),max(v for _,v in samples),samples[-1][1],
                        math.log1p((admissions[h][1]-samples[-1][0]).total_seconds()/86400)]
            lm[i,j] = True
        for j,molecule in enumerate(MOLECULES):
            med_source[i,j] = h in execution_source,h in order_source
            if h in execution_source: mm[i,j,:4] = True
            samples = administered.get((h,molecule),[])
            if samples:
                days = Counter(t.date() for t,_ in samples)
                mv[i,j,:4] = len(samples),len(days),max(days.values()),len(samples)/len(days)
                doses = [d for _,d in samples]
                if all(d is not None for d in doses):
                    mv[i,j,4:6] = sum(doses),max(doses); mm[i,j,4:6] = True
            if orders.get((h,molecule)):
                mv[i,j,6] = float(np.median(orders[h,molecule])); mm[i,j,6] = True
    train = state['patient_splits']=='train'
    ls,ln = scale(lv,lm,train); ms,mn = scale(mv,mm,train)
    for j,name in enumerate(LABS):
        coverage['labs'][name] = dict(itemid=list(LABS.values())[j][0],label=dictionary[list(LABS.values())[j][0]]['label'],
            unit=list(LABS.values())[j][1],admissions=int(lm[:,j,0].sum()),
            patients=len(np.unique(state['subject_ids'][lm[:,j,0]])))
    for j,name in enumerate(MOLECULES):
        coverage['medications'][name] = dict(recorded_execution=int((mv[:,j,0]>0).sum()),
            full_recorded_mg_dose=int(mm[:,j,4].sum()),ordered_numeric_frequency=int(mm[:,j,6].sum()))
    labs = np.concatenate((ls,lm),2).astype(np.float32)
    meds = np.concatenate((ms,mm,med_source),2).astype(np.float32)
    for a in (labs,meds,eos['eos_features']):
        if not np.isfinite(a).all(): raise ValueError('nonfinite extension')
    output.mkdir(parents=True)
    np.savez_compressed(output/'extension.npz',**{k:state[k] for k in ('hadm_ids','subject_ids','patient_splits')},
        new_labs=labs,eos=eos['eos_features'],med_details=meds,
        lab_values_raw=lv,lab_masks=lm,med_values_raw=mv,med_masks=mm)
    report = dict(status='source_checked_extension_not_trained',admissions=len(ids),
        parent_state_manifest_sha256=sha256(state_dir/'manifest.json'),eos_manifest_sha256=sha256(eos_dir/'manifest.json'),
        raw_source_sha256=hashes,lab_names=list(LABS),med_names=list(MOLECULES),
        lab_channels=list(LAB_CHANNELS),med_channels=list(MED_CHANNELS),coverage=coverage,audit=dict(audit),
        shapes=dict(new_labs=list(labs.shape),eos=list(eos['eos_features'].shape),med_details=list(meds.shape)),
        normalization=dict(labs=ln,medications=mn),output_sha256=sha256(output/'extension.npz'),
        seconds=time.perf_counter()-started,
        limits=['Counts are recorded eMAR events, not complete exposure or prescribed frequency.',
            'Doses only component-verified mg by molecule; no mL/puff/NEB or potency conversion; missing is masked.',
            'Missing execution route may use its linked visible active pharmacy order route; this is an order route, not observed route.',
            'Ordered frequency is median valid pharmacy order frequency, not executed frequency.',
            'No complete drug-lab response sequence or causal treatment effect is inferred.'])
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],coverage=coverage,seconds=report['seconds'])),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--state',type=Path,default=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2'))
    p.add_argument('--eos',type=Path,default=Path('patient_data/processed/v6_eos_med_context_2026-10-09_r3'))
    p.add_argument('--output',type=Path,default=Path('patient_data/processed/v6_detail_extension_2026-10-11_r2'))
    a=p.parse_args(); build(a.snapshot,a.state,a.eos,a.output)
