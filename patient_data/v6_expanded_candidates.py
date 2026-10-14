"""独立扩展字典和派生数据；旧已哈希来源/生成器不修改。"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import argparse
import json
import re
import numpy as np
from patient_data import medication_identity as old
from patient_data.export_mimiciv import rows, sha256
from patient_data.eos_med_context import finite
from training.v5.numeric_preprocessing import robust_parameters

NEW_DRUGS=('levalbuterol','arformoterol','umeclidinium','beclomethasone','mometasone',
 'fluticasone_vilanterol','tiotropium_olodaterol','glycopyrrolate_formoterol',
 'roflumilast','theophylline','acetylcysteine','guaifenesin',
 'amoxicillin_clavulanate','ampicillin_sulbactam','cefepime','moxifloxacin','clarithromycin')
NAMES=old.NAMES+NEW_DRUGS
ALIASES={'levalbuterol':('xopenex',),'arformoterol':('brovana',),'umeclidinium':('incruse',),
 'beclomethasone':('qvar',),'mometasone':('asmanex',),'fluticasone_vilanterol':('breo',),
 'tiotropium_olodaterol':('stiolto',),'glycopyrrolate_formoterol':('bevespi',),
 'roflumilast':('daliresp',),'theophylline':('theo-24','theodur','theo-dur'),
 'acetylcysteine':('mucomyst',),'guaifenesin':('mucinex',),
 'amoxicillin_clavulanate':('augmentin',),'ampicillin_sulbactam':('unasyn',)}
ORIGINAL_IDENTITY=old.identity
INHALED=set(old.INHALED)|set(NEW_DRUGS[:8])|{'acetylcysteine'}
# 候选理由先于结局；同一项目的跨itemid来源只在单位相同时合并。
LABS={
 'neutrophils_absolute':(('52075','53159'),'K/uL','炎症/感染细胞组成'),
 'lymphocytes_absolute':(('51133','52769','53157'),'K/uL','免疫细胞组成'),
 'monocytes_percent':(('51254',),'%','炎症细胞组成'),
 'hematocrit':(('51221',),'%','贫血/红细胞增多与缺氧背景'),
 'mcv':(('51250',),'fL','贫血/营养表型'),
 'calcium_total':(('50893',),'mg/dL','电解质和全身病情背景'),
 'phosphate':(('50970',),'mg/dL','呼吸肌能量/电解质背景'),
 'protein_total':(('50976',),'g/dL','营养/炎症背景'),
 'fibrinogen':(('51214',),'mg/dL','有COPD加重/死亡关联证据的炎症凝血候选'),
 'troponin_t':(('51003',),'ng/mL','心肺重叠及心肌损伤背景，不是COPD特异指标')}


def identity(name):
    s=str(name).lower()
    # 祛痰镇咳复方不当作纯guaifenesin累计剂量；本轮未建独立复方槽位。
    if ('guaifenesin' in s or 'mucinex' in s) and any(x in s for x in ('codeine','dextromethorphan',' dm')):return None
    combos={'fluticasone_vilanterol':('fluticasone','vilanterol'),
      'tiotropium_olodaterol':('tiotropium','olodaterol'),
      'glycopyrrolate_formoterol':('glycopyrrolate','formoterol'),
      'amoxicillin_clavulanate':('amoxicillin','clav'),
      'ampicillin_sulbactam':('ampicillin','sulbactam')}
    # 排除本轮仍未建独立接口的复方，不能把它们误认成单药。
    if ('trelegy' in s or all(x in s for x in ('fluticasone','umeclidinium','vilanterol'))
        or 'dulera' in s or all(x in s for x in ('mometasone','formoterol'))):return None
    for n,parts in combos.items():
        if all(x in s for x in parts) or any(x in s for x in ALIASES.get(n,())):return n
    hits=[n for n in NEW_DRUGS if re.search(r'\b'+n+r'\b',s) or any(x in s for x in ALIASES.get(n,()))]
    prior=ORIGINAL_IDENTITY(s)
    if prior:hits.append(prior)
    return hits[0] if len(set(hits))==1 else None


@contextmanager
def medication_catalog():
    """显式单进程字典适配：仅派生期间生效；终于恢复旧模块，不热改旧文件。"""
    saved=old.NAMES,old.INHALED,old.identity
    old.NAMES,old.INHALED,old.identity=NAMES,INHALED,identity
    try:yield
    finally:old.NAMES,old.INHALED,old.identity=saved


def summarize(observations,end):
    if not observations:return np.zeros(5,np.float32),np.zeros(5,bool)
    # 同项目同时间冲突不求任意平均；完全相同记录去重。
    buckets=defaultdict(set)
    for t,v in observations:buckets[t].add(v)
    items=sorted((t,next(iter(v))) for t,v in buckets.items() if len(v)==1)
    if not items:return np.zeros(5,np.float32),np.zeros(5,bool)
    values=[v for _,v in items]
    return np.asarray((values[0],min(values),max(values),values[-1],
           np.log1p((end-items[-1][0]).total_seconds()/86400)),np.float32),np.ones(5,bool)


def build_labs(snapshot,state_dir,output):
    if output.exists():raise FileExistsError(output)
    sm=json.loads((snapshot/'metadata/manifest.json').read_text());st=json.loads((state_dir/'manifest.json').read_text())
    if sm['status']!='validated' or sha256(state_dir/'clinical_state.npz')!=st['output_sha256']['clinical_state.npz']:raise ValueError('invalid parent')
    with np.load(state_dir/'clinical_state.npz') as z:state={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits')}
    ix={str(h):i for i,h in enumerate(state['hadm_ids'])};sources={}
    for name in ('raw/labevents','raw/d_labitems','cohort/admissions'):
        p=snapshot/(name+'.csv.gz');sources[name]=sha256(p)
        if sources[name]!=sm['sources'][name]['sha256']:raise ValueError('raw changed')
    admissions={r['hadm_id']:(datetime.fromisoformat(r['admittime']),datetime.fromisoformat(r['dischtime']))
      for r in rows(snapshot/'cohort/admissions.csv.gz') if r['hadm_id'] in ix}
    dictionary={r['itemid']:r for r in rows(snapshot/'raw/d_labitems.csv.gz')}
    mapping={item:(j,unit) for j,(items,unit,_) in enumerate(LABS.values()) for item in items}
    obs=defaultdict(list);audit=Counter();units=defaultdict(Counter)
    for r in rows(snapshot/'raw/labevents.csv.gz'):
        h=r['hadm_id']
        if h not in ix or r['itemid'] not in mapping:continue
        j,unit=mapping[r['itemid']];name=list(LABS)[j];units[name][r['valueuom']]+=1
        try:t,s=datetime.fromisoformat(r['charttime']),datetime.fromisoformat(r['storetime'])
        except ValueError:audit[name+'/invalid_time']+=1;continue
        v=finite(r['valuenum']);start,end=admissions[h]
        if not start<=t<=s<=end:audit[name+'/not_visible']+=1;continue
        if v is None or v<0 or (unit=='%' and v>100) or r['valueuom']!=unit:
            audit[name+'/invalid_value_or_unit']+=1;continue
        obs[h,j].append((t,v));audit[name+'/accepted']+=1
    raw=np.zeros((len(ix),len(LABS),5),np.float32);mask=np.zeros_like(raw,bool)
    for (h,j),items in obs.items():raw[ix[h],j],mask[ix[h],j]=summarize(items,admissions[h][1])
    train=state['patient_splits']=='train';scaled=np.zeros_like(raw);normalization={};coverage={}
    for j,name in enumerate(LABS):
        for k in range(5):
            fit=train&mask[:,j,k]
            if not fit.any():mask[:,j,k]=False;continue
            # raw age already log-days; concentration channels transform log1p.
            x=np.log1p(raw[:,j,k]) if k<4 else raw[:,j,k]
            p=robust_parameters(x[fit]);normalization[f'{name}/{k}']=p
            valid=mask[:,j,k];scaled[valid,j,k]=np.clip((x[valid]-p['median'])/p['scale'],-8,8)
        observed=mask[:,j,0];coverage[name]=dict(admissions=int(observed.sum()),
          patients=int(len(np.unique(state['subject_ids'][observed]))),train_patients=int(len(np.unique(state['subject_ids'][observed&train]))),
          missing_admissions=int((~observed).sum()),raw_units=dict(units[name]))
    features=np.concatenate((scaled,mask.astype(np.float32)),2);output.mkdir(parents=True)
    np.savez_compressed(output/'expanded_labs.npz',**state,features=features,values_raw=raw,value_masks=mask,names=np.asarray(list(LABS)))
    (output/'normalization.json').write_text(json.dumps(normalization,indent=2),encoding='utf-8')
    manifest=dict(status='source_checked_expanded_labs',parent_state_manifest_sha256=sha256(state_dir/'manifest.json'),
      output_sha256=sha256(output/'expanded_labs.npz'),sources_sha256=sources,catalog=LABS,
      dictionary={i:dictionary[i] for ids,_,_ in LABS.values() for i in ids},coverage=coverage,audit=dict(audit),
      builder_sha256=sha256(Path(__file__)),rules='charttime<=storetime<=discharge; exact units; conflicting simultaneous values excluded; train-only scaling')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'],coverage=coverage)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--state',type=Path,default=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2'))
    p.add_argument('--output',type=Path,default=Path('patient_data/processed/v6_expanded_candidates_2026-10-12_r2'));a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    with medication_catalog():old.build(a.snapshot,a.state,a.output/'medications')
    build_labs(a.snapshot,a.state,a.output/'labs')


if __name__=='__main__':main()
