"""全输入审计及独立粗细接口；只读旧来源，训练折拟合新尺度。"""
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from scipy.stats import spearmanr
from patient_data.export_mimiciv import rows, sha256
from patient_data.compact_multisource import order_available, INHALED_ROUTES, ORDER_SYSTEMIC_ROUTES
from patient_data.v6_expanded_medication_fix import identity
from patient_data.v6_expanded_candidates import INHALED
from patient_data.eos_med_context import finite
from patient_data.medication_identity import UNITS
from training.v5.numeric_preprocessing import robust_parameters
from training.v6.expanded_original_experiment import checked, parser as parent_parser
from training.v6.data import HISTORY_NAMES, CURRENT_NAMES

ROOT = Path('patient_data/processed/v6_audited_coarse_fine_2026-10-12_r2')
LAB_ZH = ('动脉低pH','动脉高二氧化碳分压','低碳酸氢根','高碳酸氢根','高白细胞','低白蛋白','低血红蛋白','高肌酐',
 '高碳酸酸中毒','低白细胞','低钠','高钠','低钾','高钾','高尿素氮','低血糖','高血糖','低血小板','低镁','高乳酸','高CRP','高NT-proBNP',
 '中性粒细胞百分比','淋巴细胞百分比','红细胞分布宽度','阴离子间隙','氯','嗜酸粒细胞绝对值','嗜酸粒细胞百分比',
 '中性粒细胞绝对值','淋巴细胞绝对值','单核细胞百分比','红细胞压积','平均红细胞体积','总钙','磷','总蛋白','纤维蛋白原','肌钙蛋白T')
LAB_NAMES = ('arterial_low_ph','arterial_high_pco2','bicarbonate_low','bicarbonate_high','wbc_high','albumin_low','hemoglobin_low',
 'creatinine_high','arterial_hypercapnic_acidosis','wbc_low','sodium_low','sodium_high','potassium_low','potassium_high',
 'urea_nitrogen_high','glucose_low','glucose_high','platelet_low','magnesium_low','lactate_high','crp_high','ntprobnp_high',
 'neutrophils_percent','lymphocytes_percent','rdw','anion_gap','chloride','eosinophils_absolute','eosinophils_percent',
 'neutrophils_absolute','lymphocytes_absolute','monocytes_percent','hematocrit','mcv','calcium_total','phosphate','protein_total','fibrinogen','troponin_t')
LAB_FAMILY = (0,0,0,0,1,2,2,3,0,1,4,4,4,4,3,5,5,2,4,5,1,6,1,1,2,0,4,1,1,1,1,1,2,2,4,4,2,1,6)
FAMILIES = ('acid_base','inflammation','hematology_nutrition','renal','electrolytes','metabolic_perfusion','cardiac')
MERGE = ((2,3),(4,9),(10,11),(12,13),(15,16))
COARSE_DRUGS = ('SABA','SAMA','LABA','LAMA','ICS','systemic_steroid','antibacterial','mucus_targeted','methylxanthine','PDE4_inhibitor')
DRUG_PARENTS = {
 'albuterol':(0,), 'levalbuterol':(0,), 'ipratropium':(1,), 'salmeterol':(2,), 'formoterol':(2,), 'arformoterol':(2,),
 'tiotropium':(3,), 'umeclidinium':(3,), 'fluticasone':(4,), 'budesonide':(4,), 'beclomethasone':(4,), 'mometasone':(4,),
 'albuterol_ipratropium':(0,1),'fluticasone_salmeterol':(4,2),'budesonide_formoterol':(4,2),
 'umeclidinium_vilanterol':(3,2),'fluticasone_vilanterol':(4,2),'tiotropium_olodaterol':(3,2),'glycopyrrolate_formoterol':(3,2),
 'prednisone':(5,), 'methylprednisolone':(5,), 'hydrocortisone':(5,), 'dexamethasone':(5,),
 'azithromycin':(6,), 'ceftriaxone':(6,), 'levofloxacin':(6,), 'doxycycline':(6,), 'piperacillin_tazobactam':(6,),
 'vancomycin':(6,), 'amoxicillin_clavulanate':(6,), 'ampicillin_sulbactam':(6,), 'cefepime':(6,), 'moxifloxacin':(6,), 'clarithromycin':(6,),
 'acetylcysteine':(7,), 'guaifenesin':(7,), 'theophylline':(8,), 'roflumilast':(9,)}


def scalar_dose(value, unit):
    """不猜范围、浓度或复方有效成分比例；质量单位统一mg，其余保留剂型单位。"""
    v=finite(value); u=str(unit).strip().lower()
    factors={'mg':1.,'mcg':.001,'ug':.001,'g':1000.,'gm':1000.}
    aliases={'puff':'puff','puffs':'puff','neb':'neb','nebs':'neb','inh':'inh','ml':'ml'}
    if v is None or v<=0:return None
    if u in factors:return v*factors[u],UNITS.index('mg')
    if u in aliases:return v,UNITS.index(aliases[u])
    return None


def order_doses(snapshot, ids, drug_names):
    """按pharmacy_id联接可用时间与状态，处方只作缺失单次剂量的补充。"""
    ix={str(h):i for i,h in enumerate(ids)}; di={n:j for j,n in enumerate(drug_names)}
    admissions={r['hadm_id']:(datetime.fromisoformat(r['admittime']),datetime.fromisoformat(r['dischtime']))
                for r in rows(snapshot/'cohort/admissions.csv.gz') if r['hadm_id'] in ix}
    pharmacy={}; observed=np.zeros(len(ids),bool)
    for r in rows(snapshot/'raw/pharmacy.csv.gz'):
        h=r['hadm_id']
        if h in ix and order_available(r,admissions[h])[0]:
            pharmacy[h,r['pharmacy_id']]=r; observed[ix[h]]=True
    obs=defaultdict(list); seen=set(); reasons=Counter(); iv_vanco_frequency=defaultdict(list)
    for r in rows(snapshot/'raw/prescriptions.csv.gz'):
        h=r['hadm_id']; p=pharmacy.get((h,r['pharmacy_id']))
        if h not in ix or p is None:continue
        drug=identity(r['drug'])
        if drug not in di:continue
        route=r.get('route','').strip().upper()
        if route not in (INHALED_ROUTES if drug in INHALED else ORDER_SYSTEMIC_ROUTES):continue
        if drug=='vancomycin' and route not in {'IV','IV DRIP','IV PUSH','IV BOLUS','INTRAVENOUS'}:
            reasons['oral_or_noniv_vancomycin_order_not_systemic']+=1;continue
        if not order_available(r,admissions[h],entered=datetime.fromisoformat(p['entertime']),status=p['status'])[0]:continue
        value=scalar_dose(r.get('dose_val_rx'),r.get('dose_unit_rx'))
        if value is None:reasons['unparseable_scalar_or_unit']+=1;continue
        key=(h,r['pharmacy_id'],drug,r['starttime'],r['stoptime'],value)
        if key in seen:reasons['duplicate_order_dose']+=1;continue
        seen.add(key);obs[ix[h],di[drug]].append(value)
        if drug=='vancomycin':
            frequency=finite(p.get('doses_per_24_hrs'))
            if frequency is not None and 0<frequency<=48:iv_vanco_frequency[ix[h]].append(frequency)
    raw=np.zeros((len(ids),len(drug_names)),np.float32);mask=np.zeros_like(raw,bool);unit=np.zeros_like(raw,np.int64)
    for (i,j),items in obs.items():
        units={u for _,u in items}
        if len(units)!=1:reasons['mixed_order_units_not_combined']+=1;continue
        unit[i,j]=next(iter(units)); raw[i,j]=max(v for v,_ in items);mask[i,j]=True
    vf=np.zeros(len(ids),np.float32);vfm=np.zeros(len(ids),bool)
    for i,values in iv_vanco_frequency.items():vf[i]=np.median(values);vfm[i]=True
    return raw,mask,unit,observed,dict(reasons),vf,vfm


def scale_per_unit(raw, masks, units, train, names):
    x=np.zeros_like(raw);parameters={}
    for j,name in enumerate(names):
        for k in range(raw.shape[2]):
            # 通道0是单次剂量，其余为记录或计划频率；不能跨剂型拟合剂量尺度。
            for unit in (range(len(UNITS)) if k==0 else (-1,)):
                valid=masks[:,j,k] & (units[:,j]==unit if unit>=0 else True)
                fitted=valid&train
                if not fitted.any():masks[valid,j,k]=False;continue
                p=robust_parameters(np.log1p(raw[fitted,j,k]));parameters[f'{name}/{k}/{unit}']=p
                x[valid,j,k]=np.clip((np.log1p(raw[valid,j,k])-p['median'])/p['scale'],-8,8)
    return x,parameters


def run(output):
    if output.exists():raise FileExistsError(output)
    args=parent_parser().parse_args([]);args.expanded=Path('patient_data/processed/v6_expanded_candidates_2026-10-12_r3')
    d,c,h,ext,added,med,prov,active,norm_checks=checked(args)
    train=d.arrays['patient_splits']=='train'; subjects=d.arrays['subject_ids']; n=len(train)
    # checked为了旧实验屏蔽稀有新药；这里重新只读加载，供完整审计与上卷类别。
    with np.load(args.expanded/'medications/medication_identity.npz') as z:med={k:z[k] for k in z.files}
    output.mkdir(parents=True)
    catalog=[]
    def coverage(group,name,observed,positive=None,**extra):
        observed=np.asarray(observed,bool);positive=observed if positive is None else np.asarray(positive,bool)
        item=dict(group=group,name=name,observed_admissions=int(observed.sum()),observed_percent=round(100*observed.mean(),3),
          missing_admissions=int((~observed).sum()),observed_patients=int(len(np.unique(subjects[observed]))),
          train_patients=int(len(np.unique(subjects[observed&train]))),positive_admissions=int(positive.sum()),
          positive_train_patients=int(len(np.unique(subjects[positive&train]))),**extra)
        catalog.append(item);return item
    lab=np.zeros((n,39,24),np.float32); lp=np.zeros((n,39),bool)
    lab[:,:22,:9]=d.arrays['lab_axis_inputs'];lp[:,:22]=lab[:,:22,6]>0
    lab[:,22:27,:10]=ext['new_labs'];lp[:,22:27]=ext['lab_masks'].any(2)
    lab[:,27:29,:20]=ext['eos'];lp[:,27:29]=ext['eos'][:,:,6:12].any(2)
    lab[:,29:,:10]=added['features'];lp[:,29:]=added['value_masks'].any(2)
    for j,name in enumerate(LAB_NAMES):
        coverage('laboratory',name,lp[:,j],lp[:,j]&(d.arrays['lab_axis_raw'][:,j,1]>0) if j<22 else None,
          chinese=LAB_ZH[j],source='labevents + d_labitems; specimen link for arterial gas',
          post_peak_rechecked=int((lab[:,j,7]>0).sum()) if j<22 else None,
          proposal='clinical_core_keep' if j<29 else 'candidate_keep',
          positive_definition='abnormal peak' if j<22 else 'observed, not abnormality classification')
    # 高低双轴合到一个概念，完整保留各9通道，不求平均。
    merged=set(v for pair in MERGE for v in pair);spec=[]
    for j in range(39):
        pair=next((p for p in MERGE if p[0]==j),None)
        if pair:spec.append((pair,LAB_NAMES[j].removesuffix('_low').removesuffix('_high'),LAB_FAMILY[j]))
        elif j not in merged:spec.append(((j,),LAB_NAMES[j],LAB_FAMILY[j]))
    merged_lab=[];merged_present=[];lnames=[];families=[]
    for indices,name,family in spec:
        x=lab[:,indices[0]].copy();p=lp[:,indices[0]].copy()
        if len(indices)==2:x[:,9:18]=lab[:,indices[1],:9];p|=lp[:,indices[1]]
        merged_lab.append(x);merged_present.append(p);lnames.append(name);families.append(family)
    merged_lab=np.stack(merged_lab,1);merged_present=np.stack(merged_present,1)
    # 先验低特异/重叠候选暂离主配方；不删除原数据，需后续同架构整组比较。
    excluded_lab={'hematocrit','mcv','monocytes_percent','protein_total'}
    lab_keep=np.array([name not in excluded_lab for name in lnames])
    duplicates=[]
    for j in range(39):
        for k in range(j+1,39):
            if np.array_equal(lp[:,j],lp[:,k]):duplicates.append(dict(kind='availability',left=LAB_NAMES[j],right=LAB_NAMES[k]))
    correlations=[]
    rawextras=np.concatenate((ext['lab_values_raw'],added['values_raw']),1)
    masks=np.concatenate((ext['lab_masks'],added['value_masks']),1);names=list(LAB_NAMES[22:27])+list(LAB_NAMES[29:])
    for j in range(len(names)):
        for k in range(j+1,len(names)):
            valid=train&masks[:,j,3]&masks[:,k,3]
            if valid.sum()<200:continue
            a,b=rawextras[valid,j,3],rawextras[valid,k,3]
            if np.std(a)==0 or np.std(b)==0:continue
            rho=float(spearmanr(a,b).statistic)
            if abs(rho)>=.8:correlations.append(dict(left=names[j],right=names[k],n=int(valid.sum()),rho=rho))
    ordered,om,ou,order_source,oa,vf,vfm=order_doses(args.snapshot,d.arrays['hadm_ids'],list(med['drug_names']))
    raw=med['values_raw'];mask=med['value_masks'];units=med['dose_unit_id']
    # 不把全期累计量当单次剂量；实际最大单次优先，医嘱中位单次补缺。
    dose=np.where(mask[:,:,5],raw[:,:,5],ordered);dose_mask=mask[:,:,5]|om
    dose_unit=np.where(mask[:,:,5],units,ou)
    freq=np.where(raw[:,:,0]>0,raw[:,:,3],raw[:,:,6]);freqmask=(raw[:,:,0]>0)|mask[:,:,6]
    # 旧万古霉素汇总混合途径，不能声称是全身抗感染。新接口只用已核对IV医嘱。
    van=list(med['drug_names']).index('vancomycin')
    dose[:,van]=ordered[:,van];dose_mask[:,van]=om[:,van];dose_unit[:,van]=ou[:,van]
    freq[:,van]=vf;freqmask[:,van]=vfm
    vals=np.stack((dose,freq),2);vm=np.stack((dose_mask,freqmask),2)
    scaled,norm=scale_per_unit(vals,vm,dose_unit,train,list(med['drug_names']))
    present=med['present']|om
    present[:,van]=om[:,van]|vfm
    med_features=np.concatenate((scaled,vm.astype(np.float32),np.eye(len(UNITS),dtype=np.float32)[dose_unit]),2)*present[:,:,None]
    # 来源及记录次数不直接入网；记录频率与医嘱频率互补，含义为documented schedule。
    coarse=np.zeros((n,10,2),np.float32);coarse[:,:7,0]=c.arrays['med_features'][:,:,:2].any(2)
    coarse[:,:,1]=c.arrays['med_features'][:,:,2:].any((1,2))[:,None]|order_source[:,None]
    drug_parent=np.zeros((len(med['drug_names']),10),np.float32);keep=[]
    for j,name in enumerate(med['drug_names']):
        drug_parent[j,list(DRUG_PARENTS[name])]=1
        for g in DRUG_PARENTS[name]:coarse[:,g,0]=np.maximum(coarse[:,g,0],present[:,j])
        row=coverage('drug_identity',str(name),present[:,j],source='emar + pharmacy + linked prescriptions',
          actual_single_dose=int(mask[:,j,5].sum()),ordered_single_dose=int(om[:,j].sum()),
          dose_after_complement=int(vm[:,j,0].sum()),frequency_after_complement=int(vm[:,j,1].sum()),
          new_order_dose_only=int((om[:,j]&~mask[:,j,5]).sum()),old_enabled=bool(active['drug'][str(name)]['in_model']))
        keep.append(row['train_patients']>=20)
        row['proposal']='fine_leaf_keep' if keep[-1] else 'parent_only_no_fine_embedding'
    for j,name in enumerate(COARSE_DRUGS):coverage('drug_class',name,coarse[:,j,1]>0,coarse[:,j,0]>0,source='union treatment evidence')
    for prefix in ('micro','support'):
        for j,name in enumerate(c.arrays[prefix+'_names']):
            coverage(prefix,str(name),c.arrays[prefix+'_masks'][:,j],c.arrays[prefix+'_values'][:,j]>0,
              source='microbiologyevents' if prefix=='micro' else 'restricted ICU chartevents',proposal='keep_compact_no_species_expansion')
    for j,name in enumerate(HISTORY_NAMES):coverage('frozen_history',name,np.ones(n,bool),d.history_raw[:,j]!=0,proposal='unchanged frozen H34')
    for j,name in enumerate(CURRENT_NAMES):coverage('frozen_current',name,d.current_masks[:,j],d.current[:,j]>0,proposal='unchanged frozen H34')
    text=np.zeros((n,4),int)
    for i,refs in enumerate(d.text_refs):
        for field,_ in refs:text[i,field]+=1
    for j,name in enumerate(('history_of_present_illness','past_medical_history','hospital_course','discharge_condition')):
        coverage('text',name,text[:,j]>0,chunks=int(text[:,j].sum()),source='saved B four-field Bio_ClinicalBERT',proposal='retain frozen vectors')
    micro=np.concatenate((c.arrays['micro_values'],c.arrays['micro_masks'][:,[0,3,4]]),1).astype(np.float32)
    if not all(np.array_equal(c.arrays['micro_masks'][:,0],c.arrays['micro_masks'][:,j]) for j in (1,2)):raise ValueError('culture masks differ')
    # 同一培养可用标记只传一次；细菌类别/病毒/耐药内容不删除。
    support=np.concatenate((c.arrays['support_values'],c.arrays['support_masks']),1).astype(np.float32)
    np.savez_compressed(output/'interface.npz',hadm_ids=d.arrays['hadm_ids'],subject_ids=subjects,patient_splits=d.arrays['patient_splits'],
      labs=merged_lab,lab_present=merged_present,lab_names=np.asarray(lnames),lab_family=np.asarray(families),lab_keep=lab_keep,
      medications=med_features,med_present=present,med_names=med['drug_names'],med_keep=np.asarray(keep),drug_parent=drug_parent,
      coarse=coarse,micro=micro,micro_present=c.arrays['micro_masks'].any(1),support=support,support_present=c.arrays['support_masks'].any(1))
    (output/'normalization.json').write_text(json.dumps(norm,indent=2),encoding='utf-8')
    keys=list(dict.fromkeys(k for row in catalog for k in row))
    with (output/'all_inputs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(catalog)
    audit=dict(status='audited_interface_not_trained',admissions=n,patients=int(len(np.unique(subjects))),rows=catalog,
      duplicate_availability=duplicates,train_observed_latest_correlations=correlations,order_dose_audit=oa,
      old_axes=39,merged_concepts=len(lnames),compact_concepts=int(lab_keep.sum()),all_drug_leaves=len(keep),compact_drug_leaves=int(sum(keep)),
      excluded_lab_candidates=sorted(excluded_lab),micro_repeated_masks_removed=2,
      medication_contract='union documented treatment; actual maximum single dose when complete, otherwise maximum scalar ordered single dose; actual or planned frequency; no origin channels or count/span; vancomycin IV orders only',
      member_sha256=sha256(args.snapshot/'cohort/admissions.csv.gz'),interface_sha256=sha256(output/'interface.npz'),
      sources=prov,source_code_sha256=sha256(Path(__file__)),normalization_errors=norm_checks,
      limits=['No raw deletion; threshold is train patients >=20 for fine leaf, parent retained.','Candidate clinical simplification not yet adopted.',
              'Prescribed dose is not certified administered; cumulative dose not synthesized from prescription.',
              'Existing vancomycin executed summaries not route-separated and excluded from new interface; only verified IV order retained, indication unknown.',
              'Latest-value correlations are feature-level, not necessarily same-specimen paired measurements.'])
    (output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k in ('status','admissions','patients','old_axes','merged_concepts','compact_concepts','compact_drug_leaves')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT);run(p.parse_args().output)
