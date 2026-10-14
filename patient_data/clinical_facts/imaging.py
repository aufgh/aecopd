"""追回明确胸片，分离胸部CT，保存全部合格报告的逐字章节。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from patient_data.cxr_report_selection import identify, HEADER, usable
from patient_data.export_mimiciv import rows, sha256

CHEST = re.compile(r'\b(chest|thorax|thoracic|lungs?)\b', re.I)
CT = re.compile(r'\b(CT|CTA|MDCT|HRCT|COMPUTED TOMOGRAPHY)\b', re.I)
OTHER = re.compile(r'\b(MRI|MR|ULTRASOUND|SONOGRAM|FLUOROSCOPY|BIOPSY)\b', re.I)
VIEW = re.compile(r'\b(radiographs?|x[ -]?rays?|portable|AP|PA|lateral|frontal|views?|port)\b',re.I)
EXPLICIT_CXR = re.compile(r'^(?:(?:portable\s+)?(?:AP|PA|frontal)(?:\s+(?:and\s+)?lateral)?\s+(?:portable\s+)?(?:chest|radiograph)|chest\s+(?:radiographs?|x[ -]?rays?|views?))\s+(?:was\s+|were\s+|is\s+|are\s+)?(?:obtained|performed|demonstrates?|shows?|reveals?)\b', re.I)


def sections(text):
    result=[]; headings=list(HEADER.finditer(text))
    for i,h in enumerate(headings):
        kind=h.group(1).upper()
        if kind not in ('FINDINGS','IMPRESSION','CONCLUSION'): continue
        start=h.end(); end=headings[i+1].start() if i+1<len(headings) else len(text)
        left=text[start:end]; offset=len(left)-len(left.lstrip()); value=left.strip()
        if value:
            start+=offset; end=start+len(value)
            result.append({'kind':kind,'start_char':start,'end_char':end,'text':value})
    return result


def modality(text):
    category,title=identify(text)
    headings=list(HEADER.finditer(text)); actual=[]
    for i,h in enumerate(headings):
        if h.group(1).upper() in ('EXAM','EXAMINATION','TECHNIQUE'):
            end=headings[i+1].start() if i+1<len(headings) else len(text)
            actual.append(text[h.end():end].strip())
    if title: actual.append(title)
    for segment in actual:
        if CT.search(segment):
            return ('chest_ct' if CHEST.search(segment) else 'other'), 'actual_CT_exam_or_technique'
        if OTHER.search(segment): return 'other','actual_non_CXR_exam_or_technique'
        if CHEST.search(segment) and re.search(r'\b(?:axial|helical)\s+(?:CT\s+)?images?\b',segment,re.I):
            return 'chest_ct','actual_axial_chest_technique'
    if category.startswith('cxr_'): return 'cxr',category
    stop=min((h.start() for h in headings),default=len(text))
    current_titles=actual+[line.strip() for line in text[:stop].splitlines()[:12]]
    for title in current_titles:
        if CHEST.search(title) and VIEW.search(title) and not CT.search(title) and not OTHER.search(title):
            return 'cxr','explicit_current_chest_views'
    # 未识别的明确非胸部EXAM不得被正文比较/指征提到的胸片改写。
    if category=='unrecognized_exam':
        if title and not CHEST.search(title): return 'other','explicit_non_chest_exam'
    for section in sections(text):
        if section['kind']=='FINDINGS' and EXPLICIT_CXR.search(' '.join(section['text'][:800].split())):
            return 'cxr','explicit_CXR_in_current_findings'
    return ('other' if category.startswith('non_cxr') else 'unknown'),category


def build(snapshot,state,output):
    if output.exists(): raise FileExistsError(output)
    raw=json.loads((snapshot/'metadata/manifest.json').read_text(encoding='utf-8'))
    sm=json.loads((state/'manifest.json').read_text(encoding='utf-8'))
    if raw['status']!='validated' or sm['status']!='derived_and_roundtrip_checked_not_prediction_validated':
        raise ValueError('unvalidated source')
    for source in ('raw/radiology','cohort/admissions'):
        if sha256(snapshot/(source+'.csv.gz'))!=raw['sources'][source]['sha256']: raise ValueError('source changed')
    if sha256(state/'clinical_state.npz')!=sm['output_sha256']['clinical_state.npz']: raise ValueError('state changed')
    with np.load(state/'clinical_state.npz',allow_pickle=False) as f: members={k:f[k] for k in ('hadm_ids','subject_ids','patient_splits')}
    admissions={r['hadm_id']:r for r in rows(snapshot/'cohort/admissions.csv.gz') if r['discharge_landmark_eligible']=='t'}
    if set(admissions)!=set(map(str,members['hadm_ids'])): raise ValueError('members differ')
    split={str(h):str(s) for h,s in zip(members['hadm_ids'],members['patient_splits'],strict=True)}
    records={'cxr':[],'chest_ct':[]}; audit=Counter(); seen=set()
    for row in rows(snapshot/'raw/radiology.csv.gz'):
        audit['raw_reports']+=1
        if row['note_id'] in seen: raise ValueError('duplicate radiology note')
        seen.add(row['note_id'])
        if row['hadm_id'] not in admissions: audit['not_eligible']+=1; continue
        kind,why=modality(row['text']); audit['modality_'+kind]+=1
        okay,reason=usable(row,admissions[row['hadm_id']])
        if not okay: audit[reason]+=1; continue
        audit['available_'+kind]+=1
        if kind not in records: continue
        spans=sections(row['text'])
        if not spans: audit['no_result_section_'+kind]+=1; continue
        for span in spans:
            if not 0<=span['start_char']<=span['end_char']<=len(row['text']) or row['text'][span['start_char']:span['end_char']]!=span['text']:
                raise ValueError('invalid span')
        record={k:row[k] for k in ('subject_id','hadm_id','note_id','charttime','storetime')}
        record.update(modality=kind,classification_basis=why,original_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                      patient_split=split[row['hadm_id']],sections=spans,text='\n'.join(s['text'] for s in spans))
        records[kind].append(record)
    if audit['raw_reports']!=raw['sources']['raw/radiology']['rows']: raise ValueError('raw count differs')
    output.mkdir(parents=True)
    for kind,values in records.items():
        values.sort(key=lambda r:(int(r['hadm_id']),r['charttime'],r['storetime'],r['note_id']))
        with (output/(kind+'_reports.jsonl')).open('x',encoding='utf-8') as f:
            for record in values: f.write(json.dumps(record,ensure_ascii=False)+'\n')
    np.savez_compressed(output/'members.npz',**members)
    result={'status':'all_pre_discharge_imaging_selected','parent_state_manifest_sha256':sha256(state/'manifest.json'),
            'source_manifest_sha256':sha256(snapshot/'metadata/manifest.json'),'raw_radiology_sha256':sha256(snapshot/'raw/radiology.csv.gz'),
            'admissions':len(admissions),'audit':dict(audit),'code_sha256':sha256(Path(__file__)),
            'coverage':{k:{'reports':len(v),'admissions':len({r['hadm_id'] for r in v})} for k,v in records.items()},
            'output_sha256':{n:sha256(output/n) for n in ('cxr_reports.jsonl','chest_ct_reports.jsonl','members.npz')},
            'limits':['Rule-based modality classification is not an expert gold standard','All results sections; no future reports; unknown not forced to CXR']}
    (output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--state',type=Path,default=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); build(a.snapshot,a.state,a.output)
