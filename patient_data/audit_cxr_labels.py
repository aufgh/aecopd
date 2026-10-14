"""独立只读回查CheXbert输入原文、时间、成员及四状态落盘一致性。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows, sha256

CONDITIONS=('Pneumonia','Consolidation','Edema','Cardiomegaly','Pleural Effusion','Pneumothorax','Atelectasis')


def verify(directory,snapshot,output):
    if output.exists(): raise FileExistsError(output)
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,digest in manifest['output_sha256'].items():
        if sha256(directory/name)!=digest: raise ValueError('CheXbert output changed')
    reports=[json.loads(line) for line in (directory/'labeled_reports.jsonl').read_text(encoding='utf-8').splitlines()]
    by_note={r['note_id']:r for r in reports}
    if len(by_note)!=len(reports): raise ValueError('duplicate selected note')
    admissions={r['hadm_id']:r for r in rows(snapshot/'cohort/admissions.csv.gz')}
    matched=0
    for original in rows(snapshot/'raw/radiology.csv.gz'):
        r=by_note.get(original['note_id'])
        if not r: continue
        if any(original[key]!=r[key] for key in ('subject_id','hadm_id','charttime','storetime')):
            raise ValueError('original identity/time differs')
        if hashlib.sha256(original['text'].encode()).hexdigest()!=r['original_sha256']:
            raise ValueError('original text hash differs')
        start,end=r['start_char'],r['end_char']
        if not 0<=start<=end<=len(original['text']) or original['text'][start:end]!=r['text']:
            raise ValueError('original section span differs')
        if hashlib.sha256(r['text'].encode()).hexdigest()!=r['section_sha256']: raise ValueError('section hash differs')
        adm=admissions[r['hadm_id']]
        begin,chart,store,discharge=map(datetime.fromisoformat,(adm['admittime'],r['charttime'],r['storetime'],adm['dischtime']))
        if not begin<=chart<=store<=discharge or adm['discharge_landmark_eligible']!='t': raise ValueError('prediction time violated')
        matched+=1
    if matched!=len(reports): raise ValueError('original source missing')
    with np.load(directory/'cxr_context.npz',allow_pickle=False) as archive: arrays={k:archive[k] for k in archive.files}
    position={str(h):i for i,h in enumerate(arrays['hadm_ids'])}
    seen=set()
    for r in reports:
        i=position[r['hadm_id']]
        if i in seen or not arrays['cxr_present'][i]: raise ValueError('duplicate or unobserved image row')
        if str(arrays['subject_ids'][i])!=r['subject_id'] or str(arrays['patient_splits'][i])!=r['patient_split']:
            raise ValueError('subject/split inconsistent')
        expected=np.array([r['labels14'][c] for c in CONDITIONS])
        if not np.array_equal(arrays['cxr_states'][i],expected): raise ValueError('labels do not match archive')
        seen.add(i)
    present=arrays['cxr_present']; states=arrays['cxr_states']
    onehot=(np.eye(4,dtype=np.float32)[states]*present[:,None,None]).reshape(len(states),28)
    if len(seen)!=int(present.sum()) or not np.array_equal(onehot,arrays['cxr_onehot']): raise ValueError('missingness/onehot inconsistent')
    result={'status':'all_selected_reports_original_time_identity_state_verified','reports':matched,
            'admissions':len(states),'manifest_sha256':sha256(directory/'manifest.json'),
            'limits':['Engineering provenance verification, not clinical semantic accuracy',
                      'Report classifier false positives/negatives and CheXbert semantic errors need separate review']}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=Path('patient_data/processed/v6_chexbert_cxr_2026-10-09'))
    p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); verify(a.directory,a.snapshot,a.output)
