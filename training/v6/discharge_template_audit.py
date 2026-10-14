"""训练折出院状态重复向量与原文核对；只保存聚合哈希/规则类别，不导出病历。"""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from training.v6.regularization_audit import checked
from training.v6.data_geometry_audit import ROOT
from training.v6.train import write_json

PATTERNS={'mental_status_normal':r'clear and coherent|alert and interactive|alert and oriented',
    'function_independent':r'ambulatory\s*[-:]?\s*independent',
    'respiratory_detail':r'dyspnea|shortness of breath|\bSOB\b|wheez|sputum|hypox|respiratory distress',
    'oxygen_or_device':r'oxygen|\bO2\b|nasal cannula|BiPAP|CPAP|\bNIV\b',
    # dependent不可作为independent的子串；这里只计词证据，仍不解释否定/临床真值。
    'dependence_or_limitation':r'\bassist\w*\b|\bdependent\b|\bwheelchair\b|\bwalker\b|\blimited\b|\bunable\b|\bbedbound\b'}


def main():
    d,_,_,_,_,_,_,_,_,_=checked()
    cfg=json.loads(Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09/config.json').read_text())
    path=Path(cfg['field_dir'])/'shared_text.csv.gz'
    training_rows={row for i,refs in enumerate(d.text_refs) if d.arrays['patient_splits'][i]=='train' for field,row in refs if field==3}
    texts=Counter();people=defaultdict(set);vector_to_text=defaultdict(set);flags=Counter();lengths={}
    for row in rows(path):
        ix=int(row['embedding_row'])
        if ix not in training_rows:continue
        text=row['chunk_text'];key=hashlib.sha256(text.encode()).hexdigest()
        if key!=row['text_sha256']:raise ValueError('text hash mismatch')
        texts[key]+=1;people[key].add(row['subject_id']);lengths[key]=len(text)
        vector_hash=hashlib.sha256(np.asarray(d.text_vectors[ix]).tobytes()).hexdigest();vector_to_text[vector_hash].add(key)
        for name,pattern in PATTERNS.items():
            if re.search(pattern,text,re.I):flags[name]+=1
    result=dict(status='training_source_hashes_matched',training_fragments=sum(texts.values()),unique_source_texts=len(texts),
        fragments_from_repeated_text_across_patients=sum(n for key,n in texts.items() if len(people[key])>1),
        vector_groups_with_multiple_source_texts=sum(len(v)>1 for v in vector_to_text.values()),
        lexical_flags=dict(flags),lexical_flags_not_clinical_truth=True,
        top_repeated_texts=[dict(text_sha256=key,fragments=n,patients=len(people[key]),characters=lengths[key]) for key,n in texts.most_common(5)],
        source_sha256=sha256(path),code_sha256=sha256(Path(__file__)),no_raw_text_or_patient_ids_in_report=True,
        limits=['repetition is not proof of irrelevance','normal function may be prognostically useful',
            'respiratory omission is not evidence of clinical absence','no deletion or predictive experiment from these lexical counts'])
    write_json(ROOT/'discharge_template_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
