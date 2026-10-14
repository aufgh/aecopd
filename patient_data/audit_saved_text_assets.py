"""只审计A/B旧向量索引覆盖，不将旧索引强行用作本次训练行号。"""
from collections import Counter
import argparse
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows,sha256


def audit(state,index,output):
    if output.exists(): raise FileExistsError(output)
    with np.load(state) as z:
        ids=set(map(str,z['hadm_ids'])); splits=dict(zip(map(str,z['hadm_ids']),map(str,z['patient_splits'])))
    coverage=set(); counts=Counter(); sections=Counter(); disagreements=Counter()
    for r in rows(index):
        counts['all_rows']+=1
        if r['document_role']=='current_discharge':
            counts['current_discharge_rows']+=1
            if r['hadm_id'] in ids:
                coverage.add(r['hadm_id']); counts['matched_current_discharge_rows']+=1
                sections[r['section_role']]+=1
                disagreements[(splits[r['hadm_id']],r['data_split'])]+=1
    result=dict(status='index_coverage_audit_not_content_or_runtime_alignment',
        source_index_sha256=sha256(index),current_cohort_admissions=len(ids),
        matched_current_discharge_admissions=len(coverage),counts=dict(counts),sections=dict(sections),
        fold_pairs={str(k):v for k,v in disagreements.items()},
        limits=['Only role/current-hadm coverage checked; original document hashes not matched yet.',
            'Fold labels match on overlap, but full cohort/prefix/section representation differs; no controlled full-text comparison.',
            'History-role arrays cannot be placed into current B chunk indices.'])
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--state',type=Path,default=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2/clinical_state.npz'))
    p.add_argument('--index',type=Path,default=Path('outputs/01a08b96-2988-7001-af1d-b40d901f3400/private/model_v2/qwen_section_embedding_index.csv.gz'))
    p.add_argument('--output',type=Path,default=Path('patient_data/text_encodings/A_full_document_sections/current_cohort_overlap_2026-10-11.json'))
    a=p.parse_args(); audit(a.state,a.index,a.output)
