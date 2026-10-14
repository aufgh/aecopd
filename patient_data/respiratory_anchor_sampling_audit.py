"""原锚点采样密度聚合审计；只计训练患者，不改变模型输入或源码合同。"""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.respiratory_transition import ROOT,SOURCE,ITEMS,NAMES
from training.v6.train import write_json


def main():
    with np.load(ROOT/'respiratory_transition.npz') as z:
        train={str(h) for h,s in zip(z['hadm_ids'],z['patient_splits']) if s=='train'}
    counts={};selected=defaultdict(int)
    path=SOURCE/'raw/respiratory_chartevents.csv.gz'
    for row in rows(path):
        if row['hadm_id'] not in train or row['itemid'] not in ITEMS or row['selection_role']=='state_record':continue
        key=(row['hadm_id'],row['stay_id'],row['itemid']);number=int(row['available_numeric_rows'])
        if key in counts and counts[key]!=number:raise ValueError('inconsistent original candidate count')
        counts[key]=number;selected[key]+=1
    report={}
    for item,name in zip(ITEMS,NAMES):
        keys=[key for key in counts if key[2]==item];number=[counts[k] for k in keys]
        chosen=[selected[k] for k in keys]
        report[name]=dict(train_hospital_icu_item_groups=len(keys),original_candidate_rows=sum(number),exported_anchor_rows=sum(chosen),
            candidate_rows_per_ICU_item_quantiles=np.quantile(number,[0,.1,.5,.9,1]).tolist(),
            exported_anchor_rows_per_ICU_item_quantiles=np.quantile(chosen,[0,.1,.5,.9,1]).tolist())
    result=dict(status='training_anchor_sampling_audited',source_sha256=sha256(path),by_concept=report,
        limits=['candidate counts are SQL finite raw values before clinical cleaning',
            'underlying ICU observations may be dense; exported state anchors are sparse, not a full low-frequency native series',
            'unexported rows are not available to pair from this snapshot; no missing oxygen inferred as room air'])
    write_json(ROOT/'anchor_sampling_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
