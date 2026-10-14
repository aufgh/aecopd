"""检查既往住院临床表示的可用性；不用未来标签，不训练或重划队列。"""
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from training.v6.regularization_audit import checked
from training.v6.regularization_diagnostics import history_counts
from training.v6.train import write_json

ROOT=Path('training_runs/v6_longitudinal_feasibility_2026-10-13')


def previous_positions(records, hadms, subjects, splits):
    """只用当前入院前结束的既往合格住院；同时入院/重叠住院不作先验。"""
    if len(set(map(str,hadms)))!=len(hadms):raise ValueError('duplicate modeled admission')
    by_id={str(r['hadm_id']):r for r in records}
    if len(by_id)!=len(records):raise ValueError('duplicate raw admission')
    by_patient=defaultdict(list)
    for i,h in enumerate(hadms):
        r=by_id[str(h)]
        if str(r['subject_id'])!=str(subjects[i]):raise ValueError('patient join mismatch')
        start,end=(datetime.fromisoformat(r[k]) for k in ('admittime','dischtime'))
        if end<=start:raise ValueError('modeled admission has invalid dates')
        by_patient[str(subjects[i])].append((i,start,end))
    neighbors=[];offsets=[0];gaps=[]
    for i,h in enumerate(hadms):
        current=datetime.fromisoformat(by_id[str(h)]['admittime'])
        past=sorted((x for x in by_patient[str(subjects[i])] if x[1]<current and x[2]<=current),key=lambda x:(x[2],x[1],x[0]))
        for p,start,end in past:
            if splits[p]!=splits[i] or p==i:raise ValueError('cross-fold/self history')
            neighbors.append(p);gaps.append((current-end).total_seconds()/86400)
        offsets.append(len(neighbors))
    return np.asarray(neighbors,np.int64),np.asarray(offsets,np.int64),np.asarray(gaps,np.float64)


def run():
    d,_,_,_,_,_,_,_,search,_=checked()
    cfg=json.loads((Path(search['input_config']['reference'])/'config.json').read_text())
    source=Path(cfg['snapshot'])/'cohort/admissions.csv.gz'
    if sha256(source)!=d.provenance['raw_context_hashes']['cohort/admissions']:
        raise ValueError('raw dates source changed')
    indices,offsets,gaps=previous_positions(list(rows(source)),d.arrays['hadm_ids'],d.arrays['subject_ids'],d.arrays['patient_splits'])
    counts=np.diff(offsets);original=history_counts(d.history_raw)
    if (counts>original).any():raise ValueError('past clinical indices exceed original historical counts')
    ROOT.mkdir(exist_ok=True)
    path=ROOT/'past_clinical_indices.npz'
    payload=dict(hadm_ids=d.arrays['hadm_ids'],subject_ids=d.arrays['subject_ids'],patient_splits=d.arrays['patient_splits'],
        previous_positions=indices,offsets=offsets,days_from_previous_discharge_to_current_admission=gaps)
    if path.exists():
        with np.load(path) as z:
            for key,value in payload.items():np.testing.assert_array_equal(z[key],value)
    else:np.savez_compressed(path,**payload)
    folds={}
    for fold in ('train','validation','test'):
        mask=d.arrays['patient_splits']==fold
        if not mask.any():raise ValueError('expected patient fold is absent: '+fold)
        folds[fold]=dict(admissions=int(mask.sum()),with_past_clinical=int((counts[mask]>0).sum()),
            with_two_or_more=int((counts[mask]>=2).sum()),past_count_quantiles=np.quantile(counts[mask],[0,.5,.9,.99,1]).tolist(),
            historical_counts_without_modeled_clinical_state=int((original[mask]-counts[mask]).sum()))
    write_json(ROOT/'feasibility.json',dict(status='date_join_and_patient_split_verified_not_trained',folds=folds,
        input_sha256=sha256(path),source_sha256=sha256(source),strict_labels_sha256=d.provenance['label_sha256'],
        no_labels_in_history_interface=True,clinical_history_tokens_consumed_by_current_V6=False,
        gap_quantiles=np.quantile(gaps,[0,.5,.9,.99,1]).tolist() if len(gaps) else [],goal_achieved=False))
    print(json.dumps(dict(status='history_feasibility_audited',folds=folds)),flush=True)


if __name__=='__main__':run()
