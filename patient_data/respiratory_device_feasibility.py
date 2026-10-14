"""下一数据路线只读可用性：全量保留设备状态vs选中数值锚点；不推断室内空气。"""
from collections import Counter,defaultdict
from datetime import datetime
import json
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.respiratory_transition import ROOT,SOURCE,ITEMS,canonical_resp
from training.v6.train import write_json


def main():
    with np.load(ROOT/'respiratory_transition.npz') as z:training={str(h) for h,s in zip(z['hadm_ids'],z['patient_splits']) if s=='train'}
    device=defaultdict(lambda:defaultdict(set));spo=defaultdict(list);counts=Counter();cross_stay=defaultdict(set)
    path=SOURCE/'raw/respiratory_chartevents.csv.gz'
    for row in rows(path):
        h=row['hadm_id']
        if h not in training:continue
        t=datetime.fromisoformat(row['charttime']);stay=row['stay_id']
        if row['itemid']=='226732':
            name=row['value'].strip();counts[name]+=1;device[h,stay][t].add(name)
        elif row['itemid']=='220277' and canonical_resp(row['itemid'],row['valuenum'],row['valueuom']) is not None:
            spo[h].append((t,stay))
        if row['itemid'] in ITEMS and row['selection_role']!='state_record':cross_stay[h,row['itemid'],t].add(stay)
    coverage=Counter();compressed=[];conflicts=0
    for key,states in device.items():
        ordered=sorted(states.items());changes=0;previous=None
        for _,names in ordered:
            if len(names)>1:conflicts+=1
            state=tuple(sorted(names))
            if state!=previous:changes+=1;previous=state
        compressed.append(changes)
    for h,records in spo.items():
        records=sorted(set(records))
        for name,target in [('first',records[0]),('latest',records[-1])]:
            t,stay=target;candidates=[(x,v) for x,v in device[h,stay].items() if abs((x-t).total_seconds())<=3600]
            if not candidates:coverage[name+':no_device_within1h']+=1;continue
            _,values=min(candidates,key=lambda r:(abs((r[0]-t).total_seconds()),r[0]))
            coverage[name+':paired']+=1
            for v in values:coverage[name+':'+v]+=1
    result=dict(status='read_only_training_device_feasibility_not_new_features',source_sha256=sha256(path),
        training_device_rows=sum(counts.values()),device_values=dict(counts),selected_spo2_pairing=dict(coverage),
        hospital_icu_device_groups=len(compressed),compressed_state_count_quantiles=np.quantile(compressed,[0,.1,.5,.9,1]).tolist(),
        simultaneous_device_conflict_timestamps=conflicts,
        numeric_same_hospital_item_time_multiple_stays=sum(len(v)>1 for v in cross_stay.values()),
        caveats=['None is explicit recorded device label, not proof of room air or FiO2=21%',
            'device transition is not automatically extubation, COPD NIV indication or treatment benefit',
            'conflicting same-time labels must not be arbitrarily sorted as transitions',
            'no new model fitting, labels or confirmation/test inspected'])
    write_json(ROOT/'device_feasibility.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
