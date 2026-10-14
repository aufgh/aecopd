"""全部合格胸片的七项观测轨迹；不将未提及当作阴性或恢复。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import sha256

SELECTED=('Pneumonia','Consolidation','Edema','Cardiomegaly','Pleural Effusion','Pneumothorax','Atelectasis')
FEATURES=('first_blank','first_positive','first_negative','first_uncertain',
          'latest_blank','latest_positive','latest_negative','latest_uncertain',
          'ever_positive','ever_uncertain','later_explicit_negative','new_positive',
          'first_conflict','latest_conflict')

def combine(states):
    states=set(map(int,states))-{0}
    conflict=1 in states and 2 in states
    return (3 if conflict else 1 if 1 in states else 3 if 3 in states else 2 if 2 in states else 0), conflict

def aggregate(records):
    """同charttime是集合而非序列；storetime不制造病理先后。"""
    if not records: return np.zeros((7,14),np.float32)
    by_time=defaultdict(list)
    for r in records: by_time[r['charttime']].append(r)
    buckets=[]
    for timestamp in sorted(by_time):
        rows=by_time[timestamp]; codes=[]; conflicts=[]
        for field in SELECTED:
            code,conflict=combine(r['labels14'][field] for r in rows)
            codes.append(code); conflicts.append(conflict or any(r['chunk_conflicts'].get(field,False) for r in rows))
        buckets.append((codes,conflicts))
    result=np.zeros((7,14),np.float32)
    for j in range(7):
        sequence=[b[0][j] for b in buckets]; first,last=sequence[0],sequence[-1]
        result[j,first]=1; result[j,4+last]=1
        result[j,8]=1 in sequence; result[j,9]=3 in sequence
        # 必须存在较晚的明确阴性；同一时刻、空白或疑似不作为改善。
        result[j,10]=any(v==2 and 1 in sequence[:i] for i,v in enumerate(sequence))
        result[j,11]=first==2 and 1 in sequence[1:]
        result[j,12]=buckets[0][1][j]; result[j,13]=buckets[-1][1][j]
    return result

def chunks(text,tokenizer):
    """全字符覆盖、保留首尾；超长段逐字缩短至官方512预算。"""
    from training.v6.chexbert_inference import encode_ids
    start=0; result=[]
    while start<len(text):
        end=min(len(text),start+1800)
        while encode_ids(text[start:end],tokenizer)[1]>500:
            end=start+(end-start)//2
            if end<=start: raise ValueError('single character exceeds token budget')
        # 在预算内的自然句边界结束；无边界则保留硬边界的审计偏移。
        if end<len(text):
            boundary=max(text.rfind('\n',start+max(1,(end-start)//2),end),text.rfind('. ',start+max(1,(end-start)//2),end))
            if boundary>start: end=boundary+1
        result.append((start,end,text[start:end])); start=end
    if ''.join(c[2] for c in result)!=text: raise ValueError('lost report text')
    return result

def build(args):
    from training.v6.chexbert_inference import load_model,label_texts,CONDITIONS
    import torch
    if args.output.exists(): raise FileExistsError(args.output)
    selection=json.loads((args.selected/'manifest.json').read_text())
    for name,value in selection['output_sha256'].items():
        if sha256(args.selected/name)!=value: raise ValueError('selection hash changed')
    records=[json.loads(s) for s in (args.selected/'cxr_reports.jsonl').read_text(encoding='utf-8').splitlines()]
    model,tokenizer,asset=load_model(args.assets)
    spans=[chunks(r['text'],tokenizer) for r in records]; texts=[c[2] for group in spans for c in group]
    started=time.perf_counter(); labels,counts=label_texts(texts,model,tokenizer,args.batch_size)
    if (counts>512).any(): raise ValueError('truncated chunk')
    elapsed=time.perf_counter()-started; pointer=0
    for record,group in zip(records,spans,strict=True):
        subset=labels[pointer:pointer+len(group)]; pointer+=len(group)
        combined=[combine(subset[:,j]) for j in range(14)]
        record.update(labels14={name:v[0] for name,v in zip(CONDITIONS,combined,strict=True)},
                      chunk_conflicts={name:bool(v[1]) for name,v in zip(CONDITIONS,combined,strict=True)},
                      chunks=[{'start_char':a,'end_char':b,'tokens':int(counts[pointer-len(group)+i]),
                               'labels14':list(map(int,subset[i]))} for i,(a,b,_) in enumerate(group)])
    with np.load(args.selected/'members.npz') as f: members={k:f[k] for k in f.files}
    grouped=defaultdict(list)
    for record in records: grouped[record['hadm_id']].append(record)
    values=np.stack([aggregate(grouped.get(str(h),[])) for h in members['hadm_ids']])
    present=np.asarray([str(h) in grouped for h in members['hadm_ids']],bool)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'cxr_trajectory.npz',**members,cxr_values=values.reshape(-1,98),
                        cxr_present=present,condition_names=np.array(SELECTED),feature_names=np.array(FEATURES))
    with (args.output/'labeled_reports.jsonl').open('x',encoding='utf-8') as f:
        for record in records: f.write(json.dumps(record,ensure_ascii=False)+'\n')
    result={'status':'all_cxr_labeled_and_aggregated','reports':len(records),'admissions_with_cxr':int(present.sum()),
            'admissions_with_repeat_timestamps':sum(len({r['charttime'] for r in group})>1 for group in grouped.values()),
            'chunks':len(texts),'seconds':elapsed,'reports_per_second':len(records)/elapsed,'truncated_chunks':0,
            'parent_state_manifest_sha256':selection['parent_state_manifest_sha256'],
            'selection_manifest_sha256':sha256(args.selected/'manifest.json'),'asset_manifest_sha256':sha256(args.assets/'asset_manifest.json'),
            'feature_names':FEATURES,'condition_names':SELECTED,
            'trajectory_counts':{field:{key:int(values[:,j,k].sum()) for k,key in enumerate(FEATURES[8:],8)} for j,field in enumerate(SELECTED)},
            'code_sha256':sha256(Path(__file__)),'output_sha256':{n:sha256(args.output/n) for n in ('cxr_trajectory.npz','labeled_reports.jsonl')},
            'limits':['Report labels not image pixels or expert gold standard','Later explicit negative is an observation, not proof of treatment effect',
                      'Unknown modality excluded; no assumption that all thoracic findings are CXR','Possible pretrained MIMIC-CXR overlap']}
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected',type=Path,default=Path('patient_data/processed/v6_imaging_all_2026-10-10_r2'))
    p.add_argument('--assets',type=Path,default=Path('training/model_cache/CheXbert_stanford_2026-10-09'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--batch-size',type=int,default=32)
    build(p.parse_args())
