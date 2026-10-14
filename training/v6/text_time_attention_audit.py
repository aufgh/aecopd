"""仅冻结已训练文本读取器的字段内权重审计；权重不是风险贡献或临床解释。"""
from collections import defaultdict
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from patient_data.export_mimiciv import rows,sha256
from patient_data.text_time_coverage import ROOT,B,STATE,checked_sources
from training.v6.train import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',choices=('a','b'),required=True);args=p.parse_args()
    sources,_=checked_sources();audit=json.loads((ROOT/'audit.json').read_text())
    if sources!=audit['source_sha256']:raise ValueError('source audit differs')
    with np.load(STATE/'clinical_state.npz') as z:train={str(h) for h,f in zip(z['hadm_ids'],z['patient_splits']) if f=='train'}
    with gzip.open(ROOT/'priority_candidate_index.jsonl.gz','rt',encoding='utf-8') as f:
        priority={int(j) for r in map(json.loads,f) for j in r['B_containing_rows']}
    fields={'history_of_present_illness':0,'past_medical_history':1,'hospital_course':2,'discharge_condition':3}
    chunks=[r for r in rows(B/'shared_text.csv.gz') if r['hadm_id'] in train]
    groups=defaultdict(list)
    for i,r in enumerate(chunks):groups[r['hadm_id'],r['field']].append(i)
    vectors=np.load(B/'bert/bioclinicalbert.float16.npy',mmap_mode='r');result={}
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for seed in (42,7):
        path=Path(f'training_runs/v6_module_surgery_2026-10-12_r3/seed{seed}/control/best.pt')
        expected=json.loads(Path('training_runs/v6_warm_residual_2026-10-13/contract.json').read_text())['teacher_checkpoint_sha256'][str(seed)]
        if sha256(path)!=expected:raise ValueError('original teacher changed')
        state=torch.load(path,map_location='cpu',weights_only=True)['state_dict'];prefix='encoder.base.'
        projection=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,128),nn.GELU(),nn.LayerNorm(128)).cuda().eval().requires_grad_(False)
        field=nn.Embedding(4,128).cuda().eval().requires_grad_(False);score=nn.Linear(128,1).cuda().eval().requires_grad_(False)
        for name,module in [('text_projection',projection),('field_ids',field),('text_score',score)]:
            weights={k.removeprefix(prefix+name+'.'):v for k,v in state.items() if k.startswith(prefix+name+'.')}
            module.load_state_dict(weights,strict=True)
        logits=[]
        with torch.inference_mode():
            for start in range(0,len(chunks),1024):
                batch=chunks[start:start+1024]
                x=torch.as_tensor(np.asarray(vectors[[int(r['embedding_row']) for r in batch]],np.float32),device='cuda')
                ids=torch.tensor([fields[r['field']] for r in batch],device='cuda')
                logits.extend(score(projection(x)+field(ids)).squeeze(-1).cpu().tolist())
        shares=[];uniforms=[];multipliers=[];group_counts=defaultdict(int)
        for (_,role),ix in groups.items():
            tagged=np.array([int(chunks[i]['embedding_row']) in priority for i in ix])
            if not tagged.any():continue
            raw=torch.tensor([logits[i] for i in ix],dtype=torch.float32);w=torch.softmax(raw,0).numpy()
            shares.append(float(w[tagged].sum()));uniforms.append(float(tagged.mean()));multipliers.append(shares[-1]/uniforms[-1]);group_counts[role]+=1
        def q(x):return np.quantile(x,[0,.1,.5,.9,1]).tolist()
        result[str(seed)]=dict(teacher_sha256=expected,groups_with_priority_bearing_full_chunk=len(shares),group_counts=dict(group_counts),
            attention_share_quantiles=q(shares),uniform_share_quantiles=q(uniforms),attention_to_uniform_ratio_quantiles=q(multipliers),
            groups_priority_weight_below_uniform=int(sum(a<b for a,b in zip(shares,uniforms))),
            logits_sha256=__import__('hashlib').sha256(np.asarray(logits,np.float32).tobytes()).hexdigest())
    proof=dict(status='frozen_original_text_reader_training_only_weight_audit',teachers=result,audit_sha256=sha256(ROOT/'audit.json'),
        code_sha256=sha256(Path(__file__)),training_chunks=len(chunks),GPU=torch.cuda.get_device_name(0),new_fit=False,
        caveats=['Subset extracted from actual verified teacher weights; no new optimization or risk metrics',
            'Only full-contained lexical priority candidate chunks tagged;54 cross-boundary units not tagged by this rule',
            'Attention weights are not causal attribution or proof of ignored clinical information',
            'FP32 score audit; does not claim entire original model risk replay',
            'Within-field learned attention; final available-field averaging is a separate operation'])
    write_json(ROOT/f'attention_audit_{args.tag}.json',proof)
    if args.tag=='b' and json.loads((ROOT/'attention_audit_a.json').read_text())!=proof:raise ValueError('independent attention audit differs')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
