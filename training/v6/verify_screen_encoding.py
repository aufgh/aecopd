"""新进程检查候选索引/原句跨度/分块覆盖，并GPU重算固定32行向量。"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from patient_data.export_mimiciv import sha256
from training.v6.encode_screen_candidates import ROUTES, selected_groups


def verify(screen, directory, model_dir):
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    m=json.loads((directory/'manifest.json').read_text())
    if m['status']!='candidate_bert_complete': raise ValueError('unfinished vectors')
    groups,counts,digest=selected_groups(screen)
    if digest!=m['source_manifest_sha256'] or dict(counts)!=m['counts']: raise ValueError('candidate selection differs')
    for name,value in m['output_sha256'].items():
        if sha256(directory/name)!=value: raise ValueError('encoded output changed')
    for name,value in m['model_files'].items():
        if sha256(model_dir/name)!=value: raise ValueError('encoder asset changed')
    chunks=[]; per_group=defaultdict(list)
    with gzip.open(directory/'chunks.jsonl.gz','rt',encoding='utf-8') as f:
        for j,line in enumerate(f):
            r=json.loads(line)
            if r['embedding_row']!=j or not 0<len(r['input_ids'])<=384 or r['token_start']<0 or r['token_end']<=r['token_start']:
                raise ValueError('invalid chunk indices')
            key=(r['hadm_id'],r['subject_id'],r['patient_split'],r['field'],r['note_id'])
            if key not in groups: raise ValueError('group not found')
            expected=sorted(groups[key],key=lambda x:x['start_char'])
            if r['source_units']!=[[u['start_char'],u['end_char']] for u in expected]: raise ValueError('source unit positions differ')
            per_group[key].append(j); chunks.append(r)
    if len(chunks)!=m['rows'] or set(per_group)!=set(groups): raise ValueError('source group coverage mismatch')
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    # 全部组重新tokenize，核对每段token与原句、前缀、special tokens完全一致。
    for key,indices in per_group.items():
        text='\n'.join(r['quote'] for r in sorted(groups[key],key=lambda x:x['start_char']))
        ids=tokenizer(text,add_special_tokens=False)['input_ids']
        prefix=tokenizer('Clinical topic: '+ROUTES[key[3]]+'\n',add_special_tokens=False)['input_ids']
        start=0
        for index in indices:
            r=chunks[index]
            if r['token_start']!=start or r['input_ids']!=[tokenizer.cls_token_id,*prefix,*ids[start:r['token_end']],tokenizer.sep_token_id]:
                raise ValueError('original token content/order differs')
            start=r['token_end']
        if start!=len(ids): raise ValueError('truncated or missing tail tokens')
    vectors=np.load(directory/'vectors.float16.npy',mmap_mode='r')
    norms=np.linalg.norm(vectors.astype(np.float32),axis=1)
    if not np.isfinite(vectors).all() or not np.allclose(norms,1,atol=.002): raise ValueError('vector finite/norm differs')
    chosen=np.linspace(0,len(chunks)-1,32,dtype=int)
    model=AutoModel.from_pretrained(model_dir,local_files_only=True).cuda().eval().requires_grad_(False)
    encoded=tokenizer.pad([{'input_ids':chunks[i]['input_ids']} for i in chosen],padding=True,return_tensors='pt')
    encoded={k:v.cuda() for k,v in encoded.items()}
    with torch.inference_mode():
        hidden=model(**encoded).last_hidden_state.float(); mask=encoded['attention_mask'].bool().clone()
        lengths=mask.sum(1); mask[:,0]=False; mask.scatter_(1,(lengths-1)[:,None],False)
        pooled=(hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
        pooled=torch.nn.functional.normalize(pooled,dim=1).cpu().numpy()
    error=float(np.abs(vectors[chosen].astype(np.float32)-pooled).max())
    if error>.002: raise ValueError('GPU frozen encoding reproduction failed')
    report=dict(status='all_chunks_source_tokens_verified_and_32_gpu_reencoded',chunks=len(chunks),groups=len(groups),
                manifest_sha256=sha256(directory/'manifest.json'),max_vector_error=error,
                limits=['Not expert semantic accuracy; conservative future-section screening not exhaustive leakage guarantee'])
    (directory/'independent_encoding_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('/workspace/training/model_cache/Bio_ClinicalBERT'))
    a=p.parse_args();verify(a.screen,a.directory,a.model)
