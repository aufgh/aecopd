"""原B训练折token完整性与固定32行GPU重编码；不是新向量或新训练。"""
from collections import Counter
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel
from patient_data.export_mimiciv import rows,sha256
from patient_data.text_time_coverage import ROOT,B,STATE,ENCODED,checked_sources
from training.v6.train import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',choices=('a','b'),required=True);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('actual GPU required')
    report=json.loads((ROOT/'audit.json').read_text());sources,_=checked_sources()
    if sources!=report['source_sha256']:raise ValueError('audit source changed')
    for tag in ('a','b'):
        v=json.loads((ROOT/f'verification_{tag}.json').read_text())
        if v['report_sha256']!=sha256(ROOT/'audit.json') or v['maximum_count_or_index_error']!=0:raise ValueError('source audit not independently replayed')
    model_dir=Path('training/model_cache/Bio_ClinicalBERT');m=json.loads((ENCODED/'manifest.json').read_text())
    for name,h in m['model_files'].items():
        if sha256(model_dir/name)!=h:raise ValueError('local encoder asset differs')
    with np.load(STATE/'clinical_state.npz') as z:train={str(h) for h,f in zip(z['hadm_ids'],z['patient_splits']) if f=='train'}
    current=[r for r in rows(B/'shared_text.csv.gz') if r['hadm_id'] in train]
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True);maximum=0;hist=Counter()
    for start in range(0,len(current),256):
        batch=current[start:start+256];ids=tokenizer([r['chunk_text'] for r in batch],add_special_tokens=True,truncation=False)['input_ids']
        for r,tokens in zip(batch,ids):
            if len(tokens)!=int(r['bert_token_count']) or len(tokens)>512:raise ValueError('actual B token length mismatch/truncation')
            maximum=max(maximum,len(tokens));hist[r['field']]+=1
    # 每字段等间距8行，固定非结局选择；不同tag独立进程同输入。
    chosen=[]
    for field in sorted(hist):
        eligible=[i for i,r in enumerate(current) if r['field']==field]
        chosen.extend(eligible[i] for i in np.linspace(0,len(eligible)-1,8,dtype=int))
    texts=[current[i]['chunk_text'] for i in chosen];indices=[int(current[i]['embedding_row']) for i in chosen]
    original=np.load(B/'bert/bioclinicalbert.float16.npy',mmap_mode='r')
    if original.shape!=(76974,768):raise ValueError('original B vector shape changed')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModel.from_pretrained(model_dir,local_files_only=True).cuda().eval().requires_grad_(False)
    encoded={k:v.cuda() for k,v in tokenizer(texts,padding=True,truncation=False,return_tensors='pt').items()}
    with torch.inference_mode():
        hidden=model(**encoded).last_hidden_state.float();mask=encoded['attention_mask'].bool().clone()
        lengths=mask.sum(1);mask[:,0]=False;mask.scatter_(1,(lengths-1)[:,None],False)
        pooled=torch.nn.functional.normalize((hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None],dim=1).cpu().numpy()
    expected=original[indices].astype(np.float32);error=float(np.abs(pooled-expected).max())
    if error>.002:raise ValueError('fixed B GPU encoding differs beyond FP16 budget')
    proof=dict(status='all_training_B_token_lengths_checked_and_fixed32_GPU_reencoded',training_rows=len(current),actual_max_B_tokens=maximum,
        field_counts=dict(hist),sample_rows=indices,maximum_vector_error=error,FP16_tolerance=.002,
        tokenization_truncation=False,source_audit_sha256=sha256(ROOT/'audit.json'),original_vectors_sha256=sha256(B/'bert/bioclinicalbert.float16.npy'),
        encoder_files=m['model_files'],code_sha256=sha256(Path(__file__)),device=torch.cuda.get_device_name(0),
        limits=['Only32 vector rows reencoded; not all76974 GPU rerun','No semantic accuracy or pooling utility proof','No new fitting or vector overwrite'])
    write_json(ROOT/f'encoding_check_{args.tag}.json',proof)
    if args.tag=='b' and json.loads((ROOT/'encoding_check_a.json').read_text())!=proof:raise ValueError('independent encoding proofs differ')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
