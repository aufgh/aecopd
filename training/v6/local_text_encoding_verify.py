"""两个独立GPU进程固定32句重放，所有句向量/索引有限值和范数核验。"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel
from patient_data.local_text_interface import ROOT,MODEL,weighted_vector
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',choices=('a','b'),required=True);a=p.parse_args()
    m=json.loads((ROOT/'manifest.json').read_text());enc=json.loads((ROOT/'encoding.json').read_text())
    if enc['source_manifest_sha256']!=sha256(ROOT/'manifest.json') or enc['output_sha256']!=sha256(ROOT/'local_vectors.float16.npy'):raise ValueError('incomplete or changed vectors')
    for p,h in m['source_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('source changed')
    for name,h in m['encoder_files'].items():
        if sha256(MODEL/name)!=h:raise ValueError('model asset changed')
    with gzip.open(ROOT/'local_tasks.jsonl.gz','rt',encoding='utf-8') as f:tasks=list(map(json.loads,f))
    vectors=np.load(ROOT/'local_vectors.float16.npy',mmap_mode='r');x=np.asarray(vectors,np.float32);norms=np.linalg.norm(x,axis=1)
    if x.shape!=(m['sentences'],768) or not np.isfinite(x).all() or not np.allclose(norms,1,atol=.002):raise ValueError('invalid sentence vectors')
    choices=set(np.linspace(0,m['sentences']-1,31,dtype=int).tolist())
    many={r['sentence_row'] for r in tasks if r['token_start']>0}
    if many:choices.add(min(many))
    else:choices.add(m['sentences']-2)
    selected=[r for r in tasks if r['sentence_row'] in choices];rows={j:[] for j in choices};weights={j:[] for j in choices}
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);model=AutoModel.from_pretrained(MODEL,local_files_only=True).cuda().eval().requires_grad_(False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for offset in range(0,len(selected),32):
        batch=selected[offset:offset+32];encoded={k:v.cuda() for k,v in tokenizer.pad([{'input_ids':r['input_ids']} for r in batch],padding=True,return_tensors='pt').items()}
        with torch.inference_mode():
            hidden=model(**encoded).last_hidden_state.float();mask=encoded['attention_mask'].bool().clone();lengths=mask.sum(1)
            mask[:,0]=False;mask.scatter_(1,(lengths-1)[:,None],False)
            out=torch.nn.functional.normalize((hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None],dim=1).cpu().numpy()
        for r,v in zip(batch,out):rows[r['sentence_row']].append(v);weights[r['sentence_row']].append(r['content_tokens'])
    error=max(float(np.abs(weighted_vector(rows[j],weights[j])-x[j]).max()) for j in choices)
    if error>.002:raise ValueError('GPU source local encoding not reproduced')
    proof=dict(status='all_local_vectors_checked_fixed32_GPU_reencoded',sentences=len(x),sample_sentences=sorted(choices),sample_long_sentence_included=bool(many),
        maximum_vector_error=error,manifest_sha256=sha256(ROOT/'manifest.json'),vector_sha256=enc['output_sha256'],
        code_sha256=sha256(Path(__file__)),model_fitted=False,GPU=torch.cuda.get_device_name(0))
    write_json(ROOT/f'encoding_verification_{a.tag}.json',proof)
    if a.tag=='b' and json.loads((ROOT/'encoding_verification_a.json').read_text())!=proof:raise ValueError('independent GPU encoding differs')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
