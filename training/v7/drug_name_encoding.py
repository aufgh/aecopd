"""离线冻结BERT只编码38个公开规范药名；不读取患者文本。"""
import json
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from patient_data.export_mimiciv import sha256

ROOT=Path('training_runs/v7_sequential_research_2026-10-15/drug_names')
SOURCE=Path('patient_data/processed/v6_audited_coarse_fine_2026-10-12_r2/interface.npz')
MODEL=Path('training/model_cache/Bio_ClinicalBERT')


if __name__=='__main__':
    if ROOT.exists():raise FileExistsError('drug name encoding exists')
    started=time.perf_counter();torch.set_num_threads(4)
    with np.load(SOURCE) as z:names=z['med_names']
    if len(names)!=38:raise ValueError('drug names changed')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModel.from_pretrained(MODEL,local_files_only=True).eval().requires_grad_(False)
    tokens=tokenizer(names.tolist(),return_tensors='pt',padding=True,truncation=True,max_length=64,return_special_tokens_mask=True)
    special=tokens.pop('special_tokens_mask')
    with torch.no_grad():
        out=model(**tokens).last_hidden_state
        mask=tokens['attention_mask'].bool()&~special.bool()
        vectors=(out*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        vectors=torch.nn.functional.normalize(vectors,dim=1).numpy()
    ROOT.mkdir(parents=True);np.savez_compressed(ROOT/'names.npz',names=names,vectors=vectors)
    report=dict(model='Bio_ClinicalBERT',drug_names=38,patient_text_encoded=False,backbone_trained=False,
        source_sha256={str(p):sha256(p) for p in (SOURCE,MODEL/'config.json',MODEL/'pytorch_model.bin',Path(__file__))},
        output_sha256=sha256(ROOT/'names.npz'),seconds=time.perf_counter()-started,
        limits='drug-name lexical embedding, not verified pharmacological manual semantics')
    (ROOT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)
