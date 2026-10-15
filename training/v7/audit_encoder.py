"""离线读取本地 BERT 的实际参数数；不编码患者文本，不微调。"""
import json
from pathlib import Path
import time

from transformers import AutoModel
from patient_data.export_mimiciv import sha256

if __name__=='__main__':
    start=time.perf_counter()
    model_path=Path('training/model_cache/Bio_ClinicalBERT')
    model=AutoModel.from_pretrained(model_path,local_files_only=True)
    total=sum(p.numel() for p in model.parameters())
    pooler=sum(p.numel() for p in model.pooler.parameters())
    if total!=108310272 or total-pooler!=107719680:
        raise ValueError('local BERT architecture differs')
    result=dict(model='Bio_ClinicalBERT',implementation=type(model).__name__,
        total_parameters_including_pooler=total,encoder_parameters_excluding_pooler=total-pooler,
        pooler_parameters=pooler,hidden_size=model.config.hidden_size,layers=model.config.num_hidden_layers,
        backbone_attention_heads=model.config.num_attention_heads,v7_projection_heads=8,
        v7_backbone_optimizer_parameters=0,backbone_training_or_encoding_performed=False,
        source_sha256={str(p):sha256(p) for p in (model_path/'config.json',model_path/'pytorch_model.bin',Path(__file__))},
        inspection_seconds=time.perf_counter()-start)
    Path('training_runs/v7_first_2026-10-15_r2/bert_parameter_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)
