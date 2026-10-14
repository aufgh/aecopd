"""官方CheXbert模型的离线GPU推理桥；保留14项类码，向V6交付7项四状态。"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer
from patient_data.export_mimiciv import sha256
from training.v6.acquire_chexbert import WEIGHT_SHA

CONDITIONS = ('Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Lung Lesion','Edema','Consolidation',
              'Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices','No Finding')
SELECTED = ('Pneumonia','Consolidation','Edema','Cardiomegaly','Pleural Effusion','Pneumothorax','Atelectasis')
STATES = ('Blank','Positive','Negative','Uncertain')


def load_model(assets):
    manifest = json.loads((assets/'asset_manifest.json').read_text())
    for name, info in manifest['sources'].items():
        if sha256(assets/name) != info['sha256']: raise ValueError('CheXbert asset changed: '+name)
    if sha256(assets/'chexbert.pth') != WEIGHT_SHA: raise ValueError('weight identity mismatch')
    if not torch.cuda.is_available(): raise RuntimeError('GPU required, no CPU encoding fallback')
    spec=importlib.util.spec_from_file_location('official_chexbert_labeler', assets/'official/src/models/bert_labeler.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    config=BertConfig.from_json_file(str(assets/'bert-base-uncased/config.json'))
    config._attn_implementation='eager'
    # 完整CheXbert权重包含BERT骨干；初始化架构不另下载普通BERT权重。
    with patch.object(module.BertModel, 'from_pretrained', side_effect=lambda *a,**kw:BertModel(config)):
        model=module.bert_labeler()
    checkpoint=torch.load(assets/'chexbert.pth',map_location='cpu',weights_only=True)
    if 'model_state_dict' not in checkpoint: raise ValueError('unexpected official checkpoint structure')
    state={k.removeprefix('module.'):v for k,v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(state,strict=True)
    del checkpoint, state
    model=model.cuda().eval().requires_grad_(False)
    tokenizer=BertTokenizer.from_pretrained(str(assets/'bert-base-uncased'),local_files_only=True)
    if (tokenizer.vocab_size,tokenizer.cls_token_id,tokenizer.sep_token_id,tokenizer.pad_token_id)!=(30522,101,102,0):
        raise ValueError('tokenizer contract differs')
    return model,tokenizer,manifest


def encode_ids(text, tokenizer):
    normalized=' '.join(text.split())
    tokens=tokenizer.tokenize(normalized)
    ids=[tokenizer.cls_token_id]+tokenizer.convert_tokens_to_ids(tokens)+[tokenizer.sep_token_id]
    length=len(ids)
    if length>512: ids=ids[:511]+[tokenizer.sep_token_id]
    return ids,length


@torch.inference_mode()
def label_texts(texts,model,tokenizer,batch_size=32):
    labels,token_counts=[],[]
    for start in range(0,len(texts),batch_size):
        encoded=[encode_ids(t,tokenizer) for t in texts[start:start+batch_size]]
        ids=[torch.tensor(v[0],dtype=torch.long,device='cuda') for v in encoded]
        padded=torch.nn.utils.rnn.pad_sequence(ids,batch_first=True,padding_value=0)
        masks=(padded!=0).to(torch.float32)
        logits=model(padded,masks)
        labels.append(torch.stack([x.argmax(1) for x in logits],1).cpu().numpy().astype(np.int8))
        token_counts.extend([v[1] for v in encoded])
    return np.concatenate(labels),np.asarray(token_counts)


def one_hot_states(labels,present):
    labels=np.asarray(labels); present=np.asarray(present)
    if labels.ndim!=2 or labels.shape[1]!=7 or present.shape!=(len(labels),) or present.dtype!=bool:
        raise ValueError('invalid CXR shape')
    if not np.isin(labels,(0,1,2,3)).all(): raise ValueError('invalid CheXbert state')
    return (np.eye(4,dtype=np.float32)[labels]*present[:,None,None]).reshape(len(labels),28)


def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    selection=json.loads((args.selected/'manifest.json').read_text())
    if selection['status']!='selected_temporally_checked_not_labeled': raise ValueError('selection unfinished')
    for name,digest in selection['output_sha256'].items():
        if sha256(args.selected/name)!=digest: raise ValueError('selection source changed')
    records=[json.loads(line) for line in (args.selected/'selected_reports.jsonl').read_text(encoding='utf-8').splitlines()]
    if not records or len(records)!=selection['reports_selected']: raise ValueError('selection count invalid')
    with np.load(args.selected/'members.npz',allow_pickle=False) as f: identities={k:f[k] for k in f.files}
    model,tokenizer,assets_manifest=load_model(args.assets)
    # 合成已知真值检查。合成检查不代表真实报告总体准确率。
    synthetic=[]
    for name in SELECTED:
        readable=name.lower()
        synthetic += [{'text':'IMPRESSION: '+readable+' is present.', 'condition':name,'expected':1},
                      {'text':'IMPRESSION: No evidence of '+readable+'.','condition':name,'expected':2},
                      {'text':'IMPRESSION: Possible '+readable+'.','condition':name,'expected':3}]
    syn_labels,_=label_texts([s['text'] for s in synthetic],model,tokenizer,args.batch_size)
    checks=[dict(s,predicted=int(l[CONDITIONS.index(s['condition'])])) for s,l in zip(synthetic,syn_labels,strict=True)]
    # 独立适配器复核官方模型forward，均用同权重和token，不重复调用同一函数冒充校验。
    text=records[0]['text']; ids,_=encode_ids(text,tokenizer)
    encoded=torch.tensor([ids],device='cuda'); mask=(encoded!=0).float()
    with torch.inference_mode():
        reference=model(encoded,mask)
        cls=model.dropout(model.bert(encoded,attention_mask=mask).last_hidden_state[:,0,:])
        adapted=[head(cls) for head in model.linear_heads]
        max_error=max(float((x-y).abs().max()) for x,y in zip(reference,adapted,strict=True))
    if max_error>1e-6: raise ValueError('official forward contract mismatch')
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    all_labels,counts=label_texts([r['text'] for r in records],model,tokenizer,args.batch_size)
    elapsed=time.perf_counter()-started
    n=len(identities['hadm_ids']); position={str(h):i for i,h in enumerate(identities['hadm_ids'])}
    codes=np.zeros((n,7),np.int8); present=np.zeros(n,bool); seen=set()
    for record,label in zip(records,all_labels,strict=True):
        pos=position[record['hadm_id']]
        if pos in seen or str(identities['subject_ids'][pos])!=record['subject_id'] or str(identities['patient_splits'][pos])!=record['patient_split']:
            raise ValueError('CXR identity or split mismatch')
        seen.add(pos); present[pos]=True
        codes[pos]=label[[CONDITIONS.index(s) for s in SELECTED]]
    inputs=one_hot_states(codes,present)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'cxr_context.npz',**identities,cxr_states=codes,cxr_onehot=inputs,cxr_present=present,
                        condition_names=np.array(SELECTED),state_names=np.array(STATES))
    with (args.output/'labeled_reports.jsonl').open('x',encoding='utf-8') as f:
        for i,(r,label,count) in enumerate(zip(records,all_labels,counts,strict=True)):
            f.write(json.dumps(dict(r,labels14={name:int(v) for name,v in zip(CONDITIONS,label,strict=True)},
                                    input_tokens=int(count),truncated=bool(count>512)),ensure_ascii=False)+'\n')
    audit={'synthetic_correct':sum(c['expected']==c['predicted'] for c in checks), 'synthetic_total':len(checks),
           'synthetic_checks':checks,'official_forward_max_error':max_error,'strict_checkpoint_loading':True,
           'note':'synthetic expected labels are not a real-note clinical gold standard'}
    (args.output/'engineering_checks.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    result={'status':'chexbert_labeled_temporally_checked_not_prediction_validated','admissions':n,'reports':len(records),
            'coverage_patients':selection['patients_with_report'],'parent_state_manifest_sha256':selection['parent_state_manifest_sha256'],
            'selection_manifest_sha256':sha256(args.selected/'manifest.json'),'asset_manifest_sha256':sha256(args.assets/'asset_manifest.json'),
            'weight_sha256':WEIGHT_SHA,'condition_names':SELECTED,'state_names':STATES,'batch_size':args.batch_size,
            'device':torch.cuda.get_device_name(0),'precision':'float32','labeling_seconds':elapsed,'reports_per_second':len(records)/elapsed,
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'truncated_reports':int((counts>512).sum()),
            'code_sha256':sha256(Path(__file__)),'output_sha256':{name:sha256(args.output/name) for name in ('cxr_context.npz','labeled_reports.jsonl','engineering_checks.json')},
            'label_distribution':{name:dict(Counter(map(int,codes[present,j]))) for j,name in enumerate(SELECTED)},
            'limits':['No real-report expert gold standard yet','Possible MIMIC-CXR pretrained data overlap; not independent external representation',
                      'No Findings-or-Impression reports conservatively absent; no CT or image pixels',
                      'Last chest report as a whole; four states preserved, missing report masked separately']}
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('output_sha256','label_distribution')}) ,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=Path('training/model_cache/CheXbert_stanford_2026-10-09'))
    p.add_argument('--selected',type=Path,default=Path('patient_data/processed/v6_cxr_reports_validated_2026-10-09'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=32)
    args=p.parse_args()
    if args.batch_size<1: p.error('positive batch size required')
    build(args)
