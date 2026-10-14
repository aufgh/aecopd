"""同安全原句的已有包含块/局部冻结编码接口，不使用高频轨迹。"""
from collections import Counter,defaultdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.text_time_coverage import SCREEN,ENCODED,STATE,time_flags,checked_sources
from training.v6.encode_screen_candidates import selected_groups,ROUTES
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_local_text_interface_2026-10-13_r2')
PROTOCOL=Path('docs/V6_LOCAL_TEXT_INTERFACE_PROTOCOL_2026-10-13.md')
MODEL=Path('training/model_cache/Bio_ClinicalBERT')
SECTIONS=('hospital_course','discharge_condition','discharge_physical_exam','discharge_instructions','transitional_issues','followup_instructions')
FLAG_KEYS=('explicit_discharge','change_or_residual','plan','historical')


def overlaps(a,b,chunks):
    if not 0<=a<b:raise ValueError('invalid sentence token span')
    result=[]
    for row,start,end in chunks:
        n=max(0,min(b,end)-max(a,start))
        if n:result.append((row,n))
    if sum(n for _,n in result)!=b-a:raise ValueError('sentence tokens missing or multiply covered')
    return result


def index_mapping(mapped):
    """落盘索引与源重建均使用JSON原生list，防止tuple/list假差异。"""
    return [[int(row),int(tokens)] for row,tokens in mapped]


def token_span(offsets,start,end):
    if not 0<=start<end:raise ValueError('invalid joined sentence character span')
    use=[i for i,(a,b) in enumerate(offsets) if a<end and b>start]
    if not use or use!=list(range(use[0],use[-1]+1)):raise ValueError('sentence token positions not contiguous')
    return use[0],use[-1]+1


def weighted_vector(vectors,weights):
    if len(vectors)!=len(weights) or not len(weights) or np.any(np.asarray(weights)<=0):raise ValueError('invalid weights')
    result=np.average(np.asarray(vectors,np.float32),axis=0,weights=np.asarray(weights,np.float32)).astype(np.float32)
    norm=np.linalg.norm(result)
    if not np.isfinite(result).all() or norm<=1e-12:raise ValueError('invalid vector')
    return result/norm


def derive():
    from transformers import AutoTokenizer
    source,_=checked_sources();groups,counts,source_hash=selected_groups(SCREEN)
    m=json.loads((ENCODED/'manifest.json').read_text());proof=json.loads((ENCODED/'independent_encoding_verification.json').read_text())
    if source_hash!=m['source_manifest_sha256'] or proof['manifest_sha256']!=sha256(ENCODED/'manifest.json'):raise ValueError('existing encoding proof differs')
    for name,h in m['model_files'].items():
        if sha256(MODEL/name)!=h:raise ValueError('model asset changed')
    with np.load(STATE/'clinical_state.npz') as z:members={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits')}
    positions={str(h):i for i,h in enumerate(members['hadm_ids'])}
    chunks=defaultdict(list)
    with gzip.open(ENCODED/'chunks.jsonl.gz','rt',encoding='utf-8') as f:
        for j,r in enumerate(map(json.loads,f)):
            if r['embedding_row']!=j:raise ValueError('existing rows reordered')
            chunks[(r['hadm_id'],r['subject_id'],r['patient_split'],r['field'],r['note_id'])].append(r)
    if set(chunks)!=set(groups):raise ValueError('source group coverage differs')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,use_fast=True)
    if not tokenizer.is_fast:raise ValueError('fast tokenizer original offsets required')
    existing=np.load(ENCODED/'vectors.float16.npy',mmap_mode='r');index=[];tasks=[];contained=[];metadata=[];sentence_positions=[];stats=Counter()
    for key,units in sorted(groups.items()):
        units=sorted(units,key=lambda r:r['start_char']);wanted=[r for r in units if time_flags(r['quote'],r['section'])['priority_candidate']]
        if not wanted:continue
        h,s,fold,route,note=key;pos=positions[h]
        if str(members['subject_ids'][pos])!=s or members['patient_splits'][pos]!=fold:raise ValueError('subject/fold differs')
        content='\n'.join(r['quote'] for r in units)
        tokenized=tokenizer(content,add_special_tokens=False,return_offsets_mapping=True,truncation=False)
        ids=tokenized['input_ids'];offsets=tokenized['offset_mapping'];prefix=tokenizer('Clinical topic: '+ROUTES[route]+'\n',add_special_tokens=False)['input_ids']
        budget=384-len(prefix)-2;expected_units=[[r['start_char'],r['end_char']] for r in units];right=0;spans=[]
        for r in sorted(chunks[key],key=lambda x:x['token_start']):
            a,b=r['token_start'],r['token_end']
            if a!=right or not a<b<=len(ids) or r['source_units']!=expected_units or r['input_ids']!=[tokenizer.cls_token_id,*prefix,*ids[a:b],tokenizer.sep_token_id]:
                raise ValueError('actual saved block tokens/source differ')
            spans.append((r['embedding_row'],a,b));right=b;stats['source_blocks_token_verified']+=1
        if right!=len(ids):raise ValueError('missing tail tokens')
        cursor=0;wanted_keys={r['unit_key'] for r in wanted}
        for unit in units:
            start,end=cursor,cursor+len(unit['quote']);cursor=end+1
            if unit['unit_key'] not in wanted_keys:continue
            a,b=token_span(offsets,start,end);mapped=overlaps(a,b,spans)
            # 单句再tokenize是局部上下文，不偷用整组hidden；原句内容不改写。
            local_ids=tokenizer(unit['quote'],add_special_tokens=False,truncation=False)['input_ids']
            if not local_ids:raise ValueError('empty local sentence')
            row=len(index);first_task=len(tasks)
            for offset in range(0,len(local_ids),budget):
                end_token=min(offset+budget,len(local_ids))
                tasks.append(dict(sentence_row=row,token_start=offset,token_end=end_token,content_tokens=end_token-offset,
                    input_ids=[tokenizer.cls_token_id,*prefix,*local_ids[offset:end_token],tokenizer.sep_token_id]))
            flags=time_flags(unit['quote'],unit['section']);meta=np.zeros(10,np.float32);meta[SECTIONS.index(unit['section'])]=1
            meta[6:]=[flags[k] for k in FLAG_KEYS];metadata.append(meta);sentence_positions.append(pos)
            record={k:unit[k] for k in ('hadm_id','subject_id','patient_split','note_id','original_sha256','section','start_char','end_char','unit_key')}
            record.update(sentence_row=row,route=route,joined_token_start=a,joined_token_end=b,containing_rows_and_tokens=index_mapping(mapped),
                local_token_count=len(local_ids),task_start=first_task,task_end=len(tasks),quote_sha256=hashlib.sha256(unit['quote'].encode()).hexdigest(),metadata=meta.tolist())
            index.append(record);contained.append(weighted_vector(existing[[j for j,_ in mapped]],[n for _,n in mapped]));stats['sentences:'+fold]+=1
            if fold=='train':stats['training_local_tokens']+=len(local_ids);stats['training_cross_block_sentences']+=int(len(mapped)>1)
    if stats['sentences:train']!=6613:raise ValueError('locked training candidate membership differs')
    arr=dict(**members,sentence_positions=np.asarray(sentence_positions,np.int64),metadata=np.asarray(metadata,np.float32),
        containing_vectors=np.asarray(contained,np.float16))
    source.update({str(p):sha256(p) for p in (Path(__file__),Path('training/v6/encode_screen_candidates.py'),Path('patient_data/text_time_coverage.py'),
        Path('training/tests/test_v6_local_text_interface.py'),PROTOCOL,Path('docs/V6_DATA_SCOPE_DECISION_2026-10-13.md'))})
    report=dict(source_sha256=source,existing_candidate_encoding_manifest_sha256=sha256(ENCODED/'manifest.json'),existing_source_counts=dict(counts),
        encoder_files=m['model_files'],statistics=dict(stats),sentences=len(index),local_tasks=len(tasks),sections=list(SECTIONS),flag_keys=list(FLAG_KEYS),
        training_unique_admissions=len({r['hadm_id'] for r in index if r['patient_split']=='train'}),
        limits=['No high-frequency trajectories used','Same exact safe sentence membership in both views','No outcome used for selection',
            'Containing vector retains other block content, not exact local sentence representation','Lexical time metadata not clinical assertion',
            'Retrospective FINAL policy, not prospective availability'])
    return arr,index,tasks,report


def build():
    if ROOT.exists():raise FileExistsError('new interface directory exists; inspect')
    arr,index,tasks,report=derive();ROOT.mkdir(parents=True)
    np.savez_compressed(ROOT/'interface.npz',**arr)
    for name,values in [('index.jsonl.gz',index),('local_tasks.jsonl.gz',tasks)]:
        with gzip.open(ROOT/name,'xt',encoding='utf-8') as f:
            for r in values:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report.update(status='source_token_interface_built_local_encoding_pending',output_sha256={n:sha256(ROOT/n) for n in ('interface.npz','index.jsonl.gz','local_tasks.jsonl.gz')})
    write_json(ROOT/'manifest.json',report);write_json(ROOT/'state.json',dict(status='local_encoding_pending',new_model_fits=0))
    print(json.dumps({k:report[k] for k in ('status','statistics','sentences','local_tasks','training_unique_admissions')}),flush=True)


def verify():
    arr,index,tasks,report=derive();m=json.loads((ROOT/'manifest.json').read_text())
    for k,v in report.items():
        if m[k]!=v:raise ValueError('source interface metadata differs: '+k)
    for name,h in m['output_sha256'].items():
        if sha256(ROOT/name)!=h:raise ValueError('interface output changed')
    with np.load(ROOT/'interface.npz') as z:
        for k,v in arr.items():np.testing.assert_array_equal(z[k],v)
    for name,values in [('index.jsonl.gz',index),('local_tasks.jsonl.gz',tasks)]:
        with gzip.open(ROOT/name,'rt',encoding='utf-8') as f:
            if list(map(json.loads,f))!=values:raise ValueError('source token index differs')
    proof=dict(status='independent_source_token_interface_verified',source_and_arrays_max_error=0,manifest_sha256=sha256(ROOT/'manifest.json'),
        sentences=len(index),tasks=len(tasks),no_model_fits=True)
    write_json(ROOT/'source_verification.json',proof);print(json.dumps(proof),flush=True)


def encode():
    import torch
    from transformers import AutoModel,AutoTokenizer
    if not torch.cuda.is_available():raise RuntimeError('actual GPU required')
    if (ROOT/'local_vectors.float16.npy').exists():raise FileExistsError('local vectors exist; inspect before any recovery')
    m=json.loads((ROOT/'manifest.json').read_text());proof=json.loads((ROOT/'source_verification.json').read_text())
    if proof['manifest_sha256']!=sha256(ROOT/'manifest.json'):raise ValueError('source token verification stale')
    for p,h in m['source_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('protected interface source changed')
    for name,h in m['encoder_files'].items():
        if sha256(MODEL/name)!=h:raise ValueError('encoder asset changed')
    with gzip.open(ROOT/'local_tasks.jsonl.gz','rt',encoding='utf-8') as f:tasks=list(map(json.loads,f))
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);model=AutoModel.from_pretrained(MODEL,local_files_only=True).cuda().eval().requires_grad_(False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sums=np.zeros((m['sentences'],768),np.float32);weight=np.zeros(m['sentences'],np.float32);start=time.perf_counter()
    for offset in range(0,len(tasks),64):
        current=tasks[offset:offset+64];encoded={k:v.cuda() for k,v in tokenizer.pad([{'input_ids':r['input_ids']} for r in current],padding=True,return_tensors='pt').items()}
        with torch.inference_mode():
            hidden=model(**encoded).last_hidden_state.float();mask=encoded['attention_mask'].bool().clone();lengths=mask.sum(1)
            mask[:,0]=False;mask.scatter_(1,(lengths-1)[:,None],False)
            vectors=torch.nn.functional.normalize((hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None],dim=1).cpu().numpy()
        for r,v in zip(current,vectors):sums[r['sentence_row']]+=v*r['content_tokens'];weight[r['sentence_row']]+=r['content_tokens']
        if offset%1024==0:write_json(ROOT/'encoding_progress.json',dict(completed_tasks=offset+len(current),tasks=len(tasks),seconds=time.perf_counter()-start))
    if not np.all(weight>0):raise ValueError('sentence encoding lost')
    sums/=weight[:,None];sums/=np.linalg.norm(sums,axis=1)[:,None]
    if not np.isfinite(sums).all():raise ValueError('nonfinite local vectors')
    np.save(ROOT/'local_vectors.float16.npy',sums.astype(np.float16),allow_pickle=False)
    write_json(ROOT/'encoding.json',dict(status='local_frozen_vectors_complete',rows=m['sentences'],width=768,tasks=len(tasks),
        seconds=time.perf_counter()-start,source_manifest_sha256=sha256(ROOT/'manifest.json'),output_sha256=sha256(ROOT/'local_vectors.float16.npy'),
        GPU=torch.cuda.get_device_name(0),code_sha256=sha256(Path(__file__)),encoder_files=m['encoder_files'],batch_size=64,model_fitted=False,
        pooling='same Clinical topic prefix; content mean minus CLS/SEP/PAD; per-task L2; content-token weighted sentence aggregate then L2'))
    write_json(ROOT/'state.json',dict(status='local_encoding_complete_pending_GPU_replay',new_model_fits=0))
    print(json.dumps(dict(status='local_vectors_complete',rows=m['sentences'],seconds=time.perf_counter()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify','encode'),required=True);a=p.parse_args()
    try:globals()[a.mode]()
    except Exception as e:
        Path('work/v6_local_text_interface_2026-10-13').mkdir(parents=True,exist_ok=True)
        write_json(Path('work/v6_local_text_interface_2026-10-13')/f'failure_{a.mode}.json',dict(error=repr(e),new_model_fits=0))
        raise
