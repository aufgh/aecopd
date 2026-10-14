"""r4已审计候选原句→四路由组分块→同一本地冻结ClinicalBERT GPU编码。"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re
import time
import numpy as np
from patient_data.export_mimiciv import sha256

ROUTES = ('acute_context', 'background_support', 'inpatient_findings_course', 'discharge_plan_state')
FUTURE = re.compile(r'\b(?:readmitted|re-admitted|returned\s+(?:back\s+)?to\s+(?:the\s+)?(?:ED|emergency)|after\s+discharge.{0,60}(?:died|expired|readmitted))\b', re.I|re.S)


def route(section):
    if section in ('chief_complaint', 'history_of_present_illness', 'review_of_systems', 'admission_physical_exam'): return 0
    if section in ('past_medical_history', 'social_history', 'admission_medications'): return 1
    if section in ('discharge_condition', 'discharge_physical_exam', 'discharge_medications', 'discharge_instructions',
                   'followup_instructions', 'transitional_issues', 'discharge_disposition', 'discharge_diagnosis'): return 3
    return 2


def selected_groups(directory):
    m = json.loads((directory/'manifest.json').read_text())
    audit = json.loads((directory/'source_audit.json').read_text())
    if (m['version'] != 'clinical-sentence-screen-v4-source-reviewed'
            or audit['status'] != 'all_original_quotes_contexts_and_section_coverage_checked'
            or audit['manifest_sha256'] != sha256(directory/'manifest.json')):
        raise ValueError('only independently source-audited r4 is accepted')
    for name in ('all_units.jsonl.gz', 'selected_sentences.jsonl.gz', 'documents.jsonl'):
        if sha256(directory/name) != m['output_sha256'][name]: raise ValueError('candidate source changed')
    banned = set(); counts = Counter()
    # 显式未来结局或回院叙述涉及的整个章节保守隔离；不编码相邻context。
    with gzip.open(directory/'all_units.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['kind'] == 'discharge' and (r['reason'] == 'observed_future_outcome' or FUTURE.search(r['quote'])):
                banned.add((r['note_id'], r['section_ordinal']))
    groups = defaultdict(list); seen = set()
    with gzip.open(directory/'selected_sentences.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['kind'] != 'discharge': continue  # 此轮只检验小结；影像已有独立对照
            if (r['note_id'], r['section_ordinal']) in banned: counts['quarantined_selected_units'] += 1; continue
            key = (r['note_id'], r['start_char'], r['end_char'])
            if key in seen: raise ValueError('duplicate candidate source unit')
            seen.add(key)
            groups[(r['hadm_id'], r['subject_id'], r['patient_split'], route(r['section']), r['note_id'])].append(r)
    counts['quarantined_sections'] = len(banned)
    counts['selected_units_used'] = len(seen)
    return groups, counts, sha256(directory/'manifest.json')


def build(args):
    import torch
    from transformers import AutoModel, AutoTokenizer
    if not torch.cuda.is_available(): raise RuntimeError('GPU required; no CPU fallback')
    if args.output.exists(): raise FileExistsError(args.output)
    started = time.perf_counter()
    groups, counts, source_hash = selected_groups(args.screen)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError('ClinicalBERT special tokens unavailable')
    tasks = []
    for (h,s,fold,field,note), units in sorted(groups.items()):
        units.sort(key=lambda x: x['start_char'])
        # 每条原句都保留于source_units；块间不重叠，长句按token切开而非静默截断。
        content = '\n'.join(r['quote'] for r in units)
        ids = tokenizer(content, add_special_tokens=False)['input_ids']
        prefix = tokenizer('Clinical topic: '+ROUTES[field]+'\n', add_special_tokens=False)['input_ids']
        budget = 384-len(prefix)-2
        for offset in range(0, len(ids), budget):
            input_ids = [tokenizer.cls_token_id, *prefix, *ids[offset:offset+budget], tokenizer.sep_token_id]
            tasks.append(dict(hadm_id=h, subject_id=s, patient_split=fold, field=field, note_id=note,
                              token_start=offset, token_end=min(offset+budget, len(ids)), input_ids=input_ids,
                              source_units=[(r['start_char'],r['end_char']) for r in units]))
    if not tasks: raise ValueError('no candidate input')
    args.output.mkdir(parents=True)
    with gzip.open(args.output/'chunks.jsonl.gz', 'xt', encoding='utf-8') as f:
        for j,r in enumerate(tasks): f.write(json.dumps(dict(embedding_row=j, **r))+'\n')
    model = AutoModel.from_pretrained(args.model, local_files_only=True).cuda().eval().requires_grad_(False)
    if model.config.hidden_size != 768: raise ValueError('encoder identity/width mismatch')
    vectors = np.lib.format.open_memmap(args.output/'vectors.float16.npy', mode='w+', dtype=np.float16, shape=(len(tasks),768))
    torch.set_num_threads(2)
    for start in range(0, len(tasks), args.batch_size):
        current = tasks[start:start+args.batch_size]
        encoded = tokenizer.pad([{'input_ids':r['input_ids']} for r in current], padding=True, return_tensors='pt')
        encoded = {k:v.cuda() for k,v in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state.float()
            mask = encoded['attention_mask'].bool().clone()
            lengths = mask.sum(1); mask[:,0] = False; mask.scatter_(1, (lengths-1)[:,None], False)
            pooled = (hidden*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
            pooled = torch.nn.functional.normalize(pooled, dim=1)
        array = pooled.cpu().numpy().astype(np.float16)
        if not np.isfinite(array).all(): raise ValueError('nonfinite candidate encoding')
        vectors[start:start+len(current)] = array
        if start % (args.batch_size*50) == 0:
            vectors.flush(); print(json.dumps(dict(completed_rows=start+len(current), rows=len(tasks), seconds=time.perf_counter()-started)), flush=True)
    vectors.flush()
    weights = {p.name:sha256(p) for p in args.model.iterdir() if p.is_file() and p.suffix in ('.json','.txt','.bin','.safetensors')}
    manifest = dict(status='candidate_bert_complete', model='Bio_ClinicalBERT', encoder_frozen=True,
                    source_manifest_sha256=source_hash, routes=ROUTES, rows=len(tasks), width=768,
                    counts=dict(counts), admissions=len(set(r['hadm_id'] for r in tasks)), model_files=weights,
                    output_sha256={n:sha256(args.output/n) for n in ('chunks.jsonl.gz','vectors.float16.npy')},
                    seconds=time.perf_counter()-started, device=torch.cuda.get_device_name(0), max_tokens=384,
                    pooling='content token mean excluding CLS/SEP/PAD; L2; FP32 forward, FP16 storage',
                    limits=['Rule candidates not expert semantic gold; conservative section quarantine not exhaustive future-leak proof',
                            'Retrospective summary policy; only exact selected quotes encoded, adjacent audit context not encoded',
                            'Four routes differ from original B four chapters; no BERT fine-tuning'])
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'], rows=len(tasks), seconds=manifest['seconds'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('/workspace/training/model_cache/Bio_ClinicalBERT'))
    p.add_argument('--batch-size', type=int, default=64)
    a = p.parse_args()
    if a.batch_size < 1: p.error('invalid batch size')
    build(a)
