"""固定同一胸片/CT清单，冻结RadBERT与真实RadGraph-XL本地GPU处理。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import time
import numpy as np
from patient_data.export_mimiciv import sha256
from patient_data.clinical_facts.cxr_trajectory import aggregate, SELECTED
from patient_data.clinical_facts.catalog import CT_FIELDS
from patient_data.clinical_facts.features import ct_vector
from patient_data.clinical_facts.guards import CT_CONCEPTS


def records(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, digest in manifest['output_sha256'].items():
        if sha256(root / name) != digest:
            raise ValueError('imaging input hash changed')
    result = []
    for kind in ('cxr', 'chest_ct'):
        with (root / (kind + '_reports.jsonl')).open(encoding='utf-8') as f:
            result.extend(json.loads(line) for line in f)
    with np.load(root / 'members.npz') as f:
        members = {k: f[k] for k in f.files}
    return result, members, manifest


def graph_chunks(text, limit=1400):
    """字符完整覆盖；尽量在句子末分界，绝不截掉长报告的后半部分。"""
    output = []; start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            boundary = max(text.rfind('\n', start + limit // 2, end), text.rfind('. ', start + limit // 2, end))
            if boundary > start:
                end = boundary + 1
        output.append((start, end, text[start:end])); start = end
    return output


def graph_facts(annotation, text):
    """专科实体+关系映射固定临床概念；类别来自模型，非关键词猜肯否。"""
    entities = annotation['entities']; tokens = annotation['text'].split()
    cursor = 0; offsets = []
    for token in tokens:
        pos = text.find(token, cursor)
        if pos < 0:
            raise ValueError('RadGraph token cannot be roundtripped to supplied source')
        offsets.append((pos, pos + len(token))); cursor = pos + len(token)
    polarity = {'Observation::definitely present': 1, 'Observation::definitely absent': 2,
                'Observation::uncertain': 3}
    cxr = {k: [] for k in SELECTED}; ct = []; evidence = []
    patterns = {'Pneumonia': r'pneumonia', 'Consolidation': r'consolidat', 'Edema': r'\bedema\b',
                'Cardiomegaly': r'cardiomegal|enlarged\s+heart', 'Pleural Effusion': r'effusion',
                'Pneumothorax': r'pneumothora', 'Atelectasis': r'atelecta'}
    for key, entity in entities.items():
        if entity['label'] not in polarity:
            continue
        state = polarity[entity['label']]
        related = [v['tokens'] for v in entities.values()
                   if any(rel == 'modify' and target == key for rel, target in v['relations'])]
        # 不能把“increase/acute”等修饰节点沿modify边复制成目标疾病并继承修饰词极性。
        related += [entities[target]['tokens'] for rel, target in entity['relations'] if rel == 'located_at']
        context = ' '.join(related + [entity['tokens']])
        lo, hi = offsets[entity['start_ix']][0], offsets[entity['end_ix']][1]
        quote = text[lo:hi]
        found = []
        for name, pattern in patterns.items():
            if re.search(pattern, context, re.I):
                cxr[name].append(state); found.append(name)
        for field, pattern in CT_CONCEPTS.items():
            if re.search(pattern, context, re.I):
                if field == 'enlarged_pulmonary_artery' and not re.search(r'enlarg|dilat|prominent|caliber|calibre|diameter|measur', context, re.I):
                    continue
                value = {1: 'present', 2: 'absent', 3: 'uncertain'}[state]
                ct.append({'field': field, 'value': value}); found.append(field)
        if found:
            evidence.append({'fields': found, 'state': state, 'quote': quote, 'start': lo, 'end': hi,
                             'entity': entity, 'related_context': context})
    def combine(values):
        if 1 in values and 2 in values: return 3
        return 1 if 1 in values else 3 if 3 in values else 2 if 2 in values else 0
    return {k: combine(v) for k, v in cxr.items()}, ct, evidence


def build(args):
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    if args.output.exists() and not args.resume: raise FileExistsError(args.output)
    if args.resume and (args.kind != 'radgraph' or (args.output / 'manifest.json').exists()):
        raise ValueError('resume only unfinished RadGraph output')
    data, members, manifest = records(args.selected)
    if args.limit: data = data[:args.limit]
    args.output.mkdir(parents=True, exist_ok=args.resume)
    n = len(members['hadm_ids']); index = {str(h): i for i, h in enumerate(members['hadm_ids'])}
    all_present = np.zeros((n, 2), bool)
    vectors = np.zeros((n, 2, 1536), np.float32)
    graphs = np.zeros_like(vectors)
    grouped = defaultdict(list); started = time.perf_counter(); chunks = 0
    completed = 0
    if args.resume:
        origin = json.loads((args.output / 'resume_origin.json').read_text())
        if origin['source_manifest_sha256'] != sha256(args.selected / 'manifest.json'): raise ValueError('resume source changed')
        with (args.output / 'reports.jsonl').open(encoding='utf-8') as f: previous = list(map(json.loads, f))
        for saved, source in zip(previous, data, strict=False):
            if any(saved[k] != source[k] for k in ('hadm_id', 'subject_id', 'note_id', 'charttime', 'original_sha256', 'modality')):
                raise ValueError('resume is not an exact matching report prefix')
            for e in saved['evidence']:
                if source['text'][e['start']:e['end']] != e['quote']: raise ValueError('resume evidence mismatch')
            kind = 0 if saved['modality'] == 'cxr' else 1; pos = index[saved['hadm_id']]
            all_present[pos, kind] = True; grouped[pos, kind].append(saved); chunks += saved['chunks']
        completed = len(previous)
        if completed > len(data): raise ValueError('resume has extra reports')
        print(json.dumps({'resuming_completed_reports': completed}), flush=True)
    if args.kind == 'radbert':
        from transformers import AutoModel, AutoTokenizer
        assets = Path('training/model_cache/RadBERT_2026-10-10')
        asset = json.loads((assets / 'asset_manifest.json').read_text())
        for name, value in asset['files'].items():
            if sha256(assets / name) != value['sha256']: raise ValueError('RadBERT assets changed')
        tokenizer = AutoTokenizer.from_pretrained(str(assets), local_files_only=True)
        # 不使用checkpoint没有训练过的随机pooler；只取冻结token隐藏状态。
        model = AutoModel.from_pretrained(str(assets), local_files_only=True, add_pooling_layer=False).eval().requires_grad_(False).cuda()
        if model.config.hidden_size != 768: raise ValueError('unexpected RadBERT width')
        boundary_ids = tokenizer.encode('', add_special_tokens=True)
        if len(boundary_ids) != 2: raise ValueError('unexpected RoBERTa special token pattern')
    else:
        from radgraph import RadGraph
        assets = Path('training/model_cache/RadGraph_XL_2026-10-10')
        asset = json.loads((assets / 'asset_manifest.json').read_text())
        # 原始配置保存在已校验的公开压缩包中；运行配置仅把骨干路径绑定本地资产。
        import tarfile
        with tarfile.open(assets / 'modern-radgraph-xl.tar.gz') as archive:
            config = json.load(archive.extractfile('config.json'))
        backbone = Path('training/model_cache/ModernBERT_base_2026-10-10')
        backbone_manifest = json.loads((backbone / 'asset_manifest.json').read_text())
        for name, meta in backbone_manifest['files'].items():
            if sha256(backbone / name) != meta['sha256']: raise ValueError('ModernBERT asset changed')
        config['dataset_reader']['token_indexers']['bert']['model_name'] = str(backbone.resolve())
        config['model']['embedder']['token_embedders']['bert']['model_name'] = str(backbone.resolve())
        (assets / 'modern-radgraph-xl/config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        model = RadGraph(model_type='modern-radgraph-xl', cuda=0, model_cache_dir=str(assets),
                         tokenizer_cache_dir='/workspace/training/model_cache/huggingface/hub').eval().requires_grad_(False)
        # 短报告/变长输入不值得为每种形状触发首次compile；权重不变，仍在GPU前向。
        for module in model.modules():
            if hasattr(module, 'config') and hasattr(module.config, 'reference_compile'):
                module.config.reference_compile = False
    with (args.output / 'reports.jsonl').open('a' if args.resume else 'x', encoding='utf-8') as out:
        for ordinal, record in enumerate(data):
            if ordinal < completed: continue
            kind = 0 if record['modality'] == 'cxr' else 1
            pos = index[record['hadm_id']]; all_present[pos, kind] = True
            saved = {k: record[k] for k in ('hadm_id', 'subject_id', 'note_id', 'charttime', 'original_sha256', 'modality')}
            if args.kind == 'radbert':
                ids = tokenizer.encode(record['text'], add_special_tokens=False)
                pieces = [ids[i:i + 510] for i in range(0, len(ids), 510)]
                sums = np.zeros(768, np.float64); count = 0
                for start in range(0, len(pieces), args.batch_size):
                    # transformers5不再提供build_inputs_with_special_tokens；边界由真实tokenizer验收。
                    batch = tokenizer.pad([{'input_ids': [boundary_ids[0]] + p + [boundary_ids[1]]}
                                           for p in pieces[start:start + args.batch_size]], return_tensors='pt')
                    batch = {k: v.cuda() for k, v in batch.items()}
                    with torch.inference_mode():
                        hidden = model(**batch).last_hidden_state
                        # 排除特殊token；最后短块按实际词token数加权，不被等块平均放大。
                        mask = batch['attention_mask'].clone()
                        mask[:, 0] = 0
                        for j, piece in enumerate(pieces[start:start + args.batch_size]): mask[j, len(piece) + 1:] = 0
                        sums += (hidden * mask[:, :, None]).sum((0, 1)).cpu().numpy()
                        count += int(mask.sum())
                if count != len(ids): raise ValueError('RadBERT token coverage differs')
                vector = (sums / max(1, count)).astype(np.float32)
                vector /= max(float(np.linalg.norm(vector)), 1e-12)
                grouped[pos, kind].append((record['charttime'], vector))
                saved.update(input_tokens=len(ids), chunks=len(pieces)); chunks += len(pieces)
            else:
                labels = {k: [] for k in SELECTED}; facts = []; evidence = []
                pieces = graph_chunks(record['text']); chunks += len(pieces)
                for offset, end, piece in pieces:
                    with torch.inference_mode(): annotation = model([piece])['0']
                    cxr, ct, quotes = graph_facts(annotation, piece)
                    for name, value in cxr.items(): labels[name].append(value)
                    facts.extend(dict(f, charttime=record['charttime']) for f in ct)
                    for q in quotes:
                        q.update(start=q['start'] + offset, end=q['end'] + offset)
                        if record['text'][q['start']:q['end']] != q['quote']: raise ValueError('quote mismatch')
                    evidence.extend(quotes)
                def merge(v):
                    return 3 if 1 in v and 2 in v else 1 if 1 in v else 3 if 3 in v else 2 if 2 in v else 0
                saved.update(labels14={k: merge(v) for k, v in labels.items()}, chunk_conflicts={}, ct_facts=facts,
                             evidence=evidence, chunks=len(pieces))
                grouped[pos, kind].append(saved)
            out.write(json.dumps(saved, ensure_ascii=False) + '\n'); out.flush()
            if (ordinal + 1) % 500 == 0 or ordinal == 0:
                print(json.dumps({'kind': args.kind, 'reports': ordinal + 1, 'total': len(data),
                                  'seconds': time.perf_counter() - started}), flush=True)
    for (pos, kind), values in grouped.items():
        if args.kind == 'radbert':
            by_time = defaultdict(list)
            for t, v in values: by_time[t].append(v)
            latest = np.mean(by_time[max(by_time)], axis=0)
            average = np.mean([np.mean(v, axis=0) for v in by_time.values()], axis=0)
            for v in (latest, average): v /= max(float(np.linalg.norm(v)), 1e-12)
            vectors[pos, kind] = np.concatenate([latest, average])
        elif kind == 0:
            graphs[pos, kind, :98] = aggregate(values).reshape(-1)
        else:
            fs = [f for r in values for f in r['ct_facts']]
            graphs[pos, kind, :60] = ct_vector(fs)
    values = vectors if args.kind == 'radbert' else graphs
    np.savez_compressed(args.output / 'imaging.npz', **members, values=values, present=all_present)
    result = {'status': 'full_radiology_processed' if not args.limit else 'pilot_not_full', 'kind': args.kind,
              'reports': len(data), 'chunks': chunks, 'seconds': time.perf_counter() - started,
              'resumed_completed_reports': completed, 'seconds_scope': 'current process only; excludes prior inference and pause' if completed else 'complete current process',
              'resume_origin_sha256': sha256(args.output / 'resume_origin.json') if completed else None,
              'source_manifest_sha256': sha256(args.selected / 'manifest.json'),
              'parent_state_manifest_sha256': manifest['parent_state_manifest_sha256'],
              'asset_manifest_sha256': sha256(assets / 'asset_manifest.json'),
              'code_sha256': sha256(Path(__file__)),
              'radgraph_backbone_manifest_sha256': sha256(backbone / 'asset_manifest.json') if args.kind == 'radgraph' else None,
              'output_sha256': {s: sha256(args.output / s) for s in ('reports.jsonl', 'imaging.npz')},
              'dimensions': [2, 1536], 'encoder_frozen': True,
              'runtime': {'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
                          'device': torch.cuda.get_device_name(0), 'radgraph_reference_compile': False if args.kind == 'radgraph' else None},
              'limits': ['Not expert accuracy; RadGraph concepts mapped from model entities and relations',
                         'Clinical fact padding and dense embedding are different representations',
                         'No report count given as an extra numerical feature']}
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected', type=Path, default=Path('patient_data/processed/v6_imaging_all_2026-10-10_r2'))
    p.add_argument('--kind', choices=('radbert', 'radgraph'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=8); p.add_argument('--limit', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    build(p.parse_args())
