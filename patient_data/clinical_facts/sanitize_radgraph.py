"""剔除跨自然章节的实体后重聚合；保留完整公开模型原输出，不重跑GPU。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from patient_data.clinical_facts.radiology_compare import records
from patient_data.clinical_facts.cxr_trajectory import SELECTED, aggregate
from patient_data.clinical_facts.features import ct_vector
from patient_data.clinical_facts.catalog import CT_FIELDS


def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    source, members, selected = records(args.selected); by_id = {r['note_id']: r for r in source}
    manifest = json.loads((args.source / 'manifest.json').read_text())
    if manifest['status'] != 'full_radiology_processed' or manifest['kind'] != 'radgraph': raise ValueError('not full RadGraph')
    for name, checksum in manifest['output_sha256'].items():
        if sha256(args.source / name) != checksum: raise ValueError('input changed')
    with (args.source / 'reports.jsonl').open(encoding='utf-8') as f: reports = list(map(json.loads, f))
    if len(reports) != len(source) or {r['note_id'] for r in reports} != by_id.keys(): raise ValueError('reports differ')
    dropped = []; groups = defaultdict(list)
    for r in reports:
        maps = []; cursor = 0
        for s in by_id[r['note_id']]['sections']:
            maps.append((cursor, cursor + len(s['text']))); cursor += len(s['text']) + 1
        valid = []
        for e in r['evidence']:
            if by_id[r['note_id']]['text'][e['start']:e['end']] != e['quote']: raise ValueError('input evidence differs')
            if any(a <= e['start'] < e['end'] <= b for a, b in maps): valid.append(e)
            else: dropped.append(dict(e, note_id=r['note_id'], reason='entity_spans_distinct_natural_sections'))
        # 必须在剔除无效实体后重建标签，而非仅改引用使旧错误标签仍然进入模型。
        labels = {k: [] for k in SELECTED}; facts = []
        for e in valid:
            for field in e['fields']:
                if field in SELECTED: labels[field].append(e['state'])
                elif field in CT_FIELDS: facts.append(dict(field=field, value={1: 'present', 2: 'absent', 3: 'uncertain'}[e['state']], charttime=r['charttime']))
        def merged(values):
            return 3 if 1 in values and 2 in values else 1 if 1 in values else 3 if 3 in values else 2 if 2 in values else 0
        r.update(evidence=valid, ct_facts=facts, labels14={k: merged(v) for k, v in labels.items()},
                 chunk_conflicts={k: 1 in v and 2 in v for k, v in labels.items()})
        groups[r['hadm_id'], 0 if r['modality'] == 'cxr' else 1].append(r)
    n = len(members['hadm_ids']); index = {str(h): i for i, h in enumerate(members['hadm_ids'])}
    values = np.zeros((n, 2, 1536), np.float32); present = np.zeros((n, 2), bool)
    for (hadm, kind), rs in groups.items():
        i = index[hadm]; present[i, kind] = True
        vector = aggregate(rs).reshape(-1) if kind == 0 else ct_vector([f for r in rs for f in r['ct_facts']])
        values[i, kind, :len(vector)] = vector
    args.output.mkdir(parents=True)
    for name, values_out in (('reports.jsonl', reports), ('excluded_entities.jsonl', dropped)):
        with (args.output / name).open('x', encoding='utf-8') as f:
            for r in values_out: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    np.savez_compressed(args.output / 'imaging.npz', **members, values=values, present=present)
    manifest.update(parent_radgraph_manifest_sha256=sha256(args.source / 'manifest.json'),
                    excluded_cross_section_entities=len(dropped), sanitization_code_sha256=sha256(Path(__file__)),
                    output_sha256={name: sha256(args.output / name) for name in ('reports.jsonl', 'excluded_entities.jsonl', 'imaging.npz')})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'reports': len(reports), 'excluded_cross_section_entities': len(dropped)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected', type=Path, default=Path('patient_data/processed/v6_imaging_all_2026-10-10_r2'))
    p.add_argument('--source', type=Path, default=Path('patient_data/processed/v6_radgraph_imaging_2026-10-10_r2'))
    p.add_argument('--output', type=Path, required=True); build(p.parse_args())
