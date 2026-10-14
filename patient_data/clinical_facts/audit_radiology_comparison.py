"""影像比较的独立原文、时间、成员和张量回查；不冒充医学准确率。"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from patient_data.clinical_facts.llm import digest
from patient_data.cxr_report_selection import usable
from patient_data.clinical_facts.radiology_compare import records
from patient_data.clinical_facts.cxr_trajectory import SELECTED, aggregate
from patient_data.clinical_facts.features import ct_vector


def audit(args):
    data, members, selection = records(args.selected)
    sources = {r['note_id']: r for r in rows(args.snapshot / 'raw/radiology.csv.gz')}
    admissions = {r['hadm_id']: r for r in rows(args.snapshot / 'cohort/admissions.csv.gz')}
    by_id = {r['note_id']: r for r in data}; index = {str(h): i for i, h in enumerate(members['hadm_ids'])}
    report = {}; matched_presence = None
    for name, folder in (('radbert', args.radbert), ('radgraph', args.radgraph)):
        m = json.loads((folder / 'manifest.json').read_text())
        if m['status'] != 'full_radiology_processed': raise ValueError('unfinished imaging')
        for file, checksum in m['output_sha256'].items():
            if sha256(folder / file) != checksum: raise ValueError('processed input changed')
        with (folder / 'reports.jsonl').open(encoding='utf-8') as f: predictions = list(map(json.loads, f))
        if len(predictions) != len(by_id) or {r['note_id'] for r in predictions} != by_id.keys(): raise ValueError('report set differs')
        a = dict(np.load(folder / 'imaging.npz')); expected = np.zeros_like(a['present']); evidence = []; groups = defaultdict(list)
        for key in ('hadm_ids', 'subject_ids', 'patient_splits'):
            if not np.array_equal(a[key], members[key]): raise ValueError('members or splits changed')
        for p in predictions:
            source = sources[p['note_id']]; selected = by_id[p['note_id']]
            if digest(source['text']) != p['original_sha256'] or not usable(source, admissions[p['hadm_id']])[0]: raise ValueError('raw source or availability differs')
            if any(p[k] != selected[k] for k in ('hadm_id', 'subject_id', 'charttime', 'modality')): raise ValueError('report identity differs')
            kind = 0 if p['modality'] == 'cxr' else 1; pos = index[p['hadm_id']]
            expected[pos, kind] = True; groups[pos, kind].append(p)
            if name == 'radgraph':
                # 合并结果章节的偏移，再映射回原报告；跨章节的实体应当拒绝。
                maps = []; cursor = 0
                for s in selected['sections']:
                    if source['text'][s['start_char']:s['end_char']] != s['text']: raise ValueError('selected source span changed')
                    maps.append((cursor, cursor + len(s['text']), s['start_char'], s['kind'])); cursor += len(s['text']) + 1
                for e in p['evidence']:
                    if selected['text'][e['start']:e['end']] != e['quote']: raise ValueError('selected evidence mismatch')
                    span = next((s for s in maps if s[0] <= e['start'] < e['end'] <= s[1]), None)
                    if span is None: raise ValueError('entity crosses original section boundary')
                    lo = span[2] + e['start'] - span[0]; hi = span[2] + e['end'] - span[0]
                    if source['text'][lo:hi] != e['quote']: raise ValueError('original evidence mismatch')
                    evidence.append(dict(e, hadm_id=p['hadm_id'], note_id=p['note_id'], original_start=lo, original_end=hi, section=span[3]))
        if not np.array_equal(a['present'], expected) or (a['values'][~expected] != 0).any(): raise ValueError('incorrect observation mask')
        if matched_presence is not None and not np.array_equal(matched_presence, expected): raise ValueError('candidate coverage differs')
        matched_presence = expected
        if name == 'radgraph':
            for (pos, kind), predictions in groups.items():
                expected_vector = aggregate(predictions).reshape(-1) if kind == 0 else ct_vector([f for r in predictions for f in r['ct_facts']])
                if not np.array_equal(a['values'][pos, kind, :len(expected_vector)], expected_vector): raise ValueError('fact aggregation mismatch')
            evidence_path = folder / 'original_evidence_audit.jsonl'
            if evidence_path.exists(): raise FileExistsError(evidence_path)
            with evidence_path.open('x', encoding='utf-8') as f:
                for e in evidence: f.write(json.dumps(e, ensure_ascii=False) + '\n')
        else:
            lengths = np.linalg.norm(a['values'][expected], axis=1)
            if not np.allclose(lengths, np.sqrt(2), atol=2e-6): raise ValueError('dense embedding normalization differs')
        report[name] = dict(reports=len(by_id), cxr_admissions=int(expected[:, 0].sum()), ct_admissions=int(expected[:, 1].sum()),
                            exact_original_evidence=len(evidence), manifest_sha256=sha256(folder / 'manifest.json'))
    value = dict(status='full_original_source_time_members_and_arrays_checked', candidates=report,
                 identical_report_sets_and_source_masks=True, limitations=['Engineering and source roundtrip, not expert semantic accuracy', 'Frozen representations, not evidence of readmission benefit'])
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.write_text(json.dumps(value, indent=2), encoding='utf-8'); print(json.dumps(value), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected', type=Path, default=Path('patient_data/processed/v6_imaging_all_2026-10-10_r2'))
    p.add_argument('--snapshot', type=Path, default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--radbert', type=Path, default=Path('patient_data/processed/v6_radbert_imaging_2026-10-10_r2'))
    p.add_argument('--radgraph', type=Path, default=Path('patient_data/processed/v6_radgraph_imaging_source_checked_2026-10-10'))
    p.add_argument('--output', type=Path, required=True)
    audit(p.parse_args())
