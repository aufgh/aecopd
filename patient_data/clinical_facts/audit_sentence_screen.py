"""从原始快照独立回查全部句段、上下文、成员及章节字符覆盖；不估计医学准确率。"""
import argparse
from collections import Counter, defaultdict
import json
import gzip
from pathlib import Path
import re
import time
from patient_data.export_mimiciv import rows, sha256
from patient_data.clinical_facts.llm import digest


def audit(args):
    result = args.output / 'source_audit.json'
    if result.exists(): raise FileExistsError(result)
    start = time.perf_counter()
    m = json.loads((args.output / 'manifest.json').read_text())
    for name, expected in m['output_sha256'].items():
        if sha256(args.output / name) != expected: raise ValueError('screen output changed')
    with (args.output / 'documents.jsonl').open(encoding='utf-8') as stream:
        docs = {r['note_id']: r for r in map(json.loads, stream)}
    # 原文仅读一次，审计原句与覆盖，不重新运行抽取器作为唯一证据。
    raw = {}
    for kind in ('discharge', 'radiology'):
        for r in rows(args.snapshot / ('raw/' + kind + '.csv.gz')):
            if r['note_id'] in docs: raw[r['note_id']] = r
    if raw.keys() != docs.keys(): raise ValueError('missing original document')
    for key, d in docs.items():
        r = raw[key]
        if r['hadm_id'] != d['hadm_id'] or r['subject_id'] != d['subject_id'] or digest(r['text']) != d['original_sha256']:
            raise ValueError('source identity or SHA mismatch')
    counts = Counter(); selected_keys = set(); covered = defaultdict(list)
    with gzip.open(args.output / 'all_units.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for row in map(json.loads, stream):
            text = raw[row['note_id']]['text']; a, b = row['start_char'], row['end_char']
            ca, cb = row['context_start'], row['context_end']
            if not 0 <= ca <= a < b <= cb <= len(text) or text[a:b] != row['quote']:
                raise ValueError('original quote/context offsets invalid')
            section = docs[row['note_id']]['section_spans'][row['section_ordinal']]
            if not section['start_char'] <= ca <= cb <= section['end_char']:
                raise ValueError('context crossed section boundary')
            covered[(row['note_id'], row['section_ordinal'])].append((a, b))
            counts[row['kind'] + '_units'] += 1
            if row['selected']:
                if row['unit_key'] in selected_keys: raise ValueError('duplicate selected unit')
                selected_keys.add(row['unit_key'])
    for key, d in docs.items():
        for j, section in enumerate(d['section_spans']):
            spans = covered[key, j]
            if any(b > c for (_, b), (c, _) in zip(spans, spans[1:])): raise ValueError('overlapping units')
            original = raw[key]['text'][section['start_char']:section['end_char']]
            actual = ''.join(raw[key]['text'][a:b] for a, b in spans)
            if re.sub(r'\s', '', original) != re.sub(r'\s', '', actual): raise ValueError('nonwhitespace section text lost')
    selected_count = 0
    with gzip.open(args.output / 'selected_sentences.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for row in map(json.loads, stream):
            if not row['selected'] or row['unit_key'] not in selected_keys: raise ValueError('selected output differs')
            selected_keys.remove(row['unit_key']); selected_count += 1
    if selected_keys: raise ValueError('selected output incomplete')
    report = dict(status='all_original_quotes_contexts_and_section_coverage_checked', documents=len(docs),
        units=dict(counts), selected_units=selected_count, manifest_sha256=sha256(args.output / 'manifest.json'),
        audit_code_sha256=sha256(Path(__file__)), seconds=time.perf_counter()-start,
        limits=['Exact source and coverage validation is not semantic precision or recall',
                'Section headers/preamble outside parser sections are not included in coverage denominator',
                'Source report selection and retrospective final-summary policy remain inherited'])
    result.write_text(json.dumps(report, indent=2), encoding='utf-8'); print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args())
