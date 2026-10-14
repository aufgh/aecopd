"""复用已核验句段边界，局部补充规则再筛；全量原句仍保留，不重读/重切原文。"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import random
import re
import time
from patient_data.export_mimiciv import rows, sha256
from patient_data.clinical_facts.sentence_rules import classify, VERSION, RULE_SHA
from patient_data.clinical_facts.sentence_screen import write


def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    parent = json.loads((args.parent / 'manifest.json').read_text())
    for name, value in parent['output_sha256'].items():
        if sha256(args.parent / name) != value: raise ValueError('parent changed')
    excluded = set()
    for path in args.development_review:
        with path.open(encoding='utf-8') as stream: excluded.update(r['note_id'] for r in map(json.loads, stream))
    started = time.perf_counter(); args.output.mkdir(parents=True)
    counts = defaultdict(Counter); topics = defaultdict(Counter); changed = 0; per_doc = Counter()
    review = defaultdict(list); seen = Counter(); rng = random.Random(44)
    # 本次规则改动只影响这些词/已选句；没有命中的未选句复用原判断。
    changed_terms = re.compile(r'opaci|scar|colonic|colon\b|bowel|rectal|vitals?|\bBP\b|\bHR\b|\bRR\b|O2\s+sat|no pathologic|ventilation|home\s+with|cream|ointment|topical|Flonase|nasal\s+spray', re.I)
    with gzip.open(args.parent / 'all_units.jsonl.gz', 'rt', encoding='utf-8') as source, \
         gzip.open(args.output / 'all_units.jsonl.gz', 'xt', encoding='utf-8', compresslevel=1) as audit, \
         gzip.open(args.output / 'selected_sentences.jsonl.gz', 'xt', encoding='utf-8', compresslevel=1) as selected:
        for row in map(json.loads, source):
            if row['selected'] or changed_terms.search(row['quote']):
                decision = classify(row['quote'], row['section'], row['kind'])
                changed += int(any(row[k] != decision[k] for k in decision)); row.update(decision)
            c = counts[row['kind']]; c['units'] += 1; c['unit_chars'] += len(row['quote']); c[row['reason']] += 1
            write(audit, row)
            if row['selected']:
                write(selected, row); c['selected_chars'] += len(row['quote']); topics[row['kind']].update(row['topics']); per_doc[row['note_id']] += 1
            if row['note_id'] in excluded: continue
            bucket = (row['kind'], row['selected']); seen[bucket] += 1
            if len(review[bucket]) < 4: review[bucket].append(row.copy())
            else:
                pos = rng.randrange(seen[bucket])
                if pos < 4: review[bucket][pos] = row.copy()
    with (args.parent / 'documents.jsonl').open(encoding='utf-8') as source, (args.output / 'documents.jsonl').open('x', encoding='utf-8') as dest:
        for row in map(json.loads, source):
            row['selected'] = per_doc[row['note_id']]; c = counts[row['kind']]
            c['documents'] += 1; c['documents_with_selection'] += int(row['selected'] > 0); write(dest, row)
    wanted = {r['note_id'] for batch in review.values() for r in batch}; originals = {}
    for name in ('discharge', 'radiology'):
        for row in rows(args.snapshot / ('raw/' + name + '.csv.gz')):
            if row['note_id'] in wanted: originals[row['note_id']] = row['text']
    if originals.keys() != wanted: raise ValueError('missing review original')
    with (args.output / 'review_sample.jsonl').open('x', encoding='utf-8') as stream:
        for bucket in sorted(review):
            for row in review[bucket]:
                row['context'] = originals[row['note_id']][row['context_start']:row['context_end']]; write(stream, row)
    m = dict(status='completed_rule_candidates_not_semantic_gold', version=VERSION, rule_sha256=RULE_SHA,
        counts={k: dict(v) for k, v in counts.items()}, topics={k: dict(v) for k, v in topics.items()},
        code_sha256=sha256(Path(__file__)), rule_code_sha256=sha256(Path(__file__).with_name('sentence_rules.py')),
        source_manifest_sha256=parent['source_manifest_sha256'], parent_manifest_sha256=sha256(args.parent / 'manifest.json'),
        elapsed_seconds=time.perf_counter()-started, changed_units=changed, development_review_excluded_notes=len(excluded),
        policy=parent['policy'] + '; source-reviewed local rule refinement reuses source-audited boundaries',
        output_sha256={n: sha256(args.output / n) for n in ('all_units.jsonl.gz','selected_sentences.jsonl.gz','documents.jsonl','review_sample.jsonl')})
    (args.output / 'manifest.json').write_text(json.dumps(m, indent=2), encoding='utf-8')
    print(json.dumps({k: m[k] for k in ('status','counts','elapsed_seconds','changed_units')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--development-review', nargs='+', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    build(parser.parse_args())
