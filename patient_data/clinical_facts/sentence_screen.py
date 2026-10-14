"""三类文书逐原句筛选：全量未选句也留审计；不调用LLM、不编码、不训练。"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import gzip
from pathlib import Path
import random
import re
import time
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from patient_data.clinical_facts.llm import digest
from patient_data.clinical_facts.sentence_rules import classify, VERSION, RULE_SHA
from training.v5.section_alignment_data.section_parser import parse_sections


def units(text):
    """保持原文字符索引；小数/常用缩写不切开，列表和空行分界。"""
    boundaries = {0, len(text)}
    for match in re.finditer(r'\n[ \t\r]*\n|\n(?=[ \t]*(?:[-*#]|\d+[.)]|[A-Za-z][A-Za-z /]{1,35}:)[ \t]*)', text):
        boundaries.add(match.start()); boundaries.add(match.end())
    for match in re.finditer(r'[.!?](?=\s+[A-Z0-9])', text):
        pos = match.start()
        before = text[max(0, pos - 18):pos + 1]
        if re.search(r'(?:^|\n)[ \t]*\d+\.$', text[:pos + 1]):
            continue  # 编号应与其条目保持同一语义单元。
        if text[pos] == '.' and (re.search(r'\b(?:Dr|Mr|Mrs|Ms|vs|e\.g|i\.e|q\.d|b\.i\.d|t\.i\.d)\.$', before, re.I)
                                 or re.search(r'\b[A-Z]\.$', before)):
            continue
        boundaries.add(pos + 1)
    result = []
    ordered = sorted(boundaries)
    for left, right in zip(ordered, ordered[1:]):
        while left < right and text[left].isspace(): left += 1
        while right > left and text[right - 1].isspace(): right -= 1
        if left < right: result.append((left, right))
    # 非空白字符必须且只能覆盖一次；标题不在传入body里。
    if ''.join(text[a:b] for a, b in result).split() != text.split():
        # 标点末尾相邻片段会影响split token；比较去空白字符串才是覆盖判据。
        if re.sub(r'\s', '', ''.join(text[a:b] for a, b in result)) != re.sub(r'\s', '', text):
            raise ValueError('sentence segmentation lost characters')
    return result


def screen(original, sections, kind):
    output = []
    for ordinal, section in enumerate(sections):
        start, end, body = section['start_char'], section['end_char'], section['text']
        role = section.get('role', section.get('kind', 'unclassified'))
        if not 0 <= start <= end <= len(original) or original[start:end] != body:
            raise ValueError('invalid original section offsets')
        spans = units(body)
        for i, (a, b) in enumerate(spans):
            # 相邻语句只作上下文，主题筛选根据主句。跨章节不能合并。
            ca = spans[max(0, i - 1)][0]; cb = spans[min(len(spans) - 1, i + 1)][1]
            row = dict(section=role, section_ordinal=ordinal, unit_ordinal=i,
                       start_char=start+a, end_char=start+b, quote=body[a:b],
                       context_start=start+ca, context_end=start+cb)
            row.update(classify(row['quote'], role, kind))
            output.append(row)
    return output


def checked_manifest(directory):
    m = json.loads((directory / 'manifest.json').read_text())
    for name, value in m['output_sha256'].items():
        if sha256(directory / name) != value: raise ValueError('source file changed: ' + name)
    return m


def sources(args):
    """旧任务清单只用来核对最终文书ID，全文章节重新切分，非旧主题裁剪。"""
    checked_manifest(args.notes); checked_manifest(args.imaging)
    with np.load(args.imaging / 'members.npz') as data:
        people = {str(h): str(s) for h, s in zip(data['hadm_ids'], data['subject_ids'], strict=True)}
        folds = {str(h): str(f) for h, f in zip(data['hadm_ids'], data['patient_splits'], strict=True)}
    with (args.notes / 'sources.jsonl').open(encoding='utf-8') as stream:
        summaries = [r for r in map(json.loads, stream) if r['channel'] == 'discharge']
    wanted = {}
    for kind in ('cxr', 'chest_ct'):
        with (args.imaging / (kind + '_reports.jsonl')).open(encoding='utf-8') as stream:
            for r in map(json.loads, stream): wanted[r['note_id']] = (kind, r)
    raw_m = json.loads((args.snapshot / 'metadata/manifest.json').read_text())
    for name in ('raw/radiology', 'raw/discharge'):
        path = args.snapshot / (name + '.csv.gz')
        if sha256(path) != raw_m['sources'][name]['sha256']: raise ValueError('original snapshot changed')
    expected = {r['note_id']: r for r in summaries}
    for r in rows(args.snapshot / 'raw/discharge.csv.gz'):
        if r['note_id'] not in expected: continue
        selected = expected.pop(r['note_id'])
        if digest(r['text']) != selected['original_sha256'] or r['text'] != selected['text']:
            raise ValueError('summary original changed')
        sections = [asdict(s) for s in parse_sections(r['text'])]
        if not sections: sections = [dict(role='unclassified', start_char=0, end_char=len(r['text']), text=r['text'])]
        yield 'discharge', r, sections, people, folds
    if expected: raise ValueError('summary not found in raw source')
    for r in rows(args.snapshot / 'raw/radiology.csv.gz'):
        if r['note_id'] not in wanted: continue
        kind, selected = wanted.pop(r['note_id'])
        if digest(r['text']) != selected['original_sha256']: raise ValueError('radiology original changed')
        yield kind, r, selected['sections'], people, folds
    if wanted: raise ValueError('radiology not found in raw source')


def write(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    started = time.perf_counter(); args.output.mkdir(parents=True)
    counts = defaultdict(Counter); selected_topics = defaultdict(Counter); review = defaultdict(list)
    rng = random.Random(43); reservoir_seen = Counter(); notes = []; unique = set()
    excluded_review_notes = set()
    if args.development_review:
        with args.development_review.open(encoding='utf-8') as stream:
            excluded_review_notes = {r['note_id'] for r in map(json.loads, stream)}
    with gzip.open(args.output / 'all_units.jsonl.gz', 'xt', encoding='utf-8', compresslevel=1) as audit, \
         gzip.open(args.output / 'selected_sentences.jsonl.gz', 'xt', encoding='utf-8', compresslevel=1) as chosen:
        for kind, raw, sections, people, folds in sources(args):
            h = raw['hadm_id']
            if h not in people or people[h] != raw['subject_id'] or raw['note_id'] in unique:
                raise ValueError('member/identity/duplicate mismatch')
            unique.add(raw['note_id']); stats = counts[kind]; stats['documents'] += 1
            doc_units = screen(raw['text'], sections, kind)
            meta = {k: raw[k] for k in ('subject_id', 'hadm_id', 'note_id', 'charttime', 'storetime')}
            meta.update(kind=kind, patient_split=folds[h], original_sha256=digest(raw['text']))
            notes.append(dict(meta, section_spans=[{k: s[k] for k in ('start_char', 'end_char')} for s in sections],
                              units=len(doc_units), selected=sum(v['selected'] for v in doc_units)))
            for unit in doc_units:
                row = dict(meta, **unit)
                row['unit_key'] = digest(raw['note_id'] + ':' + str(unit['start_char']) + ':' + str(unit['end_char']))
                if raw['text'][row['start_char']:row['end_char']] != row['quote']: raise ValueError('quote differs')
                stats['units'] += 1; stats['unit_chars'] += len(row['quote']); stats[row['reason']] += 1
                write(audit, row)
                if row['selected']:
                    write(chosen, row); stats['selected_chars'] += len(row['quote']); selected_topics[kind].update(row['topics'])
                if row['note_id'] in excluded_review_notes: continue
                bucket = (kind, row['selected'])
                reservoir_seen[bucket] += 1
                sample = dict(row, context=raw['text'][row['context_start']:row['context_end']])
                if len(review[bucket]) < 6: review[bucket].append(sample)
                else:
                    position = rng.randrange(reservoir_seen[bucket])
                    if position < 6: review[bucket][position] = sample
            stats['documents_with_selection'] += int(any(v['selected'] for v in doc_units))
    with (args.output / 'documents.jsonl').open('x', encoding='utf-8') as stream:
        for r in notes: write(stream, r)
    with (args.output / 'review_sample.jsonl').open('x', encoding='utf-8') as stream:
        for bucket in sorted(review):
            for row in review[bucket]: write(stream, row)
    result = dict(status='completed_rule_candidates_not_semantic_gold', version=VERSION, rule_sha256=RULE_SHA,
        counts={k: dict(v) for k, v in counts.items()}, topics={k: dict(v) for k, v in selected_topics.items()},
        elapsed_seconds=time.perf_counter()-started, code_sha256=sha256(Path(__file__)),
        source_manifest_sha256={name: sha256(directory / 'manifest.json') for name, directory in
                               (('notes', args.notes), ('imaging', args.imaging))},
        policy='No LLM/encoding/training; final summary policy unchanged; all unselected units retained; cues are lexical, not clinical assertion',
        development_review_excluded_notes=len(excluded_review_notes),
        output_sha256={n: sha256(args.output / n) for n in ('all_units.jsonl.gz', 'selected_sentences.jsonl.gz', 'documents.jsonl', 'review_sample.jsonl')})
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'counts', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    parser.add_argument('--notes', type=Path, default=Path('patient_data/processed/v6_bonsai_tasks_2026-10-10_validated'))
    parser.add_argument('--imaging', type=Path, default=Path('patient_data/processed/v6_imaging_all_2026-10-10_r2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--development-review', type=Path)
    build(parser.parse_args())
