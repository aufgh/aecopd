"""核对保存的A章节向量与本次原始小结；只生成派生索引，不改原始数据。"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import numpy as np
from patient_data.export_mimiciv import rows, sha256
# 旧包__init__导入已停用的模型。单独读取纯解析模块，不恢复旧训练依赖。
_parser_path = Path(__file__).resolve().parents[1]/'training/aecopd_training/text_sections.py'
_spec = importlib.util.spec_from_file_location('_saved_section_parser', _parser_path)
_parser = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _parser
_spec.loader.exec_module(_parser)
split_clinical_document = _parser.split_clinical_document
build_section_embedding_text = _parser.build_section_embedding_text

ROLES = ('chief_complaint', 'history_of_present_illness', 'past_medical_history',
         'hospital_course', 'pertinent_results', 'discharge_diagnosis',
         'discharge_condition', 'discharge_medications', 'follow_up', 'other')
FOUR = (1, 2, 3, 6)


def section_inputs(raw):
    parsed = split_clinical_document('current_discharge', raw['note_type'], raw['text'])
    result = {}
    normalized = raw['text'].replace('\r\n', '\n').replace('\r', '\n').strip()
    for section in parsed.sections:
        # parser省略部分标题，但每个正文仍必须是规范化原文的连续片段。
        start = normalized.find(section.text)
        end = start + len(section.text)
        if start < 0:
            # 历史解析器对标题同一行正文/other标题作strip，可能删去第一行尾部空格。
            # 只允许这个明确的空白变换，不容许文本改写或跨段拼接。
            first, separator, rest = section.text.partition('\n')
            pattern = re.escape(first) + (r'[ \t]*\n'+re.escape(rest) if separator else '')
            match = re.search(pattern, normalized)
            if match is None: raise ValueError('section is not a normalized source span')
            start, end = match.span()
        if not 0 <= start <= end <= len(normalized): raise ValueError('invalid source span')
        source = normalized[start:end]
        first, separator, rest = source.partition('\n')
        if source != section.text and (first.rstrip()+separator+rest) != section.text:
            raise ValueError('source span changed beyond first-line whitespace')
        result[section.section_order] = (
            section, build_section_embedding_text('current_discharge', parsed.modality, section, raw['note_type']))
    return result


def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    started = time.perf_counter()
    manifest = json.loads((args.snapshot/'metadata/manifest.json').read_text())
    raw_path = args.snapshot/'raw/discharge.csv.gz'
    if manifest['status'] != 'validated' or sha256(raw_path) != manifest['sources']['raw/discharge']['sha256']:
        raise ValueError('raw discharge snapshot changed')
    with np.load(args.state/'clinical_state.npz') as z:
        a = {k:z[k] for k in ('hadm_ids', 'subject_ids', 'patient_splits')}
    by_h = {str(h):i for i,h in enumerate(a['hadm_ids'])}
    index_path = args.assets/'model_v2/qwen_section_embedding_index.csv.gz'
    chunks_path = args.assets/'model_v2/qwen_section_text_chunks.csv.gz'
    selected, notes = {}, {}
    for count, r in enumerate(rows(index_path)):
        if int(r['embedding_row']) != count: raise ValueError('A index is not sequential')
        if r['document_role'] != 'current_discharge' or r['hadm_id'] not in by_h: continue
        p = by_h[r['hadm_id']]
        if r['data_split'] != a['patient_splits'][p] or r['section_role'] not in ROLES:
            raise ValueError('patient split/role differs')
        if int(r['chunk_index']) != 0: raise ValueError('unexpected multi-chunk A section')
        note = r['source_document_key'].rsplit(':', 1)[1]
        selected[count] = r
        notes.setdefault(note, (p, r['source_document_key']))
        if notes[note] != (p, r['source_document_key']): raise ValueError('note ownership conflict')
    print(json.dumps({'stage':'index', 'rows':len(selected), 'notes':len(notes)}), flush=True)
    expected, note_hashes = {}, {}
    for r in rows(raw_path):
        if r['note_id'] not in notes: continue
        p, key = notes[r['note_id']]
        if r['hadm_id'] != str(a['hadm_ids'][p]) or r['subject_id'] != str(a['subject_ids'][p]):
            raise ValueError('raw note ownership differs')
        if key in expected: raise ValueError('duplicate raw note')
        expected[key] = (r, section_inputs(r))
        note_hashes[r['note_id']] = hashlib.sha256(r['text'].encode()).hexdigest()
    if len(expected) != len(notes): raise ValueError('saved note absent from current snapshot')
    kept, refs, seen, roles, fold_counts = [], [], set(), Counter(), Counter()
    for number, r in enumerate(rows(chunks_path)):
        if number not in selected: continue
        idx = selected[number]
        for key in ('document_key','source_document_key','hadm_id','data_split','document_role',
                    'section_role','section_order','section_instance','chunk_index','modality'):
            if r[key] != idx[key]: raise ValueError('A chunk/index mismatch: '+key)
        raw, sections = expected[r['source_document_key']]
        section, text = sections[int(r['section_order'])]
        start, end = int(r['char_start']), int(r['char_end'])
        if (r['source_hadm_id'] != r['hadm_id'] or r['subject_id'] != raw['subject_id']
            or r['charttime'] != raw['charttime'] or r['storetime'] != raw['storetime']
            or r['section_role'] != section.section_role or int(r['section_instance']) != section.section_instance
            or not 0 <= start <= end <= len(text) or r['chunk_text'] != text[start:end]
            or text[:start].strip() or text[end:].strip()):
            raise ValueError('raw source/span/prefix mismatch')
        key = r['source_document_key'], int(r['section_order'])
        if key in seen: raise ValueError('duplicate current section')
        seen.add(key)
        p = by_h[r['hadm_id']]
        refs.append((p, ROLES.index(r['section_role']), len(kept)))
        kept.append(number); roles[r['section_role']] += 1; fold_counts[r['data_split']] += 1
    expected_keys = {(key, order) for key,(_,sections) in expected.items() for order in sections}
    if seen != expected_keys or len(kept) != len(selected):
        raise ValueError('not all source sections represented exactly once')
    vectors_path = args.assets/'bioclinicalbert_sections_2026-09-24/bioclinicalbert_section_embeddings.float16.npy'
    state_path = vectors_path.parent/'encoding_state.json'
    state = json.loads(state_path.read_text())
    if state['status'] != 'complete' or state['completed_rows'] != count+1:
        raise ValueError('incomplete A BERT')
    v = np.load(vectors_path, mmap_mode='r')
    if v.shape != (count+1,768) or v.dtype != np.float16: raise ValueError('invalid saved shape')
    subset = np.asarray(v[kept])
    norms = np.linalg.norm(subset.astype(np.float32), axis=1)
    if not np.isfinite(subset).all() or not np.all((norms > .99) & (norms < 1.01)):
        raise ValueError('invalid selected encoding')
    args.output.mkdir(parents=True)
    np.save(args.output/'vectors.float16.npy', subset)
    np.savez_compressed(args.output/'alignment.npz', **a, refs=np.asarray(refs,np.int64), source_rows=np.asarray(kept,np.int64))
    matched = np.unique(np.asarray(refs)[:,0])
    audit = dict(status='current_discharge_source_and_complete_section_coverage_verified',
        roles=list(ROLES), four_ids=list(FOUR), rows=len(kept), admissions=len(matched),
        patients=len(np.unique(a['subject_ids'][matched])), source_notes=len(notes),
        sections=dict(roles), section_rows_by_fold=dict(fold_counts),
        members_by_fold={f:int(np.sum(a['patient_splits'][matched]==f)) for f in ('train','validation','test')},
        norm_min=float(norms.min()), norm_max=float(norms.max()),
        note_text_sha256=note_hashes,
        source_sha256={str(p):sha256(p) for p in (raw_path,index_path,chunks_path,vectors_path,state_path)},
        state_manifest_sha256=sha256(args.state/'manifest.json'),
        output_sha256={f:sha256(args.output/f) for f in ('vectors.float16.npy','alignment.npz')},
        code_sha256={str(p):sha256(p) for p in (Path(__file__),Path('training/aecopd_training/text_sections.py'))},
        limitations=['Full means all natural sections of the selected saved current note, not every database note.',
            'Subset comes from an old strict cohort and then-time note selection; not population-wide coverage.',
            'Final stored discharge summary is treated as discharge information; not real-time deployment.',
            'Vectors are reused from historical completion state; this audit does not independently rerun BERT.'],
        elapsed_seconds=time.perf_counter()-started)
    (args.output/'manifest.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ('status','rows','admissions','members_by_fold','elapsed_seconds')}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--state',type=Path,default=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2'))
    p.add_argument('--assets',type=Path,default=Path('outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'))
    p.add_argument('--output',type=Path,default=Path('patient_data/processed/v6_saved_sections_aligned_2026-10-11'))
    build(p.parse_args())
