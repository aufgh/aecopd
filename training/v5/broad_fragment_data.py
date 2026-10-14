"""新片段只替换文本引用；沿用核验后的结构化输入与完整患者历史。"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from training.v5.event_data import EventData

EXCERPT_ROLE = 'clinical_excerpt'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_selection(data: EventData, selection_file: Path) -> dict:
    selection = json.loads(selection_file.read_text(encoding='utf-8'))
    if sha(selection_file.parent/selection['members_file']) != selection['member_sha256']:
        raise ValueError('Frozen batch membership changed')
    base = data.base
    by_hadm = {int(h): i for i, h in enumerate(base.hadm_ids)}
    ids = [int(r['hadm_id']) for r in selection['members']]
    if len(ids) != len(set(ids)) or len(ids) != selection['selected_notes']:
        raise ValueError('Duplicate or inconsistent batch members')
    for member in selection['members']:
        i = by_hadm.get(int(member['hadm_id']))
        if (i is None or int(base.subject_ids[i]) != int(member['subject_id']) or
                str(base.splits[i]) != member['patient_split']):
            raise ValueError('Batch patient/admission/split differs from validated model data')
    return selection


def preserve_selected_histories(data: EventData, selection: dict) -> EventData:
    """仅筛选训练/评价患者序列，保留选中患者没有小结的住院及原索引。"""
    subjects = {int(r['subject_id']) for r in selection['members']}
    base = data.base
    sequences = tuple(seq for seq in base.sequences if int(base.subject_ids[seq[0]]) in subjects)
    seen = {int(base.subject_ids[seq[0]]) for seq in sequences}
    if seen != subjects or any(len(set(base.subject_ids[seq].tolist())) != 1 or
                               len(set(base.splits[seq].tolist())) != 1 for seq in sequences):
        raise ValueError('Selected patient history missing or crosses patients/splits')
    expected = set(np.flatnonzero(np.isin(base.subject_ids, list(subjects))).tolist())
    if set(np.concatenate(sequences).tolist()) != expected:
        raise ValueError('Selected full discharge histories were truncated')
    return replace(data, base=replace(base, sequences=sequences))


def common_roles(data: EventData) -> tuple[str, ...]:
    roles = data.base.field_roles
    return roles if EXCERPT_ROLE in roles else (*roles, EXCERPT_ROLE)


def old_or_no_text(data: EventData, selection: dict, *, enabled: bool) -> EventData:
    hadms = {int(r['hadm_id']) for r in selection['members']}
    base = data.base
    refs = tuple(refs if enabled and int(hadm) in hadms else ()
                 for hadm, refs in zip(base.hadm_ids, base.text_by_admission, strict=True))
    return replace(data, base=replace(base, text_by_admission=refs, field_roles=common_roles(data)))


def attach_broad_text(data: EventData, selection: dict, directory: Path, encoder: str) -> EventData:
    if encoder not in ('bert', 'qwen'):
        raise ValueError('Unsupported encoder')
    path = directory/'shared_text.csv.gz'
    manifest_sha = sha(path)
    summary = json.loads((directory/'manifest_summary.json').read_text(encoding='utf-8'))
    validation = json.loads((directory/'vector_validation.json').read_text(encoding='utf-8'))
    state = json.loads((directory/encoder/'state.json').read_text(encoding='utf-8'))
    width = 768 if encoder == 'bert' else 2560
    filename = 'bioclinicalbert.float16.npy' if encoder == 'bert' else 'qwen3_embedding.float16.npy'
    if (summary['status'] != 'manifest_ready' or not summary['provenance_validated'] or
            summary['manifest_sha256'] != manifest_sha or validation['manifest_sha256'] != manifest_sha or
            validation['encoders'][encoder]['status'] != 'validated' or
            state['status'] != 'complete' or state['manifest_sha256'] != manifest_sha or
            state['completed_rows'] != summary['rows']):
        raise ValueError('Incomplete or changed shared manifest/encoding')
    if ((encoder == 'bert' and state.get('model') != 'Bio_ClinicalBERT') or
            (encoder == 'qwen' and state.get('encoder') != 'Qwen3-Embedding-4B')):
        raise ValueError('Encoder identity mismatch')
    vectors = np.load(directory/encoder/filename, mmap_mode='r')
    if vectors.shape != (summary['rows'], width) or vectors.dtype != np.float16:
        raise ValueError('Wrong vector width/rows/dtype')
    members = {int(r['hadm_id']): r for r in selection['members']}
    base = data.base
    by_hadm = {int(h): i for i, h in enumerate(base.hadm_ids)}
    roles = common_roles(data)
    role = roles.index(EXCERPT_ROLE)
    refs: list[list[tuple[int, int]]] = [[] for _ in base.hadm_ids]
    csv.field_size_limit(2**31-1)
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(vectors):
        raise ValueError('Vector row count mismatch')
    for row_number, row in enumerate(rows):
        hadm = int(row['hadm_id'])
        member = members.get(hadm)
        if member is None or hadm not in by_hadm:
            raise ValueError('Text outside selected batch')
        if (int(row['embedding_row']) != row_number or int(row['subject_id']) != int(member['subject_id']) or
                row['note_id'] != member['note_id'] or row['patient_split'] != member['patient_split'] or
                row['source_sha256'] != member['source_sha256'] or
                hashlib.sha256(row['chunk_text'].encode()).hexdigest() != row['text_sha256']):
            raise ValueError('Text identity/order/content mismatch')
        # 主题为检索提示，不复制一个向量给多个主题，不将其当成已证实语义标签。
        refs[by_hadm[hadm]].append((role, row_number))
    for start in range(0, len(vectors), 2048):
        values = vectors[start:start+2048].astype(np.float32)
        norms = np.linalg.norm(values, axis=1)
        if not np.isfinite(values).all() or np.any((norms < .98) | (norms > 1.02)):
            raise ValueError('Invalid or unnormalized frozen vectors')
    return replace(data, base=replace(base, text_by_admission=tuple(tuple(r) for r in refs),
                    text_vectors=vectors, text_width=width, manifest_sha256=manifest_sha, field_roles=roles))
