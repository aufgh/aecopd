"""复用通过核验的 V6 成员和历史可见性；只缓存实际片段，避免全量 padding。"""
from pathlib import Path
import json

import numpy as np
import torch

from patient_data.export_mimiciv import sha256
from training.v6.clinical_facts_context import checked
from training.v6.unified_longitudinal import make_batch
from training.v7.model import V7Batch

DRUG_ROOT = Path('patient_data/processed/v6_audited_coarse_fine_2026-10-12_r2')
RAD_ROOT = Path('patient_data/processed/v6_radbert_imaging_2026-10-10_r2')


def load_sources(d, c):
    audit = json.loads((DRUG_ROOT / 'audit.json').read_text())
    if sha256(DRUG_ROOT / 'interface.npz') != audit['interface_sha256']:
        raise ValueError('drug interface changed')
    with np.load(DRUG_ROOT / 'interface.npz') as z:
        drug = {k: z[k] for k in ('hadm_ids', 'subject_ids', 'patient_splits', 'medications', 'med_present', 'drug_parent', 'med_names')}
    for k in ('hadm_ids', 'subject_ids', 'patient_splits'):
        np.testing.assert_array_equal(drug[k], d.arrays[k])
    if drug['medications'].shape != (len(d.times), 38, 11) or not np.isfinite(drug['medications']).all():
        raise ValueError('drug schema/finite mismatch')
    rad, _ = checked(RAD_ROOT, 'imaging.npz', 'full_radiology_processed', c)
    if rad['values'].shape != (len(d.times), 2, 1536) or not np.isfinite(rad['values']).all():
        raise ValueError('radiology schema/finite mismatch')
    hashes = {str(p): sha256(p) for p in (DRUG_ROOT / 'audit.json', DRUG_ROOT / 'interface.npz', RAD_ROOT / 'manifest.json', RAD_ROOT / 'imaging.npz')}
    return drug, rad, hashes


def make_v7_batch(c, ext, lab, relation, drug, rad, positions, *, history=True):
    b = make_batch(c, ext, lab, positions, relation['neighbors'], relation['offsets'], relation['gaps'],
                   relation['note_times'], relation['admit'], history=history, device='cpu')
    compact, _, labs, eos, context, current, past, present, gaps = b
    # 与父构造器完全相同的稳定 key 顺序；一个过去住院可按文本可见状态出现两份。
    unique = list(dict.fromkeys((int(p), True) for p in positions))
    seen = set(unique)
    for p in positions:
        selected = relation['neighbors'][relation['offsets'][p]:relation['offsets'][p+1]] if history else []
        for v in selected:
            key = (int(v), bool(relation['note_times'][v] <= relation['admit'][p]))
            if key not in seen:
                unique.append(key); seen.add(key)
    rows = np.array([p for p, _ in unique], np.int64)
    if len(rows) != len(context):
        raise ValueError('V7 key expansion mismatch')
    index = compact.clinical.snippet_index
    slots = np.empty(len(index), np.int64)
    count = np.zeros(len(rows), np.int64)
    for i, v in enumerate(index[:, 0].numpy()):
        slots[i] = count[v]; count[v] += 1
    tensor = torch.as_tensor
    return V7Batch(context, labs, eos, tensor(drug['medications'][rows]), tensor(drug['med_present'][rows]),
                   compact.med_features, compact.micro_values, compact.micro_masks,
                   compact.support_values, compact.support_masks, compact.clinical.snippet_vectors,
                   index, tensor(slots), tensor(rad['values'][rows]), tensor(rad['present'][rows]),
                   current, past, present, gaps)


def cache(c, ext, lab, relation, drug, rad, indices, *, batch_size=64):
    result = []
    for start in range(0, len(indices), batch_size):
        p = indices[start:start + batch_size]
        result.append((p, make_v7_batch(c, ext, lab, relation, drug, rad, p)))
    return result
