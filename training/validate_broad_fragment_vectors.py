"""核对双编码共同清单、完成状态、有限值与归一化；不评估医学正确率。"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from training.bonsai_broad_fragments import atomic_json


def validate(directory: Path) -> dict:
    manifest = directory/'shared_text.csv.gz'
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with gzip.open(manifest, 'rt', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    for i, row in enumerate(rows):
        if int(row['embedding_row']) != i or hashlib.sha256(row['chunk_text'].encode()).hexdigest() != row['text_sha256']:
            raise ValueError('Invalid manifest row/hash')
    report = {'manifest_sha256': sha, 'rows': len(rows), 'encoders': {},
              'clinical_accuracy_validated': False, 'prediction_performance_evaluated': False}
    for name, filename, dim in [('bert', 'bioclinicalbert.float16.npy', 768), ('qwen', 'qwen3_embedding.float16.npy', 2560)]:
        state = json.loads((directory/name/'state.json').read_text(encoding='utf-8'))
        if state['status'] != 'complete' or state['manifest_sha256'] != sha or state['completed_rows'] != len(rows):
            raise ValueError('Incomplete/mismatched encoder state: '+name)
        vectors = np.load(directory/name/filename, mmap_mode='r')
        if vectors.shape != (len(rows), dim) or vectors.dtype != np.float16:
            raise ValueError('Wrong vector shape/dtype')
        norms = []
        for start in range(0, len(rows), 1024):
            values = vectors[start:start+1024].astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite vectors')
            norm = np.linalg.norm(values, axis=1)
            if np.any((norm < .98) | (norm > 1.02)):
                raise ValueError('Vectors not L2 normalized')
            norms.extend(norm.tolist())
        report['encoders'][name] = {'status': 'validated', 'shape': list(vectors.shape),
                                    'norm_min': min(norms), 'norm_max': max(norms)}
    atomic_json(directory/'vector_validation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    print(json.dumps(validate(parser.parse_args().directory)), flush=True)
