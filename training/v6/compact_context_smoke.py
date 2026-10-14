"""全队列输入合同和各折GPU验收；不训练、不运行新风险分支。"""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import numpy as np
import torch
from training.v6.data import load
from training.v6.compact_context_data import attach
from patient_data.export_mimiciv import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'state_dir', 'labels', 'compact_dir', 'output'):
        parser.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required; no CPU fallback')
    base = load(args.snapshot, args.state_dir, args.labels)
    data = attach(base, args.compact_dir); results = {}
    for fold in ('train', 'validation', 'test'):
        candidates = np.flatnonzero(base.arrays['patient_splits'] == fold)
        # 全折逐批传GPU，覆盖稀有阳性和完全缺失；没有抽样漏掉边缘行。
        checked = 0; shapes = {}
        for start in range(0, len(candidates), 256):
            ix = candidates[start:start+256]
            h, batch, t, e = data.collate(ix, 'cuda')
            for f in fields(batch):
                if f.name == 'clinical': continue
                value = getattr(batch, f.name)
                if value.device.type != 'cuda' or not torch.isfinite(value).all(): raise ValueError('invalid GPU input')
                shapes[f.name] = list(value.shape[1:])
            torch.testing.assert_close(batch.clinical.lab_axis_inputs,
                torch.as_tensor(base.arrays['lab_axis_inputs'][ix], device='cuda'))
            checked += len(ix)
        results[fold] = {'admissions_checked': checked, 'feature_shapes_per_admission': shapes}
    report = {'status': 'compact_full_cohort_gpu_interface_verified_not_trained',
              'admissions_aligned': len(base.history), 'patients_aligned': len(set(base.arrays['subject_ids'].tolist())),
              'optimizer_steps': 0, 'checkpoint_saved': False, 'text_loaded': False,
              'device': torch.cuda.get_device_name(), 'folds': results, 'provenance': data.provenance,
              'code_sha256': {name: sha256(Path(__file__).parent/name) for name in ('compact_context_data.py', 'compact_context_smoke.py')},
              'limits': ['Input representation and device contract only; current default network and old checkpoints unchanged',
                         'No predictive performance, clinical indication or causal treatment-effect validation']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'admissions_aligned', 'patients_aligned', 'optimizer_steps', 'device')}))


if __name__ == '__main__': main()
