"""独立重载四臂、重新预测全验证/历史测试折；不优化参数或选择模型。"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.data import load
from training.v6.model import HistoryBaseline, ConstantCorrection, AnchoredModel
from training.v6.train import ARMS, make_batches, predict, write_json
from training.v6.risk import predict_cif
from training.v6.evaluation import GRID


def verify(root: Path):
    report = json.loads((root / 'report.json').read_text(encoding='utf-8'))
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    if report['status'] != 'completed_single_seed_exploratory':
        raise ValueError('incomplete run')
    for name, digest in report['contract']['code_sha256'].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError(f'training code changed: {name}')
    data = load(Path(config['snapshot']), Path(config['state_dir']), Path(config['labels']),
                field_dir=Path(config['field_dir']), vectors_path=Path(config['vectors_path']))
    if data.provenance != report['contract']['data_provenance']:
        raise ValueError('source provenance changed')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    base = HistoryBaseline().cuda()
    base.load_state_dict(torch.load(root / 'history/best.pt', map_location='cuda', weights_only=True)['state_dict'])
    checks = {}
    for arm in ARMS:
        checkpoint = root / arm / 'best.pt'
        if sha256(checkpoint) != report['models'][arm]['checkpoint_sha256']:
            raise ValueError('checkpoint hash mismatch')
        model = (copy.deepcopy(base) if arm == 'history' else ConstantCorrection(copy.deepcopy(base)) if arm == 'constant'
                 else AnchoredModel(copy.deepcopy(base), data.text_vectors.shape[1])).cuda()
        model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True)['state_dict'])
        if hasattr(model, 'baseline') and not all(torch.equal(v, base.state_dict()[key]) for key, v in model.baseline.state_dict().items()):
            raise ValueError('arm has different frozen baseline')
        arm_checks = {'parameters_total': sum(p.numel() for p in model.parameters()),
                      'parameters_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'folds': {}}
        for fold in ('validation', 'test'):
            ix = np.flatnonzero(data.arrays['patient_splits'] == fold)
            batches = make_batches(data, ix, config['batch_size'], 'cuda', include_text=arm == 'clinical_bert')
            a, k, _, _ = predict(model, batches)
            curve = predict_cif(a, k, GRID, step_days=.25)
            with np.load(root / arm / (fold + '_predictions.npz')) as archive:
                saved = {name: archive[name] for name in archive.files}
            if not np.array_equal(saved['hadm_ids'], data.arrays['hadm_ids'][ix]):
                raise ValueError('prediction identity changed')
            parameter_error = max(float(np.abs(a-saved['log_amplitude']).max()), float(np.abs(k-saved['shape']).max()))
            curve_error = float(np.abs(curve-saved['curve']).max())
            if parameter_error > 1e-5 or curve_error > 5e-6:
                raise ValueError(f'prediction reproduction failed: {arm}/{fold}: {parameter_error}, {curve_error}')
            if (not np.allclose(curve.sum(2), 1, atol=1e-10) or np.any(np.diff(curve[:, :, 0], axis=1) > 1e-10)
                    or np.any(np.diff(curve[:, :, 1:], axis=1) < -1e-10)):
                raise ValueError('invalid full curve')
            arm_checks['folds'][fold] = {'admissions': len(ix), 'parameter_max_difference': parameter_error,
                                         'curve_max_difference': curve_error, 'probability_and_monotonicity_passed': True}
        checks[arm] = arm_checks
    write_json(root / 'verification.json', {'status': 'independent_gpu_reload_and_full_prediction_verified',
                                           'checks': checks, 'optimizer_steps': 0,
                                           'limits': 'Engineering reproduction, not independent statistical validation'})
    print(json.dumps({'status': 'independent_gpu_reload_and_full_prediction_verified'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if (args.run / 'verification.json').exists():
        raise FileExistsError(args.run / 'verification.json')
    verify(args.run)
