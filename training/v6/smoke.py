"""真实数据接口/GPU冒烟：不做优化器更新、不保存风险模型检查点。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.data import load
from training.v6.model import AnchoredModel, HistoryBaseline
from training.v6.risk import competing_nll, predict_cif


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'state_dir', 'labels', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--field-dir', type=Path, required=True)
    parser.add_argument('--vectors-path', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU verification requested; CPU fallback forbidden')
    torch.set_num_threads(1)
    data = load(args.snapshot, args.state_dir, args.labels, field_dir=args.field_dir, vectors_path=args.vectors_path)
    torch.manual_seed(42)
    baseline = HistoryBaseline()
    model = AnchoredModel(baseline, data.text_vectors.shape[1]).cuda().eval()
    torch.manual_seed(42)  # 文本维度变化后仍恢复后续随机流。
    reports = {}
    for fold in ('train', 'validation', 'test'):
        candidates = np.flatnonzero(data.arrays['patient_splits'] == fold)
        # 固定分散取样用于接口验收；不是评估模型性能或按结局挑人。
        positions = candidates[np.linspace(0, len(candidates) - 1, min(64, len(candidates)), dtype=int)]
        history, batch, times, events = data.collate(positions, 'cuda')
        model.zero_grad(set_to_none=True)
        output = model(history, batch)
        loss = competing_nll(output, times, events)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None for p in baseline.parameters()):
            raise ValueError('nonfinite loss or history gradient leakage')
        expected = baseline(history)
        if not torch.equal(output.log_amplitude, expected.log_amplitude) or not torch.equal(output.shape, expected.shape):
            raise ValueError('zero initialized risk differs from baseline')
        _, branches = model.encoder(batch)
        with torch.no_grad():
            pmh_vector = batch.snippet_vectors.clone()
            pmh_vector[batch.snippet_index[:, 1] == 1] = 1000
            from dataclasses import replace
            left, _ = model.encoder(batch, mask_pmh=True)
            repeated, _ = model.encoder(batch, mask_pmh=True)
            right, _ = model.encoder(replace(batch, snippet_vectors=pmh_vector), mask_pmh=True)
            pmh_difference = float((left - right).abs().max().cpu())
            repeat_difference = float((left - repeated).abs().max().cpu())
            # CUDA scatter/index_add 的并行求和次序可能造成末位浮点差异。
            # 同时报告未改输入的重复误差，不把容差称为完全位级确定性。
            if not torch.allclose(left, right, rtol=1e-5, atol=1e-6):
                raise ValueError(f'PMH mask not effective: changed={pmh_difference}, repeat={repeat_difference}')
        a, k = (x.detach().cpu().numpy() for x in (output.log_amplitude, output.shape))
        curve = predict_cif(a, k, np.array([0., 7., 30., 90., 365.]))
        if not np.allclose(curve.sum(2), 1, atol=1e-10):
            raise ValueError('probability conservation failed')
        reports[fold] = {'admissions_checked': len(positions), 'snippet_rows': len(batch.snippet_vectors),
                         'finite_initial_loss': float(loss.detach().cpu()),
                         'initial_baseline_equal': True, 'history_gradients_none': True,
                         'pmh_mask_invariance': True,
                         'pmh_mask_max_absolute_difference': pmh_difference,
                         'identical_input_repeat_max_absolute_difference': repeat_difference,
                         'maximum_probability_sum_error': float(np.abs(curve.sum(2) - 1).max()),
                         'branch_mean_l2_norm': {name: float(branches[name].detach().norm(dim=1).mean().cpu())
                                                 for name in ('structured', 'lab', 'medication', 'text')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'real_input_gpu_smoke_passed_not_trained', 'folds': reports,
              'device': torch.cuda.get_device_name(0), 'torch_version': torch.__version__,
              'optimizer_steps': 0, 'checkpoint_saved': False, 'provenance': data.provenance,
              'code_sha256': {name: sha256(Path(__file__).parent / name) for name in ('model.py', 'risk.py', 'data.py', 'smoke.py')},
              'limitations': ['Baseline is random initialization, not trained; loss is not model performance',
                              'Branch norms indicate numerical flow, not predictive contribution',
                              'Historical test fold used only for schema verification in this smoke']}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'device': report['device'], 'optimizer_steps': 0,
                      'checked_admissions': sum(r['admissions_checked'] for r in reports.values())}))


if __name__ == '__main__':
    main()
