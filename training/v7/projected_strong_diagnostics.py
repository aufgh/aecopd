"""区分截距中心化与H34可解释的患者间变异；只诊断已验证权重。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json
from training.v7 import projected_strong_experiment as run
from training.v7.projected_strong import ProjectedStrongV7


def main(seed):
    folder = run.ROOT/f'seed{seed}'; proof = json.loads((folder/'verification.json').read_text())
    if proof['maximum_error'] != 0 or proof['report_sha256'] != sha256(folder/'report.json'):
        raise ValueError('projected replay stale')
    data, anchors, *_ = run.setup(seed)
    _, _, _, base, drug, tr, _, design, _ = run.prepared(data, anchors)
    model = ProjectedStrongV7(base, drug['drug_parent'], anchors, seed).cuda().eval()
    model.load_state_dict(torch.load(folder/'best.pt', map_location='cuda', weights_only=True)['state_dict'])
    hidden = np.concatenate([model.representations(b.to('cuda')).cpu().numpy() for _, b in tr]).astype(np.float64)
    residual = hidden-design @ model.projection_coefficients.cpu().numpy().astype(np.float64)
    centered = hidden-hidden.mean(0)
    total, variance, remaining = (float(np.mean(x*x)) for x in (hidden, centered, residual))
    result = dict(selected_checkpoint_sha256=sha256(folder/'best.pt'), training_rows=len(hidden),
        hidden_total_mean_square=total, hidden_centered_mean_square=variance, residual_mean_square=remaining,
        mean_component_fraction=1-variance/total,
        residual_fraction_of_centered_variance=remaining/variance if variance else None,
        limits='training representation diagnostic only; intercept removes shared mean; energy is not clinical information',
        source_sha256=sha256(Path(__file__)))
    path = folder/'representation_diagnostics.json'
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError('representation diagnostics differ on independent recomputation')
        write_json(folder/'representation_diagnostics_verification.json', dict(maximum_error=0, diagnostics_sha256=sha256(path)))
    else:
        write_json(path, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, choices=(42, 7), default=42)
    main(parser.parse_args().seed)
