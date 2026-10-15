"""固定去重出现小幅信号后，仅补两个廉价匹配读取器种子。"""
import argparse
import json
from pathlib import Path
import torch
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json
from training.v7 import residual_controls as base
from training.v7.query_replication import fit, verify_result
from training.v7.sequential_variants import Variant

ROOT = Path('training_runs/v7_sequential_research_2026-10-15/residual_followup')
PROTOCOL = Path('docs/V7_RESIDUAL_FOLLOWUP_PROTOCOL_2026-10-15.md')
VIEWS = ('scaled_v7', 'fixed_projection')
SEEDS = (7, 101)


def setup():
    a, d, parent = base.setup()
    report = json.loads((base.ROOT/'report.json').read_text())
    proof = json.loads((base.ROOT/'verification.json').read_text())
    if proof['maximum_prediction_error'] != 0 or proof['report_sha256'] != sha256(base.ROOT/'report.json'):
        raise ValueError('residual replay stale')
    if report['feature_sha256'] != sha256(base.ROOT/'features.npz'):
        raise ValueError('residual features changed')
    paths = (Path(__file__), PROTOCOL, base.ROOT/'report.json', base.ROOT/'verification.json', base.ROOT/'features.npz')
    return a, d, dict(parent=parent, seeds=SEEDS, views=VIEWS,
                     source_sha256={str(p): sha256(p) for p in paths})


def train():
    if ROOT.exists():
        raise FileExistsError('residual followup exists')
    a, d, contract = setup()
    ROOT.mkdir(); write_json(ROOT/'contract.json', contract)
    x = base.cache.load_npz(base.ROOT/'features.npz')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    results = {}
    for seed in SEEDS:
        (ROOT/f'seed{seed}').mkdir()
        results[str(seed)] = {}
        for name in VIEWS:
            factory = lambda unused, parent, variant, seed: base.Model(variant.name, seed)
            results[str(seed)][name] = fit(None, dict(drug_parent=None), Variant(name),
                base.batches(a, x[name], a['train']), base.batches(a, x[name], a['tune']), d, a['tune'],
                model_factory=factory, folder=ROOT/f'seed{seed}'/name, seed=seed,
                max_epochs=24, min_epochs=8, patience=6)
    write_json(ROOT/'report.json', dict(results=results, default_replaced=False))


def verify():
    a, d, current = setup()
    if json.loads(json.dumps(current)) != json.loads((ROOT/'contract.json').read_text()):
        raise ValueError('residual followup source differs')
    x = base.cache.load_npz(base.ROOT/'features.npz')
    report = json.loads((ROOT/'report.json').read_text())
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    for seed in SEEDS:
        for name in VIEWS:
            m = base.Model(name, seed).cuda()
            verify_result(m, base.batches(a, x[name], a['tune']), d, a['tune'],
                          ROOT/f'seed{seed}'/name, report['results'][str(seed)][name], patience=6, min_epochs=8)
            del m; torch.cuda.empty_cache()
    write_json(ROOT/'verification.json', dict(maximum_error=0, models_replayed=4,
                                             report_sha256=sha256(ROOT/'report.json')))
    print('four residual followup best/last replay error 0', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('train', 'verify'), required=True)
    train() if parser.parse_args().mode == 'train' else verify()
