"""单独检验强基座与端到端表示去重的组合；旧实验不修改。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import competing_nll
from training.v6.train import write_json
from training.v7 import strong_anchor_experiment as parent
from training.v7.projected_strong import ProjectedStrongV7, ridge_operator, estimate_projection, refresh_projection
from training.v7.query_replication import verify_result
from training.v7.sequential_readout import outcome_statistics, paired

ROOT = Path('training_runs/v7_projected_strong_2026-10-15')
PROTOCOL = Path('docs/V7_PROJECTED_STRONG_PROTOCOL_2026-10-15.md')


def setup(seed):
    data, anchors, parent_contract = parent.setup()
    if parent_contract != json.loads((parent.ROOT/'contract.json').read_text()):
        raise ValueError('strong experiment source changed')
    control = parent.ROOT if seed == 42 else parent.ROOT.parent/'strong_replication'
    report = json.loads((control/'report.json').read_text()); proof = json.loads((control/'verification.json').read_text())
    if proof['maximum_error'] != 0 or proof['report_sha256'] != sha256(control/'report.json'):
        raise ValueError('control replay stale')
    control_info = report if seed == 42 else report['result']
    control_folder = control if seed == 42 else control/'model'
    paths = (Path(__file__), PROTOCOL, Path('training/v7/projected_strong.py'),
             Path('training/tests/test_v7_projected_strong.py'), Path('training/v7/query_replication.py'),
             Path('training/v7/sequential_readout.py'), control/'report.json', control/'verification.json')
    contract = dict(parent=parent_contract, source_sha256={str(p): sha256(p) for p in paths}, seed=seed,
        projection='train only; refresh after each epoch; eval capture; ridge=1; no gradient through fitted map',
        max_epochs=16, min_epochs=8, patience=6, frozen_anchor_seed=42)
    return data, anchors, contract, control_info, control_folder


def prepared(data, anchors):
    d, c, ext, lab, rel, train, tune, base, drug, rad, *_ = data
    tr = parent.wrapped(parent.cache(c, ext, lab, rel, drug, rad, train))
    va = parent.wrapped(parent.cache(c, ext, lab, rel, drug, rad, tune))
    np.testing.assert_array_equal(np.concatenate([p for p, _ in tr]), train)
    context = np.concatenate([b.original.context[b.original.current].numpy() for _, b in tr])
    np.testing.assert_array_equal(context, anchors['context'][train])
    design, operator = ridge_operator(context)
    return d, train, tune, base, drug, tr, va, design, operator


def preflight(m, b, anchors, seed):
    initial = {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.eval(); m.projection_enabled = False
    a = m(b); m.projection_enabled = True; z = m(b)
    torch.testing.assert_close(a.log_amplitude, z.log_amplitude, atol=0, rtol=0)
    torch.testing.assert_close(a.shape, z.shape, atol=0, rtol=0)
    torch.testing.assert_close(z.log_amplitude, m.anchor_a[b.positions], atol=0, rtol=0)
    torch.testing.assert_close(z.shape, m.anchor_k[b.positions], atol=0, rtol=0)
    groups, _ = m.optimizer_groups(); opt = torch.optim.AdamW(groups, weight_decay=.01)
    before = m.history_fusion[0].weight.detach().clone()
    torch.manual_seed(seed); m.train()
    for _ in range(3):
        opt.zero_grad(set_to_none=True); out = m(b)
        (out.log_amplitude.square().mean()+out.shape.square().mean()).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step()
    if torch.equal(before, m.history_fusion[0].weight):
        raise ValueError('projected risk path did not update the encoder')
    if not torch.equal(initial['projection_coefficients'], m.projection_coefficients):
        raise ValueError('optimizer changed fixed map')
    if any(p.grad is not None for p in m.baseline.parameters()):
        raise ValueError('frozen H34 received gradients')
    m.load_state_dict(initial); m.zero_grad(set_to_none=True)
    return dict(initial_anchor_error=0, encoder_updated=True, projection_fixed_during_steps=True,
                baseline_frozen=True, trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad))


def comparison(d, tune, folder, info, control_folder, control):
    def read(path):
        with np.load(path) as z:
            for key, value in (('subject_ids', d.arrays['subject_ids'][tune]), ('hadm_ids', d.arrays['hadm_ids'][tune]),
                               ('times', d.times[tune]), ('events', d.events[tune])):
                np.testing.assert_array_equal(z[key], value)
            return z['curve']
    c = read(folder/'best_predictions.npz'); b = read(control_folder/'best_predictions.npz')
    _, cv = outcome_statistics(c, d.times[tune], d.events[tune], info['outputs']['best']['threshold'])
    _, bv = outcome_statistics(b, d.times[tune], d.events[tune], control['outputs']['best']['threshold'])
    people, inverse = np.unique(d.arrays['subject_ids'][tune], return_inverse=True)
    rng = np.random.default_rng(42)
    counts = np.array([np.bincount(rng.integers(len(people), size=len(people)), minlength=len(people)) for _ in range(200)])
    cm, bm = info['outputs']['best']['metrics'], control['outputs']['best']['metrics']
    df = cm['annual']['f1']-bm['annual']['f1']; di = cm['readmission']['integrated_brier_7_365']-bm['readmission']['integrated_brier_7_365']
    return dict(delta_f1=df, delta_ibs=di, **paired(cv, bv, counts[:, inverse].astype(np.float64)),
        patients=len(people), replicates=200, replication_recommended=bool(df >= .0001 and di <= .0001),
        limits='fixed fitted models/thresholds; exposed tuning; no multiplicity adjustment')


def train(seed):
    folder = ROOT/f'seed{seed}'
    if folder.exists():
        raise FileExistsError('projected strong run exists; preserve it')
    started = time.perf_counter(); data, anchors, contract, control, control_folder = setup(seed)
    d, train, tune, base, drug, tr, va, design, operator = prepared(data, anchors)
    folder.mkdir(parents=True); write_json(folder/'contract.json', contract)
    m = ProjectedStrongV7(base, drug['drug_parent'], anchors, seed).cuda()
    projection = refresh_projection(m, tr, operator, design)
    write_json(folder/'preflight.json', preflight(m, tr[0][1].to('cuda'), anchors, seed))
    groups, _ = m.optimizer_groups(); opt = torch.optim.AdamW(groups, weight_decay=.01)
    torch.manual_seed(seed); rng = np.random.default_rng(seed); early = EarlyStopping(6, 8, 1e-4)
    hist = []; best = None; mapping_seconds = 0.
    for epoch in range(17):
        tick = time.perf_counter()
        if epoch:
            m.train()
            for j in rng.permutation(len(tr)):
                p, cpu = tr[j]; opt.zero_grad(set_to_none=True); out = m(cpu.to('cuda'))
                loss = competing_nll(out, torch.as_tensor(d.times[p], device='cuda'), torch.as_tensor(d.events[p], device='cuda'))+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite projected loss')
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise ValueError('nonfinite projected gradient')
                opt.step()
            mapping_started = time.perf_counter(); projection = refresh_projection(m, tr, operator, design)
            mapping_seconds += time.perf_counter()-mapping_started
        aa, kk, curve, nll = parent.first.predict(m, va, d)
        if epoch == 0:
            np.testing.assert_array_equal(aa, anchors['anchor_a'][tune]); np.testing.assert_array_equal(kk, anchors['anchor_k'][tune])
        metrics = parent.first.parent.detail(d, tune, curve, nll)
        row = dict(epoch=epoch, annual=metrics['annual'], tuning_ibs=metrics['readmission']['integrated_brier_7_365'],
            tuning_death_ibs=metrics['death']['integrated_brier_7_365'], projection=projection, seconds=time.perf_counter()-tick)
        hist.append(row); _, stop = early.observe(epoch, -row['annual']['f1']); key = parent.first.selection_key(row)
        if best is None or key > best:
            best = key; torch.save(dict(state_dict=m.state_dict(), epoch=epoch, threshold=row['annual']['threshold']), folder/'best.pt')
        torch.save(dict(state_dict=m.state_dict(), epoch=epoch, threshold=row['annual']['threshold']), folder/'last.pt')
        write_json(folder/'history.json', hist)
        print(json.dumps(dict(seed=seed, epoch=epoch, f1=row['annual']['f1'], ibs=row['tuning_ibs'], seconds=row['seconds'])), flush=True)
        if stop:
            break
    outputs = {}
    for which in ('best', 'last'):
        ck = torch.load(folder/f'{which}.pt', map_location='cuda', weights_only=True); m.load_state_dict(ck['state_dict'])
        aa, kk, curve, nll = parent.first.predict(m, va, d)
        parent.first.parent.save(folder/f'{which}_predictions.npz', d, tune, aa, kk, curve)
        outputs[which] = dict(epoch=ck['epoch'], threshold=ck['threshold'], metrics=parent.first.parent.detail(d, tune, curve, nll, ck['threshold']),
            checkpoint_sha256=sha256(folder/f'{which}.pt'), predictions_sha256=sha256(folder/f'{which}_predictions.npz'))
    info = dict(outputs=outputs, completed_epochs=epoch, early=asdict(early), seconds=time.perf_counter()-started,
        projection_refresh_seconds=mapping_seconds, stop_reason='patience' if stop else 'budget_max_epochs')
    info['comparison'] = comparison(d, tune, folder, info, control_folder, control)
    write_json(folder/'report.json', info)


def verify(seed):
    folder = ROOT/f'seed{seed}'; data, anchors, current, control, control_folder = setup(seed)
    if current != json.loads((folder/'contract.json').read_text()):
        raise ValueError('projected source changed')
    d, train, tune, base, drug, tr, va, design, operator = prepared(data, anchors)
    m = ProjectedStrongV7(base, drug['drug_parent'], anchors, seed).cuda()
    report = json.loads((folder/'report.json').read_text())
    for which in ('best', 'last'):
        ck = torch.load(folder/f'{which}.pt', map_location='cuda', weights_only=True); m.load_state_dict(ck['state_dict'])
        coefficients, _ = estimate_projection(m, tr, operator, design)
        np.testing.assert_array_equal(coefficients, m.projection_coefficients.cpu().numpy())
        for key in ('anchor_a', 'anchor_k', 'anchor_valid'):
            np.testing.assert_array_equal(getattr(m, key).cpu().numpy(), anchors['valid' if key == 'anchor_valid' else key])
    verify_result(m, va, d, tune, folder, report, patience=6, min_epochs=8)
    if comparison(d, tune, folder, report, control_folder, control) != report['comparison']:
        raise ValueError('patient paired recomputation differs')
    write_json(folder/'verification.json', dict(maximum_error=0, projection_reestimation_error=0, best_last_replayed=True,
        metrics_selection_early_equal=True, report_sha256=sha256(folder/'report.json')))
    print(f'projected strong seed{seed} independently replayed, projection and probability error 0', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=('train', 'verify'), required=True)
    parser.add_argument('--seed', type=int, choices=(42, 7), default=42); args = parser.parse_args()
    {'train': train, 'verify': verify}[args.mode](args.seed)
