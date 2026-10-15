"""完整 V6 上关闭跨源 Query；原合同不改，辅助选点只保存不反馈训练。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.early_stopping import EarlyStopping
from training.v6.regularization_search import validation_parts
from training.v6.risk import competing_nll
from training.v6.train import write_json
from training.v7 import strong_anchor_experiment as parent
from training.v7.query_replication import verify_result
from training.v7.strong_no_cross import StrongV7NoCross
from training.v7.sequential_readout import outcome_statistics, paired

ROOT = Path('training_runs/v7_strong_no_cross_2026-10-15')
PROTOCOL = Path('docs/V7_STRONG_NO_CROSS_PROTOCOL_2026-10-15.md')
PARTITION = Path('training_runs/v6_regularization_search_2026-10-12/validation_partition.npz')
ANCHOR = Path('training_runs/v6_symmetric_context_2026-10-14/symmetric/tuning_predictions.npz')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def split_summary(subjects, splits, train, tune):
    expected_tune, confirm = validation_parts(subjects, splits)
    np.testing.assert_array_equal(expected_tune, tune)
    np.testing.assert_array_equal(np.flatnonzero(splits == 'train'), train)
    with np.load(PARTITION) as z:
        np.testing.assert_array_equal(z['subject_ids'], subjects)
        np.testing.assert_array_equal(z['tuning_indices'], tune)
        np.testing.assert_array_equal(z['confirmation_indices'], confirm)
    groups = dict(train=train, tune=tune, confirmation=confirm, test=np.flatnonzero(splits == 'test'))
    people = {k: set(subjects[ix].tolist()) for k, ix in groups.items()}
    names = list(groups)
    if any(people[a] & people[b] for i, a in enumerate(names) for b in names[i+1:]):
        raise ValueError('patients overlap across model/validation partitions')
    if sum(len(ix) for ix in groups.values()) != len(subjects):
        raise ValueError('prediction cohort is not fully partitioned')
    return dict(total_discharges=len(subjects), total_patients=len(np.unique(subjects)),
                groups={k: dict(discharges=len(ix), patients=len(people[k])) for k, ix in groups.items()},
                validation_split='historical patient SHA256 parity split into tune and confirmation',
                patients_disjoint=True, confirmation_test_outcomes_evaluated=False)


def setup(seed):
    data, anchors, previous = parent.setup()
    if previous != read(parent.ROOT/'contract.json'):
        raise ValueError('locked strong parent differs')
    control_root = parent.ROOT if seed == 42 else parent.ROOT.parent/'strong_replication'
    report, proof = read(control_root/'report.json'), read(control_root/'verification.json')
    if proof['maximum_error'] != 0 or proof['report_sha256'] != sha256(control_root/'report.json'):
        raise ValueError('matched control proof is stale')
    info = report if seed == 42 else report['result']
    control_folder = control_root if seed == 42 else control_root/'model'
    d, _, _, _, _, train, tune, *_ = data
    split = split_summary(d.arrays['subject_ids'], d.arrays['patient_splits'], train, tune)
    sources = (Path(__file__), Path('training/v7/strong_no_cross.py'),
               Path('training/tests/test_v7_strong_no_cross.py'), PROTOCOL,
               Path('training/v7/query_replication.py'), Path('training/v7/sequential_readout.py'),
               PARTITION, ANCHOR, control_root/'report.json', control_root/'verification.json',
               control_folder/'best.pt', control_folder/'best_predictions.npz')
    contract = dict(parent=previous, source_sha256={str(p): sha256(p) for p in sources}, seed=seed,
                    max_epochs=16, min_epochs=8, patience=6, frozen_anchor_seed=42,
                    changed='cross=False; drug_cross/rad_cross frozen and bypassed',
                    loss='continuous competing NLL + 1e-4 delta square',
                    primary_selection='maximum F1; tie lower readmission IBS; tie earlier epoch',
                    auxiliary_snapshots='minimum readmission IBS and IBS<=anchor maximum F1; no training feedback',
                    split=split, confirmation_or_test_evaluated=False)
    return data, anchors, contract, info, control_folder


def batches(data):
    d, c, ext, lab, rel, train, tune, base, drug, rad, *_ = data
    tr = parent.wrapped(parent.cache(c, ext, lab, rel, drug, rad, train))
    va = parent.wrapped(parent.cache(c, ext, lab, rel, drug, rad, tune))
    np.testing.assert_array_equal(np.concatenate([p for p, _ in tr]), train)
    np.testing.assert_array_equal(np.concatenate([p for p, _ in va]), tune)
    return d, train, tune, base, drug, tr, va


def preflight(base, drug, anchors, batch, times, events, seed):
    control = parent.StrongV7(base, drug['drug_parent'], anchors, seed=seed).cuda().eval()
    model = StrongV7NoCross(base, drug['drug_parent'], anchors, seed=seed).cuda().eval()
    for name, value in control.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    expected = control(batch)
    del control
    calls = []
    handles = [module.register_forward_pre_hook(lambda *args: calls.append(True))
               for module in (model.drug_cross, model.rad_cross)]
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    out = model(batch)
    torch.testing.assert_close(out.log_amplitude, expected.log_amplitude, rtol=0, atol=0)
    torch.testing.assert_close(out.shape, expected.shape, rtol=0, atol=0)
    groups, _ = model.optimizer_groups()
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    torch.manual_seed(seed); model.train()
    for _ in range(3):
        opt.zero_grad(set_to_none=True); out = model(batch)
        loss = competing_nll(out, times, events) + 1e-4*out.delta.square().mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
    if calls:
        raise ValueError('disabled cross source was called')
    frozen = ('baseline.', 'drug_cross.', 'rad_cross.', 'contrast_projection.')
    for name, parameter in model.named_parameters():
        if name.startswith(frozen) and (parameter.grad is not None or not torch.equal(parameter, initial[name])):
            raise ValueError('frozen parameter changed: '+name)
    fields = {n: not torch.equal(p, initial[n]) for n, p in model.named_parameters()
              if n.startswith('field_pool.') and '.project.' in n}
    if not fields or not all(fields.values()):
        raise ValueError('M4 field compression did not update on real risk path')
    if torch.equal(model.history_fusion[0].weight, initial['history_fusion.0.weight']):
        raise ValueError('history encoder did not update')
    for handle in handles:
        handle.remove()
    return dict(shared_initial_state_equal=True, initial_anchor_error=0, disabled_cross_calls=0,
                field_compression_updates=fields, frozen_parameters_unchanged=True,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))


def aligned_curve(path, d, tune):
    with np.load(path) as z:
        for key, expected in (('hadm_ids', d.arrays['hadm_ids'][tune]), ('subject_ids', d.arrays['subject_ids'][tune]),
                              ('times', d.times[tune]), ('events', d.events[tune])):
            np.testing.assert_array_equal(z[key], expected)
        return z['curve']


def comparisons(folder, info, d, tune, control_folder, control):
    candidate = aligned_curve(folder/'best_predictions.npz', d, tune)
    people, inverse = np.unique(d.arrays['subject_ids'][tune], return_inverse=True)
    rng = np.random.default_rng(42)
    weights = np.array([np.bincount(rng.integers(len(people), size=len(people)), minlength=len(people))[inverse]
                        for _ in range(200)], np.float64)
    metric, values = outcome_statistics(candidate, d.times[tune], d.events[tune], info['outputs']['best']['threshold'])
    anchor_metrics = read(parent.first.ROOT/'report.json')['reference_metrics']['v6_symmetric']
    references = dict(matched_query=(control_folder/'best_predictions.npz', control['outputs']['best']['threshold']),
                      full_v6=(ANCHOR, anchor_metrics['annual']['threshold']))
    result = {}
    for name, (path, threshold) in references.items():
        other = aligned_curve(path, d, tune)
        other_metric, other_values = outcome_statistics(other, d.times[tune], d.events[tune], threshold)
        target = ((d.events[tune, None] == 2) & (d.times[tune, None] <= np.arange(366))).astype(float)
        delta = np.trapezoid(((candidate[:, :, 2]-target)**2-(other[:, :, 2]-target)**2)[:, 7:], dx=1., axis=1)/358
        result[name] = dict(delta_f1=metric['f1']-other_metric['f1'], delta_ibs=metric['ibs']-other_metric['ibs'],
                            delta_death_ibs=float(delta.mean()), **paired(values, other_values, weights),
                            death_ibs_95ci=np.quantile((weights@delta)/weights.sum(1), [.025, .975]).tolist())
    matched = result['matched_query']
    return dict(candidate_minus_reference=result, patients=len(people), replicates=200,
                replication_recommended=bool(matched['delta_f1'] >= .0001 and matched['delta_ibs'] <= .0001),
                limits='fixed fitted models and thresholds; exposed tune; no multiplicity adjustment')


def train(seed):
    folder = ROOT/f'seed{seed}'
    if folder.exists():
        raise FileExistsError('preserve existing no-cross experiment')
    started = time.perf_counter()
    data, anchors, contract, control, control_folder = setup(seed)
    d, train_indices, tune, base, drug, tr, va = batches(data)
    folder.mkdir(parents=True); write_json(folder/'contract.json', contract)
    p, cpu = tr[0]
    proof = preflight(base, drug, anchors, cpu.to('cuda'), torch.as_tensor(d.times[p], device='cuda'),
                      torch.as_tensor(d.events[p], device='cuda'), seed)
    if proof['trainable_parameters'] != 9337424:
        raise ValueError('unexpected active no-cross parameter count')
    write_json(folder/'preflight.json', proof); torch.cuda.empty_cache()
    model = StrongV7NoCross(base, drug['drug_parent'], anchors, seed=seed).cuda()
    groups, _ = model.optimizer_groups(); opt = torch.optim.AdamW(groups, weight_decay=.01)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    early = EarlyStopping(6, 8, 1e-4); history = []; keys = {}
    anchor_ibs = None
    for epoch in range(17):
        tick = time.perf_counter(); total = count = 0
        if epoch:
            model.train()
            for j in rng.permutation(len(tr)):
                positions, cpu = tr[j]; opt.zero_grad(set_to_none=True)
                out = model(cpu.to('cuda'))
                loss = competing_nll(out, torch.as_tensor(d.times[positions], device='cuda'),
                                     torch.as_tensor(d.events[positions], device='cuda')) + 1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite no-cross loss')
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise ValueError('nonfinite no-cross gradient')
                opt.step(); total += float(loss.detach())*len(positions); count += len(positions)
            if count != len(train_indices):
                raise ValueError('incomplete full training epoch')
        aa, kk, curve, nll = parent.first.predict(model, va, d)
        if epoch == 0:
            np.testing.assert_array_equal(aa, anchors['anchor_a'][tune])
            np.testing.assert_array_equal(kk, anchors['anchor_k'][tune])
        metrics = parent.first.parent.detail(d, tune, curve, nll)
        row = dict(epoch=epoch, annual=metrics['annual'], tuning_ibs=metrics['readmission']['integrated_brier_7_365'],
                   tuning_death_ibs=metrics['death']['integrated_brier_7_365'],
                   online_nll=total/count if count else None, seconds=time.perf_counter()-tick)
        if anchor_ibs is None:
            anchor_ibs = row['tuning_ibs']
        history.append(row); _, stop = early.observe(epoch, -row['annual']['f1'])
        choices = dict(best=parent.first.selection_key(row), min_ibs=(-row['tuning_ibs'], row['annual']['f1'], -epoch))
        if row['tuning_ibs'] <= anchor_ibs:
            choices['ibs_guard'] = parent.first.selection_key(row)
        for which, key in choices.items():
            if which not in keys or key > keys[which]:
                keys[which] = key
                torch.save(dict(state_dict=model.state_dict(), epoch=epoch, threshold=row['annual']['threshold']), folder/f'{which}.pt')
        torch.save(dict(state_dict=model.state_dict(), epoch=epoch, threshold=row['annual']['threshold']), folder/'last.pt')
        write_json(folder/'history.json', history)
        print(json.dumps(dict(seed=seed, epoch=epoch, f1=row['annual']['f1'], ibs=row['tuning_ibs'],
                              death_ibs=row['tuning_death_ibs'], seconds=row['seconds'])), flush=True)
        if stop:
            break
    outputs = {}
    for which in ('best', 'last', 'min_ibs', 'ibs_guard'):
        ck = torch.load(folder/f'{which}.pt', map_location='cuda', weights_only=True); model.load_state_dict(ck['state_dict'])
        aa, kk, curve, nll = parent.first.predict(model, va, d)
        parent.first.parent.save(folder/f'{which}_predictions.npz', d, tune, aa, kk, curve)
        outputs[which] = dict(epoch=ck['epoch'], threshold=ck['threshold'],
            metrics=parent.first.parent.detail(d, tune, curve, nll, ck['threshold']),
            checkpoint_sha256=sha256(folder/f'{which}.pt'), predictions_sha256=sha256(folder/f'{which}_predictions.npz'))
    info = dict(outputs=outputs, completed_epochs=epoch, early=asdict(early), seconds=time.perf_counter()-started,
                training_seconds=sum(row['seconds'] for row in history), default_replaced=False)
    info['comparison'] = comparisons(folder, info, d, tune, control_folder, control)
    write_json(folder/'report.json', info)
    print(json.dumps(dict(completed=True, comparison=info['comparison'])), flush=True)


def verify(seed):
    folder = ROOT/f'seed{seed}'
    data, anchors, contract, control, control_folder = setup(seed)
    if contract != read(folder/'contract.json'):
        raise ValueError('no-cross sources differ')
    d, _, tune, base, drug, _, va = batches(data)
    model = StrongV7NoCross(base, drug['drug_parent'], anchors, seed=seed).cuda()
    info = read(folder/'report.json')
    verify_result(model, va, d, tune, folder, info, patience=6, min_epochs=8)
    history = read(folder/'history.json')
    minimum = min(history, key=lambda row: (row['tuning_ibs'], -row['annual']['f1'], row['epoch']))
    eligible = [row for row in history if row['tuning_ibs'] <= history[0]['tuning_ibs']]
    guarded = max(eligible, key=parent.first.selection_key)
    if (minimum['epoch'] != info['outputs']['min_ibs']['epoch'] or guarded['epoch'] != info['outputs']['ibs_guard']['epoch']):
        raise ValueError('auxiliary checkpoint selection differs')
    for name in ('anchor_a', 'anchor_k', 'anchor_valid'):
        np.testing.assert_array_equal(getattr(model, name).cpu().numpy(), anchors['valid' if name == 'anchor_valid' else name])
    if info['comparison'] != comparisons(folder, info, d, tune, control_folder, control):
        raise ValueError('paired comparison differs')
    write_json(folder/'verification.json', dict(maximum_error=0, four_snapshots_replayed=True,
        metrics_selection_early_equal=True, report_sha256=sha256(folder/'report.json')))
    print('no-cross four snapshots independently replayed; maximum error 0', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=('train', 'verify'), required=True)
    parser.add_argument('--seed', choices=(42, 7), type=int, default=42); args = parser.parse_args()
    {'train': train, 'verify': verify}[args.mode](args.seed)
