"""有限患者级OOF基础风险诊断；不复用标签训练权重、不训练临床网络。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from patient_data.export_mimiciv import sha256
from training.v5.numeric_preprocessing import robust_parameters
from training.v6.data import HISTORY_NAMES
from training.v6.deep_interaction_experiment import runtime_setup
from training.v6.evaluation import metrics, public_metrics, paired_interval
from training.v6.module_surgery import probability_curve
from training.v6.regularization_audit import checked, REFERENCE
from training.v6.risk import RiskOutput, competing_nll
from training.v6.train import write_json

ROOT = Path('training_runs/v6_oof_baseline_2026-10-13')
PROTOCOL = Path('docs/V6_OOF_BASELINE_PROTOCOL_2026-10-13.md')
SEED, FOLDS, UPDATES = 42, 5, 600


def digest_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


def fold_assignment(subjects, folds=FOLDS):
    subjects = np.asarray(subjects)
    if subjects.ndim != 1 or not len(subjects) or folds < 2 or len(np.unique(subjects)) < folds:
        raise ValueError('invalid patient folds')
    ordered = sorted(np.unique(subjects).tolist(), key=lambda s: hashlib.sha256(f'v6-oof-v1:{s}'.encode()).digest())
    lookup = {s: j % folds for j, s in enumerate(ordered)}
    return np.array([lookup[s] for s in subjects], dtype=np.int64)


def fit_scales(raw, fit_indices):
    raw, ix = np.asarray(raw), np.asarray(fit_indices)
    if raw.ndim != 2 or raw.shape[1] != 6 or not np.isfinite(raw).all() or ix.ndim != 1 or not len(ix):
        raise ValueError('invalid raw history/scaling positions')
    if len(np.unique(ix)) != len(ix) or np.any(ix < 0) or np.any(ix >= len(raw)):
        raise ValueError('invalid scaling index')
    result = {}
    for c in (0, 2, 3, 4):
        rows = ix if c != 4 else ix[raw[ix, 5] == 0]
        result[HISTORY_NAMES[c]] = robust_parameters(raw[rows, c])
    return result


def matrix(raw, current, masks, scales):
    raw, current, masks = np.asarray(raw), np.asarray(current), np.asarray(masks, dtype=bool)
    if raw.shape != (len(current), 6) or current.shape != (len(raw), 14) or masks.shape != current.shape:
        raise ValueError('invalid H34 source shapes')
    if not np.isfinite(raw).all() or not np.isfinite(current).all():
        raise ValueError('nonfinite H34')
    h = raw.copy()
    for c in (0, 2, 3, 4):
        p = scales[HISTORY_NAMES[c]]
        valid = np.ones(len(h), dtype=bool) if c != 4 else raw[:, 5] == 0
        h[:, c] = np.where(valid, np.clip((h[:, c]-p['median'])/p['scale'], -8, 8), 0)
    return np.concatenate([h, np.where(masks, current, 0), masks.astype(float)], axis=1).astype(np.float32)


class FreshH34(nn.Module):
    def __init__(self, seed=SEED):
        super().__init__()
        # 不调用旧ExpandedBaseline(history)的复制已学习H6路径。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.amplitude = nn.Linear(34, 2)
        nn.init.constant_(self.amplitude.bias, -2.)
        self.raw_shape = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != 34:
            raise ValueError('expected H34 matrix')
        return RiskOutput(self.amplitude(x).clamp(-10, 6), (F.softplus(self.raw_shape)+.1).expand(len(x), 2))


def weight_digest(model):
    return hashlib.sha256(''.join(k+digest_array(v.detach().cpu().numpy()) for k, v in model.state_dict().items()).encode()).hexdigest()


def fit_model(x, times, events, *, updates=UPDATES):
    """只接收内训练特征/标签；无留出、调参标签或选点参数。"""
    device = x.device
    model = FreshH34().to(device)
    initial = weight_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=1e-4)
    history = []
    start = time.perf_counter()
    for step in range(updates+1):
        optimizer.zero_grad(set_to_none=True)
        loss = competing_nll(model(x), times, events)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite inner fit')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise ValueError('nonfinite inner gradient')
        if step % 25 == 0 or step == updates:
            history.append(dict(update=step, fixed_train_nll=float(loss.detach()), gradient_norm=float(norm)))
        if step != updates:
            optimizer.step()
    return model, dict(initial_weight_digest=initial, updates=updates, selected_update=updates,
                       selection='fixed terminal; no held-out score selection', history=history,
                       tail50_nll_change=history[-1]['fixed_train_nll']-history[-3]['fixed_train_nll'] if len(history)>=3 else None,
                       seconds=time.perf_counter()-start)


@torch.no_grad()
def predict(model, x):
    model.eval()
    parts = [model(chunk) for chunk in x.split(512)]
    return tuple(torch.cat([getattr(p, key) for p in parts]).cpu().numpy() for key in ('log_amplitude', 'shape'))


def data_contract():
    d, _, _, _, _, _, tune, _, original, prior = checked()
    train = np.flatnonzero(d.arrays['patient_splits'] == 'train')
    if len(train) != 13592 or len(tune) != 1531:
        raise ValueError('outer split changed')
    sources = {**original['source_sha256'], **prior['source_sha256']}
    for p in ('training/v6/oof_baseline_diagnostic.py', 'training/tests/test_v6_oof_baseline.py',
              'training/v6/data.py', 'training/v6/risk.py', 'training/v5/numeric_preprocessing.py',
              'training/v6/regularization_audit.py', 'training/v6/module_surgery.py', 'training/v6/token_model.py',
              'training/v6/evaluation.py'):
        sources[p] = sha256(Path(p))
    protected = {str(REFERENCE/f'seed{s}/control/best.pt'): sha256(REFERENCE/f'seed{s}/control/best.pt') for s in (42, 7)}
    parent = Path(original['input_config']['parent'])
    protected.update({str(parent/f'seed{s}/survival_base/best.pt'): sha256(parent/f'seed{s}/survival_base/best.pt') for s in (42, 7)})
    proof = dict(source_sha256=sources, protocol_sha256=sha256(PROTOCOL), protected_weights=protected,
                 data_provenance=d.provenance, train_indices_hash=digest_array(train), tuning_indices_hash=digest_array(tune),
                 raw_history_hash=digest_array(d.history_raw), current_hash=digest_array(d.current),
                 current_mask_hash=digest_array(d.current_masks), times_hash=digest_array(d.times), events_hash=digest_array(d.events),
                 seed=SEED, folds=FOLDS, updates=UPDATES, train_admissions=len(train), train_patients=len(np.unique(d.arrays['subject_ids'][train])),
                 initialization='fresh random; never loaded old H6/H34/clinical weights', clinical_fits=0,
                 optimizer=dict(name='AdamW', lr=.01, weight_decay=1e-4, clip=1., batching='full inner train'),
                 selection='fixed terminal600; outer tuning evaluation only', confirmation_test_evaluated=False)
    return d, train, tune, proof


def tensors(d, x, ix):
    return (torch.as_tensor(x[ix], device='cuda'), torch.as_tensor(d.times[ix], device='cuda'),
            torch.as_tensor(d.events[ix], device='cuda', dtype=torch.long))


def preflight():
    d, train, tune, contract = data_contract()
    if ROOT.exists():
        raise FileExistsError('diagnostic directory exists; inspect instead of repeat')
    ROOT.mkdir()
    write_json(ROOT/'contract.json', contract)
    assignment = fold_assignment(d.arrays['subject_ids'][train])
    np.savez_compressed(ROOT/'partition.npz', train_indices=train, tuning_indices=tune, fold=assignment)
    held = train[assignment == 0]
    inner = train[assignment != 0]
    assert not np.intersect1d(d.arrays['subject_ids'][inner], d.arrays['subject_ids'][held]).size
    scales = fit_scales(d.history_raw, inner)
    x = matrix(d.history_raw, d.current, d.current_masks, scales)
    sample = inner[:128]
    m = FreshH34().cuda()
    a, k = predict(m, torch.as_tensor(x[held[:128]], device='cuda'))
    restored = FreshH34().cuda(); restored.load_state_dict(m.state_dict())
    ar, kr = predict(restored, torch.as_tensor(x[held[:128]], device='cuda'))
    assert np.array_equal(a, ar) and np.array_equal(k, kr)
    fitted, tiny = fit_model(*tensors(d, x, sample), updates=1)
    assert weight_digest(fitted) != weight_digest(m)
    # 改动留出标签后，同一内训练的更新必须逐权重一致。
    altered_events = d.events.copy(); altered_events[held] = (altered_events[held]+1)%3
    repeated, _ = fit_model(torch.as_tensor(x[sample], device='cuda'), torch.as_tensor(d.times[sample], device='cuda'),
                            torch.as_tensor(altered_events[sample], device='cuda', dtype=torch.long), updates=1)
    assert weight_digest(fitted) == weight_digest(repeated)
    write_json(ROOT/'preflight.json', dict(status='passed', parameter_count=sum(p.numel() for p in m.parameters()),
        initial_reload_max_error=0., heldout_label_perturbation_weight_error=0., gpu=torch.cuda.get_device_name(0), single_update=tiny,
        patient_disjoint=True, no_clinical_fit=True))
    write_json(ROOT/'state.json', dict(status='preflight_passed_not_started', model_fits=0, goal_achieved=False))
    print(json.dumps(dict(status='preflight_passed', parameters=72, initial_error=0., model_fits=0)), flush=True)


def report(d, train, tune, assignment, ap, kp, full_a, full_k, tune_a, tune_k, fit_infos):
    oof, full, tuning = [probability_curve(a, k) for a, k in ((ap,kp),(full_a,full_k),(tune_a,tune_k))]
    for c in (oof, full, tuning):
        np.testing.assert_allclose(c.sum(2), 1., atol=1e-9)
        if not np.isfinite(c).all() or np.any((c<0)|(c>1)) or np.any(np.diff(c[:,:,1:],axis=1)<-1e-10):
            raise ValueError('invalid daily probabilities')
    def score(c, ix):
        return {str(cause):public_metrics(metrics(d.times[ix], d.events[ix], c[:,:,cause], cause=cause)) for cause in (1,2)}
    delta = oof[:,90,1]-full[:,90,1]
    counts = np.rint(np.expm1(d.history_raw[train,2])).astype(int)
    layers = {}
    for name, mask in [('first',counts==0),('one_two',(counts>=1)&(counts<=2)),('three_plus',counts>=3)]:
        layers[name] = dict(admissions=int(mask.sum()), mean_risk90_difference=float(delta[mask].mean()),
            risk90_difference_quantiles=np.quantile(delta[mask],[0,.1,.5,.9,1]).tolist())
    return dict(status='baseline_only_diagnostic_not_clinical_gain', fits=6, clinical_fits=0, goal_achieved=False,
        train_oof=score(oof,train), train_in_sample=score(full,train), tuning_fresh_refit=score(tuning,tune),
        readmission_paired_oof_minus_in_sample=paired_interval(d.arrays['subject_ids'][train],d.times[train],d.events[train],
            oof[:,:,1],full[:,:,1],replicates=200,seed=SEED), layers=layers, fit_information=fit_infos,
        mean_absolute_risk90_change=float(np.abs(delta).mean()),
        risk90_difference_quantiles=np.quantile(delta,[0,.1,.5,.9,1]).tolist(),
        caveats=['single seed, five fitted inner folds are not five clinical seeds',
                 '80% inner vs100% full training; composition, scale and sample-size effects not isolated',
                 'not evidence that OOF improves clinical residual; no clinical network fitted',
                 'fresh recipe differs from old pretrained/validation-selected H34; no old/new attribution',
                 'tuning exposed; no confirmation or historical test evaluation'])


def run():
    d, train, tune, contract = data_contract()
    if contract != json.loads((ROOT/'contract.json').read_text()):
        raise ValueError('contract changed after preflight')
    if json.loads((ROOT/'state.json').read_text())['status'] != 'preflight_passed_not_started':
        raise ValueError('already started; no repeated fits')
    assignment = fold_assignment(d.arrays['subject_ids'][train])
    ap, kp = np.empty((len(train),2),np.float32), np.empty((len(train),2),np.float32)
    infos = {}
    for f in range(FOLDS+1):
        name = f'fold{f}' if f<FOLDS else 'full'
        fit_ix = train[assignment!=f] if f<FOLDS else train
        predict_ix = train[assignment==f] if f<FOLDS else np.r_[train,tune]
        scales = fit_scales(d.history_raw,fit_ix)
        x = matrix(d.history_raw,d.current,d.current_masks,scales)
        folder = ROOT/name; folder.mkdir()
        write_json(ROOT/'state.json',dict(status='fitting',active=name,completed_fits=f,goal_achieved=False))
        model, info = fit_model(*tensors(d,x,fit_ix))
        torch.save(dict(state_dict=model.cpu().state_dict(),seed=SEED,updates=UPDATES),folder/'terminal.pt')
        model.cuda()
        a,k = predict(model,torch.as_tensor(x[predict_ix],device='cuda'))
        info.update(scales=scales,fit_index_hash=digest_array(fit_ix), prediction_index_hash=digest_array(predict_ix),
                    fit_matrix_hash=digest_array(x[fit_ix]), prediction_matrix_hash=digest_array(x[predict_ix]),
                    checkpoint_sha256=sha256(folder/'terminal.pt'), fit_admissions=len(fit_ix),
                    fit_patients=len(np.unique(d.arrays['subject_ids'][fit_ix])))
        with torch.no_grad():
            info['prediction_nll'] = float(competing_nll(RiskOutput(torch.as_tensor(a,device='cuda'),torch.as_tensor(k,device='cuda')),
                torch.as_tensor(d.times[predict_ix],device='cuda'),torch.as_tensor(d.events[predict_ix],device='cuda',dtype=torch.long)))
        np.savez_compressed(folder/'predictions.npz',indices=predict_ix,log_amplitude=a,shape=k)
        write_json(folder/'fit.json',info); infos[name]=info
        if f<FOLDS: ap[assignment==f],kp[assignment==f]=a,k
        else: full_a,full_k,tune_a,tune_k=a[:len(train)],k[:len(train)],a[len(train):],k[len(train):]
        print(json.dumps(dict(completed=name,nll=info['history'][-1]['fixed_train_nll'],seconds=info['seconds'])),flush=True)
        del model
    if len({i['initial_weight_digest'] for i in infos.values()})!=1:
        raise ValueError('initialization mismatch')
    np.savez_compressed(ROOT/'oof_risk_interface.npz',indices=train,fold=assignment,log_amplitude=ap,shape=kp)
    result=report(d,train,tune,assignment,ap,kp,full_a,full_k,tune_a,tune_k,infos)
    write_json(ROOT/'diagnostic.json',result)
    write_json(ROOT/'state.json',dict(status='fits_complete_pending_independent_verification',model_fits=6,clinical_fits=0,goal_achieved=False))
    print(json.dumps(dict(status=result['status'],fits=6,mean_absolute_risk90_change=result['mean_absolute_risk90_change'])),flush=True)


def verify(tag):
    d,train,tune,contract=data_contract()
    if contract != json.loads((ROOT/'contract.json').read_text()): raise ValueError('contract changed')
    assignment=fold_assignment(d.arrays['subject_ids'][train])
    with np.load(ROOT/'partition.npz') as z:
        for name,a in [('fold',assignment),('train_indices',train),('tuning_indices',tune)]: np.testing.assert_array_equal(z[name],a)
    ap,kp=np.empty((len(train),2),np.float32),np.empty((len(train),2),np.float32)
    infos,errors={},{}
    for f in range(FOLDS+1):
        name=f'fold{f}' if f<FOLDS else 'full'; folder=ROOT/name
        info=json.loads((folder/'fit.json').read_text())
        fit_ix=train[assignment!=f] if f<FOLDS else train
        pred_ix=train[assignment==f] if f<FOLDS else np.r_[train,tune]
        if f<FOLDS and np.intersect1d(d.arrays['subject_ids'][fit_ix],d.arrays['subject_ids'][pred_ix]).size:
            raise ValueError('patient overlap')
        scales=fit_scales(d.history_raw,fit_ix);x=matrix(d.history_raw,d.current,d.current_masks,scales)
        if scales!=info['scales'] or digest_array(x[fit_ix])!=info['fit_matrix_hash'] or digest_array(x[pred_ix])!=info['prediction_matrix_hash']:
            raise ValueError('inner scales/input differ')
        if (sha256(folder/'terminal.pt')!=info['checkpoint_sha256'] or info['updates']!=UPDATES or
            info['selected_update']!=UPDATES or info['initial_weight_digest']!=weight_digest(FreshH34())):
            raise ValueError('fit/checkpoint contract differs')
        m=FreshH34().cuda();m.load_state_dict(torch.load(folder/'terminal.pt',map_location='cuda',weights_only=True)['state_dict'])
        a,k=predict(m,torch.as_tensor(x[pred_ix],device='cuda'))
        with np.load(folder/'predictions.npz') as z:
            np.testing.assert_array_equal(pred_ix,z['indices'])
            error=max(float(np.abs(a-z['log_amplitude']).max()),float(np.abs(k-z['shape']).max()))
        if error>3e-7: raise ValueError('prediction replay error')
        errors[name]=error;infos[name]=info
        if f<FOLDS:ap[assignment==f],kp[assignment==f]=a,k
        else: full_a,full_k,tune_a,tune_k=a[:len(train)],k[:len(train)],a[len(train):],k[len(train):]
        del m
    with np.load(ROOT/'oof_risk_interface.npz') as z:
        for name,a in [('indices',train),('fold',assignment),('log_amplitude',ap),('shape',kp)]:np.testing.assert_array_equal(z[name],a)
    r=report(d,train,tune,assignment,ap,kp,full_a,full_k,tune_a,tune_k,infos)
    if r!=json.loads((ROOT/'diagnostic.json').read_text()):raise ValueError('numeric report differs')
    proof=dict(status='independently_verified',models=6,maximum_replay_error=max(errors.values()),
               replay_errors=errors,report_sha256=sha256(ROOT/'diagnostic.json'),contract_sha256=sha256(ROOT/'contract.json'),
               patient_partition_rebuilt=True,inner_scales_rebuilt=True,confirmation_test_not_evaluated=True)
    write_json(ROOT/f'verification_{tag}.json',proof)
    if all((ROOT/f'verification_{t}.json').exists() for t in ('a','b')):
        for t in ('a','b'):
            if json.loads((ROOT/f'verification_{t}.json').read_text())!=proof:raise ValueError('independent verification differs')
        write_json(ROOT/'state.json',dict(status='baseline_diagnostic_complete_verified',model_fits=6,clinical_fits=0,goal_achieved=False))
    print(json.dumps(proof),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('preflight','run','verify'),required=True)
    parser.add_argument('--tag',choices=('a','b'),default='a');args=parser.parse_args()
    runtime_setup()
    try:
        if args.mode=='preflight':preflight()
        elif args.mode=='run':run()
        else:verify(args.tag)
    except Exception as error:
        if ROOT.exists():write_json(ROOT/f'failure_{args.mode}_{args.tag}.json',dict(error=repr(error),goal_achieved=False))
        raise


if __name__=='__main__':main()
