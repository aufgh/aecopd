"""单seed融合前联合学习；原完整V6仅作重载参照，不重建病历。"""
import argparse
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from patient_data.export_mimiciv import sha256
from training.v6.discharge_regimen_screen import setup as parent_setup, ROOT as PARENT
from training.v6.discharge_regimen_cox import probability_readout
from training.v6.regularization_audit import REFERENCE
from training.v6.module_surgery import SurgeryModel, probability_curve
from training.v6.module_surgery_experiment import batches, parameters, summary, save
from training.v6.regimen_joint_fusion import JointFusion, RegimenBatch, ARMS
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import competing_nll
from training.v6.train import write_json

ROOT = Path('training_runs/v6_regimen_joint_fusion_2026-10-14')
PROTOCOL = Path('docs/V6_REGIMEN_JOINT_FUSION_PROTOCOL_2026-10-14.md')


def fingerprint(value):
    h = hashlib.sha256()
    def update(x):
        if isinstance(x, torch.Tensor):
            update(x.detach().cpu().numpy())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype.str, x.shape)).encode()); h.update(x.tobytes())
        elif is_dataclass(x):
            for f in fields(x):
                h.update(f.name.encode()); update(getattr(x, f.name))
        elif isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode()); update(x[k])
        elif isinstance(x, (tuple, list)):
            for y in x:
                update(y)
        else:
            h.update(repr(x).encode())
    update(value)
    return h.hexdigest()


def setup():
    data, train, tune, arrays, parent = parent_setup()
    if parent != json.loads((PARENT/'contract.json').read_text()):
        raise ValueError('verified parent contract differs')
    proof = json.loads((PARENT/'verification.json').read_text())
    if proof['maximum_error'] != 0 or proof['screening_sha256'] != sha256(PARENT/'screening.json'):
        raise ValueError('parent proof stale')
    sources = dict(parent['source_sha256'])
    for p in (Path(__file__), Path('training/v6/regimen_joint_fusion.py'), PROTOCOL,
              Path('training/tests/test_v6_regimen_joint_fusion.py'),
              Path('training/v6/discharge_regimen_cox.py'), PARENT/'verification.json',
              PARENT/'contract.json', PARENT/'screening.json'):
        sources[str(p)] = sha256(p)
    contract = dict(source_sha256=sources, interface_sha256=parent['data_sha256'],
                    teacher_sha256=parent['teacher_sha256'], base_sha256=parent['base_sha256'],
                    label_sha256=parent['label_sha256'], data_provenance=data[0].provenance,
                    seed=42, arms=list(ARMS), new_parameters=4480,
                    optimizer=dict(name='AdamW', lr=2e-5, weight_decay=1e-4,
                                   delta_penalty=1e-4, clip=1.),
                    early=dict(min_epochs=10, patience=7, min_delta=1e-5, max_epochs=45),
                    selection='specified tuning readmission IBS7-365 including epoch0',
                    no_high_frequency=True, confirmation_test_used=False,
                    clinical_trainable=True, bert_backbone_loaded=False,
                    original_inputs_unchanged=True, source_missing_new_injection_zero=True,
                    source_missing_final_risk_not_constrained=True)
    return data, train, tune, arrays, contract


def teacher_for(data):
    m = SurgeryModel(data[4], 42, 'control', data[5]).cuda()
    m.load_state_dict(torch.load(REFERENCE/'seed42/control/best.pt',
                                 map_location='cuda', weights_only=True)['state_dict'])
    return m.eval()


def wrap(cache, arrays):
    for _, h, b, _, _ in cache:
        if (h.shape[1] != 34 or b.base.base.clinical.current_masks.any()
                or b.base.medications.ne(0).any()
                or b.base.base.clinical.snippet_vectors.shape[1] != 768):
            raise ValueError('inherited H34/current/detail/text view changed')
    return [(pos, h, RegimenBatch(b, torch.as_tensor(arrays['features'][pos], device='cuda'),
                                  torch.as_tensor(arrays['section_available'][pos], device='cuda')),
             t, e) for pos, h, b, t, e in cache]


@torch.no_grad()
def predict(model, cache):
    a, k, nll = parameters(model, cache)
    return a, k, probability_curve(a, k), nll


def trainable(model):
    return {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}


def baseline_equal(model, teacher):
    return all(torch.equal(v, teacher.baseline.state_dict()[k]) for k, v in model.baseline.state_dict().items())


def preflight(teacher, original, new):
    old_a, old_k, old_curve, _ = predict(teacher, original)
    proof = {}
    for arm in ARMS:
        m = JointFusion(teacher, arm).cuda().eval()
        a, k, curve, _ = predict(m, new)
        for x, y in ((a, old_a), (k, old_k), (curve, old_curve)):
            np.testing.assert_array_equal(x, y)
        _, h, b, t, e = new[0]
        before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-5, weight_decay=1e-4)
        loss = competing_nll(m(h, b), t, e); loss.backward()
        if any(p.grad is not None for p in m.baseline.parameters()):
            raise ValueError('H34 received gradients')
        opt.step()
        groups = ('teacher.encoder.base.lab.', 'teacher.encoder.base.med.',
                  'teacher.encoder.base.microbiology.', 'teacher.encoder.base.support.',
                  'teacher.encoder.base.text_projection.', 'teacher.encoder.base.fusion.',
                  'teacher.encoder.extra_labs.', 'teacher.delta_head.', 'regimen.')
        updates = {prefix: any(not torch.equal(before[n], p) for n, p in m.named_parameters()
                               if n.startswith(prefix) and n in before) for prefix in groups}
        if not all(updates.values()) or not baseline_equal(m, teacher):
            raise ValueError('effective clinical/injection update or frozen H34 failed: '+repr(updates))
        m.eval()
        with torch.no_grad():
            m.regimen.weight.fill_(.03)
            x = torch.zeros_like(b.features)
            z = torch.zeros_like(b.available)
            if torch.count_nonzero(m.injection(torch.ones_like(x), z)):
                raise ValueError('missing new injection not zero')
            # 前一个batch内容不能留在网络中。
            first = m(h, b).log_amplitude.clone()
            m.injection(torch.ones_like(x), z)
            torch.testing.assert_close(m(h, b).log_amplitude, first, rtol=0, atol=0)
            one = m.injection(x, b.available); two = m.injection(x+1, b.available)
            if arm == 'availability':
                torch.testing.assert_close(one, two, rtol=0, atol=0)
            elif not (one != two).any():
                raise ValueError('content insensitive under nonzero projection')
        proof[arm] = dict(initial_parameter_daily_risk_error=0., real_step_updates=updates,
                          h34_no_gradient_and_unchanged=True, new_missing_injection_zero=True,
                          trainable_parameters=sum(trainable(m).values()))
    return dict(status='passed_real_gpu', arms=proof, gpu=torch.cuda.get_device_name())


def fit(teacher, arm, train_cache, tune_cache, d, tune, folder):
    folder.mkdir()
    model = JointFusion(teacher, arm).cuda()
    torch.manual_seed(42)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=1e-4)
    early = EarlyStopping(7, 10, 1e-5); rng = np.random.default_rng(42)
    history = []; reason = 'max_epochs'; steps = 0
    for epoch in range(46):
        online = None
        if epoch:
            model.train(); total = 0.; n = 0
            for j in rng.permutation(len(train_cache)):
                _, h, b, t, e = train_cache[j]
                opt.zero_grad(set_to_none=True); out = model(h, b)
                nll = competing_nll(out, t, e); loss = nll + 1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                if not torch.isfinite(norm):
                    raise ValueError('nonfinite gradient norm')
                opt.step(); steps += 1
                total += float(nll.detach())*len(t); n += len(t)
            online = total/n
        a, k, curve, nll = predict(model, tune_cache)
        score = summary(curve, d, tune, nll)['readmission']['integrated_brier_7_365']
        best, stop = early.observe(epoch, score)
        history.append(dict(epoch=epoch, tuning_ibs=score, train_nll=online))
        if best:
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch), folder/'best.pt')
        write_json(folder/'history.json', history)
        if epoch % 5 == 0:
            print(json.dumps(dict(arm=arm, epoch=epoch, best=early.best_epoch, tuning_ibs=score)), flush=True)
        if stop:
            reason = 'patience_exhausted'; break
    if not baseline_equal(model, teacher):
        raise ValueError('H34 changed during actual training')
    torch.save(dict(state_dict=model.state_dict(), epoch=history[-1]['epoch']), folder/'last.pt')
    info = dict(completed_epochs=history[-1]['epoch'], best_epoch=early.best_epoch,
                early=asdict(early), stop_reason=reason, optimizer_steps=steps,
                last_checkpoint_sha256=sha256(folder/'last.pt'),
                trainable_parameters=trainable(model), h34_unchanged=True,
                last_injection_norm=float(model.regimen.weight.detach().norm()))
    model.load_state_dict(torch.load(folder/'best.pt', map_location='cuda', weights_only=True)['state_dict'])
    a, k, curve, nll = predict(model, tune_cache)
    save(folder/'tuning_predictions.npz', d, tune, a, k, curve)
    info.update(metrics=summary(curve, d, tune, nll),
                probability_readout=probability_readout(d.times[tune], d.events[tune], curve),
                checkpoint_sha256=sha256(folder/'best.pt'))
    write_json(folder/'training_result.json', info)
    return info


def lock(report):
    scores = {'original': report['reference']['metrics']['readmission']['integrated_brier_7_365']}
    scores.update({a: report['fits'][a]['metrics']['readmission']['integrated_brier_7_365'] for a in ARMS})
    selected = min(scores, key=scores.get)
    content = report['fits']['content']['metrics']
    refs = (report['reference']['metrics'], report['fits']['availability']['metrics'])
    material = selected == 'content' and all(
        content['readmission']['integrated_brier_7_365'] <= .99*r['readmission']['integrated_brier_7_365']
        and content['readmission']['points']['90']['auc'] >= r['readmission']['points']['90']['auc']+.005
        and content['death']['integrated_brier_7_365'] <= r['death']['integrated_brier_7_365'] for r in refs)
    return dict(scores=scores, selected=selected, nominated=['content'] if selected == 'content' else [],
                material_signal=material, calibration_passed=False, default_changed=False, goal_achieved=False)


def prepare():
    data, train, tune, arrays, contract = setup()
    d, c, _, ext = data[:4]
    tc = batches(d, c, ext, train); vc = batches(d, c, ext, tune)
    new_train, new_tune = wrap(tc, arrays), wrap(vc, arrays)
    teacher = teacher_for(data)
    contract['input_sha256'] = dict(train=fingerprint(tc), tune=fingerprint(vc),
                                  new_train=fingerprint(new_train), new_tune=fingerprint(new_tune))
    contract['trainable_white_list'] = trainable(JointFusion(teacher, 'availability'))
    return d, train, tune, contract, teacher, tc, vc, new_train, new_tune


def run():
    if ROOT.exists():
        raise FileExistsError('batch exists; do not repeat fitting')
    started = time.perf_counter()
    d, train, tune, contract, teacher, tc, vc, nt, nv = prepare()
    ROOT.mkdir(); write_json(ROOT/'contract.json', contract)
    write_json(ROOT/'preflight.json', preflight(teacher, vc, nv))
    ref = {}
    for name, cache, ix in (('train', tc, train), ('tune', vc, tune)):
        a, k, curve, nll = predict(teacher, cache)
        folder = ROOT/'original'; folder.mkdir(exist_ok=True)
        save(folder/(name+'_predictions.npz'), d, ix, a, k, curve)
        ref[name] = dict(prediction_sha256=sha256(folder/(name+'_predictions.npz')))
        if name == 'tune':
            ref.update(metrics=summary(curve, d, ix, nll),
                       probability_readout=probability_readout(d.times[ix], d.events[ix], curve))
    parent = json.loads((PARENT/'screening.json').read_text())
    if ref['metrics'] != parent['reference']:
        raise ValueError('original full-input reference changed')
    report = dict(reference=ref, fits={}, single_seed_only=True)
    for arm in ARMS:
        write_json(ROOT/'state.json', dict(status='training', arm=arm, goal_achieved=False))
        report['fits'][arm] = fit(teacher, arm, nt, nv, d, tune, ROOT/arm)
    report['lock'] = lock(report); report['actual_seconds'] = time.perf_counter()-started
    write_json(ROOT/'screening.json', report); write_json(ROOT/'nomination_lock.json', report['lock'])
    write_json(ROOT/'state.json', dict(status='complete_pending_independent_replay', goal_achieved=False))
    print(json.dumps(report['lock']), flush=True)


def check_prediction(path, d, ix, a, k, curve):
    with np.load(path) as z:
        for key, value in (('hadm_ids', d.arrays['hadm_ids'][ix]), ('subject_ids', d.arrays['subject_ids'][ix]),
                           ('times', d.times[ix]), ('events', d.events[ix]),
                           ('log_amplitude', a), ('shape', k), ('curve', curve)):
            np.testing.assert_array_equal(z[key], value)
    if (not np.isfinite(curve).all() or np.any(curve < 0) or np.any(curve > 1)
            or np.max(np.abs(curve.sum(2)-1)) > 1e-9 or np.min(np.diff(curve[:, :, 1:], axis=1)) < -1e-10):
        raise ValueError('invalid daily joint probabilities')


def verify():
    d, train, tune, contract, teacher, tc, vc, nt, nv = prepare()
    if contract != json.loads((ROOT/'contract.json').read_text()):
        raise ValueError('source/input/white-list contract changed')
    report = json.loads((ROOT/'screening.json').read_text())
    for name, cache, ix in (('train', tc, train), ('tune', vc, tune)):
        a, k, curve, nll = predict(teacher, cache)
        path = ROOT/'original'/(name+'_predictions.npz')
        if sha256(path) != report['reference'][name]['prediction_sha256']:
            raise ValueError('teacher cache hash differs')
        check_prediction(path, d, ix, a, k, curve)
        if name == 'tune' and (summary(curve, d, ix, nll) != report['reference']['metrics']
                or probability_readout(d.times[ix], d.events[ix], curve) != report['reference']['probability_readout']):
            raise ValueError('teacher metrics/calibration differ')
    for arm in ARMS:
        folder = ROOT/arm; info = report['fits'][arm]
        history = json.loads((folder/'history.json').read_text())
        early = EarlyStopping(7, 10, 1e-5); first_stop = None
        for row in history:
            _, stop = early.observe(row['epoch'], row['tuning_ibs'])
            if stop and first_stop is None:
                first_stop = row['epoch']
        if (asdict(early) != info['early'] or len(history)-1 != info['completed_epochs']
                or info['optimizer_steps'] != len(tc)*info['completed_epochs']
                or (info['stop_reason'] == 'patience_exhausted' and first_stop != info['completed_epochs'])
                or (info['stop_reason'] == 'max_epochs' and (info['completed_epochs'] != 45 or first_stop is not None))):
            raise ValueError('early-stop/actual training differs')
        model = JointFusion(teacher, arm).cuda()
        for filename, hashkey in (('last.pt', 'last_checkpoint_sha256'), ('best.pt', 'checkpoint_sha256')):
            if sha256(folder/filename) != info[hashkey]:
                raise ValueError('checkpoint hash changed')
            saved = torch.load(folder/filename, map_location='cuda', weights_only=True)
            model.load_state_dict(saved['state_dict'])
            if not baseline_equal(model, teacher):
                raise ValueError('H34 differs')
            if filename == 'last.pt':
                if saved['epoch'] != info['completed_epochs'] or float(model.regimen.weight.detach().norm()) != info['last_injection_norm']:
                    raise ValueError('actual last training differs')
            elif saved['epoch'] != early.best_epoch:
                raise ValueError('checkpoint selection differs')
        a, k, curve, nll = predict(model, nv)
        check_prediction(folder/'tuning_predictions.npz', d, tune, a, k, curve)
        if (summary(curve, d, tune, nll) != info['metrics']
                or probability_readout(d.times[tune], d.events[tune], curve) != info['probability_readout']):
            raise ValueError('candidate metrics/calibration differ')
    if lock(report) != report['lock'] or report['lock'] != json.loads((ROOT/'nomination_lock.json').read_text()):
        raise ValueError('nomination changed')
    write_json(ROOT/'verification.json', dict(status='independent_full_input_teacher_weights_daily_calibration_early_lock_replay',
               maximum_error=0., no_refitting=True, screening_sha256=sha256(ROOT/'screening.json')))
    write_json(ROOT/'state.json', dict(status='single_seed_independently_verified', goal_achieved=False))
    print(json.dumps(dict(maximum_error=0., no_refitting=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', required=True, choices=('run', 'verify'))
    mode = parser.parse_args().mode
    try:
        run() if mode == 'run' else verify()
    except Exception as error:
        if ROOT.exists():
            write_json(ROOT/('failure_'+mode+'.json'), dict(error=repr(error)))
        raise
