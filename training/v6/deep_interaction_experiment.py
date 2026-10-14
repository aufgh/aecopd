"""A/B/C原位住院编码替换；正式训练、配对评价及独立重放分开。"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.deep_interaction import ARMS, CANDIDATES, DeepModel, gradient_groups
from training.v6.module_surgery import SurgeryModel, member_nll, probability_curve
from training.v6.module_surgery_experiment import inputs, batches, parameters, validation, summary, save, setup
from training.v6.horizon_objective import restore
from training.v6.evaluation import paired_interval
from training.v6.train import write_json

ROOT = Path('training_runs/v6_deep_interaction_2026-10-12')
REFERENCE = Path('training_runs/v6_module_surgery_2026-10-12_r3')
PROTOCOL = Path('docs/V6_DEEP_INTERACTION_PROTOCOL_2026-10-12.md')
SEEDS = (42, 7)
EPOCHS = 100


def model_for(base, seed, arm):
    if arm == 'control':
        prep = json.loads((REFERENCE/'preprocessing.json').read_text())
        m = SurgeryModel(base, seed, 'control', prep['bins'])
    else:
        m = DeepModel(base, seed, arm)
    m = m.cuda(); torch.manual_seed(seed)
    return m


def runtime_setup():
    setup()
    torch.backends.mha.set_fastpath_enabled(False)


def fit(model, train, val, times, events, path, seed):
    path.mkdir(parents=True)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    frozen = {k: v.detach().clone() for k, v in model.baseline.state_dict().items()}
    history, best, chosen = [], math.inf, 0
    torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    def record(epoch, online=None, norms=None, branch_grad=None, lr=0.):
        nonlocal best, chosen
        info = validation(model, val, times, events)
        _, _, fixed = parameters(model, train[:8])
        row = dict(epoch=epoch, train_online_nll=online, train_fixed2048_nll=fixed,
            gradient_norm_before_clip=norms, branch_gradient_norms=branch_grad, lr=lr,
            gate=float(torch.tanh(model.encoder.gate).detach()) if hasattr(model.encoder, 'gate') else None,
            **{'validation_'+k: v for k, v in info.items()})
        history.append(row)
        if info['ibs'] < best:
            best, chosen = info['ibs'], epoch
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch, validation_ibs=best), path/'best.pt')
        if epoch % 5 == 0:
            write_json(path/'history.json', history)
            print(json.dumps(dict(seed=seed, arm=model.arm, epoch=epoch, best_epoch=chosen,
                ibs=info['ibs'], auc90=info['auc90'], elapsed_seconds=round(time.perf_counter()-start, 1),
                peak_gpu_gib=round(torch.cuda.max_memory_allocated()/1024**3, 2))), flush=True)
    record(0)
    for epoch in range(1, EPOCHS+1):
        model.train()
        lr = 1e-4 * min(1., epoch/5) * (.1 + .45*(1+math.cos(math.pi*max(0, epoch-5)/(EPOCHS-5))))
        for g in opt.param_groups:
            g['lr'] = lr
        total, count, norms, branch_grad = 0., 0, [], None
        for step, j in enumerate(rng.permutation(len(train))):
            _, h, b, t, e = train[j]
            opt.zero_grad(set_to_none=True)
            out = model(h, b); nll = member_nll(out, t, e)
            loss = nll + 1e-4*out.delta.square().mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite objective')
            loss.backward()
            if step == 0:
                branch_grad = gradient_groups(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError('nonfinite gradients')
            opt.step(); total += float(nll.detach())*len(t); count += len(t); norms.append(float(norm))
        record(epoch, total/count, float(np.mean(norms)), branch_grad, lr)
    if any(not torch.equal(v, model.baseline.state_dict()[k]) for k, v in frozen.items()):
        raise ValueError('frozen baseline changed')
    result = dict(best_epoch=chosen, completed_epochs=EPOCHS, validation_ibs=best, lr=1e-4,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        checkpoint_sha256=sha256(path/'best.pt'), elapsed_seconds=time.perf_counter()-start,
        peak_training_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        peak_training_gpu_reserved_gib=torch.cuda.max_memory_reserved()/1024**3, frozen_baseline_verified=True)
    write_json(path/'history.json', history); write_json(path/'training_result.json', result)
    return result


def run():
    runtime_setup()
    if ROOT.exists():
        raise FileExistsError(ROOT)
    d, c, h, ext, config, _, prov = inputs()
    previous = json.loads((REFERENCE/'report.json').read_text())
    proof = json.loads((REFERENCE/'independent_verification.json').read_text())
    if previous['status'] != 'completed_and_independently_verified' or proof['report_sha256'] != sha256(REFERENCE/'report.json'):
        raise ValueError('verified reference required')
    protected = dict(previous['contract']['source_sha256'])
    for p, digest in protected.items():
        if sha256(Path(p)) != digest:
            raise ValueError('protected source changed: '+p)
    for p in ('training/v6/deep_interaction.py', 'training/v6/deep_interaction_experiment.py',
              'training/tests/test_v6_deep_interaction.py', str(PROTOCOL), str(REFERENCE/'report.json')):
        protected[p] = sha256(Path(p))
    ix = {f: np.flatnonzero(d.arrays['patient_splits'] == f) for f in ('train', 'validation', 'test')}
    ROOT.mkdir()
    contract = dict(source_sha256=protected, data_provenance=d.provenance, compact_provenance=c.provenance,
        extensions=prov, input_config=config, seeds=list(SEEDS), arms=list(ARMS), epochs=EPOCHS, lr=1e-4,
        batch_size=256, width=256, selection='validation readmission IBS7-365',
        frozen_baselines=previous['contract']['frozen_baselines'], fold_admissions={f:len(v) for f,v in ix.items()},
        patients=len(np.unique(d.arrays['subject_ids'])), endpoint=previous['contract']['endpoint'],
        device=torch.cuda.get_device_name(0), control='exact replay of prior r3, not retrained',
        text=previous['contract']['text'], bootstrap_repeats=200,
        limitations=['Reused test, not blind; no multiplicity adjustment.', 'Two seeds; fixed hyperparameters.',
                    'A/B/C inspired adapters, not full paper replications.', 'WorldA is observed in-system outcome; not complete outside followup.'])
    write_json(ROOT/'contract.json', contract)
    cache = {f:batches(d,c,ext,ix[f]) for f in ('train','validation')}
    fits, checks = {}, {}
    for seed in SEEDS:
        base = restore(Path(config['parent']), seed, 'survival_base', h, .1)
        fits[str(seed)], checks[str(seed)] = {}, {}
        for arm in CANDIDATES:
            m = model_for(base, seed, arm)
            # 所有候选公用tokenizer、输出和风险头初值；外层门控0不再是共同风险0的必要条件。
            other = model_for(base, seed, 'late_single')
            for name in ('tokens', 'score', 'output'):
                if any(not torch.equal(v, getattr(other.encoder,name).state_dict()[k]) for k,v in getattr(m.encoder,name).state_dict().items()):
                    raise ValueError('common initialization differs')
            for k,v in m.delta_head.state_dict().items():
                if not torch.equal(v,other.delta_head.state_dict()[k]):
                    raise ValueError('risk head initialization differs')
            del other
            with torch.no_grad():
                _, history, b, _, _ = cache['validation'][0]
                a, expected = m(history,b), base(history)
                error = float(max((a.log_amplitude-expected.log_amplitude).abs().max(),(a.shape-expected.shape).abs().max()))
            if error != 0:
                raise ValueError('initial risk differs from frozen baseline')
            checks[str(seed)][arm] = error
            fits[str(seed)][arm] = fit(m,cache['train'],cache['validation'],d.times[ix['validation']],d.events[ix['validation']],ROOT/f'seed{seed}'/arm,seed)
            del m; torch.cuda.empty_cache()
        fits[str(seed)]['control'] = dict(previous['models'][str(seed)]['control'], reused=True,
            checkpoint_path=str(REFERENCE/f'seed{seed}/control/best.pt'))
    scores = {a:float(np.mean([fits[str(s)][a]['validation_ibs'] for s in SEEDS])) for a in ARMS}
    choice = min(scores,key=scores.get)
    write_json(ROOT/'selected_before_test.json',dict(fits=fits,validation_scores=scores,selected_arm=choice,
        epoch0_errors=checks, all_training_complete=True))
    del cache['train']; torch.cuda.empty_cache()
    finalize(d,c,h,ext,config,contract,fits,scores,choice,checks)


def finalize(d,c,h,ext,config,contract,fits,scores,choice,checks):
    ix = {f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('validation','test')}
    cache = {f:batches(d,c,ext,ix[f]) for f in ix}
    models, curves, replay = {}, {}, {}
    for seed in SEEDS:
        base = restore(Path(config['parent']),seed,'survival_base',h,.1)
        models[str(seed)], curves[seed] = {}, {}
        for arm in ARMS:
            source = REFERENCE if arm=='control' else ROOT
            path = source/f'seed{seed}'/arm
            m = model_for(base,seed,arm)
            m.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            dest = ROOT/f'seed{seed}'/arm; dest.mkdir(parents=True,exist_ok=True)
            models[str(seed)][arm] = dict(**{k:v for k,v in fits[str(seed)][arm].items() if k!='folds'},folds={})
            curves[seed][arm] = {}
            for fold in ix:
                a,k,nll = parameters(m,cache[fold]); curve=probability_curve(a,k)
                if arm=='control':
                    with np.load(path/(fold+'_predictions.npz')) as z:
                        error = float(np.abs(curve-z['curve']).max())
                    if error>3e-7:
                        raise ValueError('reused control prediction differs')
                    replay[f'{seed}/{fold}'] = error
                models[str(seed)][arm]['folds'][fold] = summary(curve,d,ix[fold],nll)
                save(dest/(fold+'_predictions.npz'),d,ix[fold],a,k,curve)
                curves[seed][arm][fold] = curve
            del m
    pairs = [(a,'control') for a in CANDIDATES]+[('late_single','late_gated'),('A_deep','late_single'),('B_shared','late_single'),('C_latent','late_single')]
    comparisons = {str(s):{f:{f'{a} minus {b}':paired_interval(d.arrays['subject_ids'][ix[f]],d.times[ix[f]],d.events[ix[f]],
        curves[s][a][f][:,:,1],curves[s][b][f][:,:,1],seed=s,replicates=200) for a,b in pairs} for f in ix} for s in SEEDS}
    report = dict(status='completed_pending_independent_verification',contract=contract,models=models,
        comparisons=comparisons,validation_scores=scores,validation_selected_arm=choice,epoch0_errors=checks,
        reused_control_replay_errors=replay)
    write_json(ROOT/'report.json',report)
    print(json.dumps(dict(status=report['status'],selected=choice)),flush=True)


def verify():
    runtime_setup()
    r=json.loads((ROOT/'report.json').read_text()); contract=r['contract']
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:
            raise ValueError('source changed: '+p)
    d,c,h,ext,config,_,_=inputs()
    if d.provenance!=contract['data_provenance'] or c.provenance!=contract['compact_provenance']:
        raise ValueError('data provenance changed')
    errors={}
    for fold in ('validation','test'):
        ix=np.flatnonzero(d.arrays['patient_splits']==fold); cache=batches(d,c,ext,ix)
        for seed in SEEDS:
            base=restore(Path(config['parent']),seed,'survival_base',h,.1)
            if sha256(Path(config['parent'])/f'seed{seed}/survival_base/best.pt')!=contract['frozen_baselines'][str(seed)]:
                raise ValueError('baseline changed')
            for arm in ARMS:
                source=REFERENCE if arm=='control' else ROOT
                path=source/f'seed{seed}'/arm
                state=torch.load(path/'best.pt',map_location='cuda',weights_only=True)
                info=r['models'][str(seed)][arm]
                if sha256(path/'best.pt')!=info['checkpoint_sha256']:
                    raise ValueError('checkpoint changed')
                if arm!='control':
                    history=json.loads((path/'history.json').read_text())
                    if len(history)!=101 or min(history,key=lambda x:x['validation_ibs'])['epoch']!=state['epoch']:
                        raise ValueError('epochs or validation checkpoint choice differs')
                m=model_for(base,seed,arm);m.load_state_dict(state['state_dict'])
                if any(not torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):
                    raise ValueError('baseline not frozen')
                a,k,nll=parameters(m,cache);curve=probability_curve(a,k)
                from training.v6.token_model import daily_probabilities
                with np.load(ROOT/f'seed{seed}'/arm/(fold+'_predictions.npz')) as z:
                    for key,expected in (('hadm_ids',d.arrays['hadm_ids'][ix]),('subject_ids',d.arrays['subject_ids'][ix]),('times',d.times[ix]),('events',d.events[ix])):
                        np.testing.assert_array_equal(z[key],expected)
                    error=float(np.abs(curve-z['curve']).max())
                    daily,cond=daily_probabilities(curve)
                    np.testing.assert_allclose(daily,z['daily_event_probability'],atol=3e-7,rtol=0)
                    np.testing.assert_allclose(cond,z['conditional_daily_event_probability'],atol=3e-7,rtol=0)
                if error>3e-7 or not np.isfinite(curve).all():raise ValueError('replay failed')
                np.testing.assert_allclose(curve.sum(2),1,atol=1e-9)
                if np.any(np.diff(curve[:,:,1:],axis=1)<-1e-10):raise ValueError('nonmonotonic CIF')
                if summary(curve,d,ix,nll)!=info['folds'][fold]:raise ValueError('metric replay differs')
                errors[f'{seed}/{arm}/{fold}']=error
                del m
        del cache
    r['status']='completed_and_independently_verified';write_json(ROOT/'report.json',r)
    write_json(ROOT/'independent_verification.json',dict(status=r['status'],folds=len(errors),maximum_error=max(errors.values()),
        replay_errors=errors,report_sha256=sha256(ROOT/'report.json'),source_hashes_verified=len(contract['source_sha256'])))
    print(json.dumps(dict(status=r['status'],folds=len(errors),maximum_error=max(errors.values()))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('train','verify'),required=True)
    args=parser.parse_args();run() if args.mode=='train' else verify()
