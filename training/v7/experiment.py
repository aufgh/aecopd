"""V7 首轮两条件：有限训练、时间测量及新进程独立重放。"""
import argparse
from dataclasses import asdict, replace, fields
import json
from pathlib import Path
import time

import numpy as np
import torch

from patient_data.export_mimiciv import sha256
from training.v6 import unified_longitudinal_experiment as parent
from training.v6.early_stopping import EarlyStopping
from training.v6.evaluation import paired_interval
from training.v6.multihorizon_readout import horizon_readout
from training.v6.risk import competing_nll
from training.v6.train import write_json
from training.v6.unified_longitudinal import f1_at
from training.v7.data import load_sources, cache, make_v7_batch
from training.v7.losses import objective, HORIZONS
from training.v7.model import V7Model

ROOT = Path('training_runs/v7_first_2026-10-15_r2')
PROTOCOL = Path('docs/V7_FIRST_EXPERIMENT_PRD_2026-10-15.md')
ARMS = ('nll_control', 'recommended')
MAX_EPOCHS = 30


def synchronized_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def setup():
    started = time.perf_counter()
    d, c, ext, lab, relation, train, tune, _, fresh, teacher, old_contract, source = parent.setup()
    baseline = fresh.baseline.cpu()
    del fresh, teacher
    torch.cuda.empty_cache()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)
    drug, rad, hashes = load_sources(d, c)
    paths = [Path(__file__), Path('training/v7/model.py'), Path('training/v7/data.py'),
             Path('training/v7/losses.py'), Path('training/tests/test_v7_model.py'),
             Path('training/tests/test_v7_losses.py'), Path('training/v6/curve_score.py'),
             Path('training/v6/risk.py'), PROTOCOL]
    # 旧模型只读取已锁定调参概率；核验其报告/重放证据，不使用确认/测试结果。
    reference_roots = {'v6_original': parent.ROOT / 'original',
                       'v6_history_unified': parent.ROOT / 'history_unified',
                       'v6_symmetric': Path('training_runs/v6_symmetric_context_2026-10-14/symmetric')}
    for folder in (parent.ROOT, Path('training_runs/v6_symmetric_context_2026-10-14')):
        proof = json.loads((folder / 'verification.json').read_text())
        if proof['maximum_error'] != 0 or proof['report_sha256'] != sha256(folder / 'report.json'):
            raise ValueError('reference report verification changed')
        paths.extend((folder / 'report.json', folder / 'verification.json'))
    for folder in reference_roots.values():
        paths.append(folder / 'tuning_predictions.npz')
    hashes.update({str(p): sha256(p) for p in paths})
    contract = dict(parent_contract=old_contract, source_sha256=hashes,
        arms=list(ARMS), seed=42, horizons=list(HORIZONS), model=dict(d_model=256, heads=8,
        queries=4, ffn=1024, temporal_layers=2, encoder_dropout=.1, fusion_dropout=.3,
        bert_frozen=True, bert_parameters=108310272, bert_optimizer_parameters=0,
        radbert_frozen=True, cached_embeddings_reused=True, fine_drugs=38, parent_classes=10,
        radiology_categories=['CXR', 'CT'], radiology_input='category aggregate 1536, not individual report tokens'),
        optimizer=dict(name='AdamW', cached_projection_lr=1e-4, new_module_lr=3e-4,
        weight_decay=.01, clip=1., delta_penalty=1e-4), loss=dict(focal_gamma=2.,
        focal_alpha=.75, smoothing=.05, contrast_margin=.5, focal_coefficient=1., contrast_coefficient=.1,
        continuous_time_competing_nll_retained=True, contrast_sources_before_cross_attention=True),
        early=dict(patience=6, min_epochs=8, min_delta=1e-4, max_epochs=MAX_EPOCHS),
        batch_size=64, selection='maximum tuning annual F1; tie lower readmission IBS; tie earlier epoch',
        history_truncated=False, high_frequency_used=False, confirmation_or_test_used=False,
        automation_resumed=False, source_setup_seconds=time.perf_counter()-started)
    references = {}
    for name, folder in reference_roots.items():
        with np.load(folder / 'tuning_predictions.npz') as z:
            a = {k: z[k] for k in z.files}
        for k, values in (('hadm_ids', d.arrays['hadm_ids'][tune]), ('subject_ids', d.arrays['subject_ids'][tune]),
                          ('times', d.times[tune]), ('events', d.events[tune])):
            np.testing.assert_array_equal(a[k], values)
        references[name] = a
    return d, c, ext, lab, relation, train, tune, baseline, drug, rad, contract, source, references


@torch.no_grad()
def predict(model, batches, d):
    model.eval()
    aa, kk = [], []
    nll, count = 0., 0
    for p, cpu_batch in batches:
        b = cpu_batch.to('cuda')
        out = model(b)
        aa.append(out.log_amplitude.cpu().numpy()); kk.append(out.shape.cpu().numpy())
        nll += float(competing_nll(out, torch.as_tensor(d.times[p], device='cuda'),
                                 torch.as_tensor(d.events[p], device='cuda'))) * len(p)
        count += len(p)
    a, k = np.concatenate(aa), np.concatenate(kk)
    return a, k, parent.probability_curve(a, k), nll / count


def preflight(baseline, drug, b, d, positions):
    started = synchronized_time()
    model = V7Model(baseline, drug['drug_parent']).cuda().eval()
    groups, names = model.optimizer_groups()
    out = model(b)
    expected = model.baseline(b.context[b.current])
    torch.testing.assert_close(out.log_amplitude, expected.log_amplitude, rtol=0, atol=0)
    torch.testing.assert_close(out.shape, expected.shape, rtol=0, atol=0)
    second = V7Model(baseline, drug['drug_parent']).cuda().eval()
    for p, q in zip(model.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    del second
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    parts = None
    model.train()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        out, reps, masks = model(b, auxiliary=True)
        loss, parts = objective(out, torch.as_tensor(d.times[positions], device='cuda'),
            torch.as_tensor(d.events[positions], device='cuda'), reps, masks,
            torch.as_tensor(d.arrays['subject_ids'][positions], device='cuda'), recommended=True)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite GPU preflight loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise ValueError('nonfinite GPU preflight gradient')
        opt.step()
    if any(p.grad is not None or not torch.equal(p, before['baseline.' + n]) for n, p in model.baseline.named_parameters()):
        raise ValueError('frozen H34 changed/received gradient')
    prefixes = ('lab_projection.', 'drug_numeric.', 'text_projection.', 'rad_projection.',
                'drug_cross.', 'rad_cross.', 'temporal.', 'history_pool.', 'delta_head.', 'contrast_projection.')
    changed = {g: any(n.startswith(g) and not torch.equal(p, before[n]) for n, p in model.named_parameters()) for g in prefixes}
    if not all(changed.values()):
        raise ValueError('module did not update: ' + repr(changed))
    # 所有字段压缩层必须既有梯度也发生更新；不能只检查宽泛模块前缀。
    field_updates = {}
    for i, pool in enumerate(model.field_pool):
        for n, p in pool.project.named_parameters():
            name = f'field_pool.{i}.project.{n}'
            field_updates[name] = (p.grad is not None and torch.isfinite(p.grad).all().item()
                                   and p.grad.abs().sum().item() > 0 and not torch.equal(p, before[name]))
    if not all(field_updates.values()):
        raise ValueError('field compression bypassed: ' + repr(field_updates))
    disconnected = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
    if disconnected:
        raise ValueError('trainable parameter disconnected from recommended loss: '+repr(disconnected))
    model.eval()
    original = model(b).log_amplitude.detach()
    no_history = replace(b, past_present=torch.zeros_like(b.past_present))
    sensitivity = float((model(no_history).log_amplitude - original).abs().max().detach())
    if sensitivity <= 0:
        raise ValueError('history had no fitted effect')
    # 对真实过去检验求导，不能仅凭序列长度变化称为读取历史内容。
    labs = b.labs.detach().clone().requires_grad_()
    model.zero_grad(set_to_none=True)
    model(replace(b, labs=labs)).log_amplitude.sum().backward()
    past_only = torch.ones(len(b.context), dtype=torch.bool, device='cuda'); past_only[b.current] = False
    past_gradient = float(labs.grad[past_only].abs().sum())
    if past_gradient <= 0:
        raise ValueError('past clinical contents received no gradient')
    result = dict(initial_baseline_error=0, arm_initial_parameter_error=0, h34_frozen=True,
        actual_updates=changed, field_compression_updates=field_updates, disconnected_trainable_parameters=disconnected,
        no_history_sensitivity=sensitivity, past_lab_gradient= past_gradient,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_model_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        frozen_external_bert_parameters=108310272, optimizer_groups={g['group_name']:
            dict(lr=g['lr'], parameters=sum(p.numel() for p in g['params']), names=names[g['group_name']]) for g in groups},
        loss_parts={k:float(v.detach()) if torch.is_tensor(v) else v for k,v in parts.items()},
        seconds=synchronized_time()-started, device=torch.cuda.get_device_name(),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda)
    return result


def selection_key(row):
    return row['annual']['f1'], -row['tuning_ibs'], -row['epoch']


def fit(baseline, drug, arm, train_batches, tune_batches, d, tune):
    folder = ROOT / arm
    folder.mkdir()
    model = V7Model(baseline, drug['drug_parent']).cuda()
    # 构造结束后重置 dropout 与批次随机流；预检和另一条件不能污染它。
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    rng = np.random.default_rng(42)
    groups, _ = model.optimizer_groups()
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    early = EarlyStopping(6, 8, 1e-4)
    history, best_key, steps = [], None, 0
    frozen = {n:p.detach().cpu().clone() for n,p in model.baseline.named_parameters()}
    torch.cuda.reset_peak_memory_stats()
    started = synchronized_time()
    for epoch in range(MAX_EPOCHS + 1):
        train_seconds = 0.
        online = None
        if epoch:
            model.train(); totals = {k:0. for k in ('total', 'nll', 'delta_penalty', 'focal', 'contrast')}
            pair_counts = dict(positive_pairs=0, negative_pairs=0)
            count = 0; norms = []
            train_started = synchronized_time()
            for j in rng.permutation(len(train_batches)):
                p, cpu_batch = train_batches[j]
                b = cpu_batch.to('cuda')
                opt.zero_grad(set_to_none=True)
                out, reps, masks = model(b, auxiliary=True)
                loss, parts = objective(out, torch.as_tensor(d.times[p], device='cuda'),
                    torch.as_tensor(d.events[p], device='cuda'), reps, masks,
                    torch.as_tensor(d.arrays['subject_ids'][p], device='cuda'), recommended=arm=='recommended')
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise ValueError('nonfinite training gradient')
                opt.step(); steps += 1; count += len(p)
                totals['total'] += float(loss.detach()) * len(p)
                for k in ('nll', 'delta_penalty', 'focal', 'contrast'):
                    totals[k] += float(parts[k].detach()) * len(p)
                for k in pair_counts:
                    pair_counts[k] += parts[k]
                norms.append(float(norm))
            train_seconds = synchronized_time() - train_started
            online = dict(**{k:v/count for k,v in totals.items()}, **pair_counts,
                          max_gradient_norm_before_clip=max(norms), optimizer_steps=len(train_batches))
        eval_started = synchronized_time()
        a,k,curve,nll = predict(model, tune_batches, d)
        metrics = parent.detail(d, tune, curve, nll)
        eval_seconds = synchronized_time() - eval_started
        row = dict(epoch=epoch, online=online, annual=metrics['annual'],
                   tuning_nll=nll, tuning_ibs=metrics['readmission']['integrated_brier_7_365'],
                   tuning_death_ibs=metrics['death']['integrated_brier_7_365'],
                   train_seconds=train_seconds, evaluation_seconds=eval_seconds)
        history.append(row)
        _, stop = early.observe(epoch, -row['annual']['f1'])
        key = selection_key(row)
        if best_key is None or key > best_key:
            best_key = key
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch, threshold=row['annual']['threshold']), folder/'best.pt')
        write_json(folder/'history.json', history)
        # 每轮保存故障可恢复现场，但本轮入口不自动续训或越过预算。
        torch.save(dict(state_dict=model.state_dict(), epoch=epoch, optimizer_steps=steps), folder/'last.pt')
        print(json.dumps(dict(arm=arm, epoch=epoch, f1=row['annual']['f1'], ibs=row['tuning_ibs'],
                              train_seconds=train_seconds, evaluation_seconds=eval_seconds)), flush=True)
        if stop:
            break
    if any(not torch.equal(p.detach().cpu(), frozen[n]) for n,p in model.baseline.named_parameters()):
        raise ValueError('base changed during fit')
    chosen = torch.load(folder/'best.pt', map_location='cuda', weights_only=True)
    model.load_state_dict(chosen['state_dict'])
    a,k,curve,nll = predict(model, tune_batches, d)
    metrics = parent.detail(d, tune, curve, nll, chosen['threshold'])
    parent.save(folder/'tuning_predictions.npz', d, tune, a,k,curve)
    thresholds = {}
    for cause in (1,2):
        y = (d.events[tune]==cause) & (d.times[tune]<=365)
        thresholds[cause] = chosen['threshold'] if cause==1 else parent.best_f1(y,curve[:,365,cause])['threshold']
    multihorizon = {label:{str(day):horizon_readout(dict(events=d.events[tune], times=d.times[tune], curve=curve),
        day,cause,thresholds[cause]) for day in HORIZONS} for label,cause in (('readmission',1),('death',2))}
    # H6 的第一轴是归一化既往次数；按真实关系计数分层，避免猜测其尺度。
    counts = getattr(d, '_v7_prior_counts')[tune]
    strata = {}
    for label, selected in (('0',counts==0),('1',counts==1),('2_4',(counts>=2)&(counts<=4)),('5_plus',counts>=5)):
        if selected.any():
            strata[label] = dict(n=int(selected.sum()), annual=f1_at((d.events[tune][selected]==1)&(d.times[tune][selected]<=365),
                curve[selected,365,1], chosen['threshold']))
    result = dict(best_epoch=chosen['epoch'], completed_epochs=epoch, early=asdict(early),
        optimizer_steps=steps, stop_reason='patience_exhausted' if stop else 'max_epochs',
        checkpoint_sha256=sha256(folder/'best.pt'), last_sha256=sha256(folder/'last.pt'),
        predictions_sha256=sha256(folder/'tuning_predictions.npz'), metrics=metrics,
        multihorizon=multihorizon, prior_count_strata=strata,
        seconds=synchronized_time()-started, train_seconds=sum(r['train_seconds'] for r in history),
        evaluation_seconds=sum(r['evaluation_seconds'] for r in history),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(folder/'result.json', result)
    return result, curve


def run():
    if ROOT.exists():
        raise FileExistsError('V7 run already exists; preserving it, no silent rerun')
    all_started = time.perf_counter()
    d,c,ext,lab,relation,train,tune,baseline,drug,rad,contract,source,references = setup()
    ROOT.mkdir()
    write_json(ROOT/'contract.json',contract); write_json(ROOT/'source_report.json',source)
    # 明确训练 source 统计；不由调参/结局决定药物词表或尺度。
    write_json(ROOT/'input_coverage.json',dict(training_discharges=len(train),
        fine_drug_present_discharges=int(drug['med_present'][train].any(1).sum()),
        radiology_present_discharges=int(rad['present'][train].any(1).sum()),
        radiology_categories_counts=rad['present'][train].sum(0).tolist(),
        drug_names=drug['med_names'].tolist(), all_38_audited_drug_slots_used=True))
    object.__setattr__(d, '_v7_prior_counts', np.diff(relation['offsets']))
    highest = train[np.argsort(-np.diff(relation['offsets'])[train])[:16]]
    cross = train[rad['present'][train].any(1) & drug['med_present'][train].any(1)][:16]
    selected = np.unique(np.r_[highest,cross])
    b = make_v7_batch(c,ext,lab,relation,drug,rad,selected).to('cuda')
    write_json(ROOT/'preflight.json',preflight(baseline,drug,b,d,selected))
    del b
    torch.cuda.empty_cache()
    print(json.dumps(dict(status='preflight_passed',parameters=json.loads((ROOT/'preflight.json').read_text())['trainable_parameters'])),flush=True)
    cache_started = time.perf_counter()
    train_batches = cache(c,ext,lab,relation,drug,rad,train)
    tune_batches = cache(c,ext,lab,relation,drug,rad,tune)
    cache_seconds = time.perf_counter()-cache_started
    cache_bytes = sum(getattr(b,f.name).numel()*getattr(b,f.name).element_size() for _,b in train_batches+tune_batches for f in fields(b))
    write_json(ROOT/'cache_report.json',dict(seconds=cache_seconds,cpu_tensor_bytes=cache_bytes,
        train_batches=len(train_batches),tune_batches=len(tune_batches),sparse_text_cache=True,
        tensor_transfer_included_in_train_seconds=True))
    results, curves = {}, {}
    for arm in ARMS:
        results[arm],curves[arm] = fit(baseline,drug,arm,train_batches,tune_batches,d,tune)
        torch.cuda.empty_cache()
    reference_metrics = {name:parent.detail(d,tune,a['curve'],0.) for name,a in references.items()}
    for value in reference_metrics.values():
        value.pop('nll')  # 此处没有重建旧 NLL，不能用0冒充其真实结果。
    intervals = {}
    for arm in ARMS:
        for name, a in references.items():
            intervals[arm+'_minus_'+name] = paired_interval(d.arrays['subject_ids'][tune],d.times[tune],d.events[tune],
                curves[arm][:,:,1],a['curve'][:,:,1])
    intervals['recommended_minus_nll_control'] = paired_interval(d.arrays['subject_ids'][tune],d.times[tune],d.events[tune],
        curves['recommended'][:,:,1],curves['nll_control'][:,:,1])
    report = dict(results=results,reference_metrics=reference_metrics,paired_intervals=intervals,
        total_run_seconds=time.perf_counter()-all_started, source_setup_seconds=contract['source_setup_seconds'],
        cache_seconds=cache_seconds, limits=['single seed, exposed tuning fold','combination architecture/input experiment',
        'no confirmed external validation','frozen cached BERT; not BERT fine-tuning duration'],default_v6_replaced=False)
    write_json(ROOT/'report.json',report)
    print(json.dumps(dict(status='completed',seconds=report['total_run_seconds'])),flush=True)


def verify():
    started=time.perf_counter()
    report=json.loads((ROOT/'report.json').read_text());saved=json.loads((ROOT/'contract.json').read_text())
    d,c,ext,lab,relation,train,tune,baseline,drug,rad,current,source,_=setup()
    for path,digest in saved['source_sha256'].items():
        if sha256(Path(path))!=digest:raise ValueError('V7 source changed: '+path)
    # 仅时间字段因进程不同而不同；其余合同和全部父合同须相等。
    saved.pop('source_setup_seconds');current.pop('source_setup_seconds')
    if current!=saved:raise ValueError('V7 reconstruction contract differs')
    tune_batches=cache(c,ext,lab,relation,drug,rad,tune)
    results={}; maximum=0.
    for arm in ARMS:
        folder=ROOT/arm;info=report['results'][arm]
        for file,key in (('best.pt','checkpoint_sha256'),('last.pt','last_sha256'),('tuning_predictions.npz','predictions_sha256')):
            if sha256(folder/file)!=info[key]:raise ValueError('saved artifact changed')
        model=V7Model(baseline,drug['drug_parent']).cuda()
        chosen=torch.load(folder/'best.pt',map_location='cuda',weights_only=True);model.load_state_dict(chosen['state_dict'])
        for n,p in model.baseline.state_dict().items():
            torch.testing.assert_close(p.cpu(),baseline.state_dict()[n],rtol=0,atol=0)
        a,k,curve,nll=predict(model,tune_batches,d)
        with np.load(folder/'tuning_predictions.npz') as z:
            for name,value in (('hadm_ids',d.arrays['hadm_ids'][tune]),('subject_ids',d.arrays['subject_ids'][tune]),
                               ('times',d.times[tune]),('events',d.events[tune])):np.testing.assert_array_equal(z[name],value)
            errors={name:float(np.abs(z[name]-value).max()) for name,value in (('log_amplitude',a),('shape',k),('curve',curve))}
            from training.v6.token_model import daily_probabilities
            daily,conditional=daily_probabilities(curve)
            for name,value in (('daily_event_probability',daily),('conditional_daily_event_probability',conditional)):
                errors[name]=float(np.abs(z[name]-value).max())
        error=max(errors.values());maximum=max(maximum,error)
        if error!=0:raise ValueError('independent replay prediction differs: '+repr(errors))
        actual=parent.detail(d,tune,curve,nll,chosen['threshold'])
        if actual!=info['metrics']:raise ValueError('independent metrics differ')
        history=json.loads((folder/'history.json').read_text())
        selected=max(history,key=selection_key)
        if selected['epoch']!=chosen['epoch'] or chosen['epoch']!=info['best_epoch']:
            raise ValueError('checkpoint selection differs')
        early=EarlyStopping(6,8,1e-4)
        for i,row in enumerate(history):
            if row['epoch']!=i:raise ValueError('nonsequential epoch history')
            _,stop=early.observe(i,-row['annual']['f1'])
            if stop and i!=len(history)-1:raise ValueError('trained past early stop')
        if asdict(early)!=info['early'] or (not stop and history[-1]['epoch']!=MAX_EPOCHS):
            raise ValueError('early stop state differs')
        last=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        if last['epoch']!=info['completed_epochs'] or last['optimizer_steps']!=info['optimizer_steps']:
            raise ValueError('last checkpoint step/epoch differs')
        if np.max(np.abs(curve.sum(2)-1))>1e-10 or (np.diff(curve[:,:,1:],axis=1)<-1e-12).any():
            raise ValueError('curve conservation/monotonicity failed')
        results[arm]=dict(errors=errors,metrics_equal=True,selection_equal=True,early_stop_equal=True,h34_equal=True)
        del model;torch.cuda.empty_cache()
    proof=dict(maximum_error=maximum,arms=results,source_contract_equal=True,confirmation_or_test_used=False,
               report_sha256=sha256(ROOT/'report.json'),seconds=time.perf_counter()-started)
    write_json(ROOT/'verification.json',proof)
    print(json.dumps(proof),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('train','verify'),required=True)
    args=parser.parse_args()
    run() if args.mode=='train' else verify()
