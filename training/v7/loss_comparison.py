"""损失名称直接写组成；只改变原NLL/对比是否进入训练总损失。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from patient_data.export_mimiciv import sha256
from training.v6 import unified_longitudinal_experiment as parent
from training.v6.early_stopping import EarlyStopping
from training.v6.evaluation import paired_interval
from training.v6.train import write_json
from training.v6.multihorizon_readout import horizon_readout
from training.v6.curve_score import daily_cif
from training.v6.risk import competing_nll
from training.v7 import experiment as first
from training.v7.data import cache, make_v7_batch
from training.v7.losses import HORIZONS, soft_focal, contrastive
from training.v7.model import V7Model

ROOT=Path('training_runs/v7_loss_comparison_2026-10-15')
PROTOCOL=Path('docs/V7_LOSS_COMPARISON_PRD_2026-10-15.md')
CONDITIONS={
    'focal_smoothing_contrast':dict(label='Focal＋标签平滑＋对比',nll=0.,contrast=.1),
    'focal_smoothing':dict(label='Focal＋标签平滑',nll=0.,contrast=0.),
    'nll_focal_smoothing':dict(label='原竞争风险损失＋Focal＋标签平滑',nll=1.,contrast=0.),
}
OLD_LABELS={'nll_control':'原竞争风险损失',
            'recommended':'原竞争风险损失＋Focal＋标签平滑＋对比'}


def loss_terms(out,times,events,reps,masks,subjects,*,nll_weight,contrast_weight):
    if nll_weight not in (0.,1.) or contrast_weight not in (0.,.1):
        raise ValueError('unregistered loss coefficient')
    nll=competing_nll(out,times,events)
    curve=daily_cif(out)
    days=torch.as_tensor(HORIZONS,device=times.device)
    causes=torch.tensor([1,2],device=times.device)
    target=(events[:,None,None]==causes[None,None,:]) & (times[:,None,None]<=days[None,:,None])
    focal=soft_focal(curve[:,days,1:],target)
    penalty=1e-4*out.delta.square().mean()
    contrast=nll.detach()*0
    pairs=dict(positive_pairs=0,negative_pairs=0)
    if contrast_weight:
        contrast,pairs=contrastive(reps,masks,subjects)
    # 无NLL条件真的从优化图中省略该项；日志中的NLL不代表加入总损失。
    total=focal+penalty
    if nll_weight:
        total=total+nll
    if contrast_weight:
        total=total+contrast_weight*contrast
    return total,dict(nll=nll,delta_penalty=penalty,focal=focal,contrast=contrast,**pairs)


def setup():
    data=first.setup()
    old_saved=json.loads((first.ROOT/'contract.json').read_text())
    old_current=dict(data[-3])
    for path,digest in old_saved['source_sha256'].items():
        if sha256(Path(path))!=digest:raise ValueError('first V7 source changed: '+path)
    old_saved.pop('source_setup_seconds');old_current.pop('source_setup_seconds')
    if old_current!=old_saved:raise ValueError('first V7 source reconstruction differs')
    proof=json.loads((first.ROOT/'verification.json').read_text())
    if proof['maximum_error']!=0 or proof['report_sha256']!=sha256(first.ROOT/'report.json'):
        raise ValueError('first V7 proof changed')
    paths=[Path(__file__),Path('training/tests/test_v7_loss_comparison.py'),PROTOCOL,
           first.ROOT/'contract.json',first.ROOT/'report.json',first.ROOT/'verification.json',
           Path('C:/Users/Administrator/Desktop/研究 2.md')]
    # Windows原文通过额外只读挂载到容器；不依赖从外部重新取得文档。
    research=Path('/research/research_2.md')
    paths[-1]=research
    for arm in first.ARMS:
        paths.extend((first.ROOT/arm/'best.pt',first.ROOT/arm/'last.pt',first.ROOT/arm/'tuning_predictions.npz'))
    contract=dict(first_contract=old_saved,source_sha256={str(p):sha256(p) for p in paths},
        conditions=CONDITIONS,seed=42,batch_size=64,early=dict(max_epochs=30,min_epochs=8,patience=6,min_delta=1e-4),
        focal=dict(gamma=2.,alpha=.75,smoothing=.05),contrast_margin=.5,
        delta_penalty=1e-4,model_and_inputs_changed=False,nll_diagnostic_not_necessarily_optimized=True,
        original_loss_recommendation_incomplete=True,contrast_coefficient_is_implementation_choice=True,
        confirmation_or_test_used=False,automation_resumed=False)
    return data,contract


def preflight(baseline,drug,b,d,positions):
    report={}
    for arm,cfg in CONDITIONS.items():
        model=V7Model(baseline,drug['drug_parent']).cuda()
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        groups,_=model.optimizer_groups();opt=torch.optim.AdamW(groups,weight_decay=.01)
        frozen={n:p.detach().clone() for n,p in model.baseline.named_parameters()}
        before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        for _ in range(3):
            model.train();opt.zero_grad(set_to_none=True)
            out,reps,masks=model(b,auxiliary=True)
            total,parts=loss_terms(out,torch.as_tensor(d.times[positions],device='cuda'),
                torch.as_tensor(d.events[positions],device='cuda'),reps,masks,
                torch.as_tensor(d.arrays['subject_ids'][positions],device='cuda'),
                nll_weight=cfg['nll'],contrast_weight=cfg['contrast'])
            if not torch.isfinite(total):raise ValueError('nonfinite preflight loss')
            total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite preflight gradient')
            opt.step()
        if any(p.grad is not None or not torch.equal(p,frozen[n]) for n,p in model.baseline.named_parameters()):
            raise ValueError('frozen baseline changed')
        required=('delta_head.','lab_projection.','text_projection.','rad_projection.','drug_numeric.',
                  'temporal.','history_pool.','drug_cross.','rad_cross.')
        changed={prefix:any(n.startswith(prefix) and not torch.equal(p,before[n]) for n,p in model.named_parameters() if n in before) for prefix in required}
        if not all(changed.values()):raise ValueError('no-NLL path not trained: '+repr(changed))
        disconnected=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None
                      and not (cfg['contrast']==0 and n.startswith('contrast_projection.'))]
        if disconnected:raise ValueError('unexplained disconnected parameters: '+repr(disconnected))
        report[arm]=dict(label=cfg['label'],nll_optimized=bool(cfg['nll']),contrast_optimized=bool(cfg['contrast']),
            actual_updates=changed,h34_frozen=True,unexplained_disconnected_parameters=disconnected,
            loss_parts={k:float(v.detach()) if torch.is_tensor(v) else v for k,v in parts.items()})
        del model,opt;torch.cuda.empty_cache()
    # 同一初值、同一批次测各项更新信号；不能用损失值大小代替。
    model=V7Model(baseline,drug['drug_parent']).cuda().eval()
    out,reps,masks=model(b,auxiliary=True)
    _,parts=loss_terms(out,torch.as_tensor(d.times[positions],device='cuda'),
        torch.as_tensor(d.events[positions],device='cuda'),reps,masks,
        torch.as_tensor(d.arrays['subject_ids'][positions],device='cuda'),nll_weight=1.,contrast_weight=.1)
    params=[p for p in model.parameters() if p.requires_grad]
    gradients={}
    for term in ('nll','focal','contrast'):
        g=torch.autograd.grad(parts[term],params,retain_graph=True,allow_unused=True)
        gradients[term]=torch.cat([v.detach().flatten() if v is not None else torch.zeros_like(p).flatten() for v,p in zip(g,params,strict=True)])
    def cosine(a,b):
        return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30))
    report['initial_gradient_diagnostic']=dict(
        raw_norms={n:float(v.norm()) for n,v in gradients.items()},
        optimized_contrast_norm=float(gradients['contrast'].norm()*.1),
        nll_focal_cosine=cosine(gradients['nll'],gradients['focal']),
        note='common initial zero delta head; contrast updates source heads, risk losses initially update output layer; diagnostic only')
    return report


def save_metrics(folder,d,tune,a,k,curve,nll,threshold,filename):
    parent.save(folder/filename,d,tune,a,k,curve)
    m=parent.detail(d,tune,curve,nll,threshold)
    multihorizon={label:{str(day):horizon_readout(dict(times=d.times[tune],events=d.events[tune],curve=curve),
        day,cause,threshold if cause==1 else .2) for day in HORIZONS} for label,cause in (('readmission',1),('death',2))}
    return m,multihorizon


def fit(arm,cfg,baseline,drug,tr,va,d,tune):
    folder=ROOT/arm;folder.mkdir()
    model=V7Model(baseline,drug['drug_parent']).cuda()
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);rng=np.random.default_rng(42)
    groups,_=model.optimizer_groups();opt=torch.optim.AdamW(groups,weight_decay=.01)
    initial={n:p.detach().cpu().clone() for n,p in model.baseline.named_parameters()}
    early=EarlyStopping(6,8,1e-4);history=[];best=None;steps=0
    started=first.synchronized_time();torch.cuda.reset_peak_memory_stats()
    for epoch in range(31):
        online=None;train_seconds=0.
        if epoch:
            model.train();sums={k:0. for k in ('total','nll','delta_penalty','focal','contrast')};count=0
            pair_counts=dict(positive_pairs=0,negative_pairs=0);norms=[]
            t0=first.synchronized_time()
            for j in rng.permutation(len(tr)):
                p,cpu=tr[j];b=cpu.to('cuda');opt.zero_grad(set_to_none=True)
                out,reps,masks=model(b,auxiliary=True)
                total,parts=loss_terms(out,torch.as_tensor(d.times[p],device='cuda'),torch.as_tensor(d.events[p],device='cuda'),
                    reps,masks,torch.as_tensor(d.arrays['subject_ids'][p],device='cuda'),nll_weight=cfg['nll'],contrast_weight=cfg['contrast'])
                if not torch.isfinite(total):raise ValueError('nonfinite loss')
                total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
                opt.step();steps+=1;count+=len(p);norms.append(float(norm))
                sums['total']+=float(total.detach())*len(p)
                for k in ('nll','delta_penalty','focal','contrast'):sums[k]+=float(parts[k].detach())*len(p)
                for k in pair_counts:pair_counts[k]+=parts[k]
            train_seconds=first.synchronized_time()-t0
            online=dict(**{k:v/count for k,v in sums.items()},**pair_counts,max_gradient_norm_before_clip=max(norms))
        t0=first.synchronized_time();a,k,curve,nll=first.predict(model,va,d);m=parent.detail(d,tune,curve,nll)
        row=dict(epoch=epoch,online=online,annual=m['annual'],tuning_nll=nll,
            tuning_ibs=m['readmission']['integrated_brier_7_365'],tuning_death_ibs=m['death']['integrated_brier_7_365'],
            train_seconds=train_seconds,evaluation_seconds=first.synchronized_time()-t0)
        history.append(row);_,stop=early.observe(epoch,-row['annual']['f1'])
        key=first.selection_key(row)
        if best is None or key>best:
            best=key;torch.save(dict(state_dict=model.state_dict(),epoch=epoch,threshold=row['annual']['threshold']),folder/'best.pt')
        torch.save(dict(state_dict=model.state_dict(),epoch=epoch,optimizer_steps=steps),folder/'last.pt')
        write_json(folder/'history.json',history)
        print(json.dumps(dict(loss=cfg['label'],epoch=epoch,f1=row['annual']['f1'],ibs=row['tuning_ibs'],
            train_seconds=train_seconds),ensure_ascii=False),flush=True)
        if stop:break
    if any(not torch.equal(p.detach().cpu(),initial[n]) for n,p in model.baseline.named_parameters()):raise ValueError('H34 changed')
    last_metrics,last_multihorizon=save_metrics(folder,d,tune,a,k,curve,nll,row['annual']['threshold'],'last_predictions.npz')
    chosen=torch.load(folder/'best.pt',map_location='cuda',weights_only=True);model.load_state_dict(chosen['state_dict'])
    a,k,curve,nll=first.predict(model,va,d)
    metrics,multi=save_metrics(folder,d,tune,a,k,curve,nll,chosen['threshold'],'tuning_predictions.npz')
    result=dict(label=cfg['label'],coefficients=cfg,best_epoch=chosen['epoch'],completed_epochs=epoch,
        optimizer_steps=steps,early=asdict(early),stop_reason='patience_exhausted' if stop else 'max_epochs',
        metrics=metrics,multihorizon=multi,last_metrics=last_metrics,last_multihorizon=last_multihorizon,
        artifact_sha256={name:sha256(folder/name) for name in ('best.pt','last.pt','tuning_predictions.npz','last_predictions.npz','history.json')},
        seconds=first.synchronized_time()-started,train_seconds=sum(r['train_seconds'] for r in history),
        evaluation_seconds=sum(r['evaluation_seconds'] for r in history),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(folder/'result.json',result)
    return result,curve


def run():
    if ROOT.exists():raise FileExistsError('preserve existing loss comparison')
    started=time.perf_counter();data,contract=setup()
    d,c,ext,lab,rel,train,tune,baseline,drug,rad,_,source,references=data
    ROOT.mkdir();write_json(ROOT/'contract.json',contract);write_json(ROOT/'source_report.json',source)
    highest=train[np.argsort(-np.diff(rel['offsets'])[train])[:16]]
    cross=train[rad['present'][train].any(1)&drug['med_present'][train].any(1)][:16]
    selected=np.unique(np.r_[highest,cross]);b=make_v7_batch(c,ext,lab,rel,drug,rad,selected).to('cuda')
    write_json(ROOT/'preflight.json',preflight(baseline,drug,b,d,selected));del b;torch.cuda.empty_cache()
    tr=cache(c,ext,lab,rel,drug,rad,train);va=cache(c,ext,lab,rel,drug,rad,tune)
    # epoch0必须与上一轮各条件保存记录逐项相同。
    initial=V7Model(baseline,drug['drug_parent']).cuda().eval()
    _,_,_,nll=first.predict(initial,va,d);del initial;torch.cuda.empty_cache()
    results={};curves={}
    for arm,cfg in CONDITIONS.items():
        results[arm],curves[arm]=fit(arm,cfg,baseline,drug,tr,va,d,tune);torch.cuda.empty_cache()
        original=json.loads((first.ROOT/'nll_control/history.json').read_text())[0]
        incoming=json.loads((ROOT/arm/'history.json').read_text())[0]
        for k in ('annual','tuning_nll','tuning_ibs','tuning_death_ibs'):
            if incoming[k]!=original[k]:raise ValueError('initial risk/input differs from first experiment')
    old_report=json.loads((first.ROOT/'report.json').read_text())
    for arm in first.ARMS:
        with np.load(first.ROOT/arm/'tuning_predictions.npz') as z:curves[arm]=z['curve']
    comparisons={}
    pairs=(('focal_smoothing_contrast','recommended'),('focal_smoothing_contrast','focal_smoothing'),
        ('nll_focal_smoothing','recommended'),('nll_focal_smoothing','focal_smoothing'))
    for left,right in pairs:
        comparisons[left+'_minus_'+right]=paired_interval(d.arrays['subject_ids'][tune],d.times[tune],d.events[tune],
            curves[left][:,:,1],curves[right][:,:,1])
    report=dict(results=results,reused_results={k:dict(label=OLD_LABELS[k],**old_report['results'][k]) for k in first.ARMS},
        original_v6_metrics=old_report['reference_metrics'],paired_intervals=comparisons,
        seconds=time.perf_counter()-started,initial_metrics_equal_to_previous=True,
        limits='single seed; exposed tuning fold; threshold/epoch selected there; no independent validation; fixed coefficient .1',
        confirmation_or_test_used=False,automation_resumed=False)
    write_json(ROOT/'report.json',report);print(json.dumps(dict(status='completed',seconds=report['seconds'])),flush=True)


def verify():
    data,contract=setup();saved=json.loads((ROOT/'contract.json').read_text())
    if contract!=saved:raise ValueError('loss contract reconstruction differs')
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('loss source changed: '+p)
    d,c,ext,lab,rel,train,tune,baseline,drug,rad,_,_,_=data
    va=cache(c,ext,lab,rel,drug,rad,tune);report=json.loads((ROOT/'report.json').read_text());results={}
    for arm in CONDITIONS:
        folder=ROOT/arm;info=report['results'][arm]
        for name,digest in info['artifact_sha256'].items():
            if sha256(folder/name)!=digest:raise ValueError('loss artifact changed')
        model=V7Model(baseline,drug['drug_parent']).cuda()
        errors={}
        for which in ('best','last'):
            ck=torch.load(folder/(which+'.pt'),map_location='cuda',weights_only=True);model.load_state_dict(ck['state_dict'])
            for n,p in model.baseline.state_dict().items():torch.testing.assert_close(p.cpu(),baseline.state_dict()[n],rtol=0,atol=0)
            a,k,curve,nll=first.predict(model,va,d)
            name='tuning_predictions.npz' if which=='best' else 'last_predictions.npz'
            with np.load(folder/name) as z:
                for field,value in (('hadm_ids',d.arrays['hadm_ids'][tune]),('subject_ids',d.arrays['subject_ids'][tune]),
                                    ('times',d.times[tune]),('events',d.events[tune])):np.testing.assert_array_equal(z[field],value)
                difference={field:float(np.abs(z[field]-value).max()) for field,value in (('log_amplitude',a),('shape',k),('curve',curve))}
                from training.v6.token_model import daily_probabilities
                daily,conditional=daily_probabilities(curve)
                for field,value in (('daily_event_probability',daily),('conditional_daily_event_probability',conditional)):
                    difference[field]=float(np.abs(z[field]-value).max())
            if max(difference.values())!=0:raise ValueError('loss replay differs')
            if np.max(np.abs(curve.sum(2)-1))>1e-10 or (np.diff(curve[:,:,1:],axis=1)<-1e-12).any():raise ValueError('invalid curve')
            m=parent.detail(d,tune,curve,nll,ck['threshold'] if which=='best' else None)
            if m!=info['metrics' if which=='best' else 'last_metrics']:raise ValueError('loss metrics differ')
            errors[which]=difference
        hist=json.loads((folder/'history.json').read_text());best=max(hist,key=first.selection_key)
        if best['epoch']!=info['best_epoch']:raise ValueError('loss selection differs')
        early=EarlyStopping(6,8,1e-4)
        for i,row in enumerate(hist):
            if row['epoch']!=i:raise ValueError('loss epoch sequence differs')
            _,stop=early.observe(i,-row['annual']['f1'])
            if stop and i!=len(hist)-1:raise ValueError('trained after early stop')
        if asdict(early)!=info['early'] or (not stop and hist[-1]['epoch']!=30):raise ValueError('loss early stop differs')
        if ck['epoch']!=info['completed_epochs'] or ck['optimizer_steps']!=info['optimizer_steps']:raise ValueError('last counter differs')
        results[arm]=dict(label=CONDITIONS[arm]['label'],errors=errors,metrics_equal=True,early_equal=True,h34_frozen=True)
        del model;torch.cuda.empty_cache()
    proof=dict(maximum_error=0,results=results,source_contract_equal=True,report_sha256=sha256(ROOT/'report.json'),
               confirmation_or_test_used=False)
    write_json(ROOT/'verification.json',proof);print(json.dumps(proof,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('train','verify'),required=True)
    args=parser.parse_args();run() if args.mode=='train' else verify()
