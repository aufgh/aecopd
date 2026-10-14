"""固定原模型的训练/调参梯度对照；没有参数更新或确认集评价。"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.module_surgery_experiment import batches
from training.v6.warm_residual_audit import teacher
from training.v6.horizon_objective import restore
from training.v6.objective_direction_diagnostic import NAMES,cosine,compare,losses
from training.v6.train import write_json

ROOT=Path('training_runs/v6_generalization_direction_2026-10-13')
PROTOCOL=Path('docs/V6_GENERALIZATION_DIRECTION_PROTOCOL_2026-10-13.md')
SOURCE=('training/v6/generalization_direction.py','training/v6/curve_score.py',
        'training/v6/objective_direction_diagnostic.py','training/tests/test_v6_generalization_direction.py',str(PROTOCOL))


def parameter_group(name):
    if name.startswith('delta_head.'):return 'risk_head'
    if name.startswith('encoder.extra_labs.'):return 'additional_labs'
    if name.startswith('encoder.extra_meds.'):return 'inactive_fine_medications'
    prefix='encoder.base.'
    if not name.startswith(prefix):raise ValueError('unmapped trainable parameter '+name)
    sub=name.removeprefix(prefix)
    for paths,group in ((('current.','current_ids.','current_score.','branch_norms.0.'),'inactive_legacy_current'),
                        (('lab.','lab_ids.','lab_score.','branch_norms.1.'),'labs'),
                        (('med.','branch_norms.2.'),'medications'),
                        (('microbiology.','branch_norms.3.'),'microbiology'),
                        (('support.','branch_norms.4.'),'support'),
                        (('field_ids.','text_score.','text_projection.'),'text'),
                        (('fusion.',),'fusion')):
        if sub.startswith(paths):return group
    raise ValueError('unmapped clinical parameter '+name)


def gradient_direction(train,tuning):
    if train.shape!=tuning.shape or not np.isfinite(train).all() or not np.isfinite(tuning).all():
        raise ValueError('matched finite gradients required')
    return dict(cosine=cosine(train,tuning),dot=float(train.flatten()@tuning.flatten()),
                train_norm=float(np.linalg.norm(train)),tuning_norm=float(np.linalg.norm(tuning)))


def mean_gradient_sum(batch_sums,fold_count):
    if fold_count<=0:raise ValueError('positive full fold count required')
    return np.sum(batch_sums,axis=0)/fold_count


def verify_old():
    folder=Path('training_runs/v6_objective_network_2026-10-13')
    report=json.loads((folder/'diagnostic.json').read_text())
    proof=json.loads((folder/'independent_verification.json').read_text())
    if proof['report_sha256']!=sha256(folder/'diagnostic.json') or proof['maximum_error']>1e-10:
        raise ValueError('prior full-network diagnostic proof invalid')
    for p,digest in report['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('prior diagnostic dependency changed '+p)
    return report,sha256(folder/'diagnostic.json')


def measure(model,cache,count):
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if any(n.startswith('baseline.') for n,p in named):raise ValueError('H34 not frozen')
    groups=[parameter_group(n) for n,p in named]
    params=[p for n,p in named];totals={k:[torch.zeros_like(p,dtype=torch.float64) for p in params] for k in NAMES}
    values={k:0. for k in NAMES};audit=dict(current_active_records=0,fine_medication_nonzero_values=0,
        lab_observed_records=0,medication_source_records=0,microbiology_source_records=0,support_source_records=0,text_records=0)
    seen=0
    for _,history,batch,t,e in cache:
        detail=batch.base;compact=detail.base;clinical=compact.clinical
        audit['current_active_records']+=int(clinical.current_masks.any(1).sum())
        audit['fine_medication_nonzero_values']+=int(torch.count_nonzero(detail.medications))
        audit['lab_observed_records']+=int((clinical.lab_axis_inputs[:,:,6]>0).any(1).sum())
        audit['medication_source_records']+=int(compact.med_features[:,:,2:].bool().any(dim=(1,2)).sum())
        audit['microbiology_source_records']+=int(compact.micro_masks.any(1).sum())
        audit['support_source_records']+=int(compact.support_masks.any(1).sum())
        audit['text_records']+=int(torch.unique(clinical.snippet_index[:,0]).numel())
        out=model(history,batch);scored,_=losses(out,t,e)
        for j,name in enumerate(NAMES):
            values[name]+=float(scored[name].detach().sum())/count
            precision=torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32=False
                parts=torch.autograd.grad(scored[name].sum()/count,params,allow_unused=True,retain_graph=j<len(NAMES)-1)
            finally:torch.backends.cuda.matmul.allow_tf32=precision
            for total,part in zip(totals[name],parts):
                if part is not None:total.add_(part.double())
        seen+=len(t)
    if seen!=count or audit['current_active_records'] or audit['fine_medication_nonzero_values']:
        raise ValueError('fold coverage or inactive input contract differs')
    vectors={k:torch.cat([g.flatten() for g in v]).cpu().numpy() for k,v in totals.items()}
    if not all(np.isfinite(v).all() for v in vectors.values()):raise ValueError('nonfinite aggregate gradients')
    layout=[];offset=0;last_ids={id(p) for p in model.delta_head[-1].parameters()}
    for (name,p),group in zip(named,groups):
        layout.append(dict(name=name,group=group,start=offset,end=offset+p.numel(),last_projection=id(p) in last_ids))
        offset+=p.numel()
    return dict(losses=values,input_audit=audit),vectors,layout


def scopes(layout):
    length=layout[-1]['end'];groups={}
    predicates={'all_clinical':lambda r:True,'encoder':lambda r:r['group']!='risk_head',
                'risk_head':lambda r:r['group']=='risk_head','last_projection':lambda r:r['last_projection']}
    for group in sorted(set(r['group'] for r in layout)-{'risk_head'}):
        predicates[group]=lambda r,group=group:r['group']==group
    for name,predicate in predicates.items():
        mask=np.zeros(length,bool)
        for row in layout:
            if predicate(row):mask[row['start']:row['end']]=True
        groups[name]=mask
    return groups


def fold_stats(d,ix):
    return dict(records=len(ix),patients=len(np.unique(d.arrays['subject_ids'][ix])),
        events={str(e):int((d.events[ix]==e).sum()) for e in (0,1,2)},
        index_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),
        label_rows_sha256=hashlib.sha256(d.times[ix].tobytes()+d.events[ix].tobytes()).hexdigest())


def run():
    begin=time.perf_counter();data=checked();d,c,h,ext,_,bins,tune,_,search,_=data
    previous,previous_hash=verify_old();train=np.flatnonzero(d.arrays['patient_splits']=='train')
    if set(d.arrays['subject_ids'][train])&set(d.arrays['subject_ids'][tune]):raise ValueError('patient fold overlap')
    stats={fold:fold_stats(d,ix) for fold,ix in [('train',train),('tuning',tune)]}
    contract=dict(source_sha256={p:sha256(Path(p)) for p in SOURCE},prior_report_sha256=previous_hash,
        search_contract_sha256=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')),
        data_provenance=d.provenance,compact_provenance=c.provenance,label_sha256=d.provenance['label_sha256'],folds=stats,
        teacher_sha256={str(s):sha256(REFERENCE/f'seed{s}/control/best.pt') for s in (42,7)},
        baseline_sha256={str(s):sha256(Path(search['input_config']['parent'])/f'seed{s}/survival_base/best.pt') for s in (42,7)})
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'contract.json').exists():
        if json.loads((ROOT/'contract.json').read_text())!=contract:raise ValueError('generalization contract changed')
    else:write_json(ROOT/'contract.json',contract)
    cache={fold:batches(d,c,ext,ix) for fold,ix in [('train',train),('tuning',tune)]};results={};gradient_errors={}
    write_json(ROOT/'state.json',dict(status='running_diagnostic',model_fits=0,goal_achieved=False))
    for seed in (42,7):
        base=restore(Path(search['input_config']['parent']),seed,'survival_base',h,.1);model=teacher(base,seed,bins).eval()
        frozen={k:v.detach().clone() for k,v in model.state_dict().items()};vectors={};folds={};layout=None
        for fold in ('train','tuning'):
            folds[fold],vectors[fold],current=measure(model,cache[fold],stats[fold]['records'])
            if layout is not None and current!=layout:raise ValueError('parameter layouts differ')
            layout=current
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()):raise ValueError('original model mutated')
        masks=scopes(layout);measures={}
        for scope,mask in masks.items():
            tr={k:v[mask] for k,v in vectors['train'].items()};tu={k:v[mask] for k,v in vectors['tuning'].items()}
            measures[scope]=dict(parameters=int(mask.sum()),same_objective={k:gradient_direction(tr[k],tu[k]) for k in NAMES},
                cross_objective_cosines={a:{b:cosine(tr[a],tu[b]) for b in NAMES} for a in NAMES})
        old=previous['models'][str(seed)];prior_error=0.
        for name in NAMES:
            prior_error=max(prior_error,abs(folds['train']['losses'][name]-old['losses'][name]))
        for scope in ('all_clinical','encoder','risk_head','last_projection'):
            mask=masks[scope]
            for name in NAMES:
                prior_error=max(prior_error,abs(np.linalg.norm(vectors['train'][name][mask])-old['scopes'][scope]['gradient_norms'][name]))
                for other in NAMES:
                    prior_error=max(prior_error,abs(cosine(vectors['train'][name][mask],vectors['train'][other][mask])-
                                                   old['scopes'][scope]['cosines'][name][other]))
        if prior_error>1e-10:raise ValueError(f'prior training gradient reproduction differs: {prior_error}')
        artifact=ROOT/f'seed{seed}_aggregate_gradients.npz'
        flattened={fold+'_'+name:value for fold,values in vectors.items() for name,value in values.items()}
        if artifact.exists():
            with np.load(artifact) as z:
                if set(z.files)!=set(flattened):raise ValueError('saved aggregate keys differ')
                error=max(float(np.max(np.abs(z[k]-v))) for k,v in flattened.items())
            if error>1e-10:raise ValueError('aggregate gradient replay differs')
        else:np.savez_compressed(artifact,**flattened);error=None
        gradient_errors[str(seed)]=error
        results[str(seed)]=dict(folds=folds,scopes=measures,prior_training_max_error=float(prior_error),
                               parameter_layout=layout,model_weights_unchanged=True)
        print(json.dumps(dict(seed=seed,all_clinical=measures['all_clinical'],seconds=round(time.perf_counter()-begin,1))),flush=True)
        del model,vectors,frozen
    report=dict(models=results,folds=stats,contract_sha256=sha256(ROOT/'contract.json'),model_fits=0,goal_achieved=False,
        scope='Fixed original states; train/tuning gradients only; no optimization, validation-gradient updates or confirmation/test analysis.')
    target=ROOT/'diagnostic.json'
    if target.exists():report_error=compare(json.loads(target.read_text()),report)
    else:write_json(target,report);report_error=None
    complete=report_error is not None and all(v is not None for v in gradient_errors.values())
    if complete:write_json(ROOT/'independent_verification.json',dict(status='independently_replayed',
        aggregate_gradient_errors=gradient_errors,maximum_gradient_error=max(gradient_errors.values()),
        maximum_report_error=report_error,report_sha256=sha256(target),seconds=time.perf_counter()-begin))
    write_json(ROOT/'state.json',dict(status='diagnostic_complete_verified' if complete else 'saved_pending_independent_replay',
        model_fits=0,goal_achieved=False))


if __name__=='__main__':
    try:run()
    except Exception as error:
        ROOT.mkdir(exist_ok=True);write_json(ROOT/'failure.json',dict(type=type(error).__name__,error=str(error),goal_achieved=False));raise
