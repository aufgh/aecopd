"""一年F1选点的单seed四条件；当前/过去原输入共同训练，原正式模型不改。"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch

from patient_data.export_mimiciv import rows,sha256
from patient_data.unified_lab_state import ROOT as LAB_ROOT
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.module_surgery import SurgeryModel,probability_curve
from training.v6.module_surgery_experiment import summary,save
from training.v6.horizon_objective import restore
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import competing_nll
from training.v6.train import write_json
from training.v6.unified_longitudinal import ARMS,LongitudinalModel,make_batch,best_f1,f1_at

ROOT=Path('training_runs/v6_unified_longitudinal_f1_2026-10-14_r2')
PROTOCOL=Path('docs/V6_UNIFIED_LONGITUDINAL_F1_PRD_2026-10-14.md')
INDEX=Path('training_runs/v6_longitudinal_feasibility_2026-10-13')
SECTIONS=Path('outputs/01a08b96-2988-7001-af1d-b40d901f3400/private/v5_section_aligned/dataset_v4/note_sections.csv.gz')
MAX_EPOCHS=45


def seconds(value):
    return int(np.datetime64(datetime.fromisoformat(value),'s').astype(np.int64))


def setup():
    d,c,h,ext,base,bins,tune,confirm,search,_=checked()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)
    manifest=json.loads((LAB_ROOT/'manifest.json').read_text())
    if sha256(LAB_ROOT/'interface.npz')!=manifest['output_sha256']:raise ValueError('unified interface changed')
    for p,digest in manifest['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('unified source changed')
    with np.load(LAB_ROOT/'interface.npz') as z:lab={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits','values','eos_context')}
    for k in ('hadm_ids','subject_ids','patient_splits'):np.testing.assert_array_equal(lab[k],d.arrays[k])
    proof=json.loads((INDEX/'feasibility.json').read_text()); index_path=INDEX/'past_clinical_indices.npz'
    if sha256(index_path)!=proof['input_sha256'] or proof['strict_labels_sha256']!=d.provenance['label_sha256']:
        raise ValueError('historical index differs')
    with np.load(index_path) as z:
        for k in ('hadm_ids','subject_ids','patient_splits'):np.testing.assert_array_equal(z[k],d.arrays[k])
        neighbors,offsets,gaps=(z[k] for k in ('previous_positions','offsets','days_from_previous_discharge_to_current_admission'))
    reference_cfg=json.loads((Path(search['input_config']['reference'])/'config.json').read_text())
    snapshot=Path(reference_cfg['snapshot']); field_dir=Path(reference_cfg['field_dir'])
    # 使用B实际section引用绑定选中文书，不以所有小结的最大存档时间猜本次文书。
    section_note={r['section_id']:(r['hadm_id'],r['note_id']) for r in rows(SECTIONS)}
    by_id={str(v):i for i,v in enumerate(d.arrays['hadm_ids'])}; used={i:set() for i in range(len(by_id))}
    for r in rows(field_dir/'shared_text.csv.gz'):
        if r['section_id'] not in section_note:raise ValueError('B section not bound to saved note')
        hadm,note=section_note[r['section_id']]
        if hadm!=r['hadm_id']:raise ValueError('B note patient/admission mapping differs')
        used[by_id[hadm]].add(note)
    note_store={}
    for r in rows(snapshot/'raw/discharge.csv.gz'):
        try: note_store[r['note_id']]=(r['hadm_id'],seconds(r['storetime']))
        except ValueError:note_store[r['note_id']]=(r['hadm_id'],np.iinfo(np.int64).max)
    note_times=np.full(len(by_id),np.iinfo(np.int64).max,np.int64)
    for i,notes in used.items():
        if notes:
            if any(note not in note_store or note_store[note][0]!=str(d.arrays['hadm_ids'][i]) for note in notes):
                raise ValueError('selected B note absent in controlled source')
            note_times[i]=max(note_store[note][1] for note in notes)
    dates={r['hadm_id']:(seconds(r['admittime']),seconds(r['dischtime'])) for r in rows(snapshot/'cohort/admissions.csv.gz')}
    admit=np.array([dates[str(h)][0] for h in d.arrays['hadm_ids']],np.int64)
    discharge=np.array([dates[str(h)][1] for h in d.arrays['hadm_ids']],np.int64)
    for p in range(len(by_id)):
        past=neighbors[offsets[p]:offsets[p+1]]
        if (np.any(past==p) or np.any(d.arrays['subject_ids'][past]!=d.arrays['subject_ids'][p])
                or np.any(d.arrays['patient_splits'][past]!=d.arrays['patient_splits'][p])
                or np.any(discharge[past]>admit[p]) or np.any(admit[past]>=admit[p])):
            raise ValueError('future/self/cross-patient historical visit')
    train=np.flatnonzero(d.arrays['patient_splits']=='train')
    base=restore(Path(search['input_config']['parent']),42,'survival_base',h,.1)
    fresh=SurgeryModel(base,42,'control',bins).cuda()
    teacher=SurgeryModel(base,42,'control',bins).cuda()
    teacher.load_state_dict(torch.load(REFERENCE/'seed42/control/best.pt',map_location='cuda',weights_only=True)['state_dict'])
    sources=dict(manifest['source_sha256'])
    for p in (Path(__file__),Path('training/v6/unified_longitudinal.py'),Path('patient_data/unified_lab_state.py'),
              PROTOCOL,Path('training/tests/test_v6_unified_longitudinal.py'),LAB_ROOT/'interface.npz',LAB_ROOT/'manifest.json',
              index_path,INDEX/'feasibility.json',SECTIONS,field_dir/'shared_text.csv.gz',snapshot/'raw/discharge.csv.gz',
              snapshot/'cohort/admissions.csv.gz',REFERENCE/'seed42/control/best.pt'):
        sources[str(p)]=sha256(p)
    contract=dict(source_sha256=sources,data_provenance=d.provenance,compact_provenance=c.provenance,
        labels_sha256=d.provenance['label_sha256'],frozen_base_sha256=sha256(Path(search['input_config']['parent'])/'seed42/survival_base/best.pt'),
        arms=list(ARMS),seed=42,horizon=365,selection='maximum tuning annual F1; tie lower readmission IBS; tie earlier epoch',
        threshold_selection='all unique tuning scores, equal scores grouped, F1 tie highest threshold',
        optimizer=dict(name='AdamW',lr=3e-4,weight_decay=1e-4,clip=1.,delta_penalty=1e-4),
        early=dict(min_epochs=10,patience=7,min_delta=1e-4,max_epochs=45),batch_size=128,
        original_clinical_weights_not_reused=True,history_encoder_trainable=True,no_history_truncation=True,
        no_high_frequency=True,bert_backbone_frozen=True,goal_automation_resumed=False,historical_test_used=False)
    relation=dict(neighbors=neighbors,offsets=offsets,gaps=gaps,note_times=note_times,admit=admit)
    context_counts=np.diff(offsets)[train]
    text_relations=sum(int((note_times[neighbors[offsets[p]:offsets[p+1]]]<=admit[p]).sum()) for p in train)
    source_report=dict(training_discharges=len(train),training_patients=len(np.unique(d.arrays['subject_ids'][train])),
        visits_with_history=int((context_counts>0).sum()),maximum_prior_visits=int(context_counts.max()),
        prior_visit_relations=int(context_counts.sum()),text_visible_prior_relations=text_relations,
        past_text_excluded_relations=int(context_counts.sum())-text_relations,
        text_visibility='all source notes of this B visit stored by current admission; unknown excludes historical text only')
    return d,c,ext,lab,relation,train,tune,confirm,fresh,teacher,contract,source_report


def cache(c,ext,lab,relation,indices,history):
    result=[]
    for start in range(0,len(indices),128):
        p=indices[start:start+128]
        b=make_batch(c,ext,lab,p,relation['neighbors'],relation['offsets'],relation['gaps'],
                     relation['note_times'],relation['admit'],history=history)
        result.append((p,b))
    return result


@torch.no_grad()
def predict(model,cache,d):
    model.eval();a=[];k=[];nll=0.;n=0
    for p,b in cache:
        out=model(b);a.append(out.log_amplitude.cpu().numpy());k.append(out.shape.cpu().numpy())
        nll+=float(competing_nll(out,torch.as_tensor(d.times[p],device='cuda'),torch.as_tensor(d.events[p],device='cuda')))*len(p);n+=len(p)
    a,k=np.concatenate(a),np.concatenate(k)
    return a,k,probability_curve(a,k),nll/n


def detail(d,indices,curve,nll,threshold=None):
    y=(d.events[indices]==1)&(d.times[indices]<=365)
    decision=best_f1(y,curve[:,365,1]) if threshold is None else f1_at(y,curve[:,365,1],threshold)
    return dict(annual=decision,**summary(curve,d,indices,nll))


def original_predict(teacher,d,c,ext,indices):
    """旧参照保留其实际TF32/256批路径；不拿新FP32路径冒充旧保存概率。"""
    from training.v6.module_surgery_experiment import batches,parameters
    old=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=True
        a,k,nll=parameters(teacher,batches(d,c,ext,indices))
        return a,k,probability_curve(a,k),nll
    finally:torch.backends.cuda.matmul.allow_tf32=old


class OriginalView(torch.nn.Module):
    def __init__(self,teacher):super().__init__();self.teacher=teacher
    def forward(self,b):
        from training.v6.detail_extension_experiment import DetailBatch
        from training.v6.model import AnchoredModel
        compact,legacy,_,_,context,cur,_,_,_=b
        if not torch.equal(cur,torch.arange(len(cur),device=cur.device)):raise ValueError('original view only current batch')
        return AnchoredModel.forward(self.teacher,context,DetailBatch(compact,legacy,torch.zeros(len(cur),192,device='cuda')))


def preflight(fresh,relation,d,c,ext,lab,train):
    counts=np.diff(relation['offsets'])[train]
    selected=train[np.argsort(-counts)[:16]]
    # 相同当前预测点，不同条件共享基础风险及风险头；检验维度改动不会漂移初值。
    incoming={flag:cache(c,ext,lab,relation,selected,flag)[0][1] for flag in (False,True)}
    initial={};updates={}
    for arm in ARMS:
        model=LongitudinalModel(fresh,arm).cuda().eval()
        b=incoming[model.use_history]
        out=model(b);initial[arm]=(out.log_amplitude.detach(),out.shape.detach())
        for a,k in initial.values():
            torch.testing.assert_close(out.log_amplitude,a,rtol=0,atol=0);torch.testing.assert_close(out.shape,k,rtol=0,atol=0)
        before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        frozen={n:p.detach().clone() for n,p in model.original.baseline.named_parameters()}
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss=competing_nll(model(b),torch.as_tensor(d.times[selected],device='cuda'),torch.as_tensor(d.events[selected],device='cuda'))
            loss.backward();opt.step()
        if any(p.grad is not None or not torch.equal(p,frozen[n]) for n,p in model.original.baseline.named_parameters()):
            raise ValueError('H34 changed/received gradient')
        changed={group:any(not torch.equal(p,before[n]) for n,p in model.named_parameters() if n in before and n.startswith(group))
                 for group in ('original.encoder.base.lab.','original.encoder.base.text_projection.',
                               'original.delta_head.','lab_ids.','temporal.','history_head.')}
        required=['original.encoder.base.lab.','original.encoder.base.text_projection.','original.delta_head.']
        if model.unified:required.append('lab_ids.')
        if model.use_history:required+=['temporal.','history_head.']
        if not all(changed[g] for g in required):raise ValueError('effective shared/history layer not updated: '+repr(changed))
        sensitivity=0.;past_gradient=0.
        if model.use_history:
            model.eval();original=model(b).log_amplitude.detach()
            current=b[5];past_only=torch.ones(len(b[4]),dtype=torch.bool,device='cuda');past_only[current]=False
            labs=b[2].detach().clone().requires_grad_(True);legacy=b[1].detach().clone().requires_grad_(True)
            active=list(b);active[2]=labs;active[1]=legacy
            model.zero_grad(set_to_none=True)
            model(tuple(active)).log_amplitude.sum().backward()
            # 共用原核心检验也可读：统一条件用新lab梯度，原条件验证旧追加检验梯度。
            gradient=labs.grad if model.unified else legacy.grad
            if gradient is not None:past_gradient=float(gradient[past_only].abs().sum())
            without=list(b);without[7]=torch.zeros_like(b[7])
            sensitivity=float((model(tuple(without)).log_amplitude-original).abs().max().detach())
            if past_gradient<=0 or sensitivity<=0:raise ValueError('history content not differentiably consumed')
        updates[arm]=dict(actual_updates=changed,history_input_gradient_l1=past_gradient,
                          history_removal_risk_parameter_difference=sensitivity,h34_frozen=True)
        del model;torch.cuda.empty_cache()
    return dict(common_initial_risk_parameter_error=0.,conditions=updates)


def fit(fresh,arm,tr,va,d,tune,folder):
    folder.mkdir();model=LongitudinalModel(fresh,arm).cuda();torch.manual_seed(42);rng=np.random.default_rng(42)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4)
    early=EarlyStopping(7,10,1e-4);best=(-np.inf,np.inf);rows_=[];reason='max_epochs';steps=0;started=time.perf_counter()
    frozen={n:p.detach().clone() for n,p in model.original.baseline.named_parameters()}
    for epoch in range(MAX_EPOCHS+1):
        train_nll=None
        if epoch:
            model.train();total=0.;count=0
            for j in rng.permutation(len(tr)):
                p,b=tr[j];optimizer.zero_grad(set_to_none=True)
                out=model(b);nll=competing_nll(out,torch.as_tensor(d.times[p],device='cuda'),torch.as_tensor(d.events[p],device='cuda'))
                loss=nll+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                if not torch.isfinite(norm):raise ValueError('nonfinite gradients')
                optimizer.step();steps+=1;total+=float(nll.detach())*len(p);count+=len(p)
            train_nll=total/count
        a,k,curve,nll=predict(model,va,d);m=detail(d,tune,curve,nll)
        f1=m['annual']['f1'];ibs=m['readmission']['integrated_brier_7_365']
        candidate=f1>best[0]+1e-12 or (abs(f1-best[0])<=1e-12 and ibs<best[1]-1e-12)
        _,stop=early.observe(epoch,-f1)
        rows_.append(dict(epoch=epoch,train_nll=train_nll,annual=m['annual'],tuning_ibs=ibs,
                          tuning_death_ibs=m['death']['integrated_brier_7_365']))
        if candidate:
            best=(f1,ibs)
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,threshold=m['annual']['threshold']),folder/'best.pt')
        write_json(folder/'history.json',rows_)
        if epoch%5==0:print(json.dumps(dict(arm=arm,epoch=epoch,f1=f1,threshold=m['annual']['threshold'],ibs=ibs,seconds=round(time.perf_counter()-started,1))),flush=True)
        if stop:reason='patience_exhausted';break
    if any(not torch.equal(p,frozen[n]) for n,p in model.original.baseline.named_parameters()):raise ValueError('H34 changed')
    torch.save(dict(state_dict=model.state_dict(),epoch=epoch),folder/'last.pt')
    chosen=torch.load(folder/'best.pt',map_location='cuda',weights_only=True);model.load_state_dict(chosen['state_dict'])
    a,k,curve,nll=predict(model,va,d);m=detail(d,tune,curve,nll,chosen['threshold'])
    save(folder/'tuning_predictions.npz',d,tune,a,k,curve)
    info=dict(best_epoch=chosen['epoch'],completed_epochs=epoch,stop_reason=reason,early=asdict(early),optimizer_steps=steps,
              checkpoint_sha256=sha256(folder/'best.pt'),last_sha256=sha256(folder/'last.pt'),metrics=m,
              trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),seconds=time.perf_counter()-started)
    write_json(folder/'result.json',info);return info


def train():
    if ROOT.exists():raise FileExistsError('finite batch exists; do not rerun')
    data=setup();d,c,ext,lab,relation,train,tune,confirm,fresh,teacher,contract,source_report=data
    ROOT.mkdir();write_json(ROOT/'contract.json',contract);write_json(ROOT/'source_report.json',source_report)
    write_json(ROOT/'state.json',dict(status='preflight',automation_resumed=False))
    write_json(ROOT/'preflight.json',preflight(fresh,relation,d,c,ext,lab,train))
    all_caches={flag:(cache(c,ext,lab,relation,train,flag),cache(c,ext,lab,relation,tune,flag)) for flag in (False,True)}
    a,k,curve,nll=original_predict(teacher,d,c,ext,tune)
    with np.load('training_runs/v6_regimen_joint_fusion_2026-10-14/original/tune_predictions.npz') as z:
        np.testing.assert_array_equal(z['hadm_ids'],d.arrays['hadm_ids'][tune])
        np.testing.assert_array_equal(z['curve'],curve,err_msg='original actual numeric path must restore saved curve')
    reference=detail(d,tune,curve,nll);(ROOT/'original').mkdir();save(ROOT/'original/tuning_predictions.npz',d,tune,a,k,curve)
    write_json(ROOT/'original/result.json',dict(metrics=reference,newly_trained=False))
    print(json.dumps(dict(phase='cached_complete_inputs',source_report=source_report,
                         cuda_allocated_gb=round(torch.cuda.memory_allocated()/1e9,3),original_annual=reference['annual'])),flush=True)
    results={}
    for arm in ARMS:
        write_json(ROOT/'state.json',dict(status='training',arm=arm,automation_resumed=False))
        tr,va=all_caches[arm.startswith('history')]
        results[arm]=fit(fresh,arm,tr,va,d,tune,ROOT/arm)
    scores={arm:row['metrics'] for arm,row in results.items()};scores['original']=reference
    selected=sorted(scores,key=lambda n:(-scores[n]['annual']['f1'],scores[n]['readmission']['integrated_brier_7_365'],n))[0]
    nominated=selected if selected not in ('original','current_original') and scores[selected]['annual']['f1']>max(scores[n]['annual']['f1'] for n in ('original','current_original')) else None
    lock=dict(selected=selected,nominated=nominated,thresholds={n:m['annual']['threshold'] for n,m in scores.items()},
              rule='annual tuning F1; then IBS; no confirmation reselection',default_changed=False,goal_achieved=False)
    write_json(ROOT/'nomination_lock.json',lock);write_json(ROOT/'report.json',dict(reference=reference,results=results,lock=lock))
    write_json(ROOT/'state.json',dict(status='trained_pending_independent_replay',automation_resumed=False))
    print(json.dumps(lock),flush=True)


def verify():
    data=setup();d,c,ext,lab,relation,train,tune,confirm,fresh,teacher,contract,source_report=data
    if contract!=json.loads((ROOT/'contract.json').read_text()) or source_report!=json.loads((ROOT/'source_report.json').read_text()):raise ValueError('new process source/input contract differs')
    report=json.loads((ROOT/'report.json').read_text());errors={}
    for arm in ('original',*ARMS):
        va=cache(c,ext,lab,relation,tune,arm.startswith('history'))
        if arm=='original':model=None;expected=report['reference']
        else:
            model=LongitudinalModel(fresh,arm).cuda();chosen=torch.load(ROOT/arm/'best.pt',map_location='cuda',weights_only=True)
            info=report['results'][arm]
            if chosen['epoch']!=info['best_epoch'] or sha256(ROOT/arm/'best.pt')!=info['checkpoint_sha256']:raise ValueError('checkpoint differs')
            model.load_state_dict(chosen['state_dict']);expected=info['metrics']
            history=json.loads((ROOT/arm/'history.json').read_text());early=EarlyStopping(7,10,1e-4)
            for r in history:early.observe(r['epoch'],-r['annual']['f1'])
            if asdict(early)!=info['early']:raise ValueError('F1 early stop differs')
            selected=min(history,key=lambda r:(-r['annual']['f1'],r['tuning_ibs'],r['epoch']))
            if selected['epoch']!=chosen['epoch'] or selected['annual']['threshold']!=chosen['threshold']:raise ValueError('F1/IBS selection differs')
        a,k,curve,nll=original_predict(teacher,d,c,ext,tune) if arm=='original' else predict(model,va,d)
        with np.load(ROOT/arm/'tuning_predictions.npz') as z:
            for key,values in (('hadm_ids',d.arrays['hadm_ids'][tune]),('subject_ids',d.arrays['subject_ids'][tune]),('times',d.times[tune]),('events',d.events[tune])):np.testing.assert_array_equal(z[key],values)
            err=max(float(np.abs(z['curve']-curve).max()),float(np.abs(z['log_amplitude']-a).max()),float(np.abs(z['shape']-k).max()))
        if err>1e-6:raise ValueError('risk replay differs')
        actual=detail(d,tune,curve,nll,expected['annual']['threshold'])
        if actual['annual']!=expected['annual']:raise ValueError('locked threshold decision replay differs')
        for cause in ('readmission','death'):
            if abs(actual[cause]['integrated_brier_7_365']-expected[cause]['integrated_brier_7_365'])>1e-8:raise ValueError('IBS replay differs')
        errors[arm]=err;del model,va;torch.cuda.empty_cache()
    write_json(ROOT/'verification.json',dict(status='independent_inputs_weights_daily_probability_F1_early_replay',errors=errors,maximum_error=max(errors.values()),report_sha256=sha256(ROOT/'report.json')))
    write_json(ROOT/'state.json',dict(status='single_seed_independently_verified',automation_resumed=False))
    print(json.dumps(errors),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=('train','verify'))
    try: train() if p.parse_args().mode=='train' else verify()
    except Exception as e:
        if ROOT.exists():write_json(ROOT/'failure.json',dict(type=type(e).__name__,message=str(e)))
        raise
