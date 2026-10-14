"""已核对原文的A完整章节/同资产四字段配对试验，不改旧模型和报告。"""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from patient_data.export_mimiciv import sha256
from patient_data.align_saved_sections import ROLES, FOUR
from training.v6.compact_ablation import inputs
from training.v6.compact_context_data import attach
from training.v6.compact_model import CompactClinicalEncoder, CompactAnchoredModel
from training.v6.architecture_experiment import batches
from training.v6.horizon_objective import restore, labels90, fit90, predict90, binary_metrics, paired90, ranked_curves
from training.v6.evaluation import GRID, metrics, public_metrics
from training.v6.risk import predict_cif
from training.v6.train import write_json

ARMS = ('none', 'four', 'clinical9', 'full10')
PROTOCOL = Path('docs/V6_FULL_SECTION_COMPARISON_PROTOCOL_2026-10-11.md')


class SectionEncoder(CompactClinicalEncoder):
    def __init__(self):
        super().__init__(768)
        self.field_ids = nn.Embedding(len(ROLES), self.width)

    def _text(self, batch, n, *, mask_pmh=False):
        v, ix = batch.snippet_vectors, batch.snippet_index
        fields = len(ROLES)
        if (v.ndim != 2 or v.shape[1] != 768 or ix.shape != (len(v), 2)
            or ix.dtype != torch.long or (ix < 0).any() or (ix[:,0] >= n).any()
            or (ix[:,1] >= fields).any() or not torch.isfinite(v).all()):
            raise ValueError('invalid A section vectors/index')
        if mask_pmh:
            keep = ix[:,1] != ROLES.index('past_medical_history')
            v, ix = v[keep], ix[keep]
        if not len(v):
            return v.new_zeros((n,self.width)), torch.zeros(n,device=v.device,dtype=torch.bool)
        content = self.text_projection(v)+self.field_ids(ix[:,1])
        group = fields*ix[:,0]+ix[:,1]
        score = self.text_score(content).squeeze(-1)
        maximum = score.new_full((fields*n,),-torch.inf)
        maximum.scatter_reduce_(0,group,score,reduce='amax',include_self=True)
        weight = torch.exp(score-maximum[group])
        total = score.new_zeros(fields*n).scatter_add_(0,group,weight)
        weight = weight/total[group]
        pooled = content.new_zeros((fields*n,self.width)).index_add_(0,group,weight[:,None]*content)
        present = (total>0).reshape(n,fields)
        result = pooled.reshape(n,fields,self.width).sum(1)/present.sum(1).clamp_min(1)[:,None]
        return result,present.any(1)


def load_inputs(args):
    d,c,hist,_ = inputs(args.reference,args.compact_dir)
    parent = json.loads((args.parent/'report.json').read_text())
    proof = json.loads((args.parent/'independent_verification.json').read_text())
    if proof['report_sha256'] != sha256(args.parent/'report.json'): raise ValueError('parent unverified')
    if d.provenance != parent['contract']['data_provenance'] or c.provenance != parent['contract']['compact_provenance']:
        raise ValueError('parent input changed')
    for p,digest in parent['contract']['code_sha256'].items():
        if sha256(Path(p)) != digest: raise ValueError('parent code changed: '+p)
    manifest = json.loads((args.aligned/'manifest.json').read_text())
    if (manifest['status'] != 'current_discharge_source_and_complete_section_coverage_verified'
        or tuple(manifest['roles']) != ROLES or manifest['state_manifest_sha256'] != d.provenance['state_manifest_sha256']):
        raise ValueError('A alignment contract changed')
    for p,digest in manifest['source_sha256'].items():
        if sha256(Path(p.replace('\\','/'))) != digest: raise ValueError('A source changed: '+p)
    for p,digest in manifest['output_sha256'].items():
        if sha256(args.aligned/p) != digest: raise ValueError('A aligned output changed')
    with np.load(args.aligned/'alignment.npz') as z: a={k:z[k] for k in z.files}
    for k in ('hadm_ids','subject_ids','patient_splits'):
        if not np.array_equal(a[k],d.arrays[k]): raise ValueError('aligned member/split differs')
    v = np.load(args.aligned/'vectors.float16.npy',mmap_mode='r')
    refs = a['refs']
    if v.shape != (len(refs),768) or refs.shape != (len(v),3) or not np.isfinite(v).all():
        raise ValueError('aligned array shape/value differs')
    if (not np.array_equal(refs[:,2],np.arange(len(v))) or (refs[:,:2]<0).any()
        or (refs[:,0]>=len(d.history)).any() or (refs[:,1]>=len(ROLES)).any()):
        raise ValueError('invalid aligned row mapping')
    members = np.unique(refs[:,0])
    ix = {f:members[d.arrays['patient_splits'][members]==f] for f in ('train','validation','test')}
    views, contexts = {}, {}
    for arm in ARMS:
        ids = () if arm=='none' else FOUR if arm=='four' else tuple(range(9)) if arm=='clinical9' else tuple(range(10))
        mapping=[[] for _ in d.history]
        for position,field,row in refs:
            if field in ids: mapping[position].append((int(field),int(row)))
        views[arm]=replace(d,text_vectors=v,text_refs=tuple(tuple(x) for x in mapping))
        contexts[arm]=replace(c,base=views[arm])
    return d,views,contexts,hist,ix,parent,manifest


def model_for(args,seed,hist):
    base = restore(args.parent,seed,'survival_base',hist,.1)
    torch.manual_seed(seed)
    m = CompactAnchoredModel(copy.deepcopy(base),768)
    m.encoder = SectionEncoder()
    return m.cuda()


def prepared(views,contexts,ix,arm):
    return batches(views[arm],contexts[arm],ix,512,expanded=True,text='bert')


def prediction(root,seed,arm,fold):
    with np.load(root/f'seed{seed}'/arm/(fold+'_predictions.npz')) as z: return {k:z[k] for k in z.files}


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2); started=time.perf_counter()
    d,views,contexts,hist,ix,parent,manifest = load_inputs(args)
    y=labels90(d.times,d.events); args.output.mkdir(parents=True)
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    contract=dict(parent_report_sha256=sha256(args.parent/'report.json'),
        alignment_manifest_sha256=sha256(args.aligned/'manifest.json'),
        data_provenance=d.provenance,compact_provenance=contexts['none'].provenance,
        protocol_sha256=sha256(PROTOCOL),code_sha256={**parent['contract']['code_sha256'],
            'training/v6/saved_section_experiment.py':sha256(Path(__file__)),
            'patient_data/align_saved_sections.py':sha256(Path('patient_data/align_saved_sections.py'))},
        seeds=[42,7],arms=list(ARMS),selection='validation Brier90',epochs=60,patience=10,lr=3e-4,
        device=torch.cuda.get_device_name(0),folds={f:dict(admissions=len(p),patients=len(np.unique(d.arrays['subject_ids'][p])),
            positives=int(y[p].sum())) for f,p in ix.items()},sections=manifest['sections'])
    write_json(args.output/'contract.json',contract)
    fitted={}
    for seed in (42,7):
        (args.output/f'seed{seed}').mkdir(); common=None
        for arm in ARMS:
            m=model_for(args,seed,hist); initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            if common is None: common=initial
            elif not all(torch.equal(v,common[k]) for k,v in initial.items()): raise ValueError('arm initial states differ')
            cache={f:prepared(views,contexts,ix[f],arm) for f in ('train','validation')}
            write_json(args.output/'status.json',dict(status='training',seed=seed,arm=arm))
            fitted[seed,arm]=fit90(m,cache['train'],cache['validation'],y[ix['validation']],args.output/f'seed{seed}'/arm,
                epochs=60,lr=3e-4,seed=seed,patience=10,warmup=5)
            del cache,m
    write_json(args.output/'status.json',dict(status='all8_checkpoints_selected_before_test'))
    models,comparisons={},{ }
    for seed in (42,7):
        models[str(seed)]={}; comparisons[str(seed)]={}
        for arm in ARMS:
            m=model_for(args,seed,hist)
            m.load_state_dict(torch.load(args.output/f'seed{seed}'/arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            models[str(seed)][arm]={**fitted[seed,arm],'folds':{}}
            for f in ('validation','test'):
                idx=ix[f]; p=predict90(m,prepared(views,contexts,idx,arm))
                p['curve']=predict_cif(p['parameters'],p['shape'],GRID)
                if (not np.allclose(p['curve'].sum(2),1,atol=1e-9) or (np.diff(p['curve'][:,:,1:],axis=1)<-1e-10).any()
                    or not np.allclose(p['risk90'],p['curve'][:,90,1],atol=1e-8)):
                    raise ValueError('CIF contract failed')
                detail=binary_metrics(y[idx],p['risk90'])
                detail['survival_readmission']=public_metrics(metrics(d.times[idx],d.events[idx],p['curve'][:,:,1],ordering=True))
                detail['survival_death']=public_metrics(metrics(d.times[idx],d.events[idx],p['curve'][:,:,2],cause=2,ordering=True))
                models[str(seed)][arm]['folds'][f]=detail
                np.savez_compressed(args.output/f'seed{seed}'/arm/(f+'_predictions.npz'),
                    hadm_ids=d.arrays['hadm_ids'][idx],subject_ids=d.arrays['subject_ids'][idx],times=d.times[idx],events=d.events[idx],
                    y90=y[idx],**{k:v for k,v in p.items() if k!='loss'})
            del m
        for f in ('validation','test'):
            comparisons[str(seed)][f]={}
            for arm in ARMS[1:]:
                p=prediction(args.output,seed,arm,f)
                for other in ('none','four'):
                    if other==arm: continue
                    base=prediction(args.output,seed,other,f)
                    comparisons[str(seed)][f][arm+'_minus_'+other]=paired90(p['subject_ids'],p['y90'],p['risk90'],base['risk90'],seed=seed)
    report=dict(status='completed_source_aligned_same_asset_8_models',contract=contract,models=models,comparisons=comparisons,
        elapsed_seconds=time.perf_counter()-started,all_arm_initial_states_equal=True,
        limits=['Reused test; no multiplicity adjustment; matched subset only, not all19401.',
            'Full means every natural section of one saved selected note, not all database notes.',
            'Baseline trained on full training fold; clinical correction trained on matched subset.',
            'Old A full/four prefix differs from current B recipe; no direct cross-population attribution.'])
    write_json(args.output/'report.json',report); render(args.output,report)
    write_json(args.output/'status.json',dict(status=report['status'],seconds=report['elapsed_seconds']))
    print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'])),flush=True)


def verify(args):
    report=json.loads((args.output/'report.json').read_text()); contract=report['contract']
    for p,digest in contract['code_sha256'].items():
        if sha256(Path(p))!=digest: raise ValueError('source code changed')
    if sha256(PROTOCOL)!=contract['protocol_sha256']: raise ValueError('protocol changed')
    d,views,contexts,hist,ix,_,_=load_inputs(args); y=labels90(d.times,d.events); errors={}
    if sha256(args.aligned/'manifest.json')!=contract['alignment_manifest_sha256']: raise ValueError('alignment changed')
    for seed in (42,7):
        for arm in ARMS:
            path=args.output/f'seed{seed}'/arm/'best.pt'
            if sha256(path)!=report['models'][str(seed)][arm]['checkpoint_sha256']: raise ValueError('checkpoint changed')
            m=model_for(args,seed,hist); frozen=copy.deepcopy(m.baseline.state_dict())
            m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
            if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in frozen.items()): raise ValueError('baseline changed')
            for f in ('validation','test'):
                p=predict90(m,prepared(views,contexts,ix[f],arm)); saved=prediction(args.output,seed,arm,f)
                for k,val in (('hadm_ids',d.arrays['hadm_ids'][ix[f]]),('subject_ids',d.arrays['subject_ids'][ix[f]]),
                              ('times',d.times[ix[f]]),('events',d.events[ix[f]]),('y90',y[ix[f]])):
                    if not np.array_equal(saved[k],val): raise ValueError('identity/labels changed')
                for k,val in p.items():
                    if k!='loss' and not np.allclose(saved[k],val,atol=1e-7): raise ValueError('replay mismatch')
                curve=predict_cif(p['parameters'],p['shape'],GRID)
                if not np.allclose(saved['curve'],curve,atol=1e-7): raise ValueError('curve replay mismatch')
                fine=predict_cif(p['parameters'][:64],p['shape'][:64],GRID,step_days=.125)
                if np.abs(fine-curve[:64]).max()>2e-4: raise ValueError('integration convergence failed')
                measured=binary_metrics(y[ix[f]],p['risk90'])
                for k in ('auc90','average_precision','brier90','logloss90'):
                    if abs(measured[k]-report['models'][str(seed)][arm]['folds'][f][k])>1e-7: raise ValueError('metric replay mismatch')
                errors[f'{seed}/{arm}/{f}']=float(np.abs(saved['risk90']-p['risk90']).max())
            del m
    write_json(args.output/'independent_verification.json',dict(status='all8_models_all16_folds_independently_reloaded',
        report_sha256=sha256(args.output/'report.json'),frozen_baseline_unchanged=True,errors=errors))
    print('all8_models_all16_folds_independently_reloaded',flush=True)


def render(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures'; folder.mkdir()
    labels=dict(none='No text',four='Same-asset four fields',clinical9='Nine clinical sections',full10='Full ten sections')
    colors=dict(none='#888888',four='#2674ae',clinical9='#2a925b',full10='#cf7722')
    for seed in (42,7):
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        for arm in ARMS:
            p=prediction(root,seed,arm,'test'); m=report['models'][str(seed)][arm]['folds']['test']
            c=ranked_curves(p['y90'],p['risk90']); bins=m['calibration_bins']
            axes[0].plot(c['fpr'],c['tpr'],color=colors[arm],label=f'{labels[arm]}: AUC90={m["auc90"]:.3f}')
            actual=metrics(p['times'],p['events'],p['curve'][:,:,1])
            axes[1].plot(GRID[7:],actual['auc_curve'][7:],color=colors[arm],label=labels[arm])
            axes[2].plot([b['predicted'] for b in bins],[b['observed'] for b in bins],'o-',color=colors[arm],label=labels[arm])
        axes[0].plot([0,1],[0,1],'k--',alpha=.3); axes[2].plot([0,1],[0,1],'k--',alpha=.3)
        axes[2].set(xlim=(0,.7),ylim=(0,.7))
        for ax,title,x,y in zip(axes,('90-day ROC','Time-dependent AUC(t)','90-day Calibration'),
            ('False positive rate','Days after discharge','Mean predicted risk'),('True positive rate','AUC(t)','Observed fraction')):
            ax.set(title=title,xlabel=x,ylabel=y); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle(f'Same A encodings and patients; seed{seed}; reused test, not blind')
        fig.tight_layout(); fig.savefig(folder/f'test_seed{seed}_curves.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
    pairs=('four_minus_none','clinical9_minus_four','full10_minus_four')
    for j,pair in enumerate(pairs):
        for i,seed in enumerate((42,7)):
            results=report['comparisons'][str(seed)]['test'][pair]['metrics']
            for ax,k in zip(axes,('brier90','auc90')):
                r=results[k]; yy=j+(i-.5)*.15; color=('#2674ae','#cf7722')[i]
                ax.plot(r['ci95'],[yy,yy],color=color); ax.scatter(r['difference'],yy,color=color,label=f'seed{seed}' if j==0 else None)
    for ax in axes:
        ax.axvline(0,color='k',lw=1); ax.grid(axis='x',alpha=.2); ax.legend(); ax.set_yticks(range(3),pairs)
    axes[0].invert_yaxis(); axes[0].set_xlabel('Brier90 paired difference (lower better)')
    axes[1].set_xlabel('AUC90 paired difference (higher better)')
    fig.suptitle('Matched patients; 200 patient-clustered 95% intervals; no multiplicity adjustment')
    fig.tight_layout(); fig.savefig(folder/'test_paired.png',dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--compact-dir',type=Path,default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--parent',type=Path,default=Path('training_runs/v6_90day_objective_2026-10-11'))
    p.add_argument('--aligned',type=Path,default=Path('patient_data/processed/v6_saved_sections_aligned_2026-10-11'))
    p.add_argument('--output',type=Path,default=Path('training_runs/v6_saved_section_comparison_2026-10-11'))
    p.add_argument('--verify',action='store_true')
    args=p.parse_args(); (verify if args.verify else run)(args)


if __name__=='__main__': main()
