"""单种子组级留一重训和匹配解冻历史；原四臂不覆盖。"""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from patient_data.export_mimiciv import sha256
from training.v6.data import load
from training.v6.model import HistoryBaseline, AnchoredModel, ClinicalBatch
from training.v6.risk import RiskOutput, predict_cif
from training.v6.train import make_batches, fit, predict, write_json
from training.v6.evaluation import GRID, metrics, public_metrics, paired_interval

ARMS = ('full_no_text', 'without_current', 'without_labs', 'without_medications', 'unfrozen_history')


def mask_group(batch: ClinicalBatch, arm: str) -> ClinicalBatch:
    if arm not in ARMS:
        raise ValueError('unknown ablation')
    if arm == 'without_current':
        return replace(batch, current_values=torch.zeros_like(batch.current_values), current_masks=torch.zeros_like(batch.current_masks))
    if arm == 'without_labs':
        return replace(batch, lab_axis_inputs=torch.zeros_like(batch.lab_axis_inputs))
    if arm == 'without_medications':
        return replace(batch, med_exposure=torch.zeros_like(batch.med_exposure), med_source_mask=torch.zeros_like(batch.med_source_mask))
    return batch


class UnfrozenClinicalModel(AnchoredModel):
    """与冻结参照相同架构/初值；唯一训练机制变更是历史参数允许更新。"""
    def __init__(self, baseline, text_width):
        super().__init__(baseline, text_width)
        self.baseline.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.baseline.train(mode)
        return self

    def forward(self, history, batch, *, mask_pmh=False):
        base = self.baseline(history)
        h, _ = self.encoder(batch, mask_pmh=mask_pmh)
        if len(h) != len(history):
            raise ValueError('history/current alignment mismatch')
        delta = self.delta_head(h).reshape(-1, 2, 2)
        shape_delta = delta[:, :, 1] if self.patient_shape else torch.zeros_like(delta[:, :, 1])
        shape = F.softplus(self.baseline.raw_shape + shape_delta) + .1
        return RiskOutput((base.log_amplitude + delta[:, :, 0]).clamp(-10.,6.), shape, delta)


def prepared(data, ix, size, arm):
    return [(pos, h, mask_group(b, arm), t, e) for pos, h, b, t, e in
            make_batches(data, ix, size, 'cuda', include_text=False)]


def build_model(base, width, arm, seed):
    torch.manual_seed(seed)
    cls = UnfrozenClinicalModel if arm == 'unfrozen_history' else AnchoredModel
    return cls(copy.deepcopy(base), width).cuda()


def evaluate(data, ix, model, batches, directory, fold):
    a, k, nll, delta = predict(model, batches)
    curve = predict_cif(a, k, GRID, step_days=.25)
    fine = predict_cif(a[:64], k[:64], GRID, step_days=.125)
    error = float(np.max(np.abs(fine-curve[:64])))
    if error > 2e-4 or not np.allclose(curve.sum(2), 1, atol=1e-10):
        raise ValueError('CIF numerical check failed')
    readmission = public_metrics(metrics(data.times[ix], data.events[ix], curve[:, :, 1], ordering=True))
    death = public_metrics(metrics(data.times[ix], data.events[ix], curve[:, :, 2], cause=2, ordering=True))
    counts = np.expm1(data.history_raw[ix, 2]).round().astype(int)
    strata = {}
    for name, keep in (('prior_0', counts == 0), ('prior_1_2', (counts>=1)&(counts<=2)), ('prior_3_plus', counts>=3)):
        strata[name] = public_metrics(metrics(data.times[ix][keep], data.events[ix][keep], curve[keep, :, 1], ordering=True))
    np.savez_compressed(directory / (fold+'_predictions.npz'), hadm_ids=data.arrays['hadm_ids'][ix],
                        subject_ids=data.arrays['subject_ids'][ix], times=data.times[ix], events=data.events[ix],
                        log_amplitude=a, shape=k, curve=curve, delta=delta)
    return curve, {'readmission': readmission, 'death': death, 'history_strata': strata,
                   'nll': nll, 'CIF_step_error_first64': error}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required, no CPU fallback')
    torch.set_num_threads(2)
    old = json.loads((args.reference / 'report.json').read_text(encoding='utf-8'))
    old_config = json.loads((args.reference / 'config.json').read_text(encoding='utf-8'))
    if (old['status'] != 'completed_single_seed_exploratory' or old_config['seed'] != 42
            or old_config['epochs'] != 60 or old_config['batch_size'] != 512):
        raise ValueError('reference training contract changed')
    data = load(Path(old_config['snapshot']), Path(old_config['state_dir']), Path(old_config['labels']),
                field_dir=Path(old_config['field_dir']), vectors_path=Path(old_config['vectors_path']))
    if data.provenance != old['contract']['data_provenance']:
        raise ValueError('reference data provenance mismatch')
    reference_checkpoint = args.reference / 'history/best.pt'
    if sha256(reference_checkpoint) != old['models']['history']['checkpoint_sha256']:
        raise ValueError('baseline checkpoint changed')
    base = HistoryBaseline().cuda()
    base.load_state_dict(torch.load(reference_checkpoint, map_location='cuda', weights_only=True)['state_dict'])
    seed, size = 42, 512
    ix = {f: np.flatnonzero(data.arrays['patient_splits']==f) for f in ('train','validation','test')}
    args.output.mkdir(parents=True)
    contract = {'status': 'matched_group_ablation_contract', 'seed': seed, 'max_epochs': 60, 'batch_size': size,
                'patience': 10, 'learning_rate': 3e-4, 'warmup': 5, 'bootstrap': 200,
                'reference_run': str(args.reference), 'reference_report_sha256': sha256(args.reference/'report.json'),
                'baseline_checkpoint_sha256': sha256(reference_checkpoint), 'data_provenance': data.provenance,
                'arms': list(ARMS), 'text': 'none for every arm', 'only_unfrozen_history_changes_history_training': True,
                'code_sha256': {f: sha256(Path(__file__).parent/f) for f in ('group_ablation.py','train.py','model.py','risk.py','data.py','evaluation.py')},
                'limits': ['Single seed; historical test reused', 'Group omission changes its validity flags and presence as well as content',
                           'Other branches can substitute correlated information; no drug causal effect estimate']}
    write_json(args.output/'contract.json', contract)
    models, fitted, initial = {}, {}, None
    for arm in ARMS:
        model = build_model(base, data.text_vectors.shape[1], arm, seed)
        # 对照全部参数逐项完全一致；解冻只改变requires_grad，不改参数数量。
        state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if initial is None:
            initial = state
        elif not all(torch.equal(state[k], initial[k]) for k in state):
            raise ValueError('nonmatched initialization')
        train = prepared(data, ix['train'], size, arm)
        validation = prepared(data, ix['validation'], size, arm)
        write_json(args.output/'status.json', {'status':'training','arm':arm})
        fitted[arm] = fit(model, data, train, validation, ix['validation'], args.output/arm,
                          epochs=60,lr=3e-4,seed=seed,patience=10,warmup=5)
        models[arm] = model
        del train, validation
        if arm == 'full_no_text':
            saved = json.loads((args.reference/'clinical_no_text/history.json').read_text())
            now = fitted[arm]['history']
            if (len(saved)!=len(now) or fitted[arm]['best_epoch']!=old['models']['clinical_no_text']['best_epoch']
                    or max(abs(x['validation_integrated_brier']-y['validation_integrated_brier']) for x,y in zip(saved,now,strict=True))>1e-7):
                raise ValueError('matched reference full training not reproduced')
    write_json(args.output/'status.json', {'status':'all_models_selected_before_test'})
    results, predictions = {}, {}
    for arm in ARMS:
        # 真正从磁盘重建模型并重载，正确保留屏蔽/解冻配方。
        model = build_model(base, data.text_vectors.shape[1], arm, seed)
        model.load_state_dict(torch.load(args.output/arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
        baseline_change = max(float((v-base.state_dict()[k]).abs().max().detach().cpu()) for k,v in model.baseline.state_dict().items())
        if arm!='unfrozen_history' and baseline_change!=0:
            raise ValueError('frozen baseline changed')
        results[arm] = {k:v for k,v in fitted[arm].items() if k!='history'}
        results[arm]['baseline_max_parameter_change'] = baseline_change
        results[arm]['folds'] = {}
        for f in ('validation','test'):
            batches = prepared(data,ix[f],size,arm)
            curve, detail = evaluate(data,ix[f],model,batches,args.output/arm,f)
            results[arm]['folds'][f] = detail
            predictions[arm,f] = curve
    comparisons = {}
    for f in ('validation','test'):
        comparisons[f] = {arm+'_minus_full': paired_interval(data.arrays['subject_ids'][ix[f]],data.times[ix[f]],data.events[ix[f]],
                                                              predictions[arm,f][:,:,1],predictions['full_no_text',f][:,:,1],replicates=200,seed=seed)
                          for arm in ARMS[1:]}
    report = {'status':'completed_matched_group_ablation_single_seed','contract':contract,'models':results,'comparisons':comparisons,
              'full_reference_training_reproduced':True,'initial_state_identical_across_arms':True,
              'limits':contract['limits']}
    write_json(args.output/'report.json',report)
    render(args.output,report)
    write_json(args.output/'status.json',{'status':report['status']})
    print(json.dumps({'status':report['status'],'output':str(args.output)}),flush=True)


def render(root, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = root/'figures'; figures.mkdir()
    labels = {'full_no_text':'Full clinical (no text)', 'without_current':'Without current diagnoses / disposition',
              'without_labs':'Without all lab axes', 'without_medications':'Without medication exposure',
              'unfrozen_history':'Matched unfrozen history'}
    for fold in ('validation','test'):
        fig,axes=plt.subplots(1,2,figsize=(14,4))
        for i,arm in enumerate(ARMS[1:]):
            c=report['comparisons'][fold][arm+'_minus_full']
            for ax,p,b in zip(axes,('integrated_brier_difference','auc90_difference'),('integrated_brier_95ci','auc90_95ci')):
                ax.plot(c[b],[i,i],color='#2474b7',lw=2); ax.scatter(c[p],i,marker='D',color='#d95f02',zorder=3)
        for ax,title in zip(axes,('Integrated Brier: candidate - full (lower better)','AUC(90): candidate - full (higher better)')):
            ax.axvline(0,color='gray'); ax.set_yticks(range(4),[labels[a] for a in ARMS[1:]],fontsize=8)
            ax.set_title(title,fontsize=11); ax.grid(axis='x',alpha=.2)
        fig.suptitle(f'{fold}: retrained groups; one seed; patient-bootstrap 95% intervals; no multiplicity adjustment')
        fig.tight_layout(); fig.savefig(figures/(fold+'_paired_differences.png'),dpi=180,bbox_inches='tight'); plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for arm in ARMS:
            with np.load(root/arm/(fold+'_predictions.npz')) as archive:
                m=metrics(archive['times'],archive['events'],archive['curve'][:,:,1])
            axes[0].plot(GRID[7:],m['auc_curve'][7:],label=labels[arm]); axes[1].plot(GRID[7:],m['brier_curve'][7:])
        axes[0].set_title('Strict readmission AUC(t)'); axes[1].set_title('Readmission Brier(t)')
        for ax in axes: ax.set_xlabel('Days after discharge'); ax.grid(alpha=.2)
        axes[0].legend(fontsize=7); fig.suptitle(f'{fold}: matched group experiments, World A')
        fig.tight_layout(); fig.savefig(figures/(fold+'_auc_brier.png'),dpi=180,bbox_inches='tight'); plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
