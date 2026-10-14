"""只读已独立重载的六轮结果，汇总明确时点与双种子配对图；不选模不训练。"""
import argparse
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json

RUNS={'linear42':'v6_expanded_baseline_seed42_2026-10-10','linear7':'v6_expanded_baseline_seed7_2026-10-10',
      'nonlinear42':'v6_nonlinear_expanded_seed42_2026-10-10_r2','nonlinear7':'v6_nonlinear_expanded_seed7_2026-10-10',
      'explore42':'v6_architecture_screen_seed42_2026-10-10','explore7':'v6_architecture_screen_seed7_2026-10-10'}


def run(output):
    if output.exists(): raise FileExistsError(output)
    reports={};digests={}
    for key,name in RUNS.items():
        root=Path('training_runs')/name
        r=json.loads((root/'report.json').read_text());v=json.loads((root/'independent_verification.json').read_text())
        if v['status']!='all_arms_gpu_reloaded' or v['report_sha256']!=sha256(root/'report.json'): raise ValueError('run not independently verified')
        reports[key]=r;digests[key]=dict(report_sha256=sha256(root/'report.json'),verification_sha256=sha256(root/'independent_verification.json'))
    output.mkdir(parents=True)
    rows=[]
    for key,r in reports.items():
        for arm,m in r['models'].items():
            row=dict(run=key,arm=arm,seed=r['contract']['seed'],best_epoch=m['best_epoch'],epochs_trained=m['epochs_trained'])
            for fold in ('validation','test'):
                d=m['folds'][fold]['readmission']
                row[fold+'_IBS']=d['integrated_brier_7_365']
                row[fold+'_CIF_C365']=d['project_cif_concordance']['365'].get('c_index')
                for day in (30,90,180,365):
                    row[f'{fold}_AUC{day}']=d['points'][str(day)]['auc']
                    row[f'{fold}_Brier{day}']=d['points'][str(day)]['brier']
            rows.append(row)
    write_json(output/'summary.json',dict(status='verified_results_only',runs=RUNS,source_hashes=digests,rows=rows,
                 conclusions=dict(default_baseline='Keep original H+full clinical+BERT; expanded B retained as attribution reference, not proven equivalent',
                                  architecture='No stable verified gain; seed42 screen fusion wins validation but not historical test',
                                  ratios='NLR/SII do not improve consistently over same-source components',
                                  text='Screen alone and fusion do not outperform B consistently; source screening not semantic gold'),
                 limits=['H reference fixed seed42; B/delta seeds42/7; not all-pipeline retraining variance',
                         'Many hypotheses, repeated historical test, no multiplicity correction; paired patient bootstrap200',
                         'World A observed-record endpoint; no verified full individual external risk']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # 每个图都明确差值方向；均值菱形不冒充置信区间。
    pairs=[('Linear expanded frozen B','linear','expanded_full','legacy_full'),
           ('Nonlinear expanded frozen B','nonlinear','expanded_full','legacy_full'),
           ('Original BERT vs no text','explore','pool128','no_text'),
           ('CBC components vs availability','explore','components','availability'),
           ('NLR / SII vs components','explore','nlr_sii','components'),
           ('Screened BERT vs original BERT','explore','screen_only','pool128'),
           ('BERT + screened vs original BERT','explore','bert_screen','pool128')]
    for fold in ('validation','test'):
        fig,axes=plt.subplots(1,2,figsize=(16,5.5),sharey=True)
        for i,(label,family,left,right) in enumerate(pairs):
            vals=[]
            for j,seed in enumerate((42,7)):
                r=reports[family+str(seed)]
                a=r['models'][left]['folds'][fold]['readmission'];b=r['models'][right]['folds'][fold]['readmission']
                value=(a['integrated_brier_7_365']-b['integrated_brier_7_365'],a['points']['90']['auc']-b['points']['90']['auc'])
                vals.append(value)
                for axis,v in zip(axes,value): axis.scatter(v,i+(-.1 if j==0 else .1),s=45,color=('#2878b5','#d95f02')[j],label=f'Seed {seed}' if i==0 else None)
            avg=np.mean(vals,axis=0)
            for axis,v in zip(axes,avg): axis.scatter(v,i,marker='D',s=65,color='#333333',label='Two-seed mean' if i==0 else None)
        for ax in axes:
            ax.axvline(0,color='gray');ax.set_yticks(range(len(pairs)),[p[0] for p in pairs],fontsize=9);ax.grid(axis='x',alpha=.2)
        axes[0].invert_yaxis();axes[0].set_title('Integrated Brier: candidate - comparator (lower better)')
        axes[1].set_title('90-day AUC: candidate - comparator (higher better)')
        axes[0].set_xlabel('Paired difference');axes[1].set_xlabel('Paired difference');axes[1].legend(fontsize=8)
        fig.suptitle(f'{fold}: strict AECOPD; H fixed; B/delta retrained; no multiplicity adjustment\nDots = fitted seed estimates; diamonds = mean, not confidence intervals',fontsize=11)
        fig.tight_layout();fig.savefig(output/(fold+'_paired_two_seeds.png'),dpi=180);plt.close(fig)
    names=['pool64','pool256','hidden64','retained','set4','piecewise','pool128_lr1e4','retained_lr1e4','set4_lr1e4','piecewise_lr1e4']
    fig,axes=plt.subplots(1,2,figsize=(15,6),sharey=True)
    r=reports['explore42']
    for i,name in enumerate(names):
        for j,fold in enumerate(('validation','test')):
            a=r['models'][name]['folds'][fold]['readmission'];b=r['models']['pool128']['folds'][fold]['readmission']
            axes[0].scatter(a['integrated_brier_7_365']-b['integrated_brier_7_365'],i+(-.1 if j==0 else .1),color=('#2878b5','#d95f02')[j],label=fold if i==0 else None)
            axes[1].scatter(a['points']['90']['auc']-b['points']['90']['auc'],i+(-.1 if j==0 else .1),color=('#2878b5','#d95f02')[j])
    for ax in axes: ax.axvline(0,color='gray');ax.grid(axis='x',alpha=.2);ax.set_yticks(range(len(names)),names,fontsize=9)
    axes[0].invert_yaxis();axes[0].set_title('Integrated Brier difference (lower better)');axes[1].set_title('90-day AUC difference (higher better)')
    axes[0].legend();fig.suptitle('Architecture candidate - pool128; same linear B; seed42 exploratory; test reused')
    fig.tight_layout();fig.savefig(output/'architecture_paired.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(status='verified_results_only',models=len(rows),output=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
