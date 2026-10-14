"""已完成一年F1实验的图与分层读出，不能重选epoch或阈值。"""
import argparse
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.unified_longitudinal import f1_at

ROOT=Path('training_runs/v6_unified_longitudinal_f1_2026-10-14_r2')


def build():
    proof=json.loads((ROOT/'verification.json').read_text());report=json.loads((ROOT/'report.json').read_text())
    if proof['maximum_error']!=0 or proof['report_sha256']!=sha256(ROOT/'report.json'):raise ValueError('replay stale')
    with np.load('patient_data/processed/v6_model_inputs_bert_2026-10-09/context.npz') as z:
        hadm=z['hadm_ids'];raw=z['history_raw'];lookup={int(v):i for i,v in enumerate(hadm)}
    result={};a={};ref=None
    for name in ('original',*report['results']):
        with np.load(ROOT/name/'tuning_predictions.npz') as z:
            values={k:z[k] for k in ('hadm_ids','subject_ids','times','events','curve')}
        if ref is None:ref=values
        for k in ('hadm_ids','subject_ids','times','events'):np.testing.assert_array_equal(values[k],ref[k])
        count=np.rint(np.expm1(raw[np.array([lookup[int(v)] for v in values['hadm_ids']]),2])).astype(int)
        expected=report['reference'] if name=='original' else report['results'][name]['metrics']
        threshold=expected['annual']['threshold'];groups={}
        for group,mask in dict(prior_0=count==0,prior_1_2=(count>=1)&(count<=2),prior_3_plus=count>=3).items():
            y=(values['events'][mask]==1)&(values['times'][mask]<=365)
            groups[group]=dict(admissions=int(mask.sum()),patients=len(np.unique(values['subject_ids'][mask])),
                               annual=f1_at(y,values['curve'][mask,365,1],threshold))
        result[name]=groups
    out=dict(report_sha256=sha256(ROOT/'report.json'),fixed_global_threshold_per_model=True,
             strata=result,no_refitting=True,no_new_selection=True)
    (ROOT/'strata_readout.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({name:{s:x['annual']['f1'] for s,x in groups.items()} for name,groups in result.items()}),flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads((ROOT/'report.json').read_text());v=json.loads((ROOT/'verification.json').read_text())
    if v['report_sha256']!=sha256(ROOT/'report.json'):raise ValueError('replay stale')
    labels=dict(current_original='Current / original labs',current_unified='Current / unified labs',
                history_original='Full history / original labs',history_unified='Full history / unified labs')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for arm,label in labels.items():
        rows=json.loads((ROOT/arm/'history.json').read_text());epochs=[x['epoch'] for x in rows]
        axes[0].plot(epochs,[x['annual']['f1'] for x in rows],label=label)
        axes[1].plot(epochs,[x['tuning_ibs'] for x in rows],label=label)
    axes[0].axhline(r['reference']['annual']['f1'],color='black',linestyle='--',label='Retained V6')
    axes[1].axhline(r['reference']['readmission']['integrated_brier_7_365'],color='black',linestyle='--',label='Retained V6')
    for ax,title,ylabel in zip(axes,('Annual F1 selection (higher better)','Probability error (lower better)'),('F1 at tuning-selected threshold','Readmission IBS7-365'),strict=True):
        ax.set(title=title,xlabel='Actual epoch',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Seed 42; fresh clinical initialization; shared trainable history encoder; 365-day endpoint')
    fig.tight_layout();fig.savefig(ROOT/'training_f1_ibs.png',dpi=140);plt.close(fig)
    names=['original',*labels];metrics=[r['reference']['annual'],*[r['results'][n]['metrics']['annual'] for n in labels]]
    fig,axes=plt.subplots(1,2,figsize=(12,4.6));x=np.arange(len(names))
    axes[0].bar(x,[m['f1'] for m in metrics],label='F1');axes[0].set_ylim(0,1)
    for i,m in enumerate(metrics):axes[0].text(i,m['f1']+.015,f"{m['f1']:.3f}",ha='center',fontsize=8)
    axes[1].bar(x-.18,[m['precision'] for m in metrics],width=.36,label='Precision')
    axes[1].bar(x+.18,[m['recall'] for m in metrics],width=.36,label='Recall');axes[1].set_ylim(0,1);axes[1].legend()
    for ax in axes:ax.set_xticks(x,['Retained V6',*['Current\noriginal','Current\nunified','History\noriginal','History\nunified']]);ax.grid(axis='y',alpha=.2)
    axes[0].set_title('Selected annual F1');axes[1].set_title('Precision and recall at locked tuning thresholds')
    fig.suptitle('1,531 exposed tuning discharges / 359 events; no confirmation/test threshold fitting')
    fig.tight_layout();fig.savefig(ROOT/'annual_decision.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=('build','plot'))
    build() if p.parse_args().mode=='build' else plot()
