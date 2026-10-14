"""条件化候选的核验结果/校准/分层；不根据图表重新选择候选。"""
import argparse
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.contextual_residual_experiment import ROOT,SEEDS,TRIALS,inputs
from training.v6.task_optimization_report import averages,prepare_plot
from training.v6.task_optimization_diagnostics import analyze
from training.v6.regularization_diagnostics import history_counts
from training.v6.train import write_json


def fmt(x):return '不可计算' if x is None else f'{x:.6f}'


def report(mode):
    proofname='tuning_verification.json' if mode=='tuning' else 'confirmation_verification.json'
    proof=json.loads((ROOT/proofname).read_text())
    if mode=='tuning':
        if proof['status']!='tuning_independently_verified' or proof['models']!=18:
            raise ValueError('independent tuning replay needed')
        if proof['selected_sha256']!=sha256(ROOT/'selected_before_confirmation.json'):
            raise ValueError('selection proof changed')
        if proof['tuning_report_sha256']!=sha256(ROOT/'tuning_report.json'):raise ValueError('tuning report proof changed')
        r=json.loads((ROOT/'tuning_report.json').read_text());names=[t.name for t in TRIALS]
        selected=json.loads((ROOT/'selected_before_confirmation.json').read_text())
        if r['selected']!=selected['selected']:raise ValueError('report choice differs')
        for seed in SEEDS:
            for name in names:
                p=ROOT/f'seed{seed}'/name/'best.pt';info=r['models'][str(seed)][name]['training']
                if info!=selected['results'][str(seed)][name] or sha256(p)!=info['checkpoint_sha256']:raise ValueError('weight changed')
    else:
        if proof['report_sha256']!=sha256(ROOT/'confirmation_report.json'):raise ValueError('confirmation proof changed')
        r=json.loads((ROOT/'confirmation_report.json').read_text())
        names=['control','baseline_only',r['selected'],'official_V6']
    stats=averages(r['models'],names)
    lines=[f'# 条件化临床修正：{"调参" if mode=="tuning" else "锁定确认"}结果','',
        f'三临床种子42、7、101；两个固定基础来源42、7、42。当前选择`{r["selected"]}`。没有完成五种子最终门槛，不更换默认V6。',
        '', '| 模型 | AUC30↑ | AUC90↑ | AUC180↑ | AUC365↑ | Brier90↓ | 再入院IBS↓ | 死亡IBS↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        s=stats[name];p=s['readmission']['points']
        lines.append('| '+name+' | '+' | '.join([fmt(p[str(day)]['auc']) for day in (30,90,180,365)]+
            [fmt(p['90']['brier']),fmt(s['readmission']['ibs']),fmt(s['death']['ibs'])])+' |')
    lines+=['','基础再校准baseline_only没有读取临床数据；其收益不能称为检验/药物/文本增量。',
            '同队列/患者划分/严格结局/每日竞争概率，原始数据和向量不变。原验证曾曝光，不是新盲测，历史测试未用于搜索。']
    if mode=='tuning':
        plt=prepare_plot();fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
        for j,name in enumerate(names):
            deltas=[[],[]]
            for index,seed in enumerate(SEEDS):
                x=r['models'][str(seed)][name]['metrics']['readmission'];base=r['models'][str(seed)]['control']['metrics']['readmission']
                values=(x['integrated_brier_7_365']-base['integrated_brier_7_365'],x['points']['90']['auc']-base['points']['90']['auc'])
                for ax,collection,value in zip(axes,deltas,values):
                    collection.append(value);ax.scatter(value,j+(index-1)*.1,color=f'C{index}',s=23,label=f'Seed {seed}' if j==0 else None)
            for ax,values in zip(axes,deltas):ax.scatter(np.mean(values),j,marker='D',color='black',label='Three-seed mean' if j==0 else None)
        for ax in axes:ax.axvline(0,color='gray',lw=1);ax.grid(axis='x',alpha=.25);ax.set_yticks(range(len(names)),names)
        axes[0].invert_yaxis();axes[0].set_xlabel('IBS7-365: candidate minus matched V6 (negative better)')
        axes[1].set_xlabel('AUC90: candidate minus matched V6 (positive better)')
        fig.suptitle('Frozen-risk-conditioned clinical correction; tuning validation only\n'
                     'Baseline-only comparator excludes clinical inputs; no confidence intervals')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.88),ncol=4)
        fig.tight_layout(rect=(0,0,1,.80));fig.savefig(ROOT/'tuning_increment.png',dpi=180);plt.close(fig)
        lines+=['','![调参三种子差值](tuning_increment.png)','',
                '每个点是一个种子，黑色菱形为均值，不是患者95%区间；仅调参筛选，不是稳定增益证据。',
                'control或baseline_only胜出不挑选另一临床候选确认。只有候选同时优于两个参照，才按事前规则确认。']
    else:
        lines+=['','## 配对区间','',
            '| 种子 | 参照 | AUC90差值↑ | IBS差值↓ | AUC90患者95%区间 | IBS患者95%区间 |',
            '|---|---|---:|---:|---|---|']
        for seed in SEEDS:
            for reference in ('control','baseline_only','official_V6'):
                p=r['comparisons'][str(seed)][reference]
                lines.append(f'| {seed} | {reference} | {p["auc90_difference"]:.6f} | {p["integrated_brier_difference"]:.6f} | {p["auc90_95ci"]} | {p["integrated_brier_95ci"]} |')
        lines+=['','200次患者聚类配对抽样为条件于已拟合模型的探索区间；不包含重训方差、不消除选择偏倚或观察偏倚。']
    (ROOT/('TUNING_RESULTS.md' if mode=='tuning' else 'CONFIRMATION_RESULTS.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='verified_context_report_written',stage=mode,goal_achieved=False)),flush=True)


def diagnostics():
    r=json.loads((ROOT/'confirmation_report.json').read_text())
    proof=json.loads((ROOT/'confirmation_verification.json').read_text())
    if proof['report_sha256']!=sha256(ROOT/'confirmation_report.json'):raise ValueError('confirmation proof changed')
    (d,_,_,_,_,_,_,ix,_,_),_=inputs();counts=history_counts(d.history_raw[ix]);result={}
    for seed in SEEDS:
        curves={}
        for name in ['control','baseline_only',r['selected'],'official_V6']:
            with np.load(ROOT/f'seed{seed}'/name/'confirmation_predictions.npz') as z:
                for key,expected in (('hadm_ids',d.arrays['hadm_ids'][ix]),('subject_ids',d.arrays['subject_ids'][ix]),
                                     ('times',d.times[ix]),('events',d.events[ix])):
                    np.testing.assert_array_equal(z[key],expected)
                curves[name]=z['curve'].copy()
        result[str(seed)],counts_by_group=analyze(d.times[ix],d.events[ix],d.arrays['subject_ids'][ix],counts,curves,
                seed=seed,comparators=('control','baseline_only','official_V6'))
    write_json(ROOT/'calibration_and_history_strata.json',dict(models=result,strata_admissions=counts_by_group,
        report_sha256=sha256(ROOT/'confirmation_report.json'),test_not_evaluated=True,goal_achieved=False,
        definition='raw completed COPD-coded admission counts; not normalized values and not lifetime disease onset',
        limits=['No reselection from calibration or strata.','Already exposed validation; not blind.','Sparse stratum AUC may be unavailable.']))
    print(json.dumps(dict(status='context_calibration_and_strata_written',goal_achieved=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('tuning','confirmation','diagnostics'),required=True)
    mode=p.parse_args().mode
    diagnostics() if mode=='diagnostics' else report(mode)
