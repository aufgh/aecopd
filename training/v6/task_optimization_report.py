"""第二批三种子可视化与锁定确认报告；不依据图表重新选模。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.task_optimization_experiment import ROOT,SEEDS
from training.v6.task_optimization import TRIALS


def averages(models,names):
    result={}
    for name in names:
        result[name]={}
        for cause in ('readmission','death'):
            values=[models[str(s)][name]['metrics'][cause] for s in SEEDS]
            points={}
            for day in (30,90,180,365):
                points[str(day)]={}
                for key in ('auc','brier','mean_predicted','observed_world_a_fraction'):
                    xs=[x['points'][str(day)][key] for x in values]
                    points[str(day)][key]=float(np.mean(xs)) if all(v is not None for v in xs) else None
            result[name][cause]=dict(ibs=float(np.mean([x['integrated_brier_7_365'] for x in values])),points=points)
    return result


def prepare_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def tuning():
    proof=json.loads((ROOT/'tuning_verification.json').read_text())
    if (proof['status']!='tuning_independently_verified' or proof['models']!=24 or
            proof['selected_before_confirmation_sha256']!=sha256(ROOT/'selected_before_confirmation.json') or
            proof['tuning_report_sha256']!=sha256(ROOT/'tuning_report.json')):
        raise ValueError('full independent tuning verification required')
    r=json.loads((ROOT/'tuning_report.json').read_text())
    selected=json.loads((ROOT/'selected_before_confirmation.json').read_text())
    names=[x.name for x in TRIALS]
    if r['selected']!=selected['selected']:raise ValueError('report choice differs')
    for seed in SEEDS:
        for name in names:
            info=r['models'][str(seed)][name]['training']
            if info!=selected['results'][str(seed)][name] or sha256(ROOT/f'seed{seed}'/name/'best.pt')!=info['checkpoint_sha256']:
                raise ValueError('tuning result changed')
    stats=averages(r['models'],names);plt=prepare_plot()
    fig,axes=plt.subplots(1,2,figsize=(13,7),sharey=True)
    for j,name in enumerate(names):
        differences=[[],[]]
        for index,seed in enumerate(SEEDS):
            p=r['models'][str(seed)][name]['metrics']['readmission']
            base=r['models'][str(seed)]['control']['metrics']['readmission']
            values=(p['integrated_brier_7_365']-base['integrated_brier_7_365'],
                    p['points']['90']['auc']-base['points']['90']['auc'])
            for ax,delta,collection in zip(axes,values,differences):
                collection.append(delta)
                ax.scatter(delta,j+(index-1)*.12,s=23,color=f'C{index}',label=f'Seed {seed}' if j==0 else None)
        for ax,values in zip(axes,differences):
            ax.scatter(np.mean(values),j,marker='D',color='black',s=46,label='Three-seed mean' if j==0 else None)
    for ax in axes:
        ax.axvline(0,color='gray',lw=1);ax.grid(axis='x',alpha=.25);ax.set_yticks(range(len(names)),names)
    axes[0].invert_yaxis();axes[0].set_xlabel('IBS7-365: candidate minus matched V6 (negative better)')
    axes[1].set_xlabel('AUC90: candidate minus matched V6 (positive better)')
    fig.suptitle('Cause-specific heads / shared-gradient optimization; tuning validation only\n'
                 '100 full epochs per model; no confidence intervals; not confirmed')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.90),ncol=4)
    fig.tight_layout(rect=(0,0,1,.84));fig.savefig(ROOT/'tuning_increment.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(15,13))
    for row,seed in enumerate(SEEDS):
        for name in names:
            h=json.loads((ROOT/f'seed{seed}'/name/'history.json').read_text())
            axes[row,0].plot([x['epoch'] for x in h[1:]],[x['train_nll'] for x in h[1:]],label=name,lw=1.2)
            axes[row,1].plot([x['epoch'] for x in h],[x['tuning_ibs'] for x in h],label=name,lw=1.2)
        for ax in axes[row]:ax.set_xlabel('Epoch');ax.grid(alpha=.25)
        axes[row,0].set_title(f'Seed {seed}: online training joint NLL')
        axes[row,1].set_title(f'Seed {seed}: tuning validation IBS7-365')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.965),ncol=4)
    fig.suptitle('24 complete trainings; checkpoints selected by tuning IBS, not last epoch')
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(ROOT/'training_curves.png',dpi=150);plt.close(fig)
    lines=['# 第二批双原因表示与梯度：调参结果','',
           '八条件×三种子各100完整epoch已完成。仅调参验证子集，不是确认或测试结果。三临床种子复用两个固定H34来源。',
           '',f'事前规则选择：`{r["selected"]}`；确认前不依据图表或某个AUC更换选择。',
           '', '| 条件 | AUC30↑ | AUC90↑ | AUC180↑ | AUC365↑ | Brier90↓ | IBS7–365↓ | 死亡IBS↓ |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'不可计算' if x is None else f'{x:.6f}'
    for name in names:
        s=stats[name];p=s['readmission']['points']
        lines.append('| '+name+' | '+' | '.join([fmt(p[str(d)]['auc']) for d in (30,90,180,365)]+
            [fmt(p['90']['brier']),fmt(s['readmission']['ibs']),fmt(s['death']['ibs'])])+' |')
    lines+=['','## 图与边界','',
            '![三种子配对差值](tuning_increment.png)','',
            '每个小点是一个种子，黑色菱形是三种子均值；不是患者置信区间。左负右正才是改善方向。',
            '', '![完整100轮曲线](training_curves.png)','',
            '训练NLL为启用dropout的在线批次均值，验证IBS为eval完整调参子集；后期验证恶化不能靠训练损失下降解释成成功。',
            '', '24模型新进程概率重放、指标/每日接口、选点/冻结基线与源码合同核验完成。',
            '确认子集和历史测试没有用于本批搜索；原验证曾曝光，拆分不是新盲测。',
            '下一步按锁定选择一次确认，只有实质信号才进行五种子稳定评估；不覆盖原V6。']
    (ROOT/'TUNING_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='verified_tuning_figures_written',selected=r['selected'],goal_achieved=False)),flush=True)


def confirmation():
    r=json.loads((ROOT/'confirmation_report.json').read_text())
    proof=json.loads((ROOT/'confirmation_verification.json').read_text())
    if proof['report_sha256']!=sha256(ROOT/'confirmation_report.json'):
        raise ValueError('confirmation replay stale')
    names=['control',*r['lock']['nominated'],'official_V6'];stats=averages(r['models'],names)
    chosen=r['lock']['selected']
    lines=['# 第二批：锁定三种子确认','',
        f'候选`{chosen}`由三种子调参平均IBS锁定，确认仅评价该候选及匹配/旧正式V6，不允许换名挑选。',
        '', '| 模型 | AUC90↑ | Brier90↓ | 再入院IBS↓ | 死亡IBS↓ |', '|---|---:|---:|---:|---:|']
    for name in names:
        s=stats[name];p=s['readmission']['points']['90']
        lines.append(f'| {name} | {p["auc"]:.6f} | {p["brier"]:.6f} | {s["readmission"]["ibs"]:.6f} | {s["death"]["ibs"]:.6f} |')
    lines+=['', '## 种子方向与患者配对区间','',
            '| 临床种子 | 参照 | AUC90差值（正更好） | IBS差值（负更好） | AUC90患者95%区间 | IBS患者95%区间 |',
            '|---|---|---:|---:|---|---|']
    for seed in SEEDS:
        for ref in ('control','official_V6'):
            p=r['comparisons'][str(seed)][ref]
            lines.append(f'| {seed} | {ref} | {p["auc90_difference"]:.6f} | {p["integrated_brier_difference"]:.6f} | {p["auc90_95ci"]} | {p["integrated_brier_95ci"]} |')
    lines+=['', '三种子只是筛选确认，不满足长期五种子稳定标准。两个基础来源被复用，不能当作三次独立划分。',
            '患者配对抽样200次区间条件于已拟合模型，不消除配置搜索选择偏倚、随访不完整或多重比较问题。',
            '尚需校准/病史层内审计，确认出现实质信号才扩大种子，最后锁定历史测试；旧测试不是新盲测。',
            '没有稳定证据前，原V6仍是默认参照，长期目标未完成。']
    (ROOT/'CONFIRMATION_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='verified_confirmation_report_written',goal_achieved=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('tuning','confirmation'),required=True)
    tuning() if parser.parse_args().mode=='tuning' else confirmation()
