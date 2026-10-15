"""从已核验的聚合结果生成中文报告与科学图，不读取患者原文。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from patient_data.export_mimiciv import sha256

ROOT=Path('training_runs/v7_first_2026-10-15_r2')
DOC=Path('docs/V7_FIRST_EXPERIMENT_RESULTS_2026-10-15.md')


def read(path):
    return json.loads(path.read_text())


if __name__=='__main__':
    report=read(ROOT/'report.json');proof=read(ROOT/'verification.json')
    if proof['maximum_error']!=0 or proof['report_sha256']!=sha256(ROOT/'report.json'):
        raise ValueError('V7 result proof stale')
    for name,target in (('readout_verification.json','readout.json'),('diagnostics_verification.json','diagnostics.json')):
        p=read(ROOT/name)
        key='readout_sha256' if name.startswith('readout') else 'diagnostics_sha256'
        if p['maximum_error']!=0 or p[key]!=sha256(ROOT/target):raise ValueError('supplement proof stale')
    last=read(ROOT/'last_verification.json')
    if last['maximum_error']!=0 or last['report_sha256']!=sha256(ROOT/'report.json'):raise ValueError('last proof stale')
    readout=read(ROOT/'readout.json');diag=read(ROOT/'diagnostics.json')
    pre=read(ROOT/'preflight.json');coverage=read(ROOT/'input_coverage.json');cache=read(ROOT/'cache_report.json')
    source=read(ROOT/'source_report.json');bert=read(ROOT/'bert_parameter_audit.json')
    arms=('nll_control','recommended');names={'nll_control':'V7 NLL对照','recommended':'V7 推荐联合损失',
        'v6_original':'原V6','v6_history_unified':'V6统一历史','v6_symmetric':'V6对称背景'}
    histories={arm:read(ROOT/arm/'history.json') for arm in arms}
    fig,axes=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
    for arm,color in zip(arms,('#2366a6','#de7c24'),strict=True):
        h=histories[arm];x=[r['epoch'] for r in h]
        axes[0].plot(x,[r['annual']['f1'] for r in h],'-o',ms=3,color=color,label=arm)
        axes[1].plot(x,[r['tuning_ibs'] for r in h],'-o',ms=3,color=color,label=arm)
        selected=report['results'][arm]['best_epoch'];row=h[selected]
        axes[0].plot(selected,row['annual']['f1'],'*',ms=12,color=color)
        axes[1].plot(selected,row['tuning_ibs'],'*',ms=12,color=color)
        axes[2].plot(x[1:],[r['train_seconds'] for r in h[1:]],'-o',ms=3,color=color,label=arm)
    for name,color,style in (('v6_original','#888888',':'),('v6_symmetric','#222222','--')):
        m=report['reference_metrics'][name]
        axes[0].axhline(m['annual']['f1'],color=color,ls=style,label=name)
        axes[1].axhline(m['readmission']['integrated_brier_7_365'],color=color,ls=style,label=name)
    for ax,title,ylabel in zip(axes,('Annual tuning F1 (higher is better)','Readmission IBS (lower is better)','Measured training time'),
                              ('F1','IBS, days 7-365','Seconds per epoch'),strict=True):
        ax.set(title=title,xlabel='Epoch',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=7)
    png=ROOT/'first_experiment_summary.png';fig.savefig(png,dpi=180);plt.close(fig)
    rows=[]
    for name in ('v6_original','v6_history_unified','v6_symmetric',*arms):
        m=report['results'][name]['metrics'] if name in arms else report['reference_metrics'][name]
        ap=readout['metrics'][name]['readmission']['365']['average_precision']
        rows.append(f"| {names[name]} | {m['annual']['f1']:.6f} | {m['annual']['precision']:.6f} | {m['annual']['recall']:.6f} | {m['readmission']['mean_auc_7_365']:.6f} | {ap:.6f} | {m['readmission']['integrated_brier_7_365']:.6f} | {m['death']['integrated_brier_7_365']:.6f} |")
    timings=[]
    for arm in arms:
        info=report['results'][arm];h=histories[arm];times=[r['train_seconds'] for r in h if r['epoch']]
        timings.append(f"| {names[arm]} | {info['completed_epochs']} | {info['best_epoch']} | {info['optimizer_steps']} | {info['train_seconds']:.2f} | {info['evaluation_seconds']:.2f} | {info['seconds']:.2f} | {np.median(times):.2f} | {info['peak_allocated_bytes']/2**20:.1f} | {info['peak_reserved_bytes']/2**20:.1f} |")
    horizons=[]
    for day in (30,90,365):
        for arm in ('v6_original','v6_symmetric',*arms):
            p=readout['metrics'][arm]['readmission'][str(day)];f=p['tuning_best_f1']
            horizons.append(f"| {day} | {names[arm]} | {f['f1']:.6f} | {f['precision']:.6f} | {f['recall']:.6f} | {p['auc']:.6f} | {p['average_precision']:.6f} | {p['brier']:.6f} | {p['mean_predicted']:.6f} | {p['observed_fraction']:.6f} |")
    sensitivities=[]
    for arm in arms:
        for s,v in diag['results'][arm]['ablation_sensitivity'].items():
            sensitivities.append(f"| {names[arm]} | {s} | {v['annual_readmission_mean_absolute_change']:.8f} | {v['annual_readmission_max_absolute_change']:.8f} | {v['annual_death_mean_absolute_change']:.8f} |")
    intervals=[]
    for key in ('recommended_minus_nll_control','recommended_minus_v6_symmetric'):
        f=readout['annual_patient_paired'][key];b=report['paired_intervals'][key]
        intervals.append(f"- `{key}`：年度F1差{f['annual_f1_difference']:+.6f}，患者聚类95%区间{f['annual_f1_95ci']}；IBS差{b['integrated_brier_difference']:+.6f}，95%区间{b['integrated_brier_95ci']}。")
    recommended=report['results']['recommended'];control=report['results']['nll_control']
    contrast=histories['recommended'][-1]['online']
    text=f'''# V7首轮正式实验结果

2026-10-15，用户直接授权的首轮已完成。正式输出为`training_runs/v7_first_2026-10-15_r2`。V7两条件均完成实际GPU训练及早停；该单seed调参结果尚未支持替换原V6默认。

## 实际实现与冻结范围

全部有效临床表示256；8头/每头32；M4字段与住院读取、历史M4；FFN1024；历史Transformer2层；编码/attention Dropout=.1、融合/风险头=.3。冻结Bio_ClinicalBERT/RadBERT，文本768→256、影像类别聚合1536→256；BERT骨干仍是12层12头，不把V7的8头改进它内部。显式历史偏置`−softplus(lambda)×log1p(gap_days)`，是内容分数上的正值幂律时间衰减。

实际BERT参数{bert['total_parameters_including_pooler']:,}，去pooler为{bert['encoder_parameters_excluding_pooler']:,}；本轮BERT优化器参数0。V7有效可训练{pre['trainable_parameters']:,}参数，H34另冻结{pre['frozen_model_parameters']}参数。文本/影像投影{pre['optimizer_groups']['cached_projection']['parameters']:,}参数lr1e-4；其余新模块{pre['optimizer_groups']['new_modules']['parameters']:,}参数lr3e-4；AdamW衰减.01、梯度clip1。

三层读取为文本字段内M4、字段→住院M4、住院→历史M4。38药物身份、10药效父类、剂量/频率有效mask和单位因子化；保留旧七药效四来源、常规36检验九通道/EOS语境、五微生物/三支持与H34背景。药物Query读取字段查询和影像token；影像Query读取文本。冻结向量内不重建逐词交互，CXR/CT为类别聚合而非逐报告表示。

尚未接入药物升级斜率、完整route/duration或新的隐式生物标志物规则字典；本轮不能回答这些独立机制的效果。

## 数据与损失

建模队列19401出院/5351患者，训练{source['training_discharges']}出院/{source['training_patients']}患者，指定调参1531出院/402患者。全部{source['prior_visit_relations']}条训练历史关系保留，最多{source['maximum_prior_visits']}次过去住院，不截断；{source['text_visible_prior_relations']}条历史文本可见，{source['past_text_excluded_relations']}条因存档时间边界关闭文本，表格继续可用。细药训练覆盖{coverage['fine_drug_present_discharges']}出院，影像覆盖{coverage['radiology_present_discharges']}出院。严格AECOPD与竞争死亡/WorldA365标签保留，确认和历史测试不用于本轮选择。

对照：竞争NLL+1e-4修正平方。推荐：以上+1.0多期限Focal-smoothed+.1对比。Focal gamma2/alpha.75，标签平滑epsilon.05，即负/正目标.025/.975；7/14/30/60/90/180/365日×两原因等权。对比margin.5，文本↔药/检验/影像，同住院正对、不同患者负对，所有同患者负对与缺源排除，来源独立投影/L2范数。联合损失系数1/.1为首轮固定实施选择，外部报告未提供这两个系数。

两个条件同seed42新初始化，共同冻结H34，第0轮风险完全相同。最多30轮，至少8轮，patience6/min_delta1e-4，年度调参F1选点、并列选较低再入院IBS。对照最佳{control['best_epoch']}轮，推荐最佳{recommended['best_epoch']}轮；对照与推荐完成{control['completed_epochs']}/{recommended['completed_epochs']}轮。此配对分离联合损失效果；V7与V6比较仍混合了容量、输入和架构变化。

## 已选检查点结果

| 模型 | 一年F1 | PPV | 召回 | 平均AUC 7–365 | 一年AP | 再入院IBS | 死亡IBS |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

F1/阈值均在同一暴露调参折选择，不能称独立验证成绩。IBS越低越好，AUC/AP/F1越高越好。多期限和双原因完整结果见`readout.json`与各条件`result.json`，以下摘30/90/365再入院：

| 天数 | 模型 | F1 | PPV | 召回 | AUC | AP | Brier | 平均预测 | 本库观察比例 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(horizons)}

本次采用性判断：推荐方案未在年度F1与全年概率误差上超过已保存的V6对称背景参照；小幅胜过V7 NLL对照并不证明整组超参数均有效。每来源/药物语义/时间衰减的独立增量需匹配消融，不能由组合结果直接接受或否定。

{chr(10).join(intervals)}

区间为200次患者聚类配对抽样，固定已选模型/阈值；不消除调参选择偏差，单seed，未做多重比较校正。

## 实测耗时

GPU `{pre['device']}`，FP32/关闭TF32，既有BERT向量复用，不重新编码或微调BERT。正式两条件及来源核验/缓存/统计合计{report['total_run_seconds']:.2f}秒；来源核验{report['source_setup_seconds']:.2f}秒，CPU稀疏缓存{cache['seconds']:.2f}秒、{cache['cpu_tensor_bytes']/2**20:.1f}MiB。下列显存为进程Torch峰值，不含桌面等其他GPU进程。

| 条件 | 完成轮 | 最佳轮 | 更新步 | 纯训练秒 | 每轮评价合计秒 | fit总秒 | 训练中位秒/轮 | allocated MiB | reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(timings)}

这些是冻结BERT的V7训练成本；BERT微调时间尚未测量。报告、重放与实现修复时间另外记载，不把它们混入纯训练时间。

![逐轮F1、IBS与实际训练耗时](../training_runs/v7_first_2026-10-15_r2/first_experiment_summary.png)

## 来源内容敏感性

在已选best上，对当前与过去来源同时关闭其真实mask，H34基础风险保持；历史项仅关闭过去关系。以下为年度概率变化。它是内容依赖诊断，不能代替重训后的价值消融；不用于重新选点。

| 条件 | 关闭来源 | 再入院平均绝对变化 | 再入院最大绝对变化 | 死亡平均绝对变化 |
|---|---|---:|---:|---:|
{chr(10).join(sensitivities)}

已选推荐权重中，关闭药物后的年度再入院平均概率变化约0.1111个百分点，关闭过去历史约0.2898个百分点；检验/文本/影像也有非零影响。对照风险修正的记录间标准差较小，推荐权重的记录间修正变化更明显。该现象支持继续研究有效学习与来源使用，但不能把内容敏感性当作预测增量，也不能据此直接认定任一来源没有价值。

推荐末轮在线NLL{contrast['nll']:.6f}、Focal{contrast['focal']:.6f}、对比{contrast['contrast']:.6f}，正对{contrast['positive_pairs']}、合法负对{contrast['negative_pairs']}；完整分项和梯度范数见`history.json`。

## 验证与实现问题

24项相关测试全部通过，其中CPU运行暂跳过的真实CUDA精度检查另在GPU通过。真实三步预检核对所有四字段压缩参数非零梯度/更新、活跃模块更新、无悬空可训练参数、冻结H34、历史内容梯度、共同初值。新的独立进程重建原来源/合同，best每日概率/分数/选点/早停误差0；last重建对照末轮记录并再次重放每日曲线误差0；AP/校准/F1患者区间及内容敏感性也独立重算一致。

初次运行发现字段M4压缩结果被丢弃，已停止并保留源码、日志和检查点于`training_runs/v7_first_2026-10-15`。修正后两条件在r2从头初始化，不接续旧优化器或权重；初次运行不作为性能结论。该问题已加入单元回归和GPU逐参数检查。

原V6默认、旧数据/向量/权重保留；本轮未恢复长期automation，未推送。冻结的BERT骨干与现有向量仍可继续用于V7后续有明确机制的匹配实验。
'''
    DOC.write_text(text,encoding='utf-8')
    print(json.dumps(dict(report=str(DOC),plot=str(png),plot_bytes=png.stat().st_size)),flush=True)
