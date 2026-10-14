"""正式报告与直观图；旧强参照必须核对相同成员/标签后才纳入。"""
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.audited_experiment import OUTPUT,INPUT
from training.v6.evaluation import metrics,paired_interval
from training.v6.train import write_json


def run():
    root=OUTPUT;report=json.loads((root/'report.json').read_text());simple=json.loads((root/'simple_controls/report.json').read_text())
    proof=json.loads((root/'independent_verification.json').read_text())
    if proof['report_sha256']!=sha256(root/'report.json') or proof['maximum_probability_error']!=0:raise ValueError('proof missing')
    oldroot=Path('training_runs/v6_original_interaction_2026-10-12_r3');old=json.loads((oldroot/'report.json').read_text())
    oldproof=json.loads((oldroot/'independent_verification.json').read_text())
    if oldproof['report_sha256']!=sha256(oldroot/'report.json'):raise ValueError('old comparator stale')
    scores=[];oldcurves={};curves={};paired={}
    for seed in (42,7):
        with np.load(oldroot/f'seed{seed}/control/test_predictions.npz') as z:previous={k:z[k] for k in z.files}
        oldcurves[seed]=previous['curve'];score=old['models'][str(seed)]['control']['folds']['test']['readmission']
        scores.append(dict(model='original_V6',seed=seed,auc90=score['points']['90']['auc'],ibs=score['integrated_brier_7_365']))
        for arm,m in report['models'][str(seed)].items():
            with np.load(root/f'seed{seed}'/arm/'test_predictions.npz') as z:new={k:z[k] for k in z.files}
            for key in ('hadm_ids','subject_ids','times','events'):
                if not np.array_equal(new[key],previous[key]):raise ValueError('old comparison contract mismatch')
            score=m['folds']['test']['readmission'];scores.append(dict(model=arm,seed=seed,auc90=score['points']['90']['auc'],ibs=score['integrated_brier_7_365']))
            curves[seed,arm]=new['curve'];paired[f'{seed}/{arm}']=paired_interval(new['subject_ids'],new['times'],new['events'],new['curve'][:,:,1],previous['curve'][:,:,1],seed=seed)
        for name in (f'xgb{seed}',):
            with np.load(root/'simple_controls'/f'{name}_test_predictions.npz') as z:new={k:z[k] for k in z.files}
            for key in ('hadm_ids','subject_ids','times','events'):
                if not np.array_equal(new[key],previous[key]):raise ValueError('simple members wrong')
            score=simple['folds'][name]['test']['readmission'];scores.append(dict(model='XGBoost_Cox',seed=seed,auc90=score['points']['90']['auc'],ibs=score['integrated_brier_7_365']))
            curves[seed,'XGBoost_Cox']=new['curve'];paired[f'{seed}/XGBoost_Cox']=paired_interval(new['subject_ids'],new['times'],new['events'],new['curve'][:,:,1],previous['curve'][:,:,1],seed=seed)
    score=simple['folds']['cox']['test']['readmission'];scores.append(dict(model='regularized_Cox',seed=None,auc90=score['points']['90']['auc'],ibs=score['integrated_brier_7_365']))
    write_json(root/'comparison_to_strong_reference.json',dict(old_report_sha256=sha256(oldroot/'report.json'),matched_members_and_labels=True,comparisons=paired,scores=scores))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=['original_V6','full_pool','compact_pool','compact_interaction_pool','compact_interaction_cls','flat','regularized_Cox','XGBoost_Cox']
    labels=['Original V6 (strong reference)','All candidate leaves + pooling','Compact coarse/fine + pooling','Compact + interaction + pooling',
      'Compact + interaction + CLS','Shallow MLP (same chosen recipe)','Regularized cause-specific Cox','Cause-specific XGBoost-Cox']
    fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
    for j,name in enumerate(order):
        values=[r for r in scores if r['model']==name]
        for r in values:
            color='C0' if r['seed']==42 else 'C1' if r['seed']==7 else 'gray';offset=-.08 if r['seed']==42 else .08 if r['seed']==7 else 0.
            for ax,key in ((axes[0],'ibs'),(axes[1],'auc90')):ax.scatter(r[key],j+offset,c=color,s=35,label=f"Seed {r['seed']}" if j==0 else None)
    for ax in axes:ax.set_yticks(range(len(order)),labels);ax.grid(axis='x',alpha=.2);ax.legend()
    axes[0].invert_yaxis();axes[0].set_xlabel('Integrated Brier 7-365 (lower better)');axes[1].set_xlabel('AUC90 (higher better)')
    fig.suptitle('Strict AECOPD; same patients and labels; selected by validation; reused test');fig.tight_layout();fig.savefig(root/'performance_overview.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,color in (('original_V6','black'),('compact_interaction_cls','C0'),('XGBoost_Cox','C2')):
        for seed,style in ((42,'-'),(7,'--')):
            curve=oldcurves[seed] if name=='original_V6' else curves[seed,name]
            detail=metrics(previous['times'],previous['events'],curve[:,:,1]);x=np.arange(366)
            axes[0].plot(x[7:],detail['auc_curve'][7:],color=color,ls=style,label=f'{name}, seed{seed}')
            axes[1].plot(x[7:],detail['brier_curve'][7:],color=color,ls=style,label=f'{name}, seed{seed}')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7);ax.set_xlabel('Days after discharge')
    axes[0].set_ylabel('Cumulative/dynamic AUC(t)');axes[1].set_ylabel('Brier(t) (lower better)')
    fig.suptitle('Daily risk evaluation; World A; competing death included in controls');fig.tight_layout();fig.savefig(root/'daily_metrics.png',dpi=175);plt.close(fig)
    table=['|模型|AUC90：42|AUC90：7|IBS：42|IBS：7|','|---|---:|---:|---:|---:|']
    for name,label in zip(order,labels):
        values={r['seed']:r for r in scores if r['model']==name}
        if None in values:
            r=values[None];table.append(f"|{label}|{r['auc90']:.6f}（一次拟合）|—|{r['ibs']:.6f}|—|")
        else:table.append(f"|{label}|{values[42]['auc90']:.6f}|{values[7]['auc90']:.6f}|{values[42]['ibs']:.6f}|{values[7]['ibs']:.6f}|")
    text='''# 全输入审计、粗细层级及减少早期池化：完整结果

## 结论与当前决定

完成全部输入审计、新接口、四个神经候选×两种子、浅层MLP×两种子、正则双原因Cox及XGBoost-Cox对照。**没有稳定超过原V6强参照；不替换默认模型、不删除原始数据或既有编码。**粗细表示与来源互补已实现并保存，但表示正确不等于性能保证。

可以直接整理的冗余是同一分析物的高/低概念重复、同一培养mask重复，以及药物来源/计数/跨度对模型输入的膨胀。合并保留临床内容；不是把异常、罕见但重要的信号删除。简化包目前只是候选，不凭“CI包含0”证明完全无害，更不能永久删除高红细胞压积等有生物价值的指标。

## 同一任务与训练

19,401次合格出院/5,351患者，train13,592、validation2,949、test2,860，同患者不跨折。严格491.21/J44.1任意诊断位首次非转院再入院、先发生死亡为竞争事件，World A365天；不是完整院外随访。冻结同一H34（历史+当前诊断/去向），不在修正分支重复加入。保存B四字段Bio_ClinicalBERT全体片段，不新编码/替换Qwen。

每个神经臂完整100epoch，种子42/7、LR1e-4、batch256、AdamW1e-4、warmup5+cosine下限0.1、梯度clip1、风险头零初始化。所有权重按验证IBS7–365选择，不以测试挑字段/选epoch。验证可选出CLS为本轮新候选最佳，但原强参照两种子平均验证IBS0.094382仍优于它的0.094788，不能只在新的弱候选中比较后宣布突破。

## 检验、药物与其他信息

完整中文全字段表、人数/缺失/异常/阳性、来源和取舍：[审计数据README](../patient_data/processed/v6_audited_coarse_fine_2026-10-12_r2/README.md)、[CSV](../patient_data/processed/v6_audited_coarse_fine_2026-10-12_r2/all_inputs.csv)。本表是已采用+近期候选并集，不声称默认使用了全部39轴/38药物。

39轴/来源→34概念（5对高低轴合并），保留两个异常方向，不平均。候选简化包暂屏蔽红细胞压积、MCV、单核百分比、总蛋白→30概念。红细胞压积/MCV/单核并不稀疏，理由是部分重叠/非特异的复杂度测试；高红细胞压积慢性缺氧背景不能由低Hb轴完全替代，因此本轮不永久删除。CRP851次、纤维蛋白原886次、NT-proBNP1,305次、EOS绝对2,743次虽少但医学理由明确，保留。

药效父层10类+38药品身份；不足20训练患者的10细叶不生成独立细表示，父药效保留，简化候选28细叶。初始38候选并非原强参照的默认38药物。单次剂量/频率两数值+2mask+7单位类别为11维，去掉原23维中的来源、记录数/日数、跨度及累计量。实际/医嘱互补但不证明所有计划执行；单位仅可核验换算，复方不重复累计。万古霉素旧途径混合，新接口只采纳有效IV医嘱。pharmacy与prescriptions已识别药物不一致的全量关联核对为0组；新Rx剂量不是凭药名推猜浓度。

微生物5内容标记保留，培养mask只传一次。呼吸细菌培养可用1,222次住院；常见菌71、铜绿114、其他菌252阳性；病毒可用618/阳性38，耐药有可评价结果334/阳性236。没有检查不算阴性，阳性不证明定植vs感染；本轮不继续拆菌种。三类呼吸支持仍保留，不把未有ICU记录当无氧疗。

**确实还有未采用信息**：已保存低频RR3,473、SpO₂3,477、FiO₂1,792、普通氧流3,147次住院，以及通气时长/拔管；胸片12,346份/4,888次住院、胸部CT2,118份/1,872次住院及RadBERT/RadGraph输出；肺功能/家庭氧疗/吸入技术/照护等候选原句。已有源不等于已入网或已证明增量。OMR BMI/体重、ED/程序/转科是官方可用候选，但本轮实时PG查询缺进程凭据，仅核了已有导出/历史证据，不伪造当前队列覆盖；不复用旧队列OMR数字。最后ICU数值不能当出院值，混合标本pO₂不能当PaO₂，也不能不核同时FiO₂就计算P/F。

## 交互/读出的准确结构

每个检验概念24槽共享MLP24→32→128；每个药品11→32→128，加医学父类别/身份嵌入。24槽是兼容异构摘要的固定填槽，不等于每槽都有独立网络。药效父层2→128；微生物8→128、支持6→128；每个保存文本片段768→128，片段/字段身份保留。检验父层额外汇总只作上下文，细叶不丢弃。

池化臂195,397可训练参数；1层4头Transformer/FFN256交互后池化327,877；同层CLS最终读出328,005。没有人为顺序位置编码。CLS仅取消模态早期压缩，不取消最后风险读出，也未取消旧BERT片段内部编码聚合。四维修正头128→32→4，修正两种事件各自幅度与形状，冻结基线不被训练。浅MLP为完整表格+四字段向量均值→64→128→四修正，295,524参数；浅不一定总参数少，因为首层输入多。

```mermaid
flowchart LR
  H[冻结H34基线] --> R[两个原因联合连续风险]
  L[检验细概念+医学父层] --> T[保留各概念/片段]
  M[药品剂量频率+药效父层] --> T
  B[五菌标记/三支持/保存文本片段] --> T
  T --> C[候选:先交互 再池化或CLS最终读出]
  C --> D[4维零初始化风险修正]
  D --> R
  R --> P[0–365天CIF/每天事件概率/条件概率]
```

文献与外推限制见[临床/层级/池化依据](V6_AUDIT_LITERATURE_2026-10-12.md)。注意力学习条件关联，不证明药物疗效，亦不消除测量选择与适应症混杂。

## 相同测试集的实际成绩

'''+ '\n'.join(table)+'''

IBS是7–365天整个风险曲线的均方概率误差汇总，不是“第90天误差”。AUC90只作易懂的明确节点报告，模型仍每日输出。完整文件包含30/90/180/365天AUC/Brier、竞争风险C-index、死亡与校准分组。

简化包相对完整包：种子42 ΔAUC90=−0.000290、ΔIBS=+0.00000359；种子7 +0.000219、−0.00001234。200次患者聚类配对区间均跨0，支持“未看到明显损失/提升”，不支持“证明四指标无价值”。这是预定包的整组重训，不是四指标逐个效应归因，也不是单因素P值/LASSO/SHAP筛选。

交互后池化相对简化池化：两种子AUC方向不一致且全年IBS更差；CLS相对交互池化AUC两种子改善，但仍未超过原V6强参照。不能由局部比较说“不池化一定更好”。验证选择CLS和测试差异都完整公开，没有事后用测试选回其他臂。

## 简单算法对照及故障修复

神经臂选好后才执行浅MLP、双原因L2 Cox及XGBoost-Cox。Cox使用H34+训练PCA临床16+文本16（66特征），L2 10/100按验证选100；一次确定性拟合，不伪装成两种子。XGB使用原表格+H34+训练PCA文本16（1,390特征），深度2/3、最多300树、LR0.03、min_child_weight10、lambda10、subsample/colsample0.8，验证原因Cox似然early-stop30，再按验证联合CIF的IBS选配置。两原因危险率/Breslow一起计算CIF，死亡不是普通“阴性”输入，仍有每天概率。

Cox/XGB重新拟合所有H34与临床特征，不是冻结历史；PCA/最终汇总策略也不同，因此是成熟算法整体对照，不是只替换一个层或精确论文复现。XGB原sklearn接口load_model因当前sklearn1.6 tags不兼容而失败；保留已完成10神经模型，复用已拟合Cox/PCA，仅简单树候选重拟合并原生Booster UBJ保存/重载。没有降级全局依赖，也没有覆盖旧哈希源文件。

## 核验与采用边界

新进程20折神经预测/指标逐项复现，最大概率误差0；10个H34冻结权重不变；原始快照17文件哈希一致；单次/计划剂量单位与训练尺度检查、排列/缺失/零初始化/两原因质量守恒等7个新定向测试和27个相关回归测试通过。Cox/原生树save/load重放一致。Bootstrap200为条件于已拟合模型的患者聚类探索区间，无多重比较校正；测试已反复使用，不能做临床部署承诺。

目前决定：**继续保留原V6强参照，不采用新的整套层级/CLS替换。**新接口和审计成果保留，明确合并/互补规则可以复用；不因新网络未胜就永久删医学核心变量。后续优先核可用低频氧合/支持及真正独立的出院管理语义，避免继续同时改数据、架构、损失、队列后无法归因。

## 文件入口

- [性能直观图](../training_runs/v6_audited_coarse_fine_2026-10-12/performance_overview.png)
- [每日AUC/Brier曲线](../training_runs/v6_audited_coarse_fine_2026-10-12/daily_metrics.png)
- [配对差异及区间图](../training_runs/v6_audited_coarse_fine_2026-10-12/paired_comparison.png)
- [完整100轮曲线](../training_runs/v6_audited_coarse_fine_2026-10-12/training.png)
- 正式结果：training_runs/v6_audited_coarse_fine_2026-10-12/report.json、simple_controls/report.json、independent_verification.json。
- 数据：patient_data/processed/v6_audited_coarse_fine_2026-10-12_r2；无r2为初始审计，不使用其万古霉素接口。
'''
    Path('docs/V6_AUDITED_COARSE_FINE_RESULTS_2026-10-12.md').write_text(text,encoding='utf-8')
    (root/'README.md').write_text('# 正式全输入审计/粗细/交互及简单对照\n\n全部已完成，10神经模型×100轮，两原因Cox和XGB对照。没有超过原V6强参照，不覆盖默认。\n\n方法、全部表格与限制见docs/V6_AUDITED_COARSE_FINE_RESULTS_2026-10-12.md；当前数据仅用patient_data/processed/v6_audited_coarse_fine_2026-10-12_r2。\n\n新进程20折重放误差0，哈希/选轮/冻结检查通过。XGB包装接口修复记录在simple_controls/report.json。所有预测含0–365天风险，不仅90天。\n',encoding='utf-8')
    print(json.dumps(dict(status='formal_report_and_figures_completed',matched_strong_reference=True)),flush=True)


if __name__=='__main__':run()
