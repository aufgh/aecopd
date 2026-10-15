"""直接列出损失组成，重算最佳/末轮读出并生成中文结果。"""
import argparse
import json
from pathlib import Path

import numpy as np

from patient_data.export_mimiciv import sha256
from training.v6.horizon_objective import binary_metrics
from training.v6.multihorizon_readout import horizon_readout
from training.v6.unified_longitudinal import f1_at
from training.v6.evaluation import paired_interval
from training.v6.train import write_json
from training.v7 import experiment as first
from training.v7.loss_comparison import ROOT,CONDITIONS,OLD_LABELS
from training.v7.losses import HORIZONS

DOC=Path('docs/V7_LOSS_COMPARISON_RESULTS_2026-10-15.md')
NAMES={**OLD_LABELS,**{k:v['label'] for k,v in CONDITIONS.items()}}
ORDER=('nll_control','recommended','nll_focal_smoothing','focal_smoothing','focal_smoothing_contrast')


def read(path):return json.loads(path.read_text())


def load(path):
    with np.load(path) as z:return {k:z[k] for k in ('hadm_ids','subject_ids','times','events','curve')}


def calculate():
    report=read(ROOT/'report.json');proof=read(ROOT/'verification.json')
    if proof['maximum_error']!=0 or proof['report_sha256']!=sha256(ROOT/'report.json'):
        raise ValueError('loss comparison proof stale')
    arrays={};lasts={};paths=[Path(__file__),ROOT/'report.json',ROOT/'verification.json',
        Path('training/v6/horizon_objective.py'),Path('training/v6/multihorizon_readout.py'),
        Path('training/v6/evaluation.py'),Path('training/v6/unified_longitudinal.py')]
    for name in ORDER:
        path=(ROOT/name if name in CONDITIONS else first.ROOT/name)/'tuning_predictions.npz'
        arrays[name]=load(path);paths.append(path)
        if name in CONDITIONS:
            path=ROOT/name/'last_predictions.npz';lasts[name]=load(path);paths.append(path)
    base=arrays['nll_control']
    for a in [*arrays.values(),*lasts.values()]:
        for key in ('hadm_ids','subject_ids','times','events'):np.testing.assert_array_equal(a[key],base[key])
    metrics={};last_metrics={}
    for names,targets,which in ((arrays,metrics,'best'),(lasts,last_metrics,'last')):
        for name,a in names.items():
            info=(report['results'] if name in CONDITIONS else report['reused_results'])[name]
            saved=info['metrics' if which=='best' else 'last_metrics']
            targets[name]={}
            for label,cause in (('readmission',1),('death',2)):
                points={}
                for day in HORIZONS:
                    p=a['curve'][:,day,cause];y=((a['events']==cause)&(a['times']<=day)).astype(np.int8)
                    point=binary_metrics(y,p)
                    for old,new in (('auc90','auc'),('brier90','brier'),('logloss90','logloss')):point[new]=point.pop(old)
                    point['decisions']=horizon_readout(a,day,cause,saved['annual']['threshold'] if cause==1 else .2)
                    points[str(day)]=point
                    if name in CONDITIONS:
                        expected=info['multihorizon' if which=='best' else 'last_multihorizon'][label][str(day)]
                        if point['decisions']!=expected:raise ValueError('multi-horizon readout changed')
                targets[name][label]=points
    paired={}
    for key,expected in report['paired_intervals'].items():
        left,right=key.split('_minus_')
        actual=paired_interval(base['subject_ids'],base['times'],base['events'],arrays[left]['curve'][:,:,1],arrays[right]['curve'][:,:,1])
        if actual!=expected:raise ValueError('paired IBS/AUC readout differs')
        people,inverse=np.unique(base['subject_ids'],return_inverse=True)
        groups=[np.flatnonzero(inverse==i) for i in range(len(people))]
        y=(base['events']==1)&(base['times']<=365)
        thresholds={name:((report['results'] if name in CONDITIONS else report['reused_results'])[name]['metrics']['annual']['threshold']) for name in (left,right)}
        def delta(ix):
            return f1_at(y[ix],arrays[left]['curve'][ix,365,1],thresholds[left])['f1']-f1_at(y[ix],arrays[right]['curve'][ix,365,1],thresholds[right])['f1']
        rng=np.random.default_rng(42);samples=[]
        for _ in range(200):
            ix=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]);samples.append(delta(ix))
        paired[key]=dict(**actual,annual_f1_difference=delta(np.arange(len(y))),annual_f1_95ci=np.quantile(samples,[.025,.975]).tolist(),
            fixed_selected_checkpoints_and_thresholds=True)
    return dict(source_sha256={str(p):sha256(p) for p in paths},metrics=metrics,last_metrics=last_metrics,
        paired=paired,new_training=0,new_prediction_calibration=0,
        limits='single seed, exposed tuning fold, postselection descriptive calibration fit only, no new independent validation')


def publish():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=read(ROOT/'report.json');readout=read(ROOT/'readout.json');proof=read(ROOT/'readout_verification.json')
    if proof['maximum_error']!=0 or proof['readout_sha256']!=sha256(ROOT/'readout.json'):raise ValueError('readout not verified')
    probe=read(ROOT/'probability_probe.json');pre=read(ROOT/'preflight.json')
    infos={**report['reused_results'],**report['results']}
    rows=[];last_rows=[];time_rows=[];horizon_rows=[]
    for name in ORDER:
        info=infos[name];m=info['metrics'];ap=readout['metrics'][name]['readmission']['365']['average_precision']
        epoch=str(info['best_epoch'])+('（初始化）' if info['best_epoch']==0 else '')
        rows.append(f"| {NAMES[name]} | {epoch} | {m['annual']['f1']:.6f} | {m['readmission']['mean_auc_7_365']:.6f} | {ap:.6f} | {m['readmission']['integrated_brier_7_365']:.6f} | {m['death']['integrated_brier_7_365']:.6f} |")
        if name in CONDITIONS:
            m=info['last_metrics']
            last_rows.append(f"| {NAMES[name]} | {info['completed_epochs']} | {m['annual']['f1']:.6f} | {m['readmission']['integrated_brier_7_365']:.6f} | {m['death']['integrated_brier_7_365']:.6f} |")
            time_rows.append(f"| {NAMES[name]} | {info['completed_epochs']} | {info['optimizer_steps']} | {info['train_seconds']:.2f} | {info['seconds']:.2f} |")
        for day in (30,90,365):
            p=readout['metrics'][name]['readmission'][str(day)];f=p['decisions']['tuning_best_f1']
            horizon_rows.append(f"| {NAMES[name]} | {day} | {f['f1']:.6f} | {f['precision']:.6f} | {f['recall']:.6f} | {p['auc']:.6f} | {p['brier']:.6f} |")
    pair_rows=[]
    for key,v in readout['paired'].items():
        left,right=key.split('_minus_')
        pair_rows.append(f"- {NAMES[left]} 减 {NAMES[right]}：一年F1差{v['annual_f1_difference']:+.6f}、95%区间{v['annual_f1_95ci']}；再入院IBS差{v['integrated_brier_difference']:+.6f}、95%区间{v['integrated_brier_95ci']}。")
    fig,axes=plt.subplots(1,3,figsize=(15.5,4.8),layout='constrained')
    english={'nll_control':'Original survival loss','recommended':'Original + Focal + smoothing + contrast',
        'nll_focal_smoothing':'Original + Focal + smoothing','focal_smoothing':'Focal + smoothing',
        'focal_smoothing_contrast':'Focal + smoothing + contrast'}
    for name,color in zip(ORDER,('#777777','#222222','#2275b8','#c43d3d','#dd8c26'),strict=True):
        folder=ROOT/name if name in CONDITIONS else first.ROOT/name
        h=read(folder/'history.json');x=[r['epoch'] for r in h]
        axes[0].plot(x,[r['annual']['f1'] for r in h],'-o',ms=3,color=color,label=english[name])
        axes[1].plot(x,[r['tuning_ibs'] for r in h],'-o',ms=3,color=color,label=english[name])
        if name!='nll_control':axes[2].plot(x[1:],[r['online']['focal'] for r in h[1:]],'-o',ms=3,color=color,label=english[name])
    for ax,title,ylabel in zip(axes,('Annual tuning F1','Readmission probability error','Training Focal term'),('F1, higher better','IBS, lower better','Focal loss, unweighted'),strict=True):
        ax.set(title=title,xlabel='Epoch',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=6.4)
    png=ROOT/'loss_comparison.png';fig.savefig(png,dpi=170);plt.close(fig)
    probe_rows=[f"| {100*r['true_rate']:.0f}% | {100*r['focal_gamma2_alpha075_smoothing005_optimum']:.2f}% |" for r in probe['rows'] if r['true_rate'] in (.01,.05,.1)]
    gradient=pre['initial_gradient_diagnostic']
    text=f'''# V7损失组成核对与补充实验

2026-10-15。用户要求继续研究，并要求直接解释实际损失组成，追问为何没有采用外部报告推荐的损失。此次固定V7架构/输入/初值，只改变原竞争风险损失、Focal和平滑标签、对比项的组合，补做三次训练。

## 先回答原问题

上轮用过外部报告给出的Focal gamma2/alpha.75、Label Smoothing epsilon.05与对比margin.5。但实际总损失是：

`原竞争风险损失 + Focal(平滑标签) + .1×对比 + 1e-4×风险修正平方`。

其中，保留原竞争风险损失、设置对比相对权重.1、选用七个期限/两原因适配都属于实施选择，不能用“推荐联合损失”这种简称代替解释，更不能把上轮结果当作只用外部建议损失的效果。保留原损失的理由是增加对具体事件时间与竞争死亡的直接监督；它不是训练当前风险模型的必要条件，是否有益须实测。

[外部研究2第97行](<../../研究 2.md>)没有给完整总损失/对比配对及权重。0.5是距离裕度，.1才是此处对比损失相对权重。标签平滑先将二元0/1目标变为.025/.975，再计算Focal，不是额外独立加一个未知损失。本次仍固定.1来隔离原竞争风险损失是否加入，并不把.1称为原文给定值。锁定合同把该行误标为98，实际定位为97，参数内容无差别；原合同保留以便核验。

Focal减少易分类样本对训练的影响，原始证据来自分类/检测，不保证本队列PPV或风险概率必然改善。[Focal原论文](https://arxiv.org/abs/1708.02002)。均匀标签平滑按`(1-epsilon)y+epsilon/K`定义；本次K=2。[标签平滑原论文第7节](https://arxiv.org/pdf/1512.00567)。两者结合采用上轮明确的软标签公式，原报告未定义唯一结合方式。

## 实验范围与结果

同19401建模出院/5351患者，train13592、指定调参1531/402患者；冻结BERT/RadBERT/H34，256维/8头/M4/FFN1024及全部历史保持，seed42新初始化，差分LR/Dropout/优化器不变。新增三种：Focal＋标签平滑＋.1对比；Focal＋标签平滑；原竞争风险损失＋Focal＋标签平滑。共同风险修正平方系数1e-4仍保留。

上一轮两个已核验条件直接复用，不重训。所有新模型第0轮风险逐项等于上一轮；相同最多30/min8/patience6早停，仍按年度调参F1选点、并列低IBS。

| 实际损失组成 | 最佳轮 | 一年F1 | 平均AUC 7–365 | 一年AP | 再入院IBS | 死亡IBS |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

**最佳0表示初始化，没有接受训练后的临床权重；不能把该行分数当作Focal训练带来的收益。** 对应真实训练末轮另外列出：

| 实际损失组成 | 实际完成轮 | 末轮一年F1 | 末轮再入院IBS | 末轮死亡IBS |
|---|---:|---:|---:|---:|
{chr(10).join(last_rows)}

Focal直接替代原损失的两个条件均实际完成训练，未超过初始化；这只说明当前V7、参数与七期限双原因适配没有得到可接受收益，不能据此否定Focal在其他分类设计中的价值，也不能独立否定标签平滑或对比学习。已保存的原V6/对称背景仍作强参照，默认不替换。

| 实际损失组成 | 天数 | F1 | PPV | 召回 | AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(horizon_rows)}

{chr(10).join(pair_rows)}

200次患者聚类配对区间，固定已选阈值/模型，单seed暴露调参，不能替代独立验证。两个直接Focal条件均选第0轮，故两者区间为0只反映共同初始化，不能证明两种训练等效。全部7/14/30/60/90/180/365读出、AP/校准见`readout.json`。

![各组成的F1、概率误差与Focal训练项](../training_runs/v7_loss_comparison_2026-10-15/loss_comparison.png)

## 为什么损失下降，风险概率却可能更差

按本次明确公式计算一个简化例子：假设单一二元任务中，所有人具有相同真实发生率，模型只能输出一个分数。对该参数组合取期望损失并数值求最小值，可得：

| 假设真实发生率 | 本次Focal＋平滑公式偏好的分数 |
|---|---:|
{chr(10).join(probe_rows)}

这是独立数学例子，不是本队列患者预测，也不保证神经网络达到这些最优值。它说明加权分类分数不能自动当作真实风险概率；把该损失直接用于多期限风险曲线，需要另外核对概率误差。公式、数值重算和普通二元交叉熵的最小值检验保存在`probability_probe.json`及独立证明。

同一初始、含较长历史/影像的真实预检批次，各项梯度范数：原竞争风险损失{gradient['raw_norms']['nll']:.6f}，Focal {gradient['raw_norms']['focal']:.6f}，.1加权对比{gradient['optimized_contrast_norm']:.6f}；原损失与Focal方向余弦{gradient['nll_focal_cosine']:.6f}。负值表示该批次存在更新方向冲突，不能用上轮混合训练代替只用Focal。该批次并非总体随机样本，不能把此数值外推全部训练；也未据此调权重。

## 时间与核验

| 新训练损失组成 | 完成轮 | 更新步 | 纯训练秒 | 含评价保存fit秒 |
|---|---:|---:|---:|---:|
{chr(10).join(time_rows)}

此次三次训练/核验前统计合计{report['seconds']:.2f}秒，RTX4070Ti，冻结向量复用。15相关测试通过，含去掉原损失后的梯度等价/时间约束排除、旧公式一致性和来源mask回归。实际GPU三种均验证有效风险/临床路径更新、基础风险冻结；不加对比时仅其独立投影头按设计没有梯度。

新的独立GPU进程重建来源/合同及best/last，保存幅度/shape/366日概率/每日事件概率和条件概率误差0；全部基础指标/早停/选点一致。补充读出逐项核对保存的最佳/末轮多期限结果、患者配对统计并独立重算一致。未使用确认/测试选点，未恢复长期automation，原资产保留。
'''
    DOC.write_text(text,encoding='utf-8')
    print(json.dumps(dict(report=str(DOC),plot=str(png)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('calculate','verify','publish'),required=True);args=p.parse_args()
    if args.mode=='publish':publish()
    else:
        result=calculate();target=ROOT/'readout.json'
        if args.mode=='calculate':
            if target.exists():raise FileExistsError('readout exists; use verify')
            write_json(target,result);print('损失组成读出已保存')
        else:
            if read(target)!=result:raise ValueError('independent loss readout differs')
            write_json(ROOT/'readout_verification.json',dict(maximum_error=0,readout_sha256=sha256(target),new_training=0))
            print('损失组成读出独立重算一致')
