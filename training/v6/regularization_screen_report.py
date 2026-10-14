"""已核验首批调参结果图；不读取确认或测试选择条件。"""
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.regularization_search import ROOT
from training.v6.train import write_json


def finish():
    r=json.loads((ROOT/'screening_report.json').read_text())
    proof=json.loads((ROOT/'screening_verification.json').read_text())
    if proof['screening_report_sha256']!=sha256(ROOT/'screening_report.json') or proof['models']!=26:
        raise ValueError('complete independent verification required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(r['results'])
    control=r['results']['v6_control']['tuning']
    colors=['C0' if n.startswith('v6_') else 'C1' if n.startswith('single_') else 'C3' if n.startswith('A_') else 'C4' for n in names]
    fig,axes=plt.subplots(1,2,figsize=(15,11),sharey=True)
    for j,name in enumerate(names):
        x=r['results'][name]['tuning']
        axes[0].scatter(x['ibs']-control['ibs'],j,color=colors[j],marker='D')
        axes[1].scatter(x['auc90']-control['auc90'],j,color=colors[j],marker='D')
    for ax in axes:
        ax.axvline(0,color='black',lw=1);ax.set_yticks(range(len(names)),names);ax.grid(axis='x',alpha=.25)
    axes[0].invert_yaxis();axes[0].set_xlabel('Tuning IBS: candidate minus matched V6 (negative better)')
    axes[1].set_xlabel('Tuning AUC90: candidate minus matched V6 (positive better)')
    fig.suptitle('Regularization screening; seed 42; tuning subset only; not confirmed; no confidence intervals')
    fig.tight_layout();fig.savefig(ROOT/'tuning_increment.png',dpi=170);plt.close(fig)
    chosen=['v6_control',*r['nominated_before_confirmation']]
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for name in chosen:
        history=json.loads((ROOT/name/'history.json').read_text())
        axes[0].plot([x['epoch'] for x in history],[x['train_fixed2048_nll'] for x in history],label=name)
        axes[1].plot([x['epoch'] for x in history],[x['tuning_ibs'] for x in history],label=name)
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_ylabel('Fixed training sample NLL');axes[1].set_ylabel('Tuning validation IBS7-365')
    fig.suptitle('60 full epochs; nominees locked by tuning IBS before confirmation')
    fig.tight_layout();fig.savefig(ROOT/'nominee_training.png',dpi=180);plt.close(fig)
    lines=['# 首批26项正则化搜索：调参阶段结果','',
        '26条件各60完整epoch、单种子42。这里只展示调参半折，不是测试结果，也未证明稳定提升。确认名单由训练结束后的固定规则选出，未根据确认分数替换。',
        '',f'入围名单：`{r["nominated_before_confirmation"]}`。',
        '', '| 条件 | 最佳epoch | 调参IBS↓ | 调参AUC90↑ | Brier90↓ | GPU训练分钟 |',
        '|---|---:|---:|---:|---:|---:|']
    for name in r['ranked_by_tuning_ibs']:
        x=r['results'][name];p=x['tuning']
        lines.append(f'| {name} | {x["best_epoch"]} | {p["ibs"]:.6f} | {p["auc90"]:.6f} | {p["brier90"]:.6f} | {x["elapsed_seconds"]/60:.2f} |')
    lines+=['','## 证据与解释边界','',
        f'- 新进程26模型重载，最大概率误差{proof["maximum_error"]:.3g}，完整epoch、选点、冻结权重、数据和调参指标一致。',
        '- 同一患者全部出院在同一个验证半折；调参1,531次、确认1,418次，原训练13,592次不变。',
        '- 不能将这里的IBS/AUC与原2,949次完整验证或2,860次测试的指标直接比较，病例组成不同。',
        '- 全部候选与本批匹配V6共同输入与冻结H34；旧正式V6仍须在确认与最终评价中作为更强参照。',
        '- 固定排名选取最多三个优于匹配对照IBS的候选，没有实现“机制多样性”过滤，不在结果后改规则。',
        '- 原验证集已有历史曝光，拆分不能变成新的盲测。首批未读取/评价测试指标，确认仅对锁定候选计算。',
        '- 更低训练损失不等于更好泛化；多数最佳epoch早于60，不代表提前停训。',
        '- 下一步锁定确认、五临床种子与两个固定H34来源配对、校准/既往次数层内审计，最后才考虑历史测试。',
        '', '## 文件','',
        '- `screening_report.json`、`screening_verification.json`、`validation_partition.npz`、`state.json`。',
        '- `tuning_increment.png`：单种子调参差值，不是置信区间或变量重要性。',
        '- `nominee_training.png`：匹配V6与入围候选完整60轮曲线。',
        '- 每条件的best.pt、history.json、training_result.json及tuning_predictions.npz。',
        '', '长期目标仍未完成；本页是筛选证据，不更换默认模型。']
    content='\n'.join(lines)+'\n'
    (ROOT/'SCREENING_RESULTS.md').write_text(content,encoding='utf-8')
    Path('docs/V6_REGULARIZATION_SCREENING_RESULTS_2026-10-13.md').write_text(content,encoding='utf-8')
    print(json.dumps(dict(status='screening_figures_written',nominated=r['nominated_before_confirmation'],goal_achieved=False)),flush=True)


if __name__=='__main__':finish()
