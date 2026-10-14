"""英文科学图与中文结果页；只读独立核验通过的本轮结果。"""
import json
from pathlib import Path
import numpy as np
from training.v6.module_surgery_experiment import ROOT, SEEDS
from training.v6.module_surgery import ARMS
from training.v6.evaluation import metrics
from patient_data.export_mimiciv import sha256

LABELS = {'control': 'Original V6', 'fusion_resnet': 'ResNet fusion',
    'lab_ple': 'PLE lab encoding', 'lab_periodic': 'Periodic lab encoding',
    'interaction_torch': 'PyTorch interaction', 'interaction_ft': 'FT-style interaction',
    'fusion_tabm': 'TabM fusion (16 members)'}


def value(report, seed, arm, key, fold='test'):
    r = report['models'][str(seed)][arm]['folds'][fold]['readmission']
    if key.startswith('auc'):
        return r['points'][key[3:]]['auc']
    if key == 'brier90':
        return r['points']['90']['brier']
    return r[key]


def mean(report, arm, key, fold='test'):
    return float(np.mean([value(report, seed, arm, key, fold) for seed in SEEDS]))


def finish():
    r = json.loads((ROOT/'report.json').read_text())
    proof = json.loads((ROOT/'independent_verification.json').read_text())
    if r['status'] != 'completed_and_independently_verified' or proof['report_sha256'] != sha256(ROOT/'report.json'):
        raise ValueError('independent verification required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {arm: f'C{i}' for i, arm in enumerate(ARMS)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for j, arm in enumerate(ARMS):
        for seed, offset in ((42, -.08), (7, .08)):
            axes[0].scatter(value(r, seed, arm, 'auc90'), j+offset, color=colors[arm], alpha=.7)
            axes[1].scatter(value(r, seed, arm, 'integrated_brier_7_365'), j+offset, color=colors[arm], alpha=.7)
        axes[0].scatter(mean(r, arm, 'auc90'), j, marker='D', color=colors[arm], s=45)
        axes[1].scatter(mean(r, arm, 'integrated_brier_7_365'), j, marker='D', color=colors[arm], s=45)
    for ax in axes:
        ax.set_yticks(range(len(ARMS)), [LABELS[a] for a in ARMS]); ax.grid(axis='x', alpha=.25)
    axes[0].invert_yaxis(); axes[0].set_xlabel('Readmission AUC90 (higher better)')
    axes[1].set_xlabel('Readmission IBS7-365 (lower better)')
    fig.suptitle('V6 module replacements; frozen H34; dots: seeds; diamonds: mean; reused test')
    fig.tight_layout(); fig.savefig(ROOT/'performance_overview.png', dpi=180); plt.close(fig)

    names = list(ARMS[1:]) + ['ft_increment']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for j, arm in enumerate(names):
        pair = f'{arm} minus control' if arm != 'ft_increment' else 'interaction_ft minus interaction_torch'
        for seed, color, offset in ((42, 'C0', -.1), (7, 'C1', .1)):
            delta = r['comparisons'][str(seed)]['test'][pair]
            for ax, key, ci in ((axes[0], 'integrated_brier_difference', 'integrated_brier_95ci'),
                                (axes[1], 'auc90_difference', 'auc90_95ci')):
                ax.plot(delta[ci], [j+offset]*2, color=color, alpha=.65)
                ax.scatter(delta[key], j+offset, color=color, marker='D', label=f'Seed {seed}' if j == 0 else None)
    for ax in axes:
        ax.axvline(0, color='black', lw=1); ax.grid(axis='x', alpha=.25); ax.legend()
        ax.set_yticks(range(len(names)), [LABELS[a] if a in LABELS else 'FT minus PyTorch interaction' for a in names])
    axes[0].invert_yaxis(); axes[0].set_xlabel('IBS7-365: candidate minus comparator (lower better)')
    axes[1].set_xlabel('AUC90: candidate minus comparator (higher better)')
    fig.suptitle('Paired patient bootstrap 200; 95% intervals; reused test; no multiplicity adjustment')
    fig.tight_layout(); fig.savefig(ROOT/'paired_increment.png', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for row, seed in enumerate(SEEDS):
        for arm in ARMS:
            history = json.loads((ROOT/f'seed{seed}'/arm/'history.json').read_text())
            epochs = [item['epoch'] for item in history]
            axes[row, 0].plot(epochs, [item['train_fixed2048_nll'] for item in history], color=colors[arm], label=LABELS[arm])
            axes[row, 1].plot(epochs, [item['validation_ibs'] for item in history], color=colors[arm], label=LABELS[arm])
        axes[row, 0].set_title(f'Seed {seed}: fixed training sample joint NLL')
        axes[row, 1].set_title(f'Seed {seed}: validation IBS7-365')
        for ax in axes[row]:
            ax.set_xlabel('Epoch'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('All 100 epochs completed; frozen baseline; checkpoint selected by validation IBS')
    fig.tight_layout(); fig.savefig(ROOT/'training.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for arm in ARMS:
        auc, brier = [], []
        for seed in SEEDS:
            with np.load(ROOT/f'seed{seed}'/arm/'test_predictions.npz') as z:
                m = metrics(z['times'], z['events'], z['curve'][:, :, 1])
                auc.append(m['auc_curve']); brier.append(m['brier_curve'])
        axes[0].plot(np.arange(7, 366), np.mean(auc, axis=0)[7:], color=colors[arm], label=LABELS[arm])
        axes[1].plot(np.arange(366), np.mean(brier, axis=0), color=colors[arm], label=LABELS[arm])
    for ax in axes:
        ax.set_xlabel('Days after discharge'); ax.legend(fontsize=8); ax.grid(alpha=.25)
    axes[0].set_ylabel('Readmission AUC(t), mean across 2 seeds'); axes[1].set_ylabel('Readmission Brier(t), mean across 2 seeds')
    fig.suptitle('Continuous-time evaluation; 90 days is a display point, not the model output limit')
    fig.tight_layout(); fig.savefig(ROOT/'time_dependent_metrics.png', dpi=180); plt.close(fig)

    selected = r['validation_selected_arm']
    with np.load(ROOT/f'seed{SEEDS[0]}'/selected/'test_predictions.npz') as z:
        curve, daily, conditional = z['curve'][:3], z['daily_event_probability'][:3], z['conditional_daily_event_probability'][:3]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for i in range(3):
        axes[i, 0].plot(np.arange(366), curve[i, :, 1], label='Readmission CIF')
        axes[i, 0].plot(np.arange(366), curve[i, :, 2], label='Death CIF')
        axes[i, 1].plot(np.arange(1, 366), daily[i, :, 0], label='Unconditional daily readmission')
        axes[i, 2].plot(np.arange(1, 366), conditional[i, :, 0], label='Readmission given event-free yesterday')
        for ax in axes[i]:
            ax.set_xlabel('Days after discharge'); ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.suptitle(f'Daily probabilities: validation-selected {LABELS[selected]}; first 3 test rows, not outcome-selected')
    fig.tight_layout(); fig.savefig(ROOT/'daily_probability_examples.png', dpi=170); plt.close(fig)

    lines = ['# V6 内部模块原位替换：完整实验结果', '',
        '这是保留V6的模块替换，不是上一轮公开表格整模型实验。14次训练均完成100epoch，冻结H34、原输入、B四字段全部BERT片段及零初始化风险头不变。', '',
        f'验证IBS均值选择：`{selected}`。测试已反复使用，不是盲测；以下平均是两个种子的结果均值，不是为每个患者只输出一个平均风险。', '',
        '| 模块替换 | AUC30 | AUC90 | AUC180 | AUC365 | Brier90↓ | IBS7–365↓ | 最佳epoch 42/7 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for arm in ARMS:
        epochs = '/'.join(str(r['models'][str(s)][arm]['best_epoch']) for s in SEEDS)
        vals = [mean(r, arm, key) for key in ('auc30', 'auc90', 'auc180', 'auc365', 'brier90', 'integrated_brier_7_365')]
        lines.append('| '+LABELS[arm]+' | '+' | '.join(f'{v:.6f}' for v in vals)+' | '+epochs+' |')
    lines.extend(['', '## 本轮结论与采用决定', '',
        '- 按事先规定的两种子验证IBS均值，原V6为最优，默认不替换。',
        '- 原PyTorch交互的测试IBS均值点估计稍好，但AUC90几乎不变；两种子IBS配对区间都跨0，验证均值也未优于原V6，不能称稳定改善。',
        '- ResNet融合和TabM融合的AUC90在两种子均下降，探索性配对区间均在0以下；在本次固定配方/LR/容量下不采用。不能扩大为这些方法在任何任务都无效。',
        '- PLE/Periodic的验证IBS、测试IBS和AUC90均未超过原V6；只说明本次22轴编码替换未获益，不否定检验的医学价值。',
        '- FT-style没有优于原交互；原交互/FT的已训练门控绝对值约0.001–0.012，交互在该门控参数化下改动较弱。门控小不能单独证明文本/检验无信息。',
        '- 全部100轮显示训练NLL下降而验证IBS后期升高；增加训练轮数并未自动解决泛化问题，不能用训练损失更低宣称更好。',
        '- 一种固定学习率不是穷尽调参，也未测试所有模块组合。本轮不依据复用测试集继续拼装挑分，不自动换默认。',
        '', '## 其他已计算评价', '',
        '再入院与死亡分别计算全年IBS、项目竞争风险排序及30/90/180/365天校准。排序是本项目明确定义的CIF-concordance，不混称所有文献的C-index。完整定义与分位校准表在report.json。', '',
        '| 模型 | 再入院排序90天 | 再入院排序365天 | 死亡IBS7–365 |', '|---|---:|---:|---:|'])
    for arm in ARMS:
        def ordering(day):
            return float(np.mean([r['models'][str(s)][arm]['folds']['test']['readmission']['project_cif_concordance'][str(day)]['c_index'] for s in SEEDS]))
        death_ibs = np.mean([r['models'][str(s)][arm]['folds']['test']['death']['integrated_brier_7_365'] for s in SEEDS])
        lines.append(f'| {LABELS[arm]} | {ordering(90):.6f} | {ordering(365):.6f} | {death_ibs:.6f} |')
    lines.extend(['', '## 相对于原V6的配对增量', '',
        '差值＝候选−原V6；AUC为正较好，IBS为负较好。区间是按患者聚类200次重抽样，条件于已拟合模型；不是跨种子稳定性保证，未校正多重比较。', '',
        '| 候选 | seed | ΔAUC90 [95%CI] | ΔIBS [95%CI] |', '|---|---:|---|---|'])
    for arm in ARMS[1:]:
        for seed in SEEDS:
            item = r['comparisons'][str(seed)]['test'][f'{arm} minus control']
            def interval(key, ci):
                lo, hi = item[ci]; return f'{item[key]:+.6f} [{lo:+.6f}, {hi:+.6f}]'
            lines.append(f'| {LABELS[arm]} | {seed} | '+interval('auc90_difference', 'auc90_95ci')+' | '+interval('integrated_brier_difference', 'integrated_brier_95ci')+' |')
    lines.extend(['', '## 验证、运行及解释边界', '',
        f'- 病例：19,401次出院／{r["contract"]["patients"]}人；训练13,592，验证2,949，测试2,860；严格491.21/J44.1、竞争死亡、WorldA365不变。',
        '- 22轴＋既定7项目扩展检验、多来源7药效、5病原/3支持、保存的B四字段片段。当前诊断/去向已进冻结H34，不在临床分支重复；原细药扩展为0，本轮不改配方。',
        '- 仅H34冻结，其余临床模块可训练；未替换模块初值逐项相同，全部候选Epoch0风险完全相同。',
        '- PLE/Periodic只改22轴编码，既定7项扩展不动；空/常数训练列的PLE退化回退明确记录。',
        '- FT复用官方模块的参数化与实现并适配mask/token输出，不是完整FT论文复现；所有参数在训练折从头学习，不是预训练医学权重。另有PyTorch交互参照。',
        '- TabM只替换临床融合器，原修正头逐成员使用，平均成员概率，不平均Weibull参数；容量与单成员不同。',
        f'- 用时：{r["elapsed_seconds"]/60:.2f}分钟（各模型训练合计＋汇总评价）；张量显存峰值{r["peak_gpu_memory_gib"]:.2f}GiB，范围：{r.get("peak_gpu_memory_scope", "training and evaluation process")}。',
        f'- 新进程{proof["folds"]}折重载，最大概率误差{proof["maximum_replay_error"]:.3g}；{proof["source_hashes_verified"]}项源文件/合同哈希核对。',
        '- 100epoch完整性、验证选点、基线权重、患者/标签/划分、分位点训练折拟合、概率守恒/单调及每日接口均核对。',
        '- 全部114项V6回归在GPU容器通过，无跳过；代码测试不等于临床增益，证据分层见test_validation.json。',
        '- 两批入口诊断保留：已查明旧训练GPU原子聚合存在非确定性，重训不保证逐人逐位相同；正式r3统一确定性训练/推理，旧checkpoint精确重放与新对照指标近似复现分别核验。',
        '- 全部训练完成后修复NPZ关闭后的汇总读取错误；未重训/重选checkpoint。contract_training_original.json及code_snapshot保存实际训练合同/源码，最终合同同时核对训练快照与修复后的汇总代码。',
        '', '## 图与源文件', '',
        '- `performance_overview.png`：各模块两种子点估计和均值；不把种子散点称置信区间。',
        '- `paired_increment.png`：配对增量及患者聚类区间，另列FT−PyTorch交互。',
        '- `training.png`：每次完整100轮，训练NLL及验证整条曲线IBS。',
        '- `time_dependent_metrics.png`：逐天AUC(t)/Brier(t)，仍输出每天概率。',
        '- `daily_probability_examples.png`：固定前三条记录，无患者身份展示。',
        '- `report.json`、`contract.json`、`preprocessing.json`、`independent_verification.json`、`selected_before_test.json`。',
        '- 每模型目录保存best.pt、history.json、training_result.json及验证/测试每日预测。',
        '', '本轮不依据测试结果自动更换默认模型；文献机制合理不等于已证明性能提高，需看验证选择、配对区间及不同种子方向。'])
    content = '\n'.join(lines)+'\n'
    (ROOT/'README.md').write_text(content, encoding='utf-8')
    Path('docs/V6_MODULE_SURGERY_RESULTS_2026-10-12.md').write_text(content, encoding='utf-8')
    print(json.dumps(dict(status='report_and_5_figures_written', selected=selected,
        means={a: {k: mean(r, a, k) for k in ('auc90', 'brier90', 'integrated_brier_7_365')} for a in ARMS})), flush=True)


if __name__ == '__main__':
    finish()
