# 09_TRAINING_V6 · 2

续篇；文件碎片按part序号拼接可还原原文。

## training/v6/module_surgery_report.py · lines 1–177

<!-- REVIEW_FRAGMENT {"path":"training/v6/module_surgery_report.py","part":1,"start_line":1,"end_line":177,"source_sha256":"d53772f3fb7dfc1e75112b2fdc1204560c36c77114158472d2be1ce5be1d0ab0","fragment_sha256":"d53772f3fb7dfc1e75112b2fdc1204560c36c77114158472d2be1ce5be1d0ab0","added_terminal_newline":false} -->
````python
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
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/multihorizon_readout.py · lines 1–151

<!-- REVIEW_FRAGMENT {"path":"training/v6/multihorizon_readout.py","part":1,"start_line":1,"end_line":151,"source_sha256":"04dc75a37dc726cc514bccca9e4f49cb4ae27b51396c4566a278e7bd364ecb34","fragment_sha256":"04dc75a37dc726cc514bccca9e4f49cb4ae27b51396c4566a278e7bd364ecb34","added_terminal_newline":false} -->
````python
"""锁定checkpoint的多时间点读出；阈值探索不重新选择模型或训练。"""
import argparse
from pathlib import Path
import numpy as np
from training.v6.fixed_threshold_readout import load, aligned, point, read, write, sha
from training.v6.unified_longitudinal import best_f1, f1_at
from training.v6.evaluation import metrics, public_metrics, REPORT_DAYS

PARENT = Path('training_runs/v6_unified_longitudinal_f1_2026-10-14_r2')
ROOT = Path('training_runs/v6_multihorizon_readout_2026-10-14')
PROTOCOL = Path('docs/V6_MULTIHORIZON_READOUT_PROTOCOL_2026-10-14.md')
TEST = Path('training/tests/test_v6_multihorizon_readout.py')
NAMES = ('original', 'current_original', 'current_unified', 'history_original', 'history_unified')
WINDOWS = ((7, 30), (31, 90), (91, 365))


def decision(y, score, threshold):
    result = f1_at(y, score, threshold)
    result['accuracy'] = (result['tp'] + result['tn']) / len(y)
    result['alerts_per_100'] = 100 * result['alerts'] / len(y)
    return result


def horizon_readout(a, day, cause, annual_threshold):
    if day not in REPORT_DAYS or cause not in (1, 2):
        raise ValueError('unsupported horizon/cause')
    y = (a['events'] == cause) & (a['times'] <= day)
    competing = (a['events'] == 3-cause) & (a['times'] <= day)
    score = a['curve'][:, day, cause]
    optimal = best_f1(y, score)
    result = point(y, score, competing)
    result['tuning_best_f1'] = decision(y, score, optimal['threshold'])
    result['annual_threshold_readout'] = decision(y, score, annual_threshold)
    for row in result['thresholds']:
        row['f1'] = f1_at(y, score, row['threshold'])['f1']
    return result


def segmented_auc(curve):
    result = {}
    for start, end in WINDOWS:
        values = curve[start:end+1]
        good = np.isfinite(values)
        result[f'{start}_{end}'] = dict(valid_days=int(good.sum()),
            mean_auc=float(values[good].mean()) if good.any() else None)
    return result


def prepare():
    parent = read(PARENT/'report.json')
    proof = read(PARENT/'verification.json')
    if proof['maximum_error'] != 0 or proof['report_sha256'] != sha(PARENT/'report.json'):
        raise ValueError('parent proof stale')
    paths = [PARENT/'report.json', PARENT/'verification.json', PARENT/'contract.json',
             PARENT/'nomination_lock.json', Path(__file__), TEST, PROTOCOL,
             Path('training/v6/evaluation.py'), Path('training/v6/fixed_threshold_readout.py'),
             Path('training/v6/unified_longitudinal.py')]
    arrays = {}; reference = None
    for name in NAMES:
        p = PARENT/name/'tuning_predictions.npz'
        a = load(p); paths.append(p)
        if reference is None: reference = a
        aligned(reference, a)
        if name != 'original':
            checkpoint = PARENT/name/'best.pt'
            if sha(checkpoint) != parent['results'][name]['checkpoint_sha256']:
                raise ValueError('parent checkpoint changed')
            paths.append(checkpoint)
        arrays[name] = a
    if len(reference['times']) != 1531:
        raise ValueError('not specified tuning fold')
    contract = dict(source_sha256={str(p):sha(p) for p in paths},
        parent_labels_sha256=read(PARENT/'contract.json')['labels_sha256'],
        horizon_days=list(REPORT_DAYS), models=list(NAMES), new_fits=0,
        selected_checkpoints_unchanged=True, tuning_thresholds_not_independent_evaluation=True,
        confirmation_or_test_read=False, automation_resumed=False)
    return parent, arrays, contract


def calculate():
    parent, arrays, contract = prepare(); results = {}
    for name, a in arrays.items():
        expected = parent['reference'] if name == 'original' else parent['results'][name]['metrics']
        result = dict(admissions=len(a['times']), patients=len(np.unique(a['subject_ids'])), outcomes={})
        for label, cause in (('readmission',1), ('death',2)):
            m = metrics(a['times'], a['events'], a['curve'][:,:,cause], cause=cause, ordering=True)
            # 既有保存模型全部每日指标逐项复现，不重算/替换其原选点。
            if public_metrics(m) != expected[label]:
                raise ValueError('parent daily metrics differ: '+name+'/'+label)
            annual_y = (a['events'] == cause) & (a['times'] <= 365)
            annual = (expected['annual'] if cause == 1 else
                      best_f1(annual_y, a['curve'][:,365,cause]))
            points = {str(day):horizon_readout(a,day,cause,annual['threshold']) for day in REPORT_DAYS}
            if cause == 1:
                actual = f1_at(annual_y, a['curve'][:,365,1], annual['threshold'])
                if actual != expected['annual']:
                    raise ValueError('parent annual decision differs')
            result['outcomes'][label] = dict(daily_metrics=public_metrics(m),
                segmented_auc=segmented_auc(m['auc_curve']), annual_threshold=annual['threshold'], points=points)
        results[name] = result
    return contract, dict(models=results, new_fits=0, no_model_reselection=True,
        fold='same exposed tuning fold', threshold_policy='tuning-best per horizon + annual transfer + common fixed',
        calibration_bins='fixed 0/.1/.2/.3/.5/1; descriptive, not individual truth')


def build():
    if ROOT.exists(): raise FileExistsError('readout exists; use verify, do not rebuild')
    contract, report = calculate()
    ROOT.mkdir(); write(ROOT/'contract.json',contract); write(ROOT/'report.json',report)
    print({n:{d:x['tuning_best_f1']['f1'] for d,x in r['outcomes']['readmission']['points'].items()}
           for n,r in report['models'].items()},flush=True)


def verify():
    contract, report = calculate()
    if contract != read(ROOT/'contract.json') or report != read(ROOT/'report.json'):
        raise ValueError('independent readout differs')
    write(ROOT/'verification.json',dict(maximum_error=0,independent_new_process=True,
        no_refitting=True,report_sha256=sha(ROOT/'report.json'),contract_sha256=sha(ROOT/'contract.json')))
    print('all five multi-horizon readouts replay exactly; no refitting',flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    proof = read(ROOT/'verification.json')
    if proof['report_sha256'] != sha(ROOT/'report.json'): raise ValueError('readout proof stale')
    report = read(ROOT/'report.json'); days=np.array(REPORT_DAYS)
    labels=('Retained V6','Current / original','Current / unified','History / original','History / unified')
    fig, axes=plt.subplots(2,2,figsize=(12,8))
    for name,label in zip(NAMES,labels,strict=True):
        r=report['models'][name]['outcomes']['readmission']; p=r['points'];dm=r['daily_metrics']
        axes[0,0].plot(days,[dm['points'][str(d)]['auc'] for d in days],marker='o',label=label)
        axes[0,1].plot(days,[p[str(d)]['tuning_best_f1']['f1'] for d in days],marker='o',label=label)
        axes[1,0].plot(days,[p[str(d)]['brier'] for d in days],marker='o',label=label)
        axes[1,1].plot(days,[p[str(d)]['tuning_best_f1']['threshold'] for d in days],marker='o',label=label)
    titles=('AUC(t), threshold-independent','F1 at separately tuning-selected thresholds',
            'Brier(t), lower is better','Tuning-selected threshold at each horizon')
    for ax,title in zip(axes.flat,titles,strict=True):
        ax.set(title=title,xlabel='Days since discharge');ax.set_xscale('log');ax.set_xticks(days)
        ax.set_xticklabels(days);ax.grid(alpha=.2)
    axes[0,0].set_ylim(.5,1);axes[0,1].set_ylim(0,1);axes[1,1].set_ylim(0,1)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Same 1,531 exposed tuning discharges; annual-selected checkpoints locked; zero new fits')
    fig.tight_layout();fig.savefig(ROOT/'multihorizon.png',dpi=140);plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=('build','verify','plot'))
    {'build':build,'verify':verify,'plot':plot}[parser.parse_args().mode]()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/nonlinear_baseline_probe.py · lines 1–56

<!-- REVIEW_FRAGMENT {"path":"training/v6/nonlinear_baseline_probe.py","part":1,"start_line":1,"end_line":56,"source_sha256":"910e4c1832d5a42521d5ea6a9eb5ff95434fe0f6d53432b98ac5c567bd225e4d","fragment_sha256":"910e4c1832d5a42521d5ea6a9eb5ff95434fe0f6d53432b98ac5c567bd225e4d","added_terminal_newline":false} -->
````python
"""线性扩展B未达到等效证据时的小型非线性承接检查；两种子、旧H固定。"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from patient_data.export_mimiciv import sha256
from training.v6 import architecture_experiment as experiment
from training.v6.exploration_model import ExpandedBaseline


class AdditiveAmplitude(nn.Module):
    def __init__(self, history):
        super().__init__()
        self.history=nn.Linear(6,2)
        self.history.load_state_dict(history.amplitude.state_dict())
        self.current=nn.Sequential(nn.Linear(34,32),nn.GELU(),nn.Linear(32,2))
        nn.init.zeros_(self.current[-1].weight);nn.init.zeros_(self.current[-1].bias)

    def forward(self,x): return self.history(x[:,:6])+self.current(x)


class NonlinearExpandedBaseline(ExpandedBaseline):
    initialization_seed=42
    def __init__(self,history):
        nn.Module.__init__(self)
        torch.manual_seed(self.initialization_seed)
        self.raw_shape=nn.Parameter(history.raw_shape.detach().clone())
        self.amplitude=AdditiveAmplitude(history)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify',action='store_true');args=p.parse_args()
    NonlinearExpandedBaseline.initialization_seed=args.seed
    original=experiment.ExpandedBaseline;experiment.ExpandedBaseline=NonlinearExpandedBaseline
    try:
        if not args.verify:
            common=dict(reference=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'),
                compact_dir=Path('patient_data/processed/v6_compact_multisource_2026-10-09'),
                combinations=Path('patient_data/processed/v6_cbc_combinations_2026-10-10'),
                candidate=Path('patient_data/processed/v6_screen_candidate_bert_2026-10-10'),
                epochs=60,batch_size=512,bootstrap=200,seed=args.seed,verify=False,baseline_run=args.output)
            experiment.run(SimpleNamespace(**common,phase='baseline',output=args.output))
            r=json.loads((args.output/'report.json').read_text())
            r['contract']['code_sha256'][str(Path(__file__))]=sha256(Path(__file__))
            r['contract']['baseline_architecture']='H affine amplitude skip + 34->32->2 zero-final-init current MLP, global two shapes; joint trained then frozen'
            r['contract']['limits'].append('Follow-up capacity probe after linear-B result, not a prespecified blind test')
            experiment.write_json(args.output/'report.json',r);experiment.write_json(args.output/'contract.json',r['contract'])
        experiment.verify(SimpleNamespace(output=args.output))
    finally: experiment.ExpandedBaseline=original


if __name__=='__main__': main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/objective_direction_diagnostic.py · lines 1–137

<!-- REVIEW_FRAGMENT {"path":"training/v6/objective_direction_diagnostic.py","part":1,"start_line":1,"end_line":137,"source_sha256":"943f3fcd2d246da1fbc6ea1a04efee2f66069076f310ed0e1b22359e211dbcb0","fragment_sha256":"943f3fcd2d246da1fbc6ea1a04efee2f66069076f310ed0e1b22359e211dbcb0","added_terminal_newline":false} -->
````python
"""原V6同训练记录、同状态的风险头方向诊断；没有训练或验证选择。"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.module_surgery_experiment import batches
from training.v6.warm_residual_audit import teacher
from training.v6.horizon_objective import restore
from training.v6.risk import RiskOutput
from training.v6.patient_weight_loss import per_record_nll
from training.v6.curve_score import daily_cif,curve_scores
from training.v6.token_model import fast_cif
from training.v6.train import write_json

ROOT=Path('training_runs/v6_objective_direction_2026-10-13')
NAMES=('nll','readmission','death','joint')
SOURCES=('training/v6/objective_direction_diagnostic.py','training/v6/curve_score.py',
         'docs/V6_OBJECTIVE_DIRECTION_PROTOCOL_2026-10-13.md')


def output(raw,prior,base_shape):
    delta=raw.reshape(-1,2,2)
    return RiskOutput((prior+delta[:,:,0]).clamp(-10,6),F.softplus(base_shape+delta[:,:,1])+.1)


def losses(out,t,e):
    curve=daily_cif(out)
    return dict(nll=per_record_nll(out,t,e),**curve_scores(curve,t,e)),curve


def cosine(a,b):
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a.flatten()@b.flatten()/norm) if norm>1e-18 else None


def compare(saved,current,path=''):
    if isinstance(current,dict):
        if current.keys()!=saved.keys():raise ValueError('replay keys differ '+path)
        return max([compare(saved[k],v,path+'/'+k) for k,v in current.items()]+[0.])
    if isinstance(current,list):
        if len(current)!=len(saved):raise ValueError('replay list differs '+path)
        return max([compare(x,y,path) for x,y in zip(saved,current)]+[0.])
    if isinstance(current,(float,int)) and not isinstance(current,bool):
        error=abs(current-saved)
        if error>1e-10:raise ValueError('replay numeric differs '+path)
        return error
    if saved!=current:raise ValueError('replay differs '+path)
    return 0.


def run():
    start=time.perf_counter()
    d,c,h,ext,_,bins,_,_,search,_=checked()
    ix=np.flatnonzero(d.arrays['patient_splits']=='train')
    cache=batches(d,c,ext,ix);results={}
    for seed in (42,7):
        base=restore(Path(search['input_config']['parent']),seed,'survival_base',h,.1)
        model=teacher(base,seed,bins).eval();records=[]
        initial_error=0.;forward_error=0.
        with torch.no_grad():
            for _,history,batch,t,e in cache:
                state,_=model.encoder(batch.base);hidden=model.delta_head[:-1](state)
                raw=model.delta_head[-1](hidden);prior=model.baseline(history)
                predicted=output(raw,prior.log_amplitude,model.baseline.raw_shape)
                expected=model(history,batch)
                for a,b in [(predicted.log_amplitude,expected.log_amplitude),(predicted.shape,expected.shape)]:
                    initial_error=max(initial_error,float((a-b).abs().max()))
                if initial_error!=0:raise ValueError('teacher parameter reconstruction differs')
                ours=daily_cif(predicted).cpu().numpy()
                official=fast_cif(expected.log_amplitude.cpu().numpy(),expected.shape.cpu().numpy(),device='cuda')
                forward_error=max(forward_error,float(np.max(np.abs(ours-official))))
                if forward_error>1e-12:raise ValueError('official daily forward differs')
                records.append((torch.cat((hidden,torch.ones(len(t),1,device='cuda')),1),raw,
                                prior.log_amplitude,t,e))
        gradients={name:np.zeros((4,33),np.float64) for name in NAMES}
        values={name:0. for name in NAMES};count=0
        for hidden,raw,prior,t,e in records:
            raw=raw.detach().requires_grad_(True)
            scored,_=losses(output(raw,prior,model.baseline.raw_shape.detach()),t,e)
            for j,name in enumerate(NAMES):
                values[name]+=float(scored[name].detach().sum())/len(ix)
                g=torch.autograd.grad(scored[name].sum(),raw,retain_graph=j<len(NAMES)-1)[0]
                gradients[name]+=(g.double().T@hidden.double()).detach().cpu().numpy()/len(ix)
            count+=len(t)
        if count!=len(ix) or not all(np.isfinite(g).all() for g in gradients.values()):
            raise ValueError('training coverage/gradient invalid')
        pairwise={a:{b:cosine(gradients[a],gradients[b]) for b in NAMES} for a in NAMES}
        component={str(i):{name:cosine(gradients['nll'][i],gradients[name][i]) for name in NAMES[1:]}
                   for i in range(4)}
        steps={}
        for direction,g in gradients.items():
            tensor=torch.as_tensor(g,dtype=torch.float64,device='cuda')
            projected=[hidden.double()@tensor.T for hidden,*_ in records]
            rms=float(torch.sqrt(sum(p.square().sum() for p in projected)/(len(ix)*4)))
            if rms<1e-15:raise ValueError('cannot normalize zero direction')
            rows=[]
            for size in (.001,.01):
                total={name:0. for name in NAMES}
                with torch.no_grad():
                    for (hidden,raw,prior,t,e),p in zip(records,projected):
                        adjusted=raw-(p*(size/rms)).to(raw.dtype)
                        scored,_=losses(output(adjusted,prior,model.baseline.raw_shape.detach()),t,e)
                        for name in NAMES:total[name]+=float(scored[name].sum())/len(ix)
                rows.append(dict(raw_delta_rms=size,losses=total,
                    changes={name:total[name]-values[name] for name in NAMES},
                    first_order_changes={name:float(-size/rms*np.sum(gradients[name]*g)) for name in NAMES}))
            steps[direction]=rows
        results[str(seed)]=dict(losses=values,gradient_norms={k:float(np.linalg.norm(v)) for k,v in gradients.items()},
            gradient_cosines=pairwise,component_cosines=component,
            component_order=['readmission_amplitude','readmission_raw_shape','death_amplitude','death_raw_shape'],
            local_descent=steps,original_parameter_max_error=initial_error,official_curve_max_error=forward_error,
            teacher_sha256=sha256(REFERENCE/f'seed{seed}/control/best.pt'))
        print(json.dumps(dict(seed=seed,losses=values,gradient_cosines=pairwise,
                             seconds=round(time.perf_counter()-start,1))),flush=True)
        del model
    report=dict(models=results,admissions=len(ix),patients=len(np.unique(d.arrays['subject_ids'][ix])),
        training_index_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),
        label_sha256=d.provenance['label_sha256'],source_sha256={p:sha256(Path(p)) for p in SOURCES},
        search_contract_sha256=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')),
        scope='All training discharges; fixed eval upstream; final linear projection only; no fits or validation selection.',
        goal_achieved=False)
    ROOT.mkdir(exist_ok=True)
    path=ROOT/'diagnostic.json'
    if path.exists():
        error=compare(json.loads(path.read_text()),report)
        write_json(ROOT/'independent_verification.json',dict(status='independently_replayed',maximum_error=error,
            report_sha256=sha256(path),seconds=time.perf_counter()-start,models=2))
    else:write_json(path,report)
    print(json.dumps(dict(status='completed_training_only_diagnostic',seconds=round(time.perf_counter()-start,1))),flush=True)


if __name__=='__main__':run()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/objective_direction_report.py · lines 1–50

<!-- REVIEW_FRAGMENT {"path":"training/v6/objective_direction_report.py","part":1,"start_line":1,"end_line":50,"source_sha256":"23b706ce8b52531b60dc49f196ee412e4019a4fa84dda3055e7cd465e55016cd","fragment_sha256":"23b706ce8b52531b60dc49f196ee412e4019a4fa84dda3055e7cd465e55016cd","added_terminal_newline":false} -->
````python
"""经独立重放后生成英文方向图；非训练性能/验证AUC图。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json


def run():
    root=Path('training_runs/v6_objective_direction_2026-10-13')
    network=Path('training_runs/v6_objective_network_2026-10-13')
    for folder in (root,network):
        proof=json.loads((folder/'independent_verification.json').read_text())
        if proof['report_sha256']!=sha256(folder/'diagnostic.json') or proof['maximum_error']>1e-10:
            raise ValueError('independent diagnostic proof required')
    head=json.loads((root/'diagnostic.json').read_text());full=json.loads((network/'diagnostic.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(14,8.4),layout='constrained')
    pairs=[('nll','readmission'),('nll','death'),('nll','joint'),('readmission','death')]
    for row,seed in enumerate(('42','7')):
        scopes=full['models'][seed]['scopes']
        order=['all_clinical','encoder','risk_head','last_projection']
        values=np.array([[scopes[s]['cosines'][a][b] for a,b in pairs] for s in order])
        ax=axes[row,0];im=ax.imshow(values,vmin=-1,vmax=1,cmap='RdBu',aspect='auto')
        ax.set_xticks(range(4),['NLL vs\nreadmission','NLL vs\ndeath','NLL vs\njoint score','Readmission\nvs death'])
        ax.set_yticks(range(4),['All clinical parameters','Encoder','Risk correction head','Final projection'])
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',color='white' if abs(values[i,j])>.7 else 'black')
        ax.set_title(f'Source {seed}: gradient cosine (not AUC)')
        fig.colorbar(im,ax=ax,shrink=.8,label='-1 opposite | 0 orthogonal | +1 aligned')
        ax=axes[row,1];names=['nll','readmission','death','joint'];x=np.arange(4)
        delta=head['models'][seed]['local_descent']
        for offset,key,label,color in [(-.18,'readmission','Readmission IBS','#1f77b4'),(.18,'death','Death IBS','#ff7f0e')]:
            ax.bar(x+offset,[delta[n][1]['changes'][key]*1e5 for n in names],width=.36,label=label,color=color)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,['NLL','Readmission IBS','Death IBS','Joint score'])
        ax.set_ylabel('Training IBS change (x 1e-5; lower is better)')
        ax.set_xlabel('One local descent direction; matched raw-output RMS = 0.01')
        ax.set_title(f'Source {seed}: final-projection local step');ax.legend()
    fig.suptitle('Fixed original V6 states; all 13,592 training discharges; no optimization or validation selection',fontsize=13)
    fig.savefig(root/'objective_directions.png',dpi=160);plt.close(fig)
    write_json(root/'state.json',dict(status='diagnostic_complete_and_independently_replayed',new_model_fits=0,
        training_records=head['admissions'],teachers=2,goal_achieved=False,
        report_sha256=sha256(root/'diagnostic.json'),network_report_sha256=sha256(network/'diagnostic.json'),
        figure_sha256=sha256(root/'objective_directions.png')))
    print('Saved verified gradient/local-direction figure; no new performance result.')


if __name__=='__main__':run()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/objective_gradient_precision.py · lines 1–41

<!-- REVIEW_FRAGMENT {"path":"training/v6/objective_gradient_precision.py","part":1,"start_line":1,"end_line":41,"source_sha256":"bce240fec88904947f1fafa866d8d9467a61710f92c65d8eb6a569ea70ca2fbe","fragment_sha256":"bce240fec88904947f1fafa866d8d9467a61710f92c65d8eb6a569ea70ca2fbe","added_terminal_newline":false} -->
````python
"""真实首个训练batch的最小精度差分；只切换反向TF32，不改变前向。"""
import json
from pathlib import Path
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked
from training.v6.module_surgery_experiment import batches
from training.v6.warm_residual_audit import teacher
from training.v6.horizon_objective import restore
from training.v6.objective_direction_diagnostic import output
from training.v6.patient_weight_loss import per_record_nll
from training.v6.train import write_json
import numpy as np


def run():
    d,c,h,ext,_,bins,_,_,search,_=checked()
    ix=np.flatnonzero(d.arrays['patient_splits']=='train')[:256]
    _,history,batch,t,e=batches(d,c,ext,ix)[0]
    base=restore(Path(search['input_config']['parent']),42,'survival_base',h,.1)
    model=teacher(base,42,bins).eval()
    with torch.no_grad():state,_=model.encoder(batch.base);hidden=model.delta_head[:-1](state);prior=model.baseline(history)
    layer=model.delta_head[-1];raw=layer(hidden);out=output(raw,prior.log_amplitude,model.baseline.raw_shape.detach())
    loss=per_record_nll(out,t,e).sum()/13592
    raw_grad=torch.autograd.grad(loss,raw,retain_graph=True)[0]
    reference=raw_grad.double().T@hidden.double();result={}
    original=torch.backends.cuda.matmul.allow_tf32
    try:
        for enabled in (True,False):
            torch.backends.cuda.matmul.allow_tf32=enabled
            actual=torch.autograd.grad(loss,layer.weight,retain_graph=True)[0]
            difference=(actual.double()-reference).abs()
            result[str(enabled)]=dict(max_error=float(difference.max()),relative_norm_error=float(difference.norm()/reference.norm()))
    finally:torch.backends.cuda.matmul.allow_tf32=original
    root=Path('training_runs/v6_objective_network_2026-10-13');root.mkdir(exist_ok=True)
    report=dict(status='real_batch_backward_precision_probe',results=result,
        source_sha256=sha256(Path(__file__)),forward_unchanged=True)
    write_json(root/'precision_probe.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':run()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/objective_network_diagnostic.py · lines 1–86

<!-- REVIEW_FRAGMENT {"path":"training/v6/objective_network_diagnostic.py","part":1,"start_line":1,"end_line":86,"source_sha256":"429f3b8f29c0e6e0a2ea0840ca0cb98e6bc164a0c7b0bcded8b7ee3411ad08e0","fragment_sha256":"429f3b8f29c0e6e0a2ea0840ca0cb98e6bc164a0c7b0bcded8b7ee3411ad08e0","added_terminal_newline":false} -->
````python
"""同状态全可训练临床分支的四目标梯度；只读，不训练。"""
import json
import time
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.module_surgery_experiment import batches
from training.v6.warm_residual_audit import teacher
from training.v6.horizon_objective import restore
from training.v6.objective_direction_diagnostic import NAMES,cosine,compare,losses
from training.v6.train import write_json

ROOT=Path('training_runs/v6_objective_network_2026-10-13')


def run():
    start=time.perf_counter();d,c,h,ext,_,bins,_,_,search,_=checked()
    previous=Path('training_runs/v6_objective_direction_2026-10-13')
    proof=json.loads((previous/'independent_verification.json').read_text())
    if proof['report_sha256']!=sha256(previous/'diagnostic.json'):raise ValueError('head proof stale')
    before=json.loads((previous/'diagnostic.json').read_text())
    for p,digest in before['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('head diagnostic source changed')
    ix=np.flatnonzero(d.arrays['patient_splits']=='train');cache=batches(d,c,ext,ix);result={}
    for seed in (42,7):
        base=restore(Path(search['input_config']['parent']),seed,'survival_base',h,.1)
        model=teacher(base,seed,bins).eval()
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if any(n.startswith('baseline.') for n,p in named):raise ValueError('H34 not frozen')
        params=[p for n,p in named];gradients={k:[torch.zeros_like(p,dtype=torch.float64) for p in params] for k in NAMES}
        values={k:0. for k in NAMES};last_ids={id(p) for p in model.delta_head[-1].parameters()}
        for _,history,batch,t,e in cache:
            out=model(history,batch);scored,_=losses(out,t,e)
            for j,name in enumerate(NAMES):
                values[name]+=float(scored[name].detach().sum())/len(ix)
                # 原前向TF32不动；严格梯度核验只关闭反向矩阵乘法TF32舍入。
                precision=torch.backends.cuda.matmul.allow_tf32
                try:
                    torch.backends.cuda.matmul.allow_tf32=False
                    g=torch.autograd.grad(scored[name].sum()/len(ix),params,allow_unused=True,retain_graph=j<len(NAMES)-1)
                finally:torch.backends.cuda.matmul.allow_tf32=precision
                for total,part in zip(gradients[name],g):
                    if part is not None:total.add_(part.double())
        scopes={'all_clinical':[True]*len(named),
                'encoder':[n.startswith('encoder.') for n,p in named],
                'risk_head':[n.startswith('delta_head.') for n,p in named],
                'last_projection':[id(p) in last_ids for n,p in named]}
        measures={}
        for scope,mask in scopes.items():
            vectors={k:torch.cat([g.flatten() for g,active in zip(v,mask) if active]).cpu().numpy()
                     for k,v in gradients.items()}
            if not all(np.isfinite(v).all() for v in vectors.values()):raise ValueError('nonfinite full gradients')
            measures[scope]=dict(parameters=len(vectors['nll']),
                gradient_norms={k:float(np.linalg.norm(v)) for k,v in vectors.items()},
                cosines={a:{b:cosine(vectors[a],vectors[b]) for b in NAMES} for a in NAMES})
        old=before['models'][str(seed)]
        loss_error=max(abs(values[k]-old['losses'][k]) for k in NAMES)
        cosine_error=max(abs(measures['last_projection']['cosines'][a][b]-old['gradient_cosines'][a][b])
                         for a in NAMES for b in NAMES)
        if loss_error>1e-8 or cosine_error>1e-5:
            ROOT.mkdir(exist_ok=True)
            write_json(ROOT/'failure.json',dict(seed=seed,loss_error=loss_error,cosine_error=cosine_error,
                scopes=measures,losses=values,source_sha256=sha256(Path(__file__))))
            raise ValueError(f'last-projection/training objective mismatch: loss={loss_error}, cosine={cosine_error}')
        result[str(seed)]=dict(losses=values,scopes=measures,last_projection_cosine_max_error=cosine_error,
            prior_loss_max_error=loss_error,teacher_sha256=sha256(REFERENCE/f'seed{seed}/control/best.pt'))
        print(json.dumps(dict(seed=seed,scopes=measures,seconds=round(time.perf_counter()-start,1))),flush=True)
        del model,gradients
    source=('training/v6/objective_network_diagnostic.py','docs/V6_OBJECTIVE_NETWORK_PROTOCOL_2026-10-13.md',
            'training/v6/curve_score.py','training/v6/objective_gradient_precision.py')
    report=dict(models=result,admissions=len(ix),training_index_sha256=before['training_index_sha256'],
        label_sha256=before['label_sha256'],head_report_sha256=sha256(previous/'diagnostic.json'),
        source_sha256={p:sha256(Path(p)) for p in source},goal_achieved=False,
        scope='Fixed eval; all trainable clinical parameters; frozen H34; no optimization or validation selection.')
    ROOT.mkdir(exist_ok=True);path=ROOT/'diagnostic.json'
    if path.exists():
        error=compare(json.loads(path.read_text()),report)
        write_json(ROOT/'independent_verification.json',dict(status='independently_replayed',maximum_error=error,
            report_sha256=sha256(path),seconds=time.perf_counter()-start))
    else:write_json(path,report)
    print(json.dumps(dict(status='completed_full_clinical_gradient',seconds=round(time.perf_counter()-start,1))),flush=True)


if __name__=='__main__':run()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/oof_baseline_diagnostic.py · lines 1–317

<!-- REVIEW_FRAGMENT {"path":"training/v6/oof_baseline_diagnostic.py","part":1,"start_line":1,"end_line":317,"source_sha256":"2a2bad70d70cc4a4fd0ffcbf0957dad62f53bdbdda40b7e307fdaa2cb5c5327f","fragment_sha256":"2a2bad70d70cc4a4fd0ffcbf0957dad62f53bdbdda40b7e307fdaa2cb5c5327f","added_terminal_newline":false} -->
````python
"""有限患者级OOF基础风险诊断；不复用标签训练权重、不训练临床网络。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from patient_data.export_mimiciv import sha256
from training.v5.numeric_preprocessing import robust_parameters
from training.v6.data import HISTORY_NAMES
from training.v6.deep_interaction_experiment import runtime_setup
from training.v6.evaluation import metrics, public_metrics, paired_interval
from training.v6.module_surgery import probability_curve
from training.v6.regularization_audit import checked, REFERENCE
from training.v6.risk import RiskOutput, competing_nll
from training.v6.train import write_json

ROOT = Path('training_runs/v6_oof_baseline_2026-10-13')
PROTOCOL = Path('docs/V6_OOF_BASELINE_PROTOCOL_2026-10-13.md')
SEED, FOLDS, UPDATES = 42, 5, 600


def digest_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


def fold_assignment(subjects, folds=FOLDS):
    subjects = np.asarray(subjects)
    if subjects.ndim != 1 or not len(subjects) or folds < 2 or len(np.unique(subjects)) < folds:
        raise ValueError('invalid patient folds')
    ordered = sorted(np.unique(subjects).tolist(), key=lambda s: hashlib.sha256(f'v6-oof-v1:{s}'.encode()).digest())
    lookup = {s: j % folds for j, s in enumerate(ordered)}
    return np.array([lookup[s] for s in subjects], dtype=np.int64)


def fit_scales(raw, fit_indices):
    raw, ix = np.asarray(raw), np.asarray(fit_indices)
    if raw.ndim != 2 or raw.shape[1] != 6 or not np.isfinite(raw).all() or ix.ndim != 1 or not len(ix):
        raise ValueError('invalid raw history/scaling positions')
    if len(np.unique(ix)) != len(ix) or np.any(ix < 0) or np.any(ix >= len(raw)):
        raise ValueError('invalid scaling index')
    result = {}
    for c in (0, 2, 3, 4):
        rows = ix if c != 4 else ix[raw[ix, 5] == 0]
        result[HISTORY_NAMES[c]] = robust_parameters(raw[rows, c])
    return result


def matrix(raw, current, masks, scales):
    raw, current, masks = np.asarray(raw), np.asarray(current), np.asarray(masks, dtype=bool)
    if raw.shape != (len(current), 6) or current.shape != (len(raw), 14) or masks.shape != current.shape:
        raise ValueError('invalid H34 source shapes')
    if not np.isfinite(raw).all() or not np.isfinite(current).all():
        raise ValueError('nonfinite H34')
    h = raw.copy()
    for c in (0, 2, 3, 4):
        p = scales[HISTORY_NAMES[c]]
        valid = np.ones(len(h), dtype=bool) if c != 4 else raw[:, 5] == 0
        h[:, c] = np.where(valid, np.clip((h[:, c]-p['median'])/p['scale'], -8, 8), 0)
    return np.concatenate([h, np.where(masks, current, 0), masks.astype(float)], axis=1).astype(np.float32)


class FreshH34(nn.Module):
    def __init__(self, seed=SEED):
        super().__init__()
        # 不调用旧ExpandedBaseline(history)的复制已学习H6路径。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.amplitude = nn.Linear(34, 2)
        nn.init.constant_(self.amplitude.bias, -2.)
        self.raw_shape = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != 34:
            raise ValueError('expected H34 matrix')
        return RiskOutput(self.amplitude(x).clamp(-10, 6), (F.softplus(self.raw_shape)+.1).expand(len(x), 2))


def weight_digest(model):
    return hashlib.sha256(''.join(k+digest_array(v.detach().cpu().numpy()) for k, v in model.state_dict().items()).encode()).hexdigest()


def fit_model(x, times, events, *, updates=UPDATES):
    """只接收内训练特征/标签；无留出、调参标签或选点参数。"""
    device = x.device
    model = FreshH34().to(device)
    initial = weight_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=1e-4)
    history = []
    start = time.perf_counter()
    for step in range(updates+1):
        optimizer.zero_grad(set_to_none=True)
        loss = competing_nll(model(x), times, events)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite inner fit')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise ValueError('nonfinite inner gradient')
        if step % 25 == 0 or step == updates:
            history.append(dict(update=step, fixed_train_nll=float(loss.detach()), gradient_norm=float(norm)))
        if step != updates:
            optimizer.step()
    return model, dict(initial_weight_digest=initial, updates=updates, selected_update=updates,
                       selection='fixed terminal; no held-out score selection', history=history,
                       tail50_nll_change=history[-1]['fixed_train_nll']-history[-3]['fixed_train_nll'] if len(history)>=3 else None,
                       seconds=time.perf_counter()-start)


@torch.no_grad()
def predict(model, x):
    model.eval()
    parts = [model(chunk) for chunk in x.split(512)]
    return tuple(torch.cat([getattr(p, key) for p in parts]).cpu().numpy() for key in ('log_amplitude', 'shape'))


def data_contract():
    d, _, _, _, _, _, tune, _, original, prior = checked()
    train = np.flatnonzero(d.arrays['patient_splits'] == 'train')
    if len(train) != 13592 or len(tune) != 1531:
        raise ValueError('outer split changed')
    sources = {**original['source_sha256'], **prior['source_sha256']}
    for p in ('training/v6/oof_baseline_diagnostic.py', 'training/tests/test_v6_oof_baseline.py',
              'training/v6/data.py', 'training/v6/risk.py', 'training/v5/numeric_preprocessing.py',
              'training/v6/regularization_audit.py', 'training/v6/module_surgery.py', 'training/v6/token_model.py',
              'training/v6/evaluation.py'):
        sources[p] = sha256(Path(p))
    protected = {str(REFERENCE/f'seed{s}/control/best.pt'): sha256(REFERENCE/f'seed{s}/control/best.pt') for s in (42, 7)}
    parent = Path(original['input_config']['parent'])
    protected.update({str(parent/f'seed{s}/survival_base/best.pt'): sha256(parent/f'seed{s}/survival_base/best.pt') for s in (42, 7)})
    proof = dict(source_sha256=sources, protocol_sha256=sha256(PROTOCOL), protected_weights=protected,
                 data_provenance=d.provenance, train_indices_hash=digest_array(train), tuning_indices_hash=digest_array(tune),
                 raw_history_hash=digest_array(d.history_raw), current_hash=digest_array(d.current),
                 current_mask_hash=digest_array(d.current_masks), times_hash=digest_array(d.times), events_hash=digest_array(d.events),
                 seed=SEED, folds=FOLDS, updates=UPDATES, train_admissions=len(train), train_patients=len(np.unique(d.arrays['subject_ids'][train])),
                 initialization='fresh random; never loaded old H6/H34/clinical weights', clinical_fits=0,
                 optimizer=dict(name='AdamW', lr=.01, weight_decay=1e-4, clip=1., batching='full inner train'),
                 selection='fixed terminal600; outer tuning evaluation only', confirmation_test_evaluated=False)
    return d, train, tune, proof


def tensors(d, x, ix):
    return (torch.as_tensor(x[ix], device='cuda'), torch.as_tensor(d.times[ix], device='cuda'),
            torch.as_tensor(d.events[ix], device='cuda', dtype=torch.long))


def preflight():
    d, train, tune, contract = data_contract()
    if ROOT.exists():
        raise FileExistsError('diagnostic directory exists; inspect instead of repeat')
    ROOT.mkdir()
    write_json(ROOT/'contract.json', contract)
    assignment = fold_assignment(d.arrays['subject_ids'][train])
    np.savez_compressed(ROOT/'partition.npz', train_indices=train, tuning_indices=tune, fold=assignment)
    held = train[assignment == 0]
    inner = train[assignment != 0]
    assert not np.intersect1d(d.arrays['subject_ids'][inner], d.arrays['subject_ids'][held]).size
    scales = fit_scales(d.history_raw, inner)
    x = matrix(d.history_raw, d.current, d.current_masks, scales)
    sample = inner[:128]
    m = FreshH34().cuda()
    a, k = predict(m, torch.as_tensor(x[held[:128]], device='cuda'))
    restored = FreshH34().cuda(); restored.load_state_dict(m.state_dict())
    ar, kr = predict(restored, torch.as_tensor(x[held[:128]], device='cuda'))
    assert np.array_equal(a, ar) and np.array_equal(k, kr)
    fitted, tiny = fit_model(*tensors(d, x, sample), updates=1)
    assert weight_digest(fitted) != weight_digest(m)
    # 改动留出标签后，同一内训练的更新必须逐权重一致。
    altered_events = d.events.copy(); altered_events[held] = (altered_events[held]+1)%3
    repeated, _ = fit_model(torch.as_tensor(x[sample], device='cuda'), torch.as_tensor(d.times[sample], device='cuda'),
                            torch.as_tensor(altered_events[sample], device='cuda', dtype=torch.long), updates=1)
    assert weight_digest(fitted) == weight_digest(repeated)
    write_json(ROOT/'preflight.json', dict(status='passed', parameter_count=sum(p.numel() for p in m.parameters()),
        initial_reload_max_error=0., heldout_label_perturbation_weight_error=0., gpu=torch.cuda.get_device_name(0), single_update=tiny,
        patient_disjoint=True, no_clinical_fit=True))
    write_json(ROOT/'state.json', dict(status='preflight_passed_not_started', model_fits=0, goal_achieved=False))
    print(json.dumps(dict(status='preflight_passed', parameters=72, initial_error=0., model_fits=0)), flush=True)


def report(d, train, tune, assignment, ap, kp, full_a, full_k, tune_a, tune_k, fit_infos):
    oof, full, tuning = [probability_curve(a, k) for a, k in ((ap,kp),(full_a,full_k),(tune_a,tune_k))]
    for c in (oof, full, tuning):
        np.testing.assert_allclose(c.sum(2), 1., atol=1e-9)
        if not np.isfinite(c).all() or np.any((c<0)|(c>1)) or np.any(np.diff(c[:,:,1:],axis=1)<-1e-10):
            raise ValueError('invalid daily probabilities')
    def score(c, ix):
        return {str(cause):public_metrics(metrics(d.times[ix], d.events[ix], c[:,:,cause], cause=cause)) for cause in (1,2)}
    delta = oof[:,90,1]-full[:,90,1]
    counts = np.rint(np.expm1(d.history_raw[train,2])).astype(int)
    layers = {}
    for name, mask in [('first',counts==0),('one_two',(counts>=1)&(counts<=2)),('three_plus',counts>=3)]:
        layers[name] = dict(admissions=int(mask.sum()), mean_risk90_difference=float(delta[mask].mean()),
            risk90_difference_quantiles=np.quantile(delta[mask],[0,.1,.5,.9,1]).tolist())
    return dict(status='baseline_only_diagnostic_not_clinical_gain', fits=6, clinical_fits=0, goal_achieved=False,
        train_oof=score(oof,train), train_in_sample=score(full,train), tuning_fresh_refit=score(tuning,tune),
        readmission_paired_oof_minus_in_sample=paired_interval(d.arrays['subject_ids'][train],d.times[train],d.events[train],
            oof[:,:,1],full[:,:,1],replicates=200,seed=SEED), layers=layers, fit_information=fit_infos,
        mean_absolute_risk90_change=float(np.abs(delta).mean()),
        risk90_difference_quantiles=np.quantile(delta,[0,.1,.5,.9,1]).tolist(),
        caveats=['single seed, five fitted inner folds are not five clinical seeds',
                 '80% inner vs100% full training; composition, scale and sample-size effects not isolated',
                 'not evidence that OOF improves clinical residual; no clinical network fitted',
                 'fresh recipe differs from old pretrained/validation-selected H34; no old/new attribution',
                 'tuning exposed; no confirmation or historical test evaluation'])


def run():
    d, train, tune, contract = data_contract()
    if contract != json.loads((ROOT/'contract.json').read_text()):
        raise ValueError('contract changed after preflight')
    if json.loads((ROOT/'state.json').read_text())['status'] != 'preflight_passed_not_started':
        raise ValueError('already started; no repeated fits')
    assignment = fold_assignment(d.arrays['subject_ids'][train])
    ap, kp = np.empty((len(train),2),np.float32), np.empty((len(train),2),np.float32)
    infos = {}
    for f in range(FOLDS+1):
        name = f'fold{f}' if f<FOLDS else 'full'
        fit_ix = train[assignment!=f] if f<FOLDS else train
        predict_ix = train[assignment==f] if f<FOLDS else np.r_[train,tune]
        scales = fit_scales(d.history_raw,fit_ix)
        x = matrix(d.history_raw,d.current,d.current_masks,scales)
        folder = ROOT/name; folder.mkdir()
        write_json(ROOT/'state.json',dict(status='fitting',active=name,completed_fits=f,goal_achieved=False))
        model, info = fit_model(*tensors(d,x,fit_ix))
        torch.save(dict(state_dict=model.cpu().state_dict(),seed=SEED,updates=UPDATES),folder/'terminal.pt')
        model.cuda()
        a,k = predict(model,torch.as_tensor(x[predict_ix],device='cuda'))
        info.update(scales=scales,fit_index_hash=digest_array(fit_ix), prediction_index_hash=digest_array(predict_ix),
                    fit_matrix_hash=digest_array(x[fit_ix]), prediction_matrix_hash=digest_array(x[predict_ix]),
                    checkpoint_sha256=sha256(folder/'terminal.pt'), fit_admissions=len(fit_ix),
                    fit_patients=len(np.unique(d.arrays['subject_ids'][fit_ix])))
        with torch.no_grad():
            info['prediction_nll'] = float(competing_nll(RiskOutput(torch.as_tensor(a,device='cuda'),torch.as_tensor(k,device='cuda')),
                torch.as_tensor(d.times[predict_ix],device='cuda'),torch.as_tensor(d.events[predict_ix],device='cuda',dtype=torch.long)))
        np.savez_compressed(folder/'predictions.npz',indices=predict_ix,log_amplitude=a,shape=k)
        write_json(folder/'fit.json',info); infos[name]=info
        if f<FOLDS: ap[assignment==f],kp[assignment==f]=a,k
        else: full_a,full_k,tune_a,tune_k=a[:len(train)],k[:len(train)],a[len(train):],k[len(train):]
        print(json.dumps(dict(completed=name,nll=info['history'][-1]['fixed_train_nll'],seconds=info['seconds'])),flush=True)
        del model
    if len({i['initial_weight_digest'] for i in infos.values()})!=1:
        raise ValueError('initialization mismatch')
    np.savez_compressed(ROOT/'oof_risk_interface.npz',indices=train,fold=assignment,log_amplitude=ap,shape=kp)
    result=report(d,train,tune,assignment,ap,kp,full_a,full_k,tune_a,tune_k,infos)
    write_json(ROOT/'diagnostic.json',result)
    write_json(ROOT/'state.json',dict(status='fits_complete_pending_independent_verification',model_fits=6,clinical_fits=0,goal_achieved=False))
    print(json.dumps(dict(status=result['status'],fits=6,mean_absolute_risk90_change=result['mean_absolute_risk90_change'])),flush=True)


def verify(tag):
    d,train,tune,contract=data_contract()
    if contract != json.loads((ROOT/'contract.json').read_text()): raise ValueError('contract changed')
    assignment=fold_assignment(d.arrays['subject_ids'][train])
    with np.load(ROOT/'partition.npz') as z:
        for name,a in [('fold',assignment),('train_indices',train),('tuning_indices',tune)]: np.testing.assert_array_equal(z[name],a)
    ap,kp=np.empty((len(train),2),np.float32),np.empty((len(train),2),np.float32)
    infos,errors={},{}
    for f in range(FOLDS+1):
        name=f'fold{f}' if f<FOLDS else 'full'; folder=ROOT/name
        info=json.loads((folder/'fit.json').read_text())
        fit_ix=train[assignment!=f] if f<FOLDS else train
        pred_ix=train[assignment==f] if f<FOLDS else np.r_[train,tune]
        if f<FOLDS and np.intersect1d(d.arrays['subject_ids'][fit_ix],d.arrays['subject_ids'][pred_ix]).size:
            raise ValueError('patient overlap')
        scales=fit_scales(d.history_raw,fit_ix);x=matrix(d.history_raw,d.current,d.current_masks,scales)
        if scales!=info['scales'] or digest_array(x[fit_ix])!=info['fit_matrix_hash'] or digest_array(x[pred_ix])!=info['prediction_matrix_hash']:
            raise ValueError('inner scales/input differ')
        if (sha256(folder/'terminal.pt')!=info['checkpoint_sha256'] or info['updates']!=UPDATES or
            info['selected_update']!=UPDATES or info['initial_weight_digest']!=weight_digest(FreshH34())):
            raise ValueError('fit/checkpoint contract differs')
        m=FreshH34().cuda();m.load_state_dict(torch.load(folder/'terminal.pt',map_location='cuda',weights_only=True)['state_dict'])
        a,k=predict(m,torch.as_tensor(x[pred_ix],device='cuda'))
        with np.load(folder/'predictions.npz') as z:
            np.testing.assert_array_equal(pred_ix,z['indices'])
            error=max(float(np.abs(a-z['log_amplitude']).max()),float(np.abs(k-z['shape']).max()))
        if error>3e-7: raise ValueError('prediction replay error')
        errors[name]=error;infos[name]=info
        if f<FOLDS:ap[assignment==f],kp[assignment==f]=a,k
        else: full_a,full_k,tune_a,tune_k=a[:len(train)],k[:len(train)],a[len(train):],k[len(train):]
        del m
    with np.load(ROOT/'oof_risk_interface.npz') as z:
        for name,a in [('indices',train),('fold',assignment),('log_amplitude',ap),('shape',kp)]:np.testing.assert_array_equal(z[name],a)
    r=report(d,train,tune,assignment,ap,kp,full_a,full_k,tune_a,tune_k,infos)
    if r!=json.loads((ROOT/'diagnostic.json').read_text()):raise ValueError('numeric report differs')
    proof=dict(status='independently_verified',models=6,maximum_replay_error=max(errors.values()),
               replay_errors=errors,report_sha256=sha256(ROOT/'diagnostic.json'),contract_sha256=sha256(ROOT/'contract.json'),
               patient_partition_rebuilt=True,inner_scales_rebuilt=True,confirmation_test_not_evaluated=True)
    write_json(ROOT/f'verification_{tag}.json',proof)
    if all((ROOT/f'verification_{t}.json').exists() for t in ('a','b')):
        for t in ('a','b'):
            if json.loads((ROOT/f'verification_{t}.json').read_text())!=proof:raise ValueError('independent verification differs')
        write_json(ROOT/'state.json',dict(status='baseline_diagnostic_complete_verified',model_fits=6,clinical_fits=0,goal_achieved=False))
    print(json.dumps(proof),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('preflight','run','verify'),required=True)
    parser.add_argument('--tag',choices=('a','b'),default='a');args=parser.parse_args()
    runtime_setup()
    try:
        if args.mode=='preflight':preflight()
        elif args.mode=='run':run()
        else:verify(args.tag)
    except Exception as error:
        if ROOT.exists():write_json(ROOT/f'failure_{args.mode}_{args.tag}.json',dict(error=repr(error),goal_achieved=False))
        raise


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/oof_baseline_report.py · lines 1–49

<!-- REVIEW_FRAGMENT {"path":"training/v6/oof_baseline_report.py","part":1,"start_line":1,"end_line":49,"source_sha256":"8af01c31d7c7e71545ce33d19f1fcc033c30b42949a29eb142ec6f10e594b469","fragment_sha256":"8af01c31d7c7e71545ce33d19f1fcc033c30b42949a29eb142ec6f10e594b469","added_terminal_newline":false} -->
````python
"""OOF基线有限诊断图；不选择模型或更新拟合来源。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patient_data.export_mimiciv import sha256
from training.v6.oof_baseline_diagnostic import ROOT
from training.v6.module_surgery import probability_curve


def main():
    report=json.loads((ROOT/'diagnostic.json').read_text())
    for tag in ('a','b'):
        proof=json.loads((ROOT/f'verification_{tag}.json').read_text())
        if proof['report_sha256']!=sha256(ROOT/'diagnostic.json') or proof['maximum_replay_error']!=0:
            raise ValueError('unverified result')
    with np.load(ROOT/'oof_risk_interface.npz') as z:
        indices=z['indices']; a=z['log_amplitude']; k=z['shape']
    oof=probability_curve(a,k)[:,90,1]
    with np.load(ROOT/'full/predictions.npz') as z:
        np.testing.assert_array_equal(indices,z['indices'][:len(indices)])
        full=probability_curve(z['log_amplitude'][:len(indices)],z['shape'][:len(indices)])[:,90,1]
    delta=(oof-full)*100
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    for name,info in report['fit_information'].items():
        history=info['history']
        axs[0].plot([r['update'] for r in history],[r['fixed_train_nll'] for r in history],label=name,
                    linewidth=2.5 if name=='full' else 1.3)
    axs[0].set(title='Fresh H34: fixed training NLL',xlabel='Full-batch updates',ylabel='Joint survival NLL (lower better)')
    axs[0].legend(ncol=2);axs[0].grid(alpha=.2)
    ordered=np.sort(delta)
    axs[1].plot(ordered,np.arange(1,len(delta)+1)/len(delta),color='#1f77b4',linewidth=2)
    axs[1].axvline(0,color='black',linewidth=1)
    axs[1].set(title='Same training records: OOF minus in-sample risk',
               xlabel='90-day readmission probability difference (percentage points)',ylabel='Cumulative fraction of records',
               xlim=(-2.5,2.5),ylim=(0,1))
    outside=int((np.abs(delta)>2.5).sum())
    axs[1].text(.04,.82,f'Mean absolute difference: {np.abs(delta).mean():.3f} pp\n'
        f'10th / 90th percentiles: {np.quantile(delta,.1):.3f} / {np.quantile(delta,.9):.3f} pp\n'
        f'{outside} records outside displayed x-range',transform=axs[1].transAxes,fontsize=10)
    axs[1].grid(alpha=.2)
    fig.suptitle('Patient-level 5-fold baseline diagnostic; one seed; no clinical network retraining',fontsize=13)
    fig.text(.5,.015,'80% inner-fold fits vs 100% full fit: sample-size / composition effects are not isolated; not a clinical gain.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(ROOT/'oof_baseline_diagnostic.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/original_expansion_figures.py · lines 1–82

<!-- REVIEW_FRAGMENT {"path":"training/v6/original_expansion_figures.py","part":1,"start_line":1,"end_line":82,"source_sha256":"b276f4d4a186b56ffb8441b5fb26414d5c7a7aa0f2db934a85574abd01e94b29","fragment_sha256":"b276f4d4a186b56ffb8441b5fb26414d5c7a7aa0f2db934a85574abd01e94b29","added_terminal_newline":false} -->
````python
"""已保存预测的按日排序/校准和历史层分析；不选择检查点、不训练。"""
import argparse
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.expanded_original_experiment import ARMS,checked
from training.v6.evaluation import metrics,public_metrics,paired_interval
from training.v6.train import write_json


def render(root):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((root/'report.json').read_text());cfg=json.loads((root/'config.json').read_text())
    proof=json.loads((root/'fresh_process_verification.json').read_text())
    if proof['report_sha256']!=sha256(root/'report.json'):raise ValueError('independent proof stale')
    args=argparse.Namespace(**{k:Path(v) for k,v in cfg.items()});d,*_=checked(args)
    idx=np.flatnonzero(d.arrays['patient_splits']=='test');prior=np.expm1(d.history_raw[idx,2]).round().astype(int)
    strata={};curves={};selected=report['validation_selected_arm']
    labels={'control':'Original V6','more_labs':'Additional labs','more_meds':'Drug identity + dose/frequency',
       'both':'Additional labs + drug details','interaction_only':'Original V6 pre-pooling interaction','both_interaction':'Expanded inputs + interaction'}
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for column,seed in enumerate((42,7)):
        strata[str(seed)]={};curves[seed]={}
        for arm in ARMS:
            with np.load(root/f'seed{seed}/{arm}/test_predictions.npz') as z:a={k:z[k] for k in ('hadm_ids','subject_ids','times','events','curve')}
            if not np.array_equal(a['hadm_ids'],d.arrays['hadm_ids'][idx]):raise ValueError('saved members mismatch')
            curves[seed][arm]=a
            detail=metrics(a['times'],a['events'],a['curve'][:,:,1])
            axes[column].plot(np.arange(7,366),detail['auc_curve'][7:],label=labels[arm],lw=1.3)
            strata[str(seed)][arm]={}
            for name,keep in (('prior_0',prior==0),('prior_1_2',(prior>=1)&(prior<=2)),('prior_3_plus',prior>=3)):
                strata[str(seed)][arm][name]=public_metrics(metrics(a['times'][keep],a['events'][keep],a['curve'][keep,:,1]))
        axes[column].set_title(f'Seed {seed}: reused-test AUC(t)');axes[column].set_xlabel('Days after discharge');axes[column].set_ylabel('AUC(t)')
        axes[column].legend(fontsize=7);axes[column].grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'auc_by_day.png',dpi=180);plt.close(fig)
    comparisons={}
    for seed in (42,7):
        comparisons[str(seed)]={};control=curves[seed]['control']
        for arm in ('more_labs','interaction_only'):
            a=curves[seed][arm];comparisons[str(seed)][arm]={}
            for name,keep in (('prior_0',prior==0),('prior_1_2',(prior>=1)&(prior<=2)),('prior_3_plus',prior>=3)):
                comparisons[str(seed)][arm][name]=paired_interval(a['subject_ids'][keep],a['times'][keep],a['events'][keep],
                     a['curve'][keep,:,1],control['curve'][keep,:,1],seed=seed)
    write_json(root/'history_strata.json',dict(metrics=strata,comparisons=comparisons,
       definition='prior completed COPD-coded stays from fixed original history; no new cohort selection',
       limits='reused test; exploratory subgroup/multiple comparisons; low cases can prevent stable AUC'))
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    for row,seed in enumerate((42,7)):
        for column,day in enumerate((90,365)):
            ax=axes[row,column]
            for arm in dict.fromkeys(('control',selected,'interaction_only')):
                bins=report['models'][str(seed)][arm]['folds']['test']['calibration'][str(day)]
                ax.plot([b['predicted'] for b in bins],[b['observed'] for b in bins],'o-',label=labels[arm])
            ax.plot([0,1],[0,1],'k--',lw=.8);ax.set_title(f'Seed {seed}: day {day}');ax.set_xlabel('Predicted cumulative risk');ax.set_ylabel('Observed World-A fraction')
            ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Reused-test calibration; no post-hoc test recalibration');fig.tight_layout();fig.savefig(root/'calibration.png',dpi=170);plt.close(fig)
    # 训练报告的数值不改，仅重画更易读的英文标签和刻度，并补同架构21/29药物对照。
    pairs=[(labels[a],f'{a} minus control') for a in ARMS[1:]]+[
       ('Interaction on expanded inputs','both_interaction minus both')]
    width_file=root/'molecule_width_control/report.json'
    width=json.loads(width_file.read_text()) if width_file.exists() else None
    if width:pairs.append(('29 versus 21 drug identities',None))
    fig,axes=plt.subplots(1,2,figsize=(15,7),sharey=True)
    from matplotlib.ticker import MaxNLocator
    for j,(_,key) in enumerate(pairs):
        for seed,color,offset in ((42,'C0',-.08),(7,'C1',.08)):
            rr=report['comparisons'][str(seed)]['test'][key] if key else width['comparisons'][str(seed)]['test']
            for ax,value,ci in ((axes[0],'integrated_brier_difference','integrated_brier_95ci'),(axes[1],'auc90_difference','auc90_95ci')):
                ax.plot(rr[ci],[j+offset]*2,color=color,alpha=.6);ax.scatter(rr[value],j+offset,color=color,label=f'Seed {seed}' if j==0 else None)
    for ax in axes:
        ax.axvline(0,color='black',lw=1);ax.set_yticks(range(len(pairs)),[n for n,_ in pairs]);ax.grid(axis='x',alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.legend(fontsize=9)
    axes[0].invert_yaxis();axes[0].ticklabel_format(axis='x',style='sci',scilimits=(-3,-3))
    axes[0].set_xlabel('IBS7-365 difference (lower better)');axes[1].set_xlabel('AUC90 difference (higher better)')
    fig.suptitle('Original V6; candidate minus comparator; reused test\nPatient-clustered bootstrap 200; 95% intervals; no multiplicity adjustment')
    fig.tight_layout();fig.savefig(root/'paired_increment_readable.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_original_interaction_2026-10-12'));render(p.parse_args().root)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/original_interaction.py · lines 1–102

<!-- REVIEW_FRAGMENT {"path":"training/v6/original_interaction.py","part":1,"start_line":1,"end_line":102,"source_sha256":"53de0ae0eea98d0d95af09b1646d93ab376a6594abe831dc778b940e15759c56","fragment_sha256":"53de0ae0eea98d0d95af09b1646d93ab376a6594abe831dc778b940e15759c56","added_terminal_newline":false} -->
````python
"""保留原V6融合和旧投影，只在原池化前加入可控残差交互。"""
from dataclasses import dataclass
import torch
from torch import nn
from training.v6.model import masked_pool


@dataclass(frozen=True)
class ExpandedBatch:
    base: object
    extra_labs: torch.Tensor
    medication_details: torch.Tensor
    medication_present: torch.Tensor
    text_slots: torch.Tensor | None = None


def field_pool(content,index,n,score):
    if not len(content):return content.new_zeros(n,content.shape[-1]),torch.zeros(n,device=content.device,dtype=torch.bool)
    group=4*index[:,0]+index[:,1];logits=score(content).squeeze(-1)
    maximum=logits.new_full((4*n,),-torch.inf);maximum.scatter_reduce_(0,group,logits,reduce='amax',include_self=True)
    weights=torch.exp(logits-maximum[group]);total=weights.new_zeros(4*n).scatter_add_(0,group,weights)
    pooled=content.new_zeros(4*n,content.shape[-1]).index_add_(0,group,weights[:,None]/total[group,None]*content)
    present=(total>0).reshape(n,4)
    return pooled.reshape(n,4,-1).sum(1)/present.sum(1).clamp_min(1)[:,None],present.any(1)


class OriginalInteractionEncoder(nn.Module):
    def __init__(self,original,*,labs=False,medications=False,interaction=False,drug_count=38):
        super().__init__();self.original=original;self.use_labs=labs;self.use_meds=medications;self.interaction=interaction
        self.lab_extra=nn.Sequential(nn.Linear(10,32),nn.GELU(),nn.Linear(32,128),nn.LayerNorm(128))
        self.lab_ids=nn.Embedding(10,128);self.lab_score=nn.Linear(128,1)
        self.drug_extra=nn.Sequential(nn.Linear(23,32),nn.GELU(),nn.Linear(32,128),nn.LayerNorm(128))
        self.drug_ids=nn.Embedding(drug_count,128);self.drug_score=nn.Linear(128,1)
        self.old_lab_extra=nn.ModuleList((nn.Linear(10,128),nn.Linear(20,128)))
        self.old_ids=nn.Embedding(7,128)
        self.type_ids=nn.Embedding(7,128)
        self.attention=nn.TransformerEncoderLayer(128,4,256,dropout=.1,batch_first=True,norm_first=True)
        self.interaction_gate=nn.Parameter(torch.zeros(()))
        # 所有臂具有相同槽位，但不优化本臂不用的模块。
        for module in (self.lab_extra,self.lab_ids,self.lab_score):module.requires_grad_(labs)
        for module in (self.drug_extra,self.drug_ids,self.drug_score):module.requires_grad_(medications)
        for module in (self.old_lab_extra,self.old_ids,self.type_ids,self.attention):module.requires_grad_(interaction)
        self.interaction_gate.requires_grad_(interaction)

    def forward(self,batch,*,mask_pmh=False):
        if not isinstance(batch,ExpandedBatch):raise TypeError('ExpandedBatch required')
        if not (self.use_labs or self.use_meds or self.interaction):return self.original(batch.base,mask_pmh=mask_pmh)
        detail=batch.base;compact=detail.base;b=compact.clinical;n=len(b.current_values);core=self.original.base
        obs=b.lab_axis_inputs[:,:,6]>0
        lab=core.lab(torch.where(obs[:,:,None],b.lab_axis_inputs,0.))+core.lab_ids(torch.arange(22,device=obs.device))
        index=b.snippet_index;vectors=b.snippet_vectors
        if mask_pmh:
            keep=index[:,1]!=1;index=index[keep];vectors=vectors[keep]
        text=core.text_projection(vectors)+core.field_ids(index[:,1])
        text_length=max(1,max(torch.bincount(index[:,0],minlength=n).tolist()) if len(index) else 1)
        text_dense=lab.new_zeros(n,text_length,128);text_mask=torch.zeros(n,text_length,device=lab.device,dtype=torch.bool)
        # 原有片段没有时间序列位置信息；填槽仅为批处理，顺序不进入注意力。
        if batch.text_slots is None or mask_pmh:
            slots=torch.zeros(len(index),device=lab.device,dtype=torch.long)
            for i in range(n):
                r=torch.where(index[:,0]==i)[0]
                if len(r):slots[r]=torch.arange(len(r),device=lab.device)
        else:slots=batch.text_slots
        text_dense[index[:,0],slots]=text;text_mask[index[:,0],slots]=True
        oldlab=torch.cat((self.old_lab_extra[0](detail.labs[:,:50].reshape(n,5,10)),
                          self.old_lab_extra[1](detail.labs[:,50:].reshape(n,2,20))),1)+self.old_ids(torch.arange(7,device=lab.device))
        oldmask=torch.cat((detail.labs[:,:50].reshape(n,5,10)[:,:,5:].bool().any(2),
                           detail.labs[:,50:].reshape(n,2,20)[:,:,6:12].bool().any(2)),1)
        newlab=self.lab_extra(batch.extra_labs)+self.lab_ids(torch.arange(10,device=lab.device))
        newmask=batch.extra_labs[:,:,5:].bool().any(2)&self.use_labs
        named=self.drug_extra(batch.medication_details)+self.drug_ids(torch.arange(batch.medication_details.shape[1],device=lab.device))
        namedmask=batch.medication_present&self.use_meds
        med=core.med(compact.med_features.flatten(1));mp=compact.med_features[:,:,2:].bool().any((1,2))
        micro=core.microbiology(torch.cat((compact.micro_values,compact.micro_masks.float()),1));up=compact.micro_masks.any(1)
        support=core.support(torch.cat((compact.support_values,compact.support_masks.float()),1));vp=compact.support_masks.any(1)
        groups=(lab,text_dense,oldlab,newlab,named,torch.stack((med,micro),1),support[:,None])
        masks=(obs,text_mask,oldmask,newmask,namedmask,torch.stack((mp,up),1),vp[:,None])
        states=torch.cat(groups,1);present=torch.cat(masks,1);lengths=[x.shape[1] for x in groups]
        if self.interaction:
            types=torch.cat([torch.full((length,),i,device=lab.device,dtype=torch.long) for i,length in enumerate(lengths)])
            safe=present.clone();safe[~safe.any(1),0]=True
            # 明确残差门控为0时，原V6已有投影/池化输出保持不变。
            proposal=self.attention(states+self.type_ids(types),src_key_padding_mask=~safe)
            states=states+torch.tanh(self.interaction_gate)*proposal*present[:,:,None]
        lab,text_dense,updatedold,newlab,named,global_pair,support=states.split(lengths,1)
        l=masked_pool(lab,obs,core.lab_score)
        text,tp=field_pool(text_dense[index[:,0],slots],index,n,core.text_score)
        current=torch.where(b.current_masks,b.current_values,0.)
        s=masked_pool(core.current(torch.stack((current,b.current_masks.float()),-1))+
                      core.current_ids(torch.arange(14,device=lab.device)),b.current_masks,core.current_score)
        flags=(b.current_masks.any(1),obs.any(1),mp,up,vp)
        branches=[norm(x)*p[:,None] for norm,x,p in zip(core.branch_norms,(s,l,global_pair[:,0],global_pair[:,1],support[:,0]),flags)]
        presence=torch.stack((*flags,tp),1).float()
        hidden=core.fusion(torch.cat((*branches,text,presence),1))
        old_add=self.original.extra_labs(detail.labs)*detail.labs.ne(0).any(1)[:,None]
        if self.interaction:
            old_add=old_add+((updatedold-oldlab)*oldmask[:,:,None]).sum(1)/oldmask.sum(1).clamp_min(1)[:,None]
        additions=hidden.new_zeros(n,128)
        if self.use_labs:additions=additions+masked_pool(newlab,newmask,self.lab_score)
        if self.use_meds:additions=additions+masked_pool(named,namedmask,self.drug_score)
        return hidden+old_add+additions,dict(lab=l,text=text,presence=presence,extra=additions,
          interaction_gate=torch.tanh(self.interaction_gate))
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/oxygen_device_experiment.py · lines 1–231

<!-- REVIEW_FRAGMENT {"path":"training/v6/oxygen_device_experiment.py","part":1,"start_line":1,"end_line":231,"source_sha256":"cd1e8973863fdeda6f5ad0a9649afe00871afe9b18ff9dff0188974dbf42b5ab","fragment_sha256":"cd1e8973863fdeda6f5ad0a9649afe00871afe9b18ff9dff0188974dbf42b5ab","added_terminal_newline":false} -->
````python
"""呼吸低频状态转换有限早停配对；原教师完全冻结，原风险起步。"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from patient_data.oxygen_device import ROOT as DATA,derive,PROTOCOL
from training.v6.oxygen_device_model import OxygenDeviceCorrection,DeviceBatch,ARMS,view
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.curve_adaptation import FrozenBatch,caches
from training.v6.module_surgery import probability_curve
from training.v6.module_surgery_experiment import summary,save
from training.v6.risk import competing_nll
from training.v6.evaluation import metrics
from training.v6.early_stopping import EarlyStopping
from training.v6.train import write_json

ROOT=Path('training_runs/v6_oxygen_device_2026-10-13')
SEEDS=(42,7,101);CAP=60


def inputs():
    data=checked();d,c,h,ext,_,bins,tune,confirm,search,prior=data
    m=json.loads((DATA/'manifest.json').read_text());proof=json.loads((DATA/'independent_verification.json').read_text())
    if proof['output_sha256']!=sha256(DATA/'oxygen_device.npz') or proof['output_sha256']!=m['output_sha256']:
        raise ValueError('source input not independently verified')
    for p,digest in m['code_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('source builder changed')
    with np.load(DATA/'oxygen_device.npz') as z:arrays={k:z[k] for k in z.files}
    for key in ('hadm_ids','subject_ids','patient_splits'):np.testing.assert_array_equal(arrays[key],d.arrays[key])
    sources={**search['source_sha256'],**prior['source_sha256'],**m['code_sha256']}
    for p in ('training/v6/oxygen_device_model.py','training/v6/oxygen_device_experiment.py',
              'training/v6/curve_adaptation.py','training/v6/warm_residual_audit.py','training/v6/early_stopping.py',
              'training/tests/test_v6_oxygen_device_model.py'):sources[p]=sha256(Path(p))
    train=np.flatnonzero(d.arrays['patient_splits']=='train')
    if len(train)!=13592 or len(tune)!=1531:raise ValueError('outer split changed')
    contract=dict(source_sha256=sources,protocol_sha256=sha256(PROTOCOL),data_manifest_sha256=sha256(DATA/'manifest.json'),
        data_sha256=m['output_sha256'],data_provenance=d.provenance,compact_provenance=c.provenance,
        teacher_hashes={str(s):sha256(REFERENCE/f'seed{s}/control/best.pt') for s in (42,7)},
        baseline_hashes={str(s):sha256(Path(search['input_config']['parent'])/f'seed{s}/survival_base/best.pt') for s in (42,7)},
        seeds=list(SEEDS),source_mapping={'42':42,'7':7,'101':42},arms=list(ARMS),new_parameters={'global':4,'other':2436},
        optimizer={'name':'AdamW','lr':.001,'weight_decay':1e-4,'delta_penalty':1e-4,'clip':1.,'batch':512},
        early_stopping={'patience':10,'min_epochs':15,'min_delta':1e-5,'max_epochs':60},
        selection='mean tuning IBS; content must beat original/global/availability/timing for clinical nomination',
        confirmation_test_evaluated=False,goal_achieved=False)
    return data,train,arrays,contract


def augmented(data,indices,seed,arrays):
    frozen,proof=caches(data,indices,seed)
    x={key:torch.cat([getattr(b,key) for b in frozen]) for key in ('hidden','raw','prior','times','events')}
    base=frozen[0].shape_raw
    if any(not torch.equal(base,b.shape_raw) for b in frozen):raise ValueError('base shape differs across cache')
    result=[];hashes=hashlib.sha256(indices.tobytes())
    features=torch.as_tensor(arrays['features'][indices],device='cuda');observed=torch.as_tensor(arrays['observed'][indices],device='cuda')
    for value in (*x.values(),base,features,observed):hashes.update(value.cpu().numpy().tobytes())
    for offset in range(0,len(indices),512):
        sl=slice(offset,offset+512);f=FrozenBatch(x['hidden'][sl],x['raw'][sl],x['prior'][sl],base,x['times'][sl],x['events'][sl])
        result.append(DeviceBatch(f,features[sl],observed[sl]))
    return result,dict(teacher_cache=proof,augmented_cache_sha256=hashes.hexdigest(),records=len(indices))


def rebuild_data(arrays):
    rebuilt,_=derive()
    for key,value in arrays.items():np.testing.assert_array_equal(value,rebuilt[key])


def new_model(arm,seed):
    m=OxygenDeviceCorrection(arm,seed).cuda();torch.manual_seed(seed);return m


@torch.no_grad()
def prediction(model,cache):
    a,k=[],[];loss,n=0.,0
    for b in cache:
        out=model(b);a.append(out.log_amplitude.cpu().numpy());k.append(out.shape.cpu().numpy())
        loss+=float(competing_nll(out,b.frozen.times,b.frozen.events))*len(b.frozen.times);n+=len(b.frozen.times)
    aa,kk=np.concatenate(a),np.concatenate(k)
    return aa,kk,probability_curve(aa,kk),loss/n


def score(model,cache):
    a,k,curve,nll=prediction(model,cache)
    t=np.concatenate([b.frozen.times.cpu().numpy() for b in cache]);e=np.concatenate([b.frozen.events.cpu().numpy() for b in cache])
    r=metrics(t,e,curve[:,:,1]);death=metrics(t,e,curve[:,:,2],cause=2)
    return dict(nll=nll,ibs=r['integrated_brier_7_365'],auc90=r['points']['90']['auc'],death_ibs=death['integrated_brier_7_365'])


def preflight():
    data,train,arrays,contract=inputs()
    if ROOT.exists():raise FileExistsError('preflight already exists; inspect')
    ROOT.mkdir();write_json(ROOT/'contract.json',contract);rebuild_data(arrays)
    cached,proof=augmented(data,train[:128],42,arrays);b=cached[0]
    prior=None
    for arm in ARMS:
        m=new_model(arm,42);out=m(b)
        if prior is None:prior=out
        else:
            torch.testing.assert_close(out.log_amplitude,prior.log_amplitude,atol=0,rtol=0)
            torch.testing.assert_close(out.shape,prior.shape,atol=0,rtol=0)
        # 用真实训练来源存在者做单步，不能只拿全缺失批次宣称梯度启动。
        del m
    use=train[np.asarray(arrays['observed'][train],bool)][:128]
    observed,_=augmented(data,use,42,arrays);probe=new_model('content',42)
    opt=torch.optim.AdamW(probe.parameters(),lr=.001);before={k:v.detach().clone() for k,v in probe.state_dict().items()}
    out=probe(observed[0]);loss=competing_nll(out,observed[0].frozen.times,observed[0].frozen.events)
    loss.backward();torch.nn.utils.clip_grad_norm_(probe.parameters(),1.);opt.step()
    if torch.equal(before['head.2.weight'],probe.state_dict()['head.2.weight']):raise ValueError('real observed source cannot update')
    feature=torch.as_tensor(arrays['features'],device='cuda')
    for arm in ('availability','timing'):np.testing.assert_array_equal(view(feature,arm).cpu().numpy(),arrays[arm])
    write_json(ROOT/'preflight.json',dict(status='passed',common_initial_risk_error=0.,original_cache_error=proof['teacher_cache']['initial_parameter_max_error'],
        source_rebuilt=True,actual_value_time_ablation_matches=True,real_observed_GPU_update=True,gpu=torch.cuda.get_device_name(0)))
    write_json(ROOT/'state.json',dict(status='preflight_passed',fits=0,goal_achieved=False))
    print(json.dumps(dict(status='preflight_passed',original_risk_error=0.)),flush=True)


def fit(model,train_cache,tune_cache,d,tune,folder,seed,arm):
    folder.mkdir(parents=True);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);early=EarlyStopping();history=[];start=time.perf_counter();reason='max_epochs'
    for epoch in range(CAP+1):
        online=None
        if epoch:
            total,n=0.,0;model.train()
            for j in rng.permutation(len(train_cache)):
                b=train_cache[j];opt.zero_grad(set_to_none=True);out=model(b)
                nll=competing_nll(out,b.frozen.times,b.frozen.events);loss=nll+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite respiratory training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('nonfinite respiratory gradient')
                opt.step();total+=float(nll.detach())*len(b.frozen.times);n+=len(b.frozen.times)
            online=total/n
        detail=score(model,tune_cache);save_best,stop=early.observe(epoch,detail['ibs'])
        history.append(dict(epoch=epoch,train_nll=online,**{'tuning_'+k:v for k,v in detail.items()},early_stale=early.stale,early_anchor=early.anchor))
        if save_best:torch.save(dict(state_dict=model.state_dict(),epoch=epoch,seed=seed,arm=arm),folder/'best.pt')
        write_json(folder/'history.json',history)
        if epoch%5==0 or stop:print(json.dumps(dict(seed=seed,arm=arm,epoch=epoch,best=early.best_epoch,ibs=detail['ibs'],early_stop=stop)),flush=True)
        if stop:reason='patience_exhausted';break
    model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
    a,k,curve,nll=prediction(model,tune_cache);save(folder/'tuning_predictions.npz',d,tune,a,k,curve)
    info=dict(arm=arm,seed=seed,completed_epochs=history[-1]['epoch'],best_epoch=early.best_epoch,stop_reason=reason,early_state=asdict(early),
        seconds=time.perf_counter()-start,checkpoint_sha256=sha256(folder/'best.pt'),metrics=summary(curve,d,tune,nll))
    write_json(folder/'training_result.json',info);return info


def lock(report):
    scores={arm:float(np.mean([report['fits'][str(s)][arm]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS])) for arm in ARMS}
    scores['original_V6']=float(np.mean([report['references'][str(s)]['readmission']['integrated_brier_7_365'] for s in SEEDS]))
    selected=min(scores,key=scores.get)
    return dict(selected=selected,scores=scores,nominated=['content'] if selected=='content' and scores['content']<min(v for k,v in scores.items() if k!='content') else [],
        selection='predeclared3-seed tuning IBS; availability/timing/global gain not device-content gain',goal_achieved=False)


def run():
    data,train,arrays,contract=inputs();d,_,_,_,_,_,tune,_,_,_=data
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    if json.loads((ROOT/'state.json').read_text())['status']!='preflight_passed':raise ValueError('already started; inspect')
    report=dict(fits={},references={},cache_hashes={},confirmation_test_not_evaluated=True,goal_achieved=False)
    for seed in SEEDS:
        tc,tp=augmented(data,train,seed,arrays);vc,vp=augmented(data,tune,seed,arrays)
        report['cache_hashes'][str(seed)]={'train':tp,'tune':vp}
        ref=new_model('content',seed);a,k,curve,nll=prediction(ref,vc)
        report['references'][str(seed)]=summary(curve,d,tune,nll);del ref
        report['fits'][str(seed)]={}
        for arm in ARMS:
            write_json(ROOT/'state.json',dict(status='training_with_early_stop',seed=seed,arm=arm,goal_achieved=False))
            m=new_model(arm,seed);report['fits'][str(seed)][arm]=fit(m,tc,vc,d,tune,ROOT/f'seed{seed}'/arm,seed,arm);del m
        del tc,vc;torch.cuda.empty_cache()
    write_json(ROOT/'screening.json',report);write_json(ROOT/'nomination_lock.json',lock(report))
    write_json(ROOT/'state.json',dict(status='training_complete_pending_independent_replay',fits=12,goal_achieved=False))
    print(json.dumps(lock(report)),flush=True)


def verify(tag):
    data,train,arrays,contract=inputs();d,_,_,_,_,_,tune,_,_,_=data
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    rebuild_data(arrays);report=json.loads((ROOT/'screening.json').read_text());errors={}
    for seed in SEEDS:
        tc,tp=augmented(data,train,seed,arrays);vc,vp=augmented(data,tune,seed,arrays)
        if report['cache_hashes'][str(seed)]!={'train':tp,'tune':vp}:raise ValueError('teacher/current source cache differs')
        ref=new_model('content',seed);a,k,curve,nll=prediction(ref,vc)
        if summary(curve,d,tune,nll)!=report['references'][str(seed)]:raise ValueError('reference differs')
        del ref,tc
        for arm in ARMS:
            folder=ROOT/f'seed{seed}'/arm;info=json.loads((folder/'training_result.json').read_text())
            if info!=report['fits'][str(seed)][arm] or sha256(folder/'best.pt')!=info['checkpoint_sha256']:raise ValueError('checkpoint differs')
            history=json.loads((folder/'history.json').read_text());early=EarlyStopping();first_stop=None
            for row in history:
                _,stop=early.observe(row['epoch'],row['tuning_ibs'])
                if stop and first_stop is None:first_stop=row['epoch']
            if asdict(early)!=info['early_state'] or early.best_epoch!=info['best_epoch'] or info['completed_epochs']!=len(history)-1:
                raise ValueError('early stop state/selection differs')
            if info['stop_reason']=='patience_exhausted' and first_stop!=info['completed_epochs']:raise ValueError('early stop bypassed')
            m=new_model(arm,seed);stored=torch.load(folder/'best.pt',map_location='cuda',weights_only=True)
            if stored['epoch']!=early.best_epoch:raise ValueError('wrong saved epoch')
            m.load_state_dict(stored['state_dict']);a,k,curve,nll=prediction(m,vc)
            with np.load(folder/'tuning_predictions.npz') as z:
                np.testing.assert_array_equal(z['times'],d.times[tune]);np.testing.assert_array_equal(z['events'],d.events[tune])
                error=max(float(np.abs(a-z['log_amplitude']).max()),float(np.abs(k-z['shape']).max()),float(np.abs(curve-z['curve']).max()))
            if error>3e-7 or summary(curve,d,tune,nll)!=info['metrics']:raise ValueError('prediction/metric differs')
            errors[f'{seed}/{arm}']=error;del m
        del vc
    if lock(report)!=json.loads((ROOT/'nomination_lock.json').read_text()):raise ValueError('selection lock differs')
    proof=dict(status='independently_verified',models=12,maximum_error=max(errors.values()),replay_errors=errors,
        report_sha256=sha256(ROOT/'screening.json'),contract_sha256=sha256(ROOT/'contract.json'),
        source_inputs_and_full_teacher_caches_rebuilt=True,early_stop_replayed=True,confirmation_test_not_evaluated=True)
    write_json(ROOT/f'verification_{tag}.json',proof)
    if all((ROOT/f'verification_{t}.json').exists() for t in ('a','b')):
        for t in ('a','b'):
            if json.loads((ROOT/f'verification_{t}.json').read_text())!=proof:raise ValueError('independent proof differs')
        write_json(ROOT/'state.json',dict(status='screening_complete_independently_verified',fits=12,nominated=lock(report)['nominated'],goal_achieved=False))
    print(json.dumps(proof),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('preflight','run','verify'),required=True);p.add_argument('--tag',choices=('a','b'),default='a');args=p.parse_args()
    try:
        if args.mode=='preflight':preflight()
        elif args.mode=='run':run()
        else:verify(args.tag)
    except Exception as e:
        if ROOT.exists():write_json(ROOT/f'failure_{args.mode}_{args.tag}.json',dict(error=repr(e),goal_achieved=False))
        raise


if __name__=='__main__':main()


````
<!-- END REVIEW_FRAGMENT -->

## training/v6/oxygen_device_model.py · lines 1–53

<!-- REVIEW_FRAGMENT {"path":"training/v6/oxygen_device_model.py","part":1,"start_line":1,"end_line":53,"source_sha256":"48b72293463f71e8927fd101effffb9587bbd15e1b7488b866dce35500f49b8c","fragment_sha256":"48b72293463f71e8927fd101effffb9587bbd15e1b7488b866dce35500f49b8c","added_terminal_newline":false} -->
````python
"""冻结原V6风险及32维状态，新115维氧疗设备事实作有限修正。"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.risk import RiskOutput

ARMS=('global','availability','timing','content')
AVAIL=[90,91,92,93,114]
TIMING=AVAIL+list(range(94,98))+list(range(100,104))


@dataclass
class DeviceBatch:
    frozen: object
    features: torch.Tensor
    observed: torch.Tensor


def view(x,arm):
    if x.ndim!=2 or x.shape[1]!=115 or not torch.isfinite(x).all():raise ValueError('expected finite115 device input')
    if arm=='content':return x
    if arm not in ('availability','timing'):raise ValueError('unknown feature view')
    keep=AVAIL if arm=='availability' else TIMING
    out=torch.zeros_like(x);out[:,keep]=x[:,keep]
    return out


class OxygenDeviceCorrection(nn.Module):
    def __init__(self,arm,seed=42):
        super().__init__()
        if arm not in ARMS:raise ValueError('invalid arm')
        self.arm=arm
        if arm=='global':self.raw=nn.Parameter(torch.zeros(4))
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed);self.head=nn.Sequential(nn.Linear(147,16),nn.GELU(),nn.Linear(16,4))
            nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,b):
        f=b.frozen
        if b.features.shape!=(len(f.hidden),115) or b.observed.shape!=(len(f.hidden),):raise ValueError('device batch/index mismatch')
        if self.arm=='global':new=self.raw.expand(len(f.hidden),4)
        else:
            x=torch.cat((f.hidden,view(b.features,self.arm)),1)
            new=self.head(x)*b.observed[:,None]
        new=new.reshape(-1,2,2)
        delta=torch.stack((new[:,:,0],.25*torch.tanh(new[:,:,1])),2)
        original=f.raw.reshape(-1,2,2)
        return RiskOutput((f.prior+original[:,:,0]+delta[:,:,0]).clamp(-10,6),
            F.softplus(f.shape_raw+original[:,:,1]+delta[:,:,1])+.1,delta)


````
<!-- END REVIEW_FRAGMENT -->

## training/v6/oxygen_device_report.py · lines 1–77

<!-- REVIEW_FRAGMENT {"path":"training/v6/oxygen_device_report.py","part":1,"start_line":1,"end_line":77,"source_sha256":"e3bc4b21cab96f4b346d9dfde77c29680b6e9d7fdf20154fedf8832e544374e9","fragment_sha256":"e3bc4b21cab96f4b346d9dfde77c29680b6e9d7fdf20154fedf8832e544374e9","added_terminal_newline":false} -->
````python
"""已重放呼吸新增内容的配对、层内与图表；不改选点和默认模型。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patient_data.export_mimiciv import sha256
from training.v6.oxygen_device_experiment import ROOT,SEEDS,ARMS,inputs,augmented,new_model,prediction
from training.v6.evaluation import paired_interval,metrics,public_metrics
from training.v6.train import write_json


def build_report():
    saved=json.loads((ROOT/'screening.json').read_text())
    for tag in ('a','b'):
        proof=json.loads((ROOT/f'verification_{tag}.json').read_text())
        if proof['report_sha256']!=sha256(ROOT/'screening.json') or proof['maximum_error']!=0:raise ValueError('not independently verified')
    data,_,arrays,_=inputs();d,_,_,_,_,_,tune,_,_,_=data
    observed=arrays['observed'][tune];counts=np.rint(np.expm1(d.history_raw[tune,2])).astype(int)
    groups={'source_observed':observed,'source_missing':~observed,'first':counts==0,'one_two':(counts>=1)&(counts<=2),'three_plus':counts>=3}
    report=dict(pairs={},layers={},means={},actual_epochs={},unobserved_max_error={},no_confirmation_test=True,
        screening_sha256=sha256(ROOT/'screening.json'),code_sha256=sha256(Path(__file__)))
    fig,axs=plt.subplots(1,3,figsize=(15,4.5));colors={'global':'#7f7f7f','availability':'#1f77b4','timing':'#ff7f0e','content':'#2ca02c'}
    differences=[]
    for j,seed in enumerate(SEEDS):
        cache,_=augmented(data,tune,seed,arrays);original=new_model('content',seed);_,_,ref,_=prediction(original,cache);del original,cache
        curves={'original_V6':ref};report['actual_epochs'][str(seed)]={}
        for arm in ARMS:
            history=json.loads((ROOT/f'seed{seed}'/arm/'history.json').read_text());info=saved['fits'][str(seed)][arm]
            report['actual_epochs'][str(seed)][arm]={'actual':info['completed_epochs'],'best':info['best_epoch']}
            axs[j].plot([r['epoch'] for r in history],[r['tuning_ibs'] for r in history],label=arm,color=colors[arm])
            best=info['best_epoch'];axs[j].scatter(best,history[best]['tuning_ibs'],color=colors[arm],marker='D',s=30)
            with np.load(ROOT/f'seed{seed}'/arm/'tuning_predictions.npz') as z:curves[arm]=z['curve']
        axs[j].set(title=f'Seed {seed}: actual early-stopped epochs',xlabel='Epoch',ylabel='Tuning IBS7-365 (lower better)');axs[j].grid(alpha=.2);axs[j].legend(fontsize=8)
        for name in ('original_V6','availability','timing'):
            report['pairs'][f'{seed}/content minus {name}']=paired_interval(d.arrays['subject_ids'][tune],d.times[tune],d.events[tune],
                curves['content'][:,:,1],curves[name][:,:,1],replicates=200,seed=seed)
        report['unobserved_max_error'][str(seed)]=float(np.abs(curves['content'][~observed]-ref[~observed]).max())
        if report['unobserved_max_error'][str(seed)]!=0:raise ValueError('unobserved source changed original risks')
        report['layers'][str(seed)]={}
        for group,mask in groups.items():
            ix=tune[mask];values={}
            for name in ('original_V6','timing','content'):
                values[name]={'readmission':public_metrics(metrics(d.times[ix],d.events[ix],curves[name][mask,:,1])),
                    'death':public_metrics(metrics(d.times[ix],d.events[ix],curves[name][mask,:,2],cause=2))}
            report['layers'][str(seed)][group]=dict(admissions=int(mask.sum()),patients=len(np.unique(d.arrays['subject_ids'][ix])),models=values)
        p=report['pairs'][f'{seed}/content minus original_V6'];differences.append((p['integrated_brier_difference'],p['auc90_difference']))
    for arm in (*ARMS,'original_V6'):
        ms=[saved['references'][str(s)] if arm=='original_V6' else saved['fits'][str(s)][arm]['metrics'] for s in SEEDS]
        report['means'][arm]=dict(ibs=float(np.mean([m['readmission']['integrated_brier_7_365'] for m in ms])),
            auc90=float(np.mean([m['readmission']['points']['90']['auc'] for m in ms])),
            death_ibs=float(np.mean([m['death']['integrated_brier_7_365'] for m in ms])),
            timepoints={str(t):dict(auc=float(np.mean([m['readmission']['points'][str(t)]['auc'] for m in ms])),
                brier=float(np.mean([m['readmission']['points'][str(t)]['brier'] for m in ms]))) for t in (7,14,30,60,90,180,365)})
    fig.suptitle('Frozen original V6; availability / timing / actual oxygen device content; no forced100 epochs',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(ROOT/'early_stopping.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8));diff=np.asarray(differences)
    for k,ax in enumerate(axs):
        ax.axvline(0,color='black',linewidth=1);ax.scatter(diff[:,k],[-.12,0,.12],color=colors['content'],alpha=.6)
        ax.scatter(diff[:,k].mean(),0,marker='D',s=75,color=colors['content']);ax.set_yticks([0],['oxygen device content'])
        ax.set_xlabel('Paired difference across three seeds');ax.grid(alpha=.2)
    axs[0].set_title('IBS: content minus original\n(lower better)');axs[1].set_title('AUC90: content minus original\n(higher better)')
    axs[0].set_xlim(-.002,.002);axs[1].set_xlim(-.02,.02)
    axs[0].plot([],[],marker='o',linestyle='',color=colors['content'],alpha=.6,label='Clinical seed')
    axs[0].plot([],[],marker='D',linestyle='',color=colors['content'],label='Mean across three seeds');axs[0].legend(fontsize=8)
    fig.suptitle('Specified tuning set; fixed teacher; three clinical seeds / two teacher sources',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(ROOT/'paired_differences.png',dpi=150);plt.close(fig)
    report['limits']=['tuning set exposed; selected best checkpoints','fixed-model bootstrap not inner refitting or independent cohort',
        'three clinical seeds reuse two teacher sources','local sources and sparse supports; not causal recovery']
    write_json(ROOT/'paired_readout.json',report)
    print(json.dumps(dict(means=report['means'],unobserved_max_error=report['unobserved_max_error'])),flush=True)


if __name__=='__main__':build_report()


````
<!-- END REVIEW_FRAGMENT -->

## training/v6/patient_balance_audit.py · lines 1–150

<!-- REVIEW_FRAGMENT {"path":"training/v6/patient_balance_audit.py","part":1,"start_line":1,"end_line":150,"source_sha256":"44be885302a29daa403b34eda3a15345932ff33e88e33fc556350995e572d80d","fragment_sha256":"44be885302a29daa403b34eda3a15345932ff33e88e33fc556350995e572d80d","added_terminal_newline":false} -->
````python
"""患者均衡独立重放与一次确认；评价从不使用训练权重。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.patient_balance_experiment import ROOT,SEEDS,OLD
from training.v6.patient_balance_model import TRIALS,build
from training.v6.patient_weight_loss import training_weights
from training.v6.warm_residual import BASE_SEEDS,WarmTrial
from training.v6.warm_residual_audit import inputs as old_inputs,teacher,candidate as old_candidate
from training.v6.task_optimization_audit import load,check_curve,check_saved
from training.v6.module_surgery_experiment import batches,parameters,validation,summary,save
from training.v6.module_surgery import probability_curve
from training.v6.horizon_objective import restore
from training.v6.evaluation import paired_interval
from training.v6.task_optimization_diagnostics import analyze
from training.v6.regularization_diagnostics import history_counts
from training.v6.train import write_json


def inputs():
    result,_=old_inputs();contract=json.loads((ROOT/'contract.json').read_text())
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('balance protected source changed: '+p)
    for key,path in (('old_warm_contract_sha256',OLD/'contract.json'),('preflight_sha256',ROOT/'preflight.json'),
        ('training_weights_sha256',ROOT/'training_weights.npz'),('search_contract_sha256',Path('training_runs/v6_regularization_search_2026-10-12/contract.json'))):
        if sha256(path)!=contract[key]:raise ValueError('balance upstream proof changed')
    d=result[0];mask=d.arrays['patient_splits']=='train'
    with np.load(ROOT/'training_weights.npz') as z:
        for power in (0.,.5,1.):
            expected=np.zeros(len(mask),np.float32);expected[mask]=training_weights(d.arrays['subject_ids'][mask],power)
            np.testing.assert_array_equal(z[str(power)],expected)
    return result,contract


def model_for(base,seed,name,bins):
    if name=='original_teacher':return teacher(base,seed,bins),None,None
    if name=='full_uniform':
        m,info,best=old_candidate(base,seed,WarmTrial('warm_finetune',mode='finetune',lr=2e-5),bins)
        return m,dict(newly_trained=False,reused_from=str(OLD/f'seed{seed}/warm_finetune'),source_training=info),best
    trial=next(t for t in TRIALS if t.name==name)
    m,info,best=load(base,seed,trial,bins,root=ROOT,model_factory=build,baseline_seeds=BASE_SEEDS)
    if not info['frozen_end_and_best_verified']:raise ValueError('frozen-state training proof absent')
    if trial.head_only:
        reference=teacher(base,seed,bins)
        for key,value in reference.encoder.state_dict().items():torch.testing.assert_close(value,m.encoder.state_dict()[key],atol=0,rtol=0)
        if any(p.requires_grad for p in m.encoder.parameters()):raise ValueError('encoder not frozen after replay')
        del reference
    return m,info,best


def verify():
    if (ROOT/'.running.lock').exists():raise RuntimeError('balance training is running')
    (d,c,h,ext,_,bins,ix,_,search,_),contract=inputs();cache=batches(d,c,ext,ix)
    selected=json.loads((ROOT/'selected_before_confirmation.json').read_text())
    names=('original_teacher','full_uniform',*[t.name for t in TRIALS]);models={};errors={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1);models[str(seed)]={}
        for name in names:
            m,info,best=model_for(base,seed,name,bins)
            score=validation(m,cache,d.times[ix],d.events[ix])
            if best is not None:
                if any(score[k]!=best['tuning_'+k] for k in score):raise ValueError('balance best metric replay differs')
            elif score!=selected['references'][str(seed)][name]['tuning']:raise ValueError('balance reference replay differs')
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k);check_curve(curve)
            path=ROOT/f'seed{seed}'/name/'tuning_predictions.npz';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():error=check_saved(path,d,ix,a,k,curve)
            else:save(path,d,ix,a,k,curve);error=None
            errors[f'{seed}/{name}']=error;models[str(seed)][name]=dict(metrics=summary(curve,d,ix,nll),training=info);del m
    scores={name:float(np.mean([models[str(s)][name]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS])) for name in names}
    if scores!=selected['tuning_scores'] or min(scores,key=scores.get)!=selected['selected']:raise ValueError('balance choice differs')
    write_json(ROOT/'tuning_report.json',dict(models=models,tuning_scores=scores,selected=selected['selected'],
        confirmation_not_evaluated=True,test_not_evaluated=True,goal_achieved=False))
    replay=all(x is not None for x in errors.values())
    proof=dict(status='balance_tuning_independently_verified' if replay else 'balance_tuning_saved_pending_second_process',
        models=len(errors),errors=errors,maximum_error=max(errors.values()) if replay else None,
        report_sha256=sha256(ROOT/'tuning_report.json'),selection_sha256=sha256(ROOT/'selected_before_confirmation.json'),
        verifier_sha256=sha256(Path(__file__)),training_weights_recomputed=True,test_not_evaluated=True,goal_achieved=False)
    write_json(ROOT/'tuning_verification.json',proof)
    print(json.dumps(dict(status=proof['status'],models=len(errors),maximum_error=proof['maximum_error'])),flush=True)


def confirm():
    proof=json.loads((ROOT/'tuning_verification.json').read_text())
    if (proof['status']!='balance_tuning_independently_verified' or proof['models']!=21 or
        proof['verifier_sha256']!=sha256(Path(__file__)) or proof['report_sha256']!=sha256(ROOT/'tuning_report.json') or
        proof['selection_sha256']!=sha256(ROOT/'selected_before_confirmation.json')):
        raise ValueError('balance full independent proof required')
    selected=json.loads((ROOT/'selected_before_confirmation.json').read_text())['selected']
    target=ROOT/'nomination_lock.json'
    if target.exists():raise FileExistsError('balance confirmation already locked')
    ordinary=('original_teacher','full_uniform','head_uniform')
    reference='head_uniform' if selected.startswith('head_') else 'full_uniform'
    lock=dict(selected=selected,nominated=[] if selected in ordinary else [selected],matching_uniform=reference,
        selection_sha256=proof['selection_sha256'],verifier_sha256=proof['verifier_sha256'])
    write_json(target,lock)
    if not lock['nominated']:
        write_json(ROOT/'state.json',dict(status='verified_ordinary_reference_selected',selected=selected,
            no_patient_weight_candidate=True,goal_achieved=False));return
    evaluate_confirmation(lock,replay=False)


def evaluate_confirmation(lock,*,replay):
    if lock['selection_sha256']!=sha256(ROOT/'selected_before_confirmation.json') or lock['verifier_sha256']!=sha256(Path(__file__)):
        raise ValueError('balance nomination changed')
    (d,c,h,ext,_,bins,_,ix,search,_),_=inputs();cache=batches(d,c,ext,ix)
    names=('original_teacher',lock['matching_uniform'],lock['selected']);models={};errors={};pairs={};diagnostics={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1);models[str(seed)]={};curves={}
        for name in names:
            m,info,_=model_for(base,seed,name,bins);a,k,nll=parameters(m,cache);curve=probability_curve(a,k);check_curve(curve)
            path=ROOT/f'seed{seed}'/name/'confirmation_predictions.npz';path.parent.mkdir(parents=True,exist_ok=True)
            if replay:
                if not path.exists():raise FileNotFoundError('balance confirmation missing')
                errors[f'{seed}/{name}']=check_saved(path,d,ix,a,k,curve)
            else:
                if path.exists():raise FileExistsError('partial balance confirmation exists; inspect scene')
                save(path,d,ix,a,k,curve)
            models[str(seed)][name]=dict(metrics=summary(curve,d,ix,nll),training=info);curves[name]=curve;del m
        if not replay:
            pairs[str(seed)]={name:paired_interval(d.arrays['subject_ids'][ix],d.times[ix],d.events[ix],
                curves[lock['selected']][:,:,1],curves[name][:,:,1],replicates=200,seed=seed) for name in names[:2]}
            diagnostics[str(seed)],strata=analyze(d.times[ix],d.events[ix],d.arrays['subject_ids'][ix],
                history_counts(d.history_raw[ix]),curves,seed=seed,comparators=names[:2])
    if replay:
        prior=json.loads((ROOT/'confirmation_report.json').read_text())
        if models!=prior['models'] or prior['lock']!=lock:raise ValueError('balance confirmation report differs')
        write_json(ROOT/'confirmation_verification.json',dict(status='balance_confirmation_independently_verified',errors=errors,
            maximum_error=max(errors.values()),report_sha256=sha256(ROOT/'confirmation_report.json'),test_not_evaluated=True,goal_achieved=False))
        from training.v6.task_optimization_finish import material_signal
        decision=material_signal(models,lock['selected'],comparators=names[:2])
        write_json(ROOT/'continuation_decision.json',decision)
        write_json(ROOT/'state.json',dict(status='verified_balance_confirmation_complete',decision=decision,goal_achieved=False))
    else:
        write_json(ROOT/'confirmation_report.json',dict(models=models,comparisons=pairs,lock=lock,selected=lock['selected'],
            test_not_evaluated=True,goal_achieved=False))
        write_json(ROOT/'calibration_and_history_strata.json',dict(models=diagnostics,strata_admissions=strata,
            confirmation_report_sha256=sha256(ROOT/'confirmation_report.json'),goal_achieved=False))
        write_json(ROOT/'state.json',dict(status='balance_confirmation_pending_replay',goal_achieved=False))
    print(json.dumps(dict(status='balance_confirmation_replayed' if replay else 'balance_confirmation_written')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('verify','confirm','verify-confirmation'),required=True)
    mode=p.parse_args().mode
    if mode=='verify':verify()
    elif mode=='confirm':confirm()
    else:evaluate_confirmation(json.loads((ROOT/'nomination_lock.json').read_text()),replay=True)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/patient_balance_experiment.py · lines 1–144

<!-- REVIEW_FRAGMENT {"path":"training/v6/patient_balance_experiment.py","part":1,"start_line":1,"end_line":144,"source_sha256":"4a6b355948c9dfc4a97ac155b09e9258129ecf71ade04aedbd50289e64834eca","fragment_sha256":"4a6b355948c9dfc4a97ac155b09e9258129ecf71ade04aedbd50289e64834eca","added_terminal_newline":false} -->
````python
"""患者均衡×编码器可塑性有限搜索；固定逐出院验证，原V6不覆盖。"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.warm_residual_experiment import ROOT as OLD
from training.v6.warm_residual_audit import candidate,teacher,inputs as old_inputs
from training.v6.warm_residual import BASE_SEEDS,WarmTrial
from training.v6.patient_balance_model import TRIALS,build
from training.v6.patient_weight_loss import training_weights,weighted_nll
from training.v6.module_surgery_experiment import batches,validation
from training.v6.horizon_objective import restore
from training.v6.module_surgery import member_nll
from training.v6.train import write_json

ROOT=Path('training_runs/v6_patient_balance_2026-10-13')
SEEDS=(42,7,101)
PROTOCOL=Path('docs/V6_PATIENT_BALANCE_PROTOCOL_2026-10-13.md')


def fit(base,seed,trial,bins,cache,weights,d,tune,path):
    model=build(base,seed,trial,bins);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=trial.lr,weight_decay=trial.weight_decay)
    frozen={key:value.detach().clone() for key,value in model.state_dict().items()
        if key.startswith('baseline.') or (trial.head_only and key.startswith('encoder.'))}
    rows=[];best=math.inf;chosen=0;started=time.perf_counter()
    def record(epoch,nll=None,wnll=None,rate=0.):
        nonlocal best,chosen
        detail=validation(model,cache['tune'],d.times[tune],d.events[tune])
        rows.append(dict(epoch=epoch,train_nll=nll,train_weighted_nll=wnll,lr=rate,
            **{'tuning_'+key:value for key,value in detail.items()}))
        if detail['ibs']<best:
            best,chosen=detail['ibs'],epoch
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,tuning_ibs=best,trial=asdict(trial)),path/'best.pt')
        if epoch%20==0:
            write_json(path/'history.json',rows)
            print(json.dumps(dict(seed=seed,trial=trial.name,epoch=epoch,best_epoch=chosen,tuning_ibs=detail['ibs'],
                seconds=round(time.perf_counter()-started,1))),flush=True)
    path.mkdir(parents=True,exist_ok=True);record(0)
    for epoch in range(1,101):
        model.train();rate=trial.lr*min(1.,epoch/5)*(.1+.45*(1+math.cos(math.pi*max(0,epoch-5)/95)))
        for group in optimizer.param_groups:group['lr']=rate
        total=weighted_total=count=0.
        for j in rng.permutation(len(cache['train'])):
            pos,h,b,t,e=cache['train'][j];optimizer.zero_grad(set_to_none=True)
            out=model(h,b);nll=member_nll(out,t,e)
            wnll=weighted_nll(out,t,e,weights[torch.as_tensor(pos,device='cuda')])
            objective=wnll+trial.delta_penalty*out.delta.square().mean()
            if not torch.isfinite(objective):raise ValueError('nonfinite weighted clinical objective')
            objective.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite weighted clinical gradient')
            optimizer.step();total+=float(nll.detach())*len(t);weighted_total+=float(wnll.detach())*len(t);count+=len(t)
        record(epoch,total/count,weighted_total/count,rate)
    for key,value in frozen.items():
        if not torch.equal(value,model.state_dict()[key]):raise ValueError('frozen state changed during training: '+key)
    best_state=torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict']
    for key,value in frozen.items():
        if not torch.equal(value,best_state[key]):raise ValueError('frozen selected state changed: '+key)
    write_json(path/'history.json',rows)
    return dict(seed=seed,baseline_seed=BASE_SEEDS[seed],trial=asdict(trial),completed_epochs=100,best_epoch=chosen,
        tuning_ibs=best,checkpoint_sha256=sha256(path/'best.pt'),elapsed_seconds=time.perf_counter()-started,
        frozen_end_and_best_verified=True,newly_trained=True)


def run():
    (d,c,h,ext,_,bins,tune,_,search,_),_=old_inputs()
    preflight=json.loads((ROOT/'preflight.json').read_text())
    if preflight['status']!='gpu_initialization_and_ordinary_step_verified':raise ValueError('GPU preflight not completed')
    for p,digest in preflight['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('preflight source changed')
    if preflight['old_control_verification_sha256']!=sha256(OLD/'tuning_verification.json'):
        raise ValueError('ordinary reused control proof changed')
    source=('training/v6/patient_balance_model.py','training/v6/patient_weight_loss.py','training/v6/patient_balance_experiment.py',
        'training/v6/patient_balance_preflight.py','training/tests/test_v6_patient_weights.py',str(PROTOCOL),
        'training/v6/warm_residual.py','training/v6/regularization_multiseed.py')
    contract=dict(source_sha256={p:sha256(Path(p)) for p in source},preflight_sha256=sha256(ROOT/'preflight.json'),
        search_contract_sha256=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')),
        old_warm_contract_sha256=sha256(OLD/'contract.json'),trials=[asdict(t) for t in TRIALS],seeds=list(SEEDS),epochs=100,
        teachers={str(s):sha256(REFERENCE/f'seed{BASE_SEEDS[s]}/control/best.pt') for s in SEEDS},
        full_uniform_controls={str(s):sha256(OLD/f'seed{s}/warm_finetune/best.pt') for s in SEEDS},
        selection='unweighted admission tuning IBS three-seed mean',confirmation_not_evaluated=True,test_not_evaluated=True)
    train=np.flatnonzero(d.arrays['patient_splits']=='train')
    weights={}
    arrays={}
    for power in (0.,.5,1.):
        w=np.zeros(len(d.history),np.float32);w[train]=training_weights(d.arrays['subject_ids'][train],power)
        arrays[str(power)]=w;weights[power]=torch.as_tensor(w,device='cuda')
    wp=ROOT/'training_weights.npz'
    if wp.exists():
        with np.load(wp) as z:
            for key,value in arrays.items():np.testing.assert_array_equal(z[key],value)
    else:np.savez_compressed(wp,**arrays)
    contract['training_weights_sha256']=sha256(wp)
    if (ROOT/'contract.json').exists():
        if json.loads((ROOT/'contract.json').read_text())!=contract:raise ValueError('balance resume contract differs')
    else:write_json(ROOT/'contract.json',contract)
    cache=dict(train=batches(d,c,ext,train),tune=batches(d,c,ext,tune));fits={};references={}
    old_trial=WarmTrial('warm_finetune',mode='finetune',lr=2e-5)
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        references[str(seed)]={}
        for name in ('original_teacher','full_uniform'):
            if name=='original_teacher':model=teacher(base,seed,bins);info=None
            else:model,info,_=candidate(base,seed,old_trial,bins)
            references[str(seed)][name]=dict(tuning=validation(model,cache['tune'],d.times[tune],d.events[tune]),
                newly_trained=False,reused_training=info,reused_from=str(OLD/f'seed{seed}/warm_finetune') if info else None)
            del model
        fits[str(seed)]={}
        for trial in TRIALS:
            path=ROOT/f'seed{seed}'/trial.name
            if (path/'training_result.json').exists():
                info=json.loads((path/'training_result.json').read_text())
                if info['trial']!=asdict(trial) or info['completed_epochs']!=100 or info['checkpoint_sha256']!=sha256(path/'best.pt'):
                    raise ValueError('completed balance fit differs')
            else:
                if path.exists() and any((path/p).exists() for p in ('history.json','best.pt')):
                    raise RuntimeError('partial balance fit exists; preserve scene, no automatic restart')
                write_json(ROOT/'state.json',dict(status='running',seed=seed,trial=trial.name,completed=fits,goal_achieved=False))
                info=fit(base,seed,trial,bins,cache,weights[trial.power],d,tune,path)
                write_json(path/'training_result.json',info)
            fits[str(seed)][trial.name]=info
    scores={name:float(np.mean([references[str(s)][name]['tuning']['ibs'] for s in SEEDS])) for name in ('original_teacher','full_uniform')}
    scores.update({t.name:float(np.mean([fits[str(s)][t.name]['tuning_ibs'] for s in SEEDS])) for t in TRIALS})
    chosen=min(scores,key=scores.get)
    write_json(ROOT/'selected_before_confirmation.json',dict(fits=fits,references=references,tuning_scores=scores,selected=chosen,
        newly_trained_models=15,old_full_uniform_controls_reused=3,confirmation_not_evaluated=True,test_not_evaluated=True))
    write_json(ROOT/'state.json',dict(status='training_complete_pending_independent_replay',selected=chosen,goal_achieved=False))
    print(json.dumps(dict(status='balance_training_complete_pending_replay',selected=chosen)),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();ROOT.mkdir(exist_ok=True);lock=ROOT/'.running.lock'
    with lock.open('x',encoding='utf-8') as owner:owner.write(json.dumps(dict(pid=os.getpid(),started=time.time())))
    try:run()
    except Exception as error:
        write_json(ROOT/'state.json',dict(status='failed',error_type=type(error).__name__,error=str(error),goal_achieved=False));raise
    finally:lock.unlink()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/patient_balance_finish.py · lines 1–37

<!-- REVIEW_FRAGMENT {"path":"training/v6/patient_balance_finish.py","part":1,"start_line":1,"end_line":37,"source_sha256":"b9eff4d3f8f4b9a2ed2d24d01a8b0b294549c1e5170682b22a732ddd472ee2df","fragment_sha256":"b9eff4d3f8f4b9a2ed2d24d01a8b0b294549c1e5170682b22a732ddd472ee2df","added_terminal_newline":false} -->
````python
"""第七批完成后独立进程核验/锁定一次确认；不重复训练。"""
import json
import subprocess
import sys
from patient_data.export_mimiciv import sha256
from training.v6.patient_balance_experiment import ROOT
from training.v6.train import write_json


def run():
    if (ROOT/'.running.lock').exists():raise RuntimeError('patient balance training still running')
    if not (ROOT/'selected_before_confirmation.json').exists():raise RuntimeError('complete balance choice missing')
    def step(mode):
        cmd=[sys.executable,'-u','-m','training.v6.patient_balance_audit','--mode',mode]
        write_json(ROOT/'postprocess_state.json',dict(status='running',command=cmd,goal_achieved=False))
        subprocess.run(cmd,check=True)
    step('verify');step('verify')
    subprocess.run([sys.executable,'-u','-m','training.v6.patient_balance_report','--mode','tuning'],check=True)
    lockpath=ROOT/'nomination_lock.json'
    if not lockpath.exists():step('confirm')
    lock=json.loads(lockpath.read_text())
    if lock['selection_sha256']!=sha256(ROOT/'selected_before_confirmation.json'):raise ValueError('existing balance lock stale')
    if lock['nominated']:
        if not (ROOT/'confirmation_report.json').exists():raise RuntimeError('locked incomplete confirmation; inspect scene')
        step('verify-confirmation')
        subprocess.run([sys.executable,'-u','-m','training.v6.patient_balance_report','--mode','confirmation'],check=True)
    write_json(ROOT/'postprocess_state.json',dict(status='complete',selected=lock['selected'],nominated=lock['nominated'],goal_achieved=False))
    print(json.dumps(dict(status='balance_postprocess_complete',selected=lock['selected'])),flush=True)


if __name__=='__main__':
    lock=ROOT/'.postprocess.lock'
    with lock.open('x',encoding='utf-8') as owner:owner.write('patient_balance_finish')
    try:run()
    except Exception as error:
        write_json(ROOT/'postprocess_state.json',dict(status='failed',error_type=type(error).__name__,error=str(error),goal_achieved=False));raise
    finally:lock.unlink()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/patient_balance_model.py · lines 1–50

<!-- REVIEW_FRAGMENT {"path":"training/v6/patient_balance_model.py","part":1,"start_line":1,"end_line":50,"source_sha256":"2e9efceec1d0e0f0b8b1753bdcd1addcc3110144b0eb19847f14b227466c1f4e","fragment_sha256":"2e9efceec1d0e0f0b8b1753bdcd1addcc3110144b0eb19847f14b227466c1f4e","added_terminal_newline":false} -->
````python
"""仅损失权重/编码器可塑性的受控适配，不修改原V6网络。"""
from dataclasses import dataclass
import numpy as np
import torch
from training.v6.warm_residual import WarmTrial,build as warm_build


@dataclass(frozen=True)
class BalanceTrial(WarmTrial):
    mode:str='finetune'
    lr:float=2e-5
    power:float=0.
    head_only:bool=False


TRIALS=(BalanceTrial('head_uniform',head_only=True),BalanceTrial('head_sqrt',power=.5,head_only=True),
    BalanceTrial('head_inverse',power=1.,head_only=True),BalanceTrial('full_sqrt',power=.5),BalanceTrial('full_inverse',power=1.))
UNIFORM=BalanceTrial('full_uniform')


def build(base,seed,trial,bins):
    model=warm_build(base,seed,trial,bins)
    if trial.head_only:model.encoder.requires_grad_(False)
    # 不把冻结encoder强制eval，保持匹配训练dropout，仅改可塑性。
    return model


def uniform_step_proof(base,seed,bins,cache):
    """普通新损失与旧warm控制的实际GPU损失/梯度/更新对照。"""
    from training.v6.patient_weight_loss import weighted_nll
    from training.v6.module_surgery import member_nll
    previous=warm_build(base,seed,WarmTrial('warm_finetune',mode='finetune',lr=2e-5),bins)
    candidate=build(base,seed,UNIFORM,bins)
    _,history,batch,t,e=cache[np.random.default_rng(seed).permutation(len(cache))[0]]
    old_state=previous.state_dict();new_state=candidate.state_dict()
    for key,value in old_state.items():torch.testing.assert_close(value,new_state[key],atol=0,rtol=0)
    values=[];gradients=[]
    for index,model in enumerate((previous,candidate)):
        torch.manual_seed(seed);model.train()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5*.2,weight_decay=1e-4)
        out=model(history,batch)
        nll=member_nll(out,t,e) if index==0 else weighted_nll(out,t,e,torch.ones_like(t))
        objective=nll+1e-4*out.delta.square().mean();objective.backward()
        gradients.append({k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None})
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();values.append(float(objective.detach()))
    if values[0]!=values[1] or gradients[0].keys()!=gradients[1].keys():raise ValueError('ordinary balance/old warm step differs')
    for key,value in gradients[0].items():torch.testing.assert_close(value,gradients[1][key],atol=0,rtol=0)
    for key,value in previous.state_dict().items():torch.testing.assert_close(value,candidate.state_dict()[key],atol=0,rtol=0)
    return dict(loss=values[0],gradient_tensors=len(gradients[0]),maximum_gradient_error=0.,maximum_updated_weight_error=0.,
        limits='first-step and recipe equivalence; not a newly executed full 100-epoch trajectory')
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/patient_balance_preflight.py · lines 1–51

<!-- REVIEW_FRAGMENT {"path":"training/v6/patient_balance_preflight.py","part":1,"start_line":1,"end_line":51,"source_sha256":"37dd05944f05f68561b7b8c9610b47c8c4a3d6113d3f6a49b6824749c2ee6d8b","fragment_sha256":"37dd05944f05f68561b7b8c9610b47c8c4a3d6113d3f6a49b6824749c2ee6d8b","added_terminal_newline":false} -->
````python
"""第七批启动前GPU权重/初始化/旧控制首步匹配；不训练100轮。"""
import json
from pathlib import Path
import torch
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.warm_residual_audit import inputs as old_inputs
from training.v6.horizon_objective import restore
from training.v6.module_surgery_experiment import batches
from training.v6.patient_balance_model import TRIALS,build,uniform_step_proof
from training.v6.warm_residual_audit import teacher
from training.v6.warm_residual import BASE_SEEDS
from training.v6.train import write_json

ROOT=Path('training_runs/v6_patient_balance_2026-10-13')


def run():
    (d,c,h,ext,_,bins,_,_,search,_),_=old_inputs()
    old=Path('training_runs/v6_warm_residual_2026-10-13')
    p=json.loads((old/'tuning_verification.json').read_text())
    if p['status']!='tuning_independently_verified' or p['report_sha256']!=sha256(old/'tuning_report.json'):
        raise ValueError('old ordinary finetuning not independently verified')
    train=batches(d,c,ext,np.flatnonzero(d.arrays['patient_splits']=='train'));proof={}
    for seed in (42,7,101):
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        proof[str(seed)]=dict(uniform_step=uniform_step_proof(base,seed,bins,train),initial={})
        previous=teacher(base,seed,bins).eval()
        with torch.no_grad():
            _,hist,b,_,_=train[0];expected=previous(hist,b)
        for trial in TRIALS:
            m=build(base,seed,trial,bins).eval()
            with torch.no_grad():out=m(hist,b)
            torch.testing.assert_close(out.log_amplitude,expected.log_amplitude,atol=0,rtol=0)
            torch.testing.assert_close(out.shape,expected.shape,atol=0,rtol=0)
            if any(p.requires_grad for p in m.baseline.parameters()):raise ValueError('H34 not frozen')
            if trial.head_only and any(p.requires_grad for p in m.encoder.parameters()):raise ValueError('encoder not frozen')
            m.train()
            if not m.encoder.training:raise ValueError('head/full conditions differ in encoder dropout mode')
            proof[str(seed)]['initial'][trial.name]=dict(maximum_error=0.,trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad))
            del m
        del previous
    ROOT.mkdir(exist_ok=True)
    write_json(ROOT/'preflight.json',dict(status='gpu_initialization_and_ordinary_step_verified',seeds=proof,
        source_sha256={x:sha256(Path(x)) for x in ('training/v6/patient_balance_model.py','training/v6/patient_weight_loss.py','training/v6/patient_balance_preflight.py')},
        old_control_verification_sha256=sha256(old/'tuning_verification.json'),goal_achieved=False))
    print(json.dumps(dict(status='gpu_preflight_passed',seeds=3,initial_models=15,gradient_update_error=0.)),flush=True)


if __name__=='__main__':run()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/patient_balance_report.py · lines 1–59

<!-- REVIEW_FRAGMENT {"path":"training/v6/patient_balance_report.py","part":1,"start_line":1,"end_line":59,"source_sha256":"9c0aea5acc2f43aed8f5627724cbbbd6a160bf697d91ca721098a39fea851b34","fragment_sha256":"9c0aea5acc2f43aed8f5627724cbbbd6a160bf697d91ca721098a39fea851b34","added_terminal_newline":false} -->
````python
"""核验后的患者加权学习曲线与配对图；不根据图表重选。"""
import argparse
import json
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.patient_balance_experiment import ROOT,SEEDS,OLD
from training.v6.patient_balance_model import TRIALS
from training.v6.task_optimization_report import averages,prepare_plot
from training.v6.train import write_json


def run(mode):
    file='tuning_report.json' if mode=='tuning' else 'confirmation_report.json'
    p=json.loads((ROOT/('tuning_verification.json' if mode=='tuning' else 'confirmation_verification.json')).read_text())
    if p['report_sha256']!=sha256(ROOT/file):raise ValueError('balance report proof stale')
    r=json.loads((ROOT/file).read_text());plt=prepare_plot()
    if mode=='tuning':
        if p['status']!='balance_tuning_independently_verified':raise ValueError('independent tuning replay required')
        names=('original_teacher','full_uniform',*[t.name for t in TRIALS])
        fig,axes=plt.subplots(3,2,figsize=(12,10))
        for row,seed in enumerate(SEEDS):
            reference=r['models'][str(seed)]['original_teacher']['metrics']['readmission']['integrated_brier_7_365']
            for index,name in enumerate(names[1:]):
                path=OLD/f'seed{seed}/warm_finetune/history.json' if name=='full_uniform' else ROOT/f'seed{seed}'/name/'history.json'
                history=json.loads(path.read_text());style='--' if name=='full_uniform' else '-'
                axes[row,0].plot([x['epoch'] for x in history[1:]],[x['train_nll'] for x in history[1:]],ls=style,label=name,lw=1.)
                axes[row,1].plot([x['epoch'] for x in history],[x['tuning_ibs']-reference for x in history],ls=style,label=name,lw=1.)
            axes[row,1].axhline(0,color='black',ls=':',label='Original V6')
            axes[row,0].set_title(f'Seed {seed}: unweighted online training NLL',fontsize=10)
            axes[row,1].set_title(f'Seed {seed}: tuning IBS minus original V6 (negative better)',fontsize=10)
            for ax in axes[row]:ax.grid(alpha=.2);ax.set_xlabel('Epoch')
        fig.suptitle('Patient-balanced loss / encoder plasticity; fixed admission-level evaluation\n'
            '15 new complete 100-epoch fits; full_uniform is a verified reused control')
        fig.legend(*axes[0,1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.935),ncol=4)
        fig.tight_layout(rect=(0,0,1,.87));fig.savefig(ROOT/'training_curves.png',dpi=160);plt.close(fig)
    else:
        names=('original_teacher',r['lock']['matching_uniform'],r['selected'])
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for index,reference in enumerate(names[:2]):
            for row,seed in enumerate(SEEDS):
                result=r['comparisons'][str(seed)][reference]
                for ax,value,ci in zip(axes,(result['integrated_brier_difference'],result['auc90_difference']),
                    (result['integrated_brier_95ci'],result['auc90_95ci'])):
                    y=row+(index-.5)*.15;ax.plot(ci,[y,y],color=f'C{index}',label=reference if row==0 else None)
                    ax.scatter(value,y,color=f'C{index}',marker='D',s=25)
        for ax in axes:ax.axvline(0,color='gray');ax.set_yticks(range(3),[f'Seed {s}' for s in SEEDS]);ax.grid(axis='x',alpha=.2)
        axes[0].set_xlabel('IBS7-365: candidate minus comparator (negative better)')
        axes[1].set_xlabel('AUC90: candidate minus comparator (positive better)')
        fig.suptitle('Locked patient-balanced candidate; admission-level confirmation validation\n200 patient-cluster resamples; no multiplicity adjustment')
        axes[0].legend();fig.tight_layout();fig.savefig(ROOT/'confirmation_increment.png',dpi=180);plt.close(fig)
    stats=averages(r['models'],names)
    write_json(ROOT/(mode+'_readout.json'),dict(means=stats,selected=r['selected'],source_report_sha256=sha256(ROOT/file),
        goal_achieved=False,evaluation_unweighted=True))
    print(json.dumps(dict(status='verified_balance_figures_written',mode=mode,selected=r['selected'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('tuning','confirmation'),required=True)
    run(p.parse_args().mode)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/patient_gradient_diagnostic.py · lines 1–80

<!-- REVIEW_FRAGMENT {"path":"training/v6/patient_gradient_diagnostic.py","part":1,"start_line":1,"end_line":80,"source_sha256":"3af9152a92d1737ad2b62908f555ff1ed606706c992318f58832300bb5069ce4","fragment_sha256":"3af9152a92d1737ad2b62908f555ff1ed606706c992318f58832300bb5069ce4","added_terminal_newline":false} -->
````python
"""原V6风险头最后投影的真实逐记录梯度；上游固定eval，不概括全网络训练。"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.module_surgery_experiment import batches
from training.v6.warm_residual_audit import teacher
from training.v6.horizon_objective import restore
from training.v6.risk import RiskOutput
from training.v6.patient_weight_loss import per_record_nll,training_weights
from training.v6.train import write_json

ROOT=Path('training_runs/v6_patient_gradient_diagnostic_2026-10-13')


def aggregate(subjects,gradients):
    people,inverse,count=np.unique(subjects,return_inverse=True,return_counts=True)
    patient_grad=np.zeros((len(people),gradients.shape[1]),np.float64)
    np.add.at(patient_grad,inverse,gradients)
    norms=np.linalg.norm(gradients,axis=1)
    patient_norm_sum=np.bincount(inverse,weights=norms)
    net_norm=np.linalg.norm(patient_grad,axis=1)
    order=np.argsort(-count,kind='stable');groups={}
    for fraction in (1,5,10,20):
        n=max(1,int(np.ceil(len(people)*fraction/100)));mask=np.zeros(len(people),bool);mask[order[:n]]=True
        total=patient_grad.sum(0)
        groups[str(fraction)]=dict(patients=n,admission_share=float(count[mask].sum()/count.sum()),
            individual_gradient_norm_mass_share=float(patient_norm_sum[mask].sum()/patient_norm_sum.sum()),
            patient_net_gradient_norm_mass_share=float(net_norm[mask].sum()/net_norm.sum()),
            projection_on_total_gradient=float(patient_grad[mask].sum(0)@total/(total@total)) if total@total>1e-18 else None)
    schemes={}
    for power in (0.,.5,1.):
        w=training_weights(subjects,power)
        weighted=gradients*w[:,None];pg=np.zeros_like(patient_grad);np.add.at(pg,inverse,weighted)
        total=weighted.mean(0);unweighted=gradients.mean(0)
        denom=np.linalg.norm(total)*np.linalg.norm(unweighted)
        schemes[str(power)]=dict(mean_gradient_norm=float(np.linalg.norm(total)),
            cosine_with_unweighted=float(total@unweighted/denom) if denom>1e-18 else None,
            top10_patient_net_norm_share=float(np.linalg.norm(pg[order[:max(1,int(np.ceil(len(people)*.1)))]],axis=1).sum()/np.linalg.norm(pg,axis=1).sum()))
    return dict(top_patient_groups=groups,weight_schemes=schemes)


def run():
    d,c,h,ext,_,bins,_,_,search,_=checked();ix=np.flatnonzero(d.arrays['patient_splits']=='train')
    cache=batches(d,c,ext,ix);result={}
    for seed in (42,7):
        base=restore(Path(search['input_config']['parent']),seed,'survival_base',h,.1)
        model=teacher(base,seed,bins).eval();vectors=[]
        for _,history,batch,t,e in cache:
            with torch.no_grad():
                cstate,_=model.encoder(batch.base);hidden=model.delta_head[:-1](cstate)
                raw=model.delta_head[-1](hidden);prior=model.baseline(history);expected=model(history,batch)
            raw=raw.detach().requires_grad_(True);delta=raw.reshape(-1,2,2)
            out=RiskOutput((prior.log_amplitude+delta[:,:,0]).clamp(-10,6),
                F.softplus(model.baseline.raw_shape.detach()+delta[:,:,1])+.1)
            torch.testing.assert_close(out.log_amplitude,expected.log_amplitude,atol=0,rtol=0)
            torch.testing.assert_close(out.shape,expected.shape,atol=0,rtol=0)
            grad=torch.autograd.grad(per_record_nll(out,t,e).sum(),raw)[0]
            augmented=torch.cat((hidden,torch.ones(len(t),1,device='cuda')),1)
            vector=(grad[:,:,None]*augmented[:,None,:]).flatten(1)
            vectors.append(vector.detach().cpu().numpy())
        vectors=np.concatenate(vectors).astype(np.float64)
        if vectors.shape!=(len(ix),132) or not np.isfinite(vectors).all():raise ValueError('invalid gradient interface')
        result[str(seed)]=aggregate(d.arrays['subject_ids'][ix],vectors)
        result[str(seed)]['teacher_checkpoint_sha256']=sha256(REFERENCE/f'seed{seed}/control/best.pt')
        del model
    ROOT.mkdir(exist_ok=True)
    report=dict(status='last_projection_gradient_measured',models=result,patients=len(np.unique(d.arrays['subject_ids'][ix])),
        admissions=len(ix),source_sha256={p:sha256(Path(p)) for p in ('training/v6/patient_gradient_diagnostic.py','training/v6/patient_weight_loss.py')},
        label_sha256=d.provenance['label_sha256'],goal_achieved=False,
        limits=['Eval-mode upstream fixed; final risk projection only, not full-network training trajectory.',
            'Gradient directions can cancel; norm mass and projection are distinct.',
            'Only training labels analyzed; weighting does not prove better validation performance.'])
    write_json(ROOT/'diagnostic.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':run()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/patient_weight_diagnostic.py · lines 1–34

<!-- REVIEW_FRAGMENT {"path":"training/v6/patient_weight_diagnostic.py","part":1,"start_line":1,"end_line":34,"source_sha256":"a7f0d64ee7d3b13e536f8d9c65254ed6863c1c27e7c318910f88c78819adbee0","fragment_sha256":"a7f0d64ee7d3b13e536f8d9c65254ed6863c1c27e7c318910f88c78819adbee0","added_terminal_newline":false} -->
````python
"""只用训练患者住院数量诊断损失权重；不读取验证/测试结局，不训练。"""
import json
from pathlib import Path
import numpy as np
from training.v6.regularization_audit import checked
from training.v6.train import write_json

ROOT=Path('training_runs/v6_patient_weight_diagnostic_2026-10-13')


def run():
    d,_,_,_,_,_,_,_,_,_=checked()
    selected=d.arrays['patient_splits']=='train'
    people,count=np.unique(d.arrays['subject_ids'][selected],return_counts=True)
    descending=np.sort(count)[::-1]
    shares={}
    for percentile in (1,5,10,20):
        k=max(1,int(np.ceil(len(count)*percentile/100)))
        shares[str(percentile)]=dict(patients=k,admissions=int(descending[:k].sum()),
            admission_loss_share=float(descending[:k].sum()/count.sum()))
    schemes={}
    for power in (0.,.5,1.):
        mass=count.astype(float)**(1-power)
        schemes[str(power)]=dict(patient_mass_ratio_max_min=float(mass.max()/mass.min()),
            effective_patients=float(mass.sum()**2/(mass**2).sum()),
            definition=f'per-admission loss weight proportional to n_patient^(-{power})')
    ROOT.mkdir(exist_ok=True)
    r=dict(status='training_input_weight_diagnostic_not_model_results',patients=len(people),admissions=int(count.sum()),
        admission_count_quantiles=np.quantile(count,[0,.5,.9,.99,1]).tolist(),top_patient_shares=shares,
        candidate_weight_schemes=schemes,no_validation_or_test_labels_analyzed=True,no_model_training=True,goal_achieved=False)
    write_json(ROOT/'diagnostic.json',r);print(json.dumps(r),flush=True)


if __name__=='__main__':run()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/patient_weight_loss.py · lines 1–32

<!-- REVIEW_FRAGMENT {"path":"training/v6/patient_weight_loss.py","part":1,"start_line":1,"end_line":32,"source_sha256":"90c9749ffa38f76bf135c30809c2b8ad524786d70142a6f72d8140414092b3d4","fragment_sha256":"90c9749ffa38f76bf135c30809c2b8ad524786d70142a6f72d8140414092b3d4","added_terminal_newline":false} -->
````python
"""仅训练损失的患者权重；不改变模型输入、标签或评价人群。"""
import math
import numpy as np
import torch
from training.v6.risk import RiskOutput,competing_nll


def training_weights(subjects,power):
    subjects=np.asarray(subjects)
    if subjects.ndim!=1 or not len(subjects) or power not in (0.,.5,1.):
        raise ValueError('nonempty training subjects and fixed weight power required')
    _,inverse,count=np.unique(subjects,return_inverse=True,return_counts=True)
    weights=count[inverse].astype(np.float64)**(-power)
    # 全训练折均值1；不按每个minibatch重新归一化，避免改写目标。
    return (weights/weights.mean()).astype(np.float32)


def per_record_nll(out:RiskOutput,times,events):
    # 复用原校验；新向量式与原均值/梯度有显式回归对照。
    competing_nll(out,times,events)
    scaled=times.clamp_min(1e-3)[:,None]/365.
    a,k=out.log_amplitude,out.shape
    cumulative=torch.exp(a)*scaled.pow(k)
    log_hazard=a+torch.log(k)-math.log(365.)+(k-1)*torch.log(scaled)
    chosen=log_hazard.gather(1,(events-1).clamp_min(0)[:,None]).squeeze(1)
    return cumulative.sum(1)-torch.where(events>0,chosen,0.)


def weighted_nll(out,times,events,weights):
    if weights.shape!=times.shape or not torch.isfinite(weights).all() or (weights<=0).any():
        raise ValueError('positive globally normalized per-record weights required')
    return (per_record_nll(out,times,events)*weights).mean()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/pft_set_experiment.py · lines 1–246

<!-- REVIEW_FRAGMENT {"path":"training/v6/pft_set_experiment.py","part":1,"start_line":1,"end_line":246,"source_sha256":"3831173c5a0178d0f3805c79a68b5032beb1f0fea3fa2bb9e44816e1237994ea","fragment_sha256":"3831173c5a0178d0f3805c79a68b5032beb1f0fea3fa2bb9e44816e1237994ea","added_terminal_newline":false} -->
````python
"""背景PFT数值集合有限早停配对；保留原完整V6风险。"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from patient_data.pft_set_interface import ROOT as DATA,derive
from training.v6.pft_set_model import PFTCorrection,PFTBatch,ARMS
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.curve_adaptation import FrozenBatch,caches
from training.v6.module_surgery import probability_curve
from training.v6.module_surgery_experiment import summary,save
from training.v6.risk import competing_nll
from training.v6.evaluation import metrics
from training.v6.early_stopping import EarlyStopping
from training.v6.train import write_json

ROOT=Path('training_runs/v6_pft_set_2026-10-13_r2')
PROTOCOL=Path('docs/V6_PFT_SET_TRAINING_PROTOCOL_2026-10-13.md')
SEEDS=(42,7,101);CAP=60



def inputs():
    data=checked();d,c,h,ext,_,bins,tune,confirm,search,prior=data
    m=json.loads((DATA/'manifest.json').read_text())
    for tag in ('a','b'):
        p=json.loads((DATA/f'verification_{tag}.json').read_text())
        if p['maximum_error'] or p['manifest_sha256']!=sha256(DATA/'manifest.json'):
            raise ValueError('PFT interface source proof differs')
    for name,digest in m['source_sha256'].items():
        if sha256(Path(name))!=digest:raise ValueError('PFT source changed: '+name)
    for name,digest in m['output_sha256'].items():
        if sha256(DATA/name)!=digest:raise ValueError('PFT interface output changed')
    with np.load(DATA/'interface.npz') as z:arrays={k:z[k] for k in z.files}
    for key in ('hadm_ids','subject_ids','patient_splits'):np.testing.assert_array_equal(arrays[key],d.arrays[key])
    train=np.flatnonzero(d.arrays['patient_splits']=='train')
    if len(train)!=13592 or len(tune)!=1531:raise ValueError('outer split changed')
    sources={**search['source_sha256'],**prior['source_sha256'],**m['source_sha256']}
    for name in ('training/v6/pft_set_model.py','training/v6/pft_set_experiment.py','training/v6/curve_adaptation.py',
                 'training/v6/warm_residual_audit.py','training/v6/early_stopping.py','training/tests/test_v6_pft_set_model.py',
                 'training/tests/test_v6_pft_set_experiment.py'):
        sources[name]=sha256(Path(name))
    contract=dict(source_sha256=sources,protocol_sha256=sha256(PROTOCOL),data_manifest_sha256=sha256(DATA/'manifest.json'),
        data_sha256=m['output_sha256']['interface.npz'],fact_sha256=m['output_sha256']['facts.jsonl.gz'],
        scales=m['scales'],data_provenance=d.provenance,compact_provenance=c.provenance,
        teacher_hashes={str(s):sha256(REFERENCE/f'seed{s}/control/best.pt') for s in (42,7)},
        baseline_hashes={str(s):sha256(Path(search['input_config']['parent'])/f'seed{s}/survival_base/best.pt') for s in (42,7)},
        seeds=list(SEEDS),source_mapping={'42':42,'7':7,'101':42},arms=list(ARMS),new_parameters=998,
        optimizer={'name':'AdamW','lr':.001,'weight_decay':1e-4,'delta_penalty':1e-4,'clip':1.,'batch':512},
        early_stopping={'patience':10,'min_epochs':15,'min_delta':1e-5,'max_epochs':60},
        selection='mean tuning IBS; content must beat original and availability for background numeric-value nomination',
        high_frequency_trajectories_used=False,old_static_support_retained_by_user=True,
        PFT_role='documented background or unknown measurement time; not current discharge state',
        confirmation_test_evaluated=False,goal_achieved=False)
    return data,train,arrays,contract


def augmented(data,indices,seed,arrays):
    frozen,proof=caches(data,indices,seed)
    x={key:torch.cat([getattr(b,key) for b in frozen]) for key in ('hidden','raw','prior','times','events')}
    base=frozen[0].shape_raw;result=[];hashes=hashlib.sha256(indices.tobytes())
    for value in (*x.values(),base):hashes.update(value.cpu().numpy().tobytes())
    for offset in range(0,len(indices),512):
        sl=slice(offset,offset+512);use=indices[sl];owner=np.full(len(arrays['hadm_ids']),-1,np.int64);owner[use]=np.arange(len(use))
        source=owner[arrays['positions']];rows=np.flatnonzero(source>=0)
        f=FrozenBatch(x['hidden'][sl],x['raw'][sl],x['prior'][sl],base,x['times'][sl],x['events'][sl])
        tensor=lambda v:torch.as_tensor(v,device='cuda')
        b=PFTBatch(f,tensor(arrays['values'][rows]),tensor(arrays['concepts'][rows]),tensor(arrays['metadata'][rows]),
            tensor(source[rows]),tensor(arrays['aggregate'][use]),tensor(arrays['observed'][use]))
        for value in (b.values,b.concepts,b.metadata,b.owners,b.aggregate,b.observed):hashes.update(value.cpu().numpy().tobytes())
        result.append(b)
    return result,dict(teacher_cache=proof,augmented_cache_sha256=hashes.hexdigest(),records=len(indices))


def rebuild_data(arrays):
    rebuilt,_,_=derive()
    for key,value in rebuilt.items():np.testing.assert_array_equal(value,arrays[key])


def new_model(arm,seed):
    m=PFTCorrection(arm,seed).cuda();torch.manual_seed(seed);return m


@torch.no_grad()
def prediction(model,cache):
    a,k=[],[];loss,n=0.,0
    for b in cache:
        out=model(b);a.append(out.log_amplitude.cpu().numpy());k.append(out.shape.cpu().numpy())
        loss+=float(competing_nll(out,b.frozen.times,b.frozen.events))*len(b.frozen.times);n+=len(b.frozen.times)
    aa,kk=np.concatenate(a),np.concatenate(k)
    return aa,kk,probability_curve(aa,kk),loss/n


def score(model,cache):
    a,k,curve,nll=prediction(model,cache)
    t=np.concatenate([b.frozen.times.cpu().numpy() for b in cache]);e=np.concatenate([b.frozen.events.cpu().numpy() for b in cache])
    r=metrics(t,e,curve[:,:,1]);death=metrics(t,e,curve[:,:,2],cause=2)
    return dict(nll=nll,ibs=r['integrated_brier_7_365'],auc90=r['points']['90']['auc'],death_ibs=death['integrated_brier_7_365'])


def preflight():
    data,train,arrays,contract=inputs()
    if ROOT.exists():raise FileExistsError('preflight already exists; inspect')
    ROOT.mkdir();write_json(ROOT/'contract.json',contract);rebuild_data(arrays)
    cached,proof=augmented(data,train[:128],42,arrays);b=cached[0]
    prior=None
    for arm in ARMS:
        m=new_model(arm,42);out=m(b)
        if prior is None:prior=out
        else:
            torch.testing.assert_close(out.log_amplitude,prior.log_amplitude,atol=0,rtol=0)
            torch.testing.assert_close(out.shape,prior.shape,atol=0,rtol=0)
        # 用真实训练来源存在者做单步，不能只拿全缺失批次宣称梯度启动。
        del m
    use=train[np.asarray(arrays['observed'][train],bool)][:128]
    observed,_=augmented(data,use,42,arrays);probe=new_model('content',42)
    opt=torch.optim.AdamW(probe.parameters(),lr=.001);before={k:v.detach().clone() for k,v in probe.state_dict().items()}
    out=probe(observed[0]);loss=competing_nll(out,observed[0].frozen.times,observed[0].frozen.events)
    loss.backward();torch.nn.utils.clip_grad_norm_(probe.parameters(),1.);opt.step()
    if torch.equal(before['head.2.weight'],probe.state_dict()['head.2.weight']):raise ValueError('real observed source cannot update')
    write_json(ROOT/'preflight.json',dict(status='passed',common_initial_risk_error=0.,original_cache_error=proof['teacher_cache']['initial_parameter_max_error'],
        source_rebuilt=True,PFT_source_availability_numeric_views=True,real_observed_GPU_update=True,gpu=torch.cuda.get_device_name(0)))
    write_json(ROOT/'state.json',dict(status='preflight_passed',fits=0,goal_achieved=False))
    print(json.dumps(dict(status='preflight_passed',original_risk_error=0.)),flush=True)


def fit(model,train_cache,tune_cache,d,tune,folder,seed,arm):
    folder.mkdir(parents=True);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);early=EarlyStopping();history=[];start=time.perf_counter();reason='max_epochs'
    for epoch in range(CAP+1):
        online=None
        if epoch:
            total,n=0.,0;model.train()
            for j in rng.permutation(len(train_cache)):
                b=train_cache[j];opt.zero_grad(set_to_none=True);out=model(b)
                nll=competing_nll(out,b.frozen.times,b.frozen.events);loss=nll+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite PFT training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('nonfinite PFT gradient')
                opt.step();total+=float(nll.detach())*len(b.frozen.times);n+=len(b.frozen.times)
            online=total/n
        detail=score(model,tune_cache);save_best,stop=early.observe(epoch,detail['ibs'])
        history.append(dict(epoch=epoch,train_nll=online,**{'tuning_'+k:v for k,v in detail.items()},early_stale=early.stale,early_anchor=early.anchor))
        if save_best:torch.save(dict(state_dict=model.state_dict(),epoch=epoch,seed=seed,arm=arm),folder/'best.pt')
        write_json(folder/'history.json',history)
        if epoch%5==0 or stop:print(json.dumps(dict(seed=seed,arm=arm,epoch=epoch,best=early.best_epoch,ibs=detail['ibs'],early_stop=stop)),flush=True)
        if stop:reason='patience_exhausted';break
    model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
    a,k,curve,nll=prediction(model,tune_cache);save(folder/'tuning_predictions.npz',d,tune,a,k,curve)
    info=dict(arm=arm,seed=seed,completed_epochs=history[-1]['epoch'],best_epoch=early.best_epoch,stop_reason=reason,early_state=asdict(early),
        seconds=time.perf_counter()-start,checkpoint_sha256=sha256(folder/'best.pt'),metrics=summary(curve,d,tune,nll))
    write_json(folder/'training_result.json',info);return info


def lock(report):
    scores={arm:float(np.mean([report['fits'][str(s)][arm]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS])) for arm in ARMS}
    scores['original_V6']=float(np.mean([report['references'][str(s)]['readmission']['integrated_brier_7_365'] for s in SEEDS]))
    selected=min(scores,key=scores.get)
    return dict(selected=selected,scores=scores,nominated=['content'] if selected=='content' and scores['content']<min(v for k,v in scores.items() if k!='content') else [],
        selection='predeclared3-seed tuning IBS; availability gain not background numeric content gain',goal_achieved=False)


def run():
    data,train,arrays,contract=inputs();d,_,_,_,_,_,tune,_,_,_=data
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    if json.loads((ROOT/'state.json').read_text())['status']!='preflight_passed':raise ValueError('already started; inspect')
    report=dict(fits={},references={},cache_hashes={},confirmation_test_not_evaluated=True,goal_achieved=False)
    for seed in SEEDS:
        tc,tp=augmented(data,train,seed,arrays);vc,vp=augmented(data,tune,seed,arrays)
        report['cache_hashes'][str(seed)]={'train':tp,'tune':vp}
        ref=new_model('content',seed);a,k,curve,nll=prediction(ref,vc)
        report['references'][str(seed)]=summary(curve,d,tune,nll);del ref
        report['fits'][str(seed)]={}
        for arm in ARMS:
            write_json(ROOT/'state.json',dict(status='training_with_early_stop',seed=seed,arm=arm,goal_achieved=False))
            m=new_model(arm,seed);report['fits'][str(seed)][arm]=fit(m,tc,vc,d,tune,ROOT/f'seed{seed}'/arm,seed,arm);del m
        del tc,vc;torch.cuda.empty_cache()
    write_json(ROOT/'screening.json',report);write_json(ROOT/'nomination_lock.json',lock(report))
    write_json(ROOT/'state.json',dict(status='training_complete_pending_independent_replay',fits=6,goal_achieved=False))
    print(json.dumps(lock(report)),flush=True)


def verify(tag):
    data,train,arrays,contract=inputs();d,_,_,_,_,_,tune,_,_,_=data
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    rebuild_data(arrays);report=json.loads((ROOT/'screening.json').read_text());errors={}
    for seed in SEEDS:
        tc,tp=augmented(data,train,seed,arrays);vc,vp=augmented(data,tune,seed,arrays)
        if report['cache_hashes'][str(seed)]!={'train':tp,'tune':vp}:raise ValueError('teacher/current source cache differs')
        ref=new_model('content',seed);a,k,curve,nll=prediction(ref,vc)
        if summary(curve,d,tune,nll)!=report['references'][str(seed)]:raise ValueError('reference differs')
        del ref,tc
        for arm in ARMS:
            folder=ROOT/f'seed{seed}'/arm;info=json.loads((folder/'training_result.json').read_text())
            if info!=report['fits'][str(seed)][arm] or sha256(folder/'best.pt')!=info['checkpoint_sha256']:raise ValueError('checkpoint differs')
            history=json.loads((folder/'history.json').read_text());early=EarlyStopping();first_stop=None
            for row in history:
                _,stop=early.observe(row['epoch'],row['tuning_ibs'])
                if stop and first_stop is None:first_stop=row['epoch']
            if asdict(early)!=info['early_state'] or early.best_epoch!=info['best_epoch'] or info['completed_epochs']!=len(history)-1:
                raise ValueError('early stop state/selection differs')
            if info['stop_reason']=='patience_exhausted' and first_stop!=info['completed_epochs']:raise ValueError('early stop bypassed')
            m=new_model(arm,seed);stored=torch.load(folder/'best.pt',map_location='cuda',weights_only=True)
            if stored['epoch']!=early.best_epoch:raise ValueError('wrong saved epoch')
            m.load_state_dict(stored['state_dict']);a,k,curve,nll=prediction(m,vc)
            with np.load(folder/'tuning_predictions.npz') as z:
                np.testing.assert_array_equal(z['times'],d.times[tune]);np.testing.assert_array_equal(z['events'],d.events[tune])
                error=max(float(np.abs(a-z['log_amplitude']).max()),float(np.abs(k-z['shape']).max()),float(np.abs(curve-z['curve']).max()))
            if error>3e-7 or summary(curve,d,tune,nll)!=info['metrics']:raise ValueError('prediction/metric differs')
            errors[f'{seed}/{arm}']=error;del m
        del vc
    if lock(report)!=json.loads((ROOT/'nomination_lock.json').read_text()):raise ValueError('selection lock differs')
    proof=dict(status='independently_verified',models=6,maximum_error=max(errors.values()),replay_errors=errors,
        report_sha256=sha256(ROOT/'screening.json'),contract_sha256=sha256(ROOT/'contract.json'),
        source_inputs_and_full_teacher_caches_rebuilt=True,early_stop_replayed=True,confirmation_test_not_evaluated=True)
    write_json(ROOT/f'verification_{tag}.json',proof)
    if all((ROOT/f'verification_{t}.json').exists() for t in ('a','b')):
        for t in ('a','b'):
            if json.loads((ROOT/f'verification_{t}.json').read_text())!=proof:raise ValueError('independent proof differs')
        write_json(ROOT/'state.json',dict(status='screening_complete_independently_verified',fits=6,nominated=lock(report)['nominated'],goal_achieved=False))
    print(json.dumps(proof),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('preflight','run','verify'),required=True);p.add_argument('--tag',choices=('a','b'),default='a');args=p.parse_args()
    try:
        if args.mode=='preflight':preflight()
        elif args.mode=='run':run()
        else:verify(args.tag)
    except Exception as e:
        if ROOT.exists():write_json(ROOT/f'failure_{args.mode}_{args.tag}.json',dict(error=repr(e),goal_achieved=False))
        raise


if __name__=='__main__':main()






````
<!-- END REVIEW_FRAGMENT -->

## training/v6/pft_set_model.py · lines 1–61

<!-- REVIEW_FRAGMENT {"path":"training/v6/pft_set_model.py","part":1,"start_line":1,"end_line":61,"source_sha256":"5ed40b49aad7e9cac33c2fe906687b6c77ad1a75746cccfbadcccf65d5e8f9eb","fragment_sha256":"5ed40b49aad7e9cac33c2fe906687b6c77ad1a75746cccfbadcccf65d5e8f9eb","added_terminal_newline":false} -->
````python
"""背景PFT不同值集合的分层读出；原完整V6风险不变起步。"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.risk import RiskOutput

ARMS=('availability','content')


@dataclass
class PFTBatch:
    frozen:object
    values:torch.Tensor
    concepts:torch.Tensor
    metadata:torch.Tensor
    owners:torch.Tensor
    aggregate:torch.Tensor
    observed:torch.Tensor


def segment_pool(tokens,owners,size,scorer):
    out=tokens.new_zeros(size,tokens.shape[-1])
    if not len(tokens):return out
    scores=scorer(tokens).squeeze(-1)
    maximum=scores.new_full((size,),-torch.inf);maximum.scatter_reduce_(0,owners,scores,reduce='amax',include_self=True)
    weights=torch.exp(scores-maximum[owners]);total=scores.new_zeros(size).scatter_add_(0,owners,weights)
    out.index_add_(0,owners,(weights/total[owners])[:,None]*tokens)
    return out


class PFTCorrection(nn.Module):
    def __init__(self,arm,seed=42):
        super().__init__()
        if arm not in ARMS:raise ValueError('unknown PFT arm')
        self.arm=arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.concept_embedding=nn.Embedding(6,8)
            self.value_projection=nn.Sequential(nn.Linear(1,8),nn.GELU(),nn.LayerNorm(8))
            self.metadata_projection=nn.Linear(3,8)
            self.value_score=nn.Linear(8,1);self.concept_score=nn.Linear(8,1)
            self.head=nn.Sequential(nn.Linear(49,16),nn.GELU(),nn.Linear(16,4))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,b):
        f=b.frozen;n=len(f.hidden);m=len(b.values)
        if b.values.shape!=(m,) or b.concepts.shape!=(m,) or b.metadata.shape!=(m,3) or b.owners.shape!=(m,) or b.aggregate.shape!=(n,9):
            raise ValueError('PFT interface shape mismatch')
        if b.owners.dtype!=torch.long or b.concepts.dtype!=torch.long or (b.owners<0).any() or (b.owners>=n).any() or (b.concepts<0).any() or (b.concepts>=6).any():
            raise ValueError('PFT concept/owner invalid')
        values=torch.zeros_like(b.values) if self.arm=='availability' else b.values
        tokens=self.concept_embedding(b.concepts)+self.value_projection(values[:,None])+self.metadata_projection(b.metadata)
        # 同概念的多个不同值先学习表示读出，再跨概念读出，不让记录多的概念占更多池化槽位。
        group=b.owners*6+b.concepts;within=segment_pool(tokens,group,n*6,self.value_score)
        present=torch.zeros(n*6,dtype=torch.bool,device=b.values.device);present[group]=True
        slots=torch.nonzero(present,as_tuple=False).squeeze(1)
        pooled=segment_pool(within[slots],slots//6,n,self.concept_score)
        raw=self.head(torch.cat((f.hidden,pooled,b.aggregate),1))*b.observed[:,None]
        raw=raw.reshape(n,2,2);delta=torch.stack((raw[:,:,0],.25*torch.tanh(raw[:,:,1])),2);original=f.raw.reshape(n,2,2)
        return RiskOutput((f.prior+original[:,:,0]+delta[:,:,0]).clamp(-10,6),F.softplus(f.shape_raw+original[:,:,1]+delta[:,:,1])+.1,delta)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/pft_set_report.py · lines 1–103

<!-- REVIEW_FRAGMENT {"path":"training/v6/pft_set_report.py","part":1,"start_line":1,"end_line":103,"source_sha256":"fe6b124b2a052d81cd2a2c1a7bf4d2ca60baa8031dc4e12eb55849151091cfcc","fragment_sha256":"fe6b124b2a052d81cd2a2c1a7bf4d2ca60baa8031dc4e12eb55849151091cfcc","added_terminal_newline":false} -->
````python
"""背景PFT集合数值/来源配对读出；不改选点/默认。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patient_data.export_mimiciv import sha256
from training.v6.pft_set_experiment import ROOT,SEEDS,ARMS,inputs,augmented,new_model,prediction
from training.v6.evaluation import paired_interval,metrics,public_metrics,calibration_bins
from training.v6.train import write_json


def build_report():
    saved=json.loads((ROOT/'screening.json').read_text())
    for tag in ('a','b'):
        proof=json.loads((ROOT/f'verification_{tag}.json').read_text())
        if proof['report_sha256']!=sha256(ROOT/'screening.json') or proof['maximum_error']!=0:raise ValueError('not independently verified')
    data,_,arrays,_=inputs();d,_,_,_,_,_,tune,_,_,_=data
    observed=arrays['observed'][tune];counts=np.rint(np.expm1(d.history_raw[tune,2])).astype(int)
    groups={'source_observed':observed,'source_missing':~observed,'first':counts==0,'one_two':(counts>=1)&(counts<=2),'three_plus':counts>=3}
    report=dict(pairs={},layers={},means={},calibration={},actual_epochs={},unobserved_max_error={},no_confirmation_test=True,
        screening_sha256=sha256(ROOT/'screening.json'),code_sha256=sha256(Path(__file__)))
    fig,axs=plt.subplots(1,3,figsize=(15,4.5));colors={'availability':'#1f77b4','content':'#2ca02c'}
    differences=[]
    calfig,calaxs=plt.subplots(2,2,figsize=(9,8))
    for j,seed in enumerate(SEEDS):
        cache,_=augmented(data,tune,seed,arrays);original=new_model('content',seed);_,_,ref,_=prediction(original,cache);del original,cache
        curves={'original_V6':ref};report['actual_epochs'][str(seed)]={}
        for arm in ARMS:
            history=json.loads((ROOT/f'seed{seed}'/arm/'history.json').read_text());info=saved['fits'][str(seed)][arm]
            report['actual_epochs'][str(seed)][arm]={'actual':info['completed_epochs'],'best':info['best_epoch']}
            axs[j].plot([r['epoch'] for r in history],[r['tuning_ibs'] for r in history],label=arm,color=colors[arm])
            best=info['best_epoch'];axs[j].scatter(best,history[best]['tuning_ibs'],color=colors[arm],marker='D',s=30)
            with np.load(ROOT/f'seed{seed}'/arm/'tuning_predictions.npz') as z:curves[arm]=z['curve']
        axs[j].set(title=f'Seed {seed}: actual early-stopped epochs',xlabel='Epoch',ylabel='Tuning IBS7-365 (lower better)');axs[j].grid(alpha=.2);axs[j].legend(fontsize=8)
        report['calibration'][str(seed)]={}
        for name,curve in curves.items():
            report['calibration'][str(seed)][name]={}
            for cause in (1,2):
                for day in (30,90,180,365):
                    bins=calibration_bins(d.times[tune],d.events[tune],curve[:,:,cause],day=day,cause=cause)
                    report['calibration'][str(seed)][name][f'{cause}/{day}']=bins
                    if j==0 and day in (90,365):
                        ax=calaxs[cause-1,0 if day==90 else 1]
                        ax.plot([r['predicted'] for r in bins],[r['observed'] for r in bins],marker='o',markersize=3,
                            color='#777777' if name=='original_V6' else colors[name],label=name)
        for name in ('original_V6','availability'):
            report['pairs'][f'{seed}/content minus {name}']=paired_interval(d.arrays['subject_ids'][tune],d.times[tune],d.events[tune],
                curves['content'][:,:,1],curves[name][:,:,1],replicates=200,seed=seed)
        report['unobserved_max_error'][str(seed)]=float(np.abs(curves['content'][~observed]-ref[~observed]).max())
        if report['unobserved_max_error'][str(seed)]!=0:raise ValueError('unobserved source changed original risks')
        report['layers'][str(seed)]={}
        for group,mask in groups.items():
            ix=tune[mask];values={}
            for name in ('original_V6','availability','content'):
                values[name]={'readmission':public_metrics(metrics(d.times[ix],d.events[ix],curves[name][mask,:,1])),
                    'death':public_metrics(metrics(d.times[ix],d.events[ix],curves[name][mask,:,2],cause=2))}
            report['layers'][str(seed)][group]=dict(admissions=int(mask.sum()),patients=len(np.unique(d.arrays['subject_ids'][ix])),models=values)
        p=report['pairs'][f'{seed}/content minus original_V6'];differences.append((p['integrated_brier_difference'],p['auc90_difference']))
    for arm in (*ARMS,'original_V6'):
        ms=[saved['references'][str(s)] if arm=='original_V6' else saved['fits'][str(s)][arm]['metrics'] for s in SEEDS]
        report['means'][arm]=dict(ibs=float(np.mean([m['readmission']['integrated_brier_7_365'] for m in ms])),
            auc90=float(np.mean([m['readmission']['points']['90']['auc'] for m in ms])),
            death_ibs=float(np.mean([m['death']['integrated_brier_7_365'] for m in ms])),
            timepoints={str(t):dict(auc=float(np.mean([m['readmission']['points'][str(t)]['auc'] for m in ms])),
                brier=float(np.mean([m['readmission']['points'][str(t)]['brier'] for m in ms]))) for t in (7,14,30,60,90,180,365)})
    fig.suptitle('Frozen original V6; PFT availability / numeric content; early stopping',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(ROOT/'early_stopping.png',dpi=150);plt.close(fig)
    for i in range(2):
        for k,day in enumerate((90,365)):
            ax=calaxs[i,k];ax.plot([0,1],[0,1],'k--',linewidth=1)
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted cumulative probability',ylabel='Observed World-A fraction',
                   title=f'{"Readmission" if i==0 else "Death"}: day {day}');ax.grid(alpha=.2);ax.legend(fontsize=8)
    calfig.suptitle('Specified tuning set; predeclared seed 42; risk-bin descriptive calibration',fontsize=11)
    calfig.tight_layout(rect=(0,0,1,.95));calfig.savefig(ROOT/'calibration.png',dpi=150);plt.close(calfig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8));diff=np.asarray(differences)
    for k,ax in enumerate(axs):
        ax.axvline(0,color='black',linewidth=1);ax.scatter(diff[:,k],[-.12,0,.12],color=colors['content'],alpha=.6)
        ax.scatter(diff[:,k].mean(),0,marker='D',s=75,color=colors['content']);ax.set_yticks([0],['background PFT numeric content'])
        ax.set_xlabel('Paired difference across three seeds');ax.grid(alpha=.2)
    axs[0].set_title('IBS: content minus original\n(lower better)');axs[1].set_title('AUC90: content minus original\n(higher better)')
    axs[0].set_xlim(-.002,.002);axs[1].set_xlim(-.02,.02)
    axs[0].plot([],[],marker='o',linestyle='',color=colors['content'],alpha=.6,label='Clinical seed')
    axs[0].plot([],[],marker='D',linestyle='',color=colors['content'],label='Mean across three seeds')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8,loc='lower center',ncol=2)
    fig.suptitle('Specified tuning set; fixed teacher; three clinical seeds / two teacher sources',fontsize=11)
    fig.tight_layout(rect=(0,.12,1,.9));fig.savefig(ROOT/'paired_differences.png',dpi=150);plt.close(fig)
    report['limits']=['tuning set exposed; selected best checkpoints','fixed-model bootstrap not inner refitting or independent cohort',
        'three clinical seeds reuse two teacher sources','fixed PFT concepts; unknown measurement dates; not medical gold',
        'World-A observed fractions not full out-of-system ascertainment; descriptive bins not prospective calibration proof']
    write_json(ROOT/'paired_readout.json',report)
    print(json.dumps(dict(means=report['means'],unobserved_max_error=report['unobserved_max_error'])),flush=True)


if __name__=='__main__':build_report()






````
<!-- END REVIEW_FRAGMENT -->

## training/v6/plot.py · lines 1–141

<!-- REVIEW_FRAGMENT {"path":"training/v6/plot.py","part":1,"start_line":1,"end_line":141,"source_sha256":"e64296c8db08b384315019653f4de1dcf6c06ac92216db2b20200d3dc67bbd3c","fragment_sha256":"e64296c8db08b384315019653f4de1dcf6c06ac92216db2b20200d3dc67bbd3c","added_terminal_newline":false} -->
````python
"""四臂实测图；英文标签，不将探索区间当显著性结论。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from training.v6.evaluation import metrics, calibration_bins, GRID

LABELS = {'history': 'History baseline', 'constant': 'History + constant calibration',
          'clinical_no_text': 'History + clinical (no text)', 'clinical_bert': 'History + clinical + BERT'}
COLORS = dict(zip(LABELS, ('#777777', '#9c6a36', '#2474b7', '#d95f02')))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def roc(scores, y):
    order = np.argsort(-scores, kind='stable')
    s, y = scores[order], y[order]
    endings = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    positives, negatives = y.sum(), (1-y).sum()
    if positives == 0 or negatives == 0:
        return np.array([0., 1.]), np.array([np.nan, np.nan])
    tpr = np.r_[0., np.cumsum(y)[endings] / positives]
    fpr = np.r_[0., np.cumsum(1-y)[endings] / negatives]
    return fpr, tpr


def render(root: Path):
    report = json.loads((root / 'report.json').read_text(encoding='utf-8'))
    figures = root / 'figures'; figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for arm, label in LABELS.items():
        history = json.loads((root / arm / 'history.json').read_text())
        epoch = [r['epoch'] for r in history]
        axes[0].plot(epoch, [r['validation_nll'] for r in history], color=COLORS[arm], label=label)
        axes[1].plot(epoch, [r['validation_integrated_brier'] for r in history], color=COLORS[arm])
        axes[2].plot(epoch, [r['validation_mean_auc'] for r in history], color=COLORS[arm])
        best = report['models'][arm]['best_epoch']
        for ax in axes:
            ax.axvline(best, color=COLORS[arm], alpha=.2, linestyle=':')
    for ax, title in zip(axes, ('Validation NLL', 'Validation integrated Brier (lower better)', 'Validation mean AUC(t)')):
        ax.set_title(title); ax.set_xlabel('Epoch (0 = initial)'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig, figures / 'training_validation.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for arm, label in LABELS.items():
        h = json.loads((root / arm / 'history.json').read_text())
        rows = [r for r in h if r['train_nll'] is not None]
        axes[0].plot([r['epoch'] for r in rows], [r['train_nll'] for r in rows], label=label, color=COLORS[arm])
        axes[1].plot([r['epoch'] for r in rows], [r['validation_nll'] for r in rows], color=COLORS[arm])
    axes[0].set_title('Training NLL'); axes[1].set_title('Validation NLL'); axes[0].legend(fontsize=8)
    for ax in axes: ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    save(fig, figures / 'training_losses.png')
    for fold in ('validation', 'test'):
        predictions = {}
        for arm in LABELS:
            with np.load(root / arm / (fold + '_predictions.npz')) as archive:
                predictions[arm] = {k: archive[k] for k in archive.files}
        reference = predictions['history']
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for cause in (1, 2):
            for arm, label in LABELS.items():
                p = predictions[arm]
                m = metrics(p['times'], p['events'], p['curve'][:, :, cause], cause=cause)
                axes[cause-1, 0].plot(GRID[7:], m['auc_curve'][7:], color=COLORS[arm], label=label)
                axes[cause-1, 1].plot(GRID[7:], m['brier_curve'][7:], color=COLORS[arm])
            axes[cause-1, 0].set_title(('Strict readmission' if cause == 1 else 'Death before readmission') + ': AUC(t)')
            axes[cause-1, 1].set_title('Brier(t) (lower better)')
        axes[0, 0].legend(fontsize=8)
        for ax in axes.flat: ax.set_xlabel('Days after discharge'); ax.grid(alpha=.2)
        fig.suptitle(f'{fold}: record-observed competing endpoints (World A)')
        save(fig, figures / (fold + '_auc_brier.png'))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, day in zip(axes, (30, 90, 365)):
            for arm, label in LABELS.items():
                p = predictions[arm]
                y = ((p['events'] == 1) & (p['times'] <= day)).astype(int)
                x, yy = roc(p['curve'][:, day, 1], y)
                auc = report['models'][arm]['folds'][fold]['readmission']['points'][str(day)]['auc']
                ax.plot(x, yy, label=f'{label} ({auc:.3f})' if auc is not None else label, color=COLORS[arm])
            ax.plot([0, 1], [0, 1], '--', color='gray'); ax.set_title(f'Time-dependent ROC: {day} days')
            ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); ax.legend(fontsize=7)
        save(fig, figures / (fold + '_roc.png'))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, day in zip(axes, (30, 90, 365)):
            for arm, label in LABELS.items():
                p = predictions[arm]
                bins = calibration_bins(p['times'], p['events'], p['curve'][:, :, 1], day=day)
                ax.plot([b['predicted'] for b in bins], [b['observed'] for b in bins], 'o-', label=label, color=COLORS[arm], ms=4)
            ax.plot([0, 1], [0, 1], '--', color='gray'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f'Calibration: {day} days'); ax.set_xlabel('Predicted risk'); ax.set_ylabel('Observed fraction (World A)')
        axes[0].legend(fontsize=7)
        save(fig, figures / (fold + '_calibration.png'))
        pairs = report['comparisons'][fold]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for i, (label, r) in enumerate(pairs.items()):
            for ax, point, bounds in zip(axes, ('integrated_brier_difference', 'auc90_difference'), ('integrated_brier_95ci', 'auc90_95ci')):
                if None not in r[bounds]:
                    ax.plot(r[bounds], [i, i], color='#2474b7', lw=2)
                ax.scatter(r[point], i, marker='D', color='#d95f02', zorder=3)
        for ax, title in zip(axes, ('Integrated Brier difference (lower better)', 'AUC(90) difference (higher better)')):
            ax.axvline(0, color='gray'); ax.set_title(title); ax.set_yticks(range(len(pairs)), [k.replace('_minus_', ' - ').replace('_', ' ') for k in pairs], fontsize=8)
            ax.set_xlabel('Candidate - comparator'); ax.grid(axis='x', alpha=.2)
        fig.suptitle(f'{fold}: paired patient bootstrap 95% intervals; one seed; no multiplicity adjustment')
        save(fig, figures / (fold + '_paired_differences.png'))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for cause, ax in zip((1, 2), axes):
            observed = np.mean((reference['events'][:, None] == cause) & (reference['times'][:, None] <= GRID), axis=0)
            ax.plot(GRID, observed, '--', color='black', label='Observed record fraction')
            for arm, label in LABELS.items():
                ax.plot(GRID, predictions[arm]['curve'][:, :, cause].mean(0), label=label, color=COLORS[arm])
            ax.set_title('Readmission' if cause == 1 else 'Death before readmission'); ax.set_xlabel('Days after discharge'); ax.set_ylabel('Mean cumulative incidence'); ax.legend(fontsize=7)
        save(fig, figures / (fold + '_cumulative_incidence.png'))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, day in zip(axes, (30, 90)):
            y = ((reference['events'] == 1) & (reference['times'] <= day)).astype(bool)
            thresholds = np.linspace(.02, .5, 80)
            for arm, label in LABELS.items():
                score = predictions[arm]['curve'][:, day, 1]
                benefit = [np.mean((score >= pt) & y) - np.mean((score >= pt) & ~y)*pt/(1-pt) for pt in thresholds]
                ax.plot(thresholds, benefit, color=COLORS[arm], label=label)
            ax.plot(thresholds, y.mean() - (1-y.mean())*thresholds/(1-thresholds), '--', color='gray', label='Follow all')
            ax.axhline(0, color='black', linestyle=':', label='Follow none'); ax.set_title(f'Exploratory follow-up DCA: {day} days')
            ax.set_xlabel('Risk threshold (assumed harm / benefit trade-off)'); ax.set_ylabel('Net benefit'); ax.legend(fontsize=7); ax.set_ylim(-.03, max(.03, y.mean()))
        fig.suptitle('Not evidence of intervention efficacy; no selected clinical threshold')
        save(fig, figures / (fold + '_exploratory_dca.png'))
        arms = list(LABELS)
        values = np.array([[report['models'][a]['folds'][fold]['history_strata'][s]['mean_auc_7_365'] or np.nan for s in ('prior_0', 'prior_1_2', 'prior_3_plus')] for a in arms])
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(values, vmin=.5, vmax=.85, cmap='YlGnBu')
        ax.set_xticks(range(3), ['Prior 0', 'Prior 1-2', 'Prior >=3']); ax.set_yticks(range(4), [LABELS[a] for a in arms]); ax.set_title(f'{fold}: mean AUC(t) within history strata')
        for i in range(4):
            for j in range(3): ax.text(j, i, f'{values[i,j]:.3f}', ha='center', va='center')
        fig.colorbar(im, ax=ax)
        save(fig, figures / (fold + '_history_strata.png'))
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/predict_research.py · lines 1–134

<!-- REVIEW_FRAGMENT {"path":"training/v6/predict_research.py","part":1,"start_line":1,"end_line":134,"source_sha256":"fb424ea6662a97c3b740aa23d3caeda2dce31211983dabd08cfd495fb3bb2837","fragment_sha256":"fb424ea6662a97c3b740aa23d3caeda2dce31211983dabd08cfd495fb3bb2837","added_terminal_newline":false} -->
````python
"""已验收V6研究模型按住院重载预测；不训练，不使用未来标签计算预测。"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.compact_ablation import inputs
from training.v6.architecture_experiment import batches
from training.v6.text_encoder_comparison import new_model
from training.v6.horizon_objective import restore
from training.v6.risk import predict_cif
from training.v6.train import write_json

DAYS=np.arange(366,dtype=float)


@torch.no_grad()
def scores(model,prepared):
    """标签只随数据批存在，故意不读取t/e，不调用带标签的predict90。"""
    model.eval(); a,k=[],[]
    for _,history,batch,_,_ in prepared:
        out=model(history,batch) if hasattr(model,'encoder') else model(history)
        a.append(out.log_amplitude.cpu().numpy()); k.append(out.shape.cpu().numpy())
    return np.concatenate(a),np.concatenate(k)


def predict(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required; no CPU embedding')
    torch.set_num_threads(2)
    data,compact,hist,_=inputs(args.reference,args.compact_dir)
    parent=json.loads((args.parent/'report.json').read_text())
    report=json.loads((args.text_parent/'report.json').read_text())
    for root,r in ((args.parent,parent),(args.text_parent,report)):
        proof=json.loads((root/'independent_verification.json').read_text())
        if proof['report_sha256']!=sha256(root/'report.json'): raise ValueError('unverified model source')
        for p,digest in r['contract']['code_sha256'].items():
            if sha256(Path(p))!=digest: raise ValueError('model source code changed')
    if data.provenance!=report['contract']['data_provenance'] or compact.provenance!=report['contract']['compact_provenance']:
        raise ValueError('model input source changed')
    if args.hadm_ids:
        lookup={int(h):i for i,h in enumerate(data.arrays['hadm_ids'])}
        if len(set(args.hadm_ids))!=len(args.hadm_ids) or any(h not in lookup for h in args.hadm_ids):
            raise ValueError('requested admission absent or duplicated')
        ix=np.array([lookup[h] for h in args.hadm_ids],dtype=np.int64)
    else:
        ix=np.flatnonzero(data.arrays['patient_splits']=='test')
        ix=ix[np.argsort(data.arrays['hadm_ids'][ix])][:args.first_test]
    prepared=batches(data,compact,ix,512,expanded=True,text='bert')
    # 反事实修改标签作工程核验；特征、标识、时间来源全部不动。
    altered=replace(data,times=np.zeros_like(data.times),events=np.zeros_like(data.events))
    altered_batches=batches(altered,replace(compact,base=altered),ix,512,expanded=True,text='bert')
    curves,base_curves,proofs=[],[],{}
    for seed in (42,7):
        path=args.text_parent/f'seed{seed}/bert/best.pt'
        if sha256(path)!=report['models'][str(seed)]['bert']['checkpoint_sha256']: raise ValueError('checkpoint changed')
        m=new_model(args,seed,'bert',hist,.1)
        m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
        a,k=scores(m,prepared); altered_a,altered_k=scores(m,altered_batches)
        if not np.array_equal(a,altered_a) or not np.array_equal(k,altered_k): raise ValueError('future label influences prediction')
        curves.append(predict_cif(a,k,DAYS))
        fine=predict_cif(a,k,DAYS,step_days=.125)
        if np.abs(fine-curves[-1]).max()>2e-4: raise ValueError('numerical integration not converged')
        base=restore(args.parent,seed,'survival_base',hist,.1)
        ba,bk=scores(base,prepared); base_curves.append(predict_cif(ba,bk,DAYS))
        proofs[str(seed)]=dict(checkpoint_sha256=sha256(path),label_invariance_exact=True,
            fine_step_max_error=float(np.abs(fine-curves[-1]).max()))
        del m,base
    stacked=np.stack(curves); mean=stacked.mean(0); reference=np.stack(base_curves).mean(0)
    if not np.allclose(mean.sum(2),1,atol=1e-9) or (np.diff(mean[:,:,1:],axis=1)<-1e-10).any():
        raise ValueError('ensemble probability contract failed')
    args.output.mkdir(parents=True)
    records=[]
    for j,p in enumerate(ix):
        field_count={f:sum(x[0]==i for x in data.text_refs[p]) for i,f in enumerate(('HPI','PMH','Hospital course','Discharge condition'))}
        records.append(dict(hadm_id=int(data.arrays['hadm_ids'][p]),subject_id=int(data.arrays['subject_ids'][p]),
            patient_split=str(data.arrays['patient_splits'][p]),
            recorded_prior_copd_stays=int(np.rint(np.expm1(data.history_raw[p,2]))),
            observed_lab_axes=int((data.arrays['lab_axis_inputs'][p,:,6]>0).sum()),text_chunks=field_count,
            predicted_cif={str(t):dict(readmission=float(mean[j,t,1]),death_before_readmission=float(mean[j,t,2]),
                neither=float(mean[j,t,0]),baseline_readmission=float(reference[j,t,1])) for t in (7,14,30,60,90,180,365)}))
    result=dict(status='verified_research_inference_only',selection='requested admissions or smallest test hadm_ids; no outcome selection',
        source_reports_sha256={str(args.parent):sha256(args.parent/'report.json'),str(args.text_parent):sha256(args.text_parent/'report.json')},
        input_provenance=data.provenance,proofs=proofs,records=records,
        limits=['Research only; no deployment/clinical action validation or complete outside follow-up.',
            'Equal-weight two-seed mixture; not a calibrated confidence interval or new performance claim.',
            'Current final discharge text allowed even if stored after discharge.',
            'Predicts observed strict readmission cumulative probability, not an exact return date.',
            'Not a private-data adapter; raw concept mapping and normalization still need validation there.'])
    write_json(args.output/'predictions.json',result)
    np.savez_compressed(args.output/'risk_curves.npz',hadm_ids=data.arrays['hadm_ids'][ix],days=DAYS,
        full_seed_curves=stacked,full_mean=mean,baseline_mean=reference)
    plot(args.output,records,mean,reference,stacked)
    write_json(args.output/'verification.json',dict(status='source_hashes_label_invariance_mass_monotonicity_and_integration_checked',
        predictions_sha256=sha256(args.output/'predictions.json'),curve_sha256=sha256(args.output/'risk_curves.npz'),
        code_sha256=sha256(Path(__file__)),max_probability_sum_error=float(np.abs(mean.sum(2)-1).max())))
    print(json.dumps(dict(status=result['status'],admissions=len(records),output=str(args.output))),flush=True)


def plot(root,records,mean,baseline,seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(records),1,figsize=(11,3.5*len(records)),squeeze=False)
    for i,record in enumerate(records):
        ax=axes[i,0]
        ax.plot(DAYS,baseline[i,:,1],color='#999999',ls='--',label='Frozen history + diagnosis + disposition')
        ax.plot(DAYS,mean[i,:,1],color='#2674ae',label='Plus clinical state + BERT: strict readmission')
        ax.fill_between(DAYS,seeds[:,i,:,1].min(0),seeds[:,i,:,1].max(0),color='#2674ae',alpha=.15,label='Two-seed range, NOT confidence interval')
        ax.plot(DAYS,mean[i,:,2],color='#cf7722',label='Death before target readmission')
        ax.scatter([30,90,180,365],mean[i,[30,90,180,365],1],s=15,color='#2674ae')
        ax.set(xlabel='Days after this discharge',ylabel='Predicted cumulative probability',ylim=(0,1),
            title=f'Admission{record["hadm_id"]}; recorded prior COPD stays={record["recorded_prior_copd_stays"]}; readmission risk90={mean[i,90,1]:.1%}')
        ax.grid(alpha=.2); ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('Reloaded V6 research prediction; selected by ID, not outcome; no clinical deployment claim')
    fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(root/'patient_risk_curves.png',dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--compact-dir',type=Path,default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--parent',type=Path,default=Path('training_runs/v6_90day_objective_2026-10-11'))
    p.add_argument('--text-parent',type=Path,default=Path('training_runs/v6_bert_qwen_same_fields_2026-10-11'))
    p.add_argument('--hadm-ids',type=int,nargs='+'); p.add_argument('--first-test',type=int,default=3)
    p.add_argument('--output',type=Path,default=Path('training_runs/v6_research_prediction_demo_2026-10-11'))
    args=p.parse_args()
    if not 1<=args.first_test<=12: p.error('demo size1..12; use explicit hadm-ids for more')
    predict(args)


if __name__=='__main__': main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/public_tabular_corrected.py · lines 1–122

<!-- REVIEW_FRAGMENT {"path":"training/v6/public_tabular_corrected.py","part":1,"start_line":1,"end_line":122,"source_sha256":"df9ea4593a5716a11168979c572a883ef66d2dfc00e39068ee63cb27d1c7eb79","fragment_sha256":"df9ea4593a5716a11168979c572a883ef66d2dfc00e39068ee63cb27d1c7eb79","added_terminal_newline":false} -->
````python
"""完整官方读出层补正；旧骨干/旧记录不改，复用确实未受影响的训练。"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch import nn
from patient_data.export_mimiciv import sha256
from training.v6.public_tabular_model import PublicModel,ARMS,initial_bias,ensemble_curve
from training.v6.public_tabular_data import read
from training.v6.public_tabular_experiment import ROOT as FIRST,SEEDS,LRS,setup,tensors,fit,parameters,render
from training.v6.audited_experiment import OUTPUT as AUDITED,save_curve,summary
from training.v6.evaluation import paired_interval
from training.v6.train import write_json

ROOT=FIRST/'official_readout'


class CorrectedModel(PublicModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.readout=(nn.Sequential(nn.BatchNorm1d(256),nn.ReLU()) if self.arm.startswith('resnet') else
            nn.Sequential(nn.LayerNorm(96),nn.ReLU()) if self.arm=='ft_transformer' else nn.Identity())

    def forward(self,x_num,x_binary):
        if self.arm in ('ft_transformer','tabm_ple'):
            h=self.backbone(x_num,x_binary.long() if x_binary.shape[1] else None)
        else:
            x=self.embedding(x_num).flatten(1) if self.embedding is not None else x_num
            h=self.backbone(torch.cat((x,x_binary),1))
        with torch.autocast(h.device.type,enabled=False):
            raw=self.head(self.readout(h.float()))
        if raw.ndim==2:raw=raw[:,None]
        return raw[...,:2].clamp(-10.,6.),torch.nn.functional.softplus(raw[...,2:])+.1


def model_for(arm,seed,data,prep):
    torch.manual_seed(seed)
    model=CorrectedModel(arm,data['x_num'].shape[1],data['x_binary'].shape[1],
        [torch.as_tensor(b) for b in prep['bins']],initial_bias(data['events'][data['patient_splits']=='train'])).cuda()
    torch.manual_seed(seed);return model


def run():
    setup();start=time.perf_counter()
    if ROOT.exists():raise FileExistsError(ROOT)
    original=json.loads((FIRST/'report.json').read_text())
    contract=original['contract']
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('first-pass source changed')
    ROOT.mkdir()
    for file in ('matrix.npz','preprocessing.joblib','data_contract.json'):
        shutil.copy2(FIRST/file,ROOT/file)
    d,prep=read(ROOT);_,train=tensors(d,'train');vix,val=tensors(d,'validation')
    contract=dict(contract,phase='complete_official_readout',new_trainings=12,reused_trainings=8,
        first_report_sha256=sha256(FIRST/'report.json'),
        correction_protocol_sha256=sha256(Path('docs/V6_PUBLIC_TABULAR_READOUT_CORRECTION_2026-10-12.md')),
        source_sha256={**contract['source_sha256'],str(Path(__file__)):sha256(Path(__file__))})
    write_json(ROOT/'contract.json',contract);fits={}
    for arm in ARMS:
        fits[arm]={}
        for lr in LRS:
            fits[arm][str(lr)]={}
            for seed in SEEDS:
                path=ROOT/arm/f'lr{lr}'/f'seed{seed}'
                if arm in ('mlp','tabm_ple'):
                    oldpath=FIRST/arm/f'lr{lr}'/f'seed{seed}'
                    result=json.loads((oldpath/'training_result.json').read_text())
                    if result['completed_epochs']!=100 or sha256(oldpath/'best.pt')!=result['checkpoint_sha256']:
                        raise ValueError('unaffected checkpoint mismatch')
                    path.mkdir(parents=True)
                    for file in ('best.pt','history.json','training_result.json'):shutil.copy2(oldpath/file,path/file)
                    result=dict(result,reused_from=str(oldpath),new_training=False)
                    print(json.dumps(dict(reused=arm,seed=seed,lr=lr)),flush=True)
                else:
                    model=model_for(arm,seed,d,prep);result=fit(model,train,val,d,vix,path,seed,lr,100)
                    result['new_training']=True;del model;torch.cuda.empty_cache()
                fits[arm][str(lr)][str(seed)]=result
    choices={arm:min(LRS,key=lambda lr:np.mean([fits[arm][str(lr)][str(s)]['validation_ibs'] for s in SEEDS])) for arm in ARMS}
    write_json(ROOT/'selected_before_test.json',dict(fits=fits,learning_rates=choices,all_training_complete=True))
    del train;torch.cuda.empty_cache();tix,test=tensors(d,'test');models={};comparisons={}
    from types import SimpleNamespace
    holder=SimpleNamespace(times=d['times'],events=d['events'],arrays=d)
    for arm in ARMS:
        models[arm]={};comparisons[arm]={}
        for seed in SEEDS:
            lr=choices[arm];path=ROOT/arm/f'lr{lr}'/f'seed{seed}';model=model_for(arm,seed,d,prep)
            model.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            models[arm][str(seed)]=dict(**fits[arm][str(lr)][str(seed)],folds={});comparisons[arm][str(seed)]={}
            for fold,ix,batch in (('validation',vix,val),('test',tix,test)):
                a,k,nll=parameters(model,batch);curve=ensemble_curve(a,k)
                save_curve(path/(fold+'_predictions.npz'),curve,holder,ix,log_amplitude_members=a,shape_members=k)
                models[arm][str(seed)]['folds'][fold]=dict(nll=nll,**summary(curve,holder,ix))
                with np.load(AUDITED/'simple_controls'/f'xgb{seed}_{fold}_predictions.npz') as z:
                    if not np.array_equal(z['hadm_ids'],d['hadm_ids'][ix]):raise ValueError('XGB member mismatch')
                    comparator=z['curve']
                comparisons[arm][str(seed)][fold]=paired_interval(d['subject_ids'][ix],d['times'][ix],d['events'][ix],curve[:,:,1],comparator[:,:,1],seed=seed)
            del model
    r=dict(status='completed_training_and_prediction_pending_independent_verification',models=models,fits=fits,
        learning_rates=choices,comparisons_to_same_input_xgb=comparisons,elapsed_seconds=time.perf_counter()-start,contract=contract)
    write_json(ROOT/'report.json',r);render(ROOT,r);print(json.dumps(dict(status=r['status'],seconds=r['elapsed_seconds'])),flush=True)


def verify():
    # 公共验收逻辑不变，显式替换为补正工厂，源文件没有被修改。
    import training.v6.public_tabular_verify as verifier
    contract=json.loads((ROOT/'contract.json').read_text())
    if sha256(Path('docs/V6_PUBLIC_TABULAR_READOUT_CORRECTION_2026-10-12.md'))!=contract['correction_protocol_sha256']:
        raise ValueError('correction protocol changed')
    verifier.model_for=model_for
    verifier.verify(ROOT)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('train','verify','finish'),required=True);args=p.parse_args()
    if args.mode=='train':run()
    elif args.mode=='verify':verify()
    else:
        from training.v6.public_tabular_report import finish
        finish(ROOT)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/public_tabular_data.py · lines 1–123

<!-- REVIEW_FRAGMENT {"path":"training/v6/public_tabular_data.py","part":1,"start_line":1,"end_line":123,"source_sha256":"2e9dadc643b222485a8b293e5e49c8e32f20e91375b292738d7acebd34487562","fragment_sha256":"2e9dadc643b222485a8b293e5e49c8e32f20e91375b292738d7acebd34487562","added_terminal_newline":false} -->
````python
"""同XGBoost信息来源的公共矩阵；只在训练折拟合额外预处理。"""
from pathlib import Path
import joblib
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.audited_experiment import load, prepare, restrict, INPUT, OUTPUT
from training.v6.audited_model import flat_features
from training.v6.train import write_json


def numeric_valid(a, h, text_present):
    """给连续列拟合尺度/分位点的有效性；原二元mask列依然入网。"""
    n = len(h)
    masks = np.ones((n, 34), bool)
    masks[:, 4] = h[:, 5] == 0  # 无既往住院时不存在历史间隔。
    labs = np.broadcast_to(a['lab_present'][:, :, None], a['labs'].shape).copy()
    for j, name in enumerate(a['lab_names']):
        x = a['labs'][:, j]
        if name in ('eosinophils_absolute', 'eosinophils_percent'):
            labs[:, j, :6] = x[:, 6:12] > 0
        elif name in ('neutrophils_percent', 'lymphocytes_percent', 'rdw', 'anion_gap', 'chloride',
                      'neutrophils_absolute', 'lymphocytes_absolute', 'monocytes_percent', 'hematocrit',
                      'mcv', 'calcium_total', 'phosphate', 'protein_total', 'fibrinogen', 'troponin_t'):
            labs[:, j, :5] = x[:, 5:10] > 0
        else:
            for offset in (0, 9) if name in ('bicarbonate', 'wbc', 'sodium', 'potassium', 'glucose') else (0,):
                labs[:, j, offset:offset+6] = (x[:, offset+6] > 0)[:, None]
                labs[:, j, offset+3] &= x[:, offset+7] > 0
                labs[:, j, offset+5] &= x[:, offset+7] > 0
    drug = np.broadcast_to(a['med_present'][:, :, None], a['medications'].shape).copy()
    drug[:, :, :2] = a['medications'][:, :, 2:4] > 0
    tail = np.ones((n, 20+8+6), bool)
    # PCA输入已经包含四字段可用标记；“全字段空”的固定PCA表示仍有定义。
    # 不能再用是否有文本把这16坐标置零，否则与XGBoost的共同输入不一致。
    text = np.ones((n, 16), bool)
    return np.concatenate((masks, labs.reshape(n, -1), np.ones_like(a['lab_present']),
        drug.reshape(n, -1), np.ones_like(a['med_present']), tail, text), 1)


def fit_preprocessing(x, valid, train):
    if not np.isfinite(x).all() or x.shape != valid.shape:
        raise ValueError('invalid feature matrix')
    keep = np.ptp(x[train], axis=0) > 1e-7
    # 恰为0/1的列用类别嵌入，其余连续；不对稀有二元阳性除以小标准差。
    binary = np.all((x[train] == 0) | (x[train] == 1), axis=0) & keep
    num = np.flatnonzero(keep & ~binary)
    cat = np.flatnonzero(binary)
    center = np.zeros(len(num), np.float32)
    scale = np.ones(len(num), np.float32)
    bins = []
    for i, column in enumerate(num):
        values = x[train & valid[:, column], column]
        if not len(values):
            raise ValueError('nonconstant numeric column has no valid training values')
        center[i] = np.median(values)
        q25, q75 = np.quantile(values, [.25, .75])
        s = q75-q25
        if s <= 1e-7:
            s = np.std(values)
        scale[i] = s if s > 1e-7 else 1.
        scaled = np.clip((values-center[i])/scale[i], -8, 8)
        edges = np.unique(np.quantile(scaled, np.linspace(0, 1, 17))).astype(np.float32)
        # 有效观测全同但“有测/缺测”使原列变化：不从缺失零拟合区间。
        if len(edges) == 1:
            edges = np.array([edges[0]-1., edges[0]+1.], np.float32)
        bins.append(edges)
    return dict(num_columns=num, binary_columns=cat, center=center, scale=scale, bins=bins,
        original_width=x.shape[1], excluded_train_constant_columns=np.flatnonzero(~keep))


def transform(x, valid, parameters):
    num = parameters['num_columns']
    xn = np.clip((x[:, num]-parameters['center'])/parameters['scale'], -8, 8)
    xn = np.where(valid[:, num], xn, 0.).astype(np.float32)
    xb = x[:, parameters['binary_columns']].astype(np.float32)
    if not np.isin(xb, (0, 1)).all() or not np.isfinite(xn).all():
        raise ValueError('invalid transformed matrix')
    return xn, xb


def build(root):
    cfg, d, c, h, a, audit = load()
    previous = OUTPUT/'simple_controls/preprocessing.joblib'
    prep = joblib.load(previous)
    if not prep['compact']:
        raise ValueError('expected compact XGBoost recipe')
    matrices=[]; histories=[]; text=[]
    # 测试输入缓存不参与任何拟合或模型选择；测试预测仅在所有选择完成后运行。
    for start in range(0, len(d.times), 256):
        pos = np.arange(start, min(start+256, len(d.times)))
        batches = restrict(prepare(d, c, a, pos), a, True)
        for _, hi, b, _, _ in batches:
            raw = flat_features(b).cpu().numpy()
            histories.append(hi.cpu().numpy())
            matrices.append(np.concatenate((hi.cpu().numpy(), raw[:, :prep['boundary']],
                prep['text_pca'].transform(raw[:, prep['boundary']:])), 1))
            text.append(b.text_present.any(1).cpu().numpy())
    x = np.concatenate(matrices).astype(np.float32)
    valid = numeric_valid(a, np.concatenate(histories), np.concatenate(text))
    params = fit_preprocessing(x, valid, d.arrays['patient_splits']=='train')
    xn, xb = transform(x, valid, params)
    # PCA及尺度落盘用于新进程重放；不在验证/测试折重拟合。
    params['text_pca'] = prep['text_pca']
    params['boundary'] = prep['boundary']
    joblib.dump(params, root/'preprocessing.joblib')
    np.savez_compressed(root/'matrix.npz', x_num=xn, x_binary=xb, x_xgb=x,
        times=d.times, events=d.events, hadm_ids=d.arrays['hadm_ids'], subject_ids=d.arrays['subject_ids'],
        patient_splits=d.arrays['patient_splits'])
    report = dict(admissions=len(x), patients=len(np.unique(d.arrays['subject_ids'])),
        counts={f:int((d.arrays['patient_splits']==f).sum()) for f in ('train','validation','test')},
        original_features=x.shape[1], numeric_features=len(params['num_columns']), binary_features=len(params['binary_columns']),
        train_constant_excluded=len(params['excluded_train_constant_columns']), train_only_fit=True,
        input_sha256=sha256(INPUT/'interface.npz'), reused_text_pca_sha256=sha256(previous), data=d.provenance,
        matrix_sha256=sha256(root/'matrix.npz'), preprocessing_sha256=sha256(root/'preprocessing.joblib'))
    write_json(root/'data_contract.json', report)
    return report


def read(root):
    with np.load(root/'matrix.npz') as z:
        data = {key:z[key] for key in z.files}
    return data, joblib.load(root/'preprocessing.joblib')
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/public_tabular_experiment.py · lines 1–231

<!-- REVIEW_FRAGMENT {"path":"training/v6/public_tabular_experiment.py","part":1,"start_line":1,"end_line":231,"source_sha256":"0d9621490c95d6cd226a56355ba886735d95dd3876f27991a24842b2c6a66aab","fragment_sha256":"0d9621490c95d6cd226a56355ba886735d95dd3876f27991a24842b2c6a66aab","added_terminal_newline":false} -->
````python
"""公开骨干的固定预算完整训练；验证选择在测试预测之前落盘。"""
import argparse
import json
import math
from pathlib import Path
import time
import importlib.metadata
import warnings
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.public_tabular_data import build, read
from training.v6.public_tabular_model import ARMS, PublicModel, initial_bias, ensemble_nll, ensemble_curve
from training.v6.audited_experiment import save_curve, summary, OUTPUT as AUDITED
from training.v6.evaluation import paired_interval
from training.v6.train import write_json

ROOT=Path('training_runs/v6_public_tabular_2026-10-12')
LRS=(3e-4,1e-3)
SEEDS=(42,7)
SOURCES=('training/v6/public_tabular_data.py','training/v6/public_tabular_model.py',
         'training/v6/public_tabular_experiment.py','training/v6/public_tabular_verify.py',
         'training/v6/risk.py','training/v6/token_model.py','training/v6/evaluation.py')


def setup():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.use_deterministic_algorithms(True)
    warnings.filterwarnings('ignore',message=r'The .*feature has just two bin edges.*',category=UserWarning)


def model_for(arm,seed,data,prep):
    torch.manual_seed(seed)
    model=PublicModel(arm,data['x_num'].shape[1],data['x_binary'].shape[1],
        [torch.as_tensor(b) for b in prep['bins']],initial_bias(data['events'][data['patient_splits']=='train'])).cuda()
    torch.manual_seed(seed)  # 构造模块消耗的随机数不改变训练采样/dropout起点。
    return model


def tensors(data,fold):
    ix=np.flatnonzero(data['patient_splits']==fold)
    return ix,tuple(torch.as_tensor(data[key][ix],device='cuda',dtype=dtype)
        for key,dtype in (('x_num',torch.float32),('x_binary',torch.float32),('times',torch.float32),('events',torch.long)))


@torch.no_grad()
def parameters(model,batch,size=128):
    model.eval();xn,xb,t,e=batch;a=[];k=[];total=0.
    for start in range(0,len(t),size):
        end=min(start+size,len(t))
        # 推理固定FP32，避免低精度改变细小排序差；训练统一BF16混合精度。
        aa,kk=model(xn[start:end],xb[start:end])
        total+=float(ensemble_nll(aa,kk,t[start:end],e[start:end]))*(end-start)
        a.append(aa.cpu().numpy());k.append(kk.cpu().numpy())
    return np.concatenate(a),np.concatenate(k),total/len(t)


def validation(model,batch,data,ix):
    a,k,nll=parameters(model,batch)
    curve=ensemble_curve(a,k)
    y=((data['events'][ix,None]==1)&(data['times'][ix,None]<=np.arange(366))).astype(float)
    brier=((curve[:,:,1]-y)**2).mean(0)
    from training.baselines.rank_metrics import auc_rank
    return dict(nll=nll,ibs=float(np.trapezoid(brier[7:],np.arange(7,366))/358),
        brier90=float(brier[90]),auc90=float(auc_rank(curve[:,90,1],y[:,90].astype(np.int8))))


def fit(model,train,val,data,vix,path,seed,lr,epochs=100):
    path.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    generator=torch.Generator(device='cuda').manual_seed(seed)
    xn,xb,t,e=train;rows=[];best=math.inf;best_epoch=0
    def record(epoch,online=None,rate=0.):
        nonlocal best,best_epoch
        stats=validation(model,val,data,vix)
        _,_,fixed=parameters(model,tuple(x[:2048] for x in train))
        rows.append(dict(epoch=epoch,training_nll_fixed2048=fixed,training_online_nll=online,
            lr=rate,**{'validation_'+k:v for k,v in stats.items()}))
        if stats['ibs']<best:
            best=stats['ibs'];best_epoch=epoch
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,validation_ibs=best),path/'best.pt')
        if epoch%10==0 or epoch==epochs:
            write_json(path/'history.json',rows)
            print(json.dumps(dict(arm=model.arm,seed=seed,lr=lr,epoch=epoch,validation_ibs=stats['ibs'],
                best_epoch=best_epoch,seconds=round(time.perf_counter()-start,1))),flush=True)
        write_json(path/'progress.json',dict(epoch=epoch,epochs=epochs,best_epoch=best_epoch,validation_ibs=best,
            elapsed_seconds=time.perf_counter()-start))
    record(0)
    for epoch in range(1,epochs+1):
        rate=lr*(epoch/5 if epoch<=5 else .1+.9*.5*(1+math.cos(math.pi*(epoch-5)/max(1,epochs-5))))
        for group in optimizer.param_groups:group['lr']=rate
        model.train();permutation=torch.randperm(len(t),device='cuda',generator=generator);total=0.
        # 同一有效batch256；每个物理batch128的损失按实际人数加权。
        for block in range(0,len(t),256):
            indices=permutation[block:block+256];optimizer.zero_grad(set_to_none=True)
            for micro in indices.split(128):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    a,k=model(xn[micro],xb[micro]);loss=ensemble_nll(a,k,t[micro],e[micro])
                if not torch.isfinite(loss):raise ValueError('nonfinite objective')
                (loss*len(micro)/len(indices)).backward()
                total+=float(loss.detach())*len(micro)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
            optimizer.step()
        record(epoch,total/len(t),rate)
    result=dict(arm=model.arm,seed=seed,lr=lr,completed_epochs=epochs,best_epoch=best_epoch,
        validation_ibs=best,parameters=sum(p.numel() for p in model.parameters()),
        peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,elapsed_seconds=time.perf_counter()-start,
        checkpoint_sha256=sha256(path/'best.pt'))
    write_json(path/'training_result.json',result)
    return result


def profile(root):
    setup();root.mkdir(parents=True,exist_ok=True)
    if not (root/'matrix.npz').exists():print(json.dumps(build(root)),flush=True)
    data,prep=read(root);_,batch=tensors(data,'train');vix,val=tensors(data,'validation')
    profiles={};initial=[]
    for arm in ARMS:
        model=model_for(arm,42,data,prep);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
        with torch.no_grad():
            model.eval();aa,kk=model(batch[0][:128],batch[1][:128])
            initial.append((aa.cpu().numpy()[:,0],kk.cpu().numpy()[:,0]))
        model.train()
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        for step in range(5):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                a,k=model(batch[0][:128],batch[1][:128]);loss=ensemble_nll(a,k,batch[2][:128],batch[3][:128])
            loss.backward();optimizer.step()
        torch.cuda.synchronize();step_seconds=(time.perf_counter()-start)/5
        start=time.perf_counter();stats=validation(model,val,data,vix)
        profiles[arm]=dict(step_seconds=step_seconds,validation_seconds=time.perf_counter()-start,
            parameters=sum(p.numel() for p in model.parameters()),peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
            **stats)
        print(json.dumps(dict(profile=arm,**profiles[arm])),flush=True)
        del model,optimizer;torch.cuda.empty_cache()
    if not all(np.array_equal(initial[0][0],a) and np.array_equal(initial[0][1],k) for a,k in initial):
        raise ValueError('initial risk differs across backbones')
    write_json(root/'profile.json',dict(models=profiles,initial_risk_identical=True,labels_not_used_for_model_choice=True))


def run(root,epochs=100):
    setup();begin=time.perf_counter()
    if not (root/'profile.json').exists():raise ValueError('profile all backbones first')
    if (root/'contract.json').exists():raise FileExistsError('new experiment root required')
    data,prep=read(root);trix,train=tensors(data,'train');vix,val=tensors(data,'validation')
    contract=dict(arms=ARMS,seeds=SEEDS,learning_rates=LRS,epochs=epochs,effective_batch=256,micro_batch=128,
        optimizer='AdamW',weight_decay=1e-4,selection='validation IBS7-365; LR by two-seed mean',
        end_to_end=True,frozen_history=False,mixed_precision_training='BF16',inference='FP32',
        source_sha256={p:sha256(Path(p)) for p in SOURCES},
        package_versions={p:importlib.metadata.version(p) for p in ('tabm','rtdl-revisiting-models','rtdl-num-embeddings','torch')},
        package_source_sha256={str(p):sha256(p) for p in Path('training/runtime/public_tabular_2026-10-12').glob('*.py')},
        protocol_sha256=sha256(Path('docs/V6_PUBLIC_TABULAR_PROTOCOL_2026-10-12.md')),
        data=json.loads((root/'data_contract.json').read_text()),gpu=torch.cuda.get_device_name(0),reused_test=True)
    write_json(root/'contract.json',contract);fits={}
    for arm in ARMS:
        fits[arm]={}
        for lr in LRS:
            fits[arm][str(lr)]={}
            for seed in SEEDS:
                model=model_for(arm,seed,data,prep)
                path=root/arm/f'lr{lr}'/f'seed{seed}'
                fits[arm][str(lr)][str(seed)]=fit(model,train,val,data,vix,path,seed,lr,epochs)
                del model;torch.cuda.empty_cache()
    choices={arm:min(LRS,key=lambda lr:np.mean([fits[arm][str(lr)][str(s)]['validation_ibs'] for s in SEEDS])) for arm in ARMS}
    write_json(root/'selected_before_test.json',dict(fits=fits,learning_rates=choices,all_training_complete=True))
    del train;torch.cuda.empty_cache();tix,test=tensors(data,'test')
    models={};comparisons={}
    for arm in ARMS:
        models[arm]={};comparisons[arm]={}
        for seed in SEEDS:
            lr=choices[arm];path=root/arm/f'lr{lr}'/f'seed{seed}';model=model_for(arm,seed,data,prep)
            checkpoint=torch.load(path/'best.pt',map_location='cuda',weights_only=True);model.load_state_dict(checkpoint['state_dict'])
            models[arm][str(seed)]=dict(**fits[arm][str(lr)][str(seed)],folds={});comparisons[arm][str(seed)]={}
            for fold,ix,batch in (('validation',vix,val),('test',tix,test)):
                a,k,nll=parameters(model,batch);curve=ensemble_curve(a,k)
                # summary/save_curve只依赖这些属性，不重新读取整个原始底账。
                from types import SimpleNamespace
                holder=SimpleNamespace(times=data['times'],events=data['events'],arrays=data)
                save_curve(path/(fold+'_predictions.npz'),curve,holder,ix,log_amplitude_members=a,shape_members=k)
                models[arm][str(seed)]['folds'][fold]=dict(nll=nll,**summary(curve,holder,ix))
                ref=AUDITED/'simple_controls'/f'xgb{seed}_{fold}_predictions.npz'
                with np.load(ref) as z:
                    if not np.array_equal(z['hadm_ids'],data['hadm_ids'][ix]):raise ValueError('reference members differ')
                    comparator=z['curve']
                comparisons[arm][str(seed)][fold]=paired_interval(data['subject_ids'][ix],data['times'][ix],data['events'][ix],
                    curve[:,:,1],comparator[:,:,1],seed=seed)
            del model
    report=dict(status='completed_training_and_prediction_pending_independent_verification',models=models,fits=fits,
        learning_rates=choices,comparisons_to_same_input_xgb=comparisons,elapsed_seconds=time.perf_counter()-begin,contract=contract)
    write_json(root/'report.json',report)
    render(root,report)
    print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'])),flush=True)


def render(root,r):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for i,arm in enumerate(ARMS):
        for s,col,dy in ((42,'C0',-.1),(7,'C1',.1)):
            row=r['comparisons_to_same_input_xgb'][arm][str(s)]['test']
            for ax,key,ci in ((axes[0],'integrated_brier_difference','integrated_brier_95ci'),(axes[1],'auc90_difference','auc90_95ci')):
                ax.plot(row[ci],[i+dy]*2,color=col);ax.scatter(row[key],i+dy,color=col,label=f'Seed {s}' if i==0 else None)
    for ax in axes:
        ax.axvline(0,color='black',lw=1);ax.set_yticks(range(len(ARMS)),ARMS);ax.grid(axis='x',alpha=.2);ax.legend()
    axes[0].invert_yaxis();axes[0].set_xlabel('IBS7-365: candidate minus XGBoost (lower better)')
    axes[1].set_xlabel('AUC90: candidate minus XGBoost (higher better)')
    fig.suptitle('Official backbones; matched information; patient bootstrap200; reused test')
    fig.tight_layout();fig.savefig(root/'paired_xgb.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,s in enumerate(SEEDS):
        for arm in ARMS:
            lr=r['learning_rates'][arm];rows=json.loads((root/arm/f'lr{lr}'/f'seed{s}'/'history.json').read_text())
            axes[i,0].plot([v['epoch'] for v in rows],[v['training_nll_fixed2048'] for v in rows],label=arm)
            axes[i,1].plot([v['epoch'] for v in rows],[v['validation_ibs'] for v in rows],label=arm)
        for ax in axes[i]:ax.grid(alpha=.2);ax.set_xlabel('Epoch');ax.legend(fontsize=8)
        axes[i,0].set_title(f'Seed{s}: training joint NLL');axes[i,1].set_title(f'Seed{s}: validation IBS7-365')
    fig.tight_layout();fig.savefig(root/'training.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--mode',choices=('profile','train'),required=True);p.add_argument('--epochs',type=int,default=100)
    args=p.parse_args();(profile(args.output) if args.mode=='profile' else run(args.output,args.epochs))
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/public_tabular_model.py · lines 1–104

<!-- REVIEW_FRAGMENT {"path":"training/v6/public_tabular_model.py","part":1,"start_line":1,"end_line":104,"source_sha256":"36e33a07ed77aec3ae53a64becde19dfe60a85d47e235998d3fb5664e42e8f66","fragment_sha256":"36e33a07ed77aec3ae53a64becde19dfe60a85d47e235998d3fb5664e42e8f66","added_terminal_newline":false} -->
````python
"""官方表格骨干；共享联合连续时间风险头，包源码不修改。"""
import math
import types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import tabm
import rtdl_revisiting_models as rtdl
import rtdl_num_embeddings as numeric
from training.v6.risk import RiskOutput, competing_nll
from training.v6.token_model import fast_cif

ARMS = ('mlp', 'resnet_ple', 'resnet_periodic', 'tabm_ple', 'ft_transformer')


def sdpa_forward(self, x_q, x_kv):
    """与官方Attention同一投影/缩放/dropout；只替换运算内核。"""
    q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
    if self.key_compression is not None:
        k = self.key_compression(k.transpose(1, 2)).transpose(1, 2)
        compression = self.key_compression if self.value_compression is None else self.value_compression
        v = compression(v.transpose(1, 2)).transpose(1, 2)
    def heads(x):
        return x.reshape(len(x), x.shape[1], self._n_heads, -1).transpose(1, 2)
    p = self.dropout.p if self.dropout is not None and self.training else 0.
    x = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), dropout_p=p)
    x = x.transpose(1, 2).reshape(len(x_q), x_q.shape[1], -1)
    return self.W_out(x) if self.W_out is not None else x


def initial_bias(events):
    fractions = np.array([(events == e).mean() for e in (1, 2)])
    total = fractions.sum()
    if not 0 < total < 1 or np.any(fractions <= 0):
        raise ValueError('both events and event-free training observations required')
    hazard = -np.log1p(-total) * fractions / total
    return np.r_[np.log(hazard), np.repeat(np.log(np.expm1(.9)), 2)].astype(np.float32)


class PublicModel(nn.Module):
    def __init__(self, arm, n_num, n_binary, bins, bias, *, sdpa=True):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.k = 16 if arm == 'tabm_ple' else 1
        self.embedding = None
        width = 96 if arm == 'ft_transformer' else 256
        if arm in ('resnet_ple', 'tabm_ple'):
            embedding = numeric.PiecewiseLinearEmbeddings(bins, 8, activation=True, version='B')
        elif arm == 'resnet_periodic':
            embedding = numeric.PeriodicEmbeddings(n_num, 8, n_frequencies=8,
                frequency_init_scale=.01, activation=True, lite=True)
        else:
            embedding = None
        if arm == 'mlp':
            self.backbone = rtdl.MLP(d_in=n_num+n_binary, d_out=None, n_blocks=3, d_block=width, dropout=.1)
        elif arm.startswith('resnet'):
            self.embedding = embedding
            self.backbone = rtdl.ResNet(d_in=n_num*8+n_binary, d_out=None, n_blocks=2,
                d_block=width, d_hidden=None, d_hidden_multiplier=2., dropout1=.1, dropout2=.1)
        elif arm == 'tabm_ple':
            self.backbone = tabm.TabM.make(n_num_features=n_num, cat_cardinalities=[2]*n_binary,
                d_out=None, num_embeddings=embedding, n_blocks=2, d_block=width, k=self.k, dropout=.1)
        else:
            self.backbone = rtdl.FTTransformer(n_cont_features=n_num, cat_cardinalities=[2]*n_binary,
                d_out=None, n_blocks=2, d_block=width, attention_n_heads=4, attention_dropout=.1,
                ffn_d_hidden_multiplier=4/3, ffn_dropout=.1, residual_dropout=0.)
            if sdpa:
                for block in self.backbone.backbone.blocks:
                    attention = block['attention']
                    attention.forward = types.MethodType(sdpa_forward, attention)
        self.head = tabm.LinearEnsemble(width, 4, k=self.k) if self.k > 1 else nn.Linear(width, 4)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.as_tensor(bias).expand_as(self.head.bias))

    def forward(self, x_num, x_binary):
        if self.arm in ('ft_transformer', 'tabm_ple'):
            h = self.backbone(x_num, x_binary.long() if x_binary.shape[1] else None)
        else:
            x = self.embedding(x_num).flatten(1) if self.embedding is not None else x_num
            h = self.backbone(torch.cat((x, x_binary), 1))
        # 混合精度只用于骨干；头保持FP32，避免不同骨干的bias相加路径造成初始风险差。
        with torch.autocast(h.device.type, enabled=False):
            raw = self.head(h.float())
        if raw.ndim == 2:
            raw = raw[:, None]
        return raw[..., :2].clamp(-10., 6.), F.softplus(raw[..., 2:]) + .1


def ensemble_nll(a, k, times, events):
    """事件标签复制到成员，平均成员损失而非平均参数的损失。"""
    members = a.shape[1]
    return competing_nll(RiskOutput(a.flatten(0, 1), k.flatten(0, 1)),
        times.repeat_interleave(members), events.repeat_interleave(members))


def ensemble_curve(a, k, *, device='cuda'):
    if a.shape != k.shape or a.ndim != 3 or a.shape[-1] != 2:
        raise ValueError('expected [patients, ensemble, 2]')
    curves = fast_cif(a.reshape(-1, 2), k.reshape(-1, 2), device=device)
    return curves.reshape(len(a), a.shape[1], 366, 3).mean(1)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/public_tabular_report.py · lines 1–196

<!-- REVIEW_FRAGMENT {"path":"training/v6/public_tabular_report.py","part":1,"start_line":1,"end_line":196,"source_sha256":"028dd086c5259268bd3f3214ff8348c1248a929052c0ef78a3500e0754863bfa","fragment_sha256":"028dd086c5259268bd3f3214ff8348c1248a929052c0ef78a3500e0754863bfa","added_terminal_newline":false} -->
````python
"""正式结果完成后补充原始来源/树重放、汇总和图表；不重新挑模型。"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import xgboost as xgb
from patient_data.export_mimiciv import sha256
from training.v6.public_tabular_data import read
from training.v6.public_tabular_experiment import ROOT, setup
from training.v6.audited_experiment import load, OUTPUT as AUDITED
from training.v6.audited_simple_controls import tree_curve
from training.v6.finalize_audited_controls import NativeTree
from training.v6.evaluation import paired_interval, metrics
from training.v6.train import write_json

ORIGINAL=Path('training_runs/v6_original_interaction_2026-10-12_r3')


def mixture_nll(a,k,times,events):
    """最终混合联合分布NLL；区别于TabM训练用的成员平均NLL。"""
    from scipy.special import logsumexp
    scaled=np.maximum(times,1e-3)[:,None,None]/365.
    cumulative=np.exp(a)*np.power(scaled,k)
    log_hazard=a+np.log(k)-np.log(365.)+(k-1)*np.log(scaled)
    selected=log_hazard[np.arange(len(a))[:,None],np.arange(a.shape[1])[None,:],np.maximum(events-1,0)[:,None]]
    loss=cumulative.sum(2)-np.where(events[:,None]>0,selected,0.)
    return float((-logsumexp(-loss,axis=1)+np.log(a.shape[1])).mean())


def finish(root=ROOT):
    setup();r=json.loads((root/'report.json').read_text());verification=json.loads((root/'independent_verification.json').read_text())
    if verification['status']!='independently_verified' or verification['report_sha256']!=sha256(root/'report.json'):
        raise ValueError('verify before final report')
    d,prep=read(root);cfg,original,c,h,a,audit=load()
    for key in ('hadm_ids','subject_ids','patient_splits'):
        if not np.array_equal(d[key],original.arrays[key]):raise ValueError('original member mismatch')
    if not np.array_equal(d['events'],original.events) or not np.array_equal(d['times'],original.times):
        raise ValueError('original strict labels mismatch')
    if sha256(Path('docs/V6_PUBLIC_TABULAR_PROTOCOL_2026-10-12.md'))!=r['contract']['protocol_sha256']:
        raise ValueError('predeclared protocol changed')
    manifest=json.loads((cfg.snapshot/'metadata/manifest.json').read_text())
    raw_hashes={}
    for key,value in manifest['sources'].items():
        file=cfg.snapshot/(key+'.csv.gz')
        digest=value['sha256']
        actual=sha256(file)
        if actual!=digest:raise ValueError('raw snapshot changed: '+key)
        raw_hashes[key]=actual
    simple=json.loads((AUDITED/'simple_controls/report.json').read_text())
    old=json.loads((ORIGINAL/'report.json').read_text())
    tree_errors={};comparisons={};references={};predictive_nll={}
    for seed in (42,7):
        depth=simple['xgb_selected_depth'][str(seed)];models=[]
        for cause in (1,2):
            booster=xgb.Booster();booster.load_model(AUDITED/'simple_controls'/f'native_seed{seed}_depth{depth}_cause{cause}.ubj')
            models.append(NativeTree(booster))
        with np.load(AUDITED/'simple_controls'/f'xgb{seed}_baseline.npz') as z:
            hz=[(z[f'{j}_time'],z[f'{j}_increment']) for j in range(2)]
        for fold in ('validation','test'):
            ix=np.flatnonzero(d['patient_splits']==fold)
            curve=tree_curve(models,hz,d['x_xgb'][ix])
            with np.load(AUDITED/'simple_controls'/f'xgb{seed}_{fold}_predictions.npz') as z:
                if not np.array_equal(z['hadm_ids'],d['hadm_ids'][ix]):raise ValueError('tree members mismatch')
                err=float(np.max(np.abs(curve-z['curve'])))
            if err>1e-10:raise ValueError('matrix is not same XGBoost source representation')
            tree_errors[f'{seed}/{fold}']=err
        references[str(seed)]=dict(xgb=simple['folds'][f'xgb{seed}'],original_v6=old['models'][str(seed)]['control']['folds'])
    for arm,models in r['models'].items():
        comparisons[arm]={};predictive_nll[arm]={}
        for seed in (42,7):
            path=root/arm/f"lr{r['learning_rates'][arm]}"/f'seed{seed}'
            predictive_nll[arm][str(seed)]={}
            for fold in ('validation','test'):
                with np.load(path/(fold+'_predictions.npz')) as z:
                    predictive_nll[arm][str(seed)][fold]=mixture_nll(z['log_amplitude_members'].astype(float),
                        z['shape_members'].astype(float),z['times'],z['events'])
            ix=np.flatnonzero(d['patient_splits']=='test')
            with np.load(path/'test_predictions.npz') as z:curve=z['curve']
            with np.load(ORIGINAL/f'seed{seed}'/'control/test_predictions.npz') as z:
                if not np.array_equal(z['hadm_ids'],d['hadm_ids'][ix]) or not np.array_equal(z['events'],d['events'][ix]):
                    raise ValueError('original reference members/endpoint mismatch')
                reference=z['curve']
            comparisons[arm][str(seed)]=paired_interval(d['subject_ids'][ix],d['times'][ix],d['events'][ix],curve[:,:,1],reference[:,:,1],seed=seed)
    supplement=dict(status='completed_and_verified',original_labels_checked=True,raw_snapshot_files=len(raw_hashes),
        raw_snapshot_hashes=raw_hashes,xgb_shared_matrix_replay_errors=tree_errors,references=references,
        comparisons_to_original_v6=comparisons,predictive_mixture_nll=predictive_nll,source_sha256=sha256(Path(__file__)),
        limits='Original V6 differs in recipe and frozen training. Same XGB information, but NN normalization/clipping differ; not pure parameter-count comparison.')
    supplement['best_public_by_validation']=min(r['models'],key=lambda arm:np.mean([
        r['models'][arm][str(seed)]['validation_ibs'] for seed in (42,7)]))
    supplement['default_changed']=False
    supplement['text_pca_explained_variance_fraction']=float(prep['text_pca'].explained_variance_ratio_.sum())
    write_json(root/'final_summary.json',supplement)
    render(root,r,references,d)
    document(root,r,supplement)
    # 验证完的数值不改；仅将本轮新报告状态发布为完成，记录发布前后哈希。
    before=sha256(root/'report.json')
    r['status']='completed_and_independently_verified'
    write_json(root/'report.json',r)
    verification['verified_report_sha256_before_publication']=before
    verification['report_sha256']=sha256(root/'report.json')
    verification['publication_change']='report status only; no metrics/checkpoints/predictions changed'
    write_json(root/'independent_verification.json',verification)
    print(json.dumps(dict(status=supplement['status'],trainings=20,epochs_each=100,
        raw_files_unchanged=len(raw_hashes),tree_replay_errors=tree_errors)),flush=True)


def render(root,r,refs,d):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['original_v6','xgb']+list(r['models']);means={}
    for name in names:
        source=[refs[str(s)][name]['test']['readmission'] for s in (42,7)] if name in refs['42'] else [r['models'][name][str(s)]['folds']['test']['readmission'] for s in (42,7)]
        means[name]=[np.mean([m['points']['90']['auc'] for m in source]),np.mean([m['integrated_brier_7_365'] for m in source])]
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for i,name in enumerate(names):
        for j in (0,1):axes[j].scatter(means[name][j],i,s=70)
    for ax in axes:ax.set_yticks(range(len(names)),names);ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis();axes[0].set_xlabel('AUC90 (two-seed mean; higher better)');axes[1].set_xlabel('IBS7-365 (two-seed mean; lower better)')
    fig.suptitle('Strict AECOPD; historical test; original V6 recipe differs')
    fig.tight_layout();fig.savefig(root/'performance_overview.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5));ix=np.flatnonzero(d['patient_splits']=='test')
    for name in names:
        measurements=[]
        for seed in (42,7):
            if name=='original_v6':file=ORIGINAL/f'seed{seed}'/'control/test_predictions.npz'
            elif name=='xgb':file=AUDITED/'simple_controls'/f'xgb{seed}_test_predictions.npz'
            else:file=root/name/f"lr{r['learning_rates'][name]}"/f'seed{seed}'/'test_predictions.npz'
            with np.load(file) as z:curve=z['curve']
            m=metrics(d['times'][ix],d['events'][ix],curve[:,:,1]);measurements.append(m)
        axes[0].plot(np.arange(7,366),np.nanmean([m['auc_curve'][7:] for m in measurements],axis=0),label=name)
        axes[1].plot(np.arange(7,366),np.mean([m['brier_curve'][7:] for m in measurements],axis=0),label=name)
    axes[0].set_ylabel('AUC(t)');axes[1].set_ylabel('Brier(t)')
    for ax in axes:ax.set_xlabel('Days after discharge');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Complete continuous-time risk evaluation; two-seed mean; reused test')
    fig.tight_layout();fig.savefig(root/'time_dependent_metrics.png',dpi=170);plt.close(fig)


def document(root,r,s):
    def points(m):return [m['points'][str(day)]['auc'] for day in (30,90,180,365)]+[m['points']['90']['brier'],m['integrated_brier_7_365']]
    lines=['# 官方表格深度学习模块：完整受控对照结果','',
        '本轮全部20次训练均完成100epoch；10个按验证选择的模型、20折新进程重放通过。没有覆盖原V6或改变队列/严格结局/文本向量。',
        '', '[固定方案](V6_PUBLIC_TABULAR_PROTOCOL_2026-10-12.md) · [结果目录](../'+str(root).replace('\\','/')+'/README.md)',
        '', '## 输入及实际适配','',
        '19,401出院、5,351患者；训练13,592、验证2,949、历史测试2,860。共同信息原矩阵1390维；训练常量734列排除后，273连续+383二元列。检验/药物细节与有效性标记保留。四字段BERT章节均值及训练折16维PCA与XGBoost相同，不是全文原token训练，也不是重新编码。',
        '', '所有网络端到端训练，不冻结H34；每成员使用相同双原因患者相关Weibull联合似然。TabM16成员平均概率，不平均参数；其最终分布是Weibull混合，比单成员更灵活，不能将增益全归因于骨干/数值嵌入。report.json的NLL为成员平均训练损失，最终混合分布NLL单列于final_summary.json。模型/学习率选择只用验证IBS7–365；90天仅是报告点，每天累计/当日/条件当日概率均保存。',
        '', '## 历史测试结果（两种子平均，不是两种子集成）','',
        '|模型|AUC30|AUC90|AUC180|AUC365|Brier90↓|IBS7–365↓|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['original_v6','xgb']+list(r['models']):
        source=[s['references'][str(seed)][name]['test']['readmission'] for seed in (42,7)] if name in ('original_v6','xgb') else [r['models'][name][str(seed)]['folds']['test']['readmission'] for seed in (42,7)]
        values=np.mean([points(m) for m in source],axis=0)
        lines.append('|'+name+'|'+'|'.join(f'{v:.6f}' for v in values)+'|')
    lines+=['', '## 本轮结论','',
        '验证选择的公开候选为TabM+PLE。它相对同信息XGBoost的AUC90与IBS点估计在两个种子均更好，但患者配对95%区间均跨0，因此只能称小幅、待确认的增益，不能称显著突破或等效证明。',
        '', '原V6 AUC90约0.7992，TabM约0.7938；TabM的Brier90/IBS均值略好，但排序没有超过原V6，且原V6配方/历史冻结不同。当前不替换原V6默认；普通MLP、两种ResNet、FT均未胜出。',
        '', '多数单模型最佳点在2–19epoch；完成100epoch后验证反而恶化。该实证不支持“再多训练或只加深网络就会改善”的解释，但有限LR/容量预算也不能证明所有网络设置的性能上限。下一步不再无差别增加模块；TabM仅保留为有证据的候选。']
    lines+=['','## 实际训练、选择与资源','',
        '|候选|共同LR|最佳epoch（42/7）|参数量|四次训练累计分钟|最大显存GiB|验证IBS均值|',
        '|---|---:|---|---:|---:|---:|---:|']
    for arm in r['models']:
        m=[r['models'][arm][str(seed)] for seed in (42,7)]
        allruns=[value for group in r['fits'][arm].values() for value in group.values()]
        lines.append(f"|{arm}|{r['learning_rates'][arm]}|{m[0]['best_epoch']}/{m[1]['best_epoch']}|{m[0]['parameters']:,}|{sum(v['elapsed_seconds'] for v in allruns)/60:.2f}|{max(v['peak_gpu_gib'] for v in allruns):.3f}|{np.mean([v['validation_ibs'] for v in m]):.6f}|")
    lines+=['','## 与同信息XGBoost的配对差值','',
        '差值=候选−XGBoost。IBS负值更好，AUC正值更好。每种子患者聚类bootstrap200；区间条件于已经训练的模型，不计训练不确定性，也没有多重比较校正。',
        '', '|候选|种子|ΔAUC90（95%区间）|ΔIBS（95%区间）|','|---|---:|---|---|']
    for arm in r['models']:
        for seed in (42,7):
            v=r['comparisons_to_same_input_xgb'][arm][str(seed)]['test']
            lines.append(f"|{arm}|{seed}|{v['auc90_difference']:+.5f} [{v['auc90_95ci'][0]:+.5f}, {v['auc90_95ci'][1]:+.5f}]|{v['integrated_brier_difference']:+.6f} [{v['integrated_brier_95ci'][0]:+.6f}, {v['integrated_brier_95ci'][1]:+.6f}]|")
    lines+=['','## 图表','',
        '![整体性能](../'+str(root).replace('\\','/')+'/performance_overview.png)','',
        '![患者配对](../'+str(root).replace('\\','/')+'/paired_xgb.png)','',
        '![训练过程](../'+str(root).replace('\\','/')+'/training.png)','',
        '![整段时间评价](../'+str(root).replace('\\','/')+'/time_dependent_metrics.png)','',
        '## 证据限制与使用决定','',
        f"五个公开候选中，验证IBS均值最好的为 **{s['best_public_by_validation']}**。这是验证集选择，不是按测试AUC挑选；旧默认仍不覆盖。四字段16维PCA保留的训练方差比例约为{s['text_pca_explained_variance_fraction']*100:.1f}%，但方差保留不等于临床语义/预后信息保留；缺失状态也可能主导方差。本轮不能断言公开骨干已经用尽原始文本信息。",
        '',
        '本轮完成的是有限预算方法适配，不能证明所有深度学习模块的性能上限。只测试2个共同LR与1个固定容量；没有针对每个算法大规模调参。新网络与XGBoost信息来源相同，但网络数值归一化/截断不同。原V6的输入配方及历史冻结不同，作为强参照而不是纯骨干消融。',
        '', '完整100epoch不等于应该使用第100epoch：如果验证在早期最好、随后训练误差下降但验证变差，就是泛化恶化；扩大网络或延长训练不能自动解决。检验/药物的医学有效性不等于在“历史+诊断+去向”条件下对本库严格再入院标签一定有增量。',
        '', '按既定边界不自动替换默认模型、不删医学变量。日概率仍只对应MIMIC内观察到的再入院/死亡，不能视作院外完整真实风险。历史测试已被反复查看，所有优势须外部/新留出验证；不能以某一天AUC最高直接宣布突破。',
        '', f"新进程10模型20折重放、概率守恒及患者划分检查通过；原始快照{s['raw_snapshot_files']}文件哈希未变；共同XGBoost矩阵重放最大误差{max(s['xgb_shared_matrix_replay_errors'].values()):.3g}。",
        '', '## 方法的可核验来源','',
        '- [TabM：官方ICLR2025代码](https://github.com/yandex-research/tabm)：共享权重的高效集成。本项目适配为联合竞争风险，而非其原分类/回归benchmark。',
        '- [MLP/ResNet/FT-Transformer：官方NeurIPS2021代码](https://github.com/yandex-research/rtdl-revisiting-models)：使用真实公开实现，FT读CLS。',
        '- [数值嵌入：官方NeurIPS2022代码](https://github.com/yandex-research/rtdl-num-embeddings)：分段线性与周期嵌入，训练折拟合分位点；不代表可自动补齐缺失临床事实。',
        '', '复现命令与固定包见training/runtime/README.md；完整数字、全部参数候选及每日预测以本轮report.json为准。']
    if r['contract'].get('phase')=='complete_official_readout':
        lines[3:3]=['', '## 读出层补正与真实执行次数','',
            '首次d_out=None同时移除了ResNet/FT的末端归一化+ReLU，发现后保留为探索记录，不以其负结果评价完整官方模型。正式结果来自official_readout子目录：ResNet恢复BatchNorm+ReLU，FT恢复LayerNorm+ReLU，再接共同生存线性层。受影响12配置重新100epoch，MLP/TabM8配置核验后原样复用；整个过程实际32次训练，不冒充首次一次完成。',
            '', '[补正原因、范围及合同](V6_PUBLIC_TABULAR_READOUT_CORRECTION_2026-10-12.md)。测试已被反复使用；补正源于官方代码核查，不是按测试成绩选择架构。']
    Path('docs/V6_PUBLIC_TABULAR_RESULTS_2026-10-12.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':finish()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/public_tabular_verify.py · lines 1–52

<!-- REVIEW_FRAGMENT {"path":"training/v6/public_tabular_verify.py","part":1,"start_line":1,"end_line":52,"source_sha256":"00833817560bf060f23c97178eab6296c8a839a5c69f03417fd22ccc0c22e54b","fragment_sha256":"00833817560bf060f23c97178eab6296c8a839a5c69f03417fd22ccc0c22e54b","added_terminal_newline":false} -->
````python
"""另一个进程核对实际数据、依赖和选定模型重放，不修改历史实验。"""
import json
from pathlib import Path
import importlib.metadata
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.public_tabular_data import read
from training.v6.public_tabular_experiment import ROOT, setup, model_for, tensors, parameters
from training.v6.public_tabular_model import ensemble_curve
from training.v6.token_model import daily_probabilities
from training.v6.train import write_json


def verify(root=ROOT):
    setup();r=json.loads((root/'report.json').read_text());contract=r['contract'];d,prep=read(root)
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('code changed: '+p)
    for package,version in contract['package_versions'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('dependency changed')
    for p,digest in contract['package_source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('installed package source changed')
    for p,key in (('matrix.npz','matrix_sha256'),('preprocessing.joblib','preprocessing_sha256')):
        if sha256(root/p)!=contract['data'][key]:raise ValueError('input changed')
    for person in np.unique(d['subject_ids']):
        if len(np.unique(d['patient_splits'][d['subject_ids']==person]))!=1:raise ValueError('patient crossed folds')
    errors={}
    for arm,models in r['models'].items():
        for seed,value in models.items():
            path=root/arm/f"lr{r['learning_rates'][arm]}"/f'seed{seed}'
            if sha256(path/'best.pt')!=value['checkpoint_sha256']:raise ValueError('checkpoint changed')
            history=json.loads((path/'history.json').read_text())
            if len(history)!=contract['epochs']+1 or min(history,key=lambda x:x['validation_ibs'])['epoch']!=value['best_epoch']:
                raise ValueError('epochs or selection mismatch')
            model=model_for(arm,int(seed),d,prep)
            model.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            for fold in ('validation','test'):
                ix,batch=tensors(d,fold);a,k,_=parameters(model,batch);curve=ensemble_curve(a,k)
                with np.load(path/(fold+'_predictions.npz')) as z:
                    if not np.array_equal(z['hadm_ids'],d['hadm_ids'][ix]) or not np.array_equal(z['events'],d['events'][ix]) or not np.array_equal(z['times'],d['times'][ix]):
                        raise ValueError('prediction members/labels changed')
                    err=float(np.max(np.abs(z['curve']-curve)))
                if err>1e-9:raise ValueError('prediction replay differs '+str(err))
                daily_probabilities(curve);errors[f'{arm}/{seed}/{fold}']=err
            del model
    result=dict(status='independently_verified',selected_models=10,completed_trainings=20,
        replay_errors=errors,patient_split_checked=True,daily_probability_mass_checked=True,
        report_sha256=sha256(root/'report.json'))
    write_json(root/'independent_verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':verify()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/radiology_experiment.py · lines 1–190

<!-- REVIEW_FRAGMENT {"path":"training/v6/radiology_experiment.py","part":1,"start_line":1,"end_line":190,"source_sha256":"98b27ffc9417375b1d604856f83200250a66146a396e7e88cfe8ec5801ff5be9","fragment_sha256":"98b27ffc9417375b1d604856f83200250a66146a396e7e88cfe8ec5801ff5be9","added_terminal_newline":false} -->
````python
"""冻结RadBERT/RadGraph影像输入，同容量同初始化、同H、同队列配对重训。"""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from patient_data.export_mimiciv import sha256
from training.v6.compact_ablation import inputs
from training.v6.clinical_facts_context import FactsBatch, FactsData, FactsEncoder, checked
from training.v6.model import AnchoredModel
from training.v6.compact_model import CompactClinicalEncoder
from training.v6.train import fit, predict, write_json
from training.v6.group_ablation import evaluate
from training.v6.evaluation import GRID, metrics, paired_interval, public_metrics
from training.v6.risk import predict_cif
from training.v6 import facts_experiment

ARMS = ('without_new', 'availability_only', 'radbert', 'radgraph')
WIDTHS = (1536, 1536, 69)
LABELS = {'without_new': 'No new imaging', 'availability_only': 'Imaging availability only',
          'radbert': 'Frozen RadBERT: CXR + CT', 'radgraph': 'RadGraph-XL facts: CXR + CT'}


class ImagingEncoder(FactsEncoder):
    def __init__(self, text_width, width=128):
        nn.Module.__init__(self)
        self.fact_branches = nn.ModuleList([nn.Sequential(nn.Linear(v, 32), nn.GELU(), nn.Linear(32, 64), nn.LayerNorm(64)) for v in WIDTHS])
        self.fusion = nn.Sequential(nn.Linear(width + 3 * 64 + 3, width), nn.GELU(), nn.Dropout(.1), nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))
        self.core = CompactClinicalEncoder(text_width, width=width)

    def forward(self, batch, *, mask_pmh=False):
        if not isinstance(batch, FactsBatch): raise TypeError('explicit FactsBatch required')
        n = len(batch.compact.clinical.current_values)
        if batch.present.shape != (n, 3) or batch.present.dtype != torch.bool: raise ValueError('invalid source mask')
        reps = []
        for j, (value, width, branch) in enumerate(zip(batch.values, WIDTHS, self.fact_branches, strict=True)):
            if value.shape != (n, width) or not torch.isfinite(value).all(): raise ValueError('invalid imaging tensor')
            reps.append(branch(value) * batch.present[:, j, None])
        core, branches = self.core(batch.compact, mask_pmh=mask_pmh)
        return self.fusion(torch.cat([core, *reps, batch.present.to(core.dtype)], 1)), dict(branches, cxr=reps[0], ct=reps[1], discharge=reps[2])


class RadiologyAnchoredModel(AnchoredModel):
    def __init__(self, baseline, text_width, width=128):
        nn.Module.__init__(self)
        self.baseline = baseline.requires_grad_(False).eval()
        self.delta_head = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 4))
        nn.init.zeros_(self.delta_head[-1].weight); nn.init.zeros_(self.delta_head[-1].bias)
        self.encoder = ImagingEncoder(text_width, width); self.patient_shape = True


def model_for(base, width):
    torch.manual_seed(42)
    return RadiologyAnchoredModel(copy.deepcopy(base), width).cuda()


def load_sources(compact, radbert, radgraph):
    arrays = {}; provenance = {}; source_manifests = []
    for name, directory in (('radbert', radbert), ('radgraph', radgraph)):
        a, digest = checked(directory, 'imaging.npz', 'full_radiology_processed', compact)
        n = len(a['hadm_ids'])
        if a['values'].shape != (n, 2, 1536) or a['present'].shape != (n, 2): raise ValueError('imaging shape differs')
        if not np.isfinite(a['values']).all() or (a['values'][~a['present']] != 0).any(): raise ValueError('invalid imaging mask')
        arrays[name] = a; provenance[name] = digest
        m = json.loads((directory / 'manifest.json').read_text())
        if m['kind'] != name: raise ValueError('radiology model kind differs')
        source_manifests.append(m['source_manifest_sha256'])
    if len(set(source_manifests)) != 1: raise ValueError('radiology report source differs')
    if not np.array_equal(arrays['radbert']['present'], arrays['radgraph']['present']): raise ValueError('source populations differ')
    for directory in (radbert, radgraph):
        m = json.loads((directory / 'manifest.json').read_text())
        if m['reports'] != 14464 or not m['encoder_frozen']: raise ValueError('not full frozen imaging')
    return arrays, provenance


def attached(compact, sources, arm):
    a = sources['radgraph' if arm == 'radgraph' else 'radbert']; n = len(a['hadm_ids'])
    result = {k: a[k] for k in ('hadm_ids', 'subject_ids', 'patient_splits')}
    for j, key in enumerate(('cxr', 'ct')):
        result[key + '_values'] = a['values'][:, j] if arm in ('radbert', 'radgraph') else np.zeros((n, 1536), np.float32)
        result[key + '_present'] = a['present'][:, j] if arm != 'without_new' else np.zeros(n, bool)
    result['discharge_values'] = np.zeros((n, 69), np.float32); result['discharge_present'] = np.zeros(n, bool)
    return FactsData(compact, result, {}, ('cxr', 'ct'))


def prepared(data, ix):
    result = []
    for start in range(0, len(ix), 512):
        pos = ix[start:start + 512]; h, b, t, e = data.collate(pos, device='cuda')
        result.append((pos, h, b, t, e))
    return result


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2); started = time.perf_counter()
    data, compact, base, checkpoint = inputs(args.reference, args.compact_dir)
    sources, provenance = load_sources(compact, args.radbert, args.radgraph)
    ix = {f: np.flatnonzero(data.arrays['patient_splits'] == f) for f in ('train', 'validation', 'test')}
    args.output.mkdir(parents=True)
    write_json(args.output / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    contract = dict(seed=42, epochs=args.epochs, arms=ARMS, bootstrap=args.bootstrap, baseline_frozen=True,
                    zero_initialized_delta=True, batch_size=512, lr=3e-4, warmup=5, patience=10,
                    baseline_checkpoint_sha256=sha256(checkpoint), data_provenance=data.provenance,
                    compact_provenance=compact.provenance, facts_provenance=provenance, image_widths=WIDTHS,
                    text='Same existing four-field BERT; no fine-tuning', selection='Validation readmission integrated Brier 7-365',
                    endpoint='strict 491.21/J44.1, joint first readmission or death, World A, 365 days',
                    code_sha256={n: sha256(Path(__file__).parent / n) for n in ('radiology_experiment.py', 'train.py', 'risk.py', 'evaluation.py', 'compact_model.py')},
                    limits=['Single seed; historical test reused; no multiplicity correction', 'Representation comparison, not diagnostic gold-standard accuracy',
                            'Patient bootstrap does not include training variance', 'CXR/CT report availability differs; subgroup results are exploratory'])
    write_json(args.output / 'contract.json', contract); fitted = {}; initial = None
    for arm in ARMS:
        write_json(args.output / 'status.json', dict(status='training', arm=arm))
        model = model_for(base, data.text_vectors.shape[1]); state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if initial is None: initial = state
        elif not all(torch.equal(v, initial[k]) for k, v in state.items()): raise ValueError('initialization differs')
        source = attached(compact, sources, arm)
        fitted[arm] = fit(model, data, prepared(source, ix['train']), prepared(source, ix['validation']), ix['validation'], args.output / arm,
                          epochs=args.epochs, lr=3e-4, seed=42, patience=10, warmup=5)
        del model; torch.cuda.empty_cache()
    write_json(args.output / 'status.json', {'status': 'all_models_selected_before_test'})
    results = {}; predictions = {}
    for arm in ARMS:
        model = model_for(base, data.text_vectors.shape[1]); model.load_state_dict(torch.load(args.output / arm / 'best.pt', map_location='cuda', weights_only=True)['state_dict'])
        if not all(torch.equal(v, base.state_dict()[k]) for k, v in model.baseline.state_dict().items()): raise ValueError('history changed')
        source = attached(compact, sources, arm); results[arm] = {k: v for k, v in fitted[arm].items() if k != 'history'}; results[arm]['folds'] = {}
        for fold in ('validation', 'test'):
            curve, detail = evaluate(data, ix[fold], model, prepared(source, ix[fold]), args.output / arm, fold)
            results[arm]['folds'][fold] = detail; predictions[arm, fold] = curve
            for subgroup, mask in [('with_any_imaging', sources['radbert']['present'].any(1)), ('with_ct', sources['radbert']['present'][:, 1])]:
                subset = mask[ix[fold]]
                results[arm]['folds'][fold][subgroup] = public_metrics(metrics(data.times[ix[fold]][subset], data.events[ix[fold]][subset], curve[subset, :, 1], ordering=True))
    comparisons = {}
    for fold in ('validation', 'test'):
        selected = ix[fold]; comparisons[fold] = {}
        for left, right in [(a, 'without_new') for a in ARMS[1:]] + [('radbert', 'availability_only'), ('radgraph', 'availability_only'), ('radgraph', 'radbert')]:
            delta = paired_interval(data.arrays['subject_ids'][selected], data.times[selected], data.events[selected], predictions[left, fold][:, :, 1], predictions[right, fold][:, :, 1], replicates=args.bootstrap, seed=42)
            delta['mean_auc_difference'] = results[left]['folds'][fold]['readmission']['mean_auc_7_365'] - results[right]['folds'][fold]['readmission']['mean_auc_7_365']
            comparisons[fold][left + '_minus_' + right] = delta
    report = dict(status='completed_full_radiology_paired', initial_weights_identical=True, contract=contract,
                  models=results, comparisons=comparisons, elapsed_seconds=time.perf_counter() - started)
    write_json(args.output / 'report.json', report)
    facts_experiment.LABELS.update(LABELS); facts_experiment.render(args.output, report)
    write_json(args.output / 'status.json', {'status': report['status']})
    print(json.dumps({'status': report['status'], 'seconds': report['elapsed_seconds']}), flush=True)


def verify(args):
    r = json.loads((args.output / 'report.json').read_text()); c = r['contract']
    config = json.loads((args.output / 'config.json').read_text())
    for name, digest in c['code_sha256'].items():
        if sha256(Path(__file__).parent / name) != digest: raise ValueError('experiment code changed')
    data, compact, base, checkpoint = inputs(Path(config['reference']), Path(config['compact_dir']))
    sources, provenance = load_sources(compact, Path(config['radbert']), Path(config['radgraph']))
    if provenance != c['facts_provenance'] or sha256(checkpoint) != c['baseline_checkpoint_sha256']: raise ValueError('input changed')
    errors = {}
    for arm in ARMS:
        path = args.output / arm / 'best.pt'
        if sha256(path) != r['models'][arm]['checkpoint_sha256']: raise ValueError('checkpoint changed')
        model = model_for(base, data.text_vectors.shape[1]); model.load_state_dict(torch.load(path, map_location='cuda', weights_only=True)['state_dict'])
        if not all(torch.equal(v, base.state_dict()[k]) for k, v in model.baseline.state_dict().items()): raise ValueError('history changed')
        source = attached(compact, sources, arm)
        for fold in ('validation', 'test'):
            selected = np.flatnonzero(data.arrays['patient_splits'] == fold)
            a, k, _, delta = predict(model, prepared(source, selected)); curve = predict_cif(a, k, GRID)
            with np.load(args.output / arm / (fold + '_predictions.npz')) as saved:
                if not np.array_equal(saved['hadm_ids'], data.arrays['hadm_ids'][selected]) or not np.array_equal(saved['events'], data.events[selected]): raise ValueError('members or labels changed')
                if not np.allclose(curve, saved['curve'], atol=2e-6, rtol=1e-5): raise ValueError('prediction differs')
                errors[arm + '/' + fold] = float(np.max(np.abs(curve - saved['curve'])))
            actual = metrics(data.times[selected], data.events[selected], curve[:, :, 1])
            if any(abs(actual[x] - r['models'][arm]['folds'][fold]['readmission'][x]) > 1e-7 for x in ('integrated_brier_7_365', 'mean_auc_7_365')): raise ValueError('metrics differ')
    write_json(args.output / 'independent_verification.json', dict(status='all_arms_gpu_reloaded', history_unchanged=True, max_curve_errors=errors))
    print('all_arms_gpu_reloaded', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--compact-dir', type=Path, default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--radbert', type=Path, default=Path('patient_data/processed/v6_radbert_imaging_2026-10-10_r2'))
    p.add_argument('--radgraph', type=Path, default=Path('patient_data/processed/v6_radgraph_imaging_source_checked_2026-10-10'))
    p.add_argument('--output', type=Path, required=True); p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--bootstrap', type=int, default=200); p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    verify(args) if args.verify else run(args)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/radiology_plot.py · lines 1–47

<!-- REVIEW_FRAGMENT {"path":"training/v6/radiology_plot.py","part":1,"start_line":1,"end_line":47,"source_sha256":"b83b5cf83e96feeae4c2f433843f0ffce02365fd74cd67897e9d114016ee9168","fragment_sha256":"b83b5cf83e96feeae4c2f433843f0ffce02365fd74cd67897e9d114016ee9168","added_terminal_newline":false} -->
````python
"""只读完成报告，绘制有患者聚类区间的影像配对结果；不训练或选模。"""
import argparse
import json
from pathlib import Path


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    report = json.loads((root / 'report.json').read_text())
    if report['status'] != 'completed_full_radiology_paired':
        raise ValueError('completed comparison required')
    names = ('availability_only', 'radbert', 'radgraph')
    labels = ('Imaging availability only', 'Frozen RadBERT: CXR + CT', 'RadGraph-XL facts: CXR + CT')
    for fold in ('validation', 'test'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
        for i, name in enumerate(names):
            result = report['comparisons'][fold][name + '_minus_without_new']
            for ax, key, ci in zip(axes, ('integrated_brier_difference', 'auc90_difference'),
                                  ('integrated_brier_95ci', 'auc90_95ci'), strict=True):
                lo, hi = result[ci]
                ax.plot([lo, hi], [i, i], color='C' + str(i), linewidth=2)
                ax.scatter(result[key], i, marker='D', color='C' + str(i), zorder=3)
        for ax in axes:
            ax.axvline(0, color='gray', linewidth=1)
            ax.set_yticks(range(3), labels)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_powerlimits((-3, 3))
            ax.xaxis.set_major_formatter(formatter)
            ax.grid(alpha=.2)
            ax.set_xlabel('Candidate - no new imaging; patient bootstrap 95% CI')
            ax.set_ylim(-.5, 2.5)
        axes[0].set_title('Integrated Brier, days 7-365 (lower better)')
        axes[1].set_title('90-day AUC(t) (higher better)')
        fig.suptitle(f'{fold}: one training seed; 200 patient-cluster resamples; no multiplicity adjustment')
        fig.tight_layout()
        fig.savefig(root / 'figures' / (fold + '_paired_ci.png'), dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    render(parser.parse_args().output)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/reconstruction_controls.py · lines 1–31

<!-- REVIEW_FRAGMENT {"path":"training/v6/reconstruction_controls.py","part":1,"start_line":1,"end_line":31,"source_sha256":"72a1374431bc8a920d6767c72906d4743d449c70ee80d53aa7874d4069601bfe","fragment_sha256":"72a1374431bc8a920d6767c72906d4743d449c70ee80d53aa7874d4069601bfe","added_terminal_newline":false} -->
````python
"""补充真正的训练折每概念均值重构对照；不伪称归一化零值就是均值。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.v6.token_data import load_inputs,prepare
from training.v6.token_model import masked_reconstruction_batch
from training.v6.train import write_json


def run(root):
    cfg=json.loads((root/'config.json').read_text());d,c,h,ext,med,_=load_inputs(*(Path(cfg[k]) for k in ('reference','compact_dir','extension','medication')))
    cache=prepare(d,c,ext,med,np.flatnonzero(d.arrays['patient_splits']=='train'),device='cuda')
    total=torch.zeros(29,20,device='cuda');count=torch.zeros(29,20,device='cuda')
    for _,_,b,_,_ in cache:total+=(b.labs*b.numeric_valid).sum(0);count+=b.numeric_valid.sum(0)
    means=total/count.clamp_min(1);results={}
    for seed in cfg['seeds']:
        mean_loss=0.;zero_loss=0.;targets=0
        for j,(_,_,b,_,_) in enumerate(cache):
            _,mask=masked_reconstruction_batch(b,seed+cfg['ssl_epochs']*10000+j)
            mean_loss+=float(((b.labs-means).square()[mask]).sum());zero_loss+=float(b.labs.square()[mask].sum());targets+=int(mask.sum())
        results[str(seed)]=dict(mean_predictor_mse=mean_loss/targets,zero_normalized_predictor_mse=zero_loss/targets,targets=targets,
            diagnostic='training fold; same CUDA masking seeds and batches as pretraining epoch20')
    write_json(root/'self_supervised_controls.json',dict(status='completed_training_mean_reconstruction_control',mean_fit='observed numeric channels in training fold only',
        results=results,limits=['Do not infer readmission gain from reconstruction gain.','No validation/test examples used.']))
    print(json.dumps(results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'));torch.set_num_threads(2);run(p.parse_args().root)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/regimen_joint_experiment.py · lines 1–334

<!-- REVIEW_FRAGMENT {"path":"training/v6/regimen_joint_experiment.py","part":1,"start_line":1,"end_line":334,"source_sha256":"fe7083d958a961a703a394aa2b309ccd3bedeca243232e07f89eccf24cfad720","fragment_sha256":"fe7083d958a961a703a394aa2b309ccd3bedeca243232e07f89eccf24cfad720","added_terminal_newline":false} -->
````python
"""单seed融合前联合学习；原完整V6仅作重载参照，不重建病历。"""
import argparse
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from patient_data.export_mimiciv import sha256
from training.v6.discharge_regimen_screen import setup as parent_setup, ROOT as PARENT
from training.v6.discharge_regimen_cox import probability_readout
from training.v6.regularization_audit import REFERENCE
from training.v6.module_surgery import SurgeryModel, probability_curve
from training.v6.module_surgery_experiment import batches, parameters, summary, save
from training.v6.regimen_joint_fusion import JointFusion, RegimenBatch, ARMS
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import competing_nll
from training.v6.train import write_json

ROOT = Path('training_runs/v6_regimen_joint_fusion_2026-10-14')
PROTOCOL = Path('docs/V6_REGIMEN_JOINT_FUSION_PROTOCOL_2026-10-14.md')


def fingerprint(value):
    h = hashlib.sha256()
    def update(x):
        if isinstance(x, torch.Tensor):
            update(x.detach().cpu().numpy())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype.str, x.shape)).encode()); h.update(x.tobytes())
        elif is_dataclass(x):
            for f in fields(x):
                h.update(f.name.encode()); update(getattr(x, f.name))
        elif isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode()); update(x[k])
        elif isinstance(x, (tuple, list)):
            for y in x:
                update(y)
        else:
            h.update(repr(x).encode())
    update(value)
    return h.hexdigest()


def setup():
    data, train, tune, arrays, parent = parent_setup()
    if parent != json.loads((PARENT/'contract.json').read_text()):
        raise ValueError('verified parent contract differs')
    proof = json.loads((PARENT/'verification.json').read_text())
    if proof['maximum_error'] != 0 or proof['screening_sha256'] != sha256(PARENT/'screening.json'):
        raise ValueError('parent proof stale')
    sources = dict(parent['source_sha256'])
    for p in (Path(__file__), Path('training/v6/regimen_joint_fusion.py'), PROTOCOL,
              Path('training/tests/test_v6_regimen_joint_fusion.py'),
              Path('training/v6/discharge_regimen_cox.py'), PARENT/'verification.json',
              PARENT/'contract.json', PARENT/'screening.json'):
        sources[str(p)] = sha256(p)
    contract = dict(source_sha256=sources, interface_sha256=parent['data_sha256'],
                    teacher_sha256=parent['teacher_sha256'], base_sha256=parent['base_sha256'],
                    label_sha256=parent['label_sha256'], data_provenance=data[0].provenance,
                    seed=42, arms=list(ARMS), new_parameters=4480,
                    optimizer=dict(name='AdamW', lr=2e-5, weight_decay=1e-4,
                                   delta_penalty=1e-4, clip=1.),
                    early=dict(min_epochs=10, patience=7, min_delta=1e-5, max_epochs=45),
                    selection='specified tuning readmission IBS7-365 including epoch0',
                    no_high_frequency=True, confirmation_test_used=False,
                    clinical_trainable=True, bert_backbone_loaded=False,
                    original_inputs_unchanged=True, source_missing_new_injection_zero=True,
                    source_missing_final_risk_not_constrained=True)
    return data, train, tune, arrays, contract


def teacher_for(data):
    m = SurgeryModel(data[4], 42, 'control', data[5]).cuda()
    m.load_state_dict(torch.load(REFERENCE/'seed42/control/best.pt',
                                 map_location='cuda', weights_only=True)['state_dict'])
    return m.eval()


def wrap(cache, arrays):
    for _, h, b, _, _ in cache:
        if (h.shape[1] != 34 or b.base.base.clinical.current_masks.any()
                or b.base.medications.ne(0).any()
                or b.base.base.clinical.snippet_vectors.shape[1] != 768):
            raise ValueError('inherited H34/current/detail/text view changed')
    return [(pos, h, RegimenBatch(b, torch.as_tensor(arrays['features'][pos], device='cuda'),
                                  torch.as_tensor(arrays['section_available'][pos], device='cuda')),
             t, e) for pos, h, b, t, e in cache]


@torch.no_grad()
def predict(model, cache):
    a, k, nll = parameters(model, cache)
    return a, k, probability_curve(a, k), nll


def trainable(model):
    return {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}


def baseline_equal(model, teacher):
    return all(torch.equal(v, teacher.baseline.state_dict()[k]) for k, v in model.baseline.state_dict().items())


def preflight(teacher, original, new):
    old_a, old_k, old_curve, _ = predict(teacher, original)
    proof = {}
    for arm in ARMS:
        m = JointFusion(teacher, arm).cuda().eval()
        a, k, curve, _ = predict(m, new)
        for x, y in ((a, old_a), (k, old_k), (curve, old_curve)):
            np.testing.assert_array_equal(x, y)
        _, h, b, t, e = new[0]
        before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-5, weight_decay=1e-4)
        loss = competing_nll(m(h, b), t, e); loss.backward()
        if any(p.grad is not None for p in m.baseline.parameters()):
            raise ValueError('H34 received gradients')
        opt.step()
        groups = ('teacher.encoder.base.lab.', 'teacher.encoder.base.med.',
                  'teacher.encoder.base.microbiology.', 'teacher.encoder.base.support.',
                  'teacher.encoder.base.text_projection.', 'teacher.encoder.base.fusion.',
                  'teacher.encoder.extra_labs.', 'teacher.delta_head.', 'regimen.')
        updates = {prefix: any(not torch.equal(before[n], p) for n, p in m.named_parameters()
                               if n.startswith(prefix) and n in before) for prefix in groups}
        if not all(updates.values()) or not baseline_equal(m, teacher):
            raise ValueError('effective clinical/injection update or frozen H34 failed: '+repr(updates))
        m.eval()
        with torch.no_grad():
            m.regimen.weight.fill_(.03)
            x = torch.zeros_like(b.features)
            z = torch.zeros_like(b.available)
            if torch.count_nonzero(m.injection(torch.ones_like(x), z)):
                raise ValueError('missing new injection not zero')
            # 前一个batch内容不能留在网络中。
            first = m(h, b).log_amplitude.clone()
            m.injection(torch.ones_like(x), z)
            torch.testing.assert_close(m(h, b).log_amplitude, first, rtol=0, atol=0)
            one = m.injection(x, b.available); two = m.injection(x+1, b.available)
            if arm == 'availability':
                torch.testing.assert_close(one, two, rtol=0, atol=0)
            elif not (one != two).any():
                raise ValueError('content insensitive under nonzero projection')
        proof[arm] = dict(initial_parameter_daily_risk_error=0., real_step_updates=updates,
                          h34_no_gradient_and_unchanged=True, new_missing_injection_zero=True,
                          trainable_parameters=sum(trainable(m).values()))
    return dict(status='passed_real_gpu', arms=proof, gpu=torch.cuda.get_device_name())


def fit(teacher, arm, train_cache, tune_cache, d, tune, folder):
    folder.mkdir()
    model = JointFusion(teacher, arm).cuda()
    torch.manual_seed(42)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=1e-4)
    early = EarlyStopping(7, 10, 1e-5); rng = np.random.default_rng(42)
    history = []; reason = 'max_epochs'; steps = 0
    for epoch in range(46):
        online = None
        if epoch:
            model.train(); total = 0.; n = 0
            for j in rng.permutation(len(train_cache)):
                _, h, b, t, e = train_cache[j]
                opt.zero_grad(set_to_none=True); out = model(h, b)
                nll = competing_nll(out, t, e); loss = nll + 1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                if not torch.isfinite(norm):
                    raise ValueError('nonfinite gradient norm')
                opt.step(); steps += 1
                total += float(nll.detach())*len(t); n += len(t)
            online = total/n
        a, k, curve, nll = predict(model, tune_cache)
        score = summary(curve, d, tune, nll)['readmission']['integrated_brier_7_365']
        best, stop = early.observe(epoch, score)
        history.append(dict(epoch=epoch, tuning_ibs=score, train_nll=online))
        if best:
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch), folder/'best.pt')
        write_json(folder/'history.json', history)
        if epoch % 5 == 0:
            print(json.dumps(dict(arm=arm, epoch=epoch, best=early.best_epoch, tuning_ibs=score)), flush=True)
        if stop:
            reason = 'patience_exhausted'; break
    if not baseline_equal(model, teacher):
        raise ValueError('H34 changed during actual training')
    torch.save(dict(state_dict=model.state_dict(), epoch=history[-1]['epoch']), folder/'last.pt')
    info = dict(completed_epochs=history[-1]['epoch'], best_epoch=early.best_epoch,
                early=asdict(early), stop_reason=reason, optimizer_steps=steps,
                last_checkpoint_sha256=sha256(folder/'last.pt'),
                trainable_parameters=trainable(model), h34_unchanged=True,
                last_injection_norm=float(model.regimen.weight.detach().norm()))
    model.load_state_dict(torch.load(folder/'best.pt', map_location='cuda', weights_only=True)['state_dict'])
    a, k, curve, nll = predict(model, tune_cache)
    save(folder/'tuning_predictions.npz', d, tune, a, k, curve)
    info.update(metrics=summary(curve, d, tune, nll),
                probability_readout=probability_readout(d.times[tune], d.events[tune], curve),
                checkpoint_sha256=sha256(folder/'best.pt'))
    write_json(folder/'training_result.json', info)
    return info


def lock(report):
    scores = {'original': report['reference']['metrics']['readmission']['integrated_brier_7_365']}
    scores.update({a: report['fits'][a]['metrics']['readmission']['integrated_brier_7_365'] for a in ARMS})
    selected = min(scores, key=scores.get)
    content = report['fits']['content']['metrics']
    refs = (report['reference']['metrics'], report['fits']['availability']['metrics'])
    material = selected == 'content' and all(
        content['readmission']['integrated_brier_7_365'] <= .99*r['readmission']['integrated_brier_7_365']
        and content['readmission']['points']['90']['auc'] >= r['readmission']['points']['90']['auc']+.005
        and content['death']['integrated_brier_7_365'] <= r['death']['integrated_brier_7_365'] for r in refs)
    return dict(scores=scores, selected=selected, nominated=['content'] if selected == 'content' else [],
                material_signal=material, calibration_passed=False, default_changed=False, goal_achieved=False)


def prepare():
    data, train, tune, arrays, contract = setup()
    d, c, _, ext = data[:4]
    tc = batches(d, c, ext, train); vc = batches(d, c, ext, tune)
    new_train, new_tune = wrap(tc, arrays), wrap(vc, arrays)
    teacher = teacher_for(data)
    contract['input_sha256'] = dict(train=fingerprint(tc), tune=fingerprint(vc),
                                  new_train=fingerprint(new_train), new_tune=fingerprint(new_tune))
    contract['trainable_white_list'] = trainable(JointFusion(teacher, 'availability'))
    return d, train, tune, contract, teacher, tc, vc, new_train, new_tune


def run():
    if ROOT.exists():
        raise FileExistsError('batch exists; do not repeat fitting')
    started = time.perf_counter()
    d, train, tune, contract, teacher, tc, vc, nt, nv = prepare()
    ROOT.mkdir(); write_json(ROOT/'contract.json', contract)
    write_json(ROOT/'preflight.json', preflight(teacher, vc, nv))
    ref = {}
    for name, cache, ix in (('train', tc, train), ('tune', vc, tune)):
        a, k, curve, nll = predict(teacher, cache)
        folder = ROOT/'original'; folder.mkdir(exist_ok=True)
        save(folder/(name+'_predictions.npz'), d, ix, a, k, curve)
        ref[name] = dict(prediction_sha256=sha256(folder/(name+'_predictions.npz')))
        if name == 'tune':
            ref.update(metrics=summary(curve, d, ix, nll),
                       probability_readout=probability_readout(d.times[ix], d.events[ix], curve))
    parent = json.loads((PARENT/'screening.json').read_text())
    if ref['metrics'] != parent['reference']:
        raise ValueError('original full-input reference changed')
    report = dict(reference=ref, fits={}, single_seed_only=True)
    for arm in ARMS:
        write_json(ROOT/'state.json', dict(status='training', arm=arm, goal_achieved=False))
        report['fits'][arm] = fit(teacher, arm, nt, nv, d, tune, ROOT/arm)
    report['lock'] = lock(report); report['actual_seconds'] = time.perf_counter()-started
    write_json(ROOT/'screening.json', report); write_json(ROOT/'nomination_lock.json', report['lock'])
    write_json(ROOT/'state.json', dict(status='complete_pending_independent_replay', goal_achieved=False))
    print(json.dumps(report['lock']), flush=True)


def check_prediction(path, d, ix, a, k, curve):
    with np.load(path) as z:
        for key, value in (('hadm_ids', d.arrays['hadm_ids'][ix]), ('subject_ids', d.arrays['subject_ids'][ix]),
                           ('times', d.times[ix]), ('events', d.events[ix]),
                           ('log_amplitude', a), ('shape', k), ('curve', curve)):
            np.testing.assert_array_equal(z[key], value)
    if (not np.isfinite(curve).all() or np.any(curve < 0) or np.any(curve > 1)
            or np.max(np.abs(curve.sum(2)-1)) > 1e-9 or np.min(np.diff(curve[:, :, 1:], axis=1)) < -1e-10):
        raise ValueError('invalid daily joint probabilities')


def verify():
    d, train, tune, contract, teacher, tc, vc, nt, nv = prepare()
    if contract != json.loads((ROOT/'contract.json').read_text()):
        raise ValueError('source/input/white-list contract changed')
    report = json.loads((ROOT/'screening.json').read_text())
    for name, cache, ix in (('train', tc, train), ('tune', vc, tune)):
        a, k, curve, nll = predict(teacher, cache)
        path = ROOT/'original'/(name+'_predictions.npz')
        if sha256(path) != report['reference'][name]['prediction_sha256']:
            raise ValueError('teacher cache hash differs')
        check_prediction(path, d, ix, a, k, curve)
        if name == 'tune' and (summary(curve, d, ix, nll) != report['reference']['metrics']
                or probability_readout(d.times[ix], d.events[ix], curve) != report['reference']['probability_readout']):
            raise ValueError('teacher metrics/calibration differ')
    for arm in ARMS:
        folder = ROOT/arm; info = report['fits'][arm]
        history = json.loads((folder/'history.json').read_text())
        early = EarlyStopping(7, 10, 1e-5); first_stop = None
        for row in history:
            _, stop = early.observe(row['epoch'], row['tuning_ibs'])
            if stop and first_stop is None:
                first_stop = row['epoch']
        if (asdict(early) != info['early'] or len(history)-1 != info['completed_epochs']
                or info['optimizer_steps'] != len(tc)*info['completed_epochs']
                or (info['stop_reason'] == 'patience_exhausted' and first_stop != info['completed_epochs'])
                or (info['stop_reason'] == 'max_epochs' and (info['completed_epochs'] != 45 or first_stop is not None))):
            raise ValueError('early-stop/actual training differs')
        model = JointFusion(teacher, arm).cuda()
        for filename, hashkey in (('last.pt', 'last_checkpoint_sha256'), ('best.pt', 'checkpoint_sha256')):
            if sha256(folder/filename) != info[hashkey]:
                raise ValueError('checkpoint hash changed')
            saved = torch.load(folder/filename, map_location='cuda', weights_only=True)
            model.load_state_dict(saved['state_dict'])
            if not baseline_equal(model, teacher):
                raise ValueError('H34 differs')
            if filename == 'last.pt':
                if saved['epoch'] != info['completed_epochs'] or float(model.regimen.weight.detach().norm()) != info['last_injection_norm']:
                    raise ValueError('actual last training differs')
            elif saved['epoch'] != early.best_epoch:
                raise ValueError('checkpoint selection differs')
        a, k, curve, nll = predict(model, nv)
        check_prediction(folder/'tuning_predictions.npz', d, tune, a, k, curve)
        if (summary(curve, d, tune, nll) != info['metrics']
                or probability_readout(d.times[tune], d.events[tune], curve) != info['probability_readout']):
            raise ValueError('candidate metrics/calibration differ')
    if lock(report) != report['lock'] or report['lock'] != json.loads((ROOT/'nomination_lock.json').read_text()):
        raise ValueError('nomination changed')
    write_json(ROOT/'verification.json', dict(status='independent_full_input_teacher_weights_daily_calibration_early_lock_replay',
               maximum_error=0., no_refitting=True, screening_sha256=sha256(ROOT/'screening.json')))
    write_json(ROOT/'state.json', dict(status='single_seed_independently_verified', goal_achieved=False))
    print(json.dumps(dict(maximum_error=0., no_refitting=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', required=True, choices=('run', 'verify'))
    mode = parser.parse_args().mode
    try:
        run() if mode == 'run' else verify()
    except Exception as error:
        if ROOT.exists():
            write_json(ROOT/('failure_'+mode+'.json'), dict(error=repr(error)))
        raise
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/regimen_joint_fusion.py · lines 1–112

<!-- REVIEW_FRAGMENT {"path":"training/v6/regimen_joint_fusion.py","part":1,"start_line":1,"end_line":112,"source_sha256":"805c66f9de36e602a8d446a5e1d60ed401d6b06c79dd886e7106aacd62f8337d","fragment_sha256":"805c66f9de36e602a8d446a5e1d60ed401d6b06c79dd886e7106aacd62f8337d","added_terminal_newline":false} -->
````python
"""显式batch上下文的融合前方案注入；原有效临床层联合学习。"""
import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from training.v6.model import masked_pool
from training.v6.risk import RiskOutput

ARMS = ('availability', 'content')
ACTIVE_PREFIXES = (
    'encoder.base.lab.', 'encoder.base.lab_ids.', 'encoder.base.lab_score.',
    'encoder.base.med.', 'encoder.base.microbiology.', 'encoder.base.support.',
    *(f'encoder.base.branch_norms.{i}.' for i in range(1, 5)),
    'encoder.base.field_ids.', 'encoder.base.text_score.',
    'encoder.base.text_projection.', 'encoder.base.fusion.',
    'encoder.extra_labs.', 'delta_head.',
)


@dataclass(frozen=True)
class RegimenBatch:
    original: object
    features: torch.Tensor
    available: torch.Tensor


class JointFusion(nn.Module):
    def __init__(self, teacher, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('unknown joint regimen arm')
        self.arm = arm
        self.teacher = copy.deepcopy(teacher)
        for name, p in self.teacher.named_parameters():
            p.requires_grad_(name.startswith(ACTIVE_PREFIXES))
        # 不让新模块初始化耗用训练dropout的全局随机流。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.regimen = nn.Linear(35, 128, bias=False)
        nn.init.zeros_(self.regimen.weight)

    @property
    def baseline(self):
        return self.teacher.baseline

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        return self

    def injection(self, features, available):
        n = len(features)
        if (features.shape != (n, 34) or available.shape != (n,)
                or available.dtype != torch.bool or not torch.isfinite(features).all()):
            raise ValueError('explicit regimen34 and boolean chapter mask required')
        x = features if self.arm == 'content' else torch.zeros_like(features)
        x = torch.cat((x, available[:, None].to(x.dtype)), 1)
        return self.regimen(x) * available[:, None]

    def clinical(self, batch, injected):
        """沿用原compact的分支顺序和算子；只改第一仿射后的一次加法。"""
        detail = self.teacher.encoder
        core = detail.base
        compact = batch.base
        b = compact.clinical
        n = len(b.current_values)
        current = torch.where(b.current_masks, b.current_values, 0.)
        observed = b.lab_axis_inputs[:, :, 6] > 0
        labs = torch.where(observed[:, :, None], b.lab_axis_inputs, 0.)
        s = masked_pool(core.current(torch.stack((current, b.current_masks.to(current.dtype)), -1))
                        + core.current_ids(torch.arange(14, device=current.device)),
                        b.current_masks, core.current_score)
        l = masked_pool(core.lab(labs) + core.lab_ids(torch.arange(22, device=labs.device)),
                        observed, core.lab_score)
        med = compact.med_features
        med_present = med[:, :, 2:].bool().any(dim=(1, 2))
        m = core.med(med.flatten(1))
        extra, extra_present = [], []
        for values, masks, module in (
                (compact.micro_values, compact.micro_masks, core.microbiology),
                (compact.support_values, compact.support_masks, core.support)):
            extra.append(module(torch.cat((values, masks.to(values.dtype)), 1)))
            extra_present.append(masks.any(1))
        present = [b.current_masks.any(1), observed.any(1), med_present, *extra_present]
        branches = [norm(x) * p[:, None] for norm, x, p in
                    zip(core.branch_norms, (s, l, m, *extra), present, strict=True)]
        text, text_present = core._text(b, n, mask_pmh=False)
        presence = torch.stack((*present, text_present), 1).to(s.dtype)
        x = torch.cat((*branches, text, presence), 1)
        h = core.fusion[0](x) + injected
        for layer in list(core.fusion)[1:]:
            h = layer(h)
        additions = []
        for values, width, module in ((batch.labs, 90, detail.extra_labs),
                                      (batch.medications, 192, detail.extra_meds)):
            if values.shape != (n, width) or not torch.isfinite(values).all():
                raise ValueError('invalid inherited detail batch')
            additions.append(module(values) * values.ne(0).any(1)[:, None])
        return h + additions[0] + additions[1]

    def forward(self, history, batch):
        if not isinstance(batch, RegimenBatch):
            raise TypeError('RegimenBatch required; no implicit context')
        with torch.no_grad():
            base = self.baseline(history)
        hidden = self.clinical(batch.original.base, self.injection(batch.features, batch.available))
        delta = self.teacher.delta_head(hidden).reshape(-1, 2, 2)
        return RiskOutput((base.log_amplitude + delta[:, :, 0]).clamp(-10., 6.),
                          F.softplus(self.baseline.raw_shape.detach() + delta[:, :, 1]) + .1, delta)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/regimen_joint_report.py · lines 1–77

<!-- REVIEW_FRAGMENT {"path":"training/v6/regimen_joint_report.py","part":1,"start_line":1,"end_line":77,"source_sha256":"600ac3c751ae49c3b92cb162cf3e7d948a8bade7c45d26a5ec0b7528fa35c9e3","fragment_sha256":"600ac3c751ae49c3b92cb162cf3e7d948a8bade7c45d26a5ec0b7528fa35c9e3","added_terminal_newline":false} -->
````python
"""已锁定联合融合配对的描述读出与图，不重新拟合。"""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path('training_runs/v6_regimen_joint_fusion_2026-10-14')


def build():
    from patient_data.export_mimiciv import sha256
    from training.v6.regimen_joint_experiment import setup
    from training.v6.evaluation import metrics, public_metrics
    from training.v6.train import write_json
    proof = json.loads((ROOT/'verification.json').read_text())
    if proof['maximum_error'] != 0 or proof['screening_sha256'] != sha256(ROOT/'screening.json'):
        raise ValueError('independent replay not current')
    data, _, tune, arrays, _ = setup(); d = data[0]
    counts = np.rint(np.expm1(d.history_raw[tune, 2])).astype(int)
    masks = dict(first=counts == 0, one_two=(counts >= 1) & (counts <= 2), three_plus=counts >= 3,
                 chapter_available=arrays['section_available'][tune], chapter_missing=~arrays['section_available'][tune])
    with np.load(ROOT/'original/tune_predictions.npz') as z:
        ref = z['curve']
    report = dict(screening_sha256=sha256(ROOT/'screening.json'), layers={}, selected_max_risk_difference={},
                  identical_predictions_no_bootstrap=True, no_refitting=True,
                  limits='WorldA retrospective descriptive tuning calibration; no individual accuracy or new blind test')
    curves = {'original': ref}
    for arm in ('availability', 'content'):
        with np.load(ROOT/arm/'tuning_predictions.npz') as z:
            curves[arm] = z['curve']
        delta = float(np.abs(curves[arm]-ref).max())
        report['selected_max_risk_difference'][arm] = delta
        if delta != 0:
            raise ValueError('do not assume identical predictions for a changed batch')
    for group, mask in masks.items():
        ix = tune[mask]
        report['layers'][group] = dict(admissions=int(mask.sum()), patients=int(len(np.unique(d.arrays['subject_ids'][ix]))),
            models={name: {label: public_metrics(metrics(d.times[ix], d.events[ix], value[mask, :, cause], cause=cause))
                           for label, cause in (('readmission', 1), ('death', 2))} for name, value in curves.items()})
    write_json(ROOT/'descriptive_report.json', report)
    print(json.dumps(dict(maximum_selected_difference=report['selected_max_risk_difference'],
                         layer_admissions={k: v['admissions'] for k, v in report['layers'].items()})), flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads((ROOT/'screening.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ref = report['reference']['metrics']['readmission']['integrated_brier_7_365']
    for arm in ('availability', 'content'):
        rows = json.loads((ROOT/arm/'history.json').read_text())
        axes[0].plot([x['epoch'] for x in rows[1:]], [x['train_nll'] for x in rows[1:]], label=arm)
        axes[1].plot([x['epoch'] for x in rows], [x['tuning_ibs'] for x in rows], label=arm)
    axes[1].axhline(ref, color='black', linestyle='--', label='Original V6 / selected epoch 0')
    for ax, title, ylabel in zip(axes, ('Actual training: all clinical branches learn', 'Specified tuning: no gain over epoch 0'),
                                ('Online training NLL (lower better)', 'Readmission IBS7-365 (lower better)'), strict=True):
        ax.set(title=title, xlabel='Epoch', ylabel=ylabel); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Seed 42; H34 frozen; pre-GELU regimen injection; 10-epoch early stopping')
    fig.tight_layout(); fig.savefig(ROOT/'training.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for i, cause in enumerate(('readmission', 'death')):
        for j, day in enumerate(('30', '90', '365')):
            bins = report['reference']['probability_readout'][cause][day]['bins']
            ax = axes[i, j]
            ax.plot([b['predicted'] for b in bins], [b['observed'] for b in bins], 'o-', label='All three selected curves coincide')
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set(title=f'{cause.title()}: day {day}', xlabel='Mean predicted cumulative probability',
                   ylabel='Observed World-A fraction', xlim=(0, 1), ylim=(0, 1)); ax.grid(alpha=.2)
    fig.suptitle('Descriptive risk-bin calibration; not individual correctness or external validation')
    axes[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(ROOT/'calibration.png', dpi=140); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', required=True, choices=('build', 'plot'))
    build() if parser.parse_args().mode == 'build' else plot()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/regularization_audit.py · lines 1–149

<!-- REVIEW_FRAGMENT {"path":"training/v6/regularization_audit.py","part":1,"start_line":1,"end_line":149,"source_sha256":"c65f3e0a799e0ca798bba2f18630700ed6cac6b2888c2583e09fcb42a2f7e0c7","fragment_sha256":"c65f3e0a799e0ca798bba2f18630700ed6cac6b2888c2583e09fcb42a2f7e0c7","added_terminal_newline":false} -->
````python
"""正则搜索独立核验/锁定确认；无测试折加载或候选改选。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_model import TRIALS,make_model
from training.v6.regularization_search import ROOT,SEED,EPOCHS,validation_parts
from training.v6.module_surgery_experiment import inputs,batches,parameters,summary,save
from training.v6.module_surgery import probability_curve,SurgeryModel
from training.v6.horizon_objective import restore
from training.v6.deep_interaction_experiment import runtime_setup
from training.v6.evaluation import paired_interval
from training.v6.train import write_json

REFERENCE=Path('training_runs/v6_module_surgery_2026-10-12_r3')


def checked():
    runtime_setup()
    contract=json.loads((ROOT/'contract.json').read_text())
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('search source changed: '+p)
    # 补充完整旧依赖的既定哈希审计，不把检查时的新哈希冒充训练快照。
    prior=json.loads((REFERENCE/'contract.json').read_text())
    for p,digest in prior['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('official V6 dependency changed: '+p)
    d,c,h,ext,config,_,_=inputs()
    if (d.provenance!=contract['data_provenance'] or c.provenance!=contract['compact_provenance'] or
            config!=contract['input_config']):raise ValueError('input provenance differs')
    tune,confirm=validation_parts(d.arrays['subject_ids'],d.arrays['patient_splits'])
    with np.load(ROOT/'validation_partition.npz') as z:
        np.testing.assert_array_equal(tune,z['tuning_indices'])
        np.testing.assert_array_equal(confirm,z['confirmation_indices'])
        np.testing.assert_array_equal(d.arrays['subject_ids'],z['subject_ids'])
    if sha256(Path(config['parent'])/f'seed{SEED}/survival_base/best.pt')!=contract['frozen_baseline_sha256']:
        raise ValueError('baseline source differs')
    if sha256(REFERENCE/'preprocessing.json')!=contract['preprocessing_sha256']:
        raise ValueError('preprocessing source differs')
    base=restore(Path(config['parent']),SEED,'survival_base',h,.1)
    bins=json.loads((REFERENCE/'preprocessing.json').read_text())['bins']
    return d,c,h,ext,base,bins,tune,confirm,contract,prior


def candidate(base,trial,bins):
    m=make_model(base,SEED,trial,bins).cuda();torch.manual_seed(SEED)
    path=ROOT/trial.name
    info=json.loads((path/'training_result.json').read_text())
    state=torch.load(path/'best.pt',map_location='cuda',weights_only=True)
    history=json.loads((path/'history.json').read_text())
    if info['trial']!=asdict(trial) or state['trial']!=asdict(trial):raise ValueError('trial contract differs')
    if info['completed_epochs']!=EPOCHS or [x['epoch'] for x in history]!=list(range(EPOCHS+1)):
        raise ValueError('incomplete epoch history')
    if min(history,key=lambda x:x['tuning_ibs'])['epoch']!=state['epoch'] or info['best_epoch']!=state['epoch']:
        raise ValueError('checkpoint was not selected by tuning IBS')
    if sha256(path/'best.pt')!=info['checkpoint_sha256']:raise ValueError('checkpoint hash differs')
    m.load_state_dict(state['state_dict'])
    if any(not torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):
        raise ValueError('frozen baseline changed')
    return m,info


def verify(*,partial=False):
    d,c,h,ext,base,bins,tune,_,contract,prior=checked()
    cache=batches(d,c,ext,tune)
    trials=[x for x in TRIALS if (ROOT/x.name/'training_result.json').exists()]
    if not partial and len(trials)!=len(TRIALS):raise ValueError('search not finished')
    errors={}
    for trial in trials:
        m,info=candidate(base,trial,bins)
        a,k,nll=parameters(m,cache);curve=probability_curve(a,k)
        with np.load(ROOT/trial.name/'tuning_predictions.npz') as z:
            np.testing.assert_array_equal(z['times'],d.times[tune]);np.testing.assert_array_equal(z['events'],d.events[tune])
            error=float(np.abs(curve-z['curve']).max())
            np.testing.assert_allclose(a,z['log_amplitude'],atol=1e-7,rtol=0)
            np.testing.assert_allclose(k,z['shape'],atol=1e-7,rtol=0)
        if error>3e-7 or not np.isfinite(curve).all() or np.any((curve<0)|(curve>1)):
            raise ValueError('probability replay failed')
        np.testing.assert_allclose(curve.sum(2),1,atol=1e-9)
        if np.any(np.diff(curve[:,:,1:],axis=1)<-1e-10):raise ValueError('nonmonotonic probabilities')
        from training.v6.module_surgery_experiment import validation
        if validation(m,cache,d.times[tune],d.events[tune])!=info['tuning']:raise ValueError('metric replay differs')
        errors[trial.name]=error;del m
    proof=dict(status='partial_completed_models_verified' if partial else 'screening_independently_verified',
        models=len(trials),maximum_error=max(errors.values()) if errors else None,replay_errors=errors,
        source_hashes_verified=len(contract['source_sha256']),official_dependency_hashes_verified=len(prior['source_sha256']),
        validation_partition_recomputed=True,confirmation_not_evaluated=True,test_not_evaluated=True)
    proof['audit_code_sha256']=sha256(Path(__file__))
    if not partial:
        r=json.loads((ROOT/'screening_report.json').read_text())
        ranked=sorted(r['results'],key=lambda x:r['results'][x]['tuning']['ibs'])
        if ranked!=r['ranked_by_tuning_ibs']:raise ValueError('ranking differs')
        nominated=[x for x in ranked if x!='v6_control' and r['results'][x]['tuning']['ibs']<r['results']['v6_control']['tuning']['ibs']][:3]
        if nominated!=r['nominated_before_confirmation']:raise ValueError('nomination changed')
        proof['screening_report_sha256']=sha256(ROOT/'screening_report.json')
    write_json(ROOT/('partial_verification.json' if partial else 'screening_verification.json'),proof)
    print(json.dumps(proof),flush=True)


def confirm():
    proof=json.loads((ROOT/'screening_verification.json').read_text())
    if proof['screening_report_sha256']!=sha256(ROOT/'screening_report.json'):raise ValueError('screening proof stale')
    if proof['audit_code_sha256']!=sha256(Path(__file__)):raise ValueError('verification code changed')
    r=json.loads((ROOT/'screening_report.json').read_text())
    nominated=r['nominated_before_confirmation']
    lock=dict(screening_report_sha256=sha256(ROOT/'screening_report.json'),nominated=nominated,
        comparators=['v6_control','official_V6'],no_test_evaluation=True,
        rule='confirm these locked nominees once; no substitution based on confirmation scores')
    if (ROOT/'nomination_lock.json').exists():raise FileExistsError('confirmation already locked; inspect existing run')
    write_json(ROOT/'nomination_lock.json',lock)
    d,c,h,ext,base,bins,tune,confirm_indices,contract,_=checked()
    cache=batches(d,c,ext,confirm_indices)
    trials={x.name:x for x in TRIALS}
    outputs,curves={},{ }
    for name in ['v6_control',*nominated,'official_V6']:
        if name=='official_V6':
            m=SurgeryModel(base,SEED,'control',bins).cuda()
            path=REFERENCE/f'seed{SEED}/control/best.pt'
            m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
            digest=sha256(path)
        else:
            m,info=candidate(base,trials[name],bins);digest=info['checkpoint_sha256']
        a,k,nll=parameters(m,cache);curve=probability_curve(a,k)
        outputs[name]=dict(checkpoint_sha256=digest,metrics=summary(curve,d,confirm_indices,nll))
        folder=ROOT/'confirmation'/name;folder.mkdir(parents=True,exist_ok=True)
        save(folder/'predictions.npz',d,confirm_indices,a,k,curve)
        curves[name]=curve;del m
    pairs={}
    for name in nominated:
        for reference in ('v6_control','official_V6'):
            pairs[f'{name} minus {reference}']=paired_interval(d.arrays['subject_ids'][confirm_indices],
                d.times[confirm_indices],d.events[confirm_indices],curves[name][:,:,1],curves[reference][:,:,1],replicates=200,seed=SEED)
    report=dict(status='single_seed_confirmation_not_stability_proof',lock=lock,models=outputs,comparisons=pairs,
        admissions=len(confirm_indices),patients=len(np.unique(d.arrays['subject_ids'][confirm_indices])),
        test_not_evaluated=True,goal_achieved=False,
        limitations=['Confirmation subset already exposed in prior project, not blind.','Single seed and selected candidates; not proof of stable gains.'])
    write_json(ROOT/'confirmation_report.json',report)
    write_json(ROOT/'state.json',dict(status='single_seed_confirmation_complete',completed_trials=list(r['results']),
        nominated=nominated,next_phase='predeclare multiseed confirmation without changing candidate settings',goal_achieved=False))
    print(json.dumps(dict(status=report['status'],nominated=nominated,goal_achieved=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('partial','verify','confirm'),required=True)
    mode=p.parse_args().mode
    verify(partial=mode=='partial') if mode!='confirm' else confirm()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/regularization_confirmation_plot.py · lines 1–38

<!-- REVIEW_FRAGMENT {"path":"training/v6/regularization_confirmation_plot.py","part":1,"start_line":1,"end_line":38,"source_sha256":"63949ad0137498d47d998ef89429689a0e2bcfa964c5546aa7c1a0d1135055b8","fragment_sha256":"63949ad0137498d47d998ef89429689a0e2bcfa964c5546aa7c1a0d1135055b8","added_terminal_newline":false} -->
````python
"""已独立核验的锁定确认差值图；不再选择/改写名单。"""
import json
from pathlib import Path
from patient_data.export_mimiciv import sha256
from training.v6.regularization_search import ROOT


def run():
    r=json.loads((ROOT/'confirmation_report.json').read_text())
    proof=json.loads((ROOT/'confirmation_verification.json').read_text())
    if proof['confirmation_report_sha256']!=sha256(ROOT/'confirmation_report.json') or proof['maximum_error']>3e-7:
        raise ValueError('verified confirmation required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=r['lock']['nominated']
    labels={'v6_shrink10':'Correction penalty 0.1','v6_shrink01':'Correction penalty 0.01','v6_wd10':'Weight decay 0.1'}
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,key,ci,title in zip(axes,('integrated_brier_difference','auc90_difference'),
                              ('integrated_brier_95ci','auc90_95ci'),
                              ('IBS7-365 difference (negative better)','AUC90 difference (positive better)')):
        for j,name in enumerate(names):
            for ref,offset,color,marker in (('v6_control',-.1,'C0','o'),('official_V6',.1,'C1','D')):
                p=r['comparisons'][f'{name} minus {ref}'];low,high=p[ci]
                ax.hlines(j+offset,low,high,color=color,lw=2)
                ax.scatter(p[key],j+offset,color=color,marker=marker,
                           label='Versus matched V6' if j==0 and ref=='v6_control' else 'Versus official V6' if j==0 else None)
        ax.axvline(0,color='black',lw=1);ax.grid(axis='x',alpha=.25);ax.set_xlabel(title)
    axes[0].set_yticks(range(len(names)),[labels[n] for n in names]);axes[0].invert_yaxis()
    fig.suptitle('Locked regularization confirmation; seed 42; 1,418 discharges / 395 patients\n'
                 'Patient-cluster bootstrap 200; exploratory 95% intervals; no multiplicity adjustment')
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.86),ncol=2,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.81));target=ROOT/'confirmation_increment.png';fig.savefig(target,dpi=180);plt.close(fig)
    print(json.dumps(dict(status='verified_confirmation_figure_saved',path=str(target))),flush=True)


if __name__=='__main__':run()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/regularization_confirmation_verify.py · lines 1–45

<!-- REVIEW_FRAGMENT {"path":"training/v6/regularization_confirmation_verify.py","part":1,"start_line":1,"end_line":45,"source_sha256":"ff01f939082753367d056a9c832c727d3f3848914795575bb552589a6f807f32","fragment_sha256":"ff01f939082753367d056a9c832c727d3f3848914795575bb552589a6f807f32","added_terminal_newline":false} -->
````python
"""单种子确认的独立重放；不再次锁定或替换候选。"""
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,candidate,REFERENCE
from training.v6.regularization_search import ROOT,SEED
from training.v6.regularization_model import TRIALS
from training.v6.module_surgery import SurgeryModel,probability_curve
from training.v6.module_surgery_experiment import batches,parameters,summary
from training.v6.token_model import daily_probabilities
from training.v6.train import write_json


def run():
    r=json.loads((ROOT/'confirmation_report.json').read_text())
    lock=json.loads((ROOT/'nomination_lock.json').read_text())
    if r['lock']!=lock:raise ValueError('nomination differs')
    d,c,h,ext,base,bins,_,confirm,_,_=checked()
    cache=batches(d,c,ext,confirm);trials={x.name:x for x in TRIALS};errors={}
    for name in ['v6_control',*lock['nominated'],'official_V6']:
        if name=='official_V6':
            m=SurgeryModel(base,SEED,'control',bins).cuda();path=REFERENCE/f'seed{SEED}/control/best.pt'
            m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
            digest=sha256(path)
        else:m,info=candidate(base,trials[name],bins);digest=info['checkpoint_sha256']
        if digest!=r['models'][name]['checkpoint_sha256']:raise ValueError('weight changed')
        a,k,nll=parameters(m,cache);curve=probability_curve(a,k)
        with np.load(ROOT/'confirmation'/name/'predictions.npz') as z:
            for key,expected in (('hadm_ids',d.arrays['hadm_ids'][confirm]),('subject_ids',d.arrays['subject_ids'][confirm]),('times',d.times[confirm]),('events',d.events[confirm])):
                np.testing.assert_array_equal(z[key],expected)
            error=float(np.abs(curve-z['curve']).max());daily,conditional=daily_probabilities(curve)
            np.testing.assert_allclose(daily,z['daily_event_probability'],atol=3e-7,rtol=0)
            np.testing.assert_allclose(conditional,z['conditional_daily_event_probability'],atol=3e-7,rtol=0)
        if error>3e-7:raise ValueError('confirmation replay differs')
        if summary(curve,d,confirm,nll)!=r['models'][name]['metrics']:raise ValueError('metric replay differs')
        errors[name]=error;del m
    write_json(ROOT/'confirmation_verification.json',dict(status='single_seed_confirmation_independently_verified',models=len(errors),
        maximum_error=max(errors.values()),replay_errors=errors,confirmation_report_sha256=sha256(ROOT/'confirmation_report.json'),
        verifier_sha256=sha256(Path(__file__)),test_not_evaluated=True,goal_achieved=False))
    print(json.dumps(dict(status='single_seed_confirmation_independently_verified',models=len(errors),maximum_error=max(errors.values()))),flush=True)


if __name__=='__main__':run()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/regularization_diagnostics.py · lines 1–75

<!-- REVIEW_FRAGMENT {"path":"training/v6/regularization_diagnostics.py","part":1,"start_line":1,"end_line":75,"source_sha256":"28d6854c3dde936b2f1ae6f500129769de3b551eea10968a6ef867e6f0280f6d","fragment_sha256":"28d6854c3dde936b2f1ae6f500129769de3b551eea10968a6ef867e6f0280f6d","added_terminal_newline":false} -->
````python
"""确认集校准与既往次数层内增量；不在层内再次选择候选。"""
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked
from training.v6.regularization_multiseed import ROOT,SEEDS
from training.v6.evaluation import metrics,public_metrics,paired_interval,calibration_bins
from training.v6.horizon_objective import calibration_fit
from training.v6.train import write_json


def history_counts(raw):
    if raw.ndim!=2 or raw.shape[1]!=6 or not np.isfinite(raw).all():raise ValueError('six raw history fields required')
    x=np.expm1(raw[:,2])
    if np.any(x<0) or not np.allclose(x,np.rint(x),atol=1e-4,rtol=0):raise ValueError('not raw log1p prior admissions')
    return np.rint(x).astype(np.int64)


def calibration(times,events,curve,cause):
    result={}
    for day in (30,90,180,365):
        risk=curve[:,:,cause]
        bins=calibration_bins(times,events,risk,day=day,cause=cause)
        y=((events==cause)&(times<=day)).astype(float)
        result[str(day)]=dict(observed=float(y.mean()),mean_predicted=float(risk[:,day].mean()),
            absolute_mean_error=float(abs(risk[:,day].mean()-y.mean())),
            ece10=float(sum(x['n']*abs(x['predicted']-x['observed']) for x in bins)/len(times)),
            logistic=calibration_fit(y,risk[:,day]),bins=bins)
    return result


def run():
    report=json.loads((ROOT/'report.json').read_text())
    proof=json.loads((ROOT/'independent_verification.json').read_text())
    if proof['report_sha256']!=sha256(ROOT/'report.json'):raise ValueError('replay proof stale')
    d,_,_,_,_,_,_,confirm,_,_=checked()
    counts=history_counts(d.history_raw[confirm])
    strata=dict(prior0=counts==0,prior1_2=(counts>=1)&(counts<=2),prior3plus=counts>=3)
    candidates=[x['name'] for x in report['contract']['trials'] if x['name']!='v6_control']
    models={}
    for seed in SEEDS:
        models[str(seed)]={};curves={}
        for name in ['v6_control',*candidates,'official_V6']:
            with np.load(ROOT/f'seed{seed}'/name/'confirmation_predictions.npz') as z:
                for key,expected in (('hadm_ids',d.arrays['hadm_ids'][confirm]),('times',d.times[confirm]),('events',d.events[confirm])):
                    np.testing.assert_array_equal(z[key],expected)
                curve=z['curve'].copy()
            curves[name]=curve
            groups={}
            for label,mask in strata.items():
                if not mask.any():groups[label]=dict(status='empty');continue
                groups[label]=public_metrics(metrics(d.times[confirm][mask],d.events[confirm][mask],curve[mask,:,1],ordering=True))
            models[str(seed)][name]=dict(calibration={str(c):calibration(d.times[confirm],d.events[confirm],curve,c) for c in (1,2)},strata=groups)
        for name in candidates:
            pairs={}
            for reference in ('v6_control','official_V6'):
                per_group={}
                for label,mask in strata.items():
                    y=(d.events[confirm][mask]==1)&(d.times[confirm][mask]<=90)
                    if not mask.any() or min(int(y.sum()),len(y)-int(y.sum()))<10:
                        per_group[label]=dict(status='insufficient cases/controls for paired AUC; not zero effect')
                        continue
                    per_group[label]=paired_interval(d.arrays['subject_ids'][confirm][mask],d.times[confirm][mask],d.events[confirm][mask],
                        curves[name][mask,:,1],curves[reference][mask,:,1],seed=seed,replicates=200)
                pairs[reference]=per_group
            models[str(seed)][name]['paired_strata']=pairs
    write_json(ROOT/'calibration_and_history_strata.json',dict(report_sha256=sha256(ROOT/'report.json'),models=models,
        definition='prior completed retained COPD-coded admissions, raw log1p field2, not strict count and not normalized values',
        strata_admissions={k:int(v.sum()) for k,v in strata.items()},test_not_evaluated=True,goal_achieved=False,
        limitations=['Sparse strata AUC may be unavailable.','Group differences not a causal interpretation.','No candidate reselection by strata.']))
    print(json.dumps(dict(status='confirmation_calibration_and_history_strata_written',goal_achieved=False)),flush=True)


if __name__=='__main__':run()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/regularization_model.py · lines 1–135

<!-- REVIEW_FRAGMENT {"path":"training/v6/regularization_model.py","part":1,"start_line":1,"end_line":135,"source_sha256":"6c5fcdf94f5267341546ad26bca2096c6ebd6b4e7992df19ee51c4546c986db8","fragment_sha256":"6c5fcdf94f5267341546ad26bca2096c6ebd6b4e7992df19ee51c4546c986db8","added_terminal_newline":false} -->
````python
"""独立正则化适配器：不改原V6/ABC源文件，连续时间概率保持。"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from training.v6.deep_interaction import DeepModel
from training.v6.module_surgery import SurgeryModel
from training.v6.risk import RiskOutput


@dataclass(frozen=True)
class Trial:
    name: str
    architecture: str = 'control'
    dropout: float = .1
    latent_dropout: float = 0.
    text_dropout: float = 0.
    weight_decay: float = 1e-4
    lr: float = 1e-4
    delta_penalty: float = 1e-4
    consistency: float = 0.
    brier: float = 0.
    depth_drop: float = 0.


TRIALS = (
    Trial('v6_control'),
    Trial('v6_dropout30', dropout=.3, latent_dropout=.2),
    Trial('v6_dropout50', dropout=.5, latent_dropout=.3),
    Trial('v6_text30', text_dropout=.3),
    Trial('v6_wd01', weight_decay=.01),
    Trial('v6_wd10', weight_decay=.1),
    Trial('v6_shrink01', delta_penalty=.01),
    Trial('v6_shrink10', delta_penalty=.1),
    Trial('v6_low_lr', lr=3e-5),
    Trial('v6_drop_wd', dropout=.3, latent_dropout=.2, text_dropout=.2, weight_decay=.01),
    Trial('v6_consistency', dropout=.3, latent_dropout=.2, consistency=1.),
    Trial('v6_brier2', brier=2.),
    Trial('v6_brier10', brier=10.),
    Trial('v6_drop_brier', dropout=.3, latent_dropout=.2, brier=5.),
    Trial('single_control', architecture='late_single'),
    Trial('single_drop_wd', architecture='late_single', dropout=.3, latent_dropout=.2, weight_decay=.01),
    Trial('A_control', architecture='A_deep'),
    Trial('A_dropout30', architecture='A_deep', dropout=.3, latent_dropout=.2),
    Trial('A_wd10', architecture='A_deep', weight_decay=.1),
    Trial('A_stochastic', architecture='A_deep', dropout=.2, depth_drop=.3),
    Trial('A_consistency', architecture='A_deep', dropout=.2, consistency=1., weight_decay=.01),
    Trial('A_brier', architecture='A_deep', dropout=.2, brier=5.),
    Trial('B_control', architecture='B_shared'),
    Trial('B_drop_wd', architecture='B_shared', dropout=.3, latent_dropout=.2, weight_decay=.01),
    Trial('B_stochastic', architecture='B_shared', dropout=.2, depth_drop=.3),
    Trial('B_consistency', architecture='B_shared', dropout=.2, consistency=1.),
)


class StochasticResidual(nn.Module):
    """每个样本随机绕过整个块；eval用完整块，不引入行号时序。"""
    def __init__(self, module, rate):
        super().__init__(); self.module, self.rate = module, rate
        if not 0 <= rate < 1:
            raise ValueError('invalid depth dropout')

    def forward(self, x, *args, **kwargs):
        update = self.module(x, *args, **kwargs)
        if self.training and self.rate:
            keep = (torch.rand(len(x), 1, 1, device=x.device) >= self.rate).to(x.dtype)
            return x + (update-x)*keep/(1-self.rate)
        return update


class OutputDropout(nn.Module):
    def __init__(self, encoder, probability):
        super().__init__(); self.encoder=encoder; self.dropout=nn.Dropout(probability)

    def forward(self, *args, **kwargs):
        h, diagnostics = self.encoder(*args, **kwargs)
        return self.dropout(h), diagnostics


def make_model(baseline, seed, trial, bins, *, text_width=768, width=256):
    if trial.architecture=='control':
        model=SurgeryModel(baseline,seed,'control',bins,text_width=text_width)
        core=model.encoder.base
        core.fusion[2].p=trial.dropout
        if trial.text_dropout:
            core.text_projection=nn.Sequential(core.text_projection,nn.Dropout(trial.text_dropout))
    else:
        model=DeepModel(baseline,seed,trial.architecture,text_width=text_width,width=width)
        # nn.MultiheadAttention使用数值dropout属性而不是独立nn.Dropout。
        for module in model.encoder.modules():
            if isinstance(module,nn.Dropout):module.p=trial.dropout
            if isinstance(module,nn.MultiheadAttention):module.dropout=trial.dropout
        if trial.depth_drop:
            model.encoder.blocks=nn.ModuleList(StochasticResidual(m,trial.depth_drop) for m in model.encoder.blocks)
    if trial.latent_dropout:
        model.encoder=OutputDropout(model.encoder,trial.latent_dropout)
    model.arm=trial.name
    torch.manual_seed(seed)
    return model


def tensor_cif(output: RiskOutput, days=(14,30,60,90,180,365), step=.5):
    """可微CIF；正时间网格避免零点奇异，双原因共享事件质量。"""
    if step<=0 or 365/step!=int(365/step) or any(d<=0 or d>365 for d in days):
        raise ValueError('invalid quadrature grid')
    device=output.shape.device
    grid=torch.arange(step,365+step/2,step,device=device,dtype=torch.float64)
    small=torch.exp(torch.linspace(-23.025850929940457,torch.log(grid[0]).item(),80,device=device,dtype=torch.float64))
    grid=torch.unique(torch.cat((small,grid,torch.tensor(days,device=device,dtype=torch.float64))),sorted=True)
    a,k=output.log_amplitude.double(),output.shape.double()
    cumulative=torch.exp(a[:,None]+k[:,None]*torch.log(grid[None,:,None]/365))
    previous=torch.cat((torch.zeros_like(cumulative[:,:1]),cumulative[:,:-1]),1)
    increment=(cumulative-previous).clamp_min(0)
    total=increment.sum(2,keepdim=True)
    survival_before=torch.exp(-previous.sum(2,keepdim=True))
    mass=survival_before*(-torch.expm1(-total))*increment/total.clamp_min(1e-300)
    cif=mass.cumsum(1)
    selected=torch.searchsorted(grid,torch.tensor(days,device=device,dtype=grid.dtype))
    survival=torch.exp(-cumulative[:,selected].sum(2,keepdim=True))
    return torch.cat((survival,cif[:,selected]),2)


def probability_consistency(left,right):
    p,q=tensor_cif(left),tensor_cif(right)
    p=p.clamp_min(1e-12);q=q.clamp_min(1e-12)
    p=p/p.sum(2,keepdim=True);q=q/q.sum(2,keepdim=True)
    return (.5*((p-q)*(p.log()-q.log())).sum(2)).mean()


def auxiliary_brier(output,times,events):
    days=torch.tensor((14,30,60,90,180,365),device=times.device)
    p=tensor_cif(output)[:,:,1:]
    y=((events[:,None,None]==torch.tensor((1,2),device=events.device)) &
       (times[:,None,None]<=days[None,:,None])).to(p.dtype)
    return (p-y).square().mean()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/regularization_multiseed.py · lines 1–166

<!-- REVIEW_FRAGMENT {"path":"training/v6/regularization_multiseed.py","part":1,"start_line":1,"end_line":166,"source_sha256":"8530d2a776f7e9ee726319d74fdf75448b9034343ff67d4588d6e58d56ac2195","fragment_sha256":"8530d2a776f7e9ee726319d74fdf75448b9034343ff67d4588d6e58d56ac2195","added_terminal_newline":false} -->
````python
"""锁定正则候选五种子确认：同基线配对，继续不用测试选择。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_model import make_model,Trial,probability_consistency,auxiliary_brier
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.module_surgery_experiment import batches,parameters,validation,summary,save
from training.v6.module_surgery import member_nll,probability_curve,SurgeryModel
from training.v6.horizon_objective import restore
from training.v6.evaluation import paired_interval
from training.v6.train import write_json

SEARCH=Path('training_runs/v6_regularization_search_2026-10-12')
ROOT=Path('training_runs/v6_regularization_multiseed_2026-10-13')
PROTOCOL=Path('docs/V6_REGULARIZATION_CONFIRMATION_PROTOCOL_2026-10-13.md')
SEEDS=(42,7,101,202,303)
BASE_SEEDS={42:42,7:7,101:42,202:7,303:42}
EPOCHS=100


def build(base,seed,trial,bins):
    model=make_model(base,seed,trial,bins).cuda();torch.manual_seed(seed)
    return model


def fit(base,seed,trial,bins,cache,times,events,path,*,model_factory=None,backward=None):
    path.mkdir(parents=True,exist_ok=True)
    model=(model_factory or build)(base,seed,trial,bins);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=trial.lr,weight_decay=trial.weight_decay)
    frozen={k:v.detach().clone() for k,v in model.baseline.state_dict().items()}
    rows,best,chosen=[],math.inf,0;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def record(epoch,online=None,regular=None,lr=0.):
        nonlocal best,chosen
        score=validation(model,cache['tune'],times,events)
        # 小样本诊断不足10病例/对照时AUC不可计算；不能写NaN或冒充0。
        if not math.isfinite(score['auc90']):score['auc90']=None
        rows.append(dict(epoch=epoch,train_nll=online,penalty=regular,lr=lr,
                         **{'tuning_'+k:v for k,v in score.items()}))
        if score['ibs']<best:
            best,chosen=score['ibs'],epoch
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,tuning_ibs=best,trial=asdict(trial)),path/'best.pt')
        if epoch%20==0:
            write_json(path/'history.json',rows)
            print(json.dumps(dict(seed=seed,trial=trial.name,epoch=epoch,best_epoch=chosen,tuning_ibs=score['ibs'],
                seconds=round(time.perf_counter()-start,1))),flush=True)
    record(0)
    for epoch in range(1,EPOCHS+1):
        model.train()
        lr=trial.lr*min(1.,epoch/5)*(.1+.45*(1+math.cos(math.pi*max(0,epoch-5)/(EPOCHS-5))))
        for g in opt.param_groups:g['lr']=lr
        loss_sum,reg_sum,count=0.,0.,0
        for j in rng.permutation(len(cache['train'])):
            _,h,b,t,e=cache['train'][j];opt.zero_grad(set_to_none=True)
            out=model(h,b);nll=member_nll(out,t,e);penalty=trial.delta_penalty*out.delta.square().mean()
            if trial.consistency:
                other=model(h,b);nll=(nll+member_nll(other,t,e))*.5
                penalty+=trial.consistency*probability_consistency(out,other)
            if trial.brier:penalty+=trial.brier*auxiliary_brier(out,t,e)
            loss=nll+penalty
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            if backward is None:loss.backward()
            else:backward(model,out,nll,penalty,t,e,trial)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
            opt.step();loss_sum+=float(nll.detach())*len(t);reg_sum+=float(penalty.detach())*len(t);count+=len(t)
        record(epoch,loss_sum/count,reg_sum/count,lr)
    if any(not torch.equal(v,model.baseline.state_dict()[k]) for k,v in frozen.items()):raise ValueError('baseline changed')
    info=dict(seed=seed,baseline_seed=BASE_SEEDS[seed],trial=asdict(trial),completed_epochs=EPOCHS,best_epoch=chosen,
        tuning_ibs=best,checkpoint_sha256=sha256(path/'best.pt'),elapsed_seconds=time.perf_counter()-start,
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,frozen_verified=True)
    write_json(path/'history.json',rows);write_json(path/'training_result.json',info)
    del model;torch.cuda.empty_cache()
    return info


def run():
    # 用确认分数只决定是否值得继续，不重新挑/替换候选。三个锁定候选均确认。
    prior=json.loads((SEARCH/'confirmation_report.json').read_text())
    lock=json.loads((SEARCH/'nomination_lock.json').read_text())
    nominated=lock['nominated']
    if not nominated:raise ValueError('no nominated candidates; next hypothesis needed')
    d,c,h,ext,_,bins,tune,confirm,search_contract,_=checked()
    screen=json.loads((SEARCH/'screening_report.json').read_text())
    if sha256(SEARCH/'screening_report.json')!=lock['screening_report_sha256']:raise ValueError('nomination stale')
    trials=[Trial(**screen['results'][name]['trial']) for name in ['v6_control',*nominated]]
    ROOT.mkdir(exist_ok=True)
    frozen={str(s):sha256(Path(search_contract['input_config']['parent'])/f'seed{BASE_SEEDS[s]}/survival_base/best.pt') for s in SEEDS}
    contract=dict(search_contract_sha256=sha256(SEARCH/'contract.json'),search_verification_sha256=sha256(SEARCH/'screening_verification.json'),
        nomination_lock_sha256=sha256(SEARCH/'nomination_lock.json'),trials=[asdict(t) for t in trials],seeds=list(SEEDS),
        baseline_seed_mapping=BASE_SEEDS,frozen_baselines=frozen,epochs=EPOCHS,selection='same tuning-half validation IBS only',
        source_sha256={str(Path(__file__)):sha256(Path(__file__)),'training/v6/regularization_model.py':sha256(Path('training/v6/regularization_model.py')),
            str(PROTOCOL):sha256(PROTOCOL),'training/v6/regularization_multiseed_verify.py':sha256(Path('training/v6/regularization_multiseed_verify.py'))},
        no_test_evaluation=True,limits=['Repeated two fixed H34 sources, five clinical seeds; not five independently refit cohorts.',
        'Validation subsets previously exposed, not blind.','Candidates locked before confirmation.'])
    if (ROOT/'contract.json').exists():
        if json.loads((ROOT/'contract.json').read_text())!=json.loads(json.dumps(contract)):
            raise ValueError('multiseed resume contract differs')
    else:write_json(ROOT/'contract.json',contract)
    cache=dict(train=batches(d,c,ext,np.flatnonzero(d.arrays['patient_splits']=='train')),tune=batches(d,c,ext,tune))
    fits={}
    for seed in SEEDS:
        base=restore(Path(search_contract['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        fits[str(seed)]={}
        for trial in trials:
            path=ROOT/f'seed{seed}'/trial.name
            if (path/'training_result.json').exists():
                completed=json.loads((path/'training_result.json').read_text())
                if (completed['trial']!=asdict(trial) or completed['completed_epochs']!=EPOCHS or
                        sha256(path/'best.pt')!=completed['checkpoint_sha256']):
                    raise ValueError('completed training result differs')
                fits[str(seed)][trial.name]=completed
                continue
            if path.exists() and any((path/name).exists() for name in ('best.pt','history.json')):
                import shutil
                backup=path/'recovery_snapshots'/str(time.time_ns());backup.mkdir(parents=True)
                for name in ('best.pt','history.json'):
                    if (path/name).exists():shutil.copy2(path/name,backup/name)
            write_json(ROOT/'state.json',dict(status='running',seed=seed,trial=trial.name,completed=fits,goal_achieved=False))
            fits[str(seed)][trial.name]=fit(base,seed,trial,bins,cache,d.times[tune],d.events[tune],path)
    write_json(ROOT/'selected_before_confirmation.json',dict(fits=fits,epochs_complete=True,no_confirmation_used_for_checkpoint_selection=True))
    del cache['train'];del cache['tune'];torch.cuda.empty_cache()
    cache=batches(d,c,ext,confirm)
    results,comparisons={},{}
    for seed in SEEDS:
        base=restore(Path(search_contract['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        results[str(seed)],curves={},{}
        for trial in trials:
            path=ROOT/f'seed{seed}'/trial.name;m=build(base,seed,trial,bins)
            m.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k)
            results[str(seed)][trial.name]=dict(training=fits[str(seed)][trial.name],metrics=summary(curve,d,confirm,nll))
            save(path/'confirmation_predictions.npz',d,confirm,a,k,curve);curves[trial.name]=curve;del m
        old=SurgeryModel(base,BASE_SEEDS[seed],'control',bins).cuda()
        old.load_state_dict(torch.load(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt',map_location='cuda',weights_only=True)['state_dict'])
        a,k,nll=parameters(old,cache);curves['official_V6']=probability_curve(a,k)
        folder=ROOT/f'seed{seed}'/'official_V6';folder.mkdir(parents=True,exist_ok=True)
        save(folder/'confirmation_predictions.npz',d,confirm,a,k,curves['official_V6'])
        results[str(seed)]['official_V6']=dict(metrics=summary(curves['official_V6'],d,confirm,nll));del old
        comparisons[str(seed)]={f'{name} minus {ref}':paired_interval(d.arrays['subject_ids'][confirm],d.times[confirm],d.events[confirm],
            curves[name][:,:,1],curves[ref][:,:,1],replicates=200,seed=seed) for name in nominated for ref in ('v6_control','official_V6')}
    write_json(ROOT/'report.json',dict(status='multiseed_confirmation_pending_replay',contract=contract,models=results,
        comparisons=comparisons,goal_achieved=False,test_not_evaluated=True))
    write_json(ROOT/'state.json',dict(status='training_and_confirmation_complete',next_phase='fresh-process replay and audit criteria before any test evaluation',goal_achieved=False))
    print(json.dumps(dict(status='multiseed_confirmation_pending_replay',goal_achieved=False)),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    ROOT.mkdir(exist_ok=True)
    lock=ROOT/'.running.lock'
    with lock.open('x',encoding='utf-8') as owner:
        owner.write(json.dumps(dict(pid=os.getpid(),started_unix=time.time(),purpose='single locked multiseed batch')))
    try:run()
    except Exception as error:
        write_json(ROOT/'state.json',dict(status='failed',error_type=type(error).__name__,error=str(error),
            next_phase='inspect logs before a bounded recovery',goal_achieved=False))
        raise
    finally:lock.unlink()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/regularization_multiseed_verify.py · lines 1–102

<!-- REVIEW_FRAGMENT {"path":"training/v6/regularization_multiseed_verify.py","part":1,"start_line":1,"end_line":102,"source_sha256":"36f11dfc9ba9bc56850a4b850a7bf8f21fa26ffd2d2e3921bf42dfdb1e8590a4","fragment_sha256":"36f11dfc9ba9bc56850a4b850a7bf8f21fa26ffd2d2e3921bf42dfdb1e8590a4","added_terminal_newline":false} -->
````python
"""五种子确认独立重放与预定增益门槛审计；不读取测试预测。"""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.regularization_multiseed import ROOT,SEARCH,SEEDS,BASE_SEEDS,EPOCHS,build
from training.v6.regularization_model import Trial
from training.v6.module_surgery_experiment import batches,parameters,summary
from training.v6.module_surgery import SurgeryModel,probability_curve
from training.v6.token_model import daily_probabilities
from training.v6.horizon_objective import restore
from training.v6.train import write_json


def audit_gate(models,names):
    """仅生成证据与达标状态；不能自动把内部确认称盲测成功。"""
    result={}
    for name in names:
        comparisons={}
        for comparator in ('v6_control','official_V6'):
            aa,bb,dd,unsafe=[],[],[],[]
            for seed in SEEDS:
                left=models[str(seed)][name]['metrics'];right=models[str(seed)][comparator]['metrics']
                l,r=left['readmission'],right['readmission']
                left_auc,right_auc=l['points']['90']['auc'],r['points']['90']['auc']
                aa.append(left_auc-right_auc if left_auc is not None and right_auc is not None else float('nan'))
                bb.append(l['integrated_brier_7_365']-r['integrated_brier_7_365'])
                dd.append(left['death']['integrated_brier_7_365']-right['death']['integrated_brier_7_365'])
                # 多时间点不应明显退化；阈值预先固定为AUC降0.005/Brier增0.001。
                unsafe.append(any(l['points'][str(day)]['auc'] is not None and r['points'][str(day)]['auc'] is not None and
                    l['points'][str(day)]['auc']<r['points'][str(day)]['auc']-.005 for day in (30,90,180,365)) or
                    any(l['points'][str(day)]['brier']>r['points'][str(day)]['brier']+.001 for day in (30,90,180,365)))
            baseline=np.mean([models[str(s)][comparator]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS])
            passed=(np.isfinite(aa).all() and sum(a>0 and b<0 for a,b in zip(aa,bb))>=4 and np.mean(aa)>=.005 and
                    -np.mean(bb)/baseline>=.01 and sum(unsafe)<=1 and np.mean(dd)<=.001)
            comparisons[comparator]=dict(auc90_differences=[x if np.isfinite(x) else None for x in aa],ibs_differences=bb,death_ibs_differences=dd,
                joint_improved_seeds=sum(a>0 and b<0 for a,b in zip(aa,bb)),mean_auc90_difference=float(np.mean(aa)) if np.isfinite(aa).all() else None,
                mean_ibs_difference=float(np.mean(bb)),relative_ibs_reduction=float(-np.mean(bb)/baseline),
                unsafe_timepoint_seeds=sum(unsafe),predeclared_basic_gate_passed=bool(passed))
        result[name]=dict(comparisons=comparisons,basic_gate_passed=all(x['predeclared_basic_gate_passed'] for x in comparisons.values()),
            stable_superiority_proven=False,
            remaining_requirements=['calibration and history-strata audit','locked historical test comparison (not blind)','paired uncertainty across seeds'])
    return result


def verify():
    r=json.loads((ROOT/'report.json').read_text());contract=r['contract']
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('multiseed source changed: '+p)
    for file,key in (('contract.json','search_contract_sha256'),('screening_verification.json','search_verification_sha256'),('nomination_lock.json','nomination_lock_sha256')):
        if sha256(SEARCH/file)!=contract[key]:raise ValueError('locked search changed')
    d,c,h,ext,_,bins,tune,confirm,search,_=checked()
    cache=batches(d,c,ext,confirm)
    trials=[Trial(**x) for x in contract['trials']]
    errors={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        if sha256(Path(search['input_config']['parent'])/f'seed{BASE_SEEDS[seed]}/survival_base/best.pt')!=contract['frozen_baselines'][str(seed)]:
            raise ValueError('foundation checkpoint changed')
        for trial in [*trials,None]:
            name=trial.name if trial else 'official_V6';path=ROOT/f'seed{seed}'/name
            if trial:
                m=build(base,seed,trial,bins)
                state=torch.load(path/'best.pt',map_location='cuda',weights_only=True)
                history=json.loads((path/'history.json').read_text())
                fit=r['models'][str(seed)][name]['training']
                if ([x['epoch'] for x in history]!=list(range(EPOCHS+1)) or
                    min(history,key=lambda x:x['tuning_ibs'])['epoch']!=state['epoch'] or
                    sha256(path/'best.pt')!=fit['checkpoint_sha256'] or state['trial']!=asdict(trial)):
                    raise ValueError('training contract/selection differs')
                m.load_state_dict(state['state_dict'])
            else:
                m=SurgeryModel(base,BASE_SEEDS[seed],'control',bins).cuda()
                m.load_state_dict(torch.load(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt',map_location='cuda',weights_only=True)['state_dict'])
            if any(not torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):raise ValueError('foundation not frozen')
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k)
            with np.load(path/'confirmation_predictions.npz') as z:
                for key,expected in (('hadm_ids',d.arrays['hadm_ids'][confirm]),('subject_ids',d.arrays['subject_ids'][confirm]),('times',d.times[confirm]),('events',d.events[confirm])):
                    np.testing.assert_array_equal(z[key],expected)
                error=float(np.abs(curve-z['curve']).max())
                daily,conditional=daily_probabilities(curve)
                np.testing.assert_allclose(daily,z['daily_event_probability'],atol=3e-7,rtol=0)
                np.testing.assert_allclose(conditional,z['conditional_daily_event_probability'],atol=3e-7,rtol=0)
            if error>3e-7 or not np.isfinite(curve).all():raise ValueError('probability replay failed')
            np.testing.assert_allclose(curve.sum(2),1,atol=1e-9)
            if np.any(np.diff(curve[:,:,1:],axis=1)<-1e-10):raise ValueError('nonmonotonic CIF')
            if summary(curve,d,confirm,nll)!=r['models'][str(seed)][name]['metrics']:raise ValueError('metric replay differs')
            errors[f'{seed}/{name}']=error;del m
    r['status']='multiseed_confirmation_independently_verified'
    write_json(ROOT/'report.json',r)
    write_json(ROOT/'independent_verification.json',dict(status=r['status'],models=len(errors),maximum_error=max(errors.values()),
        replay_errors=errors,report_sha256=sha256(ROOT/'report.json'),verifier_sha256=sha256(Path(__file__)),test_not_evaluated=True))
    gate=audit_gate(r['models'],[x.name for x in trials if x.name!='v6_control'])
    write_json(ROOT/'basic_gate_audit.json',dict(candidate_gates=gate,goal_achieved=False,
        explanation='basic thresholds alone do not complete calibration/strata/test requirements'))
    print(json.dumps(dict(status=r['status'],models=len(errors),maximum_error=max(errors.values()),gates=gate)),flush=True)


if __name__=='__main__':verify()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/regularization_screen_report.py · lines 1–71

<!-- REVIEW_FRAGMENT {"path":"training/v6/regularization_screen_report.py","part":1,"start_line":1,"end_line":71,"source_sha256":"41eb8d642c549d7bde43619fee056f3404d605b0263e8dcf3b125baa7962f39a","fragment_sha256":"41eb8d642c549d7bde43619fee056f3404d605b0263e8dcf3b125baa7962f39a","added_terminal_newline":false} -->
````python
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
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/regularization_search.py · lines 1–169

<!-- REVIEW_FRAGMENT {"path":"training/v6/regularization_search.py","part":1,"start_line":1,"end_line":169,"source_sha256":"644129e27404b70dc47beee80e85ec4596cd7a87964500444ae6b17d60235173","fragment_sha256":"644129e27404b70dc47beee80e85ec4596cd7a87964500444ae6b17d60235173","added_terminal_newline":false} -->
````python
"""长期探索首批：固定清单，仅验证调参半折；不访问测试指标。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_model import TRIALS,make_model,probability_consistency,auxiliary_brier
from training.v6.module_surgery import member_nll,probability_curve
from training.v6.module_surgery_experiment import inputs,batches,parameters,validation,setup
from training.v6.horizon_objective import restore
from training.v6.train import write_json

ROOT=Path('training_runs/v6_regularization_search_2026-10-12')
PLAN=Path('docs/V6_LONG_TERM_OPTIMIZATION_PLAN_2026-10-12.md')
SEED=42
EPOCHS=60


def validation_parts(subjects,splits):
    if subjects.shape!=splits.shape:raise ValueError('patient/split mismatch')
    patients=np.unique(subjects[splits=='validation'])
    choose={p:int(hashlib.sha256(f'v6-regularization-confirm-20261012:{p}'.encode()).hexdigest()[:8],16)%2 for p in patients}
    selected=np.array([splits[i]=='validation' and choose[subjects[i]]==0 for i in range(len(subjects))])
    confirm=(splits=='validation') & ~selected
    if not selected.any() or not confirm.any():raise ValueError('empty validation subset')
    if set(subjects[selected]) & set(subjects[confirm]):raise ValueError('patient split overlap')
    return np.flatnonzero(selected),np.flatnonzero(confirm)


def build(base,trial,bins):
    m=make_model(base,SEED,trial,bins).cuda();torch.manual_seed(SEED)
    return m


def train_one(base,trial,bins,cache,times,events,path):
    path.mkdir(parents=True,exist_ok=True)
    m=build(base,trial,bins)
    torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=trial.lr,weight_decay=trial.weight_decay)
    frozen={k:v.detach().clone() for k,v in m.baseline.state_dict().items()}
    history,best,chosen=[],math.inf,0
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def record(epoch,online=None,penalty=None,lr=0.):
        nonlocal best,chosen
        info=validation(m,cache['tune'],times,events)
        _,_,fixed=parameters(m,cache['train'][:8])
        history.append(dict(epoch=epoch,train_online_nll=online,train_fixed2048_nll=fixed,
            auxiliary_penalty=penalty,lr=lr,**{'tuning_'+k:v for k,v in info.items()}))
        if info['ibs']<best:
            best,chosen=info['ibs'],epoch
            torch.save(dict(state_dict=m.state_dict(),epoch=epoch,tuning_ibs=best,trial=asdict(trial)),path/'best.pt')
        if epoch%10==0:
            write_json(path/'history.json',history)
            print(json.dumps(dict(trial=trial.name,epoch=epoch,best_epoch=chosen,tuning_ibs=info['ibs'],
                tuning_auc90=info['auc90'],seconds=round(time.perf_counter()-start,1))),flush=True)
    record(0)
    for epoch in range(1,EPOCHS+1):
        m.train()
        lr=trial.lr*min(1.,epoch/5)*(.1+.45*(1+math.cos(math.pi*max(0,epoch-5)/(EPOCHS-5))))
        for g in opt.param_groups:g['lr']=lr
        total,count,regularization=0.,0,0.
        for j in rng.permutation(len(cache['train'])):
            _,h,b,t,e=cache['train'][j];opt.zero_grad(set_to_none=True)
            out=m(h,b);nll=member_nll(out,t,e)
            penalty=trial.delta_penalty*out.delta.square().mean()
            if trial.consistency:
                other=m(h,b);nll=(nll+member_nll(other,t,e))*.5
                penalty+=trial.consistency*probability_consistency(out,other)
            if trial.brier:penalty+=trial.brier*auxiliary_brier(out,t,e)
            loss=nll+penalty
            if not torch.isfinite(loss):raise ValueError('nonfinite regularized loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite gradients')
            if any(p.grad is not None for p in m.baseline.parameters()):raise ValueError('baseline gradient leakage')
            opt.step();total+=float(nll.detach())*len(t);regularization+=float(penalty.detach())*len(t);count+=len(t)
        record(epoch,total/count,regularization/count,lr)
    if any(not torch.equal(v,m.baseline.state_dict()[k]) for k,v in frozen.items()):raise ValueError('baseline changed')
    chosen_state=torch.load(path/'best.pt',map_location='cuda',weights_only=True);m.load_state_dict(chosen_state['state_dict'])
    a,k,nll=parameters(m,cache['tune']);curve=probability_curve(a,k)
    np.savez_compressed(path/'tuning_predictions.npz',log_amplitude=a,shape=k,curve=curve,times=times,events=events)
    info=validation(m,cache['tune'],times,events)
    if abs(info['ibs']-best)>1e-12:raise ValueError('best checkpoint replay failed')
    result=dict(trial=asdict(trial),completed_epochs=EPOCHS,best_epoch=chosen,tuning=info,
        checkpoint_sha256=sha256(path/'best.pt'),frozen_baseline_verified=True,
        trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
        elapsed_seconds=time.perf_counter()-start,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    write_json(path/'history.json',history);write_json(path/'training_result.json',result)
    del m;torch.cuda.empty_cache()
    return result


def run():
    setup();torch.backends.mha.set_fastpath_enabled(False)
    d,c,h,ext,config,old,prov=inputs()
    tune,confirm=validation_parts(d.arrays['subject_ids'],d.arrays['patient_splits'])
    train=np.flatnonzero(d.arrays['patient_splits']=='train')
    prep=json.loads(Path('training_runs/v6_module_surgery_2026-10-12_r3/preprocessing.json').read_text())
    protected=dict(old['contract']['code_sha256'])
    for p in ('training/v6/regularization_model.py','training/v6/regularization_search.py',
              'training/v6/deep_interaction.py','training/tests/test_v6_regularization.py'):
        protected[p]=sha256(Path(p))
    contract=dict(source_sha256=protected,plan_sha256=sha256(PLAN),data_provenance=d.provenance,
        compact_provenance=c.provenance,extensions=prov,input_config=config,seed=SEED,epochs=EPOCHS,
        trials=[asdict(x) for x in TRIALS],selection='tuning-half validation readmission IBS7-365',
        admissions=dict(train=len(train),tune=len(tune),confirmation=len(confirm)),
        validation_partition='patient SHA256 modulo2; original validation previously exposed, not blind',
        frozen_baseline_sha256=sha256(Path(config['parent'])/f'seed{SEED}/survival_base/best.pt'),
        preprocessing_sha256=sha256(Path('training_runs/v6_module_surgery_2026-10-12_r3/preprocessing.json')),
        endpoint='strict491.21/J44.1; competing death; WorldA365',
        no_test_evaluation=True,confirmation_not_used_for_search=True)
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'contract.json').exists():
        if json.loads((ROOT/'contract.json').read_text())!=contract:raise ValueError('resume contract changed')
    else:
        write_json(ROOT/'contract.json',contract)
        np.savez_compressed(ROOT/'validation_partition.npz',tuning_indices=tune,confirmation_indices=confirm,
                            subject_ids=d.arrays['subject_ids'])
    cache=dict(train=batches(d,c,ext,train),tune=batches(d,c,ext,tune))
    base=restore(Path(config['parent']),SEED,'survival_base',h,.1)
    completed={}
    for trial in TRIALS:
        path=ROOT/trial.name
        if (path/'training_result.json').exists():
            result=json.loads((path/'training_result.json').read_text())
            if result['trial']!=asdict(trial) or result['checkpoint_sha256']!=sha256(path/'best.pt'):
                raise ValueError('saved completed trial differs')
            completed[trial.name]=result
            continue
        if path.exists() and any((path/name).exists() for name in ('best.pt','history.json')):
            backup=path/'recovery_snapshots'/str(time.time_ns())
            backup.mkdir(parents=True)
            for name in ('best.pt','history.json'):
                if (path/name).exists():shutil.copy2(path/name,backup/name)
        write_json(ROOT/'state.json',dict(status='running',current_trial=trial.name,completed_trials=list(completed),
            total_trials=len(TRIALS),next_phase='lock top candidates, do not evaluate test',goal_achieved=False))
        completed[trial.name]=train_one(base,trial,prep['bins'],cache,d.times[tune],d.events[tune],path)
    ranked=sorted(completed,key=lambda x:completed[x]['tuning']['ibs'])
    # 至多三个、不同机制的候选；阈值按调参折筛，不看确认/测试。
    nominated=[x for x in ranked if x!='v6_control' and completed[x]['tuning']['ibs']<completed['v6_control']['tuning']['ibs']][:3]
    write_json(ROOT/'screening_report.json',dict(status='screening_complete_not_confirmed',contract=contract,
        results=completed,ranked_by_tuning_ibs=ranked,nominated_before_confirmation=nominated,
        original_reference_checkpoint='prior official V6 retained; no test scores consumed',goal_achieved=False))
    write_json(ROOT/'state.json',dict(status='screening_complete',completed_trials=list(completed),total_trials=len(TRIALS),
        nominated=nominated,next_phase='fresh-process replay then locked candidate confirmation/multiseed',goal_achieved=False))
    print(json.dumps(dict(status='screening_complete',nominated=nominated,goal_achieved=False)),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    ROOT.mkdir(exist_ok=True)
    lock=ROOT/'.running.lock'
    with lock.open('x',encoding='utf-8') as owner:
        owner.write(json.dumps(dict(pid=os.getpid(),started_unix=time.time(),purpose='single regularization batch')))
    try:
        run()
    except Exception as error:
        write_json(ROOT/'state.json',dict(status='failed',error_type=type(error).__name__,error=str(error),
            next_phase='inspect exact logs before retry; preserve partial checkpoints',goal_achieved=False))
        raise
    finally:
        lock.unlink()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/repair_expanded_medication.py · lines 1–57

<!-- REVIEW_FRAGMENT {"path":"training/v6/repair_expanded_medication.py","part":1,"start_line":1,"end_line":57,"source_sha256":"dde77696ce35124b63a14104f0464ec98db276c0d9ccefe3cb0a721f6796392f","fragment_sha256":"dde77696ce35124b63a14104f0464ec98db276c0d9ccefe3cb0a721f6796392f","added_terminal_newline":false} -->
````python
"""保留不受字典修正影响的3臂，只重跑药物相关3臂×两种子。"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.expanded_original_experiment import checked,prepare,new_model
from training.v6.horizon_objective import restore
from training.v6.token_experiment import fit
from training.v6.train import write_json


def run(source,root,*,resume_setup=False):
    if root.exists():
        if not resume_setup or list(root.rglob('best.pt')) or list(root.rglob('history.json')):
            raise FileExistsError('not an empty failed copy setup: '+str(root))
        prior=root/'contract.json'
        if prior.exists():shutil.copy2(prior,root/'contract_copy_setup_failed.json')
    old=json.loads((source/'report.json').read_text());cfg=json.loads((source/'config.json').read_text());proof=json.loads((source/'fresh_process_verification.json').read_text())
    if proof['report_sha256']!=sha256(source/'report.json'):raise ValueError('parent not verified')
    cfg['expanded']='patient_data/processed/v6_expanded_candidates_2026-10-12_r3';cfg['output']=str(root)
    args=argparse.Namespace(**{k:Path(v) for k,v in cfg.items()});d,c,h,ext,lab,med,prov,active,norm=checked(args)
    if old['contract']['data_provenance']!=d.provenance:raise ValueError('cohort drift')
    # 修正仅涉及具体药物；原检验、原7类药效、BERT、基础参照不变。
    if old['contract']['new_sources']['expanded_labs.npz']!=prov['expanded_labs.npz']:raise ValueError('lab copy drift')
    root.mkdir(exist_ok=resume_setup);write_json(root/'config.json',cfg)
    contract=dict(old['contract']);contract.update(new_sources=prov,active=active,normalization_errors=norm,
       copied_unaffected_arms=['control','more_labs','interaction_only'],rerun_arms=['more_meds','both','both_interaction'],
       correction_source_report_sha256=sha256(source/'report.json'),
       correction='Existing umeclidinium/vilanterol was incorrectly rejected by new constituent matcher; corrected, affected arms retrained.')
    contract['code_sha256']=dict(contract['code_sha256'])
    for file in ('training/v6/repair_expanded_medication.py','patient_data/v6_expanded_medication_fix.py'):
        contract['code_sha256'][file]=sha256(Path(file))
    write_json(root/'contract.json',contract)
    for seed in (42,7):
        for arm in contract['copied_unaffected_arms']:
            # 只复制原权重和完整训练日志，所有折预测用修正后的共同接口重新生成。
            dst=root/f'seed{seed}/{arm}';dst.mkdir(parents=True,exist_ok=resume_setup)
            for file in ('best.pt','history.json'):shutil.copy2(source/f'seed{seed}/{arm}'/file,dst/file)
            if sha256(dst/'best.pt')!=old['models'][str(seed)][arm]['checkpoint_sha256']:raise ValueError('copied checkpoint changed')
    ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('train','validation')}
    cache={f:prepare(d,c,ext,lab,med,ix[f]) for f in ix}
    for seed in (42,7):
        base=restore(args.parent,seed,'survival_base',h,.1)
        for arm in contract['rerun_arms']:
            m=new_model(base,seed,arm);fit(m,cache['train'],cache['validation'],d.times[ix['validation']],d.events[ix['validation']],
                 root/f'seed{seed}/{arm}',seed=seed,lr=1e-4,epochs=100);del m
    write_json(root/'status.json',dict(status='all_corrected_checkpoints_selected_before_test'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('training_runs/v6_original_interaction_2026-10-12'))
    p.add_argument('--output',type=Path,default=Path('training_runs/v6_original_interaction_2026-10-12_r3'))
    p.add_argument('--resume-copy-setup',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;run(a.source,a.output,resume_setup=a.resume_copy_setup)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/repeat_architecture.py · lines 1–45

<!-- REVIEW_FRAGMENT {"path":"training/v6/repeat_architecture.py","part":1,"start_line":1,"end_line":45,"source_sha256":"865f7b5fdcdd5ee55b26fce6bfdf77084230b8ed98e5966a9a41f215679d358a","fragment_sha256":"865f7b5fdcdd5ee55b26fce6bfdf77084230b8ed98e5966a9a41f215679d358a","added_terminal_newline":false} -->
````python
"""用另一种子确认关键对照；旧H检查点固定，B及临床Δ重新训练。"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from patient_data.export_mimiciv import sha256
from training.v6 import architecture_experiment as experiment


def run(seed, pilot, baseline_output, explore_output):
    if seed==42: raise ValueError('confirmation seed must differ')
    pilot_report=json.loads((pilot/'report.json').read_text())
    if pilot_report['status']!='completed_explore_exploration_single_seed': raise ValueError('pilot not complete')
    # 仅用验证选优；成分/比值、原B/候选/融合、pool参照为预先列出的核心比较。
    selected=pilot_report['validation_selected']
    names={'pool128','availability','components','nlr_sii','no_text','screen_only','bert_screen',selected}
    original=experiment.configurations
    filtered={name:value for name,value in original('explore').items() if name in names}
    common=dict(reference=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'),
                compact_dir=Path('patient_data/processed/v6_compact_multisource_2026-10-09'),
                combinations=Path('patient_data/processed/v6_cbc_combinations_2026-10-10'),
                candidate=Path('patient_data/processed/v6_screen_candidate_bert_2026-10-10'),
                epochs=60,batch_size=512,bootstrap=200,seed=seed,verify=False,baseline_run=baseline_output)
    experiment.run(SimpleNamespace(**common,phase='baseline',output=baseline_output))
    experiment.configurations=lambda phase: filtered if phase=='explore' else original(phase)
    try: experiment.run(SimpleNamespace(**common,phase='explore',output=explore_output))
    finally: experiment.configurations=original
    for folder in (baseline_output,explore_output):
        report=json.loads((folder/'report.json').read_text())
        report['contract']['code_sha256'][str(Path(__file__))]=sha256(Path(__file__))
        report['contract']['confirmation_selection']='Core arms prespecified; extra architecture from seed42 validation only'
        report['contract']['confirmation_pilot_report_sha256']=sha256(pilot/'report.json')
        report['contract']['limits'].append('H is the fixed seed42 historical reference; B and delta retrained under confirmation seed, not full H retraining variance')
        experiment.write_json(folder/'report.json',report)
        experiment.write_json(folder/'contract.json',report['contract'])
        experiment.verify(SimpleNamespace(output=folder))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--pilot',type=Path,default=Path('training_runs/v6_architecture_screen_seed42_2026-10-10'))
    p.add_argument('--baseline-output',type=Path,default=Path('training_runs/v6_expanded_baseline_seed7_2026-10-10'))
    p.add_argument('--explore-output',type=Path,default=Path('training_runs/v6_architecture_screen_seed7_2026-10-10'))
    a=p.parse_args(); run(a.seed,a.pilot,a.baseline_output,a.explore_output)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/report_horizon_roc.py · lines 1–55

<!-- REVIEW_FRAGMENT {"path":"training/v6/report_horizon_roc.py","part":1,"start_line":1,"end_line":55,"source_sha256":"856ae1e40cca0ff148928483f59e7c2fa05c156d91ccc8aeebda77839343f128","fragment_sha256":"856ae1e40cca0ff148928483f59e7c2fa05c156d91ccc8aeebda77839343f128","added_terminal_newline":false} -->
````python
"""只读取已有预测，展示明确时点ROC/AUC；不重训、不改变选模口径。"""
import argparse
import json
from pathlib import Path
import numpy as np


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    report = json.loads((root / 'report.json').read_text())
    folder = root / 'horizon_reporting'
    if folder.exists(): raise FileExistsError(folder)
    folder.mkdir()
    labels = {'without_new': 'No new imaging', 'availability_only': 'Imaging availability only',
              'radbert': 'Frozen RadBERT', 'radgraph': 'RadGraph-XL facts',
              'history': 'History baseline', 'constant': 'History + constant correction',
              'clinical_no_text': 'History + current clinical state',
              'clinical_bert': 'History + current state + BERT'}
    arms = report['contract'].get('arms', list(report['models']))
    all_values = {}
    for fold in ('validation', 'test'):
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        all_values[fold] = {}
        for arm in arms:
            with np.load(root / arm / (fold + '_predictions.npz')) as a:
                times, events, curves = a['times'], a['events'], a['curve']
            all_values[fold][arm] = {}
            for ax, day in zip(axes.flat, (30, 90, 180, 365), strict=True):
                y = ((events == 1) & (times <= day)).astype(int)
                score = curves[:, day, 1]
                fpr, tpr, _ = roc_curve(y, score); auc = roc_auc_score(y, score)
                recorded = report['models'][arm]['folds'][fold]['readmission']['points'][str(day)]['auc']
                if abs(auc - recorded) > 1e-10: raise ValueError('horizon AUC disagrees with existing evaluation')
                ax.plot(fpr, tpr, label=f'{labels[arm]}: {auc:.4f}')
                all_values[fold][arm][str(day)] = dict(auc=auc, cases=int(y.sum()), controls=int((1-y).sum()))
        for ax, day in zip(axes.flat, (30, 90, 180, 365), strict=True):
            ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
            ax.set(xlabel='False positive rate', ylabel='True positive rate', title=f'{day}-day cumulative readmission ROC')
            ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.suptitle(f'{fold}: existing continuous-risk predictions; strict endpoint; World A')
        fig.tight_layout(); fig.savefig(folder / (fold + '_horizon_roc.png'), dpi=160); plt.close(fig)
    (folder / 'report.json').write_text(json.dumps(dict(status='existing_predictions_horizon_auc_reproduced',
        values=all_values, limits=['No model training or selection change',
        'Competing death is control for cumulative readmission incidence',
        'World A recorded outcomes, not verified complete external follow-up',
        'No per-horizon confidence intervals added here']), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    render(parser.parse_args().run)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/resp_fact_set_experiment.py · lines 1–261

<!-- REVIEW_FRAGMENT {"path":"training/v6/resp_fact_set_experiment.py","part":1,"start_line":1,"end_line":261,"source_sha256":"6567e6b8165bc7d27a1a3ef3a5ec789bc7549fd4afe486af24f50efaac0df0b9","fragment_sha256":"6567e6b8165bc7d27a1a3ef3a5ec789bc7549fd4afe486af24f50efaac0df0b9","added_terminal_newline":false} -->
````python
"""冻结r2候选事实集合有限早停配对；保留原完整V6风险。"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from patient_data.resp_fact_set_interface import ROOT as DATA,derive
from training.v6.resp_fact_set_model import FactCorrection,FactBatch,ARMS
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.curve_adaptation import FrozenBatch,caches
from training.v6.module_surgery import probability_curve
from training.v6.module_surgery_experiment import summary,save
from training.v6.risk import competing_nll
from training.v6.evaluation import metrics
from training.v6.early_stopping import EarlyStopping
from training.v6.train import write_json

ROOT=Path('training_runs/v6_resp_fact_set_2026-10-13_r2')
PROTOCOL=Path('docs/V6_RESP_FACT_SET_TRAINING_PROTOCOL_2026-10-13.md')
SEEDS=(42,7,101);CAP=60



def inputs():
    data=checked();d,c,h,ext,_,bins,tune,confirm,search,prior=data
    m=json.loads((DATA/'manifest.json').read_text())
    for tag in ('a','b'):
        p=json.loads((DATA/f'verification_{tag}.json').read_text())
        if p['maximum_error'] or p['manifest_sha256']!=sha256(DATA/'manifest.json'):
            raise ValueError('PFT interface source proof differs')
    for name,digest in m['source_sha256'].items():
        if sha256(Path(name))!=digest:raise ValueError('PFT source changed: '+name)
    for name,digest in m['output_sha256'].items():
        if sha256(DATA/name)!=digest:raise ValueError('PFT interface output changed')
    with np.load(DATA/'interface.npz') as z:arrays={k:z[k] for k in z.files}
    for key in ('hadm_ids','subject_ids','patient_splits'):np.testing.assert_array_equal(arrays[key],d.arrays[key])
    train=np.flatnonzero(d.arrays['patient_splits']=='train')
    if len(train)!=13592 or len(tune)!=1531:raise ValueError('outer split changed')
    sources={**search['source_sha256'],**prior['source_sha256'],**m['source_sha256']}
    for name in ('training/v6/resp_fact_set_model.py','training/v6/resp_fact_set_experiment.py','training/v6/curve_adaptation.py',
                 'training/v6/warm_residual_audit.py','training/v6/early_stopping.py','training/v6/pft_set_model.py','training/tests/test_v6_resp_fact_set_model.py',
                 'training/tests/test_v6_resp_fact_set_experiment.py'):
        sources[name]=sha256(Path(name))
    contract=dict(source_sha256=sources,protocol_sha256=sha256(PROTOCOL),data_manifest_sha256=sha256(DATA/'manifest.json'),
        data_sha256=m['output_sha256']['interface.npz'],fact_sha256=m['output_sha256']['facts.jsonl.gz'],
        scales=m['scales'],data_provenance=d.provenance,compact_provenance=c.provenance,
        teacher_hashes={str(s):sha256(REFERENCE/f'seed{s}/control/best.pt') for s in (42,7)},
        baseline_hashes={str(s):sha256(Path(search['input_config']['parent'])/f'seed{s}/survival_base/best.pt') for s in (42,7)},
        seeds=list(SEEDS),source_mapping={'42':42,'7':7,'101':42},arms=list(ARMS),new_parameters=998,
        optimizer={'name':'AdamW','lr':.001,'weight_decay':1e-4,'delta_penalty':1e-4,'clip':1.,'batch':512},
        early_stopping={'patience':10,'min_epochs':15,'min_delta':1e-5,'max_epochs':60},
        selection='mean tuning IBS; content must beat original and availability for new typed-fact content nomination',
        high_frequency_trajectories_used=False,old_static_support_retained_by_user=True,
        fact_role='explicit discharge/section and separately marked inpatient change; unknown role isolated',
        execution_image='aecopd-embedding:v1',torch_runtime='2.12.0+cu130',
        confirmation_test_evaluated=False,goal_achieved=False)
    return data,train,arrays,contract


def augmented(data,indices,seed,arrays):
    frozen,proof=caches(data,indices,seed)
    x={key:torch.cat([getattr(b,key) for b in frozen]) for key in ('hidden','raw','prior','times','events')}
    base=frozen[0].shape_raw;result=[];hashes=hashlib.sha256(indices.tobytes())
    for value in (*x.values(),base):hashes.update(value.cpu().numpy().tobytes())
    for offset in range(0,len(indices),512):
        sl=slice(offset,offset+512);use=indices[sl];owner=np.full(len(arrays['hadm_ids']),-1,np.int64);owner[use]=np.arange(len(use))
        source=owner[arrays['positions']];rows=np.flatnonzero(source>=0)
        f=FrozenBatch(x['hidden'][sl],x['raw'][sl],x['prior'][sl],base,x['times'][sl],x['events'][sl])
        tensor=lambda v:torch.as_tensor(v,device='cuda')
        b=FactBatch(f,*[tensor(arrays[key][rows]) for key in ('values','concepts','roles','polarities','units','conditions','numeric_present','numeric_valid')],
            tensor(source[rows]),tensor(arrays['aggregate'][use]),tensor(arrays['observed'][use]))
        for value in (b.values,b.concepts,b.roles,b.polarities,b.units,b.conditions,b.numeric_present,b.numeric_valid,b.owners,b.aggregate,b.observed):hashes.update(value.cpu().numpy().tobytes())
        result.append(b)
    return result,dict(teacher_cache=proof,augmented_cache_sha256=hashes.hexdigest(),records=len(indices))


def rebuild_data(arrays):
    rebuilt,_,_,_=derive()
    for key,value in rebuilt.items():np.testing.assert_array_equal(value,arrays[key])


def new_model(arm,seed):
    m=FactCorrection(arm,seed).cuda();torch.manual_seed(seed);return m


@torch.no_grad()
def prediction(model,cache):
    a,k=[],[];loss,n=0.,0
    for b in cache:
        out=model(b);a.append(out.log_amplitude.cpu().numpy());k.append(out.shape.cpu().numpy())
        loss+=float(competing_nll(out,b.frozen.times,b.frozen.events))*len(b.frozen.times);n+=len(b.frozen.times)
    aa,kk=np.concatenate(a),np.concatenate(k)
    return aa,kk,probability_curve(aa,kk),loss/n


def score(model,cache):
    a,k,curve,nll=prediction(model,cache)
    t=np.concatenate([b.frozen.times.cpu().numpy() for b in cache]);e=np.concatenate([b.frozen.events.cpu().numpy() for b in cache])
    r=metrics(t,e,curve[:,:,1]);death=metrics(t,e,curve[:,:,2],cause=2)
    return dict(nll=nll,ibs=r['integrated_brier_7_365'],auc90=r['points']['90']['auc'],death_ibs=death['integrated_brier_7_365'])


def preflight():
    data,train,arrays,contract=inputs()
    if ROOT.exists():raise FileExistsError('preflight already exists; inspect')
    ROOT.mkdir();write_json(ROOT/'contract.json',contract);rebuild_data(arrays)
    cached,proof=augmented(data,train[:128],42,arrays);b=cached[0]
    prior=None
    for arm in ARMS:
        m=new_model(arm,42);out=m(b)
        if prior is None:prior=out
        else:
            torch.testing.assert_close(out.log_amplitude,prior.log_amplitude,atol=0,rtol=0)
            torch.testing.assert_close(out.shape,prior.shape,atol=0,rtol=0)
        # 用真实训练来源存在者做单步，不能只拿全缺失批次宣称梯度启动。
        del m
    use=train[np.asarray(arrays['observed'][train],bool)][:128]
    observed,_=augmented(data,use,42,arrays);probe=new_model('content',42)
    opt=torch.optim.AdamW(probe.parameters(),lr=.001);before={k:v.detach().clone() for k,v in probe.state_dict().items()}
    out=probe(observed[0]);loss=competing_nll(out,observed[0].frozen.times,observed[0].frozen.events)
    loss.backward();torch.nn.utils.clip_grad_norm_(probe.parameters(),1.);opt.step()
    if torch.equal(before['head.2.weight'],probe.state_dict()['head.2.weight']):raise ValueError('real observed source cannot update')
    # 非零修正下检查真实GPU来源控制、numeric_valid屏蔽、集合排列和缺失gate。
    control=new_model('availability',42)
    with torch.no_grad():control.head[-1].weight.fill_(.01)
    b2=observed[0]
    changed=replace(b2,values=b2.values+50,concepts=(b2.concepts+1)%8,roles=(b2.roles+1)%3,
        polarities=1-b2.polarities,units=(b2.units+1)%3,conditions=(b2.conditions+1)%4,
        numeric_present=torch.zeros_like(b2.numeric_present),numeric_valid=torch.zeros_like(b2.numeric_valid))
    torch.testing.assert_close(control(b2).log_amplitude,control(changed).log_amplitude,atol=0,rtol=0)
    torch.testing.assert_close(control(b2).shape,control(changed).shape,atol=0,rtol=0)
    ix=torch.arange(len(b2.values)-1,-1,-1,device='cuda')
    permuted=replace(b2,**{key:getattr(b2,key)[ix] for key in
        ('values','concepts','roles','polarities','units','conditions','numeric_present','numeric_valid','owners')})
    torch.testing.assert_close(probe(b2).log_amplitude,probe(permuted).log_amplitude,atol=1e-6,rtol=1e-6)
    masked=replace(b2,values=b2.values+50*(~b2.numeric_valid))
    torch.testing.assert_close(probe(b2).log_amplitude,probe(masked).log_amplitude,atol=0,rtol=0)
    missing=~b.observed
    if not missing.any():raise ValueError('missing-source preflight has no actual rows')
    torch.testing.assert_close(probe(b).log_amplitude[missing],prior.log_amplitude[missing],atol=0,rtol=0)
    torch.testing.assert_close(probe(b).shape[missing],prior.shape[missing],atol=0,rtol=0)
    if sum(p.numel() for p in probe.parameters())!=998:raise ValueError('parameter budget changed')
    write_json(ROOT/'preflight.json',dict(status='passed',common_initial_risk_error=0.,original_cache_error=proof['teacher_cache']['initial_parameter_max_error'],
        source_rebuilt=True,availability_all_typed_content_hidden=True,real_observed_GPU_update=True,gpu=torch.cuda.get_device_name(0)))
    write_json(ROOT/'state.json',dict(status='preflight_passed',fits=0,goal_achieved=False))
    print(json.dumps(dict(status='preflight_passed',original_risk_error=0.)),flush=True)


def fit(model,train_cache,tune_cache,d,tune,folder,seed,arm):
    folder.mkdir(parents=True);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);early=EarlyStopping();history=[];start=time.perf_counter();reason='max_epochs'
    for epoch in range(CAP+1):
        online=None
        if epoch:
            total,n=0.,0;model.train()
            for j in rng.permutation(len(train_cache)):
                b=train_cache[j];opt.zero_grad(set_to_none=True);out=model(b)
                nll=competing_nll(out,b.frozen.times,b.frozen.events);loss=nll+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite PFT training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('nonfinite PFT gradient')
                opt.step();total+=float(nll.detach())*len(b.frozen.times);n+=len(b.frozen.times)
            online=total/n
        detail=score(model,tune_cache);save_best,stop=early.observe(epoch,detail['ibs'])
        history.append(dict(epoch=epoch,train_nll=online,**{'tuning_'+k:v for k,v in detail.items()},early_stale=early.stale,early_anchor=early.anchor))
        if save_best:torch.save(dict(state_dict=model.state_dict(),epoch=epoch,seed=seed,arm=arm),folder/'best.pt')
        write_json(folder/'history.json',history)
        if epoch%5==0 or stop:print(json.dumps(dict(seed=seed,arm=arm,epoch=epoch,best=early.best_epoch,ibs=detail['ibs'],early_stop=stop)),flush=True)
        if stop:reason='patience_exhausted';break
    model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
    a,k,curve,nll=prediction(model,tune_cache);save(folder/'tuning_predictions.npz',d,tune,a,k,curve)
    info=dict(arm=arm,seed=seed,completed_epochs=history[-1]['epoch'],best_epoch=early.best_epoch,stop_reason=reason,early_state=asdict(early),
        seconds=time.perf_counter()-start,checkpoint_sha256=sha256(folder/'best.pt'),metrics=summary(curve,d,tune,nll))
    write_json(folder/'training_result.json',info);return info


def lock(report):
    scores={arm:float(np.mean([report['fits'][str(s)][arm]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS])) for arm in ARMS}
    scores['original_V6']=float(np.mean([report['references'][str(s)]['readmission']['integrated_brier_7_365'] for s in SEEDS]))
    selected=min(scores,key=scores.get)
    return dict(selected=selected,scores=scores,nominated=['content'] if selected=='content' and scores['content']<min(v for k,v in scores.items() if k!='content') else [],
        selection='predeclared3-seed tuning IBS; availability gain not typed-fact content gain',goal_achieved=False)


def run():
    data,train,arrays,contract=inputs();d,_,_,_,_,_,tune,_,_,_=data
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    if json.loads((ROOT/'state.json').read_text())['status']!='preflight_passed':raise ValueError('already started; inspect')
    report=dict(fits={},references={},cache_hashes={},confirmation_test_not_evaluated=True,goal_achieved=False)
    for seed in SEEDS:
        tc,tp=augmented(data,train,seed,arrays);vc,vp=augmented(data,tune,seed,arrays)
        report['cache_hashes'][str(seed)]={'train':tp,'tune':vp}
        ref=new_model('content',seed);a,k,curve,nll=prediction(ref,vc)
        report['references'][str(seed)]=summary(curve,d,tune,nll);del ref
        report['fits'][str(seed)]={}
        for arm in ARMS:
            write_json(ROOT/'state.json',dict(status='training_with_early_stop',seed=seed,arm=arm,goal_achieved=False))
            m=new_model(arm,seed);report['fits'][str(seed)][arm]=fit(m,tc,vc,d,tune,ROOT/f'seed{seed}'/arm,seed,arm);del m
        del tc,vc;torch.cuda.empty_cache()
    write_json(ROOT/'screening.json',report);write_json(ROOT/'nomination_lock.json',lock(report))
    write_json(ROOT/'state.json',dict(status='training_complete_pending_independent_replay',fits=6,goal_achieved=False))
    print(json.dumps(lock(report)),flush=True)


def verify(tag):
    data,train,arrays,contract=inputs();d,_,_,_,_,_,tune,_,_,_=data
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    rebuild_data(arrays);report=json.loads((ROOT/'screening.json').read_text());errors={}
    for seed in SEEDS:
        tc,tp=augmented(data,train,seed,arrays);vc,vp=augmented(data,tune,seed,arrays)
        if report['cache_hashes'][str(seed)]!={'train':tp,'tune':vp}:raise ValueError('teacher/current source cache differs')
        ref=new_model('content',seed);a,k,curve,nll=prediction(ref,vc)
        if summary(curve,d,tune,nll)!=report['references'][str(seed)]:raise ValueError('reference differs')
        del ref,tc
        for arm in ARMS:
            folder=ROOT/f'seed{seed}'/arm;info=json.loads((folder/'training_result.json').read_text())
            if info!=report['fits'][str(seed)][arm] or sha256(folder/'best.pt')!=info['checkpoint_sha256']:raise ValueError('checkpoint differs')
            history=json.loads((folder/'history.json').read_text());early=EarlyStopping();first_stop=None
            for row in history:
                _,stop=early.observe(row['epoch'],row['tuning_ibs'])
                if stop and first_stop is None:first_stop=row['epoch']
            if asdict(early)!=info['early_state'] or early.best_epoch!=info['best_epoch'] or info['completed_epochs']!=len(history)-1:
                raise ValueError('early stop state/selection differs')
            if info['stop_reason']=='patience_exhausted' and first_stop!=info['completed_epochs']:raise ValueError('early stop bypassed')
            m=new_model(arm,seed);stored=torch.load(folder/'best.pt',map_location='cuda',weights_only=True)
            if stored['epoch']!=early.best_epoch:raise ValueError('wrong saved epoch')
            m.load_state_dict(stored['state_dict']);a,k,curve,nll=prediction(m,vc)
            with np.load(folder/'tuning_predictions.npz') as z:
                np.testing.assert_array_equal(z['times'],d.times[tune]);np.testing.assert_array_equal(z['events'],d.events[tune])
                error=max(float(np.abs(a-z['log_amplitude']).max()),float(np.abs(k-z['shape']).max()),float(np.abs(curve-z['curve']).max()))
            if error>3e-7 or summary(curve,d,tune,nll)!=info['metrics']:raise ValueError('prediction/metric differs')
            errors[f'{seed}/{arm}']=error;del m
        del vc
    if lock(report)!=json.loads((ROOT/'nomination_lock.json').read_text()):raise ValueError('selection lock differs')
    proof=dict(status='independently_verified',models=6,maximum_error=max(errors.values()),replay_errors=errors,
        report_sha256=sha256(ROOT/'screening.json'),contract_sha256=sha256(ROOT/'contract.json'),
        source_inputs_and_full_teacher_caches_rebuilt=True,early_stop_replayed=True,confirmation_test_not_evaluated=True)
    write_json(ROOT/f'verification_{tag}.json',proof)
    if all((ROOT/f'verification_{t}.json').exists() for t in ('a','b')):
        for t in ('a','b'):
            if json.loads((ROOT/f'verification_{t}.json').read_text())!=proof:raise ValueError('independent proof differs')
        write_json(ROOT/'state.json',dict(status='screening_complete_independently_verified',fits=6,nominated=lock(report)['nominated'],goal_achieved=False))
    print(json.dumps(proof),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('preflight','run','verify'),required=True);p.add_argument('--tag',choices=('a','b'),default='a');args=p.parse_args()
    try:
        if args.mode=='preflight':preflight()
        elif args.mode=='run':run()
        else:verify(args.tag)
    except Exception as e:
        if ROOT.exists():write_json(ROOT/f'failure_{args.mode}_{args.tag}.json',dict(error=repr(e),goal_achieved=False))
        raise


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/resp_fact_set_model.py · lines 1–64

<!-- REVIEW_FRAGMENT {"path":"training/v6/resp_fact_set_model.py","part":1,"start_line":1,"end_line":64,"source_sha256":"c7c57696ebf95ebf7197bfcf74eaa07091525fe3e90f91761b69d828e2caca54","fragment_sha256":"c7c57696ebf95ebf7197bfcf74eaa07091525fe3e90f91761b69d828e2caca54","added_terminal_newline":false} -->
````python
"""冻结原完整风险的低容量typed事实集合修正；来源对照真屏蔽。"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.pft_set_model import segment_pool
from training.v6.risk import RiskOutput

ARMS=('availability','content')

@dataclass
class FactBatch:
    frozen:object
    values:torch.Tensor
    concepts:torch.Tensor
    roles:torch.Tensor
    polarities:torch.Tensor
    units:torch.Tensor
    conditions:torch.Tensor
    numeric_present:torch.Tensor
    numeric_valid:torch.Tensor
    owners:torch.Tensor
    aggregate:torch.Tensor
    observed:torch.Tensor

class FactCorrection(nn.Module):
    def __init__(self,arm,seed=42):
        super().__init__()
        if arm not in ARMS:raise ValueError('unknown fact arm')
        self.arm=arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embeddings=nn.ModuleDict({k:nn.Embedding(n,8) for k,n in
                [('concepts',8),('roles',3),('polarities',2),('units',3),('conditions',4)]})
            self.value_projection=nn.Sequential(nn.Linear(3,8),nn.GELU(),nn.LayerNorm(8))
            self.value_score=nn.Linear(8,1);self.concept_score=nn.Linear(8,1)
            self.head=nn.Sequential(nn.Linear(43,16),nn.GELU(),nn.Linear(16,4))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,b):
        f=b.frozen;n=len(f.hidden);m=len(b.values)
        if f.hidden.shape!=(n,32) or b.aggregate.shape!=(n,3) or b.observed.shape!=(n,):raise ValueError('fact admission shape differs')
        if b.owners.shape!=(m,) or b.owners.dtype!=torch.long or (b.owners<0).any() or (b.owners>=n).any():raise ValueError('fact owners invalid')
        for name,limit in [('concepts',8),('roles',3),('polarities',2),('units',3),('conditions',4)]:
            v=getattr(b,name)
            if v.shape!=(m,) or v.dtype!=torch.long or (v<0).any() or (v>=limit).any():raise ValueError('invalid typed fact '+name)
        if b.numeric_present.shape!=(m,) or b.numeric_valid.shape!=(m,) or b.numeric_present.dtype!=torch.bool or b.numeric_valid.dtype!=torch.bool:
            raise ValueError('fact numeric masks differ')
        if (b.numeric_valid&~b.numeric_present).any() or not torch.isfinite(b.values).all():raise ValueError('invalid numeric validity/value')
        if not torch.equal(b.observed,b.aggregate.bool().any(1)):raise ValueError('source role gate differs')
        if self.arm=='availability':
            # 不读取typed属性/成员数量的表征：只保留共同三角色来源信息。
            pooled=f.hidden.new_zeros(n,8)
        else:
            tokens=sum(layer(getattr(b,key)) for key,layer in self.embeddings.items())
            numeric=torch.stack((b.values*b.numeric_valid,b.numeric_present.float(),b.numeric_valid.float()),1)
            tokens=tokens+self.value_projection(numeric)
            group=b.owners*8+b.concepts;within=segment_pool(tokens,group,n*8,self.value_score)
            present=torch.zeros(n*8,dtype=torch.bool,device=b.values.device);present[group]=True
            slots=torch.nonzero(present,as_tuple=False).squeeze(1)
            pooled=segment_pool(within[slots],slots//8,n,self.concept_score)
        raw=self.head(torch.cat((f.hidden,pooled,b.aggregate),1))*b.observed[:,None]
        raw=raw.reshape(n,2,2);delta=torch.stack((raw[:,:,0],.25*torch.tanh(raw[:,:,1])),2);original=f.raw.reshape(n,2,2)
        return RiskOutput((f.prior+original[:,:,0]+delta[:,:,0]).clamp(-10,6),F.softplus(f.shape_raw+original[:,:,1]+delta[:,:,1])+.1,delta)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/resp_fact_set_report.py · lines 1–105

<!-- REVIEW_FRAGMENT {"path":"training/v6/resp_fact_set_report.py","part":1,"start_line":1,"end_line":105,"source_sha256":"913276a5a570e63894accec6576ab8832f58d27d8e06747d50674853b8e70e88","fragment_sha256":"913276a5a570e63894accec6576ab8832f58d27d8e06747d50674853b8e70e88","added_terminal_newline":false} -->
````python
"""呼吸候选事实集合内容/来源配对读出；不改选点/默认。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patient_data.export_mimiciv import sha256
from training.v6.resp_fact_set_experiment import ROOT,SEEDS,ARMS,inputs,augmented,new_model,prediction
from training.v6.evaluation import paired_interval,metrics,public_metrics,calibration_bins
from training.v6.train import write_json


def build_report():
    saved=json.loads((ROOT/'screening.json').read_text())
    for tag in ('a','b'):
        proof=json.loads((ROOT/f'verification_{tag}.json').read_text())
        if proof['report_sha256']!=sha256(ROOT/'screening.json') or proof['maximum_error']!=0:raise ValueError('not independently verified')
    data,_,arrays,_=inputs();d,_,_,_,_,_,tune,_,_,_=data
    observed=arrays['observed'][tune];counts=np.rint(np.expm1(d.history_raw[tune,2])).astype(int)
    groups={'source_observed':observed,'source_missing':~observed,'first':counts==0,'one_two':(counts>=1)&(counts<=2),'three_plus':counts>=3}
    report=dict(pairs={},layers={},means={},calibration={},actual_epochs={},unobserved_max_error={},no_confirmation_test=True,
        screening_sha256=sha256(ROOT/'screening.json'),code_sha256=sha256(Path(__file__)))
    fig,axs=plt.subplots(1,3,figsize=(15,4.5));colors={'availability':'#1f77b4','content':'#2ca02c'}
    differences=[]
    calfig,calaxs=plt.subplots(2,2,figsize=(9,8))
    for j,seed in enumerate(SEEDS):
        cache,_=augmented(data,tune,seed,arrays);original=new_model('content',seed);_,_,ref,_=prediction(original,cache);del original,cache
        curves={'original_V6':ref};report['actual_epochs'][str(seed)]={}
        for arm in ARMS:
            history=json.loads((ROOT/f'seed{seed}'/arm/'history.json').read_text());info=saved['fits'][str(seed)][arm]
            report['actual_epochs'][str(seed)][arm]={'actual':info['completed_epochs'],'best':info['best_epoch']}
            axs[j].plot([r['epoch'] for r in history],[r['tuning_ibs'] for r in history],label=arm,color=colors[arm])
            best=info['best_epoch'];axs[j].scatter(best,history[best]['tuning_ibs'],color=colors[arm],marker='D',s=30)
            with np.load(ROOT/f'seed{seed}'/arm/'tuning_predictions.npz') as z:curves[arm]=z['curve']
        axs[j].set(title=f'Seed {seed}: actual early-stopped epochs',xlabel='Epoch',ylabel='Tuning IBS7-365 (lower better)');axs[j].grid(alpha=.2);axs[j].legend(fontsize=8)
        report['calibration'][str(seed)]={}
        for name,curve in curves.items():
            report['calibration'][str(seed)][name]={}
            for cause in (1,2):
                for day in (30,90,180,365):
                    bins=calibration_bins(d.times[tune],d.events[tune],curve[:,:,cause],day=day,cause=cause)
                    report['calibration'][str(seed)][name][f'{cause}/{day}']=bins
                    if j==0 and day in (90,365):
                        ax=calaxs[cause-1,0 if day==90 else 1]
                        ax.plot([r['predicted'] for r in bins],[r['observed'] for r in bins],marker='o',markersize=3,
                            color='#777777' if name=='original_V6' else colors[name],label=name)
        for name in ('original_V6','availability'):
            report['pairs'][f'{seed}/content minus {name}']=paired_interval(d.arrays['subject_ids'][tune],d.times[tune],d.events[tune],
                curves['content'][:,:,1],curves[name][:,:,1],replicates=200,seed=seed)
        report['unobserved_max_error'][str(seed)]=float(np.abs(curves['content'][~observed]-ref[~observed]).max())
        if report['unobserved_max_error'][str(seed)]!=0:raise ValueError('unobserved source changed original risks')
        report['layers'][str(seed)]={}
        for group,mask in groups.items():
            ix=tune[mask];values={}
            for name in ('original_V6','availability','content'):
                values[name]={'readmission':public_metrics(metrics(d.times[ix],d.events[ix],curves[name][mask,:,1])),
                    'death':public_metrics(metrics(d.times[ix],d.events[ix],curves[name][mask,:,2],cause=2))}
            report['layers'][str(seed)][group]=dict(admissions=int(mask.sum()),patients=len(np.unique(d.arrays['subject_ids'][ix])),models=values)
        p=report['pairs'][f'{seed}/content minus original_V6'];differences.append((p['integrated_brier_difference'],p['auc90_difference']))
    for arm in (*ARMS,'original_V6'):
        ms=[saved['references'][str(s)] if arm=='original_V6' else saved['fits'][str(s)][arm]['metrics'] for s in SEEDS]
        report['means'][arm]=dict(ibs=float(np.mean([m['readmission']['integrated_brier_7_365'] for m in ms])),
            auc90=float(np.mean([m['readmission']['points']['90']['auc'] for m in ms])),
            death_ibs=float(np.mean([m['death']['integrated_brier_7_365'] for m in ms])),
            timepoints={str(t):dict(auc=float(np.mean([m['readmission']['points'][str(t)]['auc'] for m in ms])),
                brier=float(np.mean([m['readmission']['points'][str(t)]['brier'] for m in ms]))) for t in (7,14,30,60,90,180,365)})
    fig.suptitle('Frozen original V6; respiratory candidate facts: availability / content; early stopping',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(ROOT/'early_stopping.png',dpi=150);plt.close(fig)
    for i in range(2):
        for k,day in enumerate((90,365)):
            ax=calaxs[i,k];ax.plot([0,1],[0,1],'k--',linewidth=1)
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted cumulative probability',ylabel='Observed World-A fraction',
                   title=f'{"Readmission" if i==0 else "Death"}: day {day}');ax.grid(alpha=.2);ax.legend(fontsize=8)
    calfig.suptitle('Specified tuning set; predeclared seed 42; risk-bin descriptive calibration',fontsize=11)
    calfig.tight_layout(rect=(0,0,1,.95));calfig.savefig(ROOT/'calibration.png',dpi=150);plt.close(calfig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8));diff=np.asarray(differences)
    for k,ax in enumerate(axs):
        ax.axvline(0,color='black',linewidth=1);ax.scatter(diff[:,k],[-.12,0,.12],color=colors['content'],alpha=.6)
        ax.scatter(diff[:,k].mean(),0,marker='D',s=75,color=colors['content']);ax.set_yticks([0],['typed respiratory fact content'])
        ax.set_xlabel('Paired difference across three seeds');ax.grid(alpha=.2)
    axs[0].set_title('IBS: content minus original\n(lower better)');axs[1].set_title('AUC90: content minus original\n(higher better)')
    axs[0].set_xlim(-.002,.002);axs[1].set_xlim(-.02,.02)
    axs[0].plot([],[],marker='o',linestyle='',color=colors['content'],alpha=.6,label='Clinical seed')
    axs[0].plot([],[],marker='D',linestyle='',color=colors['content'],label='Mean across three seeds')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8,loc='lower center',ncol=2)
    fig.suptitle('Specified tuning set; fixed teacher; three clinical seeds / two teacher sources',fontsize=11)
    fig.tight_layout(rect=(0,.12,1,.9));fig.savefig(ROOT/'paired_differences.png',dpi=150);plt.close(fig)
    report['limits']=['tuning set exposed; selected best checkpoints','fixed-model bootstrap not inner refitting or independent cohort',
        'three clinical seeds reuse two teacher sources','fixed candidate concepts and explicit roles; retrospective FINAL; not medical gold',
        'World-A observed fractions not full out-of-system ascertainment; descriptive bins not prospective calibration proof']
    write_json(ROOT/'paired_readout.json',report)
    print(json.dumps(dict(means=report['means'],unobserved_max_error=report['unobserved_max_error'])),flush=True)


if __name__=='__main__':build_report()








````
<!-- END REVIEW_FRAGMENT -->

## training/v6/respiratory_transition_experiment.py · lines 1–229

<!-- REVIEW_FRAGMENT {"path":"training/v6/respiratory_transition_experiment.py","part":1,"start_line":1,"end_line":229,"source_sha256":"e72d463c3614608c10375fde99e12c2788e9e0243090a30fcff3170ed6d09425","fragment_sha256":"e72d463c3614608c10375fde99e12c2788e9e0243090a30fcff3170ed6d09425","added_terminal_newline":false} -->
````python
"""呼吸低频状态转换有限早停配对；原教师完全冻结，原风险起步。"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from patient_data.respiratory_transition import ROOT as DATA,derive,PROTOCOL
from training.v6.respiratory_transition_model import RespiratoryCorrection,RespBatch,ARMS,view
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.curve_adaptation import FrozenBatch,caches
from training.v6.module_surgery import probability_curve
from training.v6.module_surgery_experiment import summary,save
from training.v6.risk import competing_nll
from training.v6.evaluation import metrics
from training.v6.early_stopping import EarlyStopping
from training.v6.train import write_json

ROOT=Path('training_runs/v6_respiratory_transition_2026-10-13')
SEEDS=(42,7,101);CAP=60


def inputs():
    data=checked();d,c,h,ext,_,bins,tune,confirm,search,prior=data
    m=json.loads((DATA/'manifest.json').read_text());proof=json.loads((DATA/'independent_verification.json').read_text())
    if proof['output_sha256']!=sha256(DATA/'respiratory_transition.npz') or proof['output_sha256']!=m['output_sha256']:
        raise ValueError('source input not independently verified')
    for p,digest in m['code_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('source builder changed')
    with np.load(DATA/'respiratory_transition.npz') as z:arrays={k:z[k] for k in z.files}
    for key in ('hadm_ids','subject_ids','patient_splits'):np.testing.assert_array_equal(arrays[key],d.arrays[key])
    sources={**search['source_sha256'],**prior['source_sha256'],**m['code_sha256']}
    for p in ('training/v6/respiratory_transition_model.py','training/v6/respiratory_transition_experiment.py',
              'training/v6/curve_adaptation.py','training/v6/warm_residual_audit.py','training/v6/early_stopping.py',
              'training/tests/test_v6_respiratory_transition_model.py'):sources[p]=sha256(Path(p))
    train=np.flatnonzero(d.arrays['patient_splits']=='train')
    if len(train)!=13592 or len(tune)!=1531:raise ValueError('outer split changed')
    contract=dict(source_sha256=sources,protocol_sha256=sha256(PROTOCOL),data_manifest_sha256=sha256(DATA/'manifest.json'),
        data_sha256=m['output_sha256'],data_provenance=d.provenance,compact_provenance=c.provenance,
        teacher_hashes={str(s):sha256(REFERENCE/f'seed{s}/control/best.pt') for s in (42,7)},
        baseline_hashes={str(s):sha256(Path(search['input_config']['parent'])/f'seed{s}/survival_base/best.pt') for s in (42,7)},
        seeds=list(SEEDS),source_mapping={'42':42,'7':7,'101':42},arms=list(ARMS),new_parameters={'global':4,'other':2132},
        optimizer={'name':'AdamW','lr':.001,'weight_decay':1e-4,'delta_penalty':1e-4,'clip':1.,'batch':512},
        early_stopping={'patience':10,'min_epochs':15,'min_delta':1e-5,'max_epochs':60},
        selection='mean tuning IBS; content must beat original/global/availability/timing for clinical nomination',
        confirmation_test_evaluated=False,goal_achieved=False)
    return data,train,arrays,contract


def augmented(data,indices,seed,arrays):
    frozen,proof=caches(data,indices,seed)
    x={key:torch.cat([getattr(b,key) for b in frozen]) for key in ('hidden','raw','prior','times','events')}
    base=frozen[0].shape_raw
    if any(not torch.equal(base,b.shape_raw) for b in frozen):raise ValueError('base shape differs across cache')
    result=[];hashes=hashlib.sha256(indices.tobytes())
    features=torch.as_tensor(arrays['features'][indices],device='cuda');observed=torch.as_tensor(arrays['observed'][indices],device='cuda')
    for value in (*x.values(),base,features,observed):hashes.update(value.cpu().numpy().tobytes())
    for offset in range(0,len(indices),512):
        sl=slice(offset,offset+512);f=FrozenBatch(x['hidden'][sl],x['raw'][sl],x['prior'][sl],base,x['times'][sl],x['events'][sl])
        result.append(RespBatch(f,features[sl],observed[sl]))
    return result,dict(teacher_cache=proof,augmented_cache_sha256=hashes.hexdigest(),records=len(indices))


def rebuild_data(arrays):
    rebuilt,_=derive()
    for key,value in arrays.items():np.testing.assert_array_equal(value,rebuilt[key])


def new_model(arm,seed):
    m=RespiratoryCorrection(arm,seed).cuda();torch.manual_seed(seed);return m


@torch.no_grad()
def prediction(model,cache):
    a,k=[],[];loss,n=0.,0
    for b in cache:
        out=model(b);a.append(out.log_amplitude.cpu().numpy());k.append(out.shape.cpu().numpy())
        loss+=float(competing_nll(out,b.frozen.times,b.frozen.events))*len(b.frozen.times);n+=len(b.frozen.times)
    aa,kk=np.concatenate(a),np.concatenate(k)
    return aa,kk,probability_curve(aa,kk),loss/n


def score(model,cache):
    a,k,curve,nll=prediction(model,cache)
    t=np.concatenate([b.frozen.times.cpu().numpy() for b in cache]);e=np.concatenate([b.frozen.events.cpu().numpy() for b in cache])
    r=metrics(t,e,curve[:,:,1]);death=metrics(t,e,curve[:,:,2],cause=2)
    return dict(nll=nll,ibs=r['integrated_brier_7_365'],auc90=r['points']['90']['auc'],death_ibs=death['integrated_brier_7_365'])


def preflight():
    data,train,arrays,contract=inputs()
    if ROOT.exists():raise FileExistsError('preflight already exists; inspect')
    ROOT.mkdir();write_json(ROOT/'contract.json',contract);rebuild_data(arrays)
    cached,proof=augmented(data,train[:128],42,arrays);b=cached[0]
    prior=None
    for arm in ARMS:
        m=new_model(arm,42);out=m(b)
        if prior is None:prior=out
        else:
            torch.testing.assert_close(out.log_amplitude,prior.log_amplitude,atol=0,rtol=0)
            torch.testing.assert_close(out.shape,prior.shape,atol=0,rtol=0)
        # 用真实训练来源存在者做单步，不能只拿全缺失批次宣称梯度启动。
        del m
    use=train[np.asarray(arrays['observed'][train],bool)][:128]
    observed,_=augmented(data,use,42,arrays);probe=new_model('content',42)
    opt=torch.optim.AdamW(probe.parameters(),lr=.001);before={k:v.detach().clone() for k,v in probe.state_dict().items()}
    out=probe(observed[0]);loss=competing_nll(out,observed[0].frozen.times,observed[0].frozen.events)
    loss.backward();torch.nn.utils.clip_grad_norm_(probe.parameters(),1.);opt.step()
    if torch.equal(before['head.2.weight'],probe.state_dict()['head.2.weight']):raise ValueError('real observed source cannot update')
    feature=torch.as_tensor(arrays['features'],device='cuda')
    for arm in ('availability','timing'):np.testing.assert_array_equal(view(feature,arm).cpu().numpy(),arrays[arm])
    write_json(ROOT/'preflight.json',dict(status='passed',common_initial_risk_error=0.,original_cache_error=proof['teacher_cache']['initial_parameter_max_error'],
        source_rebuilt=True,actual_value_time_ablation_matches=True,real_observed_GPU_update=True,gpu=torch.cuda.get_device_name(0)))
    write_json(ROOT/'state.json',dict(status='preflight_passed',fits=0,goal_achieved=False))
    print(json.dumps(dict(status='preflight_passed',original_risk_error=0.)),flush=True)


def fit(model,train_cache,tune_cache,d,tune,folder,seed,arm):
    folder.mkdir(parents=True);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);early=EarlyStopping();history=[];start=time.perf_counter();reason='max_epochs'
    for epoch in range(CAP+1):
        online=None
        if epoch:
            total,n=0.,0;model.train()
            for j in rng.permutation(len(train_cache)):
                b=train_cache[j];opt.zero_grad(set_to_none=True);out=model(b)
                nll=competing_nll(out,b.frozen.times,b.frozen.events);loss=nll+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite respiratory training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('nonfinite respiratory gradient')
                opt.step();total+=float(nll.detach())*len(b.frozen.times);n+=len(b.frozen.times)
            online=total/n
        detail=score(model,tune_cache);save_best,stop=early.observe(epoch,detail['ibs'])
        history.append(dict(epoch=epoch,train_nll=online,**{'tuning_'+k:v for k,v in detail.items()},early_stale=early.stale,early_anchor=early.anchor))
        if save_best:torch.save(dict(state_dict=model.state_dict(),epoch=epoch,seed=seed,arm=arm),folder/'best.pt')
        write_json(folder/'history.json',history)
        if epoch%5==0 or stop:print(json.dumps(dict(seed=seed,arm=arm,epoch=epoch,best=early.best_epoch,ibs=detail['ibs'],early_stop=stop)),flush=True)
        if stop:reason='patience_exhausted';break
    model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
    a,k,curve,nll=prediction(model,tune_cache);save(folder/'tuning_predictions.npz',d,tune,a,k,curve)
    info=dict(arm=arm,seed=seed,completed_epochs=history[-1]['epoch'],best_epoch=early.best_epoch,stop_reason=reason,early_state=asdict(early),
        seconds=time.perf_counter()-start,checkpoint_sha256=sha256(folder/'best.pt'),metrics=summary(curve,d,tune,nll))
    write_json(folder/'training_result.json',info);return info


def lock(report):
    scores={arm:float(np.mean([report['fits'][str(s)][arm]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS])) for arm in ARMS}
    scores['original_V6']=float(np.mean([report['references'][str(s)]['readmission']['integrated_brier_7_365'] for s in SEEDS]))
    selected=min(scores,key=scores.get)
    return dict(selected=selected,scores=scores,nominated=['content'] if selected=='content' and scores['content']<min(v for k,v in scores.items() if k!='content') else [],
        selection='predeclared3-seed tuning IBS; availability/timing/global gain not numeric-content gain',goal_achieved=False)


def run():
    data,train,arrays,contract=inputs();d,_,_,_,_,_,tune,_,_,_=data
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    if json.loads((ROOT/'state.json').read_text())['status']!='preflight_passed':raise ValueError('already started; inspect')
    report=dict(fits={},references={},cache_hashes={},confirmation_test_not_evaluated=True,goal_achieved=False)
    for seed in SEEDS:
        tc,tp=augmented(data,train,seed,arrays);vc,vp=augmented(data,tune,seed,arrays)
        report['cache_hashes'][str(seed)]={'train':tp,'tune':vp}
        ref=new_model('content',seed);a,k,curve,nll=prediction(ref,vc)
        report['references'][str(seed)]=summary(curve,d,tune,nll);del ref
        report['fits'][str(seed)]={}
        for arm in ARMS:
            write_json(ROOT/'state.json',dict(status='training_with_early_stop',seed=seed,arm=arm,goal_achieved=False))
            m=new_model(arm,seed);report['fits'][str(seed)][arm]=fit(m,tc,vc,d,tune,ROOT/f'seed{seed}'/arm,seed,arm);del m
        del tc,vc;torch.cuda.empty_cache()
    write_json(ROOT/'screening.json',report);write_json(ROOT/'nomination_lock.json',lock(report))
    write_json(ROOT/'state.json',dict(status='training_complete_pending_independent_replay',fits=12,goal_achieved=False))
    print(json.dumps(lock(report)),flush=True)


def verify(tag):
    data,train,arrays,contract=inputs();d,_,_,_,_,_,tune,_,_,_=data
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    rebuild_data(arrays);report=json.loads((ROOT/'screening.json').read_text());errors={}
    for seed in SEEDS:
        tc,tp=augmented(data,train,seed,arrays);vc,vp=augmented(data,tune,seed,arrays)
        if report['cache_hashes'][str(seed)]!={'train':tp,'tune':vp}:raise ValueError('teacher/current source cache differs')
        ref=new_model('content',seed);a,k,curve,nll=prediction(ref,vc)
        if summary(curve,d,tune,nll)!=report['references'][str(seed)]:raise ValueError('reference differs')
        del ref,tc
        for arm in ARMS:
            folder=ROOT/f'seed{seed}'/arm;info=json.loads((folder/'training_result.json').read_text())
            if info!=report['fits'][str(seed)][arm] or sha256(folder/'best.pt')!=info['checkpoint_sha256']:raise ValueError('checkpoint differs')
            history=json.loads((folder/'history.json').read_text());early=EarlyStopping();first_stop=None
            for row in history:
                _,stop=early.observe(row['epoch'],row['tuning_ibs'])
                if stop and first_stop is None:first_stop=row['epoch']
            if asdict(early)!=info['early_state'] or early.best_epoch!=info['best_epoch'] or info['completed_epochs']!=len(history)-1:
                raise ValueError('early stop state/selection differs')
            if info['stop_reason']=='patience_exhausted' and first_stop!=info['completed_epochs']:raise ValueError('early stop bypassed')
            m=new_model(arm,seed);stored=torch.load(folder/'best.pt',map_location='cuda',weights_only=True)
            if stored['epoch']!=early.best_epoch:raise ValueError('wrong saved epoch')
            m.load_state_dict(stored['state_dict']);a,k,curve,nll=prediction(m,vc)
            with np.load(folder/'tuning_predictions.npz') as z:
                np.testing.assert_array_equal(z['times'],d.times[tune]);np.testing.assert_array_equal(z['events'],d.events[tune])
                error=max(float(np.abs(a-z['log_amplitude']).max()),float(np.abs(k-z['shape']).max()),float(np.abs(curve-z['curve']).max()))
            if error>3e-7 or summary(curve,d,tune,nll)!=info['metrics']:raise ValueError('prediction/metric differs')
            errors[f'{seed}/{arm}']=error;del m
        del vc
    if lock(report)!=json.loads((ROOT/'nomination_lock.json').read_text()):raise ValueError('selection lock differs')
    proof=dict(status='independently_verified',models=12,maximum_error=max(errors.values()),replay_errors=errors,
        report_sha256=sha256(ROOT/'screening.json'),contract_sha256=sha256(ROOT/'contract.json'),
        source_inputs_and_full_teacher_caches_rebuilt=True,early_stop_replayed=True,confirmation_test_not_evaluated=True)
    write_json(ROOT/f'verification_{tag}.json',proof)
    if all((ROOT/f'verification_{t}.json').exists() for t in ('a','b')):
        for t in ('a','b'):
            if json.loads((ROOT/f'verification_{t}.json').read_text())!=proof:raise ValueError('independent proof differs')
        write_json(ROOT/'state.json',dict(status='screening_complete_independently_verified',fits=12,nominated=lock(report)['nominated'],goal_achieved=False))
    print(json.dumps(proof),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('preflight','run','verify'),required=True);p.add_argument('--tag',choices=('a','b'),default='a');args=p.parse_args()
    try:
        if args.mode=='preflight':preflight()
        elif args.mode=='run':run()
        else:verify(args.tag)
    except Exception as e:
        if ROOT.exists():write_json(ROOT/f'failure_{args.mode}_{args.tag}.json',dict(error=repr(e),goal_achieved=False))
        raise


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/respiratory_transition_model.py · lines 1–51

<!-- REVIEW_FRAGMENT {"path":"training/v6/respiratory_transition_model.py","part":1,"start_line":1,"end_line":51,"source_sha256":"4efd553c4b9977d41ca97607279f540a23120feadd50f239eb84b314ad364dd1","fragment_sha256":"4efd553c4b9977d41ca97607279f540a23120feadd50f239eb84b314ad364dd1","added_terminal_newline":false} -->
````python
"""冻结原V6风险及32维状态，新96维呼吸事实作有限修正。"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.risk import RiskOutput

ARMS=('global','availability','timing','content')
AVAIL=[j*21+i for j in range(4) for i in (18,19)]+list(range(90,96))
TIMING=AVAIL+[j*21+i for j in range(4) for i in (5,6,8,14,15,17)]


@dataclass
class RespBatch:
    frozen: object
    features: torch.Tensor
    observed: torch.Tensor


def view(x,arm):
    if x.ndim!=2 or x.shape[1]!=96 or not torch.isfinite(x).all():raise ValueError('expected finite96 respiratory input')
    if arm=='content':return x
    if arm not in ('availability','timing'):raise ValueError('unknown feature view')
    keep=AVAIL if arm=='availability' else TIMING
    out=torch.zeros_like(x);out[:,keep]=x[:,keep]
    return out


class RespiratoryCorrection(nn.Module):
    def __init__(self,arm,seed=42):
        super().__init__()
        if arm not in ARMS:raise ValueError('invalid arm')
        self.arm=arm
        if arm=='global':self.raw=nn.Parameter(torch.zeros(4))
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed);self.head=nn.Sequential(nn.Linear(128,16),nn.GELU(),nn.Linear(16,4))
            nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,b):
        f=b.frozen
        if b.features.shape!=(len(f.hidden),96) or b.observed.shape!=(len(f.hidden),):raise ValueError('respiratory batch/index mismatch')
        if self.arm=='global':new=self.raw.expand(len(f.hidden),4)
        else:
            x=torch.cat((f.hidden,view(b.features,self.arm)),1)
            new=self.head(x)*b.observed[:,None]
        new=new.reshape(-1,2,2)
        delta=torch.stack((new[:,:,0],.25*torch.tanh(new[:,:,1])),2)
        original=f.raw.reshape(-1,2,2)
        return RiskOutput((f.prior+original[:,:,0]+delta[:,:,0]).clamp(-10,6),
            F.softplus(f.shape_raw+original[:,:,1]+delta[:,:,1])+.1,delta)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/respiratory_transition_report.py · lines 1–72

<!-- REVIEW_FRAGMENT {"path":"training/v6/respiratory_transition_report.py","part":1,"start_line":1,"end_line":72,"source_sha256":"88f766f1065a88fbeeb1922165f561d3c74141afac34f0a61497cc4b17521721","fragment_sha256":"88f766f1065a88fbeeb1922165f561d3c74141afac34f0a61497cc4b17521721","added_terminal_newline":false} -->
````python
"""已重放呼吸新增内容的配对、层内与图表；不改选点和默认模型。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patient_data.export_mimiciv import sha256
from training.v6.respiratory_transition_experiment import ROOT,SEEDS,ARMS,inputs,augmented,new_model,prediction
from training.v6.evaluation import paired_interval,metrics,public_metrics
from training.v6.train import write_json


def build_report():
    saved=json.loads((ROOT/'screening.json').read_text())
    for tag in ('a','b'):
        proof=json.loads((ROOT/f'verification_{tag}.json').read_text())
        if proof['report_sha256']!=sha256(ROOT/'screening.json') or proof['maximum_error']!=0:raise ValueError('not independently verified')
    data,_,arrays,_=inputs();d,_,_,_,_,_,tune,_,_,_=data
    observed=arrays['observed'][tune];counts=np.rint(np.expm1(d.history_raw[tune,2])).astype(int)
    groups={'source_observed':observed,'source_missing':~observed,'first':counts==0,'one_two':(counts>=1)&(counts<=2),'three_plus':counts>=3}
    report=dict(pairs={},layers={},means={},actual_epochs={},unobserved_max_error={},no_confirmation_test=True,
        screening_sha256=sha256(ROOT/'screening.json'),code_sha256=sha256(Path(__file__)))
    fig,axs=plt.subplots(1,3,figsize=(15,4.5));colors={'global':'#7f7f7f','availability':'#1f77b4','timing':'#ff7f0e','content':'#2ca02c'}
    differences=[]
    for j,seed in enumerate(SEEDS):
        cache,_=augmented(data,tune,seed,arrays);original=new_model('content',seed);_,_,ref,_=prediction(original,cache);del original,cache
        curves={'original_V6':ref};report['actual_epochs'][str(seed)]={}
        for arm in ARMS:
            history=json.loads((ROOT/f'seed{seed}'/arm/'history.json').read_text());info=saved['fits'][str(seed)][arm]
            report['actual_epochs'][str(seed)][arm]={'actual':info['completed_epochs'],'best':info['best_epoch']}
            axs[j].plot([r['epoch'] for r in history],[r['tuning_ibs'] for r in history],label=arm,color=colors[arm])
            best=info['best_epoch'];axs[j].scatter(best,history[best]['tuning_ibs'],color=colors[arm],marker='D',s=30)
            with np.load(ROOT/f'seed{seed}'/arm/'tuning_predictions.npz') as z:curves[arm]=z['curve']
        axs[j].set(title=f'Seed {seed}: actual early-stopped epochs',xlabel='Epoch',ylabel='Tuning IBS7-365 (lower better)');axs[j].grid(alpha=.2);axs[j].legend(fontsize=8)
        for name in ('original_V6','availability','timing'):
            report['pairs'][f'{seed}/content minus {name}']=paired_interval(d.arrays['subject_ids'][tune],d.times[tune],d.events[tune],
                curves['content'][:,:,1],curves[name][:,:,1],replicates=200,seed=seed)
        report['unobserved_max_error'][str(seed)]=float(np.abs(curves['content'][~observed]-ref[~observed]).max())
        if report['unobserved_max_error'][str(seed)]!=0:raise ValueError('unobserved source changed original risks')
        report['layers'][str(seed)]={}
        for group,mask in groups.items():
            ix=tune[mask];values={}
            for name in ('original_V6','timing','content'):
                values[name]={'readmission':public_metrics(metrics(d.times[ix],d.events[ix],curves[name][mask,:,1])),
                    'death':public_metrics(metrics(d.times[ix],d.events[ix],curves[name][mask,:,2],cause=2))}
            report['layers'][str(seed)][group]=dict(admissions=int(mask.sum()),patients=len(np.unique(d.arrays['subject_ids'][ix])),models=values)
        p=report['pairs'][f'{seed}/content minus original_V6'];differences.append((p['integrated_brier_difference'],p['auc90_difference']))
    for arm in (*ARMS,'original_V6'):
        ms=[saved['references'][str(s)] if arm=='original_V6' else saved['fits'][str(s)][arm]['metrics'] for s in SEEDS]
        report['means'][arm]=dict(ibs=float(np.mean([m['readmission']['integrated_brier_7_365'] for m in ms])),
            auc90=float(np.mean([m['readmission']['points']['90']['auc'] for m in ms])),
            death_ibs=float(np.mean([m['death']['integrated_brier_7_365'] for m in ms])),
            timepoints={str(t):dict(auc=float(np.mean([m['readmission']['points'][str(t)]['auc'] for m in ms])),
                brier=float(np.mean([m['readmission']['points'][str(t)]['brier'] for m in ms]))) for t in (7,14,30,60,90,180,365)})
    fig.suptitle('Frozen original V6; availability / timing / actual respiratory content; no forced100 epochs',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(ROOT/'early_stopping.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8));diff=np.asarray(differences)
    for k,ax in enumerate(axs):
        ax.axvline(0,color='black',linewidth=1);ax.scatter(diff[:,k],[-.12,0,.12],color=colors['content'],alpha=.6)
        ax.scatter(diff[:,k].mean(),0,marker='D',s=75,color=colors['content']);ax.set_yticks([0],['respiratory content'])
        ax.set_xlabel('Paired difference across three seeds');ax.grid(alpha=.2)
    axs[0].set_title('IBS: content minus original\n(lower better)');axs[1].set_title('AUC90: content minus original\n(higher better)')
    fig.suptitle('Specified tuning set; fixed teacher; three clinical seeds / two teacher sources',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(ROOT/'paired_differences.png',dpi=150);plt.close(fig)
    report['limits']=['tuning set exposed; selected best checkpoints','fixed-model bootstrap not inner refitting or independent cohort',
        'three clinical seeds reuse two teacher sources','local sources and sparse supports; not causal recovery']
    write_json(ROOT/'paired_readout.json',report)
    print(json.dumps(dict(means=report['means'],unobserved_max_error=report['unobserved_max_error'])),flush=True)


if __name__=='__main__':build_report()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/risk.py · lines 1–68

<!-- REVIEW_FRAGMENT {"path":"training/v6/risk.py","part":1,"start_line":1,"end_line":68,"source_sha256":"1c959ea10cd6ac633bf95f202734441a28c5d7fed6178427aa7688ef77916368","fragment_sha256":"1c959ea10cd6ac633bf95f202734441a28c5d7fed6178427aa7688ef77916368","added_terminal_newline":false} -->
````python
"""患者相关双原因 Weibull 联合似然与 CIF 数值积分。"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch


@dataclass(frozen=True)
class RiskOutput:
    log_amplitude: torch.Tensor  # [N,2]，不是传统 log(scale)
    shape: torch.Tensor          # [N,2]
    delta: torch.Tensor | None = None  # [N,2,2]，每原因幅度/raw shape修正


def competing_nll(output: RiskOutput, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    n = len(times)
    if (times.shape != (n,) or events.shape != (n,) or events.dtype != torch.long
            or output.log_amplitude.shape != (n, 2) or output.shape.shape != (n, 2) or n == 0):
        raise ValueError("invalid labels/parameters shape or event dtype")
    if (not torch.isfinite(times).all() or (times < 0).any() or (times > 365).any()
            or (events < 0).any() or (events > 2).any()
            or not torch.isfinite(output.log_amplitude).all()
            or not torch.isfinite(output.shape).all() or (output.shape <= 0).any()):
        raise ValueError("invalid time/event/risk parameter")
    # 沿用 V5 对记录精度为同日事件的最小时间约定，不据此重写标签。
    scaled = times.clamp_min(1e-3).unsqueeze(-1) / 365.
    a, k = output.log_amplitude, output.shape
    cumulative = torch.exp(a) * scaled.pow(k)
    log_hazard = a + torch.log(k) - math.log(365.) + (k - 1) * torch.log(scaled)
    selected = log_hazard.gather(1, (events - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (cumulative.sum(1) - torch.where(events > 0, selected, 0.)).mean()


def predict_cif(log_amplitude: np.ndarray, shape: np.ndarray, days: np.ndarray,
                *, step_days: float = .25) -> np.ndarray:
    """[N,T,3] 为 S/F1/F2；步内按累计风险增量分配事件质量，须验证步长收敛。"""
    a, k, query = (np.asarray(x, dtype=np.float64) for x in (log_amplitude, shape, days))
    if (a.ndim != 2 or a.shape[1] != 2 or not len(a) or k.shape not in ((2,), a.shape)
            or query.ndim != 1 or not len(query) or query[0] != 0
            or np.any(np.diff(query) <= 0) or query[-1] > 365
            or not math.isfinite(step_days) or step_days <= 0
            or not all(np.isfinite(x).all() for x in (a, k, query)) or (k <= 0).any()):
        raise ValueError("invalid CIF parameters/grid")
    amplitude = np.exp(a)
    if not np.isfinite(amplitude).all():
        raise ValueError("nonfinite amplitude")
    k = np.broadcast_to(k, a.shape)
    survival, cif = np.ones(len(a)), np.zeros_like(a)
    result = np.empty((len(a), len(query), 3))
    result[:, 0] = np.column_stack((survival, cif))
    previous_h = np.zeros_like(a)
    # 每个要求时间点必须是精确网格边界，不对最终 CIF 随意插值。
    for q in range(1, len(query)):
        grid = np.linspace(query[q - 1], query[q], max(1, math.ceil((query[q] - query[q - 1]) / step_days)) + 1)
        if query[q - 1] == 0:
            # k<1 的 hazard 在零点可奇异；首个时间步细化，而非直接计算 h(0)。
            grid = np.unique(np.r_[grid, np.geomspace(1e-10, min(step_days, query[q]), 80)])
        for t in grid[1:]:
            cumulative = amplitude * (t / 365.) ** k
            increment = np.maximum(0, cumulative - previous_h)
            total = increment.sum(1)
            share = np.divide(increment, total[:, None], out=np.zeros_like(increment), where=total[:, None] > 0)
            cif += survival[:, None] * (-np.expm1(-total))[:, None] * share
            survival *= np.exp(-total)
            previous_h = cumulative
        result[:, q] = np.column_stack((survival, cif))
    return result
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/run_token_suite.py · lines 1–54

<!-- REVIEW_FRAGMENT {"path":"training/v6/run_token_suite.py","part":1,"start_line":1,"end_line":54,"source_sha256":"30ebb9da9fbbe252b556395511c14a57d92916ea6f7f2a4d3fbf4e362ba26a66","fragment_sha256":"30ebb9da9fbbe252b556395511c14a57d92916ea6f7f2a4d3fbf4e362ba26a66","added_terminal_newline":false} -->
````python
"""可重复完整入口：旧已哈希训练代码不改，显式注入数值容差核验适配器。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6 import token_experiment as engine
from training.v6.token_model import fast_cif,daily_probabilities
from training.v6.evaluation import metrics,public_metrics,calibration_bins


def tolerant_evaluation(model,cache,data,indices,path):
    a,k,nll=engine.parameters(model,cache);curve=fast_cif(a,k,device='cuda');daily,conditional=daily_probabilities(curve)
    model.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
    aa,kk,_=engine.parameters(model,cache)
    if max(np.abs(a-aa).max(),np.abs(k-kk).max())>2e-6 or np.abs(curve-fast_cif(aa,kk,device='cuda')).max()>2e-7:
        raise ValueError('checkpoint replay exceeds explicit GPU float tolerance')
    detail=dict(nll=nll,readmission=public_metrics(metrics(data.times[indices],data.events[indices],curve[:,:,1],ordering=True)),
        death=public_metrics(metrics(data.times[indices],data.events[indices],curve[:,:,2],cause=2,ordering=True)),
        calibration={str(day):calibration_bins(data.times[indices],data.events[indices],curve[:,:,1],day=day) for day in (30,90,180,365)})
    return curve,detail,dict(log_amplitude=a,shape=k,curve=curve,daily_event_probability=daily,
        conditional_daily_event_probability=conditional,conditional_daily_valid=curve[:,:-1,0]>1e-12)


def main():
    p=argparse.ArgumentParser()
    defaults=dict(reference='training_runs/v6_history_anchored_bert_seed42_2026-10-09',parent='training_runs/v6_90day_objective_2026-10-11',
        compact_dir='patient_data/processed/v6_compact_multisource_2026-10-09',extension='patient_data/processed/v6_detail_extension_2026-10-11_r2',
        medication='patient_data/processed/v6_medication_identity_2026-10-12')
    for name,value in defaults.items():p.add_argument('--'+name.replace('_','-'),type=Path,default=Path(value))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=100);p.add_argument('--ssl-epochs',type=int,default=20)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,7]);a=p.parse_args()
    if a.seeds!=[42,7] or a.epochs!=100:raise ValueError('complete suite currently contracted to two seeds and 100 epochs')
    original=engine.evaluate_model
    # 这是显式兼容接口，不热改训练、模型或历史源文件；仅替换逐位相等验收。
    engine.evaluate_model=tolerant_evaluation
    try:engine.run(a)
    finally:engine.evaluate_model=original
    manifest=dict(runner_sha256=sha256(Path(__file__)),engine_sha256=sha256(Path(engine.__file__)),
        evaluation_adapter='parameter atol2e-6; probability atol2e-7; no training changes')
    engine.write_json(a.output/'execution_adapter.json',manifest)
    from training.v6.token_finalize import finalize
    from training.v6.token_verify import verify
    from training.v6.token_cox_control import run as cox
    from training.v6.token_cox_verify import verify as verify_cox
    from training.v6.reconstruction_controls import run as reconstruction
    from training.v6.token_choose_reference import choose
    from training.v6.token_result_figures import render
    finalize(a.output);verify(a.output);cox(a.output);verify_cox(a.output);reconstruction(a.output);choose(a.output);render(a.output)
    print(json.dumps(dict(status='complete_verified_suite',output=str(a.output))))


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/saved_section_experiment.py · lines 1–268

<!-- REVIEW_FRAGMENT {"path":"training/v6/saved_section_experiment.py","part":1,"start_line":1,"end_line":268,"source_sha256":"41f442025a838350a42da97ed4db55b0c4ad0966d2e5e43ea5862ca723f56d03","fragment_sha256":"41f442025a838350a42da97ed4db55b0c4ad0966d2e5e43ea5862ca723f56d03","added_terminal_newline":false} -->
````python
"""已核对原文的A完整章节/同资产四字段配对试验，不改旧模型和报告。"""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from patient_data.export_mimiciv import sha256
from patient_data.align_saved_sections import ROLES, FOUR
from training.v6.compact_ablation import inputs
from training.v6.compact_context_data import attach
from training.v6.compact_model import CompactClinicalEncoder, CompactAnchoredModel
from training.v6.architecture_experiment import batches
from training.v6.horizon_objective import restore, labels90, fit90, predict90, binary_metrics, paired90, ranked_curves
from training.v6.evaluation import GRID, metrics, public_metrics
from training.v6.risk import predict_cif
from training.v6.train import write_json

ARMS = ('none', 'four', 'clinical9', 'full10')
PROTOCOL = Path('docs/V6_FULL_SECTION_COMPARISON_PROTOCOL_2026-10-11.md')


class SectionEncoder(CompactClinicalEncoder):
    def __init__(self):
        super().__init__(768)
        self.field_ids = nn.Embedding(len(ROLES), self.width)

    def _text(self, batch, n, *, mask_pmh=False):
        v, ix = batch.snippet_vectors, batch.snippet_index
        fields = len(ROLES)
        if (v.ndim != 2 or v.shape[1] != 768 or ix.shape != (len(v), 2)
            or ix.dtype != torch.long or (ix < 0).any() or (ix[:,0] >= n).any()
            or (ix[:,1] >= fields).any() or not torch.isfinite(v).all()):
            raise ValueError('invalid A section vectors/index')
        if mask_pmh:
            keep = ix[:,1] != ROLES.index('past_medical_history')
            v, ix = v[keep], ix[keep]
        if not len(v):
            return v.new_zeros((n,self.width)), torch.zeros(n,device=v.device,dtype=torch.bool)
        content = self.text_projection(v)+self.field_ids(ix[:,1])
        group = fields*ix[:,0]+ix[:,1]
        score = self.text_score(content).squeeze(-1)
        maximum = score.new_full((fields*n,),-torch.inf)
        maximum.scatter_reduce_(0,group,score,reduce='amax',include_self=True)
        weight = torch.exp(score-maximum[group])
        total = score.new_zeros(fields*n).scatter_add_(0,group,weight)
        weight = weight/total[group]
        pooled = content.new_zeros((fields*n,self.width)).index_add_(0,group,weight[:,None]*content)
        present = (total>0).reshape(n,fields)
        result = pooled.reshape(n,fields,self.width).sum(1)/present.sum(1).clamp_min(1)[:,None]
        return result,present.any(1)


def load_inputs(args):
    d,c,hist,_ = inputs(args.reference,args.compact_dir)
    parent = json.loads((args.parent/'report.json').read_text())
    proof = json.loads((args.parent/'independent_verification.json').read_text())
    if proof['report_sha256'] != sha256(args.parent/'report.json'): raise ValueError('parent unverified')
    if d.provenance != parent['contract']['data_provenance'] or c.provenance != parent['contract']['compact_provenance']:
        raise ValueError('parent input changed')
    for p,digest in parent['contract']['code_sha256'].items():
        if sha256(Path(p)) != digest: raise ValueError('parent code changed: '+p)
    manifest = json.loads((args.aligned/'manifest.json').read_text())
    if (manifest['status'] != 'current_discharge_source_and_complete_section_coverage_verified'
        or tuple(manifest['roles']) != ROLES or manifest['state_manifest_sha256'] != d.provenance['state_manifest_sha256']):
        raise ValueError('A alignment contract changed')
    for p,digest in manifest['source_sha256'].items():
        if sha256(Path(p.replace('\\','/'))) != digest: raise ValueError('A source changed: '+p)
    for p,digest in manifest['output_sha256'].items():
        if sha256(args.aligned/p) != digest: raise ValueError('A aligned output changed')
    with np.load(args.aligned/'alignment.npz') as z: a={k:z[k] for k in z.files}
    for k in ('hadm_ids','subject_ids','patient_splits'):
        if not np.array_equal(a[k],d.arrays[k]): raise ValueError('aligned member/split differs')
    v = np.load(args.aligned/'vectors.float16.npy',mmap_mode='r')
    refs = a['refs']
    if v.shape != (len(refs),768) or refs.shape != (len(v),3) or not np.isfinite(v).all():
        raise ValueError('aligned array shape/value differs')
    if (not np.array_equal(refs[:,2],np.arange(len(v))) or (refs[:,:2]<0).any()
        or (refs[:,0]>=len(d.history)).any() or (refs[:,1]>=len(ROLES)).any()):
        raise ValueError('invalid aligned row mapping')
    members = np.unique(refs[:,0])
    ix = {f:members[d.arrays['patient_splits'][members]==f] for f in ('train','validation','test')}
    views, contexts = {}, {}
    for arm in ARMS:
        ids = () if arm=='none' else FOUR if arm=='four' else tuple(range(9)) if arm=='clinical9' else tuple(range(10))
        mapping=[[] for _ in d.history]
        for position,field,row in refs:
            if field in ids: mapping[position].append((int(field),int(row)))
        views[arm]=replace(d,text_vectors=v,text_refs=tuple(tuple(x) for x in mapping))
        contexts[arm]=replace(c,base=views[arm])
    return d,views,contexts,hist,ix,parent,manifest


def model_for(args,seed,hist):
    base = restore(args.parent,seed,'survival_base',hist,.1)
    torch.manual_seed(seed)
    m = CompactAnchoredModel(copy.deepcopy(base),768)
    m.encoder = SectionEncoder()
    return m.cuda()


def prepared(views,contexts,ix,arm):
    return batches(views[arm],contexts[arm],ix,512,expanded=True,text='bert')


def prediction(root,seed,arm,fold):
    with np.load(root/f'seed{seed}'/arm/(fold+'_predictions.npz')) as z: return {k:z[k] for k in z.files}


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2); started=time.perf_counter()
    d,views,contexts,hist,ix,parent,manifest = load_inputs(args)
    y=labels90(d.times,d.events); args.output.mkdir(parents=True)
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    contract=dict(parent_report_sha256=sha256(args.parent/'report.json'),
        alignment_manifest_sha256=sha256(args.aligned/'manifest.json'),
        data_provenance=d.provenance,compact_provenance=contexts['none'].provenance,
        protocol_sha256=sha256(PROTOCOL),code_sha256={**parent['contract']['code_sha256'],
            'training/v6/saved_section_experiment.py':sha256(Path(__file__)),
            'patient_data/align_saved_sections.py':sha256(Path('patient_data/align_saved_sections.py'))},
        seeds=[42,7],arms=list(ARMS),selection='validation Brier90',epochs=60,patience=10,lr=3e-4,
        device=torch.cuda.get_device_name(0),folds={f:dict(admissions=len(p),patients=len(np.unique(d.arrays['subject_ids'][p])),
            positives=int(y[p].sum())) for f,p in ix.items()},sections=manifest['sections'])
    write_json(args.output/'contract.json',contract)
    fitted={}
    for seed in (42,7):
        (args.output/f'seed{seed}').mkdir(); common=None
        for arm in ARMS:
            m=model_for(args,seed,hist); initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            if common is None: common=initial
            elif not all(torch.equal(v,common[k]) for k,v in initial.items()): raise ValueError('arm initial states differ')
            cache={f:prepared(views,contexts,ix[f],arm) for f in ('train','validation')}
            write_json(args.output/'status.json',dict(status='training',seed=seed,arm=arm))
            fitted[seed,arm]=fit90(m,cache['train'],cache['validation'],y[ix['validation']],args.output/f'seed{seed}'/arm,
                epochs=60,lr=3e-4,seed=seed,patience=10,warmup=5)
            del cache,m
    write_json(args.output/'status.json',dict(status='all8_checkpoints_selected_before_test'))
    models,comparisons={},{ }
    for seed in (42,7):
        models[str(seed)]={}; comparisons[str(seed)]={}
        for arm in ARMS:
            m=model_for(args,seed,hist)
            m.load_state_dict(torch.load(args.output/f'seed{seed}'/arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            models[str(seed)][arm]={**fitted[seed,arm],'folds':{}}
            for f in ('validation','test'):
                idx=ix[f]; p=predict90(m,prepared(views,contexts,idx,arm))
                p['curve']=predict_cif(p['parameters'],p['shape'],GRID)
                if (not np.allclose(p['curve'].sum(2),1,atol=1e-9) or (np.diff(p['curve'][:,:,1:],axis=1)<-1e-10).any()
                    or not np.allclose(p['risk90'],p['curve'][:,90,1],atol=1e-8)):
                    raise ValueError('CIF contract failed')
                detail=binary_metrics(y[idx],p['risk90'])
                detail['survival_readmission']=public_metrics(metrics(d.times[idx],d.events[idx],p['curve'][:,:,1],ordering=True))
                detail['survival_death']=public_metrics(metrics(d.times[idx],d.events[idx],p['curve'][:,:,2],cause=2,ordering=True))
                models[str(seed)][arm]['folds'][f]=detail
                np.savez_compressed(args.output/f'seed{seed}'/arm/(f+'_predictions.npz'),
                    hadm_ids=d.arrays['hadm_ids'][idx],subject_ids=d.arrays['subject_ids'][idx],times=d.times[idx],events=d.events[idx],
                    y90=y[idx],**{k:v for k,v in p.items() if k!='loss'})
            del m
        for f in ('validation','test'):
            comparisons[str(seed)][f]={}
            for arm in ARMS[1:]:
                p=prediction(args.output,seed,arm,f)
                for other in ('none','four'):
                    if other==arm: continue
                    base=prediction(args.output,seed,other,f)
                    comparisons[str(seed)][f][arm+'_minus_'+other]=paired90(p['subject_ids'],p['y90'],p['risk90'],base['risk90'],seed=seed)
    report=dict(status='completed_source_aligned_same_asset_8_models',contract=contract,models=models,comparisons=comparisons,
        elapsed_seconds=time.perf_counter()-started,all_arm_initial_states_equal=True,
        limits=['Reused test; no multiplicity adjustment; matched subset only, not all19401.',
            'Full means every natural section of one saved selected note, not all database notes.',
            'Baseline trained on full training fold; clinical correction trained on matched subset.',
            'Old A full/four prefix differs from current B recipe; no direct cross-population attribution.'])
    write_json(args.output/'report.json',report); render(args.output,report)
    write_json(args.output/'status.json',dict(status=report['status'],seconds=report['elapsed_seconds']))
    print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'])),flush=True)


def verify(args):
    report=json.loads((args.output/'report.json').read_text()); contract=report['contract']
    for p,digest in contract['code_sha256'].items():
        if sha256(Path(p))!=digest: raise ValueError('source code changed')
    if sha256(PROTOCOL)!=contract['protocol_sha256']: raise ValueError('protocol changed')
    d,views,contexts,hist,ix,_,_=load_inputs(args); y=labels90(d.times,d.events); errors={}
    if sha256(args.aligned/'manifest.json')!=contract['alignment_manifest_sha256']: raise ValueError('alignment changed')
    for seed in (42,7):
        for arm in ARMS:
            path=args.output/f'seed{seed}'/arm/'best.pt'
            if sha256(path)!=report['models'][str(seed)][arm]['checkpoint_sha256']: raise ValueError('checkpoint changed')
            m=model_for(args,seed,hist); frozen=copy.deepcopy(m.baseline.state_dict())
            m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
            if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in frozen.items()): raise ValueError('baseline changed')
            for f in ('validation','test'):
                p=predict90(m,prepared(views,contexts,ix[f],arm)); saved=prediction(args.output,seed,arm,f)
                for k,val in (('hadm_ids',d.arrays['hadm_ids'][ix[f]]),('subject_ids',d.arrays['subject_ids'][ix[f]]),
                              ('times',d.times[ix[f]]),('events',d.events[ix[f]]),('y90',y[ix[f]])):
                    if not np.array_equal(saved[k],val): raise ValueError('identity/labels changed')
                for k,val in p.items():
                    if k!='loss' and not np.allclose(saved[k],val,atol=1e-7): raise ValueError('replay mismatch')
                curve=predict_cif(p['parameters'],p['shape'],GRID)
                if not np.allclose(saved['curve'],curve,atol=1e-7): raise ValueError('curve replay mismatch')
                fine=predict_cif(p['parameters'][:64],p['shape'][:64],GRID,step_days=.125)
                if np.abs(fine-curve[:64]).max()>2e-4: raise ValueError('integration convergence failed')
                measured=binary_metrics(y[ix[f]],p['risk90'])
                for k in ('auc90','average_precision','brier90','logloss90'):
                    if abs(measured[k]-report['models'][str(seed)][arm]['folds'][f][k])>1e-7: raise ValueError('metric replay mismatch')
                errors[f'{seed}/{arm}/{f}']=float(np.abs(saved['risk90']-p['risk90']).max())
            del m
    write_json(args.output/'independent_verification.json',dict(status='all8_models_all16_folds_independently_reloaded',
        report_sha256=sha256(args.output/'report.json'),frozen_baseline_unchanged=True,errors=errors))
    print('all8_models_all16_folds_independently_reloaded',flush=True)


def render(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures'; folder.mkdir()
    labels=dict(none='No text',four='Same-asset four fields',clinical9='Nine clinical sections',full10='Full ten sections')
    colors=dict(none='#888888',four='#2674ae',clinical9='#2a925b',full10='#cf7722')
    for seed in (42,7):
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        for arm in ARMS:
            p=prediction(root,seed,arm,'test'); m=report['models'][str(seed)][arm]['folds']['test']
            c=ranked_curves(p['y90'],p['risk90']); bins=m['calibration_bins']
            axes[0].plot(c['fpr'],c['tpr'],color=colors[arm],label=f'{labels[arm]}: AUC90={m["auc90"]:.3f}')
            actual=metrics(p['times'],p['events'],p['curve'][:,:,1])
            axes[1].plot(GRID[7:],actual['auc_curve'][7:],color=colors[arm],label=labels[arm])
            axes[2].plot([b['predicted'] for b in bins],[b['observed'] for b in bins],'o-',color=colors[arm],label=labels[arm])
        axes[0].plot([0,1],[0,1],'k--',alpha=.3); axes[2].plot([0,1],[0,1],'k--',alpha=.3)
        axes[2].set(xlim=(0,.7),ylim=(0,.7))
        for ax,title,x,y in zip(axes,('90-day ROC','Time-dependent AUC(t)','90-day Calibration'),
            ('False positive rate','Days after discharge','Mean predicted risk'),('True positive rate','AUC(t)','Observed fraction')):
            ax.set(title=title,xlabel=x,ylabel=y); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle(f'Same A encodings and patients; seed{seed}; reused test, not blind')
        fig.tight_layout(); fig.savefig(folder/f'test_seed{seed}_curves.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
    pairs=('four_minus_none','clinical9_minus_four','full10_minus_four')
    for j,pair in enumerate(pairs):
        for i,seed in enumerate((42,7)):
            results=report['comparisons'][str(seed)]['test'][pair]['metrics']
            for ax,k in zip(axes,('brier90','auc90')):
                r=results[k]; yy=j+(i-.5)*.15; color=('#2674ae','#cf7722')[i]
                ax.plot(r['ci95'],[yy,yy],color=color); ax.scatter(r['difference'],yy,color=color,label=f'seed{seed}' if j==0 else None)
    for ax in axes:
        ax.axvline(0,color='k',lw=1); ax.grid(axis='x',alpha=.2); ax.legend(); ax.set_yticks(range(3),pairs)
    axes[0].invert_yaxis(); axes[0].set_xlabel('Brier90 paired difference (lower better)')
    axes[1].set_xlabel('AUC90 paired difference (higher better)')
    fig.suptitle('Matched patients; 200 patient-clustered 95% intervals; no multiplicity adjustment')
    fig.tight_layout(); fig.savefig(folder/'test_paired.png',dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--compact-dir',type=Path,default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--parent',type=Path,default=Path('training_runs/v6_90day_objective_2026-10-11'))
    p.add_argument('--aligned',type=Path,default=Path('patient_data/processed/v6_saved_sections_aligned_2026-10-11'))
    p.add_argument('--output',type=Path,default=Path('training_runs/v6_saved_section_comparison_2026-10-11'))
    p.add_argument('--verify',action='store_true')
    args=p.parse_args(); (verify if args.verify else run)(args)


if __name__=='__main__': main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/smoke.py · lines 1–87

<!-- REVIEW_FRAGMENT {"path":"training/v6/smoke.py","part":1,"start_line":1,"end_line":87,"source_sha256":"98497fdd7460abd9be6227f21d11b16372b804dcf6016c99489b5c62370c5145","fragment_sha256":"98497fdd7460abd9be6227f21d11b16372b804dcf6016c99489b5c62370c5145","added_terminal_newline":false} -->
````python
"""真实数据接口/GPU冒烟：不做优化器更新、不保存风险模型检查点。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.data import load
from training.v6.model import AnchoredModel, HistoryBaseline
from training.v6.risk import competing_nll, predict_cif


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'state_dir', 'labels', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--field-dir', type=Path, required=True)
    parser.add_argument('--vectors-path', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU verification requested; CPU fallback forbidden')
    torch.set_num_threads(1)
    data = load(args.snapshot, args.state_dir, args.labels, field_dir=args.field_dir, vectors_path=args.vectors_path)
    torch.manual_seed(42)
    baseline = HistoryBaseline()
    model = AnchoredModel(baseline, data.text_vectors.shape[1]).cuda().eval()
    torch.manual_seed(42)  # 文本维度变化后仍恢复后续随机流。
    reports = {}
    for fold in ('train', 'validation', 'test'):
        candidates = np.flatnonzero(data.arrays['patient_splits'] == fold)
        # 固定分散取样用于接口验收；不是评估模型性能或按结局挑人。
        positions = candidates[np.linspace(0, len(candidates) - 1, min(64, len(candidates)), dtype=int)]
        history, batch, times, events = data.collate(positions, 'cuda')
        model.zero_grad(set_to_none=True)
        output = model(history, batch)
        loss = competing_nll(output, times, events)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None for p in baseline.parameters()):
            raise ValueError('nonfinite loss or history gradient leakage')
        expected = baseline(history)
        if not torch.equal(output.log_amplitude, expected.log_amplitude) or not torch.equal(output.shape, expected.shape):
            raise ValueError('zero initialized risk differs from baseline')
        _, branches = model.encoder(batch)
        with torch.no_grad():
            pmh_vector = batch.snippet_vectors.clone()
            pmh_vector[batch.snippet_index[:, 1] == 1] = 1000
            from dataclasses import replace
            left, _ = model.encoder(batch, mask_pmh=True)
            repeated, _ = model.encoder(batch, mask_pmh=True)
            right, _ = model.encoder(replace(batch, snippet_vectors=pmh_vector), mask_pmh=True)
            pmh_difference = float((left - right).abs().max().cpu())
            repeat_difference = float((left - repeated).abs().max().cpu())
            # CUDA scatter/index_add 的并行求和次序可能造成末位浮点差异。
            # 同时报告未改输入的重复误差，不把容差称为完全位级确定性。
            if not torch.allclose(left, right, rtol=1e-5, atol=1e-6):
                raise ValueError(f'PMH mask not effective: changed={pmh_difference}, repeat={repeat_difference}')
        a, k = (x.detach().cpu().numpy() for x in (output.log_amplitude, output.shape))
        curve = predict_cif(a, k, np.array([0., 7., 30., 90., 365.]))
        if not np.allclose(curve.sum(2), 1, atol=1e-10):
            raise ValueError('probability conservation failed')
        reports[fold] = {'admissions_checked': len(positions), 'snippet_rows': len(batch.snippet_vectors),
                         'finite_initial_loss': float(loss.detach().cpu()),
                         'initial_baseline_equal': True, 'history_gradients_none': True,
                         'pmh_mask_invariance': True,
                         'pmh_mask_max_absolute_difference': pmh_difference,
                         'identical_input_repeat_max_absolute_difference': repeat_difference,
                         'maximum_probability_sum_error': float(np.abs(curve.sum(2) - 1).max()),
                         'branch_mean_l2_norm': {name: float(branches[name].detach().norm(dim=1).mean().cpu())
                                                 for name in ('structured', 'lab', 'medication', 'text')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'real_input_gpu_smoke_passed_not_trained', 'folds': reports,
              'device': torch.cuda.get_device_name(0), 'torch_version': torch.__version__,
              'optimizer_steps': 0, 'checkpoint_saved': False, 'provenance': data.provenance,
              'code_sha256': {name: sha256(Path(__file__).parent / name) for name in ('model.py', 'risk.py', 'data.py', 'smoke.py')},
              'limitations': ['Baseline is random initialization, not trained; loss is not model performance',
                              'Branch norms indicate numerical flow, not predictive contribution',
                              'Historical test fold used only for schema verification in this smoke']}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'device': report['device'], 'optimizer_steps': 0,
                      'checked_admissions': sum(r['admissions_checked'] for r in reports.values())}))


if __name__ == '__main__':
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/source_context_data.py · lines 1–75

<!-- REVIEW_FRAGMENT {"path":"training/v6/source_context_data.py","part":1,"start_line":1,"end_line":75,"source_sha256":"7ed766e2ef7d4aa47aa9955e5b5f61d1ea279c7c613f19a2c98cef74d16b5560","fragment_sha256":"7ed766e2ef7d4aa47aa9955e5b5f61d1ea279c7c613f19a2c98cef74d16b5560","added_terminal_newline":false} -->
````python
"""显式选择补充来源的 V6 输入接口；不静默修改旧实验配方、不构造标签。"""
from __future__ import annotations
from dataclasses import dataclass, replace
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256


@dataclass(frozen=True)
class SourceContextBatch:
    clinical: object               # 原 ClinicalBatch，仅给药 source_mask 修正
    eos_features: object           # [N,2,20]，6值+6mask+首次/末次四类时间上下文
    micro_values: object           # [N,6]，未观察不等于阴性
    micro_masks: object
    resp_numeric_scaled: object    # [N,8,5]，不是连续时序
    resp_numeric_masks: object
    support_values: object         # [N,9]，记录证据，不推断急性加重病因
    support_masks: object
    recorded_duration_hours: object # [N,2] 审计值，未归一化，不默认送网络
    recorded_duration_masks: object


def checked_arrays(directory, status, filename, base):
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!=status: raise ValueError('supplement unfinished')
    parent=manifest.get('parent_clinical_state_manifest_sha256',manifest.get('parent_state_manifest_sha256'))
    if parent!=base.provenance['state_manifest_sha256']: raise ValueError('supplement/base lineage differs')
    for name,digest in manifest['output_sha256'].items():
        if sha256(directory/name)!=digest: raise ValueError('supplement output changed')
    with np.load(directory/filename,allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    for key in ('hadm_ids','subject_ids','patient_splits'):
        if not np.array_equal(arrays[key],base.arrays[key]): raise ValueError('member, order or patient split differs')
    return arrays,manifest


@dataclass(frozen=True)
class SourceContextData:
    base: object
    eos: dict
    resp_micro: dict
    provenance: dict

    def collate(self,positions,device='cpu'):
        import torch
        h,clinical,t,e=self.base.collate(positions,device=device)
        positions=np.asarray(positions,dtype=np.int64)
        tensor=lambda array:torch.as_tensor(np.asarray(array)[positions],device=device)
        clinical=replace(clinical,med_source_mask=tensor(self.eos['med_source_mask']))
        batch=SourceContextBatch(clinical,tensor(self.eos['eos_features']),
            tensor(self.resp_micro['micro_values']),tensor(self.resp_micro['micro_masks']),
            tensor(self.resp_micro['resp_numeric_scaled']).float(),tensor(self.resp_micro['resp_numeric_masks']),
            tensor(self.resp_micro['support_values']),tensor(self.resp_micro['support_masks']),
            tensor(self.resp_micro['recorded_duration_hours']),tensor(self.resp_micro['recorded_duration_masks']))
        return h,batch,t,e


def attach(base,eos_dir:Path,resp_micro_dir:Path):
    eos,em=checked_arrays(eos_dir,'eos_med_context_validated_not_trained','eos_med_context.npz',base)
    rm,mm=checked_arrays(resp_micro_dir,'resp_micro_context_validated_not_trained','resp_micro_context.npz',base)
    n=len(base.arrays['hadm_ids'])
    shapes={'eos_features':(n,2,20),'med_source_mask':(n,7),'micro_values':(n,6),'micro_masks':(n,6),
            'resp_numeric_scaled':(n,8,5),'resp_numeric_masks':(n,8,5),'support_values':(n,9),
            'support_masks':(n,9),'recorded_duration_hours':(n,2),'recorded_duration_masks':(n,2)}
    for key,shape in shapes.items():
        value=(eos if key in eos else rm)[key]
        if value.shape!=shape or not np.isfinite(value).all(): raise ValueError('invalid supplementary tensor')
        if 'mask' in key and value.dtype!=bool: raise ValueError('invalid mask type')
    if not np.array_equal(eos['med_exposure'],base.arrays['med_exposure']): raise ValueError('medication classes changed')
    if ((eos['med_exposure']>0)&~eos['med_source_mask']).any(): raise ValueError('positive medication without source')
    for values,masks in (('micro_values','micro_masks'),('support_values','support_masks')):
        if not np.isin(rm[values],(0,1)).all() or ((rm[values]>0)&~rm[masks]).any(): raise ValueError('positive flag without source')
    return SourceContextData(base,eos,rm,{'eos_manifest_sha256':sha256(eos_dir/'manifest.json'),
                                        'resp_micro_manifest_sha256':sha256(resp_micro_dir/'manifest.json')})
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/source_context_smoke.py · lines 1–49

<!-- REVIEW_FRAGMENT {"path":"training/v6/source_context_smoke.py","part":1,"start_line":1,"end_line":49,"source_sha256":"de0072664b0fa5bebb92151c52f29bb965e2d202a1bc987551ff4d439156cc56","fragment_sha256":"de0072664b0fa5bebb92151c52f29bb965e2d202a1bc987551ff4d439156cc56","added_terminal_newline":false} -->
````python
"""补充来源全量合同与 GPU 张量验收；零优化步骤，新增候选不伪称进入风险头。"""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.data import load
from training.v6.source_context_data import attach
from training.v6.model import ClinicalEncoder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('snapshot','state_dir','labels','eos_dir','resp_micro_dir','output'):
        parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required; no CPU fallback')
    base=load(args.snapshot,args.state_dir,args.labels)
    data=attach(base,args.eos_dir,args.resp_micro_dir); torch.set_num_threads(1)
    encoder=ClinicalEncoder(768).cuda().eval(); result={}
    for fold in ('train','validation','test'):
        candidates=np.flatnonzero(base.arrays['patient_splits']==fold)
        ix=candidates[np.linspace(0,len(candidates)-1,64,dtype=int)]
        h,b,t,e=data.collate(ix,'cuda')
        shapes={}
        for f in fields(b):
            if f.name=='clinical': continue
            value=getattr(b,f.name)
            if value.device.type!='cuda' or not torch.isfinite(value).all(): raise ValueError('invalid GPU supplemental input')
            shapes[f.name]=list(value.shape)
        with torch.no_grad(): state,_=encoder(b.clinical)
        if not torch.isfinite(state).all(): raise ValueError('corrected medication source mask invalid')
        result[fold]={'admissions_checked':len(ix),'supplemental_shapes':shapes,'finite_clinical_encoder':True}
    report={'status':'supplemental_data_gpu_interface_verified_not_trained','admissions_aligned':len(base.history),
            'patients_aligned':len(set(base.arrays['subject_ids'].tolist())),'device':torch.cuda.get_device_name(),
            'optimizer_steps':0,'checkpoint_saved':False,'folds':result,'provenance':data.provenance,
            'code_sha256':{n:sha256(Path(__file__).parent/n) for n in ('source_context_data.py','source_context_smoke.py')},
            'limits':['New EOS/micro/resp candidates validated as tensors, not consumed by current risk network',
                      'Only corrected medication source mask tested through existing clinical encoder',
                      'No training, encoder re-encoding, model selection or predictive performance evaluation']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','admissions_aligned','patients_aligned','optimizer_steps','device')}))


if __name__=='__main__': main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/summarize_exploration.py · lines 1–90

<!-- REVIEW_FRAGMENT {"path":"training/v6/summarize_exploration.py","part":1,"start_line":1,"end_line":90,"source_sha256":"d627e211abcaf25766159efe8f9f5b02515bea6adb7ba2d6489f58322a45cfac","fragment_sha256":"d627e211abcaf25766159efe8f9f5b02515bea6adb7ba2d6489f58322a45cfac","added_terminal_newline":false} -->
````python
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
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/symmetric_context.py · lines 1–39

<!-- REVIEW_FRAGMENT {"path":"training/v6/symmetric_context.py","part":1,"start_line":1,"end_line":39,"source_sha256":"c8fe1dc6116f9167695151d06e1151c07be5afddf91752fc2dc8a875a331f74b","fragment_sha256":"c8fe1dc6116f9167695151d06e1151c07be5afddf91752fc2dc8a875a331f74b","added_terminal_newline":false} -->
````python
"""保留父计算路径，仅校正纵向当前住院token的背景接入。"""
import torch
from torch.nn import functional as F
from training.v6.unified_longitudinal import LongitudinalModel
from training.v6.risk import RiskOutput


def visit_tokens(current, past, current_context, past_context, gap, *, symmetric):
    """背景都为同一读取器输出；不额外计算/相加过去风险。"""
    historical = past + past_context + gap
    now = current + current_context if symmetric else current
    return torch.cat((now[:, None], historical), 1)


class SymmetricContextModel(LongitudinalModel):
    def __init__(self, fresh, condition):
        if condition not in ('asymmetric', 'symmetric'):
            raise ValueError('unknown context condition')
        super().__init__(fresh, 'history_unified')
        self.condition = condition

    def forward(self, batch):
        if self.condition == 'asymmetric':
            return super().forward(batch)
        compact, legacy, labs, eos, context, current_index, past_index, present, gaps = batch
        hidden = self.visits(compact, legacy, labs, eos)
        current = hidden[current_index]
        delta = self.original.delta_head(current).reshape(-1, 2, 2)
        seq = visit_tokens(current, hidden[past_index], self.past_context(context[current_index]),
            self.past_context(context[past_index]),
            self.gap(torch.log1p(gaps.clamp_min(0))[:, :, None] / 6.), symmetric=True)
        seq = torch.cat((seq[:, :1], torch.where(present[:, :, None], seq[:, 1:], 0.)), 1)
        valid = torch.cat((torch.ones(len(current), 1, dtype=torch.bool, device=current.device), present), 1)
        read = self.temporal(seq, src_key_padding_mask=~valid)[:, 0]
        delta = delta + self.history_head(read).reshape(-1, 2, 2) * present.any(1)[:, None, None]
        with torch.no_grad():
            base = self.original.baseline(context[current_index])
        return RiskOutput((base.log_amplitude + delta[:, :, 0]).clamp(-10., 6.),
            F.softplus(self.original.baseline.raw_shape.detach() + delta[:, :, 1]) + .1, delta)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/symmetric_context_experiment.py · lines 1–182

<!-- REVIEW_FRAGMENT {"path":"training/v6/symmetric_context_experiment.py","part":1,"start_line":1,"end_line":182,"source_sha256":"81d5d751abff46fc39151ab08039d645196ab57bb914fc4ffdd0965033bbd15a","fragment_sha256":"81d5d751abff46fc39151ab08039d645196ab57bb914fc4ffdd0965033bbd15a","added_terminal_newline":false} -->
````python
"""单seed匹配背景校正；父实验与受保护模型不修改。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6 import unified_longitudinal_experiment as parent
from training.v6.symmetric_context import SymmetricContextModel
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import competing_nll
from training.v6.train import write_json
from training.v6.multihorizon_readout import horizon_readout, segmented_auc
from training.v6.evaluation import metrics, REPORT_DAYS

ROOT = Path('training_runs/v6_symmetric_context_2026-10-14')
PROTOCOL = Path('docs/V6_SYMMETRIC_CONTEXT_PROTOCOL_2026-10-14.md')
CONDITIONS = ('asymmetric', 'symmetric')


def setup():
    data = parent.setup()
    contract = dict(parent_contract=data[-2], conditions=list(CONDITIONS), no_new_encodings=True,
        source_sha256={str(p):sha256(p) for p in (Path(__file__),
            Path('training/v6/symmetric_context.py'), Path('training/tests/test_v6_symmetric_context.py'),
            PROTOCOL, parent.ROOT/'report.json', parent.ROOT/'verification.json', parent.ROOT/'contract.json',
            parent.ROOT/'history_unified/best.pt', parent.ROOT/'history_unified/tuning_predictions.npz')})
    proof=json.loads((parent.ROOT/'verification.json').read_text())
    if proof['maximum_error'] != 0 or proof['report_sha256'] != sha256(parent.ROOT/'report.json'):
        raise ValueError('parent proof changed')
    return data, contract


def readout(d, indices, curve, annual_threshold):
    a = dict(times=d.times[indices], events=d.events[indices], curve=curve)
    result = {}
    for label, cause in (('readmission', 1), ('death', 2)):
        m = metrics(a['times'], a['events'], curve[:, :, cause], cause=cause)
        y = ((a['events'][:, None] == cause) & (a['times'][:, None] <= np.arange(366)))
        brier = ((curve[:, :, cause] - y) ** 2).mean(0)
        early = {}
        for day in (1, 3):
            from training.v6.unified_longitudinal import best_f1
            early[str(day)] = dict(cases=int(y[:, day].sum()), brier=float(brier[day]),
                tuning_best_f1=best_f1(y[:, day], curve[:, day, cause]), auc=None,
                reason='fewer than 20 cases or controls')
        threshold = annual_threshold if cause == 1 else .2
        result[label] = dict(integrated_brier_1_365=float(np.trapezoid(brier[1:], np.arange(1,366))/364),
            segmented_auc=segmented_auc(m['auc_curve']), early=early,
            points={str(day):horizon_readout(a, day, cause, threshold) for day in REPORT_DAYS})
    return result


def preflight(fresh, b, d, positions):
    models={n:SymmetricContextModel(fresh,n).cuda().eval() for n in CONDITIONS}
    for p,q in zip(models['asymmetric'].parameters(),models['symmetric'].parameters(),strict=True):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    initial={n:m(b) for n,m in models.items()}
    for field in ('log_amplitude','shape'):
        torch.testing.assert_close(getattr(initial['asymmetric'],field),getattr(initial['symmetric'],field),rtol=0,atol=0)
    model=models['symmetric']; baseline={n:p.detach().clone() for n,p in model.original.baseline.named_parameters()}
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss=competing_nll(model(b),torch.as_tensor(d.times[positions],device='cuda'),
                          torch.as_tensor(d.events[positions],device='cuda'))
        loss.backward(); opt.step()
    if any(p.grad is not None or not torch.equal(p,baseline[n]) for n,p in model.original.baseline.named_parameters()):
        raise ValueError('H34 updated')
    changed={prefix:any(n.startswith(prefix) and not torch.equal(p,before[n]) for n,p in model.named_parameters() if n in before)
        for prefix in ('past_context.','temporal.','history_head.','original.encoder.base.lab.','original.delta_head.')}
    if not all(changed.values()): raise ValueError('active layer did not update')
    # nonzero fitted history head allows a real check of current-background sensitivity.
    altered=list(b); altered[4]=b[4].clone(); altered[4][b[5],0]+=.5
    # Both paths change final H34: compare only the delta, which excludes baseline output.
    delta_change=float((model(tuple(altered)).delta-model(b).delta).abs().max().detach())
    if delta_change<=0: raise ValueError('current background not consumed before baseline')
    return dict(initial_parameter_error=0, h34_frozen=True, actual_updates=changed,
                current_background_delta_sensitivity=delta_change)


def fit(fresh, condition, tr, va, d, tune):
    folder=ROOT/condition; folder.mkdir()
    model=SymmetricContextModel(fresh,condition).cuda();torch.manual_seed(42);rng=np.random.default_rng(42)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4)
    early=EarlyStopping(7,10,1e-4);best=(-np.inf,np.inf);history=[];steps=0;started=time.perf_counter()
    frozen={n:p.detach().clone() for n,p in model.original.baseline.named_parameters()}
    for epoch in range(46):
        train_nll=None
        if epoch:
            model.train(); total=0;count=0
            for j in rng.permutation(len(tr)):
                p,b=tr[j];opt.zero_grad(set_to_none=True);out=model(b)
                nll=competing_nll(out,torch.as_tensor(d.times[p],device='cuda'),torch.as_tensor(d.events[p],device='cuda'))
                loss=nll+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
                opt.step();steps+=1;total+=float(nll.detach())*len(p);count+=len(p)
            train_nll=total/count
        a,k,curve,nll=parent.predict(model,va,d);m=parent.detail(d,tune,curve,nll)
        f1=m['annual']['f1'];ibs=m['readmission']['integrated_brier_7_365']
        _,stop=early.observe(epoch,-f1)
        if f1>best[0]+1e-12 or (abs(f1-best[0])<=1e-12 and ibs<best[1]-1e-12):
            best=(f1,ibs);torch.save(dict(state_dict=model.state_dict(),epoch=epoch,threshold=m['annual']['threshold']),folder/'best.pt')
        history.append(dict(epoch=epoch,train_nll=train_nll,annual=m['annual'],tuning_ibs=ibs))
        write_json(folder/'history.json',history)
        if epoch%5==0:print(json.dumps(dict(condition=condition,epoch=epoch,f1=f1,ibs=ibs)),flush=True)
        if stop:break
    if any(not torch.equal(p,frozen[n]) for n,p in model.original.baseline.named_parameters()):raise ValueError('base changed')
    torch.save(dict(state_dict=model.state_dict(),epoch=epoch),folder/'last.pt')
    chosen=torch.load(folder/'best.pt',map_location='cuda',weights_only=True);model.load_state_dict(chosen['state_dict'])
    a,k,curve,nll=parent.predict(model,va,d);m=parent.detail(d,tune,curve,nll,chosen['threshold'])
    parent.save(folder/'tuning_predictions.npz',d,tune,a,k,curve)
    info=dict(best_epoch=chosen['epoch'],completed_epochs=epoch,early=asdict(early),
        stop_reason='patience_exhausted' if stop else 'max_epochs',optimizer_steps=steps,
        checkpoint_sha256=sha256(folder/'best.pt'),metrics=m,
        multihorizon=readout(d,tune,curve,chosen['threshold']),seconds=time.perf_counter()-started)
    write_json(folder/'result.json',info);return info


def run():
    if ROOT.exists():raise FileExistsError('experiment exists; no duplicate run')
    data,contract=setup();d,c,ext,lab,rel,train,tune,_,fresh,_,_,source=data
    ROOT.mkdir();write_json(ROOT/'contract.json',contract);write_json(ROOT/'source_report.json',source)
    selected=train[np.argsort(-np.diff(rel['offsets'])[train])[:16]]
    b=parent.cache(c,ext,lab,rel,selected,True)[0][1]
    write_json(ROOT/'preflight.json',preflight(fresh,b,d,selected))
    tr=parent.cache(c,ext,lab,rel,train,True);va=parent.cache(c,ext,lab,rel,tune,True)
    results={n:fit(fresh,n,tr,va,d,tune) for n in CONDITIONS}
    with np.load(parent.ROOT/'history_unified/tuning_predictions.npz') as old, np.load(ROOT/'asymmetric/tuning_predictions.npz') as new:
        error=float(np.abs(old['curve']-new['curve']).max())
    if error!=0:raise ValueError('matched control did not reproduce parent')
    selected=max(CONDITIONS,key=lambda n:(results[n]['metrics']['annual']['f1'],-results[n]['metrics']['readmission']['integrated_brier_7_365']))
    write_json(ROOT/'report.json',dict(results=results,matched_control_parent_error=error,selected=selected,
        default_changed=False,single_seed=True,confirmation_or_test_used=False))
    write_json(ROOT/'state.json',dict(status='trained_pending_replay'))
    print(json.dumps(dict(selected=selected,matched_control_parent_error=error)),flush=True)


def verify():
    data,contract=setup()
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract changed')
    d,c,ext,lab,rel,_,tune,_,fresh,_,_,_=data;va=parent.cache(c,ext,lab,rel,tune,True)
    report=json.loads((ROOT/'report.json').read_text());errors={}
    for n in CONDITIONS:
        info=report['results'][n];model=SymmetricContextModel(fresh,n).cuda()
        chosen=torch.load(ROOT/n/'best.pt',map_location='cuda',weights_only=True)
        if sha256(ROOT/n/'best.pt')!=info['checkpoint_sha256']:raise ValueError('checkpoint changed')
        model.load_state_dict(chosen['state_dict']);a,k,curve,nll=parent.predict(model,va,d)
        with np.load(ROOT/n/'tuning_predictions.npz') as saved:
            for key,actual in (('hadm_ids',d.arrays['hadm_ids'][tune]),('subject_ids',d.arrays['subject_ids'][tune]),('events',d.events[tune]),('times',d.times[tune])):
                np.testing.assert_array_equal(saved[key],actual)
            error=max(float(np.abs(saved['curve']-curve).max()),float(np.abs(saved['log_amplitude']-a).max()),float(np.abs(saved['shape']-k).max()))
        if error!=0:raise ValueError('prediction replay differs')
        if parent.detail(d,tune,curve,nll,chosen['threshold'])!=info['metrics']:raise ValueError('metric replay differs')
        if readout(d,tune,curve,chosen['threshold'])!=info['multihorizon']:raise ValueError('multi-horizon replay differs')
        history=json.loads((ROOT/n/'history.json').read_text());early=EarlyStopping(7,10,1e-4)
        for r in history:early.observe(r['epoch'],-r['annual']['f1'])
        if asdict(early)!=info['early']:raise ValueError('early stop differs')
        best=min(history,key=lambda r:(-r['annual']['f1'],r['tuning_ibs'],r['epoch']))
        if best['epoch']!=chosen['epoch'] or best['annual']['threshold']!=chosen['threshold']:raise ValueError('selection differs')
        model.load_state_dict(torch.load(ROOT/n/'last.pt',map_location='cuda',weights_only=True)['state_dict'])
        _,_,last_curve,last_nll=parent.predict(model,va,d);last=parent.detail(d,tune,last_curve,last_nll)
        if last['annual']!=history[-1]['annual'] or last['readmission']['integrated_brier_7_365']!=history[-1]['tuning_ibs']:
            raise ValueError('last checkpoint differs')
        errors[n]=error
    write_json(ROOT/'verification.json',dict(maximum_error=0,errors=errors,
        report_sha256=sha256(ROOT/'report.json'),new_process=True))
    write_json(ROOT/'state.json',dict(status='independently_verified'))
    print(json.dumps(errors),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('train','verify'),required=True)
    try:{'train':run,'verify':verify}[p.parse_args().mode]()
    except Exception as e:
        if ROOT.exists():write_json(ROOT/'failure.json',dict(type=type(e).__name__,message=str(e)))
        raise
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/task_gradient_diagnostic.py · lines 1–72

<!-- REVIEW_FRAGMENT {"path":"training/v6/task_gradient_diagnostic.py","part":1,"start_line":1,"end_line":72,"source_sha256":"49c6a6ddb4cf98bff937f33606713283b40017170f3235e377d45b93bf189a10","fragment_sha256":"49c6a6ddb4cf98bff937f33606713283b40017170f3235e377d45b93bf189a10","added_terminal_newline":false} -->
````python
"""训练折上双原因梯度方向诊断；不训练/选模，不读取确认或测试分数。"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.module_surgery import SurgeryModel
from training.v6.module_surgery_experiment import batches
from training.v6.risk import competing_nll
from training.v6.train import write_json

ROOT=Path('training_runs/v6_task_gradient_diagnostic_2026-10-13')


def cause_losses(out,times,events):
    scaled=times.clamp_min(1e-3)[:,None]/365.
    cumulative=torch.exp(out.log_amplitude)*scaled.pow(out.shape)
    log_hazard=out.log_amplitude+torch.log(out.shape)-math.log(365.)+(out.shape-1)*torch.log(scaled)
    return torch.stack([(cumulative[:,c]-torch.where(events==c+1,log_hazard[:,c],0.)).mean() for c in (0,1)])


def gradient_summary(left,right):
    aa=sum(float(x.detach().square().sum()) for x in left if x is not None)
    bb=sum(float(x.detach().square().sum()) for x in right if x is not None)
    dot=sum(float((a.detach()*b.detach()).sum()) for a,b in zip(left,right) if a is not None and b is not None)
    return dict(readmission_norm=math.sqrt(aa),death_norm=math.sqrt(bb),
        cosine=dot/math.sqrt(aa*bb) if aa*bb>1e-24 else None,
        death_to_readmission_norm_ratio=math.sqrt(bb/aa) if aa>1e-24 else None)


def run():
    if ROOT.exists():raise FileExistsError(ROOT)
    d,c,h,ext,base,bins,_,_,contract,_=checked()
    train=np.flatnonzero(d.arrays['patient_splits']=='train')
    cache=batches(d,c,ext,train)
    model=SurgeryModel(base,42,'control',bins).cuda().eval()
    checkpoint=REFERENCE/'seed42/control/best.pt'
    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['state_dict'])
    core=model.detail.base
    groups=dict(shared_head=model.delta_head[0],lab=core.lab,text=core.text_projection,fusion=core.fusion)
    rows=[]
    selected=np.sort(np.random.default_rng(20261013).choice(len(cache),min(16,len(cache)),replace=False))
    parameters=[p for module in groups.values() for p in module.parameters() if p.requires_grad]
    lengths=[sum(p.requires_grad for p in module.parameters()) for module in groups.values()]
    for j in selected:
        _,history,b,t,e=cache[j];out=model(history,b);loss=cause_losses(out,t,e)
        torch.testing.assert_close(loss.sum(),competing_nll(out,t,e),atol=1e-6,rtol=1e-6)
        left=torch.autograd.grad(loss[0],parameters,retain_graph=True,allow_unused=True)
        right=torch.autograd.grad(loss[1],parameters,allow_unused=True)
        start=0;result={}
        for name,length in zip(groups,lengths):
            result[name]=gradient_summary(left[start:start+length],right[start:start+length]);start+=length
        rows.append(dict(training_batch=int(j),admissions=len(t),groups=result))
    aggregate={}
    for name in groups:
        values=[x['groups'][name]['cosine'] for x in rows if x['groups'][name]['cosine'] is not None]
        ratios=[x['groups'][name]['death_to_readmission_norm_ratio'] for x in rows if x['groups'][name]['death_to_readmission_norm_ratio'] is not None]
        aggregate[name]=dict(valid_batches=len(values),negative_cosine_batches=sum(x<0 for x in values),
            mean_cosine=float(np.mean(values)) if values else None,
            median_death_to_readmission_norm_ratio=float(np.median(ratios)) if ratios else None)
    ROOT.mkdir()
    write_json(ROOT/'report.json',dict(status='training_only_gradient_diagnostic',aggregate=aggregate,batches=rows,
        checkpoint_sha256=sha256(checkpoint),source_sha256=sha256(Path(__file__)),
        data_provenance=contract['data_provenance'],test_not_evaluated=True,goal_achieved=False,
        definition='cause NLL terms sum exactly to original joint NLL; eval dropout off; fixed trained V6; no optimizer update',
        limits=['Negative cosine is not proof of a causal performance bottleneck.','Gradient scale depends on parameterization.','16 training batches only.']))
    print(json.dumps(dict(status='training_only_gradient_diagnostic',aggregate=aggregate)),flush=True)


if __name__=='__main__':run()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/task_mortality_evidence.py · lines 1–73

<!-- REVIEW_FRAGMENT {"path":"training/v6/task_mortality_evidence.py","part":1,"start_line":1,"end_line":73,"source_sha256":"574ee8733247352862a73aabd05f8bde6e52cf8d228f02ac7753237015e273b6","fragment_sha256":"574ee8733247352862a73aabd05f8bde6e52cf8d228f02ac7753237015e273b6","added_terminal_newline":false} -->
````python
"""补核第二批死亡方向；只读锁定确认，不改变再入院/选模口径。"""
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.task_optimization_experiment import ROOT,SEEDS
from training.v6.task_optimization_audit import inputs
from training.v6.evaluation import paired_interval,outcome_matrix,metrics
from training.v6.train import write_json


def death_pair(subjects,times,events,candidate,reference,*,seed):
    # 仅在评价副本互换两个事件号以复用配对代码；不改训练/标签/预测。
    swapped=np.where(events==1,2,np.where(events==2,1,0))
    np.testing.assert_array_equal(outcome_matrix(times,swapped),outcome_matrix(times,events,cause=2))
    p=paired_interval(subjects,times,swapped,candidate,reference,replicates=200,seed=seed)
    expected=metrics(times,events,candidate,cause=2)['integrated_brier_7_365']-metrics(times,events,reference,cause=2)['integrated_brier_7_365']
    np.testing.assert_allclose(p['integrated_brier_difference'],expected,atol=1e-12,rtol=0)
    return p


def run():
    r=json.loads((ROOT/'confirmation_report.json').read_text());proof=json.loads((ROOT/'confirmation_verification.json').read_text())
    if proof['report_sha256']!=sha256(ROOT/'confirmation_report.json'):raise ValueError('confirmation proof stale')
    (d,_,_,_,_,_,_,ix,_,_),_=inputs();chosen=r['lock']['selected'];pairs={}
    for seed in SEEDS:
        risks={}
        for name in (chosen,'control','official_V6'):
            with np.load(ROOT/f'seed{seed}'/name/'confirmation_predictions.npz') as z:
                for key,expected in (('hadm_ids',d.arrays['hadm_ids'][ix]),('subject_ids',d.arrays['subject_ids'][ix]),('times',d.times[ix]),('events',d.events[ix])):
                    np.testing.assert_array_equal(z[key],expected)
                risks[name]=z['curve'][:,:,2].copy()
            observed=metrics(d.times[ix],d.events[ix],risks[name],cause=2)
            if observed['integrated_brier_7_365']!=r['models'][str(seed)][name]['metrics']['death']['integrated_brier_7_365']:
                raise ValueError('death metric differs')
        pairs[str(seed)]={ref:death_pair(d.arrays['subject_ids'][ix],d.times[ix],d.events[ix],risks[chosen],risks[ref],seed=seed)
                         for ref in ('control','official_V6')}
    result=dict(candidate=chosen,comparisons=pairs,report_sha256=sha256(ROOT/'confirmation_report.json'),
        code_sha256=sha256(Path(__file__)),goal_achieved=False,test_not_evaluated=True,
        limits=['Post-hoc secondary endpoint analysis; no reselection.','Three clinical seeds reuse two base sources.',
        'Patient intervals conditional on fitted model, no multiplicity adjustment.','Not evidence of readmission improvement.'])
    write_json(ROOT/'mortality_followup.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    labels=[]
    for j,(seed,ref) in enumerate((s,ref) for s in SEEDS for ref in ('control','official_V6')):
        p=pairs[str(seed)][ref];labels.append(f'Seed {seed}: vs {ref}')
        for ax,key,ci in zip(axes,('integrated_brier_difference','auc90_difference'),('integrated_brier_95ci','auc90_95ci')):
            ax.hlines(j,*p[ci],color=f'C{SEEDS.index(seed)}',lw=2);ax.scatter(p[key],j,color=f'C{SEEDS.index(seed)}',marker='D')
    for ax in axes:ax.axvline(0,color='gray',lw=1);ax.grid(axis='x',alpha=.25);ax.set_yticks(range(len(labels)),labels)
    axes[0].invert_yaxis();axes[0].set_xlabel('Death IBS7-365 difference (negative better)')
    axes[1].set_xlabel('Death AUC90 difference (positive better)')
    fig.suptitle('Locked independent clinical branches: secondary mortality follow-up\n'
                 '200 patient-cluster bootstrap; exploratory 95% intervals; no multiplicity adjustment')
    fig.tight_layout();fig.savefig(ROOT/'mortality_increment.png',dpi=180);plt.close(fig)
    lines=['# 第二批死亡方向：补充配对核验','',
           '只读已锁定候选，不更换候选或训练标签。这里是死亡概率，不是再入院成功证据。',
           '', '| 种子 | 参照 | 死亡IBS差值↓ | 95%患者区间 | 死亡AUC90差值↑ | 95%患者区间 |',
           '|---|---|---:|---|---:|---|']
    for seed in SEEDS:
        for ref,p in pairs[str(seed)].items():
            lines.append(f'| {seed} | {ref} | {p["integrated_brier_difference"]:.6f} | {p["integrated_brier_95ci"]} | {p["auc90_difference"]:.6f} | {p["auc90_95ci"]} |')
    lines+=['','![死亡方向配对区间](mortality_increment.png)','',
            '原始住院ID/患者ID/时间/事件与确认数据逐行一致，死亡IBS重算与主报告一致。',
            '评价副本交换事件号仅复用已审计配对函数，直接cause=2指标和结局矩阵均核对一致；训练和原标签未改。',
            '属于事后次要终点探索，不替代五种子稳定确认，不用于当前第三批选模。']
    (ROOT/'MORTALITY_FOLLOWUP.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='secondary_mortality_verified',comparisons=pairs,goal_achieved=False)),flush=True)


if __name__=='__main__':run()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/task_optimization.py · lines 1–99

<!-- REVIEW_FRAGMENT {"path":"training/v6/task_optimization.py","part":1,"start_line":1,"end_line":99,"source_sha256":"5ab36b988ef5d18f080baf04ea69815a3b236e2dffc65efcf7b0cb5b12ebaf43","fragment_sha256":"5ab36b988ef5d18f080baf04ea69815a3b236e2dffc65efcf7b0cb5b12ebaf43","added_terminal_newline":false} -->
````python
"""V6双原因模块/共享梯度替换；原编码与概率族不改。"""
import copy
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.regularization_model import Trial
from training.v6.module_surgery import SurgeryModel
from training.v6.risk import RiskOutput
from training.v6.task_gradient_diagnostic import cause_losses


@dataclass(frozen=True)
class TaskTrial(Trial):
    head: str = 'shared32'
    pcgrad: bool = False
    balance: bool = False
    independent_encoders: bool = False


TRIALS=(
    TaskTrial('control'),
    TaskTrial('shared64',head='shared64'),
    TaskTrial('specific32',head='specific32'),
    TaskTrial('specific64',head='specific64'),
    TaskTrial('independent32',head='specific32',independent_encoders=True),
    TaskTrial('pcgrad_shared32',pcgrad=True),
    TaskTrial('pcgrad_specific32',head='specific32',pcgrad=True),
    TaskTrial('balanced_specific32',head='specific32',pcgrad=True,balance=True),
)


class SpecificHead(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.heads=nn.ModuleList(nn.Sequential(nn.Linear(128,width),nn.GELU(),nn.Linear(width,2)) for _ in range(2))
        for h in self.heads:
            nn.init.zeros_(h[-1].weight);nn.init.zeros_(h[-1].bias)

    def forward(self,x):return torch.cat([h(x) for h in self.heads],1)


class IndependentCauses(nn.Module):
    def __init__(self,original,width=32):
        super().__init__();self.baseline=original.baseline
        self.encoders=nn.ModuleList([original.encoder,copy.deepcopy(original.encoder)])
        self.head=SpecificHead(width)

    def train(self,mode=True):
        super().train(mode);self.baseline.eval();return self

    def forward(self,history,batch):
        with torch.no_grad():base=self.baseline(history)
        states=[encoder(batch.base)[0] for encoder in self.encoders]
        delta=torch.stack([h(x) for h,x in zip(self.head.heads,states)],1)
        return RiskOutput((base.log_amplitude+delta[:,:,0]).clamp(-10,6),
            F.softplus(self.baseline.raw_shape.detach()+delta[:,:,1])+.1,delta)


def build(base,seed,trial,bins,*,text_width=768,device='cuda'):
    model=SurgeryModel(base,seed,'control',bins,text_width=text_width)
    if trial.independent_encoders:model=IndependentCauses(model,32)
    elif trial.head.startswith('specific'):model.delta_head=SpecificHead(int(trial.head.removeprefix('specific')))
    elif trial.head=='shared64':
        model.delta_head=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,4))
        nn.init.zeros_(model.delta_head[-1].weight);nn.init.zeros_(model.delta_head[-1].bias)
    model.arm=trial.name
    model=model.to(device);torch.manual_seed(seed)
    return model


def projected_sum(left,right,*,balance=False):
    """双任务对称投影，只用于共享参数；保留原联合损失的sum尺度。"""
    aa=sum(g.square().sum() for g in left);bb=sum(g.square().sum() for g in right)
    if balance and aa>1e-24 and bb>1e-24:
        mean=(aa.sqrt()+bb.sqrt())*.5
        left=[g*(mean/aa.sqrt()).clamp(.25,4.) for g in left]
        right=[g*(mean/bb.sqrt()).clamp(.25,4.) for g in right]
        aa=sum(g.square().sum() for g in left);bb=sum(g.square().sum() for g in right)
    dot=sum((a*b).sum() for a,b in zip(left,right))
    if dot<0:
        return [(a-dot/bb.clamp_min(1e-24)*b)+(b-dot/aa.clamp_min(1e-24)*a) for a,b in zip(left,right)]
    return [a+b for a,b in zip(left,right)]


def backward(model,out,nll,penalty,t,e,trial):
    if not trial.pcgrad:
        (nll+penalty).backward();return
    shared=[p for p in model.encoder.parameters() if p.requires_grad]
    if not isinstance(model.delta_head,SpecificHead):shared.extend(model.delta_head[0].parameters())
    terms=cause_losses(out,t,e)
    left=torch.autograd.grad(terms[0],shared,retain_graph=True,allow_unused=True)
    right=torch.autograd.grad(terms[1],shared,retain_graph=True,allow_unused=True)
    extra=torch.autograd.grad(penalty,shared,retain_graph=True,allow_unused=True)
    clean=lambda gs:[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(shared,gs)]
    corrected=projected_sum(clean(left),clean(right),balance=trial.balance)
    (nll+penalty).backward()  # 专用头保持普通联合似然梯度，不投影成对方任务。
    with torch.no_grad():
        for p,g,reg in zip(shared,corrected,extra):p.grad=g+(reg.detach() if reg is not None else 0.)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/task_optimization_audit.py · lines 1–203

<!-- REVIEW_FRAGMENT {"path":"training/v6/task_optimization_audit.py","part":1,"start_line":1,"end_line":203,"source_sha256":"bf42ec7df6b744ea2364403a569d5b8cafca08df49dadd86279604d6b9595671","fragment_sha256":"bf42ec7df6b744ea2364403a569d5b8cafca08df49dadd86279604d6b9595671","added_terminal_newline":false} -->
````python
"""第二批独立重放、调参锁定及一次确认；不加载历史测试预测。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.task_optimization_experiment import ROOT,SEEDS,BASE_SEEDS
from training.v6.task_optimization import TRIALS,build
from training.v6.module_surgery_experiment import batches,parameters,validation,summary,save
from training.v6.module_surgery import SurgeryModel,probability_curve
from training.v6.token_model import daily_probabilities
from training.v6.horizon_objective import restore
from training.v6.evaluation import paired_interval
from training.v6.train import write_json


def inputs():
    result=checked()
    d,c,h,ext,base,bins,tune,confirm,search,prior=result
    contract=json.loads((ROOT/'contract.json').read_text())
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('task training source changed: '+p)
    if sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json'))!=contract['search_contract_sha256']:
        raise ValueError('source contract changed')
    if contract['seeds']!=list(SEEDS) or contract['baseline_seed_mapping']!={str(k):v for k,v in BASE_SEEDS.items()}:
        raise ValueError('seed map differs')
    if contract['trials']!=[asdict(t) for t in TRIALS]:raise ValueError('trial list changed')
    # 原正式合同已保存两个基础来源的hash，不以检查时hash冒充训练快照。
    for seed in set(BASE_SEEDS.values()):
        path=Path(search['input_config']['parent'])/f'seed{seed}/survival_base/best.pt'
        if sha256(path)!=prior['frozen_baselines'][str(seed)]:raise ValueError('baseline source changed')
    official=json.loads((REFERENCE/'report.json').read_text())
    for seed in set(BASE_SEEDS.values()):
        path=REFERENCE/f'seed{seed}/control/best.pt'
        if sha256(path)!=official['models'][str(seed)]['control']['checkpoint_sha256']:
            raise ValueError('official comparator checkpoint changed')
    return result,contract


def load(base,seed,trial,bins,*,root=None,model_factory=None,baseline_seeds=None):
    """公共检查点核验；显式注入目录/工厂，不改模块全局状态。"""
    root=ROOT if root is None else root
    model_factory=build if model_factory is None else model_factory
    baseline_seeds=BASE_SEEDS if baseline_seeds is None else baseline_seeds
    path=root/f'seed{seed}'/trial.name
    info=json.loads((path/'training_result.json').read_text())
    rows=json.loads((path/'history.json').read_text())
    state=torch.load(path/'best.pt',map_location='cuda',weights_only=True)
    if info['trial']!=asdict(trial) or state['trial']!=asdict(trial):raise ValueError('trial mismatch')
    if info['seed']!=seed or info['baseline_seed']!=baseline_seeds[seed]:raise ValueError('seed mismatch')
    if info['completed_epochs']!=100 or [x['epoch'] for x in rows]!=list(range(101)):
        raise ValueError('incomplete 100-epoch record')
    best=min(rows,key=lambda x:x['tuning_ibs'])
    if (state['epoch']!=best['epoch'] or info['best_epoch']!=best['epoch'] or
            info['tuning_ibs']!=best['tuning_ibs'] or state['tuning_ibs']!=best['tuning_ibs']):
        raise ValueError('checkpoint not selected by tuning IBS')
    if sha256(path/'best.pt')!=info['checkpoint_sha256']:raise ValueError('weight hash mismatch')
    m=model_factory(base,seed,trial,bins);m.load_state_dict(state['state_dict'])
    if any(not torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):
        raise ValueError('frozen baseline moved')
    return m,info,best


def check_curve(curve):
    if not np.isfinite(curve).all() or np.any((curve<0)|(curve>1)):raise ValueError('invalid probability')
    np.testing.assert_allclose(curve.sum(2),1,atol=1e-9,rtol=0)
    if np.any(np.diff(curve[:,:,1:],axis=1)<-1e-10):raise ValueError('CIF decreases')


def check_saved(path,d,ix,a,k,curve):
    daily,conditional=daily_probabilities(curve)
    with np.load(path) as z:
        for key,expected in (('hadm_ids',d.arrays['hadm_ids'][ix]),('subject_ids',d.arrays['subject_ids'][ix]),
                             ('times',d.times[ix]),('events',d.events[ix])):
            np.testing.assert_array_equal(z[key],expected)
        for key,expected in (('log_amplitude',a),('shape',k),('curve',curve),
                             ('daily_event_probability',daily),('conditional_daily_event_probability',conditional)):
            np.testing.assert_allclose(z[key],expected,atol=3e-7,rtol=0)
        np.testing.assert_array_equal(z['conditional_daily_valid'],curve[:,:-1,0]>1e-12)
        return float(np.abs(z['curve']-curve).max())


def verify(*,partial=False):
    (d,c,h,ext,_,bins,tune,_,search,_),contract=inputs()
    cache=batches(d,c,ext,tune);results={};errors={};models=0
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        results[str(seed)]={}
        for trial in TRIALS:
            path=ROOT/f'seed{seed}'/trial.name
            if not (path/'training_result.json').exists():
                if not partial:raise ValueError('task training incomplete')
                continue
            m,info,best=load(base,seed,trial,bins)
            # 训练日志与独立新进程计算必须相同；报告全时间点而非只看AUC90。
            score=validation(m,cache,d.times[tune],d.events[tune])
            for key,value in score.items():
                if best['tuning_'+key]!=value:raise ValueError('training metric replay mismatch: '+key)
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k);check_curve(curve)
            target=path/'tuning_predictions.npz'
            if target.exists():error=check_saved(target,d,tune,a,k,curve)
            else:save(target,d,tune,a,k,curve);error=None
            # 首次写盘再读盘并不叫独立预测重放；第二次新进程才有数值误差证据。
            errors[f'{seed}/{trial.name}']=error
            results[str(seed)][trial.name]=dict(training=info,metrics=summary(curve,d,tune,nll))
            models+=1;del m
    proof=dict(status='partial_training_metrics_replayed' if partial else 'training_metrics_replayed',models=models,
        prediction_replay_errors=errors,confirmation_not_evaluated=True,test_not_evaluated=True,goal_achieved=False,
        contract_sha256=sha256(ROOT/'contract.json'),verifier_sha256=sha256(Path(__file__)))
    if not partial:
        selected=json.loads((ROOT/'selected_before_confirmation.json').read_text())
        scores={t.name:float(np.mean([results[str(s)][t.name]['training']['tuning_ibs'] for s in SEEDS])) for t in TRIALS}
        if scores!=selected['tuning_scores'] or min(scores,key=scores.get)!=selected['selected']:
            raise ValueError('tuning nomination differs')
        proof['selected_before_confirmation_sha256']=sha256(ROOT/'selected_before_confirmation.json')
        replayed=all(x is not None for x in errors.values())
        proof['status']='tuning_independently_verified' if replayed else 'tuning_predictions_saved_pending_second_process'
        proof['maximum_error']=max(errors.values()) if replayed else None
        report=dict(models=results,tuning_scores=scores,selected=selected['selected'],
                    confirmation_not_evaluated=True,test_not_evaluated=True,goal_achieved=False)
        write_json(ROOT/'tuning_report.json',report)
        proof['tuning_report_sha256']=sha256(ROOT/'tuning_report.json')
    write_json(ROOT/('partial_verification.json' if partial else 'tuning_verification.json'),proof)
    print(json.dumps(proof),flush=True)


def confirm():
    proof=json.loads((ROOT/'tuning_verification.json').read_text())
    if proof['status']!='tuning_independently_verified' or proof['models']!=len(SEEDS)*len(TRIALS):
        raise ValueError('independent full replay needed')
    if proof['verifier_sha256']!=sha256(Path(__file__)) or proof['selected_before_confirmation_sha256']!=sha256(ROOT/'selected_before_confirmation.json'):
        raise ValueError('proof changed')
    chosen=json.loads((ROOT/'selected_before_confirmation.json').read_text())['selected']
    nominated=[] if chosen=='control' else [chosen]
    lock=dict(selected=chosen,nominated=nominated,selection='three-seed mean tuning IBS',
              selected_sha256=sha256(ROOT/'selected_before_confirmation.json'),test_not_evaluated=True)
    lockpath=ROOT/'nomination_lock.json'
    if lockpath.exists():raise FileExistsError('confirmation already locked; do not substitute nominees')
    write_json(lockpath,lock)
    if not nominated:
        write_json(ROOT/'state.json',dict(status='verified_control_selected',next_phase='another bounded hypothesis; no candidate confirmation needed',goal_achieved=False))
        print(json.dumps(dict(status='verified_control_selected',goal_achieved=False)),flush=True);return
    (d,c,h,ext,_,bins,_,ix,search,_),_=inputs();cache=batches(d,c,ext,ix)
    trials={x.name:x for x in TRIALS};models={};pairs={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        models[str(seed)]={};curves={}
        for name in ['control',*nominated,'official_V6']:
            if name=='official_V6':
                m=SurgeryModel(base,BASE_SEEDS[seed],'control',bins).cuda()
                weight=REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt'
                m.load_state_dict(torch.load(weight,map_location='cuda',weights_only=True)['state_dict']);digest=sha256(weight)
            else:
                m,info,_=load(base,seed,trials[name],bins);digest=info['checkpoint_sha256']
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k);check_curve(curve)
            models[str(seed)][name]=dict(metrics=summary(curve,d,ix,nll),checkpoint_sha256=digest)
            path=ROOT/f'seed{seed}'/name;path.mkdir(exist_ok=True)
            save(path/'confirmation_predictions.npz',d,ix,a,k,curve);curves[name]=curve;del m
        pairs[str(seed)]={ref:paired_interval(d.arrays['subject_ids'][ix],d.times[ix],d.events[ix],
            curves[chosen][:,:,1],curves[ref][:,:,1],replicates=200,seed=seed) for ref in ('control','official_V6')}
    write_json(ROOT/'confirmation_report.json',dict(status='three_seed_confirmation_pending_replay',models=models,comparisons=pairs,
        lock=lock,patients=len(np.unique(d.arrays['subject_ids'][ix])),admissions=len(ix),
        test_not_evaluated=True,goal_achieved=False,limitations=['Previously exposed validation, not blind.',
        'Three clinical seeds reuse two fixed baseline sources.', 'Not the final five-seed stability gate.']))
    write_json(ROOT/'state.json',dict(status='three_seed_confirmation_complete_pending_replay',next_phase='fresh confirmation replay, then decide if five-seed confirmation justified',goal_achieved=False))
    print(json.dumps(dict(status='three_seed_confirmation_complete_pending_replay',selected=chosen)),flush=True)


def verify_confirmation():
    r=json.loads((ROOT/'confirmation_report.json').read_text());lock=json.loads((ROOT/'nomination_lock.json').read_text())
    if r['lock']!=lock:raise ValueError('nomination changed')
    (d,c,h,ext,_,bins,_,ix,search,_),_=inputs();cache=batches(d,c,ext,ix);trials={x.name:x for x in TRIALS};errors={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        for name in ['control',*lock['nominated'],'official_V6']:
            path=ROOT/f'seed{seed}'/name
            if name=='official_V6':
                m=SurgeryModel(base,BASE_SEEDS[seed],'control',bins).cuda()
                weight=REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt'
                m.load_state_dict(torch.load(weight,map_location='cuda',weights_only=True)['state_dict']);digest=sha256(weight)
            else:m,info,_=load(base,seed,trials[name],bins);digest=info['checkpoint_sha256']
            if digest!=r['models'][str(seed)][name]['checkpoint_sha256']:raise ValueError('confirmation weight changed')
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k)
            errors[f'{seed}/{name}']=check_saved(path/'confirmation_predictions.npz',d,ix,a,k,curve)
            if summary(curve,d,ix,nll)!=r['models'][str(seed)][name]['metrics']:raise ValueError('confirmation metrics differ')
            del m
    write_json(ROOT/'confirmation_verification.json',dict(status='three_seed_confirmation_independently_verified',models=len(errors),
        maximum_error=max(errors.values()),replay_errors=errors,report_sha256=sha256(ROOT/'confirmation_report.json'),
        verifier_sha256=sha256(Path(__file__)),test_not_evaluated=True,goal_achieved=False))
    print(json.dumps(dict(status='three_seed_confirmation_independently_verified',maximum_error=max(errors.values()))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('partial','verify','confirm','verify-confirmation'),required=True)
    mode=p.parse_args().mode
    if mode in ('partial','verify'):verify(partial=mode=='partial')
    elif mode=='confirm':confirm()
    else:verify_confirmation()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/task_optimization_diagnostics.py · lines 1–60

<!-- REVIEW_FRAGMENT {"path":"training/v6/task_optimization_diagnostics.py","part":1,"start_line":1,"end_line":60,"source_sha256":"dd2a79fc9b8923dce48a0b796275d3d3218b3a21019cdee4f9d028c0a65ceaa4","fragment_sha256":"dd2a79fc9b8923dce48a0b796275d3d3218b3a21019cdee4f9d028c0a65ceaa4","added_terminal_newline":false} -->
````python
"""第二批锁定候选的确认校准与次数层内增量；不再次选模。"""
import json
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.task_optimization_experiment import ROOT,SEEDS
from training.v6.task_optimization_audit import inputs
from training.v6.regularization_diagnostics import history_counts,calibration
from training.v6.evaluation import metrics,public_metrics,paired_interval
from training.v6.train import write_json


def analyze(times,events,subjects,counts,curves,*,seed,comparators=('control','official_V6')):
    if counts.shape!=times.shape or subjects.shape!=times.shape:raise ValueError('history/patient rows differ')
    strata=dict(prior0=counts==0,prior1_2=(counts>=1)&(counts<=2),prior3plus=counts>=3)
    models={}
    for name,curve in curves.items():
        groups={}
        for label,mask in strata.items():
            if not mask.any():groups[label]=dict(status='empty');continue
            groups[label]=public_metrics(metrics(times[mask],events[mask],curve[mask,:,1],ordering=True))
        models[name]=dict(calibration={str(c):calibration(times,events,curve,c) for c in (1,2)},strata=groups)
    if not set(comparators).issubset(curves):raise ValueError('required comparator absent')
    for name in curves.keys()-set(comparators):
        models[name]['paired_strata']={}
        for ref in comparators:
            groups={}
            for label,mask in strata.items():
                y=(events[mask]==1)&(times[mask]<=90)
                if not mask.any() or min(int(y.sum()),len(y)-int(y.sum()))<10:
                    groups[label]=dict(status='insufficient cases/controls; not zero effect')
                else:groups[label]=paired_interval(subjects[mask],times[mask],events[mask],curves[name][mask,:,1],
                    curves[ref][mask,:,1],replicates=200,seed=seed)
            models[name]['paired_strata'][ref]=groups
    return models,{label:int(mask.sum()) for label,mask in strata.items()}


def run():
    r=json.loads((ROOT/'confirmation_report.json').read_text())
    proof=json.loads((ROOT/'confirmation_verification.json').read_text())
    if proof['report_sha256']!=sha256(ROOT/'confirmation_report.json'):raise ValueError('confirmation proof stale')
    (d,_,_,_,_,_,_,ix,_,_),_=inputs()
    counts=history_counts(d.history_raw[ix]);result={}
    for seed in SEEDS:
        curves={}
        for name in ['control',*r['lock']['nominated'],'official_V6']:
            with np.load(ROOT/f'seed{seed}'/name/'confirmation_predictions.npz') as z:
                for key,expected in (('hadm_ids',d.arrays['hadm_ids'][ix]),('subject_ids',d.arrays['subject_ids'][ix]),
                    ('times',d.times[ix]),('events',d.events[ix])):
                    np.testing.assert_array_equal(z[key],expected)
                curves[name]=z['curve'].copy()
        result[str(seed)],strata_counts=analyze(d.times[ix],d.events[ix],d.arrays['subject_ids'][ix],counts,curves,seed=seed)
    write_json(ROOT/'calibration_and_history_strata.json',dict(models=result,strata_admissions=strata_counts,
        confirmation_report_sha256=sha256(ROOT/'confirmation_report.json'),goal_achieved=False,test_not_evaluated=True,
        definition='previous completed retained COPD-coded admissions; raw log1p field2; no normalized-value threshold',
        limits=['Previously exposed validation, not blind.','Sparse strata may have unavailable AUC.',
        'No reselection based on strata/calibration.','Three seeds reuse two fixed baseline sources.']))
    print(json.dumps(dict(status='locked_confirmation_diagnostics_written',goal_achieved=False)),flush=True)


if __name__=='__main__':run()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/task_optimization_experiment.py · lines 1–83

<!-- REVIEW_FRAGMENT {"path":"training/v6/task_optimization_experiment.py","part":1,"start_line":1,"end_line":83,"source_sha256":"20d3c92bf61c8eb18de7bd352e2b807b1ec962460d18d70777cfb7a8f1251fc5","fragment_sha256":"20d3c92bf61c8eb18de7bd352e2b807b1ec962460d18d70777cfb7a8f1251fc5","added_terminal_newline":false} -->
````python
"""第二批：三种子原因特异/独立临床分支/梯度替换，只用调参半折。"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked
from training.v6.regularization_multiseed import fit
from training.v6.task_optimization import TRIALS,build,backward
from training.v6.module_surgery_experiment import batches,parameters,validation
from training.v6.module_surgery import probability_curve
from training.v6.horizon_objective import restore
from training.v6.train import write_json

ROOT=Path('training_runs/v6_task_optimization_2026-10-13')
PROTOCOL=Path('docs/V6_TASK_OPTIMIZATION_PROTOCOL_2026-10-13.md')
SEEDS=(42,7,101)
BASE_SEEDS={42:42,7:7,101:42}


def run():
    d,c,h,ext,_,bins,tune,_,search,_=checked()
    protected={p:sha256(Path(p)) for p in ('training/v6/task_optimization.py','training/v6/task_optimization_experiment.py',
        'training/v6/regularization_multiseed.py','training/v6/task_gradient_diagnostic.py','training/tests/test_v6_task_optimization.py',str(PROTOCOL))}
    contract=dict(source_sha256=protected,search_contract_sha256=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')),
        seeds=list(SEEDS),baseline_seed_mapping=BASE_SEEDS,trials=[asdict(x) for x in TRIALS],epochs=100,
        frozen_baselines={str(s):sha256(Path(search['input_config']['parent'])/f'seed{BASE_SEEDS[s]}/survival_base/best.pt') for s in SEEDS},
        input_provenance=search['data_provenance'],selection='three-seed tuning-half readmission IBS mean',
        confirmation_not_evaluated=True,test_not_evaluated=True)
    if (ROOT/'contract.json').exists():
        if json.loads((ROOT/'contract.json').read_text())!=json.loads(json.dumps(contract)):
            raise ValueError('task-optimization resume contract changed')
    else:write_json(ROOT/'contract.json',contract)
    cache=dict(train=batches(d,c,ext,np.flatnonzero(d.arrays['patient_splits']=='train')),tune=batches(d,c,ext,tune))
    results={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        results[str(seed)]={}
        control=build(base,seed,TRIALS[0],bins)
        for trial in TRIALS:
            path=ROOT/f'seed{seed}'/trial.name
            if (path/'training_result.json').exists():
                info=json.loads((path/'training_result.json').read_text())
                if info['trial']!=asdict(trial) or sha256(path/'best.pt')!=info['checkpoint_sha256']:raise ValueError('completed model changed')
                results[str(seed)][trial.name]=info;continue
            if path.exists() and any((path/p).exists() for p in ('best.pt','history.json')):
                import shutil
                backup=path/'recovery_snapshots'/str(time.time_ns());backup.mkdir(parents=True)
                for p in ('best.pt','history.json'):
                    if (path/p).exists():shutil.copy2(path/p,backup/p)
            probe=build(base,seed,trial,bins).eval()
            encoders=probe.encoders if trial.independent_encoders else [probe.encoder]
            for encoder in encoders:
                if any(not torch.equal(v,encoder.state_dict()[k]) for k,v in control.encoder.state_dict().items()):raise ValueError('encoder initial weights differ')
            with torch.no_grad():
                _,hist,b,_,_=cache['tune'][0];out=probe(hist,b);expected=base(hist)
                if not torch.equal(out.log_amplitude,expected.log_amplitude) or not torch.equal(out.shape,expected.shape):raise ValueError('epoch0 risk differs')
            del probe
            write_json(ROOT/'state.json',dict(status='running',seed=seed,trial=trial.name,completed=results,goal_achieved=False))
            info=fit(base,seed,trial,bins,cache,d.times[tune],d.events[tune],path,model_factory=build,backward=backward)
            results[str(seed)][trial.name]=info
        del control
    score={trial.name:float(np.mean([results[str(s)][trial.name]['tuning_ibs'] for s in SEEDS])) for trial in TRIALS}
    chosen=min(score,key=score.get)
    write_json(ROOT/'selected_before_confirmation.json',dict(results=results,tuning_scores=score,selected=chosen,
        all_100_epochs_complete=True,confirmation_not_evaluated=True,test_not_evaluated=True))
    write_json(ROOT/'state.json',dict(status='training_complete_pending_replay',selected=chosen,
        next_phase='verify all 24 models, lock nominees before confirmation',goal_achieved=False))
    print(json.dumps(dict(status='training_complete_pending_replay',selected=chosen,goal_achieved=False)),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    ROOT.mkdir(exist_ok=True);lock=ROOT/'.running.lock'
    with lock.open('x',encoding='utf-8') as owner:owner.write(json.dumps(dict(pid=os.getpid(),started_unix=time.time())))
    try:run()
    except Exception as e:
        write_json(ROOT/'state.json',dict(status='failed',error_type=type(e).__name__,error=str(e),goal_achieved=False));raise
    finally:lock.unlink()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/task_optimization_finish.py · lines 1–70

<!-- REVIEW_FRAGMENT {"path":"training/v6/task_optimization_finish.py","part":1,"start_line":1,"end_line":70,"source_sha256":"99b2ca3e9f479f7dcc8d00bf108a18c322092d5bdaa3b2a9c07f421638cd643c","fragment_sha256":"99b2ca3e9f479f7dcc8d00bf108a18c322092d5bdaa3b2a9c07f421638cd643c","added_terminal_newline":false} -->
````python
"""第二批结束后的单次后处理队列；每个核验步骤为独立Python进程。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.task_optimization_experiment import ROOT,SEEDS
from training.v6.train import write_json


def material_signal(models,chosen,comparators=('control','official_V6')):
    """仅决定是否值得五种子确认，不是长期成功门槛。"""
    result={}
    for ref in comparators:
        auc=[];ibs=[];death=[];joint=0
        for seed in SEEDS:
            x=models[str(seed)][chosen]['metrics'];b=models[str(seed)][ref]['metrics']
            a=x['readmission']['points']['90']['auc'];ba=b['readmission']['points']['90']['auc']
            ad=a-ba if a is not None and ba is not None else float('nan')
            bd=x['readmission']['integrated_brier_7_365']-b['readmission']['integrated_brier_7_365']
            auc.append(ad);ibs.append(bd);death.append(x['death']['integrated_brier_7_365']-b['death']['integrated_brier_7_365'])
            joint+=int(ad>0 and bd<0)
        denominator=np.mean([models[str(s)][ref]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS])
        ma=float(np.mean(auc)) if np.isfinite(auc).all() else None
        improvement=float(-np.mean(ibs)/denominator)
        result[ref]=dict(mean_auc90_gain=ma,relative_ibs_reduction=improvement,joint_improved_seeds=joint,
            death_ibs_change=float(np.mean(death)),worth_five_seed_confirmation=bool(ma is not None and ma>=.003 and
                improvement>=.005 and joint>=2 and np.mean(death)<=.001))
    return dict(comparators=result,worth_five_seed_confirmation=all(v['worth_five_seed_confirmation'] for v in result.values()),
                goal_achieved=False,limits='screening threshold only; final five-seed/timepoint/calibration/strata/test gate unchanged')


def run():
    state=json.loads((ROOT/'state.json').read_text())
    if state['status']=='running' or (ROOT/'.running.lock').exists():raise RuntimeError('training still running')
    if not (ROOT/'selected_before_confirmation.json').exists():raise RuntimeError('complete training selection missing')
    def step(module,mode=None):
        command=[sys.executable,'-u','-m',module]+([] if mode is None else ['--mode',mode])
        write_json(ROOT/'postprocess_state.json',dict(status='running',command=command,goal_achieved=False))
        subprocess.run(command,check=True)
    step('training.v6.task_optimization_audit','verify')
    step('training.v6.task_optimization_audit','verify')
    step('training.v6.task_optimization_report','tuning')
    lockpath=ROOT/'nomination_lock.json'
    if not lockpath.exists():step('training.v6.task_optimization_audit','confirm')
    lock=json.loads(lockpath.read_text())
    if lock['selected_sha256']!=sha256(ROOT/'selected_before_confirmation.json'):raise RuntimeError('existing lock stale')
    if lock['nominated']:
        if not (ROOT/'confirmation_report.json').exists():raise RuntimeError('locked but incomplete confirmation; inspect, do not reselect')
        step('training.v6.task_optimization_audit','verify-confirmation')
        step('training.v6.task_optimization_report','confirmation')
        step('training.v6.task_optimization_diagnostics')
        r=json.loads((ROOT/'confirmation_report.json').read_text());signal=material_signal(r['models'],lock['selected'])
    else:signal=dict(worth_five_seed_confirmation=False,reason='matched control selected on tuning',goal_achieved=False)
    write_json(ROOT/'continuation_decision.json',signal)
    write_json(ROOT/'postprocess_state.json',dict(status='complete',selected=lock['selected'],
        next_phase='inspect calibrated/stratified evidence; five seeds if material signal, otherwise next bounded hypothesis',goal_achieved=False))
    print(json.dumps(dict(status='second_batch_verified_and_review_ready',selected=lock['selected'],decision=signal)),flush=True)


if __name__=='__main__':
    lock=ROOT/'.postprocess.lock'
    with lock.open('x',encoding='utf-8') as owner:owner.write(json.dumps(dict(pid=os.getpid(),started=time.time())))
    try:run()
    except Exception as e:
        write_json(ROOT/'postprocess_state.json',dict(status='failed',error_type=type(e).__name__,error=str(e),goal_achieved=False));raise
    finally:lock.unlink()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/task_optimization_report.py · lines 1–136

<!-- REVIEW_FRAGMENT {"path":"training/v6/task_optimization_report.py","part":1,"start_line":1,"end_line":136,"source_sha256":"10654280c82982f64f43755b003ea7ba3f20df19a67ec3b345ecdd621024bedb","fragment_sha256":"10654280c82982f64f43755b003ea7ba3f20df19a67ec3b345ecdd621024bedb","added_terminal_newline":false} -->
````python
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
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/text_concept.py · lines 1–220

<!-- REVIEW_FRAGMENT {"path":"training/v6/text_concept.py","part":1,"start_line":1,"end_line":220,"source_sha256":"b6fe2394badba15314be1aa37863d2787a98d42e72e4439faf81e6f5f76665c8","fragment_sha256":"b6fe2394badba15314be1aa37863d2787a98d42e72e4439faf81e6f5f76665c8","added_terminal_newline":false} -->
````python
"""训练患者内masked检验概念辅助监督；冻结course BERT，单seed快筛。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from patient_data.export_mimiciv import sha256
from patient_data.v6_lab_state import NUMERIC,FLAGS
from training.v6.fast_multiview import digest,Correction,Batch,fit,r_curve
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.curve_adaptation import caches
from training.v6.resp_fact_set_experiment import prediction
from training.v6.module_surgery_experiment import summary
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import competing_nll
from training.v6.train import write_json
from training.baselines.rank_metrics import auc_rank

ROOT=Path('training_runs/v6_text_concept_2026-10-14')
PROTOCOL=Path('docs/V6_TEXT_CONCEPT_PROTOCOL_2026-10-14.md')
AXES=('arterial_low_ph','arterial_high_pco2','wbc_high','albumin_low','hemoglobin_low')
HEADS=tuple(f'{a}/{role}' for a in AXES for role in ('peak','latest_observed'))
ARMS=('random','concept')

def targets(arrays):
    if tuple(arrays['feature_names'])!=NUMERIC+FLAGS:raise ValueError('summary fields changed')
    raw=arrays['lab_axis_raw'];names=list(arrays['axis_names']);slots=[names.index(a) for a in AXES]
    if raw.ndim!=3 or raw.shape[2]!=9 or not np.isfinite(raw).all():raise ValueError('invalid raw summary')
    observed=raw[:,slots,6].astype(bool)
    mask=np.repeat(observed,2,axis=1)
    y=(raw[:,slots,1:3]>0).reshape(len(raw),10).astype(np.float32)
    # Unknown target slots remain masked, not forced normal evidence.
    return y,mask,raw[:,slots,4].copy()

def course(d):
    text=np.zeros((len(d.history),768),np.float32);present=np.zeros(len(text),bool)
    for p,refs in enumerate(d.text_refs):
        rows=[row for field,row in refs if field==2]
        if rows:text[p]=np.asarray(d.text_vectors[rows],np.float32).mean(0);present[p]=True
    return text,present

def masked_bce(logits,y,mask):
    if logits.shape!=y.shape or mask.shape!=y.shape or mask.dtype!=torch.bool:raise ValueError('target mask differs')
    losses=F.binary_cross_entropy_with_logits(logits,y,reduction='none');counts=mask.sum(0);active=counts>0
    means=(losses*mask).sum(0)/counts.clamp_min(1)
    return means[active].mean() if active.any() else logits.sum()*0

class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.encoder=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,64),nn.GELU(),nn.Linear(64,32))
            self.decoder=nn.Linear(32,10)
    def forward(self,x):
        h=self.encoder(x);return h,self.decoder(h)

def task_split(subjects,train,present,mask,y):
    use=train[present[train]&mask[train].any(1)];fit_ix=use[subjects[use]%5!=0];hold=use[subjects[use]%5==0]
    if set(subjects[fit_ix])&set(subjects[hold]):raise ValueError('aux patient conflict')
    count=mask[fit_ix].sum(0);positive=(y[fit_ix]*mask[fit_ix]).sum(0);negative=count-positive
    active=(count>=100)&(positive>=20)&(negative>=20)
    if not active.any() or len(hold)<100:raise ValueError('insufficient valid auxiliary targets')
    return fit_ix,hold,active

def readout(logits,y,mask,prevalence):
    probabilities=1/(1+np.exp(-np.clip(logits,-30,30)));results={}
    for j,name in enumerate(HEADS):
        use=mask[:,j];yy=y[use,j];pp=np.clip(probabilities[use,j],1e-7,1-1e-7);q=float(np.clip(prevalence[j],1e-7,1-1e-7))
        results[name]=dict(observed=int(use.sum()),positive=int(yy.sum()),
            bce=float(-(yy*np.log(pp)+(1-yy)*np.log(1-pp)).mean()) if len(yy) else None,
            training_prevalence_constant_bce=float(-(yy*np.log(q)+(1-yy)*np.log(1-q)).mean()) if len(yy) else None,
            auc=float(auc_rank(pp,yy.astype(np.int8))) if 0<yy.sum()<len(yy) else None)
    return results

def auxiliary(d,text,present,y,mask,train):
    fit_ix,hold,active=task_split(d.arrays['subject_ids'],train,present,mask,y)
    used=mask&active;tensor=lambda a:torch.as_tensor(a,device='cuda')
    x=tensor(text);yy=tensor(y);mm=tensor(used);m=Reader().cuda();rng=np.random.default_rng(42)
    groups=[fit_ix[d.arrays['subject_ids'][fit_ix]==s] for s in np.unique(d.arrays['subject_ids'][fit_ix])]
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=1e-4);early=EarlyStopping(4,5,1e-4);history=[];reason='max_epochs'
    for epoch in range(21):
        if epoch:
            indices=np.asarray([rng.choice(g) for g in groups]);rng.shuffle(indices)
            for start in range(0,len(indices),256):
                ix=indices[start:start+256];opt.zero_grad(set_to_none=True);_,logits=m(x[ix]);loss=masked_bce(logits,yy[ix],mm[ix])
                if not torch.isfinite(loss):raise ValueError('nonfinite auxiliary loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        with torch.no_grad():_,logits=m(x[hold]);value=float(masked_bce(logits,yy[hold],mm[hold]))
        best,stop=early.observe(epoch,value);history.append(dict(epoch=epoch,internal_holdout_bce=value))
        if best:torch.save(m.state_dict(),ROOT/'reader_best.pt')
        if stop:reason='patience_exhausted';break
    m.load_state_dict(torch.load(ROOT/'reader_best.pt',map_location='cuda',weights_only=True));m.eval()
    counts=used[fit_ix].sum(0);prevalence=(y[fit_ix]*used[fit_ix]).sum(0)/np.maximum(counts,1)
    with torch.no_grad():_,logits=m(x[hold])
    detail=dict(history=history,early=asdict(early),actual_epochs=history[-1]['epoch'],stop_reason=reason,active_heads=active.tolist(),
        fit_admissions=len(fit_ix),holdout_admissions=len(hold),fit_patients=len(groups),fit_indices_sha256=digest({'fit':fit_ix,'hold':hold}),
        prevalence=prevalence.tolist(),readout=readout(logits.cpu().numpy(),y[hold],used[hold],prevalence),
        reader_sha256=sha256(ROOT/'reader_best.pt'),no_outer_validation_used=True,no_readmission_death_targets=True)
    write_json(ROOT/'auxiliary.json',detail)

def representation(text,present,train,arm):
    m=Reader().cuda().eval()
    if arm=='concept':m.load_state_dict(torch.load(ROOT/'reader_best.pt',map_location='cuda',weights_only=True))
    result=[]
    with torch.no_grad():
        for start in range(0,len(text),512):result.append(m(torch.as_tensor(text[start:start+512],device='cuda'))[0].cpu().numpy())
    z=np.concatenate(result);use=train[present[train]];mean=z[use].mean(0);scale=np.maximum(z[use].std(0),.001)
    z=np.clip((z-mean)/scale,-8,8);z[~present]=0
    return z.astype(np.float32),dict(mean=mean.tolist(),scale=scale.tolist(),training_records=len(use))

def batches(frozen,z,present,ix):
    result=[];offset=0
    for f in frozen:
        use=ix[offset:offset+len(f.hidden)];offset+=len(use)
        result.append(Batch(f,torch.as_tensor(z[use],device='cuda'),torch.as_tensor(np.stack((present[use],np.ones(len(use),bool)),1),device='cuda',dtype=torch.float32)))
    if offset!=len(ix):raise ValueError('frozen cache alignment changed')
    return result

def setup():
    data=checked();d=data[0];train=np.flatnonzero(d.arrays['patient_splits']=='train');tune=data[6]
    text,present=course(d);y,mask,age=targets(d.arrays)
    sources={**data[8]['source_sha256'],**data[9]['source_sha256']}
    for p in (Path(__file__),PROTOCOL,Path('training/tests/test_v6_text_concept.py'),Path('training/v6/fast_multiview.py'),
              Path('training/v6/curve_adaptation.py'),Path('training/v6/early_stopping.py'),Path('patient_data/v6_lab_state.py')):sources[str(p)]=sha256(p)
    contract=dict(source_sha256=sources,data_provenance=d.provenance,compact_provenance=data[1].provenance,
        teacher_sha256=sha256(REFERENCE/'seed42/control/best.pt'),inputs_sha256=digest(dict(text=text,present=present,y=y,mask=mask,age=age)),
        protocol_sha256=sha256(PROTOCOL),heads=list(HEADS),seed=42,arms=list(ARMS),high_frequency_used=False,confirmation_test_used=False)
    return data,train,tune,text,present,y,mask,age,contract

def lock(r):
    scores={a:r['fits'][a]['metrics']['readmission']['integrated_brier_7_365'] for a in ARMS}
    scores['original']=r['reference']['readmission']['integrated_brier_7_365'];chosen=min(scores,key=scores.get)
    return dict(scores=scores,selected=chosen,nominated=['concept'] if chosen=='concept' and scores['concept']<min(scores['random'],scores['original']) else [],goal_achieved=False)

def run():
    if ROOT.exists():raise FileExistsError('existing batch; inspect')
    started=time.perf_counter();data,train,tune,text,present,y,mask,age,contract=setup();d=data[0]
    fit_ix,hold,active=task_split(d.arrays['subject_ids'],train,present,mask,y);ROOT.mkdir();write_json(ROOT/'contract.json',contract)
    audit={}
    for j,name in enumerate(HEADS):
        use=train[present[train]&mask[train,j]];a=age[use,j//2]
        audit[name]=dict(training_observed_with_course=len(use),positive=int(y[use,j].sum()),negative=int(len(use)-y[use,j].sum()),
            latest_measurement_age_median_days=float(np.median(np.expm1(a))) if len(a) else None,role='whole_admission_peak' if j%2==0 else 'latest_observed_not_discharge')
    write_json(ROOT/'target_audit.json',dict(heads=audit,active_heads=active.tolist(),training_text_admissions=int(present[train].sum()),
        no_target_for_unobserved=True,no_original_database_rescan=True))
    tc,tp=caches(data,train,42);vc,vp=caches(data,tune,42)
    from training.v6.curve_adaptation import Correction as Zero,prediction as original_prediction
    _,_,ref,nll=original_prediction(Zero('clinical_nll').cuda(),vc)
    z,_=representation(text,present,train,'random');cache=batches(tc,z,present,train)
    observed=np.flatnonzero(present[train]);slot=int(observed[0]);bi,ri=divmod(slot,512);probe=cache[bi]
    m=Correction().cuda();out=m(probe);old=probe.frozen.raw.reshape(-1,2,2)
    torch.testing.assert_close(out.log_amplitude,(probe.frozen.prior+old[:,:,0]).clamp(-10,6),atol=0,rtol=0)
    torch.testing.assert_close(out.shape,F.softplus(probe.frozen.shape_raw+old[:,:,1])+.1,atol=0,rtol=0)
    opt=torch.optim.AdamW(m.parameters(),lr=.001);loss=competing_nll(out,probe.frozen.times,probe.frozen.events);loss.backward();opt.step()
    if not m.head[-1].weight.ne(0).any():raise ValueError('no real update')
    missing=~probe.masks.bool().all(1);updated=m(probe)
    if missing.any():torch.testing.assert_close(updated.log_amplitude[missing],out.log_amplitude[missing],atol=0,rtol=0)
    write_json(ROOT/'preflight.json',dict(status='passed',initial_risk_error=0.,real_update=True,missing_gate=True,gpu=torch.cuda.get_device_name()))
    write_json(ROOT/'state.json',dict(status='auxiliary_training',goal_achieved=False));auxiliary(d,text,present,y,mask,train)
    r=dict(fits={},features={},teacher_caches={'train':tp,'tune':vp},reference=summary(ref,d,tune,nll),single_seed_only=True)
    for arm in ARMS:
        z,scale=representation(text,present,train,arm);r['features'][arm]={'sha256':digest({'z':z}),'scale':scale}
        tb=batches(tc,z,present,train);vb=batches(vc,z,present,tune)
        np.testing.assert_array_equal(prediction(Correction().cuda(),vb)[2],ref)
        write_json(ROOT/'state.json',dict(status='risk_training',arm=arm,goal_achieved=False));r['fits'][arm]=fit(tb,vb,d,tune,ROOT/arm)
    r['lock']=lock(r);r['seconds']=time.perf_counter()-started;write_json(ROOT/'screening.json',r);write_json(ROOT/'nomination_lock.json',r['lock'])
    write_json(ROOT/'state.json',dict(status='complete_pending_independent_replay',goal_achieved=False));print(json.dumps(r['lock']),flush=True)

def replay_early(history,info,aux=False):
    early=EarlyStopping(4,5,1e-4) if aux else EarlyStopping(7,10,1e-5);first_stop=None
    for row in history:
        _,stop=early.observe(row['epoch'],row['internal_holdout_bce'] if aux else row['tuning_ibs'])
        if stop and first_stop is None:first_stop=row['epoch']
    expected=info['actual_epochs'] if aux else info['completed_epochs']
    if asdict(early)!=info['early'] or expected!=len(history)-1:raise ValueError('early state differs')
    if info['stop_reason']=='patience_exhausted' and first_stop!=expected:raise ValueError('stopping delayed')
    return early

def verify():
    data,train,tune,text,present,y,mask,age,contract=setup();d=data[0]
    if contract!=json.loads((ROOT/'contract.json').read_text()):raise ValueError('contract/source differs')
    aux=json.loads((ROOT/'auxiliary.json').read_text());replay_early(aux['history'],aux,True)
    fi,ho,active=task_split(d.arrays['subject_ids'],train,present,mask,y)
    if digest({'fit':fi,'hold':ho})!=aux['fit_indices_sha256'] or active.tolist()!=aux['active_heads'] or sha256(ROOT/'reader_best.pt')!=aux['reader_sha256']:raise ValueError('aux source/model changed')
    m=Reader().cuda().eval();m.load_state_dict(torch.load(ROOT/'reader_best.pt',map_location='cuda',weights_only=True))
    # Re-establish the same strict FP32 policy as all frozen risk caches.
    torch.backends.cuda.matmul.allow_tf32=False
    with torch.no_grad():_,logits=m(torch.as_tensor(text[ho],device='cuda'))
    if readout(logits.cpu().numpy(),y[ho],mask[ho]&active,np.asarray(aux['prevalence']))!=aux['readout']:raise ValueError('auxiliary prediction differs')
    tc,tp=caches(data,train,42);vc,vp=caches(data,tune,42);r=json.loads((ROOT/'screening.json').read_text());errors={}
    if r['teacher_caches']!={'train':tp,'tune':vp}:raise ValueError('teacher cache differs')
    ref=r_curve(data,vc)
    for arm in ARMS:
        z,scale=representation(text,present,train,arm)
        if digest({'z':z})!=r['features'][arm]['sha256'] or scale!=r['features'][arm]['scale']:raise ValueError('representation differs')
        folder=ROOT/arm;info=r['fits'][arm];early=replay_early(json.loads((folder/'history.json').read_text()),info)
        state=torch.load(folder/'best.pt',map_location='cuda',weights_only=True)
        if sha256(folder/'best.pt')!=info['checkpoint_sha256'] or state['epoch']!=early.best_epoch:raise ValueError('saved selection differs')
        model=Correction().cuda();model.load_state_dict(state['state_dict']);a,k,curve,nll=prediction(model,batches(vc,z,present,tune))
        with np.load(folder/'tuning_predictions.npz') as zold:
            np.testing.assert_array_equal(zold['times'],d.times[tune]);np.testing.assert_array_equal(zold['events'],d.events[tune])
            error=max(float(np.abs(a-zold['log_amplitude']).max()),float(np.abs(k-zold['shape']).max()),float(np.abs(curve-zold['curve']).max()))
        if error!=0 or summary(curve,d,tune,nll)!=info['metrics']:raise ValueError('risk metrics differs')
        np.testing.assert_array_equal(curve[~present[tune]],ref[~present[tune]]);errors[arm]=error
    if lock(r)!=r['lock'] or r['lock']!=json.loads((ROOT/'nomination_lock.json').read_text()):raise ValueError('nomination differs')
    write_json(ROOT/'verification.json',dict(status='independent_replay_passed',maximum_error=max(errors.values()),errors=errors,
        auxiliary_replayed=True,full_source_targets_features_scales_teacher_caches_rebuilt=True,screening_sha256=sha256(ROOT/'screening.json')))
    write_json(ROOT/'state.json',dict(status='single_seed_independently_verified',goal_achieved=False));print(json.dumps(errors),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('run','verify'),required=True);mode=p.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as e:
        if ROOT.exists():write_json(ROOT/f'failure_{mode}.json',{'error':repr(e)})
        raise
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/text_encoder_comparison.py · lines 1–268

<!-- REVIEW_FRAGMENT {"path":"training/v6/text_encoder_comparison.py","part":1,"start_line":1,"end_line":268,"source_sha256":"8319a4c1af6fc6e5886a38ecbab06873eb139c737d48882f52e26be2ba6d7ec5","fragment_sha256":"8319a4c1af6fc6e5886a38ecbab06873eb139c737d48882f52e26be2ba6d7ec5","added_terminal_newline":false} -->
````python
"""相同B四字段冻结向量的BERT/Qwen配对重训；既有源代码不改。"""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from patient_data.v6_lab_state import AXES, NUMERIC, FLAGS, MED_GROUPS
from patient_data.compact_multisource import MED_CHANNELS, MICRO_NAMES, SUPPORT_NAMES
from training.v6.data import load, HISTORY_NAMES, CURRENT_NAMES
from training.v6.compact_context_data import attach
from training.v6.compact_ablation import inputs
from training.v6.architecture_experiment import batches
from training.v6.horizon_objective import (labels90, binary_metrics, paired90, ranked_curves,
    make_model, restore, fit90, predict90)
from training.v6.evaluation import GRID, metrics, public_metrics, paired_interval
from training.v6.risk import predict_cif
from training.v6.train import write_json

ARMS = ('bert', 'qwen', 'none_768', 'none_2560')
PAIRS = (('qwen','bert'), ('bert','none_768'), ('qwen','none_768'))
PROTOCOL = Path('docs/V6_QWEN_COMPARISON_PROTOCOL_2026-10-11.md')


def shared_parameters(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()
            if not k.startswith('encoder.text_projection.')}


def checked_inputs(args):
    parent=json.loads((args.parent/'report.json').read_text())
    proof=json.loads((args.parent/'independent_verification.json').read_text())
    if proof['report_sha256'] != sha256(args.parent/'report.json'):
        raise ValueError('parent report not verified')
    for p,digest in parent['contract']['code_sha256'].items():
        if sha256(Path(p)) != digest: raise ValueError('parent source changed: '+p)
    data,compact,hist,_=inputs(args.reference,args.compact_dir)
    if data.provenance != parent['contract']['data_provenance'] or compact.provenance != parent['contract']['compact_provenance']:
        raise ValueError('parent input changed')
    cfg=json.loads((args.reference/'config.json').read_text())
    field_dir=Path(cfg['field_dir'])
    qwen=load(Path(cfg['snapshot']),Path(cfg['state_dir']),Path(cfg['labels']),
              field_dir=field_dir,vectors_path=field_dir/'qwen/qwen3_embedding.float16.npy')
    left,right=copy.deepcopy(data.provenance),copy.deepcopy(qwen.provenance)
    left.pop('text'); right.pop('text')
    if (left != right or data.text_refs != qwen.text_refs
            or data.text_vectors.shape != (76974,768) or qwen.text_vectors.shape != (76974,2560)):
        raise ValueError('encoder content/index/input mismatch')
    for key in ('manifest_sha256','rows','text_admissions'):
        if data.provenance['text'][key] != qwen.provenance['text'][key]:
            raise ValueError('shared text mismatch')
    audit={}
    for name,item in (('bert',data),('qwen',qwen)):
        norms=np.concatenate([np.linalg.norm(np.asarray(item.text_vectors[j:j+4096],np.float32),axis=1)
                              for j in range(0,len(item.text_vectors),4096)])
        if not np.isfinite(norms).all() or not np.all((norms>.99)&(norms<1.01)):
            raise ValueError('expected complete L2 vectors')
        audit[name]=dict(**item.provenance['text'],norm_min=float(norms.min()),norm_max=float(norms.max()))
    return {'bert':data,'qwen':qwen}, {'bert':compact,'qwen':attach(qwen,args.compact_dir)}, hist,parent,audit


def prepared(data,compact,idx,size,arm):
    return batches(data,compact,idx,size,expanded=True,text='none' if arm.startswith('none') else 'bert')


def arm_source(arm):
    return 'qwen' if arm in ('qwen','none_2560') else 'bert'


def new_model(args,seed,arm,hist,prevalence):
    base=restore(args.parent,seed,'survival_base',hist,prevalence)
    return make_model('survival_full',base,2560 if arm_source(arm)=='qwen' else 768,seed)


def read_prediction(root,seed,arm,fold):
    with np.load(root/f'seed{seed}'/arm/(fold+'_predictions.npz')) as z:
        return {k:z[k] for k in z.files}


def assert_empty_equal(a,b):
    keys=('epoch','train_objective','validation_objective','validation_brier90','validation_auc90','validation_ap')
    if len(a)!=len(b): raise ValueError('empty-text training length differs')
    for x,y in zip(a,b,strict=True):
        if any(x[k]!=y[k] for k in keys): raise ValueError('empty-text RNG/optimization differs')


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2); started=time.perf_counter()
    data,compact,hist,parent,audit=checked_inputs(args)
    d=data['bert']; y=labels90(d.times,d.events)
    ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('train','validation','test')}
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    code={**parent['contract']['code_sha256'],'training/v6/text_encoder_comparison.py':sha256(Path(__file__))}
    contract=dict(parent_report_sha256=sha256(args.parent/'report.json'),text_audit=audit,
        data_provenance=d.provenance,compact_provenance=compact['bert'].provenance,
        protocol_sha256=sha256(PROTOCOL),code_sha256=code,seeds=[42,7],arms=list(ARMS),
        selection='validation Brier90',lr=3e-4,epochs=60,patience=10,warmup=5,batch_size=512,
        device=torch.cuda.get_device_name(0),endpoint='strict491.21/J44.1; World A; competing death',
        limits=['reused test; no multiplicity adjustment; not blind',
                'text projection parameter counts differ: 768 vs2560; not identity-only attribution',
                'expanded baseline is a mechanism reference, not replacement of old default H/B',
                'text may reconstruct history; no causal claim or complete outside follow-up'])
    write_json(args.output/'contract.json',contract)
    catalog=dict(history=list(HISTORY_NAMES),current=list(CURRENT_NAMES),lab_axes=list(AXES),
        lab_channels=list(NUMERIC+FLAGS),med_groups=list(MED_GROUPS),med_channels=list(MED_CHANNELS),
        micro=list(MICRO_NAMES),support=list(SUPPORT_NAMES),text=audit,
        source_coverage=json.loads((args.compact_dir/'manifest.json').read_text()),
        lab_coverage=json.loads((Path(json.loads((args.reference/'config.json').read_text())['state_dir'])/'manifest.json').read_text()))
    write_json(args.output/'input_catalog.json',catalog)
    cached={key:{f:prepared(data[key],compact[key],ix[f],512,key) for f in ('train','validation')}
            for key in ('bert','qwen')}
    fitted={}
    for seed in (42,7):
        (args.output/f'seed{seed}').mkdir(); common=None
        for arm in ARMS:
            source=arm_source(arm); m=new_model(args,seed,arm,hist,y[ix['train']].mean())
            initial=shared_parameters(m)
            if common is None: common=initial
            elif common.keys()!=initial.keys() or not all(torch.equal(v,common[k]) for k,v in initial.items()):
                raise ValueError('non-text initial state differs')
            cache=cached[source]
            if arm.startswith('none'):
                cache={f:prepared(data[source],compact[source],ix[f],512,arm) for f in ('train','validation')}
            write_json(args.output/'status.json',dict(status='training',seed=seed,arm=arm))
            fitted[seed,arm]=fit90(m,cache['train'],cache['validation'],y[ix['validation']],
                args.output/f'seed{seed}'/arm,epochs=60,lr=3e-4,seed=seed,patience=10,warmup=5)
            del m
        histories=[json.loads((args.output/f'seed{seed}'/a/'history.json').read_text()) for a in ('none_768','none_2560')]
        assert_empty_equal(*histories)
    write_json(args.output/'status.json',dict(status='all_eight_checkpoints_selected_before_test'))
    results,comparisons,parent_errors={},{},{}
    for seed in (42,7):
        results[str(seed)]={}; comparisons[str(seed)]={}
        for arm in ARMS:
            source=arm_source(arm); m=new_model(args,seed,arm,hist,y[ix['train']].mean())
            m.load_state_dict(torch.load(args.output/f'seed{seed}'/arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            results[str(seed)][arm]={**fitted[seed,arm],'folds':{}}
            for fold in ('validation','test'):
                idx=ix[fold]; pred=predict90(m,prepared(data[source],compact[source],idx,512,arm))
                pred['curve']=predict_cif(pred['parameters'],pred['shape'],GRID)
                if not np.allclose(pred['curve'].sum(2),1.,atol=1e-9): raise ValueError('mass conservation')
                fine=predict_cif(pred['parameters'][:64],pred['shape'][:64],GRID,step_days=.125)
                if np.abs(fine-pred['curve'][:64]).max()>2e-4: raise ValueError('CIF step convergence failed')
                if not np.allclose(pred['risk90'],pred['curve'][:,90,1],atol=2e-6,rtol=1e-5): raise ValueError('90-day CIF query differs')
                detail=binary_metrics(y[idx],pred['risk90'])
                detail['survival_readmission']=public_metrics(metrics(d.times[idx],d.events[idx],pred['curve'][:,:,1],ordering=True))
                detail['survival_death']=public_metrics(metrics(d.times[idx],d.events[idx],pred['curve'][:,:,2],cause=2,ordering=True))
                results[str(seed)][arm]['folds'][fold]=detail
                np.savez_compressed(args.output/f'seed{seed}'/arm/(fold+'_predictions.npz'),
                    hadm_ids=d.arrays['hadm_ids'][idx],subject_ids=d.arrays['subject_ids'][idx],times=d.times[idx],events=d.events[idx],y90=y[idx],
                    **{k:v for k,v in pred.items() if k!='loss'})
                if arm=='bert':
                    previous=read_prediction(args.parent,seed,'survival_full',fold)
                    if not np.array_equal(previous['hadm_ids'],d.arrays['hadm_ids'][idx]) or not np.allclose(previous['risk90'],pred['risk90'],atol=2e-6,rtol=1e-5):
                        raise ValueError('BERT retraining does not reproduce parent recipe')
                    parent_errors[f'{seed}/{fold}']=float(np.abs(previous['risk90']-pred['risk90']).max())
            del m
        for fold in ('validation','test'):
            idx=ix[fold]; comparisons[str(seed)][fold]={}
            a,b=(read_prediction(args.output,seed,arm,fold) for arm in ('none_768','none_2560'))
            if not np.array_equal(a['risk90'],b['risk90']) or not np.array_equal(a['curve'],b['curve']):
                raise ValueError('empty-text prediction differs across encoder widths')
            for left,right in PAIRS:
                a,b=(read_prediction(args.output,seed,arm,fold) for arm in (left,right))
                comparison=paired90(d.arrays['subject_ids'][idx],y[idx],a['risk90'],b['risk90'],replicates=200,seed=seed)
                comparison['integrated']=paired_interval(d.arrays['subject_ids'][idx],d.times[idx],d.events[idx],a['curve'][:,:,1],b['curve'][:,:,1],replicates=200,seed=seed)
                comparisons[str(seed)][fold][left+'_minus_'+right]=comparison
    report=dict(status='completed_same_text_bert_qwen_two_seeds',contract=contract,models=results,
        comparisons=comparisons,nontext_initial_states_equal=True,empty_text_histories_and_predictions_identical=True,
        bert_parent_retraining_errors=parent_errors,elapsed_seconds=time.perf_counter()-started)
    write_json(args.output/'report.json',report); render(args.output,report)
    write_json(args.output/'status.json',dict(status=report['status'],seconds=report['elapsed_seconds']))
    print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'])),flush=True)


def verify(args):
    report=json.loads((args.output/'report.json').read_text()); c=report['contract']
    for path,digest in c['code_sha256'].items():
        if sha256(Path(path))!=digest: raise ValueError('source changed: '+path)
    if sha256(PROTOCOL)!=c['protocol_sha256']: raise ValueError('protocol changed')
    data,compact,hist,parent,audit=checked_inputs(args)
    if audit!=c['text_audit'] or sha256(args.parent/'report.json')!=c['parent_report_sha256']: raise ValueError('asset changed')
    d=data['bert']; y=labels90(d.times,d.events); errors={}
    for seed in (42,7):
        for arm in ARMS:
            path=args.output/f'seed{seed}'/arm/'best.pt'
            if sha256(path)!=report['models'][str(seed)][arm]['checkpoint_sha256']: raise ValueError('checkpoint changed')
            m=new_model(args,seed,arm,hist,y[d.arrays['patient_splits']=='train'].mean())
            frozen=copy.deepcopy(m.baseline.state_dict())
            m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
            if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in frozen.items()): raise ValueError('baseline changed')
            source=arm_source(arm)
            for fold in ('validation','test'):
                idx=np.flatnonzero(d.arrays['patient_splits']==fold)
                p=predict90(m,prepared(data[source],compact[source],idx,512,arm))
                p['curve']=predict_cif(p['parameters'],p['shape'],GRID)
                saved=read_prediction(args.output,seed,arm,fold)
                for key,value in (('hadm_ids',d.arrays['hadm_ids'][idx]),('subject_ids',d.arrays['subject_ids'][idx]),('times',d.times[idx]),('events',d.events[idx]),('y90',y[idx])):
                    if not np.array_equal(saved[key],value): raise ValueError('identity/label mismatch')
                for key,value in p.items():
                    if key!='loss' and not np.allclose(saved[key],value,atol=2e-6,rtol=1e-5): raise ValueError('reload mismatch')
                detail=binary_metrics(y[idx],p['risk90'])
                for key in ('auc90','average_precision','brier90','logloss90'):
                    if abs(detail[key]-report['models'][str(seed)][arm]['folds'][fold][key])>1e-7: raise ValueError('metric mismatch')
                for name,cause in (('survival_readmission',1),('survival_death',2)):
                    actual=public_metrics(metrics(d.times[idx],d.events[idx],p['curve'][:,:,cause],cause=cause,ordering=True))
                    expected=report['models'][str(seed)][arm]['folds'][fold][name]
                    if actual!=expected: raise ValueError('survival metric mismatch')
                errors[f'{seed}/{arm}/{fold}']=float(np.abs(saved['risk90']-p['risk90']).max())
            del m
    write_json(args.output/'independent_verification.json',dict(status='all8_models_all16_folds_gpu_reloaded',
        report_sha256=sha256(args.output/'report.json'),baseline_unchanged=True,errors=errors))
    print('all8_models_all16_folds_gpu_reloaded',flush=True)


def render(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures'; folder.mkdir()
    names=('none_768','bert','qwen'); labels=('No text','Bio_ClinicalBERT','Qwen3-Embedding-4B')
    for seed in (42,7):
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        for arm,label in zip(names,labels,strict=True):
            p=read_prediction(root,seed,arm,'test'); m=report['models'][str(seed)][arm]['folds']['test']
            curves=ranked_curves(p['y90'],p['risk90'])
            axes[0].plot(curves['fpr'],curves['tpr'],label=f'{label}: AUC90={m["auc90"]:.3f}')
            actual=metrics(p['times'],p['events'],p['curve'][:,:,1])
            axes[1].plot(GRID[7:],actual['auc_curve'][7:],label=label)
            b=m['calibration_bins']; axes[2].plot([x['predicted'] for x in b],[x['observed'] for x in b],'o-',label=label)
        axes[0].plot([0,1],[0,1],'k--',alpha=.3); axes[2].plot([0,1],[0,1],'k--',alpha=.3)
        axes[2].set_xlim(0,.6); axes[2].set_ylim(0,.6)
        for ax,title,x,y in zip(axes,('90-day ROC','Time-dependent AUC(t)','90-day Calibration'),
            ('False positive rate','Days after discharge','Mean predicted risk'),('True positive rate','AUC(t)','Observed fraction')):
            ax.set(title=title,xlabel=x,ylabel=y); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle(f'Same four fields; frozen expanded baseline; seed{seed}; reused test')
        fig.tight_layout(); fig.savefig(folder/f'test_seed{seed}_curves.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
    for j,(left,right) in enumerate(PAIRS):
        for i,seed in enumerate((42,7)):
            d=report['comparisons'][str(seed)]['test'][left+'_minus_'+right]['metrics']
            for ax,key in zip(axes,('brier90','auc90')):
                r=d[key]; yy=j+(i-.5)*.16; color=('#3465a4','#e6550d')[i]
                ax.plot(r['ci95'],[yy,yy],color=color,alpha=.6); ax.scatter(r['difference'],yy,color=color,label=f'seed{seed}' if j==0 else None)
    for ax in axes:
        ax.axvline(0,color='black',linewidth=1); ax.grid(axis='x',alpha=.2); ax.legend()
        ax.set_yticks(range(3),['Qwen minus BERT','BERT minus no text','Qwen minus no text'])
    axes[0].invert_yaxis(); axes[0].set_xlabel('Paired Brier90 difference (lower better)'); axes[1].set_xlabel('Paired AUC90 difference (higher better)')
    fig.suptitle('Reused test; 200 patient-clustered 95% intervals; no multiplicity adjustment')
    fig.tight_layout(); fig.savefig(folder/'test_paired_two_seeds.png',dpi=170); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--compact-dir',type=Path,default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--parent',type=Path,default=Path('training_runs/v6_90day_objective_2026-10-11'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--verify',action='store_true')
    args=p.parse_args(); (verify if args.verify else run)(args)


if __name__=='__main__': main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/text_geometry.py · lines 1–40

<!-- REVIEW_FRAGMENT {"path":"training/v6/text_geometry.py","part":1,"start_line":1,"end_line":40,"source_sha256":"0f247dae5252b98849d55933c23d4cbc380fff6cb83d705d9c618b232a2be0c8","fragment_sha256":"0f247dae5252b98849d55933c23d4cbc380fff6cb83d705d9c618b232a2be0c8","added_terminal_newline":false} -->
````python
"""训练折拟合的章节平衡冻结向量后处理，不读取结局拟合变换。"""
import numpy as np
import torch


def fit_geometry(vectors, indices, weights, fields=4):
    if vectors.ndim!=2 or indices.shape!=(len(vectors),) or weights.shape!=(len(vectors),):
        raise ValueError('invalid geometry rows')
    means,matrices,diagnostics=[],[],[]
    for field in range(fields):
        use=indices==field
        if int(use.sum())<2:raise ValueError('insufficient training field vectors')
        x=vectors[use].double();w=weights[use].double()
        if not torch.isfinite(x).all() or not torch.isfinite(w).all() or (w<=0).any():raise ValueError('nonfinite geometry')
        w=w/w.sum();mean=(x*w[:,None]).sum(0);center=x-mean
        covariance=center.T@(center*w[:,None]);covariance=(covariance+covariance.T)*.5
        eigen,rotation=torch.linalg.eigh(covariance)
        eigen=eigen.clamp_min(0);average=eigen.mean()
        if average<=0:raise ValueError('constant field representation')
        # 固定10%球形收缩，避免小特征值噪声无限放大；不是原论文逐式复现。
        shrunk=.9*eigen+.1*average
        transform=(rotation*shrunk.rsqrt()[None,:])@rotation.T
        means.append(mean.float());matrices.append(transform.float())
        diagnostics.append(dict(field=field,training_rows=len(x),mean_norm=float(mean.norm()),
            covariance_participation_rank=float(eigen.sum().square()/eigen.square().sum()),
            top1_variance_fraction=float(eigen[-1]/eigen.sum()),top10_variance_fraction=float(eigen[-10:].sum()/eigen.sum()),
            shrinkage=.1,maximum_gain=float(shrunk.rsqrt().max())))
    return torch.stack(means),torch.stack(matrices),diagnostics


def transform_rows(vectors, field, means, matrices=None):
    result=torch.empty_like(vectors)
    for f in range(len(means)):
        keep=field==f
        if not keep.any():continue
        x=vectors[keep]-means[f]
        if matrices is not None:x=x@matrices[f]
        result[keep]=torch.nn.functional.normalize(x,p=2,dim=1,eps=1e-8)
    if not torch.isfinite(result).all():raise ValueError('nonfinite transformed vectors')
    return result
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/text_time_attention_audit.py · lines 1–68

<!-- REVIEW_FRAGMENT {"path":"training/v6/text_time_attention_audit.py","part":1,"start_line":1,"end_line":68,"source_sha256":"276ccd801ae73bb3e11aeed805fe50115772f84a977753440db59367a38afe33","fragment_sha256":"276ccd801ae73bb3e11aeed805fe50115772f84a977753440db59367a38afe33","added_terminal_newline":false} -->
````python
"""仅冻结已训练文本读取器的字段内权重审计；权重不是风险贡献或临床解释。"""
from collections import defaultdict
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from patient_data.export_mimiciv import rows,sha256
from patient_data.text_time_coverage import ROOT,B,STATE,checked_sources
from training.v6.train import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',choices=('a','b'),required=True);args=p.parse_args()
    sources,_=checked_sources();audit=json.loads((ROOT/'audit.json').read_text())
    if sources!=audit['source_sha256']:raise ValueError('source audit differs')
    with np.load(STATE/'clinical_state.npz') as z:train={str(h) for h,f in zip(z['hadm_ids'],z['patient_splits']) if f=='train'}
    with gzip.open(ROOT/'priority_candidate_index.jsonl.gz','rt',encoding='utf-8') as f:
        priority={int(j) for r in map(json.loads,f) for j in r['B_containing_rows']}
    fields={'history_of_present_illness':0,'past_medical_history':1,'hospital_course':2,'discharge_condition':3}
    chunks=[r for r in rows(B/'shared_text.csv.gz') if r['hadm_id'] in train]
    groups=defaultdict(list)
    for i,r in enumerate(chunks):groups[r['hadm_id'],r['field']].append(i)
    vectors=np.load(B/'bert/bioclinicalbert.float16.npy',mmap_mode='r');result={}
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for seed in (42,7):
        path=Path(f'training_runs/v6_module_surgery_2026-10-12_r3/seed{seed}/control/best.pt')
        expected=json.loads(Path('training_runs/v6_warm_residual_2026-10-13/contract.json').read_text())['teacher_checkpoint_sha256'][str(seed)]
        if sha256(path)!=expected:raise ValueError('original teacher changed')
        state=torch.load(path,map_location='cpu',weights_only=True)['state_dict'];prefix='encoder.base.'
        projection=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,128),nn.GELU(),nn.LayerNorm(128)).cuda().eval().requires_grad_(False)
        field=nn.Embedding(4,128).cuda().eval().requires_grad_(False);score=nn.Linear(128,1).cuda().eval().requires_grad_(False)
        for name,module in [('text_projection',projection),('field_ids',field),('text_score',score)]:
            weights={k.removeprefix(prefix+name+'.'):v for k,v in state.items() if k.startswith(prefix+name+'.')}
            module.load_state_dict(weights,strict=True)
        logits=[]
        with torch.inference_mode():
            for start in range(0,len(chunks),1024):
                batch=chunks[start:start+1024]
                x=torch.as_tensor(np.asarray(vectors[[int(r['embedding_row']) for r in batch]],np.float32),device='cuda')
                ids=torch.tensor([fields[r['field']] for r in batch],device='cuda')
                logits.extend(score(projection(x)+field(ids)).squeeze(-1).cpu().tolist())
        shares=[];uniforms=[];multipliers=[];group_counts=defaultdict(int)
        for (_,role),ix in groups.items():
            tagged=np.array([int(chunks[i]['embedding_row']) in priority for i in ix])
            if not tagged.any():continue
            raw=torch.tensor([logits[i] for i in ix],dtype=torch.float32);w=torch.softmax(raw,0).numpy()
            shares.append(float(w[tagged].sum()));uniforms.append(float(tagged.mean()));multipliers.append(shares[-1]/uniforms[-1]);group_counts[role]+=1
        def q(x):return np.quantile(x,[0,.1,.5,.9,1]).tolist()
        result[str(seed)]=dict(teacher_sha256=expected,groups_with_priority_bearing_full_chunk=len(shares),group_counts=dict(group_counts),
            attention_share_quantiles=q(shares),uniform_share_quantiles=q(uniforms),attention_to_uniform_ratio_quantiles=q(multipliers),
            groups_priority_weight_below_uniform=int(sum(a<b for a,b in zip(shares,uniforms))),
            logits_sha256=__import__('hashlib').sha256(np.asarray(logits,np.float32).tobytes()).hexdigest())
    proof=dict(status='frozen_original_text_reader_training_only_weight_audit',teachers=result,audit_sha256=sha256(ROOT/'audit.json'),
        code_sha256=sha256(Path(__file__)),training_chunks=len(chunks),GPU=torch.cuda.get_device_name(0),new_fit=False,
        caveats=['Subset extracted from actual verified teacher weights; no new optimization or risk metrics',
            'Only full-contained lexical priority candidate chunks tagged;54 cross-boundary units not tagged by this rule',
            'Attention weights are not causal attribution or proof of ignored clinical information',
            'FP32 score audit; does not claim entire original model risk replay',
            'Within-field learned attention; final available-field averaging is a separate operation'])
    write_json(ROOT/f'attention_audit_{args.tag}.json',proof)
    if args.tag=='b' and json.loads((ROOT/'attention_audit_a.json').read_text())!=proof:raise ValueError('independent attention audit differs')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/text_time_coverage_report.py · lines 1–42

<!-- REVIEW_FRAGMENT {"path":"training/v6/text_time_coverage_report.py","part":1,"start_line":1,"end_line":42,"source_sha256":"16345bdb4e76c8492a208c3a1a6dd13df33457e1f9ba89efda70ffe04bdebfc6","fragment_sha256":"16345bdb4e76c8492a208c3a1a6dd13df33457e1f9ba89efda70ffe04bdebfc6","added_terminal_newline":false} -->
````python
"""审计的英文覆盖图和阶段状态；不产生性能结论或训练。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patient_data.export_mimiciv import sha256
from patient_data.text_time_coverage import ROOT
from training.v6.train import write_json


def main():
    report=json.loads((ROOT/'audit.json').read_text())
    for tag in ('a','b'):
        proof=json.loads((ROOT/f'verification_{tag}.json').read_text())
        if proof['report_sha256']!=sha256(ROOT/'audit.json') or proof['maximum_count_or_index_error']!=0:raise ValueError('source audit not independently verified')
        enc=json.loads((ROOT/f'encoding_check_{tag}.json').read_text());att=json.loads((ROOT/f'attention_audit_{tag}.json').read_text())
        if enc['source_audit_sha256']!=sha256(ROOT/'audit.json') or att['audit_sha256']!=sha256(ROOT/'audit.json'):raise ValueError('GPU proof stale')
    if json.loads((ROOT/'encoding_check_a.json').read_text())!=json.loads((ROOT/'encoding_check_b.json').read_text()):raise ValueError('GPU encoding process differs')
    if json.loads((ROOT/'attention_audit_a.json').read_text())!=json.loads((ROOT/'attention_audit_b.json').read_text()):raise ValueError('GPU attention process differs')
    roles=('hospital_course','discharge_condition','discharge_physical_exam','discharge_instructions','transitional_issues')
    names=('Hospital course','Discharge condition','Discharge physical exam','Discharge instructions','Transitional issues')
    totals=np.array([report['by_role'][r]['priority_candidates'] for r in roles]);y=np.arange(len(roles))
    fig,ax=plt.subplots(figsize=(10,5.6))
    for offset,key,name,color in [(-.17,'B_union_full','Original B: full span retained','#1f77b4'),(.17,'encoded_candidate_source_unit','Saved candidate encoding: source retained','#2ca02c')]:
        counts=np.array([report['by_role'][r].get(key,0) for r in roles]);bars=ax.barh(y+offset,counts/totals*100,height=.3,label=name,color=color)
        for bar,n,total in zip(bars,counts,totals):ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,f'{n}/{total}',va='center',fontsize=9)
    ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,125);ax.set_xticks([0,25,50,75,100])
    ax.set_xlabel('Percent of lexical priority candidate units retained');ax.grid(axis='x',alpha=.2)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),fontsize=8,frameon=False)
    fig.suptitle('Training-only original-span audit; not semantic accuracy or predictive gain',fontsize=11)
    fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(ROOT/'source_coverage.png',dpi=150);plt.close(fig)
    write_json(ROOT/'state.json',dict(status='source_token_and_frozen_attention_audit_complete_independently_verified',
        audit_sha256=sha256(ROOT/'audit.json'),verification_tags=['a','b'],unit_tests=8,new_fits=0,new_vectors=0,
        figures={'source_coverage.png':sha256(ROOT/'source_coverage.png')},report_code_sha256=sha256(Path(__file__)),
        original_model_or_input_changed=False,goal_achieved=False,
        next='Predeclare matched same-sentence local versus containing-block/availability views; no repeated extraction/full-document rerun'))
    print(json.dumps(dict(status='audit_complete',new_fits=0,new_vectors=0,priority_candidates=report['priority_candidate_units'])),flush=True)


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/text_time_encoding_check.py · lines 1–60

<!-- REVIEW_FRAGMENT {"path":"training/v6/text_time_encoding_check.py","part":1,"start_line":1,"end_line":60,"source_sha256":"a8185f9651a6e4814a780e83aa0b9354ff736298f40d7e035e8f070db8c08a27","fragment_sha256":"a8185f9651a6e4814a780e83aa0b9354ff736298f40d7e035e8f070db8c08a27","added_terminal_newline":false} -->
````python
"""原B训练折token完整性与固定32行GPU重编码；不是新向量或新训练。"""
from collections import Counter
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel
from patient_data.export_mimiciv import rows,sha256
from patient_data.text_time_coverage import ROOT,B,STATE,ENCODED,checked_sources
from training.v6.train import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',choices=('a','b'),required=True);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('actual GPU required')
    report=json.loads((ROOT/'audit.json').read_text());sources,_=checked_sources()
    if sources!=report['source_sha256']:raise ValueError('audit source changed')
    for tag in ('a','b'):
        v=json.loads((ROOT/f'verification_{tag}.json').read_text())
        if v['report_sha256']!=sha256(ROOT/'audit.json') or v['maximum_count_or_index_error']!=0:raise ValueError('source audit not independently replayed')
    model_dir=Path('training/model_cache/Bio_ClinicalBERT');m=json.loads((ENCODED/'manifest.json').read_text())
    for name,h in m['model_files'].items():
        if sha256(model_dir/name)!=h:raise ValueError('local encoder asset differs')
    with np.load(STATE/'clinical_state.npz') as z:train={str(h) for h,f in zip(z['hadm_ids'],z['patient_splits']) if f=='train'}
    current=[r for r in rows(B/'shared_text.csv.gz') if r['hadm_id'] in train]
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True);maximum=0;hist=Counter()
    for start in range(0,len(current),256):
        batch=current[start:start+256];ids=tokenizer([r['chunk_text'] for r in batch],add_special_tokens=True,truncation=False)['input_ids']
        for r,tokens in zip(batch,ids):
            if len(tokens)!=int(r['bert_token_count']) or len(tokens)>512:raise ValueError('actual B token length mismatch/truncation')
            maximum=max(maximum,len(tokens));hist[r['field']]+=1
    # 每字段等间距8行，固定非结局选择；不同tag独立进程同输入。
    chosen=[]
    for field in sorted(hist):
        eligible=[i for i,r in enumerate(current) if r['field']==field]
        chosen.extend(eligible[i] for i in np.linspace(0,len(eligible)-1,8,dtype=int))
    texts=[current[i]['chunk_text'] for i in chosen];indices=[int(current[i]['embedding_row']) for i in chosen]
    original=np.load(B/'bert/bioclinicalbert.float16.npy',mmap_mode='r')
    if original.shape!=(76974,768):raise ValueError('original B vector shape changed')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModel.from_pretrained(model_dir,local_files_only=True).cuda().eval().requires_grad_(False)
    encoded={k:v.cuda() for k,v in tokenizer(texts,padding=True,truncation=False,return_tensors='pt').items()}
    with torch.inference_mode():
        hidden=model(**encoded).last_hidden_state.float();mask=encoded['attention_mask'].bool().clone()
        lengths=mask.sum(1);mask[:,0]=False;mask.scatter_(1,(lengths-1)[:,None],False)
        pooled=torch.nn.functional.normalize((hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None],dim=1).cpu().numpy()
    expected=original[indices].astype(np.float32);error=float(np.abs(pooled-expected).max())
    if error>.002:raise ValueError('fixed B GPU encoding differs beyond FP16 budget')
    proof=dict(status='all_training_B_token_lengths_checked_and_fixed32_GPU_reencoded',training_rows=len(current),actual_max_B_tokens=maximum,
        field_counts=dict(hist),sample_rows=indices,maximum_vector_error=error,FP16_tolerance=.002,
        tokenization_truncation=False,source_audit_sha256=sha256(ROOT/'audit.json'),original_vectors_sha256=sha256(B/'bert/bioclinicalbert.float16.npy'),
        encoder_files=m['model_files'],code_sha256=sha256(Path(__file__)),device=torch.cuda.get_device_name(0),
        limits=['Only32 vector rows reencoded; not all76974 GPU rerun','No semantic accuracy or pooling utility proof','No new fitting or vector overwrite'])
    write_json(ROOT/f'encoding_check_{args.tag}.json',proof)
    if args.tag=='b' and json.loads((ROOT/'encoding_check_a.json').read_text())!=proof:raise ValueError('independent encoding proofs differ')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/token_choose_reference.py · lines 1–24

<!-- REVIEW_FRAGMENT {"path":"training/v6/token_choose_reference.py","part":1,"start_line":1,"end_line":24,"source_sha256":"fd922c7c4511b2498cb672b03f02336a3924e5d29291bf7e4aaae60ed309500c","fragment_sha256":"fd922c7c4511b2498cb672b03f02336a3924e5d29291bf7e4aaae60ed309500c","added_terminal_newline":false} -->
````python
"""只依据验证IBS选择新研究参照，不修改旧默认、旧检查点或测试指标。"""
import argparse
import json
from pathlib import Path
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json


def choose(root):
    r=json.loads((root/'report.json').read_text());proof=json.loads((root/'independent_verification.json').read_text())
    if proof['report_sha256']!=sha256(root/'report.json'):raise ValueError('neural proof stale')
    seeds=r['contract']['seeds'];names=list(r['models'][str(seeds[0])])
    scores={n:sum(r['models'][str(s)][n]['validation_ibs'] for s in seeds)/len(seeds) for n in names}
    best=min(scores,key=scores.get)
    selected=dict(status='validation_selected_research_reference',arm=best,selection='minimum mean validation IBS7-365 over predefined two seeds',
        validation_scores=scores,checkpoints={str(s):dict(path=str(root/f'seed{s}'/best/'best.pt'),
            sha256=r['models'][str(s)][best]['checkpoint_sha256'],epoch=r['models'][str(s)][best]['best_epoch']) for s in seeds},
        report_sha256=sha256(root/'report.json'),daily_curve_interface='0..365: S,Freadmission,Fdeath plus daily mass',
        limits=['Research reference only; previous defaults and assets unchanged.','No selection using test scores; observed test gains remain exploratory.'])
    write_json(root/'research_reference.json',selected);print(json.dumps(dict(selected=best,validation_ibs=scores[best])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'));choose(p.parse_args().root)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/token_cox_control.py · lines 1–79

<!-- REVIEW_FRAGMENT {"path":"training/v6/token_cox_control.py","part":1,"start_line":1,"end_line":79,"source_sha256":"84e1d923c4c6fac9cfee11e344319da0c062f30f656457676ebd4fe8679e6d84","fragment_sha256":"84e1d923c4c6fac9cfee11e344319da0c062f30f656457676ebd4fe8679e6d84","added_terminal_newline":false} -->
````python
"""相同严格终点的成熟双原因Cox算法对照，降维只拟合训练患者。"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.decomposition import PCA
from patient_data.export_mimiciv import sha256
from training.v6.token_data import load_inputs,prepare
from training.v6.token_model import TokenEncoder,daily_probabilities
from training.v6.evaluation import metrics,public_metrics,paired_interval,calibration_bins
from training.v6.train import write_json
from training.baselines.continuous_competing_cox import fit_competing_cox,predict_cumulative_incidence


def run(root):
    output=root/'cox_control'
    if (output/'report.json').exists():raise FileExistsError(output)
    if not all((root/f'seed{seed}/flat_mlp/{fold}_predictions.npz').exists() for seed in (42,7) for fold in ('validation','test')):
        raise RuntimeError('Complete neural finalization before Cox paired comparison')
    # 初次并行运行仅留下未完成预测，允许重生成自己的临时文件；完整报告不可覆盖。
    output.mkdir(exist_ok=True)
    cfg=json.loads((root/'config.json').read_text());start=time.perf_counter()
    d,c,h,ext,med,provenance=load_inputs(*(Path(cfg[k]) for k in ('reference','compact_dir','extension','medication')))
    ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('train','validation','test')}
    # 不复用神经网络的可学习投影；全部原表格输入和四字段冻结向量均值。
    encoder=TokenEncoder(flat=True).cpu().eval();raw={};context={}
    for f in ('train','validation'):  # 选模前不读取测试输入
        cache=prepare(d,c,ext,med,ix[f],size=256,device='cpu')
        with torch.no_grad():raw[f]=np.concatenate([encoder.flat_features(b).numpy() for _,_,b,_,_ in cache])
        context[f]=np.concatenate([h.numpy() for _,h,_,_,_ in cache]);del cache
    boundary=29*20+7*4+21*23+8*2
    pcas=[PCA(16,svd_solver='randomized',random_state=42),PCA(16,svd_solver='randomized',random_state=42)]
    pcas[0].fit(raw['train'][:,:boundary]);pcas[1].fit(raw['train'][:,boundary:])
    def transform(x,h):return np.concatenate((h,pcas[0].transform(x[:,:boundary]),pcas[1].transform(x[:,boundary:])),1)
    x={f:transform(raw[f],context[f]) for f in ('train','validation')}
    center=x['train'].mean(0);scale=x['train'].std(0);scale[scale<1e-7]=1
    for f in x:x[f]=np.clip((x[f]-center)/scale,-10,10)
    candidates={};fitted={};curves={}
    for l2 in (10.,100.):
        fitted[l2]=fit_competing_cox(x['train'],d.times[ix['train']],d.events[ix['train']],horizon_days=365.,l2=l2,max_iter=80)
        curve=predict_cumulative_incidence(fitted[l2],x['validation'],np.arange(366))
        detail=public_metrics(metrics(d.times[ix['validation']],d.events[ix['validation']],curve[:,:,1]))
        candidates[str(l2)]=dict(validation=detail,readmission_converged=fitted[l2].readmission.converged,death_converged=fitted[l2].death.converged)
        curves[l2]=curve;print(json.dumps(dict(cox_l2=l2,validation_ibs=detail['integrated_brier_7_365'],converged=[fitted[l2].readmission.converged,fitted[l2].death.converged])),flush=True)
    best=min(fitted,key=lambda l2:candidates[str(l2)]['validation']['integrated_brier_7_365']);model=fitted[best]
    cache=prepare(d,c,ext,med,ix['test'],size=256,device='cpu')
    with torch.no_grad():raw['test']=np.concatenate([encoder.flat_features(b).numpy() for _,_,b,_,_ in cache])
    context['test']=np.concatenate([h.numpy() for _,h,_,_,_ in cache]);del cache
    x['test']=np.clip((transform(raw['test'],context['test'])-center)/scale,-10,10)
    curves[best]=None;folds={};comparison={}
    for f in ('validation','test'):
        curve=predict_cumulative_incidence(model,x[f],np.arange(366));daily,conditional=daily_probabilities(curve)
        np.savez_compressed(output/(f+'_predictions.npz'),hadm_ids=d.arrays['hadm_ids'][ix[f]],subject_ids=d.arrays['subject_ids'][ix[f]],
            times=d.times[ix[f]],events=d.events[ix[f]],curve=curve,daily_event_probability=daily,conditional_daily_event_probability=conditional)
        folds[f]=dict(readmission=public_metrics(metrics(d.times[ix[f]],d.events[ix[f]],curve[:,:,1],ordering=True)),
                      death=public_metrics(metrics(d.times[ix[f]],d.events[ix[f]],curve[:,:,2],cause=2,ordering=True)),
                      calibration={str(day):calibration_bins(d.times[ix[f]],d.events[ix[f]],curve[:,:,1],day=day) for day in (30,90,180,365)})
        comparison[f]={}
        for seed in cfg['seeds']:
            with np.load(root/f'seed{seed}/flat_mlp/{f}_predictions.npz') as z:reference=z['curve']
            comparison[f][str(seed)]=paired_interval(d.arrays['subject_ids'][ix[f]],d.times[ix[f]],d.events[ix[f]],curve[:,:,1],reference[:,:,1],seed=seed)
    components=dict(center=center,scale=scale)
    for j,pca in enumerate(pcas):components[f'pca{j}_mean']=pca.mean_;components[f'pca{j}_components']=pca.components_
    for event,cause in (('readmission',model.readmission),('death',model.death)):
        components[event+'_coefficients']=cause.coefficients;components[event+'_times']=cause.event_times;components[event+'_increments']=cause.baseline_hazard_increments
    np.savez_compressed(output/'model.npz',**components)
    report=dict(status='completed_cox_control',selection='validation readmission IBS7-365',selected_l2=best,candidates=candidates,folds=folds,
        comparisons_to_flat_mlp=comparison,features=66,clinical_pca=16,text_pca=16,context=34,train_only_pca=True,
        elapsed_seconds=time.perf_counter()-start,source_sha256=sha256(Path(__file__)),data_provenance=d.provenance,extensions=provenance,
        limits=['Mature cause-specific Cox algorithm adapted, not an external paper replication.',
                'PCA representation differs from token encoder; no causality or blind test claim.',
                'Deterministic Cox fitted once, not duplicated as two training seeds.'])
    write_json(output/'report.json',report);print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'));run(p.parse_args().root)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/token_cox_verify.py · lines 1–42

<!-- REVIEW_FRAGMENT {"path":"training/v6/token_cox_verify.py","part":1,"start_line":1,"end_line":42,"source_sha256":"257076cd532c09d71b556e1e7a6568bc999902fe6564398a0792ffc700c04aa1","fragment_sha256":"257076cd532c09d71b556e1e7a6568bc999902fe6564398a0792ffc700c04aa1","added_terminal_newline":false} -->
````python
"""重载保存的PCA/尺度/双原因Cox参数，不重新拟合，对照预测逐折复现。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.token_data import load_inputs,prepare
from training.v6.token_model import TokenEncoder,daily_probabilities
from training.v6.train import write_json
from training.baselines.continuous_competing_cox import CauseSpecificCox,CompetingCoxModel,predict_cumulative_incidence


def verify(root):
    folder=root/'cox_control';report=json.loads((folder/'report.json').read_text());cfg=json.loads((root/'config.json').read_text())
    if sha256(Path('training/v6/token_cox_control.py'))!=report['source_sha256']:raise ValueError('Cox source changed')
    with np.load(folder/'model.npz') as z:a={k:z[k] for k in z.files}
    causes=[CauseSpecificCox(a[n+'_coefficients'],a[n+'_times'],a[n+'_increments'],True) for n in ('readmission','death')]
    model=CompetingCoxModel(*causes,365.,report['selected_l2'])
    d,c,h,ext,med,provenance=load_inputs(*(Path(cfg[k]) for k in ('reference','compact_dir','extension','medication')))
    if d.provenance!=report['data_provenance'] or provenance!=report['extensions']:raise ValueError('Cox data lineage differs')
    encoder=TokenEncoder(flat=True).cpu().eval();boundary=29*20+7*4+21*23+8*2;errors={}
    for f in ('validation','test'):
        ix=np.flatnonzero(d.arrays['patient_splits']==f);cache=prepare(d,c,ext,med,ix,device='cpu')
        with torch.no_grad():raw=np.concatenate([encoder.flat_features(b).numpy() for _,_,b,_,_ in cache])
        context=np.concatenate([h.numpy() for _,h,_,_,_ in cache]);del cache
        blocks=[(raw[:,:boundary]-a['pca0_mean'])@a['pca0_components'].T,
                (raw[:,boundary:]-a['pca1_mean'])@a['pca1_components'].T]
        x=np.clip((np.concatenate((context,*blocks),1)-a['center'])/a['scale'],-10,10)
        curve=predict_cumulative_incidence(model,x,np.arange(366));daily,conditional=daily_probabilities(curve)
        with np.load(folder/(f+'_predictions.npz')) as z:
            if not np.array_equal(z['hadm_ids'],d.arrays['hadm_ids'][ix]) or not np.array_equal(z['events'],d.events[ix]):raise ValueError('Cox indices differ')
            err=float(np.max(np.abs(curve-z['curve'])));de=float(np.max(np.abs(daily-z['daily_event_probability'])))
        if max(err,de)>2e-6:raise ValueError('Cox/PCA replay differs')
        errors[f]=dict(curve_error=err,daily_error=de)
    proof=dict(status='independently_verified_cox',report_sha256=sha256(folder/'report.json'),model_sha256=sha256(folder/'model.npz'),
        saved_transform_reloaded_without_fitting=True,reload_errors=errors)
    write_json(folder/'independent_verification.json',proof);print(json.dumps(proof))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'));torch.set_num_threads(2);verify(p.parse_args().root)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/token_data.py · lines 1–51

<!-- REVIEW_FRAGMENT {"path":"training/v6/token_data.py","part":1,"start_line":1,"end_line":51,"source_sha256":"85dacd5122007d76d059dd9addd480735a803305b6009d6941bd2a402c315e22","fragment_sha256":"85dacd5122007d76d059dd9addd480735a803305b6009d6941bd2a402c315e22","added_terminal_newline":false} -->
````python
"""冻结数据合同的token桥接；标签只交给损失，不进入TokenBatch。"""
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.compact_ablation import inputs
from training.v6.exploration_model import expanded_inputs
from training.v6.token_model import TokenBatch


def load_inputs(reference,compact_dir,extension,medication):
    d,c,hist,_=inputs(reference,compact_dir)
    result=[]; provenance={}
    for folder,file,status in ((extension,'extension.npz','source_checked_extension_not_trained'),
                               (medication,'medication_identity.npz','validated_medication_identity')):
        m=json.loads((folder/'manifest.json').read_text())
        if m['status']!=status or m['parent_state_manifest_sha256']!=d.provenance['state_manifest_sha256'] or sha256(folder/file)!=m['output_sha256']:
            raise ValueError('token data lineage mismatch')
        with np.load(folder/file) as z: a={k:z[k] for k in z.files}
        for key in ('hadm_ids','subject_ids','patient_splits'):
            if not np.array_equal(a[key],d.arrays[key]): raise ValueError('member/split mismatch')
        result.append(a); provenance[file]=sha256(folder/'manifest.json')
    return d,c,hist,*result,provenance


def prepare(d,c,ext,med,idx,*,size=256,device='cuda'):
    cache=[]
    for start in range(0,len(idx),size):
        pos=idx[start:start+size]; h,compact,t,e=c.collate(pos,device); b=compact.clinical
        h=expanded_inputs(h,compact)
        lab=np.zeros((len(pos),29,20),np.float32); valid=np.zeros_like(lab,bool)
        lab[:,:22,:9]=d.arrays['lab_axis_inputs'][pos]
        present=lab[:,:22,6]>0
        valid[:,:22,:6]=present[:,:,None];valid[:,:22,3]&=lab[:,:22,7]>0;valid[:,:22,5]&=lab[:,:22,7]>0
        lab[:,22:27,:10]=ext['new_labs'][pos];valid[:,22:27,:5]=ext['new_labs'][pos,:,5:].astype(bool)
        lab[:,27:]=ext['eos'][pos];valid[:,27:,:6]=ext['eos'][pos,:,6:12].astype(bool)
        observed=np.concatenate((present,valid[:,22:27,:5].any(2),valid[:,27:,:6].any(2)),1)
        length=max(1,max(len(d.text_refs[p]) for p in pos));vectors=np.zeros((len(pos),length,768),np.float32)
        fields=np.zeros((len(pos),length),np.int64);tm=np.zeros((len(pos),length),bool)
        for slot,p in enumerate(pos):
            for j,(field,row) in enumerate(d.text_refs[p]):
                vectors[slot,j]=d.text_vectors[row];fields[slot,j]=field;tm[slot,j]=True
        tensor=lambda a:torch.as_tensor(a,device=device)
        other=torch.cat((torch.stack((compact.micro_values,compact.micro_masks.float()),2),
                         torch.stack((compact.support_values,compact.support_masks.float()),2)),1)
        batch=TokenBatch(tensor(lab),tensor(valid),tensor(observed),compact.med_features,
                         tensor(med['features'][pos]),tensor(med['present'][pos]),other,
                         tensor(vectors),tensor(fields),tensor(tm),tensor(np.zeros_like(observed)))
        cache.append((pos,h,batch,t,e))
    return cache
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/token_experiment.py · lines 1–267

<!-- REVIEW_FRAGMENT {"path":"training/v6/token_experiment.py","part":1,"start_line":1,"end_line":267,"source_sha256":"bf4db93f8933f0c5a00129d7b031e778a4a0d05238eb580874961b0c47af0a7a","fragment_sha256":"bf4db93f8933f0c5a00129d7b031e778a4a0d05238eb580874961b0c47af0a7a","added_terminal_newline":false} -->
````python
"""四项授权目标的独立实验入口：固定严格结局、每日风险、配对神经臂。"""
import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from patient_data.export_mimiciv import sha256
from training.v6.token_data import load_inputs,prepare
from training.v6.token_model import TokenModel,fast_cif,daily_probabilities,masked_reconstruction_batch
from training.v6.horizon_objective import restore
from training.v6.detail_extension_experiment import DetailBatch,DetailEncoder
from training.v6.compact_model import CompactAnchoredModel
from training.v6.architecture_experiment import omit_current
from training.v6.exploration_model import expanded_inputs
from training.v6.risk import competing_nll
from training.v6.evaluation import metrics,public_metrics,paired_interval,calibration_bins
from training.v6.train import write_json

PROTOCOL=Path('docs/V6_COARSE_FINE_TOKEN_PROTOCOL_2026-10-12.md')
ARMS=('pool_coarse','pool_fine','interaction_fine','interaction_ssl','flat_mlp')


def legacy_batches(d,c,ext,idx,size=256):
    result=[]
    for start in range(0,len(idx),size):
        pos=idx[start:start+size];h,b,t,e=c.collate(pos,'cuda');h=expanded_inputs(h,b);b=omit_current(b)
        labs=np.concatenate((ext['new_labs'][pos].reshape(len(pos),-1),ext['eos'][pos].reshape(len(pos),-1)),1)
        result.append((pos,h,DetailBatch(b,torch.as_tensor(labs,device='cuda'),torch.zeros(len(pos),192,device='cuda')),t,e))
    return result


def arm_batches(cache,arm):
    if arm!='pool_coarse':return cache
    return [(p,h,replace(b,named=b.named*0,named_present=b.named_present&False),t,e) for p,h,b,t,e in cache]


def make_model(base,seed,arm):
    torch.manual_seed(seed)
    if arm=='legacy':
        m=CompactAnchoredModel(copy.deepcopy(base),768)
        m.encoder=DetailEncoder(m.encoder)
    else:m=TokenModel(copy.deepcopy(base),interaction=arm.startswith('interaction'),flat=arm=='flat_mlp')
    m=m.cuda();torch.manual_seed(seed)  # 初始化耗用随机数不应改变后续dropout流。
    return m


@torch.no_grad()
def parameters(model,prepared):
    model.eval();a=[];k=[];total=0.;count=0
    for _,h,b,t,e in prepared:
        out=model(h,b) if hasattr(model,'encoder') else model(h)
        a.append(out.log_amplitude.cpu().numpy());k.append(out.shape.cpu().numpy())
        total+=float(competing_nll(out,t,e))*len(t);count+=len(t)
    return np.concatenate(a),np.concatenate(k),total/count


def validation(model,prepared,times,events):
    a,k,nll=parameters(model,prepared);curve=fast_cif(a,k,device='cuda')
    y=((events[:,None]==1)&(times[:,None]<=np.arange(366))).astype(float)
    brier=((curve[:,:,1]-y)**2).mean(0)
    from training.baselines.rank_metrics import auc_rank
    return dict(nll=nll,ibs=float(np.trapezoid(brier[7:],np.arange(7,366))/358),
                brier90=float(brier[90]),auc90=float(auc_rank(curve[:,90,1],y[:,90].astype(np.int8))),
                log_amplitude_quantiles=np.quantile(a,[.01,.5,.99],axis=0).tolist(),
                shape_quantiles=np.quantile(k,[.01,.5,.99],axis=0).tolist())


def fit(model,train,val,times,events,path,*,seed,lr,epochs):
    path.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4)
    frozen={k:v.detach().cpu().clone() for k,v in model.baseline.state_dict().items()}
    history=[];best=math.inf;selected=0;begin=time.perf_counter()
    def record(epoch,online=None,gradient=None,lr_value=0.):
        nonlocal best,selected
        detail=validation(model,val,times,events)
        # 固定训练住院样本，epoch末评估；不与在线批次均值混淆。
        _,_,train_nll=parameters(model,train[:8])
        row=dict(epoch=epoch,train_online_nll=online,train_fixed2048_nll=train_nll,
                 gradient_norm_before_clip=gradient,lr=lr_value,**{'validation_'+k:v for k,v in detail.items()})
        history.append(row)
        if detail['ibs']<best:
            best=detail['ibs'];selected=epoch
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,validation_ibs=best),path/'best.pt')
        if epoch%10==0 or epoch==epochs:
            write_json(path/'history.json',history)
            print(json.dumps(dict(arm=str(path),epoch=epoch,best_epoch=selected,ibs=detail['ibs'],auc90=detail['auc90'],
                                  seconds=round(time.perf_counter()-begin,1))),flush=True)
    record(0)
    for epoch in range(1,epochs+1):
        model.train();warm=min(1.,epoch/5);cosine=.1+.9*.5*(1+math.cos(math.pi*max(0,epoch-5)/max(1,epochs-5)))
        current_lr=lr*warm*cosine
        for g in opt.param_groups:g['lr']=current_lr
        total=0.;count=0;grad=[]
        for j in rng.permutation(len(train)):
            _,h,b,t,e=train[j];opt.zero_grad(set_to_none=True);out=model(h,b);nll=competing_nll(out,t,e)
            loss=nll+1e-4*out.delta.square().mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite training objective')
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(gn):raise ValueError('nonfinite gradients')
            opt.step();total+=float(nll.detach())*len(t);count+=len(t);grad.append(float(gn))
        record(epoch,total/count,float(np.mean(grad)),current_lr)
    if not all(torch.equal(v,model.baseline.state_dict()[k].cpu()) for k,v in frozen.items()):raise ValueError('frozen H34 changed')
    write_json(path/'history.json',history)
    return dict(best_epoch=selected,completed_epochs=epochs,lr=lr,validation_ibs=best,
        checkpoint_sha256=sha256(path/'best.pt'),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        elapsed_seconds=time.perf_counter()-begin,frozen_baseline_verified=True)


def pretrain(model,train,path,seed,epochs=20):
    """训练折重构：临床标签不进入优化；重构头丢弃后风险头仍严格为0。"""
    torch.manual_seed(seed);decoder=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,20)).cuda()
    opt=torch.optim.AdamW(list(model.encoder.parameters())+list(decoder.parameters()),lr=2e-4,weight_decay=1e-4)
    history=[];start=time.perf_counter()
    for epoch in range(epochs+1):
        model.encoder.train(epoch>0);decoder.train(epoch>0);total=0.;count=0;zero=0.
        iterator=train[:8] if epoch==0 else train
        for j,(_,_,b,_,_) in enumerate(iterator):
            masked,target=masked_reconstruction_batch(b,seed+epoch*10000+j)
            if not target.any():continue
            with torch.set_grad_enabled(epoch>0):
                h,states=model.encoder(masked);x=states['tokens'][:,:29]
                pred=decoder(torch.cat((x,h[:,None].expand(-1,29,-1)),2))
                errors=(pred-b.labs).square()[target];loss=errors.mean()
                if epoch:
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters())+list(decoder.parameters()),1.)
                    opt.step()
            total+=float(errors.detach().sum());count+=len(errors);zero+=float(b.labs.square()[target].sum())
        history.append(dict(epoch=epoch,reconstruction_mse=total/max(count,1),zero_normalized_prediction_mse=zero/max(count,1),observed_targets=count))
    if any(torch.count_nonzero(p) for p in model.delta_head[-1].parameters()):raise ValueError('SSL changed risk zero initialization')
    write_json(path/'self_supervision.json',dict(history=history,epochs=epochs,training_fold_only=True,
        diagnostic='training-fold masked reconstruction, not held-out clinical accuracy',seconds=time.perf_counter()-start))
    return history[-1]


def evaluate_model(model,cache,d,indices,path):
    a,k,nll=parameters(model,cache);curve=fast_cif(a,k,device='cuda');daily,conditional=daily_probabilities(curve)
    reread=torch.load(path/'best.pt',map_location='cuda',weights_only=True)
    model.load_state_dict(reread['state_dict']);aa,kk,_=parameters(model,cache)
    if not np.array_equal(a,aa) or not np.array_equal(k,kk):raise ValueError('checkpoint replay differs')
    return curve,dict(nll=nll,readmission=public_metrics(metrics(d.times[indices],d.events[indices],curve[:,:,1],ordering=True)),
                      death=public_metrics(metrics(d.times[indices],d.events[indices],curve[:,:,2],cause=2,ordering=True)),
                      calibration={str(day):calibration_bins(d.times[indices],d.events[indices],curve[:,:,1],day=day)
                                   for day in (30,90,180,365)}),dict(log_amplitude=a,shape=k,curve=curve,
                      daily_event_probability=daily,conditional_daily_event_probability=conditional)


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;start=time.perf_counter()
    proof=json.loads((args.parent/'independent_verification.json').read_text())
    if proof['report_sha256']!=sha256(args.parent/'report.json'):raise ValueError('parent report verification stale')
    d,c,hist,ext,med,prov=load_inputs(args.reference,args.compact_dir,args.extension,args.medication)
    ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('train','validation','test')}
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    code_paths=('training/v6/token_model.py','training/v6/token_data.py','training/v6/token_experiment.py','patient_data/medication_identity.py')
    contract=dict(protocol_sha256=sha256(PROTOCOL),code_sha256={p:sha256(Path(p)) for p in code_paths},
        data_provenance=d.provenance,compact_provenance=c.provenance,extensions=prov,seeds=args.seeds,
        epochs=args.epochs,self_supervised_epochs=args.ssl_epochs,selection='validation readmission IBS 7-365',
        baseline_parent_report_sha256=sha256(args.parent/'report.json'),
        frozen_baselines={str(s):sha256(args.parent/f'seed{s}/survival_base/best.pt') for s in args.seeds},
        endpoint='strict491.21/J44.1; competing death; WorldA365',fold_admissions={f:len(v) for f,v in ix.items()},
        device=torch.cuda.get_device_name(0),batch_size=256,arms=list(ARMS),lr_sweep=[3e-4,1e-4,3e-5],
        text='unchanged B four fields; all saved Bio_ClinicalBERT fragments; not raw/full-note comparison',
        limits=['Reused test, not blind; no multiplicity adjustment.',
                'Frozen H34 was selected by prior Brier90, not refit here.',
                'Medication/test interactions are associations, not treatment causal effects.'])
    write_json(args.output/'contract.json',contract)
    cache={f:prepare(d,c,ext,med,ix[f]) for f in ('train','validation')}
    legacy={f:legacy_batches(d,c,ext,ix[f]) for f in ('train','validation')}
    fitted={};recon={}
    for seed in args.seeds:
        base=restore(args.parent,seed,'survival_base',hist,.1)
        for lr in (3e-4,1e-4,3e-5):
            name=f'legacy_lr{lr:g}';m=make_model(base,seed,'legacy');path=args.output/f'seed{seed}'/name
            fitted[seed,name]=fit(m,legacy['train'],legacy['validation'],d.times[ix['validation']],d.events[ix['validation']],path,
                                 seed=seed,lr=lr,epochs=args.epochs);del m
        for arm in ARMS:
            m=make_model(base,seed,arm);path=args.output/f'seed{seed}'/arm
            if arm=='interaction_ssl':
                path.mkdir(parents=True);recon[str(seed)]=pretrain(m,cache['train'],path,seed,args.ssl_epochs)
                # fit负责创建目录，保留该预训练文件。
            fitted[seed,arm]=fit(m,arm_batches(cache['train'],arm),arm_batches(cache['validation'],arm),
                d.times[ix['validation']],d.events[ix['validation']],path,seed=seed,lr=1e-4,epochs=args.epochs);del m
    write_json(args.output/'status.json',dict(status='all_neural_checkpoints_selected_before_test'))
    # 所有选择结束后才首次为本次实验生成测试输入。
    cache['test']=prepare(d,c,ext,med,ix['test']);legacy['test']=legacy_batches(d,c,ext,ix['test'])
    models={};curves={};comparisons={}
    for seed in args.seeds:
        base=restore(args.parent,seed,'survival_base',hist,.1);models[str(seed)]={};curves[seed]={}
        for name in [f'legacy_lr{lr:g}' for lr in (3e-4,1e-4,3e-5)]+list(ARMS):
            m=make_model(base,seed,'legacy' if name.startswith('legacy') else name);path=args.output/f'seed{seed}'/name
            m.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            models[str(seed)][name]=dict(**fitted[seed,name],folds={});curves[seed][name]={}
            for f in ('validation','test'):
                batches=legacy[f] if name.startswith('legacy') else arm_batches(cache[f],name)
                curve,detail,pred=evaluate_model(m,batches,d,ix[f],path);models[str(seed)][name]['folds'][f]=detail;curves[seed][name][f]=curve
                np.savez_compressed(path/(f+'_predictions.npz'),hadm_ids=d.arrays['hadm_ids'][ix[f]],
                    subject_ids=d.arrays['subject_ids'][ix[f]],times=d.times[ix[f]],events=d.events[ix[f]],**pred)
            del m
        comparisons[str(seed)]={}
        pairs=(('pool_fine','pool_coarse'),('interaction_fine','pool_fine'),('interaction_ssl','interaction_fine'),
               ('flat_mlp','pool_fine'),('legacy_lr0.0001','legacy_lr0.0003'),('legacy_lr3e-05','legacy_lr0.0003'))
        for f in ('validation','test'):
            comparisons[str(seed)][f]={f'{a} minus {b}':paired_interval(d.arrays['subject_ids'][ix[f]],d.times[ix[f]],d.events[ix[f]],
                curves[seed][a][f][:,:,1],curves[seed][b][f][:,:,1],seed=seed) for a,b in pairs}
    report=dict(status='neural_completed',contract=contract,models=models,comparisons=comparisons,reconstruction=recon,
                elapsed_seconds=time.perf_counter()-start,peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
    write_json(args.output/'report.json',report);render(args.output,report)
    print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'])),flush=True)


def render(root,report):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'legacy_lr0.0003':'C0','legacy_lr0.0001':'C1','legacy_lr3e-05':'C2',
            'pool_coarse':'C3','pool_fine':'C4','interaction_fine':'C5','interaction_ssl':'C6','flat_mlp':'C7'}
    for group,names in (('learning_rate',list(colors)[:3]),('token_interaction',list(ARMS))):
        fig,ax=plt.subplots(2,2,figsize=(14,9))
        for row,seed in enumerate(report['contract']['seeds']):
            for name in names:
                hist=json.loads((root/f'seed{seed}'/name/'history.json').read_text());epochs=[x['epoch'] for x in hist]
                ax[row,0].plot(epochs,[x['train_fixed2048_nll'] for x in hist],label=name,color=colors[name])
                ax[row,1].plot(epochs,[x['validation_ibs'] for x in hist],label=name,color=colors[name])
            ax[row,0].set_title(f'Seed {seed}: fixed training sample survival NLL');ax[row,1].set_title(f'Seed {seed}: validation IBS 7-365')
            for a in ax[row]:a.set_xlabel('Epoch');a.grid(alpha=.25);a.legend(fontsize=8)
        fig.suptitle('100-epoch diagnostic; frozen H34; checkpoint selected by validation IBS');fig.tight_layout()
        fig.savefig(root/(group+'_training.png'),dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(14,5))
    for col,seed in enumerate(report['contract']['seeds']):
        for name in ('pool_coarse','pool_fine','interaction_fine','interaction_ssl','flat_mlp'):
            with np.load(root/f'seed{seed}'/name/'test_predictions.npz') as z:
                m=metrics(z['times'],z['events'],z['curve'][:,:,1]);ax[col].plot(np.arange(366),m['auc_curve'],label=name,color=colors[name])
        ax[col].set_title(f'Seed {seed}: reused-test AUC(t)');ax[col].set_xlabel('Days after discharge');ax[col].set_ylabel('AUC(t)');ax[col].legend(fontsize=8);ax[col].grid(alpha=.25)
    fig.tight_layout();fig.savefig(root/'time_dependent_auc.png',dpi=180);plt.close(fig)
    seed=report['contract']['seeds'][0];chosen=min(ARMS,key=lambda n:report['models'][str(seed)][n]['validation_ibs'])
    with np.load(root/f'seed{seed}'/chosen/'test_predictions.npz') as z:
        risk=z['curve'];daily=z['daily_event_probability'];conditional=z['conditional_daily_event_probability']
    # 固定前三条测试记录，无按结局/最高效果挑病例；不展示患者身份。
    fig,ax=plt.subplots(3,3,figsize=(14,11))
    for i in range(3):
        ax[i,0].plot(np.arange(366),risk[i,:,1],label='Readmission CIF');ax[i,0].plot(np.arange(366),risk[i,:,2],label='Death CIF')
        ax[i,1].plot(np.arange(1,366),daily[i,:,0],label='Unconditional daily readmission')
        ax[i,2].plot(np.arange(1,366),conditional[i,:,0],label='Daily readmission given event-free yesterday')
        for a in ax[i]:a.set_xlabel('Days after discharge');a.legend(fontsize=7);a.grid(alpha=.25)
    fig.suptitle(f'Daily probability interface: {chosen}; first 3 held records, not outcome-selected')
    fig.tight_layout();fig.savefig(root/'daily_probability_examples.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--parent',type=Path,default=Path('training_runs/v6_90day_objective_2026-10-11'))
    p.add_argument('--compact-dir',type=Path,default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--extension',type=Path,default=Path('patient_data/processed/v6_detail_extension_2026-10-11_r2'))
    p.add_argument('--medication',type=Path,default=Path('patient_data/processed/v6_medication_identity_2026-10-12'))
    p.add_argument('--output',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'))
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--ssl-epochs',type=int,default=20);p.add_argument('--seeds',type=int,nargs='+',default=[42,7])
    run(p.parse_args())


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/token_finalize.py · lines 1–85

<!-- REVIEW_FRAGMENT {"path":"training/v6/token_finalize.py","part":1,"start_line":1,"end_line":85,"source_sha256":"98ce140ba7f1ff05d8dc7582740e88ab3498699d564793157f7ef2c8c845f055","fragment_sha256":"98ce140ba7f1ff05d8dc7582740e88ab3498699d564793157f7ef2c8c845f055","added_terminal_newline":false} -->
````python
"""只用已完成检查点汇总；容忍GPU浮点归约误差，不续训或重选测试。"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.token_data import load_inputs,prepare
from training.v6.token_experiment import ARMS,legacy_batches,make_model,arm_batches,parameters,render
from training.v6.token_model import fast_cif,daily_probabilities
from training.v6.horizon_objective import restore
from training.v6.evaluation import metrics,public_metrics,paired_interval,calibration_bins
from training.v6.train import write_json


def finalize(root):
    started=time.perf_counter();contract=json.loads((root/'contract.json').read_text());cfg=json.loads((root/'config.json').read_text())
    names=[f'legacy_lr{lr:g}' for lr in (3e-4,1e-4,3e-5)]+list(ARMS)
    summaries={}
    # 先核对全部100轮日志，未完成不读取测试集。
    for seed in cfg['seeds']:
        for name in names:
            path=root/f'seed{seed}'/name;history=json.loads((path/'history.json').read_text())
            if len(history)!=cfg['epochs']+1 or history[-1]['epoch']!=cfg['epochs']:raise ValueError('unfinished training '+str(path))
            chosen=min(history,key=lambda r:r['validation_ibs']);best=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
            if chosen['epoch']!=best['epoch'] or abs(chosen['validation_ibs']-best['validation_ibs'])>1e-12:
                raise ValueError('checkpoint selection differs')
            summaries[seed,name]=dict(best_epoch=best['epoch'],completed_epochs=cfg['epochs'],validation_ibs=best['validation_ibs'],
                lr=1e-4 if not name.startswith('legacy') else float(name.split('lr')[1]),checkpoint_sha256=sha256(path/'best.pt'),
                fit_summary_source='recovered from saved complete epoch histories and checkpoint metadata; no new training')
    for p,digest in contract['code_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('training source changed '+p)
    d,c,h,ext,med,provenance=load_inputs(*(Path(cfg[k]) for k in ('reference','compact_dir','extension','medication')))
    if d.provenance!=contract['data_provenance'] or c.provenance!=contract['compact_provenance'] or provenance!=contract['extensions']:
        raise ValueError('input lineage changed')
    ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('validation','test')}
    token={f:prepare(d,c,ext,med,ix[f]) for f in ix};legacy={f:legacy_batches(d,c,ext,ix[f]) for f in ix}
    models={};curves={};comparisons={};recon={};errors={}
    for seed in cfg['seeds']:
        base=restore(Path(cfg['parent']),seed,'survival_base',h,.1);models[str(seed)]={};curves[seed]={}
        for name in names:
            m=make_model(base,seed,'legacy' if name.startswith('legacy') else name);path=root/f'seed{seed}'/name
            m.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):raise ValueError('frozen baseline changed')
            models[str(seed)][name]=dict(**summaries[seed,name],trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
                                       frozen_baseline_verified=True,folds={});curves[seed][name]={}
            for f in ix:
                batch=legacy[f] if name.startswith('legacy') else arm_batches(token[f],name)
                a,k,nll=parameters(m,batch);curve=fast_cif(a,k,device='cuda');daily,conditional=daily_probabilities(curve)
                fresh=make_model(base,seed,'legacy' if name.startswith('legacy') else name)
                fresh.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
                aa,kk,_=parameters(fresh,batch);replay=fast_cif(aa,kk,device='cuda')
                parameter_error=float(max(np.abs(a-aa).max(),np.abs(k-kk).max()));curve_error=float(np.abs(curve-replay).max())
                if parameter_error>2e-6 or curve_error>2e-7:raise ValueError('replay exceeds float tolerance')
                errors[f'{seed}/{name}/{f}']=dict(parameters=parameter_error,curve=curve_error)
                detail=dict(nll=nll,readmission=public_metrics(metrics(d.times[ix[f]],d.events[ix[f]],curve[:,:,1],ordering=True)),
                            death=public_metrics(metrics(d.times[ix[f]],d.events[ix[f]],curve[:,:,2],cause=2,ordering=True)),
                            calibration={str(day):calibration_bins(d.times[ix[f]],d.events[ix[f]],curve[:,:,1],day=day) for day in (30,90,180,365)})
                models[str(seed)][name]['folds'][f]=detail;curves[seed][name][f]=curve
                np.savez_compressed(path/(f+'_predictions.npz'),hadm_ids=d.arrays['hadm_ids'][ix[f]],subject_ids=d.arrays['subject_ids'][ix[f]],
                    times=d.times[ix[f]],events=d.events[ix[f]],log_amplitude=a,shape=k,curve=curve,daily_event_probability=daily,
                    conditional_daily_event_probability=conditional,conditional_daily_valid=curve[:,:-1,0]>1e-12)
                del fresh
            del m
        ssl=json.loads((root/f'seed{seed}/interaction_ssl/self_supervision.json').read_text());recon[str(seed)]=ssl
        comparisons[str(seed)]={}
        pairs=(('pool_fine','pool_coarse'),('interaction_fine','pool_fine'),('interaction_ssl','interaction_fine'),
               ('flat_mlp','pool_fine'),('legacy_lr0.0001','legacy_lr0.0003'),('legacy_lr3e-05','legacy_lr0.0003'),
               ('pool_coarse','legacy_lr0.0001'),('interaction_fine','legacy_lr0.0001'),('interaction_ssl','legacy_lr0.0001'))
        for f in ix:
            comparisons[str(seed)][f]={f'{a} minus {b}':paired_interval(d.arrays['subject_ids'][ix[f]],d.times[ix[f]],d.events[ix[f]],
                curves[seed][a][f][:,:,1],curves[seed][b][f][:,:,1],seed=seed) for a,b in pairs}
    report=dict(status='completed_long_training_and_token_experiments',contract=contract,models=models,comparisons=comparisons,
        reconstruction=recon,reload_errors=errors,finalization_seconds=time.perf_counter()-started,
        numerical_tolerance=dict(parameters=2e-6,probabilities=2e-7),
        finalizer_sha256=sha256(Path(__file__)),limits=contract['limits']+[
        'Original training completed all arms; overly strict bitwise GPU replay was replaced with explicit floating-point tolerance.',
        'No training rerun or test-selected checkpoints in this finalizer.'])
    write_json(root/'report.json',report);render(root,report)
    print(json.dumps(dict(status=report['status'],seconds=report['finalization_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'));torch.set_num_threads(2);finalize(p.parse_args().root)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/token_model.py · lines 1–128

<!-- REVIEW_FRAGMENT {"path":"training/v6/token_model.py","part":1,"start_line":1,"end_line":128,"source_sha256":"ea51c37987b25e199978efa070d058caff0ff3afd95e364efa7ebe51ef994922","fragment_sha256":"ea51c37987b25e199978efa070d058caff0ff3afd95e364efa7ebe51ef994922","added_terminal_newline":false} -->
````python
"""临床概念/药物身份/片段先交互后池化；独立候选，不改变旧V6。"""
from dataclasses import dataclass, replace
import numpy as np
import torch
from torch import nn
from training.v6.model import AnchoredModel, masked_pool


@dataclass(frozen=True)
class TokenBatch:
    labs: torch.Tensor         # [N,29,20] 保留原来源特征
    numeric_valid: torch.Tensor
    lab_present: torch.Tensor
    coarse: torch.Tensor       # [N,7,4]
    named: torch.Tensor        # [N,21,23]
    named_present: torch.Tensor
    other: torch.Tensor        # [N,8,2] 微生物/呼吸支持
    text: torch.Tensor         # [N,L,768] 不先平均
    text_fields: torch.Tensor
    text_present: torch.Tensor
    masked_labs: torch.Tensor  # 自监督遮盖标记，不是缺失=正常


class TokenEncoder(nn.Module):
    def __init__(self, *, interaction=False, flat=False, width=128):
        super().__init__(); self.width=width; self.interaction=interaction; self.flat=flat
        self.lab=nn.Sequential(nn.Linear(21,width),nn.GELU(),nn.LayerNorm(width))
        self.coarse=nn.Linear(4,width); self.named=nn.Linear(23,width); self.other=nn.Linear(2,width)
        self.ids=nn.Embedding(65,width); self.field_ids=nn.Embedding(4,width)
        self.types=nn.Embedding(5,width)
        self.local=nn.Sequential(nn.Linear(width,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,width),nn.LayerNorm(width))
        layer=nn.TransformerEncoderLayer(width,4,256,.1,batch_first=True,norm_first=True)
        self.cross=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.score=nn.Linear(width,1); self.output_norm=nn.LayerNorm(width)
        # 最小对照不作token交互，完整表格加四字段均值；宽度与层数明确记录。
        self.flat_net=nn.Sequential(nn.Linear(29*20+7*4+21*23+8*2+4*768+4,width),nn.GELU(),
                                    nn.Dropout(.1),nn.Linear(width,width),nn.GELU(),nn.LayerNorm(width))
        self.text_projection=nn.Linear(768,width)  # 最后初始化
        self.flat_net.requires_grad_(flat)
        self.cross.requires_grad_(interaction and not flat)
        if flat:
            for name,module in self.named_children():
                if name!='flat_net': module.requires_grad_(False)

    def tokens(self,b):
        n=len(b.labs); dev=b.labs.device
        if b.labs.shape!=(n,29,20) or b.named.shape!=(n,21,23): raise ValueError('token interface mismatch')
        parts=(self.lab(torch.cat((b.labs,b.masked_labs.float().unsqueeze(-1)),2)),
               self.coarse(b.coarse),self.named(b.named),self.other(b.other))
        x=torch.cat(parts,1)+self.ids(torch.arange(65,device=dev))
        types=torch.tensor([0]*29+[1]*7+[1]*21+[2]*5+[3]*3,device=dev)
        x=x+self.types(types)
        text=self.text_projection(b.text)+self.field_ids(b.text_fields)+self.types.weight[4]
        x=torch.cat((x,text),1)
        present=torch.cat((b.lab_present,b.coarse[:,:,2:].bool().any(2),b.named_present,
                           b.other[:,:,1].bool(),b.text_present),1)
        x=torch.where(present[:,:,None],x,0)
        safe=present.clone(); safe[~safe.any(1),0]=True
        # 无顺序位置编码：交换完整token及其身份/有效性不改变住院表示。
        x=x+self.local(x)
        if self.interaction: x=self.cross(x,src_key_padding_mask=~safe)
        return x,present

    def flat_features(self,b):
        n=len(b.labs); text=b.text.new_zeros(n,4,768); flags=b.text.new_zeros(n,4)
        for field in range(4):
            mask=b.text_present & (b.text_fields==field)
            text[:,field]=(b.text*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
            flags[:,field]=mask.any(1)
        return torch.cat((b.labs.flatten(1),b.coarse.flatten(1),b.named.flatten(1),b.other.flatten(1),text.flatten(1),flags),1)

    def forward(self,b,*,mask_pmh=False):
        if mask_pmh: b=replace(b,text_present=b.text_present & (b.text_fields!=1))
        if self.flat: return self.flat_net(self.flat_features(b)),{}
        x,present=self.tokens(b); h=masked_pool(x,present,self.score)
        return self.output_norm(h)*present.any(1)[:,None],{'tokens':x,'present':present}


class TokenModel(AnchoredModel):
    def __init__(self,baseline,*,interaction=False,flat=False):
        super().__init__(baseline,768,width=128)
        self.encoder=TokenEncoder(interaction=interaction,flat=flat)


def fast_cif(a,k,*,device='cpu'):
    """与既有.25天积分相同的向量化实现；独立测试核对后用于频繁验证。"""
    x=torch.as_tensor(a,dtype=torch.float64,device=device); s=torch.as_tensor(k,dtype=torch.float64,device=device)
    if x.ndim!=2 or x.shape[1]!=2 or s.shape!=x.shape or not torch.isfinite(x).all() or not torch.isfinite(s).all() or (s<=0).any():
        raise ValueError('invalid risk parameters')
    grid=np.unique(np.r_[np.arange(0,365.0001,.25),np.geomspace(1e-10,.25,80)])
    g=torch.as_tensor(grid,dtype=torch.float64,device=device)
    indices=torch.as_tensor(np.searchsorted(grid,np.arange(366)),device=device)
    chunks=[]
    for start in range(0,len(x),512):
        cumulative=x[start:start+512,None,:].exp()*(g[None,:,None]/365).pow(s[start:start+512,None,:])
        increments=torch.diff(cumulative,dim=1).clamp_min(0); total=increments.sum(2)
        survival=(-cumulative.sum(2)).exp()
        mass=survival[:,:-1,None]*(-torch.expm1(-total))[:,:,None]*increments/total[:,:,None].clamp_min(1e-300)
        cif=torch.cat((torch.zeros_like(mass[:,:1]),mass.cumsum(1)),1)
        chunks.append(torch.cat((survival[:,:,None],cif),2)[:,indices].cpu().numpy())
    return np.concatenate(chunks)


def daily_probabilities(curve):
    if curve.ndim!=3 or curve.shape[1:]!=(366,3) or not np.isfinite(curve).all(): raise ValueError('invalid daily curve')
    if not np.allclose(curve.sum(2),1,atol=1e-8) or np.any(np.diff(curve[:,:,1:],axis=1)<-1e-9):
        raise ValueError('nonconserving/nonmonotone curve')
    daily=np.diff(curve[:,:,1:],axis=1)
    conditional=np.divide(daily,curve[:,:-1,:1],out=np.zeros_like(daily),where=curve[:,:-1,:1]>1e-12)
    return daily,conditional


def masked_reconstruction_batch(b,seed,rate=.15):
    """成分相关轴同时屏蔽，避免只遮低值轴却从同一项目高值轴抄答案。"""
    generator=torch.Generator(device=b.labs.device).manual_seed(seed)
    # 22轴真实命名是 arterial_low_ph / glucose_low；不能只去除前缀。
    from patient_data.v6_lab_state import AXES
    names=list(AXES)+['neutrophils','lymphocytes','rdw','anion_gap','chloride','eosinophils','eosinophils']
    families=[]
    for name in names:
        if name.startswith('arterial_') or name.startswith('bicarbonate_'): key='blood_gas'
        else: key=name.removeprefix('low_').removeprefix('high_').removesuffix('_low').removesuffix('_high')
        families.append(key)
    keys=list(dict.fromkeys(families)); chosen=torch.rand(len(b.labs),len(keys),device=b.labs.device,generator=generator)<rate
    selected=chosen[:,torch.tensor([keys.index(f) for f in families],device=b.labs.device)] & b.lab_present
    values=torch.where(selected[:,:,None],0,b.labs)
    target_mask=b.numeric_valid & selected[:,:,None]
    return replace(b,labs=values,masked_labs=selected),target_mask
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/token_result_figures.py · lines 1–43

<!-- REVIEW_FRAGMENT {"path":"training/v6/token_result_figures.py","part":1,"start_line":1,"end_line":43,"source_sha256":"0801051de960aabc022438cf0102d519437b026bf29a86178f7c5fd13077d031","fragment_sha256":"0801051de960aabc022438cf0102d519437b026bf29a86178f7c5fd13077d031","added_terminal_newline":false} -->
````python
"""补充配对增量/校准图，不改变训练、检查点或指标。"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(root):
    r=json.loads((root/'report.json').read_text());seeds=r['contract']['seeds']
    pairs=(('pool_fine minus pool_coarse','Drug identity + actual dose/frequency'),
           ('interaction_fine minus pool_fine','Interaction before pooling'),
           ('interaction_ssl minus interaction_fine','Masked reconstruction pretraining'),
           ('flat_mlp minus pool_fine','Flat MLP versus token pooling'),
           ('legacy_lr0.0001 minus legacy_lr0.0003','Lower LR: 1e-4 versus 3e-4'),
           ('legacy_lr3e-05 minus legacy_lr0.0003','Lower LR: 3e-5 versus 3e-4'))
    fig,ax=plt.subplots(1,2,figsize=(14,6),sharey=True)
    for row,(key,label) in enumerate(pairs):
        for j,seed in enumerate(seeds):
            m=r['comparisons'][str(seed)]['test'][key];y=row+(j-.5)*.16
            for col,(point,ci) in enumerate((('integrated_brier_difference','integrated_brier_95ci'),('auc90_difference','auc90_95ci'))):
                lo,hi=m[ci];ax[col].plot([lo,hi],[y,y],color=f'C{j}',alpha=.5)
                ax[col].scatter(m[point],y,color=f'C{j}',s=32,label=f'Seed {seed}' if row==0 else None)
    ax[0].set_yticks(np.arange(len(pairs)),[label for _,label in pairs]);ax[0].invert_yaxis()
    ax[0].set_title('IBS 7-365: candidate minus comparator (lower better)')
    ax[1].set_title('AUC90: candidate minus comparator (higher better)')
    for a in ax:a.axvline(0,color='black',linewidth=1);a.grid(axis='x',alpha=.2);a.legend();a.set_xlabel('Paired difference; patient-cluster bootstrap 200; 95% CI')
    fig.suptitle('Reused-test exploratory comparisons; no multiplicity adjustment');fig.tight_layout()
    fig.savefig(root/'paired_increment.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    for col,day in enumerate((30,90,365)):
        for name in ('pool_coarse','pool_fine','interaction_fine','interaction_ssl','flat_mlp'):
            bins=r['models'][str(seeds[0])][name]['folds']['test']['calibration'][str(day)]
            ax[col].plot([b['predicted'] for b in bins],[b['observed'] for b in bins],marker='o',label=name)
        ax[col].plot([0,1],[0,1],'k--',linewidth=1);ax[col].set_title(f'Seed {seeds[0]}: day {day} calibration')
        ax[col].set_xlabel('Predicted cumulative incidence');ax[col].set_ylabel('Observed World A fraction');ax[col].grid(alpha=.2)
    ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'calibration.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'));render(p.parse_args().root)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/token_verify.py · lines 1–65

<!-- REVIEW_FRAGMENT {"path":"training/v6/token_verify.py","part":1,"start_line":1,"end_line":65,"source_sha256":"91d715a0d77c8ac43d3b65dfbe7026ec5102a25d85df0f840f51ec0c4e04de8a","fragment_sha256":"91d715a0d77c8ac43d3b65dfbe7026ec5102a25d85df0f840f51ec0c4e04de8a","added_terminal_newline":false} -->
````python
"""新进程独立重载全部检查点，验证逐日接口；不训练、不修改旧结果。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.token_data import load_inputs,prepare
from training.v6.token_experiment import legacy_batches,make_model,arm_batches,parameters
from training.v6.token_model import fast_cif,daily_probabilities
from training.v6.horizon_objective import restore
from training.v6.risk import predict_cif
from training.v6.train import write_json


def verify(root):
    report=json.loads((root/'report.json').read_text());contract=report['contract'];cfg=json.loads((root/'config.json').read_text())
    for p,digest in contract['code_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('source changed '+p)
    d,c,h,ext,med,provenance=load_inputs(*(Path(cfg[k]) for k in ('reference','compact_dir','extension','medication')))
    if d.provenance!=contract['data_provenance'] or c.provenance!=contract['compact_provenance'] or provenance!=contract['extensions']:
        raise ValueError('independent data lineage differs')
    errors={};max_integration_error=0.;initial={}
    for seed,models in report['models'].items():
        seed=int(seed);basepath=Path(cfg['parent'])/f'seed{seed}/survival_base/best.pt'
        if sha256(basepath)!=contract['frozen_baselines'][str(seed)]:raise ValueError('base checkpoint changed')
        base=restore(Path(cfg['parent']),seed,'survival_base',h,.1)
        for fold in ('validation','test'):
            ix=np.flatnonzero(d.arrays['patient_splits']==fold);token=prepare(d,c,ext,med,ix);legacy=legacy_batches(d,c,ext,ix)
            for name,item in models.items():
                path=root/f'seed{seed}'/name
                if sha256(path/'best.pt')!=item['checkpoint_sha256']:raise ValueError('checkpoint changed')
                m=make_model(base,seed,'legacy' if name.startswith('legacy') else name)
                m.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
                if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):raise ValueError('frozen baseline differs')
                cache=legacy if name.startswith('legacy') else arm_batches(token,name)
                a,k,nll=parameters(m,cache);curve=fast_cif(a,k,device='cuda');daily,conditional=daily_probabilities(curve)
                with np.load(path/(fold+'_predictions.npz')) as z:
                    if not np.array_equal(z['hadm_ids'],d.arrays['hadm_ids'][ix]) or not np.array_equal(z['times'],d.times[ix]) or not np.array_equal(z['events'],d.events[ix]):
                        raise ValueError('prediction indices/labels changed')
                    err=float(np.max(np.abs(z['curve']-curve)));de=float(np.max(np.abs(z['daily_event_probability']-daily)))
                    ce=float(np.max(np.abs(z['conditional_daily_event_probability']-conditional)))
                if max(err,de,ce)>2e-7:raise ValueError('replayed probabilities exceed GPU float tolerance')
                if (daily< -1e-9).any() or (conditional.sum(2)>1+1e-8).any():raise ValueError('invalid daily event probability')
                slow=predict_cif(a[:5],k[:5],np.arange(366));max_integration_error=max(max_integration_error,float(np.max(np.abs(slow-curve[:5]))))
                errors[f'{seed}/{name}/{fold}']=dict(curve=err,daily=de,conditional=ce,nll=nll)
                history=json.loads((path/'history.json').read_text())
                if len(history)!=contract['epochs']+1 or history[-1]['epoch']!=contract['epochs']:raise ValueError('incomplete long training')
                selected=min(history,key=lambda x:x['validation_ibs'])
                if selected['epoch']!=item['best_epoch']:raise ValueError('checkpoint not validation IBS selected')
                initial[f'{seed}/{name}']=history[0]['validation_ibs'];del m
            del token,legacy
    for seed in contract['seeds']:
        values=[v for name,v in initial.items() if name.startswith(f'{seed}/')]
        if max(values)-min(values)>1e-10:raise ValueError('zero-start risk differs')
    if max_integration_error>1e-10:raise ValueError('fast/reference integration differs')
    proof=dict(status='independently_verified_neural',report_sha256=sha256(root/'report.json'),code_sha256=contract['code_sha256'],
        max_reference_integration_error=max_integration_error,reload_errors=errors,epoch0_risks_equal=True,
        complete_long_training=True,selected_by_validation_ibs=True,daily_interface='CIF; unconditional day mass; conditional day mass',
        limits=['Independent verification is engineering correctness, not clinical utility or causal validation.'])
    write_json(root/'independent_verification.json',proof);print(json.dumps(dict(status=proof['status'],checkpoints=len(errors),integration_error=max_integration_error)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'));torch.set_num_threads(2);verify(p.parse_args().root)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/train.py · lines 1–272

<!-- REVIEW_FRAGMENT {"path":"training/v6/train.py","part":1,"start_line":1,"end_line":272,"source_sha256":"a9397e187b93a25d5d6dd92ae9b57188955b07fb5c1dbe8148137594a83a3165","fragment_sha256":"a9397e187b93a25d5d6dd92ae9b57188955b07fb5c1dbe8148137594a83a3165","added_terminal_newline":false} -->
````python
"""固定单种子四臂完整训练；验证选模后才统一读取历史测试预测。"""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from patient_data.export_mimiciv import sha256
from training.v6.data import load
from training.v6.model import HistoryBaseline, ConstantCorrection, AnchoredModel
from training.v6.risk import competing_nll, predict_cif, RiskOutput
from training.v6.evaluation import GRID, metrics, public_metrics, paired_interval

ARMS = ('history', 'constant', 'clinical_no_text', 'clinical_bert')


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def forward(model, history, batch):
    return model(history, batch) if isinstance(model, AnchoredModel) else model(history)


def make_batches(data, indices, size, device, *, include_text):
    result = []
    for start in range(0, len(indices), size):
        positions = indices[start:start + size]
        h, b, t, e = data.collate(positions, device)
        if not include_text:
            b = replace(b, snippet_vectors=b.snippet_vectors[:0], snippet_index=b.snippet_index[:0])
        result.append((positions, h, b, t, e))
    return result


@torch.no_grad()
def predict(model, batches):
    model.eval()
    aa, kk, losses, deltas = [], [], [], []
    for positions, h, b, t, e in batches:
        output = forward(model, h, b)
        aa.append(output.log_amplitude.cpu().numpy()); kk.append(output.shape.cpu().numpy())
        losses.append((float(competing_nll(output, t, e)), len(positions)))
        if output.delta is not None:
            deltas.append(output.delta.cpu().numpy())
    a, k = np.concatenate(aa), np.concatenate(kk)
    nll = sum(x*n for x, n in losses) / sum(n for _, n in losses)
    delta = np.concatenate(deltas) if deltas else np.zeros((len(a), 2, 2))
    return a, k, nll, delta


def gradient_diagnostics(model):
    report = {}
    modules = [('current', model.encoder.current), ('lab', model.encoder.lab),
                         ('medication', model.encoder.med), ('text', model.encoder.text_projection),
                         ('fusion', model.encoder.fusion)]
    for name in ('microbiology', 'support'):
        if hasattr(model.encoder, name):
            modules.append((name, getattr(model.encoder, name)))
    for name, module in modules:
        params = list(module.parameters())
        count = sum(p.numel() for p in params)
        grads = [p.grad for p in params if p.grad is not None]
        report[name] = {"gradient_rms_last_batch": math.sqrt(sum(float(g.square().sum()) for g in grads) / count),
                        "parameter_rms": math.sqrt(sum(float(p.detach().square().sum()) for p in params) / count)}
    return report


def fit(model, data, train_batches, val_batches, validation_indices, directory, *, epochs, lr,
        seed, patience, warmup=0, weight_decay=1e-4):
    directory.mkdir()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    # 构造/设备迁移后恢复共同随机流；BERT/no-text共同架构与相同非文本初值。
    torch.manual_seed(seed)
    # 默认冻结臂行为不变；独立匹配解冻候选允许历史参数收到梯度。
    frozen = ({k: v.detach().cpu().clone() for k, v in model.baseline.state_dict().items()}
              if hasattr(model, 'baseline') and not any(p.requires_grad for p in model.baseline.parameters()) else None)
    history, best_score, best_epoch, stale = [], math.inf, -1, 0
    best_state = None
    started = time.perf_counter()
    for epoch in range(epochs + 1):
        train_nll, regularization, diagnostics = None, None, {}
        if epoch > 0:
            if warmup and epoch <= warmup:
                rate = lr * epoch / warmup
            elif warmup:
                rate = lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup))))
            else:
                rate = lr
            for group in optimizer.param_groups:
                group['lr'] = rate
            model.train()
            summed, penalty_sum, total = 0., 0., 0
            for j in rng.permutation(len(train_batches)):
                positions, h, b, t, e = train_batches[j]
                optimizer.zero_grad(set_to_none=True)
                output = forward(model, h, b)
                nll = competing_nll(output, t, e)
                penalty = output.delta.square().mean() * 1e-4 if output.delta is not None else nll.new_zeros(())
                loss = nll + penalty
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                loss.backward()
                if frozen is not None and any(p.grad is not None for p in model.baseline.parameters()):
                    raise ValueError('baseline gradient leakage')
                torch.nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step()
                summed += float(nll.detach()) * len(positions)
                penalty_sum += float(penalty.detach()) * len(positions)
                total += len(positions)
            train_nll, regularization = summed / total, penalty_sum / total
            if isinstance(model, AnchoredModel):
                diagnostics = gradient_diagnostics(model)
        a, k, val_nll, delta = predict(model, val_batches)
        curves = predict_cif(a, k, GRID, step_days=.25)
        val = metrics(data.times[validation_indices], data.events[validation_indices], curves[:, :, 1])
        score = val['integrated_brier_7_365']
        row = {'epoch': epoch, 'train_nll': train_nll, 'regularization': regularization,
               'validation_nll': val_nll, 'validation_integrated_brier': score,
               'validation_mean_auc': val['mean_auc_7_365'], 'diagnostics': diagnostics,
               'delta_mean': delta.mean(axis=0).tolist(), 'delta_std': delta.std(axis=0).tolist(),
               'shape_min': float(k.min()), 'shape_max': float(k.max()),
               'elapsed_seconds': round(time.perf_counter() - started, 2)}
        history.append(row)
        write_json(directory / 'history.json', history)
        if score < best_score - 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: v.detach().cpu().clone() for key, v in model.state_dict().items()}
        elif epoch > 0:
            stale += 1
        if epoch % 5 == 0 or stale >= patience:
            print(json.dumps({'arm': directory.name, 'epoch': epoch, 'validation_IBS': score,
                              'best_epoch': best_epoch, 'seconds': row['elapsed_seconds']}), flush=True)
        if epoch > 0 and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError('no best checkpoint')
    model.load_state_dict(best_state)
    if frozen is not None:
        if not all(torch.equal(v.detach().cpu(), frozen[key]) for key, v in model.baseline.state_dict().items()):
            raise ValueError('frozen baseline changed')
    checkpoint = directory / 'best.pt'
    torch.save({'state_dict': best_state, 'best_epoch': best_epoch, 'seed': seed}, checkpoint)
    # 验证真正从磁盘检查点重载而非只保留内存模型。
    restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(restored['state_dict'])
    a1, k1, _, _ = predict(model, val_batches)
    model.load_state_dict(best_state)
    a2, k2, _, _ = predict(model, val_batches)
    if not np.allclose(a1, a2, atol=2e-6, rtol=1e-5) or not np.allclose(k1, k2, atol=2e-6, rtol=1e-5):
        raise ValueError('checkpoint prediction reproduction failed')
    return {'best_epoch': best_epoch, 'epochs_trained': len(history) - 1, 'best_validation_IBS': best_score,
            'history': history, 'checkpoint_sha256': sha256(checkpoint), 'checkpoint_reloaded': True}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required, no CPU fallback')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True)
    config = vars(args).copy()
    write_json(args.output / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in config.items()})
    write_json(args.output / 'status.json', {'status': 'loading_validated_data'})
    data = load(args.snapshot, args.state_dir, args.labels, field_dir=args.field_dir, vectors_path=args.vectors_path)
    indices = {fold: np.flatnonzero(data.arrays['patient_splits'] == fold) for fold in ('train', 'validation', 'test')}
    if any(not len(ix) for ix in indices.values()):
        raise ValueError('empty fold')
    # 验证/训练预加载；此时不构造或计算历史测试预测。
    cached = {text: {f: make_batches(data, indices[f], args.batch_size, 'cuda', include_text=text)
                      for f in ('train', 'validation')} for text in (False, True)}
    common = {'data_provenance': data.provenance, 'history_normalization': data.normalization,
              'seed': args.seed, 'selection': 'validation readmission integrated Brier 7-365 daily grid',
              'optimizer': 'AdamW; NLL plus 1e-4 delta mean square; weight decay 1e-4; gradient clip 1',
              'sampling': 'admission equal weight; fixed admission minibatches with shuffled batch order',
              'folds': {f: {'admissions': len(ix), 'patients': len(set(data.arrays['subject_ids'][ix].tolist()))} for f, ix in indices.items()},
              'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'code_sha256': {f: sha256(Path(__file__).parent / f) for f in ('train.py', 'model.py', 'risk.py', 'data.py', 'evaluation.py')}}
    write_json(args.output / 'contract.json', common)
    torch.manual_seed(args.seed)
    baseline = HistoryBaseline().cuda()
    fitted, models = {}, {'history': baseline}
    fitted['history'] = fit(baseline, data, cached[False]['train'], cached[False]['validation'], indices['validation'],
                             args.output / 'history', epochs=args.baseline_epochs, lr=.01, seed=args.seed, patience=20)
    for arm in ARMS[1:]:
        write_json(args.output / 'status.json', {'status': 'training', 'arm': arm})
        torch.manual_seed(args.seed)
        base_copy = copy.deepcopy(baseline).cuda()
        model = (ConstantCorrection(base_copy) if arm == 'constant'
                 else AnchoredModel(base_copy, data.text_vectors.shape[1])).cuda()
        models[arm] = model
        text = arm == 'clinical_bert'
        fitted[arm] = fit(model, data, cached[text]['train'], cached[text]['validation'], indices['validation'],
                          args.output / arm, epochs=args.epochs, lr=.005 if arm == 'constant' else 3e-4,
                          seed=args.seed, patience=10, warmup=0 if arm == 'constant' else 5)
    write_json(args.output / 'status.json', {'status': 'all_models_selected_before_historical_test_evaluation'})
    results, predictions = {}, {}
    for arm in ARMS:
        results[arm] = {k: v for k, v in fitted[arm].items() if k != 'history'}
        results[arm]['folds'] = {}
        for fold in ('validation', 'test'):
            batches = (cached[arm == 'clinical_bert']['validation'] if fold == 'validation' else
                       make_batches(data, indices[fold], args.batch_size, 'cuda', include_text=arm == 'clinical_bert'))
            a, k, nll, delta = predict(models[arm], batches)
            curve = predict_cif(a, k, GRID, step_days=.25)
            fine = predict_cif(a[:64], k[:64], GRID, step_days=.125)
            error = float(np.max(np.abs(fine - curve[:64])))
            if error > 2e-4:
                raise ValueError(f'CIF step convergence insufficient {arm}/{fold}: {error}')
            ix = indices[fold]
            readmission = metrics(data.times[ix], data.events[ix], curve[:, :, 1], ordering=True)
            death = metrics(data.times[ix], data.events[ix], curve[:, :, 2], cause=2, ordering=True)
            stratified = {}
            prior = np.expm1(data.history_raw[ix, 2]).round().astype(int)
            for label, keep in (('prior_0', prior == 0), ('prior_1_2', (prior >= 1) & (prior <= 2)), ('prior_3_plus', prior >= 3)):
                if keep.any():
                    stratified[label] = public_metrics(metrics(data.times[ix][keep], data.events[ix][keep], curve[keep, :, 1], ordering=True))
            results[arm]['folds'][fold] = {'readmission': public_metrics(readmission), 'death': public_metrics(death),
                                          'nll': nll, 'history_strata': stratified, 'CIF_step_convergence_max_error_first64': error,
                                          'maximum_probability_sum_error': float(np.abs(curve.sum(2) - 1).max())}
            predictions[arm, fold] = curve
            np.savez_compressed(args.output / arm / (fold + '_predictions.npz'),
                                hadm_ids=data.arrays['hadm_ids'][ix], subject_ids=data.arrays['subject_ids'][ix],
                                times=data.times[ix], events=data.events[ix], history_counts=prior,
                                curve=curve, log_amplitude=a, shape=k, delta=delta)
    comparisons = {}
    for fold in ('validation', 'test'):
        ix = indices[fold]
        comparisons[fold] = {}
        for candidate, comparator in (('constant', 'history'), ('clinical_no_text', 'constant'),
                                       ('clinical_bert', 'clinical_no_text'), ('clinical_bert', 'constant')):
            label = candidate + '_minus_' + comparator
            comparisons[fold][label] = paired_interval(data.arrays['subject_ids'][ix], data.times[ix], data.events[ix],
                                                       predictions[candidate, fold][:, :, 1], predictions[comparator, fold][:, :, 1],
                                                       replicates=args.bootstrap, seed=args.seed)
    report = {'status': 'completed_single_seed_exploratory', 'contract': common, 'models': results, 'comparisons': comparisons,
              'limits': ['World A record-observed endpoint, external readmissions unknown', 'Historical test reused, not a new blind test',
                         'One training seed; cluster intervals do not include retraining variance',
                         'No new encoder, drug-lab cycle or extra predictor introduced', 'C-index explicitly project fixed-horizon CIF score definition']}
    write_json(args.output / 'report.json', report)
    from training.v6.plot import render
    render(args.output)
    write_json(args.output / 'status.json', {'status': report['status']})
    print(json.dumps({'status': report['status'], 'output': str(args.output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'state_dir', 'labels', 'output', 'field_dir', 'vectors_path'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--baseline-epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--bootstrap', type=int, default=200)
    args = parser.parse_args()
    if args.epochs < 1 or args.baseline_epochs < 1 or args.batch_size < 1 or args.bootstrap < 20:
        parser.error('invalid training budget')
    run(args)


if __name__ == '__main__':
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/training_history_audit.py · lines 1–96

<!-- REVIEW_FRAGMENT {"path":"training/v6/training_history_audit.py","part":1,"start_line":1,"end_line":96,"source_sha256":"d758a38220214dee14a1ae22f005641f36292ac29f39ebbfed9bf48f9b114e25","fragment_sha256":"d758a38220214dee14a1ae22f005641f36292ac29f39ebbfed9bf48f9b114e25","added_terminal_newline":false} -->
````python
"""只读已有日志，比较同种训练目标的验证轨迹；不加载患者资料或训练模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path) -> tuple[dict, dict]:
    report = json.loads((root / 'report.json').read_text(encoding='utf-8'))
    contract = report['contract']
    summaries, histories, hashes = [], {}, {}
    for seed in contract['seeds']:
        for arm in contract['arms']:
            path = root / f'seed{seed}' / arm / 'history.json'
            rows = json.loads(path.read_text(encoding='utf-8'))
            if [r['epoch'] for r in rows] != list(range(len(rows))):
                raise ValueError(f'non-contiguous epoch: {path}')
            for row in rows:
                for key in ('validation_objective', 'validation_brier90'):
                    if not math.isfinite(row[key]):
                        raise ValueError(f'non-finite {key}: {path}')
                if row['epoch'] and not math.isfinite(row['train_objective']):
                    raise ValueError(f'non-finite train objective: {path}')
            epoch = report['models'][str(seed)][arm]['best_epoch']
            chosen, last = rows[epoch], rows[-1]
            if report['models'][str(seed)][arm]['epochs_trained'] != last['epoch']:
                raise ValueError(f'training length differs: {path}')
            key = f'seed{seed}/{arm}'
            histories[key] = rows
            hashes[str(path.relative_to(root))] = digest(path)
            summaries.append(dict(seed=seed, arm=arm, best_epoch=epoch,
                epochs_trained=last['epoch'],
                train_nll_at_best=chosen['train_objective'], train_nll_final=last['train_objective'],
                validation_nll_at_best=chosen['validation_objective'], validation_nll_final=last['validation_objective'],
                validation_brier90_at_best=chosen['validation_brier90'], validation_brier90_final=last['validation_brier90'],
                later_validation_nll_change=last['validation_objective']-chosen['validation_objective'],
                later_validation_brier90_change=last['validation_brier90']-chosen['validation_brier90']))
    return dict(source=str(root), source_report_sha256=digest(root/'report.json'),
        history_sha256=hashes, selection=contract['selection'], seeds=contract['seeds'], arms=contract['arms'],
        status='read_only_saved_history_audit_no_new_training', rows=summaries,
        limitations=['Training NLL is online batch mean; validation NLL is evaluated at epoch end.',
                     'No gradient/update norms were recorded; instability cause is not uniquely identified.',
                     'A later validation loss increase does not prove every new feature is harmful.']), histories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result, histories = audit(args.source)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(result['seeds']), 2, figsize=(13, 8), squeeze=False)
    colors = dict(control='tab:blue', new_labs='tab:orange', med_details='tab:green',
                  both='tab:red', availability='tab:purple')
    for row, seed in enumerate(result['seeds']):
        for col, metric in enumerate(('validation_objective', 'validation_brier90')):
            ax = axes[row, col]
            for arm in result['arms']:
                history = histories[f'seed{seed}/{arm}']
                ax.plot([x['epoch'] for x in history], [x[metric] for x in history],
                        label=arm, color=colors.get(arm))
                selected = next(x for x in result['rows'] if x['seed']==seed and x['arm']==arm)
                epoch = selected['best_epoch']
                ax.scatter(epoch, history[epoch][metric], marker='D', s=22, color=colors.get(arm))
            ax.set_title(f'Seed {seed}: '+('Validation joint survival NLL' if col==0 else 'Validation Brier90')+' (lower better)')
            ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
            if row==0:
                ax.legend(fontsize=9)
    fig.suptitle('Saved validation logs only; no new training; diamonds = selected checkpoints')
    fig.tight_layout()
    args.output.mkdir(parents=True)
    (args.output/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    fig.savefig(args.output/'validation_nll_and_brier90.png', dpi=170)
    plt.close(fig)
    for name, expected in result['history_sha256'].items():
        if digest(args.source/name) != expected:
            raise ValueError('source history changed during audit')
    if digest(args.source/'report.json') != result['source_report_sha256']:
        raise ValueError('source report changed during audit')
    print(json.dumps(dict(status=result['status'], histories=len(result['rows']), output=str(args.output))))


if __name__ == '__main__':
    main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/unified_longitudinal.py · lines 1–152

<!-- REVIEW_FRAGMENT {"path":"training/v6/unified_longitudinal.py","part":1,"start_line":1,"end_line":152,"source_sha256":"2f616b2df8a70d817f6b2df4ead08cdeca90e32e416a403cef17c07dc1e7676e","fragment_sha256":"2f616b2df8a70d817f6b2df4ead08cdeca90e32e416a403cef17c07dc1e7676e","added_terminal_newline":false} -->
````python
"""统一检验与可学习历史的2×2条件；完整输入在每步重新编码，非教师隐藏缓存。"""
import copy
from dataclasses import replace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.model import masked_pool
from training.v6.risk import RiskOutput

ARMS = ('current_original', 'current_unified', 'history_original', 'history_unified')


def f1_at(y, score, threshold):
    y, score = np.asarray(y), np.asarray(score)
    if (y.dtype != bool or y.shape != score.shape or y.ndim != 1 or not len(y)
            or not np.isfinite(score).all() or np.any((score < 0)|(score > 1)) or not np.isfinite(threshold)):
        raise ValueError('invalid annual F1 inputs')
    flag = score >= threshold
    tp, fp, fn, tn = (int(x.sum()) for x in (flag&y, flag&~y, ~flag&y, ~flag&~y))
    return dict(threshold=float(threshold), f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
                false_positive_rate=fp/(fp+tn) if fp+tn else 0., tp=tp, fp=fp, fn=fn, tn=tn, alerts=tp+fp)


def best_f1(y, score):
    f1_at(y, score, 1.)  # 同一接口验证；分数同值必须同组。
    if not y.any(): return f1_at(y, score, 1.)
    order = np.argsort(-score, kind='stable'); p = score[order]; labels = y[order]
    last = np.r_[p[:-1] != p[1:], True]
    tp = np.cumsum(labels)[last]; count = np.arange(1,len(y)+1)[last]
    values = 2*tp/(count+y.sum())
    chosen = int(np.flatnonzero(values >= values.max()-1e-12)[0])
    return f1_at(y, score, float(p[last][chosen]))


class LongitudinalModel(nn.Module):
    def __init__(self, fresh, arm):
        super().__init__()
        if arm not in ARMS: raise ValueError(arm)
        self.original = copy.deepcopy(fresh)
        self.arm = arm; self.unified = arm.endswith('unified'); self.use_history = arm.startswith('history')
        # 所有条件分配同一新模块/RNG，开关而非构造顺序决定实际信息。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.lab_ids = nn.Embedding(36,128)
            self.eos_context = nn.Sequential(nn.Linear(16,32), nn.GELU(), nn.Linear(32,128))
            self.past_context = nn.Sequential(nn.LayerNorm(34), nn.Linear(34,128), nn.GELU())
            self.gap = nn.Sequential(nn.Linear(1,128), nn.Tanh())
            layer = nn.TransformerEncoderLayer(128,4,256,.1,batch_first=True,norm_first=True)
            self.temporal = nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.history_head = nn.Linear(128,4)
            nn.init.zeros_(self.history_head.weight); nn.init.zeros_(self.history_head.bias)
        with torch.no_grad(): self.lab_ids.weight[:22].copy_(self.original.encoder.base.lab_ids.weight)
        self.original.baseline.requires_grad_(False)
        core = self.original.encoder.base
        for name, p in self.original.named_parameters():
            if name.startswith(('encoder.base.current', 'encoder.base.branch_norms.0', 'encoder.extra_meds')):
                p.requires_grad_(False)
            if self.unified and name.startswith(('encoder.extra_labs', 'encoder.base.lab_ids')):
                p.requires_grad_(False)
        self.lab_ids.requires_grad_(self.unified); self.eos_context.requires_grad_(self.unified)
        for module in (self.past_context,self.gap,self.temporal,self.history_head):
            module.requires_grad_(self.use_history)

    def train(self, mode=True):
        super().train(mode); self.original.baseline.eval(); return self

    def visits(self, compact, legacy_extra, unified, eos_context):
        core = self.original.encoder.base; b = compact.clinical; n = len(b.current_values)
        if self.unified:
            observed = unified[:,:,6] > 0
            clean = torch.where(observed[:,:,None], unified, 0.)
            l = masked_pool(core.lab(clean)+self.lab_ids(torch.arange(36,device=clean.device)), observed,core.lab_score)
            eos_present = observed[:,32:].any(1)
            l = l + self.eos_context(eos_context)*eos_present[:,None]
        else:
            observed = b.lab_axis_inputs[:,:,6] > 0
            clean = torch.where(observed[:,:,None],b.lab_axis_inputs,0.)
            l = masked_pool(core.lab(clean)+core.lab_ids(torch.arange(22,device=clean.device)), observed,core.lab_score)
        med = compact.med_features; mp = med[:,:,2:].bool().any((1,2)); m = core.med(med.flatten(1))
        micro = core.microbiology(torch.cat((compact.micro_values,compact.micro_masks.float()),1))
        support = core.support(torch.cat((compact.support_values,compact.support_masks.float()),1))
        present = [torch.zeros(n,dtype=torch.bool,device=clean.device),observed.any(1),mp,
                   compact.micro_masks.any(1),compact.support_masks.any(1)]
        s = torch.zeros_like(l)
        branches = [norm(x)*p[:,None] for norm,x,p in zip(core.branch_norms,(s,l,m,micro,support),present,strict=True)]
        text, text_present = core._text(b,n,mask_pmh=False)
        h = core.fusion(torch.cat((*branches,text,torch.stack((*present,text_present),1).float()),1))
        if not self.unified:
            h = h + self.original.encoder.extra_labs(legacy_extra)*legacy_extra.ne(0).any(1)[:,None]
        return h

    def forward(self, batch):
        compact, legacy, labs, eos, context, current_index, past_index, past_present, gaps = batch
        hidden = self.visits(compact, legacy, labs, eos)
        current = hidden[current_index]
        delta = self.original.delta_head(current).reshape(-1,2,2)
        if self.use_history:
            history = hidden[past_index] + self.past_context(context[past_index])
            history = history + self.gap(torch.log1p(gaps.clamp_min(0))[:,:,None]/6.)
            history = torch.where(past_present[:,:,None],history,0.)
            seq = torch.cat((current[:,None],history),1)
            valid = torch.cat((torch.ones(len(current),1,dtype=torch.bool,device=current.device),past_present),1)
            read = self.temporal(seq,src_key_padding_mask=~valid)[:,0]
            addition = self.history_head(read).reshape(-1,2,2)*past_present.any(1)[:,None,None]
            delta = delta + addition
        with torch.no_grad(): base = self.original.baseline(context[current_index])
        return RiskOutput((base.log_amplitude+delta[:,:,0]).clamp(-10.,6.),
                          F.softplus(self.original.baseline.raw_shape.detach()+delta[:,:,1])+.1,delta)


def make_batch(c, ext, interface, positions, neighbors, offsets, gaps, note_times, admission_times, *, history, device='cuda'):
    """按当前预测点构造已可见的历史；相同过去行的文本可用状态各自明确，不依赖batch其他人。"""
    keys = [(int(p),True) for p in positions]; unique = list(dict.fromkeys(keys)); by_key = {k:i for i,k in enumerate(unique)}
    past_keys = []; past_gaps = []
    for p in positions:
        selected = neighbors[offsets[p]:offsets[p+1]] if history else np.array([],int)
        relations = []
        for v in selected:
            key = (int(v),bool(note_times[v] <= admission_times[p]))
            if key not in by_key: by_key[key]=len(unique);unique.append(key)
            relations.append(by_key[key])
        past_keys.append(relations); past_gaps.append(gaps[offsets[p]:offsets[p+1]] if history else [])
    width = max(1,max(map(len,past_keys))); n=len(positions)
    index=np.zeros((n,width),np.int64); present=np.zeros((n,width),bool); times=np.zeros((n,width),np.float32)
    for i,(v,t) in enumerate(zip(past_keys,past_gaps,strict=True)):
        index[i,:len(v)]=v;present[i,:len(v)]=True;times[i,:len(v)]=t
    # 不直接让重复hadm进入既有collate；先唯一取值，再显式展开并重映射全部片段。
    source = np.array(sorted({p for p,_ in unique}),np.int64); source_slot={p:i for i,p in enumerate(source)}
    h, compact, _, _ = c.collate(source,device=device)
    from training.v6.exploration_model import expanded_inputs
    context = expanded_inputs(h,compact)
    expand=torch.as_tensor([source_slot[p] for p,_ in unique],device=device)
    b=compact.clinical
    vectors=[]; refs=[]
    for i,(p,allowed) in enumerate(unique):
        if not allowed: continue
        mask=b.snippet_index[:,0]==source_slot[p]
        vectors.append(b.snippet_vectors[mask]); r=b.snippet_index[mask].clone();r[:,0]=i;refs.append(r)
    fields = dict(current_values=b.current_values[expand]*0, current_masks=b.current_masks[expand]&False,
                  lab_axis_inputs=b.lab_axis_inputs[expand], med_exposure=b.med_exposure[expand],
                  med_source_mask=b.med_source_mask[expand],
                  snippet_vectors=torch.cat(vectors) if vectors else b.snippet_vectors[:0],
                  snippet_index=torch.cat(refs) if refs else b.snippet_index[:0])
    compact=replace(compact,clinical=replace(b,**fields),
        **{k:getattr(compact,k)[expand] for k in ('med_features','micro_values','micro_masks','support_values','support_masks')})
    original_positions=np.array([p for p,_ in unique])
    legacy=np.concatenate((ext['new_labs'][original_positions].reshape(len(unique),-1),ext['eos'][original_positions].reshape(len(unique),-1)),1)
    tensor=lambda x:torch.as_tensor(x,device=device)
    return (compact,tensor(legacy),tensor(interface['values'][original_positions]),tensor(interface['eos_context'][original_positions]),
            context[expand],tensor(np.array([by_key[k] for k in keys])),tensor(index),tensor(present),tensor(times))
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/unified_longitudinal_experiment.py · lines 1–311

<!-- REVIEW_FRAGMENT {"path":"training/v6/unified_longitudinal_experiment.py","part":1,"start_line":1,"end_line":311,"source_sha256":"09b549fcffd57e3fffde5f56228449ea897aa09e986f95109b461c423cf74c73","fragment_sha256":"09b549fcffd57e3fffde5f56228449ea897aa09e986f95109b461c423cf74c73","added_terminal_newline":false} -->
````python
"""一年F1选点的单seed四条件；当前/过去原输入共同训练，原正式模型不改。"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch

from patient_data.export_mimiciv import rows,sha256
from patient_data.unified_lab_state import ROOT as LAB_ROOT
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.module_surgery import SurgeryModel,probability_curve
from training.v6.module_surgery_experiment import summary,save
from training.v6.horizon_objective import restore
from training.v6.early_stopping import EarlyStopping
from training.v6.risk import competing_nll
from training.v6.train import write_json
from training.v6.unified_longitudinal import ARMS,LongitudinalModel,make_batch,best_f1,f1_at

ROOT=Path('training_runs/v6_unified_longitudinal_f1_2026-10-14_r2')
PROTOCOL=Path('docs/V6_UNIFIED_LONGITUDINAL_F1_PRD_2026-10-14.md')
INDEX=Path('training_runs/v6_longitudinal_feasibility_2026-10-13')
SECTIONS=Path('outputs/01a08b96-2988-7001-af1d-b40d901f3400/private/v5_section_aligned/dataset_v4/note_sections.csv.gz')
MAX_EPOCHS=45


def seconds(value):
    return int(np.datetime64(datetime.fromisoformat(value),'s').astype(np.int64))


def setup():
    d,c,h,ext,base,bins,tune,confirm,search,_=checked()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)
    manifest=json.loads((LAB_ROOT/'manifest.json').read_text())
    if sha256(LAB_ROOT/'interface.npz')!=manifest['output_sha256']:raise ValueError('unified interface changed')
    for p,digest in manifest['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('unified source changed')
    with np.load(LAB_ROOT/'interface.npz') as z:lab={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits','values','eos_context')}
    for k in ('hadm_ids','subject_ids','patient_splits'):np.testing.assert_array_equal(lab[k],d.arrays[k])
    proof=json.loads((INDEX/'feasibility.json').read_text()); index_path=INDEX/'past_clinical_indices.npz'
    if sha256(index_path)!=proof['input_sha256'] or proof['strict_labels_sha256']!=d.provenance['label_sha256']:
        raise ValueError('historical index differs')
    with np.load(index_path) as z:
        for k in ('hadm_ids','subject_ids','patient_splits'):np.testing.assert_array_equal(z[k],d.arrays[k])
        neighbors,offsets,gaps=(z[k] for k in ('previous_positions','offsets','days_from_previous_discharge_to_current_admission'))
    reference_cfg=json.loads((Path(search['input_config']['reference'])/'config.json').read_text())
    snapshot=Path(reference_cfg['snapshot']); field_dir=Path(reference_cfg['field_dir'])
    # 使用B实际section引用绑定选中文书，不以所有小结的最大存档时间猜本次文书。
    section_note={r['section_id']:(r['hadm_id'],r['note_id']) for r in rows(SECTIONS)}
    by_id={str(v):i for i,v in enumerate(d.arrays['hadm_ids'])}; used={i:set() for i in range(len(by_id))}
    for r in rows(field_dir/'shared_text.csv.gz'):
        if r['section_id'] not in section_note:raise ValueError('B section not bound to saved note')
        hadm,note=section_note[r['section_id']]
        if hadm!=r['hadm_id']:raise ValueError('B note patient/admission mapping differs')
        used[by_id[hadm]].add(note)
    note_store={}
    for r in rows(snapshot/'raw/discharge.csv.gz'):
        try: note_store[r['note_id']]=(r['hadm_id'],seconds(r['storetime']))
        except ValueError:note_store[r['note_id']]=(r['hadm_id'],np.iinfo(np.int64).max)
    note_times=np.full(len(by_id),np.iinfo(np.int64).max,np.int64)
    for i,notes in used.items():
        if notes:
            if any(note not in note_store or note_store[note][0]!=str(d.arrays['hadm_ids'][i]) for note in notes):
                raise ValueError('selected B note absent in controlled source')
            note_times[i]=max(note_store[note][1] for note in notes)
    dates={r['hadm_id']:(seconds(r['admittime']),seconds(r['dischtime'])) for r in rows(snapshot/'cohort/admissions.csv.gz')}
    admit=np.array([dates[str(h)][0] for h in d.arrays['hadm_ids']],np.int64)
    discharge=np.array([dates[str(h)][1] for h in d.arrays['hadm_ids']],np.int64)
    for p in range(len(by_id)):
        past=neighbors[offsets[p]:offsets[p+1]]
        if (np.any(past==p) or np.any(d.arrays['subject_ids'][past]!=d.arrays['subject_ids'][p])
                or np.any(d.arrays['patient_splits'][past]!=d.arrays['patient_splits'][p])
                or np.any(discharge[past]>admit[p]) or np.any(admit[past]>=admit[p])):
            raise ValueError('future/self/cross-patient historical visit')
    train=np.flatnonzero(d.arrays['patient_splits']=='train')
    base=restore(Path(search['input_config']['parent']),42,'survival_base',h,.1)
    fresh=SurgeryModel(base,42,'control',bins).cuda()
    teacher=SurgeryModel(base,42,'control',bins).cuda()
    teacher.load_state_dict(torch.load(REFERENCE/'seed42/control/best.pt',map_location='cuda',weights_only=True)['state_dict'])
    sources=dict(manifest['source_sha256'])
    for p in (Path(__file__),Path('training/v6/unified_longitudinal.py'),Path('patient_data/unified_lab_state.py'),
              PROTOCOL,Path('training/tests/test_v6_unified_longitudinal.py'),LAB_ROOT/'interface.npz',LAB_ROOT/'manifest.json',
              index_path,INDEX/'feasibility.json',SECTIONS,field_dir/'shared_text.csv.gz',snapshot/'raw/discharge.csv.gz',
              snapshot/'cohort/admissions.csv.gz',REFERENCE/'seed42/control/best.pt'):
        sources[str(p)]=sha256(p)
    contract=dict(source_sha256=sources,data_provenance=d.provenance,compact_provenance=c.provenance,
        labels_sha256=d.provenance['label_sha256'],frozen_base_sha256=sha256(Path(search['input_config']['parent'])/'seed42/survival_base/best.pt'),
        arms=list(ARMS),seed=42,horizon=365,selection='maximum tuning annual F1; tie lower readmission IBS; tie earlier epoch',
        threshold_selection='all unique tuning scores, equal scores grouped, F1 tie highest threshold',
        optimizer=dict(name='AdamW',lr=3e-4,weight_decay=1e-4,clip=1.,delta_penalty=1e-4),
        early=dict(min_epochs=10,patience=7,min_delta=1e-4,max_epochs=45),batch_size=128,
        original_clinical_weights_not_reused=True,history_encoder_trainable=True,no_history_truncation=True,
        no_high_frequency=True,bert_backbone_frozen=True,goal_automation_resumed=False,historical_test_used=False)
    relation=dict(neighbors=neighbors,offsets=offsets,gaps=gaps,note_times=note_times,admit=admit)
    context_counts=np.diff(offsets)[train]
    text_relations=sum(int((note_times[neighbors[offsets[p]:offsets[p+1]]]<=admit[p]).sum()) for p in train)
    source_report=dict(training_discharges=len(train),training_patients=len(np.unique(d.arrays['subject_ids'][train])),
        visits_with_history=int((context_counts>0).sum()),maximum_prior_visits=int(context_counts.max()),
        prior_visit_relations=int(context_counts.sum()),text_visible_prior_relations=text_relations,
        past_text_excluded_relations=int(context_counts.sum())-text_relations,
        text_visibility='all source notes of this B visit stored by current admission; unknown excludes historical text only')
    return d,c,ext,lab,relation,train,tune,confirm,fresh,teacher,contract,source_report


def cache(c,ext,lab,relation,indices,history):
    result=[]
    for start in range(0,len(indices),128):
        p=indices[start:start+128]
        b=make_batch(c,ext,lab,p,relation['neighbors'],relation['offsets'],relation['gaps'],
                     relation['note_times'],relation['admit'],history=history)
        result.append((p,b))
    return result


@torch.no_grad()
def predict(model,cache,d):
    model.eval();a=[];k=[];nll=0.;n=0
    for p,b in cache:
        out=model(b);a.append(out.log_amplitude.cpu().numpy());k.append(out.shape.cpu().numpy())
        nll+=float(competing_nll(out,torch.as_tensor(d.times[p],device='cuda'),torch.as_tensor(d.events[p],device='cuda')))*len(p);n+=len(p)
    a,k=np.concatenate(a),np.concatenate(k)
    return a,k,probability_curve(a,k),nll/n


def detail(d,indices,curve,nll,threshold=None):
    y=(d.events[indices]==1)&(d.times[indices]<=365)
    decision=best_f1(y,curve[:,365,1]) if threshold is None else f1_at(y,curve[:,365,1],threshold)
    return dict(annual=decision,**summary(curve,d,indices,nll))


def original_predict(teacher,d,c,ext,indices):
    """旧参照保留其实际TF32/256批路径；不拿新FP32路径冒充旧保存概率。"""
    from training.v6.module_surgery_experiment import batches,parameters
    old=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=True
        a,k,nll=parameters(teacher,batches(d,c,ext,indices))
        return a,k,probability_curve(a,k),nll
    finally:torch.backends.cuda.matmul.allow_tf32=old


class OriginalView(torch.nn.Module):
    def __init__(self,teacher):super().__init__();self.teacher=teacher
    def forward(self,b):
        from training.v6.detail_extension_experiment import DetailBatch
        from training.v6.model import AnchoredModel
        compact,legacy,_,_,context,cur,_,_,_=b
        if not torch.equal(cur,torch.arange(len(cur),device=cur.device)):raise ValueError('original view only current batch')
        return AnchoredModel.forward(self.teacher,context,DetailBatch(compact,legacy,torch.zeros(len(cur),192,device='cuda')))


def preflight(fresh,relation,d,c,ext,lab,train):
    counts=np.diff(relation['offsets'])[train]
    selected=train[np.argsort(-counts)[:16]]
    # 相同当前预测点，不同条件共享基础风险及风险头；检验维度改动不会漂移初值。
    incoming={flag:cache(c,ext,lab,relation,selected,flag)[0][1] for flag in (False,True)}
    initial={};updates={}
    for arm in ARMS:
        model=LongitudinalModel(fresh,arm).cuda().eval()
        b=incoming[model.use_history]
        out=model(b);initial[arm]=(out.log_amplitude.detach(),out.shape.detach())
        for a,k in initial.values():
            torch.testing.assert_close(out.log_amplitude,a,rtol=0,atol=0);torch.testing.assert_close(out.shape,k,rtol=0,atol=0)
        before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        frozen={n:p.detach().clone() for n,p in model.original.baseline.named_parameters()}
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss=competing_nll(model(b),torch.as_tensor(d.times[selected],device='cuda'),torch.as_tensor(d.events[selected],device='cuda'))
            loss.backward();opt.step()
        if any(p.grad is not None or not torch.equal(p,frozen[n]) for n,p in model.original.baseline.named_parameters()):
            raise ValueError('H34 changed/received gradient')
        changed={group:any(not torch.equal(p,before[n]) for n,p in model.named_parameters() if n in before and n.startswith(group))
                 for group in ('original.encoder.base.lab.','original.encoder.base.text_projection.',
                               'original.delta_head.','lab_ids.','temporal.','history_head.')}
        required=['original.encoder.base.lab.','original.encoder.base.text_projection.','original.delta_head.']
        if model.unified:required.append('lab_ids.')
        if model.use_history:required+=['temporal.','history_head.']
        if not all(changed[g] for g in required):raise ValueError('effective shared/history layer not updated: '+repr(changed))
        sensitivity=0.;past_gradient=0.
        if model.use_history:
            model.eval();original=model(b).log_amplitude.detach()
            current=b[5];past_only=torch.ones(len(b[4]),dtype=torch.bool,device='cuda');past_only[current]=False
            labs=b[2].detach().clone().requires_grad_(True);legacy=b[1].detach().clone().requires_grad_(True)
            active=list(b);active[2]=labs;active[1]=legacy
            model.zero_grad(set_to_none=True)
            model(tuple(active)).log_amplitude.sum().backward()
            # 共用原核心检验也可读：统一条件用新lab梯度，原条件验证旧追加检验梯度。
            gradient=labs.grad if model.unified else legacy.grad
            if gradient is not None:past_gradient=float(gradient[past_only].abs().sum())
            without=list(b);without[7]=torch.zeros_like(b[7])
            sensitivity=float((model(tuple(without)).log_amplitude-original).abs().max().detach())
            if past_gradient<=0 or sensitivity<=0:raise ValueError('history content not differentiably consumed')
        updates[arm]=dict(actual_updates=changed,history_input_gradient_l1=past_gradient,
                          history_removal_risk_parameter_difference=sensitivity,h34_frozen=True)
        del model;torch.cuda.empty_cache()
    return dict(common_initial_risk_parameter_error=0.,conditions=updates)


def fit(fresh,arm,tr,va,d,tune,folder):
    folder.mkdir();model=LongitudinalModel(fresh,arm).cuda();torch.manual_seed(42);rng=np.random.default_rng(42)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4)
    early=EarlyStopping(7,10,1e-4);best=(-np.inf,np.inf);rows_=[];reason='max_epochs';steps=0;started=time.perf_counter()
    frozen={n:p.detach().clone() for n,p in model.original.baseline.named_parameters()}
    for epoch in range(MAX_EPOCHS+1):
        train_nll=None
        if epoch:
            model.train();total=0.;count=0
            for j in rng.permutation(len(tr)):
                p,b=tr[j];optimizer.zero_grad(set_to_none=True)
                out=model(b);nll=competing_nll(out,torch.as_tensor(d.times[p],device='cuda'),torch.as_tensor(d.events[p],device='cuda'))
                loss=nll+1e-4*out.delta.square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                if not torch.isfinite(norm):raise ValueError('nonfinite gradients')
                optimizer.step();steps+=1;total+=float(nll.detach())*len(p);count+=len(p)
            train_nll=total/count
        a,k,curve,nll=predict(model,va,d);m=detail(d,tune,curve,nll)
        f1=m['annual']['f1'];ibs=m['readmission']['integrated_brier_7_365']
        candidate=f1>best[0]+1e-12 or (abs(f1-best[0])<=1e-12 and ibs<best[1]-1e-12)
        _,stop=early.observe(epoch,-f1)
        rows_.append(dict(epoch=epoch,train_nll=train_nll,annual=m['annual'],tuning_ibs=ibs,
                          tuning_death_ibs=m['death']['integrated_brier_7_365']))
        if candidate:
            best=(f1,ibs)
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,threshold=m['annual']['threshold']),folder/'best.pt')
        write_json(folder/'history.json',rows_)
        if epoch%5==0:print(json.dumps(dict(arm=arm,epoch=epoch,f1=f1,threshold=m['annual']['threshold'],ibs=ibs,seconds=round(time.perf_counter()-started,1))),flush=True)
        if stop:reason='patience_exhausted';break
    if any(not torch.equal(p,frozen[n]) for n,p in model.original.baseline.named_parameters()):raise ValueError('H34 changed')
    torch.save(dict(state_dict=model.state_dict(),epoch=epoch),folder/'last.pt')
    chosen=torch.load(folder/'best.pt',map_location='cuda',weights_only=True);model.load_state_dict(chosen['state_dict'])
    a,k,curve,nll=predict(model,va,d);m=detail(d,tune,curve,nll,chosen['threshold'])
    save(folder/'tuning_predictions.npz',d,tune,a,k,curve)
    info=dict(best_epoch=chosen['epoch'],completed_epochs=epoch,stop_reason=reason,early=asdict(early),optimizer_steps=steps,
              checkpoint_sha256=sha256(folder/'best.pt'),last_sha256=sha256(folder/'last.pt'),metrics=m,
              trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),seconds=time.perf_counter()-started)
    write_json(folder/'result.json',info);return info


def train():
    if ROOT.exists():raise FileExistsError('finite batch exists; do not rerun')
    data=setup();d,c,ext,lab,relation,train,tune,confirm,fresh,teacher,contract,source_report=data
    ROOT.mkdir();write_json(ROOT/'contract.json',contract);write_json(ROOT/'source_report.json',source_report)
    write_json(ROOT/'state.json',dict(status='preflight',automation_resumed=False))
    write_json(ROOT/'preflight.json',preflight(fresh,relation,d,c,ext,lab,train))
    all_caches={flag:(cache(c,ext,lab,relation,train,flag),cache(c,ext,lab,relation,tune,flag)) for flag in (False,True)}
    a,k,curve,nll=original_predict(teacher,d,c,ext,tune)
    with np.load('training_runs/v6_regimen_joint_fusion_2026-10-14/original/tune_predictions.npz') as z:
        np.testing.assert_array_equal(z['hadm_ids'],d.arrays['hadm_ids'][tune])
        np.testing.assert_array_equal(z['curve'],curve,err_msg='original actual numeric path must restore saved curve')
    reference=detail(d,tune,curve,nll);(ROOT/'original').mkdir();save(ROOT/'original/tuning_predictions.npz',d,tune,a,k,curve)
    write_json(ROOT/'original/result.json',dict(metrics=reference,newly_trained=False))
    print(json.dumps(dict(phase='cached_complete_inputs',source_report=source_report,
                         cuda_allocated_gb=round(torch.cuda.memory_allocated()/1e9,3),original_annual=reference['annual'])),flush=True)
    results={}
    for arm in ARMS:
        write_json(ROOT/'state.json',dict(status='training',arm=arm,automation_resumed=False))
        tr,va=all_caches[arm.startswith('history')]
        results[arm]=fit(fresh,arm,tr,va,d,tune,ROOT/arm)
    scores={arm:row['metrics'] for arm,row in results.items()};scores['original']=reference
    selected=sorted(scores,key=lambda n:(-scores[n]['annual']['f1'],scores[n]['readmission']['integrated_brier_7_365'],n))[0]
    nominated=selected if selected not in ('original','current_original') and scores[selected]['annual']['f1']>max(scores[n]['annual']['f1'] for n in ('original','current_original')) else None
    lock=dict(selected=selected,nominated=nominated,thresholds={n:m['annual']['threshold'] for n,m in scores.items()},
              rule='annual tuning F1; then IBS; no confirmation reselection',default_changed=False,goal_achieved=False)
    write_json(ROOT/'nomination_lock.json',lock);write_json(ROOT/'report.json',dict(reference=reference,results=results,lock=lock))
    write_json(ROOT/'state.json',dict(status='trained_pending_independent_replay',automation_resumed=False))
    print(json.dumps(lock),flush=True)


def verify():
    data=setup();d,c,ext,lab,relation,train,tune,confirm,fresh,teacher,contract,source_report=data
    if contract!=json.loads((ROOT/'contract.json').read_text()) or source_report!=json.loads((ROOT/'source_report.json').read_text()):raise ValueError('new process source/input contract differs')
    report=json.loads((ROOT/'report.json').read_text());errors={}
    for arm in ('original',*ARMS):
        va=cache(c,ext,lab,relation,tune,arm.startswith('history'))
        if arm=='original':model=None;expected=report['reference']
        else:
            model=LongitudinalModel(fresh,arm).cuda();chosen=torch.load(ROOT/arm/'best.pt',map_location='cuda',weights_only=True)
            info=report['results'][arm]
            if chosen['epoch']!=info['best_epoch'] or sha256(ROOT/arm/'best.pt')!=info['checkpoint_sha256']:raise ValueError('checkpoint differs')
            model.load_state_dict(chosen['state_dict']);expected=info['metrics']
            history=json.loads((ROOT/arm/'history.json').read_text());early=EarlyStopping(7,10,1e-4)
            for r in history:early.observe(r['epoch'],-r['annual']['f1'])
            if asdict(early)!=info['early']:raise ValueError('F1 early stop differs')
            selected=min(history,key=lambda r:(-r['annual']['f1'],r['tuning_ibs'],r['epoch']))
            if selected['epoch']!=chosen['epoch'] or selected['annual']['threshold']!=chosen['threshold']:raise ValueError('F1/IBS selection differs')
        a,k,curve,nll=original_predict(teacher,d,c,ext,tune) if arm=='original' else predict(model,va,d)
        with np.load(ROOT/arm/'tuning_predictions.npz') as z:
            for key,values in (('hadm_ids',d.arrays['hadm_ids'][tune]),('subject_ids',d.arrays['subject_ids'][tune]),('times',d.times[tune]),('events',d.events[tune])):np.testing.assert_array_equal(z[key],values)
            err=max(float(np.abs(z['curve']-curve).max()),float(np.abs(z['log_amplitude']-a).max()),float(np.abs(z['shape']-k).max()))
        if err>1e-6:raise ValueError('risk replay differs')
        actual=detail(d,tune,curve,nll,expected['annual']['threshold'])
        if actual['annual']!=expected['annual']:raise ValueError('locked threshold decision replay differs')
        for cause in ('readmission','death'):
            if abs(actual[cause]['integrated_brier_7_365']-expected[cause]['integrated_brier_7_365'])>1e-8:raise ValueError('IBS replay differs')
        errors[arm]=err;del model,va;torch.cuda.empty_cache()
    write_json(ROOT/'verification.json',dict(status='independent_inputs_weights_daily_probability_F1_early_replay',errors=errors,maximum_error=max(errors.values()),report_sha256=sha256(ROOT/'report.json')))
    write_json(ROOT/'state.json',dict(status='single_seed_independently_verified',automation_resumed=False))
    print(json.dumps(errors),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=('train','verify'))
    try: train() if p.parse_args().mode=='train' else verify()
    except Exception as e:
        if ROOT.exists():write_json(ROOT/'failure.json',dict(type=type(e).__name__,message=str(e)))
        raise
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/unified_longitudinal_report.py · lines 1–70

<!-- REVIEW_FRAGMENT {"path":"training/v6/unified_longitudinal_report.py","part":1,"start_line":1,"end_line":70,"source_sha256":"9f3b7f11277ea2ce56352a327824eeecdbcf1193e0a495d32a4e4a028f5ec9bf","fragment_sha256":"9f3b7f11277ea2ce56352a327824eeecdbcf1193e0a495d32a4e4a028f5ec9bf","added_terminal_newline":false} -->
````python
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
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/verify_audited_experiment.py · lines 1–70

<!-- REVIEW_FRAGMENT {"path":"training/v6/verify_audited_experiment.py","part":1,"start_line":1,"end_line":70,"source_sha256":"ff5bc7f34b1cce0e553d2d419574a86f0f76f1519c272fd3a114911964487e59","fragment_sha256":"ff5bc7f34b1cce0e553d2d419574a86f0f76f1519c272fd3a114911964487e59","added_terminal_newline":false} -->
````python
"""新进程独立核验哈希/选轮/冻结及逐住院每日预测；不重训。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.audited_experiment import ARMS,INPUT,OUTPUT,load,prepare,restrict,new_model
from training.v6.horizon_objective import restore
from training.v6.token_experiment import parameters
from training.v6.token_model import fast_cif,daily_probabilities
from training.v6.train import write_json


def verify(root):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.use_deterministic_algorithms(True)
    report=json.loads((root/'report.json').read_text());contract=report['contract']
    if report['status']!='completed_neural_and_simple_controls':raise ValueError('incomplete run')
    for name,digest in contract['source_sha256'].items():
        if sha256(Path(name))!=digest:raise ValueError('training source changed')
    if sha256(INPUT/'audit.json')!=contract['input_audit_sha256']:raise ValueError('audit changed')
    cfg,d,c,h,a,audit=load()
    if d.provenance!=contract['data']:raise ValueError('data lineage changed')
    snapshot=json.loads((cfg.snapshot/'metadata/manifest.json').read_text())
    raw_verified={}
    for name,item in snapshot['sources'].items():
        digest=sha256(cfg.snapshot/(name+'.csv.gz'))
        if digest!=item['sha256']:raise ValueError('raw snapshot changed')
        raw_verified[name]=digest
    idx={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('validation','test')}
    cache={f:prepare(d,c,a,idx[f]) for f in idx};compact=report['selected_arm']!='full_pool';errors={};metric_errors={}
    for seed in (42,7):
        if sha256(cfg.parent/f'seed{seed}/survival_base/best.pt')!=contract['frozen_baseline_sha256'][str(seed)]:raise ValueError('base file changed')
        base=restore(cfg.parent,seed,'survival_base',h,.1)
        for arm in (*ARMS,'flat'):
            path=root/f'seed{seed}'/arm;info=report['models'][str(seed)][arm]
            if sha256(path/'best.pt')!=info['checkpoint_sha256']:raise ValueError('checkpoint changed')
            history=json.loads((path/'history.json').read_text());state=torch.load(path/'best.pt',map_location='cuda',weights_only=True)
            if len(history)!=contract['epochs']+1 or min(history,key=lambda r:r['validation_ibs'])['epoch']!=state['epoch']:raise ValueError('selection wrong')
            model=new_model(base,seed,arm,a);model.load_state_dict(state['state_dict'])
            if not all(torch.equal(value,model.baseline.state_dict()[name]) for name,value in base.state_dict().items()):raise ValueError('frozen changed')
            for fold in idx:
                aa,kk,nll=parameters(model,restrict(cache[fold],a,compact if arm=='flat' else arm!='full_pool'))
                curve=fast_cif(aa,kk,device='cuda');daily,conditional=daily_probabilities(curve)
                with np.load(path/(fold+'_predictions.npz')) as z:previous={k:z[k] for k in z.files}
                if not np.array_equal(previous['hadm_ids'],d.arrays['hadm_ids'][idx[fold]]):raise ValueError('prediction members wrong')
                err=float(max(np.abs(curve-previous['curve']).max(),np.abs(daily-previous['daily_event_probability']).max(),
                  np.abs(conditional-previous['conditional_daily_event_probability']).max()))
                if err>1e-9:raise ValueError('new process predictions differ')
                errors[f'{seed}/{arm}/{fold}']=err
                from training.v6.evaluation import metrics
                detail=metrics(d.times[idx[fold]],d.events[idx[fold]],curve[:,:,1])
                delta=abs(detail['integrated_brier_7_365']-info['folds'][fold]['readmission']['integrated_brier_7_365'])
                if delta>1e-12:raise ValueError('metrics differ')
                metric_errors[f'{seed}/{arm}/{fold}']=delta
            del model
    simple=json.loads((root/'simple_controls/report.json').read_text())
    if simple['status']!='completed_simple_controls' or max(simple['reload_errors'].values())>1e-9:raise ValueError('simple verification incomplete')
    if sha256(Path('training/v6/audited_simple_controls.py'))!=simple['source_sha256']:raise ValueError('simple source changed')
    if sha256(Path('training/v6/finalize_audited_controls.py'))!=simple['completion_code_sha256']:raise ValueError('recovery source changed')
    write_json(root/'independent_verification.json',dict(status='new_process_exact_gpu_prediction_replay',report_sha256=sha256(root/'report.json'),
      simple_report_sha256=sha256(root/'simple_controls/report.json'),completed_neural_models=10,verified_folds=20,reload_errors=errors,
      metric_errors=metric_errors,maximum_probability_error=max(errors.values()),frozen_baseline_unchanged=True,raw_snapshot_sha256=raw_verified,
      limits=['Neural new process replay verified; simple model save/load checked by training runner.',
              'Engineering success is not clinical validation or a blind test.']))
    print(json.dumps(dict(status='independently_verified',models=10,folds=20,max_error=max(errors.values()))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=OUTPUT);verify(p.parse_args().root)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/verify_detail_extension.py · lines 1–57

<!-- REVIEW_FRAGMENT {"path":"training/v6/verify_detail_extension.py","part":1,"start_line":1,"end_line":57,"source_sha256":"e25b592a5c1959dd6f14952d64ec573b47bc4fde1889c08bb3f5fa0245ff0891","fragment_sha256":"e25b592a5c1959dd6f14952d64ec573b47bc4fde1889c08bb3f5fa0245ff0891","added_terminal_newline":false} -->
````python
"""训练结束后的独立GPU重载验收；不修改预测/检查点。"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.detail_extension_experiment import load_inputs,model_for,prepared,ARMS
from training.v6.horizon_objective import predict90,binary_metrics
from training.v6.risk import predict_cif


def verify(root):
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    report=json.loads((root/'report.json').read_text()); contract=report['contract']
    for p,digest in contract['code_sha256'].items():
        if sha256(Path(p))!=digest: raise ValueError('source changed: '+p)
    if sha256(Path('docs/V6_DETAIL_EXTENSION_PROTOCOL_2026-10-11.md'))!=contract['protocol_sha256']:
        raise ValueError('protocol changed')
    config=json.loads((root/'config.json').read_text()); args=SimpleNamespace(**{k:Path(v) for k,v in config.items()})
    d,c,hist,parent,a,_=load_inputs(args)
    if sha256(args.extension/'manifest.json')!=contract['extension_manifest_sha256']: raise ValueError('extension changed')
    evidence={}
    for seed in (42,7):
        for arm in ARMS:
            path=root/f'seed{seed}'/arm/'best.pt'
            if sha256(path)!=report['models'][str(seed)][arm]['checkpoint_sha256']: raise ValueError('checkpoint changed')
            m=model_for(args,seed,hist)
            m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
            for fold in ('validation','test'):
                idx=np.flatnonzero(d.arrays['patient_splits']==fold)
                with np.load(root/f'seed{seed}'/arm/(fold+'_predictions.npz')) as z: saved={k:z[k] for k in z.files}
                for k,actual in (('hadm_ids',d.arrays['hadm_ids'][idx]),('subject_ids',d.arrays['subject_ids'][idx]),
                                 ('times',d.times[idx]),('events',d.events[idx])):
                    if not np.array_equal(saved[k],actual): raise ValueError('saved population/labels differ')
                replay=predict90(m,prepared(d,c,a,idx,arm)); err=float(np.abs(saved['risk90']-replay['risk90']).max())
                if err>1e-7: raise ValueError('risk reload differs')
                mm=binary_metrics(saved['y90'],replay['risk90'])
                expected=report['models'][str(seed)][arm]['folds'][fold]
                errors={k:abs(mm[k]-expected[k]) for k in ('auc90','brier90','average_precision','logloss90')}
                if max(errors.values())>1e-7: raise ValueError('metrics differ')
                fine=predict_cif(replay['parameters'][:64],replay['shape'][:64],np.arange(366.),step_days=.125)
                curve_error=float(np.abs(fine-saved['curve'][:64]).max())
                if curve_error>2e-4 or not np.allclose(saved['curve'].sum(2),1,atol=1e-9): raise ValueError('integration failed')
                evidence[f'{seed}/{arm}/{fold}']=dict(risk_max_error=err,metric_errors=errors,fine_curve_error=curve_error)
            del m
    result=dict(status='independently_verified_20_folds',report_sha256=sha256(root/'report.json'),
        verifier_sha256=sha256(Path(__file__)),device=torch.cuda.get_device_name(0),folds=evidence)
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result['status'],maximum_risk_error=max(x['risk_max_error'] for x in evidence.values()))))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    verify(p.parse_args().output)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/verify_group_run.py · lines 1–72

<!-- REVIEW_FRAGMENT {"path":"training/v6/verify_group_run.py","part":1,"start_line":1,"end_line":72,"source_sha256":"a5b2ce6dc7b93ea50733c4e08d99893f0b799fb7527051f3b3a40452672b8de8","fragment_sha256":"a5b2ce6dc7b93ea50733c4e08d99893f0b799fb7527051f3b3a40452672b8de8","added_terminal_newline":false} -->
````python
"""组级实验独立GPU重载验收，恢复正确的屏蔽和解冻配方。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.data import load
from training.v6.model import HistoryBaseline
from training.v6.group_ablation import ARMS, build_model, prepared
from training.v6.train import predict, write_json
from training.v6.risk import predict_cif
from training.v6.evaluation import GRID


def verify(root):
    report=json.loads((root/'report.json').read_text(encoding='utf-8'))
    contract=report['contract']
    if report['status']!='completed_matched_group_ablation_single_seed':
        raise ValueError('incomplete run')
    for name,digest in contract['code_sha256'].items():
        if sha256(Path(__file__).parent/name)!=digest:
            raise ValueError('group experiment code changed')
    reference=Path(contract['reference_run'])
    if sha256(reference/'report.json')!=contract['reference_report_sha256']:
        raise ValueError('reference changed')
    config=json.loads((reference/'config.json').read_text(encoding='utf-8'))
    data=load(Path(config['snapshot']),Path(config['state_dir']),Path(config['labels']),
              field_dir=Path(config['field_dir']),vectors_path=Path(config['vectors_path']))
    if data.provenance!=contract['data_provenance']:
        raise ValueError('data changed')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    base=HistoryBaseline().cuda()
    checkpoint=reference/'history/best.pt'
    if sha256(checkpoint)!=contract['baseline_checkpoint_sha256']:
        raise ValueError('baseline changed')
    base.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['state_dict'])
    checks={}
    for arm in ARMS:
        model=build_model(base,data.text_vectors.shape[1],arm,contract['seed'])
        cp=root/arm/'best.pt'
        if sha256(cp)!=report['models'][arm]['checkpoint_sha256']:
            raise ValueError('checkpoint changed')
        model.load_state_dict(torch.load(cp,map_location='cuda',weights_only=True)['state_dict'])
        checks[arm]={}
        for fold in ('validation','test'):
            ix=np.flatnonzero(data.arrays['patient_splits']==fold)
            batches=prepared(data,ix,contract['batch_size'],arm)
            a,k,_,_=predict(model,batches)
            curve=predict_cif(a,k,GRID,step_days=.25)
            with np.load(root/arm/(fold+'_predictions.npz')) as archive:
                ids=archive['hadm_ids']; saved=archive['curve']
            error=float(np.abs(curve-saved).max())
            if not np.array_equal(ids,data.arrays['hadm_ids'][ix]) or error>5e-6:
                raise ValueError('group prediction reproduction failed')
            if not np.allclose(curve.sum(2),1,atol=1e-10) or np.any(np.diff(curve[:,:,0],axis=1)>1e-10) or np.any(np.diff(curve[:,:,1:],axis=1)<-1e-10):
                raise ValueError('invalid probability curve')
            checks[arm][fold]={'admissions':len(ix),'maximum_curve_error':error,'probability_checks_passed':True}
    write_json(root/'verification.json',{'status':'independent_group_gpu_predictions_verified','checks':checks,
                                         'optimizer_steps':0,'verifier_sha256':sha256(Path(__file__))})
    print(json.dumps({'status':'independent_group_gpu_predictions_verified'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    if (args.run/'verification.json').exists(): raise FileExistsError(args.run/'verification.json')
    verify(args.run)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/verify_original_expansion.py · lines 1–93

<!-- REVIEW_FRAGMENT {"path":"training/v6/verify_original_expansion.py","part":1,"start_line":1,"end_line":93,"source_sha256":"2d9be01e25b89bf2978daa431a18e70efa3e4d29144adae6e353f43a5dcb9e96","fragment_sha256":"2d9be01e25b89bf2978daa431a18e70efa3e4d29144adae6e353f43a5dcb9e96","added_terminal_newline":false} -->
````python
"""新进程读取已保存完整结果，独立核对训练选择/冻结/每日风险重放。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.expanded_original_experiment import parser,checked,prepare,new_model,ARMS
from training.v6.horizon_objective import restore
from training.v6.token_experiment import parameters
from training.v6.token_model import fast_cif,daily_probabilities
from training.v6.risk import predict_cif
from training.v6.train import write_json


def verify(root):
    cfg=json.loads((root/'config.json').read_text());r=json.loads((root/'report.json').read_text());contract=r['contract']
    for name,digest in contract['code_sha256'].items():
        if sha256(Path(name))!=digest:raise ValueError('training code changed')
    args=argparse.Namespace(**{k:Path(v) for k,v in cfg.items()});d,c,h,ext,lab,med,prov,active,norm=checked(args)
    if d.provenance!=contract['data_provenance'] or prov!=contract['new_sources'] or active!=contract['active']:raise ValueError('lineage drift')
    ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('validation','test')}
    cache={f:prepare(d,c,ext,lab,med,ix[f]) for f in ix};errors={};reference_error=0.
    for seed in (42,7):
        base=restore(args.parent,seed,'survival_base',h,.1)
        for arm in ARMS:
            path=root/f'seed{seed}'/arm;history=json.loads((path/'history.json').read_text());cp=torch.load(path/'best.pt',map_location='cuda',weights_only=True)
            if len(history)!=101 or min(history,key=lambda x:x['validation_ibs'])['epoch']!=cp['epoch']:raise ValueError('selection/epoch')
            if sha256(path/'best.pt')!=r['models'][str(seed)][arm]['checkpoint_sha256']:raise ValueError('checkpoint changed')
            m=new_model(base,seed,arm);m.load_state_dict(cp['state_dict'])
            if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):raise ValueError('frozen changed')
            for fold in ix:
                a,k,nll=parameters(m,cache[fold]);curve=fast_cif(a,k,device='cuda');daily,conditional=daily_probabilities(curve)
                with np.load(path/(fold+'_predictions.npz')) as z:
                    for name in ('hadm_ids','subject_ids'):
                        if not np.array_equal(z[name],d.arrays[name][ix[fold]]):raise ValueError('saved member differs')
                    if not np.array_equal(z['events'],d.events[ix[fold]]) or not np.array_equal(z['times'],d.times[ix[fold]]):raise ValueError('labels differ')
                    err=float(np.abs(curve-z['curve']).max())
                    if err>3e-7 or np.abs(daily-z['daily_event_probability']).max()>3e-7:raise ValueError('GPU replay differs')
                    if np.any(np.diff(curve[:,:,1:],axis=1)<-1e-10) or np.abs(curve.sum(2)-1).max()>1e-10:raise ValueError('probability')
                    if not np.array_equal(z['conditional_daily_valid'],curve[:,:-1,0]>1e-12):raise ValueError('conditional validity')
                    errors[f'{seed}/{arm}/{fold}']=err
                reference_error=max(reference_error,float(np.abs(predict_cif(a[:5],k[:5],np.arange(366))-curve[:5]).max()))
            del m
    old=args.medication/'medication_identity.npz';new=args.expanded/'medications/medication_identity.npz'
    with np.load(old) as z:oldraw=z['values_raw'];oldpresent=z['present'];names=z['drug_names']
    with np.load(new) as z:newraw=z['values_raw'][:,:21];newpresent=z['present'][:,:21]
    corrections={str(name):dict(presence_changed_admissions=int((oldpresent[:,j]!=newpresent[:,j]).sum()),
          raw_changed_admissions=int(np.any(oldraw[:,j]!=newraw[:,j],axis=1).sum())) for j,name in enumerate(names)
          if np.any(oldraw[:,j]!=newraw[:,j]) or np.any(oldpresent[:,j]!=newpresent[:,j])}
    width_errors={};width=root/'molecule_width_control'
    if (width/'report.json').exists():
        protocol=json.loads((width/'protocol.json').read_text());width_report=json.loads((width/'report.json').read_text())
        if protocol['parent_report_sha256']!=sha256(root/'report.json') or protocol['code_sha256']!=sha256(Path('training/v6/medication_width_control.py')):
            raise ValueError('width control provenance changed')
        narrow=dict(med);narrow['features']=med['features'].copy();narrow['features'][:,21:]=0
        narrow['present']=med['present'].copy();narrow['present'][:,21:]=False
        small={f:prepare(d,c,ext,lab,narrow,ix[f]) for f in ix}
        for seed in (42,7):
            base=restore(args.parent,seed,'survival_base',h,.1);p=width/f'seed{seed}/existing21'
            cp=torch.load(p/'best.pt',map_location='cuda',weights_only=True);history=json.loads((p/'history.json').read_text())
            if len(history)!=101 or min(history,key=lambda x:x['validation_ibs'])['epoch']!=cp['epoch']:raise ValueError('width selection')
            if sha256(p/'best.pt')!=width_report['models'][str(seed)]['checkpoint_sha256']:raise ValueError('width checkpoint')
            m=new_model(base,seed,'more_meds');m.load_state_dict(cp['state_dict'])
            if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):raise ValueError('width frozen')
            for fold in ix:
                a,k,_=parameters(m,small[fold]);curve=fast_cif(a,k,device='cuda')
                with np.load(p/(fold+'_predictions.npz')) as z:
                    if not np.array_equal(z['hadm_ids'],d.arrays['hadm_ids'][ix[fold]]):raise ValueError('width member')
                    error=float(np.abs(curve-z['curve']).max())
                    if error>3e-7:raise ValueError('width fresh-process replay')
                    width_errors[f'{seed}/{fold}']=error
            del m
    proof=dict(status='independently_verified_original_expansion',report_sha256=sha256(root/'report.json'),
        code_sha256=sha256(Path(__file__)),fresh_process_reload_errors=errors,max_reference_integral_error=reference_error,
        train_only_scaling_errors=norm,drug_mapping_corrections=corrections,models=12,folds=24,
        medication_width_control_reload_errors=width_errors,total_models=12+len(width_errors)//2,total_folds=24+len(width_errors),
        limits=['Clinical mapping has not been physician-adjudicated in full.', 'Reused test, not blind.'])
    write_json(root/'fresh_process_verification.json',proof)
    selected=r['validation_selected_arm']
    with np.load(root/f'seed42/{selected}/test_predictions.npz') as z:
        curve=z['curve'][0];daily=z['daily_event_probability'][0];conditional=z['conditional_daily_event_probability'][0];valid=z['conditional_daily_valid'][0]
    import csv
    with (root/'daily_selected_first_test_record.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['day','survival','readmission_cif','death_cif','readmission_daily','death_daily','readmission_given_event_free','death_given_event_free'])
        writer.writerow([0,*curve[0],0,0,0,0])
        for day in range(1,366):writer.writerow([day,*curve[day],*daily[day-1],*(conditional[day-1] if valid[day-1] else ('',''))])
    print(json.dumps(dict(status=proof['status'],max_replay_error=max(errors.values()),mapping_corrections=corrections)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_original_interaction_2026-10-12'))
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.use_deterministic_algorithms(True);verify(a.root)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/verify_progress_artifacts.py · lines 1–62

<!-- REVIEW_FRAGMENT {"path":"training/v6/verify_progress_artifacts.py","part":1,"start_line":1,"end_line":62,"source_sha256":"763a75af1c94c209e4ea7ad0fbceec1da21bd7ad645c0861ca5102fcd6153eaa","fragment_sha256":"763a75af1c94c209e4ea7ad0fbceec1da21bd7ad645c0861ca5102fcd6153eaa","added_terminal_newline":false} -->
````python
"""本轮两项试验的全曲线/排序/概率评价复算，不重训、不改报告。"""
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.evaluation import metrics,public_metrics
from training.v6.horizon_objective import labels90,binary_metrics
from training.v6.train import write_json


def difference(a,b):
    if isinstance(a,dict):
        if a.keys()!=b.keys(): raise ValueError('metric keys differ')
        return max((difference(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,list):
        if len(a)!=len(b): raise ValueError('metric lengths differ')
        return max((difference(x,y) for x,y in zip(a,b,strict=True)),default=0.)
    if a is None or isinstance(a,(str,bool)):
        if a!=b: raise ValueError('metric metadata differs')
        return 0.
    return abs(a-b)


def verify():
    result={}
    for name in ('v6_saved_section_comparison_2026-10-11','v6_frozen_augmentation_2026-10-11'):
        root=Path('training_runs')/name
        report=json.loads((root/'report.json').read_text())
        proof=json.loads((root/'independent_verification.json').read_text())
        if proof['report_sha256']!=sha256(root/'report.json'): raise ValueError('stale independent proof')
        if 'selection' in report and report['selection']!=json.loads((root/'selection.json').read_text()):
            raise ValueError('selection record changed')
        folds=0; max_error=0.; max_mass=0.
        for seed,models in report['models'].items():
            for arm,model in models.items():
                for f,recorded in model['folds'].items():
                    with np.load(root/f'seed{seed}'/arm/(f+'_predictions.npz')) as z:
                        y=labels90(z['times'],z['events']); curve=z['curve']
                        if not np.array_equal(z['y90'],y): raise ValueError('strict label encoding differs')
                        if (not np.isfinite(curve).all() or (curve<-1e-10).any() or (curve>1+1e-10).any()
                            or (np.diff(curve[:,:,1:],axis=1)<-1e-10).any() or (np.diff(curve[:,:,0],axis=1)>1e-10).any()
                            or not np.array_equal(curve[:,0],np.tile([1.,0.,0.],(len(curve),1)))
                            or not np.allclose(curve[:,90,1],z['risk90'],atol=1e-8)):
                            raise ValueError('curve boundary/monotonicity failed')
                        max_mass=max(max_mass,float(np.abs(curve.sum(2)-1).max()))
                        actual=binary_metrics(y,z['risk90'])
                        for k,v in actual.items(): max_error=max(max_error,difference(v,recorded[k]))
                        for key,cause in (('survival_readmission',1),('survival_death',2)):
                            actual=public_metrics(metrics(z['times'],z['events'],curve[:,:,cause],cause=cause,ordering=True))
                            max_error=max(max_error,difference(actual,recorded[key]))
                        folds+=1
        if folds!=16 or max_error>1e-7 or max_mass>1e-9: raise ValueError('full report recomputation failed')
        result[name]=dict(folds=folds,max_metric_error=max_error,max_probability_sum_error=max_mass,
            report_sha256=sha256(root/'report.json'),proof_sha256=sha256(root/'independent_verification.json'))
    output=Path('training_runs/v6_clinical_increment_evidence_2026-10-11/progress_artifact_verification.json')
    if output.exists(): raise FileExistsError(output)
    write_json(output,dict(status='all32_new_trial_folds_complete_metrics_and_curves_recomputed',checks=result,
        code_sha256=sha256(Path(__file__))))
    print(json.dumps(result),flush=True)


if __name__=='__main__': verify()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/verify_run.py · lines 1–78

<!-- REVIEW_FRAGMENT {"path":"training/v6/verify_run.py","part":1,"start_line":1,"end_line":78,"source_sha256":"e652aeead60f991e820459135b040d73d70efd0bbeb736ea88059778f2f0a318","fragment_sha256":"e652aeead60f991e820459135b040d73d70efd0bbeb736ea88059778f2f0a318","added_terminal_newline":false} -->
````python
"""独立重载四臂、重新预测全验证/历史测试折；不优化参数或选择模型。"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.data import load
from training.v6.model import HistoryBaseline, ConstantCorrection, AnchoredModel
from training.v6.train import ARMS, make_batches, predict, write_json
from training.v6.risk import predict_cif
from training.v6.evaluation import GRID


def verify(root: Path):
    report = json.loads((root / 'report.json').read_text(encoding='utf-8'))
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    if report['status'] != 'completed_single_seed_exploratory':
        raise ValueError('incomplete run')
    for name, digest in report['contract']['code_sha256'].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError(f'training code changed: {name}')
    data = load(Path(config['snapshot']), Path(config['state_dir']), Path(config['labels']),
                field_dir=Path(config['field_dir']), vectors_path=Path(config['vectors_path']))
    if data.provenance != report['contract']['data_provenance']:
        raise ValueError('source provenance changed')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    base = HistoryBaseline().cuda()
    base.load_state_dict(torch.load(root / 'history/best.pt', map_location='cuda', weights_only=True)['state_dict'])
    checks = {}
    for arm in ARMS:
        checkpoint = root / arm / 'best.pt'
        if sha256(checkpoint) != report['models'][arm]['checkpoint_sha256']:
            raise ValueError('checkpoint hash mismatch')
        model = (copy.deepcopy(base) if arm == 'history' else ConstantCorrection(copy.deepcopy(base)) if arm == 'constant'
                 else AnchoredModel(copy.deepcopy(base), data.text_vectors.shape[1])).cuda()
        model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True)['state_dict'])
        if hasattr(model, 'baseline') and not all(torch.equal(v, base.state_dict()[key]) for key, v in model.baseline.state_dict().items()):
            raise ValueError('arm has different frozen baseline')
        arm_checks = {'parameters_total': sum(p.numel() for p in model.parameters()),
                      'parameters_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'folds': {}}
        for fold in ('validation', 'test'):
            ix = np.flatnonzero(data.arrays['patient_splits'] == fold)
            batches = make_batches(data, ix, config['batch_size'], 'cuda', include_text=arm == 'clinical_bert')
            a, k, _, _ = predict(model, batches)
            curve = predict_cif(a, k, GRID, step_days=.25)
            with np.load(root / arm / (fold + '_predictions.npz')) as archive:
                saved = {name: archive[name] for name in archive.files}
            if not np.array_equal(saved['hadm_ids'], data.arrays['hadm_ids'][ix]):
                raise ValueError('prediction identity changed')
            parameter_error = max(float(np.abs(a-saved['log_amplitude']).max()), float(np.abs(k-saved['shape']).max()))
            curve_error = float(np.abs(curve-saved['curve']).max())
            if parameter_error > 1e-5 or curve_error > 5e-6:
                raise ValueError(f'prediction reproduction failed: {arm}/{fold}: {parameter_error}, {curve_error}')
            if (not np.allclose(curve.sum(2), 1, atol=1e-10) or np.any(np.diff(curve[:, :, 0], axis=1) > 1e-10)
                    or np.any(np.diff(curve[:, :, 1:], axis=1) < -1e-10)):
                raise ValueError('invalid full curve')
            arm_checks['folds'][fold] = {'admissions': len(ix), 'parameter_max_difference': parameter_error,
                                         'curve_max_difference': curve_error, 'probability_and_monotonicity_passed': True}
        checks[arm] = arm_checks
    write_json(root / 'verification.json', {'status': 'independent_gpu_reload_and_full_prediction_verified',
                                           'checks': checks, 'optimizer_steps': 0,
                                           'limits': 'Engineering reproduction, not independent statistical validation'})
    print(json.dumps({'status': 'independent_gpu_reload_and_full_prediction_verified'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if (args.run / 'verification.json').exists():
        raise FileExistsError(args.run / 'verification.json')
    verify(args.run)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/verify_screen_encoding.py · lines 1–74

<!-- REVIEW_FRAGMENT {"path":"training/v6/verify_screen_encoding.py","part":1,"start_line":1,"end_line":74,"source_sha256":"263c772d59deebb6d32594d795ae12c9d2e61ea32bd27e94c5addbb1c709457a","fragment_sha256":"263c772d59deebb6d32594d795ae12c9d2e61ea32bd27e94c5addbb1c709457a","added_terminal_newline":false} -->
````python
"""新进程检查候选索引/原句跨度/分块覆盖，并GPU重算固定32行向量。"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from patient_data.export_mimiciv import sha256
from training.v6.encode_screen_candidates import ROUTES, selected_groups


def verify(screen, directory, model_dir):
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    m=json.loads((directory/'manifest.json').read_text())
    if m['status']!='candidate_bert_complete': raise ValueError('unfinished vectors')
    groups,counts,digest=selected_groups(screen)
    if digest!=m['source_manifest_sha256'] or dict(counts)!=m['counts']: raise ValueError('candidate selection differs')
    for name,value in m['output_sha256'].items():
        if sha256(directory/name)!=value: raise ValueError('encoded output changed')
    for name,value in m['model_files'].items():
        if sha256(model_dir/name)!=value: raise ValueError('encoder asset changed')
    chunks=[]; per_group=defaultdict(list)
    with gzip.open(directory/'chunks.jsonl.gz','rt',encoding='utf-8') as f:
        for j,line in enumerate(f):
            r=json.loads(line)
            if r['embedding_row']!=j or not 0<len(r['input_ids'])<=384 or r['token_start']<0 or r['token_end']<=r['token_start']:
                raise ValueError('invalid chunk indices')
            key=(r['hadm_id'],r['subject_id'],r['patient_split'],r['field'],r['note_id'])
            if key not in groups: raise ValueError('group not found')
            expected=sorted(groups[key],key=lambda x:x['start_char'])
            if r['source_units']!=[[u['start_char'],u['end_char']] for u in expected]: raise ValueError('source unit positions differ')
            per_group[key].append(j); chunks.append(r)
    if len(chunks)!=m['rows'] or set(per_group)!=set(groups): raise ValueError('source group coverage mismatch')
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    # 全部组重新tokenize，核对每段token与原句、前缀、special tokens完全一致。
    for key,indices in per_group.items():
        text='\n'.join(r['quote'] for r in sorted(groups[key],key=lambda x:x['start_char']))
        ids=tokenizer(text,add_special_tokens=False)['input_ids']
        prefix=tokenizer('Clinical topic: '+ROUTES[key[3]]+'\n',add_special_tokens=False)['input_ids']
        start=0
        for index in indices:
            r=chunks[index]
            if r['token_start']!=start or r['input_ids']!=[tokenizer.cls_token_id,*prefix,*ids[start:r['token_end']],tokenizer.sep_token_id]:
                raise ValueError('original token content/order differs')
            start=r['token_end']
        if start!=len(ids): raise ValueError('truncated or missing tail tokens')
    vectors=np.load(directory/'vectors.float16.npy',mmap_mode='r')
    norms=np.linalg.norm(vectors.astype(np.float32),axis=1)
    if not np.isfinite(vectors).all() or not np.allclose(norms,1,atol=.002): raise ValueError('vector finite/norm differs')
    chosen=np.linspace(0,len(chunks)-1,32,dtype=int)
    model=AutoModel.from_pretrained(model_dir,local_files_only=True).cuda().eval().requires_grad_(False)
    encoded=tokenizer.pad([{'input_ids':chunks[i]['input_ids']} for i in chosen],padding=True,return_tensors='pt')
    encoded={k:v.cuda() for k,v in encoded.items()}
    with torch.inference_mode():
        hidden=model(**encoded).last_hidden_state.float(); mask=encoded['attention_mask'].bool().clone()
        lengths=mask.sum(1); mask[:,0]=False; mask.scatter_(1,(lengths-1)[:,None],False)
        pooled=(hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
        pooled=torch.nn.functional.normalize(pooled,dim=1).cpu().numpy()
    error=float(np.abs(vectors[chosen].astype(np.float32)-pooled).max())
    if error>.002: raise ValueError('GPU frozen encoding reproduction failed')
    report=dict(status='all_chunks_source_tokens_verified_and_32_gpu_reencoded',chunks=len(chunks),groups=len(groups),
                manifest_sha256=sha256(directory/'manifest.json'),max_vector_error=error,
                limits=['Not expert semantic accuracy; conservative future-section screening not exhaustive leakage guarantee'])
    (directory/'independent_encoding_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('/workspace/training/model_cache/Bio_ClinicalBERT'))
    a=p.parse_args();verify(a.screen,a.directory,a.model)
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/verify_text_encoder_comparison.py · lines 1–87

<!-- REVIEW_FRAGMENT {"path":"training/v6/verify_text_encoder_comparison.py","part":1,"start_line":1,"end_line":87,"source_sha256":"9637adcbfcb4a93dc77ed97af7e8e7e6fd8e6ffcc322905c3b614e8ba937dff4","fragment_sha256":"9637adcbfcb4a93dc77ed97af7e8e7e6fd8e6ffcc322905c3b614e8ba937dff4","added_terminal_newline":false} -->
````python
"""独立重载验收；指标浮点递归容差，训练源码与原合同不改。"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.text_encoder_comparison import (ARMS, PROTOCOL, checked_inputs, new_model,
    prepared, arm_source, read_prediction)
from training.v6.horizon_objective import labels90, predict90, binary_metrics
from training.v6.risk import predict_cif
from training.v6.evaluation import GRID, metrics, public_metrics
from training.v6.train import write_json


def metric_difference(actual,expected,path='root'):
    """仅数值浮点允许1e-7；结构、整数计数、状态和文本须精确。"""
    if isinstance(actual,dict):
        if not isinstance(expected,dict) or actual.keys()!=expected.keys(): raise ValueError('metric structure: '+path)
        return max([metric_difference(v,expected[k],path+'/'+k) for k,v in actual.items()]+[0.])
    if isinstance(actual,list):
        if not isinstance(expected,list) or len(actual)!=len(expected): raise ValueError('metric list: '+path)
        return max([metric_difference(a,b,path) for a,b in zip(actual,expected,strict=True)]+[0.])
    if isinstance(actual,(float,np.floating)):
        if not isinstance(expected,(float,int)) or not np.isfinite(actual) or not np.isfinite(expected): raise ValueError('metric nonfinite: '+path)
        error=abs(float(actual)-expected)
        if error>1e-7: raise ValueError(f'metric differs {path}: {actual} != {expected}')
        return error
    if actual!=expected: raise ValueError('metric differs: '+path)
    return 0.


def verify(args):
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    report=json.loads((args.output/'report.json').read_text()); c=report['contract']
    for path,digest in c['code_sha256'].items():
        if sha256(Path(path))!=digest: raise ValueError('training code changed: '+path)
    if sha256(PROTOCOL)!=c['protocol_sha256']: raise ValueError('protocol changed')
    data,compact,hist,_,audit=checked_inputs(args)
    if audit!=c['text_audit'] or sha256(args.parent/'report.json')!=c['parent_report_sha256']: raise ValueError('asset changed')
    d=data['bert']; y=labels90(d.times,d.events); errors={}; metric_errors={}
    for seed in (42,7):
        for arm in ARMS:
            path=args.output/f'seed{seed}'/arm/'best.pt'
            if sha256(path)!=report['models'][str(seed)][arm]['checkpoint_sha256']: raise ValueError('checkpoint changed')
            m=new_model(args,seed,arm,hist,y[d.arrays['patient_splits']=='train'].mean())
            frozen=copy.deepcopy(m.baseline.state_dict())
            m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
            if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in frozen.items()): raise ValueError('baseline changed')
            source=arm_source(arm)
            for fold in ('validation','test'):
                idx=np.flatnonzero(d.arrays['patient_splits']==fold)
                p=predict90(m,prepared(data[source],compact[source],idx,512,arm))
                p['curve']=predict_cif(p['parameters'],p['shape'],GRID)
                saved=read_prediction(args.output,seed,arm,fold)
                for key,value in (('hadm_ids',d.arrays['hadm_ids'][idx]),('subject_ids',d.arrays['subject_ids'][idx]),
                                  ('times',d.times[idx]),('events',d.events[idx]),('y90',y[idx])):
                    if not np.array_equal(saved[key],value): raise ValueError('identity/label mismatch')
                for key,value in p.items():
                    if key!='loss' and not np.allclose(saved[key],value,atol=2e-6,rtol=1e-5): raise ValueError('reload mismatch: '+key)
                detail=binary_metrics(y[idx],p['risk90'])
                detail['survival_readmission']=public_metrics(metrics(d.times[idx],d.events[idx],p['curve'][:,:,1],ordering=True))
                detail['survival_death']=public_metrics(metrics(d.times[idx],d.events[idx],p['curve'][:,:,2],cause=2,ordering=True))
                key=f'{seed}/{arm}/{fold}'
                metric_errors[key]=metric_difference(detail,report['models'][str(seed)][arm]['folds'][fold])
                errors[key]=float(np.abs(saved['risk90']-p['risk90']).max())
            del m
    write_json(args.output/'independent_verification.json',dict(status='all8_models_all16_folds_gpu_reloaded',
        report_sha256=sha256(args.output/'report.json'),baseline_unchanged=True,errors=errors,metric_errors=metric_errors,
        verifier_sha256=sha256(Path(__file__)),training_source_and_contract_unchanged=True,
        original_verifier_limitation='exact dict equality rejected tiny floating-point metric differences; independent verifier uses tested1e-7 tolerance'))
    print(json.dumps(dict(status='all8_models_all16_folds_gpu_reloaded',max_risk_error=max(errors.values()),
                         max_metric_error=max(metric_errors.values()))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--compact-dir',type=Path,default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--parent',type=Path,default=Path('training_runs/v6_90day_objective_2026-10-11'))
    p.add_argument('--output',type=Path,required=True)
    verify(p.parse_args())


if __name__=='__main__': main()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/warm_residual.py · lines 1–60

<!-- REVIEW_FRAGMENT {"path":"training/v6/warm_residual.py","part":1,"start_line":1,"end_line":60,"source_sha256":"c9e8fd813db0511305ce30457ef38ba6a45a2384a5acfaee1465b434675283c0","fragment_sha256":"c9e8fd813db0511305ce30457ef38ba6a45a2384a5acfaee1465b434675283c0","added_terminal_newline":false} -->
````python
"""已训练V6原位小型修正/微调，保留原预测，不重置临床表示。"""
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from training.v6.module_surgery import SurgeryModel
from training.v6.regularization_model import Trial
from training.v6.risk import RiskOutput

REFERENCE=Path('training_runs/v6_module_surgery_2026-10-12_r3')
BASE_SEEDS={42:42,7:7,101:42,202:7,303:42}


@dataclass(frozen=True)
class WarmTrial(Trial):
    mode: str = 'amp'


TRIALS=(WarmTrial('warm_amp'),WarmTrial('warm_product_amp',mode='product'),
        WarmTrial('warm_product_shape',mode='bounded_shape'),WarmTrial('warm_finetune',mode='finetune',lr=2e-5))


class WarmAdapter(nn.Module):
    def __init__(self,teacher,mode):
        super().__init__();self.teacher=teacher.requires_grad_(False).eval();self.mode=mode
        width=128 if mode=='amp' else 384
        self.head=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,2 if mode!='bounded_shape' else 4))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    @property
    def baseline(self):return self.teacher.baseline

    def train(self,mode=True):
        super().train(mode);self.teacher.eval();return self

    def forward(self,history,batch):
        with torch.no_grad():
            prior=self.teacher(history,batch)
            clinical,_=self.teacher.encoder(batch.base)
            context=torch.tanh(self.teacher.baseline(history).log_amplitude/4.)
        x=clinical if self.mode=='amp' else torch.cat((clinical,clinical*context[:,0:1],clinical*context[:,1:2]),1)
        raw=self.head(x)
        if self.mode=='bounded_shape':
            raw=raw.reshape(-1,2,2)
            delta=torch.stack((raw[:,:,0],.25*torch.tanh(raw[:,:,1])),2)
            shape=F.softplus(self.baseline.raw_shape.detach()+prior.delta[:,:,1]+delta[:,:,1])+.1
        else:
            delta=torch.stack((raw,torch.zeros_like(raw)),2);shape=prior.shape
        return RiskOutput((prior.log_amplitude+delta[:,:,0]).clamp(-10,6),shape,delta)


def build(base,seed,trial,bins,*,text_width=768,device='cuda',teacher_state=None):
    teacher=SurgeryModel(base,seed,'control',bins,text_width=text_width)
    state=teacher_state if teacher_state is not None else torch.load(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt',map_location=device,weights_only=True)['state_dict']
    teacher.load_state_dict(state)
    if trial.mode=='finetune':model=teacher
    else:model=WarmAdapter(teacher,trial.mode)
    model.arm=trial.name;model=model.to(device);torch.manual_seed(seed)
    return model
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/warm_residual_audit.py · lines 1–127

<!-- REVIEW_FRAGMENT {"path":"training/v6/warm_residual_audit.py","part":1,"start_line":1,"end_line":127,"source_sha256":"62c7fe81098a737894b628288db3cd7eee5eeab979d4dfcbdcba020b3ea302ba","fragment_sha256":"62c7fe81098a737894b628288db3cd7eee5eeab979d4dfcbdcba020b3ea302ba","added_terminal_newline":false} -->
````python
"""第四批独立重放与一次锁定确认；教师只重载，不伪造新训练。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.warm_residual_experiment import ROOT,SEEDS
from training.v6.warm_residual import TRIALS,BASE_SEEDS,build
from training.v6.task_optimization_audit import load,check_curve,check_saved
from training.v6.module_surgery import SurgeryModel,probability_curve
from training.v6.module_surgery_experiment import batches,parameters,validation,summary,save
from training.v6.horizon_objective import restore
from training.v6.evaluation import paired_interval
from training.v6.train import write_json


def inputs():
    data=checked();contract=json.loads((ROOT/'contract.json').read_text())
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('warm source changed: '+p)
    if contract['search_contract_sha256']!=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')):
        raise ValueError('warm source data contract changed')
    for seed,digest in contract['teacher_checkpoint_sha256'].items():
        if sha256(REFERENCE/f'seed{seed}/control/best.pt')!=digest:raise ValueError('teacher checkpoint changed')
    return data,contract


def teacher(base,seed,bins):
    m=SurgeryModel(base,BASE_SEEDS[seed],'control',bins).cuda()
    m.load_state_dict(torch.load(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt',map_location='cuda',weights_only=True)['state_dict'])
    return m


def candidate(base,seed,trial,bins):
    m,info,best=load(base,seed,trial,bins,root=ROOT,model_factory=build,baseline_seeds=BASE_SEEDS)
    if trial.mode!='finetune':
        reference=teacher(base,seed,bins)
        for k,v in reference.state_dict().items():torch.testing.assert_close(v,m.teacher.state_dict()[k],atol=0,rtol=0)
        if any(p.requires_grad for p in m.teacher.parameters()):raise ValueError('teacher not frozen')
        del reference
    return m,info,best


def verify():
    (d,c,h,ext,_,bins,ix,_,search,_),contract=inputs();cache=batches(d,c,ext,ix);models={};errors={}
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1);models[str(seed)]={}
        for name in ['original_teacher',*[t.name for t in TRIALS]]:
            if name=='original_teacher':m=teacher(base,seed,bins);info=None
            else:
                trial=next(t for t in TRIALS if t.name==name);m,info,best=candidate(base,seed,trial,bins)
                score=validation(m,cache,d.times[ix],d.events[ix])
                for key,value in score.items():
                    if best['tuning_'+key]!=value:raise ValueError('warm training metric differs')
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k);check_curve(curve)
            path=ROOT/f'seed{seed}'/name/'tuning_predictions.npz'
            if path.exists():error=check_saved(path,d,ix,a,k,curve)
            else:save(path,d,ix,a,k,curve);error=None
            errors[f'{seed}/{name}']=error
            models[str(seed)][name]=dict(training=info,metrics=summary(curve,d,ix,nll));del m
    selected=json.loads((ROOT/'selected_before_confirmation.json').read_text())
    scores={name:float(np.mean([models[str(s)][name]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS]))
            for name in ['original_teacher',*[t.name for t in TRIALS]]}
    if scores!=selected['tuning_scores'] or min(scores,key=scores.get)!=selected['selected']:raise ValueError('warm selection changed')
    write_json(ROOT/'tuning_report.json',dict(models=models,tuning_scores=scores,selected=selected['selected'],goal_achieved=False))
    complete=all(x is not None for x in errors.values())
    write_json(ROOT/'tuning_verification.json',dict(status='tuning_independently_verified' if complete else 'tuning_predictions_saved_pending_second_process',
        models=len(errors),errors=errors,maximum_error=max(errors.values()) if complete else None,code_sha256=sha256(Path(__file__)),
        selection_sha256=sha256(ROOT/'selected_before_confirmation.json'),report_sha256=sha256(ROOT/'tuning_report.json'),test_not_evaluated=True))
    print(json.dumps(dict(status='tuning_independently_verified' if complete else 'tuning_predictions_saved_pending_second_process',models=len(errors))),flush=True)


def confirm():
    proof=json.loads((ROOT/'tuning_verification.json').read_text())
    if proof['status']!='tuning_independently_verified' or proof['code_sha256']!=sha256(Path(__file__)):
        raise ValueError('warm complete independent proof required')
    if proof['selection_sha256']!=sha256(ROOT/'selected_before_confirmation.json'):raise ValueError('selection proof stale')
    selected=json.loads((ROOT/'selected_before_confirmation.json').read_text())['selected']
    target=ROOT/'nomination_lock.json'
    if target.exists():raise FileExistsError('warm confirmation already locked; do not reselect')
    write_json(target,dict(selected=selected,nominated=[] if selected=='original_teacher' else [selected],selection_sha256=proof['selection_sha256']))
    if selected=='original_teacher':
        write_json(ROOT/'state.json',dict(status='original_teacher_selected',next_phase='no confirmed improvement; next bounded hypothesis',goal_achieved=False));return
    (d,c,h,ext,_,bins,_,ix,search,_),_=inputs();cache=batches(d,c,ext,ix);models={};pairs={}
    trial=next(t for t in TRIALS if t.name==selected)
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1);models[str(seed)]={};curves={}
        for name in ('original_teacher',selected):
            if name=='original_teacher':m=teacher(base,seed,bins);digest=sha256(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt')
            else:m,info,_=candidate(base,seed,trial,bins);digest=info['checkpoint_sha256']
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k);check_curve(curve)
            save(ROOT/f'seed{seed}'/name/'confirmation_predictions.npz',d,ix,a,k,curve)
            models[str(seed)][name]=dict(metrics=summary(curve,d,ix,nll),checkpoint_sha256=digest);curves[name]=curve;del m
        pairs[str(seed)]=paired_interval(d.arrays['subject_ids'][ix],d.times[ix],d.events[ix],curves[selected][:,:,1],
                                       curves['original_teacher'][:,:,1],replicates=200,seed=seed)
    write_json(ROOT/'confirmation_report.json',dict(models=models,comparisons=pairs,selected=selected,
        lock_sha256=sha256(target),test_not_evaluated=True,goal_achieved=False))
    write_json(ROOT/'state.json',dict(status='confirmation_complete_pending_replay',goal_achieved=False))


def verify_confirmation():
    r=json.loads((ROOT/'confirmation_report.json').read_text())
    if r['lock_sha256']!=sha256(ROOT/'nomination_lock.json'):raise ValueError('lock changed')
    (d,c,h,ext,_,bins,_,ix,search,_),_=inputs();cache=batches(d,c,ext,ix);errors={}
    trial=next(t for t in TRIALS if t.name==r['selected'])
    for seed in SEEDS:
        base=restore(Path(search['input_config']['parent']),BASE_SEEDS[seed],'survival_base',h,.1)
        for name in ('original_teacher',r['selected']):
            if name=='original_teacher':m=teacher(base,seed,bins);digest=sha256(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt')
            else:m,info,_=candidate(base,seed,trial,bins);digest=info['checkpoint_sha256']
            if digest!=r['models'][str(seed)][name]['checkpoint_sha256']:raise ValueError('confirmation weights differ')
            a,k,nll=parameters(m,cache);curve=probability_curve(a,k)
            errors[f'{seed}/{name}']=check_saved(ROOT/f'seed{seed}'/name/'confirmation_predictions.npz',d,ix,a,k,curve)
            if summary(curve,d,ix,nll)!=r['models'][str(seed)][name]['metrics']:raise ValueError('confirmation metrics differ')
            del m
    write_json(ROOT/'confirmation_verification.json',dict(status='warm_confirmation_independently_verified',errors=errors,
        maximum_error=max(errors.values()),report_sha256=sha256(ROOT/'confirmation_report.json'),test_not_evaluated=True,goal_achieved=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('verify','confirm','verify-confirmation'),required=True)
    mode=p.parse_args().mode
    if mode=='verify':verify()
    elif mode=='confirm':confirm()
    else:verify_confirmation()
````
<!-- END REVIEW_FRAGMENT -->

## training/v6/warm_residual_experiment.py · lines 1–101

<!-- REVIEW_FRAGMENT {"path":"training/v6/warm_residual_experiment.py","part":1,"start_line":1,"end_line":101,"source_sha256":"f118b94d25aeb9871fab960d3b99c61fda2afd3654eb7e6327f415886eac44aa","fragment_sha256":"f118b94d25aeb9871fab960d3b99c61fda2afd3654eb7e6327f415886eac44aa","added_terminal_newline":false} -->
````python
"""从强V6起点作四类修正；教师仅重载，不冒称新训练。"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.regularization_multiseed import fit
from training.v6.warm_residual import TRIALS,BASE_SEEDS,build
from training.v6.module_surgery import SurgeryModel,probability_curve
from training.v6.module_surgery_experiment import batches,parameters,validation,summary,save
from training.v6.horizon_objective import restore
from training.v6.train import write_json

ROOT=Path('training_runs/v6_warm_residual_2026-10-13')
PRIOR=Path('training_runs/v6_contextual_residual_2026-10-13')
SEEDS=(42,7,101)
SOURCES=('training/v6/warm_residual.py','training/v6/warm_residual_experiment.py',
         'training/v6/regularization_multiseed.py','training/tests/test_v6_warm_residual.py',
         'docs/V6_WARM_START_PROTOCOL_2026-10-13.md')


def run():
    proof=json.loads((PRIOR/'tuning_verification.json').read_text())
    if proof['status']!='tuning_independently_verified' or proof['models']!=18:raise ValueError('context batch not verified')
    d,c,h,ext,_,bins,tune,_,search,prior=checked()
    official=json.loads((REFERENCE/'report.json').read_text())
    weights={str(s):sha256(REFERENCE/f'seed{s}/control/best.pt') for s in set(BASE_SEEDS[x] for x in SEEDS)}
    for s,digest in weights.items():
        if digest!=official['models'][s]['control']['checkpoint_sha256']:raise ValueError('teacher weight changed')
    contract=dict(source_sha256={p:sha256(Path(p)) for p in SOURCES},
        search_contract_sha256=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')),
        teacher_checkpoint_sha256=weights,trials=[asdict(t) for t in TRIALS],seeds=list(SEEDS),
        teacher_seed_mapping={str(s):BASE_SEEDS[s] for s in SEEDS},epochs=100,
        selection='three clinical-seed mean tuning IBS; original teacher included as comparator',
        confirmation_not_evaluated=True,test_not_evaluated=True,
        limits=['Teacher selected in previously exposed full validation.','Three clinical seeds reuse two teacher sources.'])
    if (ROOT/'contract.json').exists():
        if json.loads((ROOT/'contract.json').read_text())!=contract:raise ValueError('warm resume contract changed')
    else:write_json(ROOT/'contract.json',contract)
    cache=dict(train=batches(d,c,ext,np.flatnonzero(d.arrays['patient_splits']=='train')),tune=batches(d,c,ext,tune));results={};references={}
    for seed in SEEDS:
        source=BASE_SEEDS[seed];base=restore(Path(search['input_config']['parent']),source,'survival_base',h,.1)
        teacher=SurgeryModel(base,source,'control',bins).cuda().eval()
        teacher.load_state_dict(torch.load(REFERENCE/f'seed{source}/control/best.pt',map_location='cuda',weights_only=True)['state_dict'])
        a,k,nll=parameters(teacher,cache['tune']);curve=probability_curve(a,k)
        folder=ROOT/f'seed{seed}/original_teacher';folder.mkdir(parents=True,exist_ok=True)
        save(folder/'tuning_predictions.npz',d,tune,a,k,curve)
        references[str(seed)]=dict(tuning=validation(teacher,cache['tune'],d.times[tune],d.events[tune]),metrics=summary(curve,d,tune,nll),
                                 teacher_source=source,teacher_checkpoint_sha256=weights[str(source)],newly_trained=False)
        results[str(seed)]={}
        for trial in TRIALS:
            path=ROOT/f'seed{seed}'/trial.name
            if (path/'training_result.json').exists():
                info=json.loads((path/'training_result.json').read_text())
                if info['trial']!=asdict(trial) or info['completed_epochs']!=100 or sha256(path/'best.pt')!=info['checkpoint_sha256']:
                    raise ValueError('completed warm model changed')
                results[str(seed)][trial.name]=info;continue
            if path.exists() and any((path/p).exists() for p in ('best.pt','history.json')):
                import shutil
                backup=path/'recovery_snapshots'/str(time.time_ns());backup.mkdir(parents=True)
                for p in ('best.pt','history.json'):
                    if (path/p).exists():shutil.copy2(path/p,backup/p)
            probe=build(base,seed,trial,bins).eval()
            with torch.no_grad():
                _,hist,b,_,_=cache['tune'][0];out=probe(hist,b);expected=teacher(hist,b)
                if not torch.equal(out.log_amplitude,expected.log_amplitude) or not torch.equal(out.shape,expected.shape):
                    raise ValueError('warm epoch0 differs from trained V6')
            del probe
            write_json(ROOT/'state.json',dict(status='running',seed=seed,trial=trial.name,completed=results,goal_achieved=False))
            info=fit(base,seed,trial,bins,cache,d.times[tune],d.events[tune],path,model_factory=build)
            stored=torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict']
            if trial.mode!='finetune':
                for key,value in teacher.state_dict().items():
                    if not torch.equal(value,stored['teacher.'+key]):raise ValueError('full frozen teacher changed')
                info['full_teacher_frozen_verified']=True
            else:info['full_teacher_frozen_verified']=False
            info['initial_teacher_checkpoint_sha256']=weights[str(source)]
            write_json(path/'training_result.json',info);results[str(seed)][trial.name]=info
        del teacher
    scores={'original_teacher':float(np.mean([references[str(s)]['tuning']['ibs'] for s in SEEDS]))}
    scores.update({t.name:float(np.mean([results[str(s)][t.name]['tuning_ibs'] for s in SEEDS])) for t in TRIALS})
    selected=min(scores,key=scores.get)
    write_json(ROOT/'selected_before_confirmation.json',dict(results=results,references=references,tuning_scores=scores,selected=selected,
        all_100_candidate_epochs_complete=True,original_teacher_reloaded_not_retrained=True,confirmation_not_evaluated=True,test_not_evaluated=True))
    write_json(ROOT/'state.json',dict(status='training_complete_pending_replay',selected=selected,goal_achieved=False))
    print(json.dumps(dict(status='training_complete_pending_replay',selected=selected)),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();ROOT.mkdir(exist_ok=True)
    lock=ROOT/'.running.lock'
    with lock.open('x',encoding='utf-8') as owner:owner.write(json.dumps(dict(pid=os.getpid(),started=time.time())))
    try:run()
    except Exception as e:
        write_json(ROOT/'state.json',dict(status='failed',error_type=type(e).__name__,error=str(e),goal_achieved=False));raise
    finally:lock.unlink()
````
<!-- END REVIEW_FRAGMENT -->
