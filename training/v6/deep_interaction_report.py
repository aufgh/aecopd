"""只汇总已独立重放的A/B/C；英文图和中文结果页，不自动替换默认。"""
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.deep_interaction import ARMS, CANDIDATES
from training.v6.deep_interaction_experiment import ROOT, REFERENCE, SEEDS
from training.v6.evaluation import metrics

LABELS = dict(control='Original V6 (replayed)', late_gated='Late pooling + gated 1 block',
    late_single='Late pooling + ungated 1 block', A_deep='A: 6 independent blocks',
    B_shared='B: 1 shared block, 6 rounds', C_latent='C: 16 latent slots, 4 blocks')


def value(r,seed,arm,key,fold='test'):
    x=r['models'][str(seed)][arm]['folds'][fold]['readmission']
    if key.startswith('auc'):
        return x['points'][key[3:]]['auc']
    if key.startswith('brier'):
        return x['points'][key[5:]]['brier']
    return x[key]


def mean(r,arm,key,fold='test'):
    return float(np.mean([value(r,s,arm,key,fold) for s in SEEDS]))


def finish():
    r=json.loads((ROOT/'report.json').read_text())
    proof=json.loads((ROOT/'independent_verification.json').read_text())
    if r['status']!='completed_and_independently_verified' or proof['report_sha256']!=sha256(ROOT/'report.json'):
        raise ValueError('independent verification required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={a:f'C{i}' for i,a in enumerate(ARMS)}
    fig,axes=plt.subplots(1,2,figsize=(15,6),sharey=True)
    for j,a in enumerate(ARMS):
        for s,offset in ((42,-.08),(7,.08)):
            axes[0].scatter(value(r,s,a,'auc90'),j+offset,color=colors[a],alpha=.65)
            axes[1].scatter(value(r,s,a,'integrated_brier_7_365'),j+offset,color=colors[a],alpha=.65)
        axes[0].scatter(mean(r,a,'auc90'),j,color=colors[a],marker='D')
        axes[1].scatter(mean(r,a,'integrated_brier_7_365'),j,color=colors[a],marker='D')
    for ax in axes:
        ax.set_yticks(range(len(ARMS)),[LABELS[a] for a in ARMS]);ax.grid(axis='x',alpha=.25)
    axes[0].invert_yaxis();axes[0].set_xlabel('Readmission AUC90 (higher better)')
    axes[1].set_xlabel('Readmission IBS7-365 (lower better)')
    fig.suptitle('Same input information and frozen H34; dots: seeds; diamonds: mean; reused test')
    fig.tight_layout();fig.savefig(ROOT/'performance_overview.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(15,7),sharey=True)
    pairs=list(r['comparisons']['42']['test'])
    for j,pair in enumerate(pairs):
        for seed,offset in ((42,-.09),(7,.09)):
            x=r['comparisons'][str(seed)]['test'][pair]
            for ax,k,ci in ((axes[0],'integrated_brier_difference','integrated_brier_95ci'),(axes[1],'auc90_difference','auc90_95ci')):
                ax.plot(x[ci],[j+offset]*2,color='C0' if seed==42 else 'C1',alpha=.7)
                ax.scatter(x[k],j+offset,color='C0' if seed==42 else 'C1',marker='D',label=f'Seed {seed}' if j==0 else None)
    for ax in axes:
        ax.axvline(0,color='black',lw=1);ax.set_yticks(range(len(pairs)),pairs);ax.grid(axis='x',alpha=.25);ax.legend()
    axes[0].invert_yaxis();axes[0].set_xlabel('IBS: candidate minus comparator (negative better)')
    axes[1].set_xlabel('AUC90: candidate minus comparator (positive better)')
    fig.suptitle('Paired patient-cluster bootstrap, 200 resamples; no multiplicity adjustment')
    fig.tight_layout();fig.savefig(ROOT/'paired_increment.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(14,9))
    histories={}
    for row,seed in enumerate(SEEDS):
        histories[seed]={}
        for a in ARMS:
            source=REFERENCE if a=='control' else ROOT
            history=json.loads((source/f'seed{seed}'/a/'history.json').read_text())
            histories[seed][a]=history
            ep=[x['epoch'] for x in history]
            axes[row,0].plot(ep,[x['train_fixed2048_nll'] for x in history],label=LABELS[a],color=colors[a])
            axes[row,1].plot(ep,[x['validation_ibs'] for x in history],label=LABELS[a],color=colors[a])
        for col in range(2):
            axes[row,col].set_xlabel('Epoch');axes[row,col].grid(alpha=.25);axes[row,col].legend(fontsize=7)
        axes[row,0].set_title(f'Seed {seed}: fixed training-sample NLL')
        axes[row,1].set_title(f'Seed {seed}: validation IBS7-365')
    fig.suptitle('100 epochs per new model; original V6 histories reused; checkpoints selected by validation IBS')
    fig.tight_layout();fig.savefig(ROOT/'training.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for a in ARMS:
        auc,brier=[],[]
        for s in SEEDS:
            with np.load(ROOT/f'seed{s}'/a/'test_predictions.npz') as z:
                detail=metrics(z['times'],z['events'],z['curve'][:,:,1])
                auc.append(detail['auc_curve']);brier.append(detail['brier_curve'])
        axes[0].plot(np.arange(7,366),np.mean(auc,axis=0)[7:],label=LABELS[a],color=colors[a])
        axes[1].plot(np.arange(366),np.mean(brier,axis=0),label=LABELS[a],color=colors[a])
    for ax in axes:
        ax.set_xlabel('Days after discharge');ax.legend(fontsize=7);ax.grid(alpha=.25)
    axes[0].set_ylabel('Readmission AUC(t)');axes[1].set_ylabel('Readmission Brier(t)')
    fig.suptitle('Two-seed metric means; continuous-time daily predictions retained')
    fig.tight_layout();fig.savefig(ROOT/'time_dependent_metrics.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(14,8))
    for row,seed in enumerate(SEEDS):
        for a in CANDIDATES:
            history=histories[seed][a]
            ep=[x['epoch'] for x in history[1:]]
            axes[row,0].plot(ep,[x['branch_gradient_norms']['interaction'] for x in history[1:]],label=LABELS[a],color=colors[a])
        gate=[x['gate'] for x in histories[seed]['late_gated']]
        axes[row,1].plot(range(len(gate)),gate,color=colors['late_gated'],label='tanh(gate)')
        axes[row,0].set_yscale('symlog',linthresh=1e-6)
        axes[row,0].set_title(f'Seed {seed}: first-batch interaction gradient norm')
        axes[row,1].set_title(f'Seed {seed}: learned outer gate')
        for ax in axes[row]:ax.set_xlabel('Epoch');ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle('Optimization diagnostics, not causal importance or proof of medical reasoning')
    fig.tight_layout();fig.savefig(ROOT/'interaction_diagnostics.png',dpi=180);plt.close(fig)

    selected=r['validation_selected_arm']
    with np.load(ROOT/f'seed42'/selected/'test_predictions.npz') as z:
        curve=z['curve'][:3];daily=z['daily_event_probability'][:3];cond=z['conditional_daily_event_probability'][:3]
    fig,axes=plt.subplots(3,3,figsize=(14,9))
    for i in range(3):
        axes[i,0].plot(np.arange(366),curve[i,:,1],label='Readmission CIF')
        axes[i,0].plot(np.arange(366),curve[i,:,2],label='Death CIF')
        axes[i,1].plot(np.arange(1,366),daily[i,:,0],label='Unconditional daily readmission')
        axes[i,2].plot(np.arange(1,366),cond[i,:,0],label='Daily readmission given event-free yesterday')
        for ax in axes[i]:ax.set_xlabel('Days after discharge');ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle(f'Validation-selected {LABELS[selected]}; fixed first 3 test rows; no identities')
    fig.tight_layout();fig.savefig(ROOT/'daily_probability_examples.png',dpi=180);plt.close(fig)

    lines=['# V6深层交互A/B/C：完整结果','',
        '五个新增条件×两个种子，均训练完整100epoch；原V6对照精确重载此前r3检查点，不冒称本轮重训。队列、标签、输入信息、冻结H34和风险头不改，只替换住院编码器。',
        '',f'验证IBS两种子均值选择：`{selected}`。本轮未自动覆盖默认模型。',
        '', '| 条件 | 验证IBS↓ | 测试AUC30 | AUC90 | AUC180 | AUC365 | Brier90↓ | 测试IBS↓ | 最佳epoch42/7 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for a in ARMS:
        vals=[mean(r,a,'integrated_brier_7_365','validation')]+[mean(r,a,k) for k in ('auc30','auc90','auc180','auc365','brier90','integrated_brier_7_365')]
        ep='/'.join(str(r['models'][str(s)][a]['best_epoch']) for s in SEEDS)
        lines.append('| '+LABELS[a]+' | '+' | '.join(f'{v:.6f}' for v in vals)+' | '+ep+' |')
    lines+=['','## 本轮结论与采用决定','',
        '- 按事先固定的验证全年IBS，两种子均值选择原V6，A/B/C不替换默认。三个方案已实际训练，不再把它们作为未验证计划。',
        '- A六块共344万可训练参数（约原V6的12.8倍），训练损失明显降低，但验证后期误差上升；最佳点也未超过原V6。不能继续把缺少增益简单归因于网络太浅或没训练够。',
        '- 去外层门控相对有门控单块，两个种子的AUC90点估计均略降，IBS均变差；不支持门控是主要性能瓶颈。此为新晚池化共同架构内的机制对照，不能声称排除了旧架构中所有门控配置。',
        '- B相对同参数单块，两个种子的AUC90分别提高约0.00357/0.00668，但配对区间跨0，IBS方向不一致，仍低于原V6；只保留局部信号，不称稳定改进。',
        '- C在两个种子下的AUC90与IBS点估计均不如原V6。本次16槽位/四块固定配方不采用，不能外推为Perceiver方法普遍无效。',
        '- 有门控晚池化单块的测试IBS均值稍好约0.000084，但两个种子方向相反、区间均跨0、验证均值较差且AUC90均降低，不把这一项改善称整体提升。',
        '- 交互层和文本/检验/药物分支确实获得梯度；这是可训练的工程证据，不是已学到医学关系的证据。',
        '', '## 配对增量','',
        '候选−对照：AUC正较好、IBS负较好。每个种子单列患者聚类200次重抽样；非盲测、未校正多重比较，不代表完整临床随访。',
        '', '| 比较 | seed | ΔAUC90 [95%CI] | ΔIBS [95%CI] |','|---|---:|---|---|']
    for pair in pairs:
        for s in SEEDS:
            x=r['comparisons'][str(s)]['test'][pair]
            def interval(k,ci):
                lo,hi=x[ci];return f'{x[k]:+.6f} [{lo:+.6f}, {hi:+.6f}]'
            lines.append(f'| {pair} | {s} | '+interval('auc90_difference','auc90_95ci')+' | '+interval('integrated_brier_difference','integrated_brier_95ci')+' |')
    lines+=['','## 容量、速度与优化诊断','',
        '| 条件 | 可训练参数 | 两种子训练合计分钟 | 训练张量显存峰值GiB |','|---|---:|---:|---:|']
    for a in CANDIDATES:
        data=[r['models'][str(s)][a] for s in SEEDS]
        lines.append(f'| {LABELS[a]} | {data[0]["trainable_parameters"]:,} | {sum(x["elapsed_seconds"] for x in data)/60:.2f} | {max(x["peak_training_gpu_allocated_gib"] for x in data):.2f} |')
    lines+=['','显存为PyTorch已分配张量峰值，包括缓存输入；不是nvidia-smi总显存。梯度范数记录每轮首个训练批次、裁剪前；非分支贡献百分比。风险头零初始化使第一次反传的编码器梯度为零，这是设计预期，后续轮次应检查是否恢复。',
        '', '## 实现与归因边界','',
        '- 交互层使检验、药效、微生物、呼吸支持及文本块相互读取；有梯度不等于已学到医学机制。',
        '- A六个独立块；B一块六轮且参数量与单块相同；C16潜在槽位、四个潜在块、两次读取输入。宽度256，输出适配回128，旧128→32→4头保留。',
        '- 五个新条件共同tokenizer、输出层和修正头初值一致；Epoch0风险与冻结基础参照完全相同。当前诊断/去向不重复输入，历史次数没有重新放进可训练分支。',
        '- 控制相同输入信息但改变表示组织；新条件vs旧V6不能单归因于深度。A与late_single同宽度/读出，B与late_single同参数；门控比较单独列出。',
        '- 44个结构化token＋全部B四字段BERT分块；未新增用药剂量、Qwen、影像或高频时序数据，不在处理槽位中人为加入行号顺序。',
        '- 100轮完整训练＋warmup/cosine；最佳轮由验证全年IBS选，最佳轮早不是训练提前结束。',
        f'- 新进程{proof["folds"]}折重放、最大概率误差{proof["maximum_error"]:.3g}；冻结权重、患者索引/标签、每日概率、CIF单调和指标一致性核验通过。',
        '- 再入院与死亡分别记录IBS、项目CIF排序和30/90/180/365天校准；完整表在report.json。',
        '- 新增9项测试＋模块替换/旧编码/评价共34项测试通过，GPU前向反向真实运行，无跳过。',
        '- 同一个学习率和两种子不是穷尽调参。测试重复使用；不把点估计小幅改善称临床突破。',
        '', '## 可视化与文件','',
        '- performance_overview.png：两种子散点及均值，不是置信区间。',
        '- paired_increment.png：患者聚类配对区间。',
        '- training.png：100轮训练NLL与验证IBS。',
        '- time_dependent_metrics.png：逐天AUC(t)/Brier(t)。',
        '- interaction_diagnostics.png：交互梯度与门控。',
        '- daily_probability_examples.png：固定前三条的每日风险示例，无身份。',
        '- contract.json、selected_before_test.json、report.json、independent_verification.json。',
        '- seed*/条件：检查点、history.json、training_result.json及验证/测试概率。control的检查点仍在此前r3，新的预测已独立保存。',
        '', '理论来源：[Universal Transformers](https://arxiv.org/abs/1807.03819)、[Perceiver IO](https://arxiv.org/abs/2107.14795)；本地是受启发的V6编码适配器，不冒称完整论文复现。']
    content='\n'.join(lines)+'\n'
    (ROOT/'README.md').write_text(content,encoding='utf-8')
    Path('docs/V6_DEEP_INTERACTION_RESULTS_2026-10-12.md').write_text(content,encoding='utf-8')
    print(json.dumps(dict(status='six_figures_and_report_written',selected=selected,
        results={a:dict(auc90=mean(r,a,'auc90'),ibs=mean(r,a,'integrated_brier_7_365')) for a in ARMS})),flush=True)


if __name__=='__main__':
    finish()
