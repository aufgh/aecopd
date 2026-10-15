"""组合实验独立重放完成后生成聚合中文结果，不重新选点。"""
import json
from pathlib import Path
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json

ROOT = Path('training_runs/v7_projected_strong_2026-10-15')
TARGET = Path('docs/V7_PROJECTED_STRONG_RESULTS_2026-10-15.md')


def main():
    if TARGET.exists():
        raise FileExistsError('projected results already exist')
    rows = []; sources = {}; detail = []; fits = 0; epochs = 0
    original = Path('training_runs/v7_sequential_research_2026-10-15')
    for seed in (42, 7):
        folder = ROOT/f'seed{seed}'
        if not folder.exists():
            continue
        report = json.loads((folder/'report.json').read_text()); proof = json.loads((folder/'verification.json').read_text())
        if proof['maximum_error'] != 0 or proof['projection_reestimation_error'] != 0 or proof['report_sha256'] != sha256(folder/'report.json'):
            raise ValueError('projected proof stale')
        control_root = original/('strong_anchor' if seed == 42 else 'strong_replication')
        control = json.loads((control_root/'report.json').read_text()); control = control if seed == 42 else control['result']
        fits += 1; epochs += report['completed_epochs']
        for name, info in ((f'不做去重，seed{seed}（复用）', control), (f'每轮训练集去重，seed{seed}（新训练）', report)):
            b = info['outputs']['best']; m = b['metrics']
            rows.append(f"|{name}|{info['completed_epochs']}|{b['epoch']}|{m['annual']['f1']:.6f}|{m['readmission']['integrated_brier_7_365']:.6f}|{m['death']['integrated_brier_7_365']:.6f}|{m['readmission']['mean_auc_7_365']:.6f}|")
        c = report['comparison']; hist = json.loads((folder/'history.json').read_text())
        best = report['outputs']['best']; last = report['outputs']['last']
        lo, hi = c['f1_95ci']; il, ih = c['ibs_95ci']
        p = hist[best['epoch']]['projection']
        detail.append(f"""### seed{seed}

实际{report['completed_epochs']}轮，选定第{best['epoch']}轮，停止原因`{report['stop_reason']}`。相对同种子不去重：ΔF1={c['delta_f1']:+.6f}，患者配对95%区间[{lo:+.6f}, {hi:+.6f}]；Δ再入院IBS={c['delta_ibs']:+.6f}，区间[{il:+.6f}, {ih:+.6f}]。两者均为候选减对照，IBS负值较好。固定402名调参患者、200次配对重采样、固定已选阈值；条件于已拟合模型和已暴露调参，未作多重比较校正。

末轮F1={last['metrics']['annual']['f1']:.6f}，再入院IBS={last['metrics']['readmission']['integrated_brier_7_365']:.6f}。选点遵循原年度F1/并列IBS，未按其他epoch较好的单项指标事后改选。

选定权重对应的映射使用{p['training_rows']}个训练出院。原表示均方{p['hidden_mean_square']:.6f}，移除后的均方{p['residual_mean_square']:.6f}；这是表示变化的工程诊断，不表示临床信息量。入口计时{report['seconds']:.2f}秒，其中逐轮重新估计映射{report['projection_refresh_seconds']:.2f}秒（初始化映射归入入口总耗时）。

按事前预算规则是否建议追加种子：`{c['replication_recommended']}`。该规则仅控制追加计算，临床成功需要独立验证。
""")
        diagnostic_path = folder/'representation_diagnostics.json'
        if diagnostic_path.exists():
            diagnostic = json.loads(diagnostic_path.read_text())
            dp = json.loads((folder/'representation_diagnostics_verification.json').read_text())
            if dp['maximum_error'] != 0 or dp['diagnostics_sha256'] != sha256(diagnostic_path):
                raise ValueError('representation diagnostic replay stale')
            detail.append(f"训练表示补充诊断：原表示去掉共同均值后的均方为{diagnostic['hidden_centered_mean_square']:.8f}，去重残差均方为{diagnostic['residual_mean_square']:.8f}，保留患者间中心化变异的比例为{diagnostic['residual_fraction_of_centered_variance']:.6f}。原未中心化能量主要包含共同均值，不能用“总能量移除比例”当成H34解释了多少临床信息。此诊断不用于重选权重或决定追加种子。\n")
            sources[str(diagnostic_path)] = sha256(diagnostic_path)
            sources[str(folder/'representation_diagnostics_verification.json')] = sha256(folder/'representation_diagnostics_verification.json')
        for name in ('contract.json', 'report.json', 'verification.json', 'preflight.json', 'history.json'):
            sources[str(folder/name)] = sha256(folder/name)
    if not fits:
        raise ValueError('no independently verified fits')
    text = f'''# 完整V6冻结＋端到端V7＋H34训练集去重：组合实验结果

已完成{fits}次新增拟合、{epochs}个实际神经epoch；同种子不去重的对照复用上一批已独立核验结果。原模型与旧实验保持，默认未替换。

这是用户要求补做的组合：完整强V6风险固定，重新训练V7编码器，在风险修正头之前减去H34可以线性预测的表示部分。它与旧的“冻结V7表示＋16单元小读取器”是不同实验。完整V6自身已经使用检验、文本、药物和历史；新增去重不构成新数据，也不能自动解释为新增临床信息。

## 匹配结果

|条件|实际训练轮数|选定轮数|年度F1↑|再入院IBS↓|死亡IBS↓|全年平均AUC↑|
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'\n\n'+'\n'.join(detail)+'''
## 实现与解释范围

V7表示在训练中变化，因此初始化及每轮结束都仅用训练患者、eval模式重估H34→256维表示的岭映射。岭系数1，截距不惩罚；轮内映射固定，梯度通过V7表示，不穿过映射求解。当前轮验证使用当前轮训练集估计的固定映射。没有新增表示缩放、零门、辅助损失、输入或网络宽度，其他条件与同种子对照一致。

该交替更新机制不能被称为每个训练步骤都严格正交，更不能说已经减掉完整V6解释的一切信息。验证只证明这个确定配方的表现，不涵盖其他岭强度、更新频率、完整训练梯度投影或全量OOF基座。

## 验证与来源

10项相关合成测试通过，覆盖关闭投影的原路径一致性、训练/留出隔离、共线与空输入、零末层恢复完整V6、实际风险头读取变换、原始表示提取与编码器梯度。真实GPU预检确认基座风险初始误差0、编码器实际更新、固定映射没有被优化器更新、H34冻结。

训练后独立新GPU进程重载best/last；分别从各检查点重新遍历训练患者拟合映射，与保存映射误差0，逐日概率重放误差0，指标、早停、选点和患者配对统计一致。确认/历史测试未参与模型或映射拟合。

前两次合成检查因表示捕获no_grad与正常可求导前向选择不同注意力内核而出现约1e-6差异，失败容器保留；在相同no_grad执行路径核对数值变换，再单独验证正常梯度后通过。真实实验沿用原setup关闭MHA快路径，不以放宽容差掩盖风险路径问题。

本地输出为`training_runs/v7_projected_strong_2026-10-15/`，含合同、预检、逐轮记录、best/last、预测和独立证明。患者级数据及权重不公开。

- [事前合同](V7_PROJECTED_STRONG_PROTOCOL_2026-10-15.md)
- [模型](../training/v7/projected_strong.py)、[训练/验证](../training/v7/projected_strong_experiment.py)、[表示诊断](../training/v7/projected_strong_diagnostics.py)、[合成测试](../training/tests/test_v7_projected_strong.py)
- [上一批完整结果](V7_SEQUENTIAL_RESEARCH_RESULTS_2026-10-15.md)

使用原离线GPU容器环境执行`python -m training.v7.projected_strong_experiment --mode train --seed 42`及独立`--mode verify --seed 42`；只有符合合同的信号才追加seed7。已有目录拒绝覆盖。`python -m training.v7.projected_strong_report`只生成经核验的聚合报告。
'''
    TARGET.write_text(text, encoding='utf-8')
    write_json(ROOT/'report_document.json', dict(report=str(TARGET), report_sha256=sha256(TARGET),
        inputs=sources, new_fits=fits, completed_epochs=epochs, default_replaced=False))
    print(json.dumps(dict(report=str(TARGET), new_fits=fits, completed_epochs=epochs)), flush=True)


if __name__ == '__main__':
    main()
