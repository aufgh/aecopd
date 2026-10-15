# AECOPD：出院后再入院与竞争死亡风险研究

## 最新研究结果（2026-10-15）

新增[完整Markdown阅读包](review_bundle/00_READ_FIRST.md)，替代此前00–07 TXT摘录：覆盖当前全部已审查公开文本源码、测试、协议与文档，并附历史实验汇总指标目录和[数据结构快照](docs/PUBLIC_DATA_SNAPSHOT_2026-10-15.md)。各文件正文按主题拆卷，每段带原文件路径、行范围和SHA256，可逐字节重建。患者行、个体标签/预测、向量、权重与本地旧Git历史保留本地。

最新[强V6上关闭跨源Query](docs/V7_STRONG_NO_CROSS_RESULTS_2026-10-15.md)已完成seed42/7：F1一次升、一次降，未稳定复现；[固定90%V6＋10%Cox](docs/V7_FIXED_V6_COX_BLEND_RESULTS_2026-10-15.md)无需模型重训，两原因IBS改善、F1略降。主/辅助检查点分开保存并独立重放，默认模型保留。

追加完成[完整V6冻结＋端到端V7＋H34训练集去重](docs/V7_PROJECTED_STRONG_RESULTS_2026-10-15.md)：seed42实际16轮，F1 .578879低于不去重的.581683，IBS .099701略低；best/last映射与预测独立重放0，按事前规则停止此配方。

[V7现有研究线索顺序实验](docs/V7_SEQUENTIAL_RESEARCH_RESULTS_2026-10-15.md)已完成56次新拟合、519神经epoch及独立重放。完整强V6冻结后的V7修正、固定训练投影、物理检验/药物动态、临床软路由、损失/Query/衰减和两折OOF均有逐项结论；小幅F1收益伴随不确定性与概率误差权衡，原默认保留。公开部分仅源码、合成测试、合同和聚合结果。

本仓库公开**研究源码、队列/任务定义、测试和经审查的汇总说明**。不提供MIMIC病历、患者标识、患者级标签/划分/预测、编码向量、模型权重或数据库凭据。数据使用必须由使用者自行取得PhysioNet授权并遵守相关协议。

## 最新：V7包含检验的多源纵向模型

V7已完成结构实现和有限首轮：统一256维、8头、M4、FFN1024，冻结BERT/RadBERT；独立检验分支、药物因子化、药物/影像Query和完整历史时间衰减。字段M4压缩旁路问题已修复，修复后从头训练并独立重放。

此前只看到文本、影像与药物的建议，需要按已存在的36个检验异常方向×9通道重新审查。检验已进入融合与文本—检验对比；首轮默认的药物Query不直接读取检验；顺序补充实验已实现该候选连接并训练，未超过初值。阅读[检验已知时的V7适配说明与实际检查状态](docs/V7_LAB_AWARE_DESIGN_AND_STATUS_2026-10-15.md)。

[首轮结果](docs/V7_FIRST_EXPERIMENT_RESULTS_2026-10-15.md)：V7原竞争风险损失对照年度F1 .565321，加入Focal＋标签平滑＋.1对比为 .567023，未超过已有V6对称背景 .577373；不据此替换默认。[损失补充比较](docs/V7_LOSS_COMPARISON_RESULTS_2026-10-15.md)：再补三次实际训练，直接Focal＋平滑（有/无对比）均未超过初值；Focal与平滑各自作用、Query和衰减独立重训尚未完成。不能把当前实现的失败一般性归因于某个方法。

公开两张图仅含逐轮汇总指标与时间，不含患者级数据。真实训练产物仍不提供；合成测试与受控复现入口见[training/v7/README.md](training/v7/README.md)。

## 当前任务与保留的V6参照

以每次合格出院为预测时点，预测未来1–365日首次严格AECOPD非转院再入院，以及先于该再入院发生的竞争死亡。模型输出连续时间风险及逐日累计概率，不是确切事件日期、不是诊断COPD，也不包含再入院后总死亡。MIMIC库内未观察到再入院不代表完整院外阴性。

- 原V6：冻结H34背景基础＋可训练临床风险修正；检验、七药效、微生物、三呼吸支持汇总、四字段冻结BERT向量。
- H34：历史6维＋本次诊断/去向14值＋14缺失标记。不是完整V6，不包含检验/文本；冻结是权重不更新，患者基础风险不同。
- 纵向候选：共享住院编码器读取历次已结束合格住院的现有临床资料，历史文书按存档可见性隔离；与本次住院一起进行纵向比较。
- 最新对称背景候选：本次与过去都先结合各自背景，再进入纵向比较；128维宽度、检验输入、冻结基础不变。实际单seed配对校正结论见[结果](docs/V6_SYMMETRIC_CONTEXT_RESULTS_2026-10-14.md)。

这是回顾性研究，不是可直接部署的临床预警系统。单seed调参结果不等独立临床有效性；确认/历史测试已被项目使用，不冒称新盲测。默认正式模型未因弱候选自动替换，长期自动探索已取消。

## 阅读顺序

0. [V7架构、实际检验输入与待验证事项](docs/V7_LAB_AWARE_DESIGN_AND_STATUS_2026-10-15.md)、[首轮结果](docs/V7_FIRST_EXPERIMENT_RESULTS_2026-10-15.md)、[损失组成比较](docs/V7_LOSS_COMPARISON_RESULTS_2026-10-15.md)。
1. [完整输入说明](docs/V6_CURRENT_INPUT_CATALOG_2026-10-11.md)：旧V6字段、来源、单位、缺失及内部维度。
2. [统一检验与纵向实验](docs/V6_UNIFIED_LONGITUDINAL_F1_RESULTS_2026-10-14.md)：四条件有限实验、实际轮数及校准/F1取舍。
3. [多期限共同评价](docs/V6_MULTIHORIZON_READOUT_RESULTS_2026-10-14.md)：7/14/30/60/90/180/365日、全年平均AUC、概率误差及固定阈值。
4. [最新对称背景校正](docs/V6_SYMMETRIC_CONTEXT_RESULTS_2026-10-14.md)：同初始化/输入的针对性实验。
5. [通俗流程与报警文献](docs/V6_HISTORY_AND_ALERT_EXPLAINED_2026-10-14.md)、[17项原始研究核对](docs/COPD_PREDICTION_PRIMARY_STUDIES_REVIEW_2026-10-14.md)。
6. [公开范围与本地资产边界](docs/PUBLICATION_BOUNDARY_2026-10-14.md)。

## 目录

|目录|用途|
|---|---|
|`cohort_pipeline/`|队列、ICD结局、转院资格与历史定义SQL；数据库导入/查询代码|
|`patient_data/`|导出、清洗、映射、接口构建、临床事实及边界测试源码；不含数据实物|
|`training/v5/`|保留的V5实现与编码/训练接口|
|`training/v6/`|V6风险模型、有限候选实验、独立重载与多期限评价|
|`training/v7/`|已含检验的256维多源纵向模型、M4修复、损失组成比较及独立重载|
|`training/baselines/`|Cox/XGBoost等基线，不同输入范围需单列|
|`training/tests/`|合成数据单元/接口回归，不需要公开患者病历|
|`model_loader/`|数据加载/批处理工具|
|`config/`|不含患者行的临床概念合同|
|`docs/`|经审查的当前合同、汇总结果和原始文献说明|
|`work/prepare_public_snapshot.py`|公开白名单/风险扫描工具，旧本地历史不会上传|

## 方法与复现范围

Python/PyTorch/NumPy；公开表格包版本配方在`training/runtime/public_tabular_requirements.txt`。BERT编码及私有数据构建路径见源码及合同，环境需要自行配置。当前实验入口依赖本地已核验数据接口、旧基础权重与来源hash；**仅克隆仓库不能直接执行真实MIMIC训练**，这不意味着真实实验产物不存在，而是它们必须留在受控本地环境。

可先在安装依赖的环境验证合成回归：

```bash
python -m unittest training.tests.test_v6_symmetric_context training.tests.test_v6_unified_longitudinal training.tests.test_v6_multihorizon_readout -v
python -m unittest training.tests.test_v7_model training.tests.test_v7_losses training.tests.test_v7_loss_comparison -v
```

真实受控环境的有限实验入口：

```bash
python -m training.v6.symmetric_context_experiment --mode train
python -m training.v6.symmetric_context_experiment --mode verify
```

入口拒绝覆盖已存在正式实验目录；运行前先读协议、检查来源合同和数据权限。不要解除Git忽略规则上传受限资料，也不要把演示/调参概率当医疗建议。
