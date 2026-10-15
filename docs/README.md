# 公开文档导航

本文件在公开快照映射为`docs/README.md`；本地详细历史导航不覆盖。

- [全部公开源码/结果/数据结构的Markdown分卷](../review_bundle/00_READ_FIRST.md)：替代旧TXT摘录，按完整源文件清单导出，支持逐字节还原核验。
- [数据快照说明](PUBLIC_DATA_SNAPSHOT_2026-10-15.md)：19,401出院/5,351患者、四划分、17张原始表规模、66份派生数组结构；没有患者取值。
- [最新关闭跨源Query结果](V7_STRONG_NO_CROSS_RESULTS_2026-10-15.md)和[固定比例融合结果](V7_FIXED_V6_COX_BLEND_RESULTS_2026-10-15.md)：已完成两个新训练与一次零重训融合，分别报告收益与权衡。

- [追加：强V6冻结＋端到端V7＋H34训练集去重](V7_PROJECTED_STRONG_RESULTS_2026-10-15.md)与[合同](V7_PROJECTED_STRONG_PROTOCOL_2026-10-15.md)：seed42实际16轮，F1低于不去重匹配对照、IBS略低；独立映射/概率重放0，未追加种子，默认保留。

- [最新：V7现有研究线索顺序实验结果](V7_SEQUENTIAL_RESEARCH_RESULTS_2026-10-15.md)及[总合同](V7_SEQUENTIAL_RESEARCH_PRD_2026-10-15.md)：56次新拟合、519神经epoch，强V6冻结修正、固定投影、物理检验、药物动态、损失/Query/衰减及两折OOF逐项完成；默认保留。
- [外部具体数学与工程设计核对](V7_EXTERNAL_MATH_ENGINEERING_REVIEW_2026-10-15.md)：文献支持范围、公式边界及本项目修正；其“尚未做”描述为当时审查时点，最新状态以上方结果为准。

- [V7已知检验输入后的方案适配与检查状态](V7_LAB_AWARE_DESIGN_AND_STATUS_2026-10-15.md)：实际架构/数据、文本补充规则、现有与待验证连接、给其他AI的审查说明。
- [V7首轮合同](V7_FIRST_EXPERIMENT_PRD_2026-10-15.md)与[结果](V7_FIRST_EXPERIMENT_RESULTS_2026-10-15.md)，[损失比较合同](V7_LOSS_COMPARISON_PRD_2026-10-15.md)与[结果](V7_LOSS_COMPARISON_RESULTS_2026-10-15.md)。
- [V7运行与源码入口](../training/v7/README.md)。首轮与补充实验分开保存；每批都有来源合同、真实训练与独立重放状态。
- [本次对称校正协议](V6_SYMMETRIC_CONTEXT_PROTOCOL_2026-10-14.md)与[实际结果](V6_SYMMETRIC_CONTEXT_RESULTS_2026-10-14.md)。
- [原统一检验/完整历史需求](V6_UNIFIED_LONGITUDINAL_F1_PRD_2026-10-14.md)与[四条件结果](V6_UNIFIED_LONGITUDINAL_F1_RESULTS_2026-10-14.md)。
- [多期限读出合同](V6_MULTIHORIZON_READOUT_PROTOCOL_2026-10-14.md)与[结果](V6_MULTIHORIZON_READOUT_RESULTS_2026-10-14.md)。
- [输入目录](V6_CURRENT_INPUT_CATALOG_2026-10-11.md)、[通俗历史与预警](V6_HISTORY_AND_ALERT_EXPLAINED_2026-10-14.md)。
- [原始研究核对](COPD_PREDICTION_PRIMARY_STUDIES_REVIEW_2026-10-14.md)、[有限单seed预算](V6_FAST_SCREENING_POLICY_2026-10-14.md)。
- [公开边界](PUBLICATION_BOUNDARY_2026-10-14.md)。

文档中指向`patient_data/processed`、`training_runs`或`outputs`的链接是本地资产位置，不是公开下载地址；不包含受限数据或权重。历史实验说明保留其当时配方，不冒充所有候选已采用。
