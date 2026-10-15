# 01_DATA_SNAPSHOT_AND_REPORTS

源文件正文或经过字段白名单提取的汇总数据，来源身份见每段元数据。

## docs/COPD_PREDICTION_PRIMARY_STUDIES_REVIEW_2026-10-14.md · lines 1–218

<!-- REVIEW_FRAGMENT {"path":"docs/COPD_PREDICTION_PRIMARY_STUDIES_REVIEW_2026-10-14.md","part":1,"start_line":1,"end_line":218,"source_sha256":"25b76dc88bd57a333d37f89feb62025a559ac202fc1fb3cf5f8f7a8ebda48fd9","fragment_sha256":"25b76dc88bd57a333d37f89feb62025a559ac202fc1fb3cf5f8f7a8ebda48fd9","added_terminal_newline":false} -->
````markdown
# COPD/AECOPD 风险预测：17 项原始研究的架构、数据与验证核对

核对日期：2026-10-14。用途：研究比较与后续实验设计，不是临床使用许可，也不是新的实验结果。本轮未训练模型、改变原队列、更新标签或替换 V6。

## 1. 先明确比较的对象

这些论文都涉及预测，但并不预测同一种事件：未来中重度急性加重、因急性加重住院、全因再入院、再入院或死亡、频繁加重表型，以及每日事件预测不能混为一个排行榜。患者人数、住院次数、肺功能检查次数也不能互换。

- AUROC/C-statistic 衡量区分能力，不是“预测正确的患者比例”；生存 C-index 和固定时间二分类 AUROC 也不完全相同。
- Accuracy 是给定阈值后的正确比例；Sensitivity 是事件检出比例；PPV 是预警后确实发生事件的比例。
- 个体绝对风险需要校准验证；风险分层或 AUC 较高本身不足以证明“预测 20% 就真有约 20% 发生”。
- 下列数值保留各论文的指标名称和验证环境。未取得补充材料或只核对到摘要的部分明确标记，不猜层数、隐藏维度、参数量。

核验层级：正文/PDF 表示核对到原始正文的方法或结果；摘要表示仅原始摘要；补充材料未完整取得不等于完整复现。网页访问有间歇性 403/验证码，未绕过访问控制。

## 2. 临床风险模型与未来急性加重

### 01. Bertens 等，2013，International Journal of COPD

题名：Development and validation of a model to predict the risk of exacerbations in chronic obstructive pulmonary disease。

- 数据与任务：开发 240 人，独立验证 793 人；预测 24 个月内需口服激素或住院的加重。
- 架构：逻辑回归、后退筛选、系数收缩；四项输入为既往加重、吸烟包年、气流受限、血管病史，不是深度网络。
- 结果：开发 C-statistic 0.75；外部 0.66（95% CI 0.61–0.71），另评估校准。
- 边界：外部年龄范围不同；不能把开发分数当泛化性能。

证据层级：原始摘要及校准图说明。[原文记录](https://pubmed.ncbi.nlm.nih.gov/24143086/)，DOI：10.2147/COPD.S49609。

### 02. PEARL，2017，Thorax

题名：The PEARL score predicts 90-day readmission or death after hospitalisation for acute exacerbation of COPD。

- 数据：开发 824、内部验证 802、外部验证 791 人，共 2,417 人。
- 架构：逻辑回归转整数评分；既往住院、扩展 MRC 气促、年龄、右心衰、左心衰五项。
- 任务：AECOPD 出院后 90 天再入院或未再入院先死亡的复合事件，不是严格 AECOPD 再入院单终点。
- 结果：三队列 AUROC 0.73 / 0.68 / 0.70；有校准和风险分层。

证据层级：原始 PDF 方法/队列表格。[原始论文](https://thorax.bmj.com/content/72/8/686)，DOI：10.1136/thoraxjnl-2016-209298。

### 03. ACCEPT，2020，Lancet Respiratory Medicine

- 数据：三项 COPD 试验合并开发 2,380 人；ECLIPSE 外部验证。
- 架构：联合 frailty–logistic 模型；一部分估计事件发生率，另一部分估计一次事件成为重度事件的概率，输出次数及严重度分布，不是神经网络。
- 任务：未来一年加重负担。
- 结果：ECLIPSE 全部 COPD 人群，≥2 次加重 AUC 0.81、≥1 次重度 0.77；仅既往有加重者为 0.73 / 0.74。后者预测总/重度年发生率 1.80 / 0.37，观察 1.82 / 0.40。
- 边界：同一模型仅改变患者范围，AUC 就明显改变；不是“准确率 81%”。

证据层级：原始摘要，模型定义与 ACCEPT 2.0 原文交叉核对。[原始论文记录](https://pubmed.ncbi.nlm.nih.gov/32178776/)，DOI：10.1016/S2213-2600(19)30397-2。

### 04. ACCEPT 2.0，2022，eClinicalMedicine

- 数据：ECLIPSE 1,803 人再校准；TORCH 安慰剂组 1,091 人外部验证。
- 架构：保留联合发生率/严重度模型，以多变量自适应回归样条 MARS 再校准，另构建减少输入要求的版本。
- 任务：未来 12 月 ≥2 次中度或 ≥1 次重度加重。
- 结果：外部 AUC 约 0.756，单纯加重史约 0.68；报告校准及决策曲线净获益。
- 启示：区分能力之外，需证明风险概率和行动阈值可信；不是单纯更换分类头。

证据层级：原始正文和摘要。[原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9309408/)，DOI：10.1016/j.eclinm.2022.101574。[作者分析代码](https://github.com/resplab/papercode/tree/main/ACCEPT2.0)。

### 05. SERCO，2024，BMJ Open Respiratory Research

- 数据：中国 ACURE 前瞻性登记，4,065 人，开发 2,196 / 地理外部验证 1,869。
- 架构：原因别风险回归、临床变量及交互项、累积发生风险；14 项预测因子，不是深度网络。
- 任务：AECOPD 出院后 1/6/12 月严重加重和 COPD 特异再入院；严重加重包括急诊/住院，其死亡定义另有规定。
- 结果：重度加重开发 C-index 0.773/0.765/0.747，外部 0.645/0.635/0.643；再入院外部 0.638/0.631/0.641。有校准和 DCA。

证据层级：原始正文及表 3。[原始论文](https://bmjopenrespres.bmj.com/content/11/1/e001881)，DOI：10.1136/bmjresp-2023-001881。不能将开发 0.747 写成外部表现。

### 06. 无既往加重史专用评分，2026，BMJ Open Respiratory Research

题名：Development and validation of a risk stratification tool for predicting COPD exacerbations in patients without prior exacerbation history。

- 数据：五医院；真正建模的无既往加重者开发 310、外部验证 86 人，不是全部筛查人数。
- 架构：后退逻辑回归转评分，以及等权简化模型；哮喘、mMRC、嗜酸百分比、肺功能及心脑血管病等。
- 任务：未来一年加重。
- 结果：外部三版 AUROC 0.727/0.717/0.728；第一版敏感度 88.2%、特异度 46.4%，区间较宽。
- 边界：无既往事件不等于首次住院；小外部样本不能当确定解决“历史独大”。

证据层级：原始 PDF 队列图、评分表及结果图。[原始论文](https://bmjopenrespres.bmj.com/content/13/1/e004016)，DOI：10.1136/bmjresp-2025-004016。

### 07. 频繁加重表型 Stacking，2025，BMC Medical Informatics and Decision Making

- 数据：开发 1,310 人，两个外部队列 418 / 200 人；64 项变量筛至 14 项。
- 架构：单变量筛选与 Boruta；RF/SVM/XGBoost/MLP 为基学习器，LASSO 为第二层组合器，另用 SHAP 解释。
- 结果：内部测试 Stacking AUC 约 0.755；外部 0.703 / 0.694，不是外部 0.9。
- 重要限制：摘要将标签写为 12 月内两次以上住院加重，正文又描述索引住院后一年“再一次”住院；是否计入索引事件须在复现前澄清。另外模型选择参考外部集，不能当完全未参与选择的终末盲测。

证据层级：原始摘要与正文。[原始论文](https://link.springer.com/article/10.1186/s12911-025-03281-4)。

## 3. 最接近出院后再入院的电子病历研究

### 08. Min 等，2019，Scientific Reports

题名：Predictive Modeling of the Hospital Readmission Risk from Patients' Claims Data Using Machine Learning: A Case Study on COPD。

- 数据：Geisinger 原库 111,992 人；最终 27,138 人、67,771 次索引住院；患者级五折。
- 任务：出院时预测 30 日再入院，输入为出院前一年索赔记录。
- 架构：人工特征与诊断/操作/药物编码；BoW/TF-IDF；另比较代码嵌入、CNN/RNN/LSTM/GRU、时间权重与注意力。
- 结果：知识/数据特征结合最佳 AUC 0.653；复杂深度模型约 0.65，没有明确超越传统模型。
- 启示：成组编码、药物特征弱、深度模型无增益，都有直接同行先例，不只发生在本项目。

证据层级：原始正文。[原始论文](https://www.nature.com/articles/s41598-019-39071-y)。各变体精确层宽未全部核验，不提供伪精确结构。

### 09. 日本全国行政数据 LASSO/DNN，2019，Annals of the American Thoracic Society

题名：Machine Learning-Based Prediction Models for 30-Day Readmission after Hospitalization for Chronic Obstructive Pulmonary Disease。

- 数据：44,929 次 COPD 住院，3,413 次再入院；随机 70%/30% 开发与测试。
- 架构：传统逻辑回归参照、LASSO、DNN；当前只核实原始摘要，DNN 精确层数/宽度未核验。
- 任务及结果：30 日再入院；传统参照 C-statistic 0.57，LASSO 0.61，DNN 0.61。
- 边界：不是独立医院外部验证；不能用“数据大、网络深”保证强表现。

证据层级：原始摘要。[原始论文记录](https://pubmed.ncbi.nlm.nih.gov/31709851/)。

### 10. Re-Admit，2022，Scientific Reports

- 数据：Houston Methodist 八医院；3,005 人用于变量筛查，但完整四变量模型训练为 332 次患者就诊、前瞻时间验证 172 人。
- 架构：小型前馈 ANN，对单隐层 2–5 节点候选做比较；输入既往半年住院数、首日用药数量、保险、首日 Rothman Index。
- 任务：入院首日预测 30 日再入院；外部时间段 AUC 0.767、敏感度 0.75、特异度 0.67、PPV 0.61。
- 边界：应用显示风险百分位，百分位不是绝对风险；后续照护前后比较也不是随机试验的因果证据。保险变量不因此自动纳入本项目。

证据层级：原始全文 PDF。[原始论文](https://www.nature.com/articles/s41598-022-22434-3)。最终获选隐层节点数未完整核验，不猜。

### 11. 澳门决策树，2023，npj Primary Care Respiratory Medicine

- 数据：镜湖医院 2018–2019 年 782 次 AECOPD 住院，207 次 30 日再入院。
- 架构：临床特征、决策树、贝叶斯超参数优化；构建平衡样本，不是深度模型。
- 结果：平均 AUC 0.7506，precision 73.85%、recall 73.7%。
- 边界：单中心、平衡样本的 precision 不能直接外推原自然患病率；既往一年住院、吸烟和年龄仍重要。重复患者分离及全部采样顺序未完整核实。

证据层级：原始正文/摘要。[原始论文](https://www.nature.com/articles/s41533-023-00339-6)。

### 12. 加拿大七医院再入院建模，2026，MEDINFO 论文集

题名：Development of Multivariable Prediction Models for 30-Day Risk of Readmission After COPD Hospital Admission: A Retrospective Cohort Study Using Electronic Medical Record Data from 7 Hospitals。

- 数据：Toronto 七医院、16,314 名 AECOPD 患者，30 日全因再入院 15.4%；内部验证。
- 架构：六种算法；主要参照为 LASSO、逻辑回归、线性判别、XGBoost。
- 结果：四者 C-statistic 约 0.688 / 0.690 / 0.687 / 0.686；既往一年住院数最重要。
- 阈值：50% 阈值敏感度只有 14%–20%、特异度 96%–98%；20% 阈值敏感度 55%–59%、特异度 69%–73%。

证据层级：原始出版摘要，不冒称取得完整方法。[原始论文](https://journals.sagepub.com/doi/abs/10.3233/SHTI250988)。

## 4. 有新增信息载体的模型

### 13. 韩国每日加重预测，2023，Scientific Reports

- 数据：KOCOSS 590 人、18,945 条索赔记录，合并 HIRA、天气、污染、病毒资料；KNHANES 外部 2,151 人。
- 架构：LGBM/RF/XGBoost/MERF 等表格模型，采用 ADASYN；不是 Transformer。
- 结果：内部 LGBM AUC 0.935，外部 LGBM 0.720；外部 RF 0.737，Accuracy 0.766，但敏感度 0.992、特异度仅 0.390。
- 边界：内部按索赔记录而非患者划分；每日事件、记录采样及外部人群构造不同，不能把 0.935 写成可泛化的出院预测准确率。

证据层级：原始正文、表 4/原始 PDF；不使用二次综述替代。[原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10618439/)。不是建议加入用户禁止的高频 ICU 数据。

### 14. 定量 CT + ANN，2024，Frontiers in Medicine

题名：Predicting COPD exacerbations based on quantitative CT analysis: an external validation study。

- 数据：共 1,150 人，开发 519 / 内部 346 / 外部 285；两年随访。
- 架构：软件定量 CT 特征，加既往加重和 BODE；最终 ANN 消费五项特征：病史、BODE、受累肺容积比例、内脏脂肪、胸大肌面积。不是原始像素端到端 CNN。
- 结果：开发/内部/外部 AUC 0.90 / 0.88 / 0.86；主要任务为两年内至少一次加重。
- 边界：网络精确层宽在补充材料，本轮未完整取得；图像量化所需原始 CT 与报告文字不是同一种输入。

证据层级：原始正文。[原始论文](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2024.1370917/full)。

### 15. AI-PFT-Clin，2025，Journal of Medical Internet Research

题名：Deep Learning–Based Chronic Obstructive Pulmonary Disease Exacerbation Prediction Using Flow-Volume and Volume-Time Curve Imaging: Retrospective Cohort Study。

- 数据：两医院 10,492 次肺功能检查，开发 6,870 / 外部 3,622 次；按患者划分，检查次数不是患者人数。正文主表报告 1,489 人，但其他人数描述存在不一致，独立患者精确计数复现前须复核。
- 架构：流量–容积/容积–时间两路 patch/Transformer，自注意力编码、相加融合、MLP 双头；再将 AI 分数与临床变量经逻辑回归融合。
- 结果：一年中重度加重外部 AUC 0.730→0.755；重度 0.675→0.713。
- 启示：新增曲线形态信息有增量；不等于把稀少 FEV1 文字数字换成 Transformer 就能复制。

证据层级：原始正文，补充 DOCX 下载未成功。[原始论文](https://www.jmir.org/2025/1/e69785/)，DOI：10.2196/69785。

## 5. 文本模型参照：不是 COPD 专属任务

### 16. ClinicalBERT，2019 预印本 / 2020 CHIL workshop 稿

- 数据：MIMIC-III；论文队列 34,560 人，2,963 阳性及 42,358 阴性住院标签。
- 架构：BERT-base 12 层、768 维；临床预训练后监督微调，分类器 768→2048→768→1；长文分块后结合各块最大/平均概率。
- 任务与结果：30 日再入院，出院小结 AUROC 0.714±0.018，AUPRC 0.701，precision 固定 80% 时 recall 0.242。
- 边界：AUPRC/阈值数值须连同样本构造复核；此 Huang 模型不是 Alsentzer 的公开 Bio_ClinicalBERT，也不是已核实的 COPD-BERT。

证据层级：作者原始 PDF，含附录结构/训练设置。[原始论文](https://arxiv.org/abs/1904.05342)。微调三轮并早停；不是冻结向量后随意加小头的等价实验。

### 17. NYUTron，2023，Nature

正式题名：Health system-scale language models are all-purpose prediction engines。

- 数据：NYU 725 万篇笔记、387,144 人用于预训练；前瞻再入院验证 29,286 次出院。
- 架构：109M 参数、12 层/768 维/12 头 BERT-like 编码器；MLM 预训练、端到端任务微调和线性分类头，输入限制 512 token。
- 结果：30 日全因再入院时间测试 AUC 0.799；前瞻 AUC 0.787、recall 82.3%、precision 20.6%。
- 边界：非 COPD 专属；高 recall 伴大量误报。前瞻是单臂非干预验证，不是证明减少再入院。大规模院内训练不同于本项目冻结编码。

证据层级：原始正文和方法。[原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10338337/)，DOI：10.1038/s41586-023-06160-y。

## 6. 对本项目的判断：哪些可以借鉴，哪些不能直接复刻

以下为跨论文综合判断，不是任何论文保证的性能结论。

1. **历史强不是独有异常，但不能作为放弃新信息的理由。** 多项严格再入院研究中既往住院依然很强；应报告在它之外的真实增量与层内校准，不能强行压制后称其余变量变重要。
2. **读出方式与信息载体要分开。** 新小头失败只证明某种接入方式失败；曲线/图像确有新信息时，深度编码可能有效。把同一份已经压缩的向量反复换头不能等同充分探索原始信息。
3. **不把论文里的微调收益套给冻结 BERT。** 两者训练对象、规模、标签和可用性不同。可以研究安全片段、分块概率融合或融合前可学习读取，但本轮不启动训练，也不推翻用户既定不微调 BERT 的边界。
4. **可做共同任务对照，但不能暗中换题。** 将成熟 LR/树模型/临床评分在本地同患者划分和同标签上重估，才是算法对照；直接拿论文 AUC 0.70 和本地 0.80 排名无效。PEARL、ACCEPT 若缺可靠 mMRC/肺功能等输入，也不能填正常后冒称忠实复现。
5. **真正临床交付应是概率与行动表。** 保留每天连续竞争风险输出，主报告补充 30/90/180/365 日预测-观察校准、Brier、按事先阈值的 sensitivity/PPV/每百人预警数及净获益。累计到某天发生事件的概率与恰好那天发生的概率分开。
6. **当前终点必须如实命名。** 本项目目前依据院内严格编码的后续住院，不能声称捕获全部门诊/院外急性加重。原 FINAL 文本也不能冒称出院瞬时已可用；随访范围、竞争死亡及源可见性仍需单独报告。
7. **下一实验优先级建议，不是已实现：** 优先完成已计划的融合前方案输入、只冻结 H34 的匹配快筛，避免再次只测全 V6 冻结后追加头；同时以校准/概率误差为必须交付，而非仅 AUC。新队列任务、影像/PFT 原始载体、BERT 适配另需可用数据与明确授权，不自动扩范围。

## 7. 研究质量警示与尚未解决的问题

- 本次是定向原始研究扩展，不是预注册的系统综述或 meta-analysis；不能保证穷尽全部文献。
- 摘要样本量可能是源库、筛查人数或检查次数，不能当最终独立患者量。
- 论文声称“外部验证”不自动意味着该集合从未参与最终模型选择；表型标签也可能在摘要/正文不一致。
- 真正的临床概率表现需要自然事件比例与完整随访，平衡抽样的 PPV、Accuracy 和 AUPRC 不可直接移植。
- 本轮未取得所有补充材料、代码和原始数据，没有完整重现任何外部模型，也不报告“已证明优于/劣于 V6”。
- 本文只保存公开来源的摘要性核对与研究判断，不含患者原文、身份、访问凭据或新的病历导出。
````
<!-- END REVIEW_FRAGMENT -->

## docs/PUBLIC_DATA_SNAPSHOT_2026-10-15.md · lines 1–23

<!-- REVIEW_FRAGMENT {"path":"docs/PUBLIC_DATA_SNAPSHOT_2026-10-15.md","part":1,"start_line":1,"end_line":23,"source_sha256":"ea6fa6ee3fb78bef0d74084780b74be6358ea14a0452b077f2ccbee8dae0f4a0","fragment_sha256":"ea6fa6ee3fb78bef0d74084780b74be6358ea14a0452b077f2ccbee8dae0f4a0","added_terminal_newline":false} -->
````markdown
# 数据与实验汇总快照：公开范围及阅读方法

2026-10-15。公开快照包括文件级来源、表/队列规模、数据字典、数组字段与shape/dtype、文件级SHA256和历史汇总指标。没有导出数组取值或患者行；元数据不能用来代替受控原始数据，也不代表获得病历再分发许可。

| 预测队列划分 | 出院次数 | 患者数 |
|---|---:|---:|
| train | 13,592 | 3,766 |
| tune | 1,531 | 402 |
| confirmation | 1,418 | 395 |
| historical test | 2,860 | 788 |
| 合计 | 19,401 | 5,351 |

同患者不跨划分。原验证集为2,949次出院/797患者，按历史固定患者哈希拆为调参和确认。确认/历史测试曾被项目早期使用，不冒称新盲测。当前披露历史已有汇总结果，没有为本次发布重新评估患者结局。

原始导出底账保留19,774次COPD编码住院/5,390患者，预测资格为19,401次出院/5,351患者，两者不是同一个计数范围。17张原始来源表的规模/文件级hash，以及66份派生NPZ的字段、形状、dtype见`docs/public_snapshot/DATA_SNAPSHOT.json`，均已纳入Markdown分卷。NPZ仅解压文件头，没有读取数组行或导出逐人标签/划分。结构目录含历史中间/失败版本；出现在目录不表示已通过验证或正在使用，实际输入以运行合同和对应manifest为准。

`docs/public_snapshot/AGGREGATE_RESULTS.json`索引146个实验目录，从658份指定命名的汇总报告中提取70,421个数值指标条目。每条保留原报告相对路径及hash；不根据目录存在断言实验完成。字段提取排除个体键、样本示例、逐人预测、数组与自由文本；除显式两端置信区间外，不递归导出列表记录，以免把无法判定聚合层级的项目混入。2份报告没有命中所选指标字段，仍保留来源记录。历史逐epoch完整日志、检查点和患者级原始报告保留本地，因此这里的“汇总快照”不是原始运行目录的全量副本。

结果按原实验/模型路径存放，不把不同划分、结局、风险集或版本的指标混成排行榜。当前V7主比较为同一tune的365天严格AECOPD首再入院及其前的竞争死亡；IBS为7–365日积分Brier，不能把WorldA本库记录终点冒称完整院外随访。

此前公开的全部文本源码和本次增量进入[Markdown阅读包](../review_bundle/00_READ_FIRST.md)。源码按原字节写入，长文件按行切分；`source_catalog.json`列出范围，`manifest.json`绑定分卷hash。执行`python tools/build_markdown_review.py --verify`核对分卷及逐字节还原，不需要病历或训练权重。

仍未公开的9份历史原型含需保留本地的病例引用，清单见阅读包首页。可重装依赖、缓存、重复发布现场、原始/派生患者数据、向量与权重不进入公开包。原始受限文件的hash只标记来源身份，不提供文件下载。
````
<!-- END REVIEW_FRAGMENT -->

## docs/PUBLIC_GITIGNORE.txt · lines 1–90

<!-- REVIEW_FRAGMENT {"path":"docs/PUBLIC_GITIGNORE.txt","part":1,"start_line":1,"end_line":90,"source_sha256":"0b228e7850e17b86534260fc63f07e24ee3d4b9ff01359b9d8e48775988f6861","fragment_sha256":"0b228e7850e17b86534260fc63f07e24ee3d4b9ff01359b9d8e48775988f6861","added_terminal_newline":false} -->
````text
# Public repository: source and aggregate documents only.
# Never add MIMIC data, identifiers, predictions, embeddings or model weights.
.env
.env.*
*.env
*.pem
*.key
*.p12
*.pfx
.pgpass
pgpass.conf
*credentials*
*secrets*
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
node_modules/
.idea/
.vscode/
*.log
*.out
*.tmp
*.bak
*.csv
*.tsv
*.json
*.jsonl
*.ndjson
*.parquet
*.arrow
*.feather
*.sqlite*
*.db*
*.gz
*.bz2
*.xz
*.zip
*.7z
*.tar
*.rar
*.npy
*.npz
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
*.gguf
*.onnx
*.pkl
*.pickle
*.joblib
*.h5
*.hdf5
*.png
*.jpg
*.jpeg
*.webp
*.pdf
*.pptx
*.docx
*.xlsx
/outputs/
/training_runs/
/transfer/
/Bonsai2_Extraction_Pipeline/
/.agents/
/.build_architecture/
/.codex-finalizer-architecture/
/training/model_cache/
/training/cv/
/training/reports/
/training/runtime/*/
/patient_data/**
!/patient_data/**/
!/patient_data/*.py
!/patient_data/clinical_facts/*.py
/work/**
!/work/prepare_public_snapshot.py
!/config/clinical_feature_contract.csv
!/config/clinical_feature_contract_v2_diagnosis.csv

# Reviewed aggregate-only V7 figures, not patient images or record-level outputs.
!/docs/figures/v7_first_experiment_2026-10-15.png
!/docs/figures/v7_loss_comparison_2026-10-15.png
!/docs/figures/v7_sequential_research_2026-10-15.png
````
<!-- END REVIEW_FRAGMENT -->

## docs/PUBLIC_NAVIGATION.md · lines 1–24

<!-- REVIEW_FRAGMENT {"path":"docs/PUBLIC_NAVIGATION.md","part":1,"start_line":1,"end_line":24,"source_sha256":"bfb7d34cede08035407a1035362b8bd92c9c3603067a60edb041a57cd423ce51","fragment_sha256":"bfb7d34cede08035407a1035362b8bd92c9c3603067a60edb041a57cd423ce51","added_terminal_newline":false} -->
````markdown
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
````
<!-- END REVIEW_FRAGMENT -->

## docs/PUBLIC_README.md · lines 1–80

<!-- REVIEW_FRAGMENT {"path":"docs/PUBLIC_README.md","part":1,"start_line":1,"end_line":80,"source_sha256":"b555cf35890530f9f583839b85e3e18553cc83a4f46c78465a08c655849804ae","fragment_sha256":"b555cf35890530f9f583839b85e3e18553cc83a4f46c78465a08c655849804ae","added_terminal_newline":false} -->
````markdown
# AECOPD：出院后再入院与竞争死亡风险研究

## 最新研究结果（2026-10-15）

新增[完整Markdown阅读包](../review_bundle/00_READ_FIRST.md)，替代此前00–07 TXT摘录：覆盖当前全部已审查公开文本源码、测试、协议与文档，并附历史实验汇总指标目录和[数据结构快照](../docs/PUBLIC_DATA_SNAPSHOT_2026-10-15.md)。各文件正文按主题拆卷，每段带原文件路径、行范围和SHA256，可逐字节重建。患者行、个体标签/预测、向量、权重与本地旧Git历史保留本地。

最新[强V6上关闭跨源Query](../docs/V7_STRONG_NO_CROSS_RESULTS_2026-10-15.md)已完成seed42/7：F1一次升、一次降，未稳定复现；[固定90%V6＋10%Cox](../docs/V7_FIXED_V6_COX_BLEND_RESULTS_2026-10-15.md)无需模型重训，两原因IBS改善、F1略降。主/辅助检查点分开保存并独立重放，默认模型保留。

追加完成[完整V6冻结＋端到端V7＋H34训练集去重](../docs/V7_PROJECTED_STRONG_RESULTS_2026-10-15.md)：seed42实际16轮，F1 .578879低于不去重的.581683，IBS .099701略低；best/last映射与预测独立重放0，按事前规则停止此配方。

[V7现有研究线索顺序实验](../docs/V7_SEQUENTIAL_RESEARCH_RESULTS_2026-10-15.md)已完成56次新拟合、519神经epoch及独立重放。完整强V6冻结后的V7修正、固定训练投影、物理检验/药物动态、临床软路由、损失/Query/衰减和两折OOF均有逐项结论；小幅F1收益伴随不确定性与概率误差权衡，原默认保留。公开部分仅源码、合成测试、合同和聚合结果。

本仓库公开**研究源码、队列/任务定义、测试和经审查的汇总说明**。不提供MIMIC病历、患者标识、患者级标签/划分/预测、编码向量、模型权重或数据库凭据。数据使用必须由使用者自行取得PhysioNet授权并遵守相关协议。

## 最新：V7包含检验的多源纵向模型

V7已完成结构实现和有限首轮：统一256维、8头、M4、FFN1024，冻结BERT/RadBERT；独立检验分支、药物因子化、药物/影像Query和完整历史时间衰减。字段M4压缩旁路问题已修复，修复后从头训练并独立重放。

此前只看到文本、影像与药物的建议，需要按已存在的36个检验异常方向×9通道重新审查。检验已进入融合与文本—检验对比；首轮默认的药物Query不直接读取检验；顺序补充实验已实现该候选连接并训练，未超过初值。阅读[检验已知时的V7适配说明与实际检查状态](../docs/V7_LAB_AWARE_DESIGN_AND_STATUS_2026-10-15.md)。

[首轮结果](../docs/V7_FIRST_EXPERIMENT_RESULTS_2026-10-15.md)：V7原竞争风险损失对照年度F1 .565321，加入Focal＋标签平滑＋.1对比为 .567023，未超过已有V6对称背景 .577373；不据此替换默认。[损失补充比较](../docs/V7_LOSS_COMPARISON_RESULTS_2026-10-15.md)：再补三次实际训练，直接Focal＋平滑（有/无对比）均未超过初值；Focal与平滑各自作用、Query和衰减独立重训尚未完成。不能把当前实现的失败一般性归因于某个方法。

公开两张图仅含逐轮汇总指标与时间，不含患者级数据。真实训练产物仍不提供；合成测试与受控复现入口见[training/v7/README.md](../training/v7/README.md)。

## 当前任务与保留的V6参照

以每次合格出院为预测时点，预测未来1–365日首次严格AECOPD非转院再入院，以及先于该再入院发生的竞争死亡。模型输出连续时间风险及逐日累计概率，不是确切事件日期、不是诊断COPD，也不包含再入院后总死亡。MIMIC库内未观察到再入院不代表完整院外阴性。

- 原V6：冻结H34背景基础＋可训练临床风险修正；检验、七药效、微生物、三呼吸支持汇总、四字段冻结BERT向量。
- H34：历史6维＋本次诊断/去向14值＋14缺失标记。不是完整V6，不包含检验/文本；冻结是权重不更新，患者基础风险不同。
- 纵向候选：共享住院编码器读取历次已结束合格住院的现有临床资料，历史文书按存档可见性隔离；与本次住院一起进行纵向比较。
- 最新对称背景候选：本次与过去都先结合各自背景，再进入纵向比较；128维宽度、检验输入、冻结基础不变。实际单seed配对校正结论见[结果](../docs/V6_SYMMETRIC_CONTEXT_RESULTS_2026-10-14.md)。

这是回顾性研究，不是可直接部署的临床预警系统。单seed调参结果不等独立临床有效性；确认/历史测试已被项目使用，不冒称新盲测。默认正式模型未因弱候选自动替换，长期自动探索已取消。

## 阅读顺序

0. [V7架构、实际检验输入与待验证事项](../docs/V7_LAB_AWARE_DESIGN_AND_STATUS_2026-10-15.md)、[首轮结果](../docs/V7_FIRST_EXPERIMENT_RESULTS_2026-10-15.md)、[损失组成比较](../docs/V7_LOSS_COMPARISON_RESULTS_2026-10-15.md)。
1. [完整输入说明](../docs/V6_CURRENT_INPUT_CATALOG_2026-10-11.md)：旧V6字段、来源、单位、缺失及内部维度。
2. [统一检验与纵向实验](../docs/V6_UNIFIED_LONGITUDINAL_F1_RESULTS_2026-10-14.md)：四条件有限实验、实际轮数及校准/F1取舍。
3. [多期限共同评价](../docs/V6_MULTIHORIZON_READOUT_RESULTS_2026-10-14.md)：7/14/30/60/90/180/365日、全年平均AUC、概率误差及固定阈值。
4. [最新对称背景校正](../docs/V6_SYMMETRIC_CONTEXT_RESULTS_2026-10-14.md)：同初始化/输入的针对性实验。
5. [通俗流程与报警文献](../docs/V6_HISTORY_AND_ALERT_EXPLAINED_2026-10-14.md)、[17项原始研究核对](../docs/COPD_PREDICTION_PRIMARY_STUDIES_REVIEW_2026-10-14.md)。
6. [公开范围与本地资产边界](../docs/PUBLICATION_BOUNDARY_2026-10-14.md)。

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
````
<!-- END REVIEW_FRAGMENT -->

## docs/public_snapshot/DATA_SNAPSHOT.json · lines 1–5332

<!-- REVIEW_FRAGMENT {"path":"docs/public_snapshot/DATA_SNAPSHOT.json","part":1,"start_line":1,"end_line":5332,"source_sha256":"218ec2b7db8f4cb43d7b42cf74ae932e935e79dd60e3126cd91e370b4fcda9c7","fragment_sha256":"218ec2b7db8f4cb43d7b42cf74ae932e935e79dd60e3126cd91e370b4fcda9c7","added_terminal_newline":false} -->
````json
{
  "scope": "aggregate metadata and array headers only; no patient rows, ids, labels or values",
  "raw_manifest_status": "validated",
  "raw_manifest_sha256": "f458ef8da5a7724dbfb22329e68690774e996daf3cadd6aed765602881e85f0c",
  "raw_export_utc_started": "2026-09-27T13:35:44.447014+00:00",
  "raw_export_utc_finished": "2026-09-27T13:39:25.977338+00:00",
  "raw_sources": {
    "cohort/admissions": {
      "source_table": "project_aecopd.copd_coded_admissions_v1_gate",
      "sha256": "52b0199bb08970f83f0de48dbac6f5c4a68298fede3ccd45ef8b3e77a5c029dd",
      "compressed_bytes": 677972,
      "rows": 19774
    },
    "raw/patients": {
      "source_table": "mimiciv_hosp.patients",
      "sha256": "95fc69b203f2ded10e462f4bdff7079f268a6cd066b6fdf5b630cd63083f8c8b",
      "compressed_bytes": 60124,
      "rows": 5390
    },
    "raw/admissions": {
      "source_table": "mimiciv_hosp.admissions",
      "sha256": "f142107021294575dea5cb35fea3e4771268018dcc18bcd9c6239fd1259d374d",
      "compressed_bytes": 865167,
      "rows": 19774
    },
    "raw/labevents": {
      "source_table": "mimiciv_hosp.labevents",
      "sha256": "14308cecfd9fc0d08538b415ac0c6a95a4d81a3624e441526040c932fc89aa4e",
      "compressed_bytes": 61954298,
      "rows": 3152512
    },
    "raw/diagnoses_icd": {
      "source_table": "mimiciv_hosp.diagnoses_icd",
      "sha256": "008eb7d4084ddae5f3afd40224c08e77dbc7e37b7630a9e74d1bea0b3e4b83e0",
      "compressed_bytes": 2052701,
      "rows": 307709
    },
    "raw/emar": {
      "source_table": "mimiciv_hosp.emar",
      "sha256": "242dbe2206da25e19a9e90415c89a4ceab9aaf4aeddc275eb60d4f6eee25368e",
      "compressed_bytes": 43800913,
      "rows": 1926692
    },
    "raw/prescriptions": {
      "source_table": "mimiciv_hosp.prescriptions",
      "sha256": "aec14dd54baf83606aba47a56e7f4b05b7bd9485e14b513203fb0b7b0afc27c4",
      "compressed_bytes": 31404602,
      "rows": 861954
    },
    "raw/pharmacy": {
      "source_table": "mimiciv_hosp.pharmacy",
      "sha256": "9d0cc625c5ff19d2b2699f58facaccaeeb52abb71a08502fc314be7fd4fe1164",
      "compressed_bytes": 28993507,
      "rows": 780177
    },
    "raw/emar_detail": {
      "source_table": "mimiciv_hosp.emar_detail",
      "sha256": "0a435c0ef4d45dc4587348782634bebd0041e9e026cc798f8bd90870da8eae29",
      "compressed_bytes": 87047026,
      "rows": 3949754
    },
    "raw/discharge": {
      "source_table": "mimiciv_note.discharge",
      "sha256": "087a266a5b027cd5062c7d7b87dc15c1178e147df940124ae7b82d43f7fcbd11",
      "compressed_bytes": 60702289,
      "rows": 14480
    },
    "raw/radiology": {
      "source_table": "mimiciv_note.radiology",
      "sha256": "530cceb478df0e8f4fca4fc703038382619e04444e4717de12c3b0208a62ee4c",
      "compressed_bytes": 17602233,
      "rows": 47117
    },
    "raw/labevents_unassigned": {
      "source_table": "mimiciv_hosp.labevents",
      "sha256": "a162f33888b09fef4e7ee1aa86734284e2b5d64bd7fb8c048fb7fdd9e3bd5982",
      "compressed_bytes": 1053444,
      "rows": 47882
    },
    "raw/d_labitems": {
      "source_table": "mimiciv_hosp.d_labitems",
      "sha256": "f856bd2a33a163902b95d326bfa3d7c11b435f7ab73d01b3b50ba0de8ec57200",
      "compressed_bytes": 15250,
      "rows": 1650
    },
    "raw/d_icd_diagnoses": {
      "source_table": "mimiciv_hosp.d_icd_diagnoses",
      "sha256": "d025ef991579197ed930cdd5de3031428bacd55e0e4af0df37e9533c11a43309",
      "compressed_bytes": 1035477,
      "rows": 112107
    },
    "outcomes/readmission_death_labels": {
      "source_table": "project_aecopd.copd_qualified_readmission_labels_v1",
      "sha256": "779267ae9ee4dedbb5412c17c35cd136e799f168cf9a1f236ffc6f4c33e0577a",
      "compressed_bytes": 783800,
      "rows": 19401
    },
    "metadata/columns": {
      "source_table": "information_schema.columns",
      "sha256": "9f0e1745ab7abf52cabc8c687ed1b9839d4dfb456d0396e401a6af7930dbd77a",
      "compressed_bytes": 2381,
      "rows": 203
    },
    "cohort/patient_splits": {
      "sha256": "bfb95a89d06df2e87adbe57ad4835d82c09c50fda9e255a86f8f13607b117b8e",
      "rows": 5390
    }
  },
  "modeling_split": {
    "total_discharges": 19401,
    "total_patients": 5351,
    "groups": {
      "train": {
        "discharges": 13592,
        "patients": 3766
      },
      "tune": {
        "discharges": 1531,
        "patients": 402
      },
      "confirmation": {
        "discharges": 1418,
        "patients": 395
      },
      "test": {
        "discharges": 2860,
        "patients": 788
      }
    },
    "validation_split": "historical patient SHA256 parity split into tune and confirmation",
    "patients_disjoint": true,
    "confirmation_test_outcomes_evaluated": false
  },
  "processed_interfaces": [
    {
      "path": "patient_data/processed/abnormality_clinical_reviewed_2026-09-29/abnormality.npz",
      "bytes": 1115210,
      "sha256": "65e10cf27c56673143cba1f9a4783430fd7bfd4ebb99a259f5f14fe80d2e9fd6",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "features",
          "shape": [
            19401,
            132
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 10243856
        },
        {
          "name": "feature_names",
          "shape": [
            132
          ],
          "dtype": "<U53",
          "fortran_order": false,
          "uncompressed_bytes": 28112
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/abnormality_only_2026-09-29/abnormality.npz",
      "bytes": 1096473,
      "sha256": "971d2093e379cd8e9c4079a271f9390564a77c23713b21b0390a8c6bbf3aeadc",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "features",
          "shape": [
            19401,
            132
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 10243856
        },
        {
          "name": "feature_names",
          "shape": [
            132
          ],
          "dtype": "<U53",
          "fortran_order": false,
          "uncompressed_bytes": 28112
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/abnormality_uncapped_2026-09-29/abnormality.npz",
      "bytes": 1115210,
      "sha256": "65e10cf27c56673143cba1f9a4783430fd7bfd4ebb99a259f5f14fe80d2e9fd6",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "features",
          "shape": [
            19401,
            132
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 10243856
        },
        {
          "name": "feature_names",
          "shape": [
            132
          ],
          "dtype": "<U53",
          "fortran_order": false,
          "uncompressed_bytes": 28112
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/lab_recovery_2026-09-29/lab_recovery.npz",
      "bytes": 592648,
      "sha256": "bf4234262bbc8b2f6ad4e4209dc12d651d79e09605f7cf4e5b4fc39f0d76b002",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "features",
          "shape": [
            19401,
            54
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 4190744
        },
        {
          "name": "feature_names",
          "shape": [
            54
          ],
          "dtype": "<U53",
          "fortran_order": false,
          "uncompressed_bytes": 11576
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/lab_recovery_clinical_reviewed_2026-09-29/lab_recovery.npz",
      "bytes": 596149,
      "sha256": "4f280abf4219e8cd6b0e851fd53650099048171142d64b64b77f5edef0e82daf",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "features",
          "shape": [
            19401,
            54
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 4190744
        },
        {
          "name": "feature_names",
          "shape": [
            54
          ],
          "dtype": "<U53",
          "fortran_order": false,
          "uncompressed_bytes": 11576
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/lab_recovery_uncapped_2026-09-29/lab_recovery.npz",
      "bytes": 596149,
      "sha256": "4f280abf4219e8cd6b0e851fd53650099048171142d64b64b77f5edef0e82daf",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "features",
          "shape": [
            19401,
            54
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 4190744
        },
        {
          "name": "feature_names",
          "shape": [
            54
          ],
          "dtype": "<U53",
          "fortran_order": false,
          "uncompressed_bytes": 11576
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_2026-09-27/event_arrays.npz",
      "bytes": 2569255,
      "sha256": "cb07c790edd8e899acdafe74437ac540d973a36edd9d76efe17f079e1c064ba0",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            828367
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3313596
        },
        {
          "name": "values",
          "shape": [
            828367
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3313596
        },
        {
          "name": "numeric_mask",
          "shape": [
            828367
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 828495
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            828367
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3313596
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_2026-09-27_incomplete_audit/event_arrays.npz",
      "bytes": 2569255,
      "sha256": "cb07c790edd8e899acdafe74437ac540d973a36edd9d76efe17f079e1c064ba0",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            828367
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3313596
        },
        {
          "name": "values",
          "shape": [
            828367
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3313596
        },
        {
          "name": "numeric_mask",
          "shape": [
            828367
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 828495
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            828367
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3313596
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29/event_arrays.npz",
      "bytes": 2560244,
      "sha256": "994c0027efa86a3099a693093583e896a159289f1ad26becc9d34c1930c7fc5b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            825644
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3302704
        },
        {
          "name": "values",
          "shape": [
            825644
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3302704
        },
        {
          "name": "numeric_mask",
          "shape": [
            825644
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 825772
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            825644
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3302704
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_formulation_verified_2026-09-29/event_arrays.npz",
      "bytes": 2559967,
      "sha256": "a20d4e8cdc5a7d86b77ef9d5f8e388515f878e1be6c885136e79eb383eaac451",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            825766
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3303192
        },
        {
          "name": "values",
          "shape": [
            825766
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3303192
        },
        {
          "name": "numeric_mask",
          "shape": [
            825766
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 825894
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            825766
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3303192
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_gold_review_v2_2026-09-29/event_arrays.npz",
      "bytes": 2560811,
      "sha256": "0fca2ecc7d166aa72b00fb5d463d7ba3ba046aeb7a99679d31ab33632c299ac9",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            825776
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3303232
        },
        {
          "name": "values",
          "shape": [
            825776
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3303232
        },
        {
          "name": "numeric_mask",
          "shape": [
            825776
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 825904
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            825776
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3303232
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_gold_review_v2_checked_2026-09-29/event_arrays.npz",
      "bytes": 2560811,
      "sha256": "0fca2ecc7d166aa72b00fb5d463d7ba3ba046aeb7a99679d31ab33632c299ac9",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            825776
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3303232
        },
        {
          "name": "values",
          "shape": [
            825776
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3303232
        },
        {
          "name": "numeric_mask",
          "shape": [
            825776
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 825904
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            825776
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3303232
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_invalid_regex_2026-09-29/event_arrays.npz",
      "bytes": 2523971,
      "sha256": "201e01ac8c8904974d84039c128e9a2b13c0c89d2b209bafd4570094f8eb7f8e",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            815884
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3263664
        },
        {
          "name": "values",
          "shape": [
            815884
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3263664
        },
        {
          "name": "numeric_mask",
          "shape": [
            815884
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 816012
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            815884
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3263664
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_medication_screened_2026-09-29/event_arrays.npz",
      "bytes": 2560181,
      "sha256": "c1aedff2678578f49ccce7eb34f9aa32ea8dc3ade7ecb83dadbb426537fcd116",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            825894
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3303704
        },
        {
          "name": "values",
          "shape": [
            825894
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3303704
        },
        {
          "name": "numeric_mask",
          "shape": [
            825894
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 826022
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            825894
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3303704
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_needs_local_form_fix_2026-09-29/event_arrays.npz",
      "bytes": 2560486,
      "sha256": "740c3c32d7a6c6291d8e441be0c29ec7fa166dfd21752a49dbf6a3211d73afb6",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            825991
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3304092
        },
        {
          "name": "values",
          "shape": [
            825991
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3304092
        },
        {
          "name": "numeric_mask",
          "shape": [
            825991
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 826119
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            825991
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3304092
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_audited_coarse_fine_2026-10-12/interface.npz",
      "bytes": 6993064,
      "sha256": "7fccc944bb19a87951efd43a14a89a508cef999895012098c85ca4a6e2f9d3fc",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "labs",
          "shape": [
            19401,
            34,
            24
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 63324992
        },
        {
          "name": "lab_present",
          "shape": [
            19401,
            34
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 659762
        },
        {
          "name": "lab_names",
          "shape": [
            34
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 4072
        },
        {
          "name": "lab_family",
          "shape": [
            34
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 400
        },
        {
          "name": "lab_keep",
          "shape": [
            34
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 162
        },
        {
          "name": "medications",
          "shape": [
            19401,
            38,
            11
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 32438600
        },
        {
          "name": "med_present",
          "shape": [
            19401,
            38
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 737366
        },
        {
          "name": "med_names",
          "shape": [
            38
          ],
          "dtype": "<U25",
          "fortran_order": false,
          "uncompressed_bytes": 3928
        },
        {
          "name": "med_keep",
          "shape": [
            38
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 166
        },
        {
          "name": "drug_parent",
          "shape": [
            38,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 1648
        },
        {
          "name": "coarse",
          "shape": [
            19401,
            10,
            2
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 1552208
        },
        {
          "name": "micro",
          "shape": [
            19401,
            8
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 620960
        },
        {
          "name": "micro_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "support",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 465752
        },
        {
          "name": "support_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_audited_coarse_fine_2026-10-12_r2/interface.npz",
      "bytes": 6983199,
      "sha256": "19def1f6c4e22763db514ed9155cafbf09682a7ca0c1d6721c154f76a9361f06",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "labs",
          "shape": [
            19401,
            34,
            24
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 63324992
        },
        {
          "name": "lab_present",
          "shape": [
            19401,
            34
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 659762
        },
        {
          "name": "lab_names",
          "shape": [
            34
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 4072
        },
        {
          "name": "lab_family",
          "shape": [
            34
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 400
        },
        {
          "name": "lab_keep",
          "shape": [
            34
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 162
        },
        {
          "name": "medications",
          "shape": [
            19401,
            38,
            11
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 32438600
        },
        {
          "name": "med_present",
          "shape": [
            19401,
            38
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 737366
        },
        {
          "name": "med_names",
          "shape": [
            38
          ],
          "dtype": "<U25",
          "fortran_order": false,
          "uncompressed_bytes": 3928
        },
        {
          "name": "med_keep",
          "shape": [
            38
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 166
        },
        {
          "name": "drug_parent",
          "shape": [
            38,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 1648
        },
        {
          "name": "coarse",
          "shape": [
            19401,
            10,
            2
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 1552208
        },
        {
          "name": "micro",
          "shape": [
            19401,
            8
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 620960
        },
        {
          "name": "micro_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "support",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 465752
        },
        {
          "name": "support_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_bonsai_tasks_2026-10-10/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_bonsai_tasks_2026-10-10_r2/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_bonsai_tasks_2026-10-10_r3/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_bonsai_tasks_2026-10-10_validated/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_cbc_combinations_2026-10-10/combinations.npz",
      "bytes": 1565183,
      "sha256": "fdafca6b31e1b7d9a81279bd3b8321de37c7415b0cd3a3602927a3788caceeeb",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "values",
          "shape": [
            19401,
            5,
            9
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3492308
        },
        {
          "name": "raw",
          "shape": [
            19401,
            5,
            9
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 6984488
        },
        {
          "name": "names",
          "shape": [
            5
          ],
          "dtype": "<U19",
          "fortran_order": false,
          "uncompressed_bytes": 508
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_chexbert_cxr_2026-10-09/cxr_context.npz",
      "bytes": 143564,
      "sha256": "0eebf16d9a78271032286ab3e4e17a6c6a54b90a1215947b6ed5354034afdcec",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "cxr_states",
          "shape": [
            19401,
            7
          ],
          "dtype": "|i1",
          "fortran_order": false,
          "uncompressed_bytes": 135935
        },
        {
          "name": "cxr_onehot",
          "shape": [
            19401,
            28
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 2173040
        },
        {
          "name": "cxr_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "condition_names",
          "shape": [
            7
          ],
          "dtype": "<U16",
          "fortran_order": false,
          "uncompressed_bytes": 576
        },
        {
          "name": "state_names",
          "shape": [
            4
          ],
          "dtype": "<U9",
          "fortran_order": false,
          "uncompressed_bytes": 272
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_clinical_state_2026-10-09/clinical_state.npz",
      "bytes": 6070435,
      "sha256": "e5ae5208ad930fafb55c403210dd448755f3c137be79aaece9d911d7dce4a8d9",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "lab_axis_raw",
          "shape": [
            19401,
            22,
            9
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 30731312
        },
        {
          "name": "lab_axis_inputs",
          "shape": [
            19401,
            22,
            9
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 15365720
        },
        {
          "name": "lab_value_masks",
          "shape": [
            19401,
            22,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 2561060
        },
        {
          "name": "axis_names",
          "shape": [
            22
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 2680
        },
        {
          "name": "feature_names",
          "shape": [
            9
          ],
          "dtype": "<U24",
          "fortran_order": false,
          "uncompressed_bytes": 992
        },
        {
          "name": "peak_audit_days",
          "shape": [
            19401,
            22,
            3
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 10243856
        },
        {
          "name": "med_exposure",
          "shape": [
            19401,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 543356
        },
        {
          "name": "med_group_names",
          "shape": [
            7
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 940
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_clinical_state_2026-10-09_r2/clinical_state.npz",
      "bytes": 6070435,
      "sha256": "e5ae5208ad930fafb55c403210dd448755f3c137be79aaece9d911d7dce4a8d9",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "lab_axis_raw",
          "shape": [
            19401,
            22,
            9
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 30731312
        },
        {
          "name": "lab_axis_inputs",
          "shape": [
            19401,
            22,
            9
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 15365720
        },
        {
          "name": "lab_value_masks",
          "shape": [
            19401,
            22,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 2561060
        },
        {
          "name": "axis_names",
          "shape": [
            22
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 2680
        },
        {
          "name": "feature_names",
          "shape": [
            9
          ],
          "dtype": "<U24",
          "fortran_order": false,
          "uncompressed_bytes": 992
        },
        {
          "name": "peak_audit_days",
          "shape": [
            19401,
            22,
            3
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 10243856
        },
        {
          "name": "med_exposure",
          "shape": [
            19401,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 543356
        },
        {
          "name": "med_group_names",
          "shape": [
            7
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 940
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_compact_multisource_2026-10-09/compact_context.npz",
      "bytes": 238971,
      "sha256": "9c5ed7e38f6f5b81776d54d045c3ef9c1dc3ec97b84f7a54dadd992801c92da4",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "axis_names",
          "shape": [
            22
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 2680
        },
        {
          "name": "med_features",
          "shape": [
            19401,
            7,
            4
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 2173040
        },
        {
          "name": "med_group_names",
          "shape": [
            7
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 940
        },
        {
          "name": "med_channel_names",
          "shape": [
            4
          ],
          "dtype": "<U25",
          "fortran_order": false,
          "uncompressed_bytes": 528
        },
        {
          "name": "med_pharmacy_audit",
          "shape": [
            19401,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 543356
        },
        {
          "name": "med_linked_rx_audit",
          "shape": [
            19401,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 543356
        },
        {
          "name": "med_unlinked_rx_audit",
          "shape": [
            19401,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 543356
        },
        {
          "name": "micro_values",
          "shape": [
            19401,
            5
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 388148
        },
        {
          "name": "micro_masks",
          "shape": [
            19401,
            5
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 97133
        },
        {
          "name": "micro_names",
          "shape": [
            5
          ],
          "dtype": "<U32",
          "fortran_order": false,
          "uncompressed_bytes": 768
        },
        {
          "name": "support_values",
          "shape": [
            19401,
            3
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 232940
        },
        {
          "name": "support_masks",
          "shape": [
            19401,
            3
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 58331
        },
        {
          "name": "support_names",
          "shape": [
            3
          ],
          "dtype": "<U28",
          "fortran_order": false,
          "uncompressed_bytes": 464
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_cxr_reports_validated_2026-10-09/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_cxr_selected_2026-10-09/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_cxr_selected_2026-10-09_r2/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_cxr_trajectory_2026-10-10/cxr_trajectory.npz",
      "bytes": 172999,
      "sha256": "f059feadfd49eea66952ab09b60f61c830c2e11a039c937522f4c2b898f9234f",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "cxr_values",
          "shape": [
            19401,
            98
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 7605320
        },
        {
          "name": "cxr_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "condition_names",
          "shape": [
            7
          ],
          "dtype": "<U16",
          "fortran_order": false,
          "uncompressed_bytes": 576
        },
        {
          "name": "feature_names",
          "shape": [
            14
          ],
          "dtype": "<U23",
          "fortran_order": false,
          "uncompressed_bytes": 1416
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_cxr_trajectory_2026-10-10_r2/cxr_trajectory.npz",
      "bytes": 175857,
      "sha256": "00e6528fd68700f5d92c21fe89d195b24875694c014f012f47df737bfb2fb3b8",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "cxr_values",
          "shape": [
            19401,
            98
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 7605320
        },
        {
          "name": "cxr_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "condition_names",
          "shape": [
            7
          ],
          "dtype": "<U16",
          "fortran_order": false,
          "uncompressed_bytes": 576
        },
        {
          "name": "feature_names",
          "shape": [
            14
          ],
          "dtype": "<U23",
          "fortran_order": false,
          "uncompressed_bytes": 1416
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_detail_extension_2026-10-11/extension.npz",
      "bytes": 2050319,
      "sha256": "a61b71adc05bafac44339d771c7cb5f3392d68c98e62a1ae2b7dfad3e40e1585",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "new_labs",
          "shape": [
            19401,
            5,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3880328
        },
        {
          "name": "eos",
          "shape": [
            19401,
            2,
            20
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3104288
        },
        {
          "name": "med_details",
          "shape": [
            19401,
            12,
            16
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 14900096
        },
        {
          "name": "lab_values_raw",
          "shape": [
            19401,
            5,
            5
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 3880328
        },
        {
          "name": "lab_masks",
          "shape": [
            19401,
            5,
            5
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 485153
        },
        {
          "name": "med_values_raw",
          "shape": [
            19401,
            12,
            7
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 13037600
        },
        {
          "name": "med_masks",
          "shape": [
            19401,
            12,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 1629812
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_detail_extension_2026-10-11_r2/extension.npz",
      "bytes": 2329704,
      "sha256": "e20e7506054567930e660e9909617463abdf37fa907cf8a0717bd918923eb047",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "new_labs",
          "shape": [
            19401,
            5,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3880328
        },
        {
          "name": "eos",
          "shape": [
            19401,
            2,
            20
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3104288
        },
        {
          "name": "med_details",
          "shape": [
            19401,
            12,
            16
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 14900096
        },
        {
          "name": "lab_values_raw",
          "shape": [
            19401,
            5,
            5
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 3880328
        },
        {
          "name": "lab_masks",
          "shape": [
            19401,
            5,
            5
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 485153
        },
        {
          "name": "med_values_raw",
          "shape": [
            19401,
            12,
            7
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 13037600
        },
        {
          "name": "med_masks",
          "shape": [
            19401,
            12,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 1629812
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_discharge_regimen_interface_2026-10-14/interface.npz",
      "bytes": 160153,
      "sha256": "bde2092a19749fa93f4c1cfaa3c851c425dfc3a6c998c03e4dd3dad0fdc83553",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "section_available",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "candidate_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "positions",
          "shape": [
            13516
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 108256
        },
        {
          "name": "products",
          "shape": [
            13516
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 108256
        },
        {
          "name": "roles",
          "shape": [
            13516
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 108256
        },
        {
          "name": "class_memberships",
          "shape": [
            13516,
            3
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 40676
        },
        {
          "name": "as_needed",
          "shape": [
            13516
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 13644
        },
        {
          "name": "features",
          "shape": [
            19401,
            34
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 2638664
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_eos_med_context_2026-10-09/eos_med_context.npz",
      "bytes": 566848,
      "sha256": "a92f1bc2f853b80883008a1181ea6f297198229effe6259daa291559a38f584e",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "eos_values_raw",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 1862624
        },
        {
          "name": "eos_values_scaled",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 1862624
        },
        {
          "name": "eos_value_masks",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 232940
        },
        {
          "name": "eos_relation_flags",
          "shape": [
            19401,
            2,
            8
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 2483456
        },
        {
          "name": "eos_features",
          "shape": [
            19401,
            2,
            20
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3104288
        },
        {
          "name": "eos_names",
          "shape": [
            2
          ],
          "dtype": "<U8",
          "fortran_order": false,
          "uncompressed_bytes": 192
        },
        {
          "name": "med_exposure",
          "shape": [
            19401,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 543356
        },
        {
          "name": "med_source_mask",
          "shape": [
            19401,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 135935
        },
        {
          "name": "med_availability_state",
          "shape": [
            19401
          ],
          "dtype": "|i1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_eos_med_context_2026-10-09_r2/eos_med_context.npz",
      "bytes": 566848,
      "sha256": "a92f1bc2f853b80883008a1181ea6f297198229effe6259daa291559a38f584e",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "eos_values_raw",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 1862624
        },
        {
          "name": "eos_values_scaled",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 1862624
        },
        {
          "name": "eos_value_masks",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 232940
        },
        {
          "name": "eos_relation_flags",
          "shape": [
            19401,
            2,
            8
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 2483456
        },
        {
          "name": "eos_features",
          "shape": [
            19401,
            2,
            20
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3104288
        },
        {
          "name": "eos_names",
          "shape": [
            2
          ],
          "dtype": "<U8",
          "fortran_order": false,
          "uncompressed_bytes": 192
        },
        {
          "name": "med_exposure",
          "shape": [
            19401,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 543356
        },
        {
          "name": "med_source_mask",
          "shape": [
            19401,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 135935
        },
        {
          "name": "med_availability_state",
          "shape": [
            19401
          ],
          "dtype": "|i1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_eos_med_context_2026-10-09_r3/eos_med_context.npz",
      "bytes": 566848,
      "sha256": "a92f1bc2f853b80883008a1181ea6f297198229effe6259daa291559a38f584e",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "eos_values_raw",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 1862624
        },
        {
          "name": "eos_values_scaled",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 1862624
        },
        {
          "name": "eos_value_masks",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 232940
        },
        {
          "name": "eos_relation_flags",
          "shape": [
            19401,
            2,
            8
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 2483456
        },
        {
          "name": "eos_features",
          "shape": [
            19401,
            2,
            20
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3104288
        },
        {
          "name": "eos_names",
          "shape": [
            2
          ],
          "dtype": "<U8",
          "fortran_order": false,
          "uncompressed_bytes": 192
        },
        {
          "name": "med_exposure",
          "shape": [
            19401,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 543356
        },
        {
          "name": "med_source_mask",
          "shape": [
            19401,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 135935
        },
        {
          "name": "med_availability_state",
          "shape": [
            19401
          ],
          "dtype": "|i1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_expanded_candidates_2026-10-12/labs/expanded_labs.npz",
      "bytes": 2045663,
      "sha256": "4b8840e5d708c50d07f4471cddf8b0d1db2a2b358d913c56289aa2551739c8a6",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            10,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 7760528
        },
        {
          "name": "values_raw",
          "shape": [
            19401,
            10,
            5
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3880328
        },
        {
          "name": "value_masks",
          "shape": [
            19401,
            10,
            5
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 970178
        },
        {
          "name": "names",
          "shape": [
            10
          ],
          "dtype": "<U20",
          "fortran_order": false,
          "uncompressed_bytes": 928
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_expanded_candidates_2026-10-12/medications/medication_identity.npz",
      "bytes": 1501742,
      "sha256": "4b6c243c1c7f9227776f7244970c6a1737cf2cfec258a089e5397dcb7efc1c47",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            38,
            23
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 67826024
        },
        {
          "name": "present",
          "shape": [
            19401,
            38
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 737366
        },
        {
          "name": "values_raw",
          "shape": [
            19401,
            38,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 20642792
        },
        {
          "name": "value_masks",
          "shape": [
            19401,
            38,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 5160794
        },
        {
          "name": "dose_unit_id",
          "shape": [
            19401,
            38
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 5898032
        },
        {
          "name": "drug_names",
          "shape": [
            38
          ],
          "dtype": "<U25",
          "fortran_order": false,
          "uncompressed_bytes": 3928
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_expanded_candidates_2026-10-12_r2/labs/expanded_labs.npz",
      "bytes": 2045663,
      "sha256": "4b8840e5d708c50d07f4471cddf8b0d1db2a2b358d913c56289aa2551739c8a6",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            10,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 7760528
        },
        {
          "name": "values_raw",
          "shape": [
            19401,
            10,
            5
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3880328
        },
        {
          "name": "value_masks",
          "shape": [
            19401,
            10,
            5
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 970178
        },
        {
          "name": "names",
          "shape": [
            10
          ],
          "dtype": "<U20",
          "fortran_order": false,
          "uncompressed_bytes": 928
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_expanded_candidates_2026-10-12_r2/medications/medication_identity.npz",
      "bytes": 1494097,
      "sha256": "076b8a16389e375cf60da7ca7e5996f1243a5f944b43faeb747b8db439c14da7",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            38,
            23
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 67826024
        },
        {
          "name": "present",
          "shape": [
            19401,
            38
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 737366
        },
        {
          "name": "values_raw",
          "shape": [
            19401,
            38,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 20642792
        },
        {
          "name": "value_masks",
          "shape": [
            19401,
            38,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 5160794
        },
        {
          "name": "dose_unit_id",
          "shape": [
            19401,
            38
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 5898032
        },
        {
          "name": "drug_names",
          "shape": [
            38
          ],
          "dtype": "<U25",
          "fortran_order": false,
          "uncompressed_bytes": 3928
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_expanded_candidates_2026-10-12_r3/labs/expanded_labs.npz",
      "bytes": 2045663,
      "sha256": "4b8840e5d708c50d07f4471cddf8b0d1db2a2b358d913c56289aa2551739c8a6",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            10,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 7760528
        },
        {
          "name": "values_raw",
          "shape": [
            19401,
            10,
            5
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3880328
        },
        {
          "name": "value_masks",
          "shape": [
            19401,
            10,
            5
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 970178
        },
        {
          "name": "names",
          "shape": [
            10
          ],
          "dtype": "<U20",
          "fortran_order": false,
          "uncompressed_bytes": 928
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_expanded_candidates_2026-10-12_r3/medications/medication_identity.npz",
      "bytes": 1494694,
      "sha256": "06d5cdcd860a9347003aa51523fcf9cc8d5c23ca99505dfefdb1a870968b5f8c",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            38,
            23
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 67826024
        },
        {
          "name": "present",
          "shape": [
            19401,
            38
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 737366
        },
        {
          "name": "values_raw",
          "shape": [
            19401,
            38,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 20642792
        },
        {
          "name": "value_masks",
          "shape": [
            19401,
            38,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 5160794
        },
        {
          "name": "dose_unit_id",
          "shape": [
            19401,
            38
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 5898032
        },
        {
          "name": "drug_names",
          "shape": [
            38
          ],
          "dtype": "<U25",
          "fortran_order": false,
          "uncompressed_bytes": 3928
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_imaging_all_2026-10-10/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_imaging_all_2026-10-10_r2/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_local_text_interface_2026-10-13/interface.npz",
      "bytes": 10317427,
      "sha256": "845137f083949e13ec0c3c2ed829666d737ba5aa82cc0348ad9c5a451398f69b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "sentence_positions",
          "shape": [
            9339
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 74840
        },
        {
          "name": "metadata",
          "shape": [
            9339,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 373688
        },
        {
          "name": "containing_vectors",
          "shape": [
            9339,
            768
          ],
          "dtype": "<f2",
          "fortran_order": false,
          "uncompressed_bytes": 14344832
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_local_text_interface_2026-10-13_r2/interface.npz",
      "bytes": 10317427,
      "sha256": "845137f083949e13ec0c3c2ed829666d737ba5aa82cc0348ad9c5a451398f69b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "sentence_positions",
          "shape": [
            9339
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 74840
        },
        {
          "name": "metadata",
          "shape": [
            9339,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 373688
        },
        {
          "name": "containing_vectors",
          "shape": [
            9339,
            768
          ],
          "dtype": "<f2",
          "fortran_order": false,
          "uncompressed_bytes": 14344832
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_medication_identity_2026-10-12/medication_identity.npz",
      "bytes": 1187829,
      "sha256": "3dda8de610a9b11cc19bbc22b715c79eef31f936a867a001f7b52d6986b8bcc0",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            21,
            23
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 37482860
        },
        {
          "name": "present",
          "shape": [
            19401,
            21
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 407549
        },
        {
          "name": "values_raw",
          "shape": [
            19401,
            21,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 11407916
        },
        {
          "name": "value_masks",
          "shape": [
            19401,
            21,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 2852075
        },
        {
          "name": "dose_unit_id",
          "shape": [
            19401,
            21
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 3259496
        },
        {
          "name": "drug_names",
          "shape": [
            21
          ],
          "dtype": "<U23",
          "fortran_order": false,
          "uncompressed_bytes": 2060
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_model_inputs_bert_2026-10-09/context.npz",
      "bytes": 446479,
      "sha256": "d97f53555f959f5ee2d21b8c09dd8fde98c45e7466b3e2c2b1efd9f5424243bc",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "history_inputs",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 465752
        },
        {
          "name": "history_raw",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 931376
        },
        {
          "name": "current_values",
          "shape": [
            19401,
            14
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 1086584
        },
        {
          "name": "current_masks",
          "shape": [
            19401,
            14
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 271742
        },
        {
          "name": "history_names",
          "shape": [
            6
          ],
          "dtype": "<U31",
          "fortran_order": false,
          "uncompressed_bytes": 872
        },
        {
          "name": "current_names",
          "shape": [
            14
          ],
          "dtype": "<U36",
          "fortran_order": false,
          "uncompressed_bytes": 2144
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_model_inputs_qwen_2026-10-09/context.npz",
      "bytes": 446479,
      "sha256": "d97f53555f959f5ee2d21b8c09dd8fde98c45e7466b3e2c2b1efd9f5424243bc",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "history_inputs",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 465752
        },
        {
          "name": "history_raw",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 931376
        },
        {
          "name": "current_values",
          "shape": [
            19401,
            14
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 1086584
        },
        {
          "name": "current_masks",
          "shape": [
            19401,
            14
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 271742
        },
        {
          "name": "history_names",
          "shape": [
            6
          ],
          "dtype": "<U31",
          "fortran_order": false,
          "uncompressed_bytes": 872
        },
        {
          "name": "current_names",
          "shape": [
            14
          ],
          "dtype": "<U36",
          "fortran_order": false,
          "uncompressed_bytes": 2144
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_oxygen_device_2026-10-13/oxygen_device.npz",
      "bytes": 527217,
      "sha256": "c58094b5dcc813893435f67914dfc6802a898b6c6c66e66cf55bdbf85eb7ec60",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            115
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 8924588
        },
        {
          "name": "availability",
          "shape": [
            19401,
            115
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 8924588
        },
        {
          "name": "timing",
          "shape": [
            19401,
            115
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 8924588
        },
        {
          "name": "observed",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "device_multihot",
          "shape": [
            19401,
            5,
            18
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 13968848
        },
        {
          "name": "device_flags",
          "shape": [
            19401,
            4
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 620960
        },
        {
          "name": "time_raw",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 931376
        },
        {
          "name": "time_masks",
          "shape": [
            19401,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 116534
        },
        {
          "name": "procedure_evidence",
          "shape": [
            19401,
            9
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 1397000
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_pft_set_interface_2026-10-13/interface.npz",
      "bytes": 109594,
      "sha256": "d7553cc3e5539357b17f31a8891fd0cb686ee025485c22e053235e4f6b85e379",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "positions",
          "shape": [
            994
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 8080
        },
        {
          "name": "concepts",
          "shape": [
            994
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 8080
        },
        {
          "name": "raw_values",
          "shape": [
            994
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 8080
        },
        {
          "name": "values",
          "shape": [
            994
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 4104
        },
        {
          "name": "metadata",
          "shape": [
            994,
            3
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 12056
        },
        {
          "name": "aggregate",
          "shape": [
            19401,
            9
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 698564
        },
        {
          "name": "observed",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_radbert_imaging_2026-10-10_r2/imaging.npz",
      "bytes": 27091123,
      "sha256": "b0b5d497bbb94346b3e54f4727de6983e5078305e3dd96560467122d3f4cde23",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "values",
          "shape": [
            19401,
            2,
            1536
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 238399616
        },
        {
          "name": "present",
          "shape": [
            19401,
            2
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 38930
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_radgraph_imaging_2026-10-10_r2/imaging.npz",
      "bytes": 437777,
      "sha256": "d07ec662d690f318f3a440242ff211762ab38e675f049be0728201b63a755d2b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "values",
          "shape": [
            19401,
            2,
            1536
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 238399616
        },
        {
          "name": "present",
          "shape": [
            19401,
            2
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 38930
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_radgraph_imaging_source_checked_2026-10-10/imaging.npz",
      "bytes": 438027,
      "sha256": "41d6f6bfe3df5d98f72a54529a445dd332e0bef1fe165047df6db4350032b28d",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "values",
          "shape": [
            19401,
            2,
            1536
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 238399616
        },
        {
          "name": "present",
          "shape": [
            19401,
            2
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 38930
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_radgraph_smoke_2026-10-10_r2/imaging.npz",
      "bytes": 331405,
      "sha256": "15648a4a0b3bf2429c3a582b7b8a62dd38b5900dbb7af03a6cb2d42eb3a954f3",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "values",
          "shape": [
            19401,
            2,
            1536
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 238399616
        },
        {
          "name": "present",
          "shape": [
            19401,
            2
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 38930
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_resp_fact_set_interface_2026-10-13/interface.npz",
      "bytes": 109491,
      "sha256": "c494a1ab9edd3023386c939cf6d1f5d44fd37d88f43681f78ef5303aa2432b5a",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "positions",
          "shape": [
            997
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 8104
        },
        {
          "name": "concepts",
          "shape": [
            997
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 8104
        },
        {
          "name": "roles",
          "shape": [
            997
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 8104
        },
        {
          "name": "polarities",
          "shape": [
            997
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 8104
        },
        {
          "name": "units",
          "shape": [
            997
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 8104
        },
        {
          "name": "conditions",
          "shape": [
            997
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 8104
        },
        {
          "name": "numeric_present",
          "shape": [
            997
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 1125
        },
        {
          "name": "raw_values",
          "shape": [
            997
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 8104
        },
        {
          "name": "values",
          "shape": [
            997
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 4116
        },
        {
          "name": "numeric_valid",
          "shape": [
            997
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 1125
        },
        {
          "name": "aggregate",
          "shape": [
            19401,
            3
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 232940
        },
        {
          "name": "observed",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_resp_micro_context_2026-10-09/resp_micro_context.npz",
      "bytes": 586726,
      "sha256": "4da575e4a26447fd04469d3b83ccc2e0295d22f4f670c71fe1ec6ef69a4588f2",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "micro_values",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 465752
        },
        {
          "name": "micro_masks",
          "shape": [
            19401,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 116534
        },
        {
          "name": "micro_source_state",
          "shape": [
            19401
          ],
          "dtype": "|i1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "icu_stay_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "resp_numeric_raw",
          "shape": [
            19401,
            8,
            5
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 6208448
        },
        {
          "name": "resp_numeric_scaled",
          "shape": [
            19401,
            8,
            5
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 6208448
        },
        {
          "name": "resp_numeric_masks",
          "shape": [
            19401,
            8,
            5
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "support_values",
          "shape": [
            19401,
            9
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 698564
        },
        {
          "name": "support_masks",
          "shape": [
            19401,
            9
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 174737
        },
        {
          "name": "recorded_duration_hours",
          "shape": [
            19401,
            2
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 310544
        },
        {
          "name": "recorded_duration_masks",
          "shape": [
            19401,
            2
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 38930
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_resp_micro_context_2026-10-09_r2/resp_micro_context.npz",
      "bytes": 586726,
      "sha256": "4da575e4a26447fd04469d3b83ccc2e0295d22f4f670c71fe1ec6ef69a4588f2",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "micro_values",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 465752
        },
        {
          "name": "micro_masks",
          "shape": [
            19401,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 116534
        },
        {
          "name": "micro_source_state",
          "shape": [
            19401
          ],
          "dtype": "|i1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "icu_stay_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "resp_numeric_raw",
          "shape": [
            19401,
            8,
            5
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 6208448
        },
        {
          "name": "resp_numeric_scaled",
          "shape": [
            19401,
            8,
            5
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 6208448
        },
        {
          "name": "resp_numeric_masks",
          "shape": [
            19401,
            8,
            5
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "support_values",
          "shape": [
            19401,
            9
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 698564
        },
        {
          "name": "support_masks",
          "shape": [
            19401,
            9
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 174737
        },
        {
          "name": "recorded_duration_hours",
          "shape": [
            19401,
            2
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 310544
        },
        {
          "name": "recorded_duration_masks",
          "shape": [
            19401,
            2
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 38930
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_respiratory_transition_2026-10-13/respiratory_transition.npz",
      "bytes": 1126146,
      "sha256": "bdf62fa01cb2de4b95f91cacd29ac41d91f7ffe9e4122b69379a3cad62b581d7",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            96
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 7450112
        },
        {
          "name": "availability",
          "shape": [
            19401,
            96
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 7450112
        },
        {
          "name": "timing",
          "shape": [
            19401,
            96
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 7450112
        },
        {
          "name": "observed",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "numeric_raw",
          "shape": [
            19401,
            4,
            9
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 5587616
        },
        {
          "name": "numeric_masks",
          "shape": [
            19401,
            4,
            9
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 698564
        },
        {
          "name": "numeric_flags",
          "shape": [
            19401,
            4,
            3
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 1862624
        },
        {
          "name": "support_pairs_raw",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 931376
        },
        {
          "name": "support_pairs_masks",
          "shape": [
            19401,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 116534
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_saved_sections_aligned_2026-10-11/alignment.npz",
      "bytes": 454304,
      "sha256": "4802112ba42487e8afd8ace9f3dca1b64bbea26714dde8630232128d22c583f9",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "refs",
          "shape": [
            75086,
            3
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 1802192
        },
        {
          "name": "source_rows",
          "shape": [
            75086
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 600816
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_short_topic_audit_96_2026-10-10/pilot_facts.npz",
      "bytes": 2677,
      "sha256": "7448d31016c1666bc85c54d2ced4e3ee12c3e2c70905994f886250ea6f237c16",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            96
          ],
          "dtype": "<U8",
          "fortran_order": false,
          "uncompressed_bytes": 3200
        },
        {
          "name": "values",
          "shape": [
            96,
            69
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 26624
        },
        {
          "name": "names",
          "shape": [
            69
          ],
          "dtype": "<U58",
          "fortran_order": false,
          "uncompressed_bytes": 16136
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_short_topic_audit_96_2026-10-10_r2/pilot_facts.npz",
      "bytes": 2677,
      "sha256": "7448d31016c1666bc85c54d2ced4e3ee12c3e2c70905994f886250ea6f237c16",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            96
          ],
          "dtype": "<U8",
          "fortran_order": false,
          "uncompressed_bytes": 3200
        },
        {
          "name": "values",
          "shape": [
            96,
            69
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 26624
        },
        {
          "name": "names",
          "shape": [
            69
          ],
          "dtype": "<U58",
          "fortran_order": false,
          "uncompressed_bytes": 16136
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_short_topics_96_2026-10-10/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_text_geometry_2026-10-13/transform.npz",
      "bytes": 8765111,
      "sha256": "1f642052032c27430a40503d2d3956fc9006cf118c421a4b45294e7844e3192c",
      "arrays": [
        {
          "name": "means",
          "shape": [
            4,
            768
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 12416
        },
        {
          "name": "whitening",
          "shape": [
            4,
            768,
            768
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 9437312
        },
        {
          "name": "field_ids",
          "shape": [
            76974
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 615920
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_unified_lab_state_2026-10-14/interface.npz",
      "bytes": 6749960,
      "sha256": "46b896255802a1283a4fb89fc28845342fc0274cea44a240a46c64290aa27031",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "raw",
          "shape": [
            19401,
            36,
            9
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 50287520
        },
        {
          "name": "values",
          "shape": [
            19401,
            36,
            9
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 25143824
        },
        {
          "name": "valid",
          "shape": [
            19401,
            36,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 4190744
        },
        {
          "name": "eos_context",
          "shape": [
            19401,
            16
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 1241792
        },
        {
          "name": "names",
          "shape": [
            36
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 4304
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    }
  ],
  "limits": "file hashes bind local artifacts, not an authorization to redistribute them; NPY headers read without array values"
}
````
<!-- END REVIEW_FRAGMENT -->

## docs/PUBLICATION_BOUNDARY_2026-10-14.md · lines 1–36

<!-- REVIEW_FRAGMENT {"path":"docs/PUBLICATION_BOUNDARY_2026-10-14.md","part":1,"start_line":1,"end_line":36,"source_sha256":"d692b9fb3bd8d45f1078b0d74edbde2dbfdef657652132521b7a19971cd1bdfe","fragment_sha256":"d692b9fb3bd8d45f1078b0d74edbde2dbfdef657652132521b7a19971cd1bdfe","added_terminal_newline":false} -->
````markdown
# 公开Git范围与保留本地内容

用户授权发布到公开仓库`https://github.com/aufgh/aecopd`。发布对象是代码/队列定义/合成测试/审查后汇总文档，不是病历或完整研究数据包。

## 上传

- cohort_pipeline、model_loader、training及patient_data根层/clinical_facts的白名单源码。
- 两份非患者概念映射CSV、当前实验协议/汇总指标/文献说明。
- 独立公开README和导航；说明真实数据与模型产物缺失是权限边界，不宣称克隆即能复现训练。

## 不上传

- raw病历、报告、患者ID/标签/划分/病例引用、所有processed数组、向量和权重。
- outputs、training_runs、transfer、work内部现场/预览、可重装runtime包、数据库凭据和账号状态。
- 含需复核个案示例的旧抽取pilot等源码；含个案的旧文档、截图/参考图片及未经公开审查历史报告。
- 本地旧Git历史：其中有需排除的旧连接默认值等内容，仅排除当前文件不能保证历史安全。

## 发布方法

保留本地主分支、文件和实验现场，不reset/删除/改写旧历史。在独立Git索引中，只暂存已扫描白名单，公开README/docs导航采用独立内容，生成无父提交的公开快照分支`codex/public-source-2026-10-14`并推送到远程main。原远程已确认空仓库，不force push、不覆盖别人的提交。

白名单源码逐文件hash、秘密/八位标识启发扫描与明确例外审查保存到本地`work/public_release_2026-10-14`，不公开疑似秘密值。启发扫描不是绝对隐私保证；限制旧文档/数据实物/历史和公开源码内容审查一起使用。

后续公开更改应基于公开分支提交/推送，不能把原本地主分支旧历史合并回来。数据必须通过经授权的受控方式另行迁移，不能用Git LFS规避这条边界。Git账号和上传权限不等MIMIC数据再分发许可。

## 2026-10-15 V7增量公开

本次沿用上述边界，在既有公开提交之后追加V7源码、合成测试、两份实验合同与汇总结果，以及已知检验输入后的适配说明。仅额外公开两张经审查的聚合图：`docs/figures/v7_first_experiment_2026-10-15.png`和`docs/figures/v7_loss_comparison_2026-10-15.png`，不包含患者记录或个体预测。公开结果中的图路径指向这两份文件；外部AI原研究全文不随代码发布。

继续使用独立索引及明确文件白名单，以远端公开main为唯一父提交，正常快进推送；不合并本地main，不改动其暂存区、工作树或实验现场。V7实际训练源码保持原字节，发布清单、来源hash、扫描和推送核对保存于本地`work/public_release_2026-10-15`。公开导航与`.gitignore`采用独立模板，公开图仅增加两个确切文件例外，其他数据边界不扩大。

## 2026-10-15 Markdown完整阅读包

用户授权以若干Markdown替代旧TXT摘录并发布Git。本次基于既有公开main追加最新关闭Query/固定融合实验源码、测试与汇总报告，包含两个源码核对说明和两份已审查历史评价/确认说明。全部既有公开文本及本次增量原样嵌入分卷；数据快照仅限17张来源表的规模/hash、四划分计数、66份NPZ头部结构，结果快照仅限历史汇总报告的数值指标白名单。不会将患者数据、ID/标签/预测取值、文书、向量或权重转成Markdown后公开。

9份含历史病例引用的原型继续保留本地，分卷首页明确列出，不把“全部公开源码”扩大为全部私有现场。合并包带来源清单、fragment和整体hash以及逐字节还原工具；公开检出只做合成测试，不挂载患者数据。旧TXT本地归档，实际训练源码/结果及私有Git历史保持。
````
<!-- END REVIEW_FRAGMENT -->

## docs/README.md · lines 1–24

<!-- REVIEW_FRAGMENT {"path":"docs/README.md","part":1,"start_line":1,"end_line":24,"source_sha256":"bfb7d34cede08035407a1035362b8bd92c9c3603067a60edb041a57cd423ce51","fragment_sha256":"bfb7d34cede08035407a1035362b8bd92c9c3603067a60edb041a57cd423ce51","added_terminal_newline":false} -->
````markdown
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
````
<!-- END REVIEW_FRAGMENT -->

## docs/V6_CURRENT_INPUT_CATALOG_2026-10-11.md · lines 1–246

<!-- REVIEW_FRAGMENT {"path":"docs/V6_CURRENT_INPUT_CATALOG_2026-10-11.md","part":1,"start_line":1,"end_line":246,"source_sha256":"faf3246ec1c01b97a57361d0256b921fa96d8131638230b32900705f3bb4990d","fragment_sha256":"faf3246ec1c01b97a57361d0256b921fa96d8131638230b32900705f3bb4990d","added_terminal_newline":false} -->
````markdown
# V6 当前输入全清单：来源、数值、缺失与网络位置

核对日期2026-10-11。以`training/v6/data.py`、`compact_context_data.py`、`compact_model.py`及最新90天目标/文本对照为依据，不把设计候选或其他实验误列为当前输入。机器可读清单和原数据覆盖元数据见[本轮input_catalog.json](../training_runs/v6_bert_qwen_same_fields_2026-10-11/input_catalog.json)。

## 0. 两种参照必须区分

| 配方 | 基础风险 | 临床修正中的诊断/去向 | 文本 |
| --- | --- | --- | --- |
| 既有compact参照 | 历史H，6维 | 14个值＋14个标记保留 | B四字段ClinicalBERT |
| 最新90天目标／Qwen对照 | 扩展基础风险，34维=H6＋诊断/去向14＋标记14 | 值和标记全部置零，保留网络槽位 | 同一B四字段ClinicalBERT／Qwen／空文本 |

扩展基础风险是机制实验参照，未满足替换原默认参照的证据条件；旧模型不覆盖。冻结的是基础风险模型参数，不是患者输入或所有患者的风险。

## 1. 队列、预测点、标签

患者曾有一次前三位诊断包含既定v1 COPD/肺气肿清单编码即可入组，保留该患者任意诊断位含清单编码的全部住院；从最早保留住院开始。完整清单与条件见[任务合同](CURRENT_TASK_AND_COHORT.md)及队列SQL，本轮不改清单。

- 原始底账5,390人/19,774次；预测点5,351人/19,401次存活且正时长出院。
- 每次出院重预测；train/validation/test=13,592/2,949/2,860次，患者不跨折。
- 未来首次非转院严格491.21/J44.1再入院为事件1；该事件前的可观察死亡为事件2；未观察到为事件0，行政上限365天。
- 时间、事件类型只送损失和评价。患者/住院ID、未来住院、未来死亡日、划分名称不送编码器。
- 本库未观察到不等于院外无事件；最终小结可晚存档，当前不是实时出院部署验证。

## 2. 历史H：完整6项

来源：原始patients、保留住院底账、已结束历史。只计本次入院前已出院且正时长的保留COPD住院，不读取未来住院，不使用跨住院Transformer。

| 顺序/Key | 中文 | 实际构造 |
| --- | --- | --- |
| 0 `age` | 本次入院年龄 | anchor_age＋入院年−anchor_year |
| 1 `female` | 女性标志 | F=1、M=0；未知性别报错，不默认填男性 |
| 2 `log_prior_completed_copd` | 既往已结束COPD住院次数 | log(1＋次数) |
| 3 `log_prior_completed_strict` | 其中严格AECOPD次数 | log(1＋严格次数) |
| 4 `log_previous_discharge_gap_days` | 最近历史出院至本次入院间隔 | log(1＋天数)；无历史为占位0 |
| 5 `no_completed_history` | 无合格既往住院 | 无历史=1、有=0 |

年龄、两计数、历史间隔用训练折median/IQR标准化、裁剪至[-8,8]；零IQR按标准差再常数尺度回退。间隔拟合不包含无历史占位值。性别/无历史仍0/1；后者区分未知间隔与真实0间隔。底账保存历史顺序，但本模型仅消费这六项，不读取全部历史检验/文本序列。

## 3. 当前诊断与出院去向：完整14值＋14标记

来源：本次diagnoses_icd（所有诊断位、ICD9/10统一家族）和admissions.discharge_location。不是把所有非COPD编码都当既往共病。

| 序号/Key | 中文 | 定义 |
| --- | --- | --- |
| 0 `current_strict_code` | 本次严格AECOPD编码 | 底账is_strict_aecopd |
| 1 `current_code:heart_failure` | 心力衰竭 | I50/I110/I130/I132；428 |
| 2 `current_code:cardiac_other` | 缺血性心脏病/房颤 | I20–I25/I48；410–414/42731 |
| 3 `current_code:ckd` | 慢性肾病 | N18；585 |
| 4 `current_code:diabetes` | 糖尿病 | E08/E09/E10/E11/E13；250 |
| 5 `current_code:malignancy` | 恶性肿瘤 | C00–C99；140–208 |
| 6 `current_code:chronic_respiratory` | 慢性呼衰/OSA | J961/J962/G4733；51883/51884/32723 |
| 7 `current_code:acute_resp_failure` | 急性/急慢性呼衰 | J960/J962；51881/51884 |
| 8 `current_code:pneumonia_or_aspiration` | 肺炎/吸入性肺损伤 | J12–J18/J69；480–486/507 |
| 9 `current_code:sepsis` | 脓毒症 | A40/A41/R652；038/99591/99592 |
| 10 `current_code:aki` | 急性肾损伤 | N17；584 |
| 11 `disposition_home` | 出院回家 | 原词包含HOME |
| 12 `disposition_facility` | 康复/护理/长期机构 | 非HOME，包含REHAB/SKILLED/NURSING/LONG TERM |
| 13 `disposition_other` | 其他去向 | 非以上、有记录的类别 |

编码去点/空格后按前缀映射，实际定义见`training/v5/clinical_concepts.py`。急慢性呼衰可同时命中两组，并非互斥疾病。前11项mask为1，0只表示未映射到编码，不证明没有疾病。去向缺失三个值和mask均0；有记录时三个mask都1、三个值one-hot。包含HOME先归home，即HOME WITH SERVICES也在home，当前未单列照护服务。

## 4. 检验：完整22轴，不是22个不同项目

按异常方向构造轴，同一项目可贡献高/低两轴；有测正常也保留。表中“有测”指通过当前单位、时间、数值、参考范围和标本规则的住院数，不代表异常人数。来源`raw/labevents.csv.gz`及r2 manifest；动脉识别还依赖原标本校验。

| 顺序/Key | 中文轴 | itemid / 原单位 | 有测住院 |
| --- | --- | --- | ---: |
| 0 `arterial_low_ph` | 动脉pH低 | 50820 / units | 2,684 |
| 1 `arterial_high_pco2` | 动脉pCO₂高 | 50818 / mm Hg | 2,662 |
| 2 `bicarbonate_low` | 碳酸氢根低 | 50882 / mEq/L | 17,042 |
| 3 `bicarbonate_high` | 碳酸氢根高 | 同上 | 17,042 |
| 4 `wbc_high` | 白细胞高 | 51301 / K/uL | 16,972 |
| 5 `albumin_low` | 白蛋白低 | 50862 / g/dL | 4,615 |
| 6 `hemoglobin_low` | 血红蛋白低 | 51222 / g/dL | 16,988 |
| 7 `creatinine_high` | 肌酐高 | 50912 / mg/dL | 17,234 |
| 8 `arterial_hypercapnic_acidosis` | 动脉高碳酸性酸中毒 | 同标本pH＋pCO₂，两者时间差≤1h | 2,662 |
| 9 `wbc_low` | 白细胞低 | 51301 / K/uL | 16,972 |
| 10 `sodium_low` | 钠低 | 50983 / mEq/L | 17,151 |
| 11 `sodium_high` | 钠高 | 同上 | 17,151 |
| 12 `potassium_low` | 钾低 | 50971 / mEq/L | 17,228 |
| 13 `potassium_high` | 钾高 | 同上 | 17,228 |
| 14 `urea_nitrogen_high` | 尿素氮高 | 51006 / mg/dL | 17,177 |
| 15 `glucose_low` | 血糖低 | 50931 / mg/dL | 16,996 |
| 16 `glucose_high` | 血糖高 | 同上 | 16,996 |
| 17 `platelet_low` | 血小板低 | 51265 / K/uL | 17,055 |
| 18 `magnesium_low` | 镁低 | 50960 / mg/dL | 16,317 |
| 19 `lactate_high` | 乳酸高 | 50813 / mmol/L | 4,073 |
| 20 `crp_high` | CRP高 | 50889 / mg/L | 851 |
| 21 `ntprobnp_high` | NT-proBNP高 | 50963 / pg/mL | 1,305 |

乳酸是该检验项目的合格记录，**没有像pH/pCO₂一样额外保证动脉标本**。不将乳酸统称动脉血乳酸。

### 每轴实际9通道：6数值＋3标记，合计22×9=198

| 槽位/Key | 中文 | 计算/缺失 |
| --- | --- | --- |
| 0 `first_observed_severity` | 首次观测异常程度 | 全住院第一合格时点 |
| 1 `observed_peak_severity` | 最大观测异常程度 | 全住院观测最大值 |
| 2 `latest_observed_severity` | 最新观测异常程度 | 可早于出院数日，不限48h |
| 3 `recovery_delta` | 观测峰后恢复差 | 有更晚峰后复查时：峰−最新；否则占位0 |
| 4 `latest_measurement_age` | 最新观测距出院 | log(1＋天数) |
| 5 `peak_to_latest_elapsed` | 最早峰至最新观测间隔 | log(1＋天数)；不可估计置0 |
| 6 `axis_observed` | 有合格观测 | 有=1；无=0、整轴零 |
| 7 `post_peak_recheck_valid` | 有效峰后复测 | 最新时刻晚于最早峰时刻=1 |
| 8 `latest_at_abnormal_peak` | 最新观测仍处异常峰 | 峰>0且最新≈峰=1；不保证出院当日测量 |

同轴同时间取中位数，不给同刻项目强加先后。峰用未截顶异常程度定位；第8标记与第7标记可同时为1（先有峰、后来又在峰）。恢复差0需结合最新严重度和标记，不能独自判断平稳。

原始异常程度：pH=`max(0,(7.35−pH)/0.10)`；pCO₂=`max(0,(pCO₂−45)/15)`；其他低轴=`max(0,(下限−值)/(上限−下限))`、高轴=`max(0,(值−上限)/(上限−下限))`。联合酸中毒取同标本两方向异常程度的最小值，因此只有两者都异常才>0。

必须满足住院内`admittime≤charttime≤storetime≤dischtime`、单位完全匹配、有限正数、有效上下限；参考范围缺失或单位不符会被排除而非自动修补，覆盖限制保留审计。pH/pCO₂也需有效参考字段，尽管它们使用固定临床阈值。

前4数值log1p，两个时间通道已log1p；各轴各数值只以训练折有效观测拟合median/IQR并裁剪[-8,8]，零IQR回退std/1。恢复差/峰间隔只有标记7有效时参与拟合，其余占位0；三标记保持0/1。完整逐值有效mask另存，但网络输入仅这9通道，不额外送六个mask。正常值仍含有测标志；不是把未测当正常。

## 5. 药物：完整7药效类别×4通道=28

多来源：eMAR实际执行记录、pharmacy及通过pharmacy_id关联的prescriptions。时间仅用于上游可见性/住院区间核验；**没有给药次数、涉及日数、首末日期、跨度、剂量、持续时间或药物序列进入网络**。多次和同类多药按存在性OR合并；复方可以命中多个药效类。

| Key | 中文/缩写 | 映射药物例子 | 执行住院 / 仅医嘱住院 |
| --- | --- | --- | ---: |
| `short_beta_agonist` | 短效β₂激动剂 SABA | albuterol、salbutamol、levalbuterol | 5,962 / 8,340 |
| `short_muscarinic_antagonist` | 短效抗胆碱剂 SAMA | ipratropium | 4,491 / 4,840 |
| `long_beta_agonist` | 长效β₂激动剂 LABA | salmeterol、formoterol、arformoterol、indacaterol、olodaterol、vilanterol | 4,299 / 3,197 |
| `long_muscarinic_antagonist` | 长效抗胆碱剂 LAMA | tiotropium、umeclidinium、吸入glycopyrrolate、aclidinium、revefenacin | 3,559 / 2,576 |
| `inhaled_corticosteroid` | 吸入糖皮质激素 ICS | 吸入budesonide、fluticasone、beclomethasone、mometasone、ciclesonide | 4,890 / 4,103 |
| `systemic_corticosteroid` | 全身糖皮质激素 | prednisone、prednisolone、methylprednisolone、dexamethasone、hydrocortisone | 3,722 / 2,708 |
| `common_systemic_antibacterial` | 常见全身抗菌药 | 大环内酯＋β内酰胺＋氟喹诺酮＋四环素四药效族合并 | 6,004 / 4,267 |

抗菌药不是全部抗菌药表：例如vancomycin/linezolid等`other_antibiotic`不在这7类；利尿剂也不在。映射完整规则及复方商品名见`training/v5/clinical_concepts.py`，按途径/剂型排除鼻喷、眼药、外用和不合格系统治疗；不能只凭药名判吸入药。系统抗菌药记录没有自动裁定呼吸适应证。

四通道对每类均固定如下：

| Key | 含义 |
| --- | --- |
| `execution_evidence` | 至少一条被接受的实际执行证据=1 |
| `order_only_evidence` | 有有效医嘱且该类无执行证据=1；不能说实际已用 |
| `execution_source_observed` | 执行来源可用标志；不是该类完整用药观察证明 |
| `order_source_observed` | 住院内有效医嘱来源可用标志；不是实际给药 |

两个来源标记是住院级来源可用性复制到7类，不是逐药核验全部记录；实际执行与仅医嘱互斥。药房与处方同一pharmacy_id/类别去重。无法关联录入时间的处方仅审计，不补为确认给药；取消/暂停、出院后录入、区间不相交医嘱排除。出院带药方案、实际累积剂量不因有prescriptions表就自动存在于本接口。

## 6. 微生物：完整5值＋5标记=10

来源独立补导的microbiologyevents：仅合格呼吸常规培养/病毒检测、出院前结果可见、有效标本，按菌株去重，尿/便/非呼吸专项不伪装常规细菌检测。以下是阳性住院数，不等于感染人数。

| Key | 中文 | 阳性 / 来源可用住院 |
| --- | --- | ---: |
| `common_copd_bacteria` | 肺炎链球菌、卡他莫拉菌、流感嗜血杆菌任一检出 | 71 / 1,222 |
| `pseudomonas_aeruginosa` | 铜绿假单胞菌检出 | 114 / 1,222 |
| `other_named_respiratory_bacteria` | 固定其他呼吸标本细菌白名单任一检出 | 252 / 1,222 |
| `respiratory_virus_positive` | 接受的呼吸病毒检测阳性 | 38 / 618 |
| `named_bacteria_resistance` | 接受细菌任一药敏R | 236 / 334 |

其他细菌白名单来自`compact_multisource.OTHER_BACTERIA`，包含金葡菌、大肠埃希菌、肺炎/产酸克雷伯菌、黏质沙雷菌、阴沟/产气肠杆菌、不动杆菌、嗜麦芽窄食单胞菌、变形杆菌、普罗威登菌、摩根菌、哈夫尼菌、无色杆菌/相关别名、伊丽莎白菌及非铜绿假单胞菌/荧光假单胞菌。精确字符串全集在[源代码](../patient_data/compact_multisource.py)，不自行扩菌种。

各值0配mask0=无可用来源，值0配mask1=有该检查来源但未检出对应保留阳性；后者不能把“未报告”严格等同实验室明确阴性。耐药mask仅在有S/I/R时有效，R表示任一药耐药，不是多重耐药。培养检出不区分活动性感染/定植，模型不作病因裁定。

## 7. 呼吸支持：完整3值＋3标记=6

来源补导受限ICU chartevents、procedureevents及ICU关联，压缩为支持证据，不包含全量高频体征。

| Key | 中文 | 阳性 / 来源可用住院 |
| --- | --- | ---: |
| `recorded_invasive_support` | 已记录有创支持/气管导管证据 | 1,043 / 3,455 |
| `recorded_noninvasive_support` | 已记录无创支持证据 | 768 / 3,454 |
| `recorded_supplemental_oxygen` | 已记录补充给氧证据 | 3,235 / 3,457 |

气切独立存在不算有创通气，CPAP/PSV或品牌名独立存在不算急性NIV；给氧来自明确设备/氧流量/FiO₂证据。这些数值/时间只上游生成存在性，不作为新连续变量送入网络。支持0且mask0表示未知；不将非ICU无记录当无需支持，也未全面裁定COPD急性、OSA或麻醉适应证。

## 8. 文本：完整四字段，变长片段列表

模型身份：冻结`emilyalsentzer/Bio_ClinicalBERT`768维（医学BERT）；Qwen为冻结`Qwen3-Embedding-4B`2560维。不是Bonsai推理输出，不是通用BERT，也未接入未核实的COPD-BERT。

| field_id / Key | 中文 | 片段数 |
| --- | --- | ---: |
| 0 `history_of_present_illness` | 出院小结现病史 | 21,827 |
| 1 `past_medical_history` | 出院小结既往史 | 15,335 |
| 2 `hospital_course` | 出院小结住院经过 | 27,459 |
| 3 `discharge_condition` | 出院小结出院状态 | 12,353 |
| 合计 | 14,206次有文本住院；5,195次无文本预测点保留 | 76,974 |

规则筛选来源章节并分块，前缀为`Clinical field: <字段英文名称>`。同一shared_text.csv.gz行号分别对应BERT/Qwen矩阵，两个状态complete、清单SHA256一致；不新加prompt或生成摘要。ClinicalBERT为内容均值/L2策略，Qwen保存为last_token_l2、none_document_baseline；实际每行最大BERT395、Qwen531 token，不等于Qwen配置2048预算。向量FP16落盘，进网络转FP32。

片段输入`[K,768或2560]`＋`[K,2]`索引（批内住院位置、字段ID），不是每人只有一个768向量。投影后每字段内学习注意力汇总片段，再对实际存在的字段等权平均。不存在字段不填“正常”也不输入NONE嵌入；整个文本分支为空时向量0、存在标志0。

**本套不含独立影像报告向量、不单列出院药物/说明/重要检查章节**；住院经过里的相关内容可能存在但不等于单独提取这些字段。现病史是最终小结回顾入院，不能叫独立入院小结。既往史可重述频次，这轮没有屏蔽它，历史代理仍是边界。

## 9. 从张量到输出：真实底层维度

固定非文本标量按概念合计276=历史6＋当前14×2＋检验22×9＋药物7×4＋微生物5×2＋支持3×2。不是把276全部直接拼一层；最新实验将前34送冻结基础风险，诊断/去向临床槽位保持空。

| 分支 | 网络处理 |
| --- | --- |
| 冻结扩展基础 | 34→2风险幅度＋2共享raw_shape，共72参数；每患者输入不同 |
| 当前诊断/去向（最新实验为空） | 每概念[值,mask]2→32→128＋14类ID嵌入→带mask注意力汇总 |
| 检验 | 每轴9→32→128＋22轴ID嵌入→带mask注意力汇总 |
| 药物 | 28→32→128 |
| 微生物 | 10→32→128 |
| 呼吸支持 | 6→32→128 |
| 文本 | 每片段D经LayerNorm→Linear(D,128)→GELU→LayerNorm；加4字段ID；字段内注意力、字段间等权平均 |
| 融合 | 六个128分支＋六个存在标志=774→128→128，GELU、Dropout0.1、LayerNorm |
| 临床修正 | 128→32→4，末层权重/偏置为0 |
| 输出 | 两事件各[对数风险幅度修正,形状修正]，与冻结基础组合；联合Weibull NLL |

空分支在LayerNorm后再次按存在性屏蔽，防止偏置伪造观测。当前骨干是小型深度神经网络/MLP＋注意力池化；冻结文本编码器内部为Transformer，但风险骨干**不是跨模态Cross-Attention，也不是跨住院Transformer或RNN**。

风险累计量采用`H_e(t)=exp(a_e)×(t/365)^k_e`，a是幅度，不是传统Weibull尺度；k=softplus(冻结raw_shape＋个体shape修正)+0.1。联合处理死亡和再入院，积分输出无事件S(t)、再入院F1(t)、死亡F2(t)，三者和为1；不能将两个独立CDF直接相加。

```mermaid
flowchart TD
    R[原始快照和补充来源] --> H[历史6＋当前诊断/去向28]
    H --> B[训练后冻结基础风险34到2＋shape]
    R --> L[22轴异常/恢复198维]
    R --> M[药物28维]
    R --> U[微生物10维]
    R --> V[支持6维]
    T[共同四字段原文清单] --> E[冻结BERT或Qwen片段向量]
    E --> P[字段内池化及字段间平均128维]
    L --> F[各分支128维＋存在性融合774到128]
    M --> F
    U --> F
    V --> F
    P --> F
    F --> D[零起步临床修正128到32到4]
    B --> O[两事件幅度和个体形状]
    D --> O
    O --> C[连续时间S/F1/F2]
```

## 10. 已保存但不在本轮输入的资料

| 资料 | 状态和理由 |
| --- | --- |
| 完整原始315万检验 | 原始全保留；网络仅上述受控22轴，尚未映射的项目不默默输入 |
| pO₂/PaO₂及静脉/不明血气 | 原始/预处理可存在；不在22轴，避免未处理给氧背景时伪称单独氧合恢复 |
| EOS绝对值/百分比及首剂激素关系 | r3派生已核验；独立候选，未纳入最新compact网络 |
| PCT、其他炎症项目 | 不在22轴；不能仅凭临床重要就称已纳入，也不声称原库一定有可用覆盖 |
| 药物剂量/时间、出院吸入方案 | 原始/文本可有；当前药物输入未估算，医嘱不替代实际剂量 |
| FEV1/FVC、吸入技术、家庭照护事实 | 本轮没有专门事实字段；部分可能在四字段文本，但不能说已准确抽取 |
| NLR/SII及原成分 | 同标本派生/双种子对照已完成，无稳定增量，不默认入网 |
| CheXbert、RadBERT、RadGraph-XL | 影像独立实验已完成，未证明整体增益，不默认追加 |
| 规则r4候选原句/新ClinicalBERT | 已筛选、审计、编码及风险对照；未证实比旧B稳定改善 |
| Bonsai事实、其他机器抽取 | 另一路先导与审计，不自动替换B四字段；本轮不调LLM |
| A完整章节编码/MIMIC-III | 保留资料，成员/角色/行合同不同，不直接拼进本轮 |

文档中的“当前输入”由实际加载/训练合同决定，不由文件是否存在或曾提出决定。研究结果与目录入口见[当前研究状态](PROJECT_RESEARCH_STATUS_2026-10-11.md)。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V6_EVALUATION_PROTOCOL.md · lines 1–51

<!-- REVIEW_FRAGMENT {"path":"docs/V6_EVALUATION_PROTOCOL.md","part":1,"start_line":1,"end_line":51,"source_sha256":"23e6de086a3e40790616a0fad07e4208af7598c750316df17cd2f8e533b1205a","fragment_sha256":"23e6de086a3e40790616a0fad07e4208af7598c750316df17cd2f8e533b1205a","added_terminal_newline":false} -->
````markdown
# V6 第一轮评价合同：本库记录事件、单种子四臂

2026-10-09。对应 实现入口（本地或未公开引用`../training/v6/README.md`），保持严格491.21/J44.1任意诊断位非转院再入院与先死亡。没有可靠院外连续随访，不改变World A记录事件定义。

## 1. 第一轮固定比较

H：6项历史信息的线性双原因Weibull；C：冻结同一H、学习四个常数幅度/raw-shape修正；S：冻结同一H、使用14本次概念、22轴和7类二元暴露的临床修正；T：相同S结构加B四字段Bio_ClinicalBERT。

所有出院点住院等权、同患者同折。四臂seed=42；临床臂共同非文本初值，构造/设备迁移后重设随机流。所有臂按验证再入院7–365天每日积分Brier选择；允许epoch0作为最优状态，不能强行将无增益模型的更新选为最佳。H最多120epochs、patience20；其他臂最多60epochs、patience10；临床LR3e-4、5epoch warm-up/cosine，H0.01、C0.005，AdamW/weight decay1e-4/梯度裁剪1。临床及常数修正增加1e-4平均平方收缩，不对幅度与shape假定相同物理意义。每epoch完整训练折，固定住院小批、打乱批次顺序，不是微型样本训练。

阶段一、阶段二均只训练/验证；全部模型选择固定后才统一生成历史测试预测。旧历史测试已反复使用，不是新盲测；本轮bootstrap不反馈选模、调参或删字段。

## 2. AUC(t)与ROC(t)

目标原因c的病例 `D=c且T≤t`；其余所有点为对照，包括t前先发生其他原因者、t后目标事件者及World A截至365未记录事件者。因没有提前删失标记，本协议不估计IPCW，直接按已知记录终点二元秩计算；这与 `timeROC` 包所列**定义(ii)/AUC_2**的病例对照划分对应，不等于已经复现完整随访下的IPCW实现。[作者维护的timeROC说明第22–24页](https://cran.r-project.org/web/packages/timeROC/timeROC.pdf)

同分数校正半分；每日病例或对照少于20时AUC标NA。均值是7–365天有效每日AUC的算术平均，不叫整合生存AUC估计器。ROC按同分数同时纳入，30/90/365天展示；与sklearn ROC-AUC的合成等价测试通过。

死亡指标含义是“严格再入院前死亡”，不是再入院后仍继续观察的全因死亡模型。

## 3. 时间排序

调用已保护的 `training/v5/competing_concordance.py`，固定风险分数为F_c(90)或F_c(365)。对每个horizon前目标事件点i，可比j为更晚目标事件、竞争事件或World A无记录目标事件；同日目标对不比较，预测同分计半分。竞争事件可以更早发生，因为发生后已不能成为首目标事件。这一对比逻辑参照 [Wolbers等第2.1节](https://academic.oup.com/biostatistics/article/15/3/526/223919)。

**不是普通Harrell C，也不是论文全部IPCW/独立患者推断的复现。**本项目有重复出院、统一行政终止且实际外院信息未知，明确命名为项目固定horizon CIF concordance，报告可比对数。合成测试覆盖较早竞争事件、较晚目标事件和目标同时事件。它描述时间顺序，不输出“预测返院日期误差”。

## 4. 概率误差与校准

再入院/死亡各自Brier(t)为记录二元结果与对应CIF的平方误差平均；积分为7–365天梯形积分/358。均值累计发生曲线、30/90/365风险分位组预测/观察比例也交付；观察是World A记录比例，不能称确认的院外发生率或完整随访AJ校准。

CIF独立FP64积分；S/F1/F2概率守恒、单调性、范围检查；训练模型各折前64个预测点对照0.25/0.125天步长，若最大误差超过2e-4拒绝本轮数值验收，不能只凭曲线和为1断言原因分配准确。

## 5. 配对不确定性与分层

200次患者聚类配对bootstrap：同一次重采样用于两模型，抽中患者保留其全部预测点；住院等权点估计不变。给出积分Brier差及90天AUC差的百分位95%区间。没有使用普通DeLong独立样本检验或HL全曲线检验。区间条件于已训练模型，不含训练种子波动；不作多重检验校正，跨零不等于无效或等效。

统一比较C−H、S−C、T−S、T−C。按既往完成COPD住院0、1–2、≥3次分层报告相同指标及事件数；首次可预测出院不自动叫首次全生命周期入院。

## 6. DCA和解释性边界

30/90天DCA仅探索“高于阈值者加强随访”，阈值0.02–0.5作为假定误干预/漏干预权衡范围，净获益为TP/N−FP/N×pt/(1−pt)。没有挑选临床最佳阈值、核实随访成本或证明干预有效。死亡作为不发生首严格再入院的对照，仍需结合死亡风险/照护目标解释；不能据DCA制定医疗建议。

输出历史分层热力图、每epoch分支梯度RMS、幅度/shape修正分布。梯度、池化权重、热力图不是SHAP，更不是因果效应。未额外加SHAP/LLM/RAG或条件扰动候选，避免扩展本轮归因。

## 7. 验收证据

- 本库全量19,401次合格出院；训练13,592/3,766人，验证2,949/797人，历史测试2,860/788人。
- 所有臂从磁盘重载检查点，验证预测与保存最佳状态核对。
- 历史分支不更新，临床/常数修正epoch0与H完全等价。
- 全量源文件/清单/向量/检查点哈希登记，原始和V5保护。
- 一种子、无新私有数据外部验证；第一轮结果不得宣称稳定泛化或最终临床效益。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V6_FAST_SCREENING_POLICY_2026-10-14.md · lines 1–9

<!-- REVIEW_FRAGMENT {"path":"docs/V6_FAST_SCREENING_POLICY_2026-10-14.md","part":1,"start_line":1,"end_line":9,"source_sha256":"e859de8818fb1a1082b9ee6f3ec4410ec899764eb4df5a7baeb56593f3d6a5e9","fragment_sha256":"e859de8818fb1a1082b9ee6f3ec4410ec899764eb4df5a7baeb56593f3d6a5e9","added_terminal_newline":false} -->
````markdown
# 用户最新实验预算：单种子先筛选

2026-10-14用户要求加快进度，减少无必要重复实验，可只使用一个种子。新探索默认固定seed42：最小匹配对照+候选，不再先跑三种子、弱候选五种子扩张或重复确认。已有批次与锁定不重写；正式稳定胜出标准不降低。

保留必要而非机械验证：来源hash/患者划分/标签/训练尺度检查；相关单元和真实GPU初始化、缺失gate、真实单步；训练后一次独立新进程重载完整视图/权重/概率、早停与选点核验。通过数据/代码已核验且hash未变时复用，不重复全量原文抽取、BERT编码或两次相同来源重建。只有出现明确实质信号或边界疑点再增加复核、区间与多种子确认。

新增风险训练早停默认min10/patience7/min_delta1e-5/max45；预训练按独立无标签内部患者保留集早停，具体预算在每批合同锁定，不凭外调参临时延长。只用原训练/指定调参，确认/历史测试不成为搜索标签。仅单种子筛选，不称稳定提升、不改变默认模型。

原V6/所有有效输入/原数据/向量权重保留。仅常规表格与文本、旧三支持汇总保留，不使用高频96/115，不上传病历/启外部AI/推送Git。当前首项新机制见V6_FAST_MULTIVIEW_PROTOCOL_2026-10-14.md。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V6_HISTORY_AND_ALERT_EXPLAINED_2026-10-14.md · lines 1–64

<!-- REVIEW_FRAGMENT {"path":"docs/V6_HISTORY_AND_ALERT_EXPLAINED_2026-10-14.md","part":1,"start_line":1,"end_line":64,"source_sha256":"5c706f0f9b0ace7f260f65b5dbfde7108aa2daa3e7c3b971ececa43b56eb04bc","fragment_sha256":"5c706f0f9b0ace7f260f65b5dbfde7108aa2daa3e7c3b971ececa43b56eb04bc","added_terminal_newline":false} -->
````markdown
# 历次住院如何进入本次预测，以及预警文献补充

本轮仅解释、原始文献查询和已保存预测的早期聚合检查，0训练/0编码/0默认修改，不恢复长期自动研究。

## 1. 用三个住院单元理解当前代码

假设同一患者已结束两次住院，现在第三次出院。每次临床表格和可见四字段小结都用同一套可训练工具整理成128个数，作为该次住院的浓缩记录；这些数不是128个明确命名的临床指标，也不是先算一遍该次最终V6风险。

过去两次记录各加入当次34维背景转换后的128个数、距本次入院的间隔，成为两个历史单元。第三次的本次临床浓缩记录与两历史单元共同进入两层纵向网络。本次记录经这个网络处理后的结果，产生“参考过去之后”的补充修正；另保留仅从本次临床产生的修正。两种修正加到本次冻结H34的基础参数上，得到全年再入院/竞争死亡/尚无事件的概率。

过去H34是当次输入，不是过去风险概率；当前H34基础用本次输入算，权重冻结但每个患者结果不同。当前clinical单元没有同一context投影，因此该组织存在当前/过去不对称，尚不能称最优。网络不是人工写好“越来越重则更危险”，它根据训练患者真实事件时间和原因学习关系；注意力权重不是因果证据。

|名称|实际含义|
|---|---|
|新鲜本次|临床网络权重重新初始化后训练，只读本次详细临床资料；仍有H34历史摘要和小结既往史，不是完全无历史|
|完整历史|在本次资料外读取所有此前已结束的合格住院的现有临床/背景资料，安全可见过去小结；不是MIMIC所有表，也不是仅住院次数|
|原检验|核心九通道异常摘要与额外检验/EOS分支保持旧处理；额外分支含原值首/极/末与标记，不是所有项目统一异常表示|
|统一检验|核心和额外项目统一为异常/变化/复测/陈旧度九通道，保留低高方向和名称，在同一检验分支处理；EOS药物前后语境保留|

四条件就是“只本次/带完整历史”与“原检验/统一检验”的四种组合，原已训练V6另作参照。新鲜不指新患者。编码指把资料转换成模型可处理的数；投影指把不同长度数列转换到共同长度；本次解释不要求用这些术语理解流程。

## 2. 第1～6天没有被训练忽略，但旧汇总排除了它们

已核对`training/v6/risk.py`真实时间NLL及`evaluation.py`：训练未要求事件时间>=7；每天预测包括0–365；旧平均AUC/IBS是7–365。这是沿用的项目评价设置，不是“前6天不能加重”的临床假设。Brier每日数组包含前6天，但未纳入旧汇总/展示时点。

同1531调参记录的已保存标签，1/2/3/4/5/6/7日累计严格再入院为2/5/8/11/12/19/24，竞争死亡为1/4/6/6/7/9/10。来源仅原V6及history_unified保存预测，不读取确认/历史测试或原病历。

|日|原V6再入院Brier|历史统一再入院Brier|
|---|---:|---:|
|1|.001358944041452398|.001300250296148181|
|2|.003396963140047268|.003271050561567992|
|3|.005459413752925029|.005256403523558817|
|4|.007480767894290642|.007160991674691209|
|5|.008294510072711270|.007878479832172998|
|6|.012365041920181508|.011986471914404540|
|7|.015393024308570150|.014975950852946913|

仅描述同折点估计，极少病例且接近“全不发生”的低事件率，不能称早期可靠突破。预测源SHA：原`da4ab84e3523a3255e0210e7ef36ef4c5f113006db9b3fe489235c1bc838be5c`；历史统一`74369bdb8e0dec33797a9fab577917326a2366b95a4910e914efc65374d7d233`。未改既有报告/proof或声称完成新的模型独立重放。

后续评价应保留旧7–365便于对照，同时单列1/3/7日病例数、Brier、检出/误报，增加1–365概率误差汇总。若坚持至少20病例才报告AUC，当前前6天均不足20，1–365有效日平均AUC会与旧口径一样，不应用0填空；不能据此不报告早期事件。此为建议，未修改受保护评价代码或重选epoch。

## 3. 预警文献：不是所有报警都同一个任务

查询日期2026-10-14；以下来自原论文摘要/出版社可检索结果、已保存原始研究报告及官方方法文档。部分PMC/出版社正文此次403或验证码，未绕过，也不冒称本轮逐页精读全文。

### 出院/基线时预测未来风险

- **PEARL，Thorax 2017**：2417人，90日再入院或死亡复合终点。既往住院、气促、年龄、左右心衰的评分，外部AUROC .70；0–1/2–4/≥5分风险组观察比例20.7%/42.1%/66.4%。用途是照护分层，不是当天症状报警；复合事件不同于当前严格再入院。原始[PubMed](https://pubmed.ncbi.nlm.nih.gov/28235886/)及[论文](https://thorax.bmj.com/content/72/8/686)。
- **ACCEPT 2.0，2022**：ECLIPSE1803人再校准、TORCH1091人外部验证；下一年≥2中度或≥1重度加重。外部AUC .756，另校准与决策曲线，比较多个阈值下模型、病史、全处理、都不处理的净获益。不是宣布F1最大阈值就是最佳医学干预阈值。[原论文](https://doi.org/10.1016/j.eclinm.2022.101574)。本轮核对出版社索引方法/结果，正文入口403，不复制其阈值当本项目处方规则。

### 持续观察后发现即将加重

- **COPDPredict，2021**：90人、6月前瞻观察；每日主观状态/FEV1及CRP，基于个人变化阈值决策树。112次临床定义加重；报道提前中位7天、敏感度97.9%、特异度84.0%、PPV38.4%。这些敏感度等由重复观察时间点交叉表产生，不是90名患者的独立预测准确率。并非随机干预试验，不能把前后住院下降当因果疗效。[原论文](https://www.dovepress.com/validation-of-copdpredict-unique-combination-of-remote-monitoring-and--peer-reviewed-fulltext-article-COPD)。本轮核对作者原始摘要/可检索结果，全文入口403。
- **Pinnock等，BMJ 2013随机试验**：256人，症状/用药和SpO2远程监测触发警报；首次住院时间未显著改善，调整HR .98，95%CI .66–1.44。证明“会报警”与“报警后能减少住院”是不同验证层次。[原始PubMed摘要](https://pubmed.ncbi.nlm.nih.gov/24136634/)。本研究数据需求不同，不据此加入用户禁止的高频轨迹。

### 阈值选择的方法依据

[scikit-learn官方阈值文档](https://scikit-learn.org/stable/modules/classification_threshold.html)明确分开概率估计与行动决定，支持训练后按F1等指标调阈值，调阈值不改变ROC/PR曲线；强调使用独立于拟合的数据。此是通用工程依据，不等临床预警已批准。我们的训练折拟合、调参折选阈值，但同调参折的最大F1仍是选择性能，需另一折锁阈值评价才验证泛化。

## 4. 对项目的实际含义

我们目前是每次出院根据截至该次资料预测未来1–365日风险，不是每天有新居家观测后更新的“即将加重探测器”。每天一项概率不等每天重新接收新信息。对于下一步临床预警研究，除AUC/F1，应考虑锁阈值后的期限别检出、PPV、每百次出院预警量、误报/漏报、概率校准和假定代价净获益；若真正转向连续报警，还需明确提前量和每患者月误报/重复报警归并。未规定报警动作/代价时，最大F1只能作为工程参照。

此前已有风险模型17项研究及固定阈值/DCA读出，不是完全没有调研；但完整报警工作流、前瞻锁阈值和干预效果未完成，本次文献不冒充已实施功能。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V6_MULTIHORIZON_READOUT_PROTOCOL_2026-10-14.md · lines 1–15

<!-- REVIEW_FRAGMENT {"path":"docs/V6_MULTIHORIZON_READOUT_PROTOCOL_2026-10-14.md","part":1,"start_line":1,"end_line":15,"source_sha256":"66f01b6b47dfa9380de0f3068728adfe538d95da1ad8f87dc16defc5be48922c","fragment_sha256":"66f01b6b47dfa9380de0f3068728adfe538d95da1ad8f87dc16defc5be48922c","added_terminal_newline":false} -->
````markdown
# 多时间点预警与全年曲线读出合同

用户直接要求中短期和全年均报告，并与之前比较。仅复用正式一年F1实验的五份已独立重放预测，不重新拟合模型、BERT、校准器，不恢复长期自动研究。

- 同1531指定调参出院、同患者/医院顺序、同严格再入院和竞争死亡标签。
- 保留已选择的checkpoint，不按短期分数另选epoch；旧V6和四个新条件全部报告。
- 时间点7/14/30/60/90/180/365日，两原因均报告。真实标签为该原因在指定时点之前先发生；不是相互排斥的时间区间任务。
- 三类阈值读出：每模型每时点在同调参集按最大F1选择（探索性、偏乐观）；既有365日阈值原样用于各时点（描述期限错配）；共同固定10%/20%/30%阈值（直接比较相同概率行动规则）。这些都不用于重新提名模型。
- 阈值不是梯度训练的模型参数。排序AUC、平均AUC、Brier、固定箱概率误差均不受预警阈值改变影响。
- 平均AUC沿用原7–365日有效每日AUC均值；额外只做7–30/31–90/91–365分段均值，同时报告有效日数。病例或对照不足20不计算该日AUC，不用0补值。
- 固定概率箱沿用旧读出0/.1/.2/.3/.5/1，保存平均预测/事件比例/箱误差，不把分箱校准描述称单患者真概率验证。
- 新结果必须复现父报告每日评价及365日阈值/混淆数。一次独立新CPU进程重新读取五预测文件并重算，记录新源码/协议/测试、父报告/proof/contract、五预测与best权重hash。
- 不读确认或历史测试，不更改终点、队列、人群、原模型默认或已锁定提名。

历史结构说明：当前过去住院单元是clinical128 + H34输入投影128 + gap128，相加后进入纵向Transformer；不是仅历史H34，也不是过去基础风险概率。当前单元不加同一context投影，而H34当前风险在最后合成。本次只核对/解释此非对称性，不把多时点评价当成“拼接优于相加”的架构实验。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V6_MULTIHORIZON_READOUT_RESULTS_2026-10-14.md · lines 1–76

<!-- REVIEW_FRAGMENT {"path":"docs/V6_MULTIHORIZON_READOUT_RESULTS_2026-10-14.md","part":1,"start_line":1,"end_line":76,"source_sha256":"a225dc7f7aaaf62665c96532113ffa72d1638689a6f9c293200595034c906230","fragment_sha256":"a225dc7f7aaaf62665c96532113ffa72d1638689a6f9c293200595034c906230","added_terminal_newline":false} -->
````markdown
# 中短期及全年风险：五个已保存模型的共同读出

本次按用户直接要求补充评价，0新训练/0编码/0模型或epoch重选。正式目录`training_runs/v6_multihorizon_readout_2026-10-14`。复用一年F1正式r2五模型的完整每日预测，同1531指定调参出院/402患者，非独立盲测。过去实验的年度最佳checkpoint保持不变，不能把这些结果称为专门为30/90天重新训练的模型。

## 1. 历史单元到底有没有结合H34

当前代码每个过去住院读取完整现有临床输入，得到共享可训练clinical128；该次H34的**34维输入**经LayerNorm→Linear34→128→GELU投影，再与clinical128和gap投影相加，作为同一个历史token进入两层Transformer。不是只读过去H34风险，不是把过去住院的最终V6概率当历史，也不是把两条完全独立曲线相加。

当前住院的clinical128作为查询/读出位置；当前H34基础风险在最后参数合成。当前token不加同一H34上下文投影，这与历史token存在非对称性。本次只确认实际代码，不宣称此设计最佳。冻结基础权重与历史是否采用联合单元是两个可以独立控制的设计选择。

后续若比较组织方式，应保持相同历史成员/安全文本/当前住院路径/风险基础，只比较过去单元的分支相加与联合非线性融合。**先拼接再单一线性投影，与两个线性投影相加可数学等价**，不能把改张量布局当新机制。联合方法要确有非线性交互才值得训练；已有“有/无历史×原/统一检验”不等于做过这项比较。

## 2. 阈值怎样获得

风险权重用真实事件时间+事件原因的连续双原因生存NLL训练。阈值不是梯度优化参数，模型已生成概率后，在指定调参折将不同概率取值依次作为阈值，计算F1，选最大；并列取更高阈值，相同概率不得拆分。年度实验还以该年度调参F1选epoch。

本次保留checkpoint，分别为7/14/30/60/90/180/365日读出三种策略：各期限调参最佳F1阈值；既有年度阈值原样应用；固定10%/20%/30%相同行动阈值。调参最佳阈值和同折F1是探索读出，不是独立验证。死亡也按先于再入院的原因2单列，不等总死亡。AUC/平均AUC/Brier/概率校准统计不受行动阈值变化影响。

## 3. AUC与概率误差：同折直接对比

|模型|30日AUC|90日AUC|365日AUC|7–365每日平均AUC↑|再入院IBS↓|竞争死亡IBS↓|
|---|---:|---:|---:|---:|---:|---:|
|原完整V6|.797710|.796093|.794785|.802141|.098729|.064127|
|新鲜本次＋原检验|.802708|.794803|.790145|.801820|.101853|.065006|
|本次＋统一检验|.784972|.790723|.795262|.798570|.100030|.064742|
|历史＋原检验|.785126|.784031|.781437|.790788|.099804|.065668|
|历史＋统一检验|.793989|.798131|.799740|.805225|.099664|.064971|

平均AUC沿用原口径：第7–365日，每日累积病例/其余记录比较，至少20病例且20对照才有效，再对有效日AUC算术平均。再入院359日均有效；死亡350日有效，不用0填空。不是七个展示时点平均，不是五模型/五种子平均，不是准确率。

|模型|7–30平均AUC|31–90平均AUC|91–365平均AUC|
|---|---:|---:|---:|
|原V6|.802589|.793428|.804003|
|历史＋统一检验|.811615|.791771|.807603|

历史＋统一检验全年均AUC比原V6高.003084（约0.308个百分点），但30日AUC较低，31–90段均AUC较低，再入院IBS约相对恶化.947%，死亡IBS也较差。原检验历史均AUC反而低。这是“数据组织与纵向相互影响”的单seed线索，不是历史一律有效或整体稳定胜出，不更改原提名/default。

## 4. 各期限最佳调参F1

|模型|7日|14日|30日|60日|90日|180日|365日|
|---|---:|---:|---:|---:|---:|---:|---:|
|原V6|.142857|.279720|.372093|.414861|.451613|.552486|.564706|
|新鲜本次＋原检验|.147239|.271357|.371308|.416382|.449568|.530769|.587112|
|本次＋统一检验|.123894|.251969|.338983|.388489|.436975|.542314|.571429|
|历史＋原检验|.163934|.277457|.365217|.407895|.452174|.546125|.571053|
|历史＋统一检验|.173913|.303571|.380000|.424460|.471264|.539964|.573281|

事件数分别24/49/81/130/160/257/359。不同期限的事件率不同，不把F1随期限变高当模型改进。历史统一的多数期限F1点估计略高，180日低；全年最高F1是新鲜本次对照，不是历史统一。每列都在同调参折选阈值，含乐观选择效应。

|期限|原V6阈值|历史统一阈值|原V6检出/误报/漏报|历史统一检出/误报/漏报|
|---|---:|---:|---:|---:|
|30日|20.13%|13.13%|32/59/49|38/81/43|
|90日|20.74%|20.05%|84/128/76|82/106/78|
|365日|23.33%|25.47%|240/251/119|221/191/138|

30日F1高伴更多误报；90日F1高主要是减少误报，而非多检出。F1最大化不等于召回率最大化或已确定临床处置策略。

## 5. 相同阈值及概率可信度

90日固定20%时原V6预警220、检出85/160、误报135、漏75、F1 .447368；历史统一预警188、检出82、误报106、漏78、F1 .471264。新模型少误报29，同时多漏3。30日固定20%时原V6 F1 .363636、历史统一 .321168：后者不再有优势，说明“各自调阈值后的比较”不能代替相同规则比较。

若把年度25.47%阈值直接用于历史统一30日，F1只有.280702；用30日自己的13.13%时是.38。短期事件率与累计概率低，不能机械沿用年度阈值。共同固定阈值适合说明工作负担，实用阈值仍应按预测期限/误报漏报代价/验证证据确定。

|期限|原V6 Brier↓|历史统一Brier↓|原V6平均预测|历史统一平均预测|库内事件比例|
|---|---:|---:|---:|---:|---:|
|30日|.044166|.043848|5.5435%|4.5811%|5.2907%|
|90日|.077665|.077716|10.8434%|9.6063%|10.4507%|
|365日|.135978|.136764|22.1786%|21.5226%|23.4487%|

30日概率误差略有利，90/365不利，不能由平均AUC较高推断个人概率更可信。全部时点两原因的Accuracy/PPV/召回/FPR/TP/FP/FN/TN/预警负担、固定共同概率箱与误差均在report.json中，未重新校准。WorldA未在库内观察不等院外未发生；FINAL文本仍回顾性。

## 6. 实施与核验

新增`training/v6/multihorizon_readout.py`与8边界回归：日期边界/竞争事件、阈值等号、年度阈值不重调、最佳F1不低于共同阈值、无效AUC日不填0、标签行错配、无效期限/原因、非守恒曲线。8项通过。

构建/一次独立CPU进程核验均复现五模型全部原每日指标及年度阈值/混淆数，新读出完整JSON误差0。记录父report/proof/contract/nomination、五预测、四best权重、新代码/测试/合同hash。report SHA `47f07b99b1a1b20aa4b092ba365d8f8857229ad968ab141bce07d7536f4142e7`。已生成并实际查看四面板英文图。未新增训练或改保护模型、数据、标签、编码和权重；未看确认/历史测试，未恢复自动研究。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V6_SYMMETRIC_CONTEXT_PROTOCOL_2026-10-14.md · lines 1–13

<!-- REVIEW_FRAGMENT {"path":"docs/V6_SYMMETRIC_CONTEXT_PROTOCOL_2026-10-14.md","part":1,"start_line":1,"end_line":13,"source_sha256":"c475b59d6ef80da61c5042316715a9b83c470d07e08a4b5ccd1f159fa67e4bea","fragment_sha256":"c475b59d6ef80da61c5042316715a9b83c470d07e08a4b5ccd1f159fa67e4bea","added_terminal_newline":false} -->
````markdown
# 本次与历史背景一致接入：有限校正实验

用户直接授权“校正模型，实验一下”。仅检验已确认的当前/历史背景不对称，不同时改128宽度、检验表示、标签或外部数据。使用已核验统一检验及完整历史接口，不重抽病历或编码BERT。

- 两条件seed42：asymmetric保持旧历史临床+当次H34输入背景+gap，本次只临床；symmetric本次也加入同一个背景读取器，gap=0。旧本次临床直达修正头仍原样，改变仅纵向网络当前单元；H34最终基础冻结不动。
- 每次住院只使用其当次背景；所有过去入院前已结束及历史小结存档规则不变。当前H34进入历史比较不是额外相加一个风险概率。
- 相同fresh clinical/共享context读取器/时间网络/末层零初始化、RNG/batch/优化器、参数量，两臂共同H34初始风险0差。无历史者纵向修正gate=0，训练后共享临床变化仍可能影响无历史者，不称无历史曲线固定。
- AdamW lr3e-4/wd1e-4，delta平方惩罚1e-4，clip1，FP32；batch128。真实时间双原因NLL训练，年度调参F1选点、并列IBS/早轮；min10/patience7/min_delta1e-4/max45与父实验相同。不改中短期checkpoint，期限别阈值仅探索读出。
- 两条件实际重新配对训练，并核对asymmetric对父history_unified的初值/选点/预测复现。保留旧完整V6强参照，不把当前条件与原模型不同fresh状态差异归为校正效果。
- 必要回归：当前/历史单元的一致构成、control数值原样、缺失gap/历史屏蔽、内容敏感/共同初始风险、H34无梯度/权重不变；真实GPU单步及context/temporal更新。
- 来源/标签/base/teacher/新源码hash；训练后一次独立GPU进程重建输入/缓存，best与last权重每日曲线、全部指标、早停及选点重放。只有train/指定1531调参，不用确认/历史测试重选或改默认。
- 保留旧7–365平均AUC/IBS，新增1/3日病例数/Brier/阈值混淆数、1–365积分Brier、7/14/30/60/90/180/365 AUC/F1/PPV/Recall/FPR、死亡及固定箱校准。AUC不足20病例/对照仍标缺失，不以0补值。连续每日曲线不变。
- 所有阈值选取/校准图来自曝光调参折，单seed不称稳定临床成功。公开Git仅源码/汇总文档，不发布任何患者级资料/预测/向量/模型权重/病历内容。长期自动任务仍取消。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V6_SYMMETRIC_CONTEXT_RESULTS_2026-10-14.md · lines 1–59

<!-- REVIEW_FRAGMENT {"path":"docs/V6_SYMMETRIC_CONTEXT_RESULTS_2026-10-14.md","part":1,"start_line":1,"end_line":59,"source_sha256":"bec8bd6f2f397935c0fd227e2d1a25466302ae1dd5b2fd4102d625f4400cb500","fragment_sha256":"bec8bd6f2f397935c0fd227e2d1a25466302ae1dd5b2fd4102d625f4400cb500","added_terminal_newline":false} -->
````markdown
# 本次/历次背景对称接入：实际有限实验结果

用户2026-10-14直接授权“校正模型，实验一下”，本次只修当前背景未提前进入纵向对照的结构不对称，不同时加宽128维、改检验/文本/标签或恢复长期自动研究。原模型/受保护源码/输入/权重未覆盖。

## 1. 修改与真正匹配对照

旧条件：过去临床128＋各自背景读取128＋gap进入历史token，本次token仅临床128。

新条件：本次token也加入同一个背景读取器得到的128；所有住院都先含其当次背景，再一起比较。当前临床直接修正头原样保留，最后冻结本次H34基础原样保留，不重复相加风险概率；背景读取器与纵向网络可训练。无历史的纵向补充仍gate0。

两臂seed42、相同参数/初始化、相同完整过去成员/安全BERT来源、统一检验，FP32/batch128、AdamW3e-4/wd1e-4/delta惩罚1e-4、clip1、NLL训练；年度调参F1选点，并列IBS，早停min10/patience7/delta1e-4/max45。不是改为短期二分类，不重抽/重编码。

正式`training_runs/v6_symmetric_context_2026-10-14`。两臂实际各12轮、最佳各5轮，共24实际epoch。原不对称对照与上一轮history_unified保存每日曲线最大差0，确定不是换训练状态再称校正效果。

## 2. 一年结果与原V6强参照

同1531指定调参出院/402患者，严格年度事件359；各自阈值/选点在调参折决定，不是独立盲测。

|模型|年度阈值|年度F1|精确率|召回率|TP/FP/FN|平均AUC7–365↑|再入院IBS7–365↓|竞争死亡IBS↓|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|原完整V6|23.33%|.564706|48.88%|66.85%|240/251/119|.802141|.098729|.064127|
|匹配不对称历史|25.47%|.573281|53.64%|61.56%|221/191/138|.805225|.099664|.064971|
|校正对称历史|26.71%|.577373|54.15%|61.84%|222/188/137|.806332|.099614|.065008|

校正对匹配对照：年度多检出1/少误报3，F1+.004092，平均AUC+.001107；再入院IBS改善约.04985%相对，仅极小变化。死亡IBS略差。对原完整V6均AUC+.004191，但再入院IBS约相对差.8965%、死亡也差，不能称整体稳定超越。

## 3. 中短期不能只看全年

|期限|原V6 AUC|旧历史AUC|校正历史AUC|旧历史最佳调参F1|校正历史最佳调参F1|
|---|---:|---:|---:|---:|---:|
|7日|.756608|.804274|.800791|.173913|.196721|
|14日|.805654|.812319|.803299|.303571|.306306|
|30日|.797710|.793989|.791103|.380000|.358491|
|60日|.785417|.785373|.785779|.424460|.422713|
|90日|.796093|.798131|.798710|.471264|.457300|
|180日|.812747|.813340|.814705|.539964|.542056|
|365日|.794785|.799740|.802597|.573281|.577373|

每期限阈值探索不重选epoch，不等各期限独立训练。对称后30/90日F1变差，而年度略好；90日固定20% F1旧.471264→新.453258，年度固定20%旧.553377→新.545833。各自年度最大F1改善不等相同预警规则改善。

## 4. 绝对概率与首六天

|期限|原V6 Brier|旧历史Brier|校正历史Brier|校正历史平均概率|库内事件比例|
|---|---:|---:|---:|---:|---:|
|30日|.044166|.043848|.043938|4.7036%|5.2907%|
|90日|.077665|.077716|.077917|9.9132%|10.4507%|
|365日|.135978|.136764|.135964|22.3236%|23.4487%|

一年单点Brier与原V6接近且极小有利，不能取代全曲线/死亡检验。固定概率箱年度绝对差旧历史.021874→校正.014081；90日.016398→.011187（依赖箱划分）。未拟校准器，均值/分箱不能证明个人真概率准确。

新增1/3日累计事件和Brier/F1混淆记录、1–365日积分误差（旧.098141224→新.098091658），没有把前六天事件排除训练；早期AUC因病例不足20保留缺失而非补0。全部两原因7/14/30/60/90/180/365结果/固定10%20%30%/缺失role读出在本地report.json中。

## 5. 验证与结论

20相关合成回归通过。真实GPU两臂参数与初始风险0差；H34无梯度/权重不变；背景/纵向/历史头/检验/风险头真实更新，改变当前背景在baseline之前引起delta改变.000285536（仅接口证据，不是临床效应量）。

一次独立新GPU进程重建原输入/历史缓存、best/last、每日风险/所有指标/早停/选点，asymmetric与symmetric重放均0，原匹配对照父曲线也0。验证文件绑定当前report SHA，未重训核验、未读取确认/历史测试。公开Git只代码与汇总，无患者资料或权重。

**本次校正已实际实现并做实验。** 表达结构更一致，年度F1/平均AUC点估计小有利，但中短期/全曲线概率误差/死亡没有一致改善。保留为可复现候选，不自动替换原V6、不因单seed称稳定成功，也不机械扩种。下一研究若继续，应针对较早压缩或阈值/目标组织提出新机制，不能把本次弱信号当完成全部目标。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V6_TASK_OPTIMIZATION_RESULTS_2026-10-13.md · lines 1–17

<!-- REVIEW_FRAGMENT {"path":"docs/V6_TASK_OPTIMIZATION_RESULTS_2026-10-13.md","part":1,"start_line":1,"end_line":17,"source_sha256":"5ee022cfd4aec97f74791e41e4089f637f979e4a8fc14aa42962b6fc17404768","fragment_sha256":"5ee022cfd4aec97f74791e41e4089f637f979e4a8fc14aa42962b6fc17404768","added_terminal_newline":false} -->
````markdown
# 第二批双原因表示与共享梯度：完整结论

八条件×三临床种子各100完整epoch全部完成；24模型调参重放及9模型锁定确认重放最大误差0。后处理容器退出0；图表已实际打开核对。原V6未被覆盖，长期目标尚未完成。

调参IBS均值选择独立临床编码支路`independent32`。按事前名单一次确认，未在确认后更换候选。

| 同一确认子集三种子均值 | AUC90↑ | Brier90↓ | 再入院IBS↓ | 死亡IBS↓ |
|---|---:|---:|---:|---:|
| 本批匹配原V6 | 0.792171 | 0.065084 | 0.089834 | 0.062797 |
| 独立临床支路 | 0.792723 | 0.064974 | 0.089644 | 0.061227 |
| 旧正式V6 | 0.793106 | 0.064922 | 0.089576 | 0.062062 |

相对匹配对照AUC90均值仅+0.000552、IBS相对降低0.212%；相对旧正式V6AUC90−0.000383、IBS相对恶化0.076%。仅1/3种子AUC/IBS同时改善，两参照下均未达到扩种筛选规则。不据seed42单次较好结果宣传突破；患者配对区间均跨零，没有多重比较调整。

死亡IBS和90天平均概率校准出现有利线索，不能因为再入院没有突破就抹去，也不能把它说成再入院突破。原始次数层内IBS并非所有层/种子改善；首次住院层AUC病例不足不可计算，不当作零效果。现有验证有历史曝光，不是新盲测；测试没有参与本批搜索。

详见`training_runs/v6_task_optimization_2026-10-13/`中的TUNING_RESULTS、CONFIRMATION_RESULTS、calibration_and_history_strata.json、两次verification、continuation_decision.json及完整曲线/权重。保留死亡方向线索供后续，不机械进行弱候选五种子复验。接着进入已有事前合同的条件化修正，并显式控制“仅基础再校准”造成的假临床增量。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V6_UNIFIED_LONGITUDINAL_F1_PRD_2026-10-14.md · lines 1–37

<!-- REVIEW_FRAGMENT {"path":"docs/V6_UNIFIED_LONGITUDINAL_F1_PRD_2026-10-14.md","part":1,"start_line":1,"end_line":37,"source_sha256":"48a6cce3bd35cec3584fbe3412a8645540737adf8b918fe86922227bf0123404","fragment_sha256":"48a6cce3bd35cec3584fbe3412a8645540737adf8b918fe86922227bf0123404","added_terminal_newline":false} -->
````markdown
# 本轮需求：统一异常检验、可学习完整历史、全年F1

用户2026-10-14授权有限实验，不恢复已取消heartbeat/长期Goal。用户纠正主期限为一年，不是90天。

## 目标与验收

1. 将现有额外五检验及EOS与原核心检验统一为同种异常/复测格式与共同检验编码/池化，不再把额外90维在融合之后另加；保留EOS已记录激素时间的专科上下文。
2. 纵向模型在每个预测点，读取本次与所有入院前已结束的历史合格住院的现有V6输入（检验、七药效、微生物、旧三支持、全部B四字段片段与过去诊断/去向），共享可训练住院编码器、两层时间间隔Transformer。不是读取固定历史C128缓存或过去风险数值；住院表示在训练中可重新学习。BERT backbone仍冻结，不重编码；“完整”限定现有核验V6输入，不宣称全原数据库全项目。
3. 主评价全年严格再入院F1/precision/recall/FPR/PPV/预警数量，验证患者独立于训练患者。同患者不跨折，全年每日联合概率/死亡/校准仍报告。

## 最小可归因实验合同

- 2×2四条件：原检验+本次，统一检验+本次，原检验+纵向，统一检验+纵向；seed42，同H34基础权重，所有有效共享模块共同新鲜初始化，临床修正末层零，epoch0为相同H34风险。旧完整V6作为不重新拟合强参照。不从已经过最优选择的V6临床权重暖启动后再把best0当新方案无效。
- 统一采用连续双原因NLL训练；按一年F1验证选epoch和阈值，IBS仅同F1时作固定第二排序。硬F1不直接求梯度，概率学习+验证阈值是常见实现；不为迎合主指标输出失去意义的未校准概率。
- 每epoch对全部指定1531调参预测，按所有不同分数的合法阈值精确找最大F1（同分不拆，F1并列选较高阈值，减少预警），不固定0.5。记录该阈值；checkpoint先F1最大、再IBS最低、再早期轮数。早停主量-F1，min10/patience7/min_delta1e-4/max45。训练batch128，AdamW lr3e-4/weight_decay1e-4/clip1，修正平方惩罚1e-4，无网格。
- 已曝光调参的最大F1会乐观，明确不是泛化F1。四条件与旧参照按验证F1一次锁定，若新方案胜新鲜匹配control与旧V6才入围；可锁定一次1418确认折评估入围与control及旧参照，严格沿用已锁阈值，不能在确认重选threshold/epoch/model。历史测试本轮不使用，不称新盲测。

## 数据规则

- 原核心异常统一九通道继承已核验raw及现有缩放公式。额外五项和EOS各拆低/高方向，共14新增方向，与原22内部方向一起36。它们是异常方向token，不是36个独立检验项目；对外统称检验分支。
- 新方向按本记录参考上下限构造`max(0,偏离/范围宽度)`，同项目低/高分开；未提供有效范围不补造正常、不猜阈值。pH/pCO2沿现有核验生理阈值不强行改成各检验单参考值。统一的表示不等于抹去标本/单位/临床阈值差异。
- 九通道统一首测、峰异常、末测、恢复差、末测年龄、峰到末时间、观测、峰后复测、末测处异常峰；原缺失保持mask。train每方向每数值拟尺度，不借调参/确认。EOS绝对/%单位分开，其8时间类别独立上下文合入检验分支，未记激素不当未给药。
- 原记录的末测可早于出院，不冒称出院当日。只从原常规labevents必要七概念读取一次新增数据，不读高频chartevents/设备96/115。原NPZ不改。
- 历史索引只同患者同折、入院前已出院、无当前/未来，复用已有checked索引。既往文本只用其对应文书已在当前入院前存档者；不可用文本mask关闭但该历史表格保留。未知文书关联不猜。当前FINAL仍原回顾性政策，明确不是实时部署。历史H34的过去诊断/去向作为历史输入，不重复把当前预测结局送入编码器。
- 全部历史住院保留，不截取最后几次；患者序列可在住院层压缩128，这种压缩可训练，与固定原教师C128不同。参数更新需实测历史Encoder梯度，不把可运行说成历史已被利用。

## 必要验证与来源

先单元：异常方向/范围缺失/正常观测/单测/峰后恢复，F1同分/无阳性/阈值复算，历史患者/时间/缺失文本/输入白名单。真实GPU检查共同H34起点、历史内容敏感、无历史gate、有效旧分支/新检验/历史模块梯度与更新。锁源码/data/labels/base/向量/索引/protocol hash，训练后一次独立新进程重建输入/重载权重/每日概率/主指标/早停/选点。不同reshape会有舍入，预先设FP32风险重放容差1e-6（初始共同路径应0）；不无限重试。

方法出处：[scikit-learn阈值与F1](https://scikit-learn.org/stable/modules/classification_threshold.html)已读正文；[BEHRT](https://www.nature.com/articles/s41598-020-62922-y)与[时间感知临床文本](https://proceedings.mlr.press/v126/zhang20c.html)提供纵向设计启发，不称复现它们或论文保证本队列增益。

## 启动前数值路径修复

初版`v6_unified_longitudinal_f1_2026-10-14`仅预检/缓存，0拟合，因新FP32/128batch与原保存TF32/256batch参照不同退出。独立真实256出院探针：原入口TF32恢复旧幅度/shape/每日曲线误差0；关闭TF32后旧入口曲线最大差.00019509；新128入口即使开TF32仍有批形状舍入差.00003605。原输入并未被证明缺失，不能直接提高容差当修复。

正式`_r2`明确四个新模型共同FP32/128计算，旧V6强参照独立复用原TF32/256真实入口并要求保存曲线逐值相同；反向训练仍新FP32，且原精度设置在参照函数结束恢复。原初版失败/协议源码快照保留在work/unified_longitudinal_preflight_failure_2026-10-14，未改原V6或数据；正式合同记录精度差别，不称FP32是性能根因。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V6_UNIFIED_LONGITUDINAL_F1_RESULTS_2026-10-14.md · lines 1–97

<!-- REVIEW_FRAGMENT {"path":"docs/V6_UNIFIED_LONGITUDINAL_F1_RESULTS_2026-10-14.md","part":1,"start_line":1,"end_line":97,"source_sha256":"92d0e1a07c959a44b25fc8a3a49c520e51e3feb6850e0850931cb3f19449c597","fragment_sha256":"92d0e1a07c959a44b25fc8a3a49c520e51e3feb6850e0850931cb3f19449c597","added_terminal_newline":false} -->
````markdown
# 一年F1：统一异常检验与完整可学习历史的单种子实验

本轮按用户新授权完成有限任务，**没有恢复长期heartbeat或Goal**。一年365天严格AECOPD再入院为主F1结局，30/90天仅辅助；原全年每日竞争风险输出、患者队列/划分不变。

## 1. 已实际做完什么

1. 统一检验接口`patient_data/processed/v6_unified_lab_state_2026-10-14`：原核心与五额外检验/EOS都用九通道异常/复测摘要，共同检验MLP及注意力池化，不再让额外90维在融合后另加。特殊EOS激素前后语境留在同检验分支；不同单位/异常方向及生理阈值没有混为相同数值。
2. 纵向结构：每个预测点读取所有此前已经出院的合格住院的现有临床表格/全部B四字段向量，共享住院编码器可训练，再两层128维/4头/FFN256/Dropout.1的时间间隔Transformer。历史诊断/去向另外编码；不是使用固定教师C128或过去风险。BERT仍冻结，临床编码/文本投影/时间网络共同更新。
3. 四条件2×2：仅本次/完整历史 × 原检验/统一检验。每个seed42，临床全部从相同新鲜权重、共同冻结H34开始，零初始化末风险层；原完整V6权重只作强参照、不再暖启动所有候选。按一年F1选epoch及阈值，F1并列按IBS，早停10/7/1e-4/45。

**F1**为精确率与召回率的调和平均，`2TP/(2TP+FP+FN)`，不是准确率。采用常见“监督概率训练+验证F1阈值调优”，不强行对硬阈值F1求梯度；每个模型都按同样规则找一年最佳阈值，不固定0.5、不在测试调阈值。训练损失保留连续时间双原因NLL；F1成为本轮选点/早停主量，概率校准继续单列。

## 2. 主结果：F1目标选出了不同模型

同1531指定调参出院/402患者，一年内严格再入院359次。以下阈值/epoch均在该已曝光调参折选择，**不是独立泛化F1**。

|模型|实际/最佳轮|一年阈值|一年F1|精确率|召回率|误报率FP/(FP+TN)|全年再入院IBS↓|
|---|---:|---:|---:|---:|---:|---:|---:|
|保留原完整V6|本轮不训练|23.33%|.564706|48.88%|66.85%|21.42%|.098729|
|新鲜仅本次＋原检验对照|16/9|19.05%|**.587112**|51.36%|68.52%|19.88%|.101853|
|仅本次＋统一异常检验|15/8|28.06%|.571429|54.41%|60.17%|15.44%|.100030|
|完整历史＋原检验|10/1|26.47%|.571053|54.11%|60.45%|15.70%|.099804|
|完整历史＋统一异常检验|12/5|25.47%|.573281|53.64%|61.56%|16.30%|.099664|

四条件实际53epoch/5671优化器更新，无一固定满45或100轮；最佳点全非0。有效临床/新参数约251174、250342、521454、520622。原V6参数不是每轮冻结候选clinical权重。

|模型|预警次数|真阳性|假阳性|漏报|
|---|---:|---:|---:|---:|
|原V6|491|240|251|119|
|新鲜本次对照|479|246|233|113|
|统一检验本次|397|216|181|143|
|原检验历史|401|217|184|142|
|统一检验历史|412|221|191|138|

新鲜本次对照的调参F1比原V6高.022406（F1尺度约2.24点），同各自验证选阈值下少12次预警、多6次检出、少18次假阳性。但IBS相对变差约3.16%，一年平均风险17.916%低于库内事件比例23.449%（原V622.179%）。它适合更好的本折分类读出，并没有同时更好的概率输出。不能把F1提高等同个体风险校准改善。

统一检验＋历史F1 .573281比原V6小有利，也高于原检验历史.571053；但未胜新鲜本次对照.587112。统一检验本次F1反而低于同新鲜原检验。因此本轮**没有把统一数据/历史改造的效果与改变选点目标混为一谈**。

调参排序锁定`selected=current_original`，`nominated=null`。按本轮事前规则，新增数据/纵向候选需同时胜新鲜匹配对照与旧V6才能入围，当前未入围，不读取1418确认/历史测试重挑模型或阈值，不改原默认。本次有限实验已完成，不机械续跑多种子；用户若后续优先选择F1对照，再做一个独立锁阈值验证是另一有限步骤，不能声称本轮已完成该验证。

## 3. 数据真的统一了

全部检验方向token共同九通道：首测异常、观测峰异常、末测异常、峰后恢复差、末测年龄、峰至末时间、有观测、有效峰后复查、末次处于异常峰。不同检验仍保留不同名称/单位、低高异常方向和参考界限。核心pH/pCO2保留原已核验阈值，不能把静脉与动脉混同。

新方向来自五项目与EOS绝对/%各自低高方向，共14，连同旧22内部方向36；这是统一编码token数，不是36个独立检验。正常测量仍observed=true，异常程度0；缺测observed=false。每个方向使用训练尺度，未观测不补正常。原真实数值文件保留，模型现在显式强调异常，正常范围内变化可能被压缩，这是该表示的特性而非数据删除。

训练有效覆盖（低/高共用同原始观测，不可加倍当患者数）：中性比例3631、淋巴比例3631、RDW11798、阴离子间隙11856、氯11917、EOS绝对1912、EOS%3631。新记录要求本次住院内chart<=store<=discharge、单位和参考范围有效；没有有效参考范围不猜阈值。EOS时间上下文保留，但原额外原值分支删除，与原处理完全一样是本实验差异而不是归一化改名。

源3152512常规检验只为新增七概念解析一次，无高频chartevents/设备轨迹。新接口原22九通道数值对照通过，原NPZ不覆盖。

## 4. 历史不再只是H34摘要

训练13592出院/3766患者，9826次有可用过去合格住院；58107当前—过去关系，最长102个过去住院，未截断历史。43803关系的对应B选中文书已在当前入院前存档；其余14304包括无文本/未存档/未知对应文书，关闭历史文本，不去掉该住院表格。实际绑定B section→源note→storetime，不取所有小结的最大时间猜文书。

```mermaid
flowchart LR
  P[过去每次已结束住院的完整现有输入] --> E[共享可训练住院编码器]
  C[本次住院完整现有输入] --> E
  E --> S[本次与历次住院表示 + 时间间隔 + 历史诊断去向]
  S --> T[两层患者纵向Transformer]
  T --> D[两原因幅度/形状修正]
  H[同一冻结H34基础] --> R[全年每日S/再入院CIF/竞争死亡CIF]
  D --> R
  R --> F[365天阈值 → Precision/Recall/F1]
```

“完整”指所有既往合格住院都读取**现有核验V6内容**：常规检验/7药效/微生物/旧三支持/B四字段全部片段、过去诊断去向。不是原数据库所有表、所有药名剂量或全集影像。住院层128仍压缩，但每步可学习而非固定teacher表示；BERT backbone冻结是算力选择，不等历史clinical encoder冻结。

真实GPU两步证明lab/text/fusion风险头更新、纵向temporal/history_head更新；历史输入梯度L1在两历史条件分别.411171/.056186，去掉历史mask风险参数改变.113289/.093882，H34权重不变。动态读取历史确实进入训练，不能用这两个预检差值当医学效应。

## 5. 全局F1没有解决所有人群

对每个模型使用**整体已锁阈值**，不在次数层重调：

|模型|无既往层F1|1–2次层F1|≥3次层F1|
|---|---:|---:|---:|
|原V6|.252874|.463415|.637730|
|新鲜本次对照|.345679|.457831|.656514|
|统一检验本次|.225806|.427586|.648452|
|原检验历史|.000000|.417266|.649396|
|统一检验历史|.078431|.455172|.646957|

历史候选的全局阈值会漏掉很多无既往患者，不能因为整体F1略有利忽略这个实际问题。无历史仍输出本次编码风险，而非没有预测；共享encoder训练与整体阈值会影响其检出。不是依据一个层的点估计就删历史，但本轮不能采用这条全局预警策略。

## 6. 核验/开销/失败现场

- 8项合成回归：低高方向、缺参考、正常已测、单测/峰后恢复、F1同分/无病例/暴力阈值核对/非法数值，均通过。
- source/index实际核验全部19401住院的同患者/同折/入院前已出院/无自引用，无未来历史；当前FINAL仍回顾性政策。
- 四条件共同初始幅度/shape误差0，H34冻结；训练前真实共享/历史梯度更新已核验。训练期间没有修改受保护源码。
- 正式训练容器UTC06:47:01.587→06:49:03.014约121.43秒含来源/缓存/预检与四拟合；一次独立GPU核验UTC06:50:17.133→06:50:34.115约16.98秒，均退出0。缓存分配约.522GB，实际训练有动态中间张量，不把缓存大小说总显存峰值。
- 新进程重建全部输入/历史文书可见性/尺度hash、重载五模型（含旧参照）/每日概率/指标/F1阈值/早停/选点，误差0，proof绑定report SHA `51c1c81f115ad3a78017ad2e0a4a3dde83d6eadc91dfdfaab5ff6b57a6395d59`。无重复拟合。图training_f1_ibs.png/annual_decision.png已实际查看。
- 初版实际checked第5返回值是ExpandedBaseline，不是config；完整预检错误保留Docker后按定义修复，尚未拟合。随后原参照数值路径检查失败也未拟合，现场在无r2目录及work快照。正式r2旧参照保留TF32/256路径，新四条件统一FP32/128；真实探针显示差异源为精度/批形状，旧路径恢复幅度shape曲线误差0，不提高容差掩盖。新四条件重放误差0，不称FP32是模型性能改善根因。
- 原V6/data/labels/向量/索引不改，长期任务未恢复，无外部AI/高频/全文重新编码。

## 本轮结论

三个要求已实际落地并完成实验，不是仅讨论“有可能”。一年F1确实让选点从概率误差最优转向不同模型，并出现相对原V6更好的调参分类读出；统一异常与完整可学习历史完成了真实接口/网络接入，但本轮没有胜新鲜对照。下一轮若继续，应围绕已观察到的“历史候选无既往检出下降”和“F1候选概率低估”做明确机制改进，而不是否定历史方向或重复同一四条件。没有自动继续授权，后续按用户直接指令。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_CACHED_FOLLOWUP_PROTOCOL_2026-10-15.md · lines 1–5

<!-- REVIEW_FRAGMENT {"path":"docs/V7_CACHED_FOLLOWUP_PROTOCOL_2026-10-15.md","part":1,"start_line":1,"end_line":5,"source_sha256":"b26824165196772277d96f02ef79aeb2e6fcaf480a468bfc3907939e82bc8f02","fragment_sha256":"b26824165196772277d96f02ef79aeb2e6fcaf480a468bfc3907939e82bc8f02","added_terminal_newline":false} -->
````markdown
# 缓存筛查后两个追加种子

已知seed42点估计后固定五条件：global、availability、linear、v7_frozen、routing_prior。它们覆盖全局再校准、最高F1的来源信息、同原值普通读取、冻结V7表示和软路由。追加seed7/101，不根据新结果改变名单。所有条件同一个冻结强V6 seed42来源，因此是新增读取器的三随机初始化复核，不是三个独立基座或外部验证。

完全沿用cached批特征/来源、512槽/16隐藏/Dropout.3、同输出变换和联合NLL+1e-4正则、AdamW .001/wd.01/clip1、batch256、max24/min8/patience6/min_delta1e-4及年度F1选点。新进程逐项重放best/last/选择/指标；保留原12条件结果不覆盖，不扩参数网格，不默认替换。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_CACHED_RESEARCH_PROTOCOL_2026-10-15.md · lines 1–11

<!-- REVIEW_FRAGMENT {"path":"docs/V7_CACHED_RESEARCH_PROTOCOL_2026-10-15.md","part":1,"start_line":1,"end_line":11,"source_sha256":"d5cb54885d78f5ab49cdd810e7e86d22bec27e05ab5fd18bca8fb3e22852f0e9","fragment_sha256":"d5cb54885d78f5ab49cdd810e7e86d22bec27e05ab5fd18bca8fb3e22852f0e9","added_terminal_newline":false} -->
````markdown
# 强基座与新检验/临床表示：缓存有限筛查

承接顺序总合同。完整已选V6对称背景风险冻结；当前train/tune来源只缓存一次，重建后与旧保存调参概率逐项核对。V7首轮已选含对比表示冻结为一个低成本候选；这不是重新训练整个V7的替代证据。真实端到端强基座实验另运行。

配方：全局4参数再校准、冻结V7表示读取、同源观测/时间对照、九项目物理首末值线性读取、有限临床/分位点PLE、分别加物理恢复速度/采样密度/三点曲率；EOS和ICS主效应对照与加入二者记录组合；全明确临床集合的普通原因分支和软先验分支。软先验仅偏置呼吸/EOS/方案/肺功能与器官/影像组权重，两原因仍可读取全部合法来源，不detach竞争梯度。PFT只使用已有背景证据/角色presence，不平均多值冒充当前肺功能；CXR保留四状态，缺失不当阴性。

所有局部读取固定512输入槽、16隐单元、末层零初始化、Dropout.3；普通/先验原因分支同容量与公共初始化，先验只改变可学习来源权重初值。新的幅度修正相加，shape乘exp(.25*tanh(delta))，零修正严格恢复原风险。全局对照使用同样输出变换。风险监督为连续联合NLL+1e-4修正平方；AdamW lr.001/wd.01/clip1，batch256，最多24轮、至少8、patience6/min_delta1e-4。seed42，年度F1/并列低IBS选择，所有结果同时报告IBS与末轮。

特征尺度、PLE有限尾节点仅训练拟合；缺观测、同刻、无峰后测量等有明确mask。EOS/ICS组合表示已记录信息，不声称未治疗。共享健康状态读取基础34及住院时长；临床路由候选包含实际可用的检验、出院吸入方案、肺功能presence、四状态CXR及旧支持。

低成本筛查不承诺任何默认替换。单独重放best/last、指标、选择及来源hash；弱配方不继续参数网格。两折融合时不得复用这些全训练有监督表示作为OOF输入，须改用未用本队列标签训练的原始缓存，并在内折拟合全部预处理及模型。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_DRUG_CONTEXT_PROTOCOL_2026-10-15.md · lines 1–9

<!-- REVIEW_FRAGMENT {"path":"docs/V7_DRUG_CONTEXT_PROTOCOL_2026-10-15.md","part":1,"start_line":1,"end_line":9,"source_sha256":"1291d7bf7f1fc38300874e8efd7315f0698f887bab297326163e5b130944d145","fragment_sha256":"1291d7bf7f1fc38300874e8efd7315f0698f887bab297326163e5b130944d145","added_terminal_newline":false} -->
````markdown
# 剩余药物线索：低成本语义、记录动态与途径筛查

先离线冻结Bio_ClinicalBERT，只编码38个规范药名，不编码患者文本/不微调。在完整冻结V6风险上比较同容量读取器的药物来源/数量、38身份、固定随机向量和冻结药名语义向量。随机与语义向量均L2归一化并用相同固定768→256随机投影/用药集合平均；这是药名表征的有限筛查，不冒称完整药理手册语义迁移。

另从已核验普通eMAR/pharmacy/detail复用原12分子、途径、执行状态、完整剂量与单位规则，一次解析记录动态。分别比较来源/可估计mask、记录数/记录跨度/首末24小时计数主效应、增加同分子计数/剂量变化速率、增加已核验途径转换。LOS≥2日且首末24小时各有执行记录才估计变化；剂量需要每条都可核验，复方不分配剂量。记录跨度不是完整疗程，没记录不等于未治疗，不跨药物相加mg或声称等效升级。

所有候选固定512槽→16隐单元→零末层修正，冻结强V6、NLL+1e-4修正平方、lr.001/wd.01/Dropout.3/batch256/max24/min8/patience6，年度F1/并列低IBS选择，与cached批相同。只用原训练/指定调参，不覆盖旧数据/模型。词表与语义模型不由结局决定；数值尺度只训练拟合。合成边界、一次原源重建及独立best/last重放后再总结。

全量临床事实产物另外做存在性与完成状态核对。仅有catalog/提示词或部分抽取，不得当作已有全队列DNR/家庭NIV/治疗缺失字典；本轮不自动调用外部付费模型补齐不明抽取。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_ENSEMBLE_FULL_REFIT_PROTOCOL_2026-10-15.md · lines 1–5

<!-- REVIEW_FRAGMENT {"path":"docs/V7_ENSEMBLE_FULL_REFIT_PROTOCOL_2026-10-15.md","part":1,"start_line":1,"end_line":5,"source_sha256":"949faf8c568550b884b49383ba0fae691edeff95bd5b8f529fa4d34ec6fc4f67","fragment_sha256":"949faf8c568550b884b49383ba0fae691edeff95bd5b8f529fa4d34ec6fc4f67","added_terminal_newline":false} -->
````markdown
# 两折发现互补后的全训练重拟合

两折阶段的等权/加权融合平均AUC约.805，高于三个单成员，但未超强V6，且成员仅用半份训练患者。因此增加一次固定全训练重拟合（每家族一模型），不增加折数、参数网格或新候选。Cox、树、24轮紧凑神经网络及全部预处理沿用既定规则，改为使用完整13592训练；调参1531不参与拟合。

直接复用先前两折OOF得到的三家族权重和温度，不在调参上重选。分别报告三单模型、等权、原OOF加权、原OOF温度结果；这种训练规模变化可能带来OOF/部署分数分布差异，如实记录，不称新的盲测。保存所有模型、全训练预处理与概率，原生XGBoost接口独立重放，原两折源码/合同/失败重载日志不改动。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_EXTERNAL_MATH_ENGINEERING_REVIEW_2026-10-15.md · lines 1–158

<!-- REVIEW_FRAGMENT {"path":"docs/V7_EXTERNAL_MATH_ENGINEERING_REVIEW_2026-10-15.md","part":1,"start_line":1,"end_line":158,"source_sha256":"6b10e1e5705bff804bdaa5b54a12029fb685b9bc518510dc8453f06356d604eb","fragment_sha256":"6b10e1e5705bff804bdaa5b54a12029fb685b9bc518510dc8453f06356d604eb","added_terminal_newline":false} -->
````markdown
# Gemini五组数学与工程建议：逐项核对及V7适配

更新：本文保留方案审查/首轮时点的状态。后续56次拟合、独立验证及每项决定见[顺序实验结果](V7_SEQUENTIAL_RESEARCH_RESULTS_2026-10-15.md)，不再以本文的“尚未做”作为当前进度。

2026-10-15。审查对象是用户粘贴的五组方案，不把其中的实现指令自动视作改动现有模型的授权。本轮核对代码、已有结果、原始文献并运行合成数学反例；不训练临床模型，不改动已锁定V6/V7源码、特征契约、数据或选点规则。

## 结论与已有证据

值得保留的方向包括：用完整强模型作为冻结参照、检验数值分段编码、计量采样过程、带来源的生物标志物与治疗记录交互、按结局软路由，以及同队列OOF概率融合。它们都需要明确对照，不能把同一个大改造的结果归因到某一模块。

需要修正的数学结论包括：零初始化保证初始输出相同，不能保证后续泛化不退步；批次内正交不等于“仅有未解释信息”；双原因的合法竞争耦合不等于梯度污染；标签平滑后的损失不自动保持对原真实概率的严格恰当性；两个恰当损失也不保证参数梯度同向。所述AUROC增加0.012–0.025、种子方差减半没有在所列基准论文中得到对应支持，不作为本项目预期收益。

当前V7的实际基础是冻结H34、零初始化4参数风险修正头，共享256维临床/历史表示；输出双原因Weibull幅度和shape，采用连续时间联合似然。它没有把已训练完整V6作为初始函数。首轮V7原损失对照年度F1 .565321，增加Focal/平滑/对比为 .567023，已有V6对称背景为 .577373；这是同指定调参折单seed描述，尚不能定位复杂模块干扰为根因。

已有较直接的证据：V6八条件×三临床种子的原因专属头、独立编码器和梯度处理比较，未稳定改善再入院；死亡方向有线索。五核心检验首末原值的冻结追加头和Cox诊断也未取得再入院增量。这些结果不否定新表示，但应防止把旧实验重新命名为新方案。见双原因实验（本地历史报告`V6_TASK_OPTIMIZATION_RESULTS_2026-10-13.md`，未随本增量公开）、原值追加（本地历史报告`V6_CORE_LAB_CONTINUOUS_RESULTS_2026-10-14.md`，未随本增量公开）、原值Cox（本地历史报告`V6_CORE_LAB_COX_RESULTS_2026-10-14.md`，未随本增量公开）。

## 一、冻结基座、OOF、正交和零门控

### 冻结强基座：合理，但要明确究竟冻结什么

- 当前H34不是完整V6；如果要保证初始函数等于某个V6，必须冻结那个完整V6的编码、风险参数和预处理，逐条核验初始双原因曲线相同。换成较弱Cox或仅核心输入后，初始相同只相对于该Cox/核心模型成立。
- Cox的线性预测量、原因别log hazard、累计风险、CIF和log odds不是同一个量。Cox需要训练折估计的基线累计风险才能得到绝对风险；不能把不同输出直接相加。沿用现有Weibull族时可分别修正幅度与raw shape；若换离散头，应另立完整时间分箱与似然合同。
- 文件`clinical_feature_contract.csv`是语义筛选契约，不是改一行就自动生成数值特征。需要派生接口、数据桥、模型消费和测试同时闭合。

特别检查了`cohort_pipeline/sql/03_longitudinal_history.sql`：它含有`aecopd_total`、`is_last_observed_aecopd`、`next_observed_*`和`days_to_next_observed_aecopd`等分析/未来字段。不能整表送入基座。当前`training/v6/data.py::history_for`按本次入院前已经结束的住院重算过去历史；候选也必须使用这种时点白名单。既往一年次数需另按时间窗重建，不能拿终生累计数替代。

### OOF：适合减少二阶段学习的过拟合，但不是现有资产

五折按患者分组，某患者全部出院及作为索引样本的历史记录不能跨折。内折的标准化、分箱、词表、选参及任何有监督基础权重都只能在内折训练患者拟合。已在全训练患者上拟合的H34/V6不能直接嵌入内折后宣称整条链是严格OOF。冻结通用BERT本身与使用了本队列标签的基础模型需区别对待。

训练二阶段模型/融合权重使用OOF预测；外层调参/确认/测试仅由未用该层患者训练的模型预测。部署使用全训练重拟合基座或五折基座平均，需事先固定，并核对它与训练OOF分数的分布差异。当前保存的train/tune概率不自动变成五折OOF。OOF也不会把已经曝光的调参集恢复成盲测。

### 批次内投影：暂不作为第一项改造

对`X∈R^(B×p)`、`H∈R^(B×d)`，投影实际为：

`H_res=(I_B-X X^+)H`。

这里消除的是该批次样本轴上的线性分量。它不能证明消除了基座对结局已解释的全部信息，也可能去掉与基座相关、但对结局仍有用的交互。共线性使`(XᵀX)⁻¹`可能不存在，需伪逆/QR/SVD；岭投影虽稳定，却不再严格正交。

合成检查中，B=64、p=70且rank=64时，残余范数比例约4.26e-15，几乎全部信号被消除。同一患者只换同批其他样本，残余表示距离变为0.70358；单人推理且基座含非零列时也可全部归零。因此训练/推理不能每个batch临时重估投影。

若后续确需研究去除线性重复，可在训练患者上估计固定映射`B=(XᵀX+λI)⁻¹XᵀH`，推理用`h_res=h-xB`，并处理编码器训练期间H变化导致映射过时的问题；或把相关性作为有限惩罚。它们仍需独立对照，不能称为已证明的正交因果学习。第一轮可不加投影，先单独检验强基座是否改变V7的优化结果。

### 零初始化：保留，但不能“双零”

对`r=tanh(α)gθ(h)`，α=0时，`∂r/∂θ=0`、`∂r/∂α=gθ(h)`。若又沿用当前V7的零输出末层，使g=0，则门与末层梯度同时为0；在没有另外打开路径的机制下，风险修正无法启动。

可选其一：保留当前零末层而不再乘零门；或零门配非零初始残差输出，再观察门先更新、内部网络随后更新。门的收缩惩罚/有界修正可以限制偏离，但不保证临床泛化提升。初始函数相同只保证epoch0；保留epoch0候选可以保证按既定规则选出的调参指标不低于该候选，不保证未知患者表现不退步。

合成反例从初始概率0.5出发，一次训练更新后概率0.61352，在相反结局的独立例子中Brier由0.25升至0.37640。这个例子只反驳普遍数学保底，不描述真实患者。

## 二、检验的四类显式表示

[Gorishniy等原论文](https://arxiv.org/html/2203.05556v4)支持数值分段线性编码，论文主要检验训练分位点和训练标签引导的树分割。临床固定节点、恢复斜率和采样密度是针对本项目的扩展，不能统称论文已经验证。四类通道也不是统计上天然正交。

| 子建议 | 与现有V7的关系 | 适配方式 |
|---|---|---|
| 临床边界PLE | 现有异常方向转换可能折叠正常范围内差异，值得比较新编码 | 保留可信物理原值/单位/标本、缺失与项目身份；比较现有九通道、同原值普通线性投影、同原值PLE，分开信息增加与编码变化 |
| EOS 100/300、PaCO2 45/52/65、白蛋白3.0/3.5 | 部分有治疗或生理解释，但不能把所有点视作统一预后断点 | 节点需逐项记录原始依据和任务用途；PaCO2需动脉标本，EOS需绝对计数单位与激素前后角色；未知单位不猜换算 |
| `+∞`尾区间 | 不能直接作为有限区间的分母 | `clip((x-b_j)/(b_(j+1)-b_j),0,1)`的节点应有限；另定训练折有限尾界、尾部尺度或超界通道，防止尾区间全部折叠 |
| 最近值−正常边界 | 当前latest severity已经部分表达 | 按偏高/偏低分别编码，保留参考范围、最近采样年龄与观测mask；称“最近观测未恢复异常”，不直接称出院稳态 |
| 峰/入院到最近的斜率 | 当前有恢复差与峰到最近间隔，但输入时间经过log和缩放，不能直接相除充当物理速率 | 从未缩放时间与物理值派生；入院→最近和最异常→最近分开；低pH最异常是最低值；至少两个不同时点，分母过小和极值需固定处理 |
| 曲率/反弹 | 两点斜率不能给曲率，也无法区分相同首末的中间反弹 | 需要至少三个不同时间点及可用的中间轨迹；现有九通道不足时明确缺数据，不凭空计算 |
| 采样密度`log1p(N/LOS)` | 当前有测/复查/时间标记，但没有显式完整计数通道 | N按同项目、标本、时间去重，不把高低两轴或重复行当多次采血；LOS用真实正时长，预定短住院稳定规则；同时保留计数、LOS、来源覆盖，比较仅来源与真实密度 |

当前九通道的最近血气/白蛋白可能距离出院数日；已有五轴审计中末测距出院中位数约4.84/4.88/0.46/3.36/0.45日。不能把所有“最近值”解释成出院当日。密度反映检查与记录过程，也可能反映临床担忧，但不是直接测得的病情或医生意图。

“快速恢复一定预后好、持续异常一定容易再入院”只能作为待检验假设，不能写入标签或预先强制其预测方向。轨迹与治疗、慢性基础状态、测量机会相互关联，数值斜率不自动等于治疗反应或因果疗效。

`unified_lab_state.py`当前包含22原异常方向、10额外方向和4 EOS方向，统一为36×9；EOS16为时间语境，不是完整EOS原值。原值保留、PLE、斜率与计数需明确新派生接口，保持原成员/标签/输入不覆盖。

## 三、按结局软路由与竞争梯度

[DeepCompete](https://pmc.ncbi.nlm.nih.gov/articles/PMC8075516/)研究连续时间竞争风险；本轮能核对其原始摘要，PMC全文访问遇到验证码，未把未读到的细节当已证实。[TraCeR](https://arxiv.org/html/2512.18129v1)确有共享表示、原因专属子网络、多项归一化和竞争风险似然，支持原因专属结构的可行性；文中相应公式没有给出这里建议的死亡风险detach规则。

1. **共享健康状态分支**：合理候选。人口学、合并症、LOS是可用的共享候选，但不能据临床名称禁止其他信息影响共享状态。
2. **再入院专家**：病史、肺功能、EOS/吸入药、支持转换有研究价值。当前V7并未消费肺功能集合、已核验出院方案集合或完整呼吸转换接口。肺功能来源稀疏且测量时点多为未知，不能高权重强制每人使用。
3. **死亡专家**：年龄、去向、器官功能及影像可以偏重，但它们也可能影响再入院。DNR和衰弱并没有因为存在disposition脚本就自动变成已核验数值输入；需单独定义来源和时点。去向DIED/死亡标志等不能当作未来结局捷径。
4. **软路由而非硬隔离**：允许两原因读取完整合法来源，用可学习的权重偏置表达临床先验，缺源时有明确回退。对照应包含共享、普通分支、同容量软先验分支，才能区分参数增多与先验增益。先保留当前风险分布族，避免同时换分支、时间离散化、损失和输入。
5. **独立二元hazard不够**：每个区间必须`h1≥0,h2≥0,h1+h2≤1`。两个独立sigmoid可给0.8+0.6，导致无事件概率−0.4。可用`softmax([0,z1,z2])`，或从联合正强度推导区间概率。分开的编码器不要求分开的非法归一化。
6. **CIF公式有条件成立**：在合法互斥条件概率下，`F1(t)=Στ h1(τ)∏s<τ(1-h1(s)-h2(s))`正确。这种竞争影响是任务本身，不应整体消除。
7. **detach不是普遍的去污染步骤**：若两分支没有共享参数且h2不依赖θ1，原本就有`∂h2/∂θ1=0`；若共享底层或多项归一化，detach会删除实际目标的一部分导数。合成例子中前向CIF相同，完整导数0.266721与有限差分一致，detach后变0.224202，已不是该前向评分的真实梯度。

连续时间联合NLL可写为`Σk[Hk(T)-I(event=k)log λk(T)]`。死亡病例在死亡之前未发生再入院，对再入院的`H1(T)`有有效监督；竞争事件从风险集中退出不能简单当作有害样本。代码`task_gradient_diagnostic.py::cause_losses`已实现这个分解。共享梯度若冲突可以测量并独立比较处理方法，但不能由竞争定义直接推断“污染”。

V6已有普通原因专属/独立编码和梯度投影比较，不需从零重讲。真正新候选是具体临床先验、有效输入和路由方式；已有死亡方向线索也不能替代本轮再入院获益证据。

## 四、四个生物标志物与治疗记录交互

[ERS综述](https://publications.ersnet.org/content/erjor/12/3/00884-2025)讨论治疗、医疗利用及可干预目标，支持研究方向；它不是这四个乘积公式的直接验证，也未证明它们是突破瓶颈的必要条件。本轮主页面直读失败，但检索可读取该期刊原文的方法、结果与讨论段，故不声称全文细节穷尽。

这些应先命名为“生物标志物与已记录治疗的组合证据”，避免在没有适应证、禁忌、患者偏好和完整观察条件时直接给出治疗不足判断。

| 组合 | 可保留的研究内容 | 必须修正/补齐 |
|---|---|---|
| EOS高×出院ICS未记录 | 第一项较可落地；可与既往加重共同建模 | EOS绝对计数与激素时序、Max/首测/最近含义分开；ICS缺章/未提/未知/held/stopped/listed分开，不能统统编码为未治疗。当前`eos_med_context`主要是全身激素时序，不能直接当出院ICS匹配 |
| 高PaCO2/曾用NIV×无家庭NIV | 可作为住院高碳酸状态与出院支持计划的探索 | 急性期PaCO2高或曾用NIV不自动等于应带家庭NIV出院。[ATS原始指南](https://pmc.ncbi.nlm.nih.gov/articles/PMC7427384/)强调稳定持续高碳酸及急性期后复评，提出2–4周再评估；家庭NIV还需有明确出院证据。院后复评不能回填本次预测特征 |
| 既往一年≥2次×少吸入治疗 | 次数与药效覆盖交互值得比较 | “终生累计”不等于“过去一年”；库外加重不可见。按ICS/LABA/LAMA药效覆盖计，区分复方和单品、救援药和维护药；一个三联吸入器不等于只覆盖一种药效 |
| NT-proBNP/影像异常×利尿/心衰治疗未记录 | 心肺联合背景可能有条件信息 | NT-proBNP与BNP不是同一检测及统一阈值；肾功能、年龄和影像不确定状态需保留。现有出院药物读取主要ICS/LABA/LAMA，没有完整利尿剂/心衰规范治疗接口；胸水也不等于已确诊容量超负荷 |

每个交互至少配套：两个原始组成特征、来源是否可判定、时间/单位角色，以及交互值。不可判定时用独立mask，不将0混作已确认阴性。交互相对主效应的增量需要单独对照；先在训练折核对可判定人数和重叠，不用结局挑阈值。

当前`clinical_combinations.py`主要构建中性/淋巴比例、NLR和SII，不等于已经包含上述治疗交互。`oxygen_device.py`是住院设备记录，不自动证明家庭设备；`discharge_regimen_interface`保留了药物角色，但未声称完整临床用药金标准。把现有文件名称等同于所需字段已经全部可用，会高估落地程度。

即使组合预测有效，也不能从观察性记录推出改药能降低风险；适应证和记录过程会造成混杂。此处结论是预测研究设计，未形成个体治疗建议。

## 五、损失、校准与异构概率融合

### 先写清楚实际训练目标

当前V7没有高权重的结局pairwise ranking loss。它有七期限平滑Focal、系数.1的同住院特征距离对比，以及部分条件保留的原竞争NLL。特征距离对比不等于按结局排序；margin .5也不是相对权重.5。最新损失比较已固定其他条件，不应把架构未胜出全部归到一个不存在的主排序项。

[Alberge等论文](https://arxiv.org/html/2406.14085v1)证明的是在删失等假设下的特定加权评分，并讨论Brier与删失分布。它没有证明“任意平滑、温度、NLL+Brier组合都严格恰当且每步梯度同向”。

- **NLL**：当前连续联合NLL已有基线，可以继续作为清楚的主对照。离散NLL也是合理候选，但换365离散风险头会改变参数量、时间精度与优化，不能只当换一个损失。
- **标签平滑**：K类目标变为`(1-ε)p+ε/K`；在无限样本简化例子中，真实概率.1、二类ε=.05的期望平滑交叉熵最优输出是.12。因此它对原始真实概率一般不再严格恰当。可作为单独正则候选，不能用“严格恰当”保底其校准。
- **温度缩放**：训练后在OOF/专门校准集拟合温度，是独立后处理；对多项hazard需联合归一化，不能逐原因随意改CIF而破坏守恒。若温度属于网络训练参数，应讨论最终变换后概率，不能将它与平滑混为同一个操作。
- **IPCW-Brier**：需要真实可解释的删失时间及删失模型、正概率条件。当前WorldA365只有365日行政终止，在观测口径内365日前无提前删失，权重为1（末日按边界约定）；这不会自动补回院外失访。死亡是竞争事件，不是用于抬权的普通失访。改删失世界观应另建任务/标签评价合同。
- **有限时点**：30/90/180/365的评分只约束这些点，不能单独严格识别全部逐日分布。全时间似然和固定点Brier的结合需说明范围及相对系数，不把它们的数值大小直接相加当已优化权重。
- **梯度同向**：严格恰当描述总体最优概率，不描述共享参数每步更新。同一批`p=[.99,.1]、y=[0,1]`，共享logit偏移的原log loss梯度+.045，Brier梯度−.071199。这个合成例子反驳“严格同向”，不否定二者结合的试验价值。

### OOF融合：保留，但先统一任务与概率

最直接的候选是共同CIF空间的非负凸组合：

`F_ens,k(t)=Σ_m w_m F_m,k(t), w_m≥0, Σ_m w_m=1`。

全局权重在两个原因、所有时点共用；每个成员的S/F1/F2合法时，组合仍单调且守恒。若随原因分别选权重，可得到F1+F2=1.6；若随时间任意换权重，累计风险可从.9降至.11。Rank-Probit首先是排序分数，不能直接解释成概率或与CIF无条件互换。

| 家族 | 现有资产 | 融合前所需核对 |
|---|---|---|
| 线性Cox | `continuous_competing_cox.py`已用于当前逐日Cox诊断 | 最终输入、L2设置、训练患者和双原因定义；L1/ElasticNet支持需另核对，不能由名称假定已实现 |
| XGBoost/CatBoost | 有旧HAIM式XGBoost代码；不等于当前已训练OOF集成成员 | 旧入口仍是Qwen/条件区间和尾部接口，需桥接当前BERT/共同成员/严格标签/每日双原因输出；CatBoost也不能仅列名称便当已有结果 |
| V6/V7 | 当前连续每日概率与独立重载可复用作为外层参照 | 既有tune预测不是OOF；内折监督基础与预处理需重新满足患者隔离 |

权重可先限定少量自由度，用训练OOF概率按预定NLL/Brier选择；与最佳单模型、固定等权和仅再校准分别比较。不能用测试标签选权重。Brier的凸性只保证组合损失不超过成员损失的同权平均，不保证优于最佳成员，也不保证AUC增加。种子方差取决于误差相关性`wᵀΣw`，没有普遍减半保证。

所引[临床表格基准原始摘要](https://pubmed.ncbi.nlm.nih.gov/42399880/)比较TabPFN与传统方法的12个二元临床任务，并已有2026年7月期刊版；可支持强传统基线有必要。所读方法与摘要没有检验这里的Cox＋树＋纵向神经OOF组合，也没有提供0.012–0.025/方差减半的对应证据。这两个数值需要另给直接来源，当前不进入实验验收目标。

## 建议推进顺序与验收

此前已明确但尚未完成的检查仍保留：Focal有/无标签平滑、跨源Query有/无、显式时间衰减有/无。新建议不能被用来把这些检查标成已完成，也不能在同时改完五组设计后声称分清了各自作用。以下是新增方向之间的顺序；正式训练前应与这些既有待办合并为有限、事先固定的实验名单。

1. **先整理可用性与最小数学实现**：已完成本轮合成检查；下一项数据工作限定在训练折，核对EOS/ICS组合、检验原值/时间/去重计数等字段是否足够，不读确认/测试作开发。
2. **单独检验强基座假设**：完整已选V6冻结，沿用同一风险分布与原NLL，比较不加修正、仅全局再校准、V7表示残差。先不加批次投影，不叠加零门与零末层，不同时换临床输入。这个实验检验优化/起点，不能把新容量的任何收益自动归给新增临床信息。历史冻结小头已多次失败，必须明确本次机制差别。
3. **检验表示逐项比较**：同物理原值的线性投影对PLE；随后分别加速率和采样密度。原九通道/观测mask保留，缺失/同刻/短住院/异常方向/尾区间都有边界核验。
4. **先一项可判定交互**：优先EOS与已记录ICS，加组成主效应和来源对照；家庭NIV、完整心衰治疗待有可核验来源再推进。临床先验路由只与同容量普通分支匹配比较，保留合法竞争梯度。
5. **再做患者级OOF异构融合**：统一成员、标签和S/F1/F2网格，训练内五折拟合成员和少量融合权重，另层评价；先不使用Rank-Probit当风险概率，也不预设提升幅度。

现行V7选点是年度调参F1、并列较低再入院IBS、再并列较早epoch；不能因为新建议重写旧报告的选择。若后续改为概率评分优先，应在新合同中先固定，同时继续报告年度F1/PPV/召回、多期限AUC和两原因IBS/校准。配对比较以患者为单位，不把多次住院当独立患者。保留原默认、epoch0、来源hash和失败现场；没有新增临床训练结果时不写“已提升”。

## 本轮验证文件

- 合成程序：`work/v7_external_design_review_2026-10-15/mathematical_checks.py`。
- 实际输出：`work/v7_external_design_review_2026-10-15/mathematical_checks.json`。
- 在既有baseline容器中CPU执行、网络关闭，无GPU/患者文件挂载；全部数学断言通过。这只是公式与工程边界核对，不是临床实验。
- 上述论文访问限制分别记录，文献支持与本项目扩展区分；数值反例为本轮自行构造，不冒充原文结果。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_FIRST_EXPERIMENT_PRD_2026-10-15.md · lines 1–49

<!-- REVIEW_FRAGMENT {"path":"docs/V7_FIRST_EXPERIMENT_PRD_2026-10-15.md","part":1,"start_line":1,"end_line":49,"source_sha256":"06487676c15bc9c4fc84a6b40d501994e9af69c29c1c2bbb43167497fcecfee5","fragment_sha256":"06487676c15bc9c4fc84a6b40d501994e9af69c29c1c2bbb43167497fcecfee5","added_terminal_newline":false} -->
````markdown
# V7 首轮正式实验合同

2026-10-15，依据用户直接授权启动有限 V7 实验。此前取消的长期自动研究不恢复。V6 数据、模型及历史实验保留。

## 目标与固定结构

- 全部有效临床表示统一256维；注意力8头，每头32维；FFN1024；住院层及历史层均采用M=4多查询读取；患者历史Transformer两层。
- 编码器/attention Dropout=.1，融合和风险头Dropout=.3。显式过去权重为softmax(content_score−softplus(lambda_raw)×log1p(gap_days))，即内容相关的幂律时间衰减，lambda>0。
- 冻结Bio_ClinicalBERT，复用768维B四字段块向量；本轮不微调、不重编码全文。实际本地BertModel108310272参数，其中去pooler的encoder107719680；这些参数不在V7训练优化器内。
- 药物单品使用已核验38身份/10药效父类、单次剂量、频率、有效mask、七剂量单位的因子化表示；原七药效×四来源证据同时保留。现接口无完整路线/疗程/院外处方轨迹，不伪造这些字段或称已实现真实升级斜率。
- 药物token作Query读取保留的字段内M4文本及影像token；CXR/CT影像token作Query读取文本。影像复用冻结RadBERT已保存的两个1536维类别聚合，不称逐报告/逐词交互。
- 文本块→四字段M4→文本住院表示；各源→本次住院融合；全部既往合格住院→历史Transformer及显式衰减M4→风险修正。末层零初始化，共同H34基线冻结。

## 数据与信息边界

继承V6统一历史合同：5351患者/19401出院；train13592，指定调参1531；其余确认/历史测试不用于本轮选择。严格491.21/J44.1未来首次非转院再入院、其前竞争死亡及WorldA365日记录边界不变。本次FINAL文本仍回顾性；过去文本须在当前入院前存档，未知关闭历史文本，既往表格仍保留。全部58107训练过去关系保留，不截断历史。只常规检验、药物、文本、影像文本及旧三支持汇总，不接入高频96/115数组。

复用并核验：统一36检验方向九通道及EOS16语境；当前/过去H34上下文；38药物细节接口；原七药效、五微生物、三支持；B文本；已验证RadBERT影像聚合。成员、患者fold、标签、原资产hash不变，源统计只训练折。本轮是架构+已有输入+联合损失的组合实验，不把整体增量归因单一因素。

## 两个同初始化条件

1. `nll_control`：V7结构、全部相同输入，竞争风险NLL＋修正平方1e-4。
2. `recommended`：同结构/输入/初值，以上损失＋1.0×多期限双原因Focal-smoothed＋0.1×跨源对比。

两个条件均seed42，从新鲜V7参数开始，不复用已训练V6临床隐藏状态；V6旧完整及最新对称历史检查点仅作保存结果强参照。此对照分离联合损失效果，不分离V7全部架构与新增药物/影像输入各自效果。

## 推荐损失的精确定义

- 保留连续时间双原因Weibull竞争NLL，确保事件时间和死亡可训练，不用普通分类替代全年输出。
- 可微CIF沿用V6 .25天网格和近零细化，在7/14/30/60/90/180/365日读取两原因累计概率。WorldA本库未记录者为相应期限非事件；不称完整院外阴性。
- 对每原因/期限二元标签y，平滑为ys=(1−.05)y+.05/2，目标.025/.975。Focal=−.75×ys×(1−p)^2 log p−.25×(1−ys)×p^2 log(1−p)，按记录/原因/期限等权均值。这是soft-target二元Focal的明确扩展，不冒称唯一标准定义。
- 对比正对为同次当前住院的文本↔药物、文本↔检验、文本↔影像（两源均存在才启用）；使用独立256投影和L2单位范数。正对平方Euclidean距离，负对max(0,.5−distance)^2。负对仅本batch不同患者，所有同患者记录排除；无合法对返回可微0，不把缺源当负例，不读风险标签造配对。
- 辅助损失系数1.0/.1为首轮固定实施选择，原报告没有给出；不在调参上搜索或事后修改。分别报告NLL、Focal、对比及正负pair数。

## 优化、选点和时间

- AdamW weight_decay=.01，梯度clip1。已有冻结向量的文本/影像投影组lr1e-4，其他新编码/交互/融合/历史/风险组lr3e-4；BERT/RadBERT骨干及H34为0更新。参数组严格无重复、无漏项。
- batch64，最多30轮，至少8轮、patience6/min_delta1e-4早停；每轮保存记录，按一年调参F1最大选点，平局再入院IBS最低，再平局早轮。阈值只在指定调参找，保留概率评价，不用确认/测试挑。
- 第0轮共同H34风险、真实GPU三步梯度/更新和缺失源/历史敏感性检查通过后才正式拟合。预检模型丢弃，正式两条件重设同seed/RNG，不继承预检优化器。
- 记录来源核验/缓存/预检/每轮纯训练/评价/总fit/独立重放时间，GPU allocated/reserved峰值；全时长不以缓存训练时长冒充BERT微调用时。
- 训练后一次新进程重建输入与合同，重载best/last、每日曲线、阈值、F1/两原因误差、早停/选点/权重hash核验；原V6不覆盖。

## 必要测试与采用限制

覆盖空文本/影像/药物、padding内容不影响、同字段块排列、四字段M4压缩层逐参数真实梯度/更新、同患者负对排除、标签平滑/focal边界、无pair可微0、显式时间衰减、完整历史/无历史、日概率守恒和单调、两条件相同初值、差分组覆盖以及冻结基线无梯度。

实现预检修正：初次运行的字段M4压缩表示被丢弃，现场保留于`training_runs/v7_first_2026-10-15`；修正后正式两条件重新初始化于`training_runs/v7_first_2026-10-15_r2`，不接续旧权重或优化器，不把初次无效运行作为性能结论。

首轮是单seed探索；报告多期限F1/阈值/PPV/召回、AUC/AP、再入院与死亡Brier/IBS、校准、原次数层及患者配对区间。即使调参F1更高也不自动采用或声称概率/外部泛化改善。所有失败现场保留，不覆盖已有输出，不启动大网格或长期automation。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_FIRST_EXPERIMENT_RESULTS_2026-10-15.md · lines 1–97

<!-- REVIEW_FRAGMENT {"path":"docs/V7_FIRST_EXPERIMENT_RESULTS_2026-10-15.md","part":1,"start_line":1,"end_line":97,"source_sha256":"a62ac6f1d37d8fc2ea6be94fd1fa32e0b65931c0621203395c7f8a85c42e6690","fragment_sha256":"a62ac6f1d37d8fc2ea6be94fd1fa32e0b65931c0621203395c7f8a85c42e6690","added_terminal_newline":false} -->
````markdown
# V7首轮正式实验结果

2026-10-15，用户直接授权的首轮已完成。正式输出为`training_runs/v7_first_2026-10-15_r2`。V7两条件均完成实际GPU训练及早停；该单seed调参结果尚未支持替换原V6默认。

## 实际实现与冻结范围

全部有效临床表示256；8头/每头32；M4字段与住院读取、历史M4；FFN1024；历史Transformer2层；编码/attention Dropout=.1、融合/风险头=.3。冻结Bio_ClinicalBERT/RadBERT，文本768→256、影像类别聚合1536→256；BERT骨干仍是12层12头，不把V7的8头改进它内部。显式历史偏置`−softplus(lambda)×log1p(gap_days)`，是内容分数上的正值幂律时间衰减。

实际BERT参数108,310,272，去pooler为107,719,680；本轮BERT优化器参数0。V7有效可训练11,180,112参数，H34另冻结72参数。文本/影像投影591,360参数lr1e-4；其余新模块10,588,752参数lr3e-4；AdamW衰减.01、梯度clip1。

三层读取为文本字段内M4、字段→住院M4、住院→历史M4。38药物身份、10药效父类、剂量/频率有效mask和单位因子化；保留旧七药效四来源、常规36检验九通道/EOS语境、五微生物/三支持与H34背景。药物Query读取字段查询和影像token；影像Query读取文本。冻结向量内不重建逐词交互，CXR/CT为类别聚合而非逐报告表示。

尚未接入药物升级斜率、完整route/duration或新的隐式生物标志物规则字典；本轮不能回答这些独立机制的效果。

## 数据与损失

建模队列19401出院/5351患者，训练13592出院/3766患者，指定调参1531出院/402患者。全部58107条训练历史关系保留，最多102次过去住院，不截断；43803条历史文本可见，14304条因存档时间边界关闭文本，表格继续可用。细药训练覆盖11323出院，影像覆盖4005出院。严格AECOPD与竞争死亡/WorldA365标签保留，确认和历史测试不用于本轮选择。

对照：竞争NLL+1e-4修正平方。推荐：以上+1.0多期限Focal-smoothed+.1对比。Focal gamma2/alpha.75，标签平滑epsilon.05，即负/正目标.025/.975；7/14/30/60/90/180/365日×两原因等权。对比margin.5，文本↔药/检验/影像，同住院正对、不同患者负对，所有同患者负对与缺源排除，来源独立投影/L2范数。联合损失系数1/.1为首轮固定实施选择，外部报告未提供这两个系数。

两个条件同seed42新初始化，共同冻结H34，第0轮风险完全相同。最多30轮，至少8轮，patience6/min_delta1e-4，年度调参F1选点、并列选较低再入院IBS。对照最佳6轮，推荐最佳8轮；对照与推荐完成8/14轮。此配对分离联合损失效果；V7与V6比较仍混合了容量、输入和架构变化。

## 已选检查点结果

| 模型 | 一年F1 | PPV | 召回 | 平均AUC 7–365 | 一年AP | 再入院IBS | 死亡IBS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原V6 | 0.564706 | 0.488798 | 0.668524 | 0.802141 | 0.597872 | 0.098729 | 0.064127 |
| V6统一历史 | 0.573281 | 0.536408 | 0.615599 | 0.805225 | 0.595851 | 0.099664 | 0.064971 |
| V6对称背景 | 0.577373 | 0.541463 | 0.618384 | 0.806332 | 0.598865 | 0.099614 | 0.065008 |
| V7 NLL对照 | 0.565321 | 0.492754 | 0.662953 | 0.790024 | 0.586543 | 0.100361 | 0.065531 |
| V7 推荐联合损失 | 0.567023 | 0.493802 | 0.665738 | 0.790124 | 0.587351 | 0.100266 | 0.066214 |

F1/阈值均在同一暴露调参折选择，不能称独立验证成绩。IBS越低越好，AUC/AP/F1越高越好。多期限和双原因完整结果见`readout.json`与各条件`result.json`，以下摘30/90/365再入院：

| 天数 | 模型 | F1 | PPV | 召回 | AUC | AP | Brier | 平均预测 | 本库观察比例 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 原V6 | 0.372093 | 0.351648 | 0.395062 | 0.797710 | 0.255111 | 0.044166 | 0.055435 | 0.052907 |
| 30 | V6对称背景 | 0.358491 | 0.290076 | 0.469136 | 0.791103 | 0.285832 | 0.043938 | 0.047036 | 0.052907 |
| 30 | V7 NLL对照 | 0.361809 | 0.305085 | 0.444444 | 0.773359 | 0.267504 | 0.044291 | 0.060522 | 0.052907 |
| 30 | V7 推荐联合损失 | 0.358974 | 0.307018 | 0.432099 | 0.774117 | 0.266829 | 0.044304 | 0.064965 | 0.052907 |
| 90 | 原V6 | 0.451613 | 0.396226 | 0.525000 | 0.796093 | 0.380197 | 0.077665 | 0.108434 | 0.104507 |
| 90 | V6对称背景 | 0.457300 | 0.408867 | 0.518750 | 0.798710 | 0.388450 | 0.077917 | 0.099132 | 0.104507 |
| 90 | V7 NLL对照 | 0.465116 | 0.434783 | 0.500000 | 0.783789 | 0.376500 | 0.078285 | 0.113410 | 0.104507 |
| 90 | V7 推荐联合损失 | 0.457143 | 0.421053 | 0.500000 | 0.783110 | 0.380615 | 0.078266 | 0.118646 | 0.104507 |
| 365 | 原V6 | 0.564706 | 0.488798 | 0.668524 | 0.794785 | 0.597872 | 0.135978 | 0.221786 | 0.234487 |
| 365 | V6对称背景 | 0.577373 | 0.541463 | 0.618384 | 0.802597 | 0.598865 | 0.135964 | 0.223236 | 0.234487 |
| 365 | V7 NLL对照 | 0.565321 | 0.492754 | 0.662953 | 0.780660 | 0.586543 | 0.138829 | 0.223726 | 0.234487 |
| 365 | V7 推荐联合损失 | 0.567023 | 0.493802 | 0.665738 | 0.780864 | 0.587351 | 0.138720 | 0.226824 | 0.234487 |

本次采用性判断：推荐方案未在年度F1与全年概率误差上超过已保存的V6对称背景参照；小幅胜过V7 NLL对照并不证明整组超参数均有效。每来源/药物语义/时间衰减的独立增量需匹配消融，不能由组合结果直接接受或否定。

- `recommended_minus_nll_control`：年度F1差+0.001702，患者聚类95%区间[-0.004235710272275547, 0.007630462956418328]；IBS差-0.000095，95%区间[-0.00038144532610730176, 0.00018409741058113746]。
- `recommended_minus_v6_symmetric`：年度F1差-0.010351，患者聚类95%区间[-0.04227614337838799, 0.033114942673126334]；IBS差+0.000652，95%区间[-0.0016966298043252007, 0.0031975158991006153]。

区间为200次患者聚类配对抽样，固定已选模型/阈值；不消除调参选择偏差，单seed，未做多重比较校正。

## 实测耗时

GPU `NVIDIA GeForce RTX 4070 Ti`，FP32/关闭TF32，既有BERT向量复用，不重新编码或微调BERT。正式两条件及来源核验/缓存/统计合计348.56秒；来源核验11.55秒，CPU稀疏缓存0.90秒、479.1MiB。下列显存为进程Torch峰值，不含桌面等其他GPU进程。

| 条件 | 完成轮 | 最佳轮 | 更新步 | 纯训练秒 | 每轮评价合计秒 | fit总秒 | 训练中位秒/轮 | allocated MiB | reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V7 NLL对照 | 8 | 6 | 1704 | 93.86 | 5.31 | 102.62 | 11.61 | 801.6 | 932.0 |
| V7 推荐联合损失 | 14 | 8 | 2982 | 216.73 | 9.31 | 230.44 | 15.70 | 792.1 | 926.0 |

这些是冻结BERT的V7训练成本；BERT微调时间尚未测量。报告、重放与实现修复时间另外记载，不把它们混入纯训练时间。

![逐轮F1、IBS与实际训练耗时](figures/v7_first_experiment_2026-10-15.png)

## 来源内容敏感性

在已选best上，对当前与过去来源同时关闭其真实mask，H34基础风险保持；历史项仅关闭过去关系。以下为年度概率变化。它是内容依赖诊断，不能代替重训后的价值消融；不用于重新选点。

| 条件 | 关闭来源 | 再入院平均绝对变化 | 再入院最大绝对变化 | 死亡平均绝对变化 |
|---|---|---:|---:|---:|
| V7 NLL对照 | text | 0.00000810 | 0.00086564 | 0.00000333 |
| V7 NLL对照 | drugs | 0.00003448 | 0.00792910 | 0.00001979 |
| V7 NLL对照 | labs | 0.00009010 | 0.00232967 | 0.00003327 |
| V7 NLL对照 | radiology | 0.00000913 | 0.00217239 | 0.00000328 |
| V7 NLL对照 | history | 0.00008659 | 0.00247968 | 0.00003379 |
| V7 推荐联合损失 | text | 0.00010265 | 0.00085614 | 0.00019738 |
| V7 推荐联合损失 | drugs | 0.00111098 | 0.04001851 | 0.00177151 |
| V7 推荐联合损失 | labs | 0.00021039 | 0.01591291 | 0.00051647 |
| V7 推荐联合损失 | radiology | 0.00005399 | 0.00204673 | 0.00014193 |
| V7 推荐联合损失 | history | 0.00289817 | 0.01303819 | 0.00682776 |

已选推荐权重中，关闭药物后的年度再入院平均概率变化约0.1111个百分点，关闭过去历史约0.2898个百分点；检验/文本/影像也有非零影响。对照风险修正的记录间标准差较小，推荐权重的记录间修正变化更明显。该现象支持继续研究有效学习与来源使用，但不能把内容敏感性当作预测增量，也不能据此直接认定任一来源没有价值。

推荐末轮在线NLL2.752402、Focal0.119871、对比0.124361，正对22282、合法负对1122158；完整分项和梯度范数见`history.json`。

## 验证与实现问题

24项相关测试全部通过，其中CPU运行暂跳过的真实CUDA精度检查另在GPU通过。真实三步预检核对所有四字段压缩参数非零梯度/更新、活跃模块更新、无悬空可训练参数、冻结H34、历史内容梯度、共同初值。新的独立进程重建原来源/合同，best每日概率/分数/选点/早停误差0；last重建对照末轮记录并再次重放每日曲线误差0；AP/校准/F1患者区间及内容敏感性也独立重算一致。

初次运行发现字段M4压缩结果被丢弃，已停止并保留源码、日志和检查点于`training_runs/v7_first_2026-10-15`。修正后两条件在r2从头初始化，不接续旧优化器或权重；初次运行不作为性能结论。该问题已加入单元回归和GPU逐参数检查。

原V6默认、旧数据/向量/权重保留；本轮未恢复长期automation，未推送。冻结的BERT骨干与现有向量仍可继续用于V7后续有明确机制的匹配实验。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_FIXED_V6_COX_BLEND_PROTOCOL_2026-10-15.md · lines 1–11

<!-- REVIEW_FRAGMENT {"path":"docs/V7_FIXED_V6_COX_BLEND_PROTOCOL_2026-10-15.md","part":1,"start_line":1,"end_line":11,"source_sha256":"135ec6774bfa54d146cb0205ef0626af6afa39f61e612bbad39b72c933e5afee","fragment_sha256":"135ec6774bfa54d146cb0205ef0626af6afa39f61e612bbad39b72c933e5afee","added_terminal_newline":false} -->
````markdown
# 固定90%完整V6＋10%Cox的概率曲线对照

2026-10-15。顺序执行A后，按既有研究方向C做一次低成本互补性诊断。事前只固定一个比例：每日三状态概率 `0.9*完整V6+0.1*full_refit Cox`，所有原因/时间点共用权重，不搜索或学习权重，不据结果换比例，不重训成员。

使用同一1531出院/402患者的已保存调参预测；完整V6为seed42对称背景、Cox为已有全训练重拟合成员。逐条核对hadm/subject/time/event及tune位置对应；本次不访问确认/测试结局。对每条曲线检查有限性、[0,1]、三状态和1、CIF非递减。

报告365天调参最佳F1（阈值仍在暴露tune上搜索，不当作独立验证）、固定原V6阈值的F1、7–365天再入院/先死亡IBS及平均每日AUC，与完整V6和单Cox点估计比较。相对完整V6做402患者200次聚类配对重采样，固定已得阈值与模型，含两原因IBS区间。无多重比较调整。

这是固定比例的探索性CIF融合，不是OOF学权重，不称独立临床验证，不替换默认模型。仅当有值得保留的互补线索，才另论完整V6内折重训；本次不启动该耗时方向。

输出保存在独立training_runs/v7_fixed_v6_cox_blend_2026-10-15，已有结果拒绝覆盖；再次运行只重算核验，保留患者级预测在本地。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_FIXED_V6_COX_BLEND_RESULTS_2026-10-15.md · lines 1–33

<!-- REVIEW_FRAGMENT {"path":"docs/V7_FIXED_V6_COX_BLEND_RESULTS_2026-10-15.md","part":1,"start_line":1,"end_line":33,"source_sha256":"6a4d5aaa1e6cab0b5a5fb97983cbd2176d8dfc7d2b2d285a952e2c966fb3b797","fragment_sha256":"6a4d5aaa1e6cab0b5a5fb97983cbd2176d8dfc7d2b2d285a952e2c966fb3b797","added_terminal_newline":false} -->
````markdown
# 固定90%完整V6＋10%Cox：零模型重训对照

2026-10-15。按[事前单比例合同](V7_FIXED_V6_COX_BLEND_PROTOCOL_2026-10-15.md)，同一1531调参出院/402患者，每日三状态概率固定为0.9完整V6＋0.1全训练重拟合Cox。所有原因与时点共用比例；未扫描或学习权重，未访问确认/测试结局，未重训成员。

**结果与指标权衡**

| 模型 | 年度调参最佳F1↑ | 沿用原V6阈值F1↑ | 再入院IBS↓ | 先死亡IBS↓ | 平均每日AUC↑ |
|---|---:|---:|---:|---:|---:|
| 完整V6 | 0.577373 | 0.577373 | 0.099614 | 0.065008 | 0.806332 |
| Cox | 0.559229 | 0.554017 | 0.100040 | 0.064363 | 0.796891 |
| **固定90/10融合** | **0.575720** | **0.570681** | **0.099401** | **0.064770** | **0.807168** |

相对完整V6：最佳F1−0.001654、再入院IBS−0.000213、死亡IBS−0.000238、平均每日AUC约+0.000836。两原因概率误差及平均排序点估计改善，但分类F1下降；沿用旧阈值的F1进一步下降。因此不能因为概率指标更好就自动替换原分类模型，也不能把F1下降扩大为融合在所有方面无用。

同一暴露调参集上，402患者聚类配对重采样200次，固定拟合结果及阈值：

| 固定融合−完整V6 | 点差 | 患者95%区间 |
|---|---:|---|
| 年度F1 | −0.001654 | [−0.015747, +0.017748] |
| 再入院IBS | −0.000213 | [−0.000496, +0.000085] |
| 先死亡IBS | −0.000238 | [−0.000421, −0.000057] |

死亡IBS的探索区间位于0以下，是可以保留的互补线索；不称多重比较校正后的确认性证据，不代表新的患者验证或完整院外死亡随访。此次F1为365天本库严格再入院，死亡为其前的竞争事件，IBS为7–365日积分评分。最佳分类阈值仍在同一暴露tune上计算，不能作为独立临床效果。

**工程核对与边界**

已按hadm/subject/time/event与tune位置核对两来源顺序；原V6分类及每日指标完整重现。三条曲线均满足有限、[0,1]、三状态和1及CIF非递减。独立进程重算合同、完整报告和每日融合曲线，最大误差0。源码为[fixed_v6_cox_blend.py](../training/v7/fixed_v6_cox_blend.py)。

首个执行在对照报告元数据比对时发现少计算了既有排序指标字段，尚未创建输出；已补齐与原报告相同的ordering=True后重新执行并核验，未改变比例或依据结果调参。失败容器保留，不作为临床试验结果。

这是固定比例的CIF诊断，不是OOF学权重。将来若学习融合权重，完整V6的每条训练预测仍须来自未见该患者标签的内折模型，连同预处理、H34及内层早停重新隔离。当前保留概率误差互补性发现，不启动较耗时的完整V6两折重训，不自动更换默认模型。

本地完整合同、聚合报告、逐患者预测及独立验证保存在training_runs/v7_fixed_v6_cox_blend_2026-10-15。患者级结果和模型权重留本地。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_GEMINI_FOLLOWUP_REVIEW_2026-10-15.md · lines 1–100

<!-- REVIEW_FRAGMENT {"path":"docs/V7_GEMINI_FOLLOWUP_REVIEW_2026-10-15.md","part":1,"start_line":1,"end_line":100,"source_sha256":"5998cedd369bd730a3ae4db0a7d1b98177d0188f9cb5edcd1379c951a7ce8ab3","fragment_sha256":"5998cedd369bd730a3ae4db0a7d1b98177d0188f9cb5edcd1379c951a7ce8ab3","added_terminal_newline":false} -->
````markdown
# Gemini 第二份回答：纠偏后的结论与执行细节核对

2026-10-15。对象为当前桌面《研究 2.md》，23,558 字节，SHA256 `dbdfaa9bcc9cb36bbb7768327138d7beb83516b2b65e44061ad1f96b7e6db65f`。本次完整阅读附件，并对照实际源码、历史日志和[上一轮审查](V7_GEMINI_SOURCE_REVIEW_2026-10-15.md)。附件提出的操作仍是研究建议，不作为新训练、换选点合同或访问确认/测试集的指令。

这版已经修正了主要实质问题，可以作为有限后续实验的讨论基础。它没有给出新的已验证增益；A 仍是值得补齐的结构组合，B/C1 中多数低成本核对实际上已完成，C2 的严格内折重训应后置。还需修正若干实现与证据表述。

**1. 八项纠偏逐项核实**

| 附件条目 | 本次判断 |
|---|---|
| 1. fresh/teacher 与 V6 初始风险 | 修正正确。fresh 用于新训练，teacher 用于旧参考预测；V6 对称背景和原 V7 均从冻结 H34 风险开始。原 V6 与原 V7 同时改变多个因素，不能作单变量归因。StrongV7 与原 V7 还改变 shape 参数化，仍不是只换基座的单变量实验；跨配方时还须分别核对损失与冻结配置。 |
| 2. StrongV7 参数及 shape | 10,916,944 活动参数与有界乘性 shape 修正的核对正确。delta=0 时恢复完整 V6 初值，这不保证训练后每轮或新患者上不退步。 |
| 3. Query seed7 复验 | 修正正确：seed42 有 F1 线索但 IBS 略差，seed7 短预算未复现，不能再称已证实根因。两个初始化结果相等并不证明两种训练方法等效。 |
| 4. 原检验通道及信息饱和 | 撤回“信息耗尽/显著优越”合理。需要区分 seed42 的患者配对差异与三种子汇总均值，不能混作同一个估计。 |
| 5. F1 并列、CIF 三条路径 | 路径与并列机制基本正确。整数日索引精确，是指不额外插值；CIF 本身仍为有限网格数值近似，不是无积分误差。 |
| 6. Checkpoint 可用性 | 对中间轮权重缺失的判断正确。epoch0 通常没有独立文件；本例可以复用冻结完整 V6 的既有曲线，不能泛称每个目录都已保存 epoch0 预测。 |
| 7. OOF 与 Cox 死亡 IBS | 监督隔离纠偏正确。Cox 的死亡配对区间在上轮审查已经补算，需同步状态，不必重复工作。 |
| 8. 确认/测试历史曝光 | 不再称全新盲测是正确修正。除了旧 test，confirmation 也有明确历史确认评估记录，不能只解释 test 曝光。 |

第 4 项的数值区别：`linear−availability` 的 seed42 F1 点差为 −0.002218、IBS 点差约 −0.000219；三种子 F1 均值差约 −0.000361、IBS 均值差约 −0.000091。这些并不矛盾，但必须注明各自的估计对象。

**2. 建议 A 成立，但当前构造函数不能直接接收变体参数**

附件找出的组合仍然没有实际运行：冻结完整 V6，关闭 V7 的 `drug_cross`/`rad_cross`，其余保持。

具体限制在 [StrongV7 构造函数](../training/v7/strong_anchor_experiment.py#L29)：它接受 `base,parent,a,seed`，内部写死 `Variant('strong_anchor')`，没有外部 `variant` 参数。因此，仅在新脚本的工厂函数中传入 `cross=False` 不会自动生效；不能把“复用旧类”理解为已有可直接调用的开关接口。

可以在新文件中继承 StrongV7，在父构造完成后明确设置无 Query 变体，并冻结两个跨源模块，复用原 forward 和 anchor buffers；也可另建小型派生类。需要检查 `variant.cross=False`、两个模块均冻结且实际跳过，验证共同参数初值、输入、epoch0 风险一致，优化器没有纳入冻结参数。[实际变体分支](../training/v7/sequential_variants.py#L32)

活动参数应从 10,916,944 减至 9,337,424。原 StrongV7 源码和结果目录都被历史合同绑定，应全部保留，而不只是保留结果目录。A 保持原 F1 优先选择规则、共同训练预算与风险公式，才不会同时引入 B 的选点变化。

节约预算时可先做 seed42，最多16轮、最少8轮、patience6，与已有对应对照比较；有值得复验的结果再补 seed7。若声称两种子的配对结果，则应完成两组，而不能用单种子代替。

**3. 建议 B 的只读部分已经完成，本次补齐 projected 与最小 IBS 读法**

上轮已经检查 StrongV7 seed42/7、availability42、linear42 的严格 IBS 约束选点。本次按同一阈值，新增 projected42，并对每条既有轨迹额外读出最小再入院 IBS 的轮次。

所有结果均在同一已暴露 tune：1,531 次出院、402 名患者；年度 F1 对应365天严格再入院，IBS对应7–365日的本库记录终点，死亡是严格再入院前死亡。阈值为完整 V6 的再入院 IBS=0.0996143261，容差固定为0，未扫描 tol。未访问确认/测试患者结果，未拟合模型。

| 模型 | 原 F1 优先选点 | IBS≤完整 V6 后的最高 F1 选点 | 历史最低再入院 IBS 选点 |
|---|---|---|---|
| StrongV7 seed42 | e10：F1 .581683 / IBS .099908 | e6：F1 .581366 / IBS .099596 | **e5：F1 .579926 / IBS .099313** |
| StrongV7 seed7 | e2：F1 .580645 / IBS .099634 | e6：F1 .578065 / IBS .099540 | 同 e6 |
| availability42 | e15：F1 .583436 / IBS .099924 | e0：F1 .577373 / IBS .099614 | 同 e0 |
| linear42 | e1：F1 .581218 / IBS .099705 | e0：F1 .577373 / IBS .099614 | 同 e0 |
| projected42 | e10：F1 .578879 / IBS .099701 | **e0：F1 .577373 / IBS .099614** | 同 e0 |

StrongV7 seed42 第5轮在这条轨迹中再入院 IBS 最低，F1 也高于完整 V6 初值，是可保留的历史线索。不过同时要看第二种结局：

| seed42 状态 | 年度再入院 F1 | 再入院 IBS | 死亡 IBS |
|---|---:|---:|---:|
| 完整 V6 初值 | .577373 | .099614 | .065008 |
| StrongV7 e5 | .579926 | .099313 | .065195 |
| StrongV7 e6 | .581366 | .099596 | .065513 |
| StrongV7 原已选 e10 | .581683 | .099908 | .065145 |

第5、6轮均不是两个结局的 IBS 同时改善；不能将只约束再入院 IBS 的规则称为已经解决双原因概率误差问题。seed7 历史没有保存每轮死亡 IBS，审查输出保留 null，不能补造该值。

第5、6轮权重均未保存。它们的标量指标来自当时日志，不等于当前已有可用模型，也无法据此生成逐患者区间。最小 IBS 与 IBS 约束后最高 F1 是两种不同选择目标，需分别报告。

新增只读审查结果（本地资产`work/gemini_source_review_2026-10-15/followup_audit.json`，未随公开包提供）已由独立进程重算，核验文件（本地资产`work/gemini_source_review_2026-10-15/followup_verification.json`，未随公开包提供）记录最大误差0，并绑定上轮审查哈希。该计算只读标量历史，没有重训，也没有重新计算临床模型的概率曲线。

**4. “加载重放为零误差”与“从头训练轨迹已验证一致”仍需区分**

[现有 StrongV7 verify](../training/v7/strong_anchor_experiment.py#L99)和[通用 verify_result](../training/v7/query_replication.py#L65)会加载 best/last，重算预测、指标和选择/早停状态。它们没有从初始参数重跑全部优化步骤。

设置 `torch.use_deterministic_algorithms(True)` 有助于确定性，但当前零误差验证不能直接推广为“已证明从头训练到第6轮可完全相同”。未来如定点恢复，应使用相同数据和预处理、共有参数初值、批次顺序、随机数重置、优化器、软件及设备条件，在新目录逐轮比较0至目标轮历史；再检查目标轮预测。若训练轨迹不同，需如实记录，不能把新结果冒充旧中间 checkpoint。

后验增加保存逻辑可以只保存而不影响训练，但如同时更换早停规则、优化器或模型选择反馈，实验已不再是恢复原轨迹。无需为此覆盖旧结果或旧 source hash。

**5. C1 的死亡配对区间已经完成；调用现有辅助函数还要核对结局标签**

上轮审查以两模型同一患者/标签对齐、固定既有曲线、402患者聚类重采样200次，计算结果为：

| Cox−完整 V6 | IBS 点差 | 患者配对95%区间 |
|---|---:|---|
| 再入院 | +0.000425 | [−0.001700, +0.003114] |
| 严格再入院前死亡 | −0.000644 | [−0.002375, +0.001123] |

两者均跨0。Cox 死亡方向存在较低点估计的互补线索，当前没有统计稳定优势的证据。附件说“未计算区间”反映的是它看到的原实验包状态，需同步为审查阶段已补算；没有必要重新训练或重复同一计算。

另一个实现细节：[evaluation.paired_interval](../training/v6/evaluation.py#L58)没有 cause 参数，内部 `outcome_matrix(times,events)` 默认是原因1再入院。直接把死亡曲线传进去会使用错误目标。它是现有再入院专用工具，不能因为接口接受二维概率矩阵，就当作通用双结局工具。

[outcome_statistics](../training/v7/sequential_readout.py#L35)虽然同时计算两原因 IBS 点值，但用于后续 paired 的元组只返回再入院逐记录误差。因此其现有配对区间路径也不能直接当作死亡区间。上轮审查使用明确 `cause=2` 的死亡目标和对应逐记录误差计算，未套错这两个函数。

患者配对区间条件于已拟合模型，200次是快速诊断预算，没有消除反复选模或历史曝光偏差。C2 所要求的内折预处理、H34、V6网络和内层早停隔离是合理的，应排在 A 及低自由度互补性诊断之后。

**6. 曝光状态与附件版本记录**

附件已经引用旧 test 的 `Reused test, not blind` 声明。还应补上 confirmation：[历史确认结果](V6_TASK_OPTIMIZATION_RESULTS_2026-10-13.md#L3)记录24模型调参重放、9模型确认重放及确认表格。这不是仅凭函数导入推断出来的潜在访问，而是已报告的实际历史使用。

本批 V7 没使用确认/测试患者结果，不会把这些子集重新变成全新盲测。后续评估仍有用途，需如实限定历史曝光。

当前附件复用了旧文件名《研究 2.md》，但其内容不是旧 Focal 研究原件。已保存当前附件的本地固定快照（本地资产`work/gemini_source_review_2026-10-15/external_reply_2_dbdfaa9bcc9c.md`，未随公开包提供）及版本绑定记录（本地资产`work/gemini_source_review_2026-10-15/followup_attachment_binding.json`，未随公开包提供）。旧损失实验合同中的 `/research/research_2.md` 哈希为 `146e1cb432b55416fda6a4e558eb787645787f5a721079eca96e1c2d286282c5`，与当前附件不同；以后核验旧损失实验时，不能再将当前同名文件当作当时的原附件挂载。历史合同、模型、权重及预测均未修改。

**7. 本轮结论与优先顺序**

可以采纳这一版经过限定后的核心研究方向，无需再围绕已纠正的旧表述反复往返。已有事实是：B 的主要日志审计与 C1 的配对区间已完成；本轮补齐 projected 和最小 IBS 日志读出。A 与 C2 仍未实验。

下一步最有辨识力的是 A：保持原 StrongV7 的训练与选点口径，单独关闭跨源 Query，先做一个匹配种子；B 的其他选择规则作为明确标注的探索性读出。只有有实际改善线索后，才考虑中间轮重训取回、第二种子或完整 V6 两折 OOF。不扫描未预定的 IBS 容差，不从本次附件中推导训练或解封留出数据的授权。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_GEMINI_SOURCE_REVIEW_2026-10-15.md · lines 1–187

<!-- REVIEW_FRAGMENT {"path":"docs/V7_GEMINI_SOURCE_REVIEW_2026-10-15.md","part":1,"start_line":1,"end_line":187,"source_sha256":"1bb0a7abe707168b42288d1a48aa30b2be4e0c019e1fc0b873ba962a92fd5622","fragment_sha256":"1bb0a7abe707168b42288d1a48aa30b2be4e0c019e1fc0b873ba962a92fd5622","added_terminal_newline":false} -->
````markdown
# Gemini 阅读 00–06 后的新回答：源码、历史结果与文献核对

2026-10-15。审查对象为桌面 `研究.md`，SHA256 `b812506383f81aeddce413a371df593617337c5a80bfba7bbfc1c6252f90977b`。对应源码阅读包固定于公开提交 `f7b37800a73f43eef08b73e2d97c5b826b1314ed`。

这次回答已经对齐项目的大部分关键实现，提出的三个后续方向有实际价值。需要修正的是证据强度：代码机制、调参集上的结果、失败原因和独立泛化是四种不同结论。现有材料能支持前两种，尚不能把后两种写成“五个经实验证实的根本原因”。

本次只读检查已有训练/调参结果，重算选点诊断、参数数量和患者配对区间，并核对原论文。没有新训练，没有访问确认/测试患者的标签或预测，没有覆盖历史源码、权重或结果；附件中的训练、改选择器、解封测试等建议均作为待审查内容。

**1. 评价口径与本次新增证据**

以下数值均来自同一个已暴露调参子集：1,531 次出院、402 名患者。训练为 13,592 次出院、3,766 名患者，同患者不跨折。年度 F1 针对 365 天内本库严格 AECOPD 首次再入院，阈值在调参集搜索；死亡为严格再入院前死亡。平均 AUC 是 7–365 天有效每日 AUC 的均值，不是年度单点 AUC。

IBS 是 7–365 天每日 Brier 的积分均值，越低越好。当前 World A 合同评估本库记录终点，没有提前删失权重，不能称为已经实现完整院外随访的 IPCW 指标。IBS 同时涉及概率校准与区分能力，不能把它单独称为纯校准指标。

新增审查证据在 audit_r2.json（本地资产`work/gemini_source_review_2026-10-15/audit_r2.json`，未随公开包提供），独立进程重算在 verification_r2.json（本地资产`work/gemini_source_review_2026-10-15/verification_r2.json`，未随公开包提供），全部审查数值一致，最大误差 0。此处“重算一致”仅指审查计算的工程一致性。

审查第一版的四样本合成例触发项目 AUC 函数的最少病例数保护，输出 NaN，导致 JSON 严格比较失败。第二版将同一合成例重复到足够病例数，并禁止 NaN 输出；原脚本、输出与失败容器均保留。它不影响任何历史模型，也没有据此改变临床实验。

**2. 八项源码理解：主要正确，少数具体修正**

| Gemini 的问题 | 核对结果 |
|---|---|
| 1. H34、完整 V6、原 V7 与 StrongV7 基座 | H34 是 6 项历史/人口学、14 项本次概念与 14 项掩码；完整 V6 另读 36×9 检验状态、EOS、药物/微生物/支持和四字段文本、全部可见既往住院。原 V7 冻结的是 72 参数 H34 双原因基座；StrongV7 冻结的是完整已选 V6 的风险参数预测。此区分正确。 |
| 2. Query 的来源与检验风险路径 | 药物 Query 读取 16 个文本 token 与 2 个影像类别 token；影像 Query 读取文本。原检验分支以 37 个 token 独立进入融合，并非原 Query 的 Key。源码理解正确。 |
| 3. 四个修正输出、风险分布、CIF | 两原因各一项幅度与一项 shape 修正，连续时间 Weibull 联合竞争风险 NLL，正确。日曲线的常用实际调用是 `module_surgery.probability_curve → token_model.fast_cif`；与 `risk.predict_cif` 同类网格数值求积，不能把日 CIF 总积分称为无近似的精确积分。 |
| 4. Focal、平滑与对比 | NLL 是否保留的配方区分正确。gamma=2、alpha=.75、平滑=.05、距离裕度=.5；.1 是对比项权重，由本项目实现选择，不是原外部报告指定。标签平滑是在 Focal 内改变二元目标，并非另加一项无名损失。 |
| 5. 冻结小读取器与完整 V7 | 前者冻结 V7 表示，仅训练读取器；StrongV7 则训练 V7 编码器和修正头、冻结完整 V6 风险，正确。原 V7 可训练参数 11,180,112；当前 StrongV7 可训练参数是 **10,916,944**，其未使用的对比投影 263,168 参数已冻结，不是文中约 11.18M 的原数。 |
| 6. 投影拟合的时点、患者与梯度 | 初始化及每个 epoch 后仅用训练患者拟合 `[1,H34]` 的岭映射，epoch 内固定，映射拟合不反传，调参时不重拟合，正确。这里去除的是 H34 的线性可预测部分，不等同于去除完整 V6 已解释的全部信息。 |
| 7. 两折成员与隔离 | 神经成员为 179→256→32→4 的 CompactNeural，不是完整 V7；内折预处理、PCA、训练患者隔离和训练 OOF 学权重的理解正确。 |
| 8. 工程重放与泛化 | 区分两者正确。但“本批未访问 confirmation/test”不能推出“整个项目历史上从未曝光”。历史文档已记录确认评估及旧测试反复使用，不能直接称为全新盲测。 |

主要调用链证据：[V6 初始化入口](../training/v6/unified_longitudinal_experiment.py#L78)、[V6 对称背景实验](../training/v6/symmetric_context_experiment.py)、[V7 主模型](../training/v7/model.py)、[StrongV7](../training/v7/strong_anchor_experiment.py#L29)、[变体及冻结规则](../training/v7/sequential_variants.py#L32)、[实际日曲线](../training/v6/token_model.py#L85)、[评价定义](../training/v6/evaluation.py)。

阅读范围表标题写“19 个”，实际列出 28 个 Python 文件和 8 个报告，共 36 个，与原包相符。这是计数错误，不能仅凭此推断没有读正文。它明确承认缺失被调用依赖，是合理的边界说明。

**3. 对“五个根本原因”逐条判断**

**3.1 强基座值得保留；“原 V6 是已收敛编码器微调”与源码不符。**

V6 入口先构建 `fresh=SurgeryModel(...)`，另构建 `teacher` 并加载参考权重。对称实验训练的是 fresh 的副本，合同明确 `original_clinical_weights_not_reused=True`。因此，原 V6 与原 V7 的神经部分均从新初始化开始训练，差别涉及宽度、结构、优化、风险参数化等，不能把“V7 从零训练、V6 在已收敛编码器上微调”作为归因前提。

StrongV7 的确把已有完整 V6 冻结风险作为出发点，但还改变了 shape 修正形式：原 V7 对 raw-shape 加修正再 softplus；StrongV7 对已正的 anchor_k 乘 `exp(.25*tanh(delta))`。所以这不是严格“其他不变，仅替换基座”的单变量实验。

| 同一调参集 | 年度 F1 | 再入院 IBS |
|---|---:|---:|
| 原 V7，NLL | 0.565321 | 0.100361 |
| 完整 V6，也是 StrongV7 的 epoch 0 | 0.577373 | 0.099614 |
| StrongV7，seed42 已选轮 | 0.581683 | 0.099908 |
| StrongV7，seed7 已选轮 | 0.580645 | 0.099634 |

StrongV7 seed42 在完整 V6 上的修正增量是 **+0.004310 F1**，不是把原 V7 到 StrongV7 的全部 +0.016362 都算成新模块带来的增量。Kumar 的论文讨论预训练模型微调在分布外的特征扭曲；本项目 BERT 冻结、下游网络新建，不能直接套用为失败证明。[原论文](https://arxiv.org/abs/2202.10054)

**3.2 关闭 Query 是可用线索；“已证明记忆伪相关、有害无益”过强。**

seed42 原基座下关闭 Query 后 F1 0.577075、平均 AUC 0.804515，优于对照；这是保留结构减法实验的理由。但另一次匹配的 seed7 短预算复验中，对照和关闭 Query 都选择 epoch 0，F1 同为 0.565321，没有复现该增量。不能仅引用有利种子。

影像粒度和覆盖率的限制、两个跨源模块增加 1,579,520 参数，均属真实问题。尚未测量训练噪声记忆或模块的因果贡献，不能写成已证明机制。文中把“最佳 epoch 6/10”称为“第 6/10 轮早停”也不准确，对应实际训练到 8/16 轮。

**3.3 availability 是值得保留的低成本参照；信息耗尽没有被证明。**

| 同一冻结基座，读取器三种子 | F1 均值±标准差 | 再入院 IBS 均值 |
|---|---:|---:|
| 化验是否测量＋末测距出院时间 | 0.582717±0.000719 | 0.099915 |
| 再加入首末物理值的线性编码 | 0.582355±0.001177 | 0.099824 |

前者 F1 略高、后者 IBS 略好，不能根据 F1 单指标认定数值无用。原 V6 的 9 通道本身已含 `latest_measurement_age` 和 `axis_observed`；新增读取器的收益可能来自已有信号的重新表达或权重调整，不是发现 V6 从未输入的剩余信息。[原通道定义](../patient_data/v6_lab_state.py#L21)

小读取器、有限种子、同一监督基座和调参集的阴性结果，没有测量“互信息”，也不构成信息理论上界。可以停止当前低收益分箱网格，不能据此宣布所有化验数值或后续真实新信息都无价值。

药物语义与动态试验的预算止损合理；动态表征应使用同一动态分支的来源对照，不能仅拿不同输入的 `name_source` 归因。`dynamics_slope−dynamics_source` 有约 +0.000895 F1 的微弱差异，但同时加入了首末数值和斜率，不能全归于斜率。隔离斜率应比较 `dynamics_slope−dynamics_main`：F1 0.580645 对 0.577373，后者选初始化，而前者 IBS 仍较高，未形成全面优势。软路由也应首先比较 `routing_prior` 与 `routing_uniform`，后者没有同等三种子复验，不能把与 availability 的跨结构差异都归于路由先验。

信息性观测文献支持“测什么、何时测可能帮助预测”，没有证明本队列的信号来自医生已凝练的真实病情、也没有证明连续值饱和。观测流程改变还可能使这种信号失效。[Sisk 等综述](https://academic.oup.com/jamia/article/28/1/155/5961436)、[Groenwold 原文](https://link.springer.com/article/10.1186/s41512-020-00077-0)

**3.4 投影没有得到预期的 F1 收益；“因为抹掉方差所以失败”仍是假设。**

投影模型的中心化方差保留 15.0457%，方差能量相差 6.6464 倍；RMS/范数尺度相差 **2.5781 倍**，不是范数缩小近 7 倍。表示方差不等于对结局有效的信息量。

端到端投影 seed42 F1 0.578879，低于未投影 0.581683；同时再入院 IBS 从 0.099908 改善到 0.099701，死亡 IBS 也改善。应描述为指标权衡，没有整体一致改善。配对 F1 和 IBS 差异区间均跨零。

每轮更新投影、epoch 内固定且停止映射求解梯度，可能改变优化动力学，值得作为解释假设；目前没有梯度方向、逐轮抵消量或固定/交替投影的完整匹配证据，不能把“交替投影阻尼”写成已证实原因。岭回归去重也不是无正则精确正交投影，更不是强制网络只学完整 V6 尚未解释的信号。

**3.5 选点偏差值得控制；F1 不存在“浮点数所以几乎不并列”的性质。**

F1 由有限的 TP、FP、FN 计数决定，可以反复出现完全相同的值。历史中 StrongV7 seed42 的最高 F1 在 epoch **7、10** 并列；availability seed42 在 **12、14、15** 并列。因此 IBS 次级裁决确实会起作用。

本次重查已选模型中 F1>0.580 的 25 个条目，其 IBS 范围为 **0.09963425–0.10052021**，文中的上界 0.100100 不准确。它观察到的 F1 与 IBS 权衡，在这些已选点上成立；下文历史 epoch 6 又说明这不是所有 epoch 的必然规律。

重复使用同一 tune、跨模型及阈值筛选会增加乐观偏差。Cawley、Dwork 支持这一一般风险，没有证明本项目的具体 +0.005～+0.009 增益一定是假象，也没有保证改用 IBS 后测试 F1 必升。区间跨零表示证据尚不足，不能当作“没有效果/已经等效”的证明。[Cawley 与 Talbot](https://www.jmlr.org/papers/v11/cawley10a.html)、[Dwork 作者页所列 Science 论文](https://dwork.seas.harvard.edu/publications/preserving-statistical-validity-adaptive-data-analysis)

**4. 原文状态表逐项处置**

| 方向 | 保留的结论与需要调整的措辞 |
|---|---|
| Weibull 联合竞争风险 | 保留作当前主干和匹配比较；CIF 使用数值近似，不由此次实验排除所有其他风险族。 |
| 历史可见性守卫 | 已实现并应保持；它保证被选历史住院及文本的特定时间规则，不能扩大为“彻底杜绝所有泄漏”。当前住院 FINAL 文本仍需区分回顾性记录任务与真实出院当刻可用性。 |
| Focal＋平滑＋对比 | 不用它直接替代 NLL 的预算决定有充分当前实验依据；叠加 NLL 的结果为小、混合收益，不能一概写成已经破坏校准。详见下表。 |
| 跨源 Query，包括药物查询检验 | 已实验，原 H34 条件与种子结果有限；强 V6 下关闭原 Query 仍未实验。不能普遍否定所有跨源注意力。 |
| PLE、速度、曲率、密度 | 当前读取器未显示稳定优势，可以停止扩大同一网格；不能说所有连续化验已被耗尽。 |
| 药名语义与给药动态 | 当前增量小，降低优先级；应使用匹配来源对照。 |
| 原因专属路由 | 当前实现未显示相对优势，预算止损合理；部分对照种子不完整，不作因果定论。 |
| 强 V6＋V7、固定/交替去重 | 三种训练机制需分别报告；静态投影方差较大，端到端投影没有达到预定 F1 扩种门槛，保留概率误差改善线索。 |
| 两折与全量重拟合 | 已完成的传统/紧凑神经融合可作独立家族对照；缺少完整 V6 成员值得补，但弱成员也可能互补，不能用单模型排名解释必然失败。 |
| Home NIV、完整心衰治疗、频繁加重与出院方案交互 | 可靠出院治疗/确认阴性语义不全，未知不可填 0；不能把“未记录”命名为临床治疗不足。 |
| 全队列文本事实与跨数据集蒸馏 | 缺少已核验的全量事实接口或目标数据对齐桥；这是需另做数据工程的条件，不表示现有原始文本必然没有这些内容。 |

| 同一原 V7，已选轮 | 年度 F1 | 再入院 IBS |
|---|---:|---:|
| NLL 对照 | 0.565321 | 0.100361 |
| NLL＋Focal＋平滑 | 0.565321 | 0.100127 |
| NLL＋Focal，不平滑 | 0.567023 | 0.100253 |
| NLL＋Focal＋平滑＋对比 | 0.567023 | 0.100266 |

后面三行再入院 IBS 均略优于 NLL 对照；死亡指标并不都改善。直接去掉 NLL 的配方选中初始化，实际末轮明显恶化。更准确的结论是：**本次直接替代失败；叠加配方没有形成稳定且全面的提升，暂不扩大搜索**。这与“Focal 在任何任务无用”是不同判断。[损失实验原报告](V7_LOSS_COMPARISON_RESULTS_2026-10-15.md)

**5. 对 A、B、C 三项新建议的落地判断**

**B：先做不重训的历史选点检查——本次已完成。**

约束固定为调参再入院 IBS≤完整 V6 的 **0.0996143261**，在满足约束的历史轮次中取 F1 最大、IBS 次级裁决、再取较早轮；并单独检查该轮权重是否保存。

| 模型 | 原已选 epoch / F1 / IBS | 按约束重读历史的 epoch / F1 / IBS | 能否直接取对应权重 |
|---|---|---|---|
| StrongV7 seed42 | 10 / 0.581683 / 0.099908 | **6 / 0.581366 / 0.099596** | 没有保存第 6 轮权重 |
| StrongV7 seed7 | 2 / 0.580645 / 0.099634 | 6 / 0.578065 / 0.099540 | 没有保存第 6 轮权重 |
| availability seed42 | 15 / 0.583436 / 0.099924 | 0 / 0.577373 / 0.099614 | 可回到原完整 V6 初始化 |
| linear seed42 | 1 / 0.581218 / 0.099705 | 0 / 0.577373 / 0.099614 | 可回到原完整 V6 初始化 |

seed42 第 6 轮确实有值得保留的折中，只比已选轮低约 0.000317 F1；seed7 则要损失约 0.002581 F1。两个小读取器的所有已训练轮次均不满足这个严格 IBS 约束。不能承诺该办法统一保住 0.583 F1 并修好概率误差。

历史文件只保存 best/last 权重及预测，无法凭 epoch 标量日志恢复第 6 轮曲线、患者配对区间或可部署模型。零 GPU 可以完成的是诊断；若要获得对应状态，需要独立重放训练或新实验保存相应状态。重看 tune 后提出的规则也是探索性选择，不能悄悄重写旧“最佳”结果；选择规则和早停规则还应分别定义。

**A：强 V6 上关闭跨源 Query——最值得优先补齐的结构实验，尚未运行。**

它准确找到了矩阵中的空格：强基座保留 Query 与弱基座关闭 Query 都做过，**强基座关闭 Query**尚未做过。先只关闭这两个模块，与已有 StrongV7 使用相同输入、初始化、风险参数化、损失、训练预算和选点规则，才能回答 Query 在强基座下的净作用。

这次应新增独立实验入口/子类与合同，不能直接改已锁定的 `strong_anchor_experiment.py`；历史源文件被结果合同哈希绑定。活动参数可由 **10,916,944 降至 9,337,424**，尚有约 934 万参数。

“低秩修正头”应另列一个后续变量。若将 256→256→4 改为 256→16→4，只减少 66,820−4,180=62,640 参数，占原 StrongV7 活动参数约 **0.574%**，不能把这个改动等同于显著缩小整个编码器。它可能有正则化价值，但要单独验证。

**C：完整 V6 与 Cox/tree 在 CIF 上融合——合理的互补性假设，区分廉价诊断和真正 OOF。**

Cox 全量重拟合的死亡 IBS 为 0.064363，完整 V6 为 0.065008，点估计更低。此次基于同一 402 患者的固定预测，做 200 次患者聚类配对重采样：

| 差异：Cox−完整 V6 | IBS 点差 | 患者配对 95% 区间 |
|---|---:|---|
| 再入院 | +0.000425 | [−0.001700, +0.003114] |
| 再入院前死亡 | −0.000644 | [−0.002375, +0.001123] |

这支持保留死亡方向的互补线索，不能称为已证明显著更好，也不能仅凭 IBS 点值称其“校准极佳”。上述区间条件于已拟合模型、未调整反复筛选，抽样次数为快速诊断预算。

已有成员排名低，并不意味着融合必然低于强模型。本次合成例中两个成员各自最佳 F1=0.8、AUC=0.5，平均后 F1=AUC=1.0；它仅反驳“数学上不可能”，不提供本项目真实收益证据。

若只想低成本看互补性，可以在已有、严格对齐的调参预测上做**事先限定的固定权重**描述性检查，明确仍属暴露调参探索。所有原因、所有时点共用一个凸组合权重，才能直接继承 CIF 单调与总概率约束；不能任意分原因调权。

若要从监督数据学习权重并称为 OOF，完整 V6 也必须对每个训练患者给出由未见该患者的内折模型生成的预测；预处理、基座拟合、候选选择的可见性都要说明。把已在全训练集拟合的 V6 训练预测，与 Cox 的 OOF 预测混在一起拟合权重，不是严格 OOF。当前无监督基座预训练的两折协议与历史监督训练的完整 V6，是不同实验合同。

时间优先顺序建议为：**已完成的 B 只读诊断 → A 单变量试验 → C 低自由度互补性诊断 → 有希望后才做完整 V6 内折重训和学权重**。不先扩化验/药物网格；瓶颈头另立变量。A/C 仍是未实验建议，本次没有产生它们的新临床成绩。

**6. 九篇引用逐项核实**

| 原文列出的文献 | 实际支持范围与书目信息 |
|---|---|
| Kumar 等，ICLR 2022 | 论文真实，讨论预训练微调与分布外特征扭曲；不能直接证明当前冻结 BERT＋新下游 V7 的失败原因。[论文](https://arxiv.org/abs/2202.10054) |
| Cawley 与 Talbot，JMLR 2010 | 论文真实，支持有限数据下模型选择过拟合及选择准则方差的重要性；没有本项目具体 F1 增益幅度，也不保证 IBS 门控能提高外部 F1。[原文页](https://www.jmlr.org/papers/v11/cawley10a.html) |
| Dwork 等，Science 2015 | 作者机构页核实了 Science 论文身份；出版商全文此次未成功取得。支持适应性复用留出集的一般问题，不能借它证明某折历史从未曝光。[作者页](https://dwork.seas.harvard.edu/publications/preserving-statistical-validity-adaptive-data-analysis) |
| Sisk 等，JAMIA 2021 | 对应综述真实，正确文章号是 **5961436**，正确 PMC 是 **PMC7810439**；原文提供的 5940859/PMC9006687 不对应这篇。后者是 Yan 等 2022 年关于可观测性偏差的另一篇研究。综述支持信息性存在/观测，但不能证明本项目某个读取器为何胜出。[Sisk 原文](https://academic.oup.com/jamia/article/28/1/155/5961436)、[被误链的 Yan 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9006687/) |
| Groenwold，2020 | 论文真实，既讨论缺失模式的预测信息，也讨论观测机制变化与可迁移性问题。[原文](https://link.springer.com/article/10.1186/s41512-020-00077-0) |
| Lin 等，Focal，ICCV 2017 | 原始目标是密集检测类别不平衡；不能仅引用它证明本项目校准崩溃的具体机制。Focal 非严格恰当性的形式分析可参考 Charoenphakdee 等，且该文还给出后验恢复映射；“不严格恰当”不是“所有有限样本校准必然更差”。[原始 Focal](https://arxiv.org/abs/1708.02002)、[后验概率理论分析](https://arxiv.org/abs/2011.09172) |
| Szegedy 等，CVPR 2016 | 已读原文第 7 节，均匀标签平滑的目标确实向均匀分布移动。无其他重加权时，二元交叉熵的总体最优目标为 `(1−ε)p+ε/2`；实际 Focal 叠加还有额外影响。不能由目标移动推导有限样本 IBS 必然恶化。[第 7 节](https://arxiv.org/pdf/1512.00567) |
| Gorishniy 等，NeurIPS 2022 | PLE 与数值嵌入原论文真实；没有提供本项目临床阈值的指南依据，也没有证明这里的负结果来自信息饱和。[原文](https://arxiv.org/abs/2203.05556) |
| Alberge 等，2024 | 实际题目为 **Teaching Models To Survive: Proper Scoring Rule and Stochastic Optimization with Competing Risks**；作者为 **Julie Alberge、Vincent Maladière、Olivier Grisel、Judith Abécassis、Gaël Varoquaux**，原文的 Jules Alberge/Julie Josse 有误。论文支持竞争风险下恰当评分的方法设计，不保证 NLL/Brier 每步梯度同向，亦不保证 F1 同步上升。[原文](https://arxiv.org/abs/2406.14085) |

**7. 收官与盲测措辞的边界**

停止不断扩大暴露调参集上的弱配方网格、保留少量有明确差异的候选，是合理的预算管理。57 次拟合可以形成有价值的方法边界记录；并不自动构成所有方法无效的证明，也不能由零误差重放推导“只剩两个有效机制”或保证论文创新性。

候选名单中保留完整 V6、强 V6＋V7、小读取器有可解释性，但必须按当前任务目标与预定标准锁定，不能从调参后的小数差异推导“唯一最优名单”。

历史 [V6 确认结果](V6_TASK_OPTIMIZATION_RESULTS_2026-10-13.md#L3) 记录了九模型确认重放和确认指标；[V6 评价合同](V6_EVALUATION_PROTOCOL.md#L11) 明确旧历史测试已反复使用。因而现有确认/测试折可以有用途，但必须说明历史曝光，不能重新命名为全新盲测。本次只读查看这两份历史聚合文档，没有读取或运行确认/测试患者结果。

**8. 供继续讨论的依赖补充**

原 00–06 阅读包保留原样。新增 07_actual_dependencies_and_guards.txt（本地资产`outputs/gemini_plaintext_review_2026-10-15/07_actual_dependencies_and_guards.txt`，未随公开包提供），包含同一公开提交中 19 个依赖文件、3,204 行正文、216,599 字节；每文件带原始路径、行号、Git blob SHA256。它补充 V6 fresh/teacher 的实际训练入口、紧凑模型、检验原通道、曲线计算与选择/早停实现，不声称覆盖全部递归依赖。

单独的 补充清单（本地资产`outputs/gemini_plaintext_review_2026-10-15/supplement_07_manifest.json`，未随公开包提供） 和 独立还原核验（本地资产`outputs/gemini_plaintext_review_2026-10-15/supplement_07_verification.json`，未随公开包提供） 不替换原 manifest。所有正文仅从已经公开的 Git blobs 导出；未公开的历史确认/评价文档未混入 07，曝光边界在本审查中单独说明。

可以回给 Gemini 的具体问题：请保留 A、B、C 的可检验建议，结合 07 重新核对 V6 初始化、StrongV7 参数与 shape 参数化、Query seed7 复验、历史 F1 并列、B 的 checkpoint 可用性和 OOF 监督可见性；把“已证实根因/信息榨干/显著更好/完全未曝光”改为与实际证据一致的限定结论，并纠正 Sisk 与 Alberge 书目。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_LAB_AWARE_DESIGN_AND_STATUS_2026-10-15.md · lines 1–88

<!-- REVIEW_FRAGMENT {"path":"docs/V7_LAB_AWARE_DESIGN_AND_STATUS_2026-10-15.md","part":1,"start_line":1,"end_line":88,"source_sha256":"a62c24017e2a2eb679b5b982d7a05ac6357195708ed8beffdd9de5db22d57c00","fragment_sha256":"a62c24017e2a2eb679b5b982d7a05ac6357195708ed8beffdd9de5db22d57c00","added_terminal_newline":false} -->
````markdown
# 已知检验输入后，如何重新审查并适配V7

更新：本文保留方案审查/首轮时点的状态。后续56次拟合、独立验证及每项决定见[顺序实验结果](V7_SEQUENTIAL_RESEARCH_RESULTS_2026-10-15.md)，不再以本文的“尚未做”作为当前进度。

2026-10-15。本文给阅读过“文本、影像报告、处方药物”方案、但尚不了解本项目检验分支的研究者使用。已有建议仍值得验证；需要先对齐实际输入、任务与已完成实验，不能把它们整体接受或否定。本文整理方案，不新增训练、不改动已经锁定的模型或损失公式。

## 必须先说明的任务

以每次合格出院记录为预测锚点，预测未来1–365日首次严格AECOPD非转院再入院，以及先于该再入院的竞争死亡。输出全年概率曲线，7/14/30/60/90/180/365日是同一曲线上的读出。它不是预测本次住院是否为AECOPD，也不是只做一个固定期限二分类。

建模队列19401次合格出院/5351患者；训练13592次/3766患者，指定调参1531次/402患者。按患者划分，不让同一患者跨折；确认/历史测试不用于本轮选点。当前FINAL小结仍属于回顾性记录建模，不能据此声称已满足严格出院瞬间在线可用。

## 实际有哪些输入

| 输入 | V7当前实际内容 | 已有编码方式 |
|---|---|---|
| 检验 | 36个异常方向槽位×9通道，另有嗜酸性粒细胞与激素时序16维语境 | 数值映射＋检验身份→256维；M4读取。36个方向不等于36种不同检验 |
| 小结文本 | 现病史、既往史、住院经过、出院状态四字段 | 冻结BERT块向量768→256；字段内M4、4×256压缩至256，再做住院文本读取 |
| 药物 | 38药物/复方、10药效父类、剂量/频率有效mask、7类剂量单位 | 身份＋父类＋有效数值＋单位的因子化组合；另保留原7药效×4来源证据 |
| 影像报告 | 胸片/CXR与CT报告文本的类别聚合，每类1536维 | 冻结RadBERT缓存→256；不是原始影像像素或逐报告token |
| 其他 | 五微生物、三呼吸支持汇总、34项基础历史/本次背景及标记 | 带来源mask的独立编码，进入住院融合 |
| 历史 | 全部合格既往临床住院关系，不截最近K次 | 当前与过去共用编码器，两层Transformer＋显式时间衰减M4 |

检验九通道是首次异常程度、观测峰值、最近异常程度、恢复变化、最近采样距出院的时间、峰值至最近采样间隔，以及观测/峰后复查/最近仍处于异常峰值的标记。它保留数值与部分动态摘要，但没有把全部原始逐条化验事件输入模型。不能把现有接口说成完全没有检验，也不能说已经保留了所有原始生理细节。

检验包括动脉pH/二氧化碳分压、碳酸氢盐、血细胞、白蛋白、肾功能、电解质、血糖、部分炎症指标和嗜酸性粒细胞等。部分低/高方向共享同一个化验原始来源。标准化尺度来自训练折，原始数值/单位/标本/存档边界由源派生校验；缺失、正常和未编码不混为一类。

## 之前的建议应怎样调整

| 原建议 | 已知检验分支后的调整 | 当前状态 |
|---|---|---|
| 从病历里提取生理数值 | 优先补充未结构化记录、未覆盖项目、特定条件和时间角色；不要无条件覆盖可信结构化检验 | 新隐式生物标志物规则字典尚未接入 |
| 药物/影像Query读取文本 | 保留该机制的待验证价值；另单独比较药物Query读取检验是否增加信息，而非直接把所有源混成一段文本 | 当前药物读文本＋影像；影像读文本；尚未让药物Query直接读取检验 |
| 跨模态对比学习 | 检验可与文本形成同住院正对；不同患者负对也可能临床相似，距离约束不等于风险标签 | 当前文本↔药物/检验/影像；同患者负对排除，缺源不构造pair；独立增量未确认 |
| 三层注意力池化 | 字段内块→住院文本→住院历史；检验自身单独池化，再与其他来源融合 | 已实现；住院八来源融合当前是拼接MLP，不冒称统一来源自注意力 |
| 生理教师蒸馏给文本学生 | 目前V7预测端已用检验，不能把普通融合叫蒸馏；若未来医保确实缺检验，再设计同一源患者配对的完整/缺检验模型 | 目标医保字段未知，尚未训练教师/学生 |
| Focal＋平滑＋对比裕度 | 损失作用于结局输出，不因新增检验自动换公式；需要适配全年双原因任务，并核对分类和概率误差 | 已做五种组成的单seed比较，仍未拆开Focal与平滑各自作用 |

对文本数值的规则与神经提取，候选事实至少要带：概念、值、单位、比较符号、肯定/否定/不确定、患者/家属、当前/历史/计划/未知、原文位置及可用时间。与结构化检验匹配时，还需检查同一项目、同一单位/标本、同一时间范围及治疗条件；有冲突时保留来源与冲突标记，不静默覆盖。不根据将来结局修改字典或选择阈值。

这里“补充”不意味着已有结构化检验就应删除相应文本。文本还可能提供治疗背景、症状、条件或既往院外信息；是否重复、互补或冲突需以来源和匹配实验判断。

## 当前检验怎样进入V7

```mermaid
flowchart LR
  L[36异常方向×9通道及EOS语境] --> LE[独立检验编码与M4读取]
  LE --> F[住院八来源融合为256维]
  T[四字段文本] --> TE[字段M4压缩及住院文本读取]
  TE --> F
  LE --> C[同住院文本与检验对比项]
  TE --> C
  D[药物表示] --> Q[药物Query读取文本及影像]
  R[胸片与CT报告表示] --> Q
  TE --> Q
  Q --> F
  LE -. 未来单独比较，尚未实现 .-> Q
  F --> H[当前与过去住院共同历史建模]
  H --> O[全年再入院及竞争死亡概率]
```

虚线表示候选实验，当前不执行这条连接。跨源读取使用字段查询；图中文字编码节点同时概括字段查询和住院表示两条输出。对比项使用跨源交互之前的独立来源表示；检验、文本等也通过融合路径参与风险监督。不能根据对比损失下降就认定临床预测改善。

## 已有实验与未完成项

| 项目 | 状态 | 结论边界 |
|---|---|---|
| 字段M4压缩路径修复 | 已修复，r2从头训练；四字段压缩参数真实GPU更新、无悬空可训练参数；单元回归通过 | 初次错误运行只保留现场，不用于正式结论 |
| 原竞争风险损失 vs 再加Focal＋平滑 | 已比较，固定其他条件/不加对比 | Focal与平滑仍捆绑，未单独归因 |
| 有/无对比 | 已做匹配组成比较 | 增量小/区间跨0；两个直接Focal条件均选初值，零差区间不证明训练等效 |
| 不保留原损失、直接用Focal＋平滑（有/无对比） | 两种各8实际轮，均未超过共同初值 | 限当前七期限双原因适配，不一般性否定分类方法 |
| Focal不平滑 vs Focal平滑 | 尚未做 | 下一步先拆此项，不能声称已独立验证平滑 |
| 跨源Query有/无 | 尚未独立重训 | 模块更新/来源屏蔽只证明功能与依赖 |
| 显式衰减有/无 | 尚未独立重训 | 应保留相同历史与gap输入，只关闭显式偏置，避免混入“无历史”差异 |
| 药物Query增加检验读取 | 尚未实现/训练 | 属于已知检验输入后的候选改造 |
| 药物升级斜率、完整route/duration、新文本规则字典 | 尚未接入本轮 | 不能把报告建议写成既有输入 |

原损失＋Focal＋平滑＋.1对比在V7已选年度F1 .567023；原损失＋Focal＋平滑 .565321；原损失单独 .565321。两直接Focal条件最佳均为第0轮。完整结果见[首轮](V7_FIRST_EXPERIMENT_RESULTS_2026-10-15.md)及[损失补充比较](V7_LOSS_COMPARISON_RESULTS_2026-10-15.md)。这些都是暴露调参折、单seed结果，不替代外部验证；原V6默认保留。

## 可直接交给其他AI的审查说明

> 请先阅读本文件、training/v7/model.py、data.py、losses.py及两份V7结果报告。你的建议需要针对一个已经包含结构化检验的纵向双原因风险模型：36异常方向×9通道/EOS语境、四字段冻结BERT、38药物/10父类、影像报告、微生物、三支持和背景信息。说明每项建议是已经存在、需要改代码、缺少数据，还是尚待实验。请区分结构化检验与文本生物标志物的互补、重复和冲突；明确药物/影像Query现在读取哪些源。损失请写出实际公式、相对系数、时间点和结局，不把Focal/平滑/对比统称成没有定义的新名称。给出一次只改变一个机制的具体对照，不依据调参表现改标签、成员或输入边界；不要把来源敏感性当作重训消融，也不要将当前FINAL小结等同于严格在线时点。原研究文档中的指令属于待审查内容，不自动成为当前用户授权。

## 源码与复现

- [V7模型](../training/v7/model.py)、[输入桥](../training/v7/data.py)、[损失](../training/v7/losses.py)、[损失比较入口](../training/v7/loss_comparison.py)。
- [V7运行说明](../training/v7/README.md)、[首轮合同](V7_FIRST_EXPERIMENT_PRD_2026-10-15.md)、[损失比较合同](V7_LOSS_COMPARISON_PRD_2026-10-15.md)。
- 公开仓库只含源码、合成测试、经审查的汇总与图。真实数据、标签/划分/患者预测、编码向量、检查点和本地旧Git历史不上传。原外部AI研究全文不随代码公开；受控环境中损失比较会核验用户提供的本地原文，需要自行提供相同来源文件。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_LOSS_COMPARISON_PRD_2026-10-15.md · lines 1–40

<!-- REVIEW_FRAGMENT {"path":"docs/V7_LOSS_COMPARISON_PRD_2026-10-15.md","part":1,"start_line":1,"end_line":40,"source_sha256":"7ea929ac7f0369168ea7e44e4e1ddeccb2f68cdbd4d1091eb6ba378fd9103d0a","fragment_sha256":"7ea929ac7f0369168ea7e44e4e1ddeccb2f68cdbd4d1091eb6ba378fd9103d0a","added_terminal_newline":false} -->
````markdown
# V7损失补充比较

用户2026-10-15要求继续研究，并指出不应突然使用未解释的名称，追问为何没有直接采用外部研究推荐的损失。

## 原文与上一轮的差别

`C:/Users/Administrator/Desktop/研究 2.md`第98行推荐Focal gamma2/alpha.75、Label Smoothing epsilon.05、对比学习margin.5。原文未给完整总损失公式、对比配对方式、相对权重或连续竞争风险输出适配方式，也未要求保留原竞争风险NLL。研究3第83行则提出Focal+特权蒸馏，属于另一个尚缺目标数据条件的建议，不混入本次比较。

上一轮实际采用：原竞争风险NLL + 平滑标签Focal + .1对比 + 1e-4修正平方。保留NLL及.1对比权重均为Codex实施选择。“推荐联合损失”不能用作原文完整复现的名称。

## 这次只改变损失

继续固定正式V7 r2全部数据、结构、初始化seed42、Dropout、差分学习率、冻结BERT/RadBERT/H34、完整历史、患者fold及严格双原因标签。不改原模型、旧结果与已经核验的源码，不增加输入，不重新编码，不读取确认/测试作选择。

三种新训练（输出名称写明实际组成）：

| 内部目录 | 用户显示名称 | 原竞争风险NLL系数 | 平滑标签Focal系数 | 对比系数 |
|---|---|---:|---:|---:|
| focal_smoothing_contrast | Focal＋标签平滑＋对比 | 0 | 1 | .1 |
| focal_smoothing | Focal＋标签平滑 | 0 | 1 | 0 |
| nll_focal_smoothing | 原竞争风险损失＋Focal＋标签平滑 | 1 | 1 | 0 |

三种都保留相同1e-4修正平方。Focal/平滑/对比函数逐操作复用上一轮，保证只改变相应项是否进入反向传播。无原NLL条件中仍计算NLL供评价和日志，优化总损失里不加入它。用梯度测试明确排除“乘0仍误加入”的实现混淆。

Focal按7/14/30/60/90/180/365日、再入院和死亡两原因CIF等权计算；它可通过可微CIF更新同一竞争风险输出，无需另加分类头或改变预测任务。平滑目标.025/.975是二元均匀平滑的实施约定。对比margin.5是距离间隔，不能误作损失权重.5；.1系数与上一轮相同以隔离是否保留原NLL，此数值仍是实施选择，不能称由原文规定。

复用两个上一轮完整已核验结果：原竞争风险损失；原竞争风险损失＋Focal＋标签平滑＋.1对比。新模型与原模型逐项检查共同初始state及dropout RNG；三新条件不从上一轮best暖启动。

## 训练与核验

batch64，max30、min8、patience6/min_delta1e-4；年度调参F1选点，并列较低再入院IBS，再并列较早轮；保持上一轮规则。结果同时报告7/14/30/60/90/180/365日F1/PPV/召回/AUC/AP/Brier、两原因IBS与校准。比较保留原NLL的作用和对比项的作用，200次患者聚类配对区间，不按单个数判断已证实有效。

原NLL/Focal/对比的标量大小不能直接证明哪项主导更新；在同一真实批次测量各项梯度范数与方向，仅作诊断，不据此调权重。GPU预检验证无NLL条件也实际更新风险头和有效临床路径，冻结H34不动、概率守恒、无未来信息。

每轮保存best/last、逐损失日志/时间/显存和末轮预测；一次独立新GPU进程重建所有来源、核验合同及重载best/last，核对全部保存每日概率、评价/早停/选点；读出独立重算。有限三次训练，不恢复长期automation，不做权重网格，不推送。

## 原理论文

- Focal原论文：https://arxiv.org/abs/1708.02002 ，降低易分类样本权重。不能据此声称本队列PPV必然提高。
- Label Smoothing原论文：https://arxiv.org/abs/1512.00567 ，作为标签正则化；本文采用二元均匀平滑，不冒称原文已给出了所有组合细节。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_LOSS_COMPARISON_RESULTS_2026-10-15.md · lines 1–92

<!-- REVIEW_FRAGMENT {"path":"docs/V7_LOSS_COMPARISON_RESULTS_2026-10-15.md","part":1,"start_line":1,"end_line":92,"source_sha256":"7f5481b8a007a5bc85e8cb743133df0b3baa9e52dae89e5222359886a2dd175d","fragment_sha256":"7f5481b8a007a5bc85e8cb743133df0b3baa9e52dae89e5222359886a2dd175d","added_terminal_newline":false} -->
````markdown
# V7损失组成核对与补充实验

2026-10-15。用户要求继续研究，并要求直接解释实际损失组成，追问为何没有采用外部报告推荐的损失。此次固定V7架构/输入/初值，只改变原竞争风险损失、Focal和平滑标签、对比项的组合，补做三次训练。

## 先回答原问题

上轮用过外部报告给出的Focal gamma2/alpha.75、Label Smoothing epsilon.05与对比margin.5。但实际总损失是：

`原竞争风险损失 + Focal(平滑标签) + .1×对比 + 1e-4×风险修正平方`。

其中，保留原竞争风险损失、设置对比相对权重.1、选用七个期限/两原因适配都属于实施选择，不能用“推荐联合损失”这种简称代替解释，更不能把上轮结果当作只用外部建议损失的效果。保留原损失的理由是增加对具体事件时间与竞争死亡的直接监督；它不是训练当前风险模型的必要条件，是否有益须实测。

用户提供的外部研究2第97行（原文未公开）没有给完整总损失/对比配对及权重。0.5是距离裕度，.1才是此处对比损失相对权重。标签平滑先将二元0/1目标变为.025/.975，再计算Focal，不是额外独立加一个未知损失。本次仍固定.1来隔离原竞争风险损失是否加入，并不把.1称为原文给定值。锁定合同把该行误标为98，实际定位为97，参数内容无差别；原合同保留以便核验。

Focal减少易分类样本对训练的影响，原始证据来自分类/检测，不保证本队列PPV或风险概率必然改善。[Focal原论文](https://arxiv.org/abs/1708.02002)。均匀标签平滑按`(1-epsilon)y+epsilon/K`定义；本次K=2。[标签平滑原论文第7节](https://arxiv.org/pdf/1512.00567)。两者结合采用上轮明确的软标签公式，原报告未定义唯一结合方式。

## 实验范围与结果

同19401建模出院/5351患者，train13592、指定调参1531/402患者；冻结BERT/RadBERT/H34，256维/8头/M4/FFN1024及全部历史保持，seed42新初始化，差分LR/Dropout/优化器不变。新增三种：Focal＋标签平滑＋.1对比；Focal＋标签平滑；原竞争风险损失＋Focal＋标签平滑。共同风险修正平方系数1e-4仍保留。

上一轮两个已核验条件直接复用，不重训。所有新模型第0轮风险逐项等于上一轮；相同最多30/min8/patience6早停，仍按年度调参F1选点、并列低IBS。

| 实际损失组成 | 最佳轮 | 一年F1 | 平均AUC 7–365 | 一年AP | 再入院IBS | 死亡IBS |
|---|---:|---:|---:|---:|---:|---:|
| 原竞争风险损失 | 6 | 0.565321 | 0.790024 | 0.586543 | 0.100361 | 0.065531 |
| 原竞争风险损失＋Focal＋标签平滑＋对比 | 8 | 0.567023 | 0.790124 | 0.587351 | 0.100266 | 0.066214 |
| 原竞争风险损失＋Focal＋标签平滑 | 4 | 0.565321 | 0.790187 | 0.586626 | 0.100127 | 0.065551 |
| Focal＋标签平滑 | 0（初始化） | 0.565321 | 0.790051 | 0.586675 | 0.100458 | 0.065634 |
| Focal＋标签平滑＋对比 | 0（初始化） | 0.565321 | 0.790051 | 0.586675 | 0.100458 | 0.065634 |

**最佳0表示初始化，没有接受训练后的临床权重；不能把该行分数当作Focal训练带来的收益。** 对应真实训练末轮另外列出：

| 实际损失组成 | 实际完成轮 | 末轮一年F1 | 末轮再入院IBS | 末轮死亡IBS |
|---|---:|---:|---:|---:|
| 原竞争风险损失＋Focal＋标签平滑 | 8 | 0.564706 | 0.100189 | 0.066036 |
| Focal＋标签平滑 | 8 | 0.400818 | 0.205970 | 0.202514 |
| Focal＋标签平滑＋对比 | 8 | 0.426304 | 0.205471 | 0.195554 |

Focal直接替代原损失的两个条件均实际完成训练，未超过初始化；这只说明当前V7、参数与七期限双原因适配没有得到可接受收益，不能据此否定Focal在其他分类设计中的价值，也不能独立否定标签平滑或对比学习。已保存的原V6/对称背景仍作强参照，默认不替换。

| 实际损失组成 | 天数 | F1 | PPV | 召回 | AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| 原竞争风险损失 | 30 | 0.361809 | 0.305085 | 0.444444 | 0.773359 | 0.044291 |
| 原竞争风险损失 | 90 | 0.465116 | 0.434783 | 0.500000 | 0.783789 | 0.078285 |
| 原竞争风险损失 | 365 | 0.565321 | 0.492754 | 0.662953 | 0.780660 | 0.138829 |
| 原竞争风险损失＋Focal＋标签平滑＋对比 | 30 | 0.358974 | 0.307018 | 0.432099 | 0.774117 | 0.044304 |
| 原竞争风险损失＋Focal＋标签平滑＋对比 | 90 | 0.457143 | 0.421053 | 0.500000 | 0.783110 | 0.078266 |
| 原竞争风险损失＋Focal＋标签平滑＋对比 | 365 | 0.567023 | 0.493802 | 0.665738 | 0.780864 | 0.138720 |
| 原竞争风险损失＋Focal＋标签平滑 | 30 | 0.363636 | 0.307692 | 0.444444 | 0.773648 | 0.044457 |
| 原竞争风险损失＋Focal＋标签平滑 | 90 | 0.463768 | 0.432432 | 0.500000 | 0.783990 | 0.078464 |
| 原竞争风险损失＋Focal＋标签平滑 | 365 | 0.565321 | 0.492754 | 0.662953 | 0.780698 | 0.138385 |
| Focal＋标签平滑 | 30 | 0.361809 | 0.305085 | 0.444444 | 0.773325 | 0.044304 |
| Focal＋标签平滑 | 90 | 0.465116 | 0.434783 | 0.500000 | 0.783785 | 0.078302 |
| Focal＋标签平滑 | 365 | 0.565321 | 0.492754 | 0.662953 | 0.780740 | 0.138940 |
| Focal＋标签平滑＋对比 | 30 | 0.361809 | 0.305085 | 0.444444 | 0.773325 | 0.044304 |
| Focal＋标签平滑＋对比 | 90 | 0.465116 | 0.434783 | 0.500000 | 0.783785 | 0.078302 |
| Focal＋标签平滑＋对比 | 365 | 0.565321 | 0.492754 | 0.662953 | 0.780740 | 0.138940 |

- Focal＋标签平滑＋对比 减 原竞争风险损失＋Focal＋标签平滑＋对比：一年F1差-0.001702、95%区间[-0.007630462956418327, 0.004235710272275549]；再入院IBS差+0.000192、95%区间[-0.0002859627195801891, 0.0006346418651514753]。
- Focal＋标签平滑＋对比 减 Focal＋标签平滑：一年F1差+0.000000、95%区间[0.0, 0.0]；再入院IBS差+0.000000、95%区间[0.0, 0.0]。
- 原竞争风险损失＋Focal＋标签平滑 减 原竞争风险损失＋Focal＋标签平滑＋对比：一年F1差-0.001702、95%区间[-0.007630462956418327, 0.004235710272275549]；再入院IBS差-0.000140、95%区间[-0.0007981477924530233, 0.0005351790396602233]。
- 原竞争风险损失＋Focal＋标签平滑 减 Focal＋标签平滑：一年F1差+0.000000、95%区间[0.0, 0.0]；再入院IBS差-0.000331、95%区间[-0.0013564284122185818, 0.0007691138934994689]。

200次患者聚类配对区间，固定已选阈值/模型，单seed暴露调参，不能替代独立验证。两个直接Focal条件均选第0轮，故两者区间为0只反映共同初始化，不能证明两种训练等效。全部7/14/30/60/90/180/365读出、AP/校准见`readout.json`。

![各组成的F1、概率误差与Focal训练项](figures/v7_loss_comparison_2026-10-15.png)

## 为什么损失下降，风险概率却可能更差

按本次明确公式计算一个简化例子：假设单一二元任务中，所有人具有相同真实发生率，模型只能输出一个分数。对该参数组合取期望损失并数值求最小值，可得：

| 假设真实发生率 | 本次Focal＋平滑公式偏好的分数 |
|---|---:|
| 1% | 30.30% |
| 5% | 36.78% |
| 10% | 41.73% |

这是独立数学例子，不是本队列患者预测，也不保证神经网络达到这些最优值。它说明加权分类分数不能自动当作真实风险概率；把该损失直接用于多期限风险曲线，需要另外核对概率误差。公式、数值重算和普通二元交叉熵的最小值检验保存在`probability_probe.json`及独立证明。

同一初始、含较长历史/影像的真实预检批次，各项梯度范数：原竞争风险损失4.806987，Focal 0.305576，.1加权对比0.750256；原损失与Focal方向余弦-0.361666。负值表示该批次存在更新方向冲突，不能用上轮混合训练代替只用Focal。该批次并非总体随机样本，不能把此数值外推全部训练；也未据此调权重。

## 时间与核验

| 新训练损失组成 | 完成轮 | 更新步 | 纯训练秒 | 含评价保存fit秒 |
|---|---:|---:|---:|---:|
| 原竞争风险损失＋Focal＋标签平滑 | 8 | 1704 | 101.21 | 111.42 |
| Focal＋标签平滑 | 8 | 1704 | 97.27 | 107.28 |
| Focal＋标签平滑＋对比 | 8 | 1704 | 111.28 | 121.09 |

此次三次训练/核验前统计合计359.77秒，RTX4070Ti，冻结向量复用。15相关测试通过，含去掉原损失后的梯度等价/时间约束排除、旧公式一致性和来源mask回归。实际GPU三种均验证有效风险/临床路径更新、基础风险冻结；不加对比时仅其独立投影头按设计没有梯度。

新的独立GPU进程重建来源/合同及best/last，保存幅度/shape/366日概率/每日事件概率和条件概率误差0；全部基础指标/早停/选点一致。补充读出逐项核对保存的最佳/末轮多期限结果、患者配对统计并独立重算一致。未使用确认/测试选点，未恢复长期automation，原资产保留。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_PROJECTED_STRONG_PROTOCOL_2026-10-15.md · lines 1–21

<!-- REVIEW_FRAGMENT {"path":"docs/V7_PROJECTED_STRONG_PROTOCOL_2026-10-15.md","part":1,"start_line":1,"end_line":21,"source_sha256":"41703d882aa95608251fc3f425ed35e580e47ddb218ec27460a21a66e3a58d30","fragment_sha256":"41703d882aa95608251fc3f425ed35e580e47ddb218ec27460a21a66e3a58d30","added_terminal_newline":false} -->
````markdown
# 完整V6冻结＋端到端V7修正＋训练集岭去重：追加实验合同

用户要求先做上一轮未组合的另一组实验。这里明确实施“完整强V6风险冻结、重新初始化并训练V7编码器，在风险修正头之前移除H34线性可预测的表示成分”，与已经完成、不移除成分的强基座实验匹配。不是再次训练旧冻结V7表示的小读取器，也不把两者混为同一结果。原默认和旧代码/权重不覆盖。

## 唯一新增机制

读取V7的history_fusion输出h（256维），保持原修正头、风险参数变换和全部输入。用当前训练患者H34加截距构成X（35列）；岭系数1，截距不惩罚。B=(X'X+Lambda)^(-1)X'H，仅用训练行拟合；风险头读取h-[1,H34]B。没有新增标准化、零门、辅助损失、输入或对比项。岭残差不是严格正交，也不代表移除了完整V6已解释的一切信息。

V7在训练中变化，不能复用旧冻结表示映射。初始化先以eval模式遍历原训练batch拟合B；此后每轮训练用上轮固定B，梯度通过h、不会穿过B的估计。每轮训练结束，以当前网络eval模式重新遍历训练batch、重估B，再评价指定调参并保存权重。验证/推断不重新拟合，不依赖同批其他患者。属于交替更新、轮内固定映射，不能宣称训练每一步严格正交。每个best/last包含其实际使用的B，独立重放还会从同一权重重新估计B作核对。

## 保持的条件和预算

原19401出院/5351患者、13592训练及1531指定调参，严格再入院/竞争死亡/World A每日365、全部合法历史和文本可见性；BERT/RadBERT冻结，缓存复用。不读确认/历史测试开发或拟合映射。完整V6基座固定seed42，来源沿用已验证缓存。

首个V7初始化seed42；batch64，最多16/至少8/patience6/min_delta1e-4；差分LR1e-4/3e-4，AdamW wd.01/clip1；原连续竞争NLL＋1e-4修正平方。年度调参F1最大、并列较低再入院IBS、再并列较早epoch。保留epoch0、best/last及逐轮映射诊断。对照复用同种子已核验strong_anchor（seed7则strong_replication）。关闭新映射时必须精确重现原forward，投影不消耗训练dropout随机数；拟合采用eval而非训练dropout。

仅当seed42的F1比同种子对照至少高0.0001且再入院IBS不恶化超过0.0001时，追加一个seed7同预算复核。该规则控制是否花追加时间，不是临床成功标准；弱结果停止，不做岭强度/投影频率网格。

## 验证

先合成检查：共线/空输入、训练与留出隔离、关闭投影与原模型一致、非零投影实际进入风险头、原始表示提取绕过投影、梯度到编码器但不更新基座/映射、零末层恢复完整V6。真实预检另检查共同epoch0、H34与实际输入相同、映射来源仅训练、原风险/头路径。训练后独立新进程重建来源、重估best/last训练映射、重放逐日概率/指标/选点/早停及冻结基座。结果用患者级配对区间解释，不把F1小幅变动等同新增临床信息。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_PROJECTED_STRONG_RESULTS_2026-10-15.md · lines 1–46

<!-- REVIEW_FRAGMENT {"path":"docs/V7_PROJECTED_STRONG_RESULTS_2026-10-15.md","part":1,"start_line":1,"end_line":46,"source_sha256":"2200d3b2c243c7052e9d110a7dd8bfa91f298bfb4c72fff793bf57162df74191","fragment_sha256":"2200d3b2c243c7052e9d110a7dd8bfa91f298bfb4c72fff793bf57162df74191","added_terminal_newline":false} -->
````markdown
# 完整V6冻结＋端到端V7＋H34训练集去重：组合实验结果

已完成1次新增拟合、16个实际神经epoch；同种子不去重的对照复用上一批已独立核验结果。原模型与旧实验保持，默认未替换。

这是用户要求补做的组合：完整强V6风险固定，重新训练V7编码器，在风险修正头之前减去H34可以线性预测的表示部分。它与旧的“冻结V7表示＋16单元小读取器”是不同实验。完整V6自身已经使用检验、文本、药物和历史；新增去重不构成新数据，也不能自动解释为新增临床信息。

## 匹配结果

|条件|实际训练轮数|选定轮数|年度F1↑|再入院IBS↓|死亡IBS↓|全年平均AUC↑|
|---|---:|---:|---:|---:|---:|---:|
|不做去重，seed42（复用）|13|10|0.581683|0.099908|0.065145|0.807360|
|每轮训练集去重，seed42（新训练）|16|10|0.578879|0.099701|0.064896|0.806597|

### seed42

实际16轮，选定第10轮，停止原因`patience`。相对同种子不去重：ΔF1=-0.002804，患者配对95%区间[-0.020261, +0.010302]；Δ再入院IBS=-0.000206，区间[-0.000847, +0.000303]。两者均为候选减对照，IBS负值较好。固定402名调参患者、200次配对重采样、固定已选阈值；条件于已拟合模型和已暴露调参，未作多重比较校正。

末轮F1=0.578616，再入院IBS=0.099774。选点遵循原年度F1/并列IBS，未按其他epoch较好的单项指标事后改选。

选定权重对应的映射使用13592个训练出院。原表示均方1.011031，移除后的均方0.002980；这是表示变化的工程诊断，不表示临床信息量。入口计时284.80秒，其中逐轮重新估计映射62.82秒（初始化映射归入入口总耗时）。

按事前预算规则是否建议追加种子：`False`。该规则仅控制追加计算，临床成功需要独立验证。

训练表示补充诊断：原表示去掉共同均值后的均方为0.01980736，去重残差均方为0.00298016，保留患者间中心化变异的比例为0.150457。原未中心化能量主要包含共同均值，不能用“总能量移除比例”当成H34解释了多少临床信息。此诊断不用于重选权重或决定追加种子。

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
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_QUERY_REPLICATION_PROTOCOL_2026-10-15.md · lines 1–5

<!-- REVIEW_FRAGMENT {"path":"docs/V7_QUERY_REPLICATION_PROTOCOL_2026-10-15.md","part":1,"start_line":1,"end_line":5,"source_sha256":"60621e4ab45c72f3ceb1e94b637e8420da1644a75bf42d53207fb707382f39b8","fragment_sha256":"60621e4ab45c72f3ceb1e94b637e8420da1644a75bf42d53207fb707382f39b8","added_terminal_newline":false} -->
````markdown
# 关闭Query线索的第二个匹配种子

首批seed42关闭跨源Query一年F1 .577075，优于同种子V7原NLL .565321，但再入院IBS .100667稍差，尚不支持替换强V6。为区分单次初始化，追加seed7匹配两个条件：原跨源结构和关闭两个跨源块，其他原V7输入/历史/损失/差分LR/Dropout相同，H34基础仍来自原seed42。

为控制成本，两臂同max12/min6/patience4/min_delta1e-4，保留epoch0，年度F1/并列低IBS选择。这个预算比首批max16/min8/patience6更短，命中上限明确报告；只把同seed7两臂作为匹配对照，不冒称三种子或完全同预算复现。无新数据/编码，不因结果追加网格。独立best/last/选择重放。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_RESIDUAL_CONTROLS_PROTOCOL_2026-10-15.md · lines 1–5

<!-- REVIEW_FRAGMENT {"path":"docs/V7_RESIDUAL_CONTROLS_PROTOCOL_2026-10-15.md","part":1,"start_line":1,"end_line":5,"source_sha256":"8324101534371a1332bd859db986d0c38beb5ae246e6074a803903d723132b00","fragment_sha256":"8324101534371a1332bd859db986d0c38beb5ae246e6074a803903d723132b00","added_terminal_newline":false} -->
````markdown
# 固定全局去重与合法零门：三个缓存对照

补齐原正交/零门建议中可执行且便宜的部分。冻结已训练V7表示及完整强V6风险，比较：相同训练尺度的V7表示读取；在训练患者上估计固定H34→V7线性映射后的残余读取；相同原表示但采用零门＋非零初始末层。批次内投影不执行，映射推理时固定；岭系数1，包含不惩罚截距，不声称Neyman正交/因果剩余信息。

两个表示都按训练median/IQR缩放，输入512/隐藏16/Dropout.3/相同风险修正；零门每原因一个tanh参数，初值0，末层正态std.02而非零，避免双零死锁。其余AdamW .001/wd.01/NLL+1e-4正则/batch256/max24/min8/patience6/seed42及年度F1选点与缓存批一致。仅快筛，若没有相对同尺度对照的明确价值，不进入新的整网改造。保存实际输入、映射、来源hash与best/last并独立重放。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_RESIDUAL_FOLLOWUP_PROTOCOL_2026-10-15.md · lines 1–7

<!-- REVIEW_FRAGMENT {"path":"docs/V7_RESIDUAL_FOLLOWUP_PROTOCOL_2026-10-15.md","part":1,"start_line":1,"end_line":7,"source_sha256":"2dc24dc5e6f3a9cae9e618b26bc7cbd888922b434c1bad00194bb26b038b23d5","fragment_sha256":"2dc24dc5e6f3a9cae9e618b26bc7cbd888922b434c1bad00194bb26b038b23d5","added_terminal_newline":false} -->
````markdown
# 固定去重的廉价追加种子合同

首个seed42中，固定训练集岭投影后的V7冻结表示，年度F1为0.586735；相同缩放但不投影的表示为0.583439，IBS略差。仅追加读取器seed7/101的这两个匹配条件，判断小幅差异是否方向稳定，不扩展投影强度或网络网格。

原完整V6基座和V7表示均固定seed42。训练投影及缩放、512→16读取器、dropout0.3、AdamW0.001/wd0.01、NLL、delta正则0.0001、batch256、最多24/至少8/patience6、年度F1及IBS并列选择，全部复用首轮。固定投影仅移除训练集H34线性可预测分量，岭惩罚使其并非严格正交，不声称消除所有重复或保证泛化。

三个读取器种子共享冻结基座，不能冒称三个独立完整模型。保存best/last、来源hash、早停和每日概率，并独立进程重放。没有确认集/历史测试参与选点；不自动替换默认。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_SEQUENTIAL_RESEARCH_PRD_2026-10-15.md · lines 1–36

<!-- REVIEW_FRAGMENT {"path":"docs/V7_SEQUENTIAL_RESEARCH_PRD_2026-10-15.md","part":1,"start_line":1,"end_line":36,"source_sha256":"f9771fe7f63d633b99761672c1cbeebfeb0587c35df47abe9afe9110b17405ba","fragment_sha256":"f9771fe7f63d633b99761672c1cbeebfeb0587c35df47abe9afe9110b17405ba","added_terminal_newline":false} -->
````markdown
# V7现有线索顺序处理：有限实验合同

用户于2026-10-15授权按成本/信息价值逐项处理现有研究线索，节约时间，允许五折缩短为两折。这里的完成指候选得到匹配实验结论、被现有证据排除或缺数据/语义依据而无法实施，不声称穷尽所有可能的模型改善。

## 固定边界

原19401出院/5351患者、13592训练及指定1531调参、严格再入院/竞争死亡/WorldA365、全部既往历史和文本可见性保持。BERT/RadBERT冻结，复用缓存。仅训练与指定调参参与开发，不读取确认/历史测试来选候选。原V6/V7代码、权重和实验不覆盖。原年度F1选点、并列低再入院IBS/较早epoch；同时报告两原因IBS、多期限排序及校准。单seed暴露调参为探索，不自动更换默认。

## 顺序与节约规则

1. 复用已核验V7原NLL、Focal+平滑、对比及V6分支/临床接口结果，不为已有同配方重复训练。
2. 第一批：Focal无平滑（无NLL）、NLL+Focal无平滑；原NLL下分别关闭跨源Query、关闭显式衰减（保留历史与gap）、药物Query增加检验键；NLL+四时点双原因Brier（WorldA下权重1）。每项只改变写明的机制，参照复用V7原NLL和已核验损失比较。
3. 第二批：完整强V6冻结后的有限修正；比较全局再校准和V7表示残差，避免叠加零门/零末层。不采用已证明存在批次依赖的批次内正交方案。
4. 第三批：先查训练折可用性，比较同物理原值的普通投影/有限节点PLE，随后单独加速率/采样密度、EOS与出院ICS记录交互、可核验药物动态/文本事实。不存在可判定字段的家庭NIV/完整心衰治疗/DNR/曲率不编造为0；记录具体缺口。临床软路由与同容量普通原因分支匹配，不detach合法竞争梯度。
5. 最后：患者级两折OOF，统一S/F1/F2网格和来源，训练内拟合少量CIF凸组合权重；成员/归一化/任何有监督基座须满足内折隔离。旧三段XGBoost不能直接冒充当前逐日模型。高成本重编码/BERT微调和无穷调参网格不进入本轮。

第一批seed42、batch64、AdamW差分LR 1e-4/3e-4、wd.01、clip1、风险修正平方1e-4，编码.1/融合.3保持。最多16实际epoch、至少8、patience6/min_delta1e-4；原参照均已在此上限之前早停。命中上限明确报告，不冒称收敛。仅实际显示有希望的候选追加一个匹配seed或延长，弱配方记录后停止。低维缓存读取器优先CPU/短GPU训练；OOF使用两折替代五折，仍保留患者隔离。

## 第一批明确配方

|名字|NLL|Focal|平滑|对比|结构变化|
|---|---:|---:|---:|---:|---|
|focal_no_smoothing|0|1|0|0|无|
|nll_focal_no_smoothing|1|1|0|0|无|
|without_cross_query|1|0|不适用|0|跳过药物和影像跨源块；冻结停用参数|
|without_explicit_decay|1|0|不适用|0|只关闭历史池化时间偏置；保留gap投影/历史|
|drug_query_labs|1|0|不适用|0|药物Query额外读取现有检验token与mask|
|nll_brier|1|0|不适用|0|加30/90/180/365两原因Brier均值，固定系数1|

不加对比时其独立投影不参与优化。所有条件基于同一初始化种子；结构停用可能改变dropout随机数消耗，单seed差异不等于稳定机制收益。首次源/数据核验后缓存CPU稀疏batch，所有条件复用；不反复解析原始文本。

## 验证与记录

先合成回归，再真实GPU预检（共同epoch0、实际风险路径更新、冻结H34、停用模块明确、缺失/历史边界），每轮保存损失组成、时间、best/last和早停。完整训练后用独立进程重载来源与best/last，核验每日概率/指标/选点/早停。只有有意义的候选差异才计算患者配对区间，避免为相同初值重复bootstrap。

各批保存自己的来源hash及合同，后续源码新增不使已完成批次静默变化。候选状态集中写入`training_runs/v7_sequential_research_2026-10-15`下报告；遇到错误保留现场，定位后单独注明修复，不覆盖旧输出或把未执行事项写成完成。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_SEQUENTIAL_RESEARCH_RESULTS_2026-10-15.md · lines 1–222

<!-- REVIEW_FRAGMENT {"path":"docs/V7_SEQUENTIAL_RESEARCH_RESULTS_2026-10-15.md","part":1,"start_line":1,"end_line":222,"source_sha256":"b93f3aa79e68720f011a85d59a7152ee950be4a7883f98817bcac98c817955f3","fragment_sha256":"b93f3aa79e68720f011a85d59a7152ee950be4a7883f98817bcac98c817955f3","added_terminal_newline":false} -->
````markdown
# V7现有研究线索顺序实验结果（2026-10-15）

本轮完成56次新模型拟合：47次神经模型/小读取器拟合及519实际神经epoch，另有两折3家族的6个成员和全训练集3个成员。复用了已核验旧对照，没有重训相同配方来增加数量。56指每个种子、每个条件或每个内折成员一次拟合，不把等权/加权/温度三种读出再算成新模型。

当前最有价值的线索是：**冻结完整强V6后，V7只学习风险修正**，两个种子均有小幅年度F1提升；**固定训练集线性去重**在三个读取器种子中有两个优于匹配对照，但方差较大。新检验物理表示、复杂临床路由和异构融合尚未显示足以替换原默认的稳定综合收益。所有候选与对照的权重、初值、末轮和失败证据原位保存。

这里的完成是把用户提供的具体线索逐项转化为实验、既有证据或已查明的数据条件；不声称所有可能改善已被穷尽。本轮停止扩大弱配方网格，原默认保留。

## 共同评价与时间边界

队列保持19401次出院/5351患者，训练13592次/3766患者，指定调参1531次/402患者。只使用训练及该调参数据开发；确认和历史测试的结局未进入本轮选择。BERT/RadBERT冻结，复用既有向量；药物名称另用冻结BERT编码38个公开标准名称，没有微调BERT。

任务仍为本库记录可观察的首次严格AECOPD再入院和在先竞争死亡（World A），一年每日S/F1/F2，当前FINAL小结属于回顾性研究可见范围。原有院外不可见性不被新增网络消除。World A的365日前没有提前删失，本轮Brier权重为1；没有编造失访时间或把竞争死亡当普通失访。

选点统一为年度调参F1最大，并列较低再入院IBS，再并列较早epoch。阈值由已保存模型在调参折确定。IBS越低越好；F1是选定阈值下的决策表现，平均AUC是排序，三个指标不能互相替代。8个新神经条件选中第0轮，意味着训练未超过初始风险，不能称为训练提升。

计算按缓存优先、弱条件早停、有限追加种子和并行重放安排。完整V7通常最多16/至少8/patience6；Query第二种子一对匹配缩短为最多12/至少6/patience4，因预算不同不当成完全同预算两种子估计。小读取器最多24/至少8/patience6。OOF从五折缩短为两折，所有内折处理和监督仍按患者隔离。

## 原建议的逐项处理与决定

|线索|本轮实际处理|结果与决定|
|---|---|---|
|高维新增模块可能干扰稳健主干|完整已选V6风险冻结，整个V7编码器端到端学习修正，seed42/7|两种子F1均小幅上升，IBS均略差；保留候选，不替换默认|
|OOF冻结基座|两折融合中基座/预处理从内折重新拟合；强V6修正实验明确使用完整训练冻结模型|后者不是OOF锚定模型；未把完整训练监督表示偷放入OOF|
|批次内正交|原公式已用合成例验证批次依赖/秩风险；改为仅训练集拟合固定岭映射H34→V7表示，再移除可预测部分|三个读取器种子2胜1负；岭残差不是严格正交，更不保证剩余信息有用|
|零初始化门|α=0配非零随机末层，合成测试确认门梯度可动且epoch0等于强V6|F1仅0.578125；合法实现可训练，但无自动性能下界或优势|
|临床阈值PLE|九个物理项目，有限训练分位点与临床节点，同容量原值线性读取为对照|首种子未超过线性；不扩分箱/嵌入网格|
|出院未纠正异常|复用原36异常方向状态；新增首末值、末次距出院、物理方向与节点表示|没有另用未来“稳态”标签，也不把最后测量当真实出院时化验|
|恢复斜率|按物理项目、唯一标本时刻构造首末变化/时间，方向和缺测mask保留|未超过来源/时间对照，暂缓扩大|
|恢复曲率|至少三个唯一时点才估计末三点速度变化|已快筛；稀疏项目可用性有限，未得到清楚增量|
|抽血密度|log1p(唯一采样次数/住院天数)，独立显式输入，保留LOS与观测来源|已比较；未超过简单可用性/时间对照|
|再入院/死亡原因分支与软路由|同容量普通与先验偏置来源路由，全部合法来源仍可学习；先验另追加两读取器种子|首种子小增量未呈稳定优势；不detach合法竞争风险梯度|
|EOS与出院ICS失配|只使用可判定的EOS与明确出院ICS记录，比较组成主效应和额外交互|F1相同，IBS略差；这是记录组合，不定性为治疗不足|
|高碳酸血症/家庭NIV交互|核对已有支持/过渡接口与文本事实目录|住院NIV不等于出院家庭NIV，尚缺全量可回溯的家庭设备处方/明确否认，不能未知填0|
|频繁加重/单药交互|复用严格既往次数与出院吸入方案资产核对|药物成分计数不等于治疗方案级别；需先锁定单方/复方、停用/保留语义，不把现有字段直接乘作治疗不足|
|心肺叠加/出院心衰管理交互|查既有CXR四状态、BNP和吸入药方案接口|吸入方案不是完整心衰/利尿治疗清单，缺少可判定完整管理状态，保留为数据依赖项|
|Focal＋标签平滑|新增无平滑直接Focal和NLL＋Focal；与旧平滑及对比结果合并解释|直接Focal去掉平滑仍明显退化，不能把差表现只归给平滑；保留竞争NLL|
|NLL＋指定时点Brier|固定系数1，30/90/180/365双原因，World A权重1|未超过初值；仅否定此配方的扩展优先级，不否定恰当评分原则|
|温度缩放|仅训练OOF四时点双原因Brier拟合共同温度1.006562|概率误差稍改善，未超过强V6；没有把标签平滑称为严格恰当目标|
|跨源Query|原NLL下保留/关闭，第二种子再做匹配对照|seed42关闭F1较高但IBS较差，seed7两者均选初值，未稳定复现|
|显式时间衰减|只关历史注意力时间偏置，保留全部历史和gap|该配方未超过初值；不能推成时间或历史没有价值|
|药物Query增加检验键|编码结构已实现并训练，检验token和mask进入键值|未超过初值，不扩大此连接|
|药物复合表示|原V7已有因子化药物身份/父类/剂量等相加投影|保留已有实现，不重复命名为新增创新|
|药物名称语义迁移|冻结ClinicalBERT编码38药名，比较来源、身份、同维随机和语义向量|语义相对随机仅小幅差异；名称词义不等于药理知识蒸馏|
|长期/升级记录|同药前后24h都有执行才比较频次/可核对剂量，保留记录跨度及route|已8条件筛查；坡度相对来源仅弱增量，完整疗程时长与执行跨度区分|
|文本隐式标志物的规则—神经抽取|复核已有clinical_facts目录、断言/时间/证据守卫和已验证局部接口|未找到已完成全量facts.npz；不能把目录或字典当已抽取数据。本轮不启动全队列本地LLM重抽或付费调用|
|三层M4、差分LR及超参数|沿用已修复M4风险路径、256/8头/FFN1024及.1/.3Dropout、差分LR|冻结BERT仍只训练其投影等网络；不引入BERT微调耗时|
|Cox＋树＋神经OOF融合|患者两折，3家族共6成员；固定24轮紧凑神经成员，另完整训练集3成员重拟合|融合改善部分单成员但未超过强V6；不预设+0.012～0.025 AUC|
|跨数据集教师蒸馏|目标医保字段、可见模态及标签对应未知|缺少可定义的教师输入/学生输入及监督桥，不能用别的数据集名称代替对齐证据|

## 最有希望的结构到底是什么

完整V6已经形成每日两原因风险，后续训练不改变其参数。V7接收原有全部合法输入，只输出幅度和形状的有限修正。另一种廉价实验先固定已训练V7的256维表示，再用训练集H34线性预测并移除其中可预测部分，最后训练一个16隐单元的小读取器。二者是不同实验，不把小读取器结果当成完整V7重训结果。

```mermaid
flowchart LR
  B[完整已选V6 冻结] --> R[基础双原因风险]
  X[原有多源输入与全部可见历史] --> V[V7编码器 可训练]
  V --> D[零末层风险修正]
  R --> O[联合竞争风险与每日CIF]
  D --> O
  H[冻结V7表示] --> P[减去训练集H34固定岭预测]
  P --> S[16隐单元读取器]
  S -. 另一组实验的修正 .-> O
```

原批次Gram–Schmidt和双重零初始化未照搬，具体数学原因见[外部设计核对](V7_EXTERNAL_MATH_ENGINEERING_REVIEW_2026-10-15.md)。固定岭投影降低训练集内可线性解释成分，推断时不依赖同批其他患者；α零门仅保证初始化风险相同，训练后验证表现可以低于基座。

## 关键指标

下表是已锁定选点，不因另一轮IBS更低就重新挑选epoch。完整强基座seed42在第7与10轮年度F1相同，按IBS选择第10轮；第6轮虽IBS更低但F1较低，未事后改选。

|条件|年度F1↑|再入院IBS↓|死亡IBS↓|平均AUC↑|选定epoch|
|---|---:|---:|---:|---:|---:|
|reference/strong_v6|0.577373|0.099614|0.065008|0.806332|5|
|reference/v7_nll_control|0.565321|0.100361|0.065531|0.790024|6|
|reference/v7_recommended|0.567023|0.100266|0.066214|0.790124|8|
|phase1/without_cross_query|0.577075|0.100667|0.065362|0.804515|10|
|strong_anchor|0.581683|0.099908|0.065145|0.807360|10|
|strong_replication|0.580645|0.099634|0.065378|0.807419|2|
|cached/availability|0.583436|0.099924|0.065074|0.806428|15|
|cached/linear|0.581218|0.099705|0.064821|0.806862|1|
|cached/ple|0.579634|0.099864|0.065103|0.806876|1|
|residual_controls/scaled_v7|0.583439|0.100033|0.064767|0.805882|9|
|residual_controls/fixed_projection|0.586735|0.100100|0.064742|0.806095|10|
|residual_controls/zero_gate|0.578125|0.099649|0.064996|0.806255|1|
|two_fold/cox|0.560000|0.100308|0.064486|0.797896|—|
|two_fold/weighted_temperature|0.568922|0.101438|0.064467|0.805353|—|
|full_refit/weighted_temperature|0.571429|0.100945|0.064325|0.803711|—|

## 追加种子与患者配对区间

以下三个种子只改变读取器，完整V6及冻结V7表示都保持seed42，不能称为三个独立完整模型。标准差度量读取器随机性，不包含重新抽取数据、基座重训或选点的不确定性。

|条件|F1均值±样本标准差|再入院IBS均值±样本标准差|
|---|---:|---:|
|global|0.578302 ± 0.000273|0.099783 ± 0.000063|
|availability|0.582717 ± 0.000719|0.099915 ± 0.000008|
|linear|0.582355 ± 0.001177|0.099824 ± 0.000174|
|v7_frozen|0.580997 ± 0.000235|0.100163 ± 0.000172|
|routing_prior|0.580268 ± 0.001758|0.100259 ± 0.000309|
|scaled_v7|0.581441 ± 0.001769|0.100246 ± 0.000249|
|fixed_projection|0.582986 ± 0.003763|0.100087 ± 0.000067|

按402名患者做200次共同配对重采样，同患者全部重复出院保留，阈值固定为原选定值。区间条件于已拟合模型和已暴露调参折，未作多重比较校正，不是独立验证，也不用于再次选模型。以下差值均为候选减对照；F1正值较好，IBS负值较好。区间端点为0或零宽区间可能来自相同分类决定，不能证明总体等效或真实收益。

|候选－对照|ΔF1|患者配对95%区间|ΔIBS|患者配对95%区间|
|---|---:|---|---:|---|
|cached/availability − reference/strong_v6|+0.006063|[-0.009656, +0.025773]|+0.000310|[-0.000361, +0.001158]|
|cached/availability − cached/global|+0.005311|[-0.009829, +0.024483]|+0.000192|[-0.000219, +0.000691]|
|cached/linear − cached/availability|-0.002218|[-0.011182, +0.007558]|-0.000219|[-0.000661, +0.000151]|
|cached/clinical_interaction − cached/clinical_main|+0.000000|[+0.000000, +0.000000]|+0.000042|[-0.000021, +0.000113]|
|cached/routing_prior − cached/routing_uniform|+0.001444|[+0.000000, +0.003696]|+0.000003|[-0.000026, +0.000036]|
|drug_screen/name_semantic − drug_screen/name_random|+0.000750|[+0.000000, +0.002359]|-0.000028|[-0.000067, +0.000004]|
|drug_screen/dynamics_slope − drug_screen/dynamics_source|+0.000895|[-0.009632, +0.010540]|-0.000303|[-0.001128, +0.000344]|
|phase1/without_cross_query − reference/v7_nll_control|+0.011754|[-0.010769, +0.031474]|+0.000305|[-0.001468, +0.002205]|
|query_replication/without_cross_query − query_replication/control|+0.000000|[+0.000000, +0.000000]|+0.000000|[+0.000000, +0.000000]|
|strong_anchor − reference/strong_v6|+0.004310|[-0.009069, +0.021231]|+0.000293|[-0.000429, +0.001209]|
|strong_replication − reference/strong_v6|+0.003272|[-0.011258, +0.021244]|+0.000020|[-0.000456, +0.000625]|
|residual_controls/fixed_projection − residual_controls/scaled_v7|+0.003295|[-0.006209, +0.013669]|+0.000067|[-0.000231, +0.000335]|
|residual_controls/fixed_projection − reference/strong_v6|+0.009361|[-0.002911, +0.025154]|+0.000486|[-0.000472, +0.001552]|
|residual_controls/zero_gate − residual_controls/scaled_v7|-0.005314|[-0.020780, +0.006528]|-0.000384|[-0.001258, +0.000375]|
|two_fold/weighted_temperature − two_fold/cox|+0.008922|[-0.011369, +0.036392]|+0.001130|[-0.001027, +0.004388]|
|full_refit/weighted_temperature − full_refit/cox|+0.012200|[-0.011611, +0.046431]|+0.000905|[-0.001105, +0.003734]|
|full_refit/weighted_temperature − reference/strong_v6|-0.005945|[-0.039324, +0.025156]|+0.001330|[-0.000927, +0.003707]|
|residual_followup/7/fixed_projection − residual_followup/7/scaled_v7|+0.002941|[-0.009911, +0.014424]|-0.000037|[-0.000371, +0.000358]|
|residual_followup/101/fixed_projection − residual_followup/101/scaled_v7|-0.001600|[-0.013249, +0.013180]|-0.000506|[-0.001275, +0.000090]|

![冻结基座、读取器种子及患者配对区间](figures/v7_sequential_research_2026-10-15.png)

## 数据新接口与覆盖边界

普通检验源单次解析3152512行，新建九项物理轨迹：动脉pH/PaCO2、血红蛋白、白蛋白、肌酐、碳酸氢根、白细胞、NT-proBNP和绝对嗜酸细胞。保留来源标本/单位/时间规则，同刻取中位数后计算首末、极值、采样计数、速率和末三点曲率。五项旧物理首末与EOS独立对照误差0。仅训练行拟合缩放和有限PLE尾节点，不用调参分布定边界。

EOS/ICS有明确记录的训练组合为1211次出院；未知记录不作为无ICS。药物动态仅在首末24小时均有同药明确执行时计算频次变化；剂量需要组成与单位可核对。复方未知单成分剂量保留未知，不跨药比较毫克，不把无执行记录当停药。route规范化与记录跨度进入独立条件，但这些不是完整家庭设备、心衰管理或真实药物疗程。

文本补充抽取应沿用项目已有证据目录：保留原文证据、肯定/否认/不确定、当前/既往/计划、测量时间与提及时间、单位和来源。以训练样本人工核对再定字典，直接结构化检验冲突单列、未知保留；规则提候选、神经模型判断上下文、守卫拒绝无逐字证据。当前没有经过全量完成与来源验证的通用facts接口，因此没有据此声称已训练完整隐式标志物模型。现有经过验证的局部呼吸事实/PFT背景证据仍被保留。

## OOF融合的隔离和实际范围

两折成员均重新拟合自己的缩放、PCA和有监督参数。神经成员使用未在本队列标签训练的冻结BERT当前/既往文本缓存加结构化输入，是256→32→4的紧凑模型；不是11M参数完整V7的两折复现，也未用全训练V7表示充当OOF。Cox为双原因L2=10；树为100棵/深度3的离散人时风险集三分类，每段转为恒定强度，输出共同每日S/F1/F2。

OOF共同凸权重为Cox 0.601374、树0.165176、紧凑神经0.233450，两原因和全部时点共用；共同温度1.006562也仅在训练OOF估计。外层用两个内折模型概率均值；随后3个家族按相同超参数重拟合完整训练集，继续使用原OOF权重/温度。没有根据外层调参改权重。当前融合没有超过强V6，停止这一固定配方的扩展。

## 独立验证、失败修复与复现

36项相关合成回归全部通过。11个训练批次的best/last或成员权重已在新进程独立重载，源合同、每日风险、指标、早停和选点一致，最大预测误差0；检验和药物动态接口均从原始来源重建，误差0。统计汇总另用新进程重算0。不能把这些工程一致性写成临床泛化验证。

保留了两种实际错误现场：XGBoost2.1.3与sklearn1.6.1的包装器重载报`AttributeError: 'super' object has no attribute '__sklearn_tags__'`，改用原生Booster读取相同权重后独立重放0，没有重新调参；第一次残差验证容器遗漏GPU参数报无NVIDIA驱动，保留日志后用声明GPU的容器完成验证，不改训练源。原失败容器与日志均保留。

正式本地输出在`training_runs/v7_sequential_research_2026-10-15/`，包含每批合同/来源hash、best/last及每日概率、独立证明、`readout.json`和`readout_verification.json`。其中含受限派生数据和权重，不公开。代码/合成测试、合同及本报告和聚合PNG可以审阅；公开不包含患者行、标签/划分/预测、向量或权重。

复现入口和各批合同见[运行说明](../training/v7/README.md)。真实来源/hash/本地模型缺一时入口拒绝执行；克隆公开源码仅能运行不依赖患者数据的合成回归。

## 全部模型读出附表

附表包含47个新神经拟合的选点、3个复用旧参考和两组各6种融合读出。Cox/树每内折计一个家族成员；融合读出不额外计拟合。实际519神经epoch不包含epoch0，也不包含旧模型训练；树迭代和Cox优化步不混入神经epoch。

|条件/种子|实际神经epoch|选定epoch|年度F1↑|再入院IBS↓|死亡IBS↓|
|---|---:|---:|---:|---:|---:|
|reference/strong_v6|—|5|0.577373|0.099614|0.065008|
|reference/v7_nll_control|—|6|0.565321|0.100361|0.065531|
|reference/v7_recommended|—|8|0.567023|0.100266|0.066214|
|phase1/focal_no_smoothing|8|0|0.565321|0.100458|0.065634|
|phase1/nll_focal_no_smoothing|14|8|0.567023|0.100253|0.065791|
|phase1/without_cross_query|16|10|0.577075|0.100667|0.065362|
|phase1/without_explicit_decay|8|0|0.565321|0.100458|0.065634|
|phase1/drug_query_labs|8|0|0.565321|0.100458|0.065634|
|phase1/nll_brier|8|0|0.565321|0.100458|0.065634|
|cached/global|8|2|0.578125|0.099732|0.064905|
|cached/v7_frozen|10|4|0.581132|0.100298|0.064721|
|cached/availability|18|15|0.583436|0.099924|0.065074|
|cached/linear|8|1|0.581218|0.099705|0.064821|
|cached/ple|8|1|0.579634|0.099864|0.065103|
|cached/velocity|8|1|0.579634|0.099862|0.065143|
|cached/density|11|5|0.580729|0.099910|0.065646|
|cached/curvature|11|5|0.581725|0.100020|0.065543|
|cached/clinical_main|13|7|0.579017|0.099825|0.065052|
|cached/clinical_interaction|12|6|0.579017|0.099867|0.065041|
|cached/routing_uniform|8|1|0.580645|0.099906|0.065417|
|cached/routing_prior|8|1|0.582090|0.099909|0.065384|
|cached_followup/7/global|8|8|0.578164|0.099853|0.064931|
|cached_followup/7/availability|16|10|0.582716|0.099913|0.065028|
|cached_followup/7/linear|12|6|0.583569|0.100024|0.065174|
|cached_followup/7/v7_frozen|18|12|0.581132|0.099969|0.064764|
|cached_followup/7/routing_prior|8|2|0.578581|0.100491|0.065606|
|cached_followup/101/global|8|2|0.578616|0.099764|0.064908|
|cached_followup/101/availability|18|12|0.581998|0.099908|0.065071|
|cached_followup/101/linear|10|4|0.582278|0.099744|0.064899|
|cached_followup/101/v7_frozen|8|2|0.580726|0.100221|0.064823|
|cached_followup/101/routing_prior|11|5|0.580132|0.100378|0.065007|
|drug_screen/name_source|9|3|0.581725|0.099913|0.065161|
|drug_screen/name_identity|9|3|0.579897|0.099954|0.065406|
|drug_screen/name_random|9|3|0.581725|0.099910|0.065166|
|drug_screen/name_semantic|9|3|0.582474|0.099882|0.065114|
|drug_screen/dynamics_source|16|10|0.579750|0.100576|0.065495|
|drug_screen/dynamics_main|8|0|0.577373|0.099614|0.065008|
|drug_screen/dynamics_slope|11|5|0.580645|0.100273|0.065508|
|drug_screen/dynamics_route|8|0|0.577373|0.099614|0.065008|
|strong_anchor|13|10|0.581683|0.099908|0.065145|
|query_replication/control|6|0|0.565321|0.100458|0.065634|
|query_replication/without_cross_query|6|0|0.565321|0.100458|0.065634|
|residual_controls/scaled_v7|15|9|0.583439|0.100033|0.064767|
|residual_controls/fixed_projection|16|10|0.586735|0.100100|0.064742|
|residual_controls/zero_gate|8|1|0.578125|0.099649|0.064996|
|strong_replication|8|2|0.580645|0.099634|0.065378|
|residual_followup/7/scaled_v7|19|13|0.580076|0.100185|0.064857|
|residual_followup/7/fixed_projection|19|13|0.583016|0.100148|0.065035|
|residual_followup/101/scaled_v7|18|12|0.580808|0.100520|0.064930|
|residual_followup/101/fixed_projection|10|4|0.579208|0.100014|0.064395|
|two_fold/cox|—|—|0.560000|0.100308|0.064486|
|two_fold/tree|—|—|0.555233|0.103835|0.067660|
|two_fold/neural|—|—|0.504965|0.112438|0.068869|
|two_fold/equal|—|—|0.572840|0.103169|0.064840|
|two_fold/weighted|—|—|0.569270|0.101715|0.064413|
|two_fold/weighted_temperature|—|—|0.568922|0.101438|0.064467|
|full_refit/cox|—|—|0.559229|0.100040|0.064363|
|full_refit/tree|—|—|0.565149|0.102984|0.067432|
|full_refit/neural|—|—|0.489945|0.113700|0.072978|
|full_refit/equal|—|—|0.570713|0.102587|0.064832|
|full_refit/weighted|—|—|0.569987|0.101178|0.064249|
|full_refit/weighted_temperature|—|—|0.571429|0.100945|0.064325|
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_STRONG_ANCHOR_PROTOCOL_2026-10-15.md · lines 1–7

<!-- REVIEW_FRAGMENT {"path":"docs/V7_STRONG_ANCHOR_PROTOCOL_2026-10-15.md","part":1,"start_line":1,"end_line":7,"source_sha256":"20cf40d2b8a1163d4ef046295e1baf6e65d1fd6881dcf7e738a8be3b35fa1b88","fragment_sha256":"20cf40d2b8a1163d4ef046295e1baf6e65d1fd6881dcf7e738a8be3b35fa1b88","added_terminal_newline":false} -->
````markdown
# 完整V6冻结＋端到端V7修正：单配方

完整已选V6对称背景作为冻结预测函数，来源为cached批已重放的train/tune幅度/shape。重新初始化V7表示与修正头并端到端训练，仅冻结H34及外部BERT/RadBERT；不同于仅冻结V7表示的小读取器。无跨batch投影、无额外零门、无Focal/平滑/对比；仅原连续竞争NLL+1e-4修正平方。

幅度加修正；shape乘exp(.25*tanh(delta_shape))，零末层时严格恢复完整V6风险。可训练架构、8头/M4/FFN1024、历史/来源输入及差分LR/Dropout与V7原NLL相同；新shape约束与缓存全局对照一致。初始风险不冒称等于纯H34，也不承诺训练后泛化保底。

seed42、batch64、AdamW wd.01/clip1、最多16/至少8/patience6/min_delta1e-4；年度调参F1、并列低IBS、再并列较早epoch，保留epoch0、末轮及全局再校准参照。只用原13592训练/1531指定调参，患者/严格标签/全部历史/文本可见性保持。弱结果不扩网格或自动替换。独立进程重载源码、best/last、每日概率、冻结锚和选择。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_STRONG_NO_CROSS_PROTOCOL_2026-10-15.md · lines 1–17

<!-- REVIEW_FRAGMENT {"path":"docs/V7_STRONG_NO_CROSS_PROTOCOL_2026-10-15.md","part":1,"start_line":1,"end_line":17,"source_sha256":"f6a9f45ecb5fb18dd9d3e43c01639f48bcd2731d6f509b53444d1c82a957369d","fragment_sha256":"f6a9f45ecb5fb18dd9d3e43c01639f48bcd2731d6f509b53444d1c82a957369d","added_terminal_newline":false} -->
````markdown
# 完整 V6 冻结后关闭跨源 Query：有限配对实验

2026-10-15。用户授权继续实验。目标是补齐已有 StrongV7 的单变量结构对照，不改历史源码或结果。新增 StrongV7NoCross 仅把 variant.cross 设为 False，同时冻结 drug_cross/rad_cross，保持同一完整 V6 seed42 风险缓存、共有参数初值、shape 公式、所有来源/历史、256/8头/M4/FFN1024、冻结 BERT/RadBERT、Dropout 与差分 LR。

seed42 先运行；最多16完整epoch，最少8、patience6、min_delta1e-4，batch64，AdamW：缓存文本/影像投影LR1e-4，其余3e-4，wd.01、clip1；连续竞争风险 NLL＋1e-4 delta平方，不加入Focal/平滑/对比。每轮遍历全部13,592训练出院，以固定seed打乱批次；不用1,531条调参出院训练。

主选点仍是调参年度 F1 最大、并列低再入院IBS、再并列较早轮；早停也保持原规则。额外仅保存最小再入院IBS及IBS≤完整V6时最高F1的检查点，以免再次缺失中间轮权重；二者是事前声明的描述性辅助读出，不影响训练/早停/主选点，不扫描容差。

先核对共有初始state_dict逐项相同、epoch0等于完整V6风险、跨源模块实际不调用/不入优化器，且M4字段压缩与编码器仍在风险梯度路径。新活动参数9,337,424；已有StrongV7为10,916,944，差1,579,520。

只在同一指定tune上报告：1,531出院/402患者；年度严格再入院F1、7–365日再入院与先死亡IBS、有效每日平均AUC及30/90/365日读出。WorldA本库记录终点，无提前删失权重，不冒充完整院外随访。核对train/tune患者不重叠与划分源文件；confirmation/test只统计成员数量，不参加优化、候选选择或结局读出，不称新盲测。

重载best/last/辅助权重核验概率、标签、指标、冻结风险、主选择与早停；独立进程再次核验。相对同seed保留Query和完整V6分别做402患者聚类配对200次，固定模型/阈值，报告F1和两个结局IBS区间，不以跨零证明等效，也不作多重比较校正的确认性声明。

预算扩种规则固定：相对同seed保留Query，年度F1至少+0.0001且再入院IBS增量不超过+0.0001，才追加seed7匹配复验；否则停止该结构扩种，不自动替换原模型。死亡IBS作为必须同时报告的结果，不据有利单项改写门槛。若主F1不升但概率误差改善，保留该线索，不自动扩网格。

产物保存在独立training_runs/v7_strong_no_cross_2026-10-15；任何已有目录拒绝覆盖。临床数据、划分索引、逐患者预测、权重留本地。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_STRONG_NO_CROSS_RESULTS_2026-10-15.md · lines 1–74

<!-- REVIEW_FRAGMENT {"path":"docs/V7_STRONG_NO_CROSS_RESULTS_2026-10-15.md","part":1,"start_line":1,"end_line":74,"source_sha256":"c52246ee69c680276d8b01bc3f01116f87fc4d519cad1f5d7264681d8c89403f","fragment_sha256":"c52246ee69c680276d8b01bc3f01116f87fc4d519cad1f5d7264681d8c89403f","added_terminal_newline":false} -->
````markdown
# 完整 V6 上关闭跨源 Query：配对实验结果

2026-10-15。仅关闭 StrongV7 的 drug_cross/rad_cross；完整 V6 seed42 风险冻结，其余输入、共有参数初值、风险公式、NLL与修正收缩、学习率、Dropout、预算、主选点保持。历史源码和结果保留，默认模型不替换。

**为什么报告只有1,531次出院、402名患者**

这是固定调参子集，负责比较/选点；每个训练epoch遍历完整13,592条训练出院。原验证患者按固定SHA256奇偶进一步分为调参与确认，划分不是本轮临时抽小，也不因缺文本/检验缩减。

| 用途 | 出院次数 | 患者数 | 本轮用途 |
|---|---:|---:|---|
| 训练 | 13,592 | 3,766 | 优化参数，每轮全量遍历 |
| 调参 | 1,531 | 402 | 选点、早停及同折比较 |
| 历史确认 | 1,418 | 395 | 只核对划分人数，不做结局评估 |
| 历史测试 | 2,860 | 788 | 只核对划分人数，不做结局评估 |
| 合计 | 19,401 | 5,351 | 同患者不跨划分 |

历史确认/测试有过项目早期曝光，不称为全新盲测。F1针对365天本库严格AECOPD首次再入院；死亡是严格再入院前死亡。IBS为7–365日积分Brier，平均AUC为有效每日AUC均值；WorldA记录终点，没有提前删失权重，不是完整院外随访的IPCW推断。

**seed42 主选点与辅助读出**

| 状态 | epoch | 年度F1↑ | 再入院IBS↓ | 死亡IBS↓ | 平均AUC↑ |
|---|---:|---:|---:|---:|---:|
| 完整V6冻结初值 | 0 | 0.577373 | 0.099614 | 0.065008 | 0.806332 |
| 保留Query：匹配旧对照 | 10 | 0.581683 | 0.099908 | 0.065145 | 0.807360 |
| 关闭Query：主F1选点 | 16 | 0.584735 | 0.099993 | 0.065011 | 0.810035 |
| 关闭Query：辅助最低再入院IBS | 5 | 0.580645 | 0.099260 | 0.065178 | 0.807345 |
| 关闭Query：辅助IBS约束选点 | 10 | 0.584615 | 0.099382 | 0.064943 | 0.810054 |

实际完成16轮；训练/每轮评估合计174.87秒，入口截至权重导出205.14秒（含加载、预检、缓存和保存，不含其后配对计算）。主选点与早停没有受到两个辅助规则反馈，辅助结果不能当作独立确认。

| 主模型减参照 | ΔF1及患者95%区间 | Δ再入院IBS及区间 | Δ死亡IBS及区间 |
|---|---|---|---|
| 同seed保留Query | +0.003052 [-0.011699, +0.015570] | +0.000085 [-0.000266, +0.000459] | -0.000135 [-0.000260, -0.000006] |
| 完整V6 | +0.007362 [-0.004366, +0.023625] | +0.000378 [-0.000552, +0.001483] | +0.000003 [-0.001592, +0.001714] |

配对区间：402患者聚类重采样200次，保留同患者重复出院权重，固定已拟合模型及阈值；同一暴露调参集，没有多重比较调整，不代表独立泛化验证。

达到事前扩种门槛（相对匹配对照F1至少+0.0001、再入院IBS增量不超过+0.0001）。

**seed7 主选点与辅助读出**

| 状态 | epoch | 年度F1↑ | 再入院IBS↓ | 死亡IBS↓ | 平均AUC↑ |
|---|---:|---:|---:|---:|---:|
| 完整V6冻结初值 | 0 | 0.577373 | 0.099614 | 0.065008 | 0.806332 |
| 保留Query：匹配旧对照 | 2 | 0.580645 | 0.099634 | 0.065378 | 0.807419 |
| 关闭Query：主F1选点 | 2 | 0.579602 | 0.099636 | 0.065369 | 0.807406 |
| 关闭Query：辅助最低再入院IBS | 0 | 0.577373 | 0.099614 | 0.065008 | 0.806332 |
| 关闭Query：辅助IBS约束选点 | 0 | 0.577373 | 0.099614 | 0.065008 | 0.806332 |

实际完成8轮；训练/每轮评估合计86.87秒，入口截至权重导出113.04秒（含加载、预检、缓存和保存，不含其后配对计算）。主选点与早停没有受到两个辅助规则反馈，辅助结果不能当作独立确认。

| 主模型减参照 | ΔF1及患者95%区间 | Δ再入院IBS及区间 | Δ死亡IBS及区间 |
|---|---|---|---|
| 同seed保留Query | -0.001043 [-0.005880, +0.001971] | +0.000002 [-0.000008, +0.000012] | -0.000009 [-0.000026, +0.000010] |
| 完整V6 | +0.002229 [-0.012562, +0.019396] | +0.000022 [-0.000462, +0.000625] | +0.000361 [-0.001652, +0.002450] |

配对区间：402患者聚类重采样200次，保留同患者重复出院权重，固定已拟合模型及阈值；同一暴露调参集，没有多重比较调整，不代表独立泛化验证。

未达到事前扩种门槛（相对匹配对照F1至少+0.0001、再入院IBS增量不超过+0.0001），不据辅助有利结果改门槛。

**两个 V7 初始化种子的汇总与处置**

| 主选点均值 | 年度F1 | 再入院IBS | 死亡IBS | 平均AUC |
|---|---:|---:|---:|---:|
| 保留Query | 0.581164 | 0.099771 | 0.065262 | 0.807389 |
| 关闭Query | 0.582168 | 0.099814 | 0.065190 | 0.808721 |

平均F1变化+0.001004、再入院IBS变化+0.000043。seed42改善、seed7退步；两个V7初始化种子共用同一个seed42完整V6基座，不代表两个独立基座训练或两次独立患者验证。只有两种子，不能据此估计稳定的总体种子方差。不扩大该结构网格，也不把关闭Query设为默认。保留seed42辅助第10轮的概率误差改善线索，但seed7的IBS约束读取回到初始化，没有复现这个联合改善。

**实现与验证**

15项相关单元测试通过，包含真正旁路Query、共有初始化、冻结参数不入优化器、M4压缩及历史梯度。真实GPU预检进一步检查三步更新与冻结风险；活动参数10,916,944→9,337,424，减少1,579,520。独立进程重载best/last/min_ibs/ibs_guard四个状态，逐项复现每日概率、标签、指标、主/辅助选点及早停，误差0。这证明保存结果可重放，不等于已从头复现全部训练轨迹。

来源合同、全部逐轮日志、四个检查点及逐患者预测在本地training_runs/v7_strong_no_cross_2026-10-15，患者级产物不公开。[事前实验合同](V7_STRONG_NO_CROSS_PROTOCOL_2026-10-15.md)。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_STRONG_REPLICATION_PROTOCOL_2026-10-15.md · lines 1–5

<!-- REVIEW_FRAGMENT {"path":"docs/V7_STRONG_REPLICATION_PROTOCOL_2026-10-15.md","part":1,"start_line":1,"end_line":5,"source_sha256":"dae2f37d2bc3db5640cdeed466a3531a7d6c223945138833f79aa4d530a9d542","fragment_sha256":"dae2f37d2bc3db5640cdeed466a3531a7d6c223945138833f79aa4d530a9d542","added_terminal_newline":false} -->
````markdown
# 强基座端到端方案的第二个表示种子

首个强基座端到端V7在训练中出现约.004年度F1上升，但已选点概率误差存在取舍。因此只增加一个V7表示种子7，与原seed42同max16/min8/patience6及全部数据/损失/输出约束。冻结完整V6来源仍是seed42，不能称两套独立基座或外部验证。沿用原年度F1选择；不因另一个epoch的IBS更好而事后改选。

复用已测试的通用有限训练函数和StrongV7结构；独立重放best/last、来源和epoch0同锚，保留原第一模型不覆盖。若仍只有弱F1/概率取舍，停止本配方，不再扩大种子/学习率网格。
````
<!-- END REVIEW_FRAGMENT -->

## docs/V7_TWO_FOLD_ENSEMBLE_PROTOCOL_2026-10-15.md · lines 1–11

<!-- REVIEW_FRAGMENT {"path":"docs/V7_TWO_FOLD_ENSEMBLE_PROTOCOL_2026-10-15.md","part":1,"start_line":1,"end_line":11,"source_sha256":"8af0e2699c324e81208373a181068dceb4dc262beaa407e1f38778b96549fb4c","fragment_sha256":"8af0e2699c324e81208373a181068dceb4dc262beaa407e1f38778b96549fb4c","added_terminal_newline":false} -->
````markdown
# 两折OOF异构概率融合：预先固定低成本配方

用原训练患者随机种子42分为两组，患者所有出院同组。每折只用另一组拟合尺度、PCA、Cox、树和紧凑多源神经网络；禁止复用全训练监督H34/V6/V7隐藏状态。确认/测试不评价。外层指定调参只读出，融合权重仅由训练OOF预测拟合。

三成员：连续双原因Cox（H34原值重新内折标准化＋九检验末值/有测/年龄＋EOS/方案原组成）；CPU XGBoost多项条件区间风险（0/7/14/30/60/90/180/365、100树深3、lr.05/min_child_weight10/lambda10）；新初始化紧凑多源Weibull网络（256→32→双原因幅度/shape，冻结BERT的当前/全部可见历史平均摘要经内折PCA各16维，另有检验/方案/PFTpresence/CXR四状态/旧支持）。最后一项是为节约时间采用的紧凑成员，不能称为完整11M参数V7的两折重训。

网络无本队列有监督预训练基座；原联合NLL，固定24轮（不以外折OOF结局选epoch）、batch256、AdamW lr.001/wd.01/clip1、Dropout.3。每折内所有预处理只fit内折训练患者；不做内部参数网格。树对已发生其他事件后的区间不建样本，不把竞争事件当删失；区间内转换为常强度日概率，明确是时间离散近似。

外层调参预测取两个内折成员平均，避免额外全训练重拟合成本；训练OOF每人只有未见该患者标签的单折预测，两者分布差异如实报告。三家族在30/90/180/365双原因OOF Brier均值上拟合一组非负和为1权重，所有原因/时点共用。另报固定等权及三个成员，不预设提升。温度缩放作为独立候选：只用OOF每日三状态条件概率拟合共同正温度，再逐日递推，保持守恒；与未校准融合分开比较。

保存折归属/所有模型/内折预处理参数/OOF与外层预测、实际秒数及来源hash；独立进程只重载模型与保存预处理重放OOF/外层结果、权重优化和全部指标，不重复拟合临床模型。临床记录、标签、预测与权重仅本地。旧强V6另作不同训练规模参照，不能把半训练两折平均与全训练模型当同训练量算法排名。
````
<!-- END REVIEW_FRAGMENT -->

## README.md · lines 1–80

<!-- REVIEW_FRAGMENT {"path":"README.md","part":1,"start_line":1,"end_line":80,"source_sha256":"aae2e275c49acce2b171eb86c4cb1e3799f34fd22e69b725879ffd0ac0733002","fragment_sha256":"aae2e275c49acce2b171eb86c4cb1e3799f34fd22e69b725879ffd0ac0733002","added_terminal_newline":false} -->
````markdown
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
````
<!-- END REVIEW_FRAGMENT -->
