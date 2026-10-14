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
