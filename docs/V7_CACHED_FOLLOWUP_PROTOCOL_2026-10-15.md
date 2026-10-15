# 缓存筛查后两个追加种子

已知seed42点估计后固定五条件：global、availability、linear、v7_frozen、routing_prior。它们覆盖全局再校准、最高F1的来源信息、同原值普通读取、冻结V7表示和软路由。追加seed7/101，不根据新结果改变名单。所有条件同一个冻结强V6 seed42来源，因此是新增读取器的三随机初始化复核，不是三个独立基座或外部验证。

完全沿用cached批特征/来源、512槽/16隐藏/Dropout.3、同输出变换和联合NLL+1e-4正则、AdamW .001/wd.01/clip1、batch256、max24/min8/patience6/min_delta1e-4及年度F1选点。新进程逐项重放best/last/选择/指标；保留原12条件结果不覆盖，不扩参数网格，不默认替换。
