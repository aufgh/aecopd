"""从审计产物生成中文全字段说明及未用来源清单；不修改训练合同。"""
from pathlib import Path
import json
import csv
import numpy as np
from patient_data.export_mimiciv import sha256
from patient_data.v6_input_audit_compact import ROOT, LAB_NAMES, LAB_FAMILY, FAMILIES


def run():
    audit=json.loads((ROOT/'audit.json').read_text());rows=audit['rows']
    source_dir=Path('patient_data/processed/v6_resp_micro_context_2026-10-09_r2')
    source=json.loads((source_dir/'manifest.json').read_text())
    with np.load(source_dir/'resp_micro_context.npz') as z:context={k:z[k] for k in z.files}
    # 此时读取的源必须与已保存清单一致；不以历史文档中的零/未导出描述作当前事实。
    for name,digest in source['output_sha256'].items():
        if sha256(source_dir/name)!=digest:raise ValueError('source context changed')
    extra=dict(status='existing_source_inventory_not_added_to_model',respiratory_numeric=source['numeric_coverage'],
      respiratory_support_and_duration=source['support_coverage'],array_shapes={k:list(v.shape) for k,v in context.items()},
      source_manifest_sha256=sha256(source_dir/'manifest.json'),live_postgresql_checked=False,
      live_check_limitation='psql read-only query lacked process credential; no credential written or saved; existing exports independently hash checked')
    (ROOT/'other_sources.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2),encoding='utf-8')
    reasons={
      'acid_base':'通气/酸碱代偿与严重度；需标本确认，不把静脉血气当动脉',
      'inflammation':'感染/炎症及免疫细胞组成；EOS用于ICS反应背景，非单向异常或确证感染',
      'hematology_nutrition':'贫血/营养/炎症背景；临床相关不保证再入院独立增量',
      'renal':'共病/治疗安全及全身严重度，保留用于整体组比较',
      'electrolytes':'呼吸肌与治疗安全/全身病情，非COPD特异',
      'metabolic_perfusion':'糖代谢/激素影响/灌注背景，保留用于整体组比较',
      'cardiac':'心肺重叠及心肌损伤鉴别，不能混合BNP与NT-proBNP或不同肌钙蛋白测定'}
    lines=['# 全输入审计：来源、稀疏、重复与候选取舍','',
      '19,401次预测出院、5,351患者；分母、严格491.21/J44.1任意诊断位结局及患者划分未变。',
      '本表审计的是强参照及全部近期候选的并集，不声称所有39轴/38药物已被采用为默认。原强参照只有22轴+7补充来源、7类治疗等；新接口/新实验分开登记。',
      '细叶阈值仅看训练患者≥20人，不按结局P值或测试分数选字段。少于20人药物仍上卷药效父层；罕见医学核心检验不自动删除。',
      '“有记录”与“异常/阳性”分开：正常记录、未检验、明确阴性、未提及不互换。',
      '', '## 全部检验轴/来源','',
      '|项目|英文键|覆盖住院/患者|覆盖%|异常峰值住院（仅原22轴）|取舍与理由|','|---|---|---:|---:|---:|---|']
    for r in rows:
        if r['group']!='laboratory':continue
        name=r['name'];j=LAB_NAMES.index(name);reason=reasons[FAMILIES[LAB_FAMILY[j]]]
        if name in audit['excluded_lab_candidates']:
            reason='简化包中暂屏蔽，不认定无用。'+reason
            if name=='hematocrit':reason+='；高红细胞压积的慢性缺氧背景并未被低Hb轴完全替代，不能永久删除'
            if name=='protein_total':reason+='；覆盖2.70%，和白蛋白主题部分重叠，不是完全重复'
        if name in ('bicarbonate_low','bicarbonate_high','wbc_low','wbc_high','sodium_low','sodium_high','potassium_low','potassium_high','glucose_low','glucose_high'):
            reason='合并同分析物概念，两个方向的内容保留。'+reason
        abnormal=str(r['positive_admissions']) if j<22 else '未设异常标签'
        lines.append(f"|{r['chinese']}|{name}|{r['observed_admissions']:,}/{r['observed_patients']:,}|{r['observed_percent']:.2f}|{abnormal}|{reason}|")
    lines+=['','高低轴合并5对后为34概念；简化包暂屏蔽4概念后30。高碳酸酸中毒是pH/PCO2组合，保留为知识特征，不当独立测量。',
      '中性/淋巴百分比的训练折最新值Spearman约−0.90，但不是完全重复，不能据此删除其中一个。其它当前审计的额外检验未发现绝对相关≥0.80的成对关系。',
      '','## 全部药物身份与真实可用剂量','',
      '给药、药房及关联处方互补。网络只看到统一治疗记录、单次剂量/单位及频率，不看到执行/医嘱来源类别。来源留作审计，避免声称计划均被执行。',
      '不把复方成分剂量拆成两个单药重复相加；不用累计量、药物日数、首末日期或跨度。mg/mcg/g规范化；puff/inh/neb/ml不互换。',
      '实际最大单次剂量优先，缺失时使用可用医嘱最大单次剂量；实际每记录日频率优先，缺失时使用计划频率。该接口仍不能解释完整药物疗效或住院外依从性。',
      '万古霉素旧实际汇总途径不清，新接口仅采用已核验IV医嘱；不可把口服万古霉素当全身抗呼吸感染。',
      '', '|药物身份|覆盖住院/训练患者|原完整实际剂量住院|医嘱补单次剂量住院|细叶取舍|','|---|---:|---:|---:|---|']
    for r in rows:
        if r['group']=='drug_identity':
            decision='保留细叶' if r['proposal']=='fine_leaf_keep' else '仅保留父药效，无独立细叶'
            lines.append(f"|{r['name']}|{r['observed_admissions']:,}/{r['train_patients']:,}|{r['actual_single_dose']:,}|{r['new_order_dose_only']:,}|{decision}|")
    lines+=['','上表万古霉素“原实际剂量”只作审计，新接口不使用该列。吸入制剂多种单位/组合导致剂量完整度不均，不能仅凭有药名就假装已有可靠mg累计。',
      '药效父层为SABA短效β₂激动剂、SAMA短效抗胆碱能、LABA长效β₂激动剂、LAMA长效抗胆碱能、ICS吸入激素、全身激素、抗菌、黏液相关、甲基黄嘌呤及PDE4抑制剂。',
      'GOLD主要治疗类别提供候选理由；茶碱/愈创甘油醚/广谱抗菌药不等于推荐用于全部COPD患者，也不证明出院后独立预测价值。',
      '', '## 微生物、支持、静态和文本：全部现有字段','',
      '|组|字段|有记录住院/患者|阳性住院|处理|','|---|---|---:|---:|---|']
    for r in rows:
        if r['group'] in ('laboratory','drug_identity'):continue
        reason={'micro':'共享培养mask去重；内容全部保留，阳性不等于活动感染/定植',
          'support':'保留3类；仅ICU记录证据，缺失不是未支持',
          'frozen_history':'不动冻结H34基础风险', 'frozen_current':'不动冻结H34诊断/去向参照',
          'text':'保留B四字段现有冻结向量；缺失不补造正常',
          'drug_class':'上卷共享药效；不重复累计复方剂量'}[r['group']]
        lines.append(f"|{r['group']}|{r['name']}|{r['observed_admissions']:,}/{r['observed_patients']:,}|{r['positive_admissions']:,}|{reason}|")
    lines+=['','静态/文本表中“阳性”仅表示非零编码/存在文本，不是疾病结局阳性。当前诊断编码缺乏不等于临床无病；去向未填不补为回家。',
      '','## 已保存但当前强参照未利用的信息','',
      '|信息|本地核验|是否已入本轮风险网络|注意事项|','|---|---|---|---|']
    for name,r in source['numeric_coverage'].items():
        lines.append(f"|{name}|{r['admissions']:,}住院/{r['patients']:,}患者|否|已保存低频锚点，最后ICU记录不是出院状态|")
    lines+=['|通气时长、拔管/重插管|已保存resp_micro_context.npz及3,133条原始操作记录|否，当前仅3类支持|需确认区间、ICU覆盖与适应证；不是所有病人都可测|',
      '|胸片结果|12,346份报告，4,888次住院；有CheXbert、RadBERT/RadGraph派生|否，已有独立历史实验|重复报告可能用于设备检查；专科抽取不是专家金标准|',
      '|胸部CT形态|2,118份报告，1,872次住院；专科派生已保存|否，已有独立历史实验|覆盖9.65%；不能把缺CT当没有肺气肿/支扩|',
      '|FEV1/FVC、家庭氧疗、吸入技术、依从性、功能/照护|已有小结原句筛选，肺功能主题5,012次句段命中|无独立标准化事实；四字段向量可能包含部分文本|命中句段不是5,012患者或有效肺功能测量；要核日期/否定/单位|',
      '|BMI/体重及其变化|官方OMR存在；本地仅核到旧队列09_validation_omr.csv的历史统计|否|旧队列覆盖不应用作现19,401分母；本轮未成功实时查询，未假装完成覆盖|',
      '|急诊分诊、门诊治疗、服务/转科、操作代码|官方MIMIC对应模块/表；本快照来源没有全部导出|否|ED是否本机导入待确认；服务/收费不自动等同病情，不以ID预测|',
      '|PaO2、血气氧饱和度、碱剩余|此前本快照候选审计有数值，4,261/1,343/4,262次住院（混合标本）|否|不是已确认动脉PaO2；需氧疗上下文，不能随意算P/F|',
      '|PCT|本快照d_labitems的procalcitonin名称检索未见item|否|不据此断言全库或文本没有PCT|',
      '','## 来源与可复核文件','',
      '- all_inputs.csv / audit.json：全部字段覆盖、患者数、异常/阳性、候选及相关性。',
      '- interface.npz / normalization.json：独立新接口，不含结局，训练折剂量尺度。',
      '- other_sources.json：已保存来源、数组形状与本次实时核验限制。',
      '- 原始快照：patient_data/mimiciv_copd/snapshot_2026-09-27；补充来源：同级supplement_resp_micro_2026-10-09。',
      '- 初版无r2目录为开发审计，万古霉素途径修订后只使用r2训练。',
      '','临床/方法依据见docs/V6_AUDIT_LITERATURE_2026-10-12.md；正式结果见training_runs/v6_audited_coarse_fine_2026-10-12。没有删除原始数据或编码。']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='comprehensive_readme_and_source_inventory_generated',root=str(ROOT))))


if __name__=='__main__':run()
