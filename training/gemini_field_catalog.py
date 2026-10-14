"""单一的16字段定义：细分内容，不把字段名当作事件时点或临床真值。"""
from __future__ import annotations

CATALOG_VERSION = 'clinical_verbatim_fields16_2026-10-06'

# 每项为中文名称、英文检索说明、原五主题；不使用这些说明做关键词预筛。
FIELDS = {
    'acute_respiratory_diagnosis': (
        '本次呼吸诊断与病因判断',
        'Respiratory diagnoses, suspected causes or explicit exclusions in this episode. '
        'Keep suspicion and later reversal; bare chronic COPD history is not an acute diagnosis.',
        'respiratory_course'),
    'dyspnea_activity_course': (
        '气促、活动诱发及恢复情况',
        'Actual breathlessness at rest or with activity, changes and comparison with baseline. '
        'Admission, recovery and discharge are allowed; retain the stated time. Not PRN drug indications.',
        'respiratory_course'),
    'cough_sputum_clearance': (
        '咳嗽、痰液与排痰情况',
        'Actual cough, sputum amount/character, clearance difficulty, secretions, suction or improvement. '
        'Admission symptoms are allowed with their original time context.',
        'respiratory_course'),
    'wheezing_respiratory_effort': (
        '喘鸣与呼吸费力体征',
        'Actual wheeze, prolonged expiration, respiratory effort/distress or documented resolution. '
        'Keep substantive respiratory reassessment, not an isolated routine normal examination.',
        'respiratory_course'),
    'oxygen_requirement_course': (
        '氧疗需求与调整过程',
        'Actual baseline, inpatient or discharge oxygen requirement, escalation, weaning and home oxygen. '
        'Not isolated numerical vital tables or generic oxygen safety instructions.',
        'respiratory_support'),
    'ventilation_support_weaning': (
        '通气支持、撤机与耐受',
        'Actual respiratory NIV/intubation, weaning, mask tolerance and home ventilation plans. '
        'Exclude routine surgical anesthesia and uncomplicated sleep-apnea-only CPAP.',
        'respiratory_support'),
    'systemic_steroid_regimen': (
        '全身激素方案、疗程与反应',
        'Actual respiratory systemic steroid use, continuation, taper, remaining course, response or intolerance. '
        'A fixed course is valid. Keep uncertain indication context; exclude clearly unrelated indications.',
        'treatment_regimen'),
    'respiratory_antibiotic_course': (
        '呼吸抗菌治疗、疗程与反应',
        'Antibiotic treatment for respiratory infection/exacerbation, course, switch, response or intolerance. '
        'Keep indication context and uncertainty. Exclude clearly unrelated infection/prophylaxis and allergy-only entries.',
        'treatment_regimen'),
    'inhaled_medication_regimen': (
        '吸入／雾化方案及调整',
        'Actual inhaled/nebulized respiratory medication regimens, continuation, changes or response. '
        'Do not infer a new start from a medication list. Exclude nasal sprays, creams and unrelated mixed lists.',
        'treatment_regimen'),
    'chest_imaging_findings': (
        '胸部影像发现及变化',
        'Substantive chest imaging findings, change or explicit respiratory disease exclusions. '
        'Keep uncertainty and follow-up context. Exclude unrelated imaging, technique and normal device position.',
        'imaging_microbiology'),
    'respiratory_pathogen_culture': (
        '呼吸病原、耐药与定植判断',
        'Respiratory pathogen/specimen results, resistance, pending results or infection-versus-colonization interpretation. '
        'Exclude unrelated specimens and unimportant contamination.',
        'imaging_microbiology'),
    'relevant_history_comorbidity': (
        '相关病史、共病与吸烟背景',
        'Respiratory disease history/severity, smoking exposure and comorbid conditions with documented relevance '
        'to breathing, recovery or management, including cardiac overlap and aspiration/swallowing disease. '
        'Keep assessed absence of contribution; not a bare COPD label or unrelated stable diagnosis list.',
        'comorbidity_function_care'),
    'inhaler_technique_adherence': (
        '吸入操作与实际用药依从情况',
        'Actual assessed inhaler technique, ability to manage respiratory medicines, missed doses, self-discontinuation '
        'or refusal. Not medication administration records or generic take-medicines-as-prescribed advice.',
        'comorbidity_function_care'),
    'mobility_frailty_assistance': (
        '活动、自理、虚弱与认知障碍',
        'Actual mobility/ADL limitation, frailty, cachexia, assistance or cognitive/mental barriers to self-care, '
        'and explicitly described functional recovery. Exclude isolated default independent/alert/coherent templates.',
        'comorbidity_function_care'),
    'discharge_disposition_social_barriers': (
        '照护去向、支持与药物设备获取',
        'Concrete home services, nursing/rehab placement, caregiver support, living alone and medication/equipment '
        'access. Keep actual arrangements; not generic go-home advice or contacts.',
        'comorbidity_function_care'),
    'care_limits_or_ama': (
        '救治限制、拒绝治疗与自行离院',
        'Actual DNR/DNI, hospice/comfort-only, refusal of necessary treatment or AMA departure. '
        'Not Full code, hypothetical discussions or merely reluctance to visit hospital.',
        'comorbidity_function_care'),
}

# 名称是模型的检索优先级提示，不是硬性章节过滤；未知/错分章节仍能选所有字段。
ROUTE_PRIORITY = {
    'baseline': ('acute_respiratory_diagnosis', 'dyspnea_activity_course', 'cough_sputum_clearance',
                 'wheezing_respiratory_effort', 'relevant_history_comorbidity', 'inhaler_technique_adherence',
                 'mobility_frailty_assistance', 'discharge_disposition_social_barriers'),
    'course_results': tuple(FIELDS),
    'discharge_management': ('dyspnea_activity_course', 'oxygen_requirement_course',
                             'ventilation_support_weaning', 'systemic_steroid_regimen',
                             'respiratory_antibiotic_course', 'inhaled_medication_regimen',
                             'inhaler_technique_adherence', 'mobility_frailty_assistance',
                             'discharge_disposition_social_barriers', 'care_limits_or_ama'),
}


def field_table(fragments: list[dict]) -> dict[str, list[str]]:
    """固定16列的原文视图；空列表仅为未选到，失败状态由调用方单独保存。"""
    result = {key: [] for key in FIELDS}
    for item in sorted(fragments, key=lambda row: (row['start'], row['end'])):
        for key in item['fields']:
            result[key].append(item['text'])
    return result
