"""冻结的首轮100条人工式原文审阅记录；非自动临床打分器。"""
import json
from pathlib import Path
from training.report_bonsai_catalog16 import collect, ROOT

# 其余引文已逐条读过全文并审阅；下面是明确的错误/存疑项。
BAD = {
 'R014:respiratory_antibiotic_course:0': '片段尾部带入下一处方编号2，候选边界不完整。',
 'R014:respiratory_antibiotic_course:1': '片段尾部带入下一处方编号3，候选边界不完整。',
 'R014:respiratory_antibiotic_course:3': '整段急诊生命体征、检验及给药，超出目标抗感染方案，含结构化冗余。',
 'R014:chest_imaging_findings:1': '仅旧CXR正常，未明确排查的疾病；不符合本栏异常/特定排查结论。',
 'R014:respiratory_pathogen_culture:0': '正常呼吸菌群不是明确致病菌、耐药或医生定植判断。',
 'R014:respiratory_pathogen_culture:2': '仅呼吸菌群，未检出具体致病菌或临床定植判断。',
 'R018:chest_imaging_findings:2': 'CT分期目的且报告待回报，不是已知影像发现；不能推断结果。',
 'R019:inhaler_technique_adherence:0': '仅泛化自述用药依从，未明确呼吸药物或吸入技术。',
 'R020:discharge_residual_dyspnea:0': 'HPI入院前爬楼症状，不能当出院残余气促。',
 'R020:discharge_residual_dyspnea:1': 'HPI入院症状，不能当出院残余气促。',
 'R020:discharge_residual_dyspnea:2': 'PRN药物适应证不是真实出院症状。',
 'R020:cough_sputum_burden:0': '入院HPI慢性咳嗽，未记录后期/出院症状。',
 'R020:airflow_wheezing_distress:0': '入院普通正常查体，不是出院转归证据。',
 'R020:airflow_wheezing_distress:1': '入院普通阴性查体，不能转写为出院残余体征。',
 'R020:airflow_wheezing_distress:2': '入院正常呼吸努力，不是出院或恢复状态。',
 'R020:airflow_wheezing_distress:3': '同一入院查体重复抽取，且时间/范围均不符合。',
 'R020:oxygen_trajectory_discharge:0': '原文确有出院RA，但候选夹带全段无关体检，未聚焦临床片段。',
 'R020:inhaled_medication_changes:1': '原文吸入药正确，但带下一处方编号5。',
 'R020:inhaled_medication_changes:2': '原文吸入药正确，但带下一处方编号6。',
 'R020:inhaled_medication_changes:3': '原文吸入药正确，但带下一处方编号10。',
 'R020:inhaler_technique_adherence:0': '继续吸入方案不代表评估过依从性或操作技术。',
 'R020:inhaler_technique_adherence:1': '常规按时服药提醒，且夹带疼痛/称重嘱托，不是实际依从障碍。',
}
UNCERTAIN = {
 'R014:respiratory_pathogen_culture:1': '仅革兰染色形态且培养正常菌群，是否符合本栏临床病原标准需专业复核。',
 'R017:mobility_frailty_assistance:0': '残留右肩无力真实，但未明确导致出院行走/ADL受限；不直接计合格。',
 'R020:cardiopulmonary_overlap:0': '记录稳定容量状态，未明示与本次呼吸鉴别的关联；不推断独立临床贡献。',
 'R020:mobility_frailty_assistance:1': '疼痛限制查体力量，是否等同出院自理/移动受限不明确；不直接计合格。',
}


def run():
    quotes, _ = collect('four')
    if len(quotes) != 100:
        raise ValueError('Not the frozen 100-quote review; all new items need fresh review')
    seen, entries = set(), {}
    for q in quotes:
        suffix = q['id'].split(':', 1)[1]
        grade, reason = 'correct', '已对照完整原文；在本候选目录下字段、时间、适应证和原文范围可用。不是已证明预测增益。'
        if suffix in BAD:
            grade, reason = 'incorrect', BAD[suffix]; seen.add(suffix)
        if suffix in UNCERTAIN:
            grade, reason = 'uncertain', UNCERTAIN[suffix]; seen.add(suffix)
        entries[q['id']] = {'sha256': q['quote_sha256'], 'grade': grade, 'reason': reason}
    if seen != set(BAD)|set(UNCERTAIN):
        raise ValueError('Unused clinical annotations')
    output = ROOT/'catalog16_review_2026-10-04.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps({'reviewer': 'assistant_not_physician', 'all_100_quotes_reviewed_against_full_notes': True,
                                 'quotes': entries}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    run()
