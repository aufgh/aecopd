"""把已验证的事件、静态和文本审计合成一张中文概念总表。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SPECIMEN_ZH = {"arterial": "动脉血", "venous": "静脉血", "central_venous": "中心静脉血",
               "mixed": "混合标本", "unknown": "标本未确定"}
EVENT_ZH = {
    "ph": "酸碱度 pH", "pco2": "二氧化碳分压 pCO₂", "po2": "氧分压 pO₂",
    "albumin": "白蛋白", "bicarbonate": "碳酸氢盐", "creatinine": "肌酐",
    "hemoglobin": "血红蛋白", "wbc": "白细胞计数",
    "saba": "短效 β₂ 激动剂 SABA", "sama": "短效抗胆碱药 SAMA",
    "laba": "长效 β₂ 激动剂 LABA", "lama": "长效抗胆碱药 LAMA",
    "inhaled_steroid": "吸入糖皮质激素", "systemic_steroid": "全身糖皮质激素",
    "macrolide": "大环内酯类抗菌药", "beta_lactam": "β-内酰胺类抗菌药",
    "fluoroquinolone": "氟喹诺酮类抗菌药", "other_antibiotic": "其他抗菌药（异质组）",
    "loop_diuretic": "袢利尿剂", "other_diuretic": "其他利尿剂（异质组）",
}
STATIC_ZH = {
    "context:age_at_admission": "本次入院年龄", "context:male": "男性",
    "context:prior_coded_admissions": "既往符合编码的住院次数",
    "context:prior_strict_aecopd_admissions": "既往严格急性加重编码住院次数",
    "aecopd_index_code:strict_aecopd": "本次严格 AECOPD 编码",
    "chronic_comorbidity:heart_failure": "心力衰竭共病",
    "chronic_comorbidity:cardiac_other": "其他心脏共病",
    "chronic_comorbidity:ckd": "慢性肾病共病",
    "chronic_comorbidity:diabetes": "糖尿病共病",
    "chronic_comorbidity:malignancy": "恶性肿瘤共病",
    "chronic_comorbidity:chronic_respiratory": "其他慢性呼吸共病",
    "acute_diagnosis:acute_resp_failure": "本次急性呼吸衰竭",
    "acute_diagnosis:pneumonia_or_aspiration": "本次肺炎或误吸",
    "acute_diagnosis:sepsis": "本次脓毒症",
    "acute_diagnosis:aki": "本次急性肾损伤",
    "disposition:home": "出院去向：家庭",
    "disposition:facility": "出院去向：机构",
    "disposition:other": "出院去向：其他",
}
TEXT_ZH = {
    "history_of_present_illness": "现病史文本", "past_medical_history": "既往史文本",
    "hospital_course": "住院经过文本", "discharge_condition": "出院时状况文本",
}
FIELDS = ("类别", "中文临床概念", "模型字段", "原始来源", "覆盖人数", "覆盖住院数",
          "未观察住院数", "原始行或片段数", "同刻碰撞或多片段数", "重复原始ID或同住院重复行数",
          "临床理由", "计数解释")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build(directory: Path) -> Path:
    output = directory / "current_concept_catalog_zh.csv"
    if output.exists():
        raise FileExistsError(output)
    result = []
    for row in _read(directory / "current_event_concepts.csv"):
        concept = row["concept"]
        family, _, specimen = concept.partition(":")
        if family not in EVENT_ZH or (specimen and specimen not in SPECIMEN_ZH):
            raise ValueError(f"untranslated event concept: {concept}")
        name = SPECIMEN_ZH[specimen] + EVENT_ZH[family] if specimen else EVENT_ZH[family]
        result.append(("检验事件" if row["kind"] == "laboratory" else "实际给药事件",
                       name, concept, row["source"], row["patients"], row["admissions"],
                       row["unobserved_admissions"], row["events"],
                       row["same_timestamp_collisions_not_proven_duplicates"], row["repeated_source_ids"],
                       row["clinical_rationale"], "同刻多条未证实重复；复方可跨类别映射"))
    for row in _read(directory / "current_static_concepts.csv"):
        concept = row["concept"]
        if concept not in STATIC_ZH:
            raise ValueError(f"untranslated static concept: {concept}")
        result.append(("结构化静态", STATIC_ZH[concept], concept, row["source"], row["patients"],
                       row["admissions"], row["unobserved_admissions"], row["rows"], "不适用",
                       row["duplicate_same_admission_rows"], row["clinical_rationale"],
                       "既往次数为数值，诊断及去向为类别；零值不等于缺失"))
    for row in _read(directory / "current_text_fields.csv"):
        field = row["field"]
        if field not in TEXT_ZH:
            raise ValueError(f"untranslated text field: {field}")
        result.append(("出院小结文本", TEXT_ZH[field], field, row["source"], row["patients"],
                       row["admissions"], row["unobserved_admissions"], row["chunks"],
                       row["multiple_chunks_above_one_per_admission"], "不适用", row["clinical_rationale"],
                       "同住院多片段是分块，不是重复文书"))
    if len(result) != 54:
        raise ValueError(f"unexpected concept count: {len(result)}")
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(result)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    print(build(parser.parse_args().audit_dir))
