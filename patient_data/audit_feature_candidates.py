"""只读审计 V5 的原始候选、现行事件概念与静态/文本覆盖，不改训练输入。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from patient_data.export_mimiciv import NULL, rows, sha256
from training.v5.clinical_concepts import medication_classes

CURRENT_LABS = {"50818", "50820", "50821", "50862", "50882", "50912", "51222", "51301"}
LAB_ITEM_FOR_CONCEPT = {"pco2": "50818", "ph": "50820", "po2": "50821",
                        "albumin": "50862", "bicarbonate": "50882", "creatinine": "50912",
                        "hemoglobin": "51222", "wbc": "51301"}
REVIEW_LABS = {
    "50813": "乳酸/组织灌注", "50817": "血气氧饱和度/氧合", "50889": "CRP/炎症",
    "50802": "碱剩余/酸碱状态，与碳酸氢盐和pH重叠",
    "50804": "血气计算总二氧化碳，与碳酸氢盐重叠",
    "50868": "阴离子间隙/酸碱状态，和氯/碳酸氢盐相关",
    "50902": "血氯/酸碱状态，和钠/碳酸氢盐相关",
    "50931": "血糖/糖尿病与应激激素治疗",
    "50960": "血镁/支气管扩张剂及利尿剂治疗安全",
    "50963": "NT-proBNP/心衰鉴别", "50971": "血钾/治疗安全", "50983": "血钠/整体病情",
    "50924": "铁蛋白/炎症及贫血，非COPD特异且相对稀疏",
    "51003": "肌钙蛋白T/心脏事件", "51199": "嗜酸粒细胞绝对数/治疗表型",
    "51006": "尿素氮/肾功能及容量状态，与肌酐相关",
    "51200": "嗜酸粒细胞百分比/治疗表型", "51265": "血小板/全身病情",
    "51221": "红细胞压积/贫血，与血红蛋白高度相关",
    "51256": "中性粒细胞百分比/感染和激素反应，与白细胞相关",
    "51277": "RDW/贫血与衰弱代理，非COPD特异",
    "52075": "绝对中性粒细胞/感染，与百分比及白细胞相关",
    "52073": "嗜酸粒细胞绝对数/治疗表型",
}
BRAND_REVIEW = ("duoneb", "combivent", "symbicort", "advair", "breo", "trelegy",
                "spiriva", "incruse", "anoro", "stiolto", "pulmicort", "ventolin",
                "proair", "xopenex", "atrovent", "daliresp", "roflumilast")
RATIONALE = {
    "ph": "酸碱状态与通气衰竭；标本不同不能混作动脉血气",
    "pco2": "二氧化碳潴留/通气衰竭；静脉值不等于PaCO2",
    "po2": "氧合严重度；只有已核实的动脉标本才能称PaO2",
    "bicarbonate": "酸碱代偿及长期高碳酸血症线索，受肾功能等影响",
    "wbc": "感染/炎症的非特异线索，受激素影响",
    "hemoglobin": "贫血与携氧能力，也反映共病",
    "creatinine": "肾功能、合并症与治疗耐受性，非COPD特异",
    "albumin": "营养/衰弱及炎症负担，非COPD特异",
    "saba": "急性加重短效β2激动剂治疗；给药是诊疗行为而非疗效",
    "sama": "急性加重短效抗胆碱能治疗；给药是诊疗行为而非疗效",
    "laba": "长期维持支气管扩张治疗；需复核复方/商品名",
    "lama": "长期维持支气管扩张治疗；需复核复方/商品名",
    "inhaled_steroid": "预防加重相关维持治疗；需复核吸入途径与复方",
    "systemic_steroid": "急性加重系统激素治疗；需排除局部制剂",
    "macrolide": "有指征的抗菌治疗；也可能用于其他感染",
    "beta_lactam": "有指征的抗菌治疗；也可能用于其他感染",
    "fluoroquinolone": "有指征的抗菌治疗；也可能用于其他感染",
    "other_antibiotic": "宽泛抗菌药组合，异质性高，需审查具体药物",
    "loop_diuretic": "心衰/容量负荷治疗，属于共病过程信号",
    "other_diuretic": "异质性利尿药组合，非AECOPD特异，需审查",
}
STATIC_RATIONALE = {
    "context:age_at_admission": "年龄影响COPD、共病和死亡风险",
    "context:male": "性别是基本人口学变量，需看独立增量而非假定因果",
    "context:prior_coded_admissions": "既往COPD编码住院频次是再入院基线风险",
    "context:prior_strict_aecopd_admissions": "既往严格加重住院频次是严重加重史",
    "aecopd_index_code:strict_aecopd": "本次严格加重编码反映本次事件表型，不是未来标签",
    "chronic_comorbidity:heart_failure": "心衰可模仿/加重气促并改变再入院风险",
    "chronic_comorbidity:cardiac_other": "缺血性心脏病/房颤等心脏共病，编码组较宽",
    "chronic_comorbidity:ckd": "肾病影响全身风险与治疗耐受性",
    "chronic_comorbidity:diabetes": "糖尿病影响感染及整体健康风险",
    "chronic_comorbidity:malignancy": "肿瘤反映非COPD疾病负担与竞争死亡",
    "chronic_comorbidity:chronic_respiratory": "慢性呼吸衰竭/睡眠呼吸障碍反映呼吸储备",
    "acute_diagnosis:acute_resp_failure": "本次急性呼吸衰竭是严重程度线索",
    "acute_diagnosis:pneumonia_or_aspiration": "肺炎/误吸提示感染与可能的再入院机制",
    "acute_diagnosis:sepsis": "脓毒症是本次全身严重度与共病过程",
    "acute_diagnosis:aki": "急性肾损伤反映本次器官损害",
    "disposition:home": "出院去向及照护需求代理，可能受制度和医生行为影响",
    "disposition:facility": "出院去向及照护需求代理，可能受制度和医生行为影响",
    "disposition:other": "出院去向及照护需求代理，可能受制度和医生行为影响",
}
TEXT_RATIONALE = {
    "history_of_present_illness": "入院病情叙述，可能含症状/诱因但与诊断编码重叠",
    "past_medical_history": "既往病史语义，可能补充未编码疾病但与共病编码重叠",
    "hospital_course": "住院期间病情与治疗反应叙述，可能补充数值和药物事件",
    "discharge_condition": "出院时残余症状/功能状态，可能与近期再入院有关",
}


def _write_csv(path: Path, names: tuple[str, ...], records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(records)


def _family(concept: str) -> str:
    return concept.split(":", 1)[0]


def _static_source(concept: str) -> str:
    if concept == "context:age_at_admission":
        return "raw/patients.anchor_age,anchor_year + cohort/admissions.admittime"
    if concept == "context:male":
        return "raw/patients.gender"
    if concept.startswith("context:prior_"):
        return "earlier eligible cohort/admissions + raw/diagnoses_icd"
    if concept.startswith("chronic_comorbidity:"):
        return "raw/diagnoses_icd current and prior eligible admissions"
    if concept.startswith(("acute_diagnosis:", "aecopd_index_code:")):
        return "raw/diagnoses_icd current admission"
    if concept.startswith("disposition:"):
        return "cohort/admissions.discharge_location"
    raise ValueError(f"unknown static concept source: {concept}")


def audit(raw_root: Path, processed_root: Path, structured_tokens: Path,
          shared_text: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    raw_manifest = json.loads((raw_root / "metadata/manifest.json").read_text(encoding="utf-8"))
    if raw_manifest["status"] != "validated":
        raise ValueError("raw ledger is not validated")
    processed_manifest = json.loads((processed_root / "manifest.json").read_text(encoding="utf-8"))
    if processed_manifest["status"] != "preprocessed_and_checked":
        raise ValueError("processed event set is not validated")
    expected = processed_manifest["output_sha256"]["events.csv.gz"]
    if sha256(processed_root / "events.csv.gz") != expected:
        raise ValueError("processed events changed")
    patient_splits = {r["subject_id"]: r["patient_split"]
                      for r in rows(raw_root / "cohort/patient_splits.csv.gz")}
    eligible_records = {r["hadm_id"]: r for r in rows(raw_root / "cohort/admissions.csv.gz")
                        if r["discharge_landmark_eligible"] == "t"}
    eligible = {hadm: (r["subject_id"], patient_splits[r["subject_id"]])
                for hadm, r in eligible_records.items()}
    if len(eligible) != 19401:
        raise ValueError("unexpected discharge cohort")
    patients = {subject for subject, _ in eligible.values()}

    profiles: dict[str, dict[str, object]] = defaultdict(lambda: {
        "events": 0, "patients": set(), "admissions": set(), "train_admissions": set(),
        "sources": set(), "same_time_collision": 0, "same_source_collision": 0,
        "numeric_present": 0, "units": Counter(),
    })
    seen_time: set[tuple[str, str, str]] = set()
    seen_source: set[tuple[str, str]] = set()
    med_source_concepts: dict[str, set[str]] = defaultdict(set)
    for r in rows(processed_root / "events.csv.gz"):
        hadm, concept = r["hadm_id"], r["concept"]
        if hadm not in eligible or concept not in processed_manifest["concepts"]:
            raise ValueError("processed event outside frozen cohort/dictionary")
        subject, split = eligible[hadm]
        if r["subject_id"] != subject or r["patient_split"] != split:
            raise ValueError("processed event patient/split mismatch")
        p = profiles[concept]
        p["events"] += 1
        p["patients"].add(subject)
        p["admissions"].add(hadm)
        if split == "train":
            p["train_admissions"].add(hadm)
        p["sources"].add(r["source_id"])
        p["numeric_present"] += r["numeric_mask"] == "1"
        p["units"][r["value_unit"]] += 1
        time_key = (hadm, concept, r["charttime"])
        if time_key in seen_time:
            p["same_time_collision"] += 1
        seen_time.add(time_key)
        source_key = (concept, r["source_id"])
        if source_key in seen_source:
            p["same_source_collision"] += 1
        seen_source.add(source_key)
        if r["kind"] == "medication":
            med_source_concepts[r["source_id"]].add(concept)

    model_rows = []
    for concept in processed_manifest["concepts"]:
        p = profiles[concept]
        family = _family(concept)
        is_medication = family not in {"ph", "pco2", "po2", "bicarbonate", "wbc",
                                       "hemoglobin", "creatinine", "albumin"}
        model_rows.append({
            "concept": concept, "kind": "medication" if is_medication else "laboratory",
            "source": "raw/emar confirmed administrations + raw/emar_detail product/route" if is_medication else
                f"raw/labevents.itemid={LAB_ITEM_FOR_CONCEPT[family]}" +
                (" + specimen itemid=52033" if family in {"ph", "pco2", "po2"} else ""),
            "events": p["events"], "patients": len(p["patients"]),
            "admissions": len(p["admissions"]), "admission_coverage_pct": round(100 * len(p["admissions"]) / len(eligible), 2),
            "unobserved_admissions": len(eligible) - len(p["admissions"]),
            "train_admissions": len(p["train_admissions"]),
            "same_timestamp_collisions_not_proven_duplicates": p["same_time_collision"],
            "repeated_source_ids": p["same_source_collision"],
            "numeric_observations": p["numeric_present"],
            "unit_counts": json.dumps(p["units"], ensure_ascii=False),
            "clinical_rationale": RATIONALE.get(family, "需人工补充临床理由"),
        })

    lab_dictionary = {r["itemid"]: r for r in rows(raw_root / "raw/d_labitems.csv.gz")}
    lab_profiles: dict[str, dict[str, object]] = defaultdict(lambda: {
        "rows": 0, "numeric": 0, "admissions": set(), "patients": set(),
        "numeric_admissions": set(), "numeric_patients": set(), "units": Counter(),
        "same_time_collision": 0, "source_collision": 0})
    reviewed_times: set[tuple[str, str, str]] = set()
    reviewed_ids: set[str] = set()
    for r in rows(raw_root / "raw/labevents.csv.gz"):
        admission = eligible_records.get(r["hadm_id"])
        if admission is None or not (admission["admittime"] <= r["charttime"] <= r["storetime"] <= admission["dischtime"]):
            continue
        p = lab_profiles[r["itemid"]]
        p["rows"] += 1
        is_numeric = r["valuenum"] not in (NULL, "")
        p["numeric"] += is_numeric
        p["admissions"].add(r["hadm_id"])
        p["patients"].add(r["subject_id"])
        if is_numeric:
            p["numeric_admissions"].add(r["hadm_id"])
            p["numeric_patients"].add(r["subject_id"])
        p["units"][r["valueuom"]] += 1
        if r["itemid"] in REVIEW_LABS:
            time_key = (r["hadm_id"], r["itemid"], r["charttime"])
            p["same_time_collision"] += time_key in reviewed_times
            p["source_collision"] += r["labevent_id"] in reviewed_ids
            reviewed_times.add(time_key)
            reviewed_ids.add(r["labevent_id"])
    raw_labs = []
    for itemid, p in sorted(lab_profiles.items(), key=lambda x: -x[1]["rows"]):
        dictionary = lab_dictionary[itemid]
        raw_labs.append({
            "itemid": itemid, "label": dictionary["label"], "fluid": dictionary["fluid"],
            "category": dictionary["category"], "review_status":
                "current_input" if itemid in CURRENT_LABS else
                "clinical_candidate_not_in_input" if itemid in REVIEW_LABS else "unreviewed",
            "review_reason": REVIEW_LABS.get(itemid, ""), "raw_rows": p["rows"],
            "numeric_rows": p["numeric"], "patients": len(p["patients"]),
            "admissions": len(p["admissions"]),
            "numeric_patients": len(p["numeric_patients"]),
            "numeric_admissions": len(p["numeric_admissions"]),
            "numeric_unobserved_admissions": len(eligible) - len(p["numeric_admissions"]),
            "same_timestamp_collisions_not_proven_duplicates":
                p["same_time_collision"] if itemid in REVIEW_LABS else "",
            "repeated_source_ids": p["source_collision"] if itemid in REVIEW_LABS else "",
            "admission_coverage_pct": round(100 * len(p["admissions"]) / len(eligible), 2),
            "unit_counts": json.dumps(p["units"].most_common(5), ensure_ascii=False),
        })

    drug_profiles: dict[str, dict[str, object]] = defaultdict(lambda: {
        "rows": 0, "confirmed": 0, "accepted_sources": 0,
        "admissions": set(), "patients": set(), "mapped": set(), "actual_classes": set()})
    selected_emar_names: dict[str, str] = {}
    for r in rows(raw_root / "raw/emar.csv.gz"):
        hadm = r["hadm_id"]
        admission = eligible_records.get(hadm)
        if admission is None or not (admission["admittime"] <= r["charttime"] <= r["storetime"] <= admission["dischtime"]):
            continue
        name = r["medication"].strip()
        p = drug_profiles[name]
        p["rows"] += 1
        p["admissions"].add(hadm)
        p["patients"].add(r["subject_id"])
        p["mapped"].update(medication_classes(name) if name != NULL else ())
        p["confirmed"] += r["event_txt"] in {"Administered", "Started", "Applied", "Delayed Administered"}
        key = r["emar_id"] + ":" + r["emar_seq"]
        if key in med_source_concepts:
            selected_emar_names[key] = name
            p["accepted_sources"] += 1
            p["actual_classes"].update(med_source_concepts[key])
    if len(selected_emar_names) != len(med_source_concepts):
        raise ValueError("processed medication sources not linked to raw eMAR")
    drug_rows = []
    for name, p in sorted(drug_profiles.items(), key=lambda x: -x[1]["confirmed"]):
        brand_hit = any(brand in name.lower() for brand in BRAND_REVIEW)
        drug_rows.append({"medication_name": name, "emar_rows": p["rows"],
                          "confirmed_rows": p["confirmed"], "patients": len(p["patients"]),
                          "admissions": len(p["admissions"]), "name_only_classes": "|".join(sorted(p["mapped"])),
                          "accepted_emar_sources": p["accepted_sources"],
                          "actual_in_model_classes": "|".join(sorted(p["actual_classes"])),
                          "brand_review": int(brand_hit),
                          "potential_missing_brand": int(brand_hit and not p["actual_classes"])})

    source_routes: dict[str, set[str]] = defaultdict(set)
    for r in rows(raw_root / "raw/emar_detail.csv.gz"):
        key = r["emar_id"] + ":" + r["emar_seq"]
        if key in med_source_concepts and r["route"] not in (NULL, ""):
            source_routes[key].add(r["route"].strip().upper())
    route_by_concept: dict[str, Counter[str]] = defaultdict(Counter)
    name_by_concept: dict[str, Counter[str]] = defaultdict(Counter)
    for key, concepts in med_source_concepts.items():
        route = "|".join(sorted(source_routes[key])) if source_routes[key] else "UNKNOWN"
        for concept in concepts:
            route_by_concept[concept][route] += 1
            name_by_concept[concept][selected_emar_names[key]] += 1
    for record in model_rows:
        if record["concept"] in route_by_concept:
            record["routes_top10"] = json.dumps(route_by_concept[record["concept"]].most_common(10), ensure_ascii=False)
            record["drug_names_top10"] = json.dumps(name_by_concept[record["concept"]].most_common(10), ensure_ascii=False)
        else:
            record["routes_top10"] = ""
            record["drug_names_top10"] = ""

    static_profiles: dict[str, dict[str, object]] = defaultdict(lambda: {
        "rows": 0, "admissions": set(), "patients": set(), "positive_current": 0, "positive_prior": 0})
    for r in rows(structured_tokens):
        if r["hadm_id"] not in eligible:
            continue
        if r["token_type"] in {"laboratory", "administered_medication"}:
            continue
        key = r["token_type"] + ":" + r["concept"]
        p = static_profiles[key]
        p["rows"] += 1
        p["admissions"].add(r["hadm_id"])
        p["patients"].add(r["subject_id"])
        p["positive_current"] += r["mask0"] == "1" and float(r["value0"]) > 0
        p["positive_prior"] += r["mask1"] == "1" and float(r["value1"]) > 0
    static_rows = [{"concept": name, "source": _static_source(name),
                    "rows": p["rows"], "patients": len(p["patients"]),
                    "admissions": len(p["admissions"]), "unobserved_admissions": len(eligible) - len(p["admissions"]),
                    "positive_current_rows": p["positive_current"], "positive_prior_rows": p["positive_prior"],
                    "duplicate_same_admission_rows": p["rows"] - len(p["admissions"]),
                    "clinical_rationale": STATIC_RATIONALE.get(name, "需人工复核")}
                   for name, p in sorted(static_profiles.items())]

    text_profiles: dict[str, dict[str, object]] = defaultdict(lambda: {
        "chunks": 0, "admissions": set(), "patients": set()})
    for r in rows(shared_text):
        if r["hadm_id"] not in eligible:
            raise ValueError("text field outside frozen cohort")
        p = text_profiles[r["field"]]
        p["chunks"] += 1
        p["admissions"].add(r["hadm_id"])
        p["patients"].add(r["subject_id"])
    text_rows = [{"field": name, "source": "shared_text + discharge note field extraction",
                  "chunks": p["chunks"], "patients": len(p["patients"]),
                  "admissions": len(p["admissions"]), "unobserved_admissions": len(eligible) - len(p["admissions"]),
                  "multiple_chunks_above_one_per_admission": p["chunks"] - len(p["admissions"]),
                  "clinical_rationale": TEXT_RATIONALE.get(name, "需人工复核")}
                 for name, p in sorted(text_profiles.items())]

    output.mkdir(parents=True)
    _write_csv(output / "current_event_concepts.csv", tuple(model_rows[0]), model_rows)
    _write_csv(output / "all_raw_lab_items.csv", tuple(raw_labs[0]), raw_labs)
    _write_csv(output / "all_emar_medication_names.csv", tuple(drug_rows[0]), drug_rows)
    _write_csv(output / "current_static_concepts.csv", tuple(static_rows[0]), static_rows)
    _write_csv(output / "current_text_fields.csv", tuple(text_rows[0]), text_rows)
    summary = {"status": "audited", "eligible_admissions": len(eligible), "eligible_patients": len(patients),
               "current_event_concepts": len(model_rows), "raw_lab_items_with_rows": len(raw_labs),
               "raw_medication_names": len(drug_rows), "current_static_concepts": len(static_rows),
               "current_text_fields": len(text_rows),
               "unmapped_brand_names": [r for r in drug_rows if r["potential_missing_brand"] and r["confirmed_rows"] >= 10],
               "inputs_sha256": {"raw_manifest": sha256(raw_root / "metadata/manifest.json"),
                                 "processed_manifest": sha256(processed_root / "manifest.json"),
                                 "structured_tokens": sha256(structured_tokens), "shared_text": sha256(shared_text)}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw_root", "processed_root", "structured_tokens", "shared_text", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.raw_root, args.processed_root, args.structured_tokens, args.shared_text, args.output)
    print(json.dumps({key: result[key] for key in ("status", "eligible_admissions", "current_event_concepts",
                                                    "raw_lab_items_with_rows", "raw_medication_names")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
