# 06_LOADER_AND_CONTRACTS · 1

源文件正文或经过字段白名单提取的汇总数据，来源身份见每段元数据。

## config/clinical_feature_contract.csv · lines 1–111

<!-- REVIEW_FRAGMENT {"path":"config/clinical_feature_contract.csv","part":1,"start_line":1,"end_line":111,"source_sha256":"b4942c2196077b5d5fd6007d4389a6ee47c28a54f519037e3c25a7a6bef59940","fragment_sha256":"b4942c2196077b5d5fd6007d4389a6ee47c28a54f519037e3c25a7a6bef59940","added_terminal_newline":false} -->
````csv
rule_id,source_domain,source_concept,model_scope,canonical_concept,feature_layer,value_type,keep,reason
STATE_PH,检验,血液酸碱度,main_model,血液酸碱度,patient_state,numeric_trajectory,1,呼吸衰竭状态
STATE_PCO2,检验,二氧化碳分压,main_model,二氧化碳分压,patient_state,numeric_trajectory,1,呼吸衰竭状态
STATE_PO2,检验,氧分压,main_model,氧分压,patient_state,numeric_trajectory,1,呼吸衰竭状态
STATE_BLOOD_GAS_SAT,检验,血气氧饱和度,main_model,血气氧饱和度,patient_state,numeric_trajectory,1,呼吸衰竭状态
STATE_BASE_EXCESS,检验,碱剩余,main_model,碱剩余,patient_state,numeric_trajectory,1,呼吸衰竭状态
STATE_BICARBONATE,检验,碳酸氢盐,main_model,碳酸氢盐,patient_state,numeric_trajectory,1,酸碱与代谢状态
STATE_WBC,检验,白细胞计数,main_model,白细胞计数,patient_state,numeric_trajectory,1,感染与炎症状态
STATE_NEUTROPHIL,检验,中性粒细胞,main_model,中性粒细胞,patient_state,numeric_trajectory,1,感染与炎症状态
STATE_LYMPHOCYTE,检验,淋巴细胞,main_model,淋巴细胞,patient_state,numeric_trajectory,1,感染与炎症状态
STATE_EOSINOPHIL,检验,嗜酸性粒细胞,main_model,嗜酸性粒细胞,patient_state,numeric_trajectory,1,COPD炎症表型
STATE_HEMOGLOBIN,检验,血红蛋白,main_model,血红蛋白,patient_state,numeric_trajectory,1,携氧能力
STATE_HEMATOCRIT,检验,红细胞压积,main_model,红细胞压积,patient_state,numeric_trajectory,1,携氧与容量状态
STATE_PLATELET,检验,血小板计数,main_model,血小板计数,patient_state,numeric_trajectory,1,血液状态
STATE_SODIUM,检验,血钠,main_model,血钠,patient_state,numeric_trajectory,1,电解质状态
STATE_POTASSIUM,检验,血钾,main_model,血钾,patient_state,numeric_trajectory,1,电解质状态
STATE_CHLORIDE,检验,血氯,main_model,血氯,patient_state,numeric_trajectory,1,电解质状态
STATE_MAGNESIUM,检验,血镁,main_model,血镁,patient_state,numeric_trajectory,1,电解质状态
STATE_CALCIUM,检验,总钙,main_model,总钙,patient_state,numeric_trajectory,1,电解质状态
STATE_CREATININE,检验,肌酐,main_model,肌酐,patient_state,numeric_trajectory,1,肾功能
STATE_BUN,检验,尿素氮,main_model,尿素氮,patient_state,numeric_trajectory,1,肾功能
STATE_EGFR,检验,估算肾小球滤过率,main_model,估算肾小球滤过率,patient_state,numeric_trajectory,1,肾功能
STATE_GLUCOSE,检验,血糖,main_model,血糖,patient_state,numeric_trajectory,1,代谢状态
STATE_ALBUMIN,检验,白蛋白,main_model,白蛋白,patient_state,numeric_trajectory,1,营养与全身状态
STATE_AST,检验,天冬氨酸氨基转移酶,main_model,天冬氨酸氨基转移酶,patient_state,numeric_trajectory,1,肝功能
STATE_ALT,检验,丙氨酸氨基转移酶,main_model,丙氨酸氨基转移酶,patient_state,numeric_trajectory,1,肝功能
STATE_BILIRUBIN,检验,总胆红素,main_model,总胆红素,patient_state,numeric_trajectory,1,肝功能
STATE_LACTATE,检验,乳酸,main_model,乳酸,patient_state,numeric_trajectory,1,组织灌注
STATE_CRP,检验,C反应蛋白,main_model,C反应蛋白,patient_state,numeric_trajectory,1,炎症状态
STATE_BNP,检验,B型利钠肽,main_model,B型利钠肽,patient_state,numeric_trajectory,1,心衰状态
STATE_TROPONIN,检验,肌钙蛋白,main_model,肌钙蛋白,patient_state,numeric_trajectory,1,心肌损伤
STATE_URINE_OUTPUT,ICU输出,尿量,main_model,尿量,patient_state,numeric_trajectory,1,容量与肾功能
STATE_RESP_CULTURE,微生物,呼吸道标本培养,main_model,呼吸道标本培养,patient_state,clinical_state,1,感染状态
STATE_BLOOD_CULTURE,微生物,血培养,main_model,血培养,patient_state,clinical_state,1,感染状态
STATE_CULTURE_POSITIVE,微生物,培养是否阳性,main_model,培养是否阳性,patient_state,clinical_state,1,感染状态
STATE_PATHOGEN,微生物,病原体鉴定,main_model,病原体鉴定,patient_state,clinical_state,1,感染状态
STATE_MICRO_VIRAL,微生物,病毒核酸或抗原检测,main_model,病毒核酸或抗原检测,patient_state,clinical_state,1,感染状态
STATE_RESISTANCE,微生物,耐药菌标志,main_model,耐药菌标志,patient_state,clinical_state,1,感染状态
DX_SMOKING,诊断,吸烟相关诊断,main_model,吸烟相关诊断,patient_state,clinical_state,1,COPD风险背景
DX_ACUTE_RESP_FAILURE,诊断,急性呼吸衰竭,main_model,急性呼吸衰竭,patient_state,clinical_state,1,呼吸严重度
DX_CHRONIC_RESP_FAILURE,诊断,慢性呼吸衰竭,main_model,慢性呼吸衰竭,patient_state,clinical_state,1,呼吸基础状态
DX_HYPOXEMIA,诊断,低氧血症,main_model,低氧血症,patient_state,clinical_state,1,呼吸状态
DX_LONG_TERM_OXYGEN,诊断,长期氧疗依赖,main_model,长期氧疗依赖,patient_state,clinical_state,1,呼吸基础状态
DX_PNEUMONIA,诊断,肺炎,main_model,肺炎,patient_state,clinical_state,1,感染共病
DX_ASPIRATION,诊断,误吸相关疾病,main_model,误吸相关疾病,patient_state,clinical_state,1,感染风险
DX_ASTHMA,诊断,哮喘,main_model,哮喘,patient_state,clinical_state,1,呼吸共病
DX_BRONCHIECTASIS,诊断,支气管扩张症,main_model,支气管扩张症,patient_state,clinical_state,1,呼吸共病
DX_EMPHYSEMA,诊断,肺气肿,main_model,肺气肿,patient_state,clinical_state,1,呼吸共病
DX_OSA,诊断,阻塞性睡眠呼吸暂停,main_model,阻塞性睡眠呼吸暂停,patient_state,clinical_state,1,呼吸共病
DX_CHF,诊断,充血性心力衰竭,main_model,充血性心力衰竭,patient_state,clinical_state,1,心血管共病
DX_IHD,诊断,缺血性心脏病,main_model,缺血性心脏病,patient_state,clinical_state,1,心血管共病
DX_AF,诊断,心房颤动或扑动,main_model,心房颤动或扑动,patient_state,clinical_state,1,心血管共病
DX_PULM_HTN,诊断,肺动脉高压,main_model,肺动脉高压,patient_state,clinical_state,1,心肺共病
DX_VTE,诊断,静脉血栓栓塞,main_model,静脉血栓栓塞,patient_state,clinical_state,1,血栓共病
DX_PE,诊断,肺栓塞,main_model,肺栓塞,patient_state,clinical_state,1,血栓共病
DX_DIABETES,诊断,糖尿病,main_model,糖尿病,patient_state,clinical_state,1,代谢共病
DX_AKI,诊断,急性肾损伤,main_model,急性肾损伤,patient_state,clinical_state,1,肾脏状态
DX_CKD,诊断,慢性肾脏病,main_model,慢性肾脏病,patient_state,clinical_state,1,肾脏共病
DX_SEPSIS,诊断,脓毒症,main_model,脓毒症,patient_state,clinical_state,1,全身严重感染
DX_OBESITY,诊断,肥胖,main_model,肥胖,patient_state,clinical_state,1,代谢共病
DX_MALNUTRITION,诊断,营养不良或低体重,main_model,营养不良或低体重,patient_state,clinical_state,1,营养状态
DX_ANEMIA,诊断,贫血,main_model,贫血,patient_state,clinical_state,1,血液共病
DX_LUNG_CANCER,诊断,肺癌,main_model,肺癌,patient_state,clinical_state,1,恶性肿瘤
DX_OTHER_CANCER,诊断,其他恶性肿瘤,main_model,其他恶性肿瘤,patient_state,clinical_state,1,恶性肿瘤
DX_IMMUNOSUPPRESSION,诊断,免疫抑制状态,main_model,免疫抑制状态,patient_state,clinical_state,1,感染风险
DX_CHRONIC_LIVER,诊断,肝硬化或慢性肝病,main_model,肝硬化或慢性肝病,patient_state,clinical_state,1,肝脏共病
STATE_VIRAL_TEST,检验,病毒核酸或抗原检测,main_model,病毒核酸或抗原检测,patient_state,clinical_state,1,跨检验与微生物来源统一为状态概念
PROCESS_OXYGEN,检验,氧疗方式与氧流量,main_model,氧疗,care_process,binary_exposure,1,呼吸支持
PROCESS_INVASIVE_VENT_LAB,检验,有创机械通气,main_model,有创机械通气,care_process,binary_exposure,1,呼吸支持
PROCESS_NIV,ICU操作,无创通气,main_model,无创通气,care_process,binary_exposure,1,呼吸支持
PROCESS_INVASIVE_VENT,ICU操作,有创机械通气,main_model,有创机械通气,care_process,binary_exposure,1,呼吸支持
PROCESS_INTUBATION,ICU操作,气管插管,main_model,有创机械通气,care_process,binary_exposure,1,呼吸支持
PROCESS_EXTUBATION,ICU操作,拔除气管插管,main_model,拔除气管插管,care_process,binary_exposure,1,呼吸支持降级
PROCESS_DIALYSIS,ICU操作,血液透析或连续肾脏替代,main_model,血液透析或连续肾脏替代,care_process,binary_exposure,1,器官支持
MED_SABA,药物,短效β2受体激动剂,main_model,支气管扩张剂,care_process,binary_exposure,1,AECOPD治疗
MED_SAMA,药物,短效抗胆碱药,main_model,支气管扩张剂,care_process,binary_exposure,1,AECOPD治疗
MED_SABA_SAMA,药物,SABA与SAMA复方,main_model,支气管扩张剂,care_process,binary_exposure,1,AECOPD治疗
MED_LABA,药物,长效β2受体激动剂,main_model,支气管扩张剂,care_process,binary_exposure,1,AECOPD治疗
MED_LAMA,药物,长效抗胆碱药,main_model,支气管扩张剂,care_process,binary_exposure,1,AECOPD治疗
MED_ICS_LABA_BRONCHO,药物,ICS/LABA复方,main_model,支气管扩张剂,care_process,binary_exposure,1,复方吸入治疗
MED_ICS_LABA_ICS,药物,ICS/LABA复方,main_model,吸入糖皮质激素,care_process,binary_exposure,1,复方吸入治疗
MED_TRIPLE_BRONCHO,药物,三联吸入治疗,main_model,支气管扩张剂,care_process,binary_exposure,1,三联吸入治疗
MED_TRIPLE_ICS,药物,三联吸入治疗,main_model,吸入糖皮质激素,care_process,binary_exposure,1,三联吸入治疗
MED_ICS,药物,吸入糖皮质激素,main_model,吸入糖皮质激素,care_process,binary_exposure,1,AECOPD治疗
MED_PREDNISONE,药物,泼尼松或泼尼松龙,main_model,全身糖皮质激素,care_process,binary_exposure,1,AECOPD治疗
MED_METHYLPRED,药物,甲泼尼龙,main_model,全身糖皮质激素,care_process,binary_exposure,1,AECOPD治疗
MED_DEXAMETHASONE,药物,地塞米松,main_model,全身糖皮质激素,care_process,binary_exposure,1,AECOPD治疗
MED_HYDROCORTISONE,药物,氢化可的松,main_model,全身糖皮质激素,care_process,binary_exposure,1,AECOPD治疗
MED_MACROLIDE,药物,大环内酯类抗菌药,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_ANTIPSEUDOMONAL,药物,抗假单胞菌β内酰胺,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_VANCOMYCIN,药物,万古霉素,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_CEPHALOSPORIN,药物,普通头孢菌素,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_FLUOROQUINOLONE,药物,氟喹诺酮类,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_NITROIMIDAZOLE,药物,硝基咪唑类抗菌药,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_TMP_SMX,药物,磺胺甲噁唑/甲氧苄啶,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_PENICILLIN_BLI,药物,青霉素类和β内酰胺酶抑制剂,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_TETRACYCLINE,药物,四环素类,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_CARBAPENEM,药物,碳青霉烯类,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_LOOP_DIURETIC,药物,袢利尿剂,main_model,利尿剂,care_process,binary_exposure,1,容量管理
MED_OTHER_DIURETIC,药物,其他利尿剂,main_model,利尿剂,care_process,binary_exposure,1,容量管理
ICU_MACROLIDE,ICU输入,大环内酯类抗菌药,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_ANTIPSEUDOMONAL,ICU输入,抗假单胞菌β内酰胺,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_VANCOMYCIN,ICU输入,万古霉素,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_CEPHALOSPORIN,ICU输入,普通头孢菌素,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_FLUOROQUINOLONE,ICU输入,氟喹诺酮类,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_NITROIMIDAZOLE,ICU输入,硝基咪唑类抗菌药,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_TMP_SMX,ICU输入,磺胺甲噁唑/甲氧苄啶,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_PENICILLIN_BLI,ICU输入,青霉素类和β内酰胺酶抑制剂,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_TETRACYCLINE,ICU输入,四环素类,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_CARBAPENEM,ICU输入,碳青霉烯类,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_LOOP_DIURETIC,ICU输入,袢利尿剂,main_model,利尿剂,care_process,binary_exposure,1,跨来源容量管理
````
<!-- END REVIEW_FRAGMENT -->

## config/clinical_feature_contract_v2_diagnosis.csv · lines 1–111

<!-- REVIEW_FRAGMENT {"path":"config/clinical_feature_contract_v2_diagnosis.csv","part":1,"start_line":1,"end_line":111,"source_sha256":"dd7403d81071e7a22f202d2ebee3b64788572840af6fad647344d8d781ad023d","fragment_sha256":"dd7403d81071e7a22f202d2ebee3b64788572840af6fad647344d8d781ad023d","added_terminal_newline":false} -->
````csv
rule_id,source_domain,source_concept,model_scope,canonical_concept,feature_layer,value_type,keep,reason
STATE_PH,检验,血液酸碱度,main_model,血液酸碱度,patient_state,numeric_trajectory,1,呼吸衰竭状态
STATE_PCO2,检验,二氧化碳分压,main_model,二氧化碳分压,patient_state,numeric_trajectory,1,呼吸衰竭状态
STATE_PO2,检验,氧分压,main_model,氧分压,patient_state,numeric_trajectory,1,呼吸衰竭状态
STATE_BLOOD_GAS_SAT,检验,血气氧饱和度,main_model,血气氧饱和度,patient_state,numeric_trajectory,1,呼吸衰竭状态
STATE_BASE_EXCESS,检验,碱剩余,main_model,碱剩余,patient_state,numeric_trajectory,1,呼吸衰竭状态
STATE_BICARBONATE,检验,碳酸氢盐,main_model,碳酸氢盐,patient_state,numeric_trajectory,1,酸碱与代谢状态
STATE_WBC,检验,白细胞计数,main_model,白细胞计数,patient_state,numeric_trajectory,1,感染与炎症状态
STATE_NEUTROPHIL,检验,中性粒细胞,main_model,中性粒细胞,patient_state,numeric_trajectory,1,感染与炎症状态
STATE_LYMPHOCYTE,检验,淋巴细胞,main_model,淋巴细胞,patient_state,numeric_trajectory,1,感染与炎症状态
STATE_EOSINOPHIL,检验,嗜酸性粒细胞,main_model,嗜酸性粒细胞,patient_state,numeric_trajectory,1,COPD炎症表型
STATE_HEMOGLOBIN,检验,血红蛋白,main_model,血红蛋白,patient_state,numeric_trajectory,1,携氧能力
STATE_HEMATOCRIT,检验,红细胞压积,main_model,红细胞压积,patient_state,numeric_trajectory,1,携氧与容量状态
STATE_PLATELET,检验,血小板计数,main_model,血小板计数,patient_state,numeric_trajectory,1,血液状态
STATE_SODIUM,检验,血钠,main_model,血钠,patient_state,numeric_trajectory,1,电解质状态
STATE_POTASSIUM,检验,血钾,main_model,血钾,patient_state,numeric_trajectory,1,电解质状态
STATE_CHLORIDE,检验,血氯,main_model,血氯,patient_state,numeric_trajectory,1,电解质状态
STATE_MAGNESIUM,检验,血镁,main_model,血镁,patient_state,numeric_trajectory,1,电解质状态
STATE_CALCIUM,检验,总钙,main_model,总钙,patient_state,numeric_trajectory,1,电解质状态
STATE_CREATININE,检验,肌酐,main_model,肌酐,patient_state,numeric_trajectory,1,肾功能
STATE_BUN,检验,尿素氮,main_model,尿素氮,patient_state,numeric_trajectory,1,肾功能
STATE_EGFR,检验,估算肾小球滤过率,main_model,估算肾小球滤过率,patient_state,numeric_trajectory,1,肾功能
STATE_GLUCOSE,检验,血糖,main_model,血糖,patient_state,numeric_trajectory,1,代谢状态
STATE_ALBUMIN,检验,白蛋白,main_model,白蛋白,patient_state,numeric_trajectory,1,营养与全身状态
STATE_AST,检验,天冬氨酸氨基转移酶,main_model,天冬氨酸氨基转移酶,patient_state,numeric_trajectory,1,肝功能
STATE_ALT,检验,丙氨酸氨基转移酶,main_model,丙氨酸氨基转移酶,patient_state,numeric_trajectory,1,肝功能
STATE_BILIRUBIN,检验,总胆红素,main_model,总胆红素,patient_state,numeric_trajectory,1,肝功能
STATE_LACTATE,检验,乳酸,main_model,乳酸,patient_state,numeric_trajectory,1,组织灌注
STATE_CRP,检验,C反应蛋白,main_model,C反应蛋白,patient_state,numeric_trajectory,1,炎症状态
STATE_BNP,检验,B型利钠肽,main_model,B型利钠肽,patient_state,numeric_trajectory,1,心衰状态
STATE_TROPONIN,检验,肌钙蛋白,main_model,肌钙蛋白,patient_state,numeric_trajectory,1,心肌损伤
STATE_URINE_OUTPUT,ICU输出,尿量,main_model,尿量,patient_state,numeric_trajectory,1,容量与肾功能
STATE_RESP_CULTURE,微生物,呼吸道标本培养,main_model,呼吸道标本培养,patient_state,clinical_state,1,感染状态
STATE_BLOOD_CULTURE,微生物,血培养,main_model,血培养,patient_state,clinical_state,1,感染状态
STATE_CULTURE_POSITIVE,微生物,培养是否阳性,main_model,培养是否阳性,patient_state,clinical_state,1,感染状态
STATE_PATHOGEN,微生物,病原体鉴定,main_model,病原体鉴定,patient_state,clinical_state,1,感染状态
STATE_MICRO_VIRAL,微生物,病毒核酸或抗原检测,main_model,病毒核酸或抗原检测,patient_state,clinical_state,1,感染状态
STATE_RESISTANCE,微生物,耐药菌标志,main_model,耐药菌标志,patient_state,clinical_state,1,感染状态
DX_SMOKING,诊断,吸烟相关诊断,main_model,吸烟相关状态,patient_state,clinical_state,1,长期风险背景，非共病
DX_ACUTE_RESP_FAILURE,诊断,急性呼吸衰竭,main_model,本次住院呼吸功能异常,patient_state,clinical_state,1,本次住院病情，非慢性共病
DX_CHRONIC_RESP_FAILURE,诊断,慢性呼吸衰竭,main_model,慢性呼吸功能负担,patient_state,clinical_state,1,出院时已知慢性呼吸状态
DX_HYPOXEMIA,诊断,低氧血症,main_model,本次住院呼吸功能异常,patient_state,clinical_state,1,本次住院病情，非慢性共病
DX_LONG_TERM_OXYGEN,诊断,长期氧疗依赖,main_model,慢性呼吸功能负担,patient_state,clinical_state,1,出院时已知长期氧疗状态
DX_PNEUMONIA,诊断,肺炎,main_model,本次住院肺部感染或误吸,patient_state,clinical_state,1,本次住院急性病情
DX_ASPIRATION,诊断,误吸相关疾病,main_model,本次住院肺部感染或误吸,patient_state,clinical_state,1,本次住院急性病情
DX_ASTHMA,诊断,哮喘,main_model,其他慢性呼吸疾病,patient_state,clinical_state,1,出院时已知慢性共病
DX_BRONCHIECTASIS,诊断,支气管扩张症,main_model,其他慢性呼吸疾病,patient_state,clinical_state,1,出院时已知慢性共病
DX_EMPHYSEMA,诊断,肺气肿,main_model,肺气肿,patient_state,clinical_state,0,COPD表型与入组疾病高度重叠
DX_OSA,诊断,阻塞性睡眠呼吸暂停,main_model,其他慢性呼吸疾病,patient_state,clinical_state,1,出院时已知慢性共病
DX_CHF,诊断,充血性心力衰竭,main_model,心力衰竭,patient_state,clinical_state,1,保留重要独立心脏共病
DX_IHD,诊断,缺血性心脏病,main_model,其他心血管疾病,patient_state,clinical_state,1,出院时已知慢性共病
DX_AF,诊断,心房颤动或扑动,main_model,其他心血管疾病,patient_state,clinical_state,1,出院时已知慢性共病
DX_PULM_HTN,诊断,肺动脉高压,main_model,其他心血管疾病,patient_state,clinical_state,1,出院时已知慢性共病
DX_VTE,诊断,静脉血栓栓塞,main_model,静脉血栓栓塞,patient_state,clinical_state,0,原概念混合急性与慢性编码，暂不入模
DX_PE,诊断,肺栓塞,main_model,肺栓塞,patient_state,clinical_state,0,原概念混合当前与既往编码，暂不入模
DX_DIABETES,诊断,糖尿病,main_model,糖尿病,patient_state,clinical_state,1,出院时已知慢性共病
DX_AKI,诊断,急性肾损伤,main_model,本次住院急性肾损伤,patient_state,clinical_state,1,本次住院急性病情
DX_CKD,诊断,慢性肾脏病,main_model,慢性肾脏病,patient_state,clinical_state,1,出院时已知慢性共病
DX_SEPSIS,诊断,脓毒症,main_model,本次住院脓毒症,patient_state,clinical_state,1,本次住院急性病情
DX_OBESITY,诊断,肥胖,main_model,肥胖,patient_state,clinical_state,0,与BMI及体重状态重复
DX_MALNUTRITION,诊断,营养不良或低体重,main_model,营养不良或低体重,patient_state,clinical_state,0,与体重BMI及白蛋白状态重复且时间属性不明
DX_ANEMIA,诊断,贫血,main_model,贫血,patient_state,clinical_state,0,与血红蛋白状态重复且急慢性混合
DX_LUNG_CANCER,诊断,肺癌,main_model,恶性肿瘤,patient_state,clinical_state,1,出院时已知慢性共病
DX_OTHER_CANCER,诊断,其他恶性肿瘤,main_model,恶性肿瘤,patient_state,clinical_state,1,出院时已知慢性共病
DX_IMMUNOSUPPRESSION,诊断,免疫抑制状态,main_model,免疫抑制状态,patient_state,clinical_state,0,原概念混合HIV移植与药源性免疫抑制，暂不入模
DX_CHRONIC_LIVER,诊断,肝硬化或慢性肝病,main_model,慢性肝病,patient_state,clinical_state,1,出院时已知慢性共病
STATE_VIRAL_TEST,检验,病毒核酸或抗原检测,main_model,病毒核酸或抗原检测,patient_state,clinical_state,1,跨检验与微生物来源统一为状态概念
PROCESS_OXYGEN,检验,氧疗方式与氧流量,main_model,氧疗,care_process,binary_exposure,1,呼吸支持
PROCESS_INVASIVE_VENT_LAB,检验,有创机械通气,main_model,有创机械通气,care_process,binary_exposure,1,呼吸支持
PROCESS_NIV,ICU操作,无创通气,main_model,无创通气,care_process,binary_exposure,1,呼吸支持
PROCESS_INVASIVE_VENT,ICU操作,有创机械通气,main_model,有创机械通气,care_process,binary_exposure,1,呼吸支持
PROCESS_INTUBATION,ICU操作,气管插管,main_model,有创机械通气,care_process,binary_exposure,1,呼吸支持
PROCESS_EXTUBATION,ICU操作,拔除气管插管,main_model,拔除气管插管,care_process,binary_exposure,1,呼吸支持降级
PROCESS_DIALYSIS,ICU操作,血液透析或连续肾脏替代,main_model,血液透析或连续肾脏替代,care_process,binary_exposure,1,器官支持
MED_SABA,药物,短效β2受体激动剂,main_model,支气管扩张剂,care_process,binary_exposure,1,AECOPD治疗
MED_SAMA,药物,短效抗胆碱药,main_model,支气管扩张剂,care_process,binary_exposure,1,AECOPD治疗
MED_SABA_SAMA,药物,SABA与SAMA复方,main_model,支气管扩张剂,care_process,binary_exposure,1,AECOPD治疗
MED_LABA,药物,长效β2受体激动剂,main_model,支气管扩张剂,care_process,binary_exposure,1,AECOPD治疗
MED_LAMA,药物,长效抗胆碱药,main_model,支气管扩张剂,care_process,binary_exposure,1,AECOPD治疗
MED_ICS_LABA_BRONCHO,药物,ICS/LABA复方,main_model,支气管扩张剂,care_process,binary_exposure,1,复方吸入治疗
MED_ICS_LABA_ICS,药物,ICS/LABA复方,main_model,吸入糖皮质激素,care_process,binary_exposure,1,复方吸入治疗
MED_TRIPLE_BRONCHO,药物,三联吸入治疗,main_model,支气管扩张剂,care_process,binary_exposure,1,三联吸入治疗
MED_TRIPLE_ICS,药物,三联吸入治疗,main_model,吸入糖皮质激素,care_process,binary_exposure,1,三联吸入治疗
MED_ICS,药物,吸入糖皮质激素,main_model,吸入糖皮质激素,care_process,binary_exposure,1,AECOPD治疗
MED_PREDNISONE,药物,泼尼松或泼尼松龙,main_model,全身糖皮质激素,care_process,binary_exposure,1,AECOPD治疗
MED_METHYLPRED,药物,甲泼尼龙,main_model,全身糖皮质激素,care_process,binary_exposure,1,AECOPD治疗
MED_DEXAMETHASONE,药物,地塞米松,main_model,全身糖皮质激素,care_process,binary_exposure,1,AECOPD治疗
MED_HYDROCORTISONE,药物,氢化可的松,main_model,全身糖皮质激素,care_process,binary_exposure,1,AECOPD治疗
MED_MACROLIDE,药物,大环内酯类抗菌药,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_ANTIPSEUDOMONAL,药物,抗假单胞菌β内酰胺,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_VANCOMYCIN,药物,万古霉素,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_CEPHALOSPORIN,药物,普通头孢菌素,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_FLUOROQUINOLONE,药物,氟喹诺酮类,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_NITROIMIDAZOLE,药物,硝基咪唑类抗菌药,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_TMP_SMX,药物,磺胺甲噁唑/甲氧苄啶,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_PENICILLIN_BLI,药物,青霉素类和β内酰胺酶抑制剂,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_TETRACYCLINE,药物,四环素类,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_CARBAPENEM,药物,碳青霉烯类,main_model,抗菌药物,care_process,binary_exposure,1,抗菌治疗
MED_LOOP_DIURETIC,药物,袢利尿剂,main_model,利尿剂,care_process,binary_exposure,1,容量管理
MED_OTHER_DIURETIC,药物,其他利尿剂,main_model,利尿剂,care_process,binary_exposure,1,容量管理
ICU_MACROLIDE,ICU输入,大环内酯类抗菌药,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_ANTIPSEUDOMONAL,ICU输入,抗假单胞菌β内酰胺,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_VANCOMYCIN,ICU输入,万古霉素,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_CEPHALOSPORIN,ICU输入,普通头孢菌素,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_FLUOROQUINOLONE,ICU输入,氟喹诺酮类,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_NITROIMIDAZOLE,ICU输入,硝基咪唑类抗菌药,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_TMP_SMX,ICU输入,磺胺甲噁唑/甲氧苄啶,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_PENICILLIN_BLI,ICU输入,青霉素类和β内酰胺酶抑制剂,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_TETRACYCLINE,ICU输入,四环素类,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_CARBAPENEM,ICU输入,碳青霉烯类,main_model,抗菌药物,care_process,binary_exposure,1,跨来源抗菌治疗
ICU_LOOP_DIURETIC,ICU输入,袢利尿剂,main_model,利尿剂,care_process,binary_exposure,1,跨来源容量管理
````
<!-- END REVIEW_FRAGMENT -->

## model_loader/aecopd_loader.py · lines 1–771

<!-- REVIEW_FRAGMENT {"path":"model_loader/aecopd_loader.py","part":1,"start_line":1,"end_line":771,"source_sha256":"a85b53d92a6f2d3d33a87eaf52d2bfcd99f7962bd6bd8ee079d5c1fcd1ce1ac5","fragment_sha256":"a85b53d92a6f2d3d33a87eaf52d2bfcd99f7962bd6bd8ee079d5c1fcd1ce1ac5","added_terminal_newline":false} -->
````python
"""AECOPD model_v1 的框架无关流式数据加载器。

加载器按 hadm_id 合并静态字段、结构化事件、数值摘要和文本索引，输出
一次住院一个 ``AdmissionSample``。它不训练模型，也不计算评价指标。
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TypeVar

from clinical_feature_contract import ClinicalFeatureContract, MappedClinicalFeature

T = TypeVar("T")

STATIC_CONTINUOUS_FIELDS = (
    "age_at_admission", "ed_hours", "hospital_los_hours", "prior_aecopd_count",
    "prior_all_admission_count", "days_since_previous_aecopd", "icu_stay_count", "icu_los_days",
    "prior_weight_kg", "prior_height_cm", "prior_bmi", "prior_sbp", "prior_dbp",
    "days_since_prior_weight", "days_since_prior_bmi", "days_since_prior_bp",
)
STATIC_CATEGORICAL_FIELDS = (
    "gender", "admission_type", "admission_location", "discharge_location",
)
STATIC_BINARY_FIELDS = ("via_ed", "entered_icu")
SUMMARY_SCALED_FIELDS = (
    "min_value_scaled", "max_value_scaled", "mean_value_scaled", "sd_value_scaled",
    "first_value_scaled", "last_value_scaled", "first_to_last_change_scaled", "slope_per_hour_scaled",
)
RESERVED_VOCAB = ("<PAD>", "<UNK>", "<MISSING>")
DEFAULT_FEATURE_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "clinical_feature_contract.csv"
)
RETROSPECTIVE_EVIDENCE_TYPES = {
    "retrospective_discharge_code",
    "retrospective_procedure_code",
}


@dataclass(frozen=True)
class EventInput:
    """一条保留来源、时间、数值、状态和证据语义的结构化事件。"""

    event_id: str
    concept_id: int
    event_domain: str
    concept_name: str
    hours_from_admission: float
    hours_before_discharge: float
    value_scaled: float
    numeric_mask: bool
    normalization_available: bool
    normalization_method: str
    raw_code: str
    raw_name: str
    value_text: str
    unit: str
    route: str
    status: str
    evidence_type: str
    model_scope: str
    duration_hours: float = 0.0
    duration_mask: bool = False
    feature_layer: str = "patient_state"
    value_type: str = "clinical_state"
    source_priority: int = 0
    dedup_identity: str = ""
    dedup_route: str = ""


@dataclass(frozen=True)
class NumericSummaryInput:
    """一个概念在整次住院内的训练折归一化数值轨迹摘要。"""

    concept_id: int
    event_domain: str
    concept_name: str
    canonical_unit: str
    measurement_count_log1p: float
    scaled_values: dict[str, float]
    missing_mask: dict[str, bool]
    normalization_available: bool
    unseen_value_count: int


@dataclass(frozen=True)
class TextChunkInput:
    """一段文本或章节的索引与元数据；风险训练可不读取 ``chunk_text``。"""

    document_key: str
    document_role: str
    document_rank: int
    days_before_landmark: float
    chunk_index: int
    token_count: int
    chunk_text: str
    source_document_key: str = ""
    modality: str = "<UNK>"
    section_role: str = "<UNK>"
    section_order: int = 0
    section_instance: int = 1


@dataclass(frozen=True)
class AdmissionSample:
    """一次索引住院的完整模型无关样本。"""

    subject_id: int
    hadm_id: int
    data_split: str
    event_type: int
    event_name: str
    time_days: float
    horizon_labels: dict[str, int]
    static_continuous: dict[str, float]
    static_missing_mask: dict[str, bool]
    static_categorical: dict[str, int]
    static_binary: dict[str, int]
    events: list[EventInput]
    numeric_summaries: list[NumericSummaryInput]
    text_chunks: list[TextChunkInput]
    hospital_los_hours: float = math.nan


def _open_csv(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _float_or_none(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _finite_or_zero(value: str | None) -> float:
    parsed = _float_or_none(value)
    return 0.0 if parsed is None else parsed


def _int_or_zero(value: str | None) -> int:
    normalized = (value or "").strip().lower()
    if normalized in {"t", "true", "y", "yes"}:
        return 1
    if normalized in {"f", "false", "n", "no", ""}:
        return 0
    parsed = _float_or_none(value)
    return 0 if parsed is None else int(parsed)


def _bool_value(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "t", "true", "y", "yes"}


def _duration_hours(start: str | None, end: str | None) -> tuple[float, bool]:
    if not start or not end:
        return 0.0, False
    try:
        duration = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 3600.0
    except ValueError:
        return 0.0, False
    if not math.isfinite(duration) or duration < 0.0:
        return 0.0, False
    return duration, True


def _canonical(value: str | None) -> str:
    stripped = (value or "").strip()
    return stripped if stripped else "<MISSING>"


def _event_identity(row: Mapping[str, str]) -> str:
    """Return the train-fitted identity used by the structured event encoder.

    Events explicitly routed to a code/text encoder must retain the concrete raw
    item. Otherwise hundreds of distinct drugs, laboratory items, and ICD codes
    collapse to one generic placeholder concept.
    """
    base = f"{row['event_domain']}|{row['concept_name']}"
    if row.get("model_scope") != "main_model_encoder":
        return base
    raw_identity = _canonical(row.get("raw_code") or row.get("raw_name") or row.get("value_text"))
    return f"{base}|{raw_identity}"


def _mapped_event_identity(mapped: MappedClinicalFeature) -> str:
    return f"{mapped.feature_layer}|{mapped.canonical_concept}"


def _canonical_evidence(row: Mapping[str, str]) -> str:
    evidence_type = (row.get("evidence_type") or "").strip()
    if evidence_type in RETROSPECTIVE_EVIDENCE_TYPES:
        return evidence_type
    if _float_or_none(row.get("hours_from_admission")) is not None:
        return "observed_time"
    if _float_or_none(row.get("hours_before_discharge")) is not None:
        return "observed_time"
    return "unknown_time"


def _is_treatment_exposure(row: Mapping[str, str]) -> bool:
    """判断治疗记录是否足以支持实际暴露。

    处方只表示计划，不冒充实际给药；明确未执行、未开始、暂缓或
    暂停的记录也不能单独证明本次暴露。“Stopped”仍保留，因为它可以是已给药
    后的停药记录。
    """
    evidence_type = (row.get("evidence_type") or "").strip().lower()
    if evidence_type == "prescription_only":
        return False
    normalized = (row.get("status") or "").strip().lower()
    excluded_markers = (
        "not given",
        "not administered",
        "not started",
        "not confirmed",
        "hold dose",
        "held dose",
        "dose held",
        "paused",
    )
    if normalized in {"no", "delayed"}:
        return False
    return not any(marker in normalized for marker in excluded_markers)


def _has_observed_treatment_time(row: Mapping[str, str]) -> bool:
    """治疗暴露必须能定位到住院时间窗。

    无入院相对时间且无出院相对时间的记录不能默认放入第1天。
    """
    return (
        _float_or_none(row.get("hours_from_admission")) is not None
        or _float_or_none(row.get("hours_before_discharge")) is not None
    )


def _treatment_dedup_identity(row: Mapping[str, str]) -> str:
    """返回仅用于上游跨来源去重的原始药物标识。"""
    return _canonical(row.get("raw_code") or row.get("raw_name"))


def _summary_identity(mapped: MappedClinicalFeature, canonical_unit: str) -> str:
    return f"{mapped.feature_layer}|{mapped.canonical_concept}|{canonical_unit}"


def _source_priority(row: Mapping[str, str]) -> int:
    """返回仅供上游去重使用的来源优先级，不进入模型特征。"""
    source_table = (row.get("source_table") or "").strip().lower()
    evidence_type = (row.get("evidence_type") or "").strip().lower()
    if source_table == "mimiciv_hosp.emar" or evidence_type == "actual_administration":
        return 3
    if source_table == "mimiciv_icu.inputevents" or evidence_type == "actual_icu_input":
        return 2
    if source_table == "mimiciv_hosp.prescriptions" or evidence_type == "prescription_only":
        return 1
    return 0


def _vocabulary(counter: Counter[str]) -> list[str]:
    learned = [
        value
        for value, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        if value not in RESERVED_VOCAB
    ]
    return [*RESERVED_VOCAB, *learned]


def _vocab_index(vocabulary: Sequence[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate(vocabulary)}


def _quantiles(values: Sequence[float], probabilities: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantiles require at least one value")
    result = []
    for probability in probabilities:
        position = probability * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        result.append(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)
    return result


def buffered_shuffle(items: Iterable[T], buffer_size: int, seed: int) -> Iterator[T]:
    """Deterministically shuffle a stream with bounded memory."""
    if buffer_size < 1:
        raise ValueError("buffer_size must be at least 1")
    rng = random.Random(seed)
    buffer: list[T] = []
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = item
    rng.shuffle(buffer)
    yield from buffer


def _grouped_rows(path: Path) -> Iterator[tuple[int, list[dict[str, str]]]]:
    current_key: int | None = None
    rows: list[dict[str, str]] = []
    for row in _open_csv(path):
        key = int(row["hadm_id"])
        if current_key is None:
            current_key = key
        if key != current_key:
            if key < current_key:
                raise ValueError(f"{path.name} is not sorted by hadm_id")
            yield current_key, rows
            current_key = key
            rows = []
        rows.append(row)
    if current_key is not None:
        yield current_key, rows


class _GroupCursor:
    def __init__(self, path: Path) -> None:
        self._iterator = iter(_grouped_rows(path))
        self._current = next(self._iterator, None)

    def take(self, hadm_id: int) -> list[dict[str, str]]:
        while self._current is not None and self._current[0] < hadm_id:
            self._current = next(self._iterator, None)
        if self._current is not None and self._current[0] == hadm_id:
            rows = self._current[1]
            self._current = next(self._iterator, None)
            return rows
        return []


class AecopdDataBundle:
    """Stream admission samples with either v1 document or v2 section text."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        text_metadata_only: bool = False,
        text_dir: str | Path | None = None,
        feature_contract_path: str | Path | None = DEFAULT_FEATURE_CONTRACT_PATH,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.feature_contract = (
            None
            if feature_contract_path is None
            else ClinicalFeatureContract.from_csv(feature_contract_path)
        )
        text_root = self.data_dir if text_dir is None else Path(text_dir)
        if text_dir is None:
            text_name = (
                "qwen_embedding_index.csv.gz"
                if text_metadata_only
                else "qwen_text_chunks.csv.gz"
            )
        else:
            text_name = (
                "qwen_section_embedding_index.csv.gz"
                if text_metadata_only
                else "qwen_section_text_chunks.csv.gz"
            )
        self.paths = {
            "cohort": self.data_dir / "cohort_labels_static.csv.gz",
            "events": self.data_dir / "events.csv.gz",
            "summary": self.data_dir / "numeric_summary.csv.gz",
            "stats": self.data_dir / "numeric_normalization_stats.csv.gz",
            "text": text_root / text_name,
        }
        missing = [str(path) for path in self.paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing model_v1 files: {missing}")

    def fit_preprocessing(self, output_path: str | Path | None = None) -> dict[str, Any]:
        """Fit static scalers and vocabularies using train rows only."""
        continuous: dict[str, list[float]] = {field: [] for field in STATIC_CONTINUOUS_FIELDS}
        categorical: dict[str, Counter[str]] = {field: Counter() for field in STATIC_CATEGORICAL_FIELDS}
        train_admissions = 0

        for row in _open_csv(self.paths["cohort"]):
            if row["data_split"] != "train":
                continue
            train_admissions += 1
            for field in STATIC_CONTINUOUS_FIELDS:
                value = _float_or_none(row.get(field))
                if value is not None:
                    continuous[field].append(value)
            for field in STATIC_CATEGORICAL_FIELDS:
                categorical[field][_canonical(row.get(field))] += 1

        static_stats: dict[str, dict[str, float | int | str]] = {}
        for field, values in continuous.items():
            if values:
                p005, p25, median, p75, p995 = _quantiles(values, [0.005, 0.25, 0.5, 0.75, 0.995])
                if p75 > p25:
                    scale = p75 - p25
                    method = "train_iqr"
                elif p995 > p005:
                    scale = p995 - p005
                    method = "train_p005_p995_range"
                else:
                    scale = 1.0
                    method = "train_constant"
                static_stats[field] = {
                    "train_n": len(values), "p005": float(p005), "p25": float(p25),
                    "median": float(median), "p75": float(p75), "p995": float(p995),
                    "scale": float(scale), "method": method,
                }
            else:
                static_stats[field] = {
                    "train_n": 0, "p005": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0,
                    "p995": 0.0, "scale": 1.0, "method": "no_train_values",
                }

        event_counter: Counter[str] = Counter()
        event_domain_counter: Counter[str] = Counter()
        event_unit_counter: Counter[str] = Counter()
        event_route_counter: Counter[str] = Counter()
        event_evidence_counter: Counter[str] = Counter()
        source_train_event_rows = 0
        retained_train_source_rows = 0
        mapped_train_event_rows = 0
        mapped_train_layer_counts: Counter[str] = Counter()
        for row in _open_csv(self.paths["events"]):
            if row["data_split"] != "train":
                continue
            source_train_event_rows += 1
            if self.feature_contract is None:
                event_counter[_event_identity(row)] += 1
                event_domain_counter[_canonical(row.get("event_domain"))] += 1
                event_unit_counter[_canonical((row.get("unit") or "").lower())] += 1
                event_route_counter[_canonical((row.get("route") or "").lower())] += 1
                event_evidence_counter[_canonical(row.get("evidence_type"))] += 1
                retained_train_source_rows += 1
                mapped_train_event_rows += 1
                continue
            retained_current_row = False
            for mapped in self.feature_contract.map_event(row):
                if mapped.value_type == "binary_exposure":
                    if not _is_treatment_exposure(row) or not _has_observed_treatment_time(row):
                        continue
                retained_current_row = True
                event_counter[_mapped_event_identity(mapped)] += 1
                event_domain_counter[mapped.feature_layer] += 1
                unit = "" if mapped.value_type == "binary_exposure" else (row.get("unit") or "")
                event_unit_counter[_canonical(unit.lower())] += 1
                event_route_counter["<MISSING>"] += 1
                event_evidence_counter[_canonical_evidence(row)] += 1
                mapped_train_event_rows += 1
                mapped_train_layer_counts[mapped.feature_layer] += 1
            retained_train_source_rows += int(retained_current_row)

        summary_counter: Counter[str] = Counter()
        for row in _open_csv(self.paths["stats"]):
            if self.feature_contract is None:
                key = f"{row['event_domain']}|{row['concept_name']}|{row['canonical_unit']}"
                summary_counter[key] += int(row["train_n"])
                continue
            pseudo_event = {**row, "model_scope": "main_model"}
            for mapped in self.feature_contract.map_event(pseudo_event):
                if mapped.feature_layer != "patient_state" or mapped.value_type != "numeric_trajectory":
                    continue
                summary_counter[_summary_identity(mapped, row["canonical_unit"])] += int(row["train_n"])

        artifacts: dict[str, Any] = {
            "schema_version": "aecopd-curated-clinical-loader-1",
            "fit_split": "train",
            "train_admissions": train_admissions,
            "static_continuous": static_stats,
            "static_categorical": {
                field: {"vocabulary": _vocabulary(counter)} for field, counter in categorical.items()
            },
            "static_binary_fields": list(STATIC_BINARY_FIELDS),
            "event_vocabulary": _vocabulary(event_counter),
            "event_domain_vocabulary": _vocabulary(event_domain_counter),
            "event_unit_vocabulary": _vocabulary(event_unit_counter),
            "event_route_vocabulary": _vocabulary(event_route_counter),
            "event_evidence_vocabulary": _vocabulary(event_evidence_counter),
            "numeric_summary_vocabulary": _vocabulary(summary_counter),
            "reserved_ids": {"pad": 0, "unknown": 1, "missing": 2},
            "normalization_contract": {
                "event_values": "train IQR; fallback train p0.5-p99.5 range; constant groups use scale 1; unseen train groups use value 0 plus availability=false",
                "static_values": "train p0.5-p99.5 clipping followed by train robust scaling; missing values imputed with train median plus missing mask",
                "numeric_summary": "precomputed scaled columns; missing optional statistics use 0 plus missing mask",
            },
            "clinical_feature_contract": (
                None if self.feature_contract is None else self.feature_contract.metadata()
            ),
            "clinical_feature_audit": {
                "source_train_event_rows": source_train_event_rows,
                "retained_train_source_rows": retained_train_source_rows,
                "excluded_train_source_rows": source_train_event_rows - retained_train_source_rows,
                "mapped_train_event_rows": mapped_train_event_rows,
                "mapped_train_layer_counts": dict(sorted(mapped_train_layer_counts.items())),
            },
        }
        if output_path is not None:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")
        return artifacts

    @staticmethod
    def load_preprocessing(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def iter_samples(
        self,
        split: str | None,
        *,
        artifacts: Mapping[str, Any],
        include_text: bool = True,
    ) -> Iterator[AdmissionSample]:
        if split is not None and split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, test, or None")

        event_cursor = _GroupCursor(self.paths["events"])
        summary_cursor = _GroupCursor(self.paths["summary"])
        text_cursor = _GroupCursor(self.paths["text"])
        event_index = _vocab_index(artifacts["event_vocabulary"])
        summary_index = _vocab_index(artifacts["numeric_summary_vocabulary"])
        category_indices = {
            field: _vocab_index(artifacts["static_categorical"][field]["vocabulary"])
            for field in STATIC_CATEGORICAL_FIELDS
        }
        artifact_contract = artifacts.get("clinical_feature_contract")
        if self.feature_contract is not None:
            if not isinstance(artifact_contract, Mapping):
                raise ValueError(
                    "Preprocessing artifacts do not contain the active clinical feature contract; "
                    "refit preprocessing before loading samples"
                )
            if artifact_contract.get("sha256") != self.feature_contract.sha256:
                raise ValueError(
                    "Clinical feature contract changed after preprocessing was fit; refit preprocessing"
                )

        for row in _open_csv(self.paths["cohort"]):
            hadm_id = int(row["hadm_id"])
            event_rows = event_cursor.take(hadm_id)
            summary_rows = summary_cursor.take(hadm_id)
            text_rows = text_cursor.take(hadm_id)
            if split is not None and row["data_split"] != split:
                continue
            row_split = row["data_split"]
            if any(item["data_split"] != row_split for item in event_rows):
                raise ValueError(f"event split mismatch for hadm_id={hadm_id}")
            if any(item["data_split"] != row_split for item in summary_rows):
                raise ValueError(f"numeric summary split mismatch for hadm_id={hadm_id}")
            if any(item["data_split"] != row_split for item in text_rows):
                raise ValueError(f"text split mismatch for hadm_id={hadm_id}")

            static_continuous: dict[str, float] = {}
            static_missing: dict[str, bool] = {}
            for field in STATIC_CONTINUOUS_FIELDS:
                value = _float_or_none(row.get(field))
                static_missing[field] = value is None
                stats = artifacts["static_continuous"][field]
                if value is None:
                    value = float(stats["median"])
                clipped = min(float(stats["p995"]), max(float(stats["p005"]), value))
                static_continuous[field] = (clipped - float(stats["median"])) / float(stats["scale"])

            static_categorical = {}
            for field in STATIC_CATEGORICAL_FIELDS:
                value = _canonical(row.get(field))
                static_categorical[field] = category_indices[field].get(value, 1)

            events = []
            for event in event_rows:
                if self.feature_contract is not None:
                    mapped_features = self.feature_contract.map_event(event)
                else:
                    mapped_features = ()
                value_num = _float_or_none(event.get("value_num"))
                duration_hours, duration_mask = _duration_hours(event.get("event_time"), event.get("end_time"))
                if self.feature_contract is not None:
                    for mapped in mapped_features:
                        if mapped.value_type == "binary_exposure":
                            if not _is_treatment_exposure(event) or not _has_observed_treatment_time(event):
                                continue
                        binary_exposure = mapped.value_type == "binary_exposure"
                        event_key = _mapped_event_identity(mapped)
                        events.append(EventInput(
                            event_id=f"{event['event_id']}::{mapped.rule_id}",
                            concept_id=event_index.get(event_key, 1),
                            event_domain=mapped.feature_layer,
                            concept_name=mapped.canonical_concept,
                            hours_from_admission=_finite_or_zero(event.get("hours_from_admission")),
                            hours_before_discharge=_finite_or_zero(event.get("hours_before_discharge")),
                            value_scaled=(
                                0.0 if binary_exposure else _finite_or_zero(event.get("value_scaled"))
                            ),
                            numeric_mask=(value_num is not None and not binary_exposure),
                            normalization_available=(
                                _bool_value(event.get("normalization_available"))
                                if value_num is not None and not binary_exposure
                                else False
                            ),
                            normalization_method=(
                                event.get("normalization_method", "") if not binary_exposure else ""
                            ),
                            raw_code=event.get("raw_code", ""),
                            raw_name=event.get("raw_name", ""),
                            value_text=event.get("value_text", ""),
                            unit="" if binary_exposure else event.get("unit", ""),
                            route="",
                            # 保留原始状态供上游审计；二元暴露在
                            # collator 中统一表示为已暴露，原始字符串不入模。
                            status=event.get("status", ""),
                            evidence_type=_canonical_evidence(event),
                            model_scope="curated_clinical_feature",
                            duration_hours=duration_hours,
                            duration_mask=duration_mask,
                            feature_layer=mapped.feature_layer,
                            value_type=mapped.value_type,
                            source_priority=_source_priority(event),
                            dedup_identity=(
                                _treatment_dedup_identity(event) if binary_exposure else ""
                            ),
                            dedup_route=(
                                _canonical(event.get("route")) if binary_exposure else ""
                            ),
                        ))
                    continue
                specific_identity = "event_domain_vocabulary" in artifacts
                event_key = _event_identity(event) if specific_identity else f"{event['event_domain']}|{event['concept_name']}"
                events.append(EventInput(
                    event_id=event["event_id"],
                    concept_id=event_index.get(event_key, 1),
                    event_domain=event["event_domain"], concept_name=event["concept_name"],
                    hours_from_admission=_finite_or_zero(event.get("hours_from_admission")),
                    hours_before_discharge=_finite_or_zero(event.get("hours_before_discharge")),
                    value_scaled=_finite_or_zero(event.get("value_scaled")), numeric_mask=value_num is not None,
                    normalization_available=_bool_value(event.get("normalization_available")) if value_num is not None else False,
                    normalization_method=event.get("normalization_method", ""), raw_code=event.get("raw_code", ""),
                    raw_name=event.get("raw_name", ""), value_text=event.get("value_text", ""),
                    unit=event.get("unit", ""), route=event.get("route", ""), status=event.get("status", ""),
                    evidence_type=event.get("evidence_type", ""), model_scope=event.get("model_scope", ""),
                    duration_hours=duration_hours, duration_mask=duration_mask,
                ))

            summaries = []
            for summary in summary_rows:
                if self.feature_contract is not None:
                    mapped_summaries = tuple(
                        mapped
                        for mapped in self.feature_contract.map_event(
                            {**summary, "model_scope": "main_model"}
                        )
                        if mapped.feature_layer == "patient_state"
                        and mapped.value_type == "numeric_trajectory"
                    )
                    if not mapped_summaries:
                        continue
                else:
                    mapped_summaries = ()
                values: dict[str, float] = {}
                missing: dict[str, bool] = {}
                for field in SUMMARY_SCALED_FIELDS:
                    parsed = _float_or_none(summary.get(field))
                    missing[field] = parsed is None
                    values[field] = 0.0 if parsed is None else parsed
                if self.feature_contract is None:
                    mapped_rows = (
                        (
                            f"{summary['event_domain']}|{summary['concept_name']}|{summary['canonical_unit']}",
                            summary["event_domain"],
                            summary["concept_name"],
                        ),
                    )
                else:
                    mapped_rows = tuple(
                        (
                            _summary_identity(mapped, summary["canonical_unit"]),
                            mapped.feature_layer,
                            mapped.canonical_concept,
                        )
                        for mapped in mapped_summaries
                    )
                for key, event_domain, concept_name in mapped_rows:
                    summaries.append(NumericSummaryInput(
                        concept_id=summary_index.get(key, 1), event_domain=event_domain,
                        concept_name=concept_name, canonical_unit=summary["canonical_unit"],
                        measurement_count_log1p=math.log1p(_int_or_zero(summary.get("measurement_count"))),
                        scaled_values=values, missing_mask=missing,
                        normalization_available=_bool_value(summary.get("normalization_available")),
                        unseen_value_count=_int_or_zero(summary.get("unseen_value_count")),
                    ))

            texts = []
            if include_text:
                for text in text_rows:
                    texts.append(TextChunkInput(
                        document_key=text["document_key"], document_role=text["document_role"],
                        document_rank=int(text["document_rank"]),
                        days_before_landmark=_finite_or_zero(text.get("days_before_landmark")),
                        chunk_index=int(text["chunk_index"]), token_count=int(text["token_count"]),
                        chunk_text=text.get("chunk_text", ""),
                        source_document_key=text.get("source_document_key") or text["document_key"],
                        modality=text.get("modality") or "<UNK>",
                        section_role=text.get("section_role") or "<UNK>",
                        section_order=int(text.get("section_order") or text["chunk_index"]),
                        section_instance=int(text.get("section_instance") or 1),
                    ))

            horizon_fields = (
                "aecopd_by_90d", "aecopd_by_180d", "aecopd_by_365d",
                "death_by_90d", "death_by_180d", "death_by_365d",
            )
            yield AdmissionSample(
                subject_id=int(row["subject_id"]), hadm_id=hadm_id, data_split=row_split,
                event_type=int(row["event_type"]), event_name=row["event_name"],
                time_days=float(row["time_days"]),
                horizon_labels={field: _int_or_zero(row.get(field)) for field in horizon_fields},
                static_continuous=static_continuous, static_missing_mask=static_missing,
                static_categorical=static_categorical,
                static_binary={field: int(_bool_value(row.get(field))) for field in STATIC_BINARY_FIELDS},
                events=events, numeric_summaries=summaries, text_chunks=texts,
                hospital_los_hours=_finite_or_zero(row.get("hospital_los_hours")),
            )

    def iter_all_samples(
        self,
        *,
        artifacts: Mapping[str, Any],
        include_text: bool = True,
    ) -> Iterator[AdmissionSample]:
        yield from self.iter_samples(None, artifacts=artifacts, include_text=include_text)

    def iter_batches(
        self,
        split: str,
        *,
        artifacts: Mapping[str, Any],
        batch_size: int,
        shuffle_buffer_size: int = 0,
        seed: int = 0,
        include_text: bool = True,
    ) -> Iterator[list[AdmissionSample]]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        samples: Iterable[AdmissionSample] = self.iter_samples(split, artifacts=artifacts, include_text=include_text)
        if shuffle_buffer_size:
            samples = buffered_shuffle(samples, shuffle_buffer_size, seed)
        batch: list[AdmissionSample] = []
        for sample in samples:
            batch.append(sample)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def sample_to_dict(sample: AdmissionSample) -> dict[str, Any]:
    """Convert a sample to plain Python data for inspection or JSON debugging."""
    return asdict(sample)
````
<!-- END REVIEW_FRAGMENT -->

## model_loader/audit_gpu_environment.py · lines 1–54

<!-- REVIEW_FRAGMENT {"path":"model_loader/audit_gpu_environment.py","part":1,"start_line":1,"end_line":54,"source_sha256":"c615c8bc7bfac2e7566d8749fa084c976ec5c6201326237434c9349c1d5aff31","fragment_sha256":"c615c8bc7bfac2e7566d8749fa084c976ec5c6201326237434c9349c1d5aff31","added_terminal_newline":false} -->
````python
from __future__ import annotations

import json
import platform
import sys


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        print(json.dumps({
            "passed": False,
            "error": "PyTorch is not installed in this container",
            "detail": str(exc),
        }, ensure_ascii=False, indent=2))
        return 1

    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": properties.name,
                "total_vram_gib": round(properties.total_memory / 1024**3, 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
            })

        probe = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=torch.float16)
        probe_result = probe @ probe
        finite_probe = bool(torch.isfinite(probe_result).all().item())
    else:
        finite_probe = False

    report = {
        "passed": bool(torch.cuda.is_available() and devices and finite_probe),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "fp16_matrix_probe_finite": finite_probe,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
````
<!-- END REVIEW_FRAGMENT -->

## model_loader/clinical_feature_contract.py · lines 1–177

<!-- REVIEW_FRAGMENT {"path":"model_loader/clinical_feature_contract.py","part":1,"start_line":1,"end_line":177,"source_sha256":"b0c5f53aca2d236b1bde32c1836f6fc4f04d68c939eb503941fbd4ca5725cc37","fragment_sha256":"b0c5f53aca2d236b1bde32c1836f6fc4f04d68c939eb503941fbd4ca5725cc37","added_terminal_newline":false} -->
````python
"""把原始结构化事件映射为受控临床特征。

该模块是原始事件表与模型装载器之间的唯一映射 seam。调用方只需要提供
一条 CSV 事件记录；药物归类、一对多复方映射、默认排除和合同校验都封装
在此模块内。
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


VALID_FEATURE_LAYERS = frozenset({"patient_state", "care_process"})
VALID_VALUE_TYPES = frozenset(
    {"numeric_trajectory", "clinical_state", "binary_exposure"}
)


@dataclass(frozen=True)
class MappedClinicalFeature:
    """一条原始事件映射后的标准临床概念。"""

    canonical_concept: str
    feature_layer: str
    value_type: str
    rule_id: str


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    source_domain: str
    source_concept: str
    model_scope: str
    canonical_concept: str
    feature_layer: str
    value_type: str
    keep: bool

    def matches(self, row: Mapping[str, str]) -> bool:
        return all(
            expected == "*" or expected == actual
            for expected, actual in (
                (self.source_domain, (row.get("event_domain") or "").strip()),
                (self.source_concept, (row.get("concept_name") or "").strip()),
                (self.model_scope, (row.get("model_scope") or "").strip()),
            )
        )

    @property
    def specificity(self) -> int:
        return sum(
            value != "*"
            for value in (self.source_domain, self.source_concept, self.model_scope)
        )


class ClinicalFeatureContract:
    """加载并执行可审核的临床特征映射合同。

    未命中的事件默认排除。若多条最高特异度规则同时命中，则全部输出，
    从而支持 ICS/LABA 等复方治疗的一对多映射。
    """

    def __init__(self, rules: tuple[_Rule, ...], *, source_path: Path, sha256: str) -> None:
        self._rules = rules
        self.source_path = source_path
        self.sha256 = sha256

    @classmethod
    def from_csv(cls, path: str | Path) -> "ClinicalFeatureContract":
        source_path = Path(path)
        payload = source_path.read_bytes()
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Clinical feature contract is empty: {source_path}")

        rules: list[_Rule] = []
        seen_rule_ids: set[str] = set()
        required = {
            "rule_id",
            "source_domain",
            "source_concept",
            "model_scope",
            "canonical_concept",
            "feature_layer",
            "value_type",
            "keep",
        }
        missing_columns = required.difference(rows[0])
        if missing_columns:
            raise ValueError(
                f"Clinical feature contract missing columns: {sorted(missing_columns)}"
            )

        for row_number, row in enumerate(rows, start=2):
            rule_id = (row.get("rule_id") or "").strip()
            if not rule_id or rule_id in seen_rule_ids:
                raise ValueError(
                    f"Invalid or duplicate rule_id at {source_path}:{row_number}: {rule_id!r}"
                )
            seen_rule_ids.add(rule_id)
            keep_value = (row.get("keep") or "").strip().lower()
            if keep_value not in {"0", "1", "false", "true", "no", "yes", "n", "y"}:
                raise ValueError(
                    f"Invalid keep value at {source_path}:{row_number}: {keep_value!r}"
                )
            keep = keep_value in {"1", "true", "yes", "y"}
            feature_layer = (row.get("feature_layer") or "").strip()
            value_type = (row.get("value_type") or "").strip()
            canonical_concept = (row.get("canonical_concept") or "").strip()
            if keep:
                if feature_layer not in VALID_FEATURE_LAYERS:
                    raise ValueError(
                        f"Invalid feature_layer at {source_path}:{row_number}: {feature_layer!r}"
                    )
                if value_type not in VALID_VALUE_TYPES:
                    raise ValueError(
                        f"Invalid value_type at {source_path}:{row_number}: {value_type!r}"
                    )
                if not canonical_concept:
                    canonical_concept = "$SOURCE"
            rules.append(
                _Rule(
                    rule_id=rule_id,
                    source_domain=(row.get("source_domain") or "").strip() or "*",
                    source_concept=(row.get("source_concept") or "").strip() or "*",
                    model_scope=(row.get("model_scope") or "").strip() or "*",
                    canonical_concept=canonical_concept,
                    feature_layer=feature_layer,
                    value_type=value_type,
                    keep=keep,
                )
            )
        return cls(
            tuple(rules),
            source_path=source_path.resolve(),
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    def map_event(self, row: Mapping[str, str]) -> tuple[MappedClinicalFeature, ...]:
        matches = [rule for rule in self._rules if rule.matches(row)]
        if not matches:
            return ()
        maximum_specificity = max(rule.specificity for rule in matches)
        selected = [rule for rule in matches if rule.specificity == maximum_specificity]
        source_concept = (row.get("concept_name") or "").strip()
        mapped = []
        for rule in selected:
            if not rule.keep:
                continue
            canonical_concept = (
                source_concept if rule.canonical_concept == "$SOURCE" else rule.canonical_concept
            )
            mapped.append(
                MappedClinicalFeature(
                    canonical_concept=canonical_concept,
                    feature_layer=rule.feature_layer,
                    value_type=rule.value_type,
                    rule_id=rule.rule_id,
                )
            )
        return tuple(mapped)

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": "aecopd-clinical-feature-contract-1",
            "path": str(self.source_path),
            "sha256": self.sha256,
            "rule_count": len(self._rules),
            "default_policy": "exclude_unmatched",
        }
````
<!-- END REVIEW_FRAGMENT -->

## model_loader/timeline_collator.py · lines 1–712

<!-- REVIEW_FRAGMENT {"path":"model_loader/timeline_collator.py","part":1,"start_line":1,"end_line":712,"source_sha256":"75c879842936a2a5f81c26c80a8df6b7c9681d742c0f8ce46090ed9420342d7b","fragment_sha256":"75c879842936a2a5f81c26c80a8df6b7c9681d742c0f8ce46090ed9420342d7b","added_terminal_newline":false} -->
````python
"""将一次住院的结构化长表聚合为共享双锚点时间轴。

聚合会保留概念、域、单位、给药途径、证据类型和编码器标志，并用 25 个
统计通道描述同组事件。它不是把整次住院压成 25 个变量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from aecopd_loader import (
    STATIC_BINARY_FIELDS,
    STATIC_CATEGORICAL_FIELDS,
    STATIC_CONTINUOUS_FIELDS,
    SUMMARY_SCALED_FIELDS,
    AdmissionSample,
    EventInput,
    TextChunkInput,
)


TIMELINE_FEATURE_NAMES = (
    "count_log1p",
    "first_scaled",
    "last_scaled",
    "min_scaled",
    "max_scaled",
    "mean_scaled",
    "sd_scaled",
    "delta_scaled",
    "slope_scaled_per_hour",
    "hours_before_discharge_first_log1p",
    "hours_before_discharge_last_log1p",
    "observation_span_hours_log1p",
    "duration_hours_max_log1p",
    "normalization_available_fraction",
    "status_administered_fraction",
    "status_not_given_fraction",
    "status_flush_fraction",
    "status_not_flushed_fraction",
    "status_started_fraction",
    "status_stopped_fraction",
    "status_paused_fraction",
    "status_changed_fraction",
    "status_abnormal_fraction",
    "status_susceptible_fraction",
    "status_resistant_fraction",
)

WINDOW_NAMES = (
    "<PAD>",
    "入院第1天（0～24小时）",
    "入院第2天（24～48小时）",
    "入院第3天（48～72小时）",
    "入院第4天（72～96小时）",
    "入院第5天（96～120小时）",
    "入院第6天（120～144小时）",
    "入院第7天（144～168小时）",
    "长住院中间阶段",
    "出院前第7天（144～168小时）",
    "出院前第6天（120～144小时）",
    "出院前第5天（96～120小时）",
    "出院前第4天（72～96小时）",
    "出院前第3天（48～72小时）",
    "出院前第2天（24～48小时）",
    "出院前第1天（0～24小时）",
    "回顾性出院编码（无可靠发生时间）",
    "其他无可靠时间事件",
)

RETROSPECTIVE_EVIDENCE_TYPES = {
    "retrospective_discharge_code",
    "retrospective_procedure_code",
}

GLOBAL_SUMMARY_FEATURE_NAMES = (
    "measurement_count_log1p",
    *SUMMARY_SCALED_FIELDS,
    "normalization_available",
    "unseen_value_count_log1p",
)

HORIZON_LABEL_NAMES = (
    "aecopd_by_90d",
    "aecopd_by_180d",
    "aecopd_by_365d",
    "death_by_90d",
    "death_by_180d",
    "death_by_365d",
)

STATIC_CONTINUOUS_LAYERS = {
    "ed_hours": "care_process",
    "hospital_los_hours": "care_process",
    "icu_stay_count": "care_process",
    "icu_los_days": "care_process",
}
STATIC_CATEGORICAL_LAYERS = {
    "admission_type": "care_process",
    "admission_location": "care_process",
    "discharge_location": "care_process",
}
STATIC_BINARY_LAYERS = {
    "via_ed": "care_process",
    "entered_icu": "care_process",
}


@dataclass(frozen=True)
class TimelineConfig:
    """Dual-anchor daily timeline configuration for one admission."""

    max_tokens: int | None = 1280
    included_layers: tuple[str, ...] = ("patient_state", "care_process")


@dataclass(frozen=True)
class TimelineBatch:
    """一个补齐后的住院批次及其所有模型输入和审计元数据。"""

    timeline_feature_names: tuple[str, ...]
    timeline_window_names: tuple[str, ...]
    timeline_lengths: list[int]
    timeline_concept_ids: list[list[int]]
    timeline_domain_ids: list[list[int]]
    timeline_window_ids: list[list[int]]
    timeline_unit_ids: list[list[int]]
    timeline_route_ids: list[list[int]]
    timeline_evidence_ids: list[list[int]]
    timeline_encoder_flags: list[list[int]]
    timeline_features: list[list[list[float]]]
    timeline_feature_masks: list[list[list[bool]]]
    timeline_attention_mask: list[list[bool]]
    static_continuous_names: tuple[str, ...]
    static_continuous: list[list[float]]
    static_continuous_masks: list[list[bool]]
    static_categorical_names: tuple[str, ...]
    static_categorical: list[list[int]]
    static_binary_names: tuple[str, ...]
    static_binary: list[list[int]]
    global_summary_feature_names: tuple[str, ...]
    global_summary_lengths: list[int]
    global_summary_concept_ids: list[list[int]]
    global_summary_features: list[list[list[float]]]
    global_summary_feature_masks: list[list[list[bool]]]
    global_summary_attention_mask: list[list[bool]]
    horizon_label_names: tuple[str, ...]
    horizon_labels: list[list[int]]
    text_chunks: list[list[TextChunkInput]]
    subject_ids: list[int]
    hadm_ids: list[int]
    event_types: list[int]
    time_days: list[float]

    def to_torch(self, device: str | None = None) -> dict[str, object]:
        """Convert model-ready numeric fields to tensors using a lazy adapter."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required only for TimelineBatch.to_torch()") from exc

        batch_size = len(self.hadm_ids)
        timeline_width = max(self.timeline_lengths, default=0)
        summary_width = max(self.global_summary_lengths, default=0)

        def tensor(values: object, dtype: object) -> object:
            return torch.as_tensor(values, dtype=dtype, device=device)

        timeline_features = (
            torch.empty((batch_size, 0, len(self.timeline_feature_names)), dtype=torch.float32, device=device)
            if timeline_width == 0 else tensor(self.timeline_features, torch.float32)
        )
        timeline_feature_masks = (
            torch.empty((batch_size, 0, len(self.timeline_feature_names)), dtype=torch.bool, device=device)
            if timeline_width == 0 else tensor(self.timeline_feature_masks, torch.bool)
        )
        summary_features = (
            torch.empty((batch_size, 0, len(self.global_summary_feature_names)), dtype=torch.float32, device=device)
            if summary_width == 0 else tensor(self.global_summary_features, torch.float32)
        )
        summary_feature_masks = (
            torch.empty((batch_size, 0, len(self.global_summary_feature_names)), dtype=torch.bool, device=device)
            if summary_width == 0 else tensor(self.global_summary_feature_masks, torch.bool)
        )

        return {
            "timeline_concept_ids": tensor(self.timeline_concept_ids, torch.long),
            "timeline_domain_ids": tensor(self.timeline_domain_ids, torch.long),
            "timeline_window_ids": tensor(self.timeline_window_ids, torch.long),
            "timeline_unit_ids": tensor(self.timeline_unit_ids, torch.long),
            "timeline_route_ids": tensor(self.timeline_route_ids, torch.long),
            "timeline_evidence_ids": tensor(self.timeline_evidence_ids, torch.long),
            "timeline_encoder_flags": tensor(self.timeline_encoder_flags, torch.bool),
            "timeline_features": timeline_features,
            "timeline_feature_masks": timeline_feature_masks,
            "timeline_attention_mask": tensor(self.timeline_attention_mask, torch.bool),
            "static_continuous": tensor(self.static_continuous, torch.float32),
            "static_continuous_masks": tensor(self.static_continuous_masks, torch.bool),
            "static_categorical": tensor(self.static_categorical, torch.long),
            "static_binary": tensor(self.static_binary, torch.float32),
            "global_summary_concept_ids": tensor(self.global_summary_concept_ids, torch.long),
            "global_summary_features": summary_features,
            "global_summary_feature_masks": summary_feature_masks,
            "global_summary_attention_mask": tensor(self.global_summary_attention_mask, torch.bool),
            "event_types": tensor(self.event_types, torch.long),
            "time_days": tensor(self.time_days, torch.float32),
            "horizon_labels": tensor(self.horizon_labels, torch.float32),
            "subject_ids": tensor(self.subject_ids, torch.long),
            "hadm_ids": tensor(self.hadm_ids, torch.long),
        }


@dataclass
class _TokenAccumulator:
    events: list[EventInput]


@dataclass(frozen=True)
class _TimelineToken:
    concept_id: int
    domain_id: int
    window_id: int
    unit_id: int
    route_id: int
    evidence_id: int
    encoder_flag: int
    features: list[float]
    feature_masks: list[bool]


@dataclass(frozen=True)
class _GlobalSummaryRows:
    lengths: list[int]
    concept_ids: list[list[int]]
    features: list[list[list[float]]]
    feature_masks: list[list[list[bool]]]
    attention_mask: list[list[bool]]


def _canonical(value: str) -> str:
    value = value.strip().lower()
    return value if value else "<MISSING>"


def dual_anchor_window_id(hours_from_admission: float, hours_before_discharge: float) -> int:
    """Map a timestamp onto the shared admission/discharge 24-hour timeline."""
    admission_time_known = math.isfinite(hours_from_admission)
    discharge_time_known = math.isfinite(hours_before_discharge)

    # Give overlapping short stays to the admission anchor first. Consequently,
    # admissions lasting <=14 days retain every 24-hour slice without duplicate
    # events; only the middle of longer stays is temporally pooled.
    if admission_time_known:
        hours_from_admission = max(0.0, hours_from_admission)
        if hours_from_admission < 168.0:
            return min(7, int(hours_from_admission // 24.0) + 1)
    if discharge_time_known:
        hours_before_discharge = max(0.0, hours_before_discharge)
        if hours_before_discharge <= 168.0:
            return 15 - min(6, int(hours_before_discharge // 24.0))
    if admission_time_known or discharge_time_known:
        return 8
    return 17


def _window_id(event: EventInput) -> int:
    if event.evidence_type in RETROSPECTIVE_EVIDENCE_TYPES:
        return 16
    return dual_anchor_window_id(event.hours_from_admission, event.hours_before_discharge)


def _index(vocabulary: Sequence[str] | None) -> dict[str, int]:
    if vocabulary is None:
        return {}
    return {value: i for i, value in enumerate(vocabulary)}


def _lookup(index: Mapping[str, int], value: str) -> int:
    return index.get(value, 1)


def _sample_sd(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0.0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def _same_treatment_record(left: EventInput, right: EventInput) -> bool:
    """使用不入模的原始药物、途径和时间判定跨表重复记录。

    该规则故意偏保守：只删除高度疑似重复的低优先级来源，不会因为
    同一日窗出现一条 eMAR 就删掉其他时点的真实治疗。
    """
    if not left.dedup_identity or left.dedup_identity == "<MISSING>":
        return False
    if left.dedup_identity != right.dedup_identity:
        return False
    if (
        left.dedup_route
        and right.dedup_route
        and left.dedup_route != "<MISSING>"
        and right.dedup_route != "<MISSING>"
        and left.dedup_route != right.dedup_route
    ):
        return False
    admission_gap = abs(left.hours_from_admission - right.hours_from_admission)
    discharge_gap = abs(left.hours_before_discharge - right.hours_before_discharge)
    return admission_gap <= 1.0 or discharge_gap <= 1.0


def _deduplicate_treatment_events(events: Sequence[EventInput]) -> list[EventInput]:
    """保留所有非重复时点，对疑似重复记录优先保留更可靠来源。"""
    retained: list[EventInput] = []
    for event in sorted(
        events,
        key=lambda item: (-item.source_priority, item.hours_from_admission, item.event_id),
    ):
        duplicate_of_better_source = any(
            other.source_priority > event.source_priority
            and _same_treatment_record(event, other)
            for other in retained
        )
        if not duplicate_of_better_source:
            retained.append(event)
    return sorted(retained, key=lambda item: (item.hours_from_admission, item.event_id))


def _status_categories(status: str) -> set[str]:
    normalized = status.strip().lower()
    if not normalized:
        return set()
    categories: set[str] = set()
    if "not given" in normalized or "not administered" in normalized:
        categories.add("not_given")
    elif normalized == "administered" or normalized.startswith("administered ") or normalized in {"applied", "given", "infused"}:
        categories.add("administered")
    if "not flushed" in normalized:
        categories.add("not_flushed")
    elif "flushed" in normalized:
        categories.add("flush")
    if "started" in normalized:
        categories.add("started")
    if any(value in normalized for value in ("stopped", "removed", "discontinued")):
        categories.add("stopped")
    if "paused" in normalized:
        categories.add("paused")
    if "change" in normalized and any(value in normalized for value in ("dose", "rate")):
        categories.add("changed")
    if normalized == "abnormal":
        categories.add("abnormal")
    if normalized in {"s", "susceptible"}:
        categories.add("susceptible")
    if normalized in {"r", "resistant"}:
        categories.add("resistant")
    return categories


class SharedTimelineCollator:
    """Collapse raw events to padded concept-by-time-window batches.

    The public interface is deliberately framework-independent. Numeric arrays can
    be converted to PyTorch/JAX tensors by the training adapter without changing
    the grouping, masking, or leakage rules implemented here.
    """

    def __init__(
        self,
        config: TimelineConfig,
        *,
        artifacts: Mapping[str, object] | None = None,
    ) -> None:
        self.config = config
        artifacts = artifacts or {}
        self._domain_index = _index(artifacts.get("event_domain_vocabulary"))  # type: ignore[arg-type]
        self._unit_index = _index(artifacts.get("event_unit_vocabulary"))  # type: ignore[arg-type]
        self._route_index = _index(artifacts.get("event_route_vocabulary"))  # type: ignore[arg-type]
        self._evidence_index = _index(artifacts.get("event_evidence_vocabulary"))  # type: ignore[arg-type]

    def __call__(self, samples: Sequence[AdmissionSample]) -> TimelineBatch:
        token_rows = [self._aggregate_sample(sample) for sample in samples]
        if self.config.max_tokens is not None:
            for sample, row in zip(samples, token_rows):
                if len(row) > self.config.max_tokens:
                    raise ValueError(
                        f"timeline token count {len(row)} exceeds max_tokens={self.config.max_tokens} "
                        f"for hadm_id={sample.hadm_id}; revise aggregation instead of silently truncating"
                    )
        max_length = max((len(row) for row in token_rows), default=0)

        concept_ids: list[list[int]] = []
        domain_ids: list[list[int]] = []
        window_ids: list[list[int]] = []
        unit_ids: list[list[int]] = []
        route_ids: list[list[int]] = []
        evidence_ids: list[list[int]] = []
        encoder_flags: list[list[int]] = []
        features: list[list[list[float]]] = []
        feature_masks: list[list[list[bool]]] = []
        attention_masks: list[list[bool]] = []
        lengths: list[int] = []

        for row in token_rows:
            lengths.append(len(row))
            padding = max_length - len(row)
            concept_ids.append([token.concept_id for token in row] + [0] * padding)
            domain_ids.append([token.domain_id for token in row] + [0] * padding)
            window_ids.append([token.window_id for token in row] + [0] * padding)
            unit_ids.append([token.unit_id for token in row] + [0] * padding)
            route_ids.append([token.route_id for token in row] + [0] * padding)
            evidence_ids.append([token.evidence_id for token in row] + [0] * padding)
            encoder_flags.append([token.encoder_flag for token in row] + [0] * padding)
            features.append([token.features for token in row] + [[0.0] * len(TIMELINE_FEATURE_NAMES) for _ in range(padding)])
            feature_masks.append([token.feature_masks for token in row] + [[False] * len(TIMELINE_FEATURE_NAMES) for _ in range(padding)])
            attention_masks.append([True] * len(row) + [False] * padding)

        global_summaries = self._collate_global_summaries(samples)
        included_layers = frozenset(self.config.included_layers)

        def static_included(name: str, layers: Mapping[str, str]) -> bool:
            return layers.get(name, "patient_state") in included_layers

        return TimelineBatch(
            timeline_feature_names=TIMELINE_FEATURE_NAMES,
            timeline_window_names=WINDOW_NAMES,
            timeline_lengths=lengths,
            timeline_concept_ids=concept_ids,
            timeline_domain_ids=domain_ids,
            timeline_window_ids=window_ids,
            timeline_unit_ids=unit_ids,
            timeline_route_ids=route_ids,
            timeline_evidence_ids=evidence_ids,
            timeline_encoder_flags=encoder_flags,
            timeline_features=features,
            timeline_feature_masks=feature_masks,
            timeline_attention_mask=attention_masks,
            static_continuous_names=STATIC_CONTINUOUS_FIELDS,
            static_continuous=[
                [
                    sample.static_continuous.get(name, 0.0)
                    if static_included(name, STATIC_CONTINUOUS_LAYERS)
                    else 0.0
                    for name in STATIC_CONTINUOUS_FIELDS
                ]
                for sample in samples
            ],
            static_continuous_masks=[
                [
                    not sample.static_missing_mask.get(name, True)
                    and static_included(name, STATIC_CONTINUOUS_LAYERS)
                    for name in STATIC_CONTINUOUS_FIELDS
                ]
                for sample in samples
            ],
            static_categorical_names=STATIC_CATEGORICAL_FIELDS,
            static_categorical=[
                [
                    sample.static_categorical.get(name, 0)
                    if static_included(name, STATIC_CATEGORICAL_LAYERS)
                    else 2
                    for name in STATIC_CATEGORICAL_FIELDS
                ]
                for sample in samples
            ],
            static_binary_names=STATIC_BINARY_FIELDS,
            static_binary=[
                [
                    sample.static_binary.get(name, 0)
                    if static_included(name, STATIC_BINARY_LAYERS)
                    else 0
                    for name in STATIC_BINARY_FIELDS
                ]
                for sample in samples
            ],
            global_summary_feature_names=GLOBAL_SUMMARY_FEATURE_NAMES,
            global_summary_lengths=global_summaries.lengths,
            global_summary_concept_ids=global_summaries.concept_ids,
            global_summary_features=global_summaries.features,
            global_summary_feature_masks=global_summaries.feature_masks,
            global_summary_attention_mask=global_summaries.attention_mask,
            horizon_label_names=HORIZON_LABEL_NAMES,
            horizon_labels=[
                [sample.horizon_labels.get(name, 0) for name in HORIZON_LABEL_NAMES]
                for sample in samples
            ],
            text_chunks=[sample.text_chunks for sample in samples],
            subject_ids=[sample.subject_id for sample in samples],
            hadm_ids=[sample.hadm_id for sample in samples],
            event_types=[sample.event_type for sample in samples],
            time_days=[sample.time_days for sample in samples],
        )

    def timeline_length(self, sample: AdmissionSample) -> int:
        """Return the exact post-aggregation token count without padding a batch."""
        return len(self._aggregate_sample(sample))

    @staticmethod
    def _collate_global_summaries(samples: Sequence[AdmissionSample]) -> _GlobalSummaryRows:
        max_length = max((len(sample.numeric_summaries) for sample in samples), default=0)
        concept_ids: list[list[int]] = []
        features: list[list[list[float]]] = []
        masks: list[list[list[bool]]] = []
        attention: list[list[bool]] = []
        lengths: list[int] = []

        for sample in samples:
            lengths.append(len(sample.numeric_summaries))
            row_ids: list[int] = []
            row_features: list[list[float]] = []
            row_masks: list[list[bool]] = []
            for summary in sample.numeric_summaries:
                values = {
                    "measurement_count_log1p": summary.measurement_count_log1p,
                    **{name: summary.scaled_values.get(name, 0.0) for name in SUMMARY_SCALED_FIELDS},
                    "normalization_available": float(summary.normalization_available),
                    "unseen_value_count_log1p": math.log1p(summary.unseen_value_count),
                }
                present = {
                    "measurement_count_log1p": True,
                    **{name: not summary.missing_mask.get(name, True) for name in SUMMARY_SCALED_FIELDS},
                    "normalization_available": True,
                    "unseen_value_count_log1p": True,
                }
                row_ids.append(summary.concept_id)
                row_features.append([values[name] for name in GLOBAL_SUMMARY_FEATURE_NAMES])
                row_masks.append([present[name] for name in GLOBAL_SUMMARY_FEATURE_NAMES])

            padding = max_length - len(row_ids)
            concept_ids.append(row_ids + [0] * padding)
            features.append(row_features + [[0.0] * len(GLOBAL_SUMMARY_FEATURE_NAMES) for _ in range(padding)])
            masks.append(row_masks + [[False] * len(GLOBAL_SUMMARY_FEATURE_NAMES) for _ in range(padding)])
            attention.append([True] * len(row_ids) + [False] * padding)

        return _GlobalSummaryRows(
            lengths=lengths,
            concept_ids=concept_ids,
            features=features,
            feature_masks=masks,
            attention_mask=attention,
        )

    def _aggregate_sample(self, sample: AdmissionSample) -> list[_TimelineToken]:
        """按完整聚合键生成 token，并汇总数值趋势、时距和状态比例。"""
        groups: dict[tuple[int, int, int, int, int, int, int], _TokenAccumulator] = {}
        for event in sample.events:
            if event.feature_layer not in self.config.included_layers:
                continue
            window_id = _window_id(event)
            domain_id = _lookup(self._domain_index, _canonical(event.event_domain))
            if event.value_type == "binary_exposure":
                # 治疗途径和记录来源只用于上游临床判定。进入时间轴后，同一
                # 日窗、同一治疗类别只有一个二元暴露 token。
                unit_id = 2
                route_id = 2
                evidence_id = 2
                encoder_flag = 0
            else:
                unit_id = _lookup(self._unit_index, _canonical(event.unit)) if event.numeric_mask else 2
                route_id = 2
                evidence_id = _lookup(self._evidence_index, _canonical(event.evidence_type))
                encoder_flag = int(event.model_scope == "main_model_encoder")
            key = (window_id, event.concept_id, domain_id, unit_id, route_id, evidence_id, encoder_flag)
            groups.setdefault(key, _TokenAccumulator(events=[])).events.append(event)

        rows = []
        for key, accumulator in groups.items():
            window_id, concept_id, domain_id, unit_id, route_id, evidence_id, encoder_flag = key
            ordered = sorted(accumulator.events, key=lambda event: (event.hours_from_admission, event.event_id))
            value_types = {event.value_type for event in ordered}
            if len(value_types) != 1:
                raise ValueError(
                    "mixed value types in one concept-window token: "
                    f"hadm_id={sample.hadm_id}, concept_id={concept_id}, "
                    f"concept_names={sorted({event.concept_name for event in ordered})}, "
                    f"value_types={sorted(value_types)}"
                )
            binary_exposure = value_types == {"binary_exposure"}
            if binary_exposure:
                ordered = _deduplicate_treatment_events(ordered)
            numeric = [event for event in ordered if event.numeric_mask]
            values = [event.value_scaled for event in numeric]
            times = [event.hours_from_admission for event in numeric]
            sd = _sample_sd(values)
            slope = _slope(times, values)
            reliable_time = window_id not in {16, 17}
            first_hours = max(0.0, ordered[0].hours_before_discharge)
            last_hours = max(0.0, ordered[-1].hours_before_discharge)
            span = max(0.0, ordered[-1].hours_from_admission - ordered[0].hours_from_admission)
            durations = [event.duration_hours for event in ordered if event.duration_mask]
            statuses = [event.status for event in ordered if event.status.strip()]
            status_categories = [_status_categories(status) for status in statuses]

            def status_fraction(category: str) -> float:
                return sum(category in categories for categories in status_categories) / len(statuses) if statuses else 0.0

            exposure_duration_candidates = [*durations]
            if binary_exposure and len(ordered) > 1 and reliable_time:
                exposure_duration_candidates.append(span)
            exposure_duration = (
                max(exposure_duration_candidates)
                if binary_exposure and exposure_duration_candidates
                else None
            )
            feature_values = {
                "count_log1p": math.log1p(1 if binary_exposure else len(ordered)),
                "first_scaled": values[0] if values else 0.0,
                "last_scaled": values[-1] if values else 0.0,
                "min_scaled": min(values) if values else 0.0,
                "max_scaled": max(values) if values else 0.0,
                "mean_scaled": sum(values) / len(values) if values else 0.0,
                "sd_scaled": sd if sd is not None else 0.0,
                "delta_scaled": values[-1] - values[0] if values else 0.0,
                "slope_scaled_per_hour": slope if slope is not None else 0.0,
                "hours_before_discharge_first_log1p": math.log1p(first_hours) if reliable_time else 0.0,
                "hours_before_discharge_last_log1p": math.log1p(last_hours) if reliable_time else 0.0,
                "observation_span_hours_log1p": math.log1p(span) if reliable_time else 0.0,
                "duration_hours_max_log1p": (
                    math.log1p(exposure_duration)
                    if exposure_duration is not None
                    else math.log1p(max(durations)) if durations else 0.0
                ),
                "normalization_available_fraction": (
                    sum(event.normalization_available for event in numeric) / len(numeric) if numeric else 0.0
                ),
                "status_administered_fraction": (
                    1.0 if binary_exposure else status_fraction("administered")
                ),
                "status_not_given_fraction": status_fraction("not_given"),
                "status_flush_fraction": status_fraction("flush"),
                "status_not_flushed_fraction": status_fraction("not_flushed"),
                "status_started_fraction": status_fraction("started"),
                "status_stopped_fraction": status_fraction("stopped"),
                "status_paused_fraction": status_fraction("paused"),
                "status_changed_fraction": status_fraction("changed"),
                "status_abnormal_fraction": status_fraction("abnormal"),
                "status_susceptible_fraction": status_fraction("susceptible"),
                "status_resistant_fraction": status_fraction("resistant"),
            }
            feature_present = {
                "count_log1p": True,
                "first_scaled": bool(values),
                "last_scaled": bool(values),
                "min_scaled": bool(values),
                "max_scaled": bool(values),
                "mean_scaled": bool(values),
                "sd_scaled": sd is not None,
                "delta_scaled": bool(values),
                "slope_scaled_per_hour": slope is not None,
                "hours_before_discharge_first_log1p": reliable_time,
                "hours_before_discharge_last_log1p": reliable_time,
                "observation_span_hours_log1p": reliable_time,
                "duration_hours_max_log1p": (
                    exposure_duration is not None if binary_exposure else bool(durations)
                ),
                "normalization_available_fraction": bool(numeric),
                "status_administered_fraction": binary_exposure or bool(statuses),
                "status_not_given_fraction": bool(statuses),
                "status_flush_fraction": bool(statuses),
                "status_not_flushed_fraction": bool(statuses),
                "status_started_fraction": bool(statuses),
                "status_stopped_fraction": bool(statuses),
                "status_paused_fraction": bool(statuses),
                "status_changed_fraction": bool(statuses),
                "status_abnormal_fraction": bool(statuses),
                "status_susceptible_fraction": bool(statuses),
                "status_resistant_fraction": bool(statuses),
            }
            rows.append(_TimelineToken(
                concept_id=concept_id,
                domain_id=domain_id,
                window_id=window_id,
                unit_id=unit_id,
                route_id=route_id,
                evidence_id=evidence_id,
                encoder_flag=encoder_flag,
                features=[feature_values[name] for name in TIMELINE_FEATURE_NAMES],
                feature_masks=[feature_present[name] for name in TIMELINE_FEATURE_NAMES],
            ))

        rows.sort(key=lambda row: (
            row.window_id, row.domain_id, row.concept_id, row.unit_id,
            row.route_id, row.evidence_id, row.encoder_flag,
        ))
        if not rows:
            # 白名单收紧后，个别住院可能没有任何保留的结构化事件。
            # 使用全掩码的 <MISSING> 占位符避免 Transformer 收到全 padding；
            # 该 token 不表示任何临床事实。
            rows.append(_TimelineToken(
                concept_id=2,
                domain_id=2,
                window_id=17,
                unit_id=2,
                route_id=2,
                evidence_id=2,
                encoder_flag=0,
                features=[0.0] * len(TIMELINE_FEATURE_NAMES),
                feature_masks=[False] * len(TIMELINE_FEATURE_NAMES),
            ))
        return rows
````
<!-- END REVIEW_FRAGMENT -->

## model_loader/validate_loader.py · lines 1–139

<!-- REVIEW_FRAGMENT {"path":"model_loader/validate_loader.py","part":1,"start_line":1,"end_line":139,"source_sha256":"a06f1c823fa5cb36de2d1990037babf87a136e0eaae6ce1c0a42b76466f5b79f","fragment_sha256":"a06f1c823fa5cb36de2d1990037babf87a136e0eaae6ce1c0a42b76466f5b79f","added_terminal_newline":false} -->
````python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from aecopd_loader import AecopdDataBundle, buffered_shuffle


EXPECTED_ADMISSIONS = {"train": 4008, "validation": 797, "test": 851}
EXPECTED_PATIENTS = {"train": 2316, "validation": 486, "test": 524}
EXPECTED_TOTALS = {"events": 2_446_659, "numeric_summaries": 324_383, "text_chunks": 154_168}


def check(name: str, observed: Any, expected: Any, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "observed": observed, "expected": expected, "passed": bool(passed), "detail": detail}


def validate(data_dir: Path, artifacts_path: Path, report_path: Path) -> dict[str, Any]:
    bundle = AecopdDataBundle(data_dir)
    artifacts = bundle.fit_preprocessing(artifacts_path)

    admission_counts: Counter[str] = Counter()
    patients: dict[str, set[int]] = defaultdict(set)
    subject_splits: dict[int, set[str]] = defaultdict(set)
    hadm_ids: set[int] = set()
    hadm_ids_by_split: dict[str, list[int]] = defaultdict(list)
    label_counts: Counter[int] = Counter()
    normalization_methods: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    violations: Counter[str] = Counter()
    maxima = {"events_per_admission": 0, "summaries_per_admission": 0, "texts_per_admission": 0, "text_tokens": 0}

    for sample in bundle.iter_all_samples(artifacts=artifacts, include_text=True):
        admission_counts[sample.data_split] += 1
        patients[sample.data_split].add(sample.subject_id)
        subject_splits[sample.subject_id].add(sample.data_split)
        hadm_ids_by_split[sample.data_split].append(sample.hadm_id)
        label_counts[sample.event_type] += 1
        if sample.hadm_id in hadm_ids:
            violations["duplicate_hadm"] += 1
        hadm_ids.add(sample.hadm_id)
        if not 0.0 <= sample.time_days <= 365.0:
            violations["label_time_out_of_range"] += 1

        for value in sample.static_continuous.values():
            if not math.isfinite(value):
                violations["static_nonfinite"] += 1

        totals["events"] += len(sample.events)
        totals["numeric_summaries"] += len(sample.numeric_summaries)
        totals["text_chunks"] += len(sample.text_chunks)
        maxima["events_per_admission"] = max(maxima["events_per_admission"], len(sample.events))
        maxima["summaries_per_admission"] = max(maxima["summaries_per_admission"], len(sample.numeric_summaries))
        maxima["texts_per_admission"] = max(maxima["texts_per_admission"], len(sample.text_chunks))

        previous_time = -math.inf
        for event in sample.events:
            if event.hours_from_admission < previous_time:
                violations["event_time_not_sorted"] += 1
            previous_time = event.hours_from_admission
            if not math.isfinite(event.value_scaled):
                violations["event_scaled_nonfinite"] += 1
            if event.numeric_mask:
                normalization_methods[event.normalization_method] += 1
                if not event.normalization_method:
                    violations["numeric_without_method"] += 1

        for summary in sample.numeric_summaries:
            if not math.isfinite(summary.measurement_count_log1p):
                violations["summary_count_nonfinite"] += 1
            if any(not math.isfinite(value) for value in summary.scaled_values.values()):
                violations["summary_scaled_nonfinite"] += 1

        role_ranks: dict[str, list[int]] = defaultdict(list)
        for text in sample.text_chunks:
            maxima["text_tokens"] = max(maxima["text_tokens"], text.token_count)
            role_ranks[text.document_role].append(text.document_rank)
            if text.token_count <= 0 or text.token_count > 32_000:
                violations["text_token_rule"] += 1
            if text.chunk_text == "":
                violations["empty_text_chunk"] += 1
        for ranks in role_ranks.values():
            if len(ranks) != len(set(ranks)):
                violations["duplicate_text_rank"] += 1

    deterministic_checks = {}
    for split, ids in hadm_ids_by_split.items():
        first = list(buffered_shuffle(ids, 256, 20260912))[:64]
        second = list(buffered_shuffle(ids, 256, 20260912))[:64]
        deterministic_checks[split] = first == second

    checks: list[dict[str, Any]] = []
    checks.append(check("住院键总数", len(hadm_ids), 5656, len(hadm_ids) == 5656))
    for split in ("train", "validation", "test"):
        checks.append(check(f"{split}住院数", admission_counts[split], EXPECTED_ADMISSIONS[split], admission_counts[split] == EXPECTED_ADMISSIONS[split]))
        checks.append(check(f"{split}患者数", len(patients[split]), EXPECTED_PATIENTS[split], len(patients[split]) == EXPECTED_PATIENTS[split]))
        checks.append(check(f"{split}固定种子批次可重复", deterministic_checks.get(split), True, deterministic_checks.get(split) is True))
    cross_split = sum(1 for splits in subject_splits.values() if len(splits) > 1)
    checks.append(check("患者跨数据集数", cross_split, 0, cross_split == 0))
    checks.append(check("标签分布", dict(sorted(label_counts.items())), {0: 2881, 1: 1755, 2: 1020}, label_counts == Counter({0: 2881, 1: 1755, 2: 1020})))
    for key, expected in EXPECTED_TOTALS.items():
        checks.append(check(key, totals[key], expected, totals[key] == expected))
    checks.append(check("训练未见数值事件", normalization_methods["unseen_in_train"], 51, normalization_methods["unseen_in_train"] == 51, "缩放值为0且availability=false"))
    checks.append(check("所有装载器违例", sum(violations.values()), 0, sum(violations.values()) == 0, json.dumps(violations, ensure_ascii=False, sort_keys=True)))

    report = {
        "schema_version": "aecopd-model-v1-loader-validation-1",
        "passed": all(item["passed"] for item in checks),
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "normalization_methods": dict(sorted(normalization_methods.items())),
        "maxima": maxima,
        "violations": dict(sorted(violations.items())),
        "validation_scope": "model-independent data assembly only; no model, loss, prediction, discrimination, or calibration metrics",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the AECOPD model_v1 streaming data loader without training a model.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.data_dir, args.artifacts, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
````
<!-- END REVIEW_FRAGMENT -->

## model_loader/validate_timeline_collator.py · lines 1–224

<!-- REVIEW_FRAGMENT {"path":"model_loader/validate_timeline_collator.py","part":1,"start_line":1,"end_line":224,"source_sha256":"f549ff5ae85123a44744fec38fd17e017a40fa16a4192d5da0198cadf23ae70d","fragment_sha256":"f549ff5ae85123a44744fec38fd17e017a40fa16a4192d5da0198cadf23ae70d","added_terminal_newline":false} -->
````python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from aecopd_loader import AecopdDataBundle, _is_treatment_exposure
from timeline_collator import SharedTimelineCollator, TimelineConfig


def percentile(values: Sequence[int], probability: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round(probability * (len(ordered) - 1))]


def validate(
    data_dir: Path,
    artifacts_path: Path,
    report_path: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    bundle = AecopdDataBundle(data_dir)
    artifacts = bundle.fit_preprocessing(artifacts_path)
    collator = SharedTimelineCollator(TimelineConfig(max_tokens=None), artifacts=artifacts)

    admission_count = 0
    raw_event_count = 0
    timeline_token_count = 0
    global_summary_count = 0
    timeline_lengths: list[int] = []
    violations: Counter[str] = Counter()
    window_token_counts: Counter[str] = Counter()
    unknown_concept_events: Counter[str] = Counter()
    duration_event_count = 0
    duration_token_count = 0
    mapped_layer_event_counts: Counter[str] = Counter()
    binary_exposure_event_count = 0
    invalid_binary_exposure_event_count = 0

    def consume(samples: list[Any]) -> None:
        nonlocal admission_count, raw_event_count, global_summary_count
        nonlocal duration_event_count, timeline_token_count, duration_token_count
        nonlocal binary_exposure_event_count, invalid_binary_exposure_event_count
        batch = collator(samples)
        admission_count += len(samples)
        raw_event_count += sum(len(sample.events) for sample in samples)
        global_summary_count += sum(len(sample.numeric_summaries) for sample in samples)
        duration_event_count += sum(event.duration_mask for sample in samples for event in sample.events)
        mapped_layer_event_counts.update(
            event.feature_layer for sample in samples for event in sample.events
        )
        binary_exposure_event_count += sum(
            event.value_type == "binary_exposure"
            for sample in samples
            for event in sample.events
        )
        invalid_binary_exposure_event_count += sum(
            event.value_type == "binary_exposure"
            and (
                not _is_treatment_exposure(
                    {"status": event.status, "evidence_type": event.evidence_type}
                )
                or event.route != ""
            )
            for sample in samples
            for event in sample.events
        )
        timeline_lengths.extend(batch.timeline_lengths)
        timeline_token_count += sum(batch.timeline_lengths)
        for sample in samples:
            unknown_concept_events[sample.data_split] += sum(event.concept_id == 1 for event in sample.events)
        _validate_batch(batch, violations, window_token_counts)
        duration_index = batch.timeline_feature_names.index("duration_hours_max_log1p")
        duration_token_count += sum(
            batch.timeline_feature_masks[i][j][duration_index]
            for i, length in enumerate(batch.timeline_lengths)
            for j in range(length)
        )

    pending: list[Any] = []
    for sample in bundle.iter_all_samples(artifacts=artifacts, include_text=False):
        pending.append(sample)
        if len(pending) == batch_size:
            consume(pending)
            pending = []
    if pending:
        consume(pending)

    maximum = max(timeline_lengths, default=0)
    compression = 1.0 - timeline_token_count / raw_event_count if raw_event_count else 0.0
    event_vocabulary = artifacts["event_vocabulary"]
    route_vocabulary = artifacts["event_route_vocabulary"]
    feature_contract = artifacts.get("clinical_feature_contract")
    checks = [
        _check("住院数", admission_count, 5656, admission_count == 5656),
        _check("映射后结构化事件非空", raw_event_count, ">0", raw_event_count > 0),
        _check("映射后全局数值摘要非空", global_summary_count, ">0", global_summary_count > 0),
        _check("训练折未知概念事件", unknown_concept_events["train"], 0, unknown_concept_events["train"] == 0),
        _check(
            "给药途径不进入新词表",
            route_vocabulary,
            ["<PAD>", "<UNK>", "<MISSING>"],
            route_vocabulary == ["<PAD>", "<UNK>", "<MISSING>"],
        ),
        _check("临床特征合同存在", bool(feature_contract), True, bool(feature_contract)),
        _check("治疗二元暴露事件非空", binary_exposure_event_count, ">0", binary_exposure_event_count > 0),
        _check(
            "未执行/暂停/未开始/执行不确定不得形成暴露",
            invalid_binary_exposure_event_count,
            0,
            invalid_binary_exposure_event_count == 0,
        ),
        _check("默认1280-token上限", maximum, "<=1280", maximum <= 1280),
        _check("collator违例", sum(violations.values()), 0, sum(violations.values()) == 0),
    ]
    report = {
        "schema_version": "aecopd-curated-clinical-timeline-validation-1",
        "passed": all(item["passed"] for item in checks),
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "time_windows": list(collator([]).timeline_window_names),
        "timeline_feature_names": list(collator([]).timeline_feature_names),
        "counts": {
            "admissions": admission_count,
            "mapped_events": raw_event_count,
            "timeline_tokens": timeline_token_count,
            "global_numeric_summaries": global_summary_count,
            "duration_events": duration_event_count,
            "tokens_with_duration": duration_token_count,
            "binary_exposure_events": binary_exposure_event_count,
            "invalid_binary_exposure_events": invalid_binary_exposure_event_count,
            "event_vocabulary_size": len(event_vocabulary),
            "summary_vocabulary_size": len(artifacts["numeric_summary_vocabulary"]),
        },
        "feature_contract": feature_contract,
        "feature_audit": artifacts.get("clinical_feature_audit"),
        "mapped_layer_event_counts": dict(sorted(mapped_layer_event_counts.items())),
        "compression": {
            "mapped_event_to_timeline_reduction": compression,
            "mapped_events_per_admission_mean": (
                raw_event_count / admission_count if admission_count else 0.0
            ),
            "timeline_tokens_per_admission_mean": (
                timeline_token_count / admission_count if admission_count else 0.0
            ),
        },
        "timeline_length_distribution": {
            "min": min(timeline_lengths, default=0),
            "median": percentile(timeline_lengths, 0.50),
            "p90": percentile(timeline_lengths, 0.90),
            "p95": percentile(timeline_lengths, 0.95),
            "p99": percentile(timeline_lengths, 0.99),
            "max": maximum,
            "over_256": sum(value > 256 for value in timeline_lengths),
            "over_512": sum(value > 512 for value in timeline_lengths),
            "over_640": sum(value > 640 for value in timeline_lengths),
            "over_768": sum(value > 768 for value in timeline_lengths),
            "over_1024": sum(value > 1024 for value in timeline_lengths),
            "over_1280": sum(value > 1280 for value in timeline_lengths),
        },
        "window_token_counts": dict(sorted(window_token_counts.items())),
        "unknown_concept_event_counts": dict(sorted(unknown_concept_events.items())),
        "violations": dict(sorted(violations.items())),
        "validation_scope": "full 5,656-admission model-independent shared-timeline collator dry run; no model training",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _validate_batch(batch: Any, violations: Counter[str], window_counts: Counter[str]) -> None:
    for i, length in enumerate(batch.timeline_lengths):
        if sum(batch.timeline_attention_mask[i]) != length:
            violations["timeline_attention_length_mismatch"] += 1
        observed_windows = batch.timeline_window_ids[i][:length]
        if observed_windows != sorted(observed_windows):
            violations["timeline_not_sorted"] += 1
        for j in range(length):
            if batch.timeline_concept_ids[i][j] == 0:
                violations["real_token_has_pad_concept"] += 1
            if any(not math.isfinite(value) for value in batch.timeline_features[i][j]):
                violations["timeline_nonfinite"] += 1
            window_id = batch.timeline_window_ids[i][j]
            window_counts[batch.timeline_window_names[window_id]] += 1
        for j in range(length, len(batch.timeline_attention_mask[i])):
            if batch.timeline_attention_mask[i][j]:
                violations["padding_marked_real"] += 1
            if batch.timeline_concept_ids[i][j] != 0:
                violations["padding_concept_nonzero"] += 1

    for i, length in enumerate(batch.global_summary_lengths):
        if sum(batch.global_summary_attention_mask[i]) != length:
            violations["summary_attention_length_mismatch"] += 1
        for j in range(length):
            if any(not math.isfinite(value) for value in batch.global_summary_features[i][j]):
                violations["summary_nonfinite"] += 1


def _check(name: str, observed: object, expected: object, passed: bool) -> dict[str, object]:
    return {"name": name, "observed": observed, "expected": expected, "passed": passed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the final shared-timeline collator without training.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    report = validate(args.data_dir, args.artifacts, args.report, batch_size=args.batch_size)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
````
<!-- END REVIEW_FRAGMENT -->
