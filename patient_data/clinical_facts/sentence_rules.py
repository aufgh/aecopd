"""仅筛选临床原句的可审计规则；主题/词面线索不是疾病事实或预测变量。"""
import hashlib
import json
import re

VERSION = 'clinical-sentence-screen-v4-source-reviewed'
# 不沿用旧8/14字段的出院残余约束；允许原文多主题，时点另存。
TOPICS = {
    'respiratory_status': r'\b(?:COPD|AECOPD|asthma|dyspn\w*|SOB|DOE|breathless\w*|shortness\s+of\s+breath|wheez\w*|cough\w*|sputum|phlegm|respiratory\s+(?:failure|distress|insufficiency)|work of breathing|accessory muscles?|mucus\s+plug\w*)\b',
    'oxygen_ventilation': r'\b(?:oxygen|O2|SpO2|room air|nasal cannula|desatur\w*|hypox\w*|hypercap\w*|BiPAP|CPAP|NIV|NIPPV|intubat\w*|extubat\w*|ventilat\w*|wean\w*|on RA|\d+\s*L\s*NC)\b',
    'respiratory_treatment': r'\b(?:predni\w*|methylpred\w*|hydrocortisone|dexamethasone|steroid\w*|solu.?medrol|albuterol|salbutamol|ipratropium|duoneb|salmeterol|formoterol|arformoterol|tiotropium|umeclidinium|budesonide|fluticasone|advair|symbicort|spiriva|incruse|trelegy|breo|anoro|combivent|inhaler\w*|nebuliz\w*|azithro\w*|doxycycl\w*|levoflox\w*|moxiflox\w*|ceftriaxone|cefepime|amoxicillin|augmentin|piperacillin|tazobactam|vancomycin|meropenem|antibiotic\w*)\b',
    'respiratory_microbiology': r'\b(?:sputum\s+cultur\w*|BAL|bronchoalveolar|tracheal\s+aspirat\w*|pseudomon\w*|haemophilus|streptococcus pneumoniae|legionella|influenza|RSV|rhinovirus|respiratory syncytial|coloniz\w*|resistan\w*)\b',
    'pulmonary_function': r'\b(?:FEV\s*1|FVC|DLCO|PFTs?|spirometr\w*|pulmonary\s+function)\b',
    'cardiopulmonary_context': r'\b(?:CHF|heart failure|pulmonary\s+hypertension|cor pulmonale|cardiomyopath\w*|chest pain|myocardial infarct\w*|acute coronary|smok\w*|tobacco|pack.?years?)\b',
    'function_frailty': r'\b(?:ADLs?|frail\w*|cachex\w*|decondition\w*|bedbound|wheelchair|walker|ambulat\w*|mobility|functional\s+(?:status|dependence)|requires?\s+assist\w*|unable\s+to\s+walk)\b',
    'management_barriers': r'\b(?:adher\w*|non.?complian\w*|not\s+(?:taking|using)|inhaler technique|spacer|cannot afford|unable to afford|financial barrier|medication access|lives?\s+(?:alone|with)|homeless|caregiver|home\s+(?:with\s+)?(?:services?|health|nursing|oxygen)|visiting nurse|VNA|SNF|skilled nursing|rehab\w*)\b',
    'care_limits': r'\b(?:DNR|DNI|do not resuscitate|do not intubate|hospice|comfort\s+(?:care|measures)|AMA|against medical advice)\b',
    'thoracic_parenchyma': r'\b(?:emphysem\w*|hyperinfl\w*|overinfl\w*|bullae|bulla|bullous|blebs?|consolidat\w*|infiltrat\w*|pneumonia|atelecta\w*|ground.?glass|opaci\w*|scar\w*|fibro\w*|reticul\w*|interstitial|(?:pulmonary|re.?expansion)\s+edema|vascular\s+congestion|lung\s+volumes?|volume\s+loss|ventilation\s+(?:at|of|in)\s+the\s+(?:left|right|bilateral)\s+lung)\b',
    'thoracic_airway': r'\b(?:bronchiecta\w*|bronchi\w*\s+(?:dilat\w*|wall|thicken\w*)|airway\s+(?:wall|thicken\w*)|mucus\s+plug\w*|mucoid\s+impaction|endobronchial)\b',
    'thoracic_pleura_vascular': r'\b(?:effusion\w*|pneumothora\w*|pleural\s+(?:fluid|thicken\w*)|pulmonary\s+(?:arter\w*|embol\w*)|cardiomegal\w*|enlarged\s+heart)\b',
    'thoracic_other': r'\b(?:nodules?|mass(?:es)?|neoplasm\w*|malignan\w*)\b',
    'thoracic_cardiac': r'\b(?:pericardial\s+effusion|heart\s+(?:size|is|appears)|cardiac silhouette)\b',
}
COMPILED = {k: re.compile(v, re.I) for k, v in TOPICS.items()}
IMAGING_TOPICS = {k for k in TOPICS if k.startswith('thoracic_')}
CUES = {
    'negation_words': r'\b(?:no|not|without|denies?|negative for|resolved|absent)\b',
    'uncertainty_words': r'\b(?:possible|possibly|may|might|probable|likely|suggest\w*|cannot exclude)\b',
    'change_words': r'\b(?:new|worsen\w*|improv\w*|resolv\w*|persist\w*|unchanged|increas\w*|decreas\w*|wean\w*|taper\w*)\b',
    'discharge_words': r'\b(?:discharg\w*|at home|home oxygen)\b',
    'historical_words': r'\b(?:previous\w*|history of|baseline|prior|formerly)\b',
    'plan_words': r'\b(?:will|plan\w*|continue|remaining|to complete|follow.?up|recommend\w*)\b',
}
WARNING = re.compile(r'\b(?:if you (?:develop|experience|have)|call\s+911|return to (?:the )?(?:ER|emergency)|seek immediate medical)\b', re.I)
FUTURE_OUTCOME = re.compile(r'\b(?:subsequently readmitted|readmitted after discharge|died after discharge|post.?discharge admission)\b', re.I)
DEVICE = re.compile(r'\b(?:tube|line|catheter|pacemaker|port|endotracheal|ETT|PICC|lead)\b', re.I)
TECHNIQUE = re.compile(r'\b(?:portable|technique|AP view|PA and lateral|contrast|dose|motion artifact)\b', re.I)
# 所有章节都保留审计，不把实验室表、家族史、联系方式变成正文证据。
EXCLUDED_ROLES = {'metadata', 'allergies', 'family_history', 'admission_labs', 'discharge_labs'}
NONTHORACIC = re.compile(r'\b(?:adrenal|renal|kidney|hepatic|liver|pancrea\w*|splenic|spleen|thyroid|prostat\w*|pelvic|uter\w*|ovari\w*|abdom\w*|colonic|colon|bowel|rectal)\b', re.I)
THORACIC_LOCATION = re.compile(r'\b(?:lung\w*|pulmonary|bronchi\w*|airway|pleur\w*|thorac\w*|chest|mediastin\w*|hilar|lobes?|basilar|apical|pericardial|heart|cardiac)\b', re.I)
NONRESP_INDICATION = re.compile(r'\b(?:UTI|urinary|renal\s+(?:collection|abscess|infection)|pyelonephritis|cellulitis|abdominal\s+infection|adrenal insufficiency|gout|skin rash)\b', re.I)
VITAL_TABLE = re.compile(r'\b(?:vitals?|BP\s*\d|HR\s*\d|RR\s*\d|O2\s+sat\s*\d)\b', re.I)
TOPICAL_DRUG = re.compile(r'\b(?:cream|ointment|topical|pruriti\w*|eczema|Flonase|nasal\s+spray)\b', re.I)
RULE_SHA = hashlib.sha256(json.dumps({'version': VERSION, 'topics': TOPICS, 'cues': CUES,
    'excluded_roles': sorted(EXCLUDED_ROLES), 'warning': WARNING.pattern,
    'future_outcome': FUTURE_OUTCOME.pattern, 'device': DEVICE.pattern,
    'technique': TECHNIQUE.pattern, 'nonthoracic': NONTHORACIC.pattern,
    'thoracic_location': THORACIC_LOCATION.pattern, 'nonresp_indication': NONRESP_INDICATION.pattern,
    'vital_table': VITAL_TABLE.pattern, 'topical_drug': TOPICAL_DRUG.pattern}, sort_keys=True).encode()).hexdigest()


def classify(text, role, kind):
    allowed = IMAGING_TOPICS if kind != 'discharge' else TOPICS.keys()
    topics = sorted(k for k in allowed if COMPILED[k].search(text))
    flags = []
    if DEVICE.search(text): flags.append('device_mentioned')
    if TECHNIQUE.search(text): flags.append('technical_mentioned')
    if WARNING.search(text): flags.append('warning_or_hypothetical')
    reason = 'selected_candidate' if topics else 'no_topic_match'
    if NONTHORACIC.search(text) and not THORACIC_LOCATION.search(text) and topics and all(k.startswith('thoracic_') for k in topics):
        reason = 'explicit_nonthoracic_finding'
    if 'respiratory_treatment' in topics:
        flags.append('drug_indication_not_inferred')
        if NONRESP_INDICATION.search(text) and set(topics) == {'respiratory_treatment'}:
            reason = 'explicit_nonrespiratory_indication'
        if TOPICAL_DRUG.search(text) and set(topics) == {'respiratory_treatment'}:
            # 不把外用/鼻用制剂当作下气道治疗；同句含其他临床主题仍保留原句。
            reason = 'topical_or_nasal_only'
    if role in EXCLUDED_ROLES: reason = 'excluded_section'
    if kind == 'discharge' and VITAL_TABLE.search(text) and set(topics) <= {'oxygen_ventilation'}:
        if not re.search(r'\b(?:wean\w*|require\w*|increas\w*|decreas\w*|discharg\w*|exert\w*|ambulat\w*|BiPAP|CPAP|intubat\w*)\b', text, re.I):
            reason = 'isolated_vital_table'
    if FUTURE_OUTCOME.search(text): reason = 'observed_future_outcome'
    # 单纯的防范模板才过滤；有其他实际陈述的混合句保留待复核。
    if WARNING.search(text) and not re.search(r'\b(?:patient|he|she|currently|still|discharged|weaned)\b', text, re.I):
        reason = 'generic_warning'
    if kind != 'discharge' and not topics and reason == 'no_topic_match':
        if DEVICE.search(text): reason = 'device_only'
        elif TECHNIQUE.search(text): reason = 'technical_only'
        elif re.search(r'\b(?:no acute|clear lungs?|lungs? (?:are )?clear|unremarkable chest|no pathologic changes in the lung)\b', text, re.I):
            topics = ['thoracic_explicit_normal']; reason = 'selected_candidate'
    cues = {name: [m.group() for m in re.finditer(pattern, text, re.I)] for name, pattern in CUES.items()}
    # 同一句的词面线索并不代表每个主题都拥有相同的肯否/时相。
    return dict(topics=topics, selected=reason == 'selected_candidate', reason=reason,
                flags=flags, context_cues={k: v for k, v in cues.items() if v})
