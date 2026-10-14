"""保守、可审计的抽取检查，不是医学真值分类器；不改写原文。"""
from __future__ import annotations

import re

CHECK_VERSION = 'targeted_checks_2026-10-04'

# 合成示例，不使用新验证文书的答案做提示。
EXAMPLES = {
    'state': '''Examples (synthetic):
NOTE: "COPD. SpO2 97% RA. He was hypoxic on admission and was weaned off oxygen."
OUTPUT: {"oxygenation":["He was hypoxic on admission and was weaned off oxygen."],"oxygen_requirement_transition":["He was hypoxic on admission and was weaned off oxygen."]}
NOTE: "She was extubated in the PACU after elective abdominal surgery."
OUTPUT: {}
NRB/nasal cannula provide oxygen; do not classify them as NIV or intubation.
Copy the shortest COMPLETE relevant span, not a mixed unrelated paragraph.''',
    'history': '''Examples (synthetic):
NOTE: "PMH: COPD, gout, hypertension. Severe COPD requiring home oxygen. Prior admission for COPD exacerbation."
OUTPUT: {"baseline_respiratory_severity":["Severe COPD requiring home oxygen."],"prior_respiratory_exacerbations":["Prior admission for COPD exacerbation."]}
COPD alone is not severity, an exacerbation history, or another respiratory disease.
Do not copy an entire unrelated diagnosis list. Keep explicit uncertainty.''',
    'treatment': '''Examples (synthetic):
NOTE: "1. Fluticasone NASAL 1 spray NU DAILY\n2. Albuterol 2 puffs IH PRN\n# UTI: Ceftriaxone started.\n# Pneumonia: Completed azithromycin for pneumonia."
OUTPUT: {"maintenance_rescue_inhalers":["2. Albuterol 2 puffs IH PRN"],"respiratory_antiinfective_course":["Completed azithromycin for pneumonia."]}
Extract individual respiratory drug lines, not the full mixed medication list. Respiratory drug indication must be supported in the provided context. Oral roflumilast is not an inhaler; do not invent a new field for it.''',
    'tests': '''Examples (synthetic):
NOTE: "WBC-14.1* Hgb-11.8* pO2-19*. CXR showed no pneumonia. Small right pleural effusion. Sputum culture grew commensal flora."
OUTPUT: {"respiratory_infection":["CXR showed no pneumonia."],"thoracic_imaging":["Small right pleural effusion."],"respiratory_microbiology":["Sputum culture grew commensal flora."]}
Do not interpret numeric tables. Subcutaneous abdominal emphysema is not pulmonary emphysema.
Negative assessed disease findings are different from normal boilerplate.''',
    'management': '''Examples (synthetic):
NOTE: "Contact: wife. Code: Full. Return if short of breath. Lives alone and home nursing was arranged. Refuses prescribed home oxygen. Outpatient pulmonary rehab was recommended."
OUTPUT: {"home_support_barriers":["Lives alone and home nursing was arranged."],"respiratory_adherence":["Refuses prescribed home oxygen."],"respiratory_followup_rehab":["Outpatient pulmonary rehab was recommended."]}
Do not infer inhaler technique from receiving nebulizers. Keep plans as plans.
Do not output an unmapped audit field: this task is not an audit of the full catalog.''',
}


def assess(field: str, quote: str) -> tuple[str, str]:
    """reject=明确字段冲突；needs_review=不能可靠自动判断；pass=仅通过检查。"""
    clean = ' '.join(quote.split())
    if field in {'baseline_respiratory_severity', 'prior_respiratory_exacerbations',
                 'other_respiratory_conditions'} and re.fullmatch(
                     r'(?:\d+[.)]\s*)?(?:COPD|chronic obstructive pulmonary disease)[.;:]?', clean, re.I):
        return 'reject', 'bare_copd_does_not_support_this_field'
    if field == 'maintenance_rescue_inhalers':
        nasal = re.search(r'\b(?:nasal|flonase|nasonex)\b|\bNU\b', clean, re.I)
        if nasal:
            # 多药混合句可能同时含正确吸入剂，不直接删除全部。
            lines = [line for line in quote.splitlines() if line.strip()]
            if len(lines) <= 1 and not re.search(r'\b(?:albuterol|salmeterol|tiotropium|advair)\b', clean, re.I):
                return 'reject', 'nasal_not_inhaled_respiratory_regimen'
            if re.search(r'ipratropium\s*0\.06%', clean, re.I) and len(lines) <= 1:
                return 'reject', 'ipratropium_nasal_route'
            return 'needs_review', 'mixed_nasal_and_other_medication_context'
    if field == 'ventilation_support' and re.search(r'\bNRB\b|non[- ]?rebreather', clean, re.I):
        if not re.search(r'\b(?:BiPAP|CPAP|NIV|intubat\w*|tracheostom\w*)\b', clean, re.I):
            return 'reject', 'oxygen_device_not_ventilatory_support'
    if field in {'oxygenation', 'hypercapnia_acidosis', 'respiratory_lab_interpretation'}:
        if re.fullmatch(r'(?:O2\s*sat\s*[:=]?\s*|SpO2\s*[:=]?\s*|sat\s*[:=]?\s*)?\d+(?:\.\d+)?\s*%?\s*(?:RA|room air|on\s*\d+\s*L(?:\s*NC)?)', clean, re.I):
            return 'reject', 'isolated_oxygen_number_not_clinical_interpretation'
        if re.search(r'\b(?:WBC|Hgb|pCO2|pO2|HCO3)[-:]\s*\d', clean, re.I) and not re.search(r'\b(?:hypox\w*|hypoxic|acidosis|hypercap\w*|anemi\w*|leukocytosis|improv\w*|resolved)\b', clean, re.I):
            return 'needs_review', 'raw_lab_table_without_explicit_interpretation'
    if field == 'care_limits_ama' and re.fullmatch(r'(?:code\s*:?\s*)?full(?:\s*code)?[.]?', clean, re.I):
        return 'reject', 'full_code_template'
    if field == 'respiratory_adherence' and re.search(r'not (?:assessed|documented)', clean, re.I):
        return 'needs_review', 'not_assessed_is_not_adherence'
    if len(re.findall(r'(?:^|[\r\n])\s*\d+[.)]\s', quote)) >= 4:
        return 'needs_review', 'mixed_numbered_list_requires_focused_span'
    if len(clean) > 900:
        return 'needs_review', 'large_span_requires_relevance_review'
    return 'pass', 'passed_limited_checks_not_clinically_verified'


def gate_evidence(evidence: dict) -> dict:
    passed, review, rejected = {}, [], []
    for field, values in evidence.items():
        passed[field] = []
        for item in values:
            decision, reason = assess(field, item['quote'])
            audited = dict(item, field=field, check_decision=decision, check_reason=reason)
            if decision == 'pass':
                passed[field].append(audited)
            elif decision == 'needs_review':
                review.append(audited)
            else:
                rejected.append(audited)
    states = {field: ('candidate_present' if passed[field] else 'withheld_pending_review'
                     if any(q['field'] == field for q in review) else 'rejected_candidates'
                     if any(q['field'] == field for q in rejected) else 'no_quote_returned_not_normal')
              for field in passed}
    return {'version': CHECK_VERSION, 'passed_evidence': passed, 'needs_review': review,
            'rejected': rejected, 'field_states': states, 'clinical_truth_validated': False}
