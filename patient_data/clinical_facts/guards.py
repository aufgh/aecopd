"""位置、章节、时相和药理证据的保守门禁；不更改LLM事实的极性。"""
import re
from patient_data.clinical_facts.catalog import ROUTES

DRUGS={
    'inhaled_ics':r'\b(?:fluticasone|budesonide|beclomethasone|mometasone|ciclesonide|advair|symbicort|dulera|breo|trelegy|breztri|ICS)\b',
    'inhaled_laba':r'\b(?:salmeterol|formoterol|arformoterol|vilanterol|olodaterol|indacaterol|advair|symbicort|dulera|breo|trelegy|breztri|anoro|stiolto|LABA)\b',
    'inhaled_lama':r'\b(?:tiotropium|umeclidinium|glycopyrrolate|aclidinium|revefenacin|spiriva|incruse|seebri|tudorza|yupelri|anoro|stiolto|trelegy|breztri|LAMA)\b',
}
CT_CONCEPTS={
    'emphysema':r'emphysem', 'centrilobular_emphysema':r'centrilobular.*emphysem',
    'paraseptal_emphysema':r'paraseptal.*emphysem', 'bullae':r'\b(?:bullae|bulla|bullous|blebs?)\b',
    'bronchiectasis':r'bronchiecta|(?:bronchial|airway)\s+(?:dilatation|dilation)',
    'airway_wall_thickening':r'(?:bronchi\w*|airway\w*)\s+(?:wall\s+)?thicken|thicken\w*\s+(?:along|of)\s+(?:the\s+)?(?:major\s+)?airway',
    'fibrosis':r'fibro(?:sis|tic)', 'consolidation':r'consolidat',
    'pleural_effusion':r'pleural\s+(?:effusion|fluid)',
    'enlarged_pulmonary_artery':r'pulmonary\s+(?:arter(?:y|ies|ial)(?:\s+trunk)?)',
}
def allowed_roles(field):
    if field in ROUTES['chest_ct']['fields']: return ROUTES['chest_ct']['roles']
    return next(route['roles'] for key,route in ROUTES.items() if key!='discharge_all' and field in route['fields'])

def guard(fact,task):
    start,end=fact['local_start'],fact['local_end']
    mapping=next((m for m in task['source_maps'] if m['input_start']<=start<end<=m['input_end']),None)
    if mapping is None or mapping['role'] not in allowed_roles(fact['field']): return None,'wrong_source_section'
    quote=task['text'][start:end]
    if quote!=fact['quote']: raise ValueError('local quote mismatch')
    field=fact['field']; role=mapping['role']
    if field in CT_CONCEPTS:
        normalized=' '.join(quote.split())
        if not re.search(CT_CONCEPTS[field],normalized,re.I): return None,'unsupported_CT_concept'
        if field=='enlarged_pulmonary_artery' and not re.search(r'\b(?:enlarg\w*|dilat\w*|prominent|caliber|calibre|diameter|measur\w*)\b',normalized,re.I):
            return None,'pulmonary_artery_patency_not_size'
        if field=='fibrosis' and fact['value']=='present' and re.search(r'\b(?:likely|possible|suggest\w*|may|probable)\b',normalized,re.I):
            return None,'uncertain_fibrosis_called_definite'
    if re.search(r'\b(?:if you (?:develop|experience)|call 911|return to (?:the )?(?:ER|emergency))\b',quote,re.I):
        return None,'generic_warning_template'
    if field=='oxygen_at_discharge' and not re.search(r'\b(?:oxygen|O2|RA|room air|cannula|NC|LPM)\b',quote,re.I):
        return None,'no_oxygen_evidence'
    if field=='oxygen_at_discharge' and fact['value']=='room_air' and not re.search(r'\b(?:RA|room air)\b',quote,re.I):
        return None,'no_home_oxygen_not_room_air'
    if field=='ventilation_at_discharge':
        if not re.search(r'\b(?:ventilat\w*|intubat\w*|extubat\w*|BiPAP|CPAP|NIV|NIPPV)\b',quote,re.I): return None,'oxygen_is_not_ventilation'
        if fact['value']=='weaned' and re.search(r'\b(?:unable|not|failed|failure)\b.{0,50}\b(?:wean|extubat)',quote,re.I|re.S): return None,'failed_wean_not_success'
    if field=='residual_dyspnea' and not re.search(r'\b(?:dyspn\w*|SOB|breathless\w*|shortness of breath|breathing comfortably)\b',quote,re.I):
        return None,'no_dyspnea_evidence'
    if field=='living_alone' and not re.search(r'\b(?:liv\w*|resid\w*|stays?|alone|by (?:him|her|them)self)\b',quote,re.I):
        return None,'destination_not_living_arrangement'
    if field=='care_limits':
        words={'dnr':r'\bDNR\b|do not resuscitate','dni':r'\bDNI\b|do not intubate',
               'comfort_care':r'hospice|comfort (?:care|measures)|palliative','ama':r'\bAMA\b|against medical advice'}
        if not re.search(words[fact['value']],quote,re.I): return None,'unsupported_care_limit'
    if field=='steroid_discharge_course' and not re.search(r'\b(?:pred\w*|methylpred\w*|hydrocortisone|dexamethasone|decadron|solu[- ]?medrol|steroid\w*)\b',quote,re.I):
        return None,'no_systemic_steroid_evidence'
    if field=='steroid_discharge_course' and fact['value']=='completed' and re.search(r'\b(?:discontinued|stopped)\b',quote,re.I):
        return None,'stopped_not_completed_steroid_course'
    if field=='adherence_or_device_barrier' and not re.search(r'\b(?:inhaler\w*|MDI|DPI|spacer|adher\w*|complian\w*|afford\w*|access\w*|refus\w*|noncompli\w*|medication\w*|device\w*)\b',quote,re.I):
        return None,'physiologic_burden_not_adherence'
    if field=='functional_dependence' and fact['value']=='independent' and re.fullmatch(r'(?:Activity Status:\s*)?Ambulatory\s*-\s*Independent\.?',quote.strip(),re.I):
        return None,'default_independent_template'
    if field in ('residual_dyspnea','exertional_desaturation','oxygen_at_discharge') and role not in ('discharge_condition','discharge_physical_exam'):
        context=task['text'][max(mapping['input_start'],start-150):min(mapping['input_end'],end+150)]
        if not re.search(r'\b(?:discharg\w+|leaving (?:the )?hospital|at home|home oxygen|home O2)\b',context,re.I):
            return None,'no_explicit_discharge_time_anchor'
    if field in DRUGS:
        if not re.search(DRUGS[field],quote,re.I): return None,'unsupported_inhaler_component'
        if re.search(r'\b(?:nasal|nostril|flonase|nasonex)\b',quote,re.I): return None,'nasal_route_not_pulmonary'
    if field in ROUTES['medications']['fields'] and role=='hospital_course' and fact['value'] in ('planned','prescribed','stopped'):
        context=task['text'][max(mapping['input_start'],start-200):min(mapping['input_end'],end+200)]
        if not re.search(r'\b(?:discharg\w+|home|taper|continue|remaining|complete\w*)\b',context,re.I):
            return None,'inpatient_drug_not_discharge_plan'
    if field=='respiratory_antibiotic_course':
        context=task['text'][max(mapping['input_start'],start-250):min(mapping['input_end'],end+250)]
        if not re.search(r'\b(?:COPD|AECOPD|pneumonia|respiratory|bronchitis|pulmonary|lung infection)\b',context,re.I):
            return None,'antibiotic_indication_not_pulmonary'
    output=dict(fact,quote=quote,section_role=role,original_start=mapping['original_start']+start-mapping['input_start'],
                original_end=mapping['original_start']+end-mapping['input_start'])
    return output,None
