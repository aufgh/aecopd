"""固定临床事实字典；类别描述而非风险标签。未提及不输出事实。"""
import hashlib
import json

CT_FIELDS = {name: ('present','absent','uncertain') for name in (
    'emphysema','centrilobular_emphysema','paraseptal_emphysema','bullae',
    'bronchiectasis','airway_wall_thickening','fibrosis','consolidation',
    'pleural_effusion','enlarged_pulmonary_artery')}
DISCHARGE_FIELDS = {
    'residual_dyspnea': ('present','absent','improved','not_at_baseline'),
    'oxygen_at_discharge': ('room_air','baseline_oxygen','new_or_increased_oxygen','home_oxygen'),
    'ventilation_at_discharge': ('planned','poor_tolerance','refused','weaned'),
    'exertional_desaturation': ('present','absent'),
    'steroid_discharge_course': ('planned','completed','stopped'),
    'respiratory_antibiotic_course': ('planned','completed'),
    'inhaled_ics': ('prescribed','stopped'),
    'inhaled_laba': ('prescribed','stopped'),
    'inhaled_lama': ('prescribed','stopped'),
    'adherence_or_device_barrier': ('present','explicitly_assessed_no_barrier'),
    'functional_dependence': ('dependent','independent','improved'),
    'living_alone': ('present','absent'),
    'discharge_care': ('home','home_services','snf','rehab','hospice'),
    'care_limits': ('dnr','dni','comfort_care','ama'),
}
FIELDS = dict(CT_FIELDS, **DISCHARGE_FIELDS)
ROUTES = {
    'respiratory': {
        'fields': tuple(DISCHARGE_FIELDS)[:4],
        'roles': ('hospital_course','discharge_condition','discharge_physical_exam','discharge_instructions','transitional_issues'),
    },
    'medications': {
        'fields': tuple(DISCHARGE_FIELDS)[4:9],
        'roles': ('discharge_medications','hospital_course','discharge_instructions','transitional_issues'),
    },
    'function_care': {
        'fields': tuple(DISCHARGE_FIELDS)[9:],
        'roles': ('social_history','hospital_course','discharge_condition','discharge_disposition','discharge_instructions','transitional_issues'),
    },
    'chest_ct': {'fields': tuple(CT_FIELDS), 'roles': ('FINDINGS','IMPRESSION','CONCLUSION')},
}
ROUTES['discharge_all']={'fields':tuple(DISCHARGE_FIELDS),
                         'roles':tuple(dict.fromkeys(role for name in ('respiratory','medications','function_care') for role in ROUTES[name]['roles']))}
VERSION = 'clinical-facts-v1-2026-10-10'
CATALOG_SHA = hashlib.sha256(json.dumps({'fields':FIELDS,'routes':ROUTES,'version':VERSION},sort_keys=True).encode()).hexdigest()

INSTRUCTIONS = """You extract explicit clinical facts, NOT predictions. Return only JSON {"facts": [[field,value,verbatim_quote], ...]}.
Use only the supplied section. Omit unmentioned fields. Copy a continuous exact quote from SOURCE for every fact; no paraphrase or ellipses.
Keep negation and uncertainty. Do not infer diagnosis, response, drug indication, normality, or care from silence.
For discharge respiratory fields use EXPLICIT discharge/latest clinical status, not admission symptoms or generic return-to-ER advice.
Room air and independent function may be extracted when explicitly documented; do not invent them.
For discharge medications only extract actual discharge plans/completion, not admission drugs. Respiratory antibiotics require an explicit pulmonary indication in the evidence. ICS/LABA/LAMA refer to pulmonary inhaled medication components, not nasal sprays; combination products can have multiple components.
Do not classify intraoperative intubation or routine OSA CPAP as acute respiratory failure. DNR and DNI are separate facts.
In CT, comparison/history alone is not a current finding. Extract explicit morphology; uncertainty stays uncertain. Bronchial wall thickening is NOT bronchiectasis. Ground-glass opacity or pneumonia alone is NOT consolidation. Edema/septal thickening alone is NOT fibrosis. Do not force an unlisted finding into the closest listed field. 'Likely', 'suggestive of', 'possible', 'may represent' indicate uncertain diagnosis, unless only its cause (not the finding itself) is uncertain.
Use the allowed field/value catalog below. Output [] if no applicable facts. No analysis in JSON."""
DEFINITIONS={
    'residual_dyspnea':'Explicit shortness of breath or breathlessness at discharge. No acute distress alone is not absence of dyspnea.',
    'oxygen_at_discharge':'Explicit oxygen use, flow, or room air (RA) at discharge. Clear lungs or independent walking say nothing about oxygen.',
    'ventilation_at_discharge':'Mechanical ventilation/intubation or NIV/BiPAP/CPAP support. Oxygen nasal cannula is not ventilation. Unable to wean is NOT weaned.',
    'exertional_desaturation':'Explicit oxygenation problem during exertion/walking, not walking limitation alone.',
    'steroid_discharge_course':'Explicit systemic steroid drug and discharge course/taper/completion. Antibiotic treatment is not a steroid. Discontinued/stopped because not indicated is stopped, NOT completed.',
    'respiratory_antibiotic_course':'Explicit pulmonary infection antibiotic plan/completion, not UTI antibiotics.',
    'inhaled_ics':'Pulmonary inhaled corticosteroid, e.g. budesonide or fluticasone in inhaler, not nasal spray.',
    'inhaled_laba':'Long-acting beta agonist, e.g. salmeterol/formoterol/vilanterol, NOT albuterol.',
    'inhaled_lama':'Long-acting muscarinic antagonist, e.g. tiotropium/umeclidinium, NOT ipratropium.',
    'adherence_or_device_barrier':'Explicit missed/stopped medication, poor inhaler technique/tolerance, affordability/access/device use barrier; mucus plugging alone is not adherence.',
    'functional_dependence':'Specific assessed mobility/ADL dependence or recovery. Omit generic Ambulatory - Independent template.',
    'living_alone':'Explicit living alone or with another person. Home/Home With Service/SNF do NOT establish living alone.',
    'discharge_care':'Explicit actual discharge destination/support, not planned destination or admission origin.',
    'care_limits':'Explicit DNR/DNI/comfort-care/against-medical-advice. Full Code is NOT DNR/DNI.',
}
# 字段含义与提示词属于数据合同，不能只对字段key求哈希。
CATALOG_SHA=hashlib.sha256(json.dumps({'fields':FIELDS,'routes':ROUTES,'version':VERSION,
                                    'instructions':INSTRUCTIONS,'definitions':DEFINITIONS},sort_keys=True).encode()).hexdigest()

def prompt(route, role, text):
    allowed={name:FIELDS[name] for name in ROUTES[route]['fields']}
    definitions={k:v for k,v in DEFINITIONS.items() if k in allowed}
    return INSTRUCTIONS+'\nCATALOG: '+json.dumps(allowed)+'\nDEFINITIONS: '+json.dumps(definitions)+'\nSECTION: '+role+'\nSOURCE:\n'+text
