"""有限明确呼吸事实规则；未知时间不自动升格为出院状态。"""
import re

def rx(pattern):
    return re.compile(pattern, re.I)

DISCHARGE = rx(r'\b(?:(?:at|on|upon|by|before|prior to|time of)\s+(?:the\s+)?discharge|discharged|discharging)\b')
FORBIDDEN = (
    ('conditional_or_plan', rx(r'\b(?:if|unless|should|will|would|plan\w*|recommend\w*|advis\w*|scheduled|consider\w*)\b')),
    ('post_discharge', rx(r'\b(?:after|following)\s+(?:the\s+)?discharge\b|\bpost[- ]discharge\b')),
    ('historical_or_baseline', rx(r'\b(?:history of|previous\w*|prior (?!to discharge)|baseline|at home|at admission|on admission|presented with)\b')),
    ('other_experiencer', rx(r'\b(?:mother|father|wife|husband|daughter|son|family member)\b')),
    ('uncertain', rx(r'\b(?:possibly|possible|may|might|suspected|rule out)\b')),
)
CHANGE = rx(r'\b(?:improv\w*|resolv\w*|wean\w*|persist\w*|still|remained|requiring)\b')
NEG = rx(r'\b(?:no|not|never|without|denies|denied|negative for|absence of|unable to|failed to|could not)\b')
DYSPNEA = r'(?:dyspn[eo]a|shortness of breath|SOB|DOE)'
OXYGEN = r'(?:oxygen|O2)'
FLOW = rx(r'\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>L(?:/min)?|lit(?:er|re)s?)\s*(?:(?:of|via|by|on)\s+)?(?:NC\b|nasal cannula\b|oxygen\b|O2\b)')
RULES = (
    ('oxygen_failed_wean', rx(r'\b(?:unable|failed|could not|not able)\b[^;.!\n]{0,35}\bwean\w*\b'), False),
    ('oxygen_weaned_room_air', rx(r'\bwean\w*\b[^;.!\n]{0,55}?\bto\s+room air\b'), True),
    ('oxygen_weaned_off', rx(r'\bwean\w*\s+(?:successfully\s+)?off\s+(?:of\s+)?(?:supplemental\s+)?'+OXYGEN+r'\b'), True),
    ('oxygen_required', rx(r'\b(?:requiring|requires?|required|remains?|remained|discharged|sent home|continues? to require)\b[^;.!\n]{0,30}?\b'+OXYGEN+r'\b'), True),
    ('room_air_status', rx(r'\b(?:on|breathing|tolerating|maintained on)\s+room air\b'), True),
    ('dyspnea', rx(r'\b'+DYSPNEA+r'\b'), False),
)

def clauses(text):
    """保留真实字符偏移；小数点不分句，不跨换行借时间或否定。"""
    cursor = 0
    for separator in re.finditer(r'[;!\n]+|(?<!\d)\.(?!\d)', text):
        if cursor < separator.start():
            yield cursor, separator.start(), text[cursor:separator.start()]
        cursor = separator.end()
    if cursor < len(text):
        yield cursor, len(text), text[cursor:]

def temporal_role(text, section):
    if DISCHARGE.search(text):
        return 'explicit_discharge'
    if section in ('discharge_condition', 'discharge_physical_exam'):
        return 'discharge_section'
    if section == 'hospital_course' and CHANGE.search(text):
        return 'inpatient_change'
    return 'unknown'

def extract(text, section):
    """返回事实及隔离理由，未命中不表示正常。范围/比较语义不自动当单值。"""
    if section not in ('hospital_course', 'discharge_condition', 'discharge_physical_exam',
                       'discharge_instructions', 'transitional_issues', 'followup_instructions'):
        return [], [{'reason': 'section_not_current', 'start': 0, 'end': len(text)}]
    facts, rejected = [], []
    for offset, end, clause in clauses(text):
        reasons = [name for name, pattern in FORBIDDEN if pattern.search(clause)]
        if reasons:
            rejected.extend(dict(reason=r, start=offset, end=end) for r in reasons)
            continue
        role = temporal_role(clause, section)
        for concept, pattern, reject_negative in RULES:
            for match in pattern.finditer(clause):
                prior = re.split(r'\b(?:but|however|yet)\b', clause[max(0, match.start()-45):match.start()], flags=re.I)[-1]
                if reject_negative and NEG.search(prior):
                    rejected.append(dict(reason='negated_'+concept, start=offset+match.start(), end=offset+match.end()))
                    continue
                polarity, name = 'affirmed', concept
                if concept == 'dyspnea':
                    direct_neg = rx(r'\b(?:no|without|denies|denied|negative for|absence of)\s+(?:(?:any|significant|new|residual|further|ongoing|complaints of|episodes of)\s+){0,3}$')
                    if direct_neg.search(prior):
                        polarity = 'negated'
                    elif NEG.search(prior):
                        rejected.append(dict(reason='ambiguous_dyspnea_negation_scope', start=offset+match.start(), end=offset+match.end()))
                        continue
                    elif rx(r'^\s+(?:had\s+)?(?:improved|resolved|subsided)\b').search(clause[match.end():]):
                        name = 'dyspnea_improved'
                facts.append(dict(concept=name, polarity=polarity, role=role, value=None, unit=None,
                    exertional=bool(rx(r'\b(?:exertion|exertional|ambulat\w*|walking|activity|DOE)\b').search(clause)),
                    start=offset+match.start(), end=offset+match.end(), clause_start=offset, clause_end=end))
        for match in FLOW.finditer(clause):
            prior = clause[max(0, match.start()-45):match.start()]
            if rx(r'\d\s*(?:[-–]|to)\s*$|[<>]=?\s*$|\b(?:at least|at most|more than|less than)\s*$').search(prior):
                rejected.append(dict(reason='flow_range_not_scalar', start=offset+match.start(), end=offset+match.end()))
                continue
            if NEG.search(prior):
                rejected.append(dict(reason='negated_oxygen_flow', start=offset+match.start(), end=offset+match.end()))
                continue
            if rx(r'\b(?:wean\w*|reduced|decreased)\b[^;.!]{0,30}\bfrom\s*$').search(prior):
                rejected.append(dict(reason='transition_start_not_latest_flow', start=offset+match.start(), end=offset+match.end()))
                continue
            facts.append(dict(concept='oxygen_flow_litres_notation', polarity='affirmed', role=role,
                value=float(match['value']), unit='L/min' if '/min' in match['unit'].lower() else 'L_oxygen_notation', exertional=bool(rx(r'\b(?:exertion|ambulat\w*|walking|activity)\b').search(clause)),
                start=offset+match.start(), end=offset+match.end(), clause_start=offset, clause_end=end))
    return facts, rejected
