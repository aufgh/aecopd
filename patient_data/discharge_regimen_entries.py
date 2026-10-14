"""保存药物候选的窄条目读取；文书列示候选不等实际给药/专家gold。"""
import re
from patient_data.medication_identity import ALIASES as OLD_ALIASES
from patient_data.v6_expanded_candidates import ALIASES as ADDED_ALIASES, identity
from patient_data.compact_multisource import INHALED_ROUTES
from training.v5.clinical_concepts import medication_classes, _MED_PATTERNS, _BRANDS

CLASSES={'inhaled_steroid','laba','lama'}
MAINTENANCE=re.compile('|'.join(_MED_PATTERNS[k] for k in sorted(CLASSES)),re.I)
ALL_MOLECULES=re.compile('|'.join(_MED_PATTERNS.values()),re.I)
ALIASES=sorted(set(_BRANDS)|{a for aa in (*OLD_ALIASES.values(),*ADDED_ALIASES.values()) for a in aa},key=lambda s:(-len(s),s))
BRAND=re.compile(r'\b(?:'+'|'.join(map(re.escape,ALIASES))+r')\b',re.I)
BOUNDARY=re.compile(r'(?m)^[ \t]*(?:\d+[.)][ \t]+|[-*][ \t]+)(?=[A-Za-z])')
PREFIX=re.compile(r'^\s*(?:\d+[.)]\s*|[-*]\s*)?(?:(?P<action>discontinued|discontinue|stopped|stop|held|hold|continue)\b[\s:-]*)?',re.I)
NAME_END=re.compile(r'\d|\b(?:Sig|take|inhale|inhalation|oral|nasal|IH|INH|NEB|PO|daily|BID|TID|QID|QHS|puffs?|capsules?|tablets?)\b',re.I)
ROUTE=re.compile(r'\b(?:ORAL INHALATION|INHALATION|IH|INH|NEB|PO|NG|GT|PEG|OGT|nasal|topical|ophthalmic|otic)\b',re.I)
FORM=re.compile(r'\b(?:inhal\w*|nebul\w*|aerosol|diskus|ellipta|respimat|HFA|MDI|DPI|respules?|puffs?|handihaler)\b',re.I)
CONDITION=re.compile(r'\b(?:if|unless|until|only when)\b',re.I)
AS_NEEDED=re.compile(r'\b(?:PRN|as needed)\b',re.I)
STOP=re.compile(r'\b(?:stop\w*|discontinu\w*|hold|held|do not take)\b',re.I)
NONLISTING=re.compile(r'\b(?:previously|formerly|history of|used to|instead of home|formulary substitute)\b',re.I)
DOSE=re.compile(r'\b\d+(?:\.\d+)?(?:\s*[-/]\s*\d+(?:\.\d+)?)?\s*(?:mg|mcg|ug|puffs?|mL)\b',re.I)
FREQUENCY=re.compile(r'\b(?:daily|BID|TID|QID|QHS|q\s*\d+\s*h|twice|once|three times|four times)\b',re.I)


def intervals(text):
    starts={0,len(text)}
    starts.update(m.start() for m in BOUNDARY.finditer(text))
    output=[]
    ordered=sorted(starts)
    for a,b in zip(ordered,ordered[1:]):
        while a<b and text[a].isspace():a+=1
        while b>a and text[b-1].isspace():b-=1
        if a<b:output.append((a,b))
    return output


def read_entry(text):
    """只分析一个条目；返回精确证据位置与保守未知原因。"""
    base=dict(status='unresolved',classes=[],reason=None,clinically_verified=False,
              product_name=None,canonical_identity=None,product_span=None,route_spans=[],
              as_needed=False,dose_strings=[],frequency_strings=[])
    prefix=PREFIX.match(text);start=prefix.end();action=prefix.group('action')
    tail=text[start:];name_end=NAME_END.search(tail)
    end=start+(name_end.start() if name_end else len(tail))
    # 分号后另一药物不是本产品名或途径说明。
    semi=text.find(';',start)
    if semi>=0:end=min(end,semi)
    while end>start and text[end-1].isspace():end-=1
    while start<end and text[start].isspace():start+=1
    name=text[start:end]
    match=MAINTENANCE.match(name) or BRAND.match(name)
    if not match:return dict(base,reason='no_anchored_product_name')
    key=identity(name);ingredients={m.group().lower() for m in MAINTENANCE.finditer(name)}
    brands=[m.group().lower() for m in BRAND.finditer(name)]
    # and/分号是独立产品连接，不能按固定复方优先把整串吞为一个药。
    if re.search(r'\band\b|;',name,re.I):return dict(base,reason='multiple_products_in_header')
    branded_keys={identity(b) or 'brand:'+b for b in brands}
    if len(branded_keys)>1 and key is None:return dict(base,reason='multiple_product_identities')
    if key is not None:
        if any(w not in key.split('_') for w in ingredients):return dict(base,reason='extra_ingredient_in_header')
        if any((identity(b) is not None and identity(b)!=key) for b in brands):return dict(base,reason='extra_product_in_header')
    elif len(ingredients)>1 and not re.search(r'[-/]',name):
        return dict(base,reason='unbound_multiple_ingredients')
    scope_end=text.find(';',end)
    if scope_end<0:scope_end=len(text)
    scope=text[start:scope_end]
    # Sig里又明确出现独立药名时，不能将混杂途径套到本药。
    extra={m.group().lower() for m in ALL_MOLECULES.finditer(text[end:scope_end])}
    allowed=ingredients|set((key or '').split('_'))
    if extra-allowed:return dict(base,product_name=name,product_span=[start,end],canonical_identity=key,reason='other_drug_in_instruction_scope')
    routes=[m for m in ROUTE.finditer(text,end,scope_end)]
    route_values={m.group().upper() for m in routes}
    inh=route_values&INHALED_ROUTES
    nonpulm=route_values- INHALED_ROUTES
    if inh and nonpulm:return dict(base,product_name=name,product_span=[start,end],canonical_identity=key,reason='conflicting_local_route')
    normalized=tuple(sorted(route_values))
    # 与既有compact匹配方式一致：明确IH/INH/NEB补正规化描述，不改原句证据。
    descriptions=(scope,)+(('inhalation',) if inh else ())
    groups=medication_classes(name,product_descriptions=descriptions,routes=normalized,
                              mapping_version='clinical_reviewed_2026_09')&CLASSES
    if not groups and key is not None:
        groups=medication_classes(key.replace('_',' '),product_descriptions=descriptions,routes=normalized,
                                  mapping_version='clinical_reviewed_2026_09')&CLASSES
    if not groups:return dict(base,product_name=name,product_span=[start,end],canonical_identity=key,
                              reason='nonpulmonary_or_unresolved_form')
    result=dict(base,classes=sorted(groups),product_name=name,product_span=[start,end],canonical_identity=key,
                route_spans=[[m.start(),m.end(),m.group()] for m in routes],as_needed=bool(AS_NEEDED.search(scope)),
                dose_strings=[m.group() for m in DOSE.finditer(text,end,scope_end)],
                frequency_strings=[m.group() for m in FREQUENCY.finditer(text,end,scope_end)])
    if CONDITION.search(text[:scope_end]) or NONLISTING.search(text[:scope_end]):
        return dict(result,status='conditional_unknown',reason='conditional_or_historical_scope')
    if action and action.lower()!='continue':return dict(result,status='explicit_stop',reason='direct_stop_prefix')
    if STOP.search(text[end:scope_end]):return dict(result,status='conditional_unknown',reason='later_action_scope_unresolved')
    return dict(result,status='listed',reason='anchored_discharge_list_entry')


def read_unit(text):
    result=[]
    for a,b in intervals(text):
        item=read_entry(text[a:b]);item['entry_span']=[a,b];item['entry_quote']=text[a:b]
        result.append(item)
    return result


def masked(text):
    """只显示固定药物/剂型/状态词，所有数字与其他词遮蔽。"""
    allowed=set(re.findall(r'[A-Za-z]+',' '.join(ALIASES)+' '+' '.join(_MED_PATTERNS.values())))
    allowed|={'sig','ih','inh','neb','po','daily','bid','tid','qid','mg','mcg','puff','puffs','capsule','capsules',
        'inhalation','diskus','hfa','handihaler','stop','stopped','hold','held','discontinued','continue','if','unless',
        'prn','as','needed','nasal','oral','topical','bromide','device','with','propionate','take','and'}
    return re.sub(r'[A-Za-z]+|\d+',lambda m:m.group().lower() if m.group().lower() in allowed else ('N' if m.group().isdigit() else 'X'),text)
