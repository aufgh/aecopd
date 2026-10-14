"""r1不变；限定氧对象/改善/动作完成/局部活动条件语义修复。"""
import re
from patient_data.discharge_resp_rules import extract as original_extract, rx, DYSPNEA, OXYGEN, FLOW

TERMINATOR=rx(r'\b(?:but|however|yet)\b')
ATTEMPT=rx(r'\b(?:attempt\w*|try|trying|instruct\w*|goal)\b')
FAILED_OBJECT=rx(r'\b(?:unable|failed|could not|not able)(?:\s+to)?(?:\s+(?:be|successfully))?\s+wean(?:ed)?\s+(?:(?:off|from|to)\s+)?(?:supplemental\s+)?(?:oxygen|O2|room air|NC|nasal cannula)\b')
ADVERB=r'(?:(?:significantly|markedly|much|somewhat|greatly|substantially|partially)\s+)?'
IMPROVED_AFTER=rx(r'^\s+(?:(?:is|was|has|had)\s+)?(?:been\s+)?'+ADVERB+r'(?:improved|resolved|subsided)\b')
IMPROVED_BEFORE=rx(r'\b(?:improved|improving|resolved)\s+(?:(?:significant|residual)\s+)?$')
IMPROVEMENT=rx(r'\b(?:improv\w*|resolv\w*|subsided)\b')
NEED_BRIDGE=r'(?:(?:on|with|home|supplemental|additional|continuous|some|low|flow|of|via|nasal|cannula|NC|L|min|liters?|litres?)\s+|\d+(?:\.\d+)?\s*(?:L(?:/min)?\s*)?){0,12}'
NEED_OBJECT=rx(r'\b(?:requiring|requires?|required|remains?|remained|discharged|sent home|continues? to require)\s+'+NEED_BRIDGE+OXYGEN+r'\b')
REST=rx(r'\b(?:at\s+rest|resting|rest)\b')
ACTIVITY=rx(r'\b(?:exertion|exertional|ambulat\w*|walking|activity|DOE)\b')

def local_scope(clause,start,end):
    """否定/改善的转折边界；保留原始坐标，不跨前一转折。"""
    left=0;right=len(clause)
    for m in TERMINATOR.finditer(clause):
        if m.end()<=start:left=m.end()
        elif m.start()>=end:right=m.start();break
    return left,right

def activity_binding(clause,match):
    """数值条件在相邻氧量/协调边界内绑定，双条件无法分清则显式ambiguous。"""
    flows=list(FLOW.finditer(clause));previous=0;following=len(clause)
    if len(flows)==1 and REST.search(clause) and ACTIVITY.search(clause):
        return 'ambiguous'
    for f in flows:
        if f.end()<=match.start():previous=f.end()
        elif f.start()>=match.end():following=f.start();break
    before=re.split(r'\b(?:and|but|however|yet)\b|,',clause[previous:match.start()],flags=re.I)[-1]
    after=re.split(r'\b(?:and|but|however|yet)\b|,',clause[match.end():following],flags=re.I)[0]
    context=before+' '+after;rest=bool(REST.search(context));active=bool(ACTIVITY.search(context))
    return 'ambiguous' if rest and active else 'rest' if rest else 'exertion' if active else 'unknown'

def extract(text,section):
    candidates,rejected=original_extract(text,section);accepted=[]
    for raw in candidates:
        r=dict(raw);concept=r['concept'];start,end=r['start'],r['end']
        clause=text[r['clause_start']:r['clause_end']];a=start-r['clause_start'];b=end-r['clause_start']
        lo,hi=local_scope(clause,a,b);left=clause[lo:a];right=clause[b:hi];scope=clause[lo:hi]
        reason=None
        if concept=='oxygen_failed_wean':
            match=FAILED_OBJECT.match(clause,a)
            if match is None:reason='wean_failure_object_unbound'
            else:r['end']=r['clause_start']+match.end()
        elif concept.startswith('oxygen_weaned'):
            if ATTEMPT.search(scope):reason='wean_attempt_instruction_or_goal'
            elif not rx(r'^weaned\b').match(text[start:end]):reason='wean_completion_not_explicit'
        elif concept=='oxygen_required':
            match=NEED_OBJECT.match(clause,a)
            if match is None or match.end()!=b:reason='oxygen_need_object_unbound'
        elif concept in ('dyspnea','dyspnea_improved') and r['polarity']=='affirmed':
            # 原r1证据可能含改善后缀；先只找真正症状位置，再绑定修饰。
            symptom=rx(r'\b'+DYSPNEA+r'\b').match(clause,a)
            if symptom is None:raise ValueError('old dyspnea span does not start with symptom')
            b=symptom.end();right=clause[b:hi]
            post=IMPROVED_AFTER.match(right);pre=IMPROVED_BEFORE.search(left)
            if post or pre:
                r['concept']='dyspnea_improved'
                if post:r['end']=r['clause_start']+b+post.end()
                if pre:r['start']=r['clause_start']+lo+pre.start()
            elif IMPROVEMENT.search(scope):reason='dyspnea_modifier_scope_unbound'
        if ATTEMPT.search(scope) and concept in ('room_air_status','oxygen_required','oxygen_flow_litres_notation'):
            reason=reason or 'instruction_or_attempt_context'
        if reason:
            rejected.append(dict(reason=reason,start=start,end=end));continue
        if concept=='oxygen_flow_litres_notation':
            match=FLOW.match(clause,a)
            if match is None:raise ValueError('old flow span mismatch')
            condition=activity_binding(clause,match)
        else:
            active=bool(ACTIVITY.search(scope));rest=bool(REST.search(scope))
            condition='ambiguous' if active and rest else 'exertion' if active else 'rest' if rest else 'unknown'
        r['activity_condition']=condition;r['exertional']=condition=='exertion'
        r['r1_start']=start;r['r1_end']=end
        accepted.append(r)
    return accepted,rejected
