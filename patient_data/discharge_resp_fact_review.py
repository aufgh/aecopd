"""有限审计样本去标识回查：只展示白名单临床词，非人工gold。"""
from collections import Counter
import gzip
import json
from pathlib import Path
import re
from patient_data.discharge_resp_fact_audit import ROOT
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json

WORDS=set('at on upon by prior to time of discharge discharged discharging she he they patient was is were are had has have required requires requiring oxygen o2 supplemental nc nasal cannula liters litre l min room air weaned wean off from successfully unable failed could not able still dyspnea sob doe shortness breath with exertion exertional ambulation walking activity improved resolved subsided remained remains continued continue home no without denies denied any residual significant new ongoing but however yet if should will plan baseline history previously normal stable comfortable breathing tolerating and due needed need needs'.split())

def clinical_mask(text):
    return ' '.join(token.lower() if token.lower() in WORDS else '[x]' for token in re.findall(r'[A-Za-z]+|\d+(?:\.\d+)?',text))

def main():
    with gzip.open(ROOT/'review_sample.jsonl.gz','rt',encoding='utf-8') as f:sample=list(map(json.loads,f))
    summaries=[]
    for i,r in enumerate(sample):
        summaries.append(dict(sample_number=i,concept=r['concept'],role=r['role'],polarity=r['polarity'],value=r['value'],unit=r['unit'],
            clause_mask=clinical_mask(r['quote'][r['clause_start']:r['clause_end']])))
    with gzip.open(ROOT/'facts.jsonl.gz','rt',encoding='utf-8') as f:facts=list(map(json.loads,f))
    issues=Counter();flagged=[]
    for r in facts:
        clause=r['quote'][r['clause_start']:r['clause_end']];reasons=[]
        if r['concept']=='oxygen_failed_wean' and not re.search(r'\b(?:oxygen|O2|room air|NC|nasal cannula)\b',clause,re.I):
            reasons.append('failed_wean_no_explicit_oxygen_object')
        if r['concept']=='dyspnea' and r['polarity']=='affirmed' and re.search(r'\b(?:improv\w*|resolv\w*)\b',clause,re.I):
            reasons.append('dyspnea_mention_with_improvement_scope_unresolved')
        if r['concept'].startswith('oxygen_weaned') and re.search(r'\b(?:attempt\w*|try|trying|instruct\w*|goal)\b',clause,re.I):
            reasons.append('weaning_attempt_or_instruction_not_success_proof')
        if r['concept']=='oxygen_flow_litres_notation' and re.search(r'\brest\b',clause,re.I) and re.search(r'\b(?:ambulat\w*|exertion|activity|walking)\b',clause,re.I):
            reasons.append('mixed_rest_activity_flow_scope_requires_binding')
        for reason in reasons:issues[reason]+=1
        if reasons:
            flagged.append(dict(sentence_row=r['sentence_row'],unit_key=r['unit_key'],start=r['start'],end=r['end'],issues=reasons))
    write_json(ROOT/'quality_readout.json',dict(status='not_ready_for_model_interface',all_lexical_records=len(facts),
        fixed_stratified_sample_records=len(sample),masked_review_only=True,issue_counts=dict(issues),new_fits=0,new_vectors=0,
        clinical_precision_recall=None,source_audit_sha256=sha256(ROOT/'audit.json'),review_code_sha256=sha256(Path(__file__)),
        limits=['Issue flags are review targets, not exhaustive semantic error counts','Masked limited review is not clinical gold labeling',
        'All first-version lexical records retained for audit, not approved current-state facts']))
    write_json(ROOT/'quality_flagged_local.json',flagged)
    print(json.dumps(dict(samples=summaries,role_counts=dict(Counter(r['role'] for r in sample)),issue_counts=dict(issues)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
