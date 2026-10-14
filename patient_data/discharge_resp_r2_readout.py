"""r2限定修复去向及有限掩码回查，未建立临床gold。"""
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
from patient_data.discharge_resp_fact_audit_r2 import ROOT
from patient_data.discharge_resp_fact_review import clinical_mask
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json

PARENT=Path('patient_data/processed/v6_discharge_resp_fact_audit_2026-10-13')

def main():
    with gzip.open(ROOT/'facts.jsonl.gz','rt',encoding='utf-8') as f:facts=list(map(json.loads,f))
    old_flags=json.loads((PARENT/'quality_flagged_local.json').read_text())
    flags={};outcomes=defaultdict(Counter)
    for r in facts:flags[(r['unit_key'],r['r1_start'],r['r1_end'])]=r
    for old in old_flags:
        r=flags.get((old['unit_key'],old['start'],old['end']))
        disposition='isolated' if r is None else ('rebound_improvement' if r['concept']=='dyspnea_improved' else
            'retained_symptom_local_scope' if r['concept']=='dyspnea' else r['activity_condition'] if r['concept']=='oxygen_flow_litres_notation' else 'retained')
        for issue in old['issues']:outcomes[issue][disposition]+=1
    with gzip.open(ROOT/'review_sample.jsonl.gz','rt',encoding='utf-8') as f:sample=list(map(json.loads,f))
    summaries=[]
    for i,r in enumerate(sample):
        summaries.append(dict(sample_number=i,concept=r['concept'],role=r['role'],polarity=r['polarity'],value=r['value'],unit=r['unit'],
            activity_condition=r['activity_condition'],clause_mask=clinical_mask(r['quote'][r['clause_start']:r['clause_end']])))
    result=dict(status='limited_semantic_revision_source_verified_not_clinical_gold',
        original_flag_dispositions={k:dict(v) for k,v in outcomes.items()},
        activity_conditions=dict(Counter(r['activity_condition'] for r in facts if r['value'] is not None)),
        records=len(facts),masked_review_samples=len(sample),source_sha256={str(p):sha256(p) for p in
            (ROOT/'audit.json',ROOT/'facts.jsonl.gz',ROOT/'review_sample.jsonl.gz',PARENT/'quality_flagged_local.json',Path(__file__))},
        clinical_precision_recall=None,new_fits=0,new_vectors=0,
        limits=['Fixed clinical-vocabulary masked review not full gold annotation','Isolated targets not true-negative medical diagnoses',
            'Local binding conservatively handles limited surface grammar; no exhaustive accuracy claim'])
    write_json(ROOT/'repair_readout.json',result)
    print(json.dumps(dict(summary=result,samples=summaries),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
